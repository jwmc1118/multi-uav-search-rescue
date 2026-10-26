"""4-행동 action distribution (Role / Rescue / Explore 1 단계 / Explore 2 단계) — 계층 구조 env (models/role/rllib_model.py 와 약속).

입력 레이아웃: [ explore logits N*K | explore_sub N*M | rescue N*T | role N*3 |
               passthrough explore N | explore_sub N | rescue N | role N | 단계 플래그 s1 N | s2 N ]
행동 공간 Dict{role, rescue, explore, explore_sub} 은 gym 이 키를 정렬 -> child 분포 순서 = [explore | explore_sub | rescue | role] x N.
행동은 모델이 순차 디코딩으로 이미 뽑았으므로 sample / deterministic_sample 은 passthrough 를 돌려준다.
logp / entropy / kl 은 학습 헤드 {'role', 'rescue', 'explore'} -> (B, N) 딕셔너리 (training/role/ppo.py 가 헤드별 loss).
explore 헤드 = 1 단계 + 2 단계 하나의 자기회귀 행동: logp = s1·logp1 + s2·logp2 (entropy / kl 도 같은 식).
결정이 없는 step 의 행동은 dummy — loss 에서 *_train 마스크로 뺀다.
"""
import torch
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.utils.annotations import override

HEAD_ORDER = ("explore", "explore_sub", "rescue", "role")   # 평탄화 / child 분포 순서


class RoleActionDistribution(TorchMultiActionDistribution):
    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space):
        self.n_agents = len(input_lens) // 4
        N = self.n_agents
        pt = inputs[:, -6 * N:-2 * N].long().clamp(min=0)
        self.taken = {h: pt[:, k * N:(k + 1) * N] for k, h in enumerate(HEAD_ORDER)}
        fl = inputs[:, -2 * N:]
        self.s1, self.s2 = fl[:, :N], fl[:, N:]
        super().__init__(inputs[:, :-6 * N], model, child_distributions=child_distributions,
                         input_lens=input_lens, action_space=action_space)

    def _children(self):
        N, d = self.n_agents, self.flat_child_distributions
        return {h: d[k * N:(k + 1) * N] for k, h in enumerate(HEAD_ORDER)}

    def _out(self, acts):
        return {h: tuple(acts[h][:, i] for i in range(self.n_agents)) for h in HEAD_ORDER}

    @override(TorchMultiActionDistribution)
    def sample(self):
        return self._out(self.taken)

    @override(TorchMultiActionDistribution)
    def deterministic_sample(self):
        return self._out(self.taken)

    def _split(self, x):
        N = self.n_agents
        if isinstance(x, dict):                     # (B, N) 텐서 (passthrough) 또는 에이전트별 튜플 (RLlib 행동 dict)
            return {h: x[h] if torch.is_tensor(x[h]) else torch.stack(list(x[h]), dim=-1) for h in HEAD_ORDER}
        return {h: x[:, k * N:(k + 1) * N] for k, h in enumerate(HEAD_ORDER)}

    def _combine(self, per):
        """child 별 (B, N) -> 학습 헤드: explore = s1·(1 단계) + s2·(2 단계)."""
        return {"role": per["role"], "rescue": per["rescue"],
                "explore": self.s1 * per["explore"] + self.s2 * per["explore_sub"]}

    @override(TorchMultiActionDistribution)
    def logp(self, x):
        xs, hd = self._split(x), self._children()
        per = {h: torch.stack([hd[h][i].logp(xs[h][:, i].long()) for i in range(self.n_agents)], dim=-1)
               for h in HEAD_ORDER}
        return self._combine(per)

    @override(TorchMultiActionDistribution)
    def entropy(self):
        hd = self._children()
        return self._combine({h: torch.stack([d.entropy() for d in hd[h]], dim=-1) for h in HEAD_ORDER})

    @override(TorchMultiActionDistribution)
    def kl(self, other):
        a, b = self._children(), other._children()
        return self._combine({h: torch.stack([a[h][i].kl(b[h][i]) for i in range(self.n_agents)], dim=-1)
                              for h in HEAD_ORDER})

    def logp_taken(self):
        """passthrough 행동의 헤드별 logp (B, N) — rollout 시 extra_action_out 로 저장."""
        return self.logp(self.taken)

    @override(TorchMultiActionDistribution)
    def sampled_action_logp(self):
        # RLlib ACTION_LOGP 자리 = role 헤드 (B, N). rescue / explore 는 extra_action_out 로 따로 저장
        return self.logp_taken()["role"]
