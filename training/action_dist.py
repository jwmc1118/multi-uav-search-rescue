"""2-head 계층 action distribution (Role_selection 의 TorchHomogeneousMultiActionDistribution 확장).

모델 출력 레이아웃 (폭 W, models/rllib_model.py):
    [ 0       : N*A          )  alloc logits      (A = T + 2: 태스크 T | explore | depot)
    [ N*A     : N*A + N*9    )  primitive logits  (9 = 8방향 + stay)
    [ W - N   : W            )  allocs (모델이 순차 디코딩으로 뽑은 값, passthrough)

행동 공간은 Dict{alloc, primitive} (gym 이 키를 정렬 -> alloc 먼저). RLlib 은 custom_action_dist 가
TorchMultiActionDistribution 의 서브클래스이면 child_distributions / input_lens 를 행동 공간에서 자동
구성한다: sum(input_lens) = N*A + N*9 = num_outputs. 모델은 passthrough N 을 더 붙여 W = num_outputs + N.

할당 logit 은 순차 디코딩(앞 순서 에이전트의 선택에 조건부)이라 sample 은 모델이 이미 뽑은 값을 그대로 쓴다.
logp / entropy / kl 은 {'alloc', 'primitive'} 딕셔너리를 반환하고 training/multiagent_ppo.py 가 헤드별 loss 를 만든다.
"""
import torch
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.utils.annotations import override


class HierarchicalMultiActionDistribution(TorchMultiActionDistribution):
    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space):
        self.n_agents = len(input_lens) // 2
        N = self.n_agents
        self.allocs_taken = inputs[:, -N:].long().clamp_(min=0)
        super().__init__(inputs[:, :-N], model,
                         child_distributions=child_distributions,
                         input_lens=input_lens, action_space=action_space)

    # ---- child dist slices (order: [alloc]*N + [prim]*N) ----
    def _heads(self):
        N = self.n_agents
        d = self.flat_child_distributions
        return d[0:N], d[N:2 * N]

    def _out(self, prim_actions):
        N = self.n_agents
        return {"alloc": tuple(self.allocs_taken[:, i] for i in range(N)),
                "primitive": tuple(prim_actions)}

    @override(TorchMultiActionDistribution)
    def sample(self):
        _, prim_d = self._heads()
        return self._out(d.sample() for d in prim_d)

    @override(TorchMultiActionDistribution)
    def deterministic_sample(self):
        _, prim_d = self._heads()
        return self._out(d.deterministic_sample() for d in prim_d)

    @override(TorchMultiActionDistribution)
    def logp(self, x):
        alloc_d, prim_d = self._heads()
        N = self.n_agents
        if isinstance(x, dict):
            xa = torch.stack(list(x["alloc"]), dim=-1)
            xp = torch.stack(list(x["primitive"]), dim=-1)
        else:
            xa, xp = x[:, :N], x[:, N:2 * N]
        la = torch.stack([alloc_d[i].logp(xa[:, i].long()) for i in range(N)], dim=-1)
        lp = torch.stack([prim_d[i].logp(xp[:, i]) for i in range(N)], dim=-1)
        return {"alloc": la, "primitive": lp}

    @override(TorchMultiActionDistribution)
    def entropy(self):
        alloc_d, prim_d = self._heads()
        N = self.n_agents
        return {"alloc": torch.stack([alloc_d[i].entropy() for i in range(N)], dim=-1),
                "primitive": torch.stack([prim_d[i].entropy() for i in range(N)], dim=-1)}

    @override(TorchMultiActionDistribution)
    def kl(self, other):
        aa, pa = self._heads()
        ab, pb = other._heads()
        N = self.n_agents
        return {"alloc": torch.stack([aa[i].kl(ab[i]) for i in range(N)], dim=-1),
                "primitive": torch.stack([pa[i].kl(pb[i]) for i in range(N)], dim=-1)}

    @override(TorchMultiActionDistribution)
    def sampled_action_logp(self):
        # RLlib ACTION_LOGP 자리 — primitive 헤드만 (B, N). alloc 은 extra_action_out 로 저장.
        _, prim_d = self._heads()
        return torch.stack([d.sampled_action_logp() for d in prim_d], dim=-1)

    def logp_alloc(self):
        alloc_d, _ = self._heads()
        N = self.n_agents
        return torch.stack([alloc_d[i].logp(self.allocs_taken[:, i]) for i in range(N)], dim=-1)
