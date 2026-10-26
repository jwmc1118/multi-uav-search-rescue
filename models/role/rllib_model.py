"""RoleModel — models/role/net.py 를 RLlib 1.3 TorchModelV2 로 감싼 것 (계층 구조 env, env/role/gym_env.py).

행동 공간 Dict{role, rescue, explore, explore_sub} — gym 이 키를 정렬하므로 평탄화 순서 = [explore | explore_sub | rescue | role].
모델 출력 레이아웃 (training/role/action_dist.py 와 약속):
    [ explore logits N*K | explore_sub logits N*M | rescue logits N*T | role logits N*3 |
      행동 passthrough: explore N | explore_sub N | rescue N | role N | 단계 플래그: s1 N | s2 N ]
logits 는 순차 디코딩의 결정 시점 마스크가 적용된 값. 행동은 모델이 순차 디코딩으로 뽑은 값을 그대로 쓴다.
s1 / s2 = 그 UAV 가 이번 step explore 1 단계 / 2 단계를 학습 정책으로 결정했는가 (explore logp = s1·logp1 + s2·logp2,
2 단계 후보가 1 개 이하 / hungarian 이면 s2 = 0 -> log π2 = 0).
학습 forward 는 SampleBatch.ACTIONS (폭 4N, 순서 explore | explore_sub | rescue | role) 로 디코딩을 재생해 같은 logits 를 만든다.
value_function() = 실제 스케일 {'role', 'rescue', 'explore'} (B, N), value_function_norm() = critic 출력 그대로.
greedy = True 면 (평가) 세 헤드 모두 argmax.
"""
import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override

from models.role.net import RoleNet, stack_role_obs

# rollout 시 extra_action_out 으로 저장되는 헤드별 컬럼 (role 헤드는 RLlib 기본 키 VF_PREDS / ACTION_LOGP)
VF_PREDS_RESCUE = "vf_preds_rescue"
VF_PREDS_EXPLORE = "vf_preds_explore"
ACTION_LOGP_RESCUE = "action_logp_rescue"
ACTION_LOGP_EXPLORE = "action_logp_explore"
EXTRA_KEYS = (VF_PREDS_RESCUE, VF_PREDS_EXPLORE, ACTION_LOGP_RESCUE, ACTION_LOGP_EXPLORE)


def role_dims(orig):
    """obs space (original) -> RoleNet dims."""
    ag = orig["agents"][0]
    return {"role_self": ag["role_self"].shape[0], "role_group": ag["role_group"].shape[0],
            "g_res": ag["role_g_res"].shape[0], "g_exp": ag["role_g_exp"].shape[0],
            "critic_task": orig["critic_tasks"].shape[-1], "critic_team": orig["critic_team"].shape[-1],
            "critic_global": orig["critic_global"].shape[-1]}


class RoleModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        orig = obs_space.original_space
        ag = orig["agents"][0]
        self.n_agents = len(orig["agents"])
        self.max_targets = ag["s_ids"].shape[0]               # S 노드 수 = s_max (rescue 행동 공간)
        self.k_max = ag["k_pos"].shape[0]
        self.n_sub = int(round(float(ag["explore_kcap"].high.max())))
        S, _, C = ag["local"].shape
        self.net = RoleNet(self.n_agents, self.max_targets, self.k_max, S, C, orig["state"].shape[-1],
                           role_dims(orig), model_config.get("custom_model_config", {}), n_sub=self.n_sub)
        self.greedy = False
        self._values = self._values_norm = None
        self.last_out = None                                   # GUI / 분석용 (마지막 forward 결과)
        per_agent = Box(-np.inf, np.inf, (self.n_agents,), np.float32)
        for k in EXTRA_KEYS:
            self.view_requirements[k] = ViewRequirement(space=per_agent)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if not isinstance(obs, dict):
            obs = restore_original_dimensions(obs, self.obs_space, "torch")
        ob = stack_role_obs(obs)
        N = self.n_agents
        B = ob["role_self"].shape[0]
        taken = input_dict.get(SampleBatch.ACTIONS)
        taken_d = None
        if taken is not None and not torch.is_tensor(taken):
            taken = torch.as_tensor(np.asarray(taken), device=ob["role_self"].device)
        if torch.is_tensor(taken) and taken.dim() == 2 and taken.shape[1] == 4 * N:
            taken_d = {"explore": taken[:, :N].long(), "explore_sub": taken[:, N:2 * N].long(),
                       "rescue": taken[:, 2 * N:3 * N].long(), "role": taken[:, 3 * N:].long()}
        out = self.net(ob, taken=taken_d, greedy=self.greedy)
        self._values, self._values_norm = out["values"], out["values_norm"]
        self.last_out = out
        lg, act, st = out["logits"], out["actions"], out["stage"]
        logits = torch.cat([lg["explore"].reshape(B, -1), lg["explore_sub"].reshape(B, -1), lg["rescue"].reshape(B, -1),
                            lg["role"].reshape(B, -1),
                            act["explore"].float(), act["explore_sub"].float(), act["rescue"].float(), act["role"].float(),
                            st["s1"].float(), st["s2"].float()], dim=1)
        return logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._values is not None, "forward() 를 먼저 호출해야 함"
        return self._values

    def value_function_norm(self):
        assert self._values_norm is not None, "forward() 를 먼저 호출해야 함"
        return self._values_norm
