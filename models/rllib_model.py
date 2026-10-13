"""RescueModel — models/rescue_net.py 를 RLlib 1.3.0 TorchModelV2 로 감싼 것.

모델 출력 레이아웃 (training/action_dist.py 와 약속, 행동 공간 키 정렬 순서 = alloc, primitive):
    [ alloc logits N*A | primitive logits N*9 | allocs N ]
    A = T + 2. 마지막 N 은 모델이 순차 디코딩으로 뽑은 할당 (rollout 시 그대로 행동이 된다).
학습 forward 는 SampleBatch.ACTIONS (폭 2N, 순서 alloc | primitive) 의 실제 할당으로 디코딩을 재생한다.
value_function() 은 {'alloc', 'primitive'} -> (B, N) 딕셔너리.
alloc_greedy = True 로 두면 (평가) 할당을 argmax 로 고른다.
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

from models.rescue_net import RescueNet, stack_obs
from training.sample_batch_keys import VF_PREDS_ALLOC, ACTION_LOGP_ALLOC


class RescueModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        orig = obs_space.original_space
        agent_space = orig["agents"][0]
        self.n_agents = len(orig["agents"])
        self.max_targets = agent_space["task_feats"].shape[0]
        self.n_alloc = agent_space["node_mask"].shape[0]
        S, _, C = agent_space["local"].shape
        self.net = RescueNet(self.n_agents, self.max_targets, S, C, orig["state"].shape[-1],
                             model_config.get("custom_model_config", {}))
        self.alloc_greedy = False
        self._values = None
        # rollout 시 extra_action_out 으로 저장되는 alloc 헤드 컬럼이 train_batch 에 실려 오도록.
        # space 를 (N,) 로 명시해야 collector 가 object 배열이 아닌 (T, N) float 배열로 쌓는다.
        per_agent = Box(-np.inf, np.inf, (self.n_agents,), np.float32)
        for k in (VF_PREDS_ALLOC, ACTION_LOGP_ALLOC):
            self.view_requirements[k] = ViewRequirement(space=per_agent)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if not isinstance(obs, dict):
            obs = restore_original_dimensions(obs, self.obs_space, "torch")
        ob = stack_obs(obs)
        N = self.n_agents
        B = ob["self_feats"].shape[0]

        # 학습 시 RLlib 은 ACTIONS 를 numpy 로 넘기기도 한다 (SampleBatch.copy) -> 반드시 텐서로 변환.
        # 놓치면 모델이 할당을 다시 샘플링해 순차 디코딩 조건이 rollout 과 달라진다 (logp / KL 오류).
        taken = input_dict.get(SampleBatch.ACTIONS)
        taken_alloc = None
        if taken is not None and not torch.is_tensor(taken):
            taken = torch.as_tensor(np.asarray(taken), device=ob["self_feats"].device)
        if torch.is_tensor(taken) and taken.dim() == 2 and taken.shape[1] == 2 * N:
            taken_alloc = taken[:, :N].long()                    # ACTIONS = [alloc | primitive]

        out = self.net(ob, taken_alloc=taken_alloc, greedy=self.alloc_greedy)
        self._values = out["values"]
        logits = torch.cat([
            out["alloc_logits"].reshape(B, N * self.n_alloc),
            out["prim_logits"].reshape(B, -1),
            out["allocs"].float(),
        ], dim=1)
        return logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._values is not None, "forward() 를 먼저 호출해야 함"
        return self._values
