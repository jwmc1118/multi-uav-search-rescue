"""CTDE critic (학습 전용, 실행 시 제거).

    V_i = MLP( CNN(State S) ⊕ CNN(local o_i) ⊕ extra_i )

- actor 와 CNN 을 공유하지 않는다: value loss 의 gradient 가 actor 인코더를 흔들지 않게.
- 헤드(alloc / primitive)마다 인스턴스를 따로 만든다.
- extra_i: alloc critic = [ξ^i, t/t_max], primitive critic = [ξ^i, t/t_max, mode one-hot, waypoint]
  (자기 중심 local 만으로는 맵 위 위치를 모르므로 ξ^i 를 항상 넣는다).
"""
import torch
import torch.nn as nn

from models.encoders import GridCNN, MLP, per_agent


class CentralCritic(nn.Module):
    def __init__(self, state_channels, local_channels, local_hw, extra_dim, state_dim, local_dim, filters,
                 hidden=(128, 64), activation=nn.ReLU):
        super().__init__()
        self.state_cnn = GridCNN(state_channels, state_dim, filters=filters, activation=activation,
                                 global_pool=True)
        self.local_cnn = GridCNN(local_channels, local_dim, filters=filters, activation=activation,
                                 global_pool=False, input_hw=local_hw)
        self.value = MLP(state_dim + local_dim + extra_dim, hidden, 1, activation)

    def forward(self, state_map, local_map, extra):
        # state_map (B,gh,gw,C) | local_map (B,N,S,S,C) | extra (B,N,E) -> V (B,N)
        B, N = local_map.shape[:2]
        sf = self.state_cnn(state_map).unsqueeze(1).expand(B, N, -1)
        lf = per_agent(self.local_cnn, local_map)
        return self.value(torch.cat([sf, lf, extra], dim=-1)).squeeze(-1)
