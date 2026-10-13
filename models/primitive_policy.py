"""Primitive policy — 할당 결과(목적지)를 받아 한 칸 이동을 고른다.

    2D state = [local (S,S,C_l) ⊕ waypoint 채널 (S,S,1)] ──CNN(Flatten)──▶ l^i
    C_l = cfg["local_channels"] (기본 obstacle, agents — yaml 에서 frontier / explored / target 추가 가능)
    MLP( l^i ⊕ mode one-hot(3) ⊕ waypoint [dx, dy, valid] [⊕ γ^i] ) ──▶ a^i ∈ 8방향 + stay

- local    : 자기 중심 고해상도 크롭 (회전 없음, 북쪽 = 위). Flatten 종단으로 공간 배치 보존.
- waypoint : 선택된 노드(태스크 / frontier / depot)까지의 BFS 경로가 local 창을 벗어나기 직전 셀
             (목적지가 창 안이면 목적지). 환경이 노드별로 계산해 node_wp 로 주고, 모델이 고른 슬롯의
             것을 꺼내 2D one-hot 채널로 그린다 -> 창 밖 목표도 창 안의 2D 정보로 바뀐다 (Two-stage 논문 방식).
- γ^i      : 할당 decoder 의 glimpse (상황 요약). 옵션 primitive_glimpse —
             'none'(기본, 미사용) / 'detach'(사용, primitive loss 가 할당 정책을 흔들지 않게) / 'grad'.
행동: Robot.ACTIONS = 8방향 (N, NE, E, SE, S, SW, W, NW) + stay. local / waypoint / 행동이 모두 같은 월드 방위.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.agent import Robot
from env.modes import N_MODES
from models.encoders import GridCNN, MLP, per_agent

N_ACTIONS = len(Robot.ACTIONS)   # 9
WP_DIM = 3                       # [dx, dy, valid]


def slot_mode_onehot(allocs, n_tasks):
    """슬롯 -> Mode one-hot (EXPLORE=0, RESCUE=1, RETURN=2; env/modes.py 의 Mode.value)."""
    mode = torch.where(allocs < n_tasks, torch.ones_like(allocs),
                       torch.where(allocs == n_tasks, torch.zeros_like(allocs),
                                   torch.full_like(allocs, 2)))
    return F.one_hot(mode, N_MODES).float()


def render_waypoint(wp, size):
    """wp (B,N,3) [dx, dy, valid] (÷ half) -> (B,N,S,S,1) one-hot. 축: x 오른쪽 = col+, y 위쪽 = row-."""
    B, N, _ = wp.shape
    half = size // 2
    col = (half + torch.round(wp[..., 0] * half)).long().clamp(0, size - 1)
    row = (half - torch.round(wp[..., 1] * half)).long().clamp(0, size - 1)
    flat = (row * size + col).reshape(B * N)
    grid = wp.new_zeros(B * N, size * size)
    grid[torch.arange(B * N, device=wp.device), flat] = wp[..., 2].reshape(B * N)
    return grid.view(B, N, size, size, 1)


class PrimitivePolicy(nn.Module):
    def __init__(self, in_channels, local_size, local_dim, glimpse_dim, filters,
                 hidden=(128, 64), activation=nn.ReLU, glimpse="none"):
        super().__init__()
        assert glimpse in ("detach", "grad", "none")
        self.size = local_size
        self.glimpse = glimpse
        self.local_cnn = GridCNN(in_channels + 1, local_dim, filters=filters, activation=activation,
                                 global_pool=False, input_hw=(local_size, local_size))
        extra = N_MODES + WP_DIM + (0 if glimpse == "none" else glimpse_dim)
        self.actor = MLP(local_dim + extra, hidden, N_ACTIONS, activation)

    def forward(self, local_map, mode_oh, wp, gamma):
        # local_map (B,N,S,S,C), mode_oh (B,N,3), wp (B,N,3), gamma (B,N,D) -> logits (B,N,9)
        x = torch.cat([local_map, render_waypoint(wp, self.size)], dim=-1)
        lf = per_agent(self.local_cnn, x)
        feats = [lf, mode_oh, wp]
        if self.glimpse == "detach":
            feats.append(gamma.detach())
        elif self.glimpse == "grad":
            feats.append(gamma)
        return self.actor(torch.cat(feats, dim=-1))
