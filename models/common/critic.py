"""CTDE critic (학습 전용, 실행 시 제거) — models/role/net.py 가 헤드 (role / rescue / explore) 마다 하나씩 쓴다.

privileged ground truth + 팀원 행동:
    V_i = MLP( CNN_gap(S) ⊕ CNN_flat(o_i) ⊕ extra_i ⊕ task_ctx_i ⊕ team_ctx_i ⊕ global )
    task_ctx_i : critic_tasks (실제 조난자 전부) 임베딩에 query = 자기 임베딩(ξ^i) 1-layer MHA pooling ⊕ 마스크 mean pooling
                 (학습되는 null 토큰을 항상 붙여 조난자 0 명이어도 NaN 없음)
    team_ctx_i : critic_team 각 행 ⊕ [자기 자신 플래그, 자기 기준 상대위치 dx, dy] ⊕ 팀원 할당 행동 a^{-i}
                 (선택 슬롯 타입 one-hot + 선택 노드 좌표, 자기 행은 0 으로 가림 -> V(s, a^{-i})) 임베딩, mean ⊕ max pooling
    global     : critic_global

- actor 와 인코더를 공유하지 않는다 (value loss gradient 가 actor 를 흔들지 않게). 헤드마다 인스턴스 따로.
- extra_i: 헤드별 입력 (role 관측 [+ 노드 특징], models/role/net.py).
"""
import math

import torch
import torch.nn as nn

from models.common.encoders import GridCNN, MLP, per_agent

# critic 전용 관측 차원 기본값 (env/core/base_sim.py BaseSimulation.CRITIC_*_DIM 과 같아야 함)
CRITIC_TASK_DIM = 9
CRITIC_TEAM_DIM = 10
CRITIC_GLOBAL_DIM = 5
SELF_FEAT_DIM = 3          # ξ^i = [x, y, c/c_max]
TEAM_ACT_DIM = 3 + 2       # 팀원 할당: 슬롯 타입 one-hot (explore / rescue / return) + 선택 노드 좌표


def masked_mean(x, mask, dim):
    """x (..., K, d), mask (..., K) -> (..., d). 전부 마스크면 0 (NaN 없음)."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1.0)


class SetAttentionPool(nn.Module):
    """query (M, d) 가 집합 keys (M, K, d) 에 multi-head attention. 학습되는 null 토큰을 항상 앞에 붙여
    집합이 비어도 (전부 마스크) softmax 가 NaN 이 되지 않는다."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, "critic_set_dim 은 critic_set_heads 의 배수여야 함"
        self.h, self.dh = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.null = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, query, keys, mask):
        M, K, d = keys.shape
        keys = torch.cat([self.null.expand(M, 1, d), keys], dim=1)
        keep = torch.cat([mask.new_ones(M, 1), mask], dim=1) > 0.5
        q = self.q(query).view(M, self.h, 1, self.dh)
        k = self.k(keys).view(M, K + 1, self.h, self.dh).transpose(1, 2)
        v = self.v(keys).view(M, K + 1, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)                  # (M, h, 1, K+1)
        att = att.masked_fill(~keep[:, None, None, :], -1e9).softmax(dim=-1)
        return self.o((att @ v).reshape(M, d))


class RichCentralCritic(nn.Module):
    """rich: V_i = MLP( CNN_gap(S) ⊕ CNN_flat(o_i) ⊕ extra_i ⊕ task_ctx_i ⊕ team_ctx_i ⊕ global )."""

    def __init__(self, state_channels, local_channels, local_hw, extra_dim, state_dim, local_dim, filters,
                 hidden=(128, 64), activation=nn.ReLU, set_dim=64, set_heads=4,
                 task_dim=CRITIC_TASK_DIM, team_dim=CRITIC_TEAM_DIM, global_dim=CRITIC_GLOBAL_DIM):
        super().__init__()
        self.state_cnn = GridCNN(state_channels, state_dim, filters=filters, activation=activation,
                                 global_pool=True)
        self.local_cnn = GridCNN(local_channels, local_dim, filters=filters, activation=activation,
                                 global_pool=False, input_hw=local_hw)
        self.task_embed = MLP(task_dim, (set_dim,), set_dim, activation)
        self.self_embed = MLP(SELF_FEAT_DIM, (set_dim,), set_dim, activation)
        self.task_attn = SetAttentionPool(set_dim, set_heads)
        self.team_embed = MLP(team_dim + 1 + 2 + TEAM_ACT_DIM, (set_dim,), set_dim, activation)
        self.value = MLP(state_dim + local_dim + extra_dim + 4 * set_dim + global_dim, hidden, 1, activation)
        self.input_dim = state_dim + local_dim + extra_dim + 4 * set_dim + global_dim

    def forward(self, state_map, local_map, extra, self_feats, tasks, task_present, team, team_act, glob):
        """self_feats (B,N,3) | tasks (B,T,Ft) | task_present (B,T) | team (B,N,Fm) | team_act (B,N,5) | glob (B,G)
        -> V (B,N)"""
        B, N = local_map.shape[:2]
        T = tasks.shape[1]
        sf = self.state_cnn(state_map).unsqueeze(1).expand(B, N, -1)
        lf = per_agent(self.local_cnn, local_map)

        # ---- task_ctx_i: 자기 query 로 조난자 집합 attention + 마스크 mean ----
        te = self.task_embed(tasks)                                             # (B,T,d)
        d = te.shape[-1]
        q = self.self_embed(self_feats).reshape(B * N, d)
        keys = te.unsqueeze(1).expand(B, N, T, d).reshape(B * N, T, d)
        mask = task_present.unsqueeze(1).expand(B, N, T).reshape(B * N, T)
        att = self.task_attn(q, keys, mask).reshape(B, N, d)
        mean_t = masked_mean(te, task_present, dim=1).unsqueeze(1).expand(B, N, d)

        # ---- team_ctx_i: 행 j 에 [자기 플래그, (x_j - x_i, y_j - y_i), a^j (자기 행은 0)] ----
        eye = torch.eye(N, device=team.device, dtype=team.dtype)
        rows = team.unsqueeze(1).expand(B, N, N, team.shape[-1])                 # [b, i, j] = team_j
        rel = team[:, None, :, :2] - team[:, :, None, :2]                       # x_j - x_i
        act = team_act.unsqueeze(1).expand(B, N, N, team_act.shape[-1]) * (1 - eye)[None, :, :, None]
        flag = eye[None, :, :, None].expand(B, N, N, 1)
        tm = self.team_embed(torch.cat([rows, flag, rel, act], dim=-1))          # (B,N,N,d)
        team_ctx = torch.cat([tm.mean(dim=2), tm.max(dim=2).values], dim=-1)

        g = glob.unsqueeze(1).expand(B, N, glob.shape[-1])
        x = torch.cat([sf, lf, extra, att, mean_t, team_ctx, g], dim=-1)
        return self.value(x).squeeze(-1)
