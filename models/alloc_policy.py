"""통합 할당 정책 (Encoder / Context / Decoder) — 논문 그림과 1:1.

노드 슬롯 A = T + 2 : [태스크 1..T | explore | depot],  H^i = [H^i_1; ...; H^i_T; F^i_e; F^i_d]

──────────────── Encoder (에이전트 i 별, 태스크 간 k-NN CCN) ────────────────
    δ^i_j = [x_j/W, y_j/H, τ_j, d^i_j]                     ∈ R^{1x4}
    F^i_j = δ^i_j W_node + b_node                          ∈ R^{1xD}
    F^i_{𝒦_j} = [F^i_{j'}]_{j' ∈ 𝒦_j}                       ∈ R^{k_j x D}     (위치 기준 k-NN, k_j = min(k, T^i - 1))
    F̃^i_j = (F^i_{𝒦_j} - 1_{k_j} F^i_j) W_msg + 1_{k_j} b_msg   ∈ R^{k_j x D}
    F̂^i_j = [F^i_j ; F̃^i_j]                                ∈ R^{(k_j+1) x D}
    H^i_j = 1^T_{k_j+1} (F̂^i_j W_agg + 1_{k_j+1} b_agg)    ∈ R^{1xD}
    F^i_e = δ^i_e W_exp + b_exp,   δ^i_e = [T^i/N_T, E^i_t/|C_free|, d^i_e]
    F^i_d = δ^i_d W_depot + b_depot, δ^i_d = [x_d/W, y_d/H, d^i_d]

──────────────── Context ────────────────
    ξ^i = [x/W, y/H, c/c_max]                 A^i_self = ξ^i W_self + b_self
    m^{i'} = [x, y, c/c_max, g_x, g_y, o (4)]  A^i_nbr  = Σ_{i' ∈ 𝒩^i} (m^{i'} W_nbr + b_nbr)   (n-hop 이웃)
    z^i = [t/t_max, A^i_self, A^i_nbr]        ∈ R^{1x(2D+1)}
    Q^i = z^i W_q + b_q                        ∈ R^{1xD}

──────────────── Decoder ────────────────
    𝐊^i = H^i W_k,  𝐕^i = H^i W_v,  P^i = H^i W_ptr        ∈ R^{(T+2)xD}
    γ^i = MHA(Q^i, 𝐊^i, 𝐕^i)                                ∈ R^{1xD}
    u^i = C · tanh( (γ^i W_glm) P^{i T} / sqrt(D) )          ∈ R^{1x(T+2)}   (Scale & Clip)
    u^i_s = -inf  (s ∉ 𝒜^i)                                                   (Mask)
    π^i = softmax(u^i)                                                        (Log-softmax 는 분포 클래스)

──────────────── 순차(오토리그레시브) 디코딩 ────────────────
dec_rank 순서대로 한 명씩 결정한다. 순서 k 의 에이전트 i 는
  - 같은 n-hop 그룹(reach)에서 앞 순서가 이미 고른 태스크를 mask (𝒮^{<i})
  - 이웃 메시지 m^{i'} 의 (g, o) 는 앞 순서 이웃 = 이번 tick 선택, 뒤 순서 이웃 = 이전 tick 선택
학습 시에는 실제로 취한 행동(taken)으로 같은 루프를 재생하므로 logit 이 rollout 과 일치한다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttention

TASK_DIM = 4          # δ^i_j
EXPLORE_DIM = 3       # δ^i_e
DEPOT_DIM = 3         # δ^i_d
SELF_DIM = 3          # ξ^i
MASK_VALUE = -1e9


class TaskEncoder(nn.Module):
    def __init__(self, embed_dim=128, knn_k=4):
        super().__init__()
        self.k = knn_k
        self.w_node = nn.Linear(TASK_DIM, embed_dim)       # W_node, b_node
        self.w_msg = nn.Linear(embed_dim, embed_dim)       # W_msg, b_msg
        self.w_agg = nn.Linear(embed_dim, embed_dim)       # W_agg, b_agg
        self.w_exp = nn.Linear(EXPLORE_DIM, embed_dim)     # W_exp, b_exp
        self.w_depot = nn.Linear(DEPOT_DIM, embed_dim)     # W_depot, b_depot

    def forward(self, task_feats, task_present, explore_feats, depot_feats):
        """task_feats (M, T, 4), task_present (M, T), explore/depot (M, 3) -> H (M, T+2, D)."""
        M, T, _ = task_feats.shape
        Fn = self.w_node(task_feats)                                   # (M,T,D)  F^i_j
        H_tasks = Fn.new_zeros(Fn.shape)
        k = min(self.k, T - 1)
        present = task_present > 0.5
        if k > 0:
            pos = task_feats[..., :2]
            d = torch.cdist(pos, pos)                                  # (M,T,T)
            eye = torch.eye(T, dtype=torch.bool, device=d.device)[None]
            d = d.masked_fill(eye | ~present[:, None, :], 1e9)
            dval, idx = torch.topk(d, k, dim=-1, largest=False)        # (M,T,k)  𝒦_j
            nbr_valid = (dval < 1e8) & present[:, :, None]             # k_j <= k 인 경우 패딩
            F_nbr = torch.gather(Fn.unsqueeze(1).expand(M, T, T, Fn.shape[-1]), 2,
                                 idx.unsqueeze(-1).expand(M, T, k, Fn.shape[-1]))   # F^i_{𝒦_j}
            F_tilde = self.w_msg(F_nbr - Fn.unsqueeze(2)) * nbr_valid.unsqueeze(-1).float()
            F_hat = torch.cat([Fn.unsqueeze(2), F_tilde], dim=2)       # (M,T,k+1,D)
            valid = torch.cat([present[:, :, None], nbr_valid], dim=2).float()
        else:
            F_hat = Fn.unsqueeze(2)
            valid = present[:, :, None].float()
        H_tasks = (self.w_agg(F_hat) * valid.unsqueeze(-1)).sum(dim=2)  # (M,T,D)  H^i_j (bias 도 행별로 합산)
        F_e = self.w_exp(explore_feats).unsqueeze(1)
        F_d = self.w_depot(depot_feats).unsqueeze(1)
        return torch.cat([H_tasks, F_e, F_d], dim=1)                   # (M,T+2,D)


class ContextEncoder(nn.Module):
    def __init__(self, embed_dim=128, msg_dim=9):
        super().__init__()
        self.w_self = nn.Linear(SELF_DIM, embed_dim)       # W_self, b_self
        self.w_nbr = nn.Linear(msg_dim, embed_dim)         # W_nbr, b_nbr
        self.w_q = nn.Linear(2 * embed_dim + 1, embed_dim) # W_q, b_q

    def forward(self, self_feats, msgs, reach, t):
        """self_feats (B,N,3), msgs (B,N,9) 팀 테이블, reach (B,N,N) n-hop, t (B,1) -> Q (B,N,D)."""
        B, N, _ = self_feats.shape
        A_self = self.w_self(self_feats)
        A_nbr = torch.bmm(reach, self.w_nbr(msgs))                    # Σ_{i'∈𝒩^i} (m W_nbr + b_nbr)
        z = torch.cat([t[:, None, :].expand(B, N, 1), A_self, A_nbr], dim=-1)
        return self.w_q(z)


class PointerDecoder(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, clip=10.0):
        super().__init__()
        self.D = embed_dim
        self.clip = clip
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_ptr = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_glm = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MultiHeadAttention(embed_dim, n_heads)

    def keys(self, H):
        return self.w_k(H), self.w_v(H), self.w_ptr(H)                # 𝐊, 𝐕, P

    def forward(self, Q, K, V, P, present, mask):
        """Q (M,D), K/V/P (M,A,D), present (M,A) MHA key 마스크, mask (M,A) admissible.
        반환 logits (M,A) (Scale & Clip + Mask), γ (M,D)."""
        gamma = self.mha(Q.unsqueeze(1), K, V, key_mask=present).squeeze(1)
        u = torch.matmul(self.w_glm(gamma).unsqueeze(1), P.transpose(-2, -1)).squeeze(1)
        u = self.clip * torch.tanh(u / math.sqrt(self.D))
        return u.masked_fill(~mask, MASK_VALUE), gamma


class AllocationPolicy(nn.Module):
    """Encoder + Context + Decoder + 순차 디코딩."""

    def __init__(self, n_tasks, n_msg_types=4, embed_dim=128, n_heads=8, knn_k=4, clip=10.0):
        super().__init__()
        self.T = n_tasks
        self.A = n_tasks + 2
        self.n_msg_types = n_msg_types
        self.encoder = TaskEncoder(embed_dim, knn_k)
        self.context = ContextEncoder(embed_dim, msg_dim=3 + 2 + n_msg_types)
        self.decoder = PointerDecoder(embed_dim, n_heads, clip)

    def forward(self, ob, taken=None, greedy=False):
        """ob: models/rescue_net.py 가 쌓은 텐서 dict (B,N,...).
        taken (B,N) long: 학습 시 실제 행동 (None 이면 rollout — 여기서 샘플).
        반환: logits (B,N,A), allocs (B,N) long, gamma (B,N,D)."""
        B, N, T = ob["task_feats"].shape[:3]
        A, D = self.A, self.decoder.D
        M = B * N

        # ---- Encoder (선택과 무관 -> 한 번만) ----
        H = self.encoder(ob["task_feats"].reshape(M, T, -1), ob["task_present"].reshape(M, T),
                         ob["explore_feats"].reshape(M, -1), ob["depot_feats"].reshape(M, -1))
        K, V, P = self.decoder.keys(H)
        ones = H.new_ones(M, 2)
        present = torch.cat([ob["task_present"].reshape(M, T), ones], dim=1) > 0.5   # (M,A)
        base_mask = ob["node_mask"].reshape(M, A) > 0.5

        reach = ob["reach"]                                        # (B,N,N)
        rank = ob["dec_rank"].round().long()                       # (B,N)
        task_ids = ob["task_ids"]                                  # (B,N,T)
        node_type = ob["node_type"].round().long()                 # (B,N,A)
        node_xy = ob["node_xy"]                                    # (B,N,A,2)

        msg_g = ob["prev_msg"][..., :2].clone()                    # (B,N,2)
        msg_o = ob["prev_msg"][..., 2:].clone()                    # (B,N,4)
        picked = H.new_full((B, N), -1.0)                          # 고른 태스크의 target id
        done = torch.zeros(B, N, dtype=torch.bool, device=H.device)

        logits_out = H.new_full((B, N, A), MASK_VALUE)
        gamma_out = H.new_zeros(B, N, D)
        allocs = torch.zeros(B, N, dtype=torch.long, device=H.device)

        for k in range(N):
            sel = rank == k                                        # (B,N)
            if not bool(sel.any()):
                continue
            # 그룹 내 앞 순서 선택 마스크 𝒮^{<i}
            link = reach * done.float()[:, None, :]                # (B,i,j)
            same = (task_ids[:, :, None, :] == picked[:, None, :, None]) & (picked[:, None, :, None] >= 0)
            conflict = ((same.float() * link[..., None]).sum(dim=2) > 0)            # (B,N,T)
            conflict = torch.cat([conflict, conflict.new_zeros(B, N, 2)], dim=-1)   # explore/depot 공유 가능
            mask = base_mask & ~conflict.reshape(M, A)
            mask[:, A - 1] = mask[:, A - 1] | ~mask.any(dim=1)     # 안전장치: 전부 막히면 depot

            msgs = torch.cat([ob["pub"], msg_g, msg_o], dim=-1)    # (B,N,9)
            Q = self.context(ob["self_feats"], msgs, reach, ob["time"]).reshape(M, D)
            logits, gamma = self.decoder(Q, K, V, P, present, mask)
            logits = logits.reshape(B, N, A)
            gamma = gamma.reshape(B, N, D)

            if taken is not None:
                act = taken.long().clamp(0, A - 1)
            elif greedy:
                act = logits.argmax(dim=-1)
            else:
                act = torch.distributions.Categorical(logits=logits.detach()).sample()

            logits_out = torch.where(sel[..., None], logits, logits_out)
            gamma_out = torch.where(sel[..., None], gamma, gamma_out)
            allocs = torch.where(sel, act, allocs)

            # 이번에 결정한 에이전트의 메시지 / 선택 태스크 갱신
            a_sel = act.clamp(0, A - 1)
            g_new = torch.gather(node_xy, 2, a_sel[..., None, None].expand(B, N, 1, 2)).squeeze(2)
            o_new = F.one_hot(torch.gather(node_type, 2, a_sel[..., None]).squeeze(-1),
                              self.n_msg_types).float()
            tid = torch.gather(task_ids, 2, a_sel.clamp(max=T - 1)[..., None]).squeeze(-1)
            tid = torch.where(a_sel < T, tid, torch.full_like(tid, -1.0))
            msg_g = torch.where(sel[..., None], g_new, msg_g)
            msg_o = torch.where(sel[..., None], o_new, msg_o)
            picked = torch.where(sel, tid, picked)
            done = done | sel
        return logits_out, allocs, gamma_out
