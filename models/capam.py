"""CapAM Decoder — Role / Rescue / Explore head 공통 pointer decoder (노드 임베딩 = 공유 이종 그래프 GNN 출력,
models/role/gnn.py).

    Q = W_q q_in   (q_in = 호출부가 이어 붙인 쿼리 — rescue: [q_ctx ‖ mean 후보 S], explore: [q_ctx ‖ mean K ‖ h_c ‖ e_stage],
                    role: [q_ctx ‖ mean(H)])
    K = W_k F,  V = W_v F,  glimpse = MHA(Q, K, V)   (n_heads, 유효 노드만 attention; 전부 무효면 전부 열어 NaN 방지)
    u_j = C · tanh( (W_Q' glimpse)ᵀ (W_K' F_j) / √D )     -> 마스크 / log-softmax 는 호출부 (models/role/net.py)
"""
import math

import torch
import torch.nn as nn

from models.common.attention import MultiHeadAttention

MASK_VALUE = -1e9          # 마스크된 노드 / 역할의 logit


class CapAMDecoder(nn.Module):
    def __init__(self, embed_dim=128, q_dim=256, n_heads=8, clip=10.0):
        super().__init__()
        self.D = embed_dim
        self.clip = clip
        self.w_q = nn.Linear(q_dim, embed_dim)                        # W_q q_in
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MultiHeadAttention(embed_dim, n_heads)
        self.w_qp = nn.Linear(embed_dim, embed_dim, bias=False)       # W_Q'
        self.w_kp = nn.Linear(embed_dim, embed_dim, bias=False)       # W_K'

    def forward(self, q_in, F, valid):
        """q_in (M,Q), F (M,n,D), valid (M,n) -> u (M,n) (마스크 전, Scale & Clip 적용)."""
        valid = valid > 0.5 if valid.dtype != torch.bool else valid
        key_mask = valid | ~valid.any(dim=-1, keepdim=True)                     # 유효 노드 0 개면 전부 열어 NaN 방지
        Q = self.w_q(q_in)
        g = self.mha(Q.unsqueeze(1), self.w_k(F), self.w_v(F), key_mask=key_mask).squeeze(1)
        u = (self.w_qp(g).unsqueeze(1) * self.w_kp(F)).sum(-1) / math.sqrt(self.D)
        return self.clip * torch.tanh(u)
