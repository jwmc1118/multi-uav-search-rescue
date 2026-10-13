"""Multi-Head Attention (glimpse) — 논문 표기 그대로.

    head_m = softmax( (Q W^m_q)(𝐊 W^m_k)^T / sqrt(D_h) ) (𝐕 W^m_v)      W^m_* ∈ R^{D x D_h}, D_h = D / n_h
    γ      = [head_1, ..., head_{n_h}] W_o + b_o                           ∈ R^{1 x D}

𝐊 = H W_k, 𝐕 = H W_v 는 호출부(models/alloc_policy.py)가 만든다. 헤드별 투영 W^m_* 는 하나의
(D x D) 행렬을 n_h 블록으로 나눈 것과 같으므로 Linear(D, D, bias=False) 하나로 구현한다.
key_mask 로 패딩 노드(실제 후보가 아닌 태스크 슬롯)를 attention 에서 뺀다.
torch 1.8 에는 nn.MultiheadAttention(batch_first=...) 이 없어 직접 구현한다.
"""
import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "D 는 n_h 로 나누어 떨어져야 함"
        self.h = n_heads
        self.dh = embed_dim // n_heads
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)    # [W^1_q ... W^{n_h}_q]
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim)                 # W_o, b_o

    def forward(self, query, key, value, key_mask=None):
        """query (B, Lq, D), key/value (B, Lk, D), key_mask (B, Lk) True = 사용. 반환 (B, Lq, D)."""
        B, Lq, D = query.shape
        Lk = key.shape[1]
        q = self.w_q(query).view(B, Lq, self.h, self.dh).transpose(1, 2)    # (B,h,Lq,dh)
        k = self.w_k(key).view(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.w_v(value).view(B, Lk, self.h, self.dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B,h,Lq,Lk)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask.bool()[:, None, None, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Lq, D)       # concat heads
        return self.w_o(out)
