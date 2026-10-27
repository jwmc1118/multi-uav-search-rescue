"""이종 그래프 GNN (dense 텐서 + 마스크, 그래프 라이브러리 없음 — torch 1.8 / RLlib 1.3 호환). 그래프 사양은 env/role/graph_obs.py.

역할 선택 / 생존자 할당 / frontier 할당 네트워크가 각자 HeteroGNN 인스턴스를 가진다 (파라미터 공유 없음).

    MLP       g(x) = W2 ReLU(W1 x + b1) + b2   (모든 MLP 2 층)
    encoder   h_v^0 = MLP_type(x_v) (노드 종류마다), e_uv^0 = MLP_edge(x_uv) (간선 종류마다), 차원 D
    집계 (층 · 관계마다 파라미터 따로, 관계 r = src -> dst)
      attention (기본)  q_v = W_q h_v,  k_uv = W_k [h_u ‖ e_uv],  val_uv = W_v [h_u ‖ e_uv]
                         a_v^r = W_o MHA(q_v, {k_uv}, {val_uv})   (adj 마스크로 실제 이웃만, 이웃이 없으면 0)
      mean_max (ablation) m_uv = MLP([h_u ‖ h_v ‖ e_uv]),  a_v^r = [mean_u m ‖ max_u m]
    노드 갱신   h_v' = LayerNorm(h_v + MLP_upd,type([h_v ‖ a_v^{r1} ‖ a_v^{r2} ‖ ...]))
    간선 갱신 (선택, 예: 생존자 할당의 U–S)  e_us' = LayerNorm(e_us + MLP([e_us ‖ h_u ‖ h_s]))   (노드 갱신 뒤)
패딩 노드는 메시지를 보내지도 받지도 않고 (adj 마스크) 출력도 0. 집계는 이웃 순서에 무관 (순열 불변).
간선 텐서는 BASE 방향 [a, b] (a 행 · b 열) 로 들고, 관계 src -> dst 의 메시지는 [n_dst, n_src] 가 필요하므로
(dst, src) != (a, b) 면 전치한다. S–S / K–K / U–U 의 행 = 받는 노드.
"""
import math

import torch
import torch.nn as nn

AGGRS = ("attention", "mean_max")


def mlp2(n_in, n_hid, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hid), nn.ReLU(), nn.Linear(n_hid, n_out))


def masked_mean_max(m, adj):
    """m (G, n_dst, n_src, D), adj (G, n_dst, n_src) bool -> (G, n_dst, 2D) [mean ‖ max], 이웃 없으면 0."""
    a = adj.unsqueeze(-1).float()
    cnt = a.sum(2)
    mean = (m * a).sum(2) / cnt.clamp(min=1.0)
    mx = m.masked_fill(~adj.unsqueeze(-1), -1e9).max(2).values
    mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], -1)


class AttentionAgg(nn.Module):
    """관계 하나의 multi-head attention 집계. W_k / W_v 의 [h_u ‖ e_uv] 는 W h_u + W' e_uv 로 나눠 계산 (같은 값)."""

    def __init__(self, D, heads):
        super().__init__()
        assert D % heads == 0, f"embed_dim {D} 가 attn_heads {heads} 로 나눠지지 않음"
        self.H, self.dh = heads, D // heads
        self.w_q = nn.Linear(D, D, bias=False)
        self.w_ku, self.w_ke = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.w_vu, self.w_ve = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.w_o = nn.Linear(D, D)
        self.out_dim = D

    def forward(self, h_src, h_dst, e, adj):
        """h_src (G, n_src, D), h_dst (G, n_dst, D), e (G, n_dst, n_src, D), adj (G, n_dst, n_src) -> (G, n_dst, D)."""
        G, nd, ns, D = e.shape
        H, dh = self.H, self.dh
        q = self.w_q(h_dst).view(G, nd, 1, H, dh)
        k = (self.w_ku(h_src)[:, None] + self.w_ke(e)).view(G, nd, ns, H, dh)
        v = (self.w_vu(h_src)[:, None] + self.w_ve(e)).view(G, nd, ns, H, dh)
        score = (q * k).sum(-1) / math.sqrt(dh)                                  # (G, nd, ns, H)
        m = adj.unsqueeze(-1)
        score = score.masked_fill(~m, -1e9)
        w = torch.softmax(score, dim=2) * m.float()                              # 이웃 없는 행은 전부 0
        out = (w.unsqueeze(-1) * v).sum(2).reshape(G, nd, D)
        has = adj.any(-1, keepdim=True).float()
        return self.w_o(out) * has


class MeanMaxAgg(nn.Module):
    """ablation: m_uv = MLP([h_u ‖ h_v ‖ e_uv]), a_v = [mean ‖ max]."""

    def __init__(self, D, heads=None):
        super().__init__()
        self.w_u, self.w_v, self.w_e = nn.Linear(D, D), nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.out = nn.Linear(D, D)
        self.out_dim = 2 * D

    def forward(self, h_src, h_dst, e, adj):
        x = self.w_u(h_src)[:, None] + self.w_v(h_dst)[:, :, None] + self.w_e(e)
        return masked_mean_max(self.out(torch.relu(x)), adj)


class HeteroGNN(nn.Module):
    """node_dims {종류: F}, edge_dims {간선 키: E}, base {간선 키: (a, b)}, relations [(이름, src, dst, 간선 키)],
    edge_update = 층마다 갱신할 간선 키 (없으면 None)."""

    def __init__(self, node_dims, edge_dims, base, relations, D=128, n_layers=2, aggr="attention", heads=4,
                 edge_update=None):
        super().__init__()
        if aggr not in AGGRS:
            raise ValueError(f"gnn.aggr={aggr!r} — {' | '.join(AGGRS)}")
        self.types, self.base, self.rel, self.L, self.D = tuple(node_dims), dict(base), tuple(relations), int(n_layers), D
        self.edge_update = edge_update
        self.enc = nn.ModuleDict({t: mlp2(f, D, D) for t, f in node_dims.items()})
        self.enc_e = nn.ModuleDict({k: mlp2(f, D, D) for k, f in edge_dims.items()})
        Agg = AttentionAgg if aggr == "attention" else MeanMaxAgg
        self.agg = nn.ModuleList([nn.ModuleDict({r: Agg(D, heads) for r, _, _, _ in self.rel}) for _ in range(self.L)])
        a_dim = self.agg[0][self.rel[0][0]].out_dim if self.L else D
        n_in = {t: sum(1 for _, _, dst, _ in self.rel if dst == t) for t in self.types}
        self.upd = nn.ModuleList([nn.ModuleDict({t: mlp2(D + a_dim * n_in[t], D, D) for t in self.types})
                                  for _ in range(self.L)])
        self.norm = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(D) for t in self.types}) for _ in range(self.L)])
        if edge_update:
            self.e_upd = nn.ModuleList([mlp2(3 * D, D, D) for _ in range(self.L)])
            self.e_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(self.L)])

    def forward(self, x, mask, edge, adj):
        """x {종류: (G, n, F)}, mask {종류: (G, n) bool}, edge {키: (G, n_a, n_b, E)}, adj {키: (G, n_a, n_b) bool}
        (패딩 노드는 adj 에서 이미 빠져 있어야 함) -> (h {종류: (G, n, D)}, h0 (encoder 출력), e {키: (G, n_a, n_b, D)})."""
        mk = {t: mask[t].unsqueeze(-1).float() for t in self.types}
        h = {t: self.enc[t](x[t]) * mk[t] for t in self.types}
        h0 = h
        e = {k: self.enc_e[k](edge[k]) * adj[k].unsqueeze(-1).float() for k in edge}
        for layer in range(self.L):
            aggs = {t: [] for t in self.types}
            for r, src, dst, key in self.rel:
                ee, ad = e[key], adj[key]
                if (dst, src) != self.base[key]:
                    ee, ad = ee.transpose(1, 2), ad.transpose(1, 2)
                aggs[dst].append(self.agg[layer][r](h[src], h[dst], ee, ad))
            h = {t: self.norm[layer][t](h[t] + self.upd[layer][t](torch.cat([h[t]] + aggs[t], -1))) * mk[t]
                 for t in self.types}
            if self.edge_update:
                k = self.edge_update
                a, b = self.base[k]
                ha, hb = h[a][:, :, None].expand(-1, -1, e[k].shape[2], -1), h[b][:, None].expand(-1, e[k].shape[1], -1, -1)
                upd = self.e_upd[layer](torch.cat([e[k], ha, hb], -1))
                e = dict(e, **{k: self.e_norm[layer](e[k] + upd) * adj[k].unsqueeze(-1).float()})
        return h, h0, e
