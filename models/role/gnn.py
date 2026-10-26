"""이종 그래프 GNN encoder (dense 텐서 + 마스크, 그래프 라이브러리 없음 — RLlib 1.3 호환). 그래프 사양은 env/role/graph_obs.py.

노드 종류 U (UAV) / S (생존자) / K (영역 또는 세그먼트) / D (depot), 관계 (src -> dst, 방향마다 따로):
    S->U, K->U, U->U, D->U  (U 가 받음)    U->S, S->S (S)    U->K, K->K (K)    U->D (D)
B-1 초기 임베딩   h_v^(0) = ReLU(W_type x_v + b_type)                               (차원 D)
B-2 메시지 전달 (L 층, 층 · 관계 · 종류마다 따로)
    m_{u->v}^r = MLP_r([h_u ‖ h_v ‖ e_uv^r])                                       (2 층, ReLU)
    a_v^r      = [ mean_{u ∈ N_r(v)} m ‖ max_{u ∈ N_r(v)} m ]                      (이웃 없으면 0)
    h_v^(ℓ+1)  = LayerNorm(h_v^(ℓ) + MLP_upd,type(v)([h_v ‖ a_v^{r_1} ‖ a_v^{r_2} ‖ ...]))
패딩 노드는 메시지를 보내지도 받지도 않고 (간선 마스크), 출력도 0.
MLP_r 의 첫 층은 [h_u ‖ h_v ‖ e] 를 W_u h_u + W_v h_v + W_e e 로 나눠 계산한다 (같은 값, 쌍마다 큰 텐서를 만들지 않음).
"""
import torch
import torch.nn as nn

from env.role.graph_obs import EKK, ESS, EUD, EUK, EUS, EUU, FD, FK, FS, FU

NODE_DIMS = {"U": FU, "S": FS, "K": FK, "D": FD}
# (이름, src, dst, 간선 키, 간선 특징 차원). 간선 / 인접 텐서는 base [a, b] (행 a, 열 b) 로 들고, 관계 src -> dst 의 메시지는
# [n_dst, n_src] 가 필요하므로 (dst, src) != (a, b) 면 전치한다. S–S / K–K 의 행 = 받는 노드 (kNN 은 행마다 이웃).
RELATIONS = (
    ("S2U", "S", "U", "US", EUS), ("U2S", "U", "S", "US", EUS),
    ("K2U", "K", "U", "UK", EUK), ("U2K", "U", "K", "UK", EUK),
    ("U2U", "U", "U", "UU", EUU),
    ("D2U", "D", "U", "UD", EUD), ("U2D", "U", "D", "UD", EUD),
    ("S2S", "S", "S", "SS", ESS), ("K2K", "K", "K", "KK", EKK),
)
BASE = {"US": ("U", "S"), "UK": ("U", "K"), "UU": ("U", "U"), "UD": ("U", "D"), "SS": ("S", "S"), "KK": ("K", "K")}


def masked_mean_max(m, adj):
    """m (G, n_dst, n_src, D), adj (G, n_dst, n_src) bool -> (G, n_dst, 2D) [mean ‖ max], 이웃 없으면 0."""
    a = adj.unsqueeze(-1).float()
    cnt = a.sum(2)
    mean = (m * a).sum(2) / cnt.clamp(min=1.0)
    mx = m.masked_fill(~adj.unsqueeze(-1), -1e9).max(2).values
    mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], -1)


class RelationMLP(nn.Module):
    """m = W_2 ReLU(W_u h_u + W_v h_v + W_e e + b)  (= 2 층 MLP([h_u ‖ h_v ‖ e]))."""

    def __init__(self, D, edge_dim):
        super().__init__()
        self.w_u = nn.Linear(D, D)
        self.w_v = nn.Linear(D, D, bias=False)
        self.w_e = nn.Linear(edge_dim, D, bias=False)
        self.out = nn.Linear(D, D)

    def forward(self, h_src, h_dst, e):
        """h_src (G, n_src, D), h_dst (G, n_dst, D), e (G, n_dst, n_src, E) -> (G, n_dst, n_src, D)."""
        x = self.w_u(h_src)[:, None, :, :] + self.w_v(h_dst)[:, :, None, :] + self.w_e(e)
        return torch.relu(self.out(torch.relu(x)))


class HeteroGNN(nn.Module):
    def __init__(self, D=128, n_layers=2):
        super().__init__()
        self.D, self.L = D, int(n_layers)
        self.embed = nn.ModuleDict({t: nn.Linear(d, D) for t, d in NODE_DIMS.items()})
        self.rel = nn.ModuleList([nn.ModuleDict({r: RelationMLP(D, e) for r, _, _, _, e in RELATIONS})
                                  for _ in range(self.L)])
        n_in = {t: sum(1 for _, _, dst, _, _ in RELATIONS if dst == t) for t in NODE_DIMS}
        self.upd = nn.ModuleList([nn.ModuleDict({t: nn.Sequential(nn.Linear(D + 2 * D * n_in[t], D), nn.ReLU(),
                                                                  nn.Linear(D, D)) for t in NODE_DIMS})
                                  for _ in range(self.L)])
        self.norm = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(D) for t in NODE_DIMS}) for _ in range(self.L)])

    def forward(self, x, mask, edge, adj):
        """x {종류: (G, n, F)}, mask {종류: (G, n) bool}, edge {키: (G, n_a, n_b, E)}, adj {키: (G, n_a, n_b) bool}
        (키 = BASE 방향, 패딩 노드는 adj 에서 이미 빠져 있어야 함) -> h {종류: (G, n, D)}."""
        h = {t: torch.relu(self.embed[t](x[t])) * mask[t].unsqueeze(-1).float() for t in NODE_DIMS}
        for layer in range(self.L):
            aggs = {t: [] for t in NODE_DIMS}
            for r, src, dst, key, _ in RELATIONS:
                a, b = BASE[key]
                e, ad = edge[key], adj[key]
                if (dst, src) != (a, b):                                   # 받는 쪽이 b: [n_b, n_a] 로 전치
                    e, ad = e.transpose(1, 2), ad.transpose(1, 2)
                m = self.rel[layer][r](h[src], h[dst], e)
                aggs[dst].append(masked_mean_max(m, ad))
            new = {}
            for t in NODE_DIMS:
                u = self.upd[layer][t](torch.cat([h[t]] + aggs[t], -1))
                new[t] = self.norm[layer][t](h[t] + u) * mask[t].unsqueeze(-1).float()
            h = new
        return h
