"""실제 전역 연결성 — 팀 실제 위치 기준 pair_connectivity (연결된 쌍 비율) = 평가 · 진단 지표 conn_real_mean (보상 · 특징에는
쓰지 않음 — 보상 · 특징의 연결도는 통신 그룹 기준 ω^g, env/role/simulation.py group_connectivity).

    간선 (l, l'): ‖p_l - p_l'‖ <= R_c (유클리드, 셀 단위, R_c = 두 UAV 의 기본 통신 반경 중 작은 쪽 — hysteresis 없음)
    C_pair = Σ_k n_k (n_k - 1) / (N (N - 1))     (n_k = 연결 요소 크기, N = 노드 수, N <= 1 이면 0)
예 (N = 5): 5 -> 1.0, 4+1 -> 0.6, 3+2 -> 0.4, 3+1+1 -> 0.3, 2+2+1 -> 0.2, 2+1+1+1 -> 0.1, 전부 끊김 -> 0.
"""
import numpy as np


def pair_connectivity(points, r_c):
    """points (N, 2) 위치, r_c (N,) 노드별 기본 통신 반경 (스칼라도 됨) -> C_pair."""
    p = np.asarray(points, float).reshape(-1, 2)
    n = len(p)
    if n <= 1:
        return 0.0
    rc = np.broadcast_to(np.asarray(r_c, float), (n,))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    lim = np.minimum(rc[:, None], rc[None])
    for a in range(n):
        for b in range(a + 1, n):
            if d[a, b] <= lim[a, b]:
                parent[find(a)] = find(b)
    sizes = np.bincount([find(a) for a in range(n)])
    return float((sizes * (sizes - 1)).sum() / (n * (n - 1)))
