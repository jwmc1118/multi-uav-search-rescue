"""연결성 — 팀 실제 위치 기준. pair_connectivity (연결된 쌍 비율) = 진단 로그 conn/real_mean,
personal_connectivity (개인 연결도 c_i) = rescue 보너스 (옵션 동안 평균) · explore 연결 배율.

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



def personal_connectivity(points, r_c):
    """개인 연결도 c_i = (|comp(i)| - 1) / (N - 1) (N <= 1 이면 0) — 간선 규칙은 pair_connectivity 와 같다.
    rescue 보너스 (옵션 동안 평균) 와 explore 연결 배율 (1 + w_mult c) 에 쓴다. 반환 (N,) 리스트."""
    p = np.asarray(points, float).reshape(-1, 2)
    n = len(p)
    if n <= 1:
        return [0.0] * n
    rc = np.broadcast_to(np.asarray(r_c, float), (n,))
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    adj = d <= np.minimum(rc[:, None], rc[None])
    comp = [-1] * n
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s], stack = s, [s]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u]):
                if comp[v] < 0:
                    comp[v] = s
                    stack.append(v)
    size = {c: comp.count(c) for c in set(comp)}
    return [(size[comp[i]] - 1) / (n - 1) for i in range(n)]
