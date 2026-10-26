"""frontier 덩어리 (DBSCAN) + 덩어리 안 구간 분할 (측지 k-medoids) — Explore 할당의 노드.

    blobs = frontier_clusters(frontier_mask, reachable=reach_mask)      # A-1 덩어리 (크기 내림차순, 전부)
    D = geodesic_matrix(cells)                                         # 덩어리 셀끼리 8-연결 측지 거리
    segs = partition(cells, k)                                         # A-5 구간 k 개 [(cells, medoid), ...]

1) A-1 덩어리: frontier 셀 중 reachable (그룹 내 어떤 UAV 에서도 BFS 도달 가능) 만 남긴 뒤 DBSCAN.
   이웃 = 유클리드 거리 <= eps 인 격자 오프셋 (eps 1.5 -> 8-연결). 자기 포함 이웃 수 >= min_samples 면 core.
   core 에서 core 로 퍼지며 키우고, core 의 이웃인 non-core 셀은 border 로 붙인다. 나머지는 noise (버림).
   안전장치: 결과가 비었는데 frontier 셀이 있으면 (전부 noise) 8-연결 성분별로 묶는다 (deadlock 방지).
2) 대표점 medoid = 셀 좌표 평균에 가장 가까운 실제 frontier 셀. 정렬 = 크기 내림차순, 동률이면 medoid 좌표 순.
3) A-5 구간 분할: 덩어리 셀만 지나는 8-연결 BFS 측지 거리로 k-medoids -> 구간마다 연결 성분 1 개로 보정
   (contiguous_labels) -> 크기 균형 보정 (balance_labels, 연결성 유지).
   초기 medoid = farthest-point (첫 점 = 측지 거리 합 최소 셀, 이후 기존 medoid 들에서 가장 먼 셀, 동률은 셀 좌표 순)
   -> 할당 / medoid 갱신 (구간 안 측지 거리 합 최소 셀) 을 수렴할 때까지 (최대 iters 회) 반복. 결정적 (같은 입력 -> 같은 결과),
   medoid 는 항상 실제 frontier 셀. 결과 구간 순서 = medoid 좌표 순.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


@dataclass
class Cluster:
    cells: np.ndarray            # (m, 2) int (row, col)
    medoid: tuple                # (row, col)
    id: int = -1                 # 덩어리 id (env 가 step 간 대응으로 붙인다, A-2)
    origin: str = "new"          # keep / split / merge / new (A-2 판정)
    parents: tuple = ()          # 대응된 이전 step 덩어리 id
    _set: frozenset = field(default=None, repr=False)

    @property
    def size(self):
        return len(self.cells)

    def cell_set(self):
        if self._set is None:
            self._set = frozenset((int(r), int(c)) for r, c in self.cells)
        return self._set


def _offsets(eps):
    k = int(math.floor(eps))
    return [(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1) if dr * dr + dc * dc <= eps * eps + 1e-9]


def dbscan_grid(cells, eps=1.5, min_samples=3):
    """cells: [(r, c), ...] -> 클러스터별 셀 리스트 목록 (noise 제외)."""
    cells = [tuple(map(int, c)) for c in cells]
    index = {c: k for k, c in enumerate(cells)}
    offs = _offsets(eps)
    nbrs = []
    for (r, c) in cells:
        nbrs.append([index[(r + dr, c + dc)] for dr, dc in offs if (r + dr, c + dc) in index])
    core = [len(n) >= min_samples for n in nbrs]
    label = [-1] * len(cells)
    out = []
    for s in range(len(cells)):
        if not core[s] or label[s] >= 0:
            continue
        cid = len(out)
        members = []
        label[s] = cid
        stack = [s]
        while stack:
            p = stack.pop()
            members.append(p)
            if not core[p]:
                continue                       # border 는 퍼뜨리지 않음
            for q in nbrs[p]:
                if label[q] < 0:
                    label[q] = cid
                    stack.append(q)
        out.append([cells[k] for k in members])
    return out


def medoid(cells):
    pts = np.asarray(cells, float)
    k = int(np.argmin(((pts - pts.mean(0)) ** 2).sum(1)))
    return int(cells[k][0]), int(cells[k][1])


def frontier_clusters(frontier, eps=1.5, min_samples=3, k_max=None, reachable=None):
    """frontier (rows, cols) bool [& reachable] -> [Cluster] (크기 내림차순). k_max 를 주면 상위 k_max 개만."""
    mask = frontier if reachable is None else (frontier & reachable)
    cells = sorted(tuple(map(int, p)) for p in np.argwhere(mask))
    raw = dbscan_grid(cells, eps, min_samples)
    if not raw and cells:
        raw = dbscan_grid(cells, eps, 1)                     # 전부 noise -> 연결 성분 단위 (deadlock 방지)
    groups = [sorted(g) for g in raw]
    out = [Cluster(np.asarray(g, int), medoid(g)) for g in groups]
    out.sort(key=lambda cl: (-cl.size, cl.medoid))
    return out if k_max is None else out[:k_max]


# ---------------------------------------------------------------- 측지 거리 / k-medoids (A-5)

def geodesic_matrix(cells):
    """cells [(r, c)] -> (n, n) 측지 거리 (덩어리 셀만 지나는 8-연결, 한 칸 = 1). 끊긴 쌍은 inf."""
    n = len(cells)
    index = {c: k for k, c in enumerate(cells)}
    rows, cols = [], []
    for k, (r, c) in enumerate(cells):
        for dr, dc in NEIGHBORS8:
            j = index.get((r + dr, c + dc))
            if j is not None:
                rows.append(k)
                cols.append(j)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return shortest_path(g, method="D", directed=False, unweighted=True)


def kmedoids(D, k, iters=10, init=None):
    """D (n, n) 측지 거리 (inf 허용) -> (labels (n,), medoid index 리스트). 결정적 (인덱스 순 tie-break).
    init = 초기 medoid index 목록 (warm start, 중복 제거 후 앞에서 k 개). 모자라면 farthest-point 로 채운다."""
    n = len(D)
    k = max(1, min(k, n))
    big = float(np.nanmax(D[np.isfinite(D)])) + 1.0 if np.isfinite(D).any() else 1.0
    Df = np.where(np.isfinite(D), D, big * 4)                  # 끊긴 쌍 = 아주 먼 거리
    meds = [int(m) for m in dict.fromkeys(init or [])][:k]
    if not meds:
        meds = [int(np.argmin(Df.sum(1)))]                    # 1-medoid (argmin 은 첫 최솟값 = 인덱스 순)
    while len(meds) < k:
        dmin = Df[:, meds].min(1)
        dmin[meds] = -1
        meds.append(int(np.argmax(dmin)))                     # farthest-point
    labels = np.argmin(Df[:, meds], axis=1)
    for _ in range(iters):
        new = []
        for j in range(k):
            idx = np.flatnonzero(labels == j)
            if len(idx) == 0:
                new.append(meds[j])
                continue
            sub = Df[np.ix_(idx, idx)].sum(1)
            new.append(int(idx[int(np.argmin(sub))]))
        lab = np.argmin(Df[:, new], axis=1)
        if new == meds and np.array_equal(lab, labels):
            break
        meds, labels = new, lab
    return labels, meds


def contiguous_labels(cells, labels, meds):
    """구간마다 연결 성분 1 개로 보정: medoid 가 있는 성분만 남기고, 떨어진 조각은 맞닿은 (8-이웃) 다른 구간 중
    가장 많이 닿은 구간에 붙인다 (반복). 붙일 곳이 없는 조각은 -1 (어느 구간에도 안 넣음). cells 는 labels 와 같은 순서."""
    labels = np.asarray(labels).copy()
    index = {c: n for n, c in enumerate(cells)}
    nbr = [[index[(r + dr, c + dc)] for dr, dc in NEIGHBORS8 if (r + dr, c + dc) in index] for r, c in cells]
    for _ in range(len(cells)):
        changed = False
        for j, m in enumerate(meds):
            members = set(np.flatnonzero(labels == j).tolist())
            if not members:
                continue
            keep, stack = {m} if m in members else set(), [m] if m in members else []
            if not stack:                                   # medoid 가 빠졌으면 가장 큰 성분
                stack = [min(members)]
                keep = {stack[0]}
            while stack:
                u = stack.pop()
                for v in nbr[u]:
                    if v in members and v not in keep:
                        keep.add(v)
                        stack.append(v)
            rest = members - keep
            while rest:
                comp, stack = {rest.pop()}, None
                stack = list(comp)
                while stack:
                    u = stack.pop()
                    for v in nbr[u]:
                        if v in rest:
                            rest.discard(v)
                            comp.add(v)
                            stack.append(v)
                touch = {}
                for u in comp:
                    for v in nbr[u]:
                        if labels[v] >= 0 and labels[v] != j:
                            touch[int(labels[v])] = touch.get(int(labels[v]), 0) + 1
                target = max(sorted(touch), key=lambda t: touch[t]) if touch else -1
                for u in comp:
                    labels[u] = target
                changed = True
        if not changed:
            break
    return labels


def balance_labels(cells, labels, meds, D, target_cv=0.25):
    """구간 크기 균형 보정 (연결성 유지): 크기 변동계수 (std / mean) 가 target_cv 이하가 될 때까지, 가장 큰 구간부터
    경계 칸 하나를 맞닿은 (8-이웃) 더 작은 구간으로 옮긴다 — medoid 칸은 옮기지 않고, 옮겨도 원래 구간이 연결 성분
    1 개로 남는 칸만 (원래 구간 medoid 에서 측지 거리가 먼 칸 우선). 옮길 칸이 없으면 멈춘다."""
    labels = np.asarray(labels).copy()
    k = len(meds)
    if k < 2:
        return labels
    index = {c: n for n, c in enumerate(cells)}
    nbr = [[index[(r + dr, c + dc)] for dr, dc in NEIGHBORS8 if (r + dr, c + dc) in index] for r, c in cells]
    med_set = set(meds)
    Df = np.where(np.isfinite(D), D, 1e6)

    def connected_without(d, u):
        members = set(np.flatnonzero(labels == d).tolist()) - {u}
        if not members:
            return False
        start = meds[d] if meds[d] in members else min(members)
        seen, stack = {start}, [start]
        while stack:
            x = stack.pop()
            for v in nbr[x]:
                if v in members and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == len(members)

    for _ in range(len(cells)):
        sizes = np.bincount(labels[labels >= 0], minlength=k).astype(float)
        if sizes.mean() <= 0 or sizes.std() / sizes.mean() <= target_cv:
            break
        moved = False
        for d in np.argsort(-sizes, kind="stable"):
            cands = []
            for u in np.flatnonzero(labels == d):
                if u in med_set:
                    continue
                recv = [int(labels[v]) for v in nbr[u] if labels[v] >= 0 and labels[v] != d and sizes[labels[v]] < sizes[d] - 1]
                if recv:
                    r = min(recv, key=lambda t: (sizes[t], t))
                    cands.append((sizes[r], -Df[u, meds[d]], int(u), r))
            for _, _, u, r in sorted(cands):
                if connected_without(int(d), u):
                    labels[u] = r
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    return labels


def partition(cells, k, D=None, iters=10, init=None):
    """덩어리 셀을 측지 k-medoids 로 k 구간 -> [(cells tuple, medoid (r, c)), ...] (medoid 좌표 순). 빈 구간 없음.
    init = 초기 medoid 셀 목록 (warm start, cells 안의 셀; 모자라면 farthest-point)."""
    cells = sorted(tuple(map(int, c)) for c in cells)
    if k <= 1 or len(cells) <= 1:
        return [(tuple(cells), medoid(cells))] if cells else []
    D = geodesic_matrix(cells) if D is None else D
    index = {c: n for n, c in enumerate(cells)}
    seeds = [index[tuple(c)] for c in (init or []) if tuple(c) in index]
    labels, meds = kmedoids(D, k, iters, init=seeds)
    labels = contiguous_labels(cells, labels, meds)                # 구간마다 연결 성분 1 개
    labels = balance_labels(cells, labels, meds, D)                # 크기 균형 (연결성 유지)
    out = []
    for j, m in enumerate(meds):
        seg = tuple(c for c, l in zip(cells, labels) if l == j)
        if seg:
            out.append((seg, cells[m]))
    out.sort(key=lambda s: s[1])
    return out
