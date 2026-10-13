"""랜덤 맵 / 조난자 생성 — 말이 안 되는 상황이 생기지 않도록 검사 포함.

장애물 (고정 고도 UAV 기준: 건물 / 비행금지구역 = 직사각형 블록)
    1) 목표 비율 ρ ~ U[obstacle_ratio] 를 뽑고, 랜덤 크기 직사각형을 비율이 ρ 에 닿을 때까지 놓는다.
    2) depot 주변(clear_radius)은 비운다 — 출발하자마자 갇히지 않게.
    3) depot 에서 flood fill (8-연결 + 코너 컷팅 금지 = UAV 이동 규칙과 동일) 로 도달 가능한 칸을 구하고,
       도달 불가능한 빈 칸(갇힌 섬)은 장애물로 메운다 -> 모든 빈 칸은 depot 에서 갈 수 있다.
    4) 최종 비율이 범위 안이고 도달 가능 영역이 충분히 크면 채택, 아니면 다시 생성 (rejection sampling).

조난자
    - 수: num_targets 범위에서 균등 (예: [10, 20])
    - 위치: depot 에서 도달 가능한 빈 칸 중에서만 (depot 근처 min_target_dist 칸 이내 제외)
      -> 장애물에 둘러싸여 발견/도달 불가능한 조난자는 생기지 않는다.
    - 골든타임: 기존대로 target_golden_time 범위에서 매 에피소드 뽑는다 (simulation._sample_golden).
"""
from collections import deque

import numpy as np

_NEIGHBORS_8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def reachable_from(start, blocked):
    """start 에서 8-연결(코너 컷팅 금지)로 갈 수 있는 칸 = True 인 (rows, cols) bool."""
    rows, cols = blocked.shape
    seen = np.zeros_like(blocked, dtype=bool)
    if blocked[start]:
        return seen
    seen[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or seen[nr, nc] or blocked[nr, nc]:
                continue
            if dr and dc and blocked[r + dr, c] and blocked[r, c + dc]:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))
    return seen


def _sample_ratio(spec, rng):
    if isinstance(spec, (list, tuple)):
        return float(rng.uniform(spec[0], spec[1]))
    return float(spec)


def _bounds(spec):
    if isinstance(spec, (list, tuple)):
        return float(spec[0]), float(spec[1])
    return float(spec), float(spec)


def generate_obstacles(rows, cols, rng, obstacle_ratio=(0.2, 0.3), depot=None,
                       clear_radius=2, block_size=None, min_free_reachable=0.5, max_tries=300):
    """랜덤 장애물 맵 생성. 반환: (obstacles (rows, cols) bool, depot (r, c)).

    depot=None 이면 매번 랜덤 칸. block_size=(min, max) 직사각형 한 변 길이 (기본: 맵 크기에 비례).
    min_free_reachable: 도달 가능 영역이 전체 칸의 이 비율 이상이어야 채택.
    """
    lo, hi = _bounds(obstacle_ratio)
    if block_size is None:
        s = max(rows, cols)
        block_size = (2, max(3, s // 6))
    bmin, bmax = int(block_size[0]), int(block_size[1])
    n_cells = rows * cols

    for _ in range(max_tries):
        target = _sample_ratio(obstacle_ratio, rng)
        d = depot if depot is not None else (int(rng.randint(rows)), int(rng.randint(cols)))
        grid = np.zeros((rows, cols), dtype=bool)
        guard = 0
        while grid.mean() < target and guard < 2000:
            guard += 1
            h = int(rng.randint(bmin, bmax + 1))
            w = int(rng.randint(bmin, bmax + 1))
            r = int(rng.randint(0, max(1, rows - h + 1)))
            c = int(rng.randint(0, max(1, cols - w + 1)))
            grid[r:r + h, c:c + w] = True
        # depot 주변 비우기
        r0, c0 = d
        grid[max(0, r0 - clear_radius):r0 + clear_radius + 1,
             max(0, c0 - clear_radius):c0 + clear_radius + 1] = False
        # 갇힌 빈 칸 메우기
        reach = reachable_from(d, grid)
        grid |= ~reach
        ratio = grid.mean()
        if lo - 1e-9 <= ratio <= hi + 1e-9 and reach.sum() >= min_free_reachable * n_cells * (1 - hi):
            return grid, d
    raise RuntimeError(f"장애물 맵 생성 실패 ({max_tries}회): obstacle_ratio={obstacle_ratio}, 맵 {rows}x{cols}")


def sample_target_cells(obstacles, depot, rng, n, min_target_dist=3):
    """depot 에서 도달 가능한 빈 칸 중 n 개 (depot Chebyshev 거리 min_target_dist 이내 제외)."""
    blocked = np.asarray(obstacles, dtype=bool)
    reach = reachable_from(tuple(depot), blocked)
    cells = np.argwhere(reach)
    far = np.maximum(np.abs(cells[:, 0] - depot[0]), np.abs(cells[:, 1] - depot[1])) > min_target_dist
    if far.sum() >= n:
        cells = cells[far]
    n = min(n, len(cells))
    idx = rng.choice(len(cells), size=n, replace=False)
    return [tuple(int(v) for v in cells[k]) for k in idx]


def sample_count(spec, rng):
    """int 또는 [min, max] (양끝 포함)."""
    if isinstance(spec, (list, tuple)):
        return int(rng.randint(int(spec[0]), int(spec[1]) + 1))
    return int(spec)
