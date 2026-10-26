"""랜덤 맵 / 조난자 생성 — 말이 안 되는 상황이 생기지 않도록 검사 포함.

장애물 (고정 고도 UAV 기준: 건물 / 비행금지구역 = 직사각형 블록)
    1) 목표 비율 ρ ~ U[obstacle_ratio] 를 뽑고, 랜덤 크기 직사각형을 비율이 ρ 에 닿을 때까지 놓는다.
    2) depot 주변(clear_radius)은 비운다 — 출발하자마자 갇히지 않게.
    2.5) 통로 폭 (min_passage_width = 2, 기본): 모든 빈 칸이 '4 칸 모두 빈 2x2 블록' 에 속하도록 한다.
       - 1 칸 폭 틈 (블록 사이 / 블록과 맵 경계 사이) 과 대각선 끼임 (2x2 안에서 빈 칸 두 개가 대각선으로만 맞닿고
         나머지 두 칸이 장애물 — 코너 컷팅 금지라 사실상 1 칸 통로) 을 없앤다.
       - 장애물로 메우면 영역이 끊길 수 있으므로 장애물을 지워 넓힌다 (그 칸을 포함하는 2x2 중 장애물이 가장 적은 것).
       - 넓히느라 비율이 범위보다 낮아지면 블록을 더 놓고 다시 넓힌다. min_passage_width = 1 이면 이 단계를 건너뛴다.
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


def narrow_cells(grid):
    """2 칸 폭 조건 위반 (빈 칸 기준): (narrow, pinch)
    narrow (rows, cols) bool: 4 칸 모두 빈 2x2 블록에 하나도 속하지 않는 빈 칸
    pinch  (rows-1, cols-1) bool: 대각선 끼임 2x2 (빈 칸이 한 대각선에만 있고 다른 대각선은 둘 다 장애물)"""
    free = ~grid
    win = free[:-1, :-1] & free[1:, :-1] & free[:-1, 1:] & free[1:, 1:]          # 2x2 블록 왼쪽 위 기준
    covered = np.zeros_like(free)
    covered[:-1, :-1] |= win
    covered[1:, :-1] |= win
    covered[:-1, 1:] |= win
    covered[1:, 1:] |= win
    a, b, c, d = free[:-1, :-1], free[:-1, 1:], free[1:, :-1], free[1:, 1:]     # [[a, b], [c, d]]
    pinch = (a & d & ~b & ~c) | (b & c & ~a & ~d)
    return free & ~covered, pinch


def widen_passages(grid, rng, max_iter=200):
    """1 칸 통로 / 대각선 끼임을 장애물을 지워 넓힌다 (in-place). 반환: 남은 위반 수 (0 이면 완료)."""
    rows, cols = grid.shape
    for _ in range(max_iter):
        narrow, pinch = narrow_cells(grid)
        if not narrow.any() and not pinch.any():
            return 0
        for r, c in np.argwhere(pinch):                                         # 끼임: 장애물 하나를 연다
            obs = [(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1) if grid[r + dr, c + dc]]
            if obs:
                grid[obs[int(rng.randint(len(obs)))]] = False
        narrow, _ = narrow_cells(grid)
        for r, c in np.argwhere(narrow):                                        # 좁은 칸: 장애물이 가장 적은 2x2 를 비운다
            if not narrow_cells(grid)[0][r, c]:
                continue
            best = None
            for r0 in (r - 1, r):
                for c0 in (c - 1, c):
                    if 0 <= r0 < rows - 1 and 0 <= c0 < cols - 1:
                        k = int(grid[r0:r0 + 2, c0:c0 + 2].sum())
                        if best is None or k < best[0] or (k == best[0] and rng.rand() < 0.5):
                            best = (k, r0, c0)
            if best is not None:
                grid[best[1]:best[1] + 2, best[2]:best[2] + 2] = False
    narrow, pinch = narrow_cells(grid)
    return int(narrow.sum() + pinch.sum())


def _sample_ratio(spec, rng):
    if isinstance(spec, (list, tuple)):
        return float(rng.uniform(spec[0], spec[1]))
    return float(spec)


def _bounds(spec):
    if isinstance(spec, (list, tuple)):
        return float(spec[0]), float(spec[1])
    return float(spec), float(spec)


def generate_obstacles(rows, cols, rng, obstacle_ratio=(0.2, 0.3), depot=None,
                       clear_radius=2, block_size=None, min_free_reachable=0.5, max_tries=300, min_passage_width=2):
    """랜덤 장애물 맵 생성. 반환: (obstacles (rows, cols) bool, depot (r, c)).

    depot=None 이면 매번 랜덤 칸. block_size=(min, max) 직사각형 한 변 길이 (기본: 맵 크기에 비례).
    min_free_reachable: 도달 가능 영역이 전체 칸의 이 비율 이상이어야 채택.
    min_passage_width: 2 = 모든 통로 폭 2 칸 이상 (위 2.5), 1 = 제한 없음.
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
        r0, c0 = d
        for _round in range(20 if min_passage_width >= 2 else 1):
            while grid.mean() < target and guard < 2000:
                guard += 1
                h = int(rng.randint(bmin, bmax + 1))
                w = int(rng.randint(bmin, bmax + 1))
                r = int(rng.randint(0, max(1, rows - h + 1)))
                c = int(rng.randint(0, max(1, cols - w + 1)))
                grid[r:r + h, c:c + w] = True
            # depot 주변 비우기
            grid[max(0, r0 - clear_radius):r0 + clear_radius + 1,
                 max(0, c0 - clear_radius):c0 + clear_radius + 1] = False
            if min_passage_width < 2:
                break
            widen_passages(grid, rng)                         # 1 칸 통로 / 대각선 끼임 -> 넓힘 (비율이 낮아질 수 있음)
            if grid.mean() >= lo - 1e-9:
                break
        # 갇힌 빈 칸 메우기
        reach = reachable_from(d, grid)
        grid |= ~reach
        ratio = grid.mean()
        if min_passage_width >= 2:                            # 넓히기 실패 / 메운 뒤 새 위반이 있으면 다시 생성
            narrow, pinch = narrow_cells(grid)
            if narrow.any() or pinch.any():
                continue
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
