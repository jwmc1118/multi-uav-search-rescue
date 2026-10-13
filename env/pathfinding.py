"""그리드 위 BFS 유틸.

용도:
- bfs_field: 에이전트당 tick 마다 1회. 모든 셀까지의 거리 d + parent (경로 역추적)
    -> 할당 노드 특징 d^i_j / d^i_e / d^i_d, admissible 판정(d <= tau), 디코딩 우선순위,
       최근접 frontier, primitive waypoint (경로가 local 창을 벗어나기 직전 셀)
  균일 비용 8-연결이므로 BFS 거리 = A* 최단거리 (대각선도 1 step).
- shortest_path_length / next_step_towards: 보상 계산, stand-in 정책

8-연결 기준 (Robot.ACTIONS 와 동일). blocked[r][c] == True 인 칸은 통과 불가,
대각선은 양 옆 두 칸이 모두 blocked 면 불가 (Robot.can_step 의 코너 컷팅 규칙).
미탐색 영역을 어떻게 다룰지는 호출부에서 blocked 마스크를 만들 때 결정한다
(보통 '알려진 장애물만' 막고 미탐색은 낙관적으로 통과 가능으로 둔다).
"""
from collections import deque

import numpy as np

_NEIGHBORS_8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols


def _corner_cut(r, c, dr, dc, blocked):
    """(r, c) 에서 대각선 (dr, dc) 이동이 두 장애물 사이를 비집고 지나가는지."""
    return bool(dr and dc and blocked[r + dr][c] and blocked[r][c + dc])


def grid_distance(a, b):
    """장애물을 무시한 8방향 이동 칸 수 (Chebyshev). 최근접/우선순위 비교용."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def shortest_path_length(start, goal, blocked, rows, cols):
    """start->goal 최단 거리. 도달 불가면 None. goal이 blocked여도 목적지로는 허용."""
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        for dr, dc in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc, rows, cols) or (nr, nc) in seen:
                continue
            if _corner_cut(r, c, dr, dc, blocked):
                continue
            if (nr, nc) == goal:
                return d + 1
            if blocked[nr][nc]:
                continue
            seen.add((nr, nc))
            q.append(((nr, nc), d + 1))
    return None


def next_step_towards(start, goal, blocked, rows, cols):
    """start에서 goal로 가는 최단 경로의 첫 (dr, dc). 경로 없으면 None."""
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return (0, 0)
    # goal에서 역방향 BFS -> start 인접칸 중 가장 가까운 방향 선택
    dist = {goal: 0}
    q = deque([goal])
    while q:
        r, c = q.popleft()
        if (r, c) == start:
            break
        for dr, dc in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc, rows, cols) or (nr, nc) in dist:
                continue
            if _corner_cut(r, c, dr, dc, blocked):   # 대각선은 대칭이라 역방향에서도 같은 판정
                continue
            if blocked[nr][nc] and (nr, nc) != start:
                continue
            dist[(nr, nc)] = dist[(r, c)] + 1
            q.append((nr, nc))
    best = None
    best_d = None
    for dr, dc in _NEIGHBORS_8:
        nr, nc = start[0] + dr, start[1] + dc
        if _in_bounds(nr, nc, rows, cols) and _corner_cut(start[0], start[1], dr, dc, blocked):
            continue
        if (nr, nc) in dist and (best_d is None or dist[(nr, nc)] < best_d):
            best_d = dist[(nr, nc)]
            best = (dr, dc)
    return best


def bfs_field(start, blocked, rows, cols):
    """start 에서 모든 셀까지의 8-연결 최단 거리와 parent.

    blocked: (rows, cols) bool 배열 (또는 list[list[bool]]) — True 는 통과 불가.
    반환: dist (rows, cols) int32, 도달 불가 = -1 / parent (rows*cols,) int32 (flat index, start/미도달 = -1)
    goal 이 blocked 여도 목적지로는 허용하는 shortest_path_length 와 달리, 여기서는 blocked 셀은
    도달 불가로 둔다 (목적지 후보는 모두 알려진 free 셀이므로 차이 없음).
    """
    blocked = np.asarray(blocked, dtype=bool)
    n = rows * cols
    # 파이썬 list 로 돈다: numpy 원소 하나씩 접근하는 것보다 ~6배 빠르다 (결과는 동일, BFS 순서 그대로)
    bl = blocked.ravel().tolist()
    dist = [-1] * n
    parent = [-1] * n
    sr, sc = start
    s0 = sr * cols + sc
    dist[s0] = 0
    q = deque([s0])
    pop, push = q.popleft, q.append
    while q:
        k = pop()
        r = k // cols
        c = k - r * cols
        dk = dist[k] + 1
        for dr, dc in _NEIGHBORS_8:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            nk = nr * cols + nc
            if dist[nk] >= 0 or bl[nk]:
                continue
            if dr and dc and bl[nr * cols + c] and bl[r * cols + nc]:
                continue                                  # 코너 컷팅 금지
            dist[nk] = dk
            parent[nk] = k
            push(nk)
    return np.asarray(dist, np.int32).reshape(rows, cols), np.asarray(parent, np.int32)


def path_to(parent, start, goal, cols):
    """bfs_field 의 parent 로 start -> goal 경로 [(r, c), ...] (start 포함). 도달 불가면 None."""
    s0 = start[0] * cols + start[1]
    k = goal[0] * cols + goal[1]
    out = []
    while k != s0:
        if k < 0:
            return None
        out.append(divmod(int(k), cols))
        k = parent[k]
    out.append(tuple(start))
    out.reverse()
    return out


def window_waypoint(path, center, half):
    """경로를 따라가다 center 중심 (2*half+1) 정사각 창을 벗어나기 직전 셀 (Two-stage 논문의 waypoint).
    목적지가 창 안이면 목적지 자체."""
    wp = path[0]
    for cell in path:
        if max(abs(cell[0] - center[0]), abs(cell[1] - center[1])) > half:
            break
        wp = cell
    return wp
