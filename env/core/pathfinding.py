"""그리드 위 BFS 유틸.

- bfs_field: UAV 당 tick 마다 1 회 (자기 지도 기준). 모든 셀까지의 거리 + parent (경로 역추적)
    -> 노드 특징 거리, 할당 가능 판정 (d <= τ - t), 결정 순서, 목표 셀, 이동 (BFS 다음 칸)
  균일 비용 8-연결이므로 BFS 거리 = A* 최단거리 (대각선도 1 step).
- path_to: parent 로 경로 복원 / window_waypoint: 경로가 local 창을 벗어나기 직전 셀 (GUI 표시)

8-연결 기준 (Robot.ACTIONS 와 동일). blocked[r][c] == True 인 칸은 통과 불가,
대각선은 양 옆 두 칸이 모두 blocked 면 불가 (Robot.can_step 의 코너 컷팅 규칙).
미탐색 영역을 어떻게 다룰지는 호출부에서 blocked 마스크를 만들 때 결정한다
(보통 '알려진 장애물만' 막고 미탐색은 낙관적으로 통과 가능으로 둔다).
"""
from collections import deque

import numpy as np

_NEIGHBORS_8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def bfs_field(start, blocked, rows, cols):
    """start 에서 모든 셀까지의 8-연결 최단 거리와 parent.

    blocked: (rows, cols) bool 배열 (또는 list[list[bool]]) — True 는 통과 불가.
    반환: dist (rows, cols) int32, 도달 불가 = -1 / parent (rows*cols,) int32 (flat index, start/미도달 = -1)
    blocked 셀은 도달 불가 (목적지 후보는 모두 알려진 free 셀).
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
