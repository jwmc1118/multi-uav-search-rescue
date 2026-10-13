"""UAV 정의 (고정 고도 2D 격자).

각 로봇은 위치/센서 범위/통신 반경/방문 기록과 함께
할당 상태(구호품 용량 capacity, mode, 배정 타겟, lock 여부)를 들고 있다. heading 은 없다.
- 센서: 전방향(360도) LiDAR 스타일 레이캐스팅 — 장애물 뒤 셀은 가려진다.
- 이동: 8방향 한 칸 이동 + stay. 대각선은 양 옆 두 칸이 모두 장애물이면 막힌다 (센서 코너 컷팅 규칙과 동일).
sense() / move() 는 GUI(gui/grid_map.py)와 학습(env/simulation.py) 양쪽에서 공유한다.
"""
import colorsys

from env.modes import Mode, MSG_EXPLORE


def supercover_line(r0, c0, r1, c1):
    """(r0, c0)에서 (r1, c1)까지 실제 기하학적 직선이 지나가는 모든 셀을 순서대로 반환한다.

    일반 브레젠험은 장애물 모서리를 스치는 광선의 차폐 판정이 타겟마다 달라지는
    불일치가 생긴다. supercover 는 직선이 격자점을 정확히 지날 때 양쪽 칸을 모두
    포함시켜 항상 같은 차폐 판정이 나오게 한다.
    """
    dr = r1 - r0
    dc = c1 - c0
    nr, nc = abs(dr), abs(dc)
    sign_r = 1 if dr > 0 else -1
    sign_c = 1 if dc > 0 else -1

    r, c = r0, c0
    points = [(r, c)]
    ir = ic = 0
    while ir < nr or ic < nc:
        lhs = (1 + 2 * ic) * nr
        rhs = (1 + 2 * ir) * nc
        if lhs < rhs:
            c += sign_c
            ic += 1
        elif lhs > rhs:
            r += sign_r
            ir += 1
        else:
            # 격자 모서리(대각선 교차점)를 정확히 지남 -> 대각선으로 같이 이동
            r += sign_r
            c += sign_c
            ir += 1
            ic += 1
        points.append((r, c))
    return points


def _shade_color(hue, index, saturation_range=(0.55, 0.95), value_range=(0.65, 1.0)):
    """같은 색 계열(hue) 안에서 index 마다 채도/명도를 황금비 간격으로 바꾼 shade."""
    frac = (index * 0.61803398875) % 1.0
    saturation = saturation_range[0] + frac * (saturation_range[1] - saturation_range[0])
    value = value_range[1] - frac * (value_range[1] - value_range[0])
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (round(r * 255), round(g * 255), round(b * 255))


class Robot:
    """UAV.

    - pos: (row, col)
    - sensor_range / comm_range (셀 단위)
    - visited: 지나간 셀 집합 (GUI 트레일)
    """

    # 8방향(월드 방위, 북쪽 = 위 = -row 부터 시계방향) + stay. ACTIONS[k] 의 이동량 = MOVES[k]
    ACTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "stay"]
    MOVES = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0)]
    _color_hue = 0.0  # 빨간 계열
    _count = 0

    def __init__(self, start_pos, sensor_range, color=None, comm_range=4, capacity=3):
        self.start_pos = tuple(start_pos)
        self.pos = tuple(start_pos)
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.color = color if color is not None else self._auto_color()
        self.visited = {self.pos}

        # ---- 할당 상태 (통합 할당 정책의 결과) ----
        self.capacity_max = capacity
        self.capacity = capacity              # 잔여 구호품 c^i
        self._reset_alloc_state()
        self.alive = True

        # env/knowledge.py 가 sense/comm 처리 후 채운다
        self.local_window = None            # 자기 중심 고해상도 크롭 (S,S,C), 북쪽 = 위쪽
        self.global_window = None           # allocentric 저해상도 개별 누적 맵 (GUI 표시용)

    def _reset_alloc_state(self):
        self.mode = Mode.EXPLORE             # 이번 tick 고른 노드의 종류
        self.assigned_target = None          # RESCUE 일 때 Target index
        self.locked = False                  # True 면 정책 대신 환경이 같은 노드를 강제 (rescue / return 유지)
        self.goal = None                     # 현재 primitive 목적지 (row, col)
        self.msg_goal = (0.0, 0.0)           # 이웃에게 보내는 g^{i'} (정규화 월드 좌표)
        self.msg_type = MSG_EXPLORE          # 이웃에게 보내는 o^{i'} 인덱스 (env/modes.py)

    def reset_state(self, start_pos=None):
        """에피소드 리셋: 위치/방문기록/할당 상태 초기화."""
        if start_pos is not None:
            self.start_pos = tuple(start_pos)
        self.pos = tuple(self.start_pos)
        self.visited = {self.pos}
        self.capacity = self.capacity_max
        self._reset_alloc_state()
        self.alive = True
        self.local_window = None
        self.global_window = None

    def refill(self):
        self.capacity = self.capacity_max

    @classmethod
    def _auto_color(cls):
        index = cls._count
        cls._count += 1
        return _shade_color(cls._color_hue, index)

    # ---------- 이동 ----------

    def can_traverse(self, cell, obstacles, rows, cols):
        r, c = cell
        return 0 <= r < rows and 0 <= c < cols and not obstacles[r][c]

    def can_step(self, delta, obstacles, rows, cols):
        """현재 위치에서 delta=(dr, dc) 한 칸 이동이 가능한지. 대각선은 코너 컷팅 금지."""
        dr, dc = delta
        r, c = self.pos
        if not self.can_traverse((r + dr, c + dc), obstacles, rows, cols):
            return False
        if dr and dc and obstacles[r + dr][c] and obstacles[r][c + dc]:
            return False
        return True

    def valid_actions(self, obstacles, rows, cols):
        return [a for a, d in zip(self.ACTIONS, self.MOVES)
                if d == (0, 0) or self.can_step(d, obstacles, rows, cols)]

    def move(self, action, obstacles, rows, cols):
        if action not in self.ACTIONS:
            return False
        delta = self.MOVES[self.ACTIONS.index(action)]
        if delta == (0, 0):
            return True
        if not self.can_step(delta, obstacles, rows, cols):
            return False
        self.pos = (self.pos[0] + delta[0], self.pos[1] + delta[1])
        self.visited.add(self.pos)
        return True

    # ---------- 센싱 ----------

    def sense(self, obstacles, rows, cols):
        """보이는 셀 집합 (전방향). 정사각형 범위 안 각 셀로 레이를 쏴 가로막힘 없이 도달한 셀만 보이는 것으로 표시.
        경유 칸은 그 칸이 타겟인 반복에서 따로 검사하고(빛샘 방지), 대각선으로 인접한
        두 장애물 사이 틈으로 새는 코너 컷팅도 막는다."""
        visible = {self.pos}
        r0, c0 = self.pos

        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):
                if dr == 0 and dc == 0:
                    continue
                r1, c1 = r0 + dr, c0 + dc
                if not (0 <= r1 < rows and 0 <= c1 < cols):
                    continue

                prev_r, prev_c = r0, c0
                for r, c in supercover_line(r0, c0, r1, c1)[1:]:
                    if r != prev_r and c != prev_c:
                        if obstacles[prev_r][c] and obstacles[r][prev_c]:
                            break  # 코너 컷팅 차단
                    if (r, c) == (r1, c1):
                        visible.add((r, c))
                        break
                    if obstacles[r][c]:
                        break
                    prev_r, prev_c = r, c

        return visible
