"""에이전트별 사적 지식 + comm_range 게이팅된 통신 병합 + 관측 텐서 생성.

관측 채널 (docs/research_pipeline_summary.md §3, Role-selection 논문 §V-A-2):
    0 obstacle : 알려진 장애물 = 1 (미탐색은 0 으로 마스킹, local 크롭의 맵 밖 = 1)
    1 explored : 알려진 셀(free|obstacle) = 1, 단조증가
    2 frontier : 알려진 free 이면서 상하좌우에 미탐색 칸이 있는 셀 = 1
    3 target   : 발견됐고 아직 active 한 타겟 = 1
    4 agents   : 자기 위치 = 1.0, 통신 범위 안 이웃 위치 = 0.5

멀티해상도 (§3.2):
    local  : 고해상도 S x S 크롭 — 자기 위치가 중앙, 회전 없이 월드 방위 그대로 (북쪽 = 위).
             로봇에 heading 이 없고 primitive 행동도 월드 8방향이라 회전이 필요 없다.
    global : 저해상도 allocentric(고정 프레임) 개별 누적 맵 — GUI 표시용 (정책 입력 아님).
             채널별 pooling (explored 는 평균 = 블록 탐색 비율, 나머지는 max).
             critic 의 중앙집중 state 도 같은 pooling 을 쓴다 (simulation.centralized_state).

RescueSimulation 과 GUI 가 이 클래스 하나를 공유한다. 구조/발견/마감 '규칙'은
env/simulation.py 가 담당한다.
"""
import numpy as np

from env.coords import WorldFrame

CH_OBSTACLE = 0
CH_EXPLORED = 1
CH_FRONTIER = 2
CH_TARGET = 3
CH_AGENTS = 4
N_CHANNELS = 5
# 채널 이름 -> 인덱스. primitive 입력 채널 선택(cfg["local_channels"])에 쓴다.
CHANNEL_INDEX = {"obstacle": CH_OBSTACLE, "explored": CH_EXPLORED, "frontier": CH_FRONTIER,
                 "target": CH_TARGET, "agents": CH_AGENTS}

AGENT_SELF = 1.0
AGENT_NEIGHBOR = 0.5

# local 크롭 패딩값 (맵 밖): 장애물로 보이게
_LOCAL_FILL = {CH_OBSTACLE: 1.0}
# global pooling 방식
_POOL_MEAN = {CH_EXPLORED}


def frontier_map(known_free, known_obstacle):
    """known_free 이면서 상하좌우(맵 안) 중 하나라도 미탐색인 셀 = True. (rows, cols) bool."""
    unexplored = ~(known_free | known_obstacle)
    p = np.pad(unexplored, 1, mode="constant", constant_values=False)
    near_unknown = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return known_free & near_unknown


def _pool(grid, factor):
    """(H, W, C) -> (ceil(H/f), ceil(W/f), C). explored 채널은 평균, 나머지는 max."""
    if factor <= 1:
        return grid
    h, w, c = grid.shape
    ph, pw = (-h) % factor, (-w) % factor
    if ph or pw:
        grid = np.pad(grid, ((0, ph), (0, pw), (0, 0)), mode="constant")
    H, W = grid.shape[0] // factor, grid.shape[1] // factor
    blocks = grid.reshape(H, factor, W, factor, c)
    out = blocks.max(axis=(1, 3))
    for ch in _POOL_MEAN:
        out[..., ch] = blocks[..., ch].mean(axis=(1, 3))
    return out


def rel_vector(robot, cell, scale):
    """로봇 -> cell 상대 벡터 (dx, dy) / scale. 축은 월드 좌표와 같다 (x 오른쪽 +, y 위쪽 +)."""
    return (cell[1] - robot.pos[1]) / scale, (robot.pos[0] - cell[0]) / scale


class KnowledgeTracker:
    def __init__(self, agents, obstacles, rows, cols, state_size=21,
                 global_downsample=4, n_hop=1, frame=None):
        self.agents = agents
        self.frame = frame or WorldFrame(rows, cols)
        self.obstacles = obstacles
        self.rows = rows
        self.cols = cols
        self.local_size = state_size          # 고해상도 local 크롭 크기 (홀수)
        self.global_downsample = max(1, int(global_downsample))
        self.n_hop = max(1, int(n_hop))
        self.reset()

    def reset(self):
        n = len(self.agents)
        self.known_free = [np.zeros((self.rows, self.cols), dtype=bool) for _ in self.agents]
        self.known_obstacle = [np.zeros((self.rows, self.cols), dtype=bool) for _ in self.agents]
        self.frontier = [np.zeros((self.rows, self.cols), dtype=bool) for _ in self.agents]
        self.adjacency = np.zeros((n, n), dtype=bool)   # 직접 통신 가능
        self.newly_sensed = [0] * n
        self.last_visible = [set() for _ in range(n)]
        self.reach = np.zeros((n, n), dtype=bool)       # n_hop 이내 연결 (지식 공유 / 이웃 context / 디코딩 그룹)

    # ---------- Communication ----------

    def compute_adjacency(self):
        """Chebyshev 거리 기반 '실제' 통신 인접 행렬. 두 에이전트의 comm_range 중 짧은 쪽
        안에 있을 때만 연결. 정보 공유/이웃 관측은 이 raw 행렬만 쓴다."""
        n = len(self.agents)
        dists = np.zeros((n, n))
        min_ranges = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                d = max(abs(self.agents[i].pos[0] - self.agents[j].pos[0]),
                        abs(self.agents[i].pos[1] - self.agents[j].pos[1]))
                dists[i, j] = dists[j, i] = d
                r = min(self.agents[i].comm_range, self.agents[j].comm_range)
                min_ranges[i, j] = min_ranges[j, i] = r
        adjacency = dists < min_ranges
        np.fill_diagonal(adjacency, False)
        return adjacency

    def nhop_matrix(self, adjacency):
        """n_hop 이내 도달 가능 여부 (자기 자신 제외)."""
        reach = adjacency.copy()
        hop = adjacency.copy()
        for _ in range(self.n_hop - 1):
            hop = (hop.astype(int) @ adjacency.astype(int)) > 0
            reach |= hop
        np.fill_diagonal(reach, False)
        return reach

    def _merge_knowledge(self):
        """n_hop 이내로 연결된 에이전트들의 누적 맵(free/obstacle)을 같은 스텝에 합친다.
        타겟 채널은 known_free 기준이라 합쳐진 맵에 들어온 생존자도 함께 공유된다."""
        free_snapshot = [f.copy() for f in self.known_free]
        obstacle_snapshot = [o.copy() for o in self.known_obstacle]
        for i in range(len(self.agents)):
            for j in range(len(self.agents)):
                if self.reach[i, j]:
                    self.known_free[i] |= free_snapshot[j]
                    self.known_obstacle[i] |= obstacle_snapshot[j]

    # ---------- Sensing ----------

    def _sense(self, agent, index):
        """센서로 본 셀을 개인 지식에 기록하고, 본 셀 집합을 반환."""
        visible = agent.sense(self.obstacles, self.rows, self.cols)
        for r, c in visible:
            if self.obstacles[r][c]:
                self.known_obstacle[index][r, c] = True
            else:
                self.known_free[index][r, c] = True
        return visible

    # ---------- Observation ----------

    def _local_crop(self, grid_full, robot):
        """(rows, cols, C) -> (S, S, C). 자기 위치 중앙, 회전 없음. 맵 밖은 _LOCAL_FILL."""
        half = self.local_size // 2
        r, c = robot.pos
        chans = []
        for ch in range(grid_full.shape[-1]):
            padded = np.pad(grid_full[:, :, ch], half, mode="constant",
                            constant_values=_LOCAL_FILL.get(ch, 0.0))
            chans.append(padded[r:r + self.local_size, c:c + self.local_size])
        return np.stack(chans, axis=-1)

    def _build_observation(self, active_target_positions):
        for i, agent in enumerate(self.agents):
            free, obst = self.known_free[i], self.known_obstacle[i]
            self.frontier[i] = frontier_map(free, obst)

            grid = np.zeros((self.rows, self.cols, N_CHANNELS), dtype=np.float32)
            grid[..., CH_OBSTACLE] = obst
            grid[..., CH_EXPLORED] = free | obst
            grid[..., CH_FRONTIER] = self.frontier[i]
            for (tr, tc) in active_target_positions:
                if free[tr, tc]:
                    grid[tr, tc, CH_TARGET] = 1.0
            for j, other in enumerate(self.agents):
                if self.adjacency[i, j]:
                    grid[other.pos[0], other.pos[1], CH_AGENTS] = AGENT_NEIGHBOR
            grid[agent.pos[0], agent.pos[1], CH_AGENTS] = AGENT_SELF

            agent.local_window = self._local_crop(grid, agent)
            agent.global_window = _pool(grid, self.global_downsample)

    def update(self, active_target_positions=()):
        """센싱 -> comm_range 게이팅 병합 -> 관측 텐서 갱신. 에이전트 이동 직후 매번 호출.
        newly_sensed[i] = 이번 센싱에서 UAV i 가 직접 본 셀 중 직전까지 팀 전체가 몰랐던 셀 수 (|E^i_t|).
        같은 새 셀을 k 대가 동시에 보면 각자 1/k 씩 -> 팀 합 = 팀이 새로 알게 된 셀 수 (중복 계산 없음)."""
        team_known = np.zeros((self.rows, self.cols), dtype=bool)
        for f, o in zip(self.known_free, self.known_obstacle):
            team_known |= f | o
        new_sets = []
        self.last_visible = []
        for i, agent in enumerate(self.agents):
            visible = self._sense(agent, i)
            self.last_visible.append(visible)
            new_sets.append({cell for cell in visible if not team_known[cell]})
        seen_by = {}
        for cells in new_sets:
            for cell in cells:
                seen_by[cell] = seen_by.get(cell, 0) + 1
        self.newly_sensed = [sum(1.0 / seen_by[cell] for cell in cells) for cells in new_sets]
        self.adjacency = self.compute_adjacency()
        self.reach = self.nhop_matrix(self.adjacency)
        self._merge_knowledge()
        self._build_observation(list(active_target_positions))

    # ---------- Helpers ----------

    def team_known_free(self):
        """어느 에이전트든 알고 있는 free 셀 (union). 발견 판정용."""
        out = np.zeros((self.rows, self.cols), dtype=bool)
        for f in self.known_free:
            out |= f
        return out

    def blocked_mask(self, agent_index):
        """agent_index 시점의 '알려진 장애물' 마스크 (rows, cols) bool. 미탐색은 통과 가능(낙관적)."""
        return self.known_obstacle[agent_index]
