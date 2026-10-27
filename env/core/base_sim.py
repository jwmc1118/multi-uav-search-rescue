"""BaseSimulation — 탐색/구조 환경의 공통 기반 (UAV, 고정 고도 2D 격자). env/role/simulation.py RoleSimulation 이 상속한다.

여기서 담당하는 것 (결정 구조와 무관한 메커니즘):
    맵 (파일 / 랜덤 생성, GUI 편집) / depot / 조난자 배치 · 골든타임 / UAV 생성 · 초기 위치
    센싱 · 통신 병합 (env/core/knowledge.py) / 믿음 (env/core/beliefs.py)
    발견 · 구조 · 재보충 · 마감 처리 / 에피소드 종료 / info / critic 용 중앙집중 상태 · privileged 관측
결정 · 이동 · 보상 · actor 관측은 하위 클래스 (RoleSimulation) 가 정한다. 하위 클래스가 덮는 훅:
    _prepare()  : 위치 · 지식 · 시간이 바뀐 직후 다음 결정 준비 (reset / GUI 조난자 편집 때 호출)
    _dist(i, c) : UAV i 에서 셀 c 까지 BFS 거리 (도달 불가 10**9) — critic 관측의 최근접 UAV 거리
"""
import copy
import json
import os
from datetime import datetime

import numpy as np

from env.core.agent import Robot
from gui.colors import agent_rgb
from env.core.coords import WorldFrame
from env.core.knowledge import KnowledgeTracker, _pool, N_CHANNELS, CHANNEL_INDEX, CH_OBSTACLE, \
    CH_EXPLORED, CH_FRONTIER, CH_TARGET, CH_AGENTS, frontier_map
from env.core.beliefs import BeliefTracker
from env.core.modes import Mode
from env.core.targets import Target, deadline
from env.core.map_gen import generate_obstacles, sample_target_cells, sample_count
from env.core.step_context import RescueEvent

DEFAULT_OPTIONS = {
    "map_mode": "file",        # 'file' : map_path 의 맵 (None 이면 빈 그리드)
                               # 'random': 매 에피소드(reset) 장애물 맵을 새로 생성 (env/core/map_gen.py, rows x cols)
    "map_path": None,          # maps/ 아래 JSON 경로. None -> 빈 그리드
    "rows": 40,                # map_path 없을 때 / random 맵 크기
    "cols": 40,
    "obstacle_ratio": [0.2, 0.3],   # random 맵 장애물 비율 범위 (depot 에서 갈 수 없는 칸은 장애물로 메운 뒤 기준)
    "obstacle_block": None,    # random 맵 건물(직사각형) 한 변 길이 [min, max]. None -> [2, max(rows, cols) // 6]
    "min_passage_width": 2,    # random 맵 통로 최소 폭: 2 = 1 칸 틈 / 대각선 끼임 없음 (env/core/map_gen.py), 1 = 제한 없음
    "target_mode": "file",     # 'file'  : 맵 파일의 조난자 (없으면 num_targets 개를 처음 한 번만 랜덤 배치)
                               # 'random': 매 에피소드 조난자 수/위치를 새로 뽑음 (depot 에서 도달 가능한 칸만)
                               # map_mode='random' 이면 조난자는 항상 random
    "state_size": 21,          # 고해상도 local 관측 윈도우 (자기 중심, 홀수) — critic local 입력
    "local_channels": ["obstacle", "agents"],   # local 관측 채널 (critic 입력)
                               # 선택지: obstacle / explored / frontier / target / agents (env/core/knowledge.py)
    "global_downsample": 4,    # critic 중앙집중 state / GUI global 맵 축소 배율
    "n_hop": 1,                # 통신 hop 수 (지식 병합 / 통신 그룹 / 순차 디코딩 그룹)
    "max_episode_len": 300,    # t_max — 시간/거리 특징 정규화 분모
    "position_mode": "depot",  # 'depot' (모두 depot 에서 출발) / 'random'
    "capacity": 3,             # 로봇 1대의 구호품 용량
    "depot": None,             # 재보충 지점 = 모든 로봇의 출발 지점. 월드 좌표 (x, y) (env/core/coords.py)
                               # None -> 맵 파일의 depot -> 맵 파일의 start -> 원점 (0, 0)
    "target_golden_time": [80, 160],  # 타겟별 골든타임 tau_j. int 또는 [min, max]
    "deadline_mode": "from_discovery",  # 'from_discovery' / 'from_start' (§9 open item 3)
    "num_targets": 10,         # 랜덤 배치 조난자 수. int 또는 [min, max] (예: [10, 20])
    "max_targets": 16,         # T: 조난자 슬롯 수 (관측 / rescue 행동 패딩 상한)
    "agents": [
        {"sensor_range": 3, "comm_range": 6},
        {"sensor_range": 3, "comm_range": 6},
        {"sensor_range": 3, "comm_range": 6},
    ],
}

def load_map(map_path, rows, cols):
    """gui/grid_map.py 저장 스키마를 읽는다. map_path 가 None 이면 빈 그리드.

    반환: obstacles(list[list[bool]]), start, target_specs, depot, rows, cols
    파일의 좌표는 내부 (row, col). start/depot 은 없으면 None (구버전 파일의 "start" 는 depot 대체값).
    target_specs: [((r, c), golden_or_None), ...]
    """
    if map_path is None:
        obstacles = [[False] * cols for _ in range(rows)]
        return obstacles, None, [], None, rows, cols

    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows, cols = data["rows"], data["cols"]
    obstacles = [[False] * cols for _ in range(rows)]
    for r, c in data.get("obstacles", []):
        obstacles[r][c] = True

    start = tuple(data["start"]) if "start" in data else None
    depot = tuple(data["depot"]) if "depot" in data else None
    target_specs = [((e[0], e[1]), e[2] if len(e) >= 3 else None) for e in data.get("targets", [])]
    return obstacles, start, target_specs, depot, rows, cols


class BaseSimulation:
    def __init__(self, options=None):
        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(options or {})
        self.random_state = np.random.RandomState(self.cfg.get("seed"))

        self.keep_map = False      # True 면 reset() 이 random 맵/조난자를 다시 뽑지 않음 (GUI Reset = 같은 상황 재생)
        if self.cfg["map_mode"] not in ("file", "random"):
            raise ValueError(f"map_mode={self.cfg['map_mode']!r} — 'file' / 'random'")
        if self.cfg["map_mode"] == "random":
            self.rows, self.cols = int(self.cfg["rows"]), int(self.cfg["cols"])
            self.obstacles = [[False] * self.cols for _ in range(self.rows)]
            self.frame = WorldFrame(self.rows, self.cols)
            self._target_specs = []
            self._generate_map()                                       # 장애물 + depot + 조난자
        else:
            (self.obstacles, file_start, self._target_specs, file_depot,
             self.rows, self.cols) = load_map(self.cfg["map_path"], self.cfg["rows"], self.cfg["cols"])
            self.frame = WorldFrame(self.rows, self.cols)              # (row, col) <-> 월드 (x, y)
            self.depot = self._resolve_depot(file_depot, file_start)   # 내부 (row, col)
            # 맵 파일에 조난자가 없거나 target_mode='random' 이면 랜덤 배치 (map_path 가 None 이면 빈 환경 — GUI 에서 직접 배치)
            if self.cfg["map_path"] and self.cfg["num_targets"] and \
                    (not self._target_specs or self.cfg["target_mode"] == "random"):
                self._randomize_target_specs()

        self.agents = [
            Robot(start_pos=self.depot,
                  sensor_range=spec["sensor_range"],
                  comm_range=spec["comm_range"],
                  capacity=self.cfg["capacity"],
                  color=agent_rgb(k))                    # UAV 색은 gui/colors.py 한 곳에서 (GUI / 그래프 공통)
            for k, spec in enumerate(self.cfg["agents"])
        ]
        self.beliefs = BeliefTracker(len(self.agents))   # UAV 별 구조 완료 믿음 (env/core/beliefs.py)
        self.knowledge = KnowledgeTracker(self.agents, self.obstacles, self.rows, self.cols,
                                          state_size=self.cfg["state_size"], n_hop=self.cfg["n_hop"])
        self.knowledge.hysteresis = int((self.cfg.get("events") or {}).get("comm_hysteresis", 0))   # 통신 링크 해제 여유
        bad = [c for c in self.cfg["local_channels"] if c not in CHANNEL_INDEX]
        if bad or not self.cfg["local_channels"]:
            raise ValueError(f"local_channels {bad or '[]'} — 선택지: {list(CHANNEL_INDEX)}")
        self.local_ch_idx = [CHANNEL_INDEX[c] for c in self.cfg["local_channels"]]

        self.max_targets = int(self.cfg["max_targets"])
        self.recorder = None       # gui_evaluation.alloc_recorder.AllocRecorder (있으면 reset / tick 마다 기록)

        self.n_agents = len(self.agents)
        ds = self.cfg["global_downsample"]
        self.gh = -(-self.rows // ds)
        self.gw = -(-self.cols // ds)

        self.targets = []
        self.time = 0
        self.reset()

    # ---------- setup helpers ----------

    def _resolve_depot(self, file_depot, file_start, obstacles=None):
        """depot (= 모든 로봇의 출발 지점) 결정. 반환은 내부 (row, col).
        우선순위: cfg["depot"] (월드 x, y — main/yaml) > 맵 파일 depot > 맵 파일 start > 원점."""
        if self.cfg["depot"] is not None:
            xy = tuple(int(v) for v in self.cfg["depot"])
            cell = self.frame.to_cell(xy)
        else:
            cell = tuple(file_depot or file_start or self.frame.to_cell((0, 0)))
            xy = self.frame.to_xy(cell)
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            x0, x1, y0, y1 = self.frame.extent()
            raise ValueError(f"depot (x, y)={xy} 가 맵 범위 밖 (x {x0}~{x1}, y {y0}~{y1})")
        if (obstacles or self.obstacles)[r][c]:
            raise ValueError(f"depot (x, y)={xy} 가 장애물 칸")
        return cell

    def _random_free_cells(self, n, exclude=()):
        exclude = set(exclude)
        free = [(r, c) for r in range(self.rows) for c in range(self.cols)
                if not self.obstacles[r][c] and (r, c) not in exclude]
        if not free:
            return []
        idx = self.random_state.choice(len(free), size=min(n, len(free)), replace=False)
        return [free[i] for i in idx]

    def _sample_golden(self, given):
        if given is not None:
            return int(given)
        g = self.cfg["target_golden_time"]
        if isinstance(g, (list, tuple)):
            return int(self.random_state.randint(g[0], g[1] + 1))
        return int(g)

    def _sample_start_positions(self):
        mode = self.cfg["position_mode"]
        if mode == "depot":
            return [self.depot for _ in self.agents]
        if mode == "random":
            occupied, out = set(), []
            for _ in self.agents:
                cell = self._random_free_cells(1, exclude=occupied)
                pos = cell[0] if cell else self.depot
                occupied.add(pos)
                out.append(pos)
            return out
        raise ValueError(f"Unknown position_mode: {mode!r}")

    # ---------- map editing (GUI) ----------

    def load_map_file(self, path):
        """맵 JSON 을 읽어 장애물/타겟/depot 를 교체하고 리셋한다.
        그리드 크기가 다르면 무시하고 False (GUI 윈도우 크기가 고정돼 있으므로)."""
        obstacles, file_start, target_specs, file_depot, rows, cols = load_map(path, self.rows, self.cols)
        if rows != self.rows or cols != self.cols:
            return False
        new_depot = self._resolve_depot(file_depot, file_start, obstacles)   # 상태를 바꾸기 전에 검증
        for r in range(self.rows):          # in-place 교체 (knowledge 가 같은 ref 를 들고 있음)
            for c in range(self.cols):
                self.obstacles[r][c] = obstacles[r][c]
        self.depot = new_depot
        self._target_specs = target_specs
        self.reset(regenerate=False)
        return True

    def save_map_file(self, directory_map, directory_target):
        """현재 맵을 JSON 으로 저장: 장애물 + depot + 조난자 (위치, 골든타임) 전부.
        조난자가 있으면 target 디렉토리, 없으면 map 디렉토리. load_map_file / map_path 로 그대로 재현된다."""
        has_targets = len(self.targets) > 0
        directory = directory_target if has_targets else directory_map
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        data = {
            "rows": self.rows,
            "cols": self.cols,
            "cell_size": self.cfg.get("cell_size", 20),
            "obstacles": [[r, c] for r in range(self.rows) for c in range(self.cols)
                          if self.obstacles[r][c]],
            "depot": list(self.depot),                # 파일은 내부 (row, col) 로 저장
            # 조난자: [row, col, golden_time] — 골든타임까지 고정해서 같은 시나리오를 재현
            "targets": [[t.pos[0], t.pos[1], t.golden_time] for t in self.targets],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    # ---------- 랜덤 맵 / 조난자 (env/core/map_gen.py) ----------

    def _randomize_target_specs(self, n=None):
        """조난자 수 (num_targets 범위) / 위치 (depot 에서 도달 가능한 칸만) 를 새로 뽑는다. 골든타임은 reset 때."""
        count = sample_count(n if n is not None else self.cfg["num_targets"], self.random_state)
        cells = sample_target_cells(self.obstacles, self.depot, self.random_state, count)
        self._target_specs = [((r, c), None) for (r, c) in cells]

    def _generate_map(self):
        """장애물 맵을 새로 생성해 in-place 로 교체 (knowledge 가 같은 리스트를 참조) + 조난자 재배치.
        depot: cfg["depot"] (월드 x, y) 가 있으면 고정, 없으면 매 맵 랜덤."""
        fixed = None
        if self.cfg["depot"] is not None:
            fixed = self.frame.to_cell(tuple(int(v) for v in self.cfg["depot"]))
        grid, depot = generate_obstacles(self.rows, self.cols, self.random_state,
                                         obstacle_ratio=self.cfg["obstacle_ratio"], depot=fixed,
                                         block_size=self.cfg["obstacle_block"],
                                         min_passage_width=self.cfg["min_passage_width"])
        for r in range(self.rows):
            row = self.obstacles[r]
            for c in range(self.cols):
                row[c] = bool(grid[r, c])
        self.depot = tuple(depot)
        self._randomize_target_specs()

    def change_map(self):
        """새 랜덤 맵 + 새 조난자로 교체하고 리셋 (GUI 'Change Map'). 이후 Reset 은 이 맵을 그대로 재생."""
        self._generate_map()
        self.reset(regenerate=False)

    def randomize_targets(self, n=None):
        """현재 맵에서 조난자만 새로 뽑고 리셋 (GUI 'Targets')."""
        self._randomize_target_specs(n)
        self.reset(regenerate=False)

    def toggle_target(self, cell):
        cell = tuple(cell)
        if self.obstacles[cell[0]][cell[1]] or cell == self.depot:
            return
        existing = [i for i, t in enumerate(self.targets) if t.pos == cell]
        if existing:
            self.targets.pop(existing[0])
        else:
            self.targets.append(Target(pos=cell, golden_time=self._sample_golden(None)))
        self._target_specs = [((t.pos[0], t.pos[1]), t.golden_time) for t in self.targets]
        self.knowledge.update(self.active_target_positions())   # 관측/발견 상태 재계산
        self._update_discovery()
        self.beliefs.update(self, self.knowledge.last_visible)
        self._prepare()
        if self.recorder is not None and self.time == 0:   # 시작 전 조난자 편집 -> 기록도 다시
            self.recorder.on_reset(self)

    # ---------- reset ----------

    def reset(self, regenerate=True):
        """에피소드 리셋. regenerate=True 이고 keep_map 이 아니면:
             map_mode='random'   -> 장애물 맵 + 조난자 새로 생성
             target_mode='random' -> 조난자 수/위치 새로 뽑음
        골든타임은 항상 새로 뽑는다."""
        if regenerate and not self.keep_map:
            if self.cfg["map_mode"] == "random":
                self._generate_map()
            elif self.cfg["target_mode"] == "random" and self.cfg["num_targets"]:
                self._randomize_target_specs()
        self.time = 0
        self.knowledge.reset()

        self.targets = [Target(pos=tuple(pos), golden_time=self._sample_golden(g))
                        for (pos, g) in self._target_specs]
        if self.cfg["deadline_mode"] == "from_start":
            for t in self.targets:
                t.end_time = t.golden_time

        for agent, pos in zip(self.agents, self._sample_start_positions()):
            agent.reset_state(pos)

        self.beliefs.reset()
        self.knowledge.update(self.active_target_positions())
        self._update_discovery()
        self.beliefs.update(self, self.knowledge.last_visible)
        self._prepare()
        if self.recorder is not None:
            self.recorder.on_reset(self)


    # ---------- queries ----------

    def active_target_positions(self):
        return [t.pos for t in self.targets if t.active]

    def frontier_cells(self, agent_index):
        """agent_index 의 개별 지식 기준 frontier 셀 목록 [(r, c), ...]."""
        return [tuple(p) for p in np.argwhere(self.knowledge.frontier[agent_index])]

    def n_free_cells(self):
        """|C_free| — 맵 전체 free 셀 수 (탐색 비율 분모)."""
        return max(1, int(np.count_nonzero(~np.asarray(self.obstacles, dtype=bool))))

    def team_explored_ratio(self):
        return float(self.knowledge.team_known_free().sum()) / self.n_free_cells()

    def _update_discovery(self):
        team_free = self.knowledge.team_known_free()
        for t in self.targets:
            if not t.discovered and t.active and team_free[t.pos[0], t.pos[1]]:
                t.discovered_at = self.time
                if self.cfg["deadline_mode"] == "from_discovery":
                    t.end_time = self.time + t.golden_time


    def _prepare(self):
        """다음 결정 준비 훅 (위치 · 지식 · 시간이 바뀐 직후). 하위 클래스가 덮는다."""

    def _dist(self, i, cell):
        """UAV i 에서 cell 까지 BFS 거리 (도달 불가 10**9). 하위 클래스가 결정용 BFS 필드로 덮는다."""
        return 10 ** 9

    def _release(self, i):
        """UAV i 의 조난자 배정 해제 (구조 완료 · 마감)."""
        a = self.agents[i]
        ti = a.assigned_target
        if ti is not None and self.targets[ti].assigned_to == i:
            self.targets[ti].assigned_to = None
        a.assigned_target = None

    # ---------- tick 공통 부분 (발견 / 구조 / 재보충 / 마감) ----------

    def _process_rescues(self):
        events = []
        for i, a in enumerate(self.agents):
            if not a.alive or a.mode is not Mode.RESCUE or a.assigned_target is None:
                continue
            ti = a.assigned_target
            t = self.targets[ti]
            if a.pos == t.pos and t.active and a.capacity > 0:
                t.rescued_at = self.time
                t.rescued_by = i
                a.capacity -= 1
                self._release(i)
                self.beliefs.record_rescue(i, ti)
                tt = None if t.discovered_at is None else self.time - t.discovered_at
                events.append(RescueEvent(target=ti, agent=i, rescued_at=self.time,
                                          discovered_at=t.discovered_at, time_taken=tt))
        return events

    def _process_refill(self, ctx=None):
        for i, a in enumerate(self.agents):
            if a.alive and a.pos == self.depot:
                if ctx is not None and a.capacity < a.capacity_max:
                    ctx.refilled.add(i)
                a.refill()

    def _process_deadlines(self):
        expired = []
        for ti, t in enumerate(self.targets):
            # 발견 여부와 상관없이 마감이 지나면 만료. from_start 는 미발견도 end_time 이 있어 만료되고,
            # from_discovery 의 미발견은 end_time 이 아직 없어 (발견 시점 기준이라) 만료되지 않는다.
            if t.active and t.end_time is not None and self.time >= t.end_time:
                t.expired_at = self.time
                for i, a in enumerate(self.agents):
                    if a.assigned_target == ti:
                        self._release(i)
                t.assigned_to = None
                expired.append(ti)
        return expired

    def _is_done(self):
        if self.time >= self.cfg["max_episode_len"]:
            return True
        return all(not t.active for t in self.targets) and len(self.targets) > 0

    def _info(self, rescued_now, expired_now):
        return {
            "time": self.time,
            "rescued_now": len(rescued_now),
            "expired_now": len(expired_now),
            "rescued_total": sum(1 for t in self.targets if t.rescued_at is not None),
            "expired_total": sum(1 for t in self.targets if t.expired_at is not None),
            "active": sum(1 for t in self.targets if t.active),
            "discovered_total": sum(1 for t in self.targets if t.discovered),
            "explored": self.team_explored_ratio(),
        }

    # ================= critic 관측 (학습 전용) =================

    def centralized_state(self):
        """CTDE critic 용 중앙집중 상태. 팀 전체 관점 5채널, 저해상도 allocentric."""
        kt = self.knowledge
        team_free = kt.team_known_free()
        team_obst = np.zeros((self.rows, self.cols), bool)
        for ko in kt.known_obstacle:
            team_obst |= ko
        full = np.zeros((self.rows, self.cols, N_CHANNELS), np.float32)
        full[..., CH_OBSTACLE] = team_obst
        full[..., CH_EXPLORED] = team_free | team_obst
        full[..., CH_FRONTIER] = frontier_map(team_free, team_obst)
        for t in self.targets:
            if t.active and t.discovered:
                full[t.pos[0], t.pos[1], CH_TARGET] = 1.0
        for a in self.agents:
            full[a.pos[0], a.pos[1], CH_AGENTS] = 1.0
        return _pool(full, self.cfg["global_downsample"])

    # critic 전용 관측 차원 (env/role/gym_env.py 의 observation space 와 models/common/critic.py 가 같은 값을 쓴다)
    CRITIC_TASK_DIM = 9      # [x, y, 남은 골든타임, discovered, active, rescued, expired, 선택 UAV 수/N, 최근접 UAV BFS 거리]
    CRITIC_TEAM_DIM = 8      # [x, y, c/c_max, 모드 one-hot(3), 목적지 x, y]
    CRITIC_GLOBAL_DIM = 5    # [미발견/N_T, 구조/N_T, 만료/N_T, 팀 탐색 비율, t/t_max]

    def critic_observation(self):
        """CTDE critic 전용 privileged 관측 (학습에만 사용, actor 는 이 키들을 읽지 않는다).
        critic_tasks        (T, 9)  env 의 실제 조난자 전부 (미발견 포함 ground truth). 슬롯 = sim.targets 순서
                                     (조난자가 T 명보다 많으면 앞 T 명만)
            남은 골든타임 = max(0, τ_j - t) / t_max — τ_j = env.core.targets.deadline
            최근접 UAV BFS 거리 = 각 UAV 의 _dist (자기 지식 기준 BFS) 중 최소 / t_max, 도달 불가 1
        critic_task_present (T,)    실제 조난자 슬롯 1, 패딩 0
        critic_team         (N, 8)  UAV 별 [x, y, c/c_max, 모드 one-hot (Mode.value 순), 현재 목적지 x, y (없으면 0)]
        critic_global       (5,)    [미발견 수/N_T, 구조 수/N_T, 만료 수/N_T, 팀 탐색 비율, t/t_max]"""
        T, N = self.max_targets, self.n_agents
        t_max = float(self.cfg["max_episode_len"])
        n_t = max(1, len(self.targets))
        tasks = np.zeros((T, self.CRITIC_TASK_DIM), np.float32)
        present = np.zeros((T,), np.float32)
        chosen = {}
        for a in self.agents:
            if a.mode is Mode.RESCUE and a.assigned_target is not None:
                chosen[a.assigned_target] = chosen.get(a.assigned_target, 0) + 1
        for j, t in enumerate(self.targets[:T]):
            dists = [d for d in (self._dist(i, t.pos) for i in range(N)) if d < 10 ** 9]
            tasks[j] = [*self.frame.normalized(t.pos),
                        min(1.0, max(0.0, deadline(t) - self.time) / t_max),
                        float(t.discovered), float(t.active),
                        float(t.rescued_at is not None), float(t.expired_at is not None),
                        chosen.get(j, 0) / N,
                        min(1.0, min(dists) / t_max) if dists else 1.0]
            present[j] = 1.0
        team = np.zeros((N, self.CRITIC_TEAM_DIM), np.float32)
        for i, a in enumerate(self.agents):
            team[i, :3] = [*self.frame.normalized(a.pos), a.capacity / max(1, a.capacity_max)]
            team[i, 3 + a.mode.value] = 1.0
            if a.goal is not None:
                team[i, 6:8] = self.frame.normalized(a.goal)
        glob = np.array([sum(1 for t in self.targets if not t.discovered) / n_t,
                         sum(1 for t in self.targets if t.rescued_at is not None) / n_t,
                         sum(1 for t in self.targets if t.expired_at is not None) / n_t,
                         self.team_explored_ratio(),
                         min(1.0, self.time / t_max)], np.float32)
        return {"critic_tasks": tasks, "critic_task_present": present, "critic_team": team, "critic_global": glob}
