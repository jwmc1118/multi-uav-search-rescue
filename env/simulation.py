"""RescueSimulation — 탐색/구조 환경의 핵심 메커니즘 (UAV, 고정 고도 2D 격자).

한 tick 순서:
    통합 할당 (task / explore / depot 중 하나, 순차 디코딩 + lock) -> 목적지 -> primitive 이동 ->
    시간 진행 -> 센싱/통신 병합(knowledge) -> 발견 / 구조 / 재보충 / 마감 판정 -> 보상 ->
    다음 결정 준비(_prepare: 에이전트당 BFS 1회로 노드 특징 / admissible / waypoint 계산)

두 가지 구동 경로가 같은 메커니즘을 공유한다:
    step()          : hook 경로. policy(env/stand_in_policy.py)의 훅이 결정 (GUI stand-in)
    step_explicit() : 학습 경로. 신경망이 고른 (alloc, primitive) 정수를 그대로 적용
                      (env/gym_env.py, training/evaluate.py, GUI Model 모드)

학습용 관측은 rl_observation(), 보상은 env/rewards.py.
"""
import copy
import json
import os
from datetime import datetime

import numpy as np

from env.agent import Robot
from env.coords import WorldFrame
from env.modes import Mode, N_MSG_TYPES
from env.knowledge import KnowledgeTracker, _pool, N_CHANNELS, CHANNEL_INDEX, CH_OBSTACLE, \
    CH_EXPLORED, CH_FRONTIER, CH_TARGET, CH_AGENTS, frontier_map
from env.pathfinding import shortest_path_length
from env.allocation import Allocator
from env.beliefs import BeliefTracker
from env.targets import Target
from env.map_gen import generate_obstacles, sample_target_cells, sample_count
from env.stand_in_policy import RandomStandInPolicy
from env.step_context import StepContext, RescueEvent
from env.rewards import compute_rewards

DEFAULT_OPTIONS = {
    "map_mode": "file",        # 'file' : map_path 의 맵 (None 이면 빈 그리드)
                               # 'random': 매 에피소드(reset) 장애물 맵을 새로 생성 (env/map_gen.py, rows x cols)
    "map_path": None,          # maps/ 아래 JSON 경로. None -> 빈 그리드
    "rows": 40,                # map_path 없을 때 / random 맵 크기
    "cols": 40,
    "obstacle_ratio": [0.2, 0.3],   # random 맵 장애물 비율 범위 (depot 에서 갈 수 없는 칸은 장애물로 메운 뒤 기준)
    "obstacle_block": None,    # random 맵 건물(직사각형) 한 변 길이 [min, max]. None -> [2, max(rows, cols) // 6]
    "target_mode": "file",     # 'file'  : 맵 파일의 조난자 (없으면 num_targets 개를 처음 한 번만 랜덤 배치)
                               # 'random': 매 에피소드 조난자 수/위치를 새로 뽑음 (depot 에서 도달 가능한 칸만)
                               # map_mode='random' 이면 조난자는 항상 random
    "state_size": 21,          # 고해상도 local 관측 윈도우 (자기 중심, 홀수) — primitive 입력
    "local_channels": ["obstacle", "agents"],   # primitive local 입력 채널 (waypoint 채널은 모델이 항상 추가)
                               # 선택지: obstacle / explored / frontier / target / agents (env/knowledge.py)
    "global_downsample": 4,    # critic 중앙집중 state / GUI global 맵 축소 배율
    "n_hop": 1,                # 통신 hop 수 (지식 병합 / 이웃 context / 순차 디코딩 그룹)
    "max_episode_len": 300,    # t_max — 시간/거리 특징 정규화 분모
    "position_mode": "depot",  # 'depot' (모두 depot 에서 출발) / 'random'
    "capacity": 3,             # 로봇 1대의 구호품 용량
    "lock_rescue": True,       # 태스크를 고르면 구조/만료/도달불가까지 유지 (False = 매 tick 재결정, tentative)
    "lock_return": True,       # depot 을 고르면 도착할 때까지 유지
    "mask_return_infeasible": False,  # t + d_ij + d_{j->depot} > t_max 인 태스크 마스크
    "depot": None,             # 재보충 지점 = 모든 로봇의 출발 지점. 월드 좌표 (x, y) (env/coords.py)
                               # None -> 맵 파일의 depot -> 맵 파일의 start -> 원점 (0, 0)
    "target_golden_time": [80, 160],  # 타겟별 골든타임 tau_j. int 또는 [min, max]
    "deadline_mode": "from_discovery",  # 'from_discovery' / 'from_start' (§9 open item 3)
    "num_targets": 10,         # 랜덤 배치 조난자 수. int 또는 [min, max] (예: [10, 20])
    "max_targets": 16,         # T: 태스크 슬롯 수 (관측/행동 패딩 상한). 행동 = T + 2 (explore, depot)
    "reward": {},              # env/rewards.py 가 읽는 coefficient 딕셔너리
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


class RescueSimulation:
    def __init__(self, options=None, policy=None):
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
                  capacity=self.cfg["capacity"])
            for spec in self.cfg["agents"]
        ]
        self.beliefs = BeliefTracker(len(self.agents))   # UAV 별 구조 완료 / lock 믿음 (env/beliefs.py)
        self.knowledge = KnowledgeTracker(
            self.agents, self.obstacles, self.rows, self.cols,
            state_size=self.cfg["state_size"],
            global_downsample=self.cfg["global_downsample"],
            n_hop=self.cfg["n_hop"],
            frame=self.frame,
        )
        bad = [c for c in self.cfg["local_channels"] if c not in CHANNEL_INDEX]
        if bad or not self.cfg["local_channels"]:
            raise ValueError(f"local_channels {bad or '[]'} — 선택지: {list(CHANNEL_INDEX)}")
        self.local_ch_idx = [CHANNEL_INDEX[c] for c in self.cfg["local_channels"]]

        self.max_targets = int(self.cfg["max_targets"])
        self.allocator = Allocator(self.max_targets,
                                   lock_rescue=self.cfg["lock_rescue"],
                                   lock_return=self.cfg["lock_return"],
                                   mask_return_infeasible=self.cfg["mask_return_infeasible"])
        self.policy = policy or RandomStandInPolicy()

        self.n_agents = len(self.agents)
        self.n_actions_alloc = self.allocator.A
        ds = self.cfg["global_downsample"]
        self.gh = -(-self.rows // ds)
        self.gw = -(-self.cols // ds)

        self.targets = []
        self.nodes = []            # 이번 tick 의 NodeSet (env/allocation.py), _prepare() 가 갱신
        self.order = []            # 순차 디코딩 순서
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

    # ---------- 랜덤 맵 / 조난자 (env/map_gen.py) ----------

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
                                         block_size=self.cfg["obstacle_block"])
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
            agent.msg_goal = self.frame.normalized(agent.pos)

        self.beliefs.reset()
        self.knowledge.update(self.active_target_positions())
        self._update_discovery()
        self.beliefs.update(self, self.knowledge.last_visible)
        self._prepare()

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
        """다음 결정을 위한 노드 집합 / 디코딩 순서 계산 (위치·지식·시간이 바뀐 직후)."""
        self.nodes, self.released_locks = self.allocator.prepare(self)
        self.order = self.allocator.decode_order(self.nodes)

    # ---------- tick 공통 부분 ----------

    def _advance(self, ctx=None):
        """시간 진행 -> 센싱/병합 -> 발견 -> 구조 -> 재보충 -> 마감. (rescued, expired) 반환."""
        self.time += 1
        self.knowledge.update(self.active_target_positions())
        self._update_discovery()
        rescued = self._process_rescues()
        self._process_refill(ctx)
        expired = self._process_deadlines()
        self.allocator.resolve_contacts(self, ctx)       # 같은 태스크를 쫓던 UAV 들이 통신 연결 -> 먼 쪽 해제
        self.beliefs.update(self, self.knowledge.last_visible)   # 구조 완료 / lock 정보 전파 (n-hop)
        return rescued, expired

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
                self.allocator.release(self, i)
                self.beliefs.record_rescue(i, ti)
                tt = None if t.discovered_at is None else self.time - t.discovered_at
                events.append(RescueEvent(target=ti, agent=i, rescued_at=self.time,
                                          discovered_at=t.discovered_at, time_taken=tt))
        return events

    def _process_refill(self, ctx=None):
        for i, a in enumerate(self.agents):
            if a.alive and a.pos == self.depot:
                if ctx is not None:
                    ctx.at_depot.add(i)
                    if a.capacity < a.capacity_max:
                        ctx.refilled.add(i)
                a.refill()
                if a.mode is Mode.RETURN:
                    a.locked = False                     # return lock 해제 (도착)

    def _process_deadlines(self):
        expired = []
        for ti, t in enumerate(self.targets):
            if t.active and t.discovered and t.end_time is not None and self.time >= t.end_time:
                t.expired_at = self.time
                for i, a in enumerate(self.agents):
                    if a.assigned_target == ti:
                        self.allocator.release(self, i)
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
            "discovered": sum(1 for t in self.targets if t.discovered),
            "explored": self.team_explored_ratio(),
        }

    def _execute(self, allocs, choose_primitive):
        """할당 적용 -> primitive 이동 -> 시간 진행 -> 보상. 두 구동 경로의 공통 본체."""
        nodes = self.nodes
        prev_free = [int(f.sum()) for f in self.knowledge.known_free]
        was_discovered = [t.discovered for t in self.targets]
        ctx = StepContext(time=self.time, n_agents=self.n_agents)
        ctx.released_locks = list(getattr(self, "released_locks", []))
        ctx.free_choice = [a.alive and nodes[i].forced_slot is None for i, a in enumerate(self.agents)]
        ctx.team_explored_before = self.team_explored_ratio()

        # 1) 할당 (충돌 해결 + lock + 메시지)
        self.allocator.apply(self, nodes, allocs, ctx)

        ctx.chosen_mode = [a.mode for a in self.agents]

        # 2) 목적지 / primitive
        blocked = [self.knowledge.blocked_mask(i) for i in range(self.n_agents)]
        for i, a in enumerate(self.agents):
            ctx.goal.append(a.goal)
            g = a.goal
            ctx.goal_dist_before.append(None if g is None or nodes[i].dist_field[g] < 0
                                        else int(nodes[i].dist_field[g]))
        prev_pos = [a.pos for a in self.agents]
        for i, a in enumerate(self.agents):
            prev = a.pos
            blocked_try = False
            if a.alive:
                act = choose_primitive(i)
                if act != "stay":
                    blocked_try = not a.move(act, self.obstacles, self.rows, self.cols)
            ctx.moved.append(a.pos != prev)
            ctx.blocked_move.append(blocked_try)
        ctx.collided = self._collisions(prev_pos)
        ctx.duplicate_pursuit = self._duplicate_pursuit()
        for i, a in enumerate(self.agents):
            g = ctx.goal[i]
            ctx.goal_dist_after.append(None if g is None else
                                       shortest_path_length(a.pos, g, blocked[i], self.rows, self.cols))

        # 3) 시간 진행 + 센싱/병합 + 발견/구조/재보충/마감
        rescued, expired = self._advance(ctx)

        # 4) ctx 마무리
        ctx.modes = [a.mode for a in self.agents]
        ctx.capacities = [a.capacity for a in self.agents]
        ctx.assigned_target = [a.assigned_target for a in self.agents]
        ctx.rescued_now = rescued
        ctx.expired_now = expired
        ctx.newly_explored = [int(f.sum()) - p for f, p in zip(self.knowledge.known_free, prev_free)]
        ctx.newly_sensed = list(self.knowledge.newly_sensed)
        ctx.team_explored_after = self.team_explored_ratio()
        ctx.newly_discovered_targets = [ti for ti, was in enumerate(was_discovered)
                                        if not was and self.targets[ti].discovered]
        ctx.reached_goal = [g is not None and self.agents[k].pos == tuple(g)
                            for k, g in enumerate(ctx.goal)]

        # 5) 보상 + 다음 결정 준비
        rew = compute_rewards(self, ctx)
        self._prepare()
        info = self._info(rescued, expired)
        info.update({
            "reward_alloc": rew["alloc"],
            "reward_primitive": rew["primitive"],
            # 정책이 실제로 고른 에이전트 = 1 (lock 강제 tick = 0). alloc loss 마스크
            "alloc_valid": [1.0 if f else 0.0 for f in ctx.free_choice],
            "double_selected": len(ctx.double_selected),
            "alloc_invalid": len(ctx.alloc_invalid),
            "collisions": sum(ctx.collided) // 2 if ctx.collided else 0,
            "duplicate_pursuit": int(sum(ctx.duplicate_pursuit)),
            "contact_released": len(ctx.contact_released),
        })
        self.last_ctx = ctx
        return self._is_done(), info

    def _duplicate_pursuit(self):
        """중복 추적 판정 (전지적, 보상 전용). UAV i 가 쫓는 태스크를
          - 더 가까운(BFS) 다른 UAV 도 쫓고 있거나
          - 이미 다른 UAV 가 구조했는데 i 가 아직 모르고 쫓고 있으면  True (먼 쪽만)."""
        out = [False] * self.n_agents
        by_target = {}
        for i, a in enumerate(self.agents):
            if a.alive and a.mode is Mode.RESCUE and a.assigned_target is not None:
                by_target.setdefault(a.assigned_target, []).append(i)
        for ti, members in by_target.items():
            t = self.targets[ti]
            if t.rescued_at is not None:
                for k in members:
                    out[k] = True
                continue
            if len(members) < 2:
                continue

            def dist(k):
                d = shortest_path_length(self.agents[k].pos, t.pos, self.knowledge.known_obstacle[k],
                                         self.rows, self.cols)
                return (d if d is not None else 10 ** 9, k)
            winner = min(members, key=dist)
            for k in members:
                if k != winner:
                    out[k] = True
        return out

    def _collisions(self, prev_pos):
        """UAV 간 충돌 판정 (이동 후). depot 칸은 기지라 제외.
          - 같은 칸 점유: p^i_{t+1} = p^j_{t+1}
          - 자리 맞바꿈(스쳐 지나감): p^i_{t+1} = p^j_t 이고 p^j_{t+1} = p^i_t
        반환: per-agent bool. 이동 자체는 막지 않는다 (보상으로만 학습)."""
        n = self.n_agents
        pos = [a.pos for a in self.agents]
        out = [False] * n
        for i in range(n):
            for j in range(i + 1, n):
                if not (self.agents[i].alive and self.agents[j].alive):
                    continue
                same = pos[i] == pos[j] and pos[i] != self.depot
                swap = (pos[i] == prev_pos[j] and pos[j] == prev_pos[i] and pos[i] != pos[j])
                if same or swap:
                    out[i] = out[j] = True
        return out

    # ---------- step: hook 경로 (GUI stand-in) ----------

    def step(self, policy=None):
        """policy 훅이 노드/primitive 를 결정. 순차 디코딩은 환경이 직접 수행. 반환: (done, info)."""
        policy = policy or self.policy
        reach = self.knowledge.reach
        allocs = [self.allocator.D] * self.n_agents
        picks = {}
        for i in self.order:
            a = self.agents[i]
            if not a.alive:
                continue
            ns = self.nodes[i]
            mask = ns.mask & ~self.allocator.group_conflicts(i, picks, self.nodes, reach)
            s = int(policy.select_node(a, i, ns, mask, self))
            allocs[i] = s
            picks[i] = ns.task_ids[s] if s < self.max_targets else None
        return self._execute(
            allocs, lambda i: policy.select_primitive(self.agents[i], i, self.agents[i].goal, self))

    # ---------- step: 학습 경로 ----------

    def step_explicit(self, allocs, primitives):
        """신경망이 뽑은 (alloc 슬롯, primitive) 정수 액션을 그대로 적용한다.

        allocs     : (n,) int, 노드 슬롯 (0..T-1 태스크, T = explore, T+1 = depot)
        primitives : (n,) int, Robot.ACTIONS 인덱스
        반환: (done, info). info 에 계층별 보상 + alloc_valid 마스크가 들어간다.
        """
        allocs = [int(x) for x in allocs]
        primitives = [int(x) for x in primitives]

        def prim(i):
            k = primitives[i]
            return Robot.ACTIONS[k] if 0 <= k < len(Robot.ACTIONS) else "stay"
        return self._execute(allocs, prim)

    # ================= RL 관측 (env/gym_env.py 의 observation_space 와 1:1) =================

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

    def rl_observation(self):
        """고정 shape 관측 dict. 노드 슬롯 순서 = [태스크 T | explore | depot] (env/allocation.py).

        정규화: 위치 = 월드 좌표 / 맵 크기 (env/coords.py, 대략 [-0.5, 0.5]),
                시간·거리·골든타임 = / t_max (max_episode_len), 용량 = / c_max.
        """
        T, N, A = self.max_targets, self.n_agents, self.allocator.A
        t_max = float(self.cfg["max_episode_len"])
        n_targets = max(1, len(self.targets))
        n_free = self.n_free_cells()

        def nd(d):                       # 거리 정규화 (도달 불가 = 1)
            return 1.0 if d is None or d < 0 else min(1.0, d / t_max)

        agents_obs = []
        for i, a in enumerate(self.agents):
            ns = self.nodes[i]
            task_feats = np.zeros((T, 4), np.float32)     # δ^i_j = [x/W, y/H, τ_j, d^i_j]
            task_present = np.zeros((T,), np.float32)
            task_ids = np.full((T,), -1.0, np.float32)
            for s, ti in enumerate(ns.task_ids):
                t = self.targets[ti]
                rem = t.remaining(self.time)
                tau = 1.0 if rem is None else float(np.clip(rem / t_max, 0.0, 1.0))
                task_feats[s] = [*self.frame.normalized(t.pos), tau, nd(ns.task_dist[s])]
                task_present[s] = 1.0
                task_ids[s] = ti

            # δ^i_e = [T^i / N_T, E^i_t / |C_free|, d^i_e]
            explore_feats = np.array([min(1.0, ns.n_known / n_targets),
                                      float(self.knowledge.known_free[i].sum()) / n_free,
                                      nd(ns.d_explore)], np.float32)
            # δ^i_d = [x_d/W, y_d/H, d^i_d]
            depot_feats = np.array([*self.frame.normalized(self.depot), nd(ns.d_depot)], np.float32)
            # ξ^i = [x/W, y/H, c/c_max]
            self_feats = np.array([*self.frame.normalized(a.pos),
                                   a.capacity / max(1, a.capacity_max)], np.float32)

            node_xy = np.zeros((A, 2), np.float32)
            for s, g in enumerate(ns.goal):
                if g is not None:
                    node_xy[s] = self.frame.normalized(g)
            node_type = np.array([self.allocator.slot_msg_type(s) for s in range(A)], np.float32)

            agents_obs.append({
                "local": a.local_window[..., self.local_ch_idx].astype(np.float32),
                "task_feats": task_feats,
                "task_present": task_present,
                "task_ids": task_ids,
                "explore_feats": explore_feats,
                "depot_feats": depot_feats,
                "self_feats": self_feats,
                "node_mask": ns.mask.astype(np.float32),
                "node_xy": node_xy,
                "node_type": node_type,
                "node_wp": ns.waypoint.astype(np.float32),
                "free_choice": np.array([0.0 if ns.forced_slot is not None else 1.0], np.float32),
            })

        # 팀 테이블 — 모델은 reach (n-hop) 로 마스킹해 이웃 것만 쓴다
        pub = np.zeros((N, 3), np.float32)          # [x/W, y/H, c/c_max]
        prev_msg = np.zeros((N, 2 + N_MSG_TYPES), np.float32)   # [g_x, g_y, o (4)] 이전 tick 결정
        for j, a in enumerate(self.agents):
            pub[j] = [*self.frame.normalized(a.pos), a.capacity / max(1, a.capacity_max)]
            prev_msg[j, :2] = a.msg_goal
            prev_msg[j, 2 + a.msg_type] = 1.0
        rank = np.zeros((N,), np.float32)
        for k, i in enumerate(self.order):
            rank[i] = k
        return {
            "agents": tuple(agents_obs),
            "pub": pub,
            "prev_msg": prev_msg,
            "reach": self.knowledge.reach.astype(np.float32),
            "dec_rank": rank,
            "time": np.array([min(1.0, self.time / t_max)], np.float32),
            "state": self.centralized_state().astype(np.float32),
        }
