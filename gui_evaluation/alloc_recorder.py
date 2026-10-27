"""임무 할당 기록기 — 에피소드 동안 UAV 상태 / 이벤트 / 조난자 / 팀 지표를 모은다 (gui_evaluation/alloc_plots.py 의 입력).

RoleSimulation 에 붙여 쓴다:  sim.recorder = AllocRecorder(); sim.recorder.on_reset(sim)
그러면 sim.reset() 이 on_reset 을, 매 tick 의 _step_role() 끝 (step / step_explicit 공통)이 on_step 을 호출한다.
GUI 의 Reset / Change Map / Load / Targets 는 모두 sim.reset() 을 거치므로 기록도 함께 초기화된다.
학습 환경에서는 recorder 가 None 이라 비용이 없다.

시각 t 의 의미: on_step 의 행은 "tick t 를 실행한 결과" (t = 시간 진행 후 sim.time). 모드 / 배정은 그 tick 에 적용된 결정.
t = 0 행은 리셋 직후 상태 (모드 = 초기값, 결정 전).
"""
import copy
import datetime

import numpy as np


def _components(adj):
    """인접 행렬의 연결 성분 id (가장 작은 UAV 번호 순으로 0, 1, ...)."""
    n = len(adj)
    comp = [-1] * n
    cid = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = cid
        stack = [s]
        while stack:
            i = stack.pop()
            for j in range(n):
                if adj[i][j] and comp[j] < 0:
                    comp[j] = cid
                    stack.append(j)
        cid += 1
    return comp


class AllocRecorder:
    def __init__(self, policy="rules", seed=None):
        self.policy = policy          # 지금 결정을 내리는 정책 이름 (GUI 가 tick 마다 갱신)
        self.seed = seed
        self.clear()

    # ------------------------------------------------------------ 상태

    def clear(self):
        self.meta = {}
        self.steps = []           # per-step per-UAV 행 (dict)
        self.events = []          # {t, type, agent, target, detail}
        self.team = []            # per-step 팀 지표 {t, explored, discovered, rescued, expired}
        self.obstacles = None     # 정적 맵 (rows, cols) bool
        self.depot = None
        self.targets0 = []        # 리셋 시점 조난자 [(pos, golden_time, end_time)]
        self._prev = None         # UAV 별 직전 (mode, target)
        self.policies = set()

    def on_reset(self, sim):
        self.clear()
        self.obstacles = np.array(sim.obstacles, dtype=bool)
        self.depot = tuple(sim.depot)
        self.targets0 = [(tuple(t.pos), t.golden_time, t.end_time) for t in sim.targets]
        self.meta = {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": self.seed if self.seed is not None else sim.cfg.get("seed"),
            "rows": sim.rows, "cols": sim.cols,
            "n_targets": len(sim.targets), "n_agents": sim.n_agents,
            "depot": list(self.depot),
            "deadline_mode": sim.cfg.get("deadline_mode"),
            "max_episode_len": sim.cfg.get("max_episode_len"),
        }
        self._prev = [(a.mode.name, -1) for a in sim.agents]
        self._record_state(sim, t=0, free=[False] * sim.n_agents)
        self._record_team(sim)

    def on_step(self, sim, ctx, info):
        t = sim.time
        self.policies.add(self.policy)
        for i, a in enumerate(sim.agents):
            mode = ctx.chosen_mode[i].name                 # 이번 tick 에 적용된 결정 (구조 처리 전)
            tgt = ctx.chosen_target[i] if mode == "RESCUE" and ctx.chosen_target[i] is not None else -1
            if (mode, tgt) != self._prev[i]:
                self.events.append({"t": t, "type": "assign", "agent": i, "target": tgt,
                                    "detail": f"{self._label(*self._prev[i])}->{self._label(mode, tgt)}"})
                self._prev[i] = (mode, tgt)
        for ti in ctx.newly_discovered_targets:
            pos = sim.targets[ti].pos
            by = next((i for i, vis in enumerate(sim.knowledge.last_visible) if pos in vis), -1)
            self.events.append({"t": t, "type": "discover", "agent": by, "target": ti, "detail": ""})
        for ev in ctx.rescued_now:
            self.events.append({"t": t, "type": "rescue", "agent": ev.agent, "target": ev.target, "detail": ""})
        for ti in ctx.expired_now:
            holders = [i for i in range(sim.n_agents) if self._prev[i] == ("RESCUE", ti)]
            self.events.append({"t": t, "type": "expire", "agent": holders[0] if holders else -1, "target": ti,
                                "detail": ",".join(map(str, holders))})
        for i in sorted(ctx.refilled):
            self.events.append({"t": t, "type": "refill", "agent": i, "target": -1, "detail": ""})
        self._record_state(sim, t, free=ctx.free_choice, modes=ctx.chosen_mode, targets=ctx.chosen_target)
        self._record_team(sim)

    # ------------------------------------------------------------ 내부

    @staticmethod
    def _label(mode, tgt):
        return f"T{tgt}" if mode == "RESCUE" and tgt >= 0 else mode.lower()

    def _record_state(self, sim, t, free, modes=None, targets=None):
        groups = _components(np.asarray(sim.knowledge.adjacency))
        for i, a in enumerate(sim.agents):
            mode = (modes[i] if modes is not None else a.mode).name
            tgt = targets[i] if targets is not None else a.assigned_target
            self.steps.append({
                "t": t, "agent": i, "r": a.pos[0], "c": a.pos[1], "mode": mode,
                "target": tgt if mode == "RESCUE" and tgt is not None else -1,
                "free_choice": int(bool(free[i])),
                "capacity": a.capacity, "group": groups[i],
                "goal_r": a.goal[0] if a.goal is not None else -1,
                "goal_c": a.goal[1] if a.goal is not None else -1,
                "policy": self.policy,
            })

    def _record_team(self, sim):
        self.team.append({
            "t": sim.time,
            "explored": sim.team_explored_ratio(),
            "discovered": sum(1 for t in sim.targets if t.discovered),
            "rescued": sum(1 for t in sim.targets if t.rescued_at is not None),
            "expired": sum(1 for t in sim.targets if t.expired_at is not None),
        })

    # ------------------------------------------------------------ 요약 / 내보내기

    def target_summary(self, sim):
        """per-target: 위치, 골든타임 / 마감, 발견 시각, 처음 배정 시각과 UAV, 구조 시각 / UAV 또는 만료 시각."""
        first = {}
        for ev in self.events:
            if ev["type"] == "assign" and ev["target"] >= 0 and ev["target"] not in first:
                first[ev["target"]] = (ev["t"], ev["agent"])
        out = []
        for ti, t in enumerate(sim.targets):
            fa = first.get(ti, (None, None))
            out.append({
                "target": ti, "r": t.pos[0], "c": t.pos[1], "golden_time": t.golden_time,
                "deadline": t.end_time if t.end_time is not None else t.golden_time,
                "discovered_at": t.discovered_at, "first_assigned_at": fa[0], "first_assigned_by": fa[1],
                "rescued_at": t.rescued_at, "rescued_by": t.rescued_by, "expired_at": t.expired_at,
                "status": "rescued" if t.rescued_at is not None else "expired" if t.expired_at is not None
                else "discovered" if t.discovered else "undiscovered",
            })
        return out

    def snapshot(self, sim, policy_label=None):
        """그래프 생성용 사본 (GUI 가 계속 진행해도 안전하도록 별도 스레드에 넘긴다)."""
        data = {
            "meta": dict(self.meta, final_t=sim.time,
                         policy=policy_label or ("+".join(sorted(self.policies)) or self.policy),
                         agent_colors=[list(a.color[:3]) for a in sim.agents]),
            "steps": copy.deepcopy(self.steps),
            "events": copy.deepcopy(self.events),
            "team": copy.deepcopy(self.team),
            "targets": self.target_summary(sim),
            "obstacles": self.obstacles.copy() if self.obstacles is not None else np.array(sim.obstacles, bool),
            "depot": self.depot,
        }
        return data
