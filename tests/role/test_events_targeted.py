"""이벤트 체계 (Role 이벤트 1~5, 대상 UAV 만 재결정) 시나리오 점검 (ray 불필요) — 빈 맵에 frontier / 생존자 / 통신 연결을 직접 놓고
env 결정 경로를 호출. (다른 테스트가 쓰는 공용 도우미 Recorder / add_victim / fresh 도 여기)

    python -m tests.role.test_events_targeted

 [1] 새 생존자 발견 (이벤트 2) -> 그 그룹의 EXPLORE UAV 만 재결정, RESCUE / RETURN UAV 는 역할 · 목표 유지
 [2] 구조 완료 (3) / 귀환 완료 (4) / 목표 무효 (5) -> 해당 UAV 만 재결정
 [3] 이미 알던 생존자가 통신 병합으로 다시 들어와도 이벤트 2 없음 / 그룹 구성이 바뀌어도 재결정 없음
 [4] goal latch: 구간 경계가 1 step 바뀌어도 목표 유지, 목표 셀이 관측되면 변경, 구간 밖 3 step 연속이면 변경
 [5] 구간 없는 상태 (|C| < m) 가 3 step 연속이면 이벤트 5 (그 전에는 없음)
 [6] 통신 hysteresis: 거리가 R_c 와 R_c + 1 사이를 오가도 링크가 깜빡이지 않음 (R_c + h 넘으면 해제)
 [7] 같은 step 에 이벤트가 겹쳐도 재결정 1 번, 원인은 모두 기록
"""
import numpy as np

from env.core.modes import Mode
from env.core.targets import Target
from tests.role.test_explore import blank_sim, connect, line, place, set_front, tick


class Recorder:
    """_decide 에 넘기는 결정 함수들. 호출 기록 + 기본값 (지정 없으면 지금 역할 / 마스크 첫 노드)."""

    def __init__(self, sim, role=None, rescue=None, explore=None):
        self.sim, self.role_v, self.rescue_v, self.explore_v = sim, role or {}, rescue or {}, explore or {}
        self.calls = {"role": [], "rescue": [], "explore": [], "sub": []}

    def role(self, i, m):
        self.calls["role"].append(i)
        r = self.role_v.get(i, self.sim.agents[i].mode.value)
        return r if m[r] else int(np.flatnonzero(m)[0])

    def rescue(self, i, m):
        self.calls["rescue"].append(i)
        s = self.rescue_v.get(i)
        return s if s is not None and m[s] else int(np.flatnonzero(m)[0])

    def explore(self, i, m):
        self.calls["explore"].append(i)
        k = self.explore_v.get(i)
        return k if k is not None and m[k] else int(np.flatnonzero(m)[0])

    def sub(self, i, m):
        self.calls["sub"].append(i)
        return int(np.flatnonzero(m)[0])

    def run(self):
        sim = self.sim
        sim.mask_violation_step = 0
        rv, sv, _, viol = sim._decide(self.role, self.rescue, self.explore, self.sub)
        assert viol == 0 and sim.mask_violation_step == 0
        return sv


def add_victim(sim, pos, deadline=300, discovered=True):
    t = Target(pos=tuple(pos), golden_time=deadline)
    t.end_time = deadline
    if discovered:
        t.discovered_at = sim.time
        for k in range(sim.n_agents):
            sim.knowledge.known_free[k][t.pos] = True
    sim.targets.append(t)
    return len(sim.targets) - 1


def fresh(n=3, complete=False, **over):
    """빈 맵 시나리오 sim. 시나리오 생존자는 모두 발견 · 그룹이 앎으로 놓이므로, complete=False (기본) 면 exploration complete
    판정을 끈다 (EXPLORE 마스크 · 이벤트 5 가 시나리오를 바꾸지 않게). 그 기능 자체는 complete=True 로 점검
    (tests/role/test_rewards.py check_explore_mask)."""
    sim = blank_sim(n, **over)
    sim.targets = []
    sim._known = [set() for _ in range(n)]
    if not complete:
        sim.exploration_complete = lambda g: False
    return sim


def three_roles():
    """UAV 0 EXPLORE, 1 RESCUE (생존자 T0), 2 RETURN — 한 그룹."""
    sim = fresh(3)
    place(sim, [(14, 3), (14, 20), (20, 30)])
    set_front(sim, line(15, 5, 34))
    t0 = add_victim(sim, (5, 5))
    tick(sim)
    sim._assign_rescue(1, t0)
    sim._assign_return(2)
    sim.agents[2].capacity = 1
    sim._known = [sim._known_victims(i) for i in range(3)]
    return sim, t0


def check_new_survivor():
    sim, t0 = three_roles()
    a = sim.agents
    add_victim(sim, (25, 35))                                    # 새 생존자 (그룹 belief 에 처음)
    tick(sim)
    ev = sim._collect_events([x.pos for x in a], set(), {})
    assert ev == [{"2"}, set(), set()], ev
    sim.events = ev
    sim._set_decide_order()
    assert sim.kind == ["role", None, None] and sim.decide_order == [0]
    rec = Recorder(sim, role={0: Mode.RESCUE.value})
    rec.run()
    assert rec.calls["role"] == [0] and a[1].assigned_target == t0 and a[2].mode is Mode.RETURN
    assert a[0].mode is Mode.RESCUE and a[0].assigned_target != t0, "이미 맡은 생존자는 후보에서 빠짐"
    print("[1] new survivor (event 2) -> only EXPLORE R0 re-decides (took the new victim); RESCUE R1 keeps T0, "
          "RETURN R2 keeps returning")


def check_own_events():
    sim, t0 = three_roles()
    a = sim.agents
    prev = [x.pos for x in a]
    prev[2] = (sim.depot[0], sim.depot[1] + 1)
    a[2].pos = sim.depot                                         # RETURN 도착
    ev = sim._collect_events(prev, {1}, {0: "exhausted"})        # R1 구조, R0 구간 소진
    assert ev == [{"5"}, {"3"}, {"4"}], ev
    sim.events = ev
    sim._set_decide_order()
    assert sim.kind == ["role", "role", "role"]
    sim, t0 = three_roles()
    sim.targets[t0].expired_at = sim.time                        # R1 목표 만료 -> R1 만
    ev = sim._collect_events([x.pos for x in sim.agents], set(), {})
    assert ev == [set(), {"5"}, set()], ev
    sim.events = ev
    sim._set_decide_order()
    assert sim.kind == [None, "role", None]
    print("[2] rescued (3) / returned (4) / goal invalid (5) -> only that UAV re-decides "
          "(expired target -> R1 only, others untouched)")


def check_known_merge_and_group_change():
    sim = fresh(2)
    place(sim, [(14, 3), (14, 36)])
    connect(sim, [[0], [1]])
    set_front(sim, line(15, 5, 34))
    v = add_victim(sim, (5, 5))
    tick(sim)
    sim._known = [{v}, set()]                                    # R0 만 알던 생존자
    connect(sim)                                                 # 통신 병합 -> R1 도 알게 됨
    tick(sim)
    ev = sim._collect_events([x.pos for x in sim.agents], set(), {})
    assert ev == [set(), set()], f"이미 알던 생존자로 이벤트 2 가 나면 안 됨 {ev}"
    connect(sim, [[0], [1]])                                     # 그룹 구성 변경만
    tick(sim)
    ev = sim._collect_events([x.pos for x in sim.agents], set(), {})
    sim.events = ev
    sim._set_decide_order()
    assert ev == [set(), set()] and sim.kind == [None, None]
    print("[3] victim already known by R0 enters R1 by comm merge -> no event 2; group split / merge alone -> no re-decision")


def check_goal_latch():
    sim = fresh(1)
    place(sim, [(14, 10)])
    set_front(sim, line(15, 5, 34))
    tick(sim)
    a = sim.agents[0]
    a.mode, a.blob = Mode.EXPLORE, sim.clusters_of(0)[0].id
    cur = (15, 12)
    a.goal, a.goal_out = cur, 0
    seg = frozenset((15, c) for c in range(20, 35))              # 경계가 옮겨져 목표 셀이 구간 밖
    hold0 = sim.ev_stats["latch_hold"]
    assert sim._latch_goal(0, seg) == cur and sim._latch_goal(0, seg) == cur, "1~2 step 은 유지"
    assert sim.ev_stats["latch_hold"] == hold0 + 2
    new = sim._latch_goal(0, seg)                                # 3 step 연속 밖 -> 구간 안 가장 가까운 셀
    assert new in seg and new != cur, new
    a.goal, a.goal_out = cur, 0
    inside = frozenset((15, c) for c in range(5, 20))
    assert sim._latch_goal(0, inside) == cur
    sim.knowledge.frontier[0][cur] = False                       # 관측됨 -> 바로 변경
    g2 = sim._latch_goal(0, inside)
    assert g2 != cur and g2 in inside
    print(f"[4] goal latch: target outside segment kept for 2 steps (latch holds +2), changed on 3rd -> {new}; "
          f"observed target -> changed immediately -> {g2}")


def check_no_segment():
    sim = fresh(3)
    place(sim, [(14, 3), (14, 20), (14, 36)])
    set_front(sim, line(15, 5, 34))
    tick(sim)
    sim.events = [{"1"}] * 3
    sim._set_decide_order()
    Recorder(sim, role={i: Mode.EXPLORE.value for i in range(3)}).run()
    assert len({x.blob for x in sim.agents}) == 1
    set_front(sim, line(15, 5, 6))                               # 셀 2 개 < 담당자 3 -> 한 대는 구간 없음
    got = []
    for k in range(3):
        sim.time += 1
        ex = tick(sim)
        got.append(sorted(i for i, why in ex.items() if why == "no segment"))
    assert got[0] == [] and got[1] == [] and len(got[2]) == 1, got
    ev = sim._collect_events([x.pos for x in sim.agents], set(), ex)
    assert "5" in ev[got[2][0]]
    print(f"[5] |C| < m: no segment for R{got[2][0]} -> event 5 only after 3 consecutive steps ({got})")


def check_hysteresis():
    sim = fresh(2)
    kn = sim.knowledge
    R = sim.agents[0].comm_range
    h = kn.hysteresis
    assert h == 2
    seq = [R, R + 1, R, R + 1, R + 2, R + 1, R + 3, R]
    links, links0 = [], []
    prev = np.zeros((2, 2), bool)
    prev0 = np.zeros((2, 2), bool)
    for d in seq:
        place(sim, [(10, 5), (10, 5 + d)])
        adj = kn.compute_adjacency(prev)
        links.append(bool(adj[0, 1]))
        prev = adj
        kn.hysteresis = 0
        adj0 = kn.compute_adjacency(prev0)
        links0.append(bool(adj0[0, 1]))
        prev0 = adj0
        kn.hysteresis = h
    assert links == [True, True, True, True, True, True, False, True], links
    assert links0 == [True, False, True, False, False, False, False, True], links0
    print(f"[6] hysteresis (R_c {R}, h {h}): distance {seq} -> link {[int(x) for x in links]} (no flicker); "
          f"without hysteresis {[int(x) for x in links0]}")


def check_overlap_once():
    sim, t0 = three_roles()
    sim.events = [{"2", "5"}, {"3", "5"}, set()]
    n_log = len(sim.ev_log)
    sim._set_decide_order()
    assert sim.decide_order == [0, 1]
    causes = sorted(e["event"] for e in sim.ev_log[n_log:] if e["trigger"] == 1)
    rec = Recorder(sim)
    rec.run()
    assert sorted(rec.calls["role"]) == [0, 1] and causes == ["3", "5"], (rec.calls, causes)
    print(f"[7] overlapping events -> one re-decision per UAV (role calls {sorted(rec.calls['role'])}), causes logged {causes}")


if __name__ == "__main__":
    check_new_survivor()
    check_own_events()
    check_known_merge_and_group_change()
    check_goal_latch()
    check_no_segment()
    check_hysteresis()
    check_overlap_once()
    print("test_events_targeted: all ok")
