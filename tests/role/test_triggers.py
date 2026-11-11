"""Role 재결정 트리거 점검 (주기 재결정, 이벤트 "1" 생존자 증가, 목표 유효, 같은 목표 재선택) — ray 불필요.

    python -m tests.role.test_triggers

 [1] period 10: t = 0, 10, 20, ... 에 alive UAV 전원 재결정 (RETURN 중 · 용량 0 포함), 그 사이 step 에는 주기 원인 없음
 [2] 이벤트 "1": (a) 그룹 안 탐색 발견 -> 그룹 전원 (역할 무관, ev1_explore) / (b) 두 그룹 통신 합류로 한쪽이 몰랐던
     생존자 -> 합쳐진 그룹 전원 (ev1_merge) / (c) 합류했지만 아는 생존자 같음 -> 없음 / (d) 늘어난 생존자가 전부 만료 · 도달 불가 ·
     이미 맡음 -> 없음
 [3] goal_valid: periodic · "1" 재결정 UAV 는 유지, "2" / "3" / "4" 는 꺼짐
 [4] RESCUE 재결정: 자기 현재 목표는 마스크되지 않음, 재결정하지 않는 UAV 가 맡은 생존자는 막힘
 [5] 같은 생존자 재선택: rescue_valid 0 (표본 아님), 목표 전환 아님. 유효한 목표를 두고 다른 생존자로 바꾸면 새 표본 + 자발적 목표 전환
 [6] 용량 0 UAV 가 주기 재결정에서 EXPLORE 선택 가능 (역할 마스크 RETURN · EXPLORE), cap0_explore_choice 기록
 [7] 규칙 정책: periodic / "1" 은 그룹 그리디 (담당 후보 생존자), "2" / "3" / "4" 는 한 대 여유 시간 최소
     (slack = τ - t - d 작은 순 -> 생존자 번호)
 [8] 한 대 규칙 = 여유 시간 기준 (EDF 아님): t = 0, 생존자 1 (τ 100, d 80, slack 20) vs 생존자 2 (τ 90, d 10, slack 80) -> 1.
     같은 상태에서 role_front (D_0 = {A}) 의 담당 후보와 같다
 [9] slack 동률: 더 가까운 생존자의 번호가 더 클 때 번호 작은 (먼) 쪽 — rule_rescue = role_front (D_0 = {A}) claim
"""
import numpy as np

from env.core.modes import Mode
from tests.role.test_events_targeted import Recorder, add_victim, fresh, three_roles
from tests.role.test_explore import COLS, ROWS, connect, line, place, set_front, tick

E, R, D = Mode.EXPLORE.value, Mode.RESCUE.value, Mode.RETURN.value


def env_cfg(**events):
    from training.config_loader import load_config
    cfg = load_config()["env_config"]
    cfg["events"] = dict(cfg["events"], log_dir=None, **events)
    return cfg


def check_periodic():
    from env.role.simulation import RoleSimulation
    from policies.role_rules import RandomDecider
    sim = RoleSimulation(env_cfg(period=10))
    sim.random_state = np.random.RandomState(0)
    sim.reset()
    hits, special = [], {"return": 0, "cap0": 0}
    dec = RandomDecider(0)
    for _ in range(60):
        t = sim.time
        if t == 15:                                               # 용량 0 + RETURN 중인 UAV 도 주기 대상인지
            sim.agents[0].capacity = 0
            sim._assign_return(0)
        alive = [i for i, a in enumerate(sim.agents) if a.alive]
        per = [i for i in range(sim.n_agents) if "periodic" in sim.events[i]]
        if t % 10 == 0:
            assert per == alive and all(sim.kind[i] == "role" for i in alive), (t, per, sim.kind)
            special["return"] += sum(1 for i in alive if sim.agents[i].mode is Mode.RETURN)
            special["cap0"] += sum(1 for i in alive if sim.agents[i].capacity == 0)
            hits.append(t)
        else:
            assert not per, (t, per)
        done, _ = sim.step(dec)
        if done:
            break
    assert hits[:4] == [0, 10, 20, 30], hits
    print(f"[1] period 10: all alive UAVs re-decide at t = {hits} (RETURN UAV-times {special['return']}, capacity-0 "
          f"{special['cap0']} included), no periodic cause in between")


def check_new_survivor_cases():
    # (a) 그룹 안 탐색 발견 -> 그룹 전원 (EXPLORE / RESCUE / RETURN 모두)
    sim, t0 = three_roles()
    add_victim(sim, (25, 35))
    tick(sim)
    ev = sim._collect_events([x.pos for x in sim.agents], set(), {})
    assert ev == [{"1"}] * 3 and (sim.ev_stats["ev1_explore"], sim.ev_stats["ev1_merge"]) == (1, 0), ev
    # (b) 두 그룹 합류: R0 만 알던 생존자가 R1 에게 들어옴 -> 합쳐진 그룹 전원
    def merged(known1):
        sim = fresh(2)
        place(sim, [(14, 3), (14, 36)])
        connect(sim, [[0], [1]])
        set_front(sim, line(15, 5, 34))
        v = add_victim(sim, (5, 5))
        tick(sim)
        sim._known = [{v}, {v} if known1 else set()]
        connect(sim)
        tick(sim)
        return sim, sim._collect_events([x.pos for x in sim.agents], set(), {})
    sim, ev = merged(False)
    assert ev == [{"1"}, {"1"}] and (sim.ev_stats["ev1_explore"], sim.ev_stats["ev1_merge"]) == (0, 1), ev
    # (c) 합류했지만 둘 다 이미 알던 생존자 -> 없음
    sim, ev = merged(True)
    assert ev == [set(), set()], ev
    # (d) 늘어난 생존자가 전부 만료 / 도달 불가 / 이미 맡음 -> 없음
    for case in ("expired", "unreachable", "held"):
        sim, t0 = three_roles()
        k = add_victim(sim, (25, 35), deadline=300 if case != "unreachable" else 1)
        if case == "expired":
            sim.targets[k].expired_at = sim.time
        tick(sim)
        if case == "held":
            sim._known = [sim._known_victims(i) - {k} for i in range(3)]
            sim.agents[1].assigned_target = k                     # 그룹 RESCUE UAV 가 이미 맡음
        ev = sim._collect_events([x.pos for x in sim.agents], set(), {})
        assert all("1" not in e for e in ev), (case, ev)
    print("[2] event 1: (a) exploration find -> whole group incl. RESCUE / RETURN (ev1_explore); (b) merge brings a "
          "victim one side did not know -> whole merged group (ev1_merge); (c) merge with the same known set -> none; "
          "(d) grown victims all expired / unreachable / already held -> none")


def check_goal_valid():
    out = {}
    for e in ("periodic", "1", "2", "3", "4"):
        sim, t0 = three_roles()
        sim.events = [{e}, {e}, {e}]
        sim._set_decide_order()
        out[e] = sim.goal_valid[1]                               # R1 RESCUE (목표 있음)
    assert out == {"periodic": True, "1": True, "2": False, "3": False, "4": False}, out
    sim, t0 = three_roles()
    sim.events = [set(), {"periodic"}, set()]
    sim._set_decide_order()
    ob = sim.rl_observation()["agents"][1]
    assert ob["role_self"][6] == 1.0 and t0 in ob["s_ids"].astype(int).tolist()    # 관측에 기존 목표 그대로
    print(f"[3] goal_valid: {out} (periodic / 1 keep the goal in the observation, 2 / 3 / 4 invalidate)")


def check_own_target_not_masked():
    sim = fresh(3)
    sim.time = 0
    place(sim, [(15, 10), (15, 14), (15, 30)])
    set_front(sim, line(2, 2, 6))
    a = add_victim(sim, (15, 12), deadline=100)
    b = add_victim(sim, (15, 32), deadline=100)
    tick(sim)
    sim._assign_rescue(0, a)
    sim._assign_rescue(2, b)
    tick(sim)
    sim.events = [{"periodic"}, {"periodic"}, set()]             # R0 (a 를 맡음) · R1 재결정, R2 (b 를 맡음) 는 아님
    sim._set_decide_order()
    assert a in sim.victim_slots[0] and sim.rescue_mask(0)[sim.victim_slots[0].index(a)], "자기 목표는 후보"
    assert a in sim.victim_slots[1], "재결정하는 다른 UAV 의 목표도 후보"
    assert b not in sim.victim_slots[0] and b not in sim.victim_slots[1], "재결정하지 않는 R2 의 목표는 막힘"
    rec = Recorder(sim, role={0: R, 1: E}, rescue={0: sim.victim_slots[0].index(a)})
    sv = rec.run()
    assert sim.agents[0].assigned_target == a and not sv["rescue"][0]       # 같은 생존자 -> 새 표본 아님
    print("[4] RESCUE re-decision: own target a stays a candidate (re-chosen), other re-deciders' targets are candidates, "
          "non-deciding R2's target b masked")


def check_same_target_option():
    sim = fresh(1)
    sim.time = 0
    place(sim, [(15, 10)])
    set_front(sim, line(2, 2, 6))
    a = add_victim(sim, (15, 20), deadline=200)
    b = add_victim(sim, (15, 4), deadline=200)
    tick(sim)
    sim.events = [{"periodic"}]
    sim._set_decide_order()
    sv = Recorder(sim, role={0: R}, rescue={0: sim.victim_slots[0].index(a)}).run()
    assert sv["rescue"][0]
    sim.role_rewards([], [])
    sim.time = 10
    tick(sim)
    sim.events = [{"periodic"}]
    sim._set_decide_order()
    sv = Recorder(sim, role={0: R}, rescue={0: sim.victim_slots[0].index(a)}).run()
    assert not sv["rescue"][0] and sim._ts_pending == [0.0], "같은 생존자 -> 표본 아님, 목표 전환 아님"
    assert sim.ev_stats["rescue_keep"] == 1 and sim.ev_stats["rescue_redecide"] == 1
    sim.role_rewards([], [])
    sim.time = 20
    tick(sim)
    sim.events = [{"periodic"}]
    sim._set_decide_order()
    sv = Recorder(sim, role={0: R}, rescue={0: sim.victim_slots[0].index(b)}).run()   # 유효한 a 를 두고 다른 생존자
    assert sv["rescue"][0] and sim._ts_pending == [1.0]
    print("[5] same survivor re-chosen: rescue_valid 0 (keep 1/1), no target switch; switching to another survivor while "
          "a is still valid -> new sample + voluntary target switch")


def check_cap0_explore():
    sim = fresh(2)
    sim.time = 0
    place(sim, [(15, 10), (15, 30)])
    set_front(sim, line(2, 2, 20))
    tick(sim)
    a = sim.agents[0]
    a.capacity = 0
    sim._assign_return(0)
    sim.events = [{"periodic"}, set()]
    sim._set_decide_order()
    m = sim.role_mask(0)
    assert m.tolist() == [True, False, True], m
    Recorder(sim, role={0: E}).run()
    assert a.mode is Mode.EXPLORE and a.blob is not None and sim.ev_stats["cap0_explore_choice"] == 1
    print(f"[6] capacity-0 UAV at periodic re-decision: role mask {m.astype(int).tolist()} (EXPLORE / RETURN) -> EXPLORE chosen, "
          f"blob C{a.blob}")


def check_rule_paths():
    from policies.role_rules import multi_event, rule_rescue
    sim = fresh(1)
    sim.time = 0
    place(sim, [(15, 10)])
    set_front(sim, line(2, 2, 6))
    near_late = add_victim(sim, (15, 12), deadline=200)
    far_early = add_victim(sim, (15, 25), deadline=60)
    tick(sim)
    m = sim.rescue_mask(0)
    picks = {}
    for e in ("periodic", "1", "2", "3", "4"):
        sim.events = [{e}]
        sim.cur_claim = {0: near_late}                            # 담당 후보 (그룹 그리디 차례가 정한 생존자)
        picks[e] = (multi_event(sim, 0), sim.victim_slots[0][rule_rescue(sim, 0, m)])
    assert all(picks[e] == (True, near_late) for e in ("periodic", "1")), picks
    assert all(picks[e] == (False, far_early) for e in ("2", "3", "4")), picks          # slack 45 < 198
    sim.events = [{"periodic", "2"}]
    assert not multi_event(sim, 0)                                # 섞이면 한 대 규칙
    print("[7] rule policy: periodic / 1 -> group greedy (claimed survivor), 2 / 3 / 4 -> single-UAV least slack; "
          "mixed causes -> single-UAV rule")


def _front_claim(sim):
    """role_front 를 D_0 = {UAV 0} 에 적용한 담당 후보 생존자 (env _decide 와 같은 입력)."""
    import torch

    from policies.decision_order import role_front, role_sigma
    X = sim._order_inputs()
    d0 = torch.zeros(1, sim.n_agents, dtype=torch.bool)
    d0[0, 0] = True
    sig = role_sigma(X["ids"], X["slack"], X["valid"], X["group"], d0)
    sel, claim = role_front(X["group"], d0, X["cap"], X["ids"], X["rd"], sig, X["valid"])
    assert bool(sel[0, 0]) and int(claim[0, 0]) >= 0, (sel, claim)
    return int(X["ids"][0, 0, int(claim[0, 0])])


def check_rule_slack():
    from policies.role_rules import rule_rescue
    sim = fresh(1)
    place(sim, [(1, 1)])
    wall = np.ones((ROWS, COLS), bool)                            # 뱀 모양 통로만 열림: 행 1 -> 행 3 -> 행 5
    wall[1, 1:39] = wall[2:4, 38] = wall[3, 1:39] = wall[4:6, 1] = wall[5, 1:39] = False
    sim.knowledge.known_obstacle[0][:] = wall
    sim.knowledge.known_free[0][:] = ~wall
    sim.time = 0
    tick(sim)
    far = tuple(int(x) for x in np.argwhere(sim.fields[0] == 80)[0])      # BFS 거리 80 인 칸 (통로 위)
    v1 = add_victim(sim, far, deadline=100)
    v2 = add_victim(sim, (1, 11), deadline=90)
    tick(sim)
    d = {v: sim._dist(0, sim.targets[v].pos) for v in (v1, v2)}
    assert d == {v1: 80, v2: 10}, d
    sim.events = [{"2"}]
    m = sim.rescue_mask(0)
    pick = sim.victim_slots[0][rule_rescue(sim, 0, m)]
    assert pick == v1, (pick, d)                                  # EDF (마감 빠른 쪽) 였다면 v2
    assert _front_claim(sim) == pick
    print(f"[8] single-UAV rule = least slack: v1 (tau 100, d {d[v1]}, slack 20) over v2 (tau 90, d {d[v2]}, slack 80) "
          f"-> v{pick + 1}; role_front (D_0 = {{A}}) claim = same")


def check_rule_slack_tie():
    from policies.role_rules import rule_rescue
    sim = fresh(1)
    sim.time = 0
    place(sim, [(15, 10)])
    far = add_victim(sim, (15, 20), deadline=60)                 # 번호 0, d 10, slack 50
    near = add_victim(sim, (15, 12), deadline=52)                # 번호 1, d 2,  slack 50 (더 가깝고 번호 큼)
    tick(sim)
    d = {v: sim._dist(0, sim.targets[v].pos) for v in (far, near)}
    assert d == {far: 10, near: 2} and (far, near) == (0, 1), d
    sim.events = [{"2"}]
    pick = sim.victim_slots[0][rule_rescue(sim, 0, sim.rescue_mask(0))]
    front = _front_claim(sim)
    assert pick == front == far, (pick, front)                   # (slack, 거리) 순이었다면 near
    print(f"[9] slack tie (50 = 50): rule_rescue -> survivor {pick} (d {d[far]}, lower id) over survivor {near} (d {d[near]}); "
          f"role_front (D_0 = {{A}}) claim = {front}")


if __name__ == "__main__":
    check_periodic()
    check_new_survivor_cases()
    check_goal_valid()
    check_own_target_not_masked()
    check_same_target_option()
    check_cap0_explore()
    check_rule_paths()
    check_rule_slack()
    check_rule_slack_tie()
    print("test_triggers: all ok")
