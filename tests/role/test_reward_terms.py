"""역할 전환 penalty · explore 직접 / 공유 분할 · 그룹 연결도 특징 점검 (작은 빈 맵, 몇 step — ray 불필요).

    python -m tests.role.test_reward_terms

[1] role 전환: periodic / "1" 에서 이전 역할이 허용되는데 바꾸면 role 보상 -w_sw / N_T (원인별 수), 이벤트 "2" / "3" / "4"
    (periodic 과 함께여도) 또는 이전 역할이 마스크로 막힘 -> 0 (강제), 역할은 같고 목표 (생존자) 만 바뀜 -> 0
[2] explore: 이번 step 직접 본 새 free 셀 -> r_direct (같은 그룹 안 동시 관측만 1/n_c 분할, 다른 그룹은 무관),
    통신 병합으로만 들어온 셀 -> r_share 에만, 그룹이 직접 본 셀은 공유로 안 셈, 다음 step 같은 셀은 다시 안 셈
[3] 특징: U 노드 특징 omega_group, 관측 team_omega = (|U_i| - 1) / (|U| - 1), rescue 보너스의 ω̄^g 가 같은 값
"""
import math

import numpy as np

from env.core.modes import Mode
from tests.role.test_events_targeted import Recorder, add_victim, fresh
from tests.role.test_explore import connect, line, place, set_front, tick

E, R, D = Mode.EXPLORE.value, Mode.RESCUE.value, Mode.RETURN.value


def scene():
    """UAV 1 대, 생존자 2 명, frontier 한 줄 (role 보상에서 발견 보상을 빼면 팀 항 0 + 전환 항만)."""
    sim = fresh(1)
    place(sim, [(15, 10)])
    set_front(sim, line(2, 2, 6))
    a = add_victim(sim, (15, 15), deadline=200)
    b = add_victim(sim, (15, 25), deadline=200)
    tick(sim)
    return sim, a, b


def switch(prev, ev, role, cap=None, target=None):
    """이전 역할 prev 에서 원인 ev 로 재결정해 role 을 고름 (RESCUE 면 target 생존자). 반환 (role 보상 - 발견 보상, sim)."""
    sim, a, b = scene()
    u = sim.agents[0]
    u.mode = prev
    if prev is Mode.RESCUE:
        sim._assign_rescue(0, a)
    if cap is not None:
        u.capacity = cap
    sim.time = 10
    sim.events = [set(ev)]
    sim._set_decide_order()
    slot = sim.victim_slots[0].index(target if target is not None else a)
    Recorder(sim, role={0: role}, rescue={0: slot}).run()
    r = sim.role_rewards([], [])["role"][0]
    return r - sim.last_discover, sim


def check_switch():
    sim, _, _ = scene()
    pen = -sim.rewards_cfg["role"]["w_sw"] / len(sim.targets)
    r, s = switch(Mode.EXPLORE, {"periodic"}, R)
    assert abs(r - pen) < 1e-12 and s.rw_stats["sw_voluntary_periodic"] == 1 and abs(s.rw["role_switch"][0] - r) < 1e-12, r
    r, s = switch(Mode.EXPLORE, {"1"}, R)
    assert abs(r - pen) < 1e-12 and s.rw_stats["sw_voluntary_ev1"] == 1, r
    for ev in ({"2"}, {"3"}, {"4"}, {"periodic", "2"}):
        r, s = switch(Mode.EXPLORE, ev, R)
        assert abs(r) < 1e-12 and s.rw_stats["sw_forced"] == 1 and s.agents[0].mode is Mode.RESCUE, (ev, r)
    r, s = switch(Mode.RETURN, {"periodic"}, E, cap=sim.agents[0].capacity_max)    # 가득 찬 용량 -> RETURN 마스크
    assert abs(r) < 1e-12 and s.rw_stats["sw_forced"] == 1 and s.agents[0].mode is Mode.EXPLORE, r
    _, a, b = scene()
    r, s = switch(Mode.RESCUE, {"periodic"}, R, target=b)                          # 역할 같음, 생존자만 바뀜
    assert abs(r) < 1e-12 and s.agents[0].assigned_target == b and s.rw_stats["sw_forced"] == 0, r
    print(f"[1] role switch: periodic / '1' with previous role allowed -> {pen:.4f} (= -w_sw/N_T); events 2 / 3 / 4 "
          f"(also with periodic), previous role masked -> 0 (forced); same role, new survivor -> 0")


def check_explore():
    sim = fresh(3)
    n = 3
    sim.comp, sim.groups = [0, 0, 1], [[0, 1], [2]]                      # 그룹 {0, 1} / {2}
    A, B, C, X, M, O = (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)
    sim.obstacles[O[0]][O[1]] = True
    prev = [np.zeros((sim.rows, sim.cols), bool) for _ in range(n)]
    prev[0][X] = True                                                    # UAV 0 은 X 를 이미 앎
    prev[1][M] = True                                                    # UAV 1 은 M 을 이미 앎 (병합으로 0 에 감)
    sim._prev_free = [p.copy() for p in prev]
    vis = [{A, B, O}, {B, C}, {B}]                                       # O = 장애물
    sim.knowledge.last_visible = vis
    union = prev[0] | prev[1]
    for c in (A, B, C, X):
        union[c] = True
    sim.knowledge.known_free = [union.copy(), union.copy(), np.zeros_like(union)]
    sim.knowledge.known_free[2][B] = True
    direct, share = sim.explore_cells()
    # 0: A 1 + B 1/2 (0, 1 이 봄, 다른 그룹의 2 는 무관), 장애물 O 제외 / 1: B 1/2 + C 1 / 2: B 1 (혼자)
    assert direct == [1.5, 1.5, 1.0], direct
    # 0: M 만 (C 는 그룹이 직접 봄) / 1: X 만 (0 이 예전에 안 셀, A 는 그룹이 직접 봄, M 은 이미 앎) / 2: 없음
    assert share == [1.0, 1.0, 0.0], share
    n_free = float(sim.n_free_cells())
    w_share = sim.rewards_cfg["explore"]["w_share"]
    mult = [1 + sim.rewards_cfg["explore"]["w_omega"] * w for w in sim.group_connectivity()]
    out = sim.role_rewards([], [], explore_active=[True, True, False])
    assert all(abs(out["explore"][i] - (mult[i] * direct[i] + w_share * share[i]) / n_free) < 1e-15 for i in (0, 1))
    assert out["explore"][2] == 0.0 and abs(sim.rw["explore_share"][0] - w_share / n_free) < 1e-15
    sim._prev_free = [f.copy() for f in sim.knowledge.known_free]       # 다음 step: 같은 셀을 다시 봄
    assert sim.explore_cells() == ([0.0] * n, [0.0] * n)
    print(f"[2] explore: direct {direct} (split 1/n_c inside the group only), merge-only cells -> share {share} "
          f"(group-seen cells excluded), same cells next step -> 0")


def check_feature():
    from env.role.graph_obs import FU, U_FEATURES
    assert FU == len(U_FEATURES) and "omega_group" in U_FEATURES
    sim = fresh(3)
    place(sim, [(10, 5), (10, 7), (25, 35)])
    connect(sim, [[0, 1], [2]])
    tick(sim)
    want = [(len(sim.groups[sim.comp[i]]) - 1) / (sim.n_agents - 1) for i in range(3)]
    assert want == [0.5, 0.5, 0.0], want
    obs = sim.rl_observation()
    assert np.allclose(obs["team_omega"], want), obs["team_omega"]
    sim.targets = []
    add_victim(sim, (10, 9))
    sim.agents[0].mode = Mode.RESCUE
    sim._res_opt[0] = {"F": 0, "d": 0.0, "wsum": 0.0, "n": 0}
    sim.role_rewards([], [])                                             # 옵션 1 step: ω^g 누적
    o = sim._res_opt[0]
    assert o["n"] == 1 and o["wsum"] == want[0], o
    rc = sim.rewards_cfg["rescue"]
    bonus = sim._option_bonus(0, True)
    expect = rc["w_b"] * ((1 - rc["w_c"]) * math.exp(0.0) + rc["w_c"] * want[0]) / 1
    assert abs(bonus - expect) < 1e-12, (bonus, expect)
    print(f"[3] U node feature omega_group; team_omega {list(map(float, obs['team_omega']))} = (|U_i|-1)/(|U|-1); "
          f"rescue bonus uses the same ω̄^g {want[0]} -> {bonus:.3f}")


if __name__ == "__main__":
    check_switch()
    check_explore()
    check_feature()
    print("test_reward_terms: all ok")
