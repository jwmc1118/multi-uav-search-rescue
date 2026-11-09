"""역할 전환 penalty · 그룹 연결도 특징 점검 (작은 빈 맵, 몇 step — ray 불필요).

    python -m tests.role.test_reward_terms

[1] role 전환: periodic / "1" 에서 이전 역할이 허용되는데 바꾸면 role 보상 -w_sw / N_T (원인별 수), 이벤트 "2" / "3" / "4"
    (periodic 과 함께여도) 또는 이전 역할이 마스크로 막힘 -> 0 (강제), 역할은 같고 목표 (생존자) 만 바뀜 -> 0
[3] 특징: U 노드 특징 omega_group, 관측 team_omega = (|U_i| - 1) / (|U| - 1)
"""
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
    print(f"[3] U node feature omega_group; team_omega {list(map(float, obs['team_omega']))} = (|U_i|-1)/(|U|-1)")


if __name__ == "__main__":
    check_switch()
    check_feature()
    print("test_reward_terms: all ok")
