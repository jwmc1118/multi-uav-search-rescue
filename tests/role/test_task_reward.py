"""task 보상 점검 (작은 빈 맵, 사건 하나씩 — ray 불필요).

    python -m tests.role.test_task_reward

전역 집합 (S 전체, F 발견, R 구조, E 만료) 기준. EXPLORE 비율 u = |S \ F| / (N_T - |E \ F|), RESCUE 대기 수 = |F \ (R ∪ E)|.
[1] 미발견 생존자 발견 -> u 감소
[2] 미발견 생존자 만료 -> u 증가 (분모만 줄어듦)
[3] 발견 후 미구조 만료 -> u 불변, RESCUE 발견 후 만료 항 -w_x / N_T 한 번
[4] 구조 -> u 불변, RESCUE 대기 수 감소
"""
from types import SimpleNamespace


from env.core.modes import Mode
from tests.role.test_events_targeted import add_victim, fresh
from tests.role.test_explore import line, place, set_front, tick


def scene():
    """UAV 0 EXPLORE, UAV 1 RESCUE (한 그룹), 생존자 4 명: a, b 미발견 / c, d 발견."""
    sim = fresh(2)
    place(sim, [(15, 10), (15, 30)])
    set_front(sim, line(2, 2, 6))
    ids = [add_victim(sim, p, deadline=300, discovered=d)
           for p, d in (((5, 5), False), ((5, 9), False), ((15, 32), True), ((25, 20), True))]
    tick(sim)
    sim.agents[0].mode = Mode.EXPLORE
    sim._assign_rescue(1, ids[2])
    return sim, ids


def step(sim, rescued=(), expired=()):
    """한 step 의 task 보상 항목 (role_rewards 경로)과 그 뒤 (미발견 비율, 대기 수)."""
    before = {k: v.copy() for k, v in sim.rw.items()}
    sim.role_rewards(list(rescued), list(expired))
    u, w, _ = sim.task_global()
    return {k: sim.rw[k] - before[k] for k in sim.rw if k.startswith("task_")}, u, round(w * len(sim.targets))


def check_events():
    sim, (a, b, c, d) = scene()
    n_t = len(sim.targets)
    w_x = float(sim.rewards_cfg["task"]["w_x"])
    _, u0, w0 = step(sim)
    assert (u0, w0) == (2 / 4, 2), (u0, w0)
    sim.targets[a].discovered_at = sim.time                              # [1] 발견
    _, u1, w1 = step(sim)
    assert u1 < u0 and (u1, w1) == (1 / 4, 3), (u1, w1)
    sim.targets[b].expired_at = sim.time                                 # [2] 미발견 만료
    t2, u2, _ = step(sim, expired=[b])
    assert abs(u2 - 1 / 3) < 1e-12 and u2 > u1 and t2["task_found_expire"][1] == 0.0, (u2, t2)   # 1 / (4 - 1) > 1 / 4
    sim.targets[d].expired_at = sim.time                                 # [3] 발견 후 미구조 만료
    _, u_pre, _ = step(sim)
    t3, u3, _ = step(sim, expired=[d])
    assert u3 == u_pre and abs(t3["task_found_expire"][1] + w_x / n_t) < 1e-12 and t3["task_found_expire"][0] == 0.0, t3
    _, _, w_pre = step(sim)
    sim.targets[c].rescued_at = sim.time                                 # [4] 구조
    t4, u4, w4 = step(sim, rescued=[SimpleNamespace(agent=1, target=c)])
    assert u4 == u3 and w4 == w_pre - 1, (u4, u3, w4, w_pre)
    print(f"[1] found: u {u0:.3f} -> {u1:.3f}  [2] undiscovered expired: u {u1:.3f} -> {u2:.3f} (denominator 4 -> 3)  "
          f"[3] found survivor expired: u unchanged, RESCUE found_expire {t3['task_found_expire'][1]:+.3f} (= -w_x/N_T)  "
          f"[4] rescue: u unchanged, wait {w_pre} -> {w4}")


if __name__ == "__main__":
    check_events()
    print("test_task_reward: all ok")
