"""부분 귀환 (용량이 남아도 RETURN) 점검 (ray 불필요).

    python -m tests.role.test_return_mask

 [1] 역할 마스크 = 표 (return_with_cap: true): cap 0 -> RETURN, EXPLORE / 0 < cap < cap_max 이고 부족분 >= return_min_deficit
     -> RESCUE, EXPLORE, RETURN / 부족분 < return_min_deficit -> RESCUE, EXPLORE / cap_max -> RESCUE, EXPLORE
 [2] return_with_cap: false -> 기존 마스크 (cap 0 일 때만 RETURN)
 [3] depot 에서 충전 직후 RETURN 막힘 (반복 귀환 없음) + 이벤트 4
"""
import numpy as np

from env.core.modes import Mode
from tests.role.test_events_targeted import add_victim, fresh
from tests.role.test_explore import line, place, set_front, tick

E, R, D = Mode.EXPLORE.value, Mode.RESCUE.value, Mode.RETURN.value


def roles(m):
    return {name for name, k in (("EXPLORE", E), ("RESCUE", R), ("RETURN", D)) if m[k]}


def setup(**over):
    sim = fresh(2, **over)
    place(sim, [(14, 3), (25, 30)])
    set_front(sim, line(15, 5, 34))
    add_victim(sim, (5, 5))
    tick(sim)
    return sim


def check_table():
    sim = setup(role={"return_with_cap": True, "return_min_deficit": 1})
    a = sim.agents[0]
    rows = []
    for cap, want in ((0, {"RETURN", "EXPLORE"}), (1, {"RESCUE", "EXPLORE", "RETURN"}), (2, {"RESCUE", "EXPLORE", "RETURN"}),
                      (3, {"RESCUE", "EXPLORE"})):
        a.capacity = cap
        got = roles(sim.role_mask(0))
        assert got == want, (cap, got, want)
        rows.append(f"cap {cap}: {sorted(got)}")
    sim = setup(role={"return_with_cap": True, "return_min_deficit": 2})
    sim.agents[0].capacity = 2                                     # 부족분 1 < 2
    assert roles(sim.role_mask(0)) == {"RESCUE", "EXPLORE"}
    sim.agents[0].capacity = 1                                     # 부족분 2 >= 2
    assert roles(sim.role_mask(0)) == {"RESCUE", "EXPLORE", "RETURN"}
    # 할당 가능 생존자 / 덩어리가 없으면 그 역할 막힘, 전부 막히면 RETURN
    sim = setup()
    sim.targets = []
    set_front(sim, np.zeros_like(sim.knowledge.frontier[0]))
    tick(sim)
    sim.agents[0].capacity = 3
    assert roles(sim.role_mask(0)) == {"RETURN"}, "전부 막히면 RETURN fallback"
    print("[1] role mask table (cap_max 3, min_deficit 1): " + " / ".join(rows) +
          "; min_deficit 2: cap 2 -> no RETURN, cap 1 -> RETURN; nothing available -> RETURN fallback")


def check_legacy_mask():
    sim = setup(role={"return_with_cap": False})
    a = sim.agents[0]
    got = {}
    for cap in (0, 1, 3):
        a.capacity = cap
        got[cap] = roles(sim.role_mask(0))
    assert got == {0: {"RETURN", "EXPLORE"}, 1: {"RESCUE", "EXPLORE"}, 3: {"RESCUE", "EXPLORE"}}, got
    print(f"[2] return_with_cap false -> old mask {{0: RETURN/EXPLORE, 1 and 3: RESCUE/EXPLORE}}")


def check_refill():
    sim = setup()
    a = sim.agents[0]
    a.capacity = 1
    sim._assign_return(0)
    prev = [(sim.depot[0], sim.depot[1] + 1), sim.agents[1].pos]
    a.pos = sim.depot
    sim._process_refill(None)
    sim._prepare_role()
    assert a.capacity == a.capacity_max
    ev = sim._collect_events(prev, set(), {})
    assert "4" in ev[0], ev
    m = sim.role_mask(0)
    assert not m[D], "충전 직후 RETURN 이 열려 있음 (반복 귀환)"
    print(f"[3] RETURN arrival at depot -> refill to cap_max {a.capacity}, event 4; right after refill mask {sorted(roles(m))} "
          f"(RETURN blocked)")


if __name__ == "__main__":
    check_table()
    check_legacy_mask()
    check_refill()
    print("test_return_mask: all ok")
