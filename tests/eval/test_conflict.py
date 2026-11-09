"""할당 충돌 지표 점검 (eval_common/episode.py ConflictTracker + env _log_rescue_choice) — 작은 빈 맵 시나리오, ray 불필요.

    python -m tests.eval.test_conflict

UAV 0 은 생존자 V 바로 옆, UAV 1 은 멀리 (통신 범위 밖 = 다른 그룹), 미발견 생존자 하나 더 (에피소드가 안 끝나게). 같은 step 의 이벤트에서 둘 다 V 를 RESCUE 로 고르게 한다.
[1] 동시 추격: 두 번째로 고른 UAV 의 결정 1 회 (첫 step, 다른 그룹), 같은 그룹 0, 새 구조 목표 결정 수 = info rescue_valid 합
[2] UAV 0 이 V 를 구조한 뒤 UAV 1 은 모르고 계속 가다 목표가 무효가 되면 헛걸음 1 회, 낭비 step = 그동안 UAV 1 이 실제로 움직인
    step 수 (테스트가 위치로 따로 셈)
[3] 보상 기준 충돌 (_check_dup): 다른 그룹 UAV 는 메모리 기록이 없으므로 0 (전역 기준 동시 추격과 다른 판정임을 확인)
"""
from env.core.modes import Mode
from eval_common.episode import ConflictTracker
from policies.role_rules import RuleDecider
from tests.role.test_events_targeted import add_victim, fresh
from tests.role.test_explore import connect, place, tick

R = Mode.RESCUE.value


class ChaseV(RuleDecider):
    """V 를 고를 수 있으면 RESCUE · V, 아니면 규칙 정책."""

    def __init__(self, v):
        super().__init__()
        self.v = v

    def role(self, sim, i, m):
        return R if m[R] and self.v in sim.victim_slots[i] else super().role(sim, i, m)

    def rescue(self, sim, i, m):
        s = sim.victim_slots[i]
        return s.index(self.v) if self.v in s and m[s.index(self.v)] else super().rescue(sim, i, m)


def main():
    sim = fresh(2)
    place(sim, [(5, 5), (25, 35)])                       # 거리 > comm_range 10 -> 다른 그룹
    v = add_victim(sim, (5, 7), deadline=300)
    add_victim(sim, (28, 2), deadline=300, discovered=False)   # 미발견 생존자 — V 구조로 에피소드가 끝나지 않게
    connect(sim, [[0], [1]])
    tick(sim)
    assert sim.comp[0] != sim.comp[1], sim.comp
    sim.events = [{"1"}, {"1"}]
    sim._set_decide_order()
    dec = ChaseV(v)
    tr = ConflictTracker(sim)
    moved1, chasing1, step_conc, n_res = 0, True, [], 0
    for _ in range(80):
        tr.before(sim)
        p1 = sim.agents[1].pos
        done, info = sim.step(dec)
        n_res += int(sum(info["rescue_valid"]))
        step_conc.append(tr.after(sim))
        ctx = sim.last_ctx
        on_v = ctx.chosen_mode[1] is Mode.RESCUE and ctx.chosen_target[1] == v
        if chasing1 and on_v:
            moved1 += int(sim.agents[1].pos != p1)
        chasing1 = chasing1 and on_v
        if done or (not chasing1 and sim.targets[v].rescued_at is not None):
            break
    m = tr.metrics(sim)
    assert step_conc[0] == 1 and sum(step_conc) == 1, step_conc
    assert (m["conflict_concurrent"], m["conflict_concurrent_cross_group"], m["conflict_concurrent_same_group"]) == (1, 1, 0), m
    assert m["rescue_decisions"] == n_res >= 2 and m["conflict_concurrent_per_dec"] == 1 / n_res, (m, n_res)
    print(f"[1] concurrent chase: {m['conflict_concurrent']} (cross {m['conflict_concurrent_cross_group']}, same "
          f"{m['conflict_concurrent_same_group']}), rescue decisions {m['rescue_decisions']}")
    t = sim.targets[v]
    assert t.rescued_by == 0 and not chasing1, (t.rescued_by, chasing1)
    assert m["wasted_trips"] == 1 and m["wasted_steps"] == moved1 > 0, (m, moved1)
    print(f"[2] UAV 0 rescued V at t={t.rescued_at}; UAV 1 wasted trip {m['wasted_trips']}, wasted steps {m['wasted_steps']} "
          f"(= {moved1} moves counted from positions)")
    assert m["conflict_reward"] == 0, m
    print(f"[3] reward-based conflict (_check_dup, group + memory): {m['conflict_reward']}")
    print("test_conflict: all ok")


if __name__ == "__main__":
    main()
