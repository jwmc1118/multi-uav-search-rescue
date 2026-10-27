"""합류 재배정 (이번 step 누군가 고른 덩어리는 기존 배정자도 2 단계 구간을 다시 받음) 점검 — 2 단계는 항상 κ Hungarian.

    python -m tests.role.test_join_reselect

 [1] 합류: A (왼쪽 끝) + A2 (오른쪽 끝) 가 덩어리 C 탐색 중, B 가 C 를 고름 -> A, B, A2 전원이 Hungarian 으로 1:1 구간 배정,
     explore 결정 (학습 sample) 은 B 의 1 단계 하나 (기존 배정자의 재배정은 결정 아님)
 [2] 합류 덩어리 κ 헝가리안 (기존 배정자 goal 구간 d = 0) -> A / A2 가 자기 goal 쪽 구간, goal 그대로, κ 품질 (exp(-κ)) 큰 하락 없음
 [3] 합류가 없는 덩어리의 기존 배정자: 결정 없음, 매칭으로 goal 구간 유지
 [4] m 감소 (배정자가 떠남): 정책 호출 없이 매 step rebalance 매칭으로 처리, 남은 배정자 goal 유지
 [5] 참여자 2 명 · 구간 2 개: 두 번째 차례는 후보 1 개 -> Hungarian 호출 없이 남은 구간
 [6] 학습 정책 rollout: explore 결정 = 1 단계뿐 (explore_valid == explore_valid_s1), 합류 재배정된 기존 배정자는 결정 · 학습
     sample 이 아니고 (explore_valid 0) explore 옵션이 계속된다 (explore_active 1)
"""
import numpy as np

from env.core.modes import Mode
from policies.role_rules import RuleDecider
from tests.role.test_events_targeted import Recorder, fresh
from tests.role.test_explore import line, place, set_front, tick

E = Mode.EXPLORE.value


def kf(sim, l):
    """그리디 κ 비용 품질 exp(-κ) (κ = (d + 1) / (|S_eff| + 1), 구간 없으면 0) — 보상 아님, 분할 품질 확인용."""
    k = sim.kappa_cells(l, sim.agents[l].cluster) if sim.agents[l].cluster else None
    return float(np.exp(-k[0])) if k is not None else 0.0


class SubRec(Recorder):
    def __init__(self, sim, sub=None, **kw):
        super().__init__(sim, **kw)
        self.sub_fn, self.masks = sub, {}

    def sub(self, i, m):
        self.calls["sub"].append(i)
        self.masks[i] = m.copy()
        return self.sub_fn(self.sim, i, m) if self.sub_fn else int(np.flatnonzero(m)[0])


def join_scene():
    """A = 0 (왼쪽 끝), B = 1 (RETURN, 합류 예정), A2 = 2 (오른쪽 끝) — 한 그룹, frontier 한 줄 (덩어리 C)."""
    sim = fresh(3)
    place(sim, [(14, 3), (20, 20), (14, 36)])
    set_front(sim, line(15, 5, 34))
    tick(sim)
    sim.agents[1].mode = Mode.RETURN
    for i in (0, 2):
        sim.agents[i].blob = sim.clusters_of(i)[0].id
    tick(sim)                                                        # rebalance: A / A2 가 양쪽 절반
    return sim


HUNGARIAN = lambda s, i, m: RuleDecider().explore_sub(s, i, m)


def decide_join(sim, sub=HUNGARIAN):
    sim.events = [set(), {"4"}, set()]
    sim._set_decide_order()
    f0 = {l: kf(sim, l) for l in (0, 2)}
    goal0 = {l: sim.agents[l].goal for l in (0, 2)}
    rec = SubRec(sim, sub=sub, role={1: E})
    sv = rec.run()
    return rec, sv, f0, goal0


def check_join():
    sim = join_scene()
    assert sim.agents[0].goal[1] < 20 and sim.agents[2].goal[1] > 20
    rec, sv, f0, _ = decide_join(sim)
    b = sim.agents[0].blob
    assert sim.agents[1].blob == b and sim.agents[2].blob == b
    order = [j for st, j in sim.decision_seq if st == "s2"]
    assert sorted(order) == [0, 1, 2], order                             # 기존 배정자 A, A2 도 참여
    assert sv["explore_s1"] == [False, True, False]                      # explore 결정 = B 의 1 단계 하나
    segs = [sim.agents[i].cluster for i in range(3)]
    assert not (segs[0] & segs[1]) and not (segs[0] & segs[2]) and not (segs[1] & segs[2]), "1:1"
    st = sim.ev_stats
    assert st["join_events"] == 1 and st["reselect_uavs"] == 2                 # 기존 배정자 A, A2 재배정
    print(f"[1] join: all three re-assigned 1:1 by Hungarian (order {order}); the only explore decision is B's stage 1")


def check_rule_join():
    sim = join_scene()
    rec, sv, f0, goal0 = decide_join(sim)
    for l in (0, 2):
        a = sim.agents[l]
        assert a.goal == goal0[l] and tuple(goal0[l]) in a.cluster, (l, goal0[l], a.goal)
    f1 = {l: kf(sim, l) for l in (0, 2)}
    drop = {l: f0[l] - f1[l] for l in (0, 2)}
    assert all(d < 0.15 for d in drop.values()), (f0, f1)
    assert sim.ev_stats["reselect_kept_goal"] == 2 and sim.ev_stats["reselect_goal_change"] == 0
    print(f"[2] join (κ Hungarian, own-goal segment d = 0): A / A2 keep their goal side, goals unchanged; "
          f"exp(-κ) A {f0[0]:.3f}->{f1[0]:.3f}, A2 {f0[2]:.3f}->{f1[2]:.3f}; B gets the middle")


def check_no_join_keeper():
    sim = fresh(2)
    place(sim, [(14, 3), (25, 30)])
    set_front(sim, line(15, 5, 20) | line(24, 25, 38))                # 덩어리 2 개
    tick(sim)
    a = sim.agents[0]
    a.blob = next(b.id for b in sim.clusters_of(0) if (15, 5) in b.cell_set())
    tick(sim)
    g0, seg0 = a.goal, a.cluster
    sim.events = [set(), {"1"}]
    sim._set_decide_order()
    slot = next(k for k, b in enumerate(sim.clusters_of(1)) if (24, 30) in b.cell_set())
    rec = SubRec(sim, role={1: E}, explore={1: slot})
    sv = rec.run()
    assert sim.agents[1].blob != a.blob and 0 not in rec.calls["sub"] and not sv["explore_reselect"][0]
    assert a.goal == g0 and a.cluster == seg0 and not sv["explore_s1"][0] and not sv["explore_s2"][0] and sv["explore_s1"][1]
    print("[3] no join (B picks another blob): keeper A no decision, keeps segment and goal")


def check_m_decrease():
    sim = join_scene()
    g0 = sim.agents[0].goal
    calls = []
    sim._stage2_all = lambda *a, **k: calls.append(1)                 # 정책 경로가 불리면 안 됨
    sim.agents[2].mode, sim.agents[2].blob = Mode.RETURN, None        # A2 가 떠남 (m 2 -> 1)
    tick(sim)
    blob = sim.blob_index[sim.comp[0]][sim.agents[0].blob]
    assert not calls and set(sim.agents[0].cluster) == set(sim._blob_cells_for_split(sim.comp[0], blob))
    assert sim.agents[0].goal == g0
    sim.agents[2].mode, sim.agents[2].blob = Mode.EXPLORE, sim.agents[0].blob
    sim.agents[1].mode, sim.agents[1].blob = Mode.EXPLORE, sim.agents[0].blob
    tick(sim)                                                         # m 1 -> 3 (rebalance 는 합류 결정 없이 매칭만)
    sim.agents[1].mode, sim.agents[1].blob = Mode.RETURN, None
    g2 = sim.agents[2].goal
    tick(sim)                                                         # m 3 -> 2
    assert not calls and sim.agents[0].goal == g0 and sim.agents[2].goal == g2
    print("[4] m decrease (assignee leaves): handled by per-step rebalance matching, no policy call, remaining goals kept")


def check_single_candidate():
    sim = fresh(2)
    place(sim, [(20, 20), (14, 36)])                                  # B = 0 (합류), A = 1 (오른쪽 끝)
    set_front(sim, line(15, 5, 34))
    tick(sim)
    sim.agents[0].mode = Mode.RETURN
    sim.agents[1].blob = sim.clusters_of(1)[0].id
    tick(sim)
    sim.events = [{"4"}, set()]
    sim._set_decide_order()
    rec = SubRec(sim, sub=HUNGARIAN, role={0: E})
    sv = rec.run()
    order = [j for st, j in sim.decision_seq if st == "s2"]
    first, second = order
    assert rec.calls["sub"] == [first] and not sv["explore_s2"][second], (order, rec.calls)
    assert sim.agents[0].cluster and sim.agents[1].cluster and not (sim.agents[0].cluster & sim.agents[1].cluster)
    print(f"[5] two participants, two segments: order {order}, Hungarian called for the first only; second takes the "
          f"remaining segment")


def check_model():
    from tests.role.smoke_role_policy import base_config, make
    from training.catalog import register_all
    register_all()
    env, prep, policy = make(base_config(mode=("learned", "learned", "learned")))
    n_re = n_s1 = 0
    for seed in range(6):
        env.seed(seed)
        obs = env.reset()
        for t in range(300):
            actions, _, _ = policy.compute_actions(prep.transform(obs)[None], explore=True)
            act = {k: [int(np.asarray(a)[0]) for a in v] for k, v in actions.items()}
            obs, _, done, info = env.step(act)
            assert info["explore_valid"] == info["explore_valid_s1"], info["explore_valid"]
            n_s1 += int(sum(info["explore_valid_s1"]))
            for i, r in enumerate(info["explore_reselect"]):
                if r > 0:
                    n_re += 1
                    assert info["explore_valid"][i] == 0 and info["explore_train"][i] == 0 and info["explore_active"][i] == 1
            if done:
                break
    assert n_re > 0 and n_s1 > 0, (n_re, n_s1)
    print(f"[6] learned rollouts: explore decisions = stage 1 only ({n_s1}); {n_re} join re-assignments are not decisions "
          f"(explore_valid 0) and keep the explore option (explore_active 1)")


if __name__ == "__main__":
    check_join()
    check_rule_join()
    check_no_join_keeper()
    check_m_decrease()
    check_single_candidate()
    check_model()
    print("test_join_reselect: all ok")
