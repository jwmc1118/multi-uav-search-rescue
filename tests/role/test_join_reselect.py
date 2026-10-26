"""합류 재선택 (이번 step 누군가 고른 덩어리는 기존 배정자도 2 단계를 다시 고름) 점검.

    python -m tests.role.test_join_reselect

 [1] 합류: A (왼쪽 끝) + A2 (오른쪽 끝) 가 덩어리 C 탐색 중, B 가 C 를 고름 -> A, B, A2 전원이 seg_front 차례 (남은 가장
     큰 구간에 가장 가까운 참여자) 로 2 단계, 후보 2 개 이상인 앞 두 차례만 정책 호출 (기존 배정자 = stage-2-only),
     마지막 차례는 후보 1 개라 결정 아님 (explore 보상은 매 step 커버리지 — 결정 sample 만 확인)
 [2] rule 모드: 합류 덩어리는 κ 헝가리안 (기존 배정자 goal 구간 d = 0) -> A / A2 가 자기 goal 쪽 구간, goal 그대로, κ 품질 (exp(-κ)) 큰 하락 없음
 [3] 합류가 없는 덩어리의 기존 배정자: 결정 없음, 매칭으로 goal 구간 유지
 [4] m 감소 (배정자가 떠남): 정책 호출 없이 매 step rebalance 매칭으로 처리, 남은 배정자 goal 유지
 [5] 참여자 2 명 · 구간 2 개: 두 번째 차례는 후보 1 개 -> 결정 아님 (explore_valid_s2 0)
 [6] 학습 정책 rollout: 모델의 stage-2-only 플래그 = env explore_reselect, 그 sample 의 logp = logp2 (s1 0, s2 1),
     새 UAV 는 logp1 + logp2, loss 에 들어가 NaN 없음 (n_decisions_explore_s2only > 0)
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


def decide_join(sim, sub=None):
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
    assert rec.calls["sub"] == order[:2], (rec.calls["sub"], order)     # 후보 3 / 2 개인 앞 두 차례만 정책
    assert int(rec.masks[order[0]].sum()) == 3 and int(rec.masks[order[1]].sum()) == 2, rec.masks
    keep_dec = [j for j in order[:2] if j != 1]
    assert sv["explore_reselect"] == [j in keep_dec for j in range(3)] and sv["explore_s1"] == [False, True, False]
    assert sv["explore_s2"] == [j in order[:2] for j in range(3)]
    segs = [sim.agents[i].cluster for i in range(3)]
    assert not (segs[0] & segs[1]) and not (segs[0] & segs[2]) and not (segs[1] & segs[2]), "1:1"
    st = sim.ev_stats
    assert st["join_events"] == 1 and st["reselect_uavs"] == 2                 # 기존 배정자 A, A2
    print(f"[1] join: all three take part; stage-2 order {order} (largest open segment -> nearest participant), policy "
          f"calls {rec.calls['sub']} (candidates 3 / 2, last 1 -> no decision); stage-2-only {keep_dec}")


def check_rule_join():
    sim = join_scene()
    rec, sv, f0, goal0 = decide_join(sim, sub=lambda s, i, m: RuleDecider().explore_sub(s, i, m))
    for l in (0, 2):
        a = sim.agents[l]
        assert a.goal == goal0[l] and tuple(goal0[l]) in a.cluster, (l, goal0[l], a.goal)
    f1 = {l: kf(sim, l) for l in (0, 2)}
    drop = {l: f0[l] - f1[l] for l in (0, 2)}
    assert all(d < 0.15 for d in drop.values()), (f0, f1)
    assert sim.ev_stats["reselect_kept_goal"] == 2 and sim.ev_stats["reselect_goal_change"] == 0
    print(f"[2] rule join (κ Hungarian, own-goal segment d = 0): A / A2 keep their goal side, goals unchanged; "
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
    rec = SubRec(sim, role={0: E})
    sv = rec.run()
    order = [j for st, j in sim.decision_seq if st == "s2"]
    first, second = order
    assert rec.calls["sub"] == [first] and not sv["explore_s2"][second], (order, rec.calls)
    if second == 1:                                                   # 기존 배정자가 두 번째면 재선택 sample 없음
        assert not sv["explore_reselect"][1]
    print(f"[5] two participants, two segments: order {order}, only the first chooses by policy; second has one "
          f"candidate left -> no sample (explore_valid_s2 0)")


def check_model():
    import torch
    from tests.role.smoke_role_policy import base_config, make
    from training.catalog import register_all
    register_all()
    env, prep, policy = make(base_config(mode=("learned", "learned", "learned")))
    N = env.n_agents
    n_rs = n_full = mism = 0
    worst = 0.0
    rows = []
    for seed in range(6):
        env.seed(seed)
        obs = env.reset()
        for t in range(300):
            ob = prep.transform(obs)
            actions, _, extra = policy.compute_actions(ob[None], explore=True)
            act = {k: [int(np.asarray(a)[0]) for a in v] for k, v in actions.items()}
            st = policy.model.last_out["stage"]
            rs_m = st["reselect"][0].cpu().numpy()
            inp = torch.as_tensor(extra["action_dist_inputs"])
            dist = policy.dist_class(inp, policy.model)
            ta = {k: torch.as_tensor(np.stack([np.asarray(a) for a in v], -1)).reshape(1, N) for k, v in actions.items()}
            ch = dist._children()
            lp1 = torch.stack([ch["explore"][i].logp(ta["explore"][:, i]) for i in range(N)], -1)[0]
            lp2 = torch.stack([ch["explore_sub"][i].logp(ta["explore_sub"][:, i]) for i in range(N)], -1)[0]
            lp = dist.logp(ta)["explore"][0]
            for i in range(N):
                if rs_m[i]:
                    n_rs += 1
                    assert dist.s1[0, i] == 0 and dist.s2[0, i] == 1
                    worst = max(worst, float(abs(lp[i] - lp2[i])))
                elif dist.s1[0, i] > 0 and dist.s2[0, i] > 0:
                    n_full += 1
                    worst = max(worst, float(abs(lp[i] - lp1[i] - lp2[i])))
            obs, _, done, info = env.step(act)
            mism += int(np.any(rs_m != (np.asarray(info["explore_reselect"]) > 0)))
            if done:
                break
    assert mism == 0 and n_rs > 0 and worst < 1e-5, (mism, n_rs, worst)
    print(f"[6a] learned rollouts: model stage-2-only flag == env explore_reselect at every step; {n_rs} stage-2-only "
          f"samples (logp = logp2), {n_full} full samples (logp1 + logp2), max |Δlogp| {worst:.1e}")
    from tests.role.smoke_role_policy import collect
    import tests.role.smoke_role_policy as sm
    sm.STEPS = 600
    for seed in range(10):
        batch = collect(env, prep, policy, seed=seed)
        if (np.asarray(batch["explore_train_mask"]) > 0).any() and any(
                any(x > 0 for x in inf["explore_reselect"]) for inf in batch["infos"]):
            break
    stats = policy.learn_on_batch(batch)["learner_stats"]
    assert stats["n_decisions_explore_s2only"] > 0 and stats["explore_logp_finite"] == 1.0, stats
    assert all(np.isfinite(v) for k, v in stats.items() if isinstance(v, (float, np.floating))), stats
    print(f"[6b] learn_on_batch: stage-2-only samples {stats['n_decisions_explore_s2only']:.0f} "
          f"(share {stats['explore_s2only_ratio']:.3f}), total loss {stats['total_loss']:.3f}, all stats finite")


if __name__ == "__main__":
    check_join()
    check_rule_join()
    check_no_join_keeper()
    check_m_decrease()
    check_single_candidate()
    check_model()
    print("test_join_reselect: all ok")
