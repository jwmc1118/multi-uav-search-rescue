"""보상 함수 시나리오 점검 (ray 클러스터 불필요) — 빈 맵에 UAV / 생존자를 직접 놓고 env 보상 경로를 호출.

    python -m tests.role.test_rewards

 [4] 중복 추적: 그룹 밖의 더 가까운 UAV 가 같은 생존자를 쫓으면 먼 쪽만 -0.05 (결정당 1 회)
 [5] 역할 진동: A -> B -> A (10 step 안, 비강제) 면 -0.03 / RESCUE -> RETURN (용량 0) -> RESCUE (충전 직후) 는 penalty 없음
 [6] 에피소드 종료: 미구조 생존자 = 마지막 step 의 만료 X
 [7] role GAE: 결정 간격 k step 이면 γ^k (할인 누적과 bootstrap 모두)
 [9] S_eff (그리디 2 단계 Hungarian 비용 κ): 다른 UAV 구간과 겹치는 셀 중 그 UAV 가 BFS 로 더 가까운 셀만 뺀다 (동률은 유지)
[12] 결정 없는 RESCUE step 은 rescue 보상 0, 보상 항목 = REWARD_ITEMS
[13] U_ε: σ < 0 인 생존자 · G_res 밖 UAV (다른 그룹) 가 맡은 생존자는 빠짐
[14] 학습: rescue 는 옵션 합 (할인 없음, 다음 role / rescue 결정에서 끝) A = R - V / explore 는 옵션 return (합류 재배정에서는 안 끝남) (길이 3 -> a + γb + γ²c, 역할 변경 · 다음 explore
     결정에서 끝, bootstrap 없음, fragment 끝에서 열린 옵션은 마스크 0)
[16] 연결성 C_pair = Σ_k n_k (n_k - 1) / (N (N - 1)): N = 5 의 연결 요소 배치 5 / 4+1 / 3+2 / 3+1+1 / 2+2+1 / 2+1+1+1 / 1×5
     -> 1.0 / 0.6 / 0.4 / 0.3 / 0.2 / 0.1 / 0.0, N = 1 -> 0, N = 2 연결 -> 1 / 끊김 -> 0, 간선 = 거리 <= R_c (경계 포함)
[17] role step 보상 = (|R_t| - w_x |X_t|) / N_T - c_osc 1_osc 정확히 (연결성 · 커버리지 항 없음)
[18] role SMDP 할인 = config 최상위 gamma (rewards.role.gamma 없음)
[21] Δcov: 두 UAV 가 같은 새 셀을 동시에 보면 각 1/2, Σ_i Δcov = 처음 관측된 free 셀 수 (실제 센싱으로도)
[22] explore 스텝 보상 r = (1 - ψ^f) Δcov / |free| (1 + w_mult c) (EXPLORE 인 UAV 만), ψ^f = 1 이면 0
[23] rescue S - F (N_T = 4): 급한 1 + 다른 1 할당 -> S 2, F 0, 보너스 / 급한 애 빼고 2 -> S 2, F 1, 보너스 없음 /
     급한 3 명 · UAV 2 대 · 급한 2 명 할당 -> S 2, F 1 (F_unavoidable 1)
[24] 그룹 기준 EXPLORE 마스크: 그룹 belief 발견 수 < N_T (실제로는 다 찾음) -> EXPLORE 허용, = N_T -> 금지 + 그 그룹
     EXPLORE UAV 에 이벤트 5 (exploration complete), 모든 역할이 막히면 RETURN 허용 (forced 기록)
"""
import numpy as np

from env.core.modes import Mode
from tests.role.test_events_targeted import Recorder, add_victim, fresh
from tests.role.test_explore import connect, line, place, set_front, tick

E, R, D = Mode.EXPLORE.value, Mode.RESCUE.value, Mode.RETURN.value


def one_uav():
    sim = fresh(1)
    sim.time = 0
    place(sim, [(15, 10)])
    set_front(sim, line(2, 2, 6))
    a = add_victim(sim, (15, 15), deadline=5 + 3)              # d 5,  σ = 3
    b = add_victim(sim, (15, 25), deadline=15 + 20)            # d 15, σ = 20
    c = add_victim(sim, (5, 10), deadline=10 + 25)             # d 10, σ = 25
    tick(sim)
    return sim, (a, b, c)


def check_dup():
    sim = fresh(2)
    sim.time = 0
    place(sim, [(15, 5), (15, 30)])
    connect(sim, [[0], [1]])                                     # 다른 그룹 (서로 모름)
    set_front(sim, line(2, 2, 6))
    v = add_victim(sim, (15, 25), deadline=100)                  # UAV 1 이 더 가까움 (5 vs 20)
    tick(sim)
    assert sim.comp[0] != sim.comp[1]
    sim._res_pending = [0.0, 0.0]
    for i in (0, 1):
        sim._assign_rescue(i, v)
    sim._rescue_event_reward(sim.comp[0], [0])
    sim._rescue_event_reward(sim.comp[1], [1])
    assert sim._res_pending == [-0.05, 0.0], sim._res_pending
    sim._check_dup()                                             # 같은 결정에서 다시 검사해도 추가 penalty 없음
    assert sim._res_pending == [-0.05, 0.0] and sim.rw_stats["dup_out"] == 1
    print("[4] duplicate pursuit across groups: farther UAV R0 gets -0.05 once (dup_out 1), closer R1 none")


def check_oscillation():
    sim, (a, b, c) = one_uav()
    u = sim.agents[0]
    u.mode = Mode.EXPLORE
    sim.time = 10
    sim.events = [{"2"}]
    sim._set_decide_order()
    Recorder(sim, role={0: R}).run()                             # EXPLORE -> RESCUE (비강제)
    assert sim._osc_pen[0] == 0.0
    sim.time = 15
    sim.events = [{"2"}]
    sim._set_decide_order()
    Recorder(sim, role={0: E}).run()                             # RESCUE -> EXPLORE (비강제, 5 step 안) -> 진동
    assert abs(sim._osc_pen[0] - 0.03) < 1e-12, sim._osc_pen
    r = sim.role_rewards([], [])["role"][0]
    assert abs(r - (-0.03)) < 1e-12, r
    # 강제: RESCUE -> RETURN (용량 0, RESCUE 마스크) -> RESCUE (충전 직후 = 이벤트 4)
    sim, (a, b, c) = one_uav()
    u = sim.agents[0]
    sim._assign_rescue(0, a)
    u.capacity = 0
    sim.time = 20
    sim.events = [{"3"}]
    sim._set_decide_order()
    Recorder(sim, role={0: D}).run()
    assert u.mode is Mode.RETURN
    u.capacity = u.capacity_max
    u.pos = sim.depot
    tick(sim)
    sim.time = 24
    sim.events = [{"4"}]
    sim._set_decide_order()
    Recorder(sim, role={0: R}).run()
    assert u.mode is Mode.RESCUE and sim._osc_pen[0] == 0.0 and sim.rw_stats["forced"] == 2 and sim.rw_stats["osc"] == 0
    print("[5] role oscillation: EXPLORE -> RESCUE -> EXPLORE within 5 steps (voluntary) -> r_role -0.03; "
          "RESCUE -> RETURN (cap 0, forced) -> RESCUE (right after refill, forced) -> no penalty")


def check_terminal():
    sim, (a, b, c) = one_uav()
    n_t = len(sim.targets)
    left = sum(1 for t in sim.targets if t.active)
    r_end = sim.role_rewards([], [], done=True)["role"][0]
    assert abs(r_end - (-2.0 * left / n_t)) < 1e-12, r_end
    sim.rewards_cfg["role"]["terminal_unrescued_as_expired"] = False
    assert sim.role_rewards([], [], done=True)["role"][0] == 0.0
    print(f"[6] episode end with {left} unrescued -> counted as expired: r_role = -w_x · {left} / N_T = {r_end:.3f} (flag off -> 0)")


def check_smdp_gae():
    from training.role.ppo import decision_advantages
    g, k = 0.99, 7
    T = 20
    r = np.zeros(T, np.float32)
    r[:k] = 1.0                                                    # 첫 결정 구간 (k step) 매 step 보상 1
    v = np.zeros(T, np.float32)
    v[0], v[k] = 0.5, 2.0
    dec = np.zeros(T, bool)
    dec[0] = dec[k] = True
    adv, vt, ok = decision_advantages(r, v, dec, dec, True, g, lam=0.0)
    want = sum(g ** t for t in range(k)) + g ** k * 2.0 - 0.5
    assert ok[0] == 1 and abs(adv[0] - want) < 1e-5, (adv[0], want)
    adv1, _, _ = decision_advantages(r, v, dec, dec, True, g, lam=1.0)
    want1 = want + g ** k * 1.0 * adv[k]
    assert abs(adv1[0] - want1) < 1e-5
    adv2, _, ok2 = decision_advantages(r, v, dec, dec, False, g, lam=0.95)
    assert ok2[k] == 0 and ok2[0] == 1, "fragment 끝에서 미완료 구간은 제외"
    print(f"[7] role SMDP GAE: decision gap k = {k} -> δ = Σ γ^t r + γ^{k} V(s_t1) - V(s_t0) = {want:.4f}; "
          f"GAE uses γ^{k} λ Â(t1); unfinished last segment masked when truncated")


def kappa_sim(n=1):
    sim = fresh(n)
    set_front(sim, line(2, 2, 6))
    return sim


def set_seg(sim, i, cells):
    a = sim.agents[i]
    a.mode, a.cluster = Mode.EXPLORE, frozenset(cells)


def check_s_eff():
    sim = kappa_sim(2)
    place(sim, [(15, 10), (15, 30)])
    connect(sim, [[0], [1]])                                     # 다른 그룹 — S_eff 는 env 전체 기준
    tick(sim)
    s0 = [(15, c) for c in range(12, 26)]                        # 14 칸, d0 = 2..15
    s1 = [(15, c) for c in range(20, 29)]                        # 9 칸,  d1 = 10..2
    set_seg(sim, 0, s0)
    set_seg(sim, 1, s1)
    k0, e0, n0 = sim.kappa_cells(0, sim.agents[0].cluster)
    k1, e1, n1 = sim.kappa_cells(1, sim.agents[1].cluster)
    # 겹침 (15, 20..25): (15, 20) 은 d0 = d1 = 10 동률 -> 유지, (15, 21..25) 는 R1 이 더 가까움 -> R0 에서 빠짐
    assert (e0, n0) == (9, 14) and abs(k0 - 3 / 10) < 1e-12, (e0, n0, k0)
    assert (e1, n1) == (9, 9) and abs(k1 - 3 / 10) < 1e-12, (e1, n1, k1)
    sim.agents[1].mode = Mode.RESCUE                              # EXPLORE 가 아니면 겹침으로 안 침
    assert sim.kappa_cells(0, sim.agents[0].cluster)[1] == 14
    print(f"[9] S_eff (greedy κ cost): R0 segment 14 cells overlaps R1 (other group) on 6 cells, R1 closer on 5 (1 tie kept) -> "
          f"|S_eff| {e0} (κ {k0:.2f}); R1 keeps all {e1}; R1 not EXPLORE -> R0 |S_eff| 14")


def check_rescue_no_spread():
    from env.role.simulation import REWARD_ITEMS
    sim = fresh(2)
    for a in sim.agents:
        a.comm_range = 20
    v = add_victim(sim, (5, 5))
    place(sim, [(10, 10), (10, 24)])                              # 이웃 거리 0.7 R_c (예전 분산 항 최대)
    tick(sim)
    sim._assign_rescue(0, v)
    sim._dup_done = [True, True]
    r = sim.role_rewards([], [])["rescue"]
    assert r == [0.0, 0.0] and tuple(sim.rw) == REWARD_ITEMS and "rescue_spread" not in sim.rw, (r, list(sim.rw))
    print(f"[12] rescue without spread: RESCUE UAV with a neighbour at 0.7 R_c, no decision -> r_rescue {r[0]}; "
          f"reward items {list(sim.rw)}")


def layout(sizes, R=10.0):
    """연결 요소 크기 목록 -> 위치 (요소 안 = R 간격 일렬, 요소 사이 = 10 R)."""
    pts = []
    for k, n in enumerate(sizes):
        pts += [(100.0 * k * R, j * R) for j in range(n)]
    return pts


def check_pair_connectivity():
    from env.role.connectivity import pair_connectivity
    R = 10.0
    cases = {(5,): 1.0, (4, 1): 0.6, (3, 2): 0.4, (3, 1, 1): 0.3, (2, 2, 1): 0.2, (2, 1, 1, 1): 0.1, (1, 1, 1, 1, 1): 0.0}
    for sizes, want in cases.items():
        got = pair_connectivity(layout(sizes, R), R)
        assert abs(got - want) < 1e-6, (sizes, got, want)
    assert pair_connectivity([(0, 0)], R) == 0.0 and pair_connectivity([], R) == 0.0
    assert pair_connectivity([(0, 0), (0, R)], R) == 1.0                    # 경계 (= R_c) 포함
    assert pair_connectivity([(0, 0), (0, R + 1e-6)], R) == 0.0
    assert pair_connectivity([(0, 0), (0, 8)], [10, 7]) == 0.0              # R_c = 두 반경 중 작은 쪽
    print("[16] C_pair: " + ", ".join(f"{'+'.join(map(str, k))} -> {v}" for k, v in cases.items())
          + "; N=1 -> 0, N=2 linked -> 1 / cut -> 0, R_c = min of the pair")


def check_role_gamma():
    """compute_role_gae 의 role 경로가 config 최상위 gamma 를 쓰는지 (결정 t=0, 다음 결정 t=3, 보상 t=1, λ = 0)."""
    from types import SimpleNamespace
    from ray.rllib.policy.sample_batch import SampleBatch
    from env.role.simulation import REWARDS_DEFAULTS
    from training.role.ppo import ADV, VF, compute_role_gae
    assert "gamma" not in REWARDS_DEFAULTS["role"]
    n, T = 1, 5
    infos = [{"reward_role": [1.0 if t == 1 else 0.0], "reward_rescue": [0.0], "reward_explore": [0.0],
              "role_valid": [float(t in (0, 3))], "rescue_valid": [0.0], "explore_valid": [0.0],
              "role_train": [float(t in (0, 3))], "rescue_train": [0.0], "explore_train": [0.0]} for t in range(T)]
    vf = np.full((T, n), 2.0, np.float32)
    out = {}
    for g in (0.5, 0.9):
        batch = SampleBatch({SampleBatch.INFOS: np.array(infos, dtype=object), SampleBatch.REWARDS: np.zeros(T, np.float32),
                             SampleBatch.DONES: np.zeros(T, bool), VF["role"]: vf.copy(), VF["rescue"]: vf.copy(),
                             VF["explore"]: vf.copy()})
        pol = SimpleNamespace(model=SimpleNamespace(n_agents=n),
                              config={"gamma": g, "lambda": 0.0, "env_config": {"rewards": {"role": {}}}})
        out[g] = float(compute_role_gae(pol, batch)[ADV["role"]][0, 0])
        want = g * 1.0 + g ** 3 * 2.0 - 2.0
        assert abs(out[g] - want) < 1e-6, (g, out[g], want)
    print(f"[18] role SMDP uses config gamma: γ 0.5 -> Â {out[0.5]:.4f}, γ 0.9 -> Â {out[0.9]:.4f} (= γ r + γ^3 V - V)")


def check_topm_excludes_held():
    """U_ε 에서 σ < 0 · G_res 밖 UAV (다른 그룹) 가 맡은 생존자는 빠진다."""
    sim = fresh(3)
    sim.time = 0
    place(sim, [(15, 10), (15, 20), (25, 35)])
    connect(sim, [[0, 1], [2]])
    set_front(sim, line(2, 2, 6))
    u = add_victim(sim, (15, 13), deadline=6)                    # σ = 6 - 3 = 3 -> 급함
    neg = add_victim(sim, (15, 25), deadline=3)                  # min d 5 -> σ = -2 (못 닿음) -> 제외
    held = add_victim(sim, (16, 10), deadline=4)                 # σ = 3 이지만 다른 그룹 R2 가 맡음 -> 제외
    far = add_victim(sim, (5, 10), deadline=100)                 # σ = 90 -> 급하지 않음
    tick(sim)
    assert sim.comp[0] == sim.comp[1] != sim.comp[2]
    sim._assign_rescue(2, held)
    urgent, sig, _ = sim.rescue_urgent(sim.comp[0], [0, 1])
    assert urgent == [u] and sig[neg] < 0 and held not in sig and sig[far] > 5, (urgent, sig)
    print(f"[13] U_eps = {urgent}: sigma<0 (T{neg}, sigma {sig[neg]}) and other-group-held (T{held}) excluded, "
          f"non-urgent T{far} (sigma {sig[far]}) excluded")


def gae_batch(rows, n=1):
    """rows: dict 키 -> (T, n) 리스트. 나머지 info 키는 0."""
    from ray.rllib.policy.sample_batch import SampleBatch
    from training.role.ppo import VF
    T = len(next(iter(rows.values())))
    keys = ["reward_role", "reward_rescue", "reward_explore", "role_valid", "rescue_valid", "explore_valid",
            "explore_reselect", "explore_active", "role_train", "rescue_train", "explore_train"]
    infos = [{k: list(rows.get(k, [[0.0] * n] * T)[t]) for k in keys} for t in range(T)]
    vf = np.full((T, n), 0.5, np.float32)
    return SampleBatch({SampleBatch.INFOS: np.array(infos, dtype=object), SampleBatch.REWARDS: np.zeros(T, np.float32),
                        SampleBatch.DONES: np.zeros(T, bool), VF["role"]: vf.copy(), VF["rescue"]: vf.copy(),
                        VF["explore"]: vf.copy()})


def check_explore_one_shot():
    """compute_role_gae: rescue = 결정 step 의 A = r - V / explore = 옵션 return (bootstrap 없음)."""
    from types import SimpleNamespace
    from training.role.ppo import ADV, PMASK, VMASK, VT, compute_role_gae
    g = 0.9
    pol = SimpleNamespace(model=SimpleNamespace(n_agents=1), config={"gamma": g, "lambda": 0.95, "env_config": {}})
    col = lambda v: [[x] for x in v]
    a, b, c = 0.3, 0.2, 0.1
    # (1) 결정 t=1, EXPLORE t=1..3, t=4 역할 변경 (active 0) -> R = a + γb + γ²c, V(s_t4) bootstrap 없음
    rows = {"explore_valid": col([0, 1, 0, 0, 0, 0]), "explore_train": col([0, 1, 0, 0, 0, 0]),
            "explore_active": col([0, 1, 1, 1, 0, 0]), "reward_explore": col([0, a, b, c, 0, 0]),
            "rescue_valid": col([1, 0, 0, 0, 0, 0]), "rescue_train": col([1, 0, 0, 0, 0, 0]),
            "reward_rescue": col([0.7, 0, 0.1, 0, 0.5, 0]), "role_valid": col([0, 0, 0, 1, 0, 0])}   # 옵션 t 0..2 (t 3 재결정)
    bt = compute_role_gae(pol, gae_batch(rows))
    R = a + g * b + g * g * c
    assert abs(bt[VT["explore"]][1, 0] - R) < 1e-6 and abs(bt[ADV["explore"]][1, 0] - (R - 0.5)) < 1e-6
    assert bt[PMASK["explore"]].sum() == 1 and bt[VMASK["explore"]].sum() == 1
    assert abs(bt[ADV["rescue"]][0, 0] - 0.3) < 1e-6 and bt[PMASK["rescue"]].sum() == 1            # rescue: 0.7 + 0.1 - 0.5
    # (2) 다음 explore 결정 (t=3) 에서 끝 -> R(1) = a + γb, 두 번째 옵션 (t=3, 끝까지 EXPLORE) 은 fragment 끝에서 열림 -> 마스크 0
    rows2 = {"explore_valid": col([0, 1, 0, 1, 0, 0]), "explore_train": col([0, 1, 0, 1, 0, 0]),
             "explore_active": col([0, 1, 1, 1, 1, 1]), "reward_explore": col([0, a, b, c, 0.05, 0.05])}
    bt2 = compute_role_gae(pol, gae_batch(rows2))
    assert abs(bt2[VT["explore"]][1, 0] - (a + g * b)) < 1e-6
    assert bt2[PMASK["explore"]][3, 0] == 0 and bt2[PMASK["explore"]].sum() == 1
    # (3) 합류 재배정 (explore_reselect, Hungarian — 정책 결정 아님) 은 옵션을 끝내지 않는다 (역할 변경 t=4 에서 끝)
    rows3 = dict(rows, explore_reselect=col([0, 0, 1, 0, 0, 0]))
    bt3 = compute_role_gae(pol, gae_batch(rows3))
    assert abs(bt3[VT["explore"]][1, 0] - R) < 1e-6
    print(f"[14] rescue option sum (0.7 + 0.1, ends at next role decision) A = R - V; explore option return a + γb + γ²c = {R:.4f} (ends on role change, no bootstrap), "
          f"ends at next explore decision (not at a join re-assignment), open option at fragment end masked")


def check_role_reward_exact():
    """role step 보상 = (|R| - w_x |X|) / N_T - c_osc 1_osc 정확히 — 위치 (연결성) · 새 관측 셀 (커버리지) 과 무관."""
    sim = fresh(3)
    for k in range(4):
        add_victim(sim, (5, 5 + 3 * k))
    tick(sim)
    w_x = sim.rewards_cfg["role"]["w_x"]
    out = []
    for pos, new in (([(5, 5), (5, 8), (5, 11)], [set(), set(), set()]),
                     ([(5, 5), (25, 30), (35, 5)], [{(1, 1)}, {(1, 1), (2, 2)}, set()])):
        place(sim, pos)
        sim.knowledge.new_sets = new
        sim._osc_pen = [0.0, 0.03, 0.0]
        r = sim.role_rewards([object()], [object()])["role"]
        want = [(1 - w_x * 1) / 4 - p for p in (0.0, 0.03, 0.0)]
        assert all(abs(x - y) < 1e-12 for x, y in zip(r, want)), (r, want)
        out.append(r)
    assert out[0] == out[1] and "role_conn" not in sim.rw and "w_conn" not in sim.rewards_cfg["role"]
    print(f"[17] role step reward = (|R| - w_x|X|)/N_T - c_osc 1_osc = {out[0]} (unchanged by positions / coverage)")


def check_coverage_split():
    sim = fresh(2)
    sim.knowledge.new_sets = [{(3, 3)}, {(3, 3)}]
    c = sim.explore_coverage()
    assert c == [0.5, 0.5], c
    # 실제 센싱: 두 UAV 가 겹쳐 서서 지도를 모두 지운 뒤 한 번 update
    K = sim.knowledge
    place(sim, [(10, 10), (10, 12)])
    for i in range(2):
        K.known_free[i][:] = False
        K.known_obstacle[i][:] = False
    K.update(())
    c = sim.explore_coverage()
    new_free = {x for cells in K.new_sets for x in cells if not sim.obstacles[x[0]][x[1]]}
    shared = set.intersection(*K.new_sets)
    assert abs(sum(c) - len(new_free)) < 1e-9 and len(shared) > 0, (sum(c), len(new_free))
    print(f"[21] coverage split: same new cell seen by 2 -> [0.5, 0.5]; real sensing: Σ dcov {sum(c):.1f} = first-seen free "
          f"cells {len(new_free)} ({len(shared)} shared)")


def check_explore_step_reward():
    sim = fresh(2)
    for k in range(4):
        add_victim(sim, (5, 5 + 3 * k))
    sim.targets[0].discovered_at = None                           # 3 / 4 발견 -> ψ^f = 0.75
    sim.knowledge.new_sets = [{(1, 1), (1, 2)}, {(1, 2)}]
    n_free = float(sim.n_free_cells())
    r = sim.role_rewards([], [], explore_active=[True, False])["explore"]
    assert abs(sim.psi_f() - 0.75) < 1e-12
    mult = 1 + sim.rewards_cfg["explore"]["w_mult"] * sim.personal_connectivity()[0]
    assert abs(r[0] - 0.25 * 1.5 / n_free * mult) < 1e-15 and r[1] == 0.0, r   # R1 은 EXPLORE 아님 -> 0
    sim.targets[0].discovered_at = 0                              # ψ^f = 1 -> 0
    r1 = sim.role_rewards([], [], explore_active=[True, True])["explore"]
    assert r1 == [0.0, 0.0], r1
    print(f"[22] explore step reward (1 - psi_f) dcov / |free| (1 + w_mult c): psi_f 0.75, dcov 1.5, mult {mult:.2f} -> {r[0]:.3e}, non-EXPLORE UAV 0; "
          f"psi_f 1 -> {r1}")


def sf_sim(victims):
    """UAV 0 (15, 10), UAV 1 (15, 20) 한 그룹, victims [(pos, deadline)] (N_T = len)."""
    sim = fresh(2)
    sim.time = 0
    place(sim, [(15, 10), (15, 20)])
    set_front(sim, line(2, 2, 6))
    ids = [add_victim(sim, p, deadline=d) for p, d in victims]
    tick(sim)
    return sim, ids


def sf_reward(sim, picks, c=(1.0, 1.0)):
    """결정 step 에 (S - F) / N_T, 옵션을 바로 닫아 (에피소드 끝) 개인 보너스 (c̄ = c)."""
    sim._sf_pending, sim._bonus_pending = [0.0] * sim.n_agents, [0.0] * sim.n_agents
    for i, ti in picks.items():
        sim._assign_rescue(i, ti)
    sim._rescue_event_reward(sim.comp[0], list(picks))
    assert sim._bonus_pending == [0.0] * sim.n_agents                  # 보너스는 옵션이 끝날 때
    sim._rescue_options(list(c), [], True)
    st = sim.rw_stats
    return st["res_S"][-1], st["res_F"][-1], sim._sf_pending[0], sim._bonus_pending[0], st["res_F_unavoidable"][-1]


def check_rescue_sf():
    V = [((15, 13), 6), ((15, 30), 100), ((5, 10), 100), ((25, 10), 100)]       # u (σ 3), n1, n2, n3 (σ 90)
    sim, (u, n1, n2, n3) = sf_sim(V)
    S, F, sf, bonus, _ = sf_reward(sim, {0: u, 1: n1})
    assert (S, F) == (2, 0) and abs(sf - 2 / 4) < 1e-12 and bonus > 0, (S, F, sf, bonus)
    sim, (u, n1, n2, n3) = sf_sim(V)
    S, F, sf, bonus2, _ = sf_reward(sim, {0: n2, 1: n1})
    assert (S, F) == (2, 1) and abs(sf - 1 / 4) < 1e-12 and bonus2 == 0.0, (S, F, sf, bonus2)
    sim, (u, n1, n2, n3) = sf_sim(V)
    assert sf_reward(sim, {0: n2, 1: n1}, c=(0.0, 0.0))[3] == 0.0       # F > 0 -> c̄ 와 무관하게 보너스 0
    V3 = [((15, 13), 6), ((15, 17), 6), ((15, 15), 8), ((15, 30), 100)]      # 급한 3 명 (σ 3 / 3 / 3)
    sim, (u1, u2, u3, n1) = sf_sim(V3)
    S, F, sf, bonus3, unav = sf_reward(sim, {0: u1, 1: u2})
    assert (S, F, unav) == (2, 1, 1) and abs(sf - 1 / 4) < 1e-12 and bonus3 == 0.0, (S, F, unav, sf, bonus3)
    print(f"[23] rescue S - F (N_T 4): urgent + other -> S 2 F 0 r_SF 0.5 + bonus {bonus:.3f}; without urgent -> S 2 F 1 "
          f"r_SF 0.25, no bonus; 3 urgent / 2 UAVs -> S 2 F 1 (F_unavoidable {unav})")


def check_explore_mask():
    sim = fresh(3, complete=True)                                 # exploration complete 판정 켬
    place(sim, [(10, 5), (10, 20), (25, 35)])
    connect(sim, [[0, 1], [2]])
    set_front(sim, line(2, 2, 30))
    ids = [add_victim(sim, (20, 5 + 5 * k)) for k in range(3)]
    for i in (0, 1, 2):                                           # 아무 그룹도 T2 를 모름 (전역으로는 3 / 3 발견)
        sim.knowledge.known_free[i][sim.targets[ids[2]].pos] = False
    tick(sim)
    gA = sim.comp[0]
    assert sim.psi_f() == 1.0 and sim.group_discovered(gA) == 2 and not sim.exploration_complete(gA)
    assert sim.role_mask(0)[Mode.EXPLORE.value], "그룹이 다 모르면 EXPLORE 허용"
    for i in (0, 1):
        sim.knowledge.known_free[i][sim.targets[ids[2]].pos] = True
    sim.agents[1].mode = Mode.RESCUE
    sim.agents[1].assigned_target = ids[0]
    tick(sim)
    assert sim.exploration_complete(gA) and not sim.role_mask(0)[Mode.EXPLORE.value]
    ev = sim._collect_events([a.pos for a in sim.agents], set(), {})
    # 그룹 A 의 EXPLORE UAV (R0) 만 — R1 은 RESCUE, 그룹 B (R2) 는 아직 T2 를 모름
    assert "5" in ev[0] and "5" not in ev[1] and "5" not in ev[2] and sim.ev_stats["ev5_causes"]["exploration_complete"] == 1, ev
    ev2 = sim._collect_events([a.pos for a in sim.agents], set(), {})            # 처음 도달할 때 한 번만
    assert "5" not in ev2[0]
    # 모든 역할이 막힘: 용량 가득 (RETURN 없음) + 할당 가능 생존자 없음 + exploration complete -> RETURN 허용 (forced)
    a = sim.agents[0]
    a.capacity = a.capacity_max
    for t in sim.targets:
        t.rescued_at = 0
    tick(sim)
    m = sim.role_mask(0)
    assert m.tolist() == [False, False, True] and sim._mask_flags == (True, True), (m, sim._mask_flags)
    from policies.role_rules import rule_role
    assert rule_role(sim, 0, m) == Mode.RETURN.value
    print("[24] EXPLORE mask: group belief 2/3 (all found globally) -> EXPLORE allowed; 3/3 -> masked + event 5 "
          "(exploration complete) for the group's EXPLORE UAV once; all roles masked -> RETURN (forced)")


if __name__ == "__main__":
    check_dup()
    check_oscillation()
    check_terminal()
    check_smdp_gae()
    check_s_eff()
    check_rescue_no_spread()
    check_topm_excludes_held()
    check_explore_one_shot()
    check_pair_connectivity()
    check_role_reward_exact()
    check_role_gamma()
    check_coverage_split()
    check_explore_step_reward()
    check_rescue_sf()
    check_explore_mask()
    print("test_rewards: all ok")
