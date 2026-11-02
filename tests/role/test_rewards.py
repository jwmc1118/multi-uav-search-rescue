"""보상 함수 시나리오 점검 (ray 클러스터 불필요) — 빈 맵에 UAV / 생존자를 직접 놓고 env 보상 경로를 호출.

    python -m tests.role.test_rewards

 [4] 중복 추적: 그룹 밖의 더 가까운 UAV 가 같은 생존자를 쫓으면 먼 쪽만 -0.05 (결정당 1 회)
 [6] 에피소드 종료: 미구조 생존자 = 마지막 step 의 만료 E (플래그 끄면 만료 항 없음)
 [7] role GAE: 결정 간격 k step 이면 γ^k (할인 누적과 bootstrap 모두)
[12] 결정 없는 RESCUE step 은 rescue 보상 0, 보상 항목 = REWARD_ITEMS
[13] U_ε: σ < 0 인 생존자 · G_res 밖 UAV (다른 그룹) 가 맡은 생존자는 빠짐
[14] 학습: rescue 는 옵션 합 (할인 없음, 다음 rescue 결정 / RESCUE 아님 에서 끝) A = R - V / explore 는 옵션 return (합류 재배정에서는
     안 끝남) (길이 3 -> a + γb + γ²c, 역할 변경 · 다음 explore 결정에서 끝, bootstrap 없음, fragment 끝에서 열린 옵션은 마스크 0)
[16] 연결성 C_pair = Σ_k n_k (n_k - 1) / (N (N - 1)): N = 5 의 연결 요소 배치 5 / 4+1 / 3+2 / 3+1+1 / 2+2+1 / 2+1+1+1 / 1×5
     -> 1.0 / 0.6 / 0.4 / 0.3 / 0.2 / 0.1 / 0.0, N = 1 -> 0, N = 2 연결 -> 1 / 끊김 -> 0, 간선 = 거리 <= R_c (경계 포함)
[17] role step 보상 - 발견 보상 = (|ΔR| - w_x |ΔE|) / N_T 정확히, 모든 UAV 같은 값, 위치와 무관
[18] role SMDP 할인 = role_gamma (model.custom_model_config.role_gamma), explore 옵션 할인 = config gamma
[19] 실제 rollout 에서 매 step role 보상 - 그 UAV 의 전환 penalty 가 모든 UAV 같음 (팀 공통 부분)
[22] explore 스텝 보상 = ((1 + w_omega ω^g) direct + w_share share) / |C_free| (EXPLORE 인 UAV 만, 발견 수와 무관),
     explore 보상 항목 = explore_direct, explore_share (직접 / 병합 셀 분할은 tests/role/test_reward_terms.py)
[23] rescue (n_suc_i - F_excess) / N_T: n_suc · d 는 UAV 개인 기준, F_excess 는 G_res 전원 같은 값, 보너스는 구조 완료로 끝난 옵션에만
     w_b ((1 - w_c) e^{-d_i} + w_c ω̄^g_i) / N_T (에피소드 끝 · F_excess > 0 이면 0)
[24] 그룹 기준 EXPLORE 마스크: 그룹 belief 발견 수 < N_T (실제로는 다 찾음) -> EXPLORE 허용, = N_T -> 금지 + 그 그룹
     EXPLORE UAV 에 이벤트 "4" (exploration complete), 모든 역할이 막히면 RETURN 허용 (forced 기록)
[28] 늦지 않은 발견 보상 (w_d / N_T) Σ_{k ∈ ΔF_t} 1[τ_k - t > T_m]: 남은 골든타임 > T_m 이면 +w_d / N_T, = T_m · 이미 만료는 0,
     rw_stats discovered / discovered_timely, 모든 UAV 같은 값
[30] F_excess = F - max(0, |U_ε| - |G|): |G| 1 · 급한 2 명 (둘 다 닿음) -> 급한 쪽 F_excess 0 + 보너스 / 급하지 않은 쪽
     F_excess 1, 보너스 0; |G| 2 · 급한 2 명 -> F_min 0; random seed 0-19 F_excess >= 0
"""
import numpy as np

from env.core.modes import Mode
from tests.role.test_events_targeted import add_victim, fresh
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


def check_terminal():
    sim, (a, b, c) = one_uav()
    n_t = len(sim.targets)
    left = sum(1 for t in sim.targets if t.active)
    r_end = sim.role_rewards([], [], done=True)["role"][0]
    assert abs(r_end - (-2.0 * left / n_t + sim.last_discover)) < 1e-12, r_end
    sim.rewards_cfg["role"]["terminal_unrescued_as_expired"] = False
    r_off = sim.role_rewards([], [], done=True)["role"][0]
    assert abs(r_off - sim.last_discover) < 1e-12, r_off
    print(f"[6] episode end with {left} unrescued -> counted as expired: r_role = -w_x · {left} / N_T = {r_end:.3f} "
          f"(flag off -> no expiry term)")


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
    """compute_role_gae: role 경로는 role_gamma (model.custom_model_config.role_gamma), explore 는 config gamma."""
    from types import SimpleNamespace
    from ray.rllib.policy.sample_batch import SampleBatch
    from training.role.ppo import ADV, VF, compute_role_gae
    n, T = 1, 5
    infos = [{"reward_role": [1.0 if t == 1 else 0.0], "reward_rescue": [0.0], "reward_explore": [0.0],
              "role_valid": [float(t in (0, 3))], "rescue_valid": [0.0], "explore_valid": [0.0],
              "role_train": [float(t in (0, 3))], "rescue_train": [0.0], "explore_train": [0.0]} for t in range(T)]
    vf = np.full((T, n), 2.0, np.float32)
    out = {}
    for gr in (1.0, 0.5, 0.9):
        batch = SampleBatch({SampleBatch.INFOS: np.array(infos, dtype=object), SampleBatch.REWARDS: np.zeros(T, np.float32),
                             SampleBatch.DONES: np.zeros(T, bool), VF["role"]: vf.copy(), VF["rescue"]: vf.copy(),
                             VF["explore"]: vf.copy()})
        pol = SimpleNamespace(model=SimpleNamespace(n_agents=n),
                              config={"gamma": 0.99, "lambda": 0.0,
                                      "model": {"custom_model_config": {"role_gamma": gr}}})
        out[gr] = float(compute_role_gae(pol, batch)[ADV["role"]][0, 0])
        want = gr * 1.0 + gr ** 3 * 2.0 - 2.0
        assert abs(out[gr] - want) < 1e-6, (gr, out[gr], want)
    from training.config_loader import load_config
    cfg = load_config()
    assert cfg["role_gamma"] == 1.0 and cfg["model"]["custom_model_config"]["role_gamma"] == cfg["role_gamma"] \
        and cfg["gamma"] == 0.99
    print(f"[18] role SMDP uses role_gamma (config gamma 0.99 ignored): γ_r 1.0 -> Â {out[1.0]:.4f}, 0.5 -> {out[0.5]:.4f}, "
          f"0.9 -> {out[0.9]:.4f} (= γ_r r + γ_r^3 V - V); yaml role_gamma 1.0 -> custom_model_config.role_gamma 1.0")


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
            "explore_reselect", "explore_active", "rescue_active", "reward_rescue_close", "role_train", "rescue_train",
            "explore_train"]
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
            "reward_rescue": col([0.7, 0, 0.1, 0, 0.5, 0]), "role_valid": col([0, 0, 0, 1, 0, 0]),
            "rescue_active": col([1, 1, 1, 0, 0, 0])}                  # 옵션 t 0..2 (t 3 재결정 -> RESCUE 아님)
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
    print(f"[14] rescue option sum (0.7 + 0.1, ends when no longer RESCUE) A = R - V; explore option return a + γb + γ²c = {R:.4f} (ends on role change, no bootstrap), "
          f"ends at next explore decision (not at a join re-assignment), open option at fragment end masked")


def check_role_reward_exact():
    """role step 보상 - 발견 보상 = (|ΔR| - w_x |ΔE|) / N_T 정확히 — 모든 UAV 같은 값, 위치 (연결성) 와 무관."""
    sim = fresh(3)
    for k in range(4):
        add_victim(sim, (5, 5 + 3 * k))
    tick(sim)
    w_x = sim.rewards_cfg["role"]["w_x"]
    out = []
    for pos in ([(5, 5), (5, 8), (5, 11)], [(5, 5), (25, 30), (35, 5)]):
        place(sim, pos)
        r = [x - sim.last_discover for x in sim.role_rewards([object()], [object()])["role"]]
        want = [(1 - w_x * 1) / 4] * 3
        assert all(abs(x - y) < 1e-12 for x, y in zip(r, want)), (r, want)
        out.append(r)
    assert all(abs(x - y) < 1e-12 for x, y in zip(out[0], out[1]))
    print(f"[17] role step reward - discovery = (|ΔR| - w_x|ΔE|)/N_T = {[round(x, 5) for x in out[0]]} (same for all UAVs, "
          f"unchanged by positions)")


def check_role_uniform():
    from env.role.simulation import RoleSimulation
    from policies.role_rules import RandomDecider, RuleDecider
    n = 0
    for dec in (RuleDecider(), RandomDecider(0)):
        sim = RoleSimulation(env_cfg(max_episode_len=120))
        sim.random_state = np.random.RandomState(1)
        sim.reset()
        done = False
        while not done:
            sw0 = sim.rw["role_switch"].copy()
            done, info = sim.step(dec)
            r = np.asarray(info["reward_role"]) - (sim.rw["role_switch"] - sw0)
            assert np.allclose(r, r[0], atol=1e-12), r
            n += 1
    print(f"[19] role reward minus own switch penalty identical for all UAVs at every step ({n} steps, rule + random)")


def check_explore_step_reward():
    from env.role.simulation import REWARD_ITEMS
    out = []
    for found_at in (None, 0):                                    # 발견 3 / 4 vs 4 / 4 — explore 보상은 발견 수와 무관
        sim = fresh(2)
        for k in range(4):
            add_victim(sim, (5, 5 + 3 * k))
        sim.targets[0].discovered_at = found_at
        sim.explore_cells = lambda: ([1.5, 1.0], [2.0, 0.0])      # direct 1.5 칸, 병합 2 칸 (UAV 0)
        n_free = float(sim.n_free_cells())
        w_share, w_om = sim.rewards_cfg["explore"]["w_share"], sim.rewards_cfg["explore"]["w_omega"]
        om = sim.group_connectivity()[0]
        r = sim.role_rewards([], [], explore_active=[True, False])["explore"]
        assert abs(r[0] - ((1 + w_om * om) * 1.5 + w_share * 2.0) / n_free) < 1e-15 and r[1] == 0.0, r   # R1 은 EXPLORE 아님 -> 0
        out.append(r)
    assert out[0] == out[1]
    assert [k for k in REWARD_ITEMS if k.startswith("explore")] == ["explore_direct", "explore_share"]
    assert set(sim.rewards_cfg["explore"]) == {"w_share", "w_omega"}
    print(f"[22] explore step reward = ((1 + w_omega ω^g) direct + w_share share) / |C_free|: direct 1.5, share 2 -> {out[0][0]:.3e} "
          f"(same with 3/4 or 4/4 found), non-EXPLORE UAV 0; explore reward items = ['explore_direct', 'explore_share']")


def sf_sim(victims):
    """UAV 0 (15, 10), UAV 1 (15, 20) 한 그룹, victims [(pos, deadline)] (N_T = len)."""
    sim = fresh(2)
    sim.time = 0
    place(sim, [(15, 10), (15, 20)])
    set_front(sim, line(2, 2, 6))
    ids = [add_victim(sim, p, deadline=d) for p, d in victims]
    tick(sim)
    return sim, ids


def sf_reward(sim, picks, c=(1.0, 1.0), success=()):
    """결정 step 의 개인 (n_suc_i - F_excess) / N_T, 그 뒤 옵션을 닫는다 — success 의 UAV 는 구조 완료로 (보너스 대상),
    나머지는 에피소드 끝으로 (보너스 0). ω̄ = c. 반환 {sf: [UAV 별], bonus: [UAV 별], d: {UAV: 옵션 d}, F, S, F_min, F_ex}."""
    from types import SimpleNamespace
    sim._sf_pending, sim._bonus_pending = [0.0] * sim.n_agents, [0.0] * sim.n_agents
    for i, ti in picks.items():
        sim._assign_rescue(i, ti)
    sim._rescue_event_reward(sim.comp[0], list(picks))
    assert sim._bonus_pending == [0.0] * sim.n_agents                  # 보너스는 옵션이 끝날 때
    d = {i: sim._res_opt[i]["d"] for i in picks}
    sim._rescue_options(list(c), [SimpleNamespace(agent=i) for i in success], True)
    st = sim.rw_stats
    return {"sf": list(sim._sf_pending), "bonus": list(sim._bonus_pending), "d": d, "S": st["res_S"][-1], "F": st["res_F"][-1],
            "F_min": st["res_F_min"][-1], "F_ex": st["res_F_excess"][-1]}


def check_rescue_sf():
    # u: UAV 0 이 3 칸 (마감 6, σ 3, 급함) / n1: UAV 1 이 10 칸 (마감 100) / nfar: UAV 1 이 10 칸인데 마감 8 (못 닿음, σ < 0) /
    # n2, n3: 급하지 않음. N_T = 5
    V = [((15, 13), 6), ((15, 30), 100), ((5, 30), 8), ((5, 10), 100), ((25, 10), 100)]
    n_t, rc = 5, None
    sim, (u, n1, nfar, n2, n3) = sf_sim(V)
    rc = sim.rewards_cfg["rescue"]
    out = sf_reward(sim, {0: u, 1: n1}, success=(0,))
    assert out["F_ex"] == 0 and out["sf"] == [1 / n_t, 1 / n_t], out                   # 개인 n_suc 1 - 0
    assert out["d"] == {0: 3 / sim.d_max, 1: 10 / sim.d_max}, out["d"]                  # d 는 개인 BFS / d_max
    want = rc["w_b"] * ((1 - rc["w_c"]) * np.exp(-3 / sim.d_max) + rc["w_c"] * 1.0) / n_t
    assert abs(out["bonus"][0] - want) < 1e-12 and out["bonus"][1] == 0.0, out          # 구조 완료 (R0) 만 보너스, R1 은 끝 -> 0
    sim, (u, n1, nfar, n2, n3) = sf_sim(V)
    out2 = sf_reward(sim, {0: u, 1: nfar}, success=(0, 1))
    assert out2["sf"] == [1 / n_t, 0.0] and out2["S"] == 1, out2                         # R1 은 제때 못 닿음 -> n_suc 0
    sim, (u, n1, nfar, n2, n3) = sf_sim(V)
    out3 = sf_reward(sim, {0: n2, 1: n1}, success=(0, 1))                                # 급한 u 를 아무도 안 맡음 -> F 1
    assert out3["F_ex"] == 1 and out3["sf"] == [0.0, 0.0] and out3["bonus"] == [0.0, 0.0], out3   # F_excess 전원 같은 값
    print(f"[23] rescue individual: picks (u, n1) -> sf {[round(x, 3) for x in out['sf']]} (n_suc each 1), d {[round(v, 4) for v in out['d'].values()]} "
          f"(own BFS / d_max), bonus only for the rescue-completed R0 {out['bonus'][0]:.4f} (= w_b(...)/N_T), episode-end R1 0; "
          f"(u, late victim) -> sf {[round(x, 3) for x in out2['sf']]}; urgent u unassigned -> F_excess 1 for both, sf 0, bonus 0")


# ---------------------------------------------------------------- 늦지 않은 발견 / F_excess / 예전 보상 보존

def env_cfg(**over):
    from training.config_loader import load_config
    cfg = load_config()["env_config"]
    cfg["events"] = dict(cfg.get("events") or {}, log_dir=None)
    cfg.update(over)
    return cfg


def run_ep(sim, seed, decider, on_step=None):
    sim.random_state = np.random.RandomState(seed)
    sim.reset()
    done, rows = False, []
    while not done:
        done, info = sim.step(decider)
        rows.append({k: list(info[k]) for k in ("reward_role", "reward_rescue", "reward_explore")})
        if on_step:
            on_step(sim, info, done)
    return rows, info


def check_discover_reward():
    """늦지 않은 발견 보상 (w_d / N_T) Σ_{k ∈ ΔF_t} 1[τ_k - t > T_m]: 남은 골든타임 > T_m 만, = T_m · 이미 만료는 0, 팀 공통."""
    sim = fresh(3)
    sim.time = 50
    tm = float(sim.rewards_cfg["role"]["discover_margin"])
    w_d = float(sim.rewards_cfg["role"]["w_d"])
    ids = [add_victim(sim, (5, 5 + 3 * k), deadline=d, discovered=False)
           for k, d in enumerate((300, 50 + int(tm) + 1, 50 + int(tm), 40))]     # 여유 250 / T_m + 1 / T_m / 이미 만료
    n_t = len(sim.targets)
    got = {k: sim.discover_reward([ids[k]]) for k in range(4)}
    want = {0: w_d / n_t, 1: w_d / n_t, 2: 0.0, 3: 0.0}
    assert all(abs(got[k] - want[k]) < 1e-12 for k in range(4)), (got, want)
    assert abs(sim.discover_reward(ids) - 2 * w_d / n_t) < 1e-12 and sim.discover_reward([]) == 0.0
    assert sim.rw_stats["discovered"] == 8 and sim.rw_stats["discovered_timely"] == 4, sim.rw_stats
    tick(sim)
    sim.time = 50
    r = sim.role_rewards([], [], newly_found=ids[:2])["role"]
    assert all(abs(x - 2 * w_d / n_t) < 1e-12 for x in r), r                          # 팀 공통 (전환 · 구조 · 만료 없음)
    print(f"[28] timely discovery (w_d {w_d}, T_m {tm:g}, N_T {n_t}): slack 250 / T_m + 1 -> +w_d/N_T each, slack = T_m / "
          f"expired -> 0; same value for all UAVs {[round(x, 4) for x in r]}")


def check_f_excess():
    from env.role.simulation import RoleSimulation
    from policies.role_rules import RandomDecider

    def detail(sim):
        st = sim.rw_stats
        return st["res_F"][-1], st["res_F_min"][-1], st["res_F_excess"][-1], sim._sf_pending[0], sim._bonus_pending[0]
    # (a) |G| = 1: 급한 u1 (σ 3), u2 (σ 4) 둘 다 UAV 0 이 닿음, 급하지 않은 n
    V = [((15, 13), 6), ((15, 7), 7), ((5, 10), 100)]
    sim, (u1, u2, n) = sf_sim(V)
    sf_reward(sim, {0: u1}, success=(0,))
    F, Fm, Fx, sf, bonus = detail(sim)
    assert (F, Fm, Fx) == (1, 1, 0) and bonus > 0 and abs(sf - 1 / 3) < 1e-12, (F, Fm, Fx, sf, bonus)
    sim, (u1, u2, n) = sf_sim(V)
    sf_reward(sim, {0: n}, success=(0,))
    F2, Fm2, Fx2, sf2, bonus2 = detail(sim)
    assert (F2, Fm2, Fx2) == (2, 1, 1) and bonus2 == 0.0 and abs(sf2 - 0.0) < 1e-12, (F2, Fm2, Fx2, sf2, bonus2)
    # (b) |G| = 2, 급한 2 명 (σ 3) 을 UAV 0 만 제때 닿음 (UAV 1 은 7 / 13 > 6) -> F_min = max(0, 2 - 2) = 0
    V2 = [((15, 13), 6), ((15, 7), 6), ((5, 20), 100)]
    sim, (u1, u2, n) = sf_sim(V2)
    sf_reward(sim, {0: u1, 1: n})
    fm = detail(sim)[:3]
    assert fm == (1, 0, 1), fm
    # (c) random 정책 seed 0-19: F_excess >= 0 (env 안 assert), F_min <= F
    sim = RoleSimulation(env_cfg(max_episode_len=150))
    events = 0
    for seed in range(20):
        run_ep(sim, seed, RandomDecider(seed))
        st = sim.rw_stats
        assert all(x >= 0 for x in st["res_F_excess"]) and all(a <= b for a, b in zip(st["res_F_min"], st["res_F"]))
        events += len(st["res_F"])
    print(f"[30] F_excess: |G| 1, 2 urgent both reachable -> pick urgent F 1 F_min 1 F_excess 0 bonus {bonus:.3f}; "
          f"pick non-urgent F_excess 1, bonus 0; |G| 2, only A reaches both urgent -> F_min 0, F_excess 1; "
          f"random seeds 0-19 ({events} rescue events) F_excess >= 0")


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
    assert all(t.discovered for t in sim.targets) and sim.group_discovered(gA) == 2 and not sim.exploration_complete(gA)
    assert sim.role_mask(0)[Mode.EXPLORE.value], "그룹이 다 모르면 EXPLORE 허용"
    for i in (0, 1):
        sim.knowledge.known_free[i][sim.targets[ids[2]].pos] = True
    sim.agents[1].mode = Mode.RESCUE
    sim.agents[1].assigned_target = ids[0]
    tick(sim)
    assert sim.exploration_complete(gA) and not sim.role_mask(0)[Mode.EXPLORE.value]
    ev = sim._collect_events([a.pos for a in sim.agents], set(), {})
    # 그룹 A 의 EXPLORE UAV (R0) 만 — R1 은 RESCUE, 그룹 B (R2) 는 아직 T2 를 모름
    assert "4" in ev[0] and "4" not in ev[1] and "4" not in ev[2] and sim.ev_stats["ev4_causes"]["exploration_complete"] == 1, ev
    ev2 = sim._collect_events([a.pos for a in sim.agents], set(), {})            # 처음 도달할 때 한 번만
    assert "4" not in ev2[0]
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
    print("[24] EXPLORE mask: group belief 2/3 (all found globally) -> EXPLORE allowed; 3/3 -> masked + event 4 "
          "(exploration complete) for the group's EXPLORE UAV once; all roles masked -> RETURN (forced)")


if __name__ == "__main__":
    check_dup()
    check_terminal()
    check_smdp_gae()
    check_rescue_no_spread()
    check_topm_excludes_held()
    check_explore_one_shot()
    check_pair_connectivity()
    check_role_reward_exact()
    check_role_gamma()
    check_role_uniform()
    check_explore_step_reward()
    check_rescue_sf()
    check_explore_mask()
    check_discover_reward()
    check_f_excess()
    print("test_rewards: all ok")
