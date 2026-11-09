"""보상 함수 시나리오 점검 (ray 클러스터 불필요) — 빈 맵에 UAV / 생존자를 직접 놓고 env 보상 경로를 호출.

    python -m tests.role.test_rewards

 [6] 에피소드 종료: 미구조 생존자 = 마지막 step 의 만료 E (플래그 끄면 만료 항 없음)
 [7] role GAE: 결정 간격 k step 이면 γ^k (할인 누적과 bootstrap 모두)
[14] rescue / explore 학습: 결정 단위 SMDP GAE — 구간 끝 = min(같은 네트워크 다음 결정, 그 역할을 벗어나는 step).
     역할을 벗어나면 그 시점 V^k bootstrap + GAE 체인 끊음, 같은 역할 연속 결정은 체인 이어짐, 구간 보상은 그 역할 보상만,
     fragment 끝에서 구간이 끝나지 않은 결정은 마스크 0
[16] 연결성 C_pair = Σ_k n_k (n_k - 1) / (N (N - 1)): N = 5 의 연결 요소 배치 5 / 4+1 / 3+2 / 3+1+1 / 2+2+1 / 2+1+1+1 / 1×5
     -> 1.0 / 0.6 / 0.4 / 0.3 / 0.2 / 0.1 / 0.0, N = 1 -> 0, N = 2 연결 -> 1 / 끊김 -> 0, 간선 = 거리 <= R_c (경계 포함)
[17] role step 보상 - 발견 보상 = (|ΔR| - w_x |ΔE|) / N_T 정확히, 모든 UAV 같은 값, 위치와 무관
[18] role SMDP 할인 = role_gamma (model.custom_model_config.role_gamma), rescue / explore 할인 = config gamma
[19] 실제 rollout 에서 매 step role 보상 - 그 UAV 의 전환 penalty 가 모든 UAV 같음 (팀 공통 부분)
[24] 그룹 기준 EXPLORE 마스크: 그룹 belief 발견 수 < N_T (실제로는 다 찾음) -> EXPLORE 허용, = N_T -> 금지 + 그 그룹
     EXPLORE UAV 에 이벤트 "4" (exploration complete), 모든 역할이 막히면 RETURN 허용 (forced 기록)
[28] 늦지 않은 발견 보상 (w_d / N_T) Σ_{k ∈ ΔF_t} 1[τ_k - t > T_m]: 남은 골든타임 > T_m 이면 +w_d / N_T, = T_m · 이미 만료는 0,
     rw_stats discovered / discovered_timely, 모든 UAV 같은 값
(task 보상 항목은 tests/role/test_task_reward.py)
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
    adv, vt, ok, _ = decision_advantages(r, v, dec, True, g, lam=0.0)
    want = sum(g ** t for t in range(k)) + g ** k * 2.0 - 0.5
    assert ok[0] == 1 and abs(adv[0] - want) < 1e-5, (adv[0], want)
    adv1, _, _, _ = decision_advantages(r, v, dec, True, g, lam=1.0)
    want1 = want + g ** k * 1.0 * adv[k]
    assert abs(adv1[0] - want1) < 1e-5
    adv2, _, ok2, _ = decision_advantages(r, v, dec, False, g, lam=0.95)
    assert ok2[k] == 0 and ok2[0] == 1, "fragment 끝에서 미완료 구간은 제외"
    print(f"[7] role SMDP GAE: decision gap k = {k} -> δ = Σ γ^t r + γ^{k} V(s_t1) - V(s_t0) = {want:.4f}; "
          f"GAE uses γ^{k} λ Â(t1); unfinished last segment masked when truncated")


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
    """compute_role_gae: role 경로는 role_gamma (model.custom_model_config.role_gamma), rescue / explore 는 config gamma."""
    from types import SimpleNamespace
    from ray.rllib.policy.sample_batch import SampleBatch
    from training.role.ppo import ADV, VF, compute_role_gae
    n, T = 1, 5
    infos = [{"reward_role": [1.0 if t == 1 else 0.0], "reward_rescue": [0.0], "reward_explore": [0.0], "role": [0.0],
              "role_valid": [float(t in (0, 3))], "rescue_valid": [0.0], "explore_valid": [0.0],
              "role_train": [float(t in (0, 3))], "rescue_train": [0.0], "explore_train": [0.0]} for t in range(T)]
    vf = np.full((T, n), 2.0, np.float32)
    out = {}
    for gr in (1.0, 0.5, 0.9):
        batch = SampleBatch({SampleBatch.INFOS: np.array(infos, dtype=object), SampleBatch.REWARDS: np.zeros(T, np.float32),
                             SampleBatch.DONES: np.zeros(T, bool), VF["role"]: vf.copy(), VF["rescue"]: vf.copy(), VF["explore"]: vf.copy()})
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


def gae_batch(rows, n=1):
    """rows: dict 키 -> (T, n) 리스트. 나머지 info 키는 0."""
    from ray.rllib.policy.sample_batch import SampleBatch
    from training.role.ppo import VF
    T = len(next(iter(rows.values())))
    keys = ["reward_role", "reward_rescue", "reward_explore", "role", "role_valid", "rescue_valid", "explore_valid", "role_train",
            "rescue_train", "explore_train"]
    infos = [{k: list(rows.get(k, [[0.0] * n] * T)[t]) for k in keys} for t in range(T)]
    vf = np.full((T, n), 0.5, np.float32)
    return SampleBatch({SampleBatch.INFOS: np.array(infos, dtype=object), SampleBatch.REWARDS: np.zeros(T, np.float32),
                        SampleBatch.DONES: np.zeros(T, bool), VF["role"]: vf.copy(), VF["rescue"]: vf.copy(), VF["explore"]: vf.copy()})


def check_task_gae():
    """compute_role_gae rescue / explore: 구간 끝 t1 = min(같은 네트워크 다음 결정, 그 역할을 벗어나는 step).
    (1) RESCUE 결정 t=1 -> t=2 RETURN (역할 벗어남): t1 = 2, V^res(s_2) bootstrap, 체인 끊김, 그 뒤 EXPLORE step 보상 안 섞임
    (2) RESCUE 결정 t=0, t=2 (그 사이 계속 RESCUE): t1 = 2 (다음 결정), GAE 체인 이어짐
    (3) EXPLORE 결정이 fragment 끝까지 EXPLORE 로 끝나지 않음 -> 마스크 0."""
    from types import SimpleNamespace
    from training.role.ppo import ADV, PMASK, VMASK, VT, compute_role_gae
    g, lam, V = 0.9, 0.95, 0.5
    col = lambda v: [[x] for x in v]
    pol = SimpleNamespace(model=SimpleNamespace(n_agents=1), config={"gamma": g, "lambda": lam, "env_config": {}})
    # (1) + (3): 역할 R R D D E E, RESCUE 결정 t=1, EXPLORE 결정 t=4 (fragment 끝까지 EXPLORE)
    rr, re = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.2, 0.05]
    rows = {"role": col([R, R, D, D, E, E]), "reward_rescue": col(rr), "reward_explore": col(re),
            "rescue_valid": col([0, 1, 0, 0, 0, 0]), "rescue_train": col([0, 1, 0, 0, 0, 0]),
            "explore_valid": col([0, 0, 0, 0, 1, 0]), "explore_train": col([0, 0, 0, 0, 1, 0])}
    bt = compute_role_gae(pol, gae_batch(rows))
    want = rr[1] + g * V - V                                              # t1 = 2 (RETURN), 체인 없음
    assert abs(bt[ADV["rescue"]][1, 0] - want) < 1e-6 and abs(bt[VT["rescue"]][1, 0] - (want + V)) < 1e-6, bt[ADV["rescue"]]
    assert bt[PMASK["rescue"]].sum() == 1 and bt[VMASK["rescue"]].sum() == 1
    assert bt[PMASK["explore"]].sum() == 0 and bt[VMASK["explore"]].sum() == 0     # EXPLORE 구간 미완료 (fragment 끝)
    assert bt[ADV["explore"]][1, 0] == 0.0                                    # advantage 는 결정한 네트워크의 컬럼에만
    # (2): 역할 R R R R E, RESCUE 결정 t=0, t=2 -> t=2 의 구간은 t=4 (EXPLORE) 에서 잘림
    r2 = [0.1, 0.2, 0.3, 0.4, 0.0]
    rows = {"role": col([R, R, R, R, E]), "reward_rescue": col(r2), "rescue_valid": col([1, 0, 1, 0, 0]),
            "rescue_train": col([1, 0, 1, 0, 0])}
    bt = compute_role_gae(pol, gae_batch(rows))
    a2 = r2[2] + g * r2[3] + g ** 2 * V - V                                   # t=2 -> t1 = 4 (역할 벗어남)
    a0 = r2[0] + g * r2[1] + g ** 2 * (V + lam * a2) - V                     # t=0 -> t1 = 2 (다음 결정), 체인 이어짐
    assert abs(bt[ADV["rescue"]][2, 0] - a2) < 1e-6 and abs(bt[ADV["rescue"]][0, 0] - a0) < 1e-6, bt[ADV["rescue"]]
    print(f"[14] rescue / explore SMDP GAE: RESCUE decision t=1, RETURN at t=2 -> cut, Â = r + γ V(s_2) - V(s_1) = {want:.4f} "
          f"(no chain, later EXPLORE reward not mixed); consecutive RESCUE decisions t=0 -> t=2 chain: Â(0) = {a0:.4f} "
          f"(Â(2) = {a2:.4f}, cut at EXPLORE t=4); EXPLORE decision unfinished at fragment end masked")


def check_role_reward_exact():
    """role step 보상 - 발견 보상 = (|ΔR| - w_x |ΔE|) / N_T 정확히 — 모든 UAV 같은 값, 위치 (연결성) 와 무관."""
    from types import SimpleNamespace
    sim = fresh(3)
    for k in range(4):
        add_victim(sim, (5, 5 + 3 * k))
    tick(sim)
    w_x = sim.rewards_cfg["role"]["w_x"]
    out = []
    for pos in ([(5, 5), (5, 8), (5, 11)], [(5, 5), (25, 30), (35, 5)]):
        place(sim, pos)
        r = [x - sim.last_discover for x in sim.role_rewards([SimpleNamespace(agent=0, target=0)], [3])["role"]]
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


# ---------------------------------------------------------------- 늦지 않은 발견 / 마스크

def env_cfg(**over):
    from training.config_loader import load_config
    cfg = load_config()["env_config"]
    cfg["events"] = dict(cfg.get("events") or {}, log_dir=None)
    cfg.update(over)
    return cfg


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
    check_terminal()
    check_smdp_gae()
    check_task_gae()
    check_pair_connectivity()
    check_role_reward_exact()
    check_role_gamma()
    check_role_uniform()
    check_explore_mask()
    check_discover_reward()
    print("test_rewards: all ok")
