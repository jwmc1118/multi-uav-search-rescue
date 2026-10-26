"""계층 구조 env (env/role/simulation.py) 단위 점검 (ray 불필요).

    python -m tests.role.test_role_env

1) DBSCAN: 8-연결 고리 frontier 는 하나로 묶임, 고립 셀 (이웃 < min_samples) 은 noise 로 제거, 전부 noise 면 연결 성분 fallback
   (구간 추적 / 소진은 tests/role/test_explore.py, 이벤트 체계는 tests/role/test_events_targeted.py)
3) 충돌 인지: 같은 조난자를 쫓는 같은 그룹 UAV 중 먼 쪽 (동률이면 index 큰 쪽) 만 Role 이벤트 5
   + 같은 step 결정 순서: 앞 순서가 고른 생존자는 뒤 UAV 의 rescue 마스크에서 빠지고, 남은 생존자가 없으면 RESCUE 역할도 마스크
4) 보상: 전부 구조 시 role 보상 합 = 1.0, 실제 에피소드에서 Σ r_role = (구조 - w_x · (만료 + 끝날 때 미구조)) / N_T - 진동 penalty
5) 규칙 정책 에피소드: rescue 보상은 RESCUE UAV 에만 (이벤트 끝 S - F + 보너스 + 중복), explore 보상은 그 step EXPLORE 인 UAV 에만
   (매 step (1 - ψ^f) Δcov / |free| >= 0) — Σ r_explore = rw explore_cov, 팀 합 <= (1 - 0) · 새 free 셀 / |free|
"""

import numpy as np

from env.role.clusters import dbscan_grid, frontier_clusters
from env.core.modes import Mode
from env.role.simulation import RoleSimulation
from policies.role_rules import RuleDecider
from training.config_loader import load_config


def make_sim(seed=0):
    cfg = load_config()["env_config"]
    cfg["seed"] = seed
    return RoleSimulation(cfg)


def check_dbscan():
    g = np.zeros((30, 30), bool)
    g[5, 5:25] = g[24, 5:25] = True                     # 20 x 20 고리 (8-연결) = 76 셀
    g[5:25, 5] = g[5:25, 24] = True
    g[1, 1] = True                                      # 고립 셀 1 개
    g[28, 2] = g[28, 3] = True                          # 셀 2 개 (이웃 수 2 < 3) -> noise
    raw = dbscan_grid([tuple(p) for p in np.argwhere(g)], eps=1.5, min_samples=3)
    ring = int(g[5:25, 5:25].sum())
    assert len(raw) == 1 and len(raw[0]) == ring, [len(c) for c in raw]
    one = frontier_clusters(g, 1.5, 3, 10)
    assert len(one) == 1 and one[0].size == ring, "고리는 덩어리 하나"
    assert tuple(one[0].medoid) in one[0].cell_set(), "medoid 는 실제 frontier 셀"
    only_noise = np.zeros((10, 10), bool)
    only_noise[0, 7] = only_noise[3, 0] = True               # 떨어진 셀 2 개뿐 (구석 depot 시작) -> fallback
    fb = frontier_clusters(only_noise, 1.5, 3, 10)
    assert len(fb) == 2, "전부 noise 일 때 fallback 클러스터가 있어야 deadlock 이 안 생김"
    print(f"[1] DBSCAN: ring {ring} cells -> 1 blob, noise (3 cells) removed; "
          f"all-noise frontier -> fallback {len(fb)} clusters")


def check_conflict():
    sim = make_sim(2)
    t = sim.targets[0]
    t.discovered_at = 0
    for k in range(sim.n_agents):
        sim.knowledge.known_free[k][t.pos] = True
    d0 = sim.depot
    sim.agents[0].pos, sim.agents[1].pos = d0, d0
    sim._prepare_role()                                   # 같은 칸 -> 같은 그룹, 같은 거리 (동률)
    for k in (0, 1):
        sim._assign_rescue(k, 0)
    assert sim.comp[0] == sim.comp[1]
    inv = [sim._target_invalid(k, 0) for k in (0, 1)]
    assert inv == [False, True], inv                      # 동률 -> index 큰 쪽 무효
    # 1 번을 한 칸 더 가까이
    path = sim.planned_path(1)[0]
    if path and len(path) > 1:
        sim.agents[1].pos = path[1]
        sim._prepare_role()
        inv = [sim._target_invalid(k, 0) for k in (0, 1)]
        assert inv == [True, False], inv
    ev = sim._collect_events([b.pos for b in sim.agents], set(), {})
    assert "5" in ev[0] and "5" not in ev[1]
    print(f"[3] conflict awareness: tie -> higher index loses, closer one keeps; farther gets event 5")


def check_rescue_mask_conflict():
    """같은 step 에 앞 순서가 고른 생존자뿐이면 뒤 UAV 의 rescue 마스크 / role 의 RESCUE 마스크가 꺼진다 (taken)."""
    from env.core.modes import Mode
    sim = make_sim(5)
    for t in sim.targets:                                  # 조난자 0 만 발견 + 모두가 앎
        t.discovered_at = None
    t = sim.targets[0]
    t.discovered_at = 0
    for k in range(sim.n_agents):
        sim.knowledge.known_free[k][t.pos] = True
    sim._prepare_role()
    assert sim.comp[0] == sim.comp[1] and sim.role_mask(1)[Mode.RESCUE.value], "전제: 1 번도 처음엔 RESCUE 가능"
    taken = {0}                                            # 같은 step 에 앞 순서 (0 번) 가 조난자 0 을 고름
    m1, m0 = sim.role_mask(1, (), taken), sim.role_mask(0)
    assert not m1[Mode.RESCUE.value], "앞 순서가 고른 조난자뿐인데 RESCUE 가 열려 있음"
    assert not sim.rescue_mask(1, taken).any()
    assert m0[Mode.RESCUE.value] and sim.rescue_mask(1).any(), "taken 이 없으면 RESCUE 가능 (conflict 는 마스크가 아니라 이벤트 6)"
    print("[3b] earlier decider took the only victim this step -> RESCUE masked for the next UAV "
          f"(role mask R1={m1.astype(int).tolist()}, R0={m0.astype(int).tolist()})")


def check_rewards():
    sim = make_sim(3)
    w = sim.rewards_cfg
    from env.core.step_context import RescueEvent
    ev = [RescueEvent(target=k, agent=0, rescued_at=1, discovered_at=0, time_taken=1) for k in range(len(sim.targets))]
    r = sim.role_rewards(ev, [])
    want = 1.0                                                         # 전원 구조 (연결성 항 없음)
    assert all(abs(x - want) < 1e-9 for x in r["role"]), (r["role"], want)
    sim = make_sim(4)
    sums = np.zeros(3)
    n_res = n_exp = 0
    new_cells = 0.0
    done = False
    rescue_outside = explore_outside = n_exp_dec = 0
    while not done:
        done, info = sim.step(RuleDecider())
        sums += [info["reward_role"][0], sum(info["reward_rescue"]), sum(info["reward_explore"])]
        n_res += info["rescued_now"]
        n_exp += info["expired_now"]
        new_cells += sum(sim.knowledge.newly_sensed)
        rescue_outside += sum(1 for k in range(sim.n_agents) if info["reward_rescue"][k] != 0 and not info["in_rescue_seg"][k])
        for k in range(sim.n_agents):
            re_ = info["reward_explore"][k]
            n_exp_dec += int(info["explore_valid"][k] > 0)
            assert 0.0 <= re_ <= 1.0, re_
            if re_ != 0 and not info["explore_active"][k]:
                explore_outside += 1
    n_t = len(sim.targets)
    left = sum(1 for t in sim.targets if t.active)                     # 끝날 때 미구조 = 마지막 step 의 만료
    osc0 = float(sim.rw["role_osc"][0])
    assert abs(sums[0] - ((n_res - w["role"]["w_x"] * (n_exp + left)) / n_t + osc0)) < 1e-9
    ratio = float(sim.rw["explore_cov"].sum())
    assert abs(sums[2] - ratio) < 1e-6 and rescue_outside == 0 and explore_outside == 0 and n_exp_dec > 0
    print(f"[4] role reward: all rescued -> {want:.3f}; episode Σ r_role(R0) {sums[0]:.3f} = ({n_res} - "
          f"{w['role']['w_x']}·({n_exp} + {left} left at end))/{n_t} + osc {osc0:.2f}")
    print(f"[5] rescue reward only on RESCUE UAVs (Σ {sums[1]:.3f}: S - F + bonus + duplicate); explore reward only on EXPLORE "
          f"UAVs, each in [0, 1] ({n_exp_dec} explore decisions; Σ {sums[2]:.4f} = rw explore_cov {ratio:.4f})")


if __name__ == "__main__":
    check_dbscan()
    check_conflict()
    check_rescue_mask_conflict()
    check_rewards()
    print("test_role_env: all ok")
