"""rescue / explore 네트워크 · 메모리 노드 · 그래프 특징 점검 (작은 맵 / 몇 step, ray 학습 불필요).

    python -m tests.role.test_task_memory

[1] 특징: U 노드 12 차원 (이름 순서), 그룹 context 4 차원, S–F 간선 = 생존자마다 가까운 덩어리 kNN (동률 포함) · 차원 3
[2] 메모리 노드: 두 UAV 가 끊기면 서로 linked 0 · age 증가, 제3 UAV 를 통해 더 최근 기록이 중계되면 t_last 갱신,
    마스크 (역할 / 생존자 / 덩어리) · ω^g 는 메모리 노드와 무관 (memory.enabled 를 꺼도 같음)
[3] rescue / explore 네트워크: RESCUE 차례면 F logit 전부 마스크, EXPLORE 차례면 S 전부 마스크, 출력 확률 합 1
"""
import torch

from env.role.graph_obs import CONTEXT_FEATURES, EDGE_FEATURES, FC, FU, U_FEATURES
from models.role.net import MASK_VALUE
from tests.role.test_events_targeted import add_victim, fresh
from tests.role.test_explore import connect, line, place, set_front, tick
from tests.role.test_gnn import make, obs_t


def rollout_traces(steps=40, seed=1):
    """학습 모델 (무작위 가중치) 로 몇 step — 차례별 그래프 trace 와 관측."""
    env, prep, m = make()
    env.seed(seed)
    obs = env.reset()
    out = []
    for _ in range(steps):
        trace = []
        m.net.debug_trace = trace
        with torch.no_grad():
            m({"obs": obs_t(prep, obs)}, [], None)
        out.append((trace, env.sim._obs_cache, m.net.knn_k, env.sim.s_max))
        obs, _, done, _ = env.step({k: [int(x) for x in m.last_out["actions"][k][0]] for k in ("role", "task")})
        if done:
            break
    m.net.debug_trace = None
    return out


def check_features(traces):
    from models.role.net import knn_adj
    assert U_FEATURES == ("x", "y", "cap", "role_explore", "role_rescue", "role_return", "goal_x", "goal_y", "depot_bfs",
                          "omega_group", "linked", "age") and FU == 12
    assert CONTEXT_FEATURES == ("rho_undiscovered", "eta_unexplored", "nu_found_unrescued", "time_elapsed") and FC == 4
    assert EDGE_FEATURES["SF"] == ("dx", "dy", "bfs")
    n_sf = 0
    for trace, ob, k, _ in traces:
        for e in trace:
            g = e["graph"]
            assert g["x"]["U"].shape[-1] == FU and g["ctx"].shape[-1] == FC and g["edge"]["SF"].shape[-1] == 3
            for r, i in enumerate(g["ii"].tolist()):
                d = torch.as_tensor(ob["agents"][i]["sf_dist"])[None]
                mS, mF = g["mask"]["S"][r:r + 1], g["mask"]["F"][r:r + 1]
                want = knn_adj(d, mS, k, mF)[0]
                assert torch.equal(g["adj"]["SF"][r], want)
                if mS.any() and mF.any():                              # 유효 생존자마다 min(k, 덩어리 수) 개 이상 (동률 포함)
                    deg = want.sum(-1)[mS[0]]
                    assert int(deg.min()) >= min(k, int(mF.sum()))
                    n_sf += int(want.sum())
    assert n_sf > 0
    print(f"[1] features: U {FU} ({', '.join(U_FEATURES)}), context {FC} ({', '.join(CONTEXT_FEATURES)}), "
          f"S–F edges {n_sf} (each survivor -> nearest {traces[0][2]} blobs incl. ties, dim 3)")


def check_memory():
    sim = fresh(3)
    place(sim, [(10, 5), (10, 8), (10, 11)])
    set_front(sim, line(2, 2, 20))
    add_victim(sim, (12, 6))

    def at(t, groups, pos=None):
        sim.time = t
        if pos:
            place(sim, pos)
        connect(sim, groups)
        tick(sim)
        sim._memory_update()
    at(5, [[0, 1, 2]])
    assert all(sim.memory[i][k]["t"] == 5 for i in range(3) for k in range(3))
    at(10, [[0, 2], [1]], [(10, 5), (25, 30), (10, 11)])                  # 0 · 1 끊김 (2 는 0 쪽)
    ob = sim.rl_observation()["agents"]
    assert ob[0]["mem_valid"][1] == 1 and ob[1]["mem_valid"][0] == 1 and ob[0]["mem_valid"][2] == 0
    assert abs(ob[0]["mem_age"][1] - 5 / 50) < 1e-6 and abs(ob[1]["mem_age"][0] - 5 / 50) < 1e-6
    at(20, [[0], [1, 2]], [(10, 5), (25, 31), (24, 30)])                  # 2 가 1 쪽으로: 1 의 지금 기록 (t 20) 을 가짐
    assert sim.memory[2][1]["t"] == 20 and sim.memory[0][1]["t"] == 5
    age_before = sim.rl_observation()["agents"][0]["mem_age"][1]
    at(30, [[0, 2], [1]], [(10, 5), (25, 31), (10, 7)])                  # 2 가 0 쪽으로 돌아옴: 1 의 기록 중계 (t 20)
    rec = sim.memory[0][1]
    assert rec["t"] == 20 and rec["pos"] == (25, 31), rec
    age_after = sim.rl_observation()["agents"][0]["mem_age"][1]
    assert abs(age_before - 15 / 50) < 1e-6 and abs(age_after - 10 / 50) < 1e-6
    # 마스크 · ω^g 는 메모리 노드와 무관
    masks = lambda: ([sim.role_mask(i).tolist() for i in range(3)], [sim.rescue_mask(i).tolist() for i in range(3)],
                     [sim.explore_mask(i).tolist() for i in range(3)], sim.group_connectivity())
    on = masks()
    sim.memory_cfg["enabled"] = False
    off = masks()
    assert on == off and sim.rl_observation()["agents"][0]["mem_valid"].sum() == 0
    sim.memory_cfg["enabled"] = True
    print(f"[2] memory: 0 · 1 split -> linked 0 for each other, age (t - t_last) / 50 = 0.1; record of 1 relayed to 0 via 2 "
          f"(t_last 5 -> 20, age {age_before:.2f} -> {age_after:.2f}); role / rescue / explore masks and ω^g identical with "
          f"memory off")


def check_task_masks(traces):
    seen = {"rescue": 0, "s1": 0}
    for trace, ob, _, T in traces:
        for e in trace:
            st, lg = e["stage"], e["logits"]
            if st not in seen:
                continue
            seen[st] += lg.shape[0]
            blocked = lg[:, T:] if st == "rescue" else lg[:, :T]
            assert (blocked <= MASK_VALUE / 2).all(), st
            p = torch.softmax(lg, -1)
            assert torch.allclose(p.sum(-1), torch.ones(lg.shape[0]), atol=1e-5)
            assert float(p[:, T:].sum() if st == "rescue" else p[:, :T].sum()) < 1e-6
    assert seen["rescue"] > 0 and seen["s1"] > 0, seen
    print(f"[3] task network: RESCUE rows {seen['rescue']} -> all F logits masked; EXPLORE rows {seen['s1']} -> all S "
          f"masked; probabilities sum to 1")


if __name__ == "__main__":
    from training.catalog import register_all
    register_all()
    tr = rollout_traces()
    check_features(tr)
    check_memory()
    check_task_masks(tr)
    print("test_task_memory: all ok")
