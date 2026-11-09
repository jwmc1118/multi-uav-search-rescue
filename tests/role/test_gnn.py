"""세 네트워크 (역할 선택 / rescue / explore) 의 이종 그래프 GNN + 연결성 보상 점검 (CPU 로 충분).

    python -m tests.role.test_gnn

[1] 그래프 구성 (모델이 직접 env 를 굴리며 순차 디코딩 trace 를 매 step 검사):
    노드 · 간선 차원 (U 12, S 3, F 3 / UU 4, US 3, UF 2, SS 3, FF 3, SF 3), U 노드 = 그룹 (linked 1) + 메모리 노드 (linked 0,
    역할 무관, 세 네트워크 같음), S 노드 = 그룹이 아는 생존자만, 값: 연결 노드 절대 위치 / 메모리 노드 기록 위치, depot BFS, ω^g,
    그룹 context [ρ, η, ν, t / T_max], U–S BFS 와 ETA 기반 σ (메모리 노드 = 기록 위치 · 용량), 목표 중복 간선 (차례 처음 = 관측의 목표 /
    슬롯 기준, 메모리 노드는 기록된 목표), 앞 차례 RESCUE 결정이 뒤 그래프에 반영
[1b] σ: 용량 0 이면 depot 경유 ETA = BFS(l, depot) + BFS(depot, s), 용량 > 0 이면 ETA = BFS(l, s)
[2] attention 집계 순열 불변, 패딩 노드 무영향 (출력 0), U–S 간선 갱신도 순열 불변
[3] 2-hop: A–B–C 사슬에서 L = 1 이면 C 가 A 에 안 닿고 L = 2 면 닿음
[4] 그룹 연결도 ω^g_i = (|U_i| - 1) / (|U| - 1) (U_i = 통신 그룹): 그룹 {3, 2} -> 3 쪽 0.5, 2 쪽 0.25
[6] 삭제된 / 모르는 모델 옵션은 에러 없이 무시
[7] kNN 동률 (S–S / F–F / S–F): k 번째 이웃과 거리가 같은 이웃은 모두 포함, 노드 순서를 섞어도 인접 · GNN 출력이 같음
[8] 크기 특징 size_feat(n) = min(1, log(1 + n) / log(1 + 100))
[9] 세 네트워크는 파라미터를 공유하지 않고, 각 네트워크 loss 는 자기 네트워크에만 gradient 를 준다
"""
import copy

import numpy as np
import torch
from ray.rllib.models import ModelCatalog

from env.core.modes import Mode
from env.core.targets import deadline
from env.role.graph_obs import EFF, ESF, ESS, EUF, EUS, EUU, FF, FS, FU
from env.role.gym_env import RoleEnv
from models.role.gnn import HeteroGNN
from models.role.net import BASE
from models.role.rllib_model import RoleModel
from training.catalog import register_all
from training.config_loader import load_config

U_CAP, U_ROLE, U_GOAL, U_DEPOT, U_OMEGA, U_LINK, U_AGE = 2, slice(3, 6), slice(6, 8), 8, 9, 10, 11
REL_US = (("S2U", "S", "U", "US"), ("U2S", "U", "S", "US"), ("U2U", "U", "U", "UU"), ("S2S", "S", "S", "SS"))   # GNN 단위 점검용


def make(n_agents=None, seed=0):
    cfg = load_config()
    ec = copy.deepcopy(cfg["env_config"])
    if n_agents:
        ec["agents"] = [dict(ec["agents"][0]) for _ in range(n_agents)]
    ec["components"] = {"role": "learn", "rescue": "learn", "explore": "learn"}
    env = RoleEnv(ec)
    prep = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    torch.manual_seed(seed)
    m = RoleModel(prep.observation_space, env.action_space, 1, {"custom_model_config": cfg["model"]["custom_model_config"]}, "m")
    return env, prep, m


def obs_t(prep, obs):
    return torch.as_tensor(prep.transform(obs))[None]


def check_graph(steps=200):
    env, prep, m = make()
    env.seed(1)
    obs = env.reset()
    sim = env.sim
    T = sim.s_max
    n_rows = n_seq = n_ov = n_mem = 0
    kinds = set()
    for t in range(steps):
        trace = []
        m.net.debug_trace = trace
        with torch.no_grad():
            m({"obs": obs_t(prep, obs)}, [], None)
        ob = sim._obs_cache
        group = ob["group"] > 0.5
        bel = sim.uav_beliefs()
        rescued_before = {}                                          # 이번 forward 의 앞 role 차례가 RESCUE 로 고른 (UAV -> 생존자 위치)
        acts = m.last_out["actions"]
        first_role = True
        for e in trace:
            g, st = e["graph"], e["stage"]
            kinds.add(st)
            assert g["x"]["U"].shape[-1] == FU and g["edge"]["UU"].shape[-1] == EUU
            assert g["x"]["S"].shape[-1] == FS and g["x"]["F"].shape[-1] == FF
            assert tuple(g["edge"][k].shape[-1] for k in ("US", "UF", "SS", "FF", "SF")) == (EUS, EUF, ESS, EFF, ESF)
            for r, i in enumerate(g["ii"].tolist()):
                n_rows += 1
                mU = g["mask"]["U"][r].numpy()
                mem = set(sim.memory_nodes(i))
                link = group[i]
                assert (mU == (link | np.isin(np.arange(sim.n_agents), list(mem)))).all() and mU[i], (st, mU)
                xU = g["x"]["U"][r].numpy()
                role_now = xU[:, U_ROLE].argmax(-1)
                assert np.allclose(g["ctx"][r].numpy(), [bel["team_rho"][i], bel["team_eta"][i], bel["team_nu"][i], bel["team_time"][i]], atol=1e-6)
                for l in np.flatnonzero(mU):
                    assert xU[l, U_LINK] == float(link[l])
                    if link[l]:
                        assert np.allclose(xU[l, 0:2], sim.frame.normalized(sim.agents[l].pos), atol=1e-6) and xU[l, U_AGE] == 0
                        assert abs(xU[l, U_DEPOT] - sim._nd(sim._dist(l, sim.depot))) < 1e-6
                        assert abs(xU[l, U_OMEGA] - sim.group_connectivity()[l]) < 1e-6
                    else:
                        rec = sim.memory[i][l]
                        n_mem += 1
                        assert np.allclose(xU[l, 0:2], sim.frame.normalized(rec["pos"]), atol=1e-6)
                        assert abs(xU[l, U_AGE] - min(1.0, (sim.time - rec["t"]) / sim.memory_cfg["T_age"])) < 1e-6
                        assert int(role_now[l]) == rec["role"] and abs(xU[l, U_OMEGA] - rec["omega"]) < 1e-6
                sid = [int(x) for x in ob["agents"][i]["s_ids"] if x >= 0]
                assert set(sid) <= set(sim._group_known(sim.comp[i])), "그룹이 모르는 생존자"
                eUS = g["edge"]["US"][r].numpy()
                for s_, ti in enumerate(sid):
                    tg = sim.targets[ti]
                    tau = deadline(tg)
                    assert abs(g["x"]["S"][r, s_, 2] - min(1.0, sim.time / tau)) < 1e-6
                    d_ds = int(sim.group_depot_field[sim.comp[i]][tg.pos])
                    for l in np.flatnonzero(mU):
                        if link[l]:
                            d = sim._dist(l, tg.pos)
                            eta = sim.eta(l, d, d_ds)
                        else:                                       # 메모리 노드: 기록 위치 · 용량 (그룹 지도 BFS)
                            rec = sim.memory[i][l]
                            d = int(sim._memory_field(sim.comp[i], rec["pos"])[tg.pos])
                            eta = None if d < 0 else (float(d) if rec["cap"] > 0 else
                                                      (None if rec["depot_d"] is None or d_ds < 0 else float(rec["depot_d"] + d_ds)))
                        assert abs(eUS[l, s_, 0] - sim._nd(d)) < 1e-6, "U–S BFS"
                        sig = -1.0 if eta is None else float(np.clip((tau - sim.time - eta) / tau, -1, 1))
                        assert abs(eUS[l, s_, 1] - sig) < 1e-5, ("σ", eUS[l, s_, 1], sig)
                if st == "role" and first_role:                    # 첫 차례: 목표 중복 = 관측의 목표 생존자 / 영역 슬롯 기준
                    tgt = [ob["team_target"][l] if link[l] else ob["agents"][i]["mem_target"][l] for l in range(sim.n_agents)]
                    slot = [ob["team_slot"][l] if link[l] else -1 for l in range(sim.n_agents)]
                    ov = g["edge"]["UU"][r, :, :, 3].numpy()
                    for a in np.flatnonzero(mU):
                        for b in np.flatnonzero(mU):
                            want = (tgt[a] == tgt[b] and tgt[a] >= 0) or (slot[a] == slot[b] and slot[a] >= 0)
                            assert ov[a, b] == float(want), (a, b, ov[a, b], want)
                            n_ov += int(want and a != b)
                if st == "role":                                     # 앞 차례 RESCUE 결정 반영 (역할 · 목표)
                    for j, gp in rescued_before.items():
                        if mU[j] and link[j]:
                            n_seq += 1
                            assert int(role_now[j]) == Mode.RESCUE.value and np.allclose(xU[j, U_GOAL], gp, atol=1e-6)
            if st == "role":
                first_role = False
            if st == "rescue":
                for i in g["ii"].tolist():
                    s_ = int(acts["task"][0, i])
                    assert s_ < T
                    rescued_before[i] = ob["agents"][i]["s_pos"][s_]
        act = {k: [int(x) for x in m.last_out["actions"][k][0]] for k in ("role", "task")}
        obs, _, done, info = env.step(act)
        assert info["mask_violations"] == 0
        if done:
            obs = env.reset()
    m.net.debug_trace = None
    assert kinds == {"role", "rescue", "s1"} and n_seq > 0 and n_ov > 0 and n_mem > 0, (kinds, n_seq, n_ov, n_mem)
    print(f"[1] graph: {n_rows} decision graphs (role / rescue / frontier) checked — dims, U = group + memory nodes "
          f"({n_mem} memory-node rows: recorded pos / role / ω^g / age, U–S σ from record), context [ρ, η, ν, t / T_max], depot BFS, "
          f"U–S BFS + ETA sigma, {n_ov} goal-overlap pairs, {n_seq} sequential RESCUE updates")


def check_sigma_depot():
    from tests.role.test_events_targeted import add_victim, fresh
    from tests.role.test_explore import place, tick
    sim = fresh(2)
    sim.time = 10
    tick(sim)
    dr, dc = sim.depot
    p0 = (5, 35) if dc < 20 else (5, 5)                               # depot 에서 먼 쪽에 UAV 0, 바로 옆에 생존자
    place(sim, [p0, (12, 20)])
    v = add_victim(sim, (p0[0] + 2, p0[1]), deadline=200)
    tick(sim)
    a0 = sim.agents[0]
    out = {}
    for cap in (a0.capacity_max, 0):
        a0.capacity = cap
        ob = sim.rl_observation()
        s_ = list(ob["agents"][0]["s_ids"]).index(v)
        d = sim._dist(0, sim.targets[v].pos)
        eta = d if cap > 0 else sim._dist(0, sim.depot) + int(sim.group_depot_field[sim.comp[0]][sim.targets[v].pos])
        want = float(np.clip((200 - 10 - eta) / 200, -1, 1))
        got = float(ob["agents"][0]["us_sigma"][0, s_])
        assert abs(got - want) < 1e-6, (cap, got, want)
        out[cap] = (eta, got)
    assert out[0][0] > out[a0.capacity_max][0]
    print(f"[1b] sigma: cap > 0 -> ETA = BFS {out[a0.capacity_max][0]} (sigma {out[a0.capacity_max][1]:.3f}); cap 0 -> via depot "
          f"ETA {out[0][0]} (sigma {out[0][1]:.3f})")


def us_gnn(D=32, L=2, edge_update=None):
    return HeteroGNN({"U": FU, "S": FS}, {"UU": EUU, "US": EUS, "SS": ESS}, BASE, REL_US, D, L, 4,
                     edge_update=edge_update).eval()


def rand_graph(G=2, N=5, S=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    x = {"U": r(G, N, FU), "S": r(G, S, FS)}
    mask = {"U": torch.ones(G, N, dtype=torch.bool), "S": torch.ones(G, S, dtype=torch.bool)}
    sym = lambda n: (lambda a: a | a.transpose(1, 2))(r(G, n, n) < 0.5) & ~torch.eye(n, dtype=torch.bool)[None]
    edge = {"UU": r(G, N, N, EUU), "US": r(G, N, S, EUS), "SS": r(G, S, S, ESS)}
    adj = {"UU": sym(N), "US": r(G, N, S) < 0.7, "SS": r(G, S, S) < 0.5}
    return x, mask, edge, adj


def permute(x, mask, edge, adj, pu, ps):
    P = {"U": pu, "S": ps}
    return ({t: v[:, P[t]] for t, v in x.items()}, {t: v[:, P[t]] for t, v in mask.items()},
            {k: v[:, P[BASE[k][0]]][:, :, P[BASE[k][1]]] for k, v in edge.items()},
            {k: v[:, P[BASE[k][0]]][:, :, P[BASE[k][1]]] for k, v in adj.items()})


def pad(x, mask, edge, adj, extra=3):
    """종류마다 패딩 노드 extra 개 (무작위 특징, 마스크 0, 간선은 무작위 — adj 는 유효 노드끼리만)."""
    G = x["U"].shape[0]
    x2 = {t: torch.cat([v, torch.rand(G, extra, v.shape[-1]) * 9], 1) for t, v in x.items()}
    m2 = {t: torch.cat([v, torch.zeros(G, extra, dtype=torch.bool)], 1) for t, v in mask.items()}
    e2, a2 = {}, {}
    for k in edge:
        a, b = BASE[k]
        na, nb = x2[a].shape[1], x2[b].shape[1]
        e = torch.rand(G, na, nb, edge[k].shape[-1]) * 9
        e[:, :edge[k].shape[1], :edge[k].shape[2]] = edge[k]
        ad = torch.rand(G, na, nb) < 0.5
        ad[:, :adj[k].shape[1], :adj[k].shape[2]] = adj[k]
        e2[k], a2[k] = e, ad & m2[a][:, :, None] & m2[b][:, None, :]
    return x2, m2, e2, a2


def check_invariance():
    torch.manual_seed(0)
    x, mask, edge, adj = rand_graph()
    pu, ps = torch.randperm(5), torch.randperm(7)
    for eu in (None, "US"):
        gnn = us_gnn(edge_update=eu)
        h, _, e = gnn(x, mask, edge, adj)
        hp, _, ep = gnn(*permute(x, mask, edge, adj, pu, ps))
        for t, p in (("U", pu), ("S", ps)):
            assert torch.allclose(h[t][:, p], hp[t], atol=1e-5), f"순열 불변 실패 ({eu}, {t})"
        assert torch.allclose(e["US"][:, pu][:, :, ps], ep["US"], atol=1e-5), "간선 갱신 순열 불변 실패"
        hpad, _, _ = gnn(*pad(x, mask, edge, adj))
        for t in ("U", "S"):
            n = h[t].shape[1]
            assert torch.allclose(h[t], hpad[t][:, :n], atol=1e-5), f"패딩이 결과를 바꿈 ({eu}, {t})"
            assert (hpad[t][:, n:] == 0).all(), "패딩 노드 출력이 0 이 아님"
    print("[2] attention (and U–S edge update) permutation invariant over U / S; padding has no effect "
          "(padding outputs 0)")


def check_two_hop():
    G, N = 1, 3
    x = {"U": torch.rand(G, N, FU), "S": torch.zeros(G, 1, FS)}
    mask = {"U": torch.ones(G, N, dtype=torch.bool), "S": torch.zeros(G, 1, dtype=torch.bool)}
    uu = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)[None]
    edge = {"UU": torch.rand(G, N, N, EUU), "US": torch.zeros(G, N, 1, EUS), "SS": torch.zeros(G, 1, 1, ESS)}
    adj = {"UU": uu, "US": torch.zeros(G, N, 1, dtype=torch.bool), "SS": torch.zeros(G, 1, 1, dtype=torch.bool)}
    x2 = {k: v.clone() for k, v in x.items()}
    x2["U"][0, 2] += 1.0                                          # C 만 바꿈
    d = {}
    for L in (1, 2):
        torch.manual_seed(0)
        gnn = us_gnn(L=L)
        d[L] = float((gnn(x, mask, edge, adj)[0]["U"][0, 0] - gnn(x2, mask, edge, adj)[0]["U"][0, 0]).abs().max())
    assert d[1] < 1e-7 and d[2] > 1e-5, d
    print(f"[3] 2-hop A–B–C (attention): L=1 |dh_A| {d[1]:.1e}, L=2 |dh_A| {d[2]:.2e}")


def check_independent_nets():
    env, prep, m = make()
    net = m.net
    nets = net.head_networks()
    ids = {h: {id(p) for p in n.parameters()} for h, n in nets.items()}
    hs = list(ids)
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            assert not (ids[hs[a]] & ids[hs[b]]), f"{hs[a]} / {hs[b]} 가 파라미터를 공유"
    params = {h: list(n.parameters()) for h, n in nets.items()}
    got = {}
    for seed in range(5):
        env.seed(seed)
        obs = env.reset()
        for t in range(300):
            m({"obs": obs_t(prep, obs)}, [], None)
            out = m.last_out
            ob = env.sim._obs_cache
            needed = torch.as_tensor(np.array([a["role_needed"][0] for a in ob["agents"]]) > 0.5)[None]
            masks = {"role": needed, "rescue": out["stage"]["rescue"], "explore": out["stage"]["s1"]}
            for h, mk in masks.items():
                key = "role" if h == "role" else "task"                       # rescue / explore 는 task 행동 칸을 나눠 씀
                lg = out["logits"][key]
                mk = mk & ((lg > -1e8).sum(-1) >= 2)
                if h in got or not bool(mk.any()):
                    continue
                lp = torch.log_softmax(lg, -1).gather(-1, out["actions"][key].clamp(0, lg.shape[-1] - 1)[..., None])[..., 0]
                loss = -(lp * mk.float()).sum()
                norms = {}
                for h2, ps in params.items():
                    gr = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
                    norms[h2] = float(sum((x ** 2).sum() for x in gr if x is not None) ** 0.5) if any(x is not None for x in gr) else 0.0
                got[h] = norms
            act = {k: [int(x) for x in out["actions"][k][0]] for k in ("role", "task")}
            obs, _, done, _ = env.step(act)
            if len(got) == 3 or done:
                break
        if len(got) == 3:
            break
    assert len(got) == 3, got
    for h, norms in got.items():
        assert norms[h] > 0 and all(v == 0.0 for h2, v in norms.items() if h2 != h), (h, norms)
    print("[9] three networks (role / rescue / explore) share no parameters; each loss reaches only its own network: "
          + ", ".join(f"{h} -> own {n[h]:.2e}, others {max(v for k, v in n.items() if k != h):.0e}" for h, n in got.items()))



def check_connectivity():
    """그룹 연결도 ω^g: 5 대가 통신 그룹 {3, 2} 로 나뉘면 3 쪽 0.5, 2 쪽 0.25."""
    env, _, _ = make()
    sim = env.sim
    env.seed(0)
    env.reset()
    sim.comp, sim.groups = [0, 0, 0, 1, 1], [[0, 1, 2], [3, 4]]
    c = sim.group_connectivity()
    assert c == [0.5, 0.5, 0.5, 0.25, 0.25], c
    print(f"[4] group connectivity: groups {{3, 2}} -> {c}")


def check_size_feat():
    from env.role.graph_obs import size_feat
    want = {2: 0.238, 5: 0.388, 26: 0.714, 74: 0.935, 100: 1.0, 150: 1.0, 1000: 1.0, 0: 0.0}
    got = {n: size_feat(n, 100) for n in want}
    assert all(abs(got[n] - w) < 1e-3 for n, w in want.items()), got
    env, _, _ = make()
    assert env.sim.explore_cfg["size_ref"] == 100 and abs(env.sim.size_feat(26) - 0.714) < 1e-3
    print("[8] size_feat: " + ", ".join(f"{n} -> {v:.3f}" for n, v in got.items()))


def check_removed_options():
    """삭제된 / 모르는 모델 옵션 (예전 config 에 남은 키) 은 에러 없이 무시."""
    from models.role.net import resolve_model_options as rmo
    out = rmo({"rescue": {"n_heads": 8}, "explore": {"clip": 5.0}, "gnn": {"aggr": "mean_max"}, "share_context": False,
               "task": {"embed_dim": 128, "entropy_coeff_rescue": 0.5}})
    assert "n_heads" not in out["rescue"] and "clip" not in out["explore"] and "aggr" not in out["gnn"] and "task" not in out
    print("[6] removed / unknown model options (old unified task keys, rescue.n_heads, explore.clip, gnn.aggr, ...) -> ignored")


def check_knn_ties():
    from models.role.net import knn_adj
    # 격자 위 점 (맨해튼 거리 -> 동률 많음) 7 개 + 패딩 1 개, k = 2
    pts = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [2, 2], [3, 3], [9, 9]], dtype=torch.float32)
    d = (pts[:, None] - pts[None]).abs().sum(-1)[None] / 10.0
    mask = torch.tensor([[True] * 7 + [False]])
    adj = knn_adj(d, mask, 2)
    assert adj[0, 0].tolist() == [False, True, True, True, True, False, False, False], adj[0, 0]   # 4 개 동률 모두
    assert int(adj[0].sum(-1).max()) > 2 and not adj[0, :, 7].any() and not adj[0, 7].any()
    torch.manual_seed(1)
    perm = torch.cat([torch.randperm(7), torch.tensor([7])])
    adj_p = knn_adj(d[:, perm][:, :, perm], mask[:, perm], 2)
    assert torch.equal(adj[:, perm][:, :, perm], adj_p), "kNN 동률이 노드 순서에 따라 달라짐"
    # GNN 출력도 순서와 무관 (S–S 간선 = 이 인접, 간선 특징 [Δx, Δy, BFS])
    gnn = us_gnn()
    x, m, edge, a = rand_graph(G=1, N=3, S=8, seed=3)
    m["S"] = mask.clone()
    rel = (pts[None, :, :] - pts[:, None, :])[None] / 10.0
    edge["SS"] = torch.cat([rel, d[..., None]], -1)
    a["SS"] = adj
    a["US"] = a["US"] & m["S"][:, None, :]
    h, _, _ = gnn(x, m, edge, a)
    pu = torch.arange(3)
    x2, m2, e2, a2 = permute(x, m, edge, a, pu, perm)
    a2["SS"] = knn_adj(e2["SS"][..., 2], m2["S"], 2)                    # 섞은 순서에서 kNN 을 다시 계산
    h2, _, _ = gnn(x2, m2, e2, a2)
    assert torch.allclose(h["S"][:, perm], h2["S"], atol=1e-5) and torch.allclose(h["U"], h2["U"], atol=1e-5)
    print(f"[7] kNN ties: node 0 (k = 2) gets all {int(adj[0, 0].sum())} equidistant neighbours; adjacency and GNN output "
          f"unchanged under node shuffling")


if __name__ == "__main__":
    register_all()
    check_invariance()
    check_two_hop()
    check_knn_ties()
    check_size_feat()
    check_removed_options()
    check_connectivity()
    check_sigma_depot()
    check_independent_nets()
    check_graph()
    print("test_gnn: all ok")
