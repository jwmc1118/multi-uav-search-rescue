"""이종 그래프 GNN encoder + 연결성 보상 점검 (CPU 로 충분).

    python -m tests.role.test_gnn

[1] 그래프 구성 (모델이 직접 env 를 굴리며 순차 디코딩 trace 를 매 step 검사):
    U 노드 = 결정 UAV 의 통신 그룹만, S 노드 = 그룹이 아는 생존자만, U–S BFS / slack 값 = env 거리,
    같은 이벤트 앞 차례 결정 반영 (결정함 플래그, RESCUE 목표 지점, 고른 생존자 맡겨짐 / targets 간선)
[2] 순열 불변 (U / S / K 노드 순서), 패딩 노드 무영향, 2-hop 전파 (A–B–C 사슬: 1 층이면 C 가 A 에 안 닿고 2 층이면 닿음)
[3] GNN 하나를 세 head 가 공유, head 별 loss 의 gradient 가 모두 GNN 에 도달
[4] 개인 연결도 c_i = (|comp(i)| - 1) / (N - 1): 5 대가 {3, 2} -> 3 쪽 0.5, 2 쪽 0.25, 한 대 -> 0
[5] 보상: rescue F > 0 이면 개인 보너스 0 (c̄ 무관), F = 0 이면 c̄ 0 -> 1 에 w_c 만큼, explore Δcov = 0 이면 c 와 무관하게 0
[7] kNN 동률 (S–S / K–K): k 번째 이웃과 거리가 같은 이웃은 모두 포함, 노드 순서를 섞어도 인접 · GNN 출력이 같음
[8] 크기 특징 size_feat(n) = min(1, log(1 + n) / log(1 + 100)): 2 -> 0.238, 5 -> 0.388, 26 -> 0.714 (사양의 0.712 는 식 값과 0.002 차이), 74 -> 0.935, >= 100 -> 1, 0 -> 0
[6] 제거된 모델 옵션 (role.arch / 헤드별 embed_dim · knn_k / share_context / env_features ...) 을 주면 읽기 쉬운 에러
"""
import copy

import numpy as np
import torch
from ray.rllib.models import ModelCatalog

from env.core.modes import Mode
from env.core.targets import deadline
from env.role.graph_obs import EUS, EUU, FU
from env.role.gym_env import RoleEnv
from models.role.gnn import HeteroGNN
from models.role.rllib_model import RoleModel
from training.catalog import register_all
from training.config_loader import load_config

U_DECIDED, U_GOAL = 9, slice(7, 9)


def make(n_agents=None, seed=0):
    cfg = load_config()
    ec = copy.deepcopy(cfg["env_config"])
    if n_agents:
        ec["agents"] = [dict(ec["agents"][0]) for _ in range(n_agents)]
    ec["policy_mode"] = {"role": "learned", "rescue": "learned", "explore": "learned"}
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
    n_entries = n_seq = n_split = 0
    for t in range(steps):
        trace = []
        m.net.debug_trace = trace
        with torch.no_grad():
            m({"obs": obs_t(prep, obs)}, [], None)
        ob = sim._obs_cache
        group = ob["group"] > 0.5
        n_split += int(len(sim.groups) > 1 and any(e["stage"] == "role" for e in trace))
        prev = set()                                                    # 앞 role 차례에 결정한 UAV
        for e in trace:
            g = e["graph"]
            for r, (b, i) in enumerate(zip(g["bi"].tolist(), g["ii"].tolist())):
                n_entries += 1
                mU = g["mask"]["U"][r].numpy()
                assert (mU == group[i]).all(), "그룹 밖 UAV 가 U 노드에 있음"
                sid = [int(x) for x in ob["agents"][i]["s_ids"] if x >= 0]
                known = set(sim._group_known(sim.comp[i]))
                assert set(sid) <= known, f"그룹이 모르는 생존자가 S 노드에 있음 {set(sid) - known}"
                assert int(g["mask"]["S"][r].sum()) == len(sid)
                eUS = g["edge"]["US"][r].numpy()
                for s_, ti in enumerate(sid):
                    for l in np.flatnonzero(mU):
                        d = sim._dist(l, sim.targets[ti].pos)
                        assert abs(eUS[l, s_, 0] - sim._nd(d)) < 1e-5, "U–S BFS 값 불일치"
                        if d < 10 ** 9:
                            sl = np.clip((deadline(sim.targets[ti]) - sim.time - d) / sim.cfg["max_episode_len"], -1, 1)
                            assert abs(eUS[l, s_, 1] - sl) < 1e-5, "U–S slack 값 불일치"
                if e["stage"] == "role":                                 # 결정함 플래그 = 앞 차례에 결정한 그룹원
                    for j in np.flatnonzero(mU):
                        want = 1.0 if j in prev else 0.0
                        assert float(g["x"]["U"][r, j, U_DECIDED]) == want, "앞 차례 결정이 U 노드에 반영 안 됨"
                        n_seq += int(j in prev)
            if e["stage"] == "role":
                prev |= set(g["ii"].tolist())
        out = m.last_out
        act = {k: [int(x) for x in out["actions"][k][0]] for k in ("role", "rescue", "explore", "explore_sub")}
        obs, _, done, info = env.step(act)
        assert info["mask_violations"] == 0
        if done:
            obs = env.reset()
    m.net.debug_trace = None
    assert n_seq > 0 and n_entries > 0
    print(f"[1] graph: {n_entries} decision graphs checked (group-only U, known-only S, U–S BFS / slack), "
          f"{n_seq} sequential-update checks, {n_split} steps with >1 comm group and role decisions")


def check_sequential_rescue():
    """같은 forward 안에서 RESCUE 를 고른 앞 차례의 목표 지점 / S 맡겨짐이 다음 차례 그래프에 들어가는지 (직접 재생)."""
    env, prep, m = make()
    found = 0
    for seed in range(6):
        env.seed(seed)
        obs = env.reset()
        for t in range(250):
            trace = []
            m.net.debug_trace = trace
            with torch.no_grad():
                m({"obs": obs_t(prep, obs)}, [], None)
            out = m.last_out
            ob = env.sim._obs_cache
            ra, rs = out["actions"]["role"][0], out["actions"]["rescue"][0]
            roles = [e for e in trace if e["stage"] == "role"]
            for k in range(1, len(roles)):
                prev = set()
                for e in roles[:k]:
                    prev |= set(e["graph"]["ii"].tolist())
                g = roles[k]["graph"]
                for r, i in enumerate(g["ii"].tolist()):
                    for j in prev:
                        if ob["group"][i, j] > 0.5 and int(ra[j]) == 1:
                            vid = ob["agents"][j]["s_ids"][int(rs[j])]
                            sid = list(ob["agents"][i]["s_ids"])
                            assert g["x"]["U"][r, j, U_DECIDED] == 1.0
                            gp = torch.as_tensor(ob["agents"][j]["s_pos"][int(rs[j])]) - torch.as_tensor(ob["team_ctx"][i, 6:8])
                            assert torch.allclose(g["x"]["U"][r, j, U_GOAL], gp, atol=1e-5), "RESCUE 목표 지점 미반영"
                            if vid in sid:
                                s = sid.index(vid)
                                assert g["x"]["S"][r, s, 3] == 1.0, "앞 차례가 고른 생존자가 맡겨짐으로 안 바뀜"
                                assert g["edge"]["US"][r, j, s, 3] == 1.0, "targets 간선 미반영"
                            found += 1
            act = {k: [int(x) for x in out["actions"][k][0]] for k in ("role", "rescue", "explore", "explore_sub")}
            obs, _, done, _ = env.step(act)
            if done:
                break
        if found >= 5:
            break
    m.net.debug_trace = None
    assert found > 0, "같은 이벤트에서 RESCUE 앞 차례가 있는 경우를 못 찾음"
    print(f"[1b] sequential RESCUE updates (decided / goal point / S held / targets edge): {found} checks")


def rand_graph(G=2, N=5, S=7, K=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    x = {"U": r(G, N, FU), "S": r(G, S, 4), "K": r(G, K, 4), "D": r(G, 1, 2)}
    mask = {"U": torch.ones(G, N, dtype=torch.bool), "S": torch.ones(G, S, dtype=torch.bool),
            "K": torch.ones(G, K, dtype=torch.bool), "D": torch.ones(G, 1, dtype=torch.bool)}
    sym = lambda n: (lambda a: a | a.transpose(1, 2))(r(G, n, n) < 0.5) & ~torch.eye(n, dtype=torch.bool)[None]
    edge = {"US": r(G, N, S, EUS), "UK": r(G, N, K, 3), "UU": r(G, N, N, EUU), "UD": r(G, N, 1, 1),
            "SS": r(G, S, S, 1), "KK": r(G, K, K, 1)}
    adj = {"US": r(G, N, S) < 0.7, "UK": r(G, N, K) < 0.7, "UU": sym(N), "UD": torch.ones(G, N, 1, dtype=torch.bool),
           "SS": r(G, S, S) < 0.5, "KK": r(G, K, K) < 0.5}
    return x, mask, edge, adj


def permute(x, mask, edge, adj, pu, ps, pk):
    P = {"U": pu, "S": ps, "K": pk, "D": torch.arange(1)}
    x2 = {t: v[:, P[t]] for t, v in x.items()}
    m2 = {t: v[:, P[t]] for t, v in mask.items()}
    base = {"US": "US", "UK": "UK", "UU": "UU", "UD": "UD", "SS": "SS", "KK": "KK"}
    e2, a2 = {}, {}
    for k in base:
        a, b = k[0], k[1]
        e2[k] = edge[k][:, P[a]][:, :, P[b]]
        a2[k] = adj[k][:, P[a]][:, :, P[b]]
    return x2, m2, e2, a2


def pad(x, mask, edge, adj, extra=3):
    """종류마다 패딩 노드 extra 개 (무작위 특징, 마스크 0, 간선은 무작위 — adj 는 유효 노드끼리만)."""
    G = x["U"].shape[0]
    x2 = {t: torch.cat([v, torch.rand(G, extra if t != "D" else 0, v.shape[-1]) * 9], 1) for t, v in x.items()}
    m2 = {t: torch.cat([v, torch.zeros(G, extra if t != "D" else 0, dtype=torch.bool)], 1) for t, v in mask.items()}
    e2, a2 = {}, {}
    for k in edge:
        a, b = k[0], k[1]
        na, nb = x2[a].shape[1], x2[b].shape[1]
        e = torch.rand(G, na, nb, edge[k].shape[-1]) * 9
        e[:, :edge[k].shape[1], :edge[k].shape[2]] = edge[k]
        ad = torch.rand(G, na, nb) < 0.5                   # 패딩 쪽 adj 는 무작위 -> 마스크로 거른다
        ad[:, :adj[k].shape[1], :adj[k].shape[2]] = adj[k]
        ad = ad & m2[a][:, :, None] & m2[b][:, None, :]
        e2[k], a2[k] = e, ad
    return x2, m2, e2, a2


def check_invariance():
    torch.manual_seed(0)
    gnn = HeteroGNN(32, 2).eval()
    x, mask, edge, adj = rand_graph()
    h = gnn(x, mask, edge, adj)
    pu, ps, pk = torch.randperm(5), torch.randperm(7), torch.randperm(4)
    hp = gnn(*permute(x, mask, edge, adj, pu, ps, pk))
    for t, p in (("U", pu), ("S", ps), ("K", pk)):
        assert torch.allclose(h[t][:, p], hp[t], atol=1e-5), f"순열 불변 실패 ({t})"
    hpad = gnn(*pad(x, mask, edge, adj))
    for t in ("U", "S", "K", "D"):
        n = h[t].shape[1]
        assert torch.allclose(h[t], hpad[t][:, :n], atol=1e-5), f"패딩이 결과를 바꿈 ({t})"
        assert (hpad[t][:, n:] == 0).all(), "패딩 노드 출력이 0 이 아님"
    # 2-hop: U 노드 A(0)–B(1)–C(2) 사슬만 (다른 종류 / 간선 없음)
    G, N = 1, 3
    x = {"U": torch.rand(G, N, FU), "S": torch.zeros(G, 1, 4), "K": torch.zeros(G, 1, 4), "D": torch.zeros(G, 1, 2)}
    mask = {"U": torch.ones(G, N, dtype=torch.bool), "S": torch.zeros(G, 1, dtype=torch.bool),
            "K": torch.zeros(G, 1, dtype=torch.bool), "D": torch.zeros(G, 1, dtype=torch.bool)}
    uu = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)[None]
    edge = {"US": torch.zeros(G, N, 1, EUS), "UK": torch.zeros(G, N, 1, 3), "UU": torch.rand(G, N, N, EUU),
            "UD": torch.zeros(G, N, 1, 1), "SS": torch.zeros(G, 1, 1, 1), "KK": torch.zeros(G, 1, 1, 1)}
    adj = {"US": torch.zeros(G, N, 1, dtype=torch.bool), "UK": torch.zeros(G, N, 1, dtype=torch.bool), "UU": uu,
           "UD": torch.zeros(G, N, 1, dtype=torch.bool), "SS": torch.zeros(G, 1, 1, dtype=torch.bool),
           "KK": torch.zeros(G, 1, 1, dtype=torch.bool)}
    x2 = {k: v.clone() for k, v in x.items()}
    x2["U"][0, 2] += 1.0                                          # C 만 바꿈
    g1, g2 = HeteroGNN(32, 1).eval(), HeteroGNN(32, 2).eval()
    d1 = (g1(x, mask, edge, adj)["U"][0, 0] - g1(x2, mask, edge, adj)["U"][0, 0]).abs().max()
    d2 = (g2(x, mask, edge, adj)["U"][0, 0] - g2(x2, mask, edge, adj)["U"][0, 0]).abs().max()
    assert d1 < 1e-7 and d2 > 1e-5, (float(d1), float(d2))
    print(f"[2] permutation invariant (U/S/K), padding has no effect, 2-hop A–B–C: L=1 |Δh_A| {float(d1):.1e}, "
          f"L=2 |Δh_A| {float(d2):.2e}")


def check_sharing():
    env, prep, m = make()
    net = m.net
    params = list(net.gnn.parameters())
    assert net.shared_encoder_parameters() == params
    # GNN 은 net.gnn 하나 (head 별 encoder 없음)
    assert sum(isinstance(mod, HeteroGNN) for mod in net.modules()) == 1
    got = {}
    for seed in range(5):
        env.seed(seed)
        obs = env.reset()
        for t in range(300):
            m({"obs": obs_t(prep, obs)}, [], None)
            out = m.last_out
            st = out["stage"]
            ob = env.sim._obs_cache
            needed = torch.as_tensor(np.array([a["role_needed"][0] for a in ob["agents"]]) > 0.5)[None]
            masks = {"role": needed, "rescue": needed & (out["actions"]["role"] == 1), "explore": st["s1"]}
            for h, mk in masks.items():
                lg = out["logits"][h]
                mk = mk & ((lg > -1e8).sum(-1) >= 2)                       # 후보 1 개면 logp ≡ 0 (gradient 0)
                if h in got or not bool(mk.any()):
                    continue
                lp = torch.log_softmax(lg, -1).gather(-1, out["actions"][h].clamp(0, lg.shape[-1] - 1)[..., None])[..., 0]
                loss = -(lp * mk.float()).sum()
                gr = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                got[h] = float(torch.sqrt(sum((x ** 2).sum() for x in gr if x is not None)))
            act = {k: [int(x) for x in out["actions"][k][0]] for k in ("role", "rescue", "explore", "explore_sub")}
            obs, _, done, _ = env.step(act)
            if len(got) == 3 or done:
                break
        if len(got) == 3:
            break
    assert len(got) == 3 and all(v > 0 for v in got.values()), got
    print(f"[3] one shared GNN; grad norm on GNN from each head: " + ", ".join(f"{h} {v:.3e}" for h, v in got.items()))


def check_connectivity():
    """개인 연결도 c_i: 5 대가 {3, 2} 로 나뉘면 3 쪽 0.5, 2 쪽 0.25, 한 대면 0."""
    from env.role.connectivity import personal_connectivity
    R = 10.0
    pts = [(0, 0), (0, R), (0, 2 * R), (100, 0), (100, R)]
    c = personal_connectivity(pts, R)
    assert c == [0.5, 0.5, 0.5, 0.25, 0.25], c
    assert personal_connectivity([(0, 0)], R) == [0.0]
    print(f"[4] personal connectivity: 5 UAVs split {{3, 2}} -> {c}; single UAV -> [0.0]")


def check_rewards():
    """rescue 개인 보너스: F > 0 이면 c̄ 와 무관하게 0, F = 0 이면 c̄ 0 -> 1 에 w_c 만큼. explore: Δcov = 0 이면 c 와 무관하게 0."""
    env, _, _ = make()
    sim = env.sim
    env.seed(2)
    env.reset()
    w_c = float(sim.rewards_cfg["rescue"]["w_c"])
    n = sim.n_agents

    def bonus(F, c):
        sim._bonus_pending = [0.0] * n
        sim._res_opt = [{"F": F, "d": 0.2, "csum": 0.0, "n": 0}] + [None] * (n - 1)
        sim._rescue_options([c] + [0.0] * (n - 1), [], True)                 # 옵션 1 step, 에피소드 끝으로 닫힘
        return sim._bonus_pending[0]
    f_lo, f_hi = bonus(1, 0.0), bonus(1, 1.0)
    z_lo, z_hi = bonus(0, 0.0), bonus(0, 1.0)
    assert f_lo == f_hi == 0.0 and abs((z_hi - z_lo) - w_c) < 1e-12, (f_lo, f_hi, z_lo, z_hi)
    sim.knowledge.new_sets = [set() for _ in range(n)]                       # Δcov = 0
    for c in ([0.0] * n, [1.0] * n):
        sim.personal_connectivity = lambda c=c: c
        e = sim.role_rewards([], [], explore_active=[True] * n)["explore"]
        assert e == [0.0] * n, e
    del sim.personal_connectivity
    print(f"[5] rescue F > 0: bonus {f_lo} (c̄ 0) / {f_hi} (c̄ 1); F = 0: Δbonus(c̄ 0 -> 1) = {z_hi - z_lo:.3f} (w_c {w_c}); "
          f"explore dcov = 0 -> reward 0 for any c")


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
    # GNN 출력도 순서와 무관 (S–S 간선 = 이 인접)
    gnn = HeteroGNN(32, 2).eval()
    x, m, edge, a = rand_graph(G=1, N=3, S=8, K=2, seed=3)
    m["S"] = mask.clone()
    edge["SS"] = d[..., None]
    a["SS"] = adj
    a["US"] = a["US"] & m["S"][:, None, :]
    h = gnn(x, m, edge, a)
    pu, pk = torch.arange(3), torch.arange(2)
    x2, m2, e2, a2 = permute(x, m, edge, a, pu, perm, pk)
    a2["SS"] = knn_adj(e2["SS"][..., 0], m2["S"], 2)                    # 섞은 순서에서 kNN 을 다시 계산
    h2 = gnn(x2, m2, e2, a2)
    assert torch.allclose(h["S"][:, perm], h2["S"], atol=1e-5) and torch.allclose(h["U"], h2["U"], atol=1e-5)
    print(f"[7] kNN ties: node 0 (k = 2) gets all {int(adj[0, 0].sum())} equidistant neighbours; adjacency and GNN output "
          f"unchanged under node shuffling")


def check_size_feat():
    from env.role.graph_obs import size_feat
    want = {2: 0.238, 5: 0.388, 26: 0.714, 74: 0.935, 100: 1.0, 150: 1.0, 1000: 1.0, 0: 0.0}
    got = {n: size_feat(n, 100) for n in want}
    assert all(abs(got[n] - w) < 1e-3 for n, w in want.items()), got
    env, _, _ = make()
    assert env.sim.explore_cfg["size_ref"] == 100 and abs(env.sim.size_feat(26) - 0.714) < 1e-3
    print("[8] size_feat: " + ", ".join(f"{n} -> {v:.3f}" for n, v in got.items()))


def check_removed_options():
    from models.role.net import resolve_model_options
    for opt in ({"role": {"arch": "capam"}}, {"rescue": {"embed_dim": 128}}, {"explore": {"knn_k": 6}},
                {"share_context": False}, {"env_features": {}}, {"role": {"role_encoder": "shared"}}):
        try:
            resolve_model_options(opt)
        except ValueError as e:
            assert "제거된 옵션" in str(e), e
        else:
            raise AssertionError(f"{opt} 에러 안 남")
    print("[6] removed model options -> readable error")


if __name__ == "__main__":
    register_all()
    check_invariance()
    check_connectivity()
    check_rewards()
    check_sharing()
    check_graph()
    check_sequential_rescue()
    check_removed_options()
    check_knn_ties()
    check_size_feat()
    print("test_gnn: all ok")
