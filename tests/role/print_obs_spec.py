"""계층 구조 env 관측 명세 출력 — 관측 키 / shape, 행동 공간, 그리고 GNN 이 보는 이종 그래프 (노드 U / S / K / D, 간선
U–S / U–K / U–U / U–D / S–S / K–K) 특징의 이름 · 값 범위 (사양 env/role/graph_obs.py).

    python -m tests.role.print_obs_spec [seed] [steps]

env 는 그리디 정책으로 진행하고, 매 step 같은 관측으로 모델 (무작위 가중치) 을 돌려 순차 디코딩 중 실제로 만든 결정 그래프
(models/role/net.py debug_trace) 의 유효 노드 / 간선 (adj = 1) 값만 모은다. 2 단계 그래프의 K = 구간.
"""
import sys

import numpy as np
import torch
from ray.rllib.models import ModelCatalog

from env.role.explore import seg_offset
from env.role.graph_obs import D_FEATURES, EDGE_FEATURES, K_FEATURES, S_FEATURES, U_FEATURES
from env.role.gym_env import RoleEnv
from env.role.simulation import ROLE_G_EXP_FEATURES, ROLE_G_RES_FEATURES, ROLE_GROUP_FEATURES, ROLE_SELF_FEATURES
from models.role.rllib_model import RoleModel
from policies.role_rules import RuleDecider
from training.catalog import register_all
from training.config_loader import load_config


def collect(env, model, prep, steps):
    sim = env.sim
    acc = {}
    add = lambda k, v: acc.setdefault(k, []).append(v.reshape(-1, v.shape[-1]).numpy())
    n_graph = {"role": 0, "s1": 0, "s2": 0}
    for _ in range(steps):
        obs = sim.rl_observation()
        trace = []
        model.net.debug_trace = trace
        with torch.no_grad():
            model({"obs": torch.as_tensor(prep.transform(obs))[None]}, [], None)
        for e in trace:
            g, st = e["graph"], e["stage"]
            n_graph[st] += int(g["ii"].shape[0])
            for t in ("U", "S", "D"):
                add(f"{t}", g["x"][t][g["mask"][t]])
            add("K2" if st == "s2" else "K1", g["x"]["K"][g["mask"]["K"]])
            for k in EDGE_FEATURES:
                key = k + ("2" if st == "s2" and k in ("UK", "KK") else "")
                add(key, g["edge"][k][g["adj"][k]])
        for i, ob in enumerate(obs["agents"]):
            for k in ("role_self", "role_group", "role_g_res", "role_g_exp"):
                acc.setdefault(k, []).append(ob[k][None])
        done, _ = sim.step(RuleDecider())
        if done:
            break
    model.net.debug_trace = None
    return {k: np.concatenate(v) for k, v in acc.items()}, n_graph


def table(title, names, vals):
    print(f"\n== {title} ==")
    for k, name in enumerate(names):
        if vals is None or not len(vals):
            rng = "(no sample)"
        else:
            v = vals[:, k]
            rng = f"[{v.min():+.3f}, {v.max():+.3f}]  mean {v.mean():+.3f}"
        print(f"  {k:>2} {name:<22} observed {rng}")


def main(seed=0, steps=120):
    register_all()
    cfg = load_config()
    env = RoleEnv(cfg["env_config"])
    prep = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    model = RoleModel(prep.observation_space, env.action_space, 1,
                      {"custom_model_config": cfg["model"]["custom_model_config"]}, "spec")
    env.seed(seed)
    env.reset()
    sim = env.sim
    vals, n_graph = collect(env, model, prep, steps)
    ag = env.observation_space["agents"][0].spaces
    print(f"정규화: 좌표 (x / W, y / H), 거리 BFS / d_max (d_max = {sim.d_max:.0f}), U–U 거리 / R_c (상한 4), "
          f"시간 / T (T = {sim.cfg['max_episode_len']}), 크기 = min(1, log(1+n) / log(1+{sim.explore_cfg['size_ref']})), 배정 수 / N (N = {sim.n_agents})")
    print(f"그래프 크기: U = N {sim.n_agents}, S = s_max {sim.s_max}, K = k_max {sim.k_max} (1 단계) / n_sub_max "
          f"{sim.n_sub_max} (2 단계), D = 1.  결정 그래프 수 {n_graph}")
    print("== per-agent keys ==")
    for k, sp in ag.items():
        print(f"  {k:<18} {str(sp.shape):<14}")
    print("== team keys ==")
    for k, sp in env.observation_space.spaces.items():
        if k != "agents":
            print(f"  {k:<18} {str(sp.shape):<14}")
    print("== action ==")
    for k, sp in env.action_space.spaces.items():
        print(f"  {k:<12} N x Discrete({sp.spaces[0].n})")
    table(f"U 노드 (UAV, 그룹만) {len(U_FEATURES)}", U_FEATURES, vals.get("U"))
    table(f"S 노드 (생존자, 그룹이 아는 것만) {len(S_FEATURES)}", S_FEATURES, vals.get("S"))
    table(f"K 노드 1 단계 (영역) {len(K_FEATURES)}", K_FEATURES, vals.get("K1"))
    table(f"K 노드 2 단계 (구간) {len(K_FEATURES)}", K_FEATURES, vals.get("K2"))
    table(f"D 노드 (depot) {len(D_FEATURES)}", D_FEATURES, vals.get("D"))
    for k, names in EDGE_FEATURES.items():
        table(f"{k[0]}–{k[1]} 간선{' (1 단계)' if k in ('UK', 'KK') else ''}", names, vals.get(k))
        if k in ("UK", "KK"):
            table(f"{k[0]}–{k[1]} 간선 (2 단계 구간)", names, vals.get(k + "2"))
    table("Role g_res (role_g_res)", ROLE_G_RES_FEATURES, vals.get("role_g_res"))
    table("Role g_exp (role_g_exp)", ROLE_G_EXP_FEATURES, vals.get("role_g_exp"))
    table("critic role_self", ROLE_SELF_FEATURES, vals.get("role_self"))
    table("critic role_group", ROLE_GROUP_FEATURES, vals.get("role_group"))
    print(f"\n2 단계 표 seg_* (K={sim.k_max}, P={sim.n_seg_slots}): "
          + ", ".join(f"kk={kk} -> [{seg_offset(kk)}:{seg_offset(kk) + kk}]" for kk in range(2, sim.n_sub_max + 1)))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
