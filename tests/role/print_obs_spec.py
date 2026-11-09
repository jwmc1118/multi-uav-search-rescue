"""계층 구조 env 관측 명세 출력 — 관측 키 / shape, 행동 공간, 그리고 세 네트워크 (역할 선택 / rescue / explore) 가 보는 이종 그래프
(노드 U / S / F, 간선 U–U / U–S / U–F / S–S / F–F / S–F, 그룹 context) 특징의 이름 · 값 범위 (사양 env/role/graph_obs.py).
결정 그래프 종류: role = 역할 선택 네트워크, rescue = rescue 네트워크, s1 = explore 네트워크 (1 단계). 세 그래프는 같은 구성.

    python -m tests.role.print_obs_spec [seed] [steps]

env 는 그리디 정책으로 진행하고, 매 step 같은 관측으로 모델 (무작위 가중치) 을 돌려 순차 디코딩 중 실제로 만든 결정 그래프
(models/role/net.py debug_trace) 의 유효 노드 / 간선 (adj = 1) 값만 네트워크별로 모은다.
"""
import sys

import numpy as np
import torch
from ray.rllib.models import ModelCatalog

from env.role.graph_obs import CONTEXT_FEATURES, EDGE_FEATURES, F_FEATURES, S_FEATURES, U_FEATURES
from env.role.gym_env import RoleEnv
from env.role.simulation import ROLE_GROUP_FEATURES, ROLE_SELF_FEATURES
from models.role.rllib_model import RoleModel
from policies.role_rules import RuleDecider
from training.catalog import register_all
from training.config_loader import load_config


def collect(env, model, prep, steps):
    sim = env.sim
    acc = {}
    add = lambda k, v: acc.setdefault(k, []).append(v.reshape(-1, v.shape[-1]).numpy())
    n_graph = {"role": 0, "rescue": 0, "s1": 0}
    for _ in range(steps):
        obs = sim.rl_observation()
        trace = []
        model.net.debug_trace = trace
        with torch.no_grad():
            model({"obs": torch.as_tensor(prep.transform(obs))[None]}, [], None)
        for e in trace:
            g, st = e["graph"], e["stage"]
            n_graph[st] += int(g["ii"].shape[0])
            for t in g["x"]:
                add(f"{st}:{t}", g["x"][t][g["mask"][t]])
            for k in g["edge"]:
                add(f"{st}:{k}", g["edge"][k][g["adj"][k]])
            add(f"{st}:ctx", g["ctx"])
        for i, ob in enumerate(obs["agents"]):
            for k in ("role_self", "role_group"):
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
    print(f"그래프 크기: U = N {sim.n_agents} (연결 + 메모리 노드), S = s_max {sim.s_max}, F = k_max {sim.k_max}.  "
          f"결정 그래프 수 {n_graph} (role = 역할 선택, rescue = rescue 네트워크, s1 = explore 네트워크)")
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
    names = {"U": U_FEATURES, "S": S_FEATURES, "F": F_FEATURES}
    for st, title in (("role", "역할 선택 (U · S · F)"), ("rescue", "rescue 네트워크 (U · S · F)"),
                      ("s1", "explore 네트워크 — 1 단계 (U · S · F)")):
        print(f"\n######## {title} ########")
        for t in ("U", "S", "F"):
            if f"{st}:{t}" in vals:
                table(f"{t} 노드 {len(names[t])}", names[t], vals[f"{st}:{t}"])
        if f"{st}:ctx" in vals:
            table(f"그룹 context {len(CONTEXT_FEATURES)}", CONTEXT_FEATURES, vals[f"{st}:ctx"])
        for k, nm in EDGE_FEATURES.items():
            if f"{st}:{k}" in vals:
                table(f"{k[0]}–{k[1]} 간선 {len(nm)}", nm, vals[f"{st}:{k}"])
    table("critic role_self", ROLE_SELF_FEATURES, vals.get("role_self"))
    table("critic role_group", ROLE_GROUP_FEATURES, vals.get("role_group"))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
