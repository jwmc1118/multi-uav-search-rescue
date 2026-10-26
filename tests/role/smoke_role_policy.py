"""계층 구조 학습 경로 점검 (ray 클러스터 없이 RolePPOTorchPolicy 를 직접 만든다).

    python -m tests.role.smoke_role_policy

1) policy_mode 4 조합 x (rollout -> 헤드별 GAE -> learn_on_batch 2 회): NaN 없음, 헤드별 stats 키, rule 헤드는 마스크 0
2) 학습 재생: SampleBatch.ACTIONS 로 순차 디코딩을 재생한 logits = rollout logits (결정한 헤드 / 에이전트에서)
   + 모델이 계산한 explore 1 단계 / 2 단계 결정 (s1 / s2d) = env 가 실제로 한 결정 (explore_valid_s1 / s2), 마스크 위반 0
3) 모델 파라미터가 N 과 무관 (N=4 / N=5 state_dict shape 동일), N=4 가중치를 N=5 모델에 실어 한 에피소드 실행
4) 헤드별 파라미터 수
"""
import copy

import numpy as np
import torch
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from env.role.gym_env import RoleEnv
from training.catalog import register_all
from training.role.ppo import RolePPOTorchPolicy
from training.config_loader import load_config

STEPS = 120
COMBOS = [("rule", "rule", "rule"), ("learned", "rule", "rule"), ("learned", "learned", "rule"),
          ("learned", "learned", "learned")]
HEAD_KEYS = [f"{s}_{h}" for h in ("role", "rescue", "explore")
             for s in ("vf_explained_var", "policy_loss", "vf_loss", "entropy", "kl", "n_decisions",
                       "value_target_mean", "value_target_std", "value_norm_mean")] + \
    ["n_decisions_explore_s1", "n_decisions_explore_s2"]
ACT_ORDER = ("explore", "explore_sub", "rescue", "role")


def base_config(n_agents=None, mode=None):
    y = load_config()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in y.items():
        if k == "model":
            cfg["model"].update(copy.deepcopy(v))
        elif k != "env":
            cfg[k] = copy.deepcopy(v)
    cfg.update(num_workers=0, num_gpus=0, train_batch_size=STEPS, sgd_minibatch_size=60, num_sgd_iter=1)
    if n_agents:
        cfg["env_config"]["agents"] = [dict(cfg["env_config"]["agents"][0]) for _ in range(n_agents)]
    if mode:
        cfg["env_config"]["policy_mode"] = dict(zip(("role", "rescue", "explore"), mode))
    return cfg


def make(cfg):
    env = RoleEnv(cfg["env_config"])
    prep = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    return env, prep, RolePPOTorchPolicy(prep.observation_space, env.action_space, cfg)


def collect(env, prep, policy, seed=0):
    env.seed(seed)
    obs = env.reset()
    rows = {k: [] for k in ("obs", "new_obs", "actions", "rewards", "dones", "infos", "logits")}
    extras = []
    mism = {"s1": 0, "s2": 0, "steps": 0, "s1_n": 0, "s2_n": 0}
    N = env.n_agents
    for t in range(STEPS):
        ob = prep.transform(obs)
        actions, _, extra = policy.compute_actions(ob[None], explore=True)
        act = {k: [int(np.asarray(a)[0]) for a in v] for k, v in actions.items()}
        flags = np.asarray(extra["action_dist_inputs"])[0][-2 * N:]
        st = policy.model.last_out["stage"]
        s1_m = st["s1"][0].cpu().numpy()
        part_m = st["s2d"][0].cpu().numpy()
        nobs, r, done, info = env.step(act)
        mode = env.sim.policy_mode
        if mode["role"] == "learned" and mode["explore"] == "learned":   # 모델 = env 결정 경로가 같아야 하는 조합
            mism["steps"] += 1
            mism["s1"] += int(np.any(s1_m != (np.asarray(info["explore_valid_s1"]) > 0)))
            mism["s2"] += int(np.any(part_m != (np.asarray(info["explore_valid_s2"]) > 0)))
            mism["s1_n"] += int(np.sum(info["explore_valid_s1"]))
            mism["s2_n"] += int(np.sum(info["explore_valid_s2"]))
            assert info["mask_violations"] == 0, (t, info["mask_violations"])
        rows["obs"].append(ob); rows["new_obs"].append(prep.transform(nobs))
        rows["actions"].append(np.concatenate([np.asarray(act[k], np.float32) for k in ACT_ORDER]))
        rows["rewards"].append(r); rows["dones"].append(done); rows["infos"].append(info)
        extras.append({k: np.asarray(v)[0] for k, v in extra.items()})
        obs = env.reset() if done else nobs
    rows["dones"][-1] = False
    batch = SampleBatch({
        SampleBatch.OBS: np.array(rows["obs"], np.float32), SampleBatch.NEXT_OBS: np.array(rows["new_obs"], np.float32),
        SampleBatch.ACTIONS: np.array(rows["actions"], np.float32), SampleBatch.REWARDS: np.array(rows["rewards"], np.float32),
        SampleBatch.DONES: np.array(rows["dones"]), SampleBatch.INFOS: np.array(rows["infos"], dtype=object),
        SampleBatch.EPS_ID: np.zeros(STEPS, np.int64), SampleBatch.AGENT_INDEX: np.zeros(STEPS, np.int64),
        **{k: np.stack([e[k] for e in extras]) for k in extras[0]},
    })
    collect.mismatch = mism
    return policy.postprocess_trajectory(batch)


def check_replay(policy, batch):
    """학습 forward (ACTIONS 로 재생) 의 logits 가 rollout 때 저장된 logits 와 같은지 (결정한 헤드 · 에이전트에서)."""
    m = policy.model
    N, T, K, M = m.n_agents, m.max_targets, m.k_max, m.n_sub
    t = policy._lazy_tensor_dict(batch.copy())
    with torch.no_grad():
        lg, _ = m.from_batch(t, is_training=True)
    lg = lg.cpu().numpy()
    prev = np.asarray(batch[SampleBatch.ACTION_DIST_INPUTS])
    o = [0, N * K, N * K + N * M, N * K + N * M + N * T, N * K + N * M + N * T + N * 3]
    seg = {"explore": (o[0], o[1], K, "explore"), "explore_sub": (o[1], o[2], M, "explore"),
           "rescue": (o[2], o[3], T, "rescue"), "role": (o[3], o[4], 3, "role")}
    s2 = prev[:, -N:] > 0.5
    worst = 0.0
    for h, (a, b, w, head) in seg.items():
        mask = np.asarray(batch[f"{head}_train_mask"]) > 0
        if h == "explore_sub":
            mask = mask & s2
        x, y = lg[:, a:b].reshape(-1, N, w), prev[:, a:b].reshape(-1, N, w)
        if mask.any():
            worst = max(worst, float(np.abs(x[mask] - y[mask]).max()))
    assert worst < 1e-3, f"재생 logits 불일치 {worst}"
    return worst


def check_combos():
    for mode in COMBOS:
        torch.manual_seed(0)
        env, prep, policy = make(base_config(mode=mode))
        batch = collect(env, prep, policy)
        for h in ("role", "rescue", "explore"):
            for k in (f"advantages_{h}" if h != "role" else "advantages", f"{h}_train_mask"):
                assert np.isfinite(batch[k]).all(), (mode, k)
        dec = {h: int(np.sum(batch[f"{h}_train_mask"])) for h in ("role", "rescue", "explore")}
        for h, m in zip(("role", "rescue", "explore"), mode):
            if m == "rule":
                assert dec[h] == 0 and np.sum(batch[f"{h}_value_mask"]) == 0, (mode, h)
        err = check_replay(policy, batch)
        for _ in range(2):
            st = policy.learn_on_batch(batch)
        ls = st.get("learner_stats", st)
        missing = [k for k in HEAD_KEYS if k not in ls]
        assert not missing, missing
        bad = [k for k, v in ls.items() if np.size(v) == 1 and not np.isfinite(float(v))]
        assert not bad, (mode, bad)
        mm = collect.mismatch
        if mm["steps"]:
            assert mm["s1"] == 0 and mm["s2"] == 0, f"모델 / env explore 단계 불일치 {mm}"
        print(f"    explore stage check: steps {mm['steps']}, s1 {mm['s1_n']} s2 {mm['s2_n']} decisions, mismatched steps "
              f"s1 {mm['s1']} / s2 {mm['s2']}")
        print(f"[1] {mode}: decisions {dec}  total_loss {float(ls['total_loss']):.4f}  "
              f"ev role/res/exp {float(ls['vf_explained_var_role']):+.2f}/{float(ls['vf_explained_var_rescue']):+.2f}/"
              f"{float(ls['vf_explained_var_explore']):+.2f}  replay max|Δlogit| {err:.1e}")


def check_n_invariance():
    _, _, p4 = make(base_config(n_agents=4))
    env5, prep5, p5 = make(base_config(n_agents=5))
    s4, s5 = p4.model.state_dict(), p5.model.state_dict()
    assert s4.keys() == s5.keys() and all(s4[k].shape == s5[k].shape for k in s4), "N 의존 파라미터"
    p5.model.load_state_dict(s4, strict=True)
    p5.model.greedy = True
    env5.seed(3)
    obs = env5.reset()
    done, steps = False, 0
    while not done:
        actions, _, _ = p5.compute_actions(prep5.transform(obs)[None], explore=False)
        obs, _, done, info = env5.step({k: [int(np.asarray(a)[0]) for a in v] for k, v in actions.items()})
        steps += 1
    print(f"[3] N-invariant ({len(s4)} tensors); N=4 weights -> N=5 model: episode done t={steps}, "
          f"rescued {info['rescued_total']}/{len(env5.sim.targets)}, mask violations {env5.sim.mask_violation_total}")
    groups = p4.model.net.module_groups()
    print("[4] params: " + ", ".join(f"{k} {sum(p.numel() for m in ms for p in m.parameters()):,}" for k, ms in groups.items()))


if __name__ == "__main__":
    register_all()
    check_combos()
    print("[2] replay logits == rollout logits (decided heads)")
    check_n_invariance()
    print("smoke_role_policy: all ok")
