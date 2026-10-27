"""학습 중 주기적 결정적 평가 — 고정 seed (run_rules 와 같은 seed 0..n-1), 세 head 모두 argmax 행동.

training/callbacks.py RoleCallbacks.on_train_result 가 eval.every iteration 마다 driver (local worker) 의 policy 로 부른다.
평가 전용 env 를 따로 만들어 쓰므로 학습 rollout 워커 · 그 env 에는 영향이 없다 (모델 greedy 플래그는 끝나면 되돌림).
seed 설정은 rule_evaluation/run_rules.py run_episode 와 같다: sim.random_state = RandomState(seed), sim.reset().
"""
import copy
import time

import numpy as np

ACT_KEYS = ("role", "rescue", "explore")


def evaluate_greedy(policy, env_config, n_seeds=15):
    """반환 {eval/rescue_rate, eval/expired_rate, eval/episode_len, eval/explored_rate (평균), ..._std, eval/time_s}."""
    from ray.rllib.models import ModelCatalog
    from env.role.gym_env import RoleEnv
    t0 = time.perf_counter()
    cfg = copy.deepcopy(env_config)
    cfg["events"] = dict(cfg.get("events") or {}, log_dir=None)       # 평가는 에피소드 로그를 쓰지 않는다
    env = RoleEnv(cfg)
    prep = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    model = policy.model
    greedy_prev = model.greedy
    model.greedy = True
    rows = []
    try:
        for seed in range(n_seeds):
            sim = env.sim
            sim.random_state = np.random.RandomState(seed)
            sim.episode_tag = seed
            obs = env.reset()
            done = False
            while not done:
                actions, _, _ = policy.compute_actions(prep.transform(obs)[None], explore=False)
                obs, _, done, info = env.step({k: [int(np.asarray(a)[0]) for a in actions[k]] for k in ACT_KEYS})
            n_t = max(1, len(sim.targets))
            rows.append({"rescue_rate": info["rescued_total"] / n_t, "expired_rate": info["expired_total"] / n_t,
                         "episode_len": sim.time, "explored_rate": info["explored"]})
    finally:
        model.greedy = greedy_prev
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], float)
        out[f"eval/{k}"], out[f"eval/{k}_std"] = float(v.mean()), float(v.std())
    out["eval/time_s"] = time.perf_counter() - t0
    return out
