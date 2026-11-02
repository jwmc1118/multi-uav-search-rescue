"""학습 중 주기적 결정적 평가 — 고정 seed (run_rules 와 같은 seed 0..n-1), 두 head (role / task (하위 할당)) 모두 argmax 행동.

training/callbacks.py RoleCallbacks.on_train_result 가 eval.every iteration 마다 driver (local worker) 의 policy 로 부른다.
평가 전용 env 를 따로 만들어 쓰므로 학습 rollout 워커 · 그 env 에는 영향이 없다 (모델 greedy 플래그는 끝나면 되돌림).
seed 설정은 rule_evaluation/run_rules.py run_episode 와 같다: sim.random_state = RandomState(seed), sim.reset().
같은 seed 의 규칙 정책 기록 (training/behavior.py rule_reference, logdir/rule_eval.json 캐시) 과 seed 별 짝 비교를 하고,
에피소드 기록 (곡선 · 귀환 시작 용량 · 미발견 생존자) 은 logdir/eval_latest.json 에 쓴다 (fig_behavior · fig_undiscovered).
"""
import copy
import json
import os
import time

import numpy as np

ACT_KEYS = ("role", "task")


def evaluate_greedy(policy, env_config, n_seeds=20, logdir=None, iteration=None):
    """반환 {eval/rescue_rate, eval/expired_rate, eval/episode_len, eval/explored_rate (평균), ..._std, eval/n_seeds,
    eval/diff_mean · eval/diff_lo · eval/diff_hi (learned - rule seed 별 구조율 차이 평균 · bootstrap 95 % CI, logdir 있을 때),
    eval/time_s}."""
    from training.behavior import EVAL_LATEST, episode_record, paired_diff, rule_reference
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
    recs = []
    try:
        for seed in range(n_seeds):
            sim = env.sim
            sim.random_state = np.random.RandomState(seed)
            sim.episode_tag = seed
            obs = env.reset()
            done, infos = False, []
            while not done:
                actions, _, _ = policy.compute_actions(prep.transform(obs)[None], explore=False)
                obs, _, done, info = env.step({k: [int(np.asarray(a)[0]) for a in actions[k]] for k in ACT_KEYS})
                infos.append(info)
            recs.append(episode_record(sim, infos))
    finally:
        model.greedy = greedy_prev
    out = {}
    for k in ("rescue_rate", "expired_rate", "episode_len", "explored_rate"):
        v = np.array([r[k] for r in recs], float)
        out[f"eval/{k}"], out[f"eval/{k}_std"] = float(v.mean()), float(v.std())
    out["eval/n_seeds"] = n_seeds
    if logdir:
        rule = rule_reference(env_config, n_seeds, logdir)
        m, lo, hi = paired_diff([r["rescue_rate"] for r in recs], [r["rescue_rate"] for r in rule])
        out.update({"eval/diff_mean": m, "eval/diff_lo": lo, "eval/diff_hi": hi})
        with open(os.path.join(logdir, EVAL_LATEST), "w") as f:
            json.dump({"iteration": iteration, "n_seeds": n_seeds, "episodes": recs}, f)
    out["eval/time_s"] = time.perf_counter() - t0
    return out
