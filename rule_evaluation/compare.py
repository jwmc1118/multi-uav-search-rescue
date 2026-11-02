"""규칙 정책 vs 학습 모델 비교 로직 (main_rule.py) — 같은 seed 몬테카를로.

    run_parallel(cfg, seeds, workers, run)   seed 단위 병렬 실행 (eval_common/parallel.py, 경량 체크포인트 정책) -> per / ts 행,
                               실제 env_config, 강제된 키, 체크포인트 경로. 같은 seed 의 초기 상태 지문이 다르면 AssertionError.
                               run = both (규칙 + 모델) / rule (규칙만, 체크포인트 없이) / learned (모델만)
    eval_env_config(files)     평가 env = files (기본 rule_evaluation/config 의 env_rule → role_rule → task_rule.yaml) 를 합친 env_config
    summarize(...)             지표별 정책 평균 ± 표준편차 · 95% bootstrap CI. 두 정책이 다 있으면 짝 차이 (모델 - 규칙) 평균 · CI,
                               승 / 무 / 패, Wilcoxon p (+ 도달 가능한 최소 p · n 부족 표시, unanimous, 효과 크기 d_z · CI)
    timeseries_stats(...)      step 별 평균 · CI (누적 지표는 에피소드가 끝난 뒤 마지막 값 유지, 역할 비율은 끝난 뒤 제외)
    convergence(...)           앞 k 개 에피소드의 구조율 짝 차이 평균 · CI (k = 1..n, 두 정책이 다 있을 때만)
평가 env (policy_mode 포함) 는 --run 과 무관하게 항상 eval_env_config. 체크포인트는 가중치와 관측 shape 키 (SHAPE_KEYS +
cluster.k_max, eval_common/checkpoint_policy.py runtime_env_config) 에만 쓰고, 그 키가 다르면 체크포인트 값으로 고정 (경고).
"""
import os

import numpy as np

from eval_common.stats import bootstrap_ci, mean_std, paired
from training.config_loader import RULE_FILES, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = ("rule", "model")
RUN_POLICIES = {"both": ("rule", "model"), "rule": ("rule",), "learned": ("model",)}   # main_rule --run
CUMULATIVE = ("cum_rescue_rate", "cum_discover_rate", "explored")
ROLES = ("role_explore", "role_rescue", "role_return")


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def eval_env_config(files=RULE_FILES):
    """평가 env_config = files (env → role → task 순) 를 training/config_loader.load_config 로 합친 값."""
    return load_config(files)["env_config"]


def present(per):
    """per_episode 에 있는 정책 (POLICIES 순)."""
    have = {r["policy"] for r in per}
    return tuple(p for p in POLICIES if p in have)


def run_parallel(cfg, seeds, workers, run="both", env_config=None, free_keys=(), timeseries=True, label="rule"):
    """반환 (per_episode, timeseries, 실제 env_config, 강제된 키 목록, 체크포인트 경로 (run rule 이면 None)).
    결과는 seed 순 (workers 수와 무관)."""
    from eval_common.checkpoint_policy import load_train_config, resolve_checkpoint, runtime_env_config
    from eval_common.parallel import Task, run_tasks
    policies = RUN_POLICIES[run]
    ckpt = resolve_checkpoint(_abs(cfg["checkpoint"])) if "model" in policies else None   # 규칙만: 체크포인트 없이
    train_env = load_train_config(ckpt).get("env_config", {}) if ckpt else {}
    env = env_config if env_config is not None else eval_env_config()
    env = dict(env, events=dict(env.get("events") or {}, log_dir=None))
    used, warnings = runtime_env_config(train_env, env, free_keys) if ckpt else (env, [])
    res = run_tasks(ckpt, cfg.get("mode", "greedy"),
                    [Task(None, env, tuple(free_keys), s, timeseries, policies) for s in seeds], workers, label)
    per = [r for x in res for r in x.per]
    per.sort(key=lambda r: (POLICIES.index(r["policy"]), r["seed"]))
    ts = [r for x in res for r in x.ts]
    ts.sort(key=lambda r: (POLICIES.index(r["policy"]), r["seed"], r["step"]))
    assert all(x.env_used == res[0].env_used for x in res) and res[0].env_used == used
    return per, ts, used, warnings, ckpt


def summarize(per, metrics, lower_is_better, boot):
    """지표별 요약 행: 정책별 mean / std / ci_lo / ci_hi. 두 정책이 다 있으면 짝 차이 (model - rule) diff / ci,
    win / tie / loss (모델 기준), p 등."""
    pols = present(per)
    by = {p: {r["seed"]: r for r in per if r["policy"] == p} for p in pols}
    seeds = sorted(set.intersection(*[set(by[p]) for p in pols]))
    rows = []
    for m in metrics:
        row = {"metric": m, "higher_is_better": m not in lower_is_better, "n": len(seeds)}
        for p in pols:
            x = np.array([by[p][s][m] for s in seeds], float)
            mu, sd = mean_std(x)
            lo, hi = bootstrap_ci(x, boot["n"], boot["alpha"], boot["seed"])
            row.update({f"{p}_mean": mu, f"{p}_std": sd, f"{p}_ci_lo": lo, f"{p}_ci_hi": hi})
        if len(pols) < 2:
            rows.append(row)
            continue
        pr = paired([by["rule"][s][m] for s in seeds], [by["model"][s][m] for s in seeds], boot["n"], boot["alpha"],
                    boot["seed"], higher_better=m not in lower_is_better)
        row.update({"diff_mean": pr["diff_mean"], "diff_ci_lo": pr["diff_lo"], "diff_ci_hi": pr["diff_hi"],
                    "model_win": pr["win"], "tie": pr["tie"], "model_loss": pr["loss"], "p_wilcoxon": pr["p_wilcoxon"],
                    "n_nonzero": pr["n_eff"], "p_min": pr["p_min"], "p_note": pr["p_note"], "unanimous": pr["unanimous"],
                    "d_z": pr["d_z"], "d_z_ci_lo": pr["d_z_lo"], "d_z_ci_hi": pr["d_z_hi"]})
        rows.append(row)
    return rows


def timeseries_stats(ts, boot, max_step=None):
    """step 별 정책 평균 · CI. 반환 [dict(policy, step, <지표>_mean / _lo / _hi)]."""
    out = []
    T = max_step or max(r["step"] for r in ts)                     # 두 정책 공통 시간 축
    for p in present(ts):
        rows = [r for r in ts if r["policy"] == p]
        seeds = sorted({r["seed"] for r in rows})
        keys = CUMULATIVE + ROLES
        arr = {k: np.full((len(seeds), T), np.nan) for k in keys}
        for si, s in enumerate(seeds):
            ep = sorted((r for r in rows if r["seed"] == s), key=lambda r: r["step"])
            for r in ep:
                if 1 <= r["step"] <= T:
                    for k in keys:
                        arr[k][si, r["step"] - 1] = r[k]
            last = ep[-1]
            for k in CUMULATIVE:                                 # 끝난 뒤에도 마지막 값 유지
                arr[k][si, last["step"]:] = last[k]
        for t in range(T):
            row = {"policy": p, "step": t + 1}
            for k in keys:
                col = arr[k][:, t]
                col = col[np.isfinite(col)]
                if len(col) == 0:
                    continue
                lo, hi = bootstrap_ci(col, boot["n_timeseries"], boot["alpha"], boot["seed"])
                row.update({f"{k}_mean": float(col.mean()), f"{k}_lo": lo, f"{k}_hi": hi, f"{k}_n": int(len(col))})
            out.append(row)
    return out


def convergence(per, boot, metric="rescue_rate"):
    """앞 k 개 seed (seed 순서) 의 짝 차이 (model - rule) 평균 · CI, k = 1..n. 한 정책만 있으면 []."""
    if len(present(per)) < 2:
        return []
    by = {p: {r["seed"]: r[metric] for r in per if r["policy"] == p} for p in POLICIES}
    seeds = sorted(set(by["rule"]) & set(by["model"]))
    d = np.array([by["model"][s] - by["rule"][s] for s in seeds], float)
    out = []
    for k in range(1, len(d) + 1):
        lo, hi = bootstrap_ci(d[:k], boot["n_convergence"], boot["alpha"], boot["seed"])
        out.append({"episodes": k, "diff_mean": float(d[:k].mean()), "ci_lo": lo, "ci_hi": hi})
    return out
