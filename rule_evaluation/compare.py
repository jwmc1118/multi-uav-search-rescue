"""규칙 정책 vs 학습 모델 비교 로직 (main_rule.py) — 같은 seed 몬테카를로.

    run_parallel(cfg, seeds, workers)   seed 단위 병렬 실행 (eval_common/parallel.py, 경량 체크포인트 정책) -> per / ts 행,
                               실제 env_config, 강제된 키, 체크포인트 경로. 같은 seed 의 초기 상태 지문이 다르면 AssertionError
    build_policies(cfg)        (순차 · ray 경로) 체크포인트 env_config -> yaml env.files -> env.overrides 로 병합한 설정으로 CheckpointPolicy
                               (env_priority=False: SHAPE_KEYS 는 체크포인트 값) 와 RulePolicy, 둘이 같이 쓸 RoleSimulation
    run(sim, policies, seeds)  정책마다 seed 목록 실행 -> per_episode / timeseries 행. 같은 seed 의 초기 상태 지문이 다르면 AssertionError
    summarize(...)             지표별 정책 평균 ± 표준편차 · 95% bootstrap CI, 짝 차이 (모델 - 규칙) 평균 · CI, 승 / 무 / 패, Wilcoxon p
                               (+ 도달 가능한 최소 p · n 부족 표시, 전부 같은 쪽 (unanimous), 효과 크기 d_z · CI)
    timeseries_stats(...)      step 별 평균 · CI (누적 지표는 에피소드가 끝난 뒤 마지막 값 유지, 역할 비율은 끝난 뒤 제외)
    convergence(...)           앞 k 개 에피소드의 구조율 짝 차이 평균 · CI (k = 1..n)
"""
import copy
import os

import numpy as np
import yaml

from eval_common.episode import RulePolicy, run_episode_detailed
from eval_common.stats import bootstrap_ci, mean_std, paired
from training.config_loader import merge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = ("rule", "model")
CUMULATIVE = ("cum_rescue_rate", "cum_discover_rate", "explored")
ROLES = ("role_explore", "role_rescue", "role_return")


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def merged_env_config(train_env, env_cfg):
    """체크포인트 env_config -> env.files (각 파일의 env_config) -> env.overrides (dict 는 key 단위 깊은 병합)."""
    out = copy.deepcopy(train_env)
    for f in env_cfg.get("files") or []:
        with open(_abs(f)) as fh:
            out = merge(out, (yaml.safe_load(fh) or {}).get("env_config") or {})
    return merge(out, env_cfg.get("overrides") or {})


def build_policies(cfg, env_config=None, quiet=False, free_keys=()):
    """반환 (sim, {'rule': RulePolicy, 'model': CheckpointPolicy}, checkpoint 경로, 실제 env_config, 경고 목록).
    free_keys: 체크포인트 값으로 고정하지 않을 SHAPE_KEYS (main_scale)."""
    from env.role.simulation import RoleSimulation
    from eval_common.checkpoint_policy import CheckpointPolicy, load_train_config, resolve_checkpoint
    ckpt = resolve_checkpoint(_abs(cfg["checkpoint"]))
    train_env = load_train_config(ckpt).get("env_config", {})
    env = env_config if env_config is not None else merged_env_config(train_env, cfg.get("env") or {})
    env = dict(env, events=dict(env.get("events") or {}, log_dir=None))
    model = CheckpointPolicy(ckpt, env_config=env, mode=cfg.get("mode", "greedy"), env_priority=False,
                             free_keys=free_keys)
    model.name = "model"
    sim = RoleSimulation(model.env_config)
    return sim, {"rule": RulePolicy(), "model": model}, ckpt, model.env_config, list(model.warnings)


def run_parallel(cfg, seeds, workers, env_config=None, free_keys=(), timeseries=True, label="rule"):
    """반환 (per_episode, timeseries, 실제 env_config, 강제된 키 목록, 체크포인트 경로). 결과는 seed 순 (workers 수와 무관)."""
    from eval_common.checkpoint_policy import load_train_config, resolve_checkpoint, runtime_env_config
    from eval_common.parallel import Task, run_tasks
    ckpt = resolve_checkpoint(_abs(cfg["checkpoint"]))
    train_env = load_train_config(ckpt).get("env_config", {})
    env = env_config if env_config is not None else merged_env_config(train_env, cfg.get("env") or {})
    env = dict(env, events=dict(env.get("events") or {}, log_dir=None))
    used, warnings = runtime_env_config(train_env, env, free_keys)
    res = run_tasks(ckpt, cfg.get("mode", "greedy"), [Task(None, env, tuple(free_keys), s, timeseries) for s in seeds],
                    workers, label)
    per = [r for x in res for r in x.per]
    per.sort(key=lambda r: (POLICIES.index(r["policy"]), r["seed"]))
    ts = [r for x in res for r in x.ts]
    ts.sort(key=lambda r: (POLICIES.index(r["policy"]), r["seed"], r["step"]))
    assert all(x.env_used == res[0].env_used for x in res) and res[0].env_used == used
    return per, ts, used, warnings, ckpt


def run(sim, policies, seeds, label="", timeseries=True):
    """반환 (per_episode [dict], timeseries [dict]). 정책마다 같은 seed 순서로, seed 별 초기 상태 지문 일치 확인."""
    per, ts, sigs = [], [], {}
    for name, pol in policies.items():
        for k, s in enumerate(seeds):
            m, rows, sig = run_episode_detailed(sim, pol, s, timeseries=timeseries)
            if s in sigs:
                assert sigs[s] == sig, f"seed {s}: 정책 {name} 의 초기 상태가 다른 정책과 다름 (맵 / 조난자 / 골든타임)"
            sigs[s] = sig
            per.append(dict(policy=name, seed=s, **m))
            ts += [dict(policy=name, seed=s, **r) for r in rows]
            print(f"\r  {label}[{name}] {k + 1}/{len(seeds)}", end="", flush=True)
        print()
    return per, ts


def summarize(per, metrics, lower_is_better, boot):
    """지표별 요약 행: 정책별 mean / std / ci_lo / ci_hi, 짝 차이 (model - rule) diff / ci, win / tie / loss (모델 기준), p."""
    by = {p: {r["seed"]: r for r in per if r["policy"] == p} for p in POLICIES}
    seeds = sorted(set(by["rule"]) & set(by["model"]))
    rows = []
    for m in metrics:
        row = {"metric": m, "higher_is_better": m not in lower_is_better, "n": len(seeds)}
        for p in POLICIES:
            x = np.array([by[p][s][m] for s in seeds], float)
            mu, sd = mean_std(x)
            lo, hi = bootstrap_ci(x, boot["n"], boot["alpha"], boot["seed"])
            row.update({f"{p}_mean": mu, f"{p}_std": sd, f"{p}_ci_lo": lo, f"{p}_ci_hi": hi})
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
    for p in POLICIES:
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
    """앞 k 개 seed (seed 순서) 의 짝 차이 (model - rule) 평균 · CI, k = 1..n."""
    by = {p: {r["seed"]: r[metric] for r in per if r["policy"] == p} for p in POLICIES}
    seeds = sorted(set(by["rule"]) & set(by["model"]))
    d = np.array([by["model"][s] - by["rule"][s] for s in seeds], float)
    out = []
    for k in range(1, len(d) + 1):
        lo, hi = bootstrap_ci(d[:k], boot["n_convergence"], boot["alpha"], boot["seed"])
        out.append({"episodes": k, "diff_mean": float(d[:k].mean()), "ci_lo": lo, "ci_hi": hi})
    return out
