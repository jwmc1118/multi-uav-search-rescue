"""확장성 비교 로직 (main_scale.py) — N 마다 크기 조정한 환경에서 규칙 정책 vs 학습 체크포인트, 같은 seed.

    base_env(cfg)                   평가 env (scale_evaluation/config 의 env_scale → role_scale → task_scale.yaml, 크기 조정 전 기준),
                                    N_ref (scale.yaml n_ref, null = 평가 env UAV 수), 체크포인트 경로
    run_all(cfg, envs, seeds, workers)   모든 (N, seed) 를 한 워커 풀에서 병렬 실행 (eval_common/parallel.py — 워커가 체크포인트
                                    가중치를 한 번만 읽고 N 이 바뀌면 모델만 새로 만듦, FREE_KEYS 는 고정 안 함) -> per_episode 행
                                    (n_agents 열 포함, N · 정책 · seed 순), N 별 실제 env_config, 강제된 키. 초기 상태 지문 일치 확인
    summarize_n(per, ...)           N 별 rule_evaluation.compare.summarize 행 + 크기 정보 (scale.csv 의 긴 형식)
    infer_summary(per, boot)        N 별 모델 추론 시간 (ms / 결정) 평균 · 표준편차 · CI
"""
import numpy as np

from eval_common.stats import bootstrap_ci, mean_std
from rule_evaluation import compare as rc
from scale_evaluation.scaling import FREE_KEYS, describe
from training.config_loader import SCALE_FILES


def base_env(cfg):
    from eval_common.checkpoint_policy import resolve_checkpoint
    ckpt = resolve_checkpoint(rc._abs(cfg["checkpoint"]))
    env = rc.eval_env_config(SCALE_FILES)
    n_ref = int(cfg.get("n_ref") or len(env.get("agents", [])))
    return env, n_ref, ckpt


def run_all(cfg, envs, seeds, workers):
    from eval_common.checkpoint_policy import load_train_config, resolve_checkpoint, runtime_env_config
    from eval_common.parallel import Task, run_tasks
    ckpt = resolve_checkpoint(rc._abs(cfg["checkpoint"]))
    train_env = load_train_config(ckpt).get("env_config", {})
    ns = sorted(envs)
    envs = {n: dict(e, events=dict(e.get("events") or {}, log_dir=None)) for n, e in envs.items()}
    tasks = [Task(n, envs[n], FREE_KEYS, s, False) for n in sorted(ns, reverse=True) for s in seeds]   # 큰 N 먼저 (부하 균형)
    res = run_tasks(ckpt, cfg.get("mode", "greedy"), tasks, workers, "scale")
    used, warns = {}, {}
    for n in ns:
        used[n], warns[n] = runtime_env_config(train_env, envs[n], FREE_KEYS)
        got = [x.env_used for x in res if x.key == n]
        assert all(g == used[n] for g in got), f"N={n}: 워커 실행 설정이 다름"
        for k in FREE_KEYS:                                    # 크기 조정 값이 실제 실행 설정에 들어갔는지
            assert used[n].get(k) == envs[n].get(k), f"N={n}: {k} {envs[n].get(k)} -> {used[n].get(k)}"
        assert len(used[n]["agents"]) == n
    per = [r for x in res for r in x.per]
    per.sort(key=lambda r: (r["n_agents"], rc.POLICIES.index(r["policy"]), r["seed"]))
    return per, used, warns


def summarize_n(per, ns, envs, metrics, lower_is_better, boot):
    out = []
    for n in ns:
        rows = rc.summarize([r for r in per if r["n_agents"] == n], metrics, lower_is_better, boot)
        info = describe(envs[n])
        out += [dict(n_agents=n, **info, **r) for r in rows]
    return out


def infer_summary(per, ns, boot):
    out = []
    for n in ns:
        x = np.array([r["infer_ms_per_decision"] for r in per if r["n_agents"] == n and r["policy"] == "model"], float)
        mu, sd = mean_std(x)
        lo, hi = bootstrap_ci(x, boot["n"], boot["alpha"], boot["seed"])
        out.append({"n_agents": n, "infer_ms_mean": mu, "infer_ms_std": sd, "infer_ms_ci_lo": lo, "infer_ms_ci_hi": hi,
                    "decisions_mean": float(np.mean([r["decisions"] for r in per
                                                     if r["n_agents"] == n and r["policy"] == "model"]))})
    return out
