"""여러 정책 비교 로직 (main_rule.py) — 같은 seed 몬테카를로, 정책 N 개 + 짝 비교.

    resolve_policies(items)    [{label, path}] -> 정책 정보 (path "rule" = 규칙 정책, 그 외 = eval_common/model_loader.py 로 조합 ·
                               가중치를 읽어 로드 가능 여부까지 확인 — 실패하면 ValueError). label 중복 금지
    parse_pairs(pairs, labels) "all" (모든 쌍, 뒤 정책 - 앞 정책) | [[A, B], ...] | "A:B,C:D" -> [(A, B)] (A - B)
    run_env_config(policies)   실행 env_config (평가 yaml + 모델이 고정하는 관측 shape 키, FREE_KEYS 는 yaml 값) + 고정된 키 목록.
                               평가 env 의 max_targets 가 num_targets 상한보다 작으면 ValueError
    run_parallel(policies, seeds, workers, mode)   (정책 × seed) 작업을 한 워커 풀에서 병렬 실행 (eval_common/parallel.py,
                               경량 정책) -> per 행 (policy = label, 정책 · seed 순), step 별 시계열 행, 실제 env_config, 강제된 키.
                               같은 seed 의 초기 상태 지문 (맵 / depot / 조난자 / 골든타임) 이 정책마다 다르면 AssertionError
    run_env_summary(env)       실제 실행 env 의 크기 · 조건 (rows, cols, max_targets, max_episode_len, d_max, UAV 수, num_targets,
                               target_golden_time) — 콘솔 · meta.json
    eval_env_config(files)     평가 env = files (기본 rule_evaluation/config 의 env_rule → role_rule → task_rule.yaml) 를 합친 env_config
    summarize(per, labels, ...)    정책 · 지표별 평균, 표준편차, bootstrap 95% CI
    pairwise(per, pairs, ...)      쌍 · 지표별 seed 차이 (A - B) 평균 · bootstrap CI, win / tie / loss, Wilcoxon p, Holm p
                                   (같은 지표 안에서 쌍 수만큼 보정), Cohen's d_z (eval_common/stats.py diff_stats · holm)
    progress_bins(ts, n_bins)      (정책, seed) 별 에피소드 진행률 (step / 에피소드 길이) 구간 값: 누적 발견율 · 구조율 = 구간 끝 값,
                                   역할 비율 (EXPLORE / RESCUE / RETURN) = 구간 안 step 평균, 동시 추격 = 구간 안 합
    progress_stats(rows, ...)      정책 · 구간별 seed 평균 + bootstrap 95% CI (fig_compare_cumulative · _roles · _conflict_time)
평가 env 는 모든 정책에 같다 (구성요소 모드만 정책마다). 모델은 가중치 · 구성요소 모드와 관측 shape 키 (SHAPE_KEYS + cluster.k_max,
eval_common/checkpoint_policy.py runtime_env_config) 에만 쓰고, 그 키가 평가 env 와 다르면 모델 값으로 고정 — 단 FREE_KEYS
(rows / cols / max_targets / max_episode_len, 모델 파라미터와 무관) 는 평가 yaml 값을 그대로 쓴다. 모델끼리 고정 값이
다르면 같은 조건으로 비교할 수 없으므로 ValueError. 규칙 정책도 그 고정된 env 로 실행한다.
"""
import itertools
import os

import numpy as np

from eval_common.stats import bootstrap_ci, diff_stats, holm, mean_std
from training.config_loader import RULE_FILES, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULE = "rule"
RULE_LABEL = "rule_rule_rule"


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def eval_env_config(files=RULE_FILES):
    """평가 env_config = files (env → role → task 순) 를 training/config_loader.load_config 로 합친 값."""
    return load_config(files)["env_config"]


def resolve_policies(items):
    """items = [{label (없으면 조합 이름), path}] -> [{label, path (절대 | None = 규칙), combo, describe, model, iteration,
    checkpoint}]. 모델은 가중치까지 읽어 로드 가능한지 확인한다."""
    from eval_common.model_loader import load_model, rel
    out = []
    for it in items:
        path = str(it["path"])
        if path == RULE:
            info = {"path": None, "combo": RULE_LABEL, "describe": "role rule · rescue rule · explore rule", "model": RULE,
                    "iteration": None, "checkpoint": None}
        else:
            p = _abs(os.path.expanduser(path))
            if not os.path.exists(p):
                raise ValueError(f"모델 경로가 없음: {path}")
            try:
                spec = load_model(p)
            except Exception as e:
                raise ValueError(f"모델을 로드할 수 없음 ({path}): {str(e).splitlines()[0]}") from None
            info = {"path": p, "combo": spec.combo, "describe": spec.describe(), "model": rel(p), "iteration": spec.iteration,
                    "checkpoint": rel(spec.checkpoint)}
        out.append(dict(label=str(it.get("label") or info["combo"]), **info))
    labels = [p["label"] for p in out]
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        raise ValueError(f"정책 label 중복: {dup} — --labels (또는 compare.policies label) 로 구분하세요")
    return out


def parse_pairs(pairs, labels):
    """반환 [(A, B)] — 짝 차이 = A - B. "all" = 모든 쌍 (목록 뒤 정책 - 앞 정책)."""
    if pairs in (None, "all"):
        return [(labels[j], labels[i]) for i, j in itertools.combinations(range(len(labels)), 2)]
    if isinstance(pairs, str):
        pairs = [x.split(":") for x in pairs.split(",") if x.strip()]
    out = []
    for pr in pairs:
        if len(pr) != 2:
            raise ValueError(f"쌍 형식 오류: {pr} — [A, B] 또는 'A:B'")
        a, b = (str(x).strip() for x in pr)
        bad = [x for x in (a, b) if x not in labels]
        if bad or a == b:
            raise ValueError(f"쌍 {a}:{b} — 없는 label {bad}" if bad else f"쌍 {a}:{b} — 같은 정책끼리")
        out.append((a, b))
    return out


def run_env_config(policies, env_config=None):
    """모든 정책이 쓸 실행 env_config + 모델 값으로 고정된 키 목록. env_config None = 평가 yaml (eval_env_config).
    모델이 있으면 FREE_KEYS 를 뺀 관측 shape 키를 체크포인트 값으로 고정 (모델끼리 다르면 ValueError).
    max_targets < num_targets 상한이면 ValueError (조난자 슬롯이 조용히 잘리지 않게)."""
    from eval_common.checkpoint_policy import FREE_KEYS, load_train_config, runtime_env_config
    from eval_common.model_loader import find_checkpoint
    env = env_config if env_config is not None else eval_env_config()
    env = dict(env, events=dict(env.get("events") or {}, log_dir=None))
    nt = env.get("num_targets")
    nt_max = max(nt) if isinstance(nt, (list, tuple)) else nt
    if nt_max and int(nt_max) > int(env["max_targets"]):
        raise ValueError(f"max_targets {env['max_targets']} < num_targets 상한 {nt_max} (num_targets {nt}) — 조난자 슬롯이 "
                         f"잘리지 않게 평가 env yaml 의 max_targets 를 {nt_max} 이상으로")
    run_env, warnings = None, []
    for p in policies:
        if p["path"]:
            e, w = runtime_env_config(load_train_config(find_checkpoint(p["path"])).get("env_config", {}), env, FREE_KEYS)
            e = {k: v for k, v in e.items() if k != "components"}
            if run_env is not None and e != run_env:
                raise ValueError(f"{p['label']}: 관측 shape 키가 다른 모델과 달라 같은 env 로 비교할 수 없음 ({'; '.join(w)})")
            run_env = e
            warnings += [f"{p['label']}: {x}" for x in w]
    return (run_env if run_env is not None else env), warnings


def run_parallel(policies, seeds, workers, mode="greedy", env_config=None, label="rule"):
    """(정책 × seed) 를 한 풀에서 실행 (env = run_env_config). 반환 (per_episode [policy, seed, 지표...], 시계열 [policy, seed,
    step, 누적 발견 · 구조율, 역할 비율], 실제 env_config, 강제된 키 목록).
    결과는 정책 (목록 순) · seed 순 (workers 수와 무관)."""
    from eval_common.checkpoint_policy import FREE_KEYS
    from eval_common.parallel import Task, run_tasks
    run_env, warnings = run_env_config(policies, env_config)
    labels = [p["label"] for p in policies]
    tasks = [Task(k, run_env, FREE_KEYS, s, True, (lab,)) for s in seeds for k, lab in enumerate(labels)]
    res = run_tasks({p["label"]: p["path"] for p in policies}, mode, tasks, workers, label)
    for s in seeds:
        sig = {x.sig for x in res if x.seed == s}
        assert len(sig) == 1, f"seed {s}: 정책마다 초기 상태가 다름 (맵 / 조난자 / 골든타임)"
    assert all(x.env_used == res[0].env_used for x in res), "정책마다 실행 env 가 다름"
    per = [r for x in res for r in x.per]
    ts = [r for x in res for r in x.ts]
    return per, ts, res[0].env_used, warnings


def run_env_summary(env):
    """실제 실행 env 의 크기 · 조건 (yaml 값이 들어갔는지 확인용). d_max = 설정값 (null 이면 rows + cols)."""
    return {"rows": env["rows"], "cols": env["cols"], "max_targets": env["max_targets"],
            "max_episode_len": env["max_episode_len"], "d_max": env.get("d_max") or env["rows"] + env["cols"],
            "n_uav": len(env.get("agents") or []), "num_targets": env.get("num_targets"),
            "target_golden_time": env.get("target_golden_time")}


def _by(per, label):
    return {r["seed"]: r for r in per if r["policy"] == label}


def summarize(per, labels, metrics, boot):
    """[{policy, metric, n, mean, std, ci_lo, ci_hi}] (지표 순 → 정책 순)."""
    rows = []
    for m in metrics:
        for lab in labels:
            x = np.array([r[m] for r in _by(per, lab).values()], float)
            mu, sd = mean_std(x)
            lo, hi = bootstrap_ci(x, boot["n"], boot["alpha"], boot["seed"])
            rows.append({"policy": lab, "metric": m, "n": int(np.isfinite(x).sum()), "mean": mu, "std": sd, "ci_lo": lo,
                         "ci_hi": hi})
    return rows


def pairwise(per, pairs, metrics, lower_is_better, boot):
    """[{pair "A - B", a, b, metric, higher_is_better, n, mean, ci_lo, ci_hi, win, tie, loss, p_wilcoxon, p_holm, d_z}]
    (지표 순 → 쌍 순). Holm 보정은 같은 지표 안의 쌍끼리."""
    rows = []
    for m in metrics:
        hb = m not in lower_is_better
        block = []
        for a, b in pairs:
            ba, bb = _by(per, a), _by(per, b)
            seeds = sorted(set(ba) & set(bb))
            d = [ba[s][m] - bb[s][m] for s in seeds]
            st = diff_stats(d, boot["n"], boot["alpha"], boot["seed"], higher_better=hb)
            block.append({"pair": f"{a} - {b}", "a": a, "b": b, "metric": m, "higher_is_better": hb, **st})
        for r, ph in zip(block, holm([r["p_wilcoxon"] for r in block])):
            r["p_holm"] = ph
        rows += block
    return rows


CUMULATIVE = ("cum_discover_rate", "cum_rescue_rate")
ROLES = ("role_explore", "role_rescue", "role_return")
SUMS = ("conflict_concurrent",)                    # 구간 안 합 (step 별 발생 수)


def progress_bins(ts, n_bins=10):
    """[{policy, seed, bin, progress (%), cum_*, role_*, conflict_concurrent}] — 구간 k = 진행률 ((k-1)/n, k/n] (에피소드 길이 =
    마지막 step)."""
    eps = {}
    for r in ts:
        eps.setdefault((r["policy"], r["seed"]), []).append(r)
    out = []
    for (pol, seed), rows in eps.items():
        rows.sort(key=lambda r: r["step"])
        L = rows[-1]["step"]
        for k in range(1, n_bins + 1):
            inb = [r for r in rows if (k - 1) * L < r["step"] * n_bins <= k * L]
            last = inb[-1] if inb else max((r for r in rows if r["step"] * n_bins <= k * L), key=lambda r: r["step"],
                                           default=rows[0])
            row = {"policy": pol, "seed": seed, "bin": k, "progress": 100.0 * k / n_bins}
            row.update({c: last[c] for c in CUMULATIVE})
            row.update({c: float(np.mean([r[c] for r in inb])) if inb else float("nan") for c in ROLES})
            row.update({c: int(sum(r[c] for r in inb)) for c in SUMS})
            out.append(row)
    return out


def progress_stats(rows, labels, boot):
    """[{policy, bin, progress, <키>_mean / _lo / _hi}] (정책 순 → 구간 순), seed 평균 + bootstrap CI."""
    out = []
    for lab in labels:
        mine = [r for r in rows if r["policy"] == lab]
        for k in sorted({r["bin"] for r in mine}):
            b = [r for r in mine if r["bin"] == k]
            row = {"policy": lab, "bin": k, "progress": b[0]["progress"], "n": len(b)}
            for c in CUMULATIVE + ROLES + SUMS:
                x = np.array([r[c] for r in b], float)
                lo, hi = bootstrap_ci(x, boot["n"], boot["alpha"], boot["seed"])
                row.update({f"{c}_mean": float(np.nanmean(x)), f"{c}_lo": lo, f"{c}_hi": hi})
            out.append(row)
    return out
