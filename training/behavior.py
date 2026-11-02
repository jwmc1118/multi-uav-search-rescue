"""고정 seed 평가의 에피소드 기록 — 고정 seed 평가 (training/fixed_seed_eval.py) 와 학습 그래프 (training/reward_plot.py) 공용.

    rec = episode_record(sim, step_infos)            # 한 에피소드 -> 구조율 · 진행률 곡선 · 귀환 시작 용량 · 미발견 생존자 · 지도
    rule_reference(env_config, n_seeds, logdir)       # 같은 고정 seed 의 규칙 정책 기록 (logdir/rule_eval.json 캐시)
    paired_diff(learned, rule)                        # seed 별 구조율 차이 평균 + bootstrap 95 % 신뢰구간

진행률 곡선 (10 % 구간 평균) 키: role_explore / role_rescue / role_return (UAV 역할 비율), found (누적 발견율 / N_T),
rescued (누적 구조율 / N_T), omega (UAV 평균 그룹 연결도 ω^g). seed 설정은 rule_evaluation/run_rules.py 와 같다
(sim.random_state = RandomState(seed), sim.reset()).
"""
import hashlib
import json
import os

import numpy as np

N_BINS = 10
CURVE_KEYS = ("role_explore", "role_rescue", "role_return", "found", "rescued", "omega")
RULE_CACHE = "rule_eval.json"
EVAL_LATEST = "eval_latest.json"


def progress_bins(values, n_bins=N_BINS):
    """step 순 값 -> 진행률 n_bins 구간 평균 (step 이 구간보다 적으면 가장 가까운 step 값)."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return [float("nan")] * n_bins
    edges = np.linspace(0, len(v), n_bins + 1).astype(int)
    return [float(v[edges[b]:max(edges[b + 1], edges[b] + 1)].mean()) for b in range(n_bins)]


def episode_curves(infos, n_targets):
    """infos = 한 에피소드의 step 별 info (env/role/simulation.py _step_role) -> {CURVE_KEYS: 10 구간 평균}."""
    n_t = max(1, n_targets)
    roles = np.array([np.bincount(np.asarray(f["role"], int), minlength=3) / max(1, len(f["role"])) for f in infos])
    out = {f"role_{name}": progress_bins(roles[:, j]) for j, name in enumerate(("explore", "rescue", "return"))}
    out["found"] = progress_bins([f["discovered_total"] / n_t for f in infos])
    out["rescued"] = progress_bins([f["rescued_total"] / n_t for f in infos])
    out["omega"] = progress_bins([f.get("omega_mean", float("nan")) for f in infos])
    return out


def mean_curves(curves):
    """seed 별 curves 평균."""
    return {k: [float(np.nanmean([c[k][b] for c in curves])) for b in range(N_BINS)] for k in CURVE_KEYS}


def episode_record(sim, infos):
    """끝난 에피소드 (sim, step 별 info) -> 평가 기록 dict (JSON 저장 가능)."""
    info = infos[-1]
    n_t = max(1, len(sim.targets))
    rw = ((info.get("event_stats") or {}).get("rewards") or {})
    return {"rescue_rate": info["rescued_total"] / n_t, "expired_rate": info["expired_total"] / n_t,
            "episode_len": sim.time, "explored_rate": info["explored"],
            "curves": episode_curves(infos, len(sim.targets)),
            "return_start_cap_hist": list(rw.get("return_start_cap_hist") or []),
            "undiscovered": [[int(t.pos[0]), int(t.pos[1])] for t in sim.targets if not t.discovered],
            "n_targets": len(sim.targets),
            "obstacles": ["".join("1" if v else "0" for v in row) for row in np.asarray(sim.obstacles, bool)],
            "depot": [int(sim.depot[0]), int(sim.depot[1])]}


def _cache_key(env_config, n_seeds):
    return hashlib.md5(json.dumps([env_config, int(n_seeds)], sort_keys=True, default=str).encode()).hexdigest()


def rule_reference(env_config, n_seeds, logdir=None):
    """고정 seed 0 .. n_seeds - 1 의 규칙 정책 (policies/role_rules.py RuleDecider) 에피소드 기록 목록. logdir 이 있으면
    logdir/rule_eval.json 에 캐시 (env_config · seed 수가 같으면 다시 돌리지 않음)."""
    key = _cache_key(env_config, n_seeds)
    path = os.path.join(logdir, RULE_CACHE) if logdir else None
    if path and os.path.isfile(path):
        try:
            with open(path) as f:
                cached = json.load(f)
            if cached.get("key") == key:
                return cached["episodes"]
        except (OSError, ValueError):
            pass
    from env.role.simulation import RoleSimulation
    from policies.role_rules import RuleDecider
    cfg = dict(env_config, events=dict(env_config.get("events") or {}, log_dir=None))
    sim = RoleSimulation(cfg)
    recs = []
    for seed in range(int(n_seeds)):
        sim.random_state = np.random.RandomState(seed)
        sim.reset()
        done, infos = False, []
        while not done:
            done, info = sim.step(RuleDecider())
            infos.append(info)
        recs.append(episode_record(sim, infos))
    if path:
        os.makedirs(logdir, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"key": key, "n_seeds": int(n_seeds), "episodes": recs}, f)
    return recs


def paired_diff(learned, rule, n_boot=2000, seed=0):
    """같은 seed 짝 (learned_k - rule_k) 의 평균과 bootstrap (seed 재표집 n_boot 회) 95 % 신뢰구간 -> (mean, lo, hi)."""
    d = np.asarray(learned, float) - np.asarray(rule, float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    boot = d[rng.randint(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
