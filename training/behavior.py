"""고정 seed 평가의 에피소드 기록 — 고정 seed 평가 (training/fixed_seed_eval.py, 학습 중인 모델만) 와 학습 그래프
(training/reward_plot.py) 공용.

    rec = episode_record(sim, step_infos)            # 한 에피소드 -> 구조율 · 진행률 곡선 · 귀환 시작 용량 · 미발견 생존자 · 지도
    BASELINES                                         # env.yaml eval.baselines 키 (조합 이름) -> (표시 이름, 색, 선 스타일) — [eval] 로그 · 그림 공용

진행률 곡선 (10 % 구간 평균) 키: role_explore / role_rescue / role_return (UAV 역할 비율), found (누적 발견율 / N_T),
rescued (누적 구조율 / N_T), omega (UAV 평균 그룹 연결도 ω^g). seed 설정은 rule_evaluation/run_rules.py 와 같다
(sim.random_state = RandomState(seed), sim.reset()).
"""
import numpy as np

N_BINS = 10
CURVE_KEYS = ("role_explore", "role_rescue", "role_return", "found", "rescued", "omega")
EVAL_LATEST = "eval_latest.json"
BASELINES = {"rule_rule_rule": ("rule_rule_rule", "0.35", "--"),          # 규칙 정책 (role · rescue · explore 모두 rule)
             "learn_rule_rule": ("learn_rule_rule", "#8c564b", "-."),       # 학습 role + 규칙 rescue / explore
             "learn_learn_learn": ("learn_learn_learn", "#e377c2", ":")}    # 세 네트워크 모두 학습


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
