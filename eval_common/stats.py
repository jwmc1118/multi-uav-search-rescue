"""평가 통계 — 평균 · 표준편차, 95% bootstrap CI, 짝 차이, 승 / 무 / 패, Wilcoxon signed-rank (scipy 있으면).

bootstrap 은 seed 고정 (재현 가능), 평균의 percentile CI.
cohen_dz(d)    효과 크기 Cohen's d_z = 평균(d) / 표준편차(d) (ddof 1) + bootstrap percentile CI (같은 resample 방식)
diff_stats(d)  main_rule 여러 정책 짝 비교 (쌍 A - B 의 seed 별 차이 d): 평균 · bootstrap CI, win / tie / loss 개수, Wilcoxon p
               (차이가 전부 0 이면 1), d_z (표준편차 0 이면 nan). holm(p 목록) = Holm-Bonferroni 보정 p (nan 은 그대로, 개수에서 뺌)
"""
import warnings

import numpy as np

try:
    from scipy.stats import wilcoxon as _wilcoxon
except ImportError:                                   # scipy 없으면 p-value 생략 (출력에 명시)
    _wilcoxon = None

HAS_SCIPY = _wilcoxon is not None


def mean_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0) if len(x) else (float("nan"), float("nan"))


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    """평균의 (1 - alpha) percentile bootstrap CI. 값이 1 개 이하면 (평균, 평균)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), size=(int(n_boot), len(x)))].mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def cohen_dz(d, n_boot=10000, alpha=0.05, seed=0):
    """짝 차이 d 의 Cohen's d_z = mean / std (ddof 1) 와 bootstrap percentile CI. 표준편차 0 이면 nan."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan"), float("nan")
    dz = float(d.mean() / d.std(ddof=1))
    rng = np.random.RandomState(seed)
    b = d[rng.randint(0, len(d), size=(int(n_boot), len(d)))]
    sd = b.std(1, ddof=1)
    ok = sd > 0
    z = b.mean(1)[ok] / sd[ok]
    if len(z) == 0:
        return dz, float("nan"), float("nan")
    return dz, float(np.percentile(z, 100 * alpha / 2)), float(np.percentile(z, 100 * (1 - alpha / 2)))


def diff_stats(d, n_boot=10000, alpha=0.05, seed=0, higher_better=True, tol=1e-9):
    """짝 차이 d (A - B, seed 별). 반환 {n, mean, ci_lo, ci_hi, win, tie, loss (A 기준 개수 — higher_better 면 d > 0 이 win),
    p_wilcoxon (전부 0 이면 1.0, scipy 없으면 nan), d_z (표준편차 0 이면 nan)}."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    lo, hi = bootstrap_ci(d, n_boot, alpha, seed)
    s = np.sign(d) * (1 if higher_better else -1)
    s[np.abs(d) <= tol] = 0
    if not np.any(np.abs(d) > tol):
        p = 1.0
    elif HAS_SCIPY:
        with warnings.catch_warnings():                       # 0 차이 · 작은 n 근사 경고 (지표 · 쌍마다 반복되어 출력만 어지럽힘)
            warnings.simplefilter("ignore", UserWarning)
            p = float(_wilcoxon(d, zero_method="wilcox").pvalue)
    else:
        p = float("nan")
    return {"n": int(len(d)), "mean": float(d.mean()) if len(d) else float("nan"), "ci_lo": lo, "ci_hi": hi,
            "win": int((s > 0).sum()), "tie": int((s == 0).sum()), "loss": int((s < 0).sum()), "p_wilcoxon": p,
            "d_z": cohen_dz(d, 1)[0]}


def holm(ps):
    """Holm-Bonferroni 보정 p (입력 순서 그대로). nan 은 보정에서 빼고 nan 으로 둔다."""
    ps = np.asarray(ps, float)
    out = np.full(len(ps), np.nan)
    idx = [i for i in np.argsort(ps, kind="stable") if np.isfinite(ps[i])]
    m, run = len(idx), 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return [float(x) for x in out]
