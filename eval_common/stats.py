"""평가 통계 — 평균 · 표준편차, 95% bootstrap CI, 짝 차이, 승 / 무 / 패, Wilcoxon signed-rank (scipy 있으면).

bootstrap 은 seed 고정 (재현 가능), 평균의 percentile CI.
"""
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


def paired(a, b, n_boot=10000, alpha=0.05, seed=0, higher_better=True, tol=1e-9):
    """짝 비교 (같은 seed): d = b - a. 반환 {diff_mean, diff_lo, diff_hi, win, tie, loss (b 기준 비율), p_wilcoxon}."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = b - a
    lo, hi = bootstrap_ci(d, n_boot, alpha, seed)
    s = np.sign(d) * (1 if higher_better else -1)
    s[np.abs(d) <= tol] = 0
    n = max(1, len(d))
    p = float("nan")
    if HAS_SCIPY and len(d) > 0 and np.any(np.abs(d) > tol):
        try:
            p = float(_wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            p = float("nan")
    return {"diff_mean": float(d.mean()) if len(d) else float("nan"), "diff_lo": lo, "diff_hi": hi,
            "win": float((s > 0).sum() / n), "tie": float((s == 0).sum() / n), "loss": float((s < 0).sum() / n),
            "p_wilcoxon": p, "n": int(len(d))}
