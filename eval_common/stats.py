"""평가 통계 — 평균 · 표준편차, 95% bootstrap CI, 짝 차이, 승 / 무 / 패, Wilcoxon signed-rank (scipy 있으면).

bootstrap 은 seed 고정 (재현 가능), 평균의 percentile CI.
짝 비교 (paired) 추가 정보:
    p_min      Wilcoxon signed-rank 로 도달 가능한 최소 양측 p = 2 / 2^n_eff (n_eff = 0 아닌 차이 수; n = 5 -> 0.0625).
               n_eff < 6 이면 p_note = "n too small (min p = ...)" — 아무리 일관된 차이여도 p < 0.05 가 나올 수 없다
    unanimous  모든 짝 차이가 같은 쪽 (승 / 무 / 패 기준, 무승부 없음): "model" (전부 모델 우세) / "rule" (전부 규칙 우세) / ""
    d_z        효과 크기 Cohen's d_z = 평균(d) / 표준편차(d) (ddof 1) + bootstrap percentile CI (같은 resample 방식)
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
    n_eff = int((np.abs(d) > tol).sum())
    p_min = min(1.0, 2.0 * 0.5 ** n_eff) if n_eff else float("nan")
    unanimous = "model" if len(d) and (s > 0).all() else ("rule" if len(d) and (s < 0).all() else "")
    dz, dz_lo, dz_hi = cohen_dz(d, n_boot, alpha, seed)
    return {"diff_mean": float(d.mean()) if len(d) else float("nan"), "diff_lo": lo, "diff_hi": hi,
            "win": float((s > 0).sum() / n), "tie": float((s == 0).sum() / n), "loss": float((s < 0).sum() / n),
            "p_wilcoxon": p, "n": int(len(d)), "n_eff": n_eff, "p_min": p_min,
            "p_note": f"n too small (min p = {p_min:.4f})" if 0 < n_eff < 6 else "",
            "unanimous": unanimous, "d_z": dz, "d_z_lo": dz_lo, "d_z_hi": dz_hi}


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
