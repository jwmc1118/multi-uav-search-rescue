"""main_rule 짝 비교 통계 점검 (eval_common/stats.py diff_stats · holm). scipy 필요 (requirements.txt).

    python -m tests.eval.test_stats

[1] 알려진 차이 배열: 평균, win / tie / loss (higher_better 와 반대 방향), d_z = 평균 / 표준편차 (ddof 1), Wilcoxon p = scipy 와 같음
[2] 차이가 전부 0: p = 1, tie = n, d_z = nan (표준편차 0) / 상수 차이: d_z = nan
[3] Holm 보정: 손으로 계산한 값 (단조 보정 · 1 상한 · nan 은 제외) 과 같음
"""
import numpy as np
from scipy.stats import wilcoxon

from eval_common.stats import diff_stats, holm


def main():
    d = [0.1, 0.0, -0.05, 0.2, 0.0, 0.15, 0.05, -0.1]
    st = diff_stats(d, n_boot=2000)
    assert abs(st["mean"] - 0.04375) < 1e-12, st
    assert (st["win"], st["tie"], st["loss"]) == (4, 2, 2), st
    assert abs(st["d_z"] - np.mean(d) / np.std(d, ddof=1)) < 1e-12, st
    assert abs(st["p_wilcoxon"] - wilcoxon(d, zero_method="wilcox").pvalue) < 1e-12, st
    assert st["ci_lo"] < st["mean"] < st["ci_hi"], st
    lo = diff_stats(d, n_boot=2000, higher_better=False)
    assert (lo["win"], lo["tie"], lo["loss"]) == (2, 2, 4) and lo["mean"] == st["mean"], lo
    print(f"[1] mean {st['mean']:.5f}, W/T/L {st['win']}/{st['tie']}/{st['loss']}, d_z {st['d_z']:.4f}, p {st['p_wilcoxon']:.4f}")

    z = diff_stats([0.0] * 5, n_boot=100)
    assert z["p_wilcoxon"] == 1.0 and (z["win"], z["tie"], z["loss"]) == (0, 5, 0) and z["d_z"] != z["d_z"], z
    c = diff_stats([0.2] * 4, n_boot=100)
    assert c["d_z"] != c["d_z"] and c["win"] == 4, c
    print("[2] all-zero differences: p = 1, ties = n, d_z = nan; constant differences: d_z = nan")

    # p = [0.01, 0.04, 0.03, nan, 0.5] -> 유효 4 개: 0.01*4 = 0.04, 0.03*3 = 0.09, 0.04*2 = 0.08 -> 단조 0.09, 0.5*1 = 0.5
    h = holm([0.01, 0.04, 0.03, float("nan"), 0.5])
    want = [0.04, 0.09, 0.09, float("nan"), 0.5]
    assert all((a != a and b != b) or abs(a - b) < 1e-12 for a, b in zip(h, want)), h
    assert holm([0.4, 0.6]) == [0.8, 0.8] and holm([0.9, 0.7, 0.8]) == [1.0, 1.0, 1.0], (holm([0.4, 0.6]), holm([0.9, 0.7, 0.8]))
    print(f"[3] Holm {['%.2f' % x for x in h]}")
    print("test_stats: all ok")


if __name__ == "__main__":
    main()
