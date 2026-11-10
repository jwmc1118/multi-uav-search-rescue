"""main_rule 그림 · 표 (eval_common/paper_style.py 스타일, 그림마다 PDF + PNG + 원자료 CSV). 정책 색은 label 별 고정
(paper_style.policy_styles — 모든 그림에서 같은 색).

    fig_compare_bar      지표별 (패널) 정책 평균 + 95% bootstrap CI 막대
    fig_compare_paired   쌍별 seed 차이 (A - B) 분포 — 구조율 · 에피소드 길이 패널, strip + 평균 · 95% CI, 0 기준선
    fig_compare_seed     seed 별 구조율 선 (정책별 색), seed 는 기준 정책 (목록 첫 정책) 구조율 순으로 정렬
    fig_compare_cumulative  에피소드 진행률 (%) 에 따른 누적 발견율 (실선) · 구조율 (점선), 정책별 색, seed 평균 + 95% CI 음영
    fig_compare_roles    정책별 패널: 진행률에 따른 EXPLORE / RESCUE / RETURN 비율 (seed 평균, stacked area)
    (진행률 = step / 그 에피소드 길이, 10 % 구간 — rule_evaluation/compare.py progress_bins · progress_stats)
    fig_compare_conflict    할당 충돌: (a) 에피소드당 동시 추격 (b) 헛걸음 (c) 낭비 step
                         (d) 구조 결정당 동시 추격 — 평균 + 95% CI
    fig_compare_conflict_time   진행률 10 % 구간별 동시 추격 수, 정책별 선 + CI 음영
    (fig_compare_bar 는 BAR_METRICS 만 — 충돌 지표는 위 두 그림)
    table_main.tex       (1) 행 = 정책, 열 = 구조율 · 만료율 · 에피소드 길이 · 발견→구조 시간 (평균 [95% CI], 열마다 가장 좋은 평균 굵게)
                         (2) 짝 비교: 쌍, Δ구조율 [95% CI], W/T/L, Holm p, d_z
    table_conflict.tex   (1) 행 = 정책, 열 = 동시 추격 · 결정당 동시 추격 · 헛걸음 · 낭비 step · 보상 기준 충돌 (적을수록 굵게)
                         (2) 짝 비교: 동시 추격 · 헛걸음 · 낭비 step 의 쌍별 Δ [CI], W/T/L (W = A 가 더 적음), Holm p, d_z
"""
import os

import numpy as np

from eval_common import paper_style as ps
from eval_common.io import latex_block
from eval_common.paper_style import LABEL

TABLE_METRICS = ("rescue_rate", "expire_rate", "episode_len", "discover_to_rescue_mean")
BAR_METRICS = ("rescue_rate", "expire_rate", "discover_rate", "episode_len", "discover_to_rescue_mean", "undiscovered")
CONFLICT_TABLE = ("conflict_concurrent", "conflict_concurrent_per_dec", "wasted_trips", "wasted_steps", "conflict_reward")
CONFLICT_PAIRS = ("conflict_concurrent", "wasted_trips", "wasted_steps")
DIGITS = {"episode_len": 1, "discover_to_rescue_mean": 1, "undiscovered": 2, "conflict_concurrent": 2, "wasted_trips": 2,
          "wasted_steps": 1, "conflict_reward": 2}


def _rows(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def fig_compare_bar(summary, labels, metrics, out, width="double", dpi=300):
    sty = ps.policy_styles(labels)
    ncols = 3 if len(metrics) > 4 else len(metrics)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = ps.figure(width, 1.75 * nrows + 0.35, ncols=ncols, nrows=nrows, squeeze=False)
    axes = axes.ravel()
    data = []
    for ax, m in zip(axes, metrics):
        for k, lab in enumerate(labels):
            r = _rows(summary, policy=lab, metric=m)[0]
            mu, lo, hi = r["mean"], r["ci_lo"], r["ci_hi"]
            ax.bar([k], [mu], width=0.7, color=sty[lab]["color"], alpha=0.85, label=lab,
                   yerr=[[mu - lo], [hi - mu]], error_kw={"elinewidth": 0.8, "capsize": 2})
            data.append({"metric": m, "policy": lab, "mean": mu, "ci_lo": lo, "ci_hi": hi, "n": r["n"]})
        ax.set_xticks([])
        ax.set_ylabel(LABEL.get(m, m))
        ax.grid(axis="x", visible=False)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=len(labels))
    return ps.save(fig, os.path.join(out, "fig_compare_bar"), data, dpi)


def fig_compare_paired(per, pw, pairs, out, metrics=("rescue_rate", "episode_len"), width="double", dpi=300):
    sty = ps.policy_styles(dict.fromkeys(x for pr in pairs for x in pr))
    fig, axes = ps.figure(width, 2.4, ncols=len(metrics), squeeze=False)
    data = []
    for ax, metric in zip(axes[0], metrics):
        rng = np.random.RandomState(0)
        ax.axhline(0, color="0.4", lw=0.8, zorder=0)
        for k, (a, b) in enumerate(pairs):
            ba = {r["seed"]: r[metric] for r in per if r["policy"] == a}
            bb = {r["seed"]: r[metric] for r in per if r["policy"] == b}
            seeds = sorted(set(ba) & set(bb))
            d = np.array([ba[s] - bb[s] for s in seeds], float)
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(d)), d, s=7, color=sty[a]["color"], alpha=0.6, linewidths=0)
            st = _rows(pw, a=a, b=b, metric=metric)[0]
            ax.errorbar([k + 0.3], [st["mean"]], yerr=[[st["mean"] - st["ci_lo"]], [st["ci_hi"] - st["mean"]]], fmt="o",
                        color="black", ms=3.5, elinewidth=1.0, capsize=2.5)
            data += [{"metric": metric, "pair": f"{a} - {b}", "seed": s, "diff": x} for s, x in zip(seeds, d)]
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([f"{a}\n$-$ {b}" for a, b in pairs], fontsize=5.5)
        ax.set_xlim(-0.5, len(pairs) - 0.3)
        ax.set_ylabel(r"$\Delta$ " + LABEL.get(metric, metric).lower() + " (per seed)")
        ax.grid(axis="x", visible=False)
    return ps.save(fig, os.path.join(out, "fig_compare_paired"), data, dpi)


def fig_compare_seed(per, labels, out, metric="rescue_rate", width="double", dpi=300):
    sty = ps.policy_styles(labels)
    by = {lab: {r["seed"]: r[metric] for r in per if r["policy"] == lab} for lab in labels}
    seeds = sorted(set.intersection(*[set(v) for v in by.values()]), key=lambda s: (by[labels[0]][s], s))
    x = np.arange(len(seeds))
    fig, ax = ps.figure(width, 2.2)
    for lab in labels:
        ax.plot(x, [by[lab][s] for s in seeds], color=sty[lab]["color"], marker=sty[lab]["marker"], ms=2.5, lw=0.9,
                alpha=0.9, label=lab)
    if len(seeds) <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds], fontsize=6)
    ax.set_xlabel(f"Seed (sorted by {labels[0]} {LABEL.get(metric, metric).lower()})")
    ax.set_ylabel(LABEL.get(metric, metric))
    ax.legend(loc="lower right", ncol=len(labels))
    return ps.save(fig, os.path.join(out, "fig_compare_seed"),
                   [{"rank": i, "seed": s, **{lab: by[lab][s] for lab in labels}} for i, s in enumerate(seeds)], dpi)


def fig_compare_cumulative(pstats, labels, out, width="single", dpi=300):
    sty = ps.policy_styles(labels)
    fig, ax = ps.figure(width, 2.5)
    for lab in labels:
        rows = [r for r in pstats if r["policy"] == lab]
        x = [r["progress"] for r in rows]
        for key, ls in (("cum_discover_rate", "-"), ("cum_rescue_rate", ":")):
            ax.plot(x, [r[f"{key}_mean"] for r in rows], color=sty[lab]["color"], ls=ls, marker=sty[lab]["marker"], ms=2.5)
            ax.fill_between(x, [r[f"{key}_lo"] for r in rows], [r[f"{key}_hi"] for r in rows], color=sty[lab]["color"],
                            alpha=0.15, linewidth=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=sty[lab]["color"], marker=sty[lab]["marker"], ms=3, label=lab) for lab in labels]
    handles += [Line2D([], [], color="0.3", ls="-", label="discovered"), Line2D([], [], color="0.3", ls=":", label="rescued")]
    ax.legend(handles=handles, loc="lower right", fontsize=6)
    ax.set_xlabel("Episode progress (%)")
    ax.set_ylabel("Cumulative ratio of survivors")
    ax.set_xlim(0, 100)
    return ps.save(fig, os.path.join(out, "fig_compare_cumulative"),
                   [{k: v for k, v in r.items() if k in ("policy", "bin", "progress", "n") or k.startswith("cum_")}
                    for r in pstats], dpi)


def fig_compare_roles(pstats, labels, out, width="double", dpi=300):
    keys = ("explore", "rescue", "return")
    fig, axes = ps.figure(width, 2.0, ncols=len(labels), squeeze=False, sharey=True)
    for ax, lab in zip(axes[0], labels):
        rows = [r for r in pstats if r["policy"] == lab]
        x = [0.0] + [r["progress"] for r in rows]                 # 구간 평균을 구간 오른쪽 끝에 두고 0 % 는 첫 구간 값
        ys = [[rows[0][f"role_{k}_mean"]] + [r[f"role_{k}_mean"] for r in rows] for k in keys]
        ax.stackplot(x, *ys, colors=[ps.ROLE_COLORS[k] for k in keys], labels=["EXPLORE", "RESCUE", "RETURN"], alpha=0.85,
                     linewidth=0, step="pre")
        ax.set_title(lab, fontsize=7.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Episode progress (%)")
        ax.grid(False)
    axes[0][0].set_ylabel("Role share of UAVs")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    return ps.save(fig, os.path.join(out, "fig_compare_roles"),
                   [{k: v for k, v in r.items() if k in ("policy", "bin", "progress", "n") or k.startswith("role_")}
                    for r in pstats], dpi)


def _tex(label):
    return label.replace("_", r"\_")


def _policy_table(summary, labels, cols, lower_is_better, caption, label):
    """행 = 정책, 열 = 지표 (평균 [95% CI], 열마다 가장 좋은 평균 굵게)."""
    best = {}
    for m in cols:
        ok = {lab: _rows(summary, policy=lab, metric=m)[0]["mean"] for lab in labels}
        ok = {k: v for k, v in ok.items() if v == v}
        if ok and len(set(ok.values())) > 1:
            target = (min if m in lower_is_better else max)(ok.values())
            best[m] = {k for k, v in ok.items() if v == target}
    rows = []
    for lab in labels:
        cells = [_tex(lab)]
        for m in cols:
            r, d = _rows(summary, policy=lab, metric=m)[0], DIGITS.get(m, 3)
            c = f"{ps.fmt(r['mean'], d)} [{ps.fmt(r['ci_lo'], d)}, {ps.fmt(r['ci_hi'], d)}]"
            cells.append(f"\\textbf{{{c}}}" if lab in best.get(m, ()) else c)
        rows.append(cells)
    return latex_block(["Policy"] + [LABEL[m] for m in cols], rows, "l" + "c" * len(cols), caption=caption, label=label)


def _pair_table(pw, pairs, cols, caption, label):
    """행 = 쌍 (지표가 여럿이면 쌍 × 지표), 열 = Δ [95% CI], W/T/L, Holm p, d_z."""
    rows = []
    for m in cols:
        for a, b in pairs:
            r, d = _rows(pw, a=a, b=b, metric=m)[0], DIGITS.get(m, 3)
            rows.append(([LABEL[m]] if len(cols) > 1 else []) +
                        [f"{_tex(a)} $-$ {_tex(b)}", f"{ps.fmt(r['mean'], d)} [{ps.fmt(r['ci_lo'], d)}, {ps.fmt(r['ci_hi'], d)}]",
                         f"{r['win']}/{r['tie']}/{r['loss']}", ps.fmt(r["p_holm"], 3), ps.fmt(r["d_z"], 2)])
    head = (["Metric"] if len(cols) > 1 else []) + ["Pair", (r"$\Delta$ [95\% CI]" if len(cols) > 1 else
                                                              r"$\Delta$ " + LABEL[cols[0]].lower() + r" [95\% CI]"),
                                                     "W/T/L", r"Holm $p$", "$d_z$"]
    return latex_block(head, rows, "l" * (len(head) - 4) + "cccc", caption=caption, label=label)


_PAIR_NOTE = ("Wilcoxon signed-rank $p$ with Holm correction over the {k} pairs per metric; $d_z$ = mean / std of paired "
              "differences, -- if the differences are constant")


def table_main(summary, pw, labels, pairs, lower_is_better, path, n):
    """table_main.tex — 정책 표 + 구조율 짝 비교 표 (한 파일에 table 두 개)."""
    cols = [m for m in TABLE_METRICS if _rows(summary, metric=m)]
    text = _policy_table(summary, labels, cols, lower_is_better,
                         f"Policy comparison on the same {n} seeds (mean [95\\% bootstrap CI]; best mean in bold).",
                         "tab:policy_compare")
    if pairs:
        text += "\n" + _pair_table(pw, pairs, ["rescue_rate"],
                                   f"Paired differences in rescue rate over {n} seeds (A $-$ B; W/T/L = seeds where A is higher / "
                                   f"equal / lower; {_PAIR_NOTE.format(k=len(pairs))}).", "tab:policy_pairs")
    with open(path, "w") as f:
        f.write(text)
    return path


def table_conflict(summary, pw, labels, pairs, lower_is_better, path, n):
    """table_conflict.tex — 할당 충돌 정책 표 + 짝 비교 표 (모두 작을수록 좋음: W = A 가 더 적은 seed)."""
    cols = [m for m in CONFLICT_TABLE if _rows(summary, metric=m)]
    text = _policy_table(summary, labels, cols, lower_is_better,
                         f"Rescue-assignment conflicts per episode on the same {n} seeds (mean [95\\% bootstrap CI]; "
                         f"fewest in bold). Concurrent chase: a new rescue target already pursued by another UAV at decision "
                         f"time; wasted trip / steps: pursuit that ended after another UAV rescued the target, and the moves "
                         f"made during it.", "tab:policy_conflict")
    pcols = [m for m in CONFLICT_PAIRS if _rows(pw, metric=m)]
    if pairs and pcols:
        text += "\n" + _pair_table(pw, pairs, pcols,
                                   f"Paired differences in conflicts over {n} seeds (A $-$ B; lower is better, so W/T/L = seeds "
                                   f"where A has fewer / equal / more; {_PAIR_NOTE.format(k=len(pairs))}).",
                                   "tab:policy_conflict_pairs")
    with open(path, "w") as f:
        f.write(text)
    return path


def fig_compare_conflict(summary, labels, out, width="double", dpi=300):
    """(a) 동시 추격, (b) 헛걸음, (c) 낭비 step, (d) 구조 결정당 동시 추격 — 에피소드당 평균 + CI."""
    sty = ps.policy_styles(labels)
    fig, axes = ps.figure(width, 2.1, ncols=4)
    data = []
    g = lambda lab, m: _rows(summary, policy=lab, metric=m)[0]
    for p, ax, m, ylab in zip("abcd", axes, ("conflict_concurrent", "wasted_trips", "wasted_steps", "conflict_concurrent_per_dec"),
                              ("Concurrent chases / episode", "Wasted trips / episode", "Wasted steps / episode",
                               "Concurrent chases / rescue decision")):
        for k, lab in enumerate(labels):
            r = g(lab, m)
            ax.bar([k], [r["mean"]], width=0.7, color=sty[lab]["color"], alpha=0.85, label=lab,
                   yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], error_kw={"elinewidth": 0.8, "capsize": 2})
            data.append({"panel": p, "metric": m, "policy": lab, "mean": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
        ax.set_ylabel(ylab)
    for k, ax in enumerate(axes):
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"({'abcd'[k]})", fontsize=7.5, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=len(labels))
    return ps.save(fig, os.path.join(out, "fig_compare_conflict"), data, dpi)


def fig_compare_conflict_time(pstats, labels, out, width="single", dpi=300):
    sty = ps.policy_styles(labels)
    fig, ax = ps.figure(width, 2.3)
    for lab in labels:
        rows = [r for r in pstats if r["policy"] == lab]
        x = [r["progress"] for r in rows]
        ax.plot(x, [r["conflict_concurrent_mean"] for r in rows], color=sty[lab]["color"], marker=sty[lab]["marker"], ms=2.5,
                label=lab)
        ax.fill_between(x, [r["conflict_concurrent_lo"] for r in rows], [r["conflict_concurrent_hi"] for r in rows],
                        color=sty[lab]["color"], alpha=0.15, linewidth=0)
    ax.set_xlabel("Episode progress (%)")
    ax.set_ylabel("Concurrent chases per 10% bin")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right", fontsize=6)
    return ps.save(fig, os.path.join(out, "fig_compare_conflict_time"),
                   [{k: v for k, v in r.items() if k in ("policy", "bin", "progress", "n") or k.startswith("conflict_")}
                    for r in pstats], dpi)


def make_all(per, summary, pw, pstats, labels, pairs, out_dir, fig_cfg):
    fdir = os.path.join(out_dir, "figures")
    width, dpi = fig_cfg.get("width", "double"), int(fig_cfg.get("dpi", 300))
    files = fig_compare_bar(summary, labels, [m for m in BAR_METRICS if _rows(summary, metric=m)], fdir, width, dpi)
    if pairs:
        files += fig_compare_paired(per, pw, pairs, fdir, width=width, dpi=dpi)
    files += fig_compare_seed(per, labels, fdir, width=width, dpi=dpi)
    files += fig_compare_cumulative(pstats, labels, fdir, dpi=dpi)
    files += fig_compare_roles(pstats, labels, fdir, width=width, dpi=dpi)
    if _rows(summary, metric="conflict_concurrent"):
        files += fig_compare_conflict(summary, labels, fdir, width, dpi)
        files += fig_compare_conflict_time(pstats, labels, fdir, dpi=dpi)
    return files
