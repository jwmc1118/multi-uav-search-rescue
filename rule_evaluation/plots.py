"""main_rule 그림 · 표 (eval_common/paper_style.py 스타일, 그림마다 PDF + PNG + 원자료 CSV).

    1 fig_metrics_ci       rescue / discover / expire rate, explored — 정책별 평균 · 95% CI 점그래프 (패널 4 개)
    2 fig_rescue_violin    구조율 분포 violin + seed 별 점
    3 fig_paired_scatter   seed 별 구조율 (x = 규칙, y = 모델), y = x 대각선 — 위쪽이 모델 우세
    4 fig_cumulative       step 에 따른 누적 구조율 · 발견율 (평균 + 95% CI 음영)
    5 fig_role_mix         모델의 역할 비율 시계열 (누적 영역)
    6 fig_convergence      에피소드 수에 따른 구조율 짝 차이 (모델 - 규칙) 평균 · 95% CI
    summary_table.tex      지표별 평균 ± 표준편차, 차이 [95% CI], p-value (booktabs, 더 좋은 평균 굵게)
"""
import os

import numpy as np
from matplotlib.ticker import MaxNLocator

from eval_common import paper_style as ps
from eval_common.io import latex_table

LABEL = {"rescue_rate": "Rescue rate", "expire_rate": "Expire rate", "discover_rate": "Discover rate",
         "explored": "Explored ratio", "episode_len": "Episode length", "rescue_time_mean": "Mean rescue time",
         "golden_slack_mean": "Golden-time slack", "same_cell": "Same-cell overlaps", "conflict_cross": "Cross-group conflicts",
         "escalations": "Escalations", "rescued_per_uav": "Rescued per UAV", "infer_ms_per_decision": "Inference (ms/decision)"}
POLICIES = ("rule", "model")


def _sty(p):
    return ps.POLICY_STYLE[p]


def fig_metrics_ci(summary, out, width="double"):
    keys = ["rescue_rate", "discover_rate", "expire_rate", "explored"]
    rows = [r for r in summary if r["metric"] in keys]
    fig, axes = ps.figure(width, 1.9, ncols=len(rows))
    data = []
    for ax, r in zip(np.atleast_1d(axes), rows):
        for k, p in enumerate(POLICIES):
            s = _sty(p)
            mu, lo, hi = r[f"{p}_mean"], r[f"{p}_ci_lo"], r[f"{p}_ci_hi"]
            ax.errorbar([k], [mu], yerr=[[mu - lo], [hi - mu]], fmt=s["marker"], color=s["color"], label=s["label"])
            data.append({"metric": r["metric"], "policy": p, "mean": mu, "ci_lo": lo, "ci_hi": hi, "n": r["n"]})
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Rule", "Learned"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(LABEL[r["metric"]])
    return ps.save(fig, os.path.join(out, "fig_metrics_ci"), data)


def fig_rescue_violin(per, out):
    fig, ax = ps.figure("single", 2.3)
    rng = np.random.RandomState(0)
    data = []
    for k, p in enumerate(POLICIES):
        s = _sty(p)
        x = np.array([r["rescue_rate"] for r in per if r["policy"] == p], float)
        vp = ax.violinplot([x], positions=[k], widths=0.7, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(s["color"])
            b.set_alpha(0.25)
            b.set_edgecolor(s["color"])
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(x)), x, s=7, color=s["color"], marker=s["marker"], alpha=0.8,
                   linewidths=0)
        data += [{"policy": p, "seed": r["seed"], "rescue_rate": r["rescue_rate"]} for r in per if r["policy"] == p]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rule", "Learned"])
    ax.set_ylabel("Rescue rate")
    return ps.save(fig, os.path.join(out, "fig_rescue_violin"), data)


def fig_paired_scatter(per, out):
    by = {p: {r["seed"]: r["rescue_rate"] for r in per if r["policy"] == p} for p in POLICIES}
    seeds = sorted(set(by["rule"]) & set(by["model"]))
    x = np.array([by["rule"][s] for s in seeds])
    y = np.array([by["model"][s] for s in seeds])
    fig, ax = ps.figure("single", 2.6)
    lo, hi = min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls=":")
    ax.scatter(x, y, s=10, color=_sty("model")["color"], marker="o", alpha=0.8, linewidths=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Rule rescue rate")
    ax.set_ylabel("Learned rescue rate")
    return ps.save(fig, os.path.join(out, "fig_paired_scatter"),
                   [{"seed": s, "rule": a, "model": b} for s, a, b in zip(seeds, x, y)])


def fig_cumulative(tstats, out, width="double"):
    fig, axes = ps.figure(width, 2.0, ncols=2)
    for ax, key, lab in zip(axes, ("cum_rescue_rate", "cum_discover_rate"), ("Cumulative rescue rate", "Cumulative discover rate")):
        for p in POLICIES:
            s = _sty(p)
            rows = [r for r in tstats if r["policy"] == p and f"{key}_mean" in r]
            t = [r["step"] for r in rows]
            ax.plot(t, [r[f"{key}_mean"] for r in rows], color=s["color"], ls=s["linestyle"], label=s["label"])
            ax.fill_between(t, [r[f"{key}_lo"] for r in rows], [r[f"{key}_hi"] for r in rows], color=s["color"], alpha=0.2,
                            linewidth=0)
        ax.set_xlabel("Step")
        ax.set_ylabel(lab)
    axes[0].legend(loc="lower right")
    data = [{k: v for k, v in r.items() if k == "policy" or k == "step" or k.startswith(("cum_rescue", "cum_discover"))}
            for r in tstats]
    return ps.save(fig, os.path.join(out, "fig_cumulative"), data)


def fig_role_mix(tstats, out):
    rows = [r for r in tstats if r["policy"] == "model" and "role_explore_mean" in r]
    t = [r["step"] for r in rows]
    ys = [[r[f"role_{k}_mean"] for r in rows] for k in ("explore", "rescue", "return")]
    fig, ax = ps.figure("single", 2.2)
    ax.stackplot(t, *ys, colors=[ps.ROLE_COLORS[k] for k in ("explore", "rescue", "return")],
                 labels=["Explore", "Rescue", "Return"], alpha=0.85, linewidth=0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Role share (learned)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return ps.save(fig, os.path.join(out, "fig_role_mix"),
                   [{"step": a, "explore": b, "rescue": c, "return": d} for a, b, c, d in zip(t, *ys)])


def fig_convergence(conv, out):
    fig, ax = ps.figure("single", 2.2)
    k = [r["episodes"] for r in conv]
    ax.axhline(0, color="0.4", lw=0.8)
    ax.plot(k, [r["diff_mean"] for r in conv], color=_sty("model")["color"])
    ax.fill_between(k, [r["ci_lo"] for r in conv], [r["ci_hi"] for r in conv], color=_sty("model")["color"], alpha=0.2,
                    linewidth=0)
    ax.set_xlabel("Number of episodes")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(r"$\Delta$ rescue rate (learned $-$ rule)")
    return ps.save(fig, os.path.join(out, "fig_convergence"), conv)


def _fmt(x, digits=3):
    return "--" if x is None or x != x else f"{x:.{digits}f}"


def summary_table(summary, path, n):
    header = ["Metric", "Rule", "Learned", r"$\Delta$ (learned $-$ rule) [95\% CI]", "$p$"]
    rows = []
    for r in summary:
        d = 1 if r["metric"] in ("episode_len", "rescue_time_mean", "same_cell", "conflict_cross", "escalations") else 3
        cells = {p: f"{_fmt(r[f'{p}_mean'], d)} $\\pm$ {_fmt(r[f'{p}_std'], d)}" for p in POLICIES}
        better = None
        if r["rule_mean"] == r["rule_mean"] and r["model_mean"] == r["model_mean"] and r["rule_mean"] != r["model_mean"]:
            better = ("model" if r["model_mean"] > r["rule_mean"] else "rule") if r["higher_is_better"] else \
                ("model" if r["model_mean"] < r["rule_mean"] else "rule")
        if better:
            cells[better] = f"\\textbf{{{cells[better]}}}"
        rows.append([LABEL.get(r["metric"], r["metric"]), cells["rule"], cells["model"],
                     f"{_fmt(r['diff_mean'], d)} [{_fmt(r['diff_ci_lo'], d)}, {_fmt(r['diff_ci_hi'], d)}]",
                     _fmt(r["p_wilcoxon"], 3)])
    return latex_table(path, header, rows, "lcccc",
                       caption=f"Rule vs.\\ learned policy (mean $\\pm$ std over {n} paired episodes; "
                               f"$\\Delta$ with 95\\% bootstrap CI; Wilcoxon signed-rank $p$).", label="tab:rule_vs_model")


def make_all(per, summary, tstats, conv, out_dir, fig_cfg):
    fdir = os.path.join(out_dir, "figures")
    width = fig_cfg.get("width", "double")
    files = []
    files += fig_metrics_ci(summary, fdir, width)
    files += fig_rescue_violin(per, fdir)
    files += fig_paired_scatter(per, fdir)
    files += fig_cumulative(tstats, fdir, width)
    files += fig_role_mix(tstats, fdir)
    files += fig_convergence(conv, fdir)
    return files
