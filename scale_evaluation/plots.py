"""main_scale 그림 · 표 (eval_common/paper_style.py 스타일, 그림마다 PDF + PNG + 원자료 CSV).

x 축 = UAV 수 N (목록 순서대로 같은 간격), N_ref 는 세로 점선. 크기 조정이 있으면 위쪽 보조 눈금에 맵 크기 / 조난자 수.
    1 fig_rates_vs_n       구조율 · 발견율 · 만료율 — 정책별 평균 · 95% CI
    2 fig_diff_vs_n        짝 차이 (모델 - 규칙) 구조율 · 발견율 평균 · 95% CI, 0 기준선
    3 fig_rescued_per_uav  UAV 당 구조 수 — 정책별 평균 · 95% CI
    4 fig_inference        모델 추론 시간 (ms / 결정, CPU) 평균 · 95% CI
    5 fig_upper_bound      구조율 (규칙 · 모델) 과 이론적 상한 UB_loose (+ 참고 UB_greedy), 상한 대비 구조율 (구조율 / UB_loose)
    scale_table.tex        N 별 맵 / 조난자, 구조율 (규칙 · 모델, 평균 ± 표준편차), 차이 [95% CI], d_z [95% CI], p, 최소 p
                           (더 좋은 평균 굵게, † = 0 아닌 차이 < 6 개라 p < 0.05 불가, ‡ = 모든 짝 차이가 같은 쪽)
"""
import os

import numpy as np

from eval_common import paper_style as ps
from eval_common.io import latex_table
from rule_evaluation.plots import LABEL, POLICIES, _fmt


def _xpos(ns, n_ref):
    x = np.arange(len(ns), dtype=float)
    ref = float(np.interp(n_ref, ns, x)) if ns[0] <= n_ref <= ns[-1] else None
    return x, ref


def _axis(ax, ns, n_ref, info, secondary):
    x, ref = _xpos(ns, n_ref)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(-0.4, len(ns) - 0.6)
    ax.set_xlabel("Number of UAVs $N$")
    if ref is not None:
        ax.axvline(ref, color="0.35", lw=0.8, ls="--", zorder=0)
    if secondary:
        top = ax.secondary_xaxis("top")
        top.set_xticks(x)
        top.set_xticklabels([f"{info[n]['rows']}×{info[n]['cols']}\n{info[n]['survivors']}" for n in ns], fontsize=5.5)
        top.tick_params(length=2, pad=1)
        top.set_xlabel("Map / survivors", fontsize=6.5, labelpad=2)
        for side in ("left", "right", "bottom"):
            top.spines[side].set_visible(False)
        ax.spines["right"].set_visible(False)
    return x


def _by(summary, metric):
    return {r["n_agents"]: r for r in summary if r["metric"] == metric}


def _info(summary):
    return {r["n_agents"]: r for r in summary}


def _policy_panel(ax, summary, metric, ns, n_ref, secondary):
    rows = _by(summary, metric)
    x = _axis(ax, ns, n_ref, _info(summary), secondary)
    data = []
    for k, p in enumerate(POLICIES):
        s = ps.POLICY_STYLE[p]
        mu = np.array([rows[n][f"{p}_mean"] for n in ns])
        lo = np.array([rows[n][f"{p}_ci_lo"] for n in ns])
        hi = np.array([rows[n][f"{p}_ci_hi"] for n in ns])
        dx = (k - 0.5) * 0.12                                 # 두 정책 오차 막대가 겹치지 않게 살짝 비킴
        ax.errorbar(x + dx, mu, yerr=[mu - lo, hi - mu], color=s["color"], marker=s["marker"], ls=s["linestyle"],
                    label=s["label"])
        data += [{"n_agents": n, "metric": metric, "policy": p, "mean": a, "ci_lo": b, "ci_hi": c}
                 for n, a, b, c in zip(ns, mu, lo, hi)]
    ax.set_ylabel(LABEL[metric])
    return data


def fig_rates_vs_n(summary, ns, n_ref, out, width, secondary):
    keys = ["rescue_rate", "discover_rate", "expire_rate"]
    fig, axes = ps.figure(width, 2.6 if secondary else 2.3, ncols=len(keys))
    data = []
    for ax, m in zip(axes, keys):
        data += _policy_panel(ax, summary, m, ns, n_ref, secondary)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    return ps.save(fig, os.path.join(out, "fig_rates_vs_n"), data)


def fig_diff_vs_n(summary, ns, n_ref, out, width, secondary):
    keys = ["rescue_rate", "discover_rate"]
    fig, axes = ps.figure(width, 2.3 if secondary else 2.0, ncols=len(keys))
    c = ps.POLICY_STYLE["model"]["color"]
    data = []
    for ax, m in zip(axes, keys):
        rows = _by(summary, m)
        x = _axis(ax, ns, n_ref, _info(summary), secondary)
        mu = np.array([rows[n]["diff_mean"] for n in ns])
        lo = np.array([rows[n]["diff_ci_lo"] for n in ns])
        hi = np.array([rows[n]["diff_ci_hi"] for n in ns])
        ax.axhline(0, color="0.4", lw=0.8)
        ax.plot(x, mu, color=c, marker="o")
        ax.fill_between(x, lo, hi, color=c, alpha=0.2, linewidth=0)
        ax.set_ylabel(r"$\Delta$ " + LABEL[m].lower() + r" (learned $-$ rule)")
        data += [{"n_agents": n, "metric": m, "diff_mean": a, "ci_lo": b, "ci_hi": d, "p_wilcoxon": rows[n]["p_wilcoxon"]}
                 for n, a, b, d in zip(ns, mu, lo, hi)]
    return ps.save(fig, os.path.join(out, "fig_diff_vs_n"), data)


def fig_rescued_per_uav(summary, ns, n_ref, out, secondary):
    fig, ax = ps.figure("single", 2.8 if secondary else 2.5)
    data = _policy_panel(ax, summary, "rescued_per_uav", ns, n_ref, secondary)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2)
    return ps.save(fig, os.path.join(out, "fig_rescued_per_uav"), data)


def fig_inference(infer, summary, ns, n_ref, out, secondary):
    by = {r["n_agents"]: r for r in infer}
    fig, ax = ps.figure("single", 2.5 if secondary else 2.2)
    x = _axis(ax, ns, n_ref, _info(summary), secondary)
    mu = np.array([by[n]["infer_ms_mean"] for n in ns])
    lo = np.array([by[n]["infer_ms_ci_lo"] for n in ns])
    hi = np.array([by[n]["infer_ms_ci_hi"] for n in ns])
    s = ps.POLICY_STYLE["model"]
    ax.errorbar(x, mu, yerr=[mu - lo, hi - mu], color=s["color"], marker=s["marker"])
    ax.set_ylabel("Inference time (ms / decision)")
    ax.set_ylim(bottom=0)
    return ps.save(fig, os.path.join(out, "fig_inference"), infer)


def fig_upper_bound(summary, ns, n_ref, out, width, secondary):
    fig, axes = ps.figure(width, 2.6 if secondary else 2.3, ncols=2)
    ub = _by(summary, "ub_loose")
    x = _axis(axes[0], ns, n_ref, _info(summary), secondary)
    mu = np.array([ub[n]["rule_mean"] for n in ns])                       # 상한은 정책과 무관 (같은 (seed, N))
    lo = np.array([ub[n]["rule_ci_lo"] for n in ns])
    hi = np.array([ub[n]["rule_ci_hi"] for n in ns])
    axes[0].plot(x, mu, color="k", lw=1.0, marker="^", label=r"$UB_\mathrm{loose}$")
    axes[0].fill_between(x, lo, hi, color="k", alpha=0.12, linewidth=0)
    data = [{"n_agents": n, "metric": "ub_loose", "policy": "-", "mean": a, "ci_lo": b, "ci_hi": c}
            for n, a, b, c in zip(ns, mu, lo, hi)]
    ug = _by(summary, "ub_greedy")
    if ug:                                                                # 전지적 EDF 스케줄 (참고 — UB_loose 가 1 에 붙을 때 유용)
        mg = np.array([ug[n]["rule_mean"] for n in ns])
        axes[0].plot(x, mg, color="k", lw=0.9, ls=":", marker="v", label=r"$UB_\mathrm{greedy}$")
        data += [{"n_agents": n, "metric": "ub_greedy", "policy": "-", "mean": a, "ci_lo": ug[n]["rule_ci_lo"],
                  "ci_hi": ug[n]["rule_ci_hi"]} for n, a in zip(ns, mg)]
    data += _policy_panel(axes[0], summary, "rescue_rate", ns, n_ref, False)
    axes[0].set_ylabel("Rescue rate")
    data += _policy_panel(axes[1], summary, "rescue_over_ub", ns, n_ref, secondary)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    return ps.save(fig, os.path.join(out, "fig_upper_bound"), data)


def scale_table(summary, path, n_seeds, metric="rescue_rate"):
    rows_by = _by(summary, metric)
    header = ["$N$", "Map", "Survivors", "Rule", "Learned", r"$\Delta$ [95\% CI]", r"$d_z$ [95\% CI]", "$p$", r"min $p$"]
    rows = []
    for n in sorted(rows_by):
        r = rows_by[n]
        cells = {p: f"{_fmt(r[f'{p}_mean'])} $\\pm$ {_fmt(r[f'{p}_std'])}" for p in POLICIES}
        if r["rule_mean"] != r["model_mean"]:
            better = ("model" if r["model_mean"] > r["rule_mean"] else "rule") if r["higher_is_better"] else \
                ("model" if r["model_mean"] < r["rule_mean"] else "rule")
            cells[better] = f"\\textbf{{{cells[better]}}}"
        rows.append([str(n), f"{r['rows']}$\\times${r['cols']}", r["survivors"].replace("–", "--"), cells["rule"],
                     cells["model"], f"{_fmt(r['diff_mean'])} [{_fmt(r['diff_ci_lo'])}, {_fmt(r['diff_ci_hi'])}]"
                     + (r"$^{\ddagger}$" if r.get("unanimous") else ""),
                     f"{_fmt(r.get('d_z'), 2)} [{_fmt(r.get('d_z_ci_lo'), 2)}, {_fmt(r.get('d_z_ci_hi'), 2)}]",
                     _fmt(r["p_wilcoxon"]) + (r"$^{\dagger}$" if r.get("p_note") else ""), _fmt(r.get("p_min"), 4)])
    return latex_table(path, header, rows, "rcccccccc",
                       caption=f"{LABEL[metric]} vs.\\ number of UAVs (mean $\\pm$ std over {n_seeds} paired episodes per $N$; "
                               f"$\\Delta$ = learned $-$ rule with 95\\% bootstrap CI; effect size $d_z$ with bootstrap CI; "
                               f"Wilcoxon signed-rank $p$ and the smallest attainable $p$. $^{{\\dagger}}$fewer than 6 non-zero "
                               f"differences: $p < 0.05$ is unattainable. $^{{\\ddagger}}$all paired differences favour the same "
                               f"policy).", label="tab:scale")


def make_all(summary, infer, ns, n_ref, out_dir, fig_cfg, secondary):
    fdir = os.path.join(out_dir, "figures")
    width = fig_cfg.get("width", "double")
    files = []
    files += fig_rates_vs_n(summary, ns, n_ref, fdir, width, secondary)
    files += fig_diff_vs_n(summary, ns, n_ref, fdir, width, secondary)
    files += fig_rescued_per_uav(summary, ns, n_ref, fdir, secondary)
    files += fig_inference(infer, summary, ns, n_ref, fdir, secondary)
    if any(r["metric"] == "ub_loose" for r in summary) and any(r["metric"] == "rescue_over_ub" for r in summary):
        files += fig_upper_bound(summary, ns, n_ref, fdir, width, secondary)
    return files
