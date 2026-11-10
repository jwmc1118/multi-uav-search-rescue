"""논문용 그림 스타일 (IEEE) — main_rule 평가 그림 (rule_evaluation/plots.py).

    apply_style()                 serif (Times 계열 -> Nimbus Roman -> STIX), mathtext stix, 축 라벨 8.5 pt, 눈금 7.5 pt,
                                  위 · 오른쪽 테두리 없음, 얇은 격자
    figure(width="single"|"double", height)   단단 3.5 in / 양단 7.16 in
    policy_styles(labels)         정책 label 별 고정 색 · 마커 (Okabe–Ito 기반, 흑백 인쇄에서도 구분 — COMBO_STYLE 의 조합 이름은
                                  항상 같은 색, 그 밖은 남은 색 순서)
    LABEL, fmt(x, digits)         지표 표시 이름, 표 숫자 형식 (nan -> --)
    save(fig, path_no_ext, data_rows)   PDF (벡터) + PNG (300 dpi) + 원자료 CSV (같은 이름)
그림 안에는 제목을 넣지 않는다 (캡션은 논문에서).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_common.io import write_csv  # noqa: E402

WIDTH = {"single": 3.5, "double": 7.16}
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F"}
COMBO_STYLE = {"rule_rule_rule": (OKABE_ITO["gray"], "s"), "learn_rule_rule": (OKABE_ITO["orange"], "^"),
               "learn_learn_rule": (OKABE_ITO["green"], "D"), "learn_learn_learn": (OKABE_ITO["blue"], "o")}
_EXTRA = [(OKABE_ITO[k], m) for k, m in (("vermillion", "v"), ("purple", "P"), ("skyblue", "X"), ("black", "*"),
                                         ("yellow", "h"))]
LABEL = {"rescue_rate": "Rescue rate", "expire_rate": "Expire rate", "discover_rate": "Discover rate",
         "explored": "Explored ratio", "episode_len": "Episode length", "rescue_time_mean": "Mean rescue time",
         "discover_to_rescue_mean": "Discovery-to-rescue time", "undiscovered": "Undiscovered survivors",
         "golden_slack_mean": "Golden-time slack", "same_cell": "Same-cell overlaps", "conflict_cross": "Cross-group conflicts",
         "escalations": "Escalations", "rescued_per_uav": "Rescued per UAV", "infer_ms_per_decision": "Inference (ms/decision)",
         "trunc_decisions_S": "Truncated decisions (S)", "trunc_frac_S": "Truncated-decision ratio (S)",
         "dropped_S": "Dropped candidates (S)", "dropped_S_per_dec": "Dropped per decision (S)",
         "dropped_urgent_S": "Dropped urgent (S)", "best_dropped_S": "EDF choice dropped (S)",
         "trunc_decisions_K": "Truncated decisions (K)", "trunc_frac_K": "Truncated-decision ratio (K)",
         "dropped_K": "Dropped blobs (K)", "dropped_K_cells": "Dropped frontier cells (K)",
         "s_nodes_overflow": "S-node overflow", "ub_loose": r"Upper bound $UB_\mathrm{loose}$",
         "ub_greedy": r"$UB_\mathrm{greedy}$", "rescue_over_ub": r"Rescue rate / $UB_\mathrm{loose}$",
         "group_size_mean": "Mean comm. group size", "n_groups_mean": "Mean number of groups",
         "isolated_frac": "Isolated-UAV ratio", "expire_undiscovered_rate": "Expired undiscovered",
         "expire_unassigned_rate": "Expired unassigned", "expire_assigned_rate": "Expired while assigned",
         "conflict_concurrent": "Concurrent chases", "conflict_concurrent_same_group": "Concurrent chases (same group)",
         "conflict_concurrent_cross_group": "Concurrent chases (cross-group)", "wasted_trips": "Wasted trips",
         "wasted_steps": "Wasted steps", "conflict_reward": "Reward-rule conflicts", "rescue_decisions": "Rescue decisions",
         "conflict_concurrent_per_dec": "Concurrent chases / decision",
         "conflict_concurrent_same_group_per_dec": "Same-group chases / decision",
         "conflict_concurrent_cross_group_per_dec": "Cross-group chases / decision",
         "wasted_trips_per_dec": "Wasted trips / decision", "wasted_steps_per_dec": "Wasted steps / decision",
         "conflict_reward_per_dec": "Reward-rule conflicts / decision"}
ROLE_COLORS = {"explore": OKABE_ITO["skyblue"], "rescue": OKABE_ITO["orange"], "return": OKABE_ITO["gray"]}


def policy_styles(labels):
    """{label: {"color", "marker"}} — 조합 이름 (COMBO_STYLE) 은 항상 같은 색, 그 밖의 label 은 순서대로 남은 색."""
    extra = iter(_EXTRA)
    out = {}
    for lab in labels:
        c, m = COMBO_STYLE[lab] if lab in COMBO_STYLE else next(extra)
        out[lab] = {"color": c, "marker": m}
    return out


def fmt(x, digits=3):
    return "--" if x is None or x != x else f"{x:.{digits}f}"


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.4, "grid.linestyle": "-",
        "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 2,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(width="single", height=2.3, ncols=1, nrows=1, **kw):
    apply_style()
    kw.setdefault("constrained_layout", True)                # 패널 라벨 · 눈금이 겹치지 않게
    return plt.subplots(nrows, ncols, figsize=(WIDTH[width], height), **kw)


def save(fig, path_no_ext, data_rows=None, dpi=300):
    """PDF + PNG + (원자료가 있으면) 같은 이름의 CSV. 반환: 저장한 파일 목록."""
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    out = [path_no_ext + ".pdf", path_no_ext + ".png"]
    fig.savefig(out[0])
    fig.savefig(out[1], dpi=dpi)
    plt.close(fig)
    if data_rows is not None:
        out.append(write_csv(path_no_ext + ".csv", data_rows))
    return out
