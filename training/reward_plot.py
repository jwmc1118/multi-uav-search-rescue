"""학습 곡선을 result.json / progress(_prev).csv 로부터 그린다 (x 축 = training iteration).

    reward_curve.png   메인 3 x 3 (train.py --plot-every 마다): 고정 seed 평가 구조율 · 학습 rollout 결과 · 에피소드 길이 ·
                       role / rescue / explore 보상 항목 · head 별 엔트로피 + KL · critic explained variance · 역할 비율
    diagnostics.png    나머지 진단 패널 (train.py --diag-every 마다, 기본 50): 학습 안정성 / 통신 / explore / rescue / 이벤트 행
매번 기록 전체를 다시 읽으므로 --resume 으로 이어서 학습해도 앞부분 곡선이 그대로 이어진다. 원시 값은 얇고 흐리게,
EMA 평활선 (smooth = α, train.py --plot-smooth, 0 = 끔) 을 굵게. 값이 없는 패널 (첫 iteration, rule head, 보상 수정 전 기록) 은
"no data" 로 표시한다.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")   # 화면 없이 파일로만 그린다
import matplotlib.pyplot as plt
import numpy as np

_STATS = "info/learner/default_policy/learner_stats/"




def _col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "")))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


PREV_PROGRESS = "progress_prev.csv"   # 이어서 학습하기 전 기록 (train.py 가 옮겨 둠)


RESULT_JSON = "result.json"


def _flatten(d, prefix=""):
    """중첩 result dict -> 'a/b/c' 키의 평탄 dict (숫자만, hist_stats 같은 리스트는 뺀다)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def read_progress(logdir):
    """progress_prev.csv (이어서 학습하기 전 기록) + 이번 기록을 이어 붙인 행 목록 (열 구성이 달라도 됨).
    이번 기록은 result.json (매 iteration 전체 결과) 을 우선 읽는다 — RLlib 의 progress.csv 는 첫 iteration 의 키로 열이
    고정돼, 첫 iteration 에 끝난 에피소드가 없으면 에피소드 custom_metrics 열이 끝까지 빠진다. result.json 이 없으면
    progress.csv. 같은 training_iteration 이 여러 번 있으면 마지막 것."""
    rows = []
    prev = os.path.join(logdir, PREV_PROGRESS)
    if os.path.isfile(prev):
        with open(prev, newline="") as f:
            rows += list(csv.DictReader(f))
    js = os.path.join(logdir, RESULT_JSON)
    if os.path.isfile(js):
        with open(js) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(_flatten(json.loads(line)))
                    except ValueError:
                        continue                                # 쓰는 중인 마지막 줄
    elif os.path.isfile(os.path.join(logdir, "progress.csv")):
        with open(os.path.join(logdir, "progress.csv"), newline="") as f:
            rows += list(csv.DictReader(f))
    by_it = {}
    for r in rows:
        try:
            by_it[int(float(r["training_iteration"]))] = r
        except (KeyError, TypeError, ValueError):
            continue
    return [by_it[k] for k in sorted(by_it)]


def greedy_baseline():
    """Eval 그래프 그리디 기준선 = yaml eval.greedy_baseline (training/config/env.yaml, run_rules 15 seed 구조율). 없으면 None."""
    try:
        from training.config_loader import load_config
        return (load_config().get("eval") or {}).get("greedy_baseline")
    except Exception:
        return None



CM = "custom_metrics/"
HEADS = ("role", "rescue", "explore")
HEAD_COLOR = {"role": "#0072B2", "rescue": "#E69F00", "explore": "#009E73"}       # 모든 패널 같은 색 (Okabe–Ito)
C = {"rescue": "#0072B2", "expire": "#D55E00", "discover": "#009E73", "explored": "#56B4E9", "len": "#000000",
     "team": "#0072B2", "discover_shaping": "#009E73", "osc": "#CC79A7", "sf": "#E69F00", "bonus": "#0072B2",
     "dup": "#D55E00", "f_excess": "#555555", "f_min": "#999999", "explore": "#009E73", "cells": "#555555",
     "lr": "#555555", "gray": "#999999"}
ROLE_COLOR = {"explore": "#56B4E9", "rescue": "#E69F00", "return": "#999999"}
SHAPING_TOL = 1e-6


def _style():
    matplotlib.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                                "grid.alpha": 0.3, "grid.linewidth": 0.5, "legend.fontsize": 7, "legend.framealpha": 0.6,
                                "axes.titlesize": 10, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})


def ema(y, alpha):
    """NaN 을 건너뛰는 EMA (alpha 0 / None = 그대로)."""
    if not alpha:
        return list(y)
    out, m = [], None
    for v in y:
        if v == v:
            m = v if m is None else alpha * v + (1 - alpha) * m
        out.append(m if m is not None else float("nan"))
    return out


def _has(y):
    return any(v == v for v in y)


def _plot(ax, x, y, color, label, smooth, ls="-", band=None):
    """원시 값 (얇고 흐리게) + EMA (굵게). band = (lo, hi) 면 음영."""
    if smooth:
        ax.plot(x, y, color=color, lw=0.6, alpha=0.35, ls=ls)
        ax.plot(x, ema(y, smooth), color=color, lw=1.6, ls=ls, label=label)
    else:
        ax.plot(x, y, color=color, lw=1.3, ls=ls, label=label)
    if band is not None:
        ax.fill_between(x, band[0], band[1], color=color, alpha=0.12, linewidth=0)


def _no_data(ax):
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")


def _lines(ax, x, rows, series, title, smooth=0.0, band=False, hline=None, legend="best"):
    """series: [(열 이름, 범례, 색 (None = 자동), 선 모양)] — band 면 열 + _mean / _min / _max. 반환: 그린 선이 있나."""
    has = False
    for k, item in enumerate(series):
        key, label = item[0], item[1]
        color = item[2] if len(item) > 2 and item[2] else f"C{k}"
        ls = item[3] if len(item) > 3 else "-"
        y = _col(rows, key + ("_mean" if band else ""))
        if not _has(y):
            continue
        has = True
        b = (_col(rows, key + "_min"), _col(rows, key + "_max")) if band else None
        _plot(ax, x, y, color, label, smooth, ls, b)
    if hline is not None:
        ax.axhline(hline, color="0.5", lw=0.8)
    if has and legend:
        ax.legend(loc=legend)
    if not has:
        _no_data(ax)
    ax.set_title(title)
    return has


def _twin(ax, x, rows, key, label, color, smooth):
    y = _col(rows, key)
    if not _has(y):
        return None
    tw = ax.twinx()
    tw.spines["right"].set_visible(True)
    tw.grid(False)
    _plot(tw, x, y, color, label, smooth, ls=":")
    tw.set_ylabel(label, fontsize=7, color=color)
    return tw


def _best(rows):
    """고정 seed 평가 best (구조율 최대, 같으면 에피소드 길이 짧은 쪽) -> (iteration, 구조율) / None."""
    best = None
    for r in rows:
        v = _col([r], CM + "eval/rescue_rate")[0]
        if v != v:
            continue
        ln = _col([r], CM + "eval/episode_len")[0]
        key = (v, -(ln if ln == ln else 1e9))
        if best is None or key > best[0]:
            best = (key, int(float(r["training_iteration"])), v)
    return None if best is None else (best[1], best[2])


def _summary_line(logdir, rows):
    last = rows[-1]
    it = last.get("training_iteration")
    el = _col([last], "time_total_s")[0]
    lr = _col([last], CM + "learner/cur_lr")[0]
    if lr != lr:
        lr = _col([last], _STATS + "cur_lr")[0]
    modes = {h: last.get(f"config/env_config/policy_mode/{h}") for h in HEADS}
    if not any(modes.values()):
        try:
            with open(os.path.join(logdir, "params.json")) as f:
                pm = (json.load(f).get("env_config") or {}).get("policy_mode") or {}
            modes = {h: pm.get(h) for h in HEADS}
        except (OSError, ValueError):
            pass
    el_s = f"{int(el) // 3600}h {int(el) % 3600 // 60:02d}m" if el == el else "?"
    lr_s = f"{lr:.2e}" if lr == lr else "?"
    steps = _col([last], "timesteps_total")[0]
    st = f", {steps / 1e6:.2f}M steps" if steps == steps else ""
    return (f"{os.path.basename(os.path.normpath(logdir))} — iter {it}{st}, elapsed {el_s}, lr {lr_s}, policy_mode "
            + " / ".join(f"{h} {modes.get(h) or '?'}" for h in HEADS))


def save_main_plot(logdir, rows, filename="reward_curve.png", baseline=None, smooth=0.1):
    """메인 3 x 3."""
    _style()
    x = _col(rows, "training_iteration")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    a = axes.flat
    # 1 Eval rescue rate
    ax = next(a)
    er, es = _col(rows, CM + "eval/rescue_rate"), _col(rows, CM + "eval/rescue_rate_std")
    pts = [(xx, r, sd) for xx, r, sd in zip(x, er, es) if r == r]
    title = "Eval rescue rate (fixed seeds, greedy)"
    if pts:
        px, pr, psd = zip(*pts)
        ax.errorbar(px, pr, yerr=[v if v == v else 0.0 for v in psd], color=C["rescue"], marker="o", ms=3, lw=1.2,
                    capsize=2, label="eval (mean ± std)")
        b = _best(rows)
        if b:
            ax.plot([b[0]], [b[1]], marker="*", ms=14, color="#D55E00", ls="none", label="best")
            title += f" — best {b[1]:.3f} @ iter {b[0]}"
    baseline = greedy_baseline() if baseline is None else baseline
    if baseline is not None:
        ax.axhline(float(baseline), color="0.3", ls="--", lw=1, label=f"rule baseline {float(baseline):.3f}")
    if pts or baseline is not None:
        ax.legend(loc="lower right")
    if not pts:
        _no_data(ax)
    ax.set_title(title)
    # 2 Train outcome
    _lines(next(a), x, rows, [(CM + "rescue_rate", "rescue", C["rescue"]), (CM + "expire_rate", "expire", C["expire"]),
                              (CM + "discover_rate", "discover", C["discover"])],
           "Train outcome (/ N_T)", smooth, band=True)
    # 3 Episode length
    _lines(next(a), x, rows, [("episode_len_mean", "episode length", C["len"])], "Episode length", smooth)
    # 4 Role reward components
    ax = next(a)
    _lines(ax, x, rows, [(CM + "reward/role_rescue_term_mean", "team (|R| - w_x|X|)/N_T", C["team"]),
                         (CM + "reward/role_discover_term_mean", "discover shaping", C["discover_shaping"]),
                         (CM + "rw_role_osc_mean", "oscillation penalty", C["osc"])],
           "Role reward / episode", smooth, hline=0.0)
    mx = _col(rows, CM + "shaping/disc_sum_abs_max")
    if _has(mx) and max(v for v in mx if v == v) > SHAPING_TOL:
        ax.text(0.02, 0.95, f"shaping discounted sum |Σγ^t F_t| max {max(v for v in mx if v == v):.1e} > {SHAPING_TOL:g}",
                transform=ax.transAxes, color="red", fontsize=8, va="top")
    # 5 Rescue reward components
    ax = next(a)
    _lines(ax, x, rows, [(CM + "reward/rescue_SF_term_mean", "(S - F_excess)/N_T", C["sf"]),
                         (CM + "reward/rescue_bonus_term_mean", "bonus", C["bonus"]),
                         (CM + "rw_rescue_dup_mean", "duplicate penalty", C["dup"])],
           "Rescue reward / episode", smooth, hline=0.0, legend="upper left")
    tw = _twin(ax, x, rows, CM + "rescue/F_excess_mean", "F_excess / event (dotted)", C["f_excess"], smooth)
    if tw is not None:
        y = _col(rows, CM + "rescue/F_min_mean")
        if _has(y):
            tw.plot(x, ema(y, smooth), color=C["f_min"], ls="--", lw=1.0)
            tw.set_ylabel("F_excess (dotted) / F_min (dashed) per event", fontsize=7)
    # 6 Explore
    ax = next(a)
    _lines(ax, x, rows, [(CM + "reward/explore_term_mean", "explore reward (team)", C["explore"])],
           "Explore reward / episode", smooth, legend="upper left")
    _twin(ax, x, rows, CM + "new_cells_per_explore_step_mean", "new cells / EXPLORE step (dotted)", C["cells"], smooth)
    # 7 Entropy + KL
    ax = next(a)
    has = _lines(ax, x, rows, [(_STATS + f"entropy_{h}", f"entropy {h}", HEAD_COLOR[h]) for h in HEADS],
                 "Policy entropy (solid) / KL (dotted)", smooth, legend=None)
    for h in HEADS:
        y = _col(rows, _STATS + f"kl_{h}")
        if _has(y):
            ax.plot(x, ema(y, smooth), color=HEAD_COLOR[h], ls=":", lw=1.2, label=f"KL {h}")
            has = True
    if has:
        ax.legend(loc="best", ncol=2)
    # 8 Critic EV
    _lines(next(a), x, rows, [(_STATS + f"vf_explained_var_{h}", h, HEAD_COLOR[h]) for h in HEADS],
           "Critic explained variance", smooth, hline=0.0)
    # 9 Role ratio (에피소드 평균 = 진행 10 구간 평균)
    ax = next(a)
    fr = {r: [] for r in ROLE_COLOR}
    for row in rows:
        for r in ROLE_COLOR:
            v = [_col([row], f"{CM}role_{r}_bin{b}_mean")[0] for b in range(10)]
            v = [u for u in v if u == u]
            fr[r].append(float(np.mean(v)) if v else float("nan"))
    if all(_has(fr[r]) for r in ROLE_COLOR):
        ys = [np.nan_to_num(np.array(fr[r])) for r in ROLE_COLOR]
        ax.stackplot(x, *ys, colors=[ROLE_COLOR[r] for r in ROLE_COLOR], labels=list(ROLE_COLOR), alpha=0.85, linewidth=0)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", ncol=3)
    else:
        _no_data(ax)
    ax.set_title("Role ratio (episode mean)")
    for ax in axes[-1]:
        ax.set_xlabel("training iteration")
    fig.suptitle(_summary_line(logdir, rows), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(logdir, filename), dpi=100)
    plt.close(fig)


def _bar_latest(ax, rows, keys, labels, colors, title):
    last = rows[-1]
    v = [_col([last], k)[0] for k in keys]
    if not any(u == u for u in v):
        _no_data(ax)
    else:
        ax.bar(np.arange(len(v)), np.nan_to_num(v), color=colors)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title)


def save_diagnostics_plot(logdir, rows, filename="diagnostics.png", smooth=0.1):
    """메인에 없는 진단 패널 — 행: 학습 안정성 / 통신 / explore / rescue / 이벤트."""
    _style()
    x = _col(rows, "training_iteration")
    st = _STATS
    layout = {
        "stability": 7, "comm": 4, "explore": 6, "rescue": 5, "events": 4}
    ncol = max(layout.values())
    fig, axes = plt.subplots(len(layout), ncol, figsize=(4.0 * ncol, 3.4 * len(layout)))
    row = {k: list(axes[i]) for i, k in enumerate(layout)}
    for k, n in layout.items():
        for ax in row[k][n:]:
            ax.axis("off")
    L = lambda ax, series, title, **kw: _lines(ax, x, rows, series, title, smooth, **kw)
    # ---- 학습 안정성 ----
    r = row["stability"]
    L(r[0], [("episode_reward", "total", C["len"])], "Episode reward (sum of heads)", band=True)
    L(r[1], [(CM + f"reward_{h}_mean", h, HEAD_COLOR[h]) for h in HEADS], "Reward per head (per UAV)")
    ok = L(r[2], [(st + f"grad_norm_enc_from_{k}", h, HEAD_COLOR[h]) for k, h in (("role", "role"), ("res", "rescue"), ("exp", "explore"))],
           "Encoder grad norm (dashed = EMA)")
    for k, h in (("role", "role"), ("res", "rescue"), ("exp", "explore")):
        y = _col(rows, CM + f"enc_grad_ema_{k}")
        if _has(y):
            r[2].plot(x, y, ls="--", color=HEAD_COLOR[h], lw=1.0)
    L(r[3], [(st + f"vf_loss_{h}", h, HEAD_COLOR[h]) for h in HEADS], "Critic value loss")
    L(r[4], [(st + f"value_target_std_{h}", h, HEAD_COLOR[h]) for h in HEADS], "Value target std")
    L(r[5], [(CM + f"decisions_per_batch_{h}", h, HEAD_COLOR[h]) for h in HEADS], "Decisions per train batch", hline=200)
    L(r[6], [(CM + "learner/cur_lr", "learning rate", C["lr"])], "Learning rate")
    # ---- 통신 ----
    r = row["comm"]
    L(r[0], [(CM + "link_broken_mean", "link breaks"), (CM + "link_made_mean", "link makes"),
             (CM + "group_size_ep_mean", "mean group size"), (CM + "dec_interval_mean", "decision interval"),
             (CM + "latch_hold_mean", "goal latch holds")], "Comm links / group size")
    L(r[1], [(CM + "n_groups_mean", "comm groups"), (CM + "group_size_ep_mean", "group size"),
             (CM + "dup_out_mean", "cross-group dup"), (CM + "ev2_rescue_nearest_mean", "ev2 RESCUE chooser nearest"),
             (CM + "obs/survivor_truncated_mean", "S-node overflow (decisions)")], "Groups / cross-group")
    L(r[2], [(CM + "conn/real_mean_mean", "real C_t pair ratio"), (CM + "rescue/c_bar_mean", "rescue c̄ over option"),
             (CM + "explore/conn_mult_mean_mean", "explore multiplier (1 + w_mult c)")], "Connectivity")
    L(r[3], [(CM + "same_cell_mean", "same cell"), (CM + "conflict_cross_mean", "cross-group conflict"),
             (CM + "dup_in_mean", "dup (in group)"), (CM + "dup_out_mean", "dup (other group)")], "Overlap / conflict / duplicate")
    # ---- explore ----
    r = row["explore"]
    L(r[0], [(CM + "explore_s1_mean", "stage 1 / ep"), (CM + "explore_s2_mean", "stage 2 Hungarian / ep"),
             (CM + "explore_reselect_mean", "join re-assignments / ep")], "Explore decisions")
    L(r[1], [(CM + "explore/cov_own_return_mean", "coverage return / option"),
             (st + "value_target_mean_explore", "learner discounted R")], "Explore options")
    _twin(r[1], x, rows, CM + "explore/option_len_mean", "option length (dotted)", C["gray"], smooth)
    L(r[2], [(CM + "blob_dist_ratio_mean", "(chosen d + 1) / (nearest d + 1)"),
             (CM + "reselect_kept_goal_mean", "reselect kept own goal")], "Blob choice / join reselect", hline=1.0)
    L(r[3], [(CM + "new_cells_per_explore_step_mean", "new cells / EXPLORE step"),
             (CM + "s1_empty_blobs_mean", "unassigned blobs at stage 1"), (CM + "s1_participants_mean", "stage-1 participants"),
             (CM + "ev2_retained_blob_changed_mean", "kept EXPLORE but blob changed"),
             (CM + "preempt_candidates_mean", "preempt candidates")], "Explore coverage / preempt")
    ax = r[4]
    last = rows[-1]
    pf = [_col([last], f"{CM}psi_f_bin{b}_mean")[0] for b in range(10)]
    if _has(pf):
        ax.plot(np.arange(10) * 10 + 5, pf, marker="o", color=C["discover"])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("episode progress (%)")
    else:
        _no_data(ax)
    ax.set_title("psi_f by episode progress (latest)")
    L(r[5], [(CM + "role/explore_masked_decisions_mean", "EXPLORE masked decisions"),
             (CM + "role/forced_return_all_masked_mean", "forced RETURN"),
             (CM + "ev5_exploration_complete_mean", "ev5 exploration complete")], "Exploration complete")
    # ---- rescue ----
    r = row["rescue"]
    L(r[0], [(CM + "rescue/S_mean", "S"), (CM + "rescue/F_mean", "F"), (CM + "rescue/F_min_mean", "F_min"),
             (CM + "rescue/F_excess_mean", "F_excess"), (CM + "rescue/U_eps_size_mean", "|U_eps|"),
             (CM + "rescue/F_unavoidable_mean", "max(0, |U_eps| - |G|)"), (CM + "rescue/bonus_rate_mean", "bonus rate")],
      "Rescue S / F per event")
    _twin(r[0], x, rows, CM + "known_to_rescue_mean", "discovery -> rescue steps (dotted)", C["gray"], smooth)
    L(r[1], [(CM + "ev2_rescue_by_claim_mean", "ev2 RESCUE by claim holder"), (CM + "ev2_rescue_other_mean", "ev2 non-claimed"),
             (CM + "s1_claim_kept_mean", "stage 1 kept claim"), (CM + "claim_not_rescue_mean", "claim holder not RESCUE"),
             (CM + "ev2_redecide_mean", "ev2 re-deciders / event"), (CM + "ev2_rescue_mean", "ev2 RESCUE / event")],
      "Decision-order claims")
    _bar_latest(r[2], rows, [f"{CM}res_redecide_{k}_mean" for k in range(1, 6)], [f"ev {k}" for k in range(1, 6)],
                ["#999999", "#999999", "#E69F00", "#999999", "#E69F00"], "RESCUE re-decision causes (latest, 3 / 5 only)")
    L(r[3], [(CM + "partial_return_mean", "partial returns"), (CM + "rescued_after_partial_mean", "rescued until refill"),
             (CM + "seg_life_mean", "segment lifetime")], "Partial return / segment lifetime")
    L(r[4], [(CM + "shaping/discovered_mean", "discovered in steps"), (CM + "shaping/discovered_feasible_mean", "feasible at discovery"),
             (CM + "reward/role_discover_term_mean", "discover shaping / ep")], "Discover shaping")
    # ---- 이벤트 / 역할 ----
    r = row["events"]
    _bar_latest(r[0], rows, [f"{CM}ev_{k}_mean" for k in ("1", "2", "3", "4", "5", "split", "merge")],
                ["1", "2", "3", "4", "5", "split", "merge"], ["#0072B2"] * 5 + ["#999999"] * 2, "Role events / episode (latest)")
    L(r[1], [(CM + "goal_change_mean", "goal changes"), (CM + "oscillation_mean", "goal oscillations"),
             (CM + "role_osc_mean", "role oscillations"), (CM + "role_forced_mean", "forced switches"),
             (CM + "role_voluntary_mean", "voluntary switches")], "Goal / role changes")
    ax = r[2]
    fr = np.array([[_col([last], f"{CM}role_{q}_bin{b}_mean")[0] for b in range(10)] for q in ROLE_COLOR])
    if np.isfinite(fr).all():
        bottom = np.zeros(10)
        for j, q in enumerate(ROLE_COLOR):
            ax.bar(np.arange(10), fr[j], bottom=bottom, color=ROLE_COLOR[q], label=q)
            bottom += fr[j]
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"{b * 10}" for b in range(10)], fontsize=7)
        ax.legend(loc="lower left", ncol=3)
    else:
        _no_data(ax)
    ax.set_title("Role ratio by episode progress (%) — latest")
    L(r[3], [(CM + "gnn_ms_per_decision_mean", "GNN ms / decision"), (CM + "gnn_ms_per_call_mean", "GNN ms / call")],
      "GNN forward time (rollout)")
    fig.suptitle("Diagnostics — " + _summary_line(logdir, rows), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(logdir, filename), dpi=80)
    plt.close(fig)


def save_reward_plot(logdir, filename="reward_curve.png", baseline=None, smooth=0.1):
    rows = read_progress(logdir)
    if rows:
        save_main_plot(logdir, rows, filename, baseline, smooth)


def save_diagnostics(logdir, filename="diagnostics.png", smooth=0.1):
    rows = read_progress(logdir)
    if rows:
        save_diagnostics_plot(logdir, rows, filename, smooth)
