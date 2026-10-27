"""학습 곡선(reward_curve.png)을 progress.csv 로부터 그린다.

train.py 가 --plot-every iteration 마다 호출한다. 매번 progress(_prev).csv 전체를 다시 읽으므로
--restore 로 이어서 학습해도 앞부분 곡선이 그대로 이어진다.
TensorBoard 없이 이미지 하나로 학습이 잘 되는지 바로 확인하는 용도.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")   # 화면 없이 파일로만 그린다
import matplotlib.pyplot as plt

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


def _lines(ax, x, rows, series, title, band=False, hline=None):
    """series: [(열 이름, 범례)]. band=True 면 열 이름 + _mean / _min / _max."""
    has = False
    for key, label in series:
        y = _col(rows, key + ("_mean" if band else ""))
        if all(v != v for v in y):
            continue
        has = True
        line, = ax.plot(x, y, label=label)
        if band:
            ax.fill_between(x, _col(rows, key + "_min"), _col(rows, key + "_max"), alpha=0.15, color=line.get_color())
    if hline is not None:
        ax.axhline(hline, color="0.5", lw=0.8)
    if has:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)


def greedy_baseline():
    """Eval 그래프 그리디 기준선 = yaml eval.greedy_baseline (training/config/env.yaml, run_rules 15 seed 구조율). 없으면 None."""
    try:
        from training.config_loader import load_config
        return (load_config().get("eval") or {}).get("greedy_baseline")
    except Exception:
        return None


def save_role_plot(logdir, rows, filename, baseline=None):
    """계층 구조 (Role / Rescue / Explore) 학습 곡선: 보상 / 임무 지표 / 헤드별 critic · 정책 / 역할 비율 · 이벤트 /
    explore 2 단계 (1·2 단계 결정 수, 평균 m, 쪼개짐 · 합쳐짐, 통신 이탈, 소진까지 step, 역할 전환) / 목표 변경 · 진동 / 부분 귀환 /
    explore 옵션 (커버리지 return · 길이) · 고른 덩어리 거리 비 · 새 관측 셀 / RESCUE 재결정 원인 / rescue S - F / ψ^f 진행."""
    import numpy as np
    x = [t / 1e6 for t in _col(rows, "timesteps_total")]
    heads = ("role", "rescue", "explore")
    fig, axes = plt.subplots(8, 4, figsize=(22, 32))
    cm = "custom_metrics/"
    _lines(axes[0][0], x, rows, [("episode_reward", "total")], "Episode reward (sum of heads, UAVs)", band=True)
    parts = {"role": [("rw_role_team_mean", "team (rescued - w_x expired)"), ("rw_role_osc_mean", "oscillation penalty")],
             "rescue": [("rw_rescue_sf_mean", "(S - F) / N_T"), ("rw_rescue_bonus_mean", "bonus (F = 0)"),
                        ("rw_rescue_dup_mean", "duplicate")],
             "explore": [("rw_explore_cov_mean", "coverage (1 - psi_f) dcov / |free|")]}
    for ax, h in zip(axes[0][1:], heads):
        _lines(ax, x, rows, [(cm + f"reward_{h}", h)] + [(cm + k, lab) for k, lab in parts[h]],
               f"{h} reward (per UAV) and parts", band=False)
    _lines(axes[1][0], x, rows, [(cm + "rescue_rate", "rescued"), (cm + "expire_rate", "expired"),
                                 (cm + "discover_rate", "discovered"), (cm + "explored", "explored")],
           "Mission (/ N_T)", band=True)
    _lines(axes[1][1], x, rows, [("episode_len_mean", "episode length")], "Episode length")
    _lines(axes[1][2], x, rows, [(cm + "same_cell_mean", "same cell"), (cm + "conflict_cross_mean", "cross-group conflict steps"),
                                 (cm + "dup_in_mean", "dup penalty (in group)"), (cm + "dup_out_mean", "dup penalty (other group)")],
           "Overlap / conflict / duplicate pursuit")
    _lines(axes[1][3], x, rows, [(_STATS + f"grad_norm_enc_from_{k}", k) for k in ("role", "exp", "res")],
           "Encoder grad norm per head network (dashed = EMA)")
    for line_ in list(axes[1][3].get_lines()):
        k = line_.get_label()
        y = _col(rows, cm + f"enc_grad_ema_{k}")
        if not all(v != v for v in y):
            axes[1][3].plot(x, y, ls="--", color=line_.get_color(), lw=1.2)
    r = _col(rows, cm + "enc_grad_ratio")
    if not all(v != v for v in r):
        tw = axes[1][3].twinx()
        tw.plot(x, r, color="0.4", lw=0.8, ls=":")
        tw.axhline(10, color="0.6", lw=0.6)
        tw.set_ylabel("EMA max/min ratio (dotted)", fontsize=8)
    _lines(axes[2][0], x, rows, [(_STATS + f"vf_explained_var_{h}", h) for h in heads], "Critic explained variance", hline=0.0)
    _lines(axes[2][1], x, rows, [(_STATS + f"vf_loss_{h}", h) for h in heads], "Critic value loss")
    _lines(axes[2][2], x, rows, [(_STATS + f"entropy_{h}", h) for h in heads] + [(_STATS + f"kl_{h}", f"kl {h}") for h in heads],
           "Entropy / KL")
    _lines(axes[2][3], x, rows, [(cm + f"decisions_per_batch_{h}", h) for h in heads], "Decisions per train batch", hline=200)
    # 역할 비율 (마지막 iteration, 에피소드 진행 10 구간)
    ax = axes[3][0]
    last = rows[-1]
    fr = np.array([[float(last.get(f"{cm}role_{r}_bin{b}_mean", "nan") or "nan") for b in range(10)]
                   for r in ("explore", "rescue", "return")])
    if np.isfinite(fr).all():
        bottom = np.zeros(10)
        for j, (r, c) in enumerate(zip(("explore", "rescue", "return"), ("#56B4E9", "#E69F00", "#999999"))):
            ax.bar(np.arange(10), fr[j], bottom=bottom, color=c, label=r)
            bottom += fr[j]
        ax.set_xticks(range(10)); ax.set_xticklabels([f"{b * 10}" for b in range(10)], fontsize=7)
        ax.legend(fontsize=8)
    ax.set_title("Role ratio by episode progress (%) — latest iter")
    _lines(axes[3][1], x, rows, [(cm + "rescue/S_mean", "S (assigned in time)"), (cm + "rescue/F_mean", "F (urgent missed)"),
                                 (cm + "rescue/U_eps_size_mean", "|U_eps| (urgent)"),
                                 (cm + "rescue/F_unavoidable_mean", "F unavoidable max(0, |U_eps| - |G_res|)"),
                                 (cm + "rescue/bonus_rate_mean", "bonus rate (F = 0)"),
                                 (cm + "ev2_rescue_by_claim_mean", "ev2 RESCUE by claim holder"),
                                 (cm + "ev2_rescue_other_mean", "ev2 RESCUE of non-claimed victim"),
                                 (cm + "s1_claim_kept_mean", "stage 1 kept claimed blob"),
                                 (cm + "claim_not_rescue_mean", "claim holder not RESCUE"),
                                 (cm + "ev2_redecide_mean", "ev2 re-deciders / event"), (cm + "ev2_rescue_mean", "ev2 RESCUE / event")],
           "Rescue S - F per event / decision order claims")
    tw3 = axes[3][1].twinx()
    y = _col(rows, cm + "known_to_rescue_mean")
    if not all(v != v for v in y):
        tw3.plot(x, y, color="0.4", ls=":")
        tw3.set_ylabel("discovery -> rescue steps (dotted)", fontsize=8)
    ax = axes[3][2]
    labels = [str(k) for k in range(1, 6)] + ["split", "merge"]
    ev = [float(last.get(f"{cm}ev_{k}_mean", "nan") or "nan") for k in labels]
    ax.bar(labels, ev, color=["#0072B2"] * 5 + ["#999999"] * 2)
    ax.set_title("Role events 1-5 (+ blob split / merge, no re-decision) per episode — latest iter")
    _lines(axes[3][3], x, rows, [(_STATS + f"value_target_std_{h}", h) for h in heads], "Value target std")
    # ---- explore 2 단계 ----
    _lines(axes[4][0], x, rows, [(cm + "explore_s1_mean", "stage 1 (blob, policy) / episode"),
                                 (cm + "explore_s2_mean", "stage 2 Hungarian assignments / episode"),
                                 (cm + "explore_reselect_mean", "join re-assignments (Hungarian) / episode"),
                                 (cm + "decisions_per_batch_explore", "explore decisions / train batch")], "Explore decisions")
    _lines(axes[4][1], x, rows, [(cm + "link_broken_mean", "link breaks"), (cm + "link_made_mean", "link makes"),
                                 (cm + "group_size_ep_mean", "mean group size"), (cm + "dec_interval_mean", "decision interval (steps)"),
                                 (cm + "latch_hold_mean", "goal latch holds")],
           "Comm links / group size / decision interval")
    _lines(axes[4][2], x, rows, [(cm + "goal_change_mean", "goal changes"), (cm + "oscillation_mean", "goal oscillations (A-B-A)"),
                                 (cm + "role_osc_mean", "role oscillations (penalized)"), (cm + "role_forced_mean", "forced role switches"),
                                 (cm + "role_voluntary_mean", "voluntary role switches")],
           "Goal / role change and oscillation per episode")
    _lines(axes[4][3], x, rows, [(cm + "partial_return_mean", "partial returns"),
                                 (cm + "rescued_after_partial_mean", "rescued until next refill"),
                                 (cm + "seg_life_mean", "steps until segment exhausted")], "Partial return / segment lifetime")
    # ---- explore ratio / 할당 지표 ----
    _lines(axes[5][0], x, rows, [(cm + "explore/cov_own_return_mean", "own coverage return per option (undiscounted)"),
                                 (_STATS + "value_target_mean_explore", "learner discounted R mean")],
           "Explore options: own coverage return")
    tw50 = axes[5][0].twinx()
    y = _col(rows, cm + "explore/option_len_mean")
    if not all(v != v for v in y):
        tw50.plot(x, y, color="0.3", ls=":")
        tw50.set_ylabel("option length t1 - t0 (dotted)", fontsize=8)
    _lines(axes[5][1], x, rows, [(cm + "blob_dist_ratio_mean", "(chosen blob d + 1) / (nearest blob d + 1)"),
                                 (cm + "reselect_kept_goal_mean", "join re-assignment: kept own-goal segment")],
           "Explore blob choice / join reselect", hline=1.0)
    ax5 = axes[5][1].twinx()
    for k, lab, c in (("join_events_mean", "join events / ep", "0.3"), ("reselect_goal_change_mean", "goal changes by reselect / ep", "0.6")):
        y = _col(rows, cm + k)
        if not all(v != v for v in y):
            ax5.plot(x, y, color=c, ls=":", label=lab)
    if ax5.get_lines():
        ax5.legend(fontsize=7, loc="lower right")
    _lines(axes[5][2], x, rows, [(cm + "new_cells_per_explore_step_mean", "new cells / EXPLORE step"),
                                 (cm + "s1_empty_blobs_mean", "unassigned blobs at stage-1 event"),
                                 (cm + "s1_participants_mean", "stage-1 participants / event"),
                                 (cm + "ev2_retained_blob_changed_mean", "kept EXPLORE but blob changed (0)"),
                                 (cm + "preempt_candidates_mean", "preempt candidates / episode")],
           "Explore coverage / preempt")
    ax = axes[5][3]
    rr = np.nan_to_num([float(last.get(f"{cm}res_redecide_{k}_mean", "nan") or "nan") for k in range(1, 6)])
    ax.bar(np.arange(5), rr, color=["#999999", "#999999", "#E69F00", "#999999", "#E69F00"])
    ax.set_xticks(range(5)); ax.set_xticklabels([f"ev {k}" for k in range(1, 6)])
    ax.set_title("RESCUE re-decision causes / episode (3, 5 only)")
    # ---- 연결성 C / 그룹 / GNN ----
    _lines(axes[6][0], x, rows, [(cm + "conn/real_mean_mean", "real C_t pair ratio (diagnostic)"),
                                 (cm + "rescue/c_bar_mean", "rescue: mean personal c over option"),
                                 (cm + "explore/conn_mult_mean_mean", "explore: multiplier (1 + w_mult c) on new cells")],
           "Connectivity: personal c in rewards / team pair ratio")
    _lines(axes[6][1], x, rows, [(cm + "reward/role_rescue_term_mean", "role: (|R| - w_x|X|)/N_T"),
                                 (cm + "reward/rescue_SF_term_mean", "rescue: (S - F)/N_T"),
                                 (cm + "reward/rescue_bonus_term_mean", "rescue: bonus"),
                                 (cm + "reward/explore_term_mean", "explore: coverage (team)")],
           "Reward terms per episode")
    _lines(axes[6][2], x, rows, [(cm + "n_groups_mean", "comm groups"), (cm + "group_size_ep_mean", "group size"),
                                 (cm + "dup_out_mean", "cross-group dup"),
                                 (cm + "ev2_rescue_nearest_mean", "ev2 RESCUE chooser nearest (ratio)"),
                                 (cm + "obs/survivor_truncated_mean", "survivor slots truncated (decisions / ep)")],
           "Groups / cross-group dup / ev2 nearest chooser")
    _lines(axes[6][3], x, rows, [(cm + "gnn_ms_per_decision_mean", "GNN ms / decision"),
                                 (cm + "gnn_ms_per_call_mean", "GNN ms / call")], "GNN forward time (rollout)")
    tw6 = axes[6][3].twinx()
    y = _col(rows, cm + "gnn_params")
    if not all(v != v for v in y):
        tw6.plot(x, y, color="0.4", ls=":")
        tw6.set_ylabel("GNN params (dotted)", fontsize=8)
    # ---- ψ^f 진행 / exploration complete 마스크 ----
    ax = axes[7][0]
    pf = [float(last.get(f"{cm}psi_f_bin{b}_mean", "nan") or "nan") for b in range(10)]
    ax.plot(np.arange(10) * 10 + 5, pf, marker="o")
    ax.set_ylim(0, 1.05)
    ax.set_title("psi_f (team discovered / N_T) by episode progress (%) — latest iter")
    _lines(axes[7][1], x, rows, [(cm + "role/explore_masked_decisions_mean", "role decisions with EXPLORE masked"),
                                 (cm + "role/forced_return_all_masked_mean", "forced RETURN (all masked)"),
                                 (cm + "ev5_exploration_complete_mean", "event 5: exploration complete")],
           "Exploration complete: masks / events per episode")
    # ---- 고정 seed 평가 (eval.every iteration 마다, 세 head argmax) / 학습률 ----
    ax = axes[7][2]
    er, es = _col(rows, cm + "eval/rescue_rate"), _col(rows, cm + "eval/rescue_rate_std")
    pts = [(xx, r, sd) for xx, r, sd in zip(x, er, es) if r == r]
    if pts:
        px, pr, ps = zip(*pts)
        ax.errorbar(px, pr, yerr=[v if v == v else 0.0 for v in ps], marker="o", capsize=3, label="eval rescue rate (mean ± std)")
        el = [(xx, v) for xx, v in zip(x, _col(rows, cm + "eval/episode_len")) if v == v]
        tw72 = ax.twinx()
        tw72.plot([a for a, _ in el], [b for _, b in el], color="0.4", ls=":", marker=".")
        tw72.set_ylabel("eval episode length (dotted)", fontsize=8)
    baseline = greedy_baseline() if baseline is None else baseline
    if baseline is not None:
        ax.axhline(float(baseline), color="r", ls="--", lw=1, label=f"greedy baseline {float(baseline):.3f}")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Eval (fixed seeds, greedy actions)")
    _lines(axes[7][3], x, rows, [(cm + "learner/cur_lr", "learning rate")], "Learning rate (lr_schedule)")
    for ax in list(axes.flat):
        if ax not in (axes[3][0], axes[3][2], axes[5][3], axes[7][0]):
            ax.set_xlabel("timesteps (M)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{os.path.basename(logdir)} — iter {rows[-1].get('training_iteration')}, {x[-1]:.2f}M steps")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(logdir, filename), dpi=100)
    plt.close(fig)


def save_reward_plot(logdir, filename="reward_curve.png", baseline=None):
    rows = read_progress(logdir)
    if rows:
        save_role_plot(logdir, rows, filename, baseline)
