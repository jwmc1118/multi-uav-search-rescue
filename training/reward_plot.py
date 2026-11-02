"""학습 그래프 (논문용) — result.json / progress(_prev).csv 와 평가 기록 (eval_latest.json, rule_eval.json) 으로 그림 6 개
(논문용 5 개 + 진단 1 개).

    fig_training_performance  (a) 고정 seed 평가 구조율 (평균 ± std, rule baseline, best) (b) 학습 에피소드 구조 · 만료 · 발견율
                              (c) 평균 에피소드 길이 (d) 평균 구조 시각 (발견 -> 구조 step)
                              (e) learned - rule 짝 비교 (같은 고정 seed 의 seed 별 구조율 차이 평균 · bootstrap 95 % CI)
                              (f) UAV 당 에피소드 역할 전환 수 (자발적 / 강제)
    fig_rewards               head 별 한 행, 항목별 한 패널 (UAV 당 에피소드 합): Role 합계 | 팀 항 | 늦지 않은 발견 | 전환 penalty /
                              Rescue 합계 | (n_suc - F_excess)/N_T | 보너스 | 중복 penalty / Explore 합계 | 직접 탐색 (배율 적용
                              후) | 정보 공유 | 직접 탐색 step 의 평균 배율 1 + w_omega ω^g
    fig_learning              (a) critic explained variance (value head 별) (b) 정책 entropy (c) iteration 당 결정 (학습 표본) 수
                              — 선 3 개: role / task–rescue / task–explore (entropy · 표본 수는 할당 표본의 역할별)
    fig_behavior              최근 평가 (고정 seed, greedy) 의 에피소드 진행률 10 % 구간 평균 (규칙 정책 = 같은 seed):
                              (a) 역할 비율 learned (b) 역할 비율 rule (c) 누적 발견 · 구조율 (d) 평균 그룹 연결도 ω^g
                              (e) 귀환 시작 시점의 남은 용량 분포 (f) iteration 별 에피소드 평균 역할 비율
    fig_undiscovered          최근 평가에서 끝까지 발견 못 한 생존자 위치 (여러 seed 누적, learned / rule 두 패널)
    fig_diagnostics           (진단, 논문용 아님) head 별 approx KL · clip 비율, 결정 시점 메모리 노드 수 · 평균 age
각 그림은 .png (dpi 200) + .pdf. 원시값은 얇고 연하게 (alpha 0.25), EMA (창 = env.yaml plot.ema_window) 는 진하게. 보조 축 없음,
범위 음영은 평가 ± std 만. 값이 없는 패널은 "no data". 규칙 정책 기록은 평가와 같은 고정 seed 로 한 번 계산해 rule_eval.json 에 캐시
(training/behavior.py rule_reference). train.py 가 --plot-every iteration 마다 save_figures 를 부른다.
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
    """fig_training_performance (a) 의 rule baseline = yaml eval.greedy_baseline (training/config/env.yaml). 없으면 None."""
    try:
        from training.config_loader import load_config
        return (load_config().get("eval") or {}).get("greedy_baseline")
    except Exception:
        return None



CM = "custom_metrics/"
HEAD_COLOR = {"role": "#1f77b4", "rescue": "#ff7f0e", "explore": "#2ca02c"}           # 파랑 / 주황 / 초록
LEARN_LINES = (("role", "role", HEAD_COLOR["role"]), ("task_rescue", "task–rescue", HEAD_COLOR["rescue"]),
               ("task_explore", "task–explore", HEAD_COLOR["explore"]))             # fig_learning 선 (stats 이름, 범례, 색)
POLICY_HEAD_COLOR = {"role": "#1f77b4", "task": "#9467bd"}                            # fig_diagnostics head 색
ROLE_COLOR = {"explore": "#2ca02c", "rescue": "#ff7f0e", "return": "#7f7f7f"}         # EXPLORE 초록 / RESCUE 주황 / RETURN 회색
OUT_COLOR = {"rescue": "#1f77b4", "expire": "#d62728", "found": "#2ca02c"}            # 구조 파랑 / 만료 빨강 / 발견 초록
SWITCH_COLOR = {"voluntary": "#9467bd", "forced": "#8c564b"}                      # 자발적 보라 / 강제 갈색
SRC_STYLE = {"learned": "-", "rule": "--"}                                            # 학습 모델 실선 / 규칙 정책 점선 (범례에 표기)
FIGS = ("fig_training_performance", "fig_rewards", "fig_learning", "fig_behavior", "fig_undiscovered", "fig_diagnostics")


def _style():
    matplotlib.rcParams.update({
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "legend.frameon": False, "axes.ymargin": 0.05, "pdf.fonttype": 42})


def ema(y, window):
    """NaN 을 건너뛰는 EMA (α = 2 / (window + 1))."""
    a = 2.0 / (max(1, int(window)) + 1.0)
    out, m = [], None
    for v in y:
        if v == v:
            m = v if m is None else a * v + (1 - a) * m
        out.append(m if m is not None else float("nan"))
    return out


def _has(y):
    return any(v == v for v in y)


def _no_data(ax, title, xlabel="training iteration", ylabel=""):
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="0.5")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot(ax, x, y, color, label, window, ls="-"):
    """원시값 (얇고 연하게) + EMA (진하게, 범례 이름). 반환: 그렸나."""
    pts = [(a, b) for a, b in zip(x, y) if b == b]
    if not pts:
        return False
    px, py = zip(*pts)
    ax.plot(px, py, color=color, lw=0.8, alpha=0.25, ls=ls)
    ax.plot(px, ema(py, window), color=color, lw=1.8, ls=ls, label=label,
            marker="o" if len(px) < 3 else None, ms=4)                 # 점이 1-2 개면 선이 안 보이므로 표시
    return True


def _panel(ax, x, rows, series, title, ylabel, window, legend=True):
    """series = [(열 이름, 범례, 색)] — 한 양의 선들. 없으면 no data."""
    drawn = [_plot(ax, x, _col(rows, key), color, label, window) for key, label, color in series]
    if not any(drawn):
        _no_data(ax, title, ylabel=ylabel)
        return False
    ax.set_title(title)
    ax.set_xlabel("training iteration")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best")
    return True


def _save(fig, logdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(logdir, name + ".png"), dpi=200)
    fig.savefig(os.path.join(logdir, name + ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- 평가 기록

def _env_config(logdir):
    try:
        with open(os.path.join(logdir, "params.json")) as f:
            return json.load(f).get("env_config")
    except (OSError, ValueError):
        return None


def _load_json(logdir, name):
    try:
        with open(os.path.join(logdir, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def rule_records(logdir, n_seeds):
    """같은 고정 seed (0 .. n_seeds - 1) 의 규칙 정책 에피소드 기록. 평가가 만든 rule_eval.json (seed 수가 같으면) 을 그대로
    쓰고, 없으면 params.json 의 env_config 로 한 번 계산해 캐시 (training/behavior.py rule_reference). 못 구하면 None."""
    from training.behavior import RULE_CACHE, rule_reference
    cached = _load_json(logdir, RULE_CACHE)
    if cached and int(cached.get("n_seeds", -1)) == int(n_seeds):
        return cached["episodes"]
    env = _env_config(logdir)
    if env is None:
        return None
    try:
        return rule_reference(env, n_seeds, logdir)
    except Exception as e:                                                  # 규칙 정책 기록 실패는 그 곡선 · 패널만 빠짐
        print(f"[plot] rule records skipped: {e}")
        return None


def _eval_latest(logdir):
    """최근 고정 seed 평가 기록 (fixed_seed_eval 이 쓴 eval_latest.json) -> (iteration, 에피소드 목록) / (None, None)."""
    from training.behavior import EVAL_LATEST
    d = _load_json(logdir, EVAL_LATEST)
    if not d or not d.get("episodes"):
        return None, None
    return d.get("iteration"), d["episodes"]


# ---------------------------------------------------------------- 그림 1

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


def _eval_points(rows, x, keys):
    """평가가 있었던 iteration 만 -> (x, {키: 값}) (값이 NaN 인 행은 뺀다)."""
    cols = {k: _col(rows, CM + k) for k in keys}
    idx = [i for i in range(len(x)) if cols[keys[0]][i] == cols[keys[0]][i]]
    return [x[i] for i in idx], {k: np.array([cols[k][i] for i in idx]) for k in keys}


def fig_training_performance(logdir, rows, x, baseline, window):
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
    a = ax[0][0]
    t_a = "Evaluation rescue rate (fixed seeds, greedy)"
    px, v = _eval_points(rows, x, ("eval/rescue_rate", "eval/rescue_rate_std"))
    if px:
        pr, ps = v["eval/rescue_rate"], np.nan_to_num(v["eval/rescue_rate_std"])
        a.plot(px, pr, color=OUT_COLOR["rescue"], lw=1.8, marker="o", ms=3, label="learned (mean)")
        a.fill_between(px, pr - ps, pr + ps, color=OUT_COLOR["rescue"], alpha=0.2, linewidth=0, label="learned ± std")
        if baseline is not None:
            a.axhline(float(baseline), color="0.3", lw=1.2, ls="--", label="rule baseline (same seeds)")
        b = _best(rows)
        if b:
            a.plot([b[0]], [b[1]], marker="*", ms=13, color="#d62728", ls="none", label=f"best (iter {b[0]})")
        a.set_title(t_a)
        a.set_xlabel("training iteration")
        a.set_ylabel("rescue rate (/N_T)")
        a.legend(loc="best")
    else:
        _no_data(a, t_a, ylabel="rescue rate (/N_T)")
    _panel(ax[0][1], x, rows, [(CM + "rescue_rate_mean", "rescued", OUT_COLOR["rescue"]),
                               (CM + "expire_rate_mean", "expired", OUT_COLOR["expire"]),
                               (CM + "discover_rate_mean", "found", OUT_COLOR["found"])],
           "Training episode outcome", "fraction of survivors (/N_T)", window)
    _panel(ax[0][2], x, rows, [("episode_len_mean", "episode length", "#000000")],
           "Mean episode length", "steps", window)
    _panel(ax[1][0], x, rows, [(CM + "known_to_rescue_mean", "discovery -> rescue", OUT_COLOR["rescue"])],
           "Mean rescue time of rescued survivors", "steps (discovery -> rescue)", window)
    a = ax[1][1]
    t_e = "Learned − rule rescue rate (paired by seed)"
    px, v = _eval_points(rows, x, ("eval/diff_mean", "eval/diff_lo", "eval/diff_hi"))
    if px:
        m = v["eval/diff_mean"]
        err = np.vstack([m - v["eval/diff_lo"], v["eval/diff_hi"] - m])
        a.errorbar(px, m, yerr=err, color=HEAD_COLOR["role"], lw=1.8, marker="o", ms=3, capsize=3,
                   label="mean difference (95 % bootstrap CI)")
        a.axhline(0.0, color="0.3", lw=1.2, ls="--", label="no difference")
        a.set_title(t_e)
        a.set_xlabel("training iteration")
        a.set_ylabel("rescue rate difference (/N_T)")
        a.legend(loc="best")
    else:
        _no_data(a, t_e, ylabel="rescue rate difference (/N_T)")
    _panel(ax[1][2], x, rows, [(CM + "switch_voluntary_mean", "voluntary", SWITCH_COLOR["voluntary"]),
                               (CM + "switch_forced_mean", "forced", SWITCH_COLOR["forced"])],
           "Role switches per UAV per episode", "switches per UAV per episode", window)
    _save(fig, logdir, "fig_training_performance")


# ---------------------------------------------------------------- 그림 2

REWARD_ROWS = (
    ("role", "Role", [("total", ("role_team", "role_discover", "role_switch")), ("(|ΔR| − w_x|ΔE|)/N_T", ("role_team",)),
                      ("timely discovery", ("role_discover",)), ("switch penalty", ("role_switch",))]),
    ("rescue", "Rescue", [("total", ("rescue_sf", "rescue_bonus", "rescue_dup")), ("(n_suc − F_excess)/N_T", ("rescue_sf",)),
                          ("bonus", ("rescue_bonus",)), ("duplicate penalty", ("rescue_dup",))]),
    ("explore", "Explore", [("total", ("explore_direct", "explore_share")),
                            ("direct (× multiplier)", ("explore_direct",)),
                            ("information sharing", ("explore_share",)),
                            ("multiplier 1 + w_ω ω^g", "explore_mult")]),
)                                       # 키가 문자열이면 보상 항목이 아닌 custom metric (그 패널은 단위 없는 배율)


def _sum_cols(rows, items):
    cols = [_col(rows, f"{CM}rw_{k}_mean") for k in items]
    out = []
    for vals in zip(*cols):
        out.append(sum(vals) if all(v == v for v in vals) else float("nan"))
    return out


def fig_rewards(logdir, rows, x, window):
    ncol = max(len(r[2]) for r in REWARD_ROWS)
    fig, ax = plt.subplots(len(REWARD_ROWS), ncol, figsize=(3.6 * ncol, 3.0 * len(REWARD_ROWS)), squeeze=False)
    for r, (head, name, items) in enumerate(REWARD_ROWS):
        for c in range(ncol):
            a = ax[r][c]
            if c >= len(items):
                a.axis("off")
                continue
            label, keys = items[c]
            metric = isinstance(keys, str)
            y = _col(rows, f"{CM}{keys}_mean") if metric else _sum_cols(rows, keys)
            ylabel = "multiplier (per direct-exploration step)" if metric else "reward per UAV per episode"
            title = f"{name}: {label}"
            if not _plot(a, x, y, HEAD_COLOR[head], label, window):
                _no_data(a, title, ylabel=ylabel)
                continue
            a.set_title(title)
            a.set_xlabel("training iteration")
            a.set_ylabel(ylabel)
            a.legend(loc="best")
    _save(fig, logdir, "fig_rewards")


# ---------------------------------------------------------------- 그림 3

def fig_learning(logdir, rows, x, window):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    _panel(ax[0], x, rows, [(_STATS + f"vf_explained_var_{k}", lb, c) for k, lb, c in LEARN_LINES],
           "Critic explained variance (per value head)", "explained variance", window)
    _panel(ax[1], x, rows, [(_STATS + f"entropy_{k}", lb, c) for k, lb, c in LEARN_LINES],
           "Policy entropy (per sample role)", "entropy (nats)", window)
    _panel(ax[2], x, rows, [(CM + f"decisions_per_batch_{k}", lb, c) for k, lb, c in LEARN_LINES],
           "Training samples per iteration", "decisions per train batch", window)
    _save(fig, logdir, "fig_learning")


def fig_diagnostics(logdir, rows, x, window):
    """진단 (논문용 아님): head 별 approx KL · clip 비율, 결정 시점 메모리 노드 수 · 평균 age."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    heads = tuple(POLICY_HEAD_COLOR)
    _panel(ax[0][0], x, rows, [(_STATS + f"approx_kl_{h}", h, POLICY_HEAD_COLOR[h]) for h in heads],
           "Approx. KL (old → new policy)", "approx KL (nats)", window)
    _panel(ax[0][1], x, rows, [(_STATS + f"clip_frac_{h}", h, POLICY_HEAD_COLOR[h]) for h in heads],
           "PPO clip fraction", "fraction of samples clipped", window)
    _panel(ax[1][0], x, rows, [(CM + "mem_nodes_mean", "memory nodes", "#7f7f7f")],
           "Memory nodes per decision", "nodes per decision graph", window)
    _panel(ax[1][1], x, rows, [(CM + "mem_age_mean", "memory age", "#7f7f7f")],
           "Mean memory-node age at decision", "age (t − t_last) / T_age", window)
    _save(fig, logdir, "fig_diagnostics")


# ---------------------------------------------------------------- 그림 4

PROG_LABEL = "episode progress (%)"


def _role_stack(a, prog, curves, title):
    if not curves:
        _no_data(a, title, PROG_LABEL, "share of UAVs")
        return
    a.stackplot(prog, *[curves[f"role_{r}"] for r in ROLE_COLOR], colors=list(ROLE_COLOR.values()),
                labels=[r.upper() for r in ROLE_COLOR], alpha=0.85, linewidth=0)
    a.set_ylim(0, 1)
    a.set_xlim(0, 100)
    a.set_title(title)
    a.set_xlabel(PROG_LABEL)
    a.set_ylabel("share of UAVs")
    a.legend(loc="lower left")


def _cap_hist(recs):
    """귀환 시작 시점 남은 용량 분포 (seed 합, 비율) -> (용량 0 .. c_max 비율 배열, 귀환 수) / None."""
    hs = [r.get("return_start_cap_hist") or [] for r in recs or []]
    width = max((len(h) for h in hs), default=0)
    if width == 0:
        return None
    tot = np.zeros(width)
    for h in hs:
        tot[:len(h)] += h
    return (tot / tot.sum() if tot.sum() > 0 else tot), int(tot.sum())


def fig_behavior(logdir, rows, x, window, eval_seeds):
    from training.behavior import N_BINS, mean_curves
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
    prog = np.arange(N_BINS) * (100 / N_BINS) + 50 / N_BINS                # 구간 중앙 (%)
    it, learned = _eval_latest(logdir)
    n_seeds = len(learned) if learned else eval_seeds
    rule = rule_records(logdir, n_seeds)
    cur = mean_curves([r["curves"] for r in learned]) if learned else None
    rc = mean_curves([r["curves"] for r in rule]) if rule else None
    sub = f" — eval iter {it}" if it is not None else ""
    _role_stack(ax[0][0], prog, cur, "(a) Role share over episode (learned)" + sub)
    _role_stack(ax[0][1], prog, rc, f"(b) Role share over episode (rule, {n_seeds} seeds)")
    a = ax[0][2]
    t_c = "(c) Cumulative found / rescued over episode"
    drawn = False
    for src, data in (("learned", cur), ("rule", rc)):
        if data:
            ls = SRC_STYLE[src]
            a.plot(prog, data["found"], color=OUT_COLOR["found"], lw=1.8, ls=ls, marker="o", ms=3, label=f"found ({src})")
            a.plot(prog, data["rescued"], color=OUT_COLOR["rescue"], lw=1.8, ls=ls, marker="o", ms=3, label=f"rescued ({src})")
            drawn = True
    if drawn:
        a.set_title(t_c)
        a.set_xlabel(PROG_LABEL)
        a.set_ylabel("fraction of survivors (/N_T)")
        a.legend(loc="best")
    else:
        _no_data(a, t_c, PROG_LABEL, "fraction of survivors (/N_T)")
    a = ax[1][0]
    t_d = "(d) Mean group connectivity over episode"
    drawn = False
    for src, data, color in (("learned", cur, HEAD_COLOR["role"]), ("rule", rc, "0.3")):
        if data and _has(data["omega"]):
            a.plot(prog, data["omega"], color=color, lw=1.8, ls=SRC_STYLE[src], marker="o", ms=3, label=src)
            drawn = True
    if drawn:
        a.set_ylim(0, 1)
        a.set_title(t_d)
        a.set_xlabel(PROG_LABEL)
        a.set_ylabel("group connectivity ω^g")
        a.legend(loc="best")
    else:
        _no_data(a, t_d, PROG_LABEL, "group connectivity ω^g")
    a = ax[1][1]
    t_e = "(e) Remaining capacity at RETURN start"
    hl, hr = _cap_hist(learned), _cap_hist(rule)
    if hl or hr:
        width = max(len(h[0]) for h in (hl, hr) if h)
        caps = np.arange(width)
        for k, (src, h, color) in enumerate((("learned", hl, HEAD_COLOR["role"]), ("rule", hr, "0.55"))):
            if h:
                a.bar(caps + (k - 0.5) * 0.4, np.pad(h[0], (0, width - len(h[0]))), width=0.4, color=color,
                      label=f"{src} ({h[1]} returns)")
        a.set_xticks(caps)
        a.set_title(t_e)
        a.set_xlabel("remaining capacity (supplies)")
        a.set_ylabel("share of RETURN starts")
        a.legend(loc="best")
    else:
        _no_data(a, t_e, "remaining capacity (supplies)", "share of RETURN starts")
    a = ax[1][2]
    t_f = "(f) Role share during training (episode mean)"
    ys = {r: _col(rows, f"{CM}role_{r}_mean") for r in ROLE_COLOR}
    pts = [i for i in range(len(x)) if all(ys[r][i] == ys[r][i] for r in ROLE_COLOR)]
    if pts:
        xs = [x[i] for i in pts]
        a.stackplot(xs, *[ema([ys[r][i] for i in pts], window) for r in ROLE_COLOR], colors=list(ROLE_COLOR.values()),
                    labels=[r.upper() for r in ROLE_COLOR], alpha=0.85, linewidth=0)
        a.set_ylim(0, 1)
        a.set_title(t_f + f" (EMA {window})")
        a.set_xlabel("training iteration")
        a.set_ylabel("share of UAV-steps")
        a.legend(loc="lower left")
    else:
        _no_data(a, t_f, ylabel="share of UAV-steps")
    _save(fig, logdir, "fig_behavior")


# ---------------------------------------------------------------- 그림 5

def _undiscovered_panel(a, recs, title):
    """recs 의 미발견 생존자 (빨간 점, seed 누적) 를 지도 위에 — 장애물 = seed 별 장애물 비율 (회색, 고정 맵이면 0 / 1), depot = 검은 사각."""
    if not recs:
        _no_data(a, title, "x (cell)", "y (cell)")
        return
    obs = np.mean([np.array([[c == "1" for c in row] for row in r["obstacles"]], float) for r in recs], 0)
    a.imshow(obs, cmap="Greys", vmin=0, vmax=1.5, origin="upper", interpolation="nearest")
    depots = sorted({tuple(r["depot"]) for r in recs})
    a.plot([d[1] for d in depots], [d[0] for d in depots], ls="none", marker="s", ms=6, color="black", label="depot")
    pts = [p for r in recs for p in r["undiscovered"]]
    if pts:
        a.plot([p[1] for p in pts], [p[0] for p in pts], ls="none", marker="o", ms=4, color=OUT_COLOR["expire"], alpha=0.7,
               label="undiscovered survivor")
    a.set_title(f"{title}: {len(pts)} undiscovered ({len(recs)} seeds)")
    a.set_xlabel("x (cell)")
    a.set_ylabel("y (cell)")
    a.grid(False)
    a.legend(loc="upper right")


def fig_undiscovered(logdir, eval_seeds):
    it, learned = _eval_latest(logdir)
    rule = rule_records(logdir, len(learned) if learned else eval_seeds)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    _undiscovered_panel(ax[0], learned, "Learned" + (f" (eval iter {it})" if it is not None else ""))
    _undiscovered_panel(ax[1], rule, "Rule")
    _save(fig, logdir, "fig_undiscovered")


def save_figures(logdir, baseline=None, ema_window=10, eval_seeds=20):
    """그림 6 개 (FIGS, .png + .pdf) 를 logdir 에 저장. 기록이 없으면 아무것도 안 함. 반환: 저장한 그림 이름.
    fig_training_performance (a) 의 rule baseline = 같은 고정 seed 규칙 정책 구조율 평균 (rule_eval.json), 없으면 baseline
    (yaml eval.greedy_baseline)."""
    rows = read_progress(logdir)
    if not rows:
        return []
    _style()
    x = _col(rows, "training_iteration")
    _, learned = _eval_latest(logdir)
    rule = rule_records(logdir, len(learned) if learned else eval_seeds)
    if rule:
        baseline = float(np.mean([r["rescue_rate"] for r in rule]))
    elif baseline is None:
        baseline = greedy_baseline()
    fig_training_performance(logdir, rows, x, baseline, ema_window)
    fig_rewards(logdir, rows, x, ema_window)
    fig_learning(logdir, rows, x, ema_window)
    fig_behavior(logdir, rows, x, ema_window, eval_seeds)
    fig_undiscovered(logdir, eval_seeds)
    fig_diagnostics(logdir, rows, x, ema_window)
    return list(FIGS)
