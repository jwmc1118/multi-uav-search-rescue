"""학습 그래프 (논문용) — result.json / progress(_prev).csv 와 학습 모델의 평가 기록 (eval_latest.json) 으로 그림 6 개
(논문용 5 개 + 진단 1 개). 기준선 정책은 실행하지 않는다 — 기준선 구조율은 env.yaml eval.baselines 의 숫자
(조합 이름: rule_rule_rule / learn_rule_rule / learn_learn_learn, null 은 표시 안 함, 색 · 선 스타일 = training/behavior.py BASELINES).

    fig_training_performance  (a) 고정 seed 평가 구조율 (평균 ± std, 기준선 수평선, best) (b) 학습 에피소드 구조 · 만료 · 발견율
                              (c) 평균 에피소드 길이 (d) 평균 구조 시각 (발견 -> 구조 step)
                              (e) 평가 구조율 − 각 기준선 (평가 iteration 마다, 기준선이 모두 null 이면 숨김)
                              (f) UAV 당 에피소드 역할 전환 수 (자발적 / 강제)
    fig_rewards               항목별 한 패널 (UAV 당 에피소드 합, 패널마다 y 범위 독립): Role 합계 | 팀 항 | 늦지 않은 발견 |
                              전환 penalty / Rescue 합계 | 대기 | 발견 후 만료 | 할당 충돌 | 목표 전환 / Explore 합계 | 미발견 |
                              목표 전환 (rule 인 구성요소 행은 제목에 (rule))
    fig_learning              (a) critic explained variance (V^role / V^res / V^exp) (b) 정책 entropy (c) iteration 당 결정
                              (학습 표본) 수 — 네트워크별 role / rescue / explore 세 줄 (learn 이 아닌 구성요소 줄은 숨김, run_meta.json)
    fig_behavior              최근 평가 (고정 seed, greedy, 학습 모델) 의 에피소드 진행률 10 % 구간 평균:
                              (a) 역할 비율 (b) 누적 발견 · 구조율 (c) 평균 그룹 연결도 ω^g
                              (d) 귀환 시작 시점의 남은 용량 분포 (e) iteration 별 에피소드 평균 역할 비율
    fig_undiscovered          최근 평가에서 끝까지 발견 못 한 생존자 위치 (여러 seed 누적, 학습 모델)
    fig_diagnostics           (진단, 논문용 아님) 네트워크별 approx KL · clip 비율, 결정 시점 메모리 노드 수 · 평균 age
각 그림은 .png (dpi 200) + .pdf. 원시값은 얇고 연하게 (alpha 0.25), EMA (창 = env.yaml plot.ema_window) 는 진하게. 보조 축 없음,
범위 음영은 평가 ± std 만. 값이 없는 패널은 "no data". train.py 가 --plot-every iteration 마다 save_figures 를 부른다.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")   # 화면 없이 파일로만 그린다
import matplotlib.pyplot as plt
import numpy as np

from env.role.simulation import EXPLORE_ITEMS, RESCUE_ITEMS  # noqa: E402  (fig_rewards Rescue / Explore 합계 항목)

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


CM = "custom_metrics/"
HEAD_COLOR = {"role": "#1f77b4", "rescue": "#ff7f0e", "explore": "#2ca02c"}   # 네트워크 색: 파랑 / 주황 / 초록
NET_LABEL = {"role": "role", "rescue": "rescue", "explore": "explore"}
ROLE_COLOR = {"explore": "#2ca02c", "rescue": "#ff7f0e", "return": "#7f7f7f"}         # EXPLORE 초록 / RESCUE 주황 / RETURN 회색
OUT_COLOR = {"rescue": "#1f77b4", "expire": "#d62728", "found": "#2ca02c"}            # 구조 파랑 / 만료 빨강 / 발견 초록
SWITCH_COLOR = {"voluntary": "#9467bd", "forced": "#8c564b"}                      # 자발적 보라 / 강제 갈색
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

def _load_json(logdir, name):
    try:
        with open(os.path.join(logdir, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
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


def _baselines(baselines):
    """config eval.baselines -> [(키, 값)] (null 제외, BASELINES 순서)."""
    from training.behavior import BASELINES
    b = baselines or {}
    return [(k, float(b[k])) for k in BASELINES if b.get(k) is not None]


def _baseline_diff_panel(a, rows, x, baselines):
    """(e) 평가 iteration 마다 학습 평가 평균 − 각 기준선 (config 숫자), 0 점선. CI 없음."""
    from training.behavior import BASELINES
    title, ylabel = "Learned − baselines (evaluation rescue rate)", "rescue rate difference (/N_T)"
    px, v = _eval_points(rows, x, ("eval/rescue_rate",))
    if not px:
        _no_data(a, title, ylabel=ylabel)
        return
    for k, b in baselines:
        name, color, ls = BASELINES[k]
        a.plot(px, v["eval/rescue_rate"] - b, color=color, lw=1.8, ls=ls, marker="o", ms=3, label=f"learned − {name}")
    a.axhline(0.0, color="0.5", lw=1.0, ls=":")
    a.set_title(title)
    a.set_xlabel("training iteration")
    a.set_ylabel(ylabel)
    a.legend(loc="best")


def fig_training_performance(logdir, rows, x, baselines, window):
    from training.behavior import BASELINES
    base = _baselines(baselines)
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
    a = ax[0][0]
    t_a = "Evaluation rescue rate (fixed seeds, greedy)"
    px, v = _eval_points(rows, x, ("eval/rescue_rate", "eval/rescue_rate_std"))
    if px:
        pr, ps = v["eval/rescue_rate"], np.nan_to_num(v["eval/rescue_rate_std"])
        a.plot(px, pr, color=OUT_COLOR["rescue"], lw=1.8, marker="o", ms=3, label="learned (mean)")
        a.fill_between(px, pr - ps, pr + ps, color=OUT_COLOR["rescue"], alpha=0.2, linewidth=0, label="learned ± std")
        for k, b in base:
            name, color, ls = BASELINES[k]
            a.axhline(b, color=color, lw=1.2, ls=ls, label=f"{name} ({b:.3f})")
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
    if base:
        _baseline_diff_panel(ax[1][1], rows, x, base)
    else:
        ax[1][1].axis("off")
    _panel(ax[1][2], x, rows, [(CM + "switch_voluntary_mean", "voluntary", SWITCH_COLOR["voluntary"]),
                               (CM + "switch_forced_mean", "forced", SWITCH_COLOR["forced"])],
           "Role switches per UAV per episode", "switches per UAV per episode", window)
    _save(fig, logdir, "fig_training_performance")


# ---------------------------------------------------------------- 그림 2

REWARD_ROWS = (
    ("role", "Role", [("total", ("role_team", "role_discover", "role_switch")), ("(|ΔR| − w_x|ΔE|)/N_T", ("role_team",)),
                      ("timely discovery", ("role_discover",)), ("switch penalty", ("role_switch",))]),
    ("rescue", "Rescue", [("total", RESCUE_ITEMS), ("wait", ("task_wait",)), ("found expiry", ("task_found_expire",)),
                          ("assignment conflict", ("task_conflict",)), ("target switch", ("task_switch_rescue",))]),
    ("explore", "Explore", [("total", EXPLORE_ITEMS), ("undiscovered", ("task_undiscovered",)),
                            ("target switch", ("task_switch_explore",))]),
)
MODE_TAG = {"rule": " (rule)", "learn": ""}                                   # 학습하지 않는 구성요소 행 제목 꼬리


def _sum_cols(rows, items):
    cols = [_col(rows, f"{CM}rw_{k}_mean") for k in items]
    out = []
    for vals in zip(*cols):
        out.append(sum(vals) if all(v == v for v in vals) else float("nan"))
    return out


def fig_rewards(logdir, rows, x, window):
    """Role / Rescue / Explore 행, 보상 항목별 패널. rule 인 구성요소 행은 제목에 (rule) (값은 로그용으로 유지)."""
    modes = _modes(logdir)
    ncol = max(len(r[2]) for r in REWARD_ROWS)
    fig, ax = plt.subplots(len(REWARD_ROWS), ncol, figsize=(3.6 * ncol, 3.0 * len(REWARD_ROWS)), squeeze=False)
    for r, (head, name, items) in enumerate(REWARD_ROWS):
        for c in range(ncol):
            a = ax[r][c]
            if c >= len(items):
                a.axis("off")
                continue
            label, keys = items[c]
            y = _sum_cols(rows, keys)
            ylabel = "reward per UAV per episode"
            title = f"{name}{MODE_TAG[modes[head]]}: {label}"
            if not _plot(a, x, y, HEAD_COLOR[head], label, window):
                _no_data(a, title, ylabel=ylabel)
                continue
            a.set_title(title)
            a.set_xlabel("training iteration")
            a.set_ylabel(ylabel)
            a.legend(loc="best")
    _save(fig, logdir, "fig_rewards")


# ---------------------------------------------------------------- 그림 3

def _modes(logdir):
    """run 의 구성요소 모드 {role, rescue, explore: rule | learn} (run_meta.json -> 폴더 이름 -> params.json)."""
    from eval_common.model_loader import components_from_meta
    comp = components_from_meta(_load_json(logdir, "run_meta.json"), logdir, _load_json(logdir, "params.json"))
    return {k: v["mode"] for k, v in comp.items()}


def _nets(logdir):
    """그릴 네트워크 (role / rescue / explore) — learn 인 구성요소만 (rule 은 학습 지표가 0 이라 뺀다)."""
    modes = _modes(logdir)
    return [k for k in HEAD_COLOR if modes[k] == "learn"]


def _net_lines(nets, key):
    return [(key.format(k), NET_LABEL[k], HEAD_COLOR[k]) for k in nets]


def fig_learning(logdir, rows, x, window):
    """(a) critic EV (V^role / V^res / V^exp) (b) 정책 entropy (c) train batch 당 학습 결정 수 — 네트워크별 (rule 인 네트워크 제외)."""
    nets = _nets(logdir)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    _panel(ax[0], x, rows, _net_lines(nets, _STATS + "vf_explained_var_{}"),
           "Critic explained variance (per network)", "explained variance", window)
    _panel(ax[1], x, rows, _net_lines(nets, _STATS + "entropy_{}"), "Policy entropy (per network)", "entropy (nats)", window)
    _panel(ax[2], x, rows, _net_lines(nets, CM + "decisions_per_batch_{}"),
           "Training samples per iteration", "decisions per train batch", window)
    _save(fig, logdir, "fig_learning")


def fig_diagnostics(logdir, rows, x, window):
    """진단 (논문용 아님): 네트워크별 approx KL · clip 비율 (rule 인 네트워크 제외), 결정 시점 메모리 노드 수 · 평균 age."""
    nets = _nets(logdir)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    _panel(ax[0][0], x, rows, _net_lines(nets, _STATS + "approx_kl_{}"), "Approx. KL (old → new policy)", "approx KL (nats)",
           window)
    _panel(ax[0][1], x, rows, _net_lines(nets, _STATS + "clip_frac_{}"), "PPO clip fraction", "fraction of samples clipped",
           window)
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


def fig_behavior(logdir, rows, x, window):
    from training.behavior import N_BINS, mean_curves
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
    ax[1][2].axis("off")
    prog = np.arange(N_BINS) * (100 / N_BINS) + 50 / N_BINS                # 구간 중앙 (%)
    it, learned = _eval_latest(logdir)
    cur = mean_curves([r["curves"] for r in learned]) if learned else None
    sub = f" — eval iter {it}, {len(learned)} seeds" if learned and it is not None else ""
    _role_stack(ax[0][0], prog, cur, "(a) Role share over episode" + sub)
    a = ax[0][1]
    t_b = "(b) Cumulative found / rescued over episode"
    if cur:
        a.plot(prog, cur["found"], color=OUT_COLOR["found"], lw=1.8, marker="o", ms=3, label="found")
        a.plot(prog, cur["rescued"], color=OUT_COLOR["rescue"], lw=1.8, marker="o", ms=3, label="rescued")
        a.set_title(t_b)
        a.set_xlabel(PROG_LABEL)
        a.set_ylabel("fraction of survivors (/N_T)")
        a.legend(loc="best")
    else:
        _no_data(a, t_b, PROG_LABEL, "fraction of survivors (/N_T)")
    a = ax[0][2]
    t_c = "(c) Mean group connectivity over episode"
    if cur and _has(cur["omega"]):
        a.plot(prog, cur["omega"], color=HEAD_COLOR["role"], lw=1.8, marker="o", ms=3)
        a.set_ylim(0, 1)
        a.set_title(t_c)
        a.set_xlabel(PROG_LABEL)
        a.set_ylabel("group connectivity ω^g")
    else:
        _no_data(a, t_c, PROG_LABEL, "group connectivity ω^g")
    a = ax[1][0]
    t_d = "(d) Remaining capacity at RETURN start"
    h = _cap_hist(learned)
    if h:
        caps = np.arange(len(h[0]))
        a.bar(caps, h[0], width=0.6, color=HEAD_COLOR["role"], label=f"{h[1]} returns")
        a.set_xticks(caps)
        a.set_title(t_d)
        a.set_xlabel("remaining capacity (supplies)")
        a.set_ylabel("share of RETURN starts")
        a.legend(loc="best")
    else:
        _no_data(a, t_d, "remaining capacity (supplies)", "share of RETURN starts")
    a = ax[1][1]
    t_e = "(e) Role share during training (episode mean)"
    ys = {r: _col(rows, f"{CM}role_{r}_mean") for r in ROLE_COLOR}
    pts = [i for i in range(len(x)) if all(ys[r][i] == ys[r][i] for r in ROLE_COLOR)]
    if pts:
        xs = [x[i] for i in pts]
        a.stackplot(xs, *[ema([ys[r][i] for i in pts], window) for r in ROLE_COLOR], colors=list(ROLE_COLOR.values()),
                    labels=[r.upper() for r in ROLE_COLOR], alpha=0.85, linewidth=0)
        a.set_ylim(0, 1)
        a.set_title(t_e + f" (EMA {window})")
        a.set_xlabel("training iteration")
        a.set_ylabel("share of UAV-steps")
        a.legend(loc="lower left")
    else:
        _no_data(a, t_e, ylabel="share of UAV-steps")
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


def fig_undiscovered(logdir):
    it, learned = _eval_latest(logdir)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.2))
    _undiscovered_panel(ax, learned, "Learned" + (f" (eval iter {it})" if it is not None else ""))
    _save(fig, logdir, "fig_undiscovered")


def save_figures(logdir, baselines=None, ema_window=10):
    """그림 6 개 (FIGS, .png + .pdf) 를 logdir 에 저장. 기록이 없으면 아무것도 안 함. 반환: 저장한 그림 이름.
    baselines = env.yaml eval.baselines ({rule_rule_rule, learn_rule_rule, learn_learn_learn: 구조율 또는 None})."""
    rows = read_progress(logdir)
    if not rows:
        return []
    _style()
    x = _col(rows, "training_iteration")
    fig_training_performance(logdir, rows, x, baselines, ema_window)
    fig_rewards(logdir, rows, x, ema_window)
    fig_learning(logdir, rows, x, ema_window)
    fig_behavior(logdir, rows, x, ema_window)
    fig_undiscovered(logdir)
    fig_diagnostics(logdir, rows, x, ema_window)
    return list(FIGS)
