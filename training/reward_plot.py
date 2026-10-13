"""학습 곡선(reward_curve.png)을 progress.csv 로부터 그린다.

train.py 가 --plot-every iteration 마다 호출한다. 매번 progress(_prev).csv 전체를 다시 읽으므로
--restore 로 이어서 학습해도 앞부분 곡선이 그대로 이어진다.
TensorBoard 없이 이미지 하나로 학습이 잘 되는지 바로 확인하는 용도.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")   # 화면 없이 파일로만 그린다
import matplotlib.pyplot as plt

_STATS = "info/learner/default_policy/learner_stats/"
# (그래프 제목, progress.csv 열 이름 앞부분 — 뒤에 _mean / _min / _max)
_REWARD_PANELS = [
    ("Episode reward (total: alloc + primitive, sum over UAVs)", "episode_reward"),
    ("Allocation reward (per UAV)", "custom_metrics/reward_alloc"),
    ("Primitive reward (per UAV)", "custom_metrics/reward_primitive"),
]
# 임무 성능 (그래프 제목, [(열 이름 앞부분, 범례)]) — 보상 계수와 상관없는 지표
_MISSION_PANELS = [
    ("Targets (/ N_T)", [("custom_metrics/rescue_rate", "rescued"), ("custom_metrics/expire_rate", "expired"),
                         ("custom_metrics/discover_rate", "discovered")]),
    ("Explored ratio (episode end)", [("custom_metrics/explored", "explored")]),
    ("UAV collisions per episode", [("custom_metrics/collisions", "collisions")]),
]
# (그래프 제목, learner_stats 키 목록)
_LEARNER_PANELS = [
    ("Loss", ["policy_loss", "vf_loss"]),
    ("Entropy / KL / VF explained var", ["entropy", "kl", "vf_explained_var"]),
]


def _col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "")))
        except ValueError:
            out.append(float("nan"))
    return out


PREV_PROGRESS = "progress_prev.csv"   # 이어서 학습하기 전 기록 (train.py 가 옮겨 둠)


def read_progress(logdir):
    """progress_prev.csv + progress.csv 를 이어 붙인 행 목록 (열 구성이 달라도 됨)."""
    rows = []
    for fn in (PREV_PROGRESS, "progress.csv"):
        path = os.path.join(logdir, fn)
        if os.path.isfile(path):
            with open(path, newline="") as f:
                rows += list(csv.DictReader(f))
    return rows


def save_reward_plot(logdir, filename="reward_curve.png"):
    rows = read_progress(logdir)
    if not rows:
        return

    x = [t / 1e6 for t in _col(rows, "timesteps_total")]
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))

    # 윗줄: 보상. 전체 = env 스칼라(할당 + primitive, UAV 합), 나머지 = 각 헤드가 최대화하는 보상 (UAV 평균)
    for ax, (title, prefix) in zip(axes[0], _REWARD_PANELS):
        mean = _col(rows, prefix + "_mean")
        if all(v != v for v in mean):   # 열이 없음 (metrics 콜백 추가 전 기록)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.fill_between(x, _col(rows, prefix + "_min"), _col(rows, prefix + "_max"),
                            alpha=0.2, label="min-max")
            ax.plot(x, mean, label="mean")
            ax.legend()
        ax.set_title(title)

    # 가운데 줄: 임무 성능
    for ax, (title, series) in zip(axes[1], _MISSION_PANELS):
        has = False
        for prefix, label in series:
            mean = _col(rows, prefix + "_mean")
            if all(v != v for v in mean):
                continue
            has = True
            line, = ax.plot(x, mean, label=label)
            ax.fill_between(x, _col(rows, prefix + "_min"), _col(rows, prefix + "_max"),
                            alpha=0.15, color=line.get_color())
        if has:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)

    # 아랫줄: 에피소드 길이 + 학습 통계
    ax = axes[2][0]
    ax.plot(x, _col(rows, "episode_len_mean"))
    ax.set_title("Episode length (mean)")

    for ax, (title, keys) in zip(axes[2][1:], _LEARNER_PANELS):
        for k in keys:
            ax.plot(x, _col(rows, _STATS + k), label=k)
        ax.set_title(title)
        ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("timesteps (M)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{os.path.basename(logdir)} — iter {rows[-1].get('training_iteration')}, {x[-1]:.2f}M steps")
    fig.tight_layout(rect=(0, 0, 1, 0.97))   # 위 제목과 겹치지 않게
    fig.savefig(os.path.join(logdir, filename), dpi=100)
    plt.close(fig)
