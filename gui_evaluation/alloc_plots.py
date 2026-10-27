"""임무 할당 분석 그래프 (논문용) — gui_evaluation/alloc_recorder.py 의 snapshot() 을 받아 그림 + 원자료를 저장한다.

    out_dir = save_alloc_plots(recorder.snapshot(sim), figures_root)

저장 (out_dir = figures_root/YYYYMMDD_HHMMSS_<policy>_t<최종 step>/):
    gantt          UAV 역할 Gantt (행 = UAV, 모드 띠 + 태스크 id, ★ 구조 / ✕ 배정 중 만료 / ▲ 재보충)
    targets_timeline  조난자 타임라인 (행 = 조난자, 마감 순: 미발견 → 발견·미배정 → 배정 UAV 색, 중복 배정 빗금)
    trajectories   궤적 지도 (UAV 경로, 시간이 갈수록 진하게 / 조난자 최종 상태)
    cumulative     발견 / 구조 / 만료 누적 (/ N_T) + 팀 탐색률
    capacity       UAV 별 구호품 잔량 (계단형)
    snapshots      균등 간격 K 시점의 할당 (UAV → 목표 화살표)
    각각 .png (300 dpi) + .pdf (벡터),  steps.csv / events.csv / targets.csv / team.csv / meta.json
UAV 색은 gui/colors.py (GUI 와 같은 색).
"""
import csv
import datetime
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from gui.colors import agent_hex, hex_to_rgb, lighten

STYLE = {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 300, "pdf.fonttype": 42}
C_RESCUED, C_EXPIRED, C_UNDISC, C_WAIT = "#2CA02C", "#D62728", "#BBBBBB", "#F2D24B"


# ---------------------------------------------------------------- 공통

def _rgb01(h):
    return tuple(v / 255 for v in hex_to_rgb(h))


def _light(h, amount):
    return tuple(v / 255 for v in lighten(hex_to_rgb(h), amount))


def _save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _by_agent(steps, n):
    rows = [[] for _ in range(n)]
    for s in steps:
        rows[s["agent"]].append(s)
    for r in rows:
        r.sort(key=lambda s: s["t"])
    return rows


def segments(agent_rows):
    """연속으로 같은 (mode, target) 인 tick 묶음 -> [(t0, t1, mode, target)]. tick t 는 구간 [t-1, t]."""
    out = []
    for s in agent_rows:
        if s["t"] == 0:
            continue
        key = (s["mode"], s["target"])
        if out and out[-1][2:] == key and out[-1][1] == s["t"] - 1:
            out[-1] = (out[-1][0], s["t"], *key)
        else:
            out.append((s["t"] - 1, s["t"], *key))
    return out


def check_consistency(data):
    """기록 일관성: Gantt rescue 구간 수 = 배정 이벤트 수, 구조 ★ 수 = 실제 구조 수, 조난자 상태 = 시뮬레이션 최종 상태."""
    n = data["meta"]["n_agents"]
    rows = _by_agent(data["steps"], n)
    n_rescue_seg = sum(1 for r in rows for seg in segments(r) if seg[2] == "RESCUE" and seg[3] >= 0)
    n_assign = sum(1 for e in data["events"] if e["type"] == "assign" and e["target"] >= 0)
    assert n_rescue_seg == n_assign, f"rescue 구간 {n_rescue_seg} != 배정 이벤트 {n_assign}"
    n_star = sum(1 for e in data["events"] if e["type"] == "rescue")
    n_rescued = sum(1 for t in data["targets"] if t["status"] == "rescued")
    assert n_star == n_rescued, f"구조 ★ {n_star} != 실제 구조 {n_rescued}"
    for t in data["targets"]:
        ev = {e["type"]: e for e in data["events"] if e["target"] == t["target"] and e["type"] in ("rescue", "expire")}
        status = "rescued" if "rescue" in ev else "expired" if "expire" in ev else None
        if status is not None or t["status"] in ("rescued", "expired"):
            assert status == t["status"], f"T{t['target']}: 타임라인 {status} != 시뮬레이션 {t['status']}"
    return {"rescue_segments": n_rescue_seg, "assign_events": n_assign, "rescued": n_rescued}


# ---------------------------------------------------------------- 1) Gantt

def plot_gantt(data, out_dir):
    meta, n = data["meta"], data["meta"]["n_agents"]
    T = meta["final_t"]
    rows = _by_agent(data["steps"], n)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * n + 1.3))
    h = 0.62
    for i in range(n):
        col = agent_hex(i)
        y = n - 1 - i
        for t0, t1, mode, tgt in segments(rows[i]):
            if mode == "EXPLORE":
                kw = dict(facecolor=_light(col, 0.72), hatch=None)
            elif mode == "RESCUE":
                kw = dict(facecolor=_rgb01(col), hatch=None)
            else:
                kw = dict(facecolor="white", hatch="////")
            ax.add_patch(Rectangle((t0, y - h / 2), t1 - t0, h, edgecolor=_rgb01(col), linewidth=0.6, **kw))
            if mode == "RESCUE" and tgt >= 0 and t1 - t0 >= max(4, T * 0.025):
                ax.text((t0 + t1) / 2, y, f"T{tgt}", ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
    for e in data["events"]:
        if e["agent"] is None or e["agent"] < 0:
            continue
        y = n - 1 - e["agent"]
        if e["type"] == "rescue":
            ax.scatter(e["t"], y + h / 2 + 0.1, marker="*", s=55, color=C_RESCUED, edgecolor="black",
                       linewidths=0.4, zorder=5)
        elif e["type"] == "refill":
            ax.scatter(e["t"], y - h / 2 - 0.12, marker="^", s=18, color="0.35", zorder=5, linewidths=0)
    for e in data["events"]:
        if e["type"] == "expire" and e["detail"]:
            for k in map(int, e["detail"].split(",")):
                ax.scatter(e["t"], n - 1 - k + h / 2 + 0.1, marker="x", s=30, color=C_EXPIRED, linewidths=1.2, zorder=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"R{i}" for i in reversed(range(n))])
    for lab, i in zip(ax.get_yticklabels(), reversed(range(n))):
        lab.set_color(_rgb01(agent_hex(i)) if agent_hex(i) != "#F0E442" else "#9A8F00")
        lab.set_fontweight("bold")
    ax.set_xlim(0, max(1, T))
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("time step")
    ax.set_title("UAV role timeline (allocation)")
    handles = [Patch(facecolor="0.85", edgecolor="0.4", label="explore"),
               Patch(facecolor="0.35", edgecolor="0.4", label="rescue (T = task id)"),
               Patch(facecolor="white", edgecolor="0.4", hatch="////", label="return"),
               Line2D([], [], marker="*", ls="", color=C_RESCUED, markeredgecolor="black", ms=8, label="rescued"),
               Line2D([], [], marker="x", ls="", color=C_EXPIRED, ms=6, label="expired while assigned"),
               Line2D([], [], marker="^", ls="", color="0.35", ms=5, label="refill")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out_dir, "gantt")


# ---------------------------------------------------------------- 2) 조난자 타임라인

def _target_intervals(data):
    """조난자 -> {agent: [(t0, t1)]} 배정 구간, 그리고 동시에 2 대 이상이 맡은 tick 목록."""
    n = data["meta"]["n_agents"]
    owners = {}
    for s in data["steps"]:
        if s["t"] > 0 and s["mode"] == "RESCUE" and s["target"] >= 0:
            owners.setdefault(s["target"], {}).setdefault(s["t"], []).append(s["agent"])
    out = {}
    for ti, by_t in owners.items():
        per = {k: [] for k in range(n)}
        for t in sorted(by_t):
            for k in by_t[t]:
                iv = per[k]
                if iv and iv[-1][1] == t - 1:
                    iv[-1] = (iv[-1][0], t)
                else:
                    iv.append((t - 1, t))
        dup = sorted(t for t, ks in by_t.items() if len(ks) > 1)
        out[ti] = (per, dup)
    return out


def plot_targets_timeline(data, out_dir):
    T = data["meta"]["final_t"]
    tg = sorted(data["targets"], key=lambda t: (t["deadline"], t["target"]))
    iv = _target_intervals(data)
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(tg) + 1.4))
    h = 0.66
    for row, t in enumerate(tg):
        y = len(tg) - 1 - row
        end = t["rescued_at"] or t["expired_at"] or T
        disc = t["discovered_at"]
        ax.add_patch(Rectangle((0, y - h / 2), (disc if disc is not None else end), h, facecolor=C_UNDISC,
                               edgecolor="none"))
        if disc is not None:
            fa = t["first_assigned_at"]
            wait_end = (fa - 1) if fa is not None else end
            if wait_end > disc:
                ax.add_patch(Rectangle((disc, y - h / 2), wait_end - disc, h, facecolor=C_WAIT, edgecolor="none"))
        per, dup = iv.get(t["target"], ({}, []))
        for k, spans in per.items():
            for t0, t1 in spans:
                ax.add_patch(Rectangle((t0, y - h / 2), t1 - t0, h, facecolor=_rgb01(agent_hex(k)), edgecolor="none"))
        for d in dup:                                           # 중복 배정 (같은 조난자를 2 대 이상)
            ax.add_patch(Rectangle((d - 1, y - h / 2), 1, h, facecolor="none", edgecolor="black", hatch="xxxx",
                                   linewidth=0))
        ax.plot([t["deadline"]] * 2, [y - h / 2 - 0.1, y + h / 2 + 0.1], ls=":", color="black", lw=0.9)
        if t["status"] == "rescued":
            ax.scatter(t["rescued_at"], y, marker="*", s=60, color=C_RESCUED, edgecolor="black", linewidths=0.4, zorder=5)
        elif t["status"] == "expired":
            ax.scatter(t["expired_at"], y, marker="x", s=36, color=C_EXPIRED, linewidths=1.4, zorder=5)
    ax.set_yticks(range(len(tg)))
    ax.set_yticklabels([f"T{t['target']}" for t in reversed(tg)])
    ax.set_xlim(0, max(1, T, max((t["deadline"] for t in tg), default=0) + 5))
    ax.set_ylim(-0.8, len(tg) - 0.2)
    ax.set_xlabel("time step")
    ax.set_title("Victim timeline (sorted by deadline)")
    n = data["meta"]["n_agents"]
    handles = [Patch(facecolor=C_UNDISC, label="undiscovered"), Patch(facecolor=C_WAIT, label="discovered, unassigned")]
    handles += [Patch(facecolor=_rgb01(agent_hex(k)), label=f"assigned R{k}") for k in range(n)]
    handles += [Patch(facecolor="white", edgecolor="black", hatch="xxxx", label="duplicate assignment"),
                Line2D([], [], ls=":", color="black", label="deadline"),
                Line2D([], [], marker="*", ls="", color=C_RESCUED, markeredgecolor="black", ms=8, label="rescued"),
                Line2D([], [], marker="x", ls="", color=C_EXPIRED, ms=6, label="expired")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out_dir, "targets_timeline")


# ---------------------------------------------------------------- 3) 궤적 지도

def _draw_map(ax, data):
    obst = np.asarray(data["obstacles"], bool)
    ax.imshow(np.where(obst, 0.25, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    dr, dc = data["depot"]
    ax.scatter(dc, dr, marker="s", s=60, color="#1E9E5A", edgecolor="black", linewidths=0.6, zorder=6)
    ax.text(dc, dr, "D", ha="center", va="center", fontsize=6, color="white", fontweight="bold", zorder=7)


def _target_marker(ax, t, at=None, label=True, size=1.0):
    """at=None 이면 최종 상태, 아니면 시각 at 의 상태로 표시."""
    def happened(x):
        return x is not None and (at is None or x <= at)
    r, c = t["r"], t["c"]
    if happened(t["rescued_at"]):
        ax.scatter(c, r, marker="*", s=70 * size, color=C_RESCUED, edgecolor="black", linewidths=0.4, zorder=8)
        if label:
            ax.annotate(f"R{t['rescued_by']}@{t['rescued_at']}", (c, r), xytext=(3, 3), textcoords="offset points",
                        fontsize=5.5, zorder=9)
    elif happened(t["expired_at"]):
        ax.scatter(c, r, marker="x", s=40 * size, color=C_EXPIRED, linewidths=1.3, zorder=8)
    elif happened(t["discovered_at"]):
        ax.scatter(c, r, marker="o", s=26 * size, facecolor=C_WAIT, edgecolor="black", linewidths=0.5, zorder=8)
    else:
        ax.scatter(c, r, marker="o", s=26 * size, facecolor="none", edgecolor="0.45", linewidths=0.8, zorder=8)


def plot_trajectories(data, out_dir):
    n = data["meta"]["n_agents"]
    rows = _by_agent(data["steps"], n)
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * data["meta"]["rows"] / max(1, data["meta"]["cols"]) + 0.6))
    _draw_map(ax, data)
    for i in range(n):
        pts = np.array([[s["c"], s["r"]] for s in rows[i]], float)
        if len(pts) < 2:
            continue
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        rgb = _rgb01(agent_hex(i))
        alpha = np.linspace(0.15, 1.0, len(segs))
        lc = LineCollection(segs, colors=[(*rgb, a) for a in alpha], linewidths=1.6, zorder=5)
        ax.add_collection(lc)
        ax.scatter(pts[-1, 0], pts[-1, 1], s=40, color=rgb, edgecolor="white", linewidths=1.2, zorder=7)
        ax.annotate(f"R{i}", (pts[-1, 0], pts[-1, 1]), xytext=(4, -8), textcoords="offset points", fontsize=7,
                    color=rgb, fontweight="bold", zorder=9)
    for t in data["targets"]:
        _target_marker(ax, t)
    handles = [Line2D([], [], color=_rgb01(agent_hex(i)), lw=2, label=f"R{i}") for i in range(n)]
    handles += [Line2D([], [], marker="s", ls="", color="#1E9E5A", markeredgecolor="black", label="depot / start"),
                Line2D([], [], marker="*", ls="", color=C_RESCUED, markeredgecolor="black", ms=9, label="rescued (UAV@t)"),
                Line2D([], [], marker="x", ls="", color=C_EXPIRED, ms=6, label="expired"),
                Line2D([], [], marker="o", ls="", markerfacecolor=C_WAIT, markeredgecolor="black", label="discovered"),
                Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="0.45", label="undiscovered")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False)
    ax.set_title(f"UAV trajectories (t = 0 → {data['meta']['final_t']}, darker = later)")
    _save(fig, out_dir, "trajectories")


# ---------------------------------------------------------------- 4) 누적 곡선 / 5) 용량

def plot_cumulative(data, out_dir):
    team = data["team"]
    n_t = max(1, data["meta"]["n_targets"])
    t = [r["t"] for r in team]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(t, [r["discovered"] / n_t for r in team], color="#E69F00", lw=1.8, label="discovered")
    ax.plot(t, [r["rescued"] / n_t for r in team], color=C_RESCUED, lw=1.8, label="rescued")
    ax.plot(t, [r["expired"] / n_t for r in team], color=C_EXPIRED, lw=1.8, label="expired")
    ax.plot(t, [r["explored"] for r in team], color="0.3", lw=1.4, ls="--", label="explored ratio")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(1, t[-1]))
    ax.set_xlabel("time step")
    ax.set_ylabel(f"fraction (N_T = {n_t})")
    ax.set_title("Cumulative discovery / rescue / expiry")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "cumulative")


def plot_capacity(data, out_dir):
    n = data["meta"]["n_agents"]
    rows = _by_agent(data["steps"], n)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    cmax = max((s["capacity"] for s in data["steps"]), default=1)
    for i in range(n):
        off = (i - (n - 1) / 2) * 0.05                      # 겹침 방지 미세 오프셋
        ax.step([s["t"] for s in rows[i]], [s["capacity"] + off for s in rows[i]], where="post",
                color=_rgb01(agent_hex(i)), lw=1.6, label=f"R{i}")
    ax.set_yticks(range(int(cmax) + 1))
    ax.set_xlim(0, max(1, data["meta"]["final_t"]))
    ax.set_xlabel("time step")
    ax.set_ylabel("remaining supplies")
    ax.set_title("UAV capacity")
    ax.legend(frameon=False, ncol=min(n, 8), loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "capacity")


# ---------------------------------------------------------------- 6) 할당 스냅샷

def plot_snapshots(data, out_dir, k=6):
    n = data["meta"]["n_agents"]
    T = data["meta"]["final_t"]
    times_avail = sorted({s["t"] for s in data["steps"]})
    want = np.linspace(0, T, k)
    times = [min(times_avail, key=lambda x: abs(x - w)) for w in want]
    by_t = {}
    for s in data["steps"]:
        by_t.setdefault(s["t"], {})[s["agent"]] = s
    ncol = 3
    nrow = int(np.ceil(k / ncol))
    aspect = data["meta"]["rows"] / max(1, data["meta"]["cols"])
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, (7.2 / ncol * aspect + 0.45) * nrow + 0.8),
                             gridspec_kw={"hspace": 0.35})
    for ax, t in zip(np.ravel(axes), times):
        _draw_map(ax, data)
        for tt in data["targets"]:
            _target_marker(ax, tt, at=t, label=False, size=0.55)
        for i in range(n):
            s = by_t[t][i]
            rgb = _rgb01(agent_hex(i))
            if s["goal_r"] >= 0:
                ax.annotate("", xy=(s["goal_c"], s["goal_r"]), xytext=(s["c"], s["r"]),
                            arrowprops=dict(arrowstyle="-|>", color=rgb, lw=1.2, shrinkA=0, shrinkB=1), zorder=9)
            ax.scatter(s["c"], s["r"], s=22, color=rgb, edgecolor="white", linewidths=0.8, zorder=10)
        ax.set_title(f"t = {t}", fontsize=9)
    for ax in np.ravel(axes)[len(times):]:
        ax.axis("off")
    handles = [Line2D([], [], color=_rgb01(agent_hex(i)), marker="o", lw=1.2, label=f"R{i} → goal") for i in range(n)]
    fig.legend(handles=handles, loc="lower center", ncol=min(n, 8), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Allocation snapshots (arrow = current goal: task / frontier / depot)", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.1, wspace=0.08, hspace=0.35)
    _save(fig, out_dir, "snapshots")


# ---------------------------------------------------------------- 저장

def _write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def folder_name(meta, now=None):
    policy = re.sub(r"[^A-Za-z0-9_.+-]", "", str(meta.get("policy", "policy")).replace("stand-in", "standin"))
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{policy or 'policy'}_t{meta.get('final_t', 0)}"


def save_alloc_plots(data, figures_root):
    """그래프 6 종 (png + pdf) + 원자료 (csv / json) 저장. 반환: 저장 폴더."""
    counts = check_consistency(data)
    out_dir = os.path.join(figures_root, folder_name(data["meta"]))
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(STYLE):
        plot_gantt(data, out_dir)
        plot_targets_timeline(data, out_dir)
        plot_trajectories(data, out_dir)
        plot_cumulative(data, out_dir)
        plot_capacity(data, out_dir)
        plot_snapshots(data, out_dir)
    _write_csv(os.path.join(out_dir, "steps.csv"), data["steps"])
    _write_csv(os.path.join(out_dir, "events.csv"), data["events"])
    _write_csv(os.path.join(out_dir, "targets.csv"), data["targets"])
    _write_csv(os.path.join(out_dir, "team.csv"), data["team"])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(data["meta"], consistency=counts), f, indent=2, default=str)
    return out_dir
