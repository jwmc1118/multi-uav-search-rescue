"""그리디 정책 (policies/role_rules.py RuleDecider) 으로 계층 구조 env 를 끝까지 돌리는 baseline / 점검 스크립트.

    python -m evaluation.run_rules                     # env.yaml + role.yaml + explore.yaml + rescue.yaml, seed 0..199, 그리디
    python evaluation/run_rules.py --episodes 50 --policy random
    python -m evaluation.run_rules --episodes 20 --tag check   # 결과 파일 이름 = *_check (기준선 파일을 덮어쓰지 않음)

에피소드 k 는 seed k 로 맵 / 조난자 / 골든타임 고정. 출력: 구조율 · 발견율 · 탐색률, 역할 비율 시계열, 이벤트별 횟수,
평균 결정 빈도, explore 2 단계 통계 (1·2 단계 결정 수, 덩어리당 평균 m, 쪼개짐 · 합쳐짐, 통신 이탈 m 감소,
구간 소진까지 step, 역할 전환, m >= 2 덩어리 구간 크기 변동계수 (std / mean) 와 <= 0.5 비율, 결정 후 목표 없이 멈춘
UAV-step 수), 이벤트 통계 (재결정 UAV 수, 목표 변경, 진동, RESCUE 재결정 원인), explore 지표 (κ 평균 · 중앙값, |S_eff|/|S|,
고른 / 가장 가까운 덩어리 거리 비, EXPLORE step 당 새 관측 셀), 담당 후보 지표 (이벤트 2 RESCUE 중 담당 후보 /
담당 아닌 생존자, 담당 후보가 RESCUE 안 고름, 1 단계 담당 후보 덩어리 그대로, 이벤트 시점 배정 0 덩어리 수, 재선택 goal 유지),
부분 귀환 통계 (횟수, 부족분, 다음 충전까지 구조 수, 용량 상태별 역할 선택 비율), 그리고 점검 — 예외 없음 / 마스크 위반 0 /
같은 step 재결정 <= max_redecide / deadlock 없음 (제자리인 UAV 가 IDLE_LIMIT step 넘게 이어지지 않음. 그룹 지도에
frontier 가 남았는데 depot 에서 대기하는 것도 포함 — frontier 가 없을 때의 depot 대기만 정상).
결과: <repo>/results/rules/  episodes_<policy>.csv, summary_<policy>.csv (지표 + 에피소드 진행 10 구간 역할 비율),
      role_timeseries_<policy>.png, events_<policy>.png, event_logs/<tag>/episode_<seed>.json (이벤트 / 목표 변경 로그)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import argparse
import csv

import numpy as np

OUT_DIR = os.path.join(ROOT, "results", "rules")
IDLE_LIMIT = 50                  # 제자리 연속 step 이 이보다 길면 deadlock 으로 본다


def run_episode(sim, decider, seed):
    from env.core.modes import Mode
    from env.role.simulation import EVENTS, EXPLORE_EVENTS
    sim.random_state = np.random.RandomState(seed)
    sim.episode_tag = seed
    sim.reset()
    t_max = int(sim.cfg["max_episode_len"])
    roles = np.zeros((t_max + 1, 3))                 # step 별 역할 비율
    idle = [0] * sim.n_agents
    max_idle = 0
    events = {k: 0 for k in EVENTS + EXPLORE_EVENTS}
    m_vals, cvs, idle_ng = [], [], 0
    n_role = n_sub = redecide_steps = violations = 0
    max_redecide = 0
    done = False
    while not done:
        prev = [a.pos for a in sim.agents]
        done, info = sim.step(decider)
        for k, v in info["events"].items():
            events[k] += v
        n_role += int(sum(info["role_valid"]))
        n_sub += int(sum(info["rescue_valid"]) + sum(info["explore_valid"]))
        if info["m_mean"] > 0:
            m_vals.append(info["m_mean"])
        idle_ng += info["idle_no_goal"]
        for g, members in enumerate(sim.groups):                 # m >= 2 덩어리의 구간 크기 변동계수 (std / mean)
            by = {}
            for i in members:
                a = sim.agents[i]
                if a.mode is Mode.EXPLORE and a.blob is not None and a.cluster:
                    by.setdefault(a.blob, []).append(len(a.cluster))
            for sizes in by.values():
                if len(sizes) >= 2:
                    cvs.append(float(np.std(sizes) / np.mean(sizes)))
        redecide_steps += int(info["redecide"] > 0)
        max_redecide = max(max_redecide, info["redecide"])
        violations += info["mask_violations"]
        for r in info["role"]:
            roles[sim.time, r] += 1.0 / sim.n_agents
        for i, a in enumerate(sim.agents):
            # depot 대기는 정상 — 단 그룹 지도에 frontier 가 남아 있는데 대기하면 deadlock 으로 센다
            waiting = a.mode is Mode.RETURN and a.pos == sim.depot and not sim.knowledge.frontier[i].any()
            stuck = a.pos == prev[i] and not waiting and (a.goal is None or a.goal != a.pos or a.mode is Mode.RETURN)
            idle[i] = idle[i] + 1 if stuck else 0
            max_idle = max(max_idle, idle[i])
    n_t = max(1, len(sim.targets))
    st = info["explore_stats"]
    es = info["event_stats"]
    cap_ratio = {}
    for c, cnt in es["role_by_cap"].items():
        tot = sum(cnt)
        for j, name in enumerate(("explore", "rescue", "return")):
            cap_ratio[f"role_{c}_{name}"] = cnt[j] / tot if tot else float("nan")
    rs = es["rewards"]
    return {
        **{f"rw_{k}": v for k, v in rs["sum"].items()},
        "rescue_events": rs["rescue_events"], "rescue_S": rs["rescue_S"], "rescue_F": rs["rescue_F"],
        "rescue_U_eps_size": rs["rescue_U_eps_size"], "rescue_F_unavoidable": rs["rescue_F_unavoidable"],
        "rescue_bonus_rate": rs["rescue_bonus_rate"], "rescue_SF_term": rs["rescue_SF_term"],
        "rescue_bonus_term": rs["rescue_bonus_term"], "role_rescue_term": rs["role_rescue_term"],
        "explore_term": rs["explore_term"], "explore_cov_own_return": rs["explore_return"],
        "explore_option_len": rs["explore_option_len"],
        **{f"psi_f_bin{b}": v for b, v in enumerate(rs["psi_f_bins"])},
        "explore_masked_decisions": es["explore_masked_decisions"], "forced_return_all_masked": es["forced_return_all_masked"],
        "ev5_exploration_complete": es["ev5_causes"]["exploration_complete"],
        "role_osc": rs["osc"], "role_forced": rs["forced"], "role_voluntary": rs["voluntary"],
        "preempt_candidates": es["preempt_candidates"], "dec_interval": es["dec_interval"], "latch_hold": es["latch_hold"],
        "link_made": es["link_made"], "link_broken": es["link_broken"], "group_size_ep": es["group_size"],
        **{f"nbr_bin{b}": v for b, v in enumerate(es["nbr_dist_hist"])},
        "dup_in": rs["dup_in"], "dup_out": rs["dup_out"],
        "conn_real_mean": rs["conn_real_mean"], "rescue_c_bar": rs["rescue_c_bar"],
        "explore_conn_mult": rs["explore_conn_mult"],
        "n_groups": es["n_groups"], "survivor_truncated": es["survivor_truncated"], "ev2_rescue_nearest": es["ev2_rescue_nearest"],
        "seed": seed, "n_targets": len(sim.targets), "episode_len": sim.time,
        "rescue_rate": info["rescued_total"] / n_t, "expire_rate": info["expired_total"] / n_t,
        "discover_rate": info["discovered_total"] / n_t, "explored": info["explored"],
        "same_cell": info["same_cell_total"],
        "role_decisions_per_uav_100": 100.0 * n_role / sim.n_agents / sim.time,
        "sub_decisions_per_uav_100": 100.0 * n_sub / sim.n_agents / sim.time,
        "redecide_steps": redecide_steps, "max_redecide": max_redecide, "mask_violations": violations,
        "max_idle_streak": max_idle, **{f"ev_{k}": v for k, v in events.items()},
        "explore_s1": st["s1"], "explore_s2": st["s2"], "m_mean": float(np.mean(m_vals)) if m_vals else 0.0,
        "ghost_drop": st["ghost_drop"], "role_switch": st["role_switch"],
        "seg_life": float(np.mean(st["seg_life"])) if st["seg_life"] else float("nan"),
        "redecide_uavs": es["redecide_uavs"], "goal_change": es["goal_change"], "oscillation": es["oscillation"],
        "partial_return": es["partial_return"],
        "join_events": es["join_events"], "reselect_uavs": es["reselect_uavs"], "reselect_kept_goal": es["reselect_kept_goal"],
        "reselect_goal_change": es["reselect_goal_change"], "explore_reselect": st["reselect"],
        "ev2_rescue_by_claim": es["ev2_rescue_by_claim"], "ev2_rescue_other": es["ev2_rescue_other"],
        "claim_not_rescue": es["claim_not_rescue"], "s1_claim_kept": es["s1_claim_kept"], "s1_empty_blobs": es["s1_empty_blobs"],
        "s1_participants": es["s1_participants"], "ev2_retained": es["ev2_retained"],
        "ev2_retained_blob_changed": es["ev2_retained_blob_changed"],
        "ev2_redecide": es["ev2_redecide"], "ev2_rescue": es["ev2_rescue"], "known_to_rescue": es["known_to_rescue"],
        "blob_dist_ratio": es["blob_dist_ratio"], "new_cells_per_explore_step": es["new_cells_per_explore_step"],
        **{f"res_redecide_{e}": v for e, v in es["rescue_redecide_causes"].items()},
        "partial_deficit": float(np.mean(es["partial_deficits"])) if es["partial_deficits"] else float("nan"),
        "rescued_after_partial": float(np.mean(es["rescued_after_partial"])) if es["rescued_after_partial"] else float("nan"),
        **cap_ratio,
        "_deficits": list(es["partial_deficits"]),
        "seg_cv_mean": float(np.mean(cvs)) if cvs else float("nan"),
        "seg_cv_ok": float(np.mean([c <= 0.5 for c in cvs])) if cvs else float("nan"),
        "idle_no_goal": idle_ng,
        "_roles": roles[1:sim.time + 1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", help="config yaml (생략하면 training/config/env.yaml + role.yaml + explore.yaml + rescue.yaml)")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--tag", default=None, help="결과 파일 이름 꼬리 (기본 = policy 이름)")
    ap.add_argument("--policy", default="default", choices=["default", "random"],
                    help="default = 그리디 정책 / random = 마스크 안 무작위")
    ap.add_argument("--return-with-cap", type=int, choices=[0, 1], default=None,
                    help="role.return_with_cap 덮어쓰기 (생략 = yaml)")
    ap.add_argument("--no-log", action="store_true", help="에피소드 이벤트 로그 JSON 저장 안 함")
    args = ap.parse_args()

    from env.role.simulation import EVENTS, EXPLORE_EVENTS, RoleSimulation
    from policies.role_rules import RandomDecider, RuleDecider
    from training.config_loader import load_config
    cfg = load_config(args.config and [os.path.join(ROOT, p) for p in args.config])["env_config"]
    if args.return_with_cap is not None:
        cfg["role"] = dict(cfg.get("role") or {}, return_with_cap=bool(args.return_with_cap))
    tag_name = args.tag or args.policy
    cfg["events"] = dict(cfg.get("events") or {},
                         log_dir=None if args.no_log else os.path.join(OUT_DIR, "event_logs", tag_name))
    sim = RoleSimulation(cfg)
    decider = {"default": RuleDecider(), "random": RandomDecider(seed=0)}[args.policy]
    eps = []
    for s in range(args.episodes):
        eps.append(run_episode(sim, decider, s))
        print(f"\r  {args.policy}: {s + 1}/{args.episodes}", end="", flush=True)
    print()

    keys = ["rescue_rate", "expire_rate", "discover_rate", "explored", "episode_len", "same_cell",
            "role_decisions_per_uav_100", "sub_decisions_per_uav_100", "redecide_steps",
            "explore_s1", "explore_s2", "m_mean", "ghost_drop", "seg_life", "role_switch",
            "seg_cv_mean", "seg_cv_ok", "idle_no_goal",
            "redecide_uavs", "goal_change", "oscillation",
            "partial_return", "partial_deficit", "rescued_after_partial",
            "rw_role_team", "rw_role_osc", "rw_rescue_sf", "rw_rescue_bonus", "rw_rescue_dup", "rw_explore_cov",
            "role_rescue_term", "rescue_SF_term", "rescue_bonus_term", "explore_term",
            "rescue_S", "rescue_F", "rescue_U_eps_size", "rescue_F_unavoidable", "rescue_bonus_rate",
            "explore_cov_own_return", "explore_option_len", "explore_masked_decisions", "forced_return_all_masked",
            "ev5_exploration_complete",
            "join_events", "reselect_uavs", "reselect_kept_goal", "reselect_goal_change", "explore_reselect",
            "ev2_rescue_by_claim", "ev2_rescue_other", "claim_not_rescue", "s1_claim_kept", "s1_empty_blobs",
            "s1_participants", "ev2_retained", "ev2_retained_blob_changed", "ev2_redecide", "ev2_rescue", "known_to_rescue", "blob_dist_ratio", "new_cells_per_explore_step",
            "res_redecide_1", "res_redecide_2", "res_redecide_3", "res_redecide_4", "res_redecide_5",
            "preempt_candidates", "dec_interval", "latch_hold", "link_made", "link_broken", "group_size_ep",
            "nbr_bin0", "nbr_bin1", "nbr_bin2", "nbr_bin3", "nbr_bin4",
            "rescue_events",
            "role_osc", "role_forced", "role_voluntary", "dup_in", "dup_out",
            "conn_real_mean", "rescue_c_bar", "explore_conn_mult", "n_groups", "ev2_rescue_nearest", "survivor_truncated"] + \
        [f"role_{c}_{r}" for c in ("zero", "partial", "full") for r in ("explore", "rescue", "return")]
    print(f"\n=== {args.policy} policy, {len(eps)} episodes, N={sim.n_agents}, return_with_cap="
          f"{sim.role_cfg['return_with_cap']} ===")
    summary = []
    for k in keys + [f"ev_{e}" for e in EVENTS + EXPLORE_EVENTS]:
        v = np.array([e[k] for e in eps], float)
        v = v[np.isfinite(v)] if np.isfinite(v).any() else np.zeros(1)
        summary.append({"metric": k, "mean": v.mean(), "std": v.std(), "min": v.min(), "max": v.max()})
        print(f"  {k:<28} {v.mean():9.3f} ± {v.std():7.3f}   (min {v.min():.3f}, max {v.max():.3f})")
    viol = sum(e["mask_violations"] for e in eps)
    max_red = max(e["max_redecide"] for e in eps)
    max_idle = max(e["max_idle_streak"] for e in eps)
    deficits = [d for e in eps for d in e["_deficits"]]
    if deficits:
        vals, cnts = np.unique(deficits, return_counts=True)
        print("  partial-return deficit distribution: " + ", ".join(f"{int(v)}: {int(c)}" for v, c in zip(vals, cnts)))
    print("\n  checks:")
    print(f"    mask violations           {viol}  {'OK' if viol == 0 else 'FAIL'}")
    print(f"    max fallback in one step  {max_red}  {'OK' if max_red <= sim.role_cfg['max_redecide'] else 'FAIL'}")
    print(f"    max idle streak (steps)   {max_idle}  {'OK' if max_idle <= IDLE_LIMIT else 'FAIL'} (limit {IDLE_LIMIT})")

    # 에피소드 진행 10 구간 (각 에피소드 길이를 10 등분) 별 역할 비율 평균
    bins = np.zeros((10, 3))
    for e in eps:
        r = e["_roles"]
        edges = np.linspace(0, len(r), 11).astype(int)
        bins += np.array([r[edges[k]:max(edges[k + 1], edges[k] + 1)].mean(0) for k in range(10)])
    bins /= len(eps)
    print("\n  role ratio by episode progress (10 bins):")
    print("    bin   " + "".join(f"{k * 10:>3}-{k * 10 + 10:<3}%" for k in range(10)))
    for j, name in enumerate(("explore", "rescue", "return")):
        print(f"    {name:<7}" + "".join(f"{bins[k, j]:>9.3f}" for k in range(10)))
    for k in range(10):
        for j, name in enumerate(("explore", "rescue", "return")):
            summary.append({"metric": f"role_{name}_bin{k}", "mean": bins[k, j], "std": "", "min": "", "max": ""})

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "_" + tag_name
    with open(os.path.join(OUT_DIR, f"episodes{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, [k for k in eps[0] if not k.startswith("_")])
        w.writeheader()
        w.writerows([{k: v for k, v in e.items() if not k.startswith("_")} for e in eps])
    with open(os.path.join(OUT_DIR, f"summary{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = max(len(e["_roles"]) for e in eps)
    acc, cnt = np.zeros((T, 3)), np.zeros(T)
    for e in eps:
        acc[:len(e["_roles"])] += e["_roles"]
        cnt[:len(e["_roles"])] += 1
    frac = acc / np.maximum(cnt, 1)[:, None]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.stackplot(np.arange(1, T + 1), frac.T, labels=["explore", "rescue", "return"],
                 colors=["#56B4E9", "#E69F00", "#999999"], alpha=0.9)
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, T + 1), cnt / len(eps), color="black", lw=1, ls="--", label="episodes still running")
    ax.set_xlim(1, T); ax.set_ylim(0, 1); ax2.set_ylim(0, 1.05)
    ax.set_xlabel("time step"); ax.set_ylabel("fraction of UAVs"); ax2.set_ylabel("fraction of episodes running")
    ax.set_title(f"Role ratio over time ({args.policy}, {len(eps)} episodes)", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"role_timeseries{tag}.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    labels = list(EVENTS + EXPLORE_EVENTS)
    m = [np.mean([e[f"ev_{k}"] for e in eps]) for k in labels]
    ax.bar(labels, m, color=["#0072B2"] * len(EVENTS) + ["#D55E00"] * len(EXPLORE_EVENTS))
    for k, v in enumerate(m):
        ax.text(k, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("per episode (sum over UAVs)")
    ax.set_title(f"Role events 1-5 / blob split, merge (no re-decision) ({tag_name})", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"events{tag}.png"), dpi=200); plt.close(fig)
    print(f"\nsaved: {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
