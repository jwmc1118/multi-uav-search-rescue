"""이론적 상한 / 실패 원인 분석 (학습 코드와 독립).

    python rule_evaluation/upper_bound.py --episodes 500
    python rule_evaluation/upper_bound.py --episodes 200 --checkpoint marl_models/RolePPO/checkpoints/checkpoint_000460
    python rule_evaluation/upper_bound.py --episodes 200 --checkpoint <ckpt> --n_agents 6   # 학습과 다른 UAV 수

목적: 정책의 구조율(≈0.85)이 정책 문제인지, 원리적으로 못 구하는 조난자 때문인지 구분.

[상한] 체크포인트 없이 동작. 학습과 같은 env 코드로 seed 0..M-1 을 reset 만 해서 맵 / depot / 조난자(위치, τ_j) 를 얻는다
       (--checkpoint 가 있으면 그 체크포인트의 학습 env_config, 없으면 --config (기본 training/config/env.yaml + role.yaml + explore.yaml + rescue.yaml)).
  구조 판정 = env 그대로: UAV 가 조난자와 같은 칸, 같은 tick 에 구조가 마감 처리보다 먼저 -> 도착 시각 <= τ_j 면 구조.
  UAV 는 depot 에서 t = 0 출발, tick 당 8방향 한 칸 (코너 컷팅 금지) -> 도착 시각 = ground truth 장애물 BFS 거리.
  d_j = depot -> 조난자 j BFS 거리,  feasible_j = d_j <= τ_j  (τ_j = env.core.targets.deadline: end_time, 없으면 golden_time)
  UB_loose  = Σ feasible_j / N_T — 위치 전부 앎, 용량 무제한, 조난자마다 UAV 하나씩 간다고 가정한 낙관적 상한
  UB_greedy = 위치 전부 앎, UAV N 대 · 용량 c · depot 재보충 준수, 매 결정마다 "도착 시각 <= τ_j" 인 조난자 중
              마감이 가장 임박한 곳으로 이동 (용량 0 이면 depot 복귀, 이동시간 = BFS 거리, UAV 간 충돌 무시) 의 구조 비율.
              greedy 휴리스틱이라 엄밀한 최적값은 아니다 (전지적 정보 + 단순 스케줄 기준선).
  deadline_mode = from_discovery 면 τ_j 를 발견 시각과 무관하게 golden_time 으로 근사 (경고 출력).

[실패 원인] --checkpoint 를 주면 같은 seed 로 evaluate 와 같은 방식 (세 헤드 greedy) 에피소드 실행,
  만료된 조난자를 (a) infeasible (d_j > τ_j) (b) feasible 인데 끝까지 미발견 (c) 발견했지만 아무도 할당 안 함
  (d) 할당됐지만 시간 안에 도착 못함 (중복 할당 여부 별도 카운트) 로 분류해 N_T 대비 비율로.

결과: <repo>/results/upper_bound/  (csv + PNG)
"""
import os
import sys

# 스크립트를 직접 실행해도 패키지를 찾도록 레포 루트를 경로에 추가 (training/train.py 와 같은 방식).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
_CALLER_CWD = os.getcwd()
os.chdir(ROOT)

import argparse
import csv

import numpy as np

OUT_DIR = os.path.join(ROOT, "results", "upper_bound")
REF_RESCUE = 0.85          # 히스토그램 기준선 (현재 정책 구조율 근처)
GOLDEN_BINS = [(0, 150), (150, 200), (200, 250), (250, 10 ** 9)]


# ---------------------------------------------------------------- 상한

def episode_instance(sim, seed):
    """seed 로 reset -> (장애물, depot, 조난자 [(pos, τ, golden)], c, N, t_max). 학습 분포와 같은 env 코드."""
    from env.core.targets import deadline as _deadline
    sim.random_state = np.random.RandomState(seed)
    sim.reset()
    targets = [(tuple(t.pos), int(_deadline(t)), int(t.golden_time)) for t in sim.targets]
    return {"obstacles": np.array(sim.obstacles, bool), "depot": tuple(sim.depot), "targets": targets,
            "capacity": int(sim.cfg["capacity"]), "n_agents": sim.n_agents,
            "t_max": int(sim.cfg["max_episode_len"])}


def bounds(inst):
    """UB_loose, UB_greedy 와 조난자별 (d_j, τ_j, feasible)."""
    from env.core.pathfinding import bfs_field
    obst, depot, tg = inst["obstacles"], inst["depot"], inst["targets"]
    rows, cols = obst.shape
    n_t = len(tg)
    field = {"D": bfs_field(depot, obst, rows, cols)[0]}
    for j, (pos, _, _) in enumerate(tg):
        field[j] = bfs_field(pos, obst, rows, cols)[0]

    def dist(a, j):                                   # a = 'D' 또는 조난자 index, j = 조난자 index 또는 'D'
        cell = depot if j == "D" else tg[j][0]
        d = int(field[a][cell])
        return d if d >= 0 else 10 ** 9

    limit = lambda j: min(tg[j][1], inst["t_max"])
    d = [dist("D", j) for j in range(n_t)]
    feasible = [d[j] <= limit(j) for j in range(n_t)]

    # greedy 스케줄러
    uavs = [{"pos": "D", "t": 0, "cap": inst["capacity"]} for _ in range(inst["n_agents"])]
    remaining, active, rescued = set(range(n_t)), list(range(len(uavs))), 0
    while active and remaining:
        k = min(active, key=lambda k: (uavs[k]["t"], k))
        u = uavs[k]
        if u["cap"] == 0:
            u["t"] += dist(u["pos"], "D"); u["pos"] = "D"; u["cap"] = inst["capacity"]
            continue
        cands = [j for j in remaining if u["t"] + dist(u["pos"], j) <= limit(j)]
        if not cands:
            active.remove(k)
            continue
        j = min(cands, key=lambda j: (tg[j][1], dist(u["pos"], j), j))
        u["t"] += dist(u["pos"], j); u["pos"] = j; u["cap"] -= 1
        remaining.discard(j)
        rescued += 1
    return {"ub_loose": sum(feasible) / max(1, n_t), "ub_greedy": rescued / max(1, n_t),
            "d": d, "feasible": feasible}


def run_bounds(cfg, seeds):
    from env.role.simulation import RoleSimulation
    sim = RoleSimulation(cfg)
    if sim.cfg["deadline_mode"] != "from_start":
        print("[warn] deadline_mode = from_discovery: τ_j 를 golden_time 으로 근사 (발견 시각 무시) — 상한은 근사값")
    eps, tgt_rows = [], []
    for s in seeds:
        inst = episode_instance(sim, s)
        b = bounds(inst)
        eps.append({"seed": s, "n_targets": len(inst["targets"]), "ub_loose": b["ub_loose"], "ub_greedy": b["ub_greedy"],
                    "_inst": inst, "_b": b})
        for j, (pos, tau, golden) in enumerate(inst["targets"]):
            tgt_rows.append({"seed": s, "target": j, "r": pos[0], "c": pos[1], "deadline": tau, "golden_time": golden,
                             "d_depot": b["d"][j], "feasible": int(b["feasible"][j])})
    return eps, tgt_rows


def _stat(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std():.3f} (min {v.min():.3f})"


def report_bounds(eps, tgt_rows):
    print(f"\n=== 상한 ({len(eps)} episodes) ===")
    print(f"  UB_loose  : {_stat([e['ub_loose'] for e in eps])}")
    print(f"  UB_greedy : {_stat([e['ub_greedy'] for e in eps])}")
    print("\n  N_T 별 infeasible 비율 (d_j > τ_j)")
    for n in sorted({e["n_targets"] for e in eps}):
        rows = [r for r in tgt_rows if sum(1 for e in eps if e["seed"] == r["seed"] and e["n_targets"] == n)]
        if rows:
            print(f"    N_T={n:>2}: {1 - np.mean([r['feasible'] for r in rows]):.3f}  ({len(rows)} victims)")
    print("  골든타임 구간별 infeasible 비율")
    for lo, hi in GOLDEN_BINS:
        rows = [r for r in tgt_rows if lo <= r["golden_time"] < hi]
        if rows:
            label = f"[{lo}, {hi})" if hi < 10 ** 8 else f"[{lo}, ∞)"
            print(f"    golden {label:<11}: {1 - np.mean([r['feasible'] for r in rows]):.3f}  ({len(rows)} victims)")


def save_bounds(eps, tgt_rows, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    _write(os.path.join(OUT_DIR, f"ub_episodes{tag}.csv"), [{k: v for k, v in e.items() if not k.startswith("_")} for e in eps])
    _write(os.path.join(OUT_DIR, f"ub_targets{tag}.csv"), tgt_rows)
    for key, title in (("ub_loose", "UB_loose (positions known, unlimited capacity)"),
                       ("ub_greedy", "UB_greedy (positions known, N UAVs, capacity + refill)")):
        v = [e[key] for e in eps]
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.hist(v, bins=np.linspace(0, 1, 26), color="#56B4E9", edgecolor="white")
        ax.axvline(REF_RESCUE, color="black", ls=":", lw=1.4, label=f"policy ≈ {REF_RESCUE}")
        ax.axvline(np.mean(v), color="#D55E00", lw=1.4, label=f"mean {np.mean(v):.3f}")
        ax.set_xlabel("rescue ratio per episode"); ax.set_ylabel("episodes"); ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"{key}_hist{tag}.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- 실패 원인

def run_policy(ckpt, cfg, eps):
    from gui_evaluation.alloc_recorder import AllocRecorder
    from env.role.simulation import RoleSimulation
    from eval_common.checkpoint_policy import CheckpointPolicy
    policy = CheckpointPolicy(ckpt, env_config=cfg)
    print(f"  {policy.describe()}")
    sim = RoleSimulation(policy.env_config)
    sim.recorder = AllocRecorder(policy="model")
    out = []
    for k, e in enumerate(eps):
        sim.random_state = np.random.RandomState(e["seed"])
        sim.reset()
        assert [tuple(t.pos) for t in sim.targets] == [p for p, _, _ in e["_inst"]["targets"]], "seed 재현 실패"
        policy.seed(e["seed"])
        done = False
        while not done:
            done, _ = policy.step(sim)
        owners = {}                                   # 조난자 -> {t: 맡은 UAV 수}
        for s in sim.recorder.steps:
            if s["t"] > 0 and s["mode"] == "RESCUE" and s["target"] >= 0:
                owners.setdefault(s["target"], {}).setdefault(s["t"], 0)
                owners[s["target"]][s["t"]] += 1
        c = {"rescued": 0, "a_infeasible": 0, "b_undiscovered": 0, "c_unassigned": 0,
             "d_late": 0, "d_late_dup": 0, "unresolved": 0}
        for j, t in enumerate(sim.targets):
            if t.rescued_at is not None:
                c["rescued"] += 1
            elif t.expired_at is None:
                c["unresolved"] += 1
            elif not e["_b"]["feasible"][j]:
                c["a_infeasible"] += 1
            elif not t.discovered:
                c["b_undiscovered"] += 1
            elif j not in owners:
                c["c_unassigned"] += 1
            else:
                c["d_late"] += 1
                c["d_late_dup"] += int(max(owners[j].values()) > 1)
        n_t = max(1, len(sim.targets))
        out.append({"seed": e["seed"], "n_targets": len(sim.targets), "ub_loose": e["ub_loose"],
                    "ub_greedy": e["ub_greedy"], "policy_rescue": c["rescued"] / n_t,
                    **{f"{key}_ratio": v / n_t for key, v in c.items() if key != "rescued"}, "episode_len": sim.time})
        print(f"\r  policy episodes {k + 1}/{len(eps)}", end="", flush=True)
    print()
    return out


def report_failures(rows, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("policy_rescue", "policy rescued"), ("ub_greedy", "UB_greedy"), ("ub_loose", "UB_loose"),
            ("a_infeasible_ratio", "(a) infeasible (d_j > τ_j)"),
            ("b_undiscovered_ratio", "(b) feasible, never discovered"),
            ("c_unassigned_ratio", "(c) discovered, never assigned"),
            ("d_late_ratio", "(d) assigned, arrived too late"),
            ("d_late_dup_ratio", "    of which duplicate-assigned"),
            ("unresolved_ratio", "not resolved at episode end")]
    print(f"\n=== 정책 vs 상한, 실패 원인 ({len(rows)} episodes, / N_T) ===")
    for k, label in keys:
        v = np.array([r[k] for r in rows])
        print(f"  {label:<36} {v.mean():.3f} ± {v.std():.3f}")
    _write(os.path.join(OUT_DIR, f"failures{tag}.csv"), rows)
    _write(os.path.join(OUT_DIR, f"failures_summary{tag}.csv"),
           [{"metric": k, "label": label.strip(), "mean": float(np.mean([r[k] for r in rows])),
             "std": float(np.std([r[k] for r in rows]))} for k, label in keys])

    m = {k: float(np.mean([r[k] for r in rows])) for k, _ in keys}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    ax1.bar(["policy", "UB_greedy", "UB_loose"], [m["policy_rescue"], m["ub_greedy"], m["ub_loose"]],
            color=["#E69F00", "#56B4E9", "#009E73"])
    for i, v in enumerate([m["policy_rescue"], m["ub_greedy"], m["ub_loose"]]):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_ylim(0, 1.08); ax1.set_ylabel("rescue ratio (/ N_T)"); ax1.set_title("Policy vs upper bounds", fontsize=10)
    parts = [("a_infeasible_ratio", "(a) infeasible", "#999999"), ("b_undiscovered_ratio", "(b) undiscovered", "#CC79A7"),
             ("c_unassigned_ratio", "(c) unassigned", "#F0E442"), ("d_late_ratio", "(d) assigned, late", "#D55E00"),
             ("unresolved_ratio", "unresolved", "#0072B2")]
    left = 0.0
    for k, label, col in parts:
        ax2.barh(["expired"], [m[k]], left=left, color=col, label=f"{label} {m[k]:.3f}")
        left += m[k]
    ax2.barh(["expired"], [m["d_late_dup_ratio"]], left=left - m["d_late_ratio"] - m["unresolved_ratio"],
             color="none", edgecolor="black", hatch="xx", label=f"  (d) of which duplicate {m['d_late_dup_ratio']:.3f}")
    ax2.set_xlim(0, max(0.05, left) * 1.15); ax2.set_xlabel("fraction of victims (/ N_T)")
    ax2.set_title("Why victims expired", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"failures{tag}.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- 공통

def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500, help="seed 0..M-1")
    ap.add_argument("--checkpoint", default=None, help="주면 실패 원인 분해 (checkpoint_XXXXXX 폴더 또는 checkpoint-XX 파일)")
    ap.add_argument("--config", nargs="*",
                    help="체크포인트 없이 상한만 계산할 때 config yaml (기본 env.yaml + role.yaml + explore.yaml + rescue.yaml)")
    ap.add_argument("--n_agents", type=int, default=None,
                    help="UAV 수 (기본 = 체크포인트 / yaml 값). 센서 / 통신 범위는 첫 UAV 값 복제")
    args = ap.parse_args()
    seeds = list(range(args.episodes))

    ckpt, tag = None, ""
    if args.checkpoint:
        from eval_common.checkpoint_policy import load_train_config, resolve_checkpoint
        ckpt = resolve_checkpoint(os.path.join(_CALLER_CWD, os.path.expanduser(args.checkpoint)))
        cfg = load_train_config(ckpt)["env_config"]
        tag = "_" + os.path.basename(os.path.dirname(ckpt))
        print(f"env_config: checkpoint params.json ({os.path.relpath(ckpt, ROOT)})")
    else:
        from training.config_loader import load_config
        y = load_config(args.config and [os.path.join(ROOT, p) for p in args.config])
        cfg = y["env_config"]
        print(f"env_config: {', '.join(args.config) if args.config else 'env.yaml + role.yaml + explore.yaml + rescue.yaml'}")

    from eval_common.checkpoint_policy import with_n_agents
    cfg = with_n_agents(cfg, args.n_agents)
    if args.n_agents:
        tag += f"_N{args.n_agents}"
    print(f"UAVs: {len(cfg['agents'])}")
    eps, tgt_rows = run_bounds(cfg, seeds)
    report_bounds(eps, tgt_rows)
    save_bounds(eps, tgt_rows, tag)
    if ckpt:
        report_failures(run_policy(ckpt, cfg, eps), tag)
    print(f"\nsaved: {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
