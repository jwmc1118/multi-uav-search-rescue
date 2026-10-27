"""확장성 실험: 체크포인트 하나로 UAV 수 N 을 바꿔가며 평가 (학습 때와 다른 N 실행).

    python scale_evaluation/scale_eval.py <checkpoint>                          # N = 2,3,4,5,6,8, seed 0..199
    python scale_evaluation/scale_eval.py <checkpoint> --n 2,4,8 --seeds 0-49

행동 선택 = 세 헤드 greedy (evaluate 기본값), 에피소드 k 는 seed k 로 맵 / 조난자 / 골든타임 고정
-> N 끼리 같은 상황에서 비교된다. 센서 / 통신 범위는 체크포인트 첫 UAV 값 복제, n_hop 등 나머지는 체크포인트 값.
결과: <repo>/results/scale_eval/  scale_<ckpt>.csv (N 별 평균 / 표준편차), scale_<ckpt>.png (선그래프)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
_CALLER_CWD = os.getcwd()
os.chdir(ROOT)

import argparse
import csv

import numpy as np

OUT_DIR = os.path.join(ROOT, "results", "scale_eval")
METRICS = [("rescue_rate", "rescue rate"), ("discover_rate", "discover rate"), ("explored", "explored ratio"),
           ("same_cell", "same-cell overlaps / episode"), ("episode_len", "episode length")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--n", default="2,3,4,5,6,8", help="UAV 수 목록")
    ap.add_argument("--seeds", default="0-199", help="'0-199' / '0,3,7'")
    args = ap.parse_args()

    from env.role.simulation import RoleSimulation
    from eval_common.checkpoint_policy import CheckpointPolicy, load_train_config, resolve_checkpoint, with_n_agents
    from eval_common.episode import parse_seeds, run_episode

    ckpt = resolve_checkpoint(os.path.join(_CALLER_CWD, os.path.expanduser(args.checkpoint)))
    train_env = load_train_config(ckpt)["env_config"]
    ns = [int(x) for x in args.n.split(",")]
    seeds = parse_seeds(args.seeds)
    print(f"checkpoint: {os.path.relpath(ckpt, ROOT)}  trained N={len(train_env['agents'])}  "
          f"N {ns}  episodes {len(seeds)} each")

    rows = []
    for n in ns:
        policy = CheckpointPolicy(ckpt, env_config=with_n_agents(train_env, n))
        sim = RoleSimulation(policy.env_config)
        metrics = METRICS
        eps = []
        for k, s in enumerate(seeds):
            eps.append(run_episode(sim, policy, s))
            print(f"\r  N={n}: {k + 1}/{len(seeds)}", end="", flush=True)
        print()
        row = {"n_agents": n, "episodes": len(seeds)}
        for m, _ in metrics:
            v = np.array([e[m] for e in eps], float)
            row[f"{m}_mean"], row[f"{m}_std"] = float(v.mean()), float(v.std())
        rows.append(row)
        policy.trainer.stop()

    print(f"\n{'N':>3}" + "".join(f"{label:>24}" for _, label in metrics))
    for r in rows:
        cells = ["{:.3f} ± {:.3f}".format(r[m + "_mean"], r[m + "_std"]) for m, _ in metrics]
        print(f"{r['n_agents']:>3}" + "".join(f"{c:>24}" for c in cells))

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = os.path.basename(os.path.dirname(ckpt))
    with open(os.path.join(OUT_DIR, f"scale_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trained = len(train_env["agents"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.0))
    for ax, (m, label) in zip(axes, metrics):
        x = [r["n_agents"] for r in rows]
        ax.errorbar(x, [r[m + "_mean"] for r in rows], yerr=[r[m + "_std"] for r in rows], marker="o", capsize=3,
                    color="#0072B2")
        ax.axvline(trained, color="0.5", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xlabel("UAVs (N)"); ax.set_title(label, fontsize=10); ax.grid(alpha=0.25)
        if m.endswith("rate") or m == "explored":
            ax.set_ylim(0, 1.02)
    fig.suptitle(f"Scalability: {tag} (trained N={trained}, dotted), {len(seeds)} seeds per N", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT_DIR, f"scale_{tag}.png"), dpi=200)
    print(f"saved: {os.path.relpath(OUT_DIR, ROOT)}/scale_{tag}.csv / .png")


if __name__ == "__main__":
    main()
