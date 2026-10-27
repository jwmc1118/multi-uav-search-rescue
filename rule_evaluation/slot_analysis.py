"""S (생존자) / K (frontier 덩어리) 슬롯 잘림 분석 — 규칙 정책만 (ray / 체크포인트 불필요).

    python -m rule_evaluation.slot_analysis                              # 학습 설정 (training/config), seed 0-49, s_max 16
    python -m rule_evaluation.slot_analysis --s_max 16,24,32,48 --seeds 0-49 --workers 8

환경 = training.config_loader.load_config() 의 env_config (학습 설정) 에 s_max 만 바꿔서 실행 (규칙 정책도 rescue 후보를
victim_slots (BFS 가까운 순 상위 s_max) 안에서 고르므로 s_max 가 행동에 영향). 지표는 eval_common/episode.py slot_snapshot.
출력 (콘솔 표 + results/rule_evaluation/<시각>_slots/):
    per_episode.csv  decisions.csv (결정마다 S / K 잘림)  tables.md (s_max 별 요약 / 조난자 수 구간별 / 시간대별)
"""
import argparse
import copy
import multiprocessing as mp
import os
import time

import numpy as np

T_BINS = ((20, 29), (30, 39), (40, 50))
PHASES = (("early", 0.0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("late", 2 / 3, 1.01))


def _work(args):
    env_cfg, seeds = args
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from env.role.simulation import RoleSimulation
    from eval_common.episode import RulePolicy, run_episode_detailed
    sim, pol = RoleSimulation(env_cfg), RulePolicy()
    pol.name = "rule"
    per, dec = [], []
    for s in seeds:
        m, _, _ = run_episode_detailed(sim, pol, s, timeseries=False, decision_log=dec)
        per.append(dict(seed=s, s_max=env_cfg["s_max"], **m))
    for d in dec:
        d["s_max"] = env_cfg["s_max"]
    return per, dec


def _mean(rows, k):
    x = np.array([r[k] for r in rows], float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def _dec_stats(dec):
    """결정 행 -> S (역할 결정) / K (1 단계 결정) 요약."""
    r = [d for d in dec if d["role"]]
    k = [d for d in dec if d["s1"]]
    nr, nk = max(1, len(r)), max(1, len(k))
    return {"role_dec": len(r), "trunc_frac_S": sum(d["dropped"] > 0 for d in r) / nr,
            "dropped_S_per_dec": sum(d["dropped"] for d in r) / nr,
            "dropped_S_per_trunc": sum(d["dropped"] for d in r) / max(1, sum(d["dropped"] > 0 for d in r)),
            "urgent_per_dec": sum(d["urgent"] for d in r) / nr, "best_dropped_frac": sum(d["best_dropped"] for d in r) / nr,
            "n_alloc_mean": sum(d["n_alloc"] for d in r) / nr, "n_alloc_max": max((d["n_alloc"] for d in r), default=0),
            "s1_dec": len(k), "trunc_frac_K": sum(d["k_dropped"] > 0 for d in k) / nk,
            "dropped_K_per_dec": sum(d["k_dropped"] for d in k) / nk, "dropped_K_cells_per_dec": sum(d["k_dropped_cells"] for d in k) / nk}


def _table(title, header, rows):
    out = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(c if isinstance(c, str) else f"{c:.3f}" for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_max", default="16")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    ap.add_argument("--tag", default="slots")
    args = ap.parse_args()
    from eval_common.episode import parse_seeds
    from eval_common.io import git_info, make_run_dir, write_csv, write_json
    from training.config_loader import load_config

    base = load_config()["env_config"]
    base = dict(base, events=dict(base.get("events") or {}, log_dir=None))
    seeds = parse_seeds(args.seeds)
    s_list = [int(x) for x in args.s_max.split(",")]
    out = make_run_dir("rule_evaluation", args.tag)
    t0 = time.time()
    per, dec = [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        jobs = [(dict(copy.deepcopy(base), s_max=S), seeds[w::args.workers]) for S in s_list for w in range(args.workers)
                if seeds[w::args.workers]]
        for k, (p, d) in enumerate(pool.imap_unordered(_work, jobs)):
            per += p
            dec += d
            print(f"\r  {k + 1}/{len(jobs)} chunks  {time.time() - t0:.0f}s", end="", flush=True)
    print()
    per.sort(key=lambda r: (r["s_max"], r["seed"]))
    dec.sort(key=lambda r: (r["s_max"], r["seed"], r["time"], r["uav"]))
    write_csv(os.path.join(out, "per_episode.csv"), per)
    write_csv(os.path.join(out, "decisions.csv"), [{k: v for k, v in d.items()} for d in dec])

    md = [f"N = {len(base['agents'])}, num_targets {base['num_targets']}, max_episode_len {base['max_episode_len']}, "
          f"k_max {base.get('cluster', {}).get('k_max')}, seeds {args.seeds} ({len(seeds)}), rule policy\n"]
    hdr = ["s_max", "rescue", "expire", "role dec/ep", "trunc S frac", "dropped S/dec", "dropped S/trunc dec",
           "urgent/dec", "EDF dropped frac", "n_alloc mean", "n_alloc max", "S-node overflow/ep",
           "trunc K frac", "dropped K/dec", "dropped K cells/dec", "reachable==dropped"]
    rows = []
    for S in s_list:
        p = [r for r in per if r["s_max"] == S]
        d = _dec_stats([x for x in dec if x["s_max"] == S])
        same = all(r["dropped_reachable_S"] == r["dropped_S"] for r in p)
        rows.append([str(S), _mean(p, "rescue_rate"), _mean(p, "expire_rate"), _mean(p, "role_decisions"),
                     d["trunc_frac_S"], d["dropped_S_per_dec"], d["dropped_S_per_trunc"], d["urgent_per_dec"],
                     d["best_dropped_frac"], d["n_alloc_mean"], str(d["n_alloc_max"]), _mean(p, "s_nodes_overflow"),
                     d["trunc_frac_K"], d["dropped_K_per_dec"], d["dropped_K_cells_per_dec"], str(same)])
    md.append(_table("s_max 별 (에피소드 평균 / 결정 단위 비율)", hdr, rows))
    S0 = s_list[0]
    d0 = [x for x in dec if x["s_max"] == S0]
    p0 = [r for r in per if r["s_max"] == S0]
    hdr2 = ["bin", "episodes", "rescue", "trunc S frac", "dropped S/dec", "urgent/dec", "EDF dropped frac", "n_alloc mean",
            "trunc K frac", "dropped K/dec"]
    rows = []
    for lo, hi in T_BINS:
        ss = {r["seed"] for r in p0 if lo <= r["n_targets"] <= hi}
        d = _dec_stats([x for x in d0 if x["seed"] in ss])
        rows.append([f"{lo}-{hi}", str(len(ss)), _mean([r for r in p0 if r["seed"] in ss], "rescue_rate"), d["trunc_frac_S"],
                     d["dropped_S_per_dec"], d["urgent_per_dec"], d["best_dropped_frac"], d["n_alloc_mean"],
                     d["trunc_frac_K"], d["dropped_K_per_dec"]])
    md.append(_table(f"조난자 수 구간별 (s_max {S0})", hdr2, rows))
    hdr3 = ["phase (t / max_len)", "role dec", "trunc S frac", "dropped S/dec", "urgent/dec", "EDF dropped frac",
            "n_alloc mean", "s1 dec", "trunc K frac", "dropped K/dec"]
    rows = []
    for name, a, b in PHASES:
        d = _dec_stats([x for x in d0 if a <= x["time"] / x["max_episode_len"] < b])
        rows.append([name, str(d["role_dec"]), d["trunc_frac_S"], d["dropped_S_per_dec"], d["urgent_per_dec"],
                     d["best_dropped_frac"], d["n_alloc_mean"], str(d["s1_dec"]), d["trunc_frac_K"], d["dropped_K_per_dec"]])
    md.append(_table(f"에피소드 시간대별 (s_max {S0})", hdr3, rows))
    text = "\n".join(md)
    with open(os.path.join(out, "tables.md"), "w") as f:
        f.write(text)
    write_json(os.path.join(out, "meta.json"), {"script": "rule_evaluation/slot_analysis.py", "args": vars(args),
                                                "env_config": base, "git": git_info(), "elapsed_s": time.time() - t0})
    print(text)
    print(f"[slots] saved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
