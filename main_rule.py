"""규칙 정책 vs 학습 모델 몬테카를로 비교 (GUI 없음, 결과 파일만 저장).

    python main_rule.py --checkpoint marl_models/RolePPO/checkpoints/checkpoint_000200
    python main_rule.py --checkpoint <ckpt> --seeds 0-9 --mode sample --tag quick
    python main_rule.py --config rule_evaluation/config/rule.yaml --no-figs
    python main_rule.py --checkpoint <ckpt> --workers 8          # seed 단위 병렬 (기본 = CPU 코어 수 - 2)

설정: rule_evaluation/config/rule.yaml (CLI 인자가 우선). 환경 = 체크포인트 env_config -> rule.yaml env.files -> env.overrides
(관측 shape 키는 체크포인트 값). 두 정책 (규칙 = policies/role_rules.py RuleDecider, 모델 = 체크포인트) 을 같은 seed 목록으로
실행하고 seed 별 초기 상태 (맵 / depot / 조난자 / 골든타임) 가 같은지 확인한다.
병렬: seed 단위로 나눠 워커 (spawn, CPU, torch 스레드 1) 가 체크포인트 가중치를 한 번만 읽어 처리 (eval_common/parallel.py,
ray 없음). 결과는 seed 순 정렬 — workers 수와 무관하게 같다 (추론 시간 제외).
결과: results/rule_evaluation/<YYYYMMDD_HHMMSS>_<tag>/
    per_episode.csv  timeseries.csv  summary.csv  summary_table.tex  figures/*.pdf|png|csv  meta.json
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # 평가는 CPU 만 (학습이 GPU 를 쓰는 중이어도 방해하지 않게)
_CALLER_CWD = os.getcwd()
os.chdir(ROOT)

import argparse  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "rule_evaluation", "config", "rule.yaml")


def main():
    ap = argparse.ArgumentParser(description="rule vs learned Monte-Carlo comparison")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default="marl_models/RolePPO/checkpoints/checkpoint_000220")
    ap.add_argument("--seeds", default=None, help="'0-99' / '0,3,7'")
    ap.add_argument("--mode", default=None, choices=["greedy", "sample"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_boot", type=int, default=None, help="bootstrap 횟수 (rule.yaml bootstrap.n)")
    ap.add_argument("--workers", type=int, default=None, help="병렬 워커 수 (rule.yaml workers, null = CPU 코어 수 - 2)")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    from eval_common.episode import parse_seeds
    from eval_common.io import git_info, make_run_dir, write_csv, write_json
    from eval_common.parallel import default_workers
    from eval_common.stats import HAS_SCIPY
    from rule_evaluation import compare, plots

    with open(os.path.join(_CALLER_CWD, args.config) if not os.path.isabs(args.config) else args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ("checkpoint", "seeds", "mode", "tag"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.checkpoint:
        cfg["checkpoint"] = os.path.join(_CALLER_CWD, os.path.expanduser(args.checkpoint))
    if not cfg.get("checkpoint"):
        raise SystemExit("체크포인트가 없음 — --checkpoint 또는 rule.yaml checkpoint")
    boot = dict(cfg.get("bootstrap") or {})
    if args.n_boot:
        boot["n"] = args.n_boot
    seeds = parse_seeds(str(cfg["seeds"]))
    if args.workers is not None:
        cfg["workers"] = args.workers
    workers = int(cfg.get("workers") or default_workers())
    t0 = time.time()

    out = make_run_dir("rule_evaluation", cfg.get("tag") or "rule_vs_model")
    print(f"[rule] checkpoint {cfg['checkpoint']}  seeds {cfg['seeds']} ({len(seeds)})  mode {cfg['mode']}  "
          f"workers {workers}  -> {os.path.relpath(out, ROOT)}")
    per, ts, env_cfg, warnings, ckpt = compare.run_parallel(cfg, seeds, workers)
    if warnings:
        print(f"[rule] checkpoint 값으로 고정한 키: {'; '.join(warnings)}")
    summary = compare.summarize(per, cfg["metrics"], set(cfg.get("lower_is_better") or []), boot)
    fig_cfg = cfg.get("figures") or {}
    tstats = compare.timeseries_stats(ts, boot, fig_cfg.get("timeseries_max_step"))
    conv = compare.convergence(per, boot)

    write_csv(os.path.join(out, "per_episode.csv"), per)
    write_csv(os.path.join(out, "timeseries.csv"), ts)
    write_csv(os.path.join(out, "summary.csv"), summary)
    plots.summary_table(summary, os.path.join(out, "summary_table.tex"), len(seeds))
    figs = []
    if fig_cfg.get("enabled", True) and not args.no_figs:
        figs = plots.make_all(per, summary, tstats, conv, out, fig_cfg)
    write_json(os.path.join(out, "meta.json"), {
        "script": "main_rule.py", "argv": sys.argv[1:], "config": cfg, "checkpoint": ckpt,
        "env_config_used": env_cfg, "forced_checkpoint_keys": warnings, "git": git_info(),
        "seeds": seeds, "n_episodes_per_policy": len(seeds), "workers": workers,
        "inference_timing": "CPU, torch 1 thread per worker, model act() wall time / decisions",
        "statistics": f"mean ± 95% bootstrap CI ({boot['n']} resamples, seed {boot['seed']}), n = {len(seeds)} episodes "
                      f"per policy; paired difference = learned - rule on the same seeds; Wilcoxon signed-rank "
                      f"{'(scipy)' if HAS_SCIPY else 'skipped (scipy not installed)'}",
        "figures": [os.path.relpath(f, out) for f in figs], "elapsed_s": time.time() - t0})

    print(f"\n{'metric':<26}{'rule':>20}{'learned':>20}{'Δ [95% CI]':>30}{'win/tie/loss':>16}{'d_z [95% CI]':>24}{'p':>8}")
    for r in summary:
        ci = f"{r['diff_mean']:+.3f} [{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}]"
        wtl = f"{r['model_win']:.2f}/{r['tie']:.2f}/{r['model_loss']:.2f}"
        p = "n/a" if r["p_wilcoxon"] != r["p_wilcoxon"] else f"{r['p_wilcoxon']:.3f}"
        dz = "n/a" if r["d_z"] != r["d_z"] else f"{r['d_z']:+.2f} [{r['d_z_ci_lo']:+.2f}, {r['d_z_ci_hi']:+.2f}]"
        note = "  " + "; ".join(x for x in (r["p_note"], f"all favour {r['unanimous']}" if r["unanimous"] else "") if x)
        print(f"{r['metric']:<26}{r['rule_mean']:>10.3f} ± {r['rule_std']:<7.3f}{r['model_mean']:>10.3f} ± "
              f"{r['model_std']:<7.3f}{ci:>30}{wtl:>16}{dz:>24}{p:>8}{note.rstrip()}")
    if not HAS_SCIPY:
        print("(Wilcoxon p-value 생략: scipy 없음)")
    print(f"\n[rule] saved {os.path.relpath(out, ROOT)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
