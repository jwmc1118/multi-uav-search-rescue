"""확장성 평가: UAV 수 N 과 함께 조난자 수 · 맵 크기 · 시간을 키워가며 규칙 정책 vs 학습 모델 (GUI 없음, 결과 파일만 저장).

    python main_scale.py --checkpoint marl_models/RolePPO/checkpoints/checkpoint_000200
    python main_scale.py --checkpoint <ckpt> --n 2,4,8 --seeds 0-9 --tag quick
    python main_scale.py --checkpoint <ckpt> --preset fixed          # UAV 수만 바꿈
    python main_scale.py --checkpoint <ckpt> --workers 8             # (N, seed) 단위 병렬 (기본 = CPU 코어 수 - 2)

설정: scale_evaluation/config/scale.yaml (CLI 인자가 우선). N 마다 환경 = 체크포인트 env_config -> scale.yaml env.files ->
env.overrides -> presets 크기 조정 (r = N / N_ref, scale_evaluation/scaling.py). 체크포인트 가중치를 그대로 싣고 관측 shape 를
정하는 키 (state_size / local_channels / s_max / k_max / n_sub_max) 는 체크포인트 값, rows / cols / max_targets /
max_episode_len 만 풀어 준다. N 마다 두 정책을 같은 seed 로 실행하고 seed 별 초기 상태가 같은지 확인한다.
추론 시간 (ms / 결정) 은 CPU 벽시계 (결정이 있던 step 의 모델 act() 시간 합 / 결정 수, 워커당 torch 스레드 1).
병렬: 모든 (N, seed) 를 한 워커 풀 (spawn, CPU) 에서 처리 — 워커는 체크포인트 가중치를 한 번만 읽고 N 이 바뀌면 모델만 새로
만든다 (eval_common/parallel.py, ray 없음). 결과는 N · seed 순 정렬 — workers 수와 무관하게 같다 (추론 시간 제외).
결과: results/scale_evaluation/<YYYYMMDD_HHMMSS>_<tag>_<preset>/
    scale.csv  per_episode.csv  inference.csv  scale_table.tex  figures/*.pdf|png|csv  meta.json
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

DEFAULT_CONFIG = os.path.join(ROOT, "scale_evaluation", "config", "scale.yaml")


def main():
    ap = argparse.ArgumentParser(description="scalability: rule vs learned over N (with scaled map / survivors / time)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", default=None, help="UAV 수 목록 '2,4,8' (scale.yaml n_list)")
    ap.add_argument("--seeds", default=None, help="'0-49' / '0,3,7'")
    ap.add_argument("--preset", default=None, help="scale.yaml presets 이름 (default / fixed ...)")
    ap.add_argument("--mode", default=None, choices=["greedy", "sample"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_boot", type=int, default=None, help="bootstrap 횟수 (scale.yaml bootstrap.n)")
    ap.add_argument("--workers", type=int, default=None, help="병렬 워커 수 (scale.yaml workers, null = CPU 코어 수 - 2)")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    from eval_common.episode import parse_seeds
    from eval_common.io import git_info, make_run_dir, write_csv, write_json
    from eval_common.parallel import default_workers
    from eval_common.stats import HAS_SCIPY
    from scale_evaluation import compare, plots
    from scale_evaluation.scaling import describe, scaled_env_config

    with open(os.path.join(_CALLER_CWD, args.config) if not os.path.isabs(args.config) else args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ("seeds", "preset", "mode", "tag"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.checkpoint:
        cfg["checkpoint"] = os.path.join(_CALLER_CWD, os.path.expanduser(args.checkpoint))
    if args.n:
        cfg["n_list"] = [int(x) for x in args.n.split(",")]
    if not cfg.get("checkpoint"):
        raise SystemExit("체크포인트가 없음 — --checkpoint 또는 scale.yaml checkpoint")
    if cfg["preset"] not in (cfg.get("presets") or {}):
        raise SystemExit(f"preset {cfg['preset']!r} 없음 — {sorted(cfg.get('presets') or {})}")
    rule = cfg["presets"][cfg["preset"]] or {}
    boot = dict(cfg.get("bootstrap") or {})
    if args.n_boot:
        boot["n"] = args.n_boot
    seeds = parse_seeds(str(cfg["seeds"]))
    ns = sorted(int(n) for n in cfg["n_list"])
    if args.workers is not None:
        cfg["workers"] = args.workers
    workers = int(cfg.get("workers") or default_workers())
    t0 = time.time()

    base, n_ref, ckpt = compare.base_env(cfg)
    if rule.get("map") is not None and base.get("map_mode") != "random":
        raise SystemExit(f"맵 크기 조정은 map_mode: random 에서만 (지금 {base.get('map_mode')!r}) — preset fixed 를 쓰거나 env.overrides")
    envs = {n: scaled_env_config(base, n, n_ref, rule) for n in ns}
    out = make_run_dir("scale_evaluation", f"{cfg.get('tag') or 'scale'}_{cfg['preset']}")
    print(f"[scale] checkpoint {os.path.relpath(ckpt, ROOT)}  N_ref {n_ref}  N {ns}  seeds {cfg['seeds']} ({len(seeds)})  "
          f"preset {cfg['preset']} {rule}  mode {cfg['mode']}  workers {workers}  -> {os.path.relpath(out, ROOT)}")
    for n in ns:
        d = describe(envs[n])
        print(f"  N={n:<3} map {d['rows']}x{d['cols']}  survivors {d['survivors']}  golden {d['golden']}  "
              f"T {d['max_episode_len']}  max_targets {envs[n]['max_targets']}")

    per, used, warns = compare.run_all(cfg, envs, seeds, workers)
    metrics = list(cfg["metrics"])
    summary = compare.summarize_n(per, ns, used, metrics, set(cfg.get("lower_is_better") or []), boot)
    infer = compare.infer_summary(per, ns, boot)

    write_csv(os.path.join(out, "per_episode.csv"), per)
    write_csv(os.path.join(out, "scale.csv"), summary)
    write_csv(os.path.join(out, "inference.csv"), infer)
    plots.scale_table(summary, os.path.join(out, "scale_table.tex"), len(seeds))
    figs = []
    fig_cfg = cfg.get("figures") or {}
    if fig_cfg.get("enabled", True) and not args.no_figs:
        secondary = any(rule.get(k) is not None for k in ("map", "survivors"))
        figs = plots.make_all(summary, infer, ns, n_ref, out, fig_cfg, secondary)
    write_json(os.path.join(out, "meta.json"), {
        "script": "main_scale.py", "argv": sys.argv[1:], "config": cfg, "checkpoint": ckpt, "n_ref": n_ref,
        "preset_rule": rule, "env_config_base": base, "env_config_used": {str(n): used[n] for n in ns},
        "forced_checkpoint_keys": {str(n): warns[n] for n in ns}, "git": git_info(), "seeds": seeds,
        "workers": workers, "torch_threads_per_worker": 1, "inference_device": "cpu",
        "statistics": f"per N: mean ± 95% bootstrap CI ({boot['n']} resamples, seed {boot['seed']}), n = {len(seeds)} "
                      f"episodes per policy; paired difference = learned - rule on the same (seed, N); Wilcoxon signed-rank "
                      f"{'(scipy)' if HAS_SCIPY else 'skipped (scipy not installed)'}",
        "figures": [os.path.relpath(f, out) for f in figs], "elapsed_s": time.time() - t0})

    by = {(r["n_agents"], r["metric"]): r for r in summary}
    inf = {r["n_agents"]: r for r in infer}
    print(f"\n{'N':>3}{'map':>9}{'surv':>8}{'rule rescue':>17}{'learned rescue':>17}{'Δ [95% CI]':>28}{'d_z':>7}{'p':>7}"
          f"{'UB':>7}{'r/UB r|m':>13}{'resc/UAV r|m':>14}{'grp':>6}{'iso':>6}{'exp u/n/a (r)':>17}{'trunc S|K (m)':>15}"
          f"{'ms/dec':>8}")
    notes = []
    for n in ns:
        r = by[(n, "rescue_rate")]
        g = lambda m, p="rule": by[(n, m)][f"{p}_mean"]
        p = "n/a" if r["p_wilcoxon"] != r["p_wilcoxon"] else f"{r['p_wilcoxon']:.3f}"
        dz = "n/a" if r["d_z"] != r["d_z"] else f"{r['d_z']:+.2f}"
        print(f"{n:>3}{r['rows']:>4}x{r['cols']:<4}{r['survivors']:>8}{r['rule_mean']:>9.3f}±{r['rule_std']:<6.3f}"
              f"{r['model_mean']:>9.3f}±{r['model_std']:<6.3f}"
              f"{r['diff_mean']:>+12.3f} [{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}]{dz:>7}{p:>7}"
              f"{g('ub_loose'):>7.3f}{g('rescue_over_ub'):>7.3f}|{g('rescue_over_ub', 'model'):<5.3f}"
              f"{g('rescued_per_uav'):>8.2f}|{g('rescued_per_uav', 'model'):<5.2f}"
              f"{g('group_size_mean'):>6.2f}{g('isolated_frac'):>6.2f}"
              f"{g('expire_undiscovered_rate'):>7.3f}/{g('expire_unassigned_rate'):.3f}/{g('expire_assigned_rate'):.3f}"
              f"{g('trunc_frac_S', 'model'):>8.3f}|{g('trunc_frac_K', 'model'):<6.3f}{inf[n]['infer_ms_mean']:>8.2f}")
        if r["p_note"] or r["unanimous"]:
            notes.append(f"N={n}: " + "; ".join(x for x in (r["p_note"], f"all favour {r['unanimous']}" if r["unanimous"] else "") if x))
    print("(UB = UB_loose, r/UB = 구조율 / UB_loose, grp = 통신 그룹 평균 크기 (규칙), iso = 고립 UAV 비율 (규칙), "
          "exp u/n/a = 규칙의 만료 원인 미발견 / 미배정 / 배정 중 (N_T 대비), trunc S|K = 모델의 잘림 결정 비율)")
    for x in notes:
        print("  " + x)
    if not HAS_SCIPY:
        print("(Wilcoxon p-value 생략: scipy 없음)")
    print(f"\n[scale] saved {os.path.relpath(out, ROOT)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
