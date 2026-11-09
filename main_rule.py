"""여러 정책을 같은 seed 몬테카를로로 비교 (GUI 없음, 결과 파일만 저장) — 정책별 요약 + 쌍별 짝 비교.

    python main_rule.py                                                     # rule.yaml compare.policies (기본 3 정책), seed 0-99
    python main_rule.py --seeds 0-9 --tag quick --workers 8
    python main_rule.py --policies rule marl_models/best_model/learn_learn_rule/model_v01 --labels rule_rule_rule learn_learn_rule
    python main_rule.py --pairs "learn_learn_learn:rule_rule_rule,learn_learn_learn:learn_rule_rule"

정책 = (label, path). path rule = 규칙 정책 (role · rescue · explore 모두 rule), 그 외 = 모델 경로 (best 모델 폴더
best_model/<조합>/model_vXX 또는 학습 체크포인트) — eval_common/model_loader.py 가 구성요소 (role / rescue / explore: rule | learn)
를 정하고 rule 인 구성요소는 env 가 규칙 정책으로 결정한다. 시작 전에 모든 모델을 로드해 보고 하나라도 실패하면 바로 종료.
--policies 를 주면 rule.yaml 목록을 대체 (--labels 를 안 주면 label = 조합 이름, 주면 개수가 같아야 함).
--pairs all (기본, 모든 쌍: 목록 뒤 정책 - 앞 정책) | "A:B,C:D" (A - B).
평가: (정책 × seed) 작업을 한 워커 풀 (spawn, CPU, torch 스레드 1, eval_common/parallel.py) 에서 실행 — 같은 seed 는 모든 정책에서
같은 맵 · 생존자 · depot (초기 상태 지문 확인). 결과는 정책 · seed 순 정렬 — workers 수와 무관하게 같다.
통계 (rule_evaluation/compare.py, eval_common/stats.py): 정책별 평균 · 표준편차 · bootstrap 95% CI, 쌍별 seed 차이 평균 · CI,
win / tie / loss, Wilcoxon signed-rank p (차이가 전부 0 이면 1), Holm 보정 p (같은 지표 안에서 쌍 수만큼), Cohen's d_z.
설정: 실행 설정 = rule_evaluation/config/rule.yaml (CLI 인자가 우선). 평가 env = rule_evaluation/config/env_rule.yaml →
role_rule.yaml → task_rule.yaml 병합 (training/config 와 같은 값으로 유지, tests/role/test_config.py) — 모든 정책에 같은 env.
결과: results/rule_evaluation/<YYYYMMDD_HHMMSS>_<tag>/  summary.txt (콘솔 출력과 같음), summary.csv, pairwise.csv, per_seed.csv,
      progress.csv (정책 · seed 별 진행률 10 % 구간 누적 발견 · 구조율, 역할 비율, 동시 추격 수), table_main.tex,
      table_conflict.tex (할당 충돌), figures/fig_compare_{bar,paired,seed,cumulative,roles,conflict,conflict_time}.pdf|png|csv,
      config.yaml, command.txt, meta.json
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
import shlex  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "rule_evaluation", "config", "rule.yaml")
PER_SEED_KEYS = ("n_targets",)                          # per_seed.csv 에 지표와 같이 남기는 열


def _cell(r, d=3):
    return f"{r['mean']:.{d}f} ± {r['std']:.{d}f} [{r['ci_lo']:.{d}f}, {r['ci_hi']:.{d}f}]" if r["mean"] == r["mean"] else "--"


def summary_text(policies, summary, pw, metrics, lower, reference, n, boot):
    """summary.txt / 콘솔 — 정책 표 (지표 × 정책) + 짝 비교 표 (지표 · 쌍). 지표 이름 뒤 방향: (+) 클수록 좋음, (-) 작을수록
    좋음 (W = 더 적음), (ref) 참고 지표 (짝 비교 안 함)."""
    tag = lambda m: f"{m} ({'ref' if m in reference else '-' if m in lower else '+'})"
    from eval_common.stats import HAS_SCIPY
    labels = [p["label"] for p in policies]
    lines = [f"policies (n = {n} seeds, mean ± std [95% bootstrap CI, {boot['n']} resamples])"]
    lines += [f"  {p['label']:<20} {p['combo']} ({p['describe']})  {p['model']}"
              + (f"  iteration {p['iteration']}" if p["iteration"] is not None else "") for p in policies]
    w = 34
    lines += ["", f"{'metric (+ higher / - lower is better)':<46}" + "".join(f"{lab:>{w}}" for lab in labels)]
    for m in metrics:
        d = 1 if m in ("episode_len", "discover_to_rescue_mean", "wasted_steps") else 3
        by = {r["policy"]: r for r in summary if r["metric"] == m}
        lines.append(f"{tag(m):<46}" + "".join(f"{_cell(by[lab], d):>{w}}" for lab in labels))
    if pw:
        lines += ["", "paired differences A - B on the same seeds (W/T/L = A better / equal / worse — for (-) metrics better = "
                      "fewer; Holm over pairs per metric)",
                  f"{'metric':<46}{'pair':<40}{'Δ mean [95% CI]':>30}{'W/T/L':>12}{'p':>9}{'p_holm':>9}{'d_z':>8}"]
        for m in metrics:
            for r in (x for x in pw if x["metric"] == m):
                ci = f"{r['mean']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]" if r["mean"] == r["mean"] else "--"
                fp = lambda x: "n/a" if x != x else f"{x:.4f}"
                dz = "--" if r["d_z"] != r["d_z"] else f"{r['d_z']:+.2f}"
                lines.append(f"{tag(m):<46}{r['pair']:<40}{ci:>30}{str(r['win']) + '/' + str(r['tie']) + '/' + str(r['loss']):>12}"
                             f"{fp(r['p_wilcoxon']):>9}{fp(r['p_holm']):>9}{dz:>8}")
        if not HAS_SCIPY:
            lines.append("(Wilcoxon p 생략: scipy 없음)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="multi-policy Monte-Carlo comparison on the same seeds",
                                 epilog="examples:\n"
                                        "  python main_rule.py                                    # rule.yaml compare.policies\n"
                                        "  python main_rule.py --policies rule <model> --labels rule_rule_rule mine\n"
                                        "  python main_rule.py --pairs \"learn_learn_learn:rule_rule_rule\" --seeds 0-9",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--policies", nargs="+", default=None,
                    help="정책 경로 목록 (rule | 모델 경로) — rule.yaml compare.policies 를 대체")
    ap.add_argument("--labels", nargs="+", default=None, help="정책 이름 (--policies 와 개수 같게, 없으면 조합 이름)")
    ap.add_argument("--pairs", default=None, help="all | \"A:B,C:D\" (A - B, rule.yaml compare.pairs)")
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
    from training.config_loader import RULE_FILES

    with open(os.path.join(_CALLER_CWD, args.config) if not os.path.isabs(args.config) else args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k in ("seeds", "mode", "tag", "workers"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cmp_cfg = dict(cfg.get("compare") or {})
    if args.labels and not args.policies:
        raise SystemExit("--labels 는 --policies 와 같이 주세요")
    if args.policies:
        if args.labels and len(args.labels) != len(args.policies):
            raise SystemExit(f"--labels {len(args.labels)} 개 != --policies {len(args.policies)} 개")
        cmp_cfg["policies"] = [{"label": args.labels[i] if args.labels else None,
                                "path": p if p == compare.RULE else os.path.join(_CALLER_CWD, os.path.expanduser(p))}
                               for i, p in enumerate(args.policies)]
    if args.pairs:
        cmp_cfg["pairs"] = args.pairs
    if not cmp_cfg.get("policies"):
        raise SystemExit("비교할 정책이 없음 — rule.yaml compare.policies 또는 --policies")
    try:
        policies = compare.resolve_policies(cmp_cfg["policies"])
        labels = [p["label"] for p in policies]
        pairs = compare.parse_pairs(cmp_cfg.get("pairs", "all"), labels)
    except ValueError as e:
        raise SystemExit(f"[rule] {e}")
    cmp_cfg["policies"] = [{"label": p["label"], "path": p["model"]} for p in policies]
    cmp_cfg["pairs"] = [list(x) for x in pairs]
    cfg["compare"] = cmp_cfg
    boot = dict(cfg.get("bootstrap") or {})
    if args.n_boot:
        boot["n"] = args.n_boot
    cfg["bootstrap"] = boot
    seeds = parse_seeds(str(cfg["seeds"]))
    workers = int(cfg.get("workers") or default_workers())
    metrics = list(cfg["metrics"])
    lower = set(cfg.get("lower_is_better") or [])
    reference = set(cfg.get("reference_metrics") or [])
    t0 = time.time()

    out = make_run_dir("rule_evaluation", cfg.get("tag") or "compare")
    print(f"[rule] {len(policies)} policies  seeds {cfg['seeds']} ({len(seeds)})  mode {cfg['mode']}  workers {workers}  -> "
          f"{os.path.relpath(out, ROOT)}")
    for p in policies:
        print(f"       {p['label']:<20} {p['combo']} ({p['describe']})  {p['model']}"
              + (f"  iteration {p['iteration']}" if p["iteration"] is not None else ""))
    print(f"       pairs: {', '.join(f'{a} - {b}' for a, b in pairs) or '-'}")
    try:
        per, ts, env_cfg, warnings = compare.run_parallel(policies, seeds, workers, cfg.get("mode", "greedy"))
    except ValueError as e:
        raise SystemExit(f"[rule] {e}")
    if warnings:
        print(f"[rule] 모델 값으로 고정한 키: {'; '.join(warnings)}")
    summary = compare.summarize(per, labels, metrics, boot)
    pw = compare.pairwise(per, pairs, [m for m in metrics if m not in reference], lower, boot)
    prog = compare.progress_bins(ts)
    pstats = compare.progress_stats(prog, labels, boot)

    order = {lab: k for k, lab in enumerate(labels)}
    per_seed = sorted(({"seed": r["seed"], "policy": r["policy"], **{k: r[k] for k in PER_SEED_KEYS + tuple(metrics)}}
                       for r in per), key=lambda r: (r["seed"], order[r["policy"]]))
    write_csv(os.path.join(out, "per_seed.csv"), per_seed)
    write_csv(os.path.join(out, "summary.csv"), summary)
    write_csv(os.path.join(out, "pairwise.csv"), pw)
    write_csv(os.path.join(out, "progress.csv"), sorted(prog, key=lambda r: (r["seed"], order[r["policy"]], r["bin"])))
    plots.table_main(summary, pw, labels, pairs, lower, os.path.join(out, "table_main.tex"), len(seeds))
    plots.table_conflict(summary, pw, labels, pairs, lower, os.path.join(out, "table_conflict.tex"), len(seeds))
    figs = []
    fig_cfg = cfg.get("figures") or {}
    if fig_cfg.get("enabled", True) and not args.no_figs:
        figs = plots.make_all(per, summary, pw, pstats, labels, pairs, out, fig_cfg)
    text = summary_text(policies, summary, pw, metrics, lower, reference, len(seeds), boot)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    with open(os.path.join(out, "command.txt"), "w") as f:
        f.write(f"cd {shlex.quote(_CALLER_CWD)}\n{shlex.join([os.path.basename(sys.executable), *sys.argv])}\n")
    write_json(os.path.join(out, "meta.json"), {
        "script": "main_rule.py", "argv": sys.argv[1:], "policies": policies, "pairs": pairs,
        "env_config_files": [os.path.relpath(f, ROOT) for f in RULE_FILES], "env_config_used": env_cfg,
        "forced_checkpoint_keys": warnings, "git": git_info(), "seeds": seeds, "n_episodes_per_policy": len(seeds),
        "workers": workers,
        "statistics": f"mean ± std, 95% bootstrap CI ({boot['n']} resamples, seed {boot['seed']}), n = {len(seeds)} seeds per "
                      f"policy; paired difference A - B on the same seeds; Wilcoxon signed-rank "
                      f"{'(scipy)' if HAS_SCIPY else 'skipped (scipy not installed)'} (p = 1 if all differences are 0), "
                      f"Holm correction over pairs within each metric; Cohen's d_z = mean / std of differences",
        "figures": [os.path.relpath(f, out) for f in figs], "elapsed_s": time.time() - t0})
    print("\n" + text)
    print(f"[rule] saved {os.path.relpath(out, ROOT)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
