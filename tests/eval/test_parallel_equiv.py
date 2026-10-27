"""병렬 평가 = 순차 평가 점검 (eval_common/parallel.py). CPU 만, ray 불필요.

    python -m tests.eval.test_parallel_equiv --checkpoint <checkpoint_XXXXXX 폴더 또는 파일> [--workers 4]

[1] main_rule 경로 (rule_evaluation.compare.run_parallel), seed 0-9: workers = 1 과 workers = W 의 per_episode / timeseries 행이
    완전히 같다 (추론 시간 infer_ms_per_decision 제외), 행 순서 = 정책 · seed 순
[2] main_scale 경로 (scale_evaluation.compare.run_all), N = 2, 5 · seed 0-3: 같은 비교 (한 워커가 여러 N 처리)
[3] 경량 정책 (LightPolicy) = ray CheckpointPolicy: seed 0-1 greedy 에서 지표 · 시계열 · 초기 상태 지문이 같다 (--skip-ray 로 생략)
--checkpoint 를 안 주면 marl_models/RolePPO/checkpoints/best/checkpoint_* 를 쓴다.
"""
import argparse
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

SKIP = ("infer_ms_per_decision",)


def _strip(rows):
    return [{k: v for k, v in r.items() if k not in SKIP} for r in rows]


def _same(a, b):
    """NaN 끼리는 같다고 본다."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.keys() != y.keys():
            return False
        for k in x:
            u, v = x[k], y[k]
            if isinstance(u, float) and isinstance(v, float) and u != u and v != v:
                continue
            if u != v:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-ray", action="store_true")
    args = ap.parse_args()
    ckpt = args.checkpoint or sorted(glob.glob("marl_models/RolePPO/checkpoints/best/checkpoint_*"))[-1]
    ckpt = os.path.abspath(ckpt)
    from rule_evaluation import compare as rc
    from scale_evaluation import compare as sc
    from scale_evaluation.scaling import scaled_env_config
    cfg = {"checkpoint": ckpt, "mode": "greedy", "env": {}}

    seeds = list(range(10))
    a = rc.run_parallel(cfg, seeds, 1, label="w=1")
    b = rc.run_parallel(cfg, seeds, args.workers, label=f"w={args.workers}")
    assert _same(_strip(a[0]), _strip(b[0])), "per_episode 가 workers 수에 따라 다름"
    assert _same(a[1], b[1]), "timeseries 가 workers 수에 따라 다름"
    assert [(r["policy"], r["seed"]) for r in a[0]] == [(p, s) for p in rc.POLICIES for s in seeds]
    print(f"[1] main_rule path: workers 1 == {args.workers} ({len(a[0])} episodes, {len(a[1])} timeseries rows)")

    base, n_ref, _ = sc.base_env(cfg)
    rule = {"survivors": 1.0, "map": 0.5, "golden_time": 0.5, "max_episode_len": 0.5}
    envs = {n: scaled_env_config(base, n, n_ref, rule) for n in (2, 5)}
    a = sc.run_all(cfg, envs, list(range(4)), 1)
    b = sc.run_all(cfg, envs, list(range(4)), args.workers)
    assert _same(_strip(a[0]), _strip(b[0])) and a[1] == b[1], "main_scale 결과가 workers 수에 따라 다름"
    print(f"[2] main_scale path: workers 1 == {args.workers} (N 2, 5 x 4 seeds, {len(a[0])} episodes)")

    if not args.skip_ray:
        from env.role.simulation import RoleSimulation
        from eval_common.checkpoint_policy import CheckpointPolicy, resolve_checkpoint
        from eval_common.episode import run_episode_detailed
        from eval_common.light_policy import LightPolicy
        heavy = CheckpointPolicy(resolve_checkpoint(ckpt), env_config=base, mode="greedy")
        light = LightPolicy(resolve_checkpoint(ckpt), mode="greedy")
        assert light.prepare(base) == heavy.env_config
        sim = RoleSimulation(heavy.env_config)
        for s in (0, 1):
            x, tx, gx = run_episode_detailed(sim, heavy, s)
            y, ty, gy = run_episode_detailed(sim, light, s)
            assert _same(_strip([x]), _strip([y])) and tx == ty and gx == gy, f"seed {s}: LightPolicy != CheckpointPolicy"
        heavy.trainer.stop()
        print("[3] LightPolicy == CheckpointPolicy (ray trainer), seeds 0-1 greedy")
    print("test_parallel_equiv: all ok")


if __name__ == "__main__":
    main()
