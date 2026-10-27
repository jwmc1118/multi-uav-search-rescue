"""규칙 vs 학습 모델 평가의 seed 단위 병렬 실행 (main_rule / main_scale).

    run_tasks(checkpoint, mode, tasks, workers)   tasks = [Task(key, env_config, free_keys, seed, timeseries)]
        -> [Result(key, seed, per [rule 행, model 행], ts, env_used)] (key, seed 순 정렬)

워커 (multiprocessing spawn) 는 시작할 때 CUDA_VISIBLE_DEVICES="" · torch.set_num_threads(1) 로 맞추고 체크포인트 가중치를
한 번만 읽는다 (eval_common/light_policy.py — ray.init / trainer 없음). 받은 task 마다 그 env_config 로 모델을 준비하고
(같은 관측 shape 면 재사용 — main_scale 은 한 워커가 여러 N 을 처리) 규칙 정책 -> 모델 순으로 같은 seed 를 실행해
초기 상태 지문이 같은지 확인한다. 결과는 seed 순으로 정렬하므로 workers 수와 무관하게 같다 (추론 시간 열 제외).
workers = 1 이면 같은 함수를 현재 프로세스에서 실행 (torch 스레드 1).
"""
import json
import multiprocessing as mp
import os
import sys
import time
from collections import namedtuple

Task = namedtuple("Task", "key env_config free_keys seed timeseries")
Result = namedtuple("Result", "key seed per ts env_used")

_W = {}


def default_workers():
    return max(1, (os.cpu_count() or 1) - 2)


def _init(checkpoint, mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import signal

    import torch
    torch.set_num_threads(1)
    import ray.rllib.models  # noqa: F401  (ray import 가 SIGTERM 핸들러를 바꾼다 -> 풀 종료 때 traceback, 기본값으로 되돌림)
    from eval_common.light_policy import LightPolicy
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _W["policy"] = LightPolicy(checkpoint, mode=mode)
    _W["sims"] = {}


def _run(task):
    from env.role.simulation import RoleSimulation
    from eval_common.episode import RulePolicy, run_episode_detailed
    model = _W["policy"]
    env = model.prepare(task.env_config, task.free_keys)
    key = json.dumps(env, sort_keys=True, default=str)
    if key not in _W["sims"]:
        _W["sims"] = {key: RoleSimulation(env)}                 # 최근 env 하나만 유지 (main_scale 의 큰 맵 메모리)
    sim = _W["sims"][key]
    rule = RulePolicy()
    rule.name = "rule"
    per, ts, sig = [], [], None
    for name, pol in (("rule", rule), ("model", model)):
        m, rows, s = run_episode_detailed(sim, pol, task.seed, timeseries=task.timeseries)
        assert sig is None or s == sig, f"seed {task.seed}: 정책 {name} 의 초기 상태가 규칙 정책과 다름 (맵 / 조난자 / 골든타임)"
        sig = s
        per.append(dict(policy=name, seed=task.seed, **m))
        ts += [dict(policy=name, seed=task.seed, **r) for r in rows]
    return Result(task.key, task.seed, per, ts, env)


def _fmt_t(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def run_tasks(checkpoint, mode, tasks, workers, label="eval"):
    """반환 [Result] — (key, seed) 순. 진행률: 완료 task / 전체, 경과, 예상 남은 시간."""
    workers = max(1, min(int(workers), len(tasks)))
    t0 = time.time()
    out = []

    def progress(k):
        el = time.time() - t0
        eta = el / k * (len(tasks) - k)
        print(f"\r  [{label}] {k}/{len(tasks)} seeds  elapsed {_fmt_t(el)}  ETA {_fmt_t(eta)}   ", end="", flush=True)

    if workers == 1:
        _init(checkpoint, mode)
        for k, t in enumerate(tasks):
            out.append(_run(t))
            progress(k + 1)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init, initargs=(checkpoint, mode)) as pool:
            for k, r in enumerate(pool.imap_unordered(_run, tasks)):
                out.append(r)
                progress(k + 1)
    print(file=sys.stdout)
    out.sort(key=lambda r: (r.key, r.seed))
    return out
