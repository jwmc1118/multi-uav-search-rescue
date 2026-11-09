"""정책 평가의 seed 단위 병렬 실행 (main_rule / main_scale).

    run_tasks(models, mode, tasks, workers)   models = {정책 슬롯 이름: 모델 경로 | None (= 규칙 정책)}
        (main_rule = 정책 label, main_scale = "baseline" / "learned"),
        tasks = [Task(key, env_config, free_keys, seed, timeseries, policies)]
        -> [Result(key, seed, per [정책별 행], ts, env_used, sig)] (key, seed 순 정렬). policies 기본 (baseline, learned)

워커 (multiprocessing spawn) 는 시작할 때 CUDA_VISIBLE_DEVICES="" · torch.set_num_threads(1) 로 맞추고 모델마다 가중치를 한 번만
읽는다 (eval_common/light_policy.py — eval_common/model_loader.py 로 구성요소 모드 자동 구성, ray.init / trainer 없음).
받은 task 마다 그 env_config 로 모델을 준비하고 (같은 관측 shape 면 재사용 — main_scale 은 한 워커가 여러 N 을 처리)
policies 순서대로 같은 seed 를 실행해 초기 상태 지문이 같은지 확인한다. 정책마다 그 구성요소 모드 (rule 인 구성요소는 규칙
정책) 로 실행하고, 두 모델의 실행 env (구성요소 모드 제외) 는 같아야 한다. 결과는 seed 순으로 정렬하므로 workers 수와 무관하게
같다 (추론 시간 열 제외). workers = 1 이면 같은 함수를 현재 프로세스에서 실행 (torch 스레드 1).
run_jobs(fn, tasks, workers) = run_tasks 가 쓰는 범용 풀 실행 (진행률 출력).
"""
import json
import multiprocessing as mp
import os
import sys
import time
from collections import namedtuple

Task = namedtuple("Task", "key env_config free_keys seed timeseries policies", defaults=(("baseline", "learned"),))
# policies: 실행할 정책 슬롯 (models 의 키, 각각 모델 경로 또는 None = 규칙 정책) — 순서대로 같은 seed 로 실행
Result = namedtuple("Result", "key seed per ts env_used sig")     # sig = seed 의 초기 상태 지문 (task 끼리 비교용)

_W = {}


def default_workers():
    return max(1, (os.cpu_count() or 1) - 2)


def _init(models, mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import signal

    import torch
    torch.set_num_threads(1)
    import ray.rllib.models  # noqa: F401  (ray import 가 SIGTERM 핸들러를 바꾼다 -> 풀 종료 때 traceback, 기본값으로 되돌림)
    from eval_common.light_policy import LightPolicy
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _W["pols"] = {name: (LightPolicy(path, mode=mode) if path else None) for name, path in models.items()}
    _W["sims"] = {}


def _strip(env):
    return {k: v for k, v in env.items() if k != "components"}


def _run(task):
    from env.role.simulation import RoleSimulation
    from eval_common.episode import RulePolicy, run_episode_detailed
    pols = {}
    env = None
    for name in task.policies:
        model = _W["pols"].get(name)
        if model is None:
            pol = RulePolicy()
            pol.name, pol.components = name, None
        else:
            run_env = _strip(model.prepare(task.env_config, task.free_keys))
            assert env is None or run_env == env, "두 모델의 실행 env (관측 shape 키) 가 다름 — 같은 조건으로 비교할 수 없음"
            env, pol = run_env, model
        pols[name] = pol
    env = env if env is not None else _strip(dict(task.env_config))
    key = json.dumps(env, sort_keys=True, default=str)
    if key not in _W["sims"]:
        _W["sims"] = {key: RoleSimulation(env)}                 # 최근 env 하나만 유지 (main_scale 의 큰 맵 메모리)
    sim = _W["sims"][key]
    per, ts, sig = [], [], None
    for name in task.policies:
        pol = pols[name]
        if pol.components is not None:                          # 그 모델의 구성요소 모드 (rule 구성요소는 env 규칙 정책)
            sim.set_components(pol.components)
        pol.name = name
        m, rows, s = run_episode_detailed(sim, pol, task.seed, timeseries=task.timeseries)
        assert sig is None or s == sig, f"seed {task.seed}: 정책 {name} 의 초기 상태가 다른 정책과 다름 (맵 / 조난자 / 골든타임)"
        sig = s
        per.append(dict(policy=name, seed=task.seed, **m))
        ts += [dict(policy=name, seed=task.seed, **r) for r in rows]
    return Result(task.key, task.seed, per, ts, env, sig)


def _fmt_t(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def run_tasks(models, mode, tasks, workers, label="eval"):
    """models = {슬롯: 모델 경로 | None}. 반환 [Result] — (key, seed) 순. 진행률: 완료 task / 전체, 경과, 예상 남은 시간.
    워커를 띄우기 전에 모델을 이 프로세스에서 한 번 읽어 로드 가능한지 확인한다 (불가능하면 바로 RuntimeError — 워커 초기화가
    실패하면 풀이 워커를 계속 다시 띄우며 멈추지 않기 때문)."""
    from eval_common.model_loader import load_model
    for name, path in models.items():
        if path:
            try:
                load_model(path)
            except Exception as e:
                raise RuntimeError(f"{name} 모델을 로드할 수 없음 ({path}): {str(e).splitlines()[0]}") from None
    out = run_jobs(_run, tasks, workers, label, init=_init, initargs=(dict(models), mode))
    out.sort(key=lambda r: (r.key, r.seed))
    return out


def _init_cpu(init=None, initargs=()):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if init is not None:
        init(*initargs)


def run_jobs(fn, tasks, workers, label="eval", init=None, initargs=()):
    """범용 병렬 실행: fn(task) 를 spawn 워커 풀에서 (CPU, torch 스레드 1, init(*initargs) 를 워커마다 한 번). fn / init 은
    모듈 최상위 함수 (spawn 에서 import 가능). 반환 = 완료 순서의 결과 목록 (정렬은 호출부). workers = 1 이면 현재 프로세스."""
    workers = max(1, min(int(workers), len(tasks)))
    t0 = time.time()
    out = []

    def progress(k):
        el = time.time() - t0
        eta = el / k * (len(tasks) - k)
        print(f"\r  [{label}] {k}/{len(tasks)} tasks  elapsed {_fmt_t(el)}  ETA {_fmt_t(eta)}   ", end="", flush=True)

    if workers == 1:
        _init_cpu(init, initargs)
        for k, t in enumerate(tasks):
            out.append(fn(t))
            progress(k + 1)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_cpu, initargs=(init, initargs)) as pool:
            for k, r in enumerate(pool.imap_unordered(fn, tasks)):
                out.append(r)
                progress(k + 1)
    print(file=sys.stdout)
    return out
