"""학습된 체크포인트 평가 (헤드리스).

    python evaluation/evaluate.py <checkpoint>                         # 세 헤드 greedy, seed 0~99
    python evaluation/evaluate.py <checkpoint> --compare --seeds 0-9  # 학습 정책 vs 그리디 정책 표 + CSV
    python evaluation/evaluate.py <checkpoint> --mode sample --seeds 0,3,7
    python evaluation/evaluate.py <checkpoint> --render                # pygame 으로 재생
    python evaluation/evaluate.py <checkpoint> --n_agents 6           # 학습 때와 다른 UAV 수로 (기본 = 체크포인트 값)

<checkpoint> = checkpoint_XXXXXX 폴더 또는 그 안의 checkpoint-XX 파일.
환경 설정은 체크포인트의 학습 config(params.json 의 env_config)를 그대로 쓴다 (--map 을 주면 그 맵 파일 고정).
에피소드 k 는 seed k 로 맵 / 조난자 / 골든타임을 고정 생성하고, 정책 샘플링도 같은 seed 로 고정한다
-> 정책끼리 같은 상황에서 비교된다.

지표 (에피소드 끝 기준, 평균 ± 표준편차):
    rescue_rate / expire_rate / discover_rate (/ N_T), explored (팀 탐색 비율), same_cell (같은 칸 누적),
    conflict_cross (서로 다른 통신 그룹이 같은 조난자를 쫓은 step), escalations (같은 step 재결정), episode_len
"""
import os
import sys

# 스크립트를 직접 실행해도 (python evaluation/evaluate.py, IDE 실행 버튼) 패키지를 찾도록 레포 루트를 경로에 추가.
# Ray 워커 프로세스도 같은 경로를 쓰도록 PYTHONPATH 에도 넣고, 상대 경로(maps/, config)를 위해 루트로 이동한다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
_CALLER_CWD = os.getcwd()
os.chdir(ROOT)

import argparse
import csv

import numpy as np

METRICS = ["rescue_rate", "expire_rate", "discover_rate", "explored", "same_cell", "conflict_cross",
           "escalations", "episode_len"]


class RulePolicy:
    """그리디 정책을 CheckpointPolicy 와 같은 인터페이스로 (step / seed / set_mode) — 비교 평가용."""

    def __init__(self):
        from policies.role_rules import RuleDecider
        self.decider = RuleDecider()
        self.name = "rule-greedy"

    def step(self, sim):
        return sim.step(self.decider)

    def seed(self, seed):
        pass

    def set_mode(self, *a, **k):
        pass


def parse_seeds(text):
    """'0-99' / '0,3,7' / '0-4,10' -> [int]."""
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_episode(sim, policy, seed):
    sim.random_state = np.random.RandomState(seed)   # 맵 / 조난자 / 골든타임 / 시작 위치
    sim.reset()
    policy.seed(seed)                                # 정책 샘플링
    conflict = esc = 0
    done, info = False, {}
    while not done:
        done, info = policy.step(sim)                # 이동은 env 의 BFS 경로 추종
        conflict += int(info["rescue_conflict_cross"])
        esc += int(info["redecide"])
    n_t = max(1, len(sim.targets))
    return {"rescue_rate": info["rescued_total"] / n_t, "expire_rate": info["expired_total"] / n_t,
            "discover_rate": info["discovered_total"] / n_t, "explored": info["explored"],
            "same_cell": info["same_cell_total"], "conflict_cross": conflict, "escalations": esc,
            "episode_len": info["time"]}


def evaluate(sim, policy, seeds, label, mode="greedy"):
    policy.set_mode(mode)
    rows = []
    for k, s in enumerate(seeds):
        rows.append(run_episode(sim, policy, s))
        print(f"\r  [{label}] {k + 1}/{len(seeds)}", end="", flush=True)
    print()
    return {m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows]))) for m in METRICS}


def print_table(results):
    w = 18
    keys = list(results[0][1].keys())
    print(f"{'policy':<16}" + "".join(f"{m:>{w}}" for m in keys))
    for label, res in results:
        cells = ["{:.3f} ± {:.3f}".format(*res[m]) for m in keys]
        print(f"{label:<16}" + "".join(f"{c:>{w}}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--mode", default="greedy", choices=["greedy", "sample"], help="세 헤드 행동 선택 (argmax / 샘플링)")
    ap.add_argument("--seeds", default="0-99", help="에피소드 seed 목록: '0-99' / '0,3,7'")
    ap.add_argument("--compare", action="store_true",
                    help="학습 정책 (--mode) vs 그리디 정책을 같은 seed 로 비교")
    ap.add_argument("--csv", default=None, help="--compare 결과 CSV 경로 (기본: 체크포인트 폴더/eval_compare.csv)")
    ap.add_argument("--map", default=None, help="고정 맵 파일 (기본: 학습 config 대로 랜덤 맵)")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--n_agents", type=int, default=None,
                    help="UAV 수 (기본 = 체크포인트 값). 센서 / 통신 범위는 체크포인트 첫 UAV 값 복제")
    args = ap.parse_args()

    from env.role.simulation import RoleSimulation
    from evaluation.checkpoint_policy import CheckpointPolicy, load_train_config, resolve_checkpoint, with_n_agents

    ckpt = resolve_checkpoint(os.path.join(_CALLER_CWD, os.path.expanduser(args.checkpoint)))
    env_cfg = load_train_config(ckpt).get("env_config", {})     # 학습 때와 같은 환경 (관측 크기도 일치)
    if args.map:
        env_cfg.update(map_mode="file", map_path=args.map)
    env_cfg = with_n_agents(env_cfg, args.n_agents)
    policy = CheckpointPolicy(ckpt, env_config=env_cfg, mode=args.mode)
    sim = RoleSimulation(policy.env_config)

    if args.render:
        from gui.grid_map import GridMap
        GridMap(sim, model_policy=policy, model_mode=args.mode).run()
        return

    seeds = parse_seeds(args.seeds)
    print(f"checkpoint: {os.path.relpath(ckpt, ROOT)}   episodes: {len(seeds)} (seeds {args.seeds})   {policy.describe()}")
    results = [(f"learned/{args.mode}", evaluate(sim, policy, seeds, f"learned/{args.mode}", args.mode))]
    if args.compare:
        results += [("rule/greedy", evaluate(sim, RulePolicy(), seeds, "rule/greedy"))]
    print_table(results)

    if args.compare:
        path = args.csv or os.path.join(os.path.dirname(ckpt), "eval_compare.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            keys = list(results[0][1].keys())
            w.writerow(["policy", "episodes"] + [f"{m}_{s}" for m in keys for s in ("mean", "std")])
            for label, res in results:
                w.writerow([label, len(seeds)] + [v for m in keys for v in res[m]])
        print(f"saved: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
