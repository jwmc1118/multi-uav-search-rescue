"""학습된 체크포인트 평가 (헤드리스).

    python -m training.evaluate <checkpoint_path> -n 20
    python -m training.evaluate <checkpoint_path> --render      # pygame 으로 재생
"""
import os
import sys

# 스크립트를 직접 실행해도 (python training/evaluate.py, IDE 실행 버튼) 패키지를 찾도록 레포 루트를 경로에 추가.
# Ray 워커 프로세스도 같은 경로를 쓰도록 PYTHONPATH 에도 넣고, 상대 경로(maps/, config)를 위해 루트로 이동한다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
_CALLER_CWD = os.getcwd()
os.chdir(ROOT)

import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("-n", "--episodes", type=int, default=20)
    ap.add_argument("--map", default="maps/map_target/map_20260720_153619.json")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    from training.checkpoint_policy import CheckpointPolicy
    from env.simulation import RescueSimulation

    env_cfg = {"map_path": args.map}
    ckpt = CheckpointPolicy(os.path.join(_CALLER_CWD, os.path.expanduser(args.checkpoint)), env_config=env_cfg)

    if args.render:
        from gui.grid_map import GridMap
        sim = RescueSimulation(env_cfg)
        GridMap(sim, model_policy=ckpt).run()
        return

    sim = RescueSimulation(env_cfg)
    rescued, expired, steps, explored = [], [], [], []
    for _ in range(args.episodes):
        sim.reset()
        done = False
        while not done:
            allocs, prims = ckpt.act(sim)
            done, info = sim.step_explicit(allocs, prims)
        rescued.append(info["rescued_total"])
        expired.append(info["expired_total"])
        steps.append(info["time"])
        explored.append(info["explored"])
    print(f"episodes={args.episodes}")
    print(f"  rescued : {np.mean(rescued):.2f} +- {np.std(rescued):.2f}")
    print(f"  expired : {np.mean(expired):.2f} +- {np.std(expired):.2f}")
    print(f"  steps   : {np.mean(steps):.1f}")
    print(f"  explored: {np.mean(explored):.3f}")


if __name__ == "__main__":
    main()
