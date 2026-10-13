"""학습 엔트리포인트.

    python -m training.train training/config/rescue.yaml -t 10        # 10M steps
    python -m training.train training/config/rescue.yaml --smoke      # 1 iteration 만
    python -m training.train training/config/rescue.yaml --local --smoke   # 워커 없이 로컬 디버그
    python training/train.py                                            # 직접 실행 / IDE 실행 버튼 (config 기본값 사용)

체크포인트는 ~/ray_results/MultiPPO/... 에 저장된다. GUI 에서 불러오려면 그 경로를 쓴다.
"""
import os
import sys

# 스크립트를 직접 실행해도 (python training/train.py, IDE 실행 버튼) 패키지를 찾도록 레포 루트를 경로에 추가.
# Ray 워커 프로세스도 같은 경로를 쓰도록 PYTHONPATH 에도 넣고, 상대 경로(maps/, config)를 위해 루트로 이동한다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
os.chdir(ROOT)

import argparse

import yaml
import ray
from ray import tune

from training.catalog import register_all
from training.multiagent_ppo import MultiPPOTrainer


def _ray_init(num_gpus):
    """대시보드 끄고 object store 를 작게 잡아 Windows 에서 기동을 가볍게 한다.
    참고: 'Redis has started but no raylets have registered yet' 가 나고 raylet.err 에
    'node_manager_hostname contains invalid UTF-8' 가 쌓이면 컴퓨터 이름에 한글이 있는 것 -> PC 이름을 영문으로 변경."""
    try:
        ray.init(num_gpus=num_gpus,
                 include_dashboard=False,
                 object_store_memory=int(2e9))   # 기본(RAM 의 30%)보다 작게
    except RuntimeError:
        tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "ray", "session_latest", "logs")
        print(f"[train] ray.init 실패. raylet 로그 확인: {tmp}  (raylet.out / raylet.err)\n"
              f"        raylet.err 에 'invalid UTF-8' 이 보이면 Windows 컴퓨터 이름을 영문으로 바꾸고 재부팅.")
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="training/config/rescue.yaml",
                    help="config yaml 경로 (생략하면 training/config/rescue.yaml)")
    ap.add_argument("-t", "--timesteps", type=float, default=10, help="총 step 수 (백만)")
    ap.add_argument("--smoke", action="store_true", help="1 iteration 만 돌려 파이프라인 점검")
    ap.add_argument("--restore", default=None, help="이어서 학습할 체크포인트 경로")
    ap.add_argument("--local", action="store_true", help="num_workers=0 로컬 디버그")
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    if args.local:
        config["num_workers"] = 0
        config["num_envs_per_worker"] = 1

    stop = {"training_iteration": 1} if args.smoke else {"timesteps_total": int(args.timesteps * 1e6)}

    # GPU 수를 직접 넘긴다. 안 넘기면 Ray 1.3 이 Windows 에서 WMIC 명령으로 GPU 를 찾는데,
    # Windows 11 최신 버전에는 WMIC 가 없어서 FileNotFoundError 가 난다.
    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    want = float(config.get("num_gpus", 0) or 0)
    if want > n_gpus:
        print(f"[train] yaml num_gpus={want} 이지만 torch 가 찾은 GPU 는 {n_gpus} 개 -> num_gpus={n_gpus} 로 학습")
        config["num_gpus"] = n_gpus
    print(f"[train] CUDA GPU {n_gpus} 개 감지, 학습 num_gpus={config.get('num_gpus', 0)}")
    _ray_init(num_gpus=n_gpus)
    register_all()
    tune.run(
        MultiPPOTrainer,
        config=config,
        stop=stop,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        keep_checkpoints_num=3,
        restore=args.restore,
    )


if __name__ == "__main__":
    main()
