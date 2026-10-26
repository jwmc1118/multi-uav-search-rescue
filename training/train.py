"""학습 엔트리포인트.

    python -m training.train -t 10                  # 계층 구조: training/config/env.yaml + role.yaml + explore.yaml + rescue.yaml, 10M steps
    python -m training.train --smoke                # 1 iteration 만
    python -m training.train --local --smoke        # 워커 없이 로컬 디버그
    python training/train.py                                            # 직접 실행 / IDE 실행 버튼 (config 기본값 사용)
    python -m training.train training/config/env.yaml ... my.yaml       # 뒤 파일이 앞 파일 값을 덮어씀

학습 구조 = 계층 구조 3 헤드 Role / Rescue / Explore (training/role/ppo.py, config env: rescue_role).
best 기준 = 고정 seed 평가 (eval.every iteration 마다, run_rules 와 같은 seed, 세 head argmax) 의 구조율 최대,
          같으면 에피소드 길이가 짧은 쪽 (evaluation/fixed_seed_eval.py, training/callbacks.py).

이어서 학습 (특정 체크포인트 기준):
    python training/train.py --resume marl_models/RolePPO/checkpoints/checkpoint_000200
    python training/train.py --resume marl_models/best_model/2026-10-16_20-50-03_role/checkpoint_000180

저장 구조:
    marl_models/RolePPO/                        새로 학습하면 덮어쓴다 (RolePPO/checkpoints 안의 체크포인트로 이어서 학습하면 유지)
        checkpoints/checkpoint_XXXXXX/          CHECKPOINT_FREQ(10) iteration 마다 저장, 전부 유지
        checkpoints/best/checkpoint_XXXXXX/     eval 기준 best (갱신될 때마다 교체, best_info.json 포함)
        reward_curve.png                        --plot-every iteration 마다 갱신
        progress.csv, result.json, events.out.tfevents.*, params.json
    marl_models/best_model/<학습 시작 시각>_role/ 이번 학습에서 지금까지 가장 좋은 모델. 학습마다 새 폴더라 덮어쓰지 않는다
        checkpoint_XXXXXX/, params.json, best_info.json, reward_curve.png
체크포인트를 저장할 때마다 바로 디스크에 남으므로 학습이 중간에 멈춰도 그때까지의 체크포인트와 best 모델은 남는다.
Ctrl+C 로 멈추면 그 시점도 한 번 더 저장한다. --smoke 는 marl_models/RolePPO_smoke/ 에 따로 쓰고 best_model/ 은 만들지 않는다
(checkpoints/best/ 는 smoke 에도 저장).
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
import csv
import glob
import json
import shutil
import time

import ray
from ray.tune.logger import UnifiedLogger

from training.callbacks import RoleCallbacks
from training.catalog import register_all
from evaluation.checkpoint_policy import resolve_checkpoint, restore_or_explain
from training.config_loader import load_config, reward_settings_line
from training.role.ppo import RolePPOTrainer
from training.reward_plot import PREV_PROGRESS, read_progress, save_reward_plot

# 이어서 학습할 체크포인트. None = 새로 학습 (RolePPO/ 덮어씀, best_model/ 은 유지)
# 실행 인자 --resume 을 주면 그쪽이 우선. 예)
#   RESUME = "marl_models/RolePPO/checkpoints/checkpoint_000200"
#   RESUME = "marl_models/best_model/2026-10-16_20-50-03_role/checkpoint_000180"
RESUME = None

OUT_DIR = os.path.join(ROOT, "marl_models", "RolePPO")
BEST_ROOT = os.path.join(ROOT, "marl_models", "best_model")
CHECKPOINT_FREQ = 10   # iteration
# Ray object store 크기 (GB). 워커가 보내는 샘플 = 관측 (약 130 KB / step, obs + new_obs 면 2 배) x rollout_fragment_length
# x num_envs_per_worker -> 워커 하나가 한 번에 약 100 MB. num_workers 를 늘리면 이 값도 키울 것 (모자라면 ObjectStoreFullError,
# 2 GB 면 워커 8 개까지만 버팀)
OBJECT_STORE_GB = 10


def _ray_init(num_gpus):
    """대시보드 끄고 object store 를 작게 잡아 Windows 에서 기동을 가볍게 한다.
    참고: 'Redis has started but no raylets have registered yet' 가 나고 raylet.err 에
    'node_manager_hostname contains invalid UTF-8' 가 쌓이면 컴퓨터 이름에 한글이 있는 것 -> PC 이름을 영문으로 변경."""
    try:
        ray.init(num_gpus=num_gpus,
                 include_dashboard=False,
                 object_store_memory=int(OBJECT_STORE_GB * 1e9))   # 기본 (RAM 의 30%) 보다 작게, 워커 수에 맞춰 OBJECT_STORE_GB
    except RuntimeError:
        tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "ray", "session_latest", "logs")
        print(f"[train] ray.init 실패. raylet 로그 확인: {tmp}  (raylet.out / raylet.err)\n"
              f"        raylet.err 에 'invalid UTF-8' 이 보이면 Windows 컴퓨터 이름을 영문으로 바꾸고 재부팅.")
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="*",
                    help="config yaml (생략하면 training/config/env.yaml + role.yaml + explore.yaml + rescue.yaml)")
    ap.add_argument("-t", "--timesteps", type=float, default=10, help="총 step 수 (백만)")
    ap.add_argument("--smoke", action="store_true", help="1 iteration 만 돌려 파이프라인 점검")
    ap.add_argument("--smoke-iters", type=int, default=1, help="--smoke 일 때 돌릴 iteration 수")
    ap.add_argument("--resume", "--restore", dest="restore", default=RESUME,
                    help="이 체크포인트에서 이어서 학습 (checkpoint_XXXXXX 폴더 또는 checkpoint-XX 파일)")
    ap.add_argument("--local", action="store_true", help="num_workers=0 로컬 디버그")
    ap.add_argument("--plot-every", type=int, default=10, help="보상 그래프(reward_curve.png) 저장 주기 (iteration)")
    args = ap.parse_args()

    config = load_config(args.config)                      # 여러 파일이면 앞에서부터 합침
    print(f"[train] config {', '.join(args.config) if args.config else 'env.yaml + role.yaml + explore.yaml + rescue.yaml'}")

    if config.get("env") != "rescue_role":
        raise SystemExit(f"[train] config env={config.get('env')!r} — 계층 구조 env 'rescue_role' 만 지원")
    print(reward_settings_line(config, "train"))         # 실제 적용 보상 값 (env 기본값 위에 yaml)
    config["callbacks"] = RoleCallbacks          # 에피소드 지표 (custom_metrics). yaml 에는 넣지 않는다
    config.pop("main", None)                     # main.py 전용 섹션 (*_main.yaml 과 key 구조를 맞추려고 env.yaml 에도 있음) — RLlib 이 모르는 키
    RoleCallbacks.eval_cfg = config.pop("eval", None)   # 고정 seed 평가 설정 (RLlib 이 모르는 키) -> driver 콜백
    ev = RoleCallbacks.eval_cfg or {}
    print(f"[train] eval: every {ev.get('every')} iterations, {ev.get('episodes')} fixed seeds, greedy (best = eval rescue rate, "
          f"tie -> shorter episode)")
    ls = config.get("lr_schedule")
    print(f"[train] lr {config.get('lr')}" + (f", lr_schedule {ls} (linear, then last value)" if ls else " (constant)"))
    if args.local:
        config["num_workers"] = 0
        config["num_envs_per_worker"] = 1


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

    out_dir = OUT_DIR + "_smoke" if args.smoke else OUT_DIR    # 점검 실행이 학습 결과를 지우지 않도록
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    best_dir = None if args.smoke else os.path.join(BEST_ROOT, time.strftime("%Y-%m-%d_%H-%M-%S") + "_role")

    # RolePPO/checkpoints 안의 체크포인트로 이어서 학습하면 폴더를 그대로 쓰고, 그 외에는 새로 덮어쓴다.
    # (best_model/ 은 절대 지우지 않는다)
    if args.restore:
        args.restore = resolve_checkpoint(args.restore)   # checkpoint_XXXXXX 폴더 경로도 받는다
        print(f"[train] 이어서 학습: {os.path.relpath(args.restore, ROOT)}")
    resume_here = bool(args.restore) and os.path.abspath(args.restore).startswith(ckpt_dir + os.sep)
    if not resume_here and os.path.isdir(out_dir):
        print(f"[train] 새 학습 -> {os.path.relpath(out_dir, ROOT)} 덮어씀 (best_model/ 은 유지)")
        shutil.rmtree(out_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    if resume_here:   # logger 가 progress.csv 를 열기 전에 기존 기록을 옮겨 둔다
        _archive_progress(out_dir, int(os.path.basename(os.path.dirname(args.restore)).rsplit("_", 1)[1]))

    trainer = RolePPOTrainer(config=config, logger_creator=lambda cfg: UnifiedLogger(cfg, out_dir, loggers=None))
    if args.restore:
        restore_or_explain(trainer, args.restore)

    saver = _Saver(trainer, out_dir, ckpt_dir, best_dir)
    stop_steps = int(args.timesteps * 1e6)
    try:
        while True:
            result = trainer.train()
            it = result["training_iteration"]
            cm = result.get("custom_metrics", {})
            extra = (f"  rescue {cm.get('rescue_rate_mean', float('nan')):.3f}"
                     f"  role decisions/batch {cm.get('decisions_per_batch_role', float('nan')):.0f}")
            print(f"[train] iter {it}  steps {result['timesteps_total']:,}  "
                  f"reward {result['episode_reward_mean']:.3f}  len {result['episode_len_mean']:.1f}{extra}  "
                  f"({result['time_this_iter_s']:.1f}s)")
            if "eval/rescue_rate" in cm:
                saver.eval_best(result, cm)
            done = (args.smoke and it >= args.smoke_iters) or result["timesteps_total"] >= stop_steps
            if it % args.plot_every == 0 or done:
                save_reward_plot(out_dir, baseline=ev.get("greedy_baseline"))
            if it % CHECKPOINT_FREQ == 0 or done:
                saver.save(result)
            if done:
                break
    except KeyboardInterrupt:
        print("\n[train] 중단 — 현재 상태를 저장합니다")
        save_reward_plot(out_dir, baseline=ev.get("greedy_baseline"))
        if trainer.iteration > saver.last_saved_iter:
            saver.save(_last_result(out_dir))
    finally:
        trainer.stop()   # logger flush / 워커 종료
        ray.shutdown()


def _last_result(out_dir):
    """progress.csv 의 마지막 행 (Ctrl+C 시점의 지표)."""
    rows = read_progress(out_dir)
    return rows[-1] if rows else {}


def _archive_progress(out_dir, iteration):
    """이어서 학습할 때 기존 기록 (result.json / progress.csv + 예전 progress_prev.csv) 을 progress_prev.csv 하나로 합쳐 옮긴다.
    - 되돌아간 체크포인트 뒤의 iteration 기록은 잘라 곡선이 겹치지 않게 한다.
    - 새 기록은 logger 가 새 progress.csv 에 쓴다. 코드가 바뀌어 열 구성이 달라져도 (예: custom_metrics 추가)
      한 파일에 섞이지 않아 깨지지 않는다. 읽을 때는 reward_plot.read_progress 가 두 파일을 이어 붙인다."""
    rows = [r for r in read_progress(out_dir) if int(float(r["training_iteration"])) <= iteration]
    for fn in ("progress.csv", "result.json", PREV_PROGRESS):         # result.json 은 logger 가 이어 쓰므로 지운다
        if os.path.isfile(os.path.join(out_dir, fn)):
            os.remove(os.path.join(out_dir, fn))
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out_dir, PREV_PROGRESS), "w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)


class _Saver:
    """체크포인트 저장 (지우지 않고 전부 유지) + eval 기준 best 를 checkpoints/best/ 에 저장 (best_model/<시각>/ 에도 복사)."""

    def __init__(self, trainer, out_dir, ckpt_dir, best_dir):
        self.trainer, self.out_dir, self.ckpt_dir, self.best_dir = trainer, out_dir, ckpt_dir, best_dir
        self.best_key = None                                  # (eval 구조율, -에피소드 길이) — 클수록 좋음
        self.last_saved_iter = trainer.iteration
        info = os.path.join(ckpt_dir, "best", "best_info.json")
        if os.path.isfile(info):                              # 이어서 학습: 기존 best 를 기준으로 (더 좋을 때만 교체)
            with open(info) as f:
                b = json.load(f)
            self.best_key = (b["eval_rescue_rate"], -b["eval_episode_len"])

    def save(self, result):
        path = self.trainer.save(self.ckpt_dir)
        self.last_saved_iter = self.trainer.iteration
        print(f"[train] checkpoint: {os.path.relpath(path, ROOT)}")

    def eval_best(self, result, cm):
        """eval 구조율 최대, 같으면 에피소드 길이가 짧은 쪽이 best. 갱신되면 checkpoints/best/ 를 이 체크포인트로 교체."""
        key = (cm["eval/rescue_rate"], -cm["eval/episode_len"])
        if self.best_key is not None and key <= self.best_key:
            return
        self.best_key = key
        best = os.path.join(self.ckpt_dir, "best")
        tmp = best + "_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        path = self.trainer.save(tmp)                          # 새 best 를 먼저 쓴 뒤 교체 (중간에 멈춰도 best 가 비지 않게)
        shutil.rmtree(best, ignore_errors=True)
        os.rename(tmp, best)
        src = os.path.join(best, os.path.basename(os.path.dirname(path)))
        info = {"checkpoint": os.path.basename(src), "training_iteration": self.trainer.iteration,
                "timesteps_total": result.get("timesteps_total"), "best_metric": "eval/rescue_rate (tie: shorter eval/episode_len)",
                "eval_rescue_rate": cm["eval/rescue_rate"], "eval_episode_len": cm["eval/episode_len"],
                "eval_expired_rate": cm["eval/expired_rate"], "eval_explored_rate": cm["eval/explored_rate"],
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(best, "best_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"[train] best 갱신 (iter {self.trainer.iteration}, eval rescue {cm['eval/rescue_rate']:.3f}, "
              f"len {cm['eval/episode_len']:.1f}): {os.path.relpath(src, ROOT)}")
        if self.best_dir:
            self._update_best(src, info)

    def _update_best(self, src, info):
        """best_model/<시각>/ 에 복사: 새 best 를 먼저 복사한 뒤 예전 best 를 지운다."""
        os.makedirs(self.best_dir, exist_ok=True)
        old = glob.glob(os.path.join(self.best_dir, "checkpoint_*"))
        dst = os.path.join(self.best_dir, os.path.basename(src))
        if os.path.abspath(dst) in map(os.path.abspath, old):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        for d in old:
            if os.path.abspath(d) != os.path.abspath(dst):
                shutil.rmtree(d, ignore_errors=True)
        for fn in ("params.json", "params.pkl", "reward_curve.png"):
            if os.path.isfile(os.path.join(self.out_dir, fn)):
                shutil.copy2(os.path.join(self.out_dir, fn), self.best_dir)
        with open(os.path.join(self.best_dir, "best_info.json"), "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
