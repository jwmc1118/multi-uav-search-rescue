"""학습된 체크포인트를 GUI/평가에서 굴리기 위한 래퍼.

gui/grid_map.py (Model 버튼) 와 training/evaluate.py 가 이 객체의 .act(sim) 가 반환하는 (allocs, primitives) 를
sim.step_explicit() 로 그대로 넣는다. 평가 시 할당은 argmax (model.alloc_greedy = True). ray 가 없거나 복원 실패 시 예외를 던지므로
호출부에서 잡아 상태메시지로 처리한다.
"""
import glob
import os

import numpy as np


def resolve_checkpoint(path):
    """checkpoint_XXXXXX 폴더면 안의 checkpoint-XX 파일 경로로 바꾼다."""
    if os.path.isdir(path):
        files = [f for f in glob.glob(os.path.join(path, "checkpoint-*"))
                 if not f.endswith(".tune_metadata")]
        if not files:
            raise FileNotFoundError(f"체크포인트 파일이 없음: {path}")
        path = files[0]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"체크포인트가 없음: {path}")
    return path


def find_params_json(checkpoint_path):
    """체크포인트의 학습 config(params.json) 위치. 저장 구조에 따라 한두 단계 위에 있다.
    marl_models/MultiPPO/checkpoints/checkpoint_X/checkpoint-X -> marl_models/MultiPPO/params.json
    marl_models/best_model/<날짜>/checkpoint_X/checkpoint-X    -> marl_models/best_model/<날짜>/params.json"""
    d = os.path.dirname(os.path.abspath(checkpoint_path))
    for _ in range(3):
        d = os.path.dirname(d)
        pj = os.path.join(d, "params.json")
        if os.path.isfile(pj):
            return pj
    return None


class CheckpointPolicy:
    def __init__(self, checkpoint_path, env_config=None):
        import ray
        import json
        import os

        from training.catalog import register_all, ENV_NAME
        from training.multiagent_ppo import MultiPPOTrainer

        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True, num_gpus=0,
                     include_dashboard=False)
        register_all()

        # 체크포인트 옆 params.json 에서 학습 config 복원
        cfg = {}
        pj = find_params_json(checkpoint_path)
        if pj:
            with open(pj) as f:
                cfg = json.load(f)
        cfg.pop("callbacks", None)     # params.json 에는 클래스가 문자열로 저장됨 -> 추론엔 필요 없으니 기본값 사용
        cfg["num_workers"] = 0
        cfg["num_envs_per_worker"] = 1
        cfg["explore"] = False
        cfg["num_gpus"] = 0            # 추론은 CPU (학습 때 num_gpus: 1 이어도)
        if env_config:
            cfg.setdefault("env_config", {}).update(env_config)

        self.trainer = MultiPPOTrainer(config=cfg, env=ENV_NAME)
        self.trainer.restore(checkpoint_path)
        self.trainer.get_policy().model.alloc_greedy = True

    def act(self, sim):
        obs = sim.rl_observation()
        action = self.trainer.compute_action(obs, explore=False)   # Ray 1.3 API (compute_single_action 은 1.4+)
        allocs = np.asarray(action["alloc"]).reshape(-1)
        prims = np.asarray(action["primitive"]).reshape(-1)
        return allocs, prims
