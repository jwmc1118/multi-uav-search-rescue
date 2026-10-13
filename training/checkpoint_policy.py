"""학습된 체크포인트를 GUI/평가에서 굴리기 위한 래퍼.

gui/grid_map.py (Model 버튼) 와 training/evaluate.py 가 이 객체의 .act(sim) 가 반환하는 (allocs, primitives) 를
sim.step_explicit() 로 그대로 넣는다. 평가 시 할당은 argmax (model.alloc_greedy = True). ray 가 없거나 복원 실패 시 예외를 던지므로
호출부에서 잡아 상태메시지로 처리한다.
"""
import numpy as np


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
        pj = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "params.json")
        if os.path.isfile(pj):
            with open(pj) as f:
                cfg = json.load(f)
        cfg["num_workers"] = 0
        cfg["num_envs_per_worker"] = 1
        cfg["explore"] = False
        if env_config:
            cfg.setdefault("env_config", {}).update(env_config)

        self.trainer = MultiPPOTrainer(config=cfg, env=ENV_NAME)
        self.trainer.restore(checkpoint_path)
        self.trainer.get_policy().model.alloc_greedy = True

    def act(self, sim):
        obs = sim.rl_observation()
        action = self.trainer.compute_single_action(obs, explore=False)
        allocs = np.asarray(action["alloc"]).reshape(-1)
        prims = np.asarray(action["primitive"]).reshape(-1)
        return allocs, prims
