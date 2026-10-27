"""경량 체크포인트 정책 — ray.init / RLlib trainer 없이 가중치만 torch 로 올려 actor 를 실행 (평가 병렬 워커용).

CheckpointPolicy (eval_common/checkpoint_policy.py) 와 같은 결정을 낸다:
    관측 = sim.rl_observation() -> RLlib 전처리기 (Dict 평탄화, 학습 config observation_filter = NoFilter) ->
    RoleModel (models/role/rllib_model.py, 학습과 같은 클래스) forward -> 모델이 순차 디코딩한 행동 (greedy = argmax,
    sample = 학습 때와 같은 샘플링) -> sim.step_explicit. RLlib 행동 분포는 모델 행동을 그대로 통과시키므로 생략해도 같다.
가중치는 체크포인트 파일에서 한 번만 읽고 (read_weights), 실행 env_config 가 바뀌면 (UAV 수 N / 맵 / s_max) 모델만 새로
만들어 같은 가중치를 싣는다 (모델 파라미터는 N · 맵 크기 · s_max 와 무관 — actor 는 strict, critic 은 shape 다르면 건너뜀).

    p = LightPolicy(checkpoint, mode="greedy")
    env = p.prepare(env_config, free_keys=())      # 실행 env_config (SHAPE_KEYS 는 체크포인트 값, free_keys 제외) + 모델 준비
    sim = RoleSimulation(env); p.seed(s); done, info = p.step(sim)
"""
import copy
import random

import numpy as np

from eval_common.checkpoint_policy import (MODES, check_compatible, is_role_config, load_train_config, load_weights,
                                           read_weights, runtime_env_config)


class LightPolicy:
    MODES = MODES

    def __init__(self, checkpoint_path, mode="greedy"):
        self.checkpoint_path = checkpoint_path
        self.train_cfg = load_train_config(checkpoint_path)
        if not is_role_config(self.train_cfg):
            raise RuntimeError(f"예전 구조 체크포인트는 지원하지 않음: {checkpoint_path}")
        check_compatible(self.train_cfg, checkpoint_path)
        if (self.train_cfg.get("observation_filter") or "NoFilter") != "NoFilter":
            raise RuntimeError(f"observation_filter={self.train_cfg['observation_filter']!r} — 경량 경로는 NoFilter 만 지원")
        self.train_env = self.train_cfg.get("env_config", {})
        self.weights = read_weights(checkpoint_path)
        self.trained_n = len(self.train_env.get("agents", []))
        self.name = "model"
        self._models = {}                                  # env 키 -> (prep, model, skipped)
        self.model = self.prep = None
        self.mode = mode

    @staticmethod
    def _key(env):
        return (len(env["agents"]), int(env.get("s_max") or 16), int((env.get("cluster") or {}).get("k_max", 10)),
                int(env.get("max_targets") or 0))

    def prepare(self, env_config, free_keys=()):
        """실행 env_config 로 모델 준비 (같은 관측 shape 면 재사용). 반환 실제 env_config. self.warnings = 강제 키."""
        run_env, self.warnings = runtime_env_config(self.train_env, env_config, free_keys)
        self.env_config = run_env
        key = self._key(run_env)
        if key not in self._models:
            self._models[key] = self._build(run_env)
        self.prep, self.model, self.skipped = self._models[key]
        self.set_mode(self.mode)
        return run_env

    def _build(self, env_config):
        import torch
        from ray.rllib.models import ModelCatalog
        from ray.rllib.models.catalog import MODEL_DEFAULTS

        from env.role.gym_env import RoleEnv
        from models.role.rllib_model import RoleModel
        env = RoleEnv(dict(env_config, events=dict(env_config.get("events") or {}, log_dir=None)))
        prep = ModelCatalog.get_preprocessor_for_space(env.observation_space)
        mc = copy.deepcopy(MODEL_DEFAULTS)
        mc.update(copy.deepcopy(self.train_cfg.get("model") or {}))
        cmc = mc.setdefault("custom_model_config", {})
        cmc["gnn"] = dict(cmc.get("gnn") or {}, s_max=int(env_config.get("s_max") or 16))
        model = RoleModel(prep.observation_space, env.action_space, 0, mc, "eval_light")
        model.eval()
        skipped = load_weights(model, self.weights)
        del env
        torch.set_grad_enabled(False)
        return prep, model, skipped

    def set_mode(self, mode="greedy"):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} — {MODES}")
        self.mode = mode
        if self.model is not None:
            self.model.greedy = mode == "greedy"

    @staticmethod
    def seed(seed):
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def act(self, sim):
        import torch
        flat = self.prep.transform(sim.rl_observation())
        with torch.no_grad():
            self.model({"obs": torch.as_tensor(np.asarray(flat, np.float32)[None])}, [], None)
        a = self.model.last_out["actions"]
        return {h: a[h][0].cpu().numpy().reshape(-1) for h in ("role", "rescue", "explore")}

    def step(self, sim):
        return sim.step_explicit(self.act(sim))
