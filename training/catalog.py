"""RLlib 등록 (env / custom model / custom action dist)."""
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from env.role.gym_env import RoleEnv
from models.role.rllib_model import RoleModel
from training.role.action_dist import RoleActionDistribution

ROLE_ENV_NAME = "rescue_role"          # 계층 구조 env (env/role/simulation.py)
ROLE_MODEL_NAME = "role_model"         # models/role/rllib_model.py
ROLE_DIST_NAME = "role_multi_action"   # training/role/action_dist.py


def register_all():
    register_env(ROLE_ENV_NAME, lambda cfg: RoleEnv(cfg))
    ModelCatalog.register_custom_model(ROLE_MODEL_NAME, RoleModel)
    ModelCatalog.register_custom_action_dist(ROLE_DIST_NAME, RoleActionDistribution)
