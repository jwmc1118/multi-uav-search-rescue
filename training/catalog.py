"""RLlib 등록 (env / custom model / custom action dist)."""
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from env.gym_env import RescueMultiEnv
from models.rllib_model import RescueModel
from training.action_dist import HierarchicalMultiActionDistribution

ENV_NAME = "rescue"
MODEL_NAME = "rescue_model"
DIST_NAME = "rescue_multi_action"


def register_all():
    register_env(ENV_NAME, lambda cfg: RescueMultiEnv(cfg))
    ModelCatalog.register_custom_model(MODEL_NAME, RescueModel)
    ModelCatalog.register_custom_action_dist(DIST_NAME, HierarchicalMultiActionDistribution)
