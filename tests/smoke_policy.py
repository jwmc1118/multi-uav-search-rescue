"""ray 클러스터 없이 RLlib Policy 를 직접 만들어 학습 1 step 을 돌린다.

    python -m tests.smoke_policy            (Exploration_rescue/ 에서)

ray.init() 이 안 되는 환경(raylet 미기동 등)에서도 실제 trainer 와 같은 코드 경로를 점검:
  MultiPPOTorchPolicy 생성 (dummy batch 로 view_requirements / loss 초기화)
  -> compute_actions 로 rollout -> postprocess_trajectory (헤드별 GAE + alloc_valid)
  -> learn_on_batch (L = L_alloc + L_primitive backward + optimizer step)
"""
import copy

import numpy as np
import yaml
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from env.gym_env import RescueMultiEnv
from training.catalog import register_all
from training.multiagent_ppo import MultiPPOTorchPolicy

STEPS = 64


def main():
    with open("training/config/rescue.yaml", "rb") as f:
        ycfg = yaml.safe_load(f)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in ycfg.items():
        if k == "model":
            cfg["model"].update(v)
        elif k not in ("env",):
            cfg[k] = v
    cfg.update(num_workers=0, num_gpus=0, train_batch_size=STEPS,
               sgd_minibatch_size=32, num_sgd_iter=1)

    register_all()
    env = RescueMultiEnv(cfg["env_config"])
    prep = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    policy = MultiPPOTorchPolicy(prep.observation_space, env.action_space, cfg)
    print("[policy] created (dummy-batch loss init ok)")

    obs = env.reset()
    rows = {k: [] for k in ("obs", "new_obs", "actions", "rewards", "dones", "infos")}
    extras = []
    for t in range(STEPS):
        ob = prep.transform(obs)
        actions, _, extra = policy.compute_actions(ob[None], explore=True)
        act = {k: [int(np.asarray(a)[0]) for a in v] for k, v in actions.items()}
        nobs, r, done, info = env.step(act)
        flat = np.concatenate([np.asarray(act[k], np.float32) for k in ("alloc", "primitive")])
        rows["obs"].append(ob); rows["new_obs"].append(prep.transform(nobs))
        rows["actions"].append(flat); rows["rewards"].append(r)
        rows["dones"].append(done or t == STEPS - 1); rows["infos"].append(info)
        extras.append({k: np.asarray(v)[0] for k, v in extra.items()})
        obs = env.reset() if done else nobs

    batch = SampleBatch({
        SampleBatch.OBS: np.array(rows["obs"], np.float32),
        SampleBatch.NEXT_OBS: np.array(rows["new_obs"], np.float32),
        SampleBatch.ACTIONS: np.array(rows["actions"], np.float32),
        SampleBatch.REWARDS: np.array(rows["rewards"], np.float32),
        SampleBatch.DONES: np.array(rows["dones"]),
        SampleBatch.INFOS: np.array(rows["infos"], dtype=object),
        SampleBatch.EPS_ID: np.zeros(STEPS, np.int64),
        SampleBatch.AGENT_INDEX: np.zeros(STEPS, np.int64),
        **{k: np.stack([e[k] for e in extras]) for k in extras[0]},
    })
    batch = policy.postprocess_trajectory(batch)
    print("[postprocess] keys:", sorted(k for k in batch.keys() if k not in ("obs", "new_obs", "infos")))
    # 회귀 점검: 학습 시 RLlib 은 ACTIONS 를 numpy 로 넘긴다 -> 모델이 그대로 재생해야 함 (재샘플링 금지)
    import torch
    from ray.rllib.models.modelv2 import restore_original_dimensions
    dev = policy.device                     # CUDA torch 면 모델이 GPU 에 있다 -> 입력도 같은 장치로
    obs_t = restore_original_dimensions(torch.as_tensor(batch[SampleBatch.OBS], device=dev),
                                        policy.observation_space, "torch")
    lt, _ = policy.model({"obs": obs_t, SampleBatch.ACTIONS: torch.as_tensor(batch[SampleBatch.ACTIONS], device=dev)},
                         [], None)
    ln, _ = policy.model({"obs": obs_t, SampleBatch.ACTIONS: np.asarray(batch[SampleBatch.ACTIONS])}, [], None)
    assert torch.allclose(lt, ln), "numpy ACTIONS 를 모델이 재생하지 못함"
    prev = np.asarray(batch[SampleBatch.ACTION_DIST_INPUTS])
    assert np.allclose(prev[:, :-env.n_agents], ln.detach().cpu().numpy()[:, :-env.n_agents], atol=1e-4), \
        "학습 재생 logit != rollout logit"
    print("[replay] numpy/tensor ACTIONS 재생 일치, rollout logit 과 일치")

    stats = policy.learn_on_batch(batch)
    ls = stats.get("learner_stats", stats)
    assert np.isfinite(float(ls["kl"])) and np.isfinite(float(ls["total_loss"])), "KL / loss 가 유한하지 않음"
    print("[learn] " + ", ".join(f"{k}={float(v):.4g}" for k, v in ls.items()
                                 if np.isscalar(v) or np.asarray(v).size == 1))
    print("\nOK: policy init + rollout + postprocess + learn_on_batch 통과.")


if __name__ == "__main__":
    main()
