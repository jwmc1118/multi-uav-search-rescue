"""trainer 의 GAE + loss 함수를 손으로 만든 배치로 점검 (ray 클러스터 불필요).

    python -m tests.smoke_loss            (Exploration_rescue/ 에서)
"""
import numpy as np
import torch
from types import SimpleNamespace
from gym.spaces import Box

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor

from env.gym_env import RescueMultiEnv
from models.rllib_model import RescueModel
from models.primitive_policy import N_ACTIONS
from training.action_dist import HierarchicalMultiActionDistribution
from training.multiagent_ppo import compute_gae_for_sample_batch, ppo_surrogate_loss
from training.sample_batch_keys import VF_PREDS_ALLOC, ACTION_LOGP_ALLOC, ALLOC_VALID, VALUE_TARGETS_ALLOC


def main():
    T_STEPS = 60
    env = RescueMultiEnv({"map_path": "maps/map_target/map_20260720_153619.json", "max_targets": 16})
    N, A = env.n_agents, env.n_alloc

    prep = DictFlatteningPreprocessor(env.observation_space)
    obs_space = Box(-np.inf, np.inf, (prep.size,), np.float32)
    obs_space.original_space = env.observation_space
    num_outputs = N * A + N * N_ACTIONS
    model = RescueModel(obs_space, env.action_space, num_outputs, {"custom_model_config": {}}, "m")
    input_lens = [A] * N + [N_ACTIONS] * N
    child = [TorchCategorical] * (2 * N)

    def dist_class(logits, m):
        return HierarchicalMultiActionDistribution(
            logits, m, child_distributions=child, input_lens=input_lens,
            action_space=env.action_space)

    # ---- roll a short trajectory ----
    obs = env.reset()
    obs_flat, acts, infos, dist_inputs = [], [], [], []
    vf, vfa, alp, alpa, dones = [], [], [], [], []
    for k in range(T_STEPS):
        ob_t = prep.transform(obs)
        obs_flat.append(ob_t)
        rd = restore_original_dimensions(torch.as_tensor(ob_t[None]), obs_space, "torch")
        out, _ = model({"obs": rd}, [], None)
        d = dist_class(out, model)
        a = d.sample()
        flat = torch.cat([torch.stack(list(a["alloc"]), -1).float(),
                          torch.stack(list(a["primitive"]), -1).float()], -1)[0]
        acts.append(flat.numpy())
        dist_inputs.append(out[0].detach().numpy())
        v = model.value_function()
        vf.append(v["primitive"][0].detach().numpy())
        vfa.append(v["alloc"][0].detach().numpy())
        alp.append(d.sampled_action_logp()[0].detach().numpy())
        alpa.append(d.logp_alloc()[0].detach().numpy())
        act_env = {"alloc": [int(x) for x in a["alloc"]],
                   "primitive": [int(x) for x in a["primitive"]]}
        obs, _, done, info = env.step(act_env)
        infos.append(info)
        dones.append(k == T_STEPS - 1)

    sb = SampleBatch({
        SampleBatch.OBS: np.array(obs_flat, np.float32),
        SampleBatch.ACTIONS: np.array(acts, np.float32),
        SampleBatch.ACTION_DIST_INPUTS: np.array(dist_inputs, np.float32),
        SampleBatch.ACTION_LOGP: np.array(alp, np.float32),
        SampleBatch.REWARDS: np.zeros(T_STEPS, np.float32),
        SampleBatch.DONES: np.array(dones, np.bool_),
        SampleBatch.INFOS: np.array(infos, dtype=object),
        SampleBatch.VF_PREDS: np.array(vf, np.float32),
        VF_PREDS_ALLOC: np.array(vfa, np.float32),
        ACTION_LOGP_ALLOC: np.array(alpa, np.float32),
    })

    policy = SimpleNamespace(
        model=model,
        config={"gamma": 0.99, "lambda": 0.95, "clip_param": 0.2, "vf_clip_param": 20.0,
                "vf_loss_coeff": 0.5},
        kl_coeff=0.2, entropy_coeff=0.01,
    )

    sb = compute_gae_for_sample_batch(policy, sb)
    print("[gae] ADVANTAGES", sb[Postprocessing.ADVANTAGES].shape,
          "VALUE_TARGETS_ALLOC", sb[VALUE_TARGETS_ALLOC].shape,
          "ALLOC_VALID mean", round(float(sb[ALLOC_VALID].mean()), 3))

    tb = SampleBatch({k: torch.as_tensor(np.asarray(v)) for k, v in sb.items()
                      if k != SampleBatch.INFOS})
    loss = ppo_surrogate_loss(policy, model, dist_class, tb)
    print("[loss] total =", float(loss),
          " policy =", float(policy._mean_policy_loss),
          " vf =", float(policy._mean_vf_loss),
          " ent =", float(policy._mean_entropy),
          " kl =", float(policy._mean_kl))
    assert abs(float(policy._mean_kl)) < 1e-4, "같은 파라미터인데 KL != 0 -> 학습 재생이 rollout 과 다름"
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print("[loss] backward ok, grad-sum =", round(g, 3))
    print("\nOK: GAE + loss + backward 통과.")


if __name__ == "__main__":
    main()
