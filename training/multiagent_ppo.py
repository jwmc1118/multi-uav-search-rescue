"""MultiPPOTrainer — 2계층(할당 / primitive) 분리 학습 PPO.

Role_selection 의 multiagent_ppo.py 패턴을 계승:
- 헤드마다 별도 reward(env info) -> 별도 GAE/advantage/value target
- 최종 loss  L = L_alloc + L_primitive  를 한 번의 backward 로 업데이트
- CTDE: critic 2개는 학습 시 중앙집중 state 를 보고, actor 는 로컬 + n-hop 통신 관측만 사용
- alloc 헤드의 policy/entropy/KL 항은 alloc_valid(정책이 실제로 고른 tick, lock 강제 아님)에서만 평균.
  alloc critic 은 모든 tick 에서 학습.

주의: RLlib 1.3.0 의 trajectory-view / SampleBatch 세부와 맞물리는 부분이 많아
      실제 학습 전 tests/smoke_loss.py, tests/smoke_policy.py 로 빠르게 점검한다.
"""
import numpy as np

import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import (
    kl_and_loss_stats, setup_mixins, KLCoeffMixin,
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, explained_variance

from training.sample_batch_keys import (
    REWARDS_ALLOC, VF_PREDS_ALLOC, ACTION_LOGP_ALLOC, ADVANTAGES_ALLOC, VALUE_TARGETS_ALLOC,
    ALLOC_VALID,
)

torch, nn = try_import_torch()


# ---------------------------------------------------------------- GAE

def _gae(rewards, vpreds, last_r, gamma, lam):
    vpred_t = np.concatenate([vpreds, np.array([last_r])])
    delta = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    adv = discount_cumsum(delta, gamma * lam).astype(np.float32)
    value_targets = (adv + vpreds).astype(np.float32)
    return adv, value_targets


def _agent_infos(sample_batch, n_agents):
    """SampleBatch.INFOS -> (reward_primitive, reward_alloc, alloc_valid) 각 (T, N)."""
    infos = sample_batch.get(SampleBatch.INFOS)
    T = sample_batch[SampleBatch.REWARDS].shape[0]
    rp = np.zeros((T, n_agents), np.float32)
    ra = np.zeros((T, n_agents), np.float32)
    av = np.zeros((T, n_agents), np.float32)
    if infos is None or (hasattr(infos, "dtype") and infos.dtype == np.float32):
        return rp, ra, av   # trajectory-view API 의 첫 더미 호출
    for k, info in enumerate(infos):
        if not isinstance(info, dict):
            continue
        rp[k] = np.asarray(info.get("reward_primitive", np.zeros(n_agents)), np.float32)
        ra[k] = np.asarray(info.get("reward_alloc", np.zeros(n_agents)), np.float32)
        av[k] = np.asarray(info.get("alloc_valid", np.zeros(n_agents)), np.float32)
    return rp, ra, av


def compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches=None, episode=None):
    n_agents = policy.model.n_agents
    rp, ra, av = _agent_infos(sample_batch, n_agents)

    vf = np.asarray(sample_batch[SampleBatch.VF_PREDS], np.float32).reshape(-1, n_agents)
    vf_a = np.asarray(sample_batch[VF_PREDS_ALLOC], np.float32).reshape(-1, n_agents)

    gamma = policy.config["gamma"]
    lam = policy.config["lambda"]
    done = bool(sample_batch[SampleBatch.DONES][-1])

    if done:
        last = {"p": np.zeros(n_agents), "a": np.zeros(n_agents)}
    else:
        input_dict = policy.model.get_input_dict(sample_batch, index="last")
        vals = policy._value(**input_dict)
        last = {
            "p": vals["primitive"][0].detach().cpu().numpy().reshape(-1),
            "a": vals["alloc"][0].detach().cpu().numpy().reshape(-1),
        }

    adv_p = np.zeros_like(vf); vt_p = np.zeros_like(vf)
    adv_a = np.zeros_like(vf); vt_a = np.zeros_like(vf)
    for i in range(n_agents):
        adv_p[:, i], vt_p[:, i] = _gae(rp[:, i], vf[:, i], last["p"][i], gamma, lam)
        adv_a[:, i], vt_a[:, i] = _gae(ra[:, i], vf_a[:, i], last["a"][i], gamma, lam)

    sample_batch[REWARDS_ALLOC] = ra
    sample_batch[ALLOC_VALID] = av
    sample_batch[SampleBatch.VF_PREDS] = vf
    sample_batch[VF_PREDS_ALLOC] = vf_a
    sample_batch[Postprocessing.ADVANTAGES] = adv_p
    sample_batch[Postprocessing.VALUE_TARGETS] = vt_p
    sample_batch[ADVANTAGES_ALLOC] = adv_a
    sample_batch[VALUE_TARGETS_ALLOC] = vt_a
    return sample_batch


# ---------------------------------------------------------------- loss

def _head_loss(policy, adv, vt, prev_vf, cur_vf, logp, old_logp, kl, ent, mask=None):
    """한 헤드 x 한 에이전트의 PPO loss. mask 가 있으면 policy/KL/entropy 항만 마스크 평균
    (value 항은 항상 전체 평균)."""
    cfg = policy.config
    ratio = torch.exp(logp - old_logp)
    surr = torch.min(adv * ratio,
                     adv * torch.clamp(ratio, 1 - cfg["clip_param"], 1 + cfg["clip_param"]))
    vf1 = torch.pow(cur_vf - vt, 2.0)
    vf_clipped = prev_vf + torch.clamp(cur_vf - prev_vf, -cfg["vf_clip_param"], cfg["vf_clip_param"])
    vf2 = torch.pow(vf_clipped - vt, 2.0)
    vf_loss = torch.max(vf1, vf2).mean()

    if mask is None:
        mean = torch.mean
    else:
        denom = mask.sum().clamp(min=1.0)
        keep = mask > 0
        # where 로 걸러야 마스크된 행의 inf (lock 강제 / 더미 배치) 가 inf*0 = nan 이 되지 않는다
        mean = lambda x: torch.where(keep, x, torch.zeros_like(x)).sum() / denom
    pol, kl_m, ent_m = mean(-surr), mean(kl), mean(ent)
    total = pol + policy.kl_coeff * kl_m + cfg["vf_loss_coeff"] * vf_loss - policy.entropy_coeff * ent_m
    return total, pol, vf_loss, ent_m, kl_m


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch, is_training=True)
    curr = dist_class(logits, model)
    prev = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

    logps = curr.logp(train_batch[SampleBatch.ACTIONS])
    ent = curr.entropy()
    kl = prev.kl(curr)
    values = model.value_function()
    N = model.n_agents

    old_logp = {"primitive": train_batch[SampleBatch.ACTION_LOGP],
                "alloc": train_batch[ACTION_LOGP_ALLOC]}
    adv = {"primitive": train_batch[Postprocessing.ADVANTAGES],
           "alloc": train_batch[ADVANTAGES_ALLOC]}
    vt = {"primitive": train_batch[Postprocessing.VALUE_TARGETS],
          "alloc": train_batch[VALUE_TARGETS_ALLOC]}
    prev_vf = {"primitive": train_batch[SampleBatch.VF_PREDS],
               "alloc": train_batch[VF_PREDS_ALLOC]}
    alloc_valid = train_batch[ALLOC_VALID].float()

    totals, pols, vfs, ents, kls = [], [], [], [], []
    for head in ("primitive", "alloc"):
        for i in range(N):
            tot, pl, vl, en, k = _head_loss(
                policy,
                adv[head][..., i], vt[head][..., i],
                prev_vf[head][..., i], values[head][..., i],
                logps[head][:, i], old_logp[head][..., i],
                kl[head][:, i], ent[head][:, i],
                mask=alloc_valid[..., i] if head == "alloc" else None,
            )
            totals.append(tot); pols.append(pl); vfs.append(vl); ents.append(en); kls.append(k)

    policy._total_loss = torch.stack(totals).sum()
    policy._mean_policy_loss = torch.stack(pols).mean()
    policy._mean_vf_loss = torch.stack(vfs).mean()
    policy._mean_entropy = torch.stack(ents).mean()
    policy._mean_kl = torch.stack(kls).mean()
    policy._vf_explained_var = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS][:, 0], values["primitive"][:, 0])
    return policy._total_loss


# ---------------------------------------------------------------- fetches / mixins

def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    v = model.value_function()
    return {
        SampleBatch.VF_PREDS: v["primitive"],
        VF_PREDS_ALLOC: v["alloc"],
        ACTION_LOGP_ALLOC: action_dist.logp_alloc(),
    }


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        def value(**input_dict):
            input_dict = SampleBatch(input_dict)
            input_dict = self._lazy_tensor_dict(input_dict)
            self.model(input_dict)
            return self.model.value_function()
        self._value = value


def setup_mixins_override(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)


MultiPPOTorchPolicy = build_policy_class(
    name="MultiPPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=extra_action_out_fn,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    before_loss_init=setup_mixins_override,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin],
)


def get_policy_class(config):
    return MultiPPOTorchPolicy


MultiPPOTrainer = build_trainer(
    name="MultiPPO",
    default_config=ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    validate_config=ray.rllib.agents.ppo.ppo.validate_config,
    default_policy=MultiPPOTorchPolicy,
    get_policy_class=get_policy_class,
    execution_plan=ray.rllib.agents.ppo.ppo.execution_plan,
)
