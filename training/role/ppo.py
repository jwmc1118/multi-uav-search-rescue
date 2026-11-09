"""RolePPOTrainer — 계층 구조 env (env/role/gym_env.py) 의 Role / Rescue / Explore 3 네트워크 MAPPO (CTDE).

하나의 RLlib policy (default_policy) 안의 모델 (models/role/net.py RoleNet). 네트워크 k ∈ {role, rescue, explore} 마다 독립된
actor · critic (encoder · 이종 그래프 GNN · decoder · critic trunk 공유 없음) — 각 loss 는 자기 네트워크에만 gradient 를 준다.
UAV i 마다 따로 보상 / advantage / value target. rescue · explore 결정은 env 의 task 행동 하나를 나눠 쓴다
(rescue = S 칸, explore = F 칸 1 단계 덩어리 선택만 — 2 단계 구간 배정은 env 의 거리 Hungarian, 같은 목표 재선택은 표본 아님).
    보상    info["reward_role" | "reward_rescue" | "reward_explore"]  (N,), 매 step (env/role/simulation.py role_rewards /
            task_rewards — rescue = RESCUE 인 step 의 task 보상, explore = EXPLORE 인 step 의 task 보상)
    advantage — 결정 단위 SMDP GAE, 네트워크 k 의 결정 step = info role_valid / rescue_valid / explore_valid:
            결정 t0 ~ t1:
                δ_k(t0) = Σ_{t0 <= t < t1} γ^(t - t0) r^k(t) + γ^(t1 - t0) V^k(s_t1) - V^k(s_t0)
            role    : t1 = 같은 UAV 의 다음 role 결정, γ = γ_r (role_gamma — env.yaml 최상위, model.custom_model_config.role_gamma)
            rescue / explore : t1 = min(같은 UAV 의 같은 네트워크 다음 결정, 그 UAV 가 그 역할 (info role: RESCUE / EXPLORE) 을
                      벗어나는 step (역할 변경 · RETURN 포함)), γ = 최상위 gamma. 역할을 벗어나 끊긴 구간은 그 시점 상태의 V^k(s_t1) 로
                      bootstrap (역할 변경 = role 이 정하는 외부 사건 -> 잘림) 하고 GAE 체인도 끊는다 (다음 같은 역할 결정은 새 체인).
                      구간 안 보상은 그 역할의 보상만 (다른 역할 구간 보상이 섞이지 않음)
            Â(t0) = δ(t0) + γ^(t1 - t0) λ Â(다음 같은 네트워크 결정)   (체인이 이어질 때만)
            에피소드 끝이면 V(s_t1) = 0. fragment (truncate_episodes) 끝에서 구간이 끝나지 않은 결정은 loss 에서 뺀다
            (마스크 0) — 다음 fragment 에서 그 뒤 결정들이 이어진다.
    loss    L = Σ_k λ_k L^k,  L^k = N ( L^clip_k + β KL_k + c_v L_V^k - c_e^k H_k )
            (λ = model.custom_model_config.loss.lambda_<k>, c_e^k = <k>.entropy_coeff, clip_k = rescue / explore.clip_param,
            role 은 최상위 clip_param — training/config/role.yaml / task.yaml)
            policy / KL / entropy 마스크 = info <k>_train (결정을 실제로 했고 모델 출력 그대로 실행) & 구간 완료
            value 마스크 = 결정 step & 구간 완료. 학습하지 않는 네트워크 (env_config.components 가 rule) 의 표본은
            두 마스크 모두 0
    rule 네트워크: components 가 rule 인 네트워크는 파라미터 requires_grad False
            (models/role/net.py) 이고 optimizer_fn 이 requires_grad 인 파라미터만 넘겨 optimizer 에서도 빠진다
            advantage 는 네트워크별 마스크 표준화 (평균 0, 표준편차 1, 미니배치 단위). 마스크 합이 0 이면 그 항은 0 (NaN 없음)
            grad_clip (config) 은 optimizer param group 마다 (= 네트워크별) gradient norm clip (RLlib apply_grad_clipping)
    학습률  optimizer param group 이 네트워크별 (actor + critic): role = 최상위 lr, rescue / explore = <k>.lr. 최상위 lr_schedule 은
            배율로 모든 group 에 적용: lr_k(t) = lr_k · lr_schedule(t) / lr (NetworkLRSchedule)
    ValueNorm (네트워크별): rollout 의 VF_PREDS* 와 bootstrap 은 실제 스케일, loss 는 미니배치마다 갱신 후 정규화 공간에서.
    진단 stats: 네트워크별 approx KL (mean(logp_old - logp_new)) · clip 비율 (|ratio - 1| > clip_k) — fig_diagnostics.
"""
import numpy as np

import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import kl_and_loss_stats, setup_mixins, KLCoeffMixin
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, explained_variance

from models.role.rllib_model import ACTION_LOGP_TASK, VF_PREDS_EXPLORE, VF_PREDS_RESCUE
from models.role.net import NETWORKS, ROLE_EXPLORE, ROLE_RESCUE

torch, nn = try_import_torch()

# postprocess 가 채우는 컬럼 (네트워크별, role 의 advantage / value target 은 RLlib 기본 키)
ADV = {"role": Postprocessing.ADVANTAGES, "rescue": "advantages_rescue", "explore": "advantages_explore"}
VT = {"role": Postprocessing.VALUE_TARGETS, "rescue": "value_targets_rescue", "explore": "value_targets_explore"}
VF = {"role": SampleBatch.VF_PREDS, "rescue": VF_PREDS_RESCUE, "explore": VF_PREDS_EXPLORE}
LOGP = {"role": SampleBatch.ACTION_LOGP, "rescue": ACTION_LOGP_TASK, "explore": ACTION_LOGP_TASK}
ACT = {"role": "role", "rescue": "task", "explore": "task"}          # 네트워크 -> 행동 공간 키 (rescue / explore 는 task 공용)
ROLE_OF = {"rescue": ROLE_RESCUE, "explore": ROLE_EXPLORE}           # 구간을 끊는 역할 (info role)
PMASK = {k: f"{k}_train_mask" for k in NETWORKS}                # policy / KL / entropy 마스크
VMASK = {k: f"{k}_value_mask" for k in NETWORKS}                # value loss 마스크
PARAM_PREFIX = {"role": ("net.role_net.", "net.critic_role."), "rescue": ("net.rescue_net.", "net.critic_rescue."),
                "explore": ("net.explore_net.", "net.critic_explore.")}   # optimizer param group (model.named_parameters)


# ---------------------------------------------------------------- 결정 단위 advantage

def _infos(batch, n, keys):
    infos = batch.get(SampleBatch.INFOS)
    T = batch[SampleBatch.REWARDS].shape[0]
    out = {k: np.zeros((T, n), np.float32) for k in keys}
    if infos is None or (hasattr(infos, "dtype") and infos.dtype == np.float32):
        return out                                            # trajectory-view 더미 호출
    for t, info in enumerate(infos):
        if isinstance(info, dict):
            for k in keys:
                if k in info:
                    out[k][t] = np.asarray(info[k], np.float32)
    return out


def role_gamma(config):
    """role SMDP 할인 γ_r = model.custom_model_config.role_gamma (config_loader 가 최상위 role_gamma 를 넣음, 없으면 1.0)."""
    g = ((config.get("model") or {}).get("custom_model_config") or {}).get("role_gamma")
    return 1.0 if g is None else float(g)


def decision_advantages(r, v, dec, done, gamma, lam, inside=None):
    """한 UAV 의 결정 단위 SMDP GAE. r / v (T,), dec (T,) bool 결정 step, inside (T,) bool = 그 네트워크의 역할인 step
    (None = 끊지 않음). 결정 t0 의 구간 끝 t1 = min(다음 결정, t0 뒤 처음 inside 가 아닌 step):
    δ(t0) = Σ γ^(t - t0) r(t) + γ^(t1 - t0) V(s_t1) - V(s_t0) (에피소드 끝이면 V = 0),
    Â(t0) = δ(t0) + γ^(t1 - t0) λ Â(다음 결정) — 역할을 벗어나 끊긴 구간은 체인도 끊음.
    반환 (adv, value target, 완료 마스크, 구간 끝 t1) — 결정 step 에만 값 (t1 은 미완료면 -1). fragment 끝에서 구간이 끝나지
    않은 결정은 마스크 0."""
    T = len(r)
    adv = np.zeros(T, np.float32)
    vt = np.zeros(T, np.float32)
    ok = np.zeros(T, np.float32)
    end = np.full(T, -1, np.int64)
    steps = np.flatnonzero(dec)
    nxt_adv = 0.0
    for k in range(len(steps) - 1, -1, -1):
        t0 = int(steps[k])
        nd = int(steps[k + 1]) if k + 1 < len(steps) else T
        t1 = nd
        if inside is not None:
            out = np.flatnonzero(~inside[t0 + 1:nd])
            if len(out):
                t1 = t0 + 1 + int(out[0])                          # 역할을 벗어난 step (잘림)
        if t1 >= T and not done:
            nxt_adv = 0.0
            continue                                           # fragment 끝에서 구간 미완료 -> loss 에서 제외
        disc = gamma ** np.arange(t1 - t0)
        G = float(np.sum(disc * r[t0:t1]))
        boot = v[t1] if t1 < T else 0.0
        chain = lam * nxt_adv if t1 == nd and t1 < T else 0.0
        a = G + gamma ** (t1 - t0) * (boot + chain) - v[t0]
        nxt_adv = a
        adv[t0], vt[t0], ok[t0], end[t0] = a, a + v[t0], 1.0, t1
    return adv, vt, ok, end


def compute_role_gae(policy, batch, other_agent_batches=None, episode=None):
    n = policy.model.n_agents
    keys = ["reward_role", "reward_rescue", "reward_explore", "role", "role_valid", "rescue_valid", "explore_valid",
            "role_train", "rescue_train", "explore_train"]
    inf = _infos(batch, n, keys)
    env_cfg = policy.config.get("env_config") or {}
    lam = policy.config["lambda"]
    gamma = {"role": role_gamma(policy.config), "rescue": float(policy.config["gamma"]),
             "explore": float(policy.config["gamma"])}          # role γ_r / rescue · explore 최상위 gamma
    done = bool(batch[SampleBatch.DONES][-1])
    comp = env_cfg.get("components") or {}
    trained = {k: comp.get(k, "learn") == "learn" for k in NETWORKS}    # 네트워크별 학습 여부 (rule 은 표본 없음)
    role_now = np.round(inf["role"]).astype(int)                 # step 의 역할 (결정 후)
    for k in NETWORKS:
        vf = np.asarray(batch[VF[k]], np.float32).reshape(-1, n)
        T = vf.shape[0]
        dec = inf[f"{k}_valid"] > 0
        adv, vt, okh = np.zeros((T, n), np.float32), np.zeros((T, n), np.float32), np.zeros((T, n), np.float32)
        for i in range(n):
            inside = (role_now[:, i] == ROLE_OF[k]) if k in ROLE_OF else None
            adv[:, i], vt[:, i], okh[:, i], _ = decision_advantages(inf[f"reward_{k}"][:, i], vf[:, i], dec[:, i], done,
                                                                    gamma[k], lam, inside)
        sel = dec.astype(np.float32)
        on = float(trained[k])
        batch[VT[k]] = np.where(okh > 0, vt, vf).astype(np.float32)
        batch[VF[k]] = vf
        batch[VMASK[k]] = okh * sel * on
        batch[ADV[k]] = (adv * sel).astype(np.float32)
        batch[PMASK[k]] = inf[f"{k}_train"] * okh * sel * on
    return batch


# ---------------------------------------------------------------- loss

def _mmean(x, m):
    """마스크 평균. 마스크 합 0 이면 0 (where 로 거르므로 마스크된 -inf / nan 이 섞이지 않는다)."""
    keep = m > 0
    return torch.where(keep, x, torch.zeros_like(x)).sum() / m.sum().clamp(min=1.0)


def _standardize(adv, m):
    """마스크된 원소로 평균 0 / 표준편차 1 (원소가 2 개 미만이면 평균만 뺌)."""
    keep = m > 0
    cnt = keep.sum()
    if int(cnt) == 0:
        return adv
    mu = _mmean(adv, m)
    if int(cnt) < 2:
        return adv - mu
    var = _mmean((adv - mu) ** 2, m)
    return (adv - mu) / (var.sqrt() + 1e-8)


def clip_params(policy, model):
    """{네트워크: PPO clip 범위} — role 은 최상위 clip_param, rescue / explore 는 <k>.clip_param."""
    return {k: float(policy.config["clip_param"] if k == "role" else model.net.cfg[k]["clip_param"]) for k in NETWORKS}


def role_ppo_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch, is_training=True)
    curr = dist_class(logits, model)
    prev = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    logps = curr.logp(train_batch[SampleBatch.ACTIONS])
    ent, kl = curr.entropy(), prev.kl(curr)
    values, values_n = model.value_function(), model.value_function_norm()
    vnorm = model.net.value_norms()
    cfg = policy.config
    ent_c = model.net.entropy_coeffs()
    lam_w = model.net.cfg["loss"]
    clips = clip_params(policy, model)
    n = float(model.n_agents)

    stats = {}
    total, pols, kls, vls, ents = 0.0, [], [], [], []
    for k in NETWORKS:
        pm, vm = train_batch[PMASK[k]].float(), train_batch[VMASK[k]].float()
        # value loss
        vt, prev_vf = train_batch[VT[k]], train_batch[VF[k]]
        vnh = vnorm.get(k)
        if vnh is not None:
            if getattr(policy, "_value_norm_ready", False) and bool((vm > 0).any()):
                vnh.update(vt[vm > 0])
            vt_l, prev_l, cur_l = vnh.normalize(vt), vnh.normalize(prev_vf), values_n[k]
        else:
            vt_l, prev_l, cur_l = vt, prev_vf, values[k]
        v1 = (cur_l - vt_l) ** 2
        v2 = (prev_l + torch.clamp(cur_l - prev_l, -cfg["vf_clip_param"], cfg["vf_clip_param"]) - vt_l) ** 2
        l_v = _mmean(torch.max(v1, v2), vm)
        # policy (clip) / KL / entropy
        a = _standardize(train_batch[ADV[k]], pm)
        clip = clips[k]
        ratio = torch.exp(logps[ACT[k]] - train_batch[LOGP[k]])
        surr = torch.min(a * ratio, a * torch.clamp(ratio, 1 - clip, 1 + clip))
        l_pi, l_kl, l_ent = _mmean(-surr, pm), _mmean(kl[ACT[k]], pm), _mmean(ent[ACT[k]], pm)
        lk = n * (l_pi + policy.kl_coeff * l_kl + cfg["vf_loss_coeff"] * l_v - ent_c[k] * l_ent)   # Σ_i = 마스크 평균 x N
        total = total + float(lam_w[f"lambda_{k}"]) * lk
        pols.append(l_pi)
        kls.append(l_kl)
        vls.append(l_v)
        ents.append(l_ent)
        with torch.no_grad():
            keep = vm > 0
            ev = explained_variance(vt[keep], values[k][keep]) if int(keep.sum()) > 1 else torch.tensor(0.0)
            d = train_batch[LOGP[k]] - logps[ACT[k]]
            stats.update({
                f"vf_explained_var_{k}": ev, f"vf_loss_{k}": l_v, f"value_target_mean_{k}": _mmean(vt, vm),
                f"value_target_std_{k}": (_mmean(vt ** 2, vm) - _mmean(vt, vm) ** 2).clamp(min=0).sqrt(),
                f"entropy_{k}": l_ent.detach(), f"n_decisions_{k}": pm.sum(),
                f"policy_loss_{k}": l_pi, f"kl_{k}": l_kl, f"approx_kl_{k}": _mmean(d, pm),
                f"clip_frac_{k}": _mmean(((ratio - 1).abs() > clip).float(), pm)})
            if vnh is not None:
                mu, sd = vnh.stats()
                stats[f"value_norm_mean_{k}"], stats[f"value_norm_std_{k}"] = mu.reshape(()), sd.reshape(())
    if not (torch.is_tensor(total) and total.requires_grad):   # 학습 대상 표본이 없어 그래프에 안 이어진 미니배치:
        total = total + sum((p * 0.0).sum() for p in model.parameters() if p.requires_grad)   # 값 그대로 · gradient 0
    policy._total_loss = total
    policy._mean_policy_loss = torch.stack(pols).mean()
    policy._mean_vf_loss = torch.stack(vls).mean()
    policy._mean_entropy = torch.stack(ents).mean()
    policy._mean_kl = torch.stack(kls).mean()
    policy._vf_explained_var = stats["vf_explained_var_role"]
    policy._head_stats = {k: v.detach() if torch.is_tensor(v) else v for k, v in stats.items()}
    return total


def stats_fn(policy, train_batch):
    out = kl_and_loss_stats(policy, train_batch)
    out.update(getattr(policy, "_head_stats", {}))
    return out


# ---------------------------------------------------------------- fetches / mixins

def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    v = model.value_function()
    lp = action_dist.logp_taken()
    return {SampleBatch.VF_PREDS: v["role"], VF_PREDS_RESCUE: v["rescue"], VF_PREDS_EXPLORE: v["explore"],
            ACTION_LOGP_TASK: lp["task"]}


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        def value(**input_dict):
            # bootstrap: input_dict 에 ACTIONS 가 없으므로 모델이 이 관측으로 행동을 다시 뽑아 critic 의 a^{-i} 로 쓴다
            input_dict = self._lazy_tensor_dict(SampleBatch(input_dict))
            self.model(input_dict)
            return self.model.value_function()
        self._value = value


def setup_mixins_override(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)


def after_loss_init(policy, obs_space, action_space, config):
    policy._value_norm_ready = True        # 더미 배치 loss 초기화 이후부터 ValueNorm 갱신


def optimizer_fn(policy, config):
    """네트워크별 param group (actor + critic, 학습하는 파라미터만 — rule 네트워크는 빠짐). group 의 base_lr:
    role = 최상위 lr, rescue / explore = <k>.lr (NetworkLRSchedule 이 lr_schedule 배율을 곱한다)."""
    groups = {k: [] for k in NETWORKS}
    for name, p in policy.model.named_parameters():
        if not p.requires_grad:
            continue
        k = next((k for k in NETWORKS if name.startswith(PARAM_PREFIX[k])), None)
        if k is None:
            raise RuntimeError(f"optimizer: 네트워크에 속하지 않는 학습 파라미터 {name}")
        groups[k].append(p)
    net_cfg = policy.model.net.cfg
    lr = {k: float(config["lr"] if k == "role" else net_cfg[k]["lr"]) for k in NETWORKS}
    pg = [{"params": groups[k], "lr": lr[k], "base_lr": lr[k], "net": k} for k in NETWORKS if groups[k]]
    if not pg:                       # learn 이 없는 정책 (전부 rule — 평가 · 테스트용 정책 생성): 갱신할 파라미터 없음
        pg = [{"params": list(policy.model.parameters()), "lr": 0.0, "base_lr": 0.0, "net": "none"}]
    return torch.optim.Adam(pg, lr=config["lr"])


class NetworkLRSchedule(LearningRateSchedule):
    """최상위 lr_schedule 을 네트워크별 lr 에 배율로: group lr = base_lr · cur_lr / lr (lr_schedule 이 없으면 base_lr 그대로)."""

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        scale = self.cur_lr / float(self.config["lr"])
        for opt in self._optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale


def before_init(policy, obs_space, action_space, config):
    setup_config(policy, obs_space, action_space, config)


RolePPOTorchPolicy = build_policy_class(
    name="RolePPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=role_ppo_loss,
    stats_fn=stats_fn,
    extra_action_out_fn=extra_action_out_fn,
    postprocess_fn=compute_role_gae,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    before_init=before_init,
    before_loss_init=setup_mixins_override,
    _after_loss_init=after_loss_init,
    mixins=[NetworkLRSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin],
)

RolePPOTrainer = build_trainer(
    name="RolePPO",
    default_config=ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    validate_config=ray.rllib.agents.ppo.ppo.validate_config,
    default_policy=RolePPOTorchPolicy,
    get_policy_class=lambda config: RolePPOTorchPolicy,
    execution_plan=ray.rllib.agents.ppo.ppo.execution_plan,
)
