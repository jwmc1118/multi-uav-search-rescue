"""RolePPOTrainer — 계층 구조 env (env/role/gym_env.py) 의 Role / Rescue / Explore 3 헤드 MAPPO (CTDE).

하나의 RLlib policy (default_policy) 안의 multi-head 모델 (models/role/net.py RoleNet). 헤드 h ∈ {role, rescue, explore},
UAV i 마다 따로 보상 / advantage / value target / critic. 헤드마다 독립된 actor 네트워크 (encoder · 이종 그래프 GNN 공유 없음,
models/role/net.py) — 각 헤드 loss 는 자기 네트워크에만 gradient 를 준다.
    explore 헤드 = 1 단계 (덩어리 선택) 만. 2 단계 (같은 덩어리 안 구간 배정) 는 env 의 κ Hungarian (학습 행동 아님).
    보상    info["reward_role" | "reward_rescue" | "reward_explore"]  (N,)  (env/role/simulation.py role_rewards)
    advantage — 결정 단위 (결정 step = info["{h}_valid"]):
            role   : SMDP. 결정 t0 ~ 다음 role 결정 t1:  δ(t0) = Σ_{t0 <= t < t1} γ_r^(t - t0) r(t) + γ_r^(t1 - t0) V(s_t1) - V(s_t0)
                     GAE 도 결정 단위: Â(t0) = δ(t0) + γ_r^(t1 - t0) λ Â(t1)   (γ_r = config gamma — 최상위, env.yaml)
                     에피소드 끝이면 V(s_t1) = 0
            rescue : 옵션 단위. 결정 t0 ~ t1 = 다음 role / rescue 결정 (구조 완료 · 이벤트 5 뒤 재결정) / 에피소드 끝:
                     R = Σ_{t0 <= t < t1} r(t) (할인 없음 — 결정 step 의 (S - F) / N_T · 중복 + 옵션 끝 step 의 개인 보너스),
                     Â = R - V(s_t0), bootstrap 없음
            explore: 옵션 단위. 결정 t0 (1 단계) ~ t1 = 다음 explore 결정 / 역할 변경 (info explore_active = 0) /
                     에피소드 끝 중 먼저 (합류로 Hungarian 이 구간을 다시 배정해도 옵션은 계속):  R = Σ_{t0 <= t < t1} γ^(t - t0) r(t),
                     Â = R - V(s_t0), bootstrap 없음 (γ = config gamma, r = 매 step 커버리지 보상)
            fragment (truncate_episodes) 끝에서 구간이 아직 안 끝난 role / rescue / explore 결정은 loss 에서 뺀다 (마스크 0) — 다음
            fragment 에서 그 뒤 결정들이 이어진다 (bootstrap 하지 않음).
    loss    L = Σ_h λ_h Σ_i ( L_π + β KL + c_v L_V - c_e^h H )        (λ_h = model.custom_model_config.loss.lambda_*)
            policy / KL / entropy 마스크 = info *_train (결정을 실제로 했고 모델 출력 그대로 실행) & 구간 완료
            value 마스크 = 결정 step & 구간 완료. policy_mode 가 rule 인 헤드는 두 마스크 모두 0
            advantage 는 head 별로 마스크된 표준화 (평균 0, 표준편차 1, 미니배치 단위)
            마스크 합이 0 이면 그 항은 0 (NaN 없음). c_e^h = 헤드별 entropy_coeff (training/config/role.yaml 등)
            grad_clip (config) 은 전체 파라미터에 한 번 (apply_grad_clipping)
    ValueNorm (헤드별): rollout 의 VF_PREDS* 와 bootstrap 은 실제 스케일, loss 는 미니배치마다 갱신 후 정규화 공간에서.
    encoder gradient norm: train iteration 마다 1 회 (그 iteration 의 첫 loss 호출) head 별 loss 로 그 헤드 네트워크의
            encoder · GNN 파라미터의 gradient norm (grad_norm_enc_from_role / _exp / _res). EMA / 불균형 경고는 training/callbacks.py
            EncGradMonitor (EMA 최대/최소 비율 10 배 이상이 10 iteration 연속일 때만)
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

from models.role.rllib_model import ACTION_LOGP_EXPLORE, ACTION_LOGP_RESCUE, VF_PREDS_EXPLORE, VF_PREDS_RESCUE
from models.role.net import ROLE_HEADS

torch, nn = try_import_torch()

# postprocess 가 채우는 헤드별 컬럼 (role 헤드의 advantage / value target 은 RLlib 기본 키)
ADV = {"role": Postprocessing.ADVANTAGES, "rescue": "advantages_rescue", "explore": "advantages_explore"}
VT = {"role": Postprocessing.VALUE_TARGETS, "rescue": "value_targets_rescue", "explore": "value_targets_explore"}
VF = {"role": SampleBatch.VF_PREDS, "rescue": VF_PREDS_RESCUE, "explore": VF_PREDS_EXPLORE}
LOGP = {"role": SampleBatch.ACTION_LOGP, "rescue": ACTION_LOGP_RESCUE, "explore": ACTION_LOGP_EXPLORE}
PMASK = {h: f"{h}_train_mask" for h in ROLE_HEADS}            # policy / KL / entropy 마스크
VMASK = {h: f"{h}_value_mask" for h in ROLE_HEADS}            # value loss 마스크
SHORT = {"role": "role", "explore": "exp", "rescue": "res"}    # grad_norm_enc_from_<short>


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


def decision_advantages(r, v, dec, boundary, done, gamma, lam=None):
    """한 UAV 의 결정 단위 advantage. r / v (T,), dec (T,) bool 결정 step, boundary (T,) bool 구간을 끝내는 step
    (그 step 에 새 결정이 있음). lam=None 이면 구간 끝 종료 (Â = G - V), 아니면 SMDP GAE (다음 결정의 V 로 bootstrap,
    γ^(t1 - t0), Â(t0) = δ(t0) + γ^(t1 - t0) λ Â(t1)). 반환 (adv, value target, 완료 마스크) — 결정 step 에만 값."""
    T = len(r)
    adv = np.zeros(T, np.float32)
    vt = np.zeros(T, np.float32)
    ok = np.zeros(T, np.float32)
    steps = np.flatnonzero(dec)
    nxt_adv = 0.0
    for k in range(len(steps) - 1, -1, -1):
        t0 = int(steps[k])
        later = np.flatnonzero(boundary[t0 + 1:])
        t1 = t0 + 1 + int(later[0]) if len(later) else T
        if t1 >= T and not done:
            nxt_adv = 0.0
            continue                                           # fragment 끝에서 구간 미완료 -> loss 에서 제외
        disc = gamma ** np.arange(t1 - t0)
        G = float(np.sum(disc * r[t0:t1]))
        if lam is None:
            a = G - v[t0]
        else:
            boot = v[t1] if t1 < T else 0.0                  # t1 = 다음 role 결정 (dec 와 boundary 가 같음)
            delta = G + gamma ** (t1 - t0) * boot - v[t0]
            a = delta + (gamma ** (t1 - t0)) * lam * (nxt_adv if t1 < T else 0.0)
            nxt_adv = a
        adv[t0], vt[t0], ok[t0] = a, a + v[t0], 1.0
    return adv, vt, ok


def compute_role_gae(policy, batch, other_agent_batches=None, episode=None):
    n = policy.model.n_agents
    keys = [f"reward_{h}" for h in ROLE_HEADS] + [f"{h}_train" for h in ROLE_HEADS] + \
        ["role_valid", "rescue_valid", "explore_valid", "explore_active"]
    inf = _infos(batch, n, keys)
    env_cfg = policy.config.get("env_config") or {}
    lam = policy.config["lambda"]
    gamma = float(policy.config["gamma"])                      # role SMDP / explore 옵션 할인 (rescue 옵션은 할인 없음)
    done = bool(batch[SampleBatch.DONES][-1])
    vf = {h: np.asarray(batch[VF[h]], np.float32).reshape(-1, n) for h in ROLE_HEADS}
    mode = env_cfg.get("policy_mode") or {}
    T = vf["role"].shape[0]
    for h in ROLE_HEADS:
        dec = inf[f"{h}_valid"] > 0
        adv, vt, ok = np.zeros((T, n), np.float32), np.zeros((T, n), np.float32), np.zeros((T, n), np.float32)
        for i in range(n):
            if h == "role":
                adv[:, i], vt[:, i], ok[:, i] = decision_advantages(inf["reward_role"][:, i], vf[h][:, i], dec[:, i],
                                                                    dec[:, i], done, gamma, lam)
            elif h == "explore":                               # 옵션: 다음 explore 결정 / 재선택 / EXPLORE 아님 에서 끝, bootstrap 없음
                boundary = dec[:, i] | (inf["explore_active"][:, i] < 0.5)
                adv[:, i], vt[:, i], ok[:, i] = decision_advantages(inf["reward_explore"][:, i], vf[h][:, i], dec[:, i],
                                                                    boundary, done, gamma)
            else:                                              # rescue 옵션: 다음 role / rescue 결정에서 끝, 할인 없음, bootstrap 없음
                boundary = dec[:, i] | (inf["role_valid"][:, i] > 0)
                adv[:, i], vt[:, i], ok[:, i] = decision_advantages(inf["reward_rescue"][:, i], vf[h][:, i], dec[:, i],
                                                                    boundary, done, 1.0)
        vt = np.where(ok > 0, vt, vf[h]).astype(np.float32)
        rule = mode.get(h) == "rule"
        batch[ADV[h]] = adv
        batch[VT[h]] = vt
        batch[VF[h]] = vf[h]
        batch[PMASK[h]] = np.zeros_like(adv) if rule else inf[f"{h}_train"] * ok
        batch[VMASK[h]] = np.zeros_like(adv) if rule else ok
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


def role_ppo_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch, is_training=True)
    curr = dist_class(logits, model)
    prev = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    logps = curr.logp(train_batch[SampleBatch.ACTIONS])
    ent, kl = curr.entropy(), prev.kl(curr)
    values, values_n = model.value_function(), model.value_function_norm()
    vnorm = model.net.value_norms()
    cfg = policy.config
    ent_c = model.net.cfg["entropy_coeff_heads"]
    lam_w = model.net.cfg["loss"]

    total, stats, head_loss = 0.0, {}, {}
    pols, vfs, ents, kls = [], [], [], []
    n = float(model.n_agents)
    for h in ROLE_HEADS:
        pm, vm = train_batch[PMASK[h]].float(), train_batch[VMASK[h]].float()
        adv, vt, prev_vf = _standardize(train_batch[ADV[h]], pm), train_batch[VT[h]], train_batch[VF[h]]
        vnh = vnorm.get(h) if vnorm else None
        if vnh is not None:
            if getattr(policy, "_value_norm_ready", False) and bool((vm > 0).any()):
                vnh.update(vt[vm > 0])
            vt_l, prev_l, cur_l = vnh.normalize(vt), vnh.normalize(prev_vf), values_n[h]
        else:
            vt_l, prev_l, cur_l = vt, prev_vf, values[h]
        ratio = torch.exp(logps[h] - train_batch[LOGP[h]])
        surr = torch.min(adv * ratio, adv * torch.clamp(ratio, 1 - cfg["clip_param"], 1 + cfg["clip_param"]))
        v1 = (cur_l - vt_l) ** 2
        v2 = (prev_l + torch.clamp(cur_l - prev_l, -cfg["vf_clip_param"], cfg["vf_clip_param"]) - vt_l) ** 2
        l_pi, l_kl, l_ent = _mmean(-surr, pm), _mmean(kl[h], pm), _mmean(ent[h], pm)
        l_v = _mmean(torch.max(v1, v2), vm)
        # Σ_i: 에이전트별 항의 합 = 마스크 평균 x N (에이전트 수에 비례하는 스케일), λ_h 로 head 가중
        lh = n * (l_pi + policy.kl_coeff * l_kl + cfg["vf_loss_coeff"] * l_v - ent_c[h] * l_ent)
        head_loss[h] = lh
        total = total + float(lam_w[f"lambda_{SHORT[h]}"]) * lh
        pols.append(l_pi); vfs.append(l_v); ents.append(l_ent); kls.append(l_kl)
        with torch.no_grad():
            keep = vm > 0
            ev = explained_variance(vt[keep], values[h][keep]) if int(keep.sum()) > 1 else torch.tensor(0.0)
            stats.update({
                f"vf_explained_var_{h}": ev, f"policy_loss_{h}": l_pi, f"vf_loss_{h}": l_v,
                f"entropy_{h}": l_ent, f"kl_{h}": l_kl, f"n_decisions_{h}": pm.sum(),
                f"value_target_mean_{h}": _mmean(vt, vm),
                f"value_target_std_{h}": (_mmean(vt ** 2, vm) - _mmean(vt, vm) ** 2).clamp(min=0).sqrt(),
            })
            if h == "explore":
                stats["explore_logp_finite"] = torch.isfinite(torch.where(pm > 0, logps[h], torch.zeros_like(logps[h]))).all().float()
            if vnh is not None:
                mu, sd = vnh.stats()
                stats[f"value_norm_mean_{h}"], stats[f"value_norm_std_{h}"] = mu.reshape(()), sd.reshape(())
    _probe_grad_norms(policy, model, head_loss)
    stats.update(getattr(policy, "_grad_probe", {}))
    policy._total_loss = total
    policy._mean_policy_loss = torch.stack(pols).mean()
    policy._mean_vf_loss = torch.stack(vfs).mean()
    policy._mean_entropy = torch.stack(ents).mean()
    policy._mean_kl = torch.stack(kls).mean()
    policy._vf_explained_var = stats["vf_explained_var_role"]
    policy._head_stats = {k: v.detach() if torch.is_tensor(v) else v for k, v in stats.items()}
    return total


def _probe_grad_norms(policy, model, head_loss):
    """iteration 마다 1 회 (callbacks 의 on_train_result 가 policy._probe_pending 을 다시 켠다) head 별 loss 로 그 헤드
    네트워크의 encoder · GNN 파라미터 gradient norm. 불균형 판정은 EMA 로 callbacks.EncGradMonitor 에서."""
    enc = model.net.encoder_parameters()
    if not enc or not getattr(policy, "_probe_pending", True):
        return
    policy._probe_pending = False
    out = {}
    for h, l in head_loss.items():
        params = [p for p in enc[h] if p.requires_grad]
        if not l.requires_grad or not params:
            out[h] = 0.0
            continue
        g = torch.autograd.grad(l, params, retain_graph=True, allow_unused=True)
        out[h] = float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)) if any(x is not None for x in g) else 0.0)
    policy._grad_probe = {f"grad_norm_enc_from_{SHORT[h]}": v for h, v in out.items()}


def stats_fn(policy, train_batch):
    out = kl_and_loss_stats(policy, train_batch)
    out.update(getattr(policy, "_head_stats", {}))
    return out


# ---------------------------------------------------------------- fetches / mixins

def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    v = model.value_function()
    lp = action_dist.logp_taken()
    return {SampleBatch.VF_PREDS: v["role"], VF_PREDS_RESCUE: v["rescue"], VF_PREDS_EXPLORE: v["explore"],
            ACTION_LOGP_RESCUE: lp["rescue"], ACTION_LOGP_EXPLORE: lp["explore"]}


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
    policy._probe_pending = True           # 더미 배치가 써 버린 encoder gradient 측정을 첫 iteration 에 다시


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
    before_init=before_init,
    before_loss_init=setup_mixins_override,
    _after_loss_init=after_loss_init,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin],
)

RolePPOTrainer = build_trainer(
    name="RolePPO",
    default_config=ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    validate_config=ray.rllib.agents.ppo.ppo.validate_config,
    default_policy=RolePPOTorchPolicy,
    get_policy_class=lambda config: RolePPOTorchPolicy,
    execution_plan=ray.rllib.agents.ppo.ppo.execution_plan,
)
