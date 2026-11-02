"""에피소드 지표를 RLlib custom_metrics 로 기록하는 콜백 (progress.csv / result.json -> training/reward_plot.py 그림 5 개).

RLlib 의 episode_reward 는 env 가 넘기는 스칼라 (헤드 보상 합, UAV 전체 합) 라서 헤드를 구분할 수 없다.
여기서 info 의 헤드별 보상과 임무 지표를 따로 모아 result["custom_metrics"] 에 넣는다
(progress.csv 에는 custom_metrics/rescue_rate_mean 처럼 _mean / _min / _max 로 기록된다).

env 는 N 대를 묶은 단일 gym.Env 라 RLlib 1.3 이 dummy agent 하나로 감싸고, episode.last_info_for() 로 그 step 의 info 를 읽는다.
training/train.py 가 config["callbacks"] 로 지정한다.
"""
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks

ENC_HEADS = ("role", "task")                          # learner_stats grad_norm_enc_from_<head>
ENC_EV = {"role": ("role",), "task": ("task_rescue", "task_explore")}   # head -> stats 이름 (EV / entropy 경고 표시)


class EncGradMonitor:
    """head 별 (자기 네트워크 encoder · GNN) gradient norm 의 EMA (α) 와 불균형 경고. iteration 마다 update(norms, ev, ent).
    EMA 최대/최소 비율 >= ratio_th 인 iteration 이 streak 번 연속이면 경고 (그 뒤 streak 번마다 다시) — 단발 스파이크는
    EMA 가 흡수하고 연속 조건도 못 채워 경고 없음. 경고에는 그 기간 head 별 EV / entropy 시작 -> 끝."""

    def __init__(self, alpha=0.1, ratio_th=10.0, streak=10):
        self.alpha, self.ratio_th, self.streak_len = alpha, ratio_th, streak
        self.ema, self.streak, self.hist = {}, 0, []

    def update(self, norms, ev=None, ent=None):
        """norms {head: 이번 iteration norm}. 반환 (ema dict, ratio, 경고 문자열 또는 None)."""
        for h, v in norms.items():
            if v is None or not np.isfinite(v) or v <= 0:          # 0 = 이번 iteration 측정 없음
                continue
            self.ema[h] = v if h not in self.ema else (1 - self.alpha) * self.ema[h] + self.alpha * v
        vals = [v for v in self.ema.values() if v > 0]
        ratio = max(vals) / min(vals) if len(vals) >= 2 else 1.0
        self.hist = (self.hist + [(dict(ev or {}), dict(ent or {}))])[-self.streak_len:]
        self.streak = self.streak + 1 if ratio >= self.ratio_th else 0
        warn = None
        if self.streak >= self.streak_len and self.streak % self.streak_len == 0:
            (ev0, en0), (ev1, en1) = self.hist[0], self.hist[-1]
            fmt = lambda a, b, h: f"{h} {a.get(h, float('nan')):.2f}->{b.get(h, float('nan')):.2f}"
            warn = (f"[warn] head 별 encoder gradient norm 불균형: EMA 최대/최소 {ratio:.1f} 배가 {self.streak} iteration 연속 (EMA "
                    + ", ".join(f"{h} {v:.3g}" for h, v in self.ema.items()) + ")\n"
                    + "       최근 " + str(len(self.hist)) + " iteration  EV " + ", ".join(fmt(ev0, ev1, h) for h in ev1)
                    + "  |  entropy " + ", ".join(fmt(en0, en1, h) for h in en1))
        return dict(self.ema), ratio, warn


class RoleCallbacks(DefaultCallbacks):
    """계층 구조 env (env/role/gym_env.py) 지표 — training/reward_plot.py 그림이 쓰는 값만 기록한다.

    에피소드 (custom_metrics, RLlib 이 _mean / _min / _max 로 기록)
        rescue_rate / expire_rate / discover_rate : 구조 · 만료 · 발견 생존자 / N_T                  (fig_training_performance b)
        known_to_rescue  : 구조된 생존자의 발견 -> 구조 step 평균                                    (fig_training_performance d)
        rw_<REWARD_ITEMS> : 보상 항목별 UAV 당 에피소드 합 (env/role/simulation.py REWARD_ITEMS — role_team / role_discover /
                           role_switch / rescue_sf / rescue_bonus / rescue_dup / explore_direct / explore_share)   (fig_rewards)
        role_explore / role_rescue / role_return : 에피소드 평균 역할 비율 (UAV-step)                 (fig_behavior f)
        explore_mult : 직접 탐색 step 의 평균 배율 1 + w_omega ω^g                                    (fig_rewards)
        mem_nodes / mem_age : 결정 시점 그래프의 메모리 노드 수 · 평균 age (step 평균)               (fig_diagnostics)
        switch_voluntary / switch_forced : UAV 당 에피소드 역할 전환 수 (자발적 = periodic + "1", 강제)  (fig_training_performance f)
    iteration (on_train_result)
        decisions_per_batch_{role,task_rescue,task_explore} : train batch 당 표본 종류별 결정 수    (fig_learning c)
        eval/rescue_rate (+ _std), eval/episode_len, eval/n_seeds, eval/diff_mean · diff_lo · diff_hi : eval.every iteration
                           마다 고정 seed · 두 head argmax 평가 + 같은 seed 규칙 정책과 짝 비교 (training/fixed_seed_eval.py)
                           (fig_training_performance a e). 에피소드 기록은 logdir/eval_latest.json (fig_behavior a c d e,
                           fig_undiscovered), 규칙 정책 기록은 logdir/rule_eval.json (fig_behavior b-e, fig_undiscovered)
    (에피소드 길이 = RLlib episode_len_mean, critic EV / entropy = learner_stats — fig_training_performance c, fig_learning a b)
    head 별 encoder gradient norm 불균형은 EncGradMonitor 로 콘솔 경고만 (기록하지 않음).
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        episode.user_data["roles"] = []
        episode.user_data["mem"] = []

    def on_episode_step(self, *, worker, base_env, episode, env_index=None, **kwargs):
        info = episode.last_info_for() or {}
        if "role" in info:
            episode.user_data["roles"].append(np.bincount(np.asarray(info["role"], int), minlength=3) / len(info["role"]))
        if "mem_nodes_dec" in info:
            episode.user_data["mem"].append((info["mem_nodes_dec"], info["mem_age_dec"]))

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        u, m = episode.user_data, episode.custom_metrics
        info = episode.last_info_for() or {}
        if "rescued_total" not in info or not u.get("roles"):
            return
        n_t = max(1, info["rescued_total"] + info["expired_total"] + info["active"])
        m.update(rescue_rate=info["rescued_total"] / n_t, expire_rate=info["expired_total"] / n_t,
                 discover_rate=info["discovered_total"] / n_t)
        roles = np.mean(np.array(u["roles"]), axis=0)
        for j, name in enumerate(("explore", "rescue", "return")):
            m[f"role_{name}"] = float(roles[j])
        es = info.get("event_stats")
        if es:
            if es["known_to_rescue"] == es["known_to_rescue"]:
                m["known_to_rescue"] = es["known_to_rescue"]
            rw = es["rewards"]
            m.update({f"rw_{k}": v for k, v in rw["sum"].items()})
            m["switch_voluntary"] = rw["switch_voluntary_periodic"] + rw["switch_voluntary_ev1"]
            m["switch_forced"] = rw["switch_forced"]
            if rw["explore_mult_mean"] == rw["explore_mult_mean"]:
                m["explore_mult"] = rw["explore_mult_mean"]
        mem = np.array(u.get("mem") or [], float).reshape(-1, 2)
        for j, k in enumerate(("mem_nodes", "mem_age")):
            if np.isfinite(mem[:, j]).any():
                m[k] = float(np.nanmean(mem[:, j]))

    @staticmethod
    def _enc_grad(trainer, st):
        """head 별 encoder gradient norm EMA 불균형 경고 (콘솔) + 다음 iteration 측정 예약 (policy._probe_pending)."""
        mon = getattr(trainer, "_enc_monitor", None)
        if mon is None:
            mon = trainer._enc_monitor = EncGradMonitor()
        norms = {h: st.get(f"grad_norm_enc_from_{h}") for h in ENC_HEADS if f"grad_norm_enc_from_{h}" in st}
        if norms:
            ev = {k: float(st.get(f"vf_explained_var_{k}", float("nan"))) for h in ENC_HEADS for k in ENC_EV[h]}
            ent = {k: float(st.get(f"entropy_{k}", float("nan"))) for h in ENC_HEADS for k in ENC_EV[h]}
            _, _, warn = mon.update({h: float(v) for h, v in norms.items()}, ev, ent)
            if warn:
                print(warn)
        try:
            trainer.workers.local_worker().get_policy()._probe_pending = True
        except Exception:
            pass

    # train.py 가 trainer 를 만들기 전에 넣는다 (yaml eval: {every, n_seeds}). None 이면 평가 안 함
    eval_cfg = None

    def _fixed_seed_eval(self, trainer, result):
        """eval.every iteration 마다 driver 의 policy 로 고정 seed 평가 -> result custom_metrics (이번 iteration 행에 기록)."""
        ec = type(self).eval_cfg
        if not ec or int(ec.get("every") or 0) <= 0 or result["training_iteration"] % int(ec["every"]):
            return
        from training.fixed_seed_eval import evaluate_greedy
        n_seeds = int(ec.get("n_seeds", 20))
        out = evaluate_greedy(trainer.get_policy(), trainer.config["env_config"], n_seeds, logdir=trainer.logdir,
                              iteration=result["training_iteration"])
        result["custom_metrics"].update(out)
        print(f"[eval] iter {result['training_iteration']}: rescue {out['eval/rescue_rate']:.3f} ± {out['eval/rescue_rate_std']:.3f}"
              f"  expired {out['eval/expired_rate']:.3f}  len {out['eval/episode_len']:.1f}  "
              f"explored {out['eval/explored_rate']:.3f}  learned - rule {out['eval/diff_mean']:+.3f} "
              f"[{out['eval/diff_lo']:+.3f}, {out['eval/diff_hi']:+.3f}]  ({out['eval/time_s']:.1f}s, {n_seeds} seeds, greedy)")

    def on_train_result(self, *, trainer, result, **kwargs):
        """헤드별 결정 수: stats 는 미니배치 평균이라 train batch 당 = 미니배치 값 x (train_batch / minibatch)."""
        st = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
        if "n_decisions_role" not in st:
            return
        cfg = trainer.config
        scale = cfg["train_batch_size"] / max(1, cfg["sgd_minibatch_size"])
        per_batch = {h: float(st[f"n_decisions_{h}"]) * scale for h in ("role", "task_rescue", "task_explore")}
        result.setdefault("custom_metrics", {}).update({f"decisions_per_batch_{h}": v for h, v in per_batch.items()})
        self._enc_grad(trainer, st)
        self._fixed_seed_eval(trainer, result)
        mode = (cfg.get("env_config") or {}).get("policy_mode") or {}
        if mode.get("role", "learned") == "learned" and per_batch["role"] < 200:
            print(f"[warn] role 결정이 train batch 당 {per_batch['role']:.0f} 개 (< 200) — "
                  f"training/config/env.yaml 의 train_batch_size 를 키우세요 (현재 {cfg['train_batch_size']})")
