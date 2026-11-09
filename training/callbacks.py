"""에피소드 지표를 RLlib custom_metrics 로 기록하는 콜백 (progress.csv / result.json -> training/reward_plot.py 그림 5 개).

RLlib 의 episode_reward 는 env 가 넘기는 스칼라 (네트워크별 보상 합, UAV 전체 합) 라서 네트워크를 구분할 수 없다.
여기서 info 의 네트워크별 보상과 임무 지표를 따로 모아 result["custom_metrics"] 에 넣는다
(progress.csv 에는 custom_metrics/rescue_rate_mean 처럼 _mean / _min / _max 로 기록된다).

env 는 N 대를 묶은 단일 gym.Env 라 RLlib 1.3 이 dummy agent 하나로 감싸고, episode.last_info_for() 로 그 step 의 info 를 읽는다.
training/train.py 가 config["callbacks"] 로 지정한다.
"""
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks

from training.behavior import BASELINES

NETS = ("role", "rescue", "explore")                  # 네트워크 (learner_stats <stat>_<net>)


class RoleCallbacks(DefaultCallbacks):
    """계층 구조 env (env/role/gym_env.py) 지표 — training/reward_plot.py 그림이 쓰는 값만 기록한다.

    에피소드 (custom_metrics, RLlib 이 _mean / _min / _max 로 기록)
        rescue_rate / expire_rate / discover_rate : 구조 · 만료 · 발견 생존자 / N_T                  (fig_training_performance b)
        known_to_rescue  : 구조된 생존자의 발견 -> 구조 step 평균                                    (fig_training_performance d)
        rw_<REWARD_ITEMS> : 보상 항목별 UAV 당 에피소드 합 (env/role/simulation.py REWARD_ITEMS — role_team / role_discover /
                           role_switch / task_undiscovered / task_wait / task_found_expire / task_conflict /
                           task_switch_rescue / task_switch_explore)      (fig_rewards Role / Rescue / Explore 행)
        role_explore / role_rescue / role_return : 에피소드 평균 역할 비율 (UAV-step)                 (fig_behavior f)
        mem_nodes / mem_age : 결정 시점 그래프의 메모리 노드 수 · 평균 age (step 평균)               (fig_diagnostics)
        switch_voluntary / switch_forced : UAV 당 에피소드 역할 전환 수 (자발적 = periodic + "1", 강제)  (fig_training_performance f)
    iteration (on_train_result)
        decisions_per_batch_{role,rescue,explore} : train batch 당 네트워크별 학습 결정 수            (fig_learning c)
        eval/rescue_rate (+ _std), eval/expired_rate, eval/episode_len, eval/explored_rate, eval/n_seeds : eval.every iteration
                           마다 학습 중인 모델만 고정 seed · 모든 네트워크 argmax 평가 (training/fixed_seed_eval.py)
                           (fig_training_performance a e). 에피소드 기록은 logdir/eval_latest.json (fig_behavior a-d,
                           fig_undiscovered). 기준선 (eval.baselines, config 숫자) 대비 차이는 [eval] 로그 · 그림에만
    (에피소드 길이 = RLlib episode_len_mean, critic EV / entropy = learner_stats — fig_training_performance c, fig_learning a b)
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
        mem = np.array(u.get("mem") or [], float).reshape(-1, 2)
        for j, k in enumerate(("mem_nodes", "mem_age")):
            if np.isfinite(mem[:, j]).any():
                m[k] = float(np.nanmean(mem[:, j]))

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
        r = out["eval/rescue_rate"]
        vs = "".join(f"  vs {BASELINES[k][0]} {r - float(v):+.3f}" for k, v in (ec.get("baselines") or {}).items()
                     if v is not None)
        print(f"[eval] iter {result['training_iteration']}: rescue {r:.3f} ± {out['eval/rescue_rate_std']:.3f}"
              f"  expired {out['eval/expired_rate']:.3f}  len {out['eval/episode_len']:.1f}  "
              f"explored {out['eval/explored_rate']:.3f}{vs}  ({out['eval/time_s']:.1f}s, {n_seeds} seeds, greedy)")

    def on_train_result(self, *, trainer, result, **kwargs):
        """네트워크별 결정 수: stats 는 미니배치 평균이라 train batch 당 = 미니배치 값 x (train_batch / minibatch)."""
        st = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
        if "n_decisions_role" not in st:
            return
        cfg = trainer.config
        scale = cfg["train_batch_size"] / max(1, cfg["sgd_minibatch_size"])
        per_batch = {h: float(st[f"n_decisions_{h}"]) * scale for h in NETS}
        result.setdefault("custom_metrics", {}).update({f"decisions_per_batch_{h}": v for h, v in per_batch.items()})
        self._fixed_seed_eval(trainer, result)
        comp = (cfg.get("env_config") or {}).get("components") or {}
        if comp.get("role", "learn") == "learn" and per_batch["role"] < 200:
            print(f"[warn] role 결정이 train batch 당 {per_batch['role']:.0f} 개 (< 200) — "
                  f"training/config/env.yaml 의 train_batch_size 를 키우세요 (현재 {cfg['train_batch_size']})")
