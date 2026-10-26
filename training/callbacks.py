"""에피소드 지표를 RLlib custom_metrics 로 기록하는 콜백 (TensorBoard / progress.csv / reward_curve.png).

RLlib 의 episode_reward 는 env 가 넘기는 스칼라 (헤드 보상 합, UAV 전체 합) 라서 헤드를 구분할 수 없다.
여기서 info 의 헤드별 보상과 임무 지표를 따로 모아 result["custom_metrics"] 에 넣는다
(progress.csv 에는 custom_metrics/rescue_rate_mean 처럼 _mean / _min / _max 로 기록된다).

env 는 N 대를 묶은 단일 gym.Env 라 RLlib 1.3 이 dummy agent 하나로 감싸고, episode.last_info_for() 로 그 step 의 info 를 읽는다.
training/train.py 가 config["callbacks"] 로 지정한다.
"""
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks

ENC_HEADS = ("role", "exp", "res")                     # learner_stats grad_norm_enc_from_<head>
ENC_EV = {"role": "role", "exp": "explore", "res": "rescue"}


def _policy_net(policies):
    """rollout worker 의 RoleNet (없으면 None)."""
    try:
        return policies["default_policy"].model.net
    except Exception:
        return None


class EncGradMonitor:
    """공유 encoder head 별 gradient norm 의 EMA (α) 와 불균형 경고. iteration 마다 update(norms, ev, ent).
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
            warn = (f"[warn] 공유 encoder gradient norm 불균형: EMA 최대/최소 {ratio:.1f} 배가 {self.streak} iteration 연속 (EMA "
                    + ", ".join(f"{h} {v:.3g}" for h, v in self.ema.items()) + ")\n"
                    + "       최근 " + str(len(self.hist)) + " iteration  EV " + ", ".join(fmt(ev0, ev1, h) for h in ev1)
                    + "  |  entropy " + ", ".join(fmt(en0, en1, h) for h in en1))
        return dict(self.ema), ratio, warn


class RoleCallbacks(DefaultCallbacks):
    """계층 구조 env (env/role/gym_env.py) 에피소드 지표.

    eval/rescue_rate / eval/expired_rate / eval/episode_len / eval/explored_rate (+ _std) / eval/time_s :
                       (train iteration, eval.every 마다) 고정 seed · 세 head argmax 평가 (evaluation/fixed_seed_eval.py)
    learner/cur_lr   : (train iteration) 현재 학습률 (lr_schedule 적용 값)

    reward_{role,rescue,explore} : 에피소드 보상 합 (UAV 평균)
    rescue_rate / expire_rate / discover_rate / explored / episode_len
    same_cell        : 같은 칸 2 대 이상 누적 (depot 제외)
    conflict_cross   : 서로 다른 통신 그룹의 UAV 가 같은 조난자를 쫓은 step 수
    group_size       : 평균 통신 그룹 크기 (UAV 별 |G_i| 의 step 평균)
    ev_1..ev_5 / escalation : Role 이벤트 번호별 횟수 (트리거 UAV 기준) / 하위 할당 실패 대체 횟수
    ev_split / ev_merge : 덩어리 쪼개짐 / 합쳐짐 횟수 (재결정 없음, 기록용)
    dec_interval / latch_hold / preempt_candidates : UAV 당 평균 결정 간격 (step) / goal latch 가 목표를 유지시킨 횟수 /
                       이벤트 2 의 새 생존자 중 EXPLORE 는 시간 안에 못 가지만 RESCUE UAV 는 갈 수 있던 수
    link_made / link_broken / group_size_ep / nbr_bin{0..4} : 통신 링크 생성 / 해제, 평균 그룹 크기, 1-hop 이웃 거리 / R_c
                       구간 비율 (0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1, 1 초과)
    explore_s1 / explore_s2 : explore 1 단계 (덩어리) / 2 단계 (구간, 정책 결정) 결정 수
    m_mean           : 담당자가 있는 덩어리당 평균 m (step 평균)
    ghost_drop       : 통신 이탈로 m 이 줄어든 (추정 담당으로 남긴) 횟수
    seg_life         : 구간을 받은 뒤 소진까지 평균 step
    redecide_uavs / goal_change / oscillation : 재결정한 UAV 수 / 목표 변경 / 진동 (osc_window 안 목표 A -> B -> A)
    partial_return / partial_deficit / rescued_after_partial : 부분 귀환 횟수 / 그때 부족분 평균 /
                       부분 귀환 충전 뒤 다음 충전까지 구조 수 평균
    role_{zero,partial,full}_{explore,rescue,return} : 용량 상태별 역할 선택 비율
    rw_role_team / rw_role_osc / rw_rescue_sf / rw_rescue_bonus / rw_rescue_dup / rw_explore_cov : 보상 항목별 에피소드 합 (UAV 평균)
    blob_dist_ratio  : 1 단계 (고른 덩어리 BFS + 1) / (가능한 덩어리 중 가장 가까운 BFS + 1) 평균
    new_cells_per_explore_step : EXPLORE UAV-step 당 새로 관측한 셀 수
    res_redecide_{1..5} : RESCUE 중이던 UAV 를 재결정시킨 이벤트 원인별 횟수 (3 / 5 만 나와야 함)
    join_events / reselect_uavs / reselect_kept_goal / reselect_goal_change : 에피소드당 합류 (재선택) 횟수 / 2 단계를 다시
                       고른 기존 배정자 수 / 그중 자기 goal 이 든 구간을 고른 비율 / 재선택 때문에 바뀐 goal 수
    ev2_rescue_by_claim / ev2_rescue_other / claim_not_rescue : 이벤트 2 에서 RESCUE 를 고른 UAV 중 차례가 됐을 때의 담당 후보
                       생존자를 고른 비율 / 담당이 아닌 생존자를 고른 비율 / 담당 후보가 있던 UAV 가 RESCUE 를 고르지 않은 비율
    s1_claim_kept / s1_empty_blobs : 1 단계에서 담당 후보 덩어리를 그대로 고른 비율 / 1 단계 이벤트 시점 배정 0 덩어리 수 평균
    s1_participants  : 1 단계 이벤트 (그룹) 당 참여 UAV 수 평균 (영역이 없는 신규 EXPLORE UAV 만)
    ev2_retained / ev2_retained_blob_changed : 이벤트 2 에서 EXPLORE 를 유지한 UAV 수 / 그중 영역이 바뀐 수 (0 이어야 함)
    ev2_redecide / ev2_rescue : 이벤트 2 한 번당 재결정 UAV 수 / 그중 RESCUE 수
    known_to_rescue  : 생존자 발견 -> 구조 (담당 UAV 도착) 까지 step 평균
    explore_reselect : 에피소드당 stage-2-only 결정 (정책 호출) 수, explore_s2only_ratio (learner) = 학습 explore sample 중 비율
    conn/real_mean   : (진단) 에피소드 평균 실제 연결성 C_t (팀 전체 현재 위치, 쌍 비율)
    rescue/c_bar     : rescue 결정당 c̄ (옵션 동안 개인 연결도 c_{i,t} 평균)
    explore/conn_mult_mean : 새 셀이 있는 EXPLORE step 의 연결 배율 (1 + w_mult c_{i,t}) 평균
    reward/role_rescue_term : role step 보상 (|R| - w_x |X|) / N_T 의 에피소드 합 (= rw_role_team)
    rescue/S / rescue/F / rescue/U_eps_size / rescue/F_unavoidable / rescue/bonus_rate : rescue 이벤트 평균 S, F, |U_ε|,
                       max(0, |U_ε| - |G_res|), F = 0 비율
    reward/rescue_SF_term / reward/rescue_bonus_term : (S - F) / N_T · 보너스의 에피소드 합 (그룹 이벤트마다 1 회)
    explore/cov_own_return / explore/option_len / reward/explore_term : explore 옵션 (결정 ~ 다음 explore 결정 / 역할 변경 / 끝)
                       당 커버리지 보상 합 (할인 없음, 학습의 할인 R 평균은 learner value_target_mean_explore) / 옵션 길이 평균 /
                       팀 전체 explore 보상의 에피소드 합
    psi_f_bin{0..9}  : 에피소드 진행 0~100% 를 10 구간으로 나눈 ψ^f (팀 전체 발견 비율) 평균
    role/explore_masked_decisions / role/forced_return_all_masked : exploration complete 로 EXPLORE 가 막힌 role 결정 수 /
                       모든 역할이 막혀 RETURN 을 허용한 role 결정 수
    ev5_exploration_complete / ev5_no_segment / ev5_rescue_invalid : 이벤트 5 원인별 횟수
    obs/survivor_truncated : 그룹이 아는 생존자가 s_max 를 넘어 S 노드가 잘린 role 결정 횟수 (에피소드당)
    n_groups / group_size_ep : step 평균 통신 그룹 수 / 크기 (dup_out = 그룹 밖 중복, s1_empty_blobs = 빈 영역,
                       new_cells_per_explore_step = step 당 새 셀)
    ev2_rescue_nearest : 이벤트 2 에서 RESCUE 를 고른 UAV 가 그 생존자에 가장 가까운 그룹 UAV 였던 비율
    gnn_ms_per_decision / gnn_ms_per_call : (rollout) GNN forward 시간 — 결정 (그래프) 1 개당 / 호출 1 회당 ms
    model_params / gnn_params : (train iteration) 모델 전체 / 공유 GNN 파라미터 수
    enc_grad_ema_{role,exp,res} / enc_grad_ratio / enc_grad_streak : (train iteration) 공유 encoder gradient norm EMA,
                       EMA 최대/최소 비율, 10 배 이상 연속 iteration 수
    rescue_events    : rescue 보상 이벤트 (그룹) 수
    role_osc / role_forced / role_voluntary : 역할 진동 (penalty) / 강제 전환 / 비강제 전환 횟수
    dup_in / dup_out : 중복 추적 penalty 횟수 (같은 그룹 / 그룹 밖의 더 가까운 UAV)
    (공유 encoder head 별 gradient norm 은 learner_stats 의 grad_norm_enc_from_role / _exp / _res)
    role_switch      : 에피소드당 역할 전환 횟수
    role_{explore,rescue,return}_bin{0..9} : 에피소드 진행 0~100% 를 10 구간으로 나눈 역할 비율
    cap0_explore     : 용량 0 인 UAV-step 중 EXPLORE 비율 (return vs explore 판단 분석용)
    """
    EVENTS = ("1", "2", "3", "4", "5", "split", "merge")

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        u = episode.user_data
        u.update({"r_role": 0.0, "r_rescue": 0.0, "r_explore": 0.0, "conflict": 0, "group": [], "roles": [],
                  "events": {k: 0 for k in self.EVENTS}, "esc": 0, "cap0": 0, "cap0_explore": 0, "m": []})
        net = _policy_net(policies)
        if net is not None and net.timing is None:
            net.timing = {}                                   # GNN forward 시간 집계 (on_episode_end 에서 읽고 비움)

    def on_episode_step(self, *, worker, base_env, episode, env_index=None, **kwargs):
        info = episode.last_info_for() or {}
        if "reward_role" not in info:
            return
        u = episode.user_data
        for h in ("role", "rescue", "explore"):
            u[f"r_{h}"] += float(np.mean(info[f"reward_{h}"]))
        u["conflict"] += int(info["rescue_conflict_cross"])
        u["group"].append(info["group_size_mean"])
        u["roles"].append(np.bincount(np.asarray(info["role"], int), minlength=3) / len(info["role"]))
        for k, v in info["events"].items():
            if k in u["events"]:
                u["events"][k] += v
        if info.get("m_mean", 0) > 0:
            u["m"].append(info["m_mean"])
        u["esc"] += int(info["redecide"])
        u["cap0"] += info["cap0"]
        u["cap0_explore"] += info["cap0_explore"]

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        u, m = episode.user_data, episode.custom_metrics
        info = episode.last_info_for() or {}
        if "rescued_total" not in info or not u.get("roles"):
            return
        n_t = max(1, info["rescued_total"] + info["expired_total"] + info["active"])
        for h in ("role", "rescue", "explore"):
            m[f"reward_{h}"] = u[f"r_{h}"]
        m.update(rescue_rate=info["rescued_total"] / n_t, expire_rate=info["expired_total"] / n_t,
                 discover_rate=info["discovered_total"] / n_t, explored=info["explored"], episode_len=episode.length,
                 same_cell=info["same_cell_total"], conflict_cross=u["conflict"], group_size=float(np.mean(u["group"])),
                 escalation=u["esc"], cap0_explore=u["cap0_explore"] / max(1, u["cap0"]))
        for k, v in u["events"].items():
            m[f"ev_{k}"] = v
        st = info.get("explore_stats")
        if st:
            m.update(explore_s1=st["s1"], explore_s2=st["s2"], explore_reselect=st["reselect"], ghost_drop=st["ghost_drop"],
                     role_switch=st["role_switch"])
            if st["seg_life"]:
                m["seg_life"] = float(np.mean(st["seg_life"]))
        es = info.get("event_stats")
        if es and "rewards" in es:
            rs = es["rewards"]
            m.update({f"rw_{k}": v for k, v in rs["sum"].items()})
            m["reward/role_rescue_term"] = rs["role_rescue_term"]
            m["reward/rescue_SF_term"] = rs["rescue_SF_term"]
            m["reward/rescue_bonus_term"] = rs["rescue_bonus_term"]
            m["reward/explore_term"] = rs["explore_term"]
            m.update(rescue_events=rs["rescue_events"], role_osc=rs["osc"], role_forced=rs["forced"],
                     role_voluntary=rs["voluntary"], dup_in=rs["dup_in"], dup_out=rs["dup_out"])
            for k, name in (("conn_real_mean", "conn/real_mean"), ("rescue_c_bar", "rescue/c_bar"),
                            ("explore_conn_mult", "explore/conn_mult_mean"),
                            ("rescue_S", "rescue/S"), ("rescue_F", "rescue/F"), ("rescue_U_eps_size", "rescue/U_eps_size"),
                            ("rescue_F_unavoidable", "rescue/F_unavoidable"), ("rescue_bonus_rate", "rescue/bonus_rate"),
                            ("explore_return", "explore/cov_own_return"), ("explore_option_len", "explore/option_len")):
                if rs[k] == rs[k]:
                    m[name] = rs[k]
            for b, v in enumerate(rs["psi_f_bins"]):
                if v == v:
                    m[f"psi_f_bin{b}"] = v
        if es and "link_made" in es:
            m.update(preempt_candidates=es["preempt_candidates"], latch_hold=es["latch_hold"], link_made=es["link_made"],
                     link_broken=es["link_broken"])
            if es["dec_interval"] == es["dec_interval"]:
                m["dec_interval"] = es["dec_interval"]
            if es["group_size"] == es["group_size"]:
                m["group_size_ep"] = es["group_size"]
            for b, v in enumerate(es["nbr_dist_hist"]):
                m[f"nbr_bin{b}"] = v
        if es:
            m.update(redecide_uavs=es["redecide_uavs"], goal_change=es["goal_change"], oscillation=es["oscillation"],
                     partial_return=es["partial_return"])
            m["obs/survivor_truncated"] = es.get("survivor_truncated", 0)
            m["role/explore_masked_decisions"] = es["explore_masked_decisions"]
            m["role/forced_return_all_masked"] = es["forced_return_all_masked"]
            for c, v in es["ev5_causes"].items():
                m[f"ev5_{c}"] = v
            m.update(ev2_retained=es["ev2_retained"], ev2_retained_blob_changed=es["ev2_retained_blob_changed"])
            m.update(join_events=es["join_events"], reselect_uavs=es["reselect_uavs"],
                     reselect_goal_change=es["reselect_goal_change"])
            for k in ("blob_dist_ratio", "new_cells_per_explore_step", "reselect_kept_goal", "ev2_rescue_by_claim",
                      "ev2_rescue_other", "claim_not_rescue", "s1_claim_kept", "s1_empty_blobs", "s1_participants",
                      "ev2_redecide", "ev2_rescue", "known_to_rescue", "n_groups", "ev2_rescue_nearest"):
                if es.get(k, float("nan")) == es.get(k, float("nan")):
                    m[k] = es[k]
            for e, v in es["rescue_redecide_causes"].items():
                m[f"res_redecide_{e}"] = v
            if es["partial_deficits"]:
                m["partial_deficit"] = float(np.mean(es["partial_deficits"]))
            if es["rescued_after_partial"]:
                m["rescued_after_partial"] = float(np.mean(es["rescued_after_partial"]))
            for c, cnt in es["role_by_cap"].items():
                tot = sum(cnt)
                if tot:
                    for j, name in enumerate(("explore", "rescue", "return")):
                        m[f"role_{c}_{name}"] = cnt[j] / tot
        if u["m"]:
            m["m_mean"] = float(np.mean(u["m"]))
        net = _policy_net(policies)
        tm = getattr(net, "timing", None)
        if tm and tm.get("gnn_rows"):
            m["gnn_ms_per_decision"] = 1000.0 * tm["gnn_s"] / tm["gnn_rows"]
            m["gnn_ms_per_call"] = 1000.0 * tm["gnn_s"] / tm["gnn_calls"]
            tm.clear()
        roles = np.array(u["roles"])
        edges = np.linspace(0, len(roles), 11).astype(int)
        for b in range(10):
            seg = roles[edges[b]:max(edges[b + 1], edges[b] + 1)].mean(0)
            for j, name in enumerate(("explore", "rescue", "return")):
                m[f"role_{name}_bin{b}"] = float(seg[j])

    @staticmethod
    def _enc_grad(trainer, st, result):
        """공유 encoder gradient norm EMA / 불균형 경고 + 다음 iteration 측정 예약 (policy._probe_pending)."""
        mon = getattr(trainer, "_enc_monitor", None)
        if mon is None:
            mon = trainer._enc_monitor = EncGradMonitor()
        norms = {h: st.get(f"grad_norm_enc_from_{h}") for h in ENC_HEADS if f"grad_norm_enc_from_{h}" in st}
        if norms:
            ev = {ENC_EV[h]: float(st.get(f"vf_explained_var_{ENC_EV[h]}", float("nan"))) for h in ENC_HEADS}
            ent = {ENC_EV[h]: float(st.get(f"entropy_{ENC_EV[h]}", float("nan"))) for h in ENC_HEADS}
            ema, ratio, warn = mon.update({h: float(v) for h, v in norms.items()}, ev, ent)
            cm = result.setdefault("custom_metrics", {})
            cm.update({f"enc_grad_ema_{h}": v for h, v in ema.items()})
            cm.update(enc_grad_ratio=ratio, enc_grad_streak=mon.streak)
            if warn:
                print(warn)
        try:
            trainer.workers.local_worker().get_policy()._probe_pending = True
        except Exception:
            pass

    # train.py 가 trainer 를 만들기 전에 넣는다 (yaml eval: {every, episodes}). None 이면 평가 안 함
    eval_cfg = None

    def _fixed_seed_eval(self, trainer, result):
        """eval.every iteration 마다 driver 의 policy 로 고정 seed 평가 -> result custom_metrics (이번 iteration 행에 기록)."""
        ec = type(self).eval_cfg
        if not ec or int(ec.get("every") or 0) <= 0 or result["training_iteration"] % int(ec["every"]):
            return
        from evaluation.fixed_seed_eval import evaluate_greedy
        out = evaluate_greedy(trainer.get_policy(), trainer.config["env_config"], int(ec.get("episodes", 15)))
        result["custom_metrics"].update(out)
        print(f"[eval] iter {result['training_iteration']}: rescue {out['eval/rescue_rate']:.3f} ± {out['eval/rescue_rate_std']:.3f}"
              f"  expired {out['eval/expired_rate']:.3f}  len {out['eval/episode_len']:.1f}  "
              f"explored {out['eval/explored_rate']:.3f}  ({out['eval/time_s']:.1f}s, {int(ec.get('episodes', 15))} seeds, greedy)")

    def on_train_result(self, *, trainer, result, **kwargs):
        """헤드별 결정 수: stats 는 미니배치 평균이라 train batch 당 = 미니배치 값 x (train_batch / minibatch)."""
        st = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
        if "n_decisions_role" not in st:
            return
        cfg = trainer.config
        scale = cfg["train_batch_size"] / max(1, cfg["sgd_minibatch_size"])
        per_batch = {h: float(st[f"n_decisions_{h}"]) * scale for h in ("role", "rescue", "explore")}
        for k in ("explore_s1", "explore_s2", "explore_s2only"):
            if f"n_decisions_{k}" in st:
                per_batch[k] = float(st[f"n_decisions_{k}"]) * scale
        result.setdefault("custom_metrics", {}).update({f"decisions_per_batch_{h}": v for h, v in per_batch.items()})
        self._enc_grad(trainer, st, result)
        if "cur_lr" in st:
            result["custom_metrics"]["learner/cur_lr"] = float(st["cur_lr"])
        self._fixed_seed_eval(trainer, result)
        try:
            net = trainer.get_policy().model.net
            result["custom_metrics"].update(model_params=sum(p.numel() for p in trainer.get_policy().model.parameters()),
                                            gnn_params=sum(p.numel() for p in net.gnn.parameters()))
        except Exception:
            pass
        mode = (cfg.get("env_config") or {}).get("policy_mode") or {}
        if mode.get("role", "learned") == "learned" and per_batch["role"] < 200:
            print(f"[warn] role 결정이 train batch 당 {per_batch['role']:.0f} 개 (< 200) — "
                  f"training/config/env.yaml 의 train_batch_size 를 키우세요 (현재 {cfg['train_batch_size']})")
