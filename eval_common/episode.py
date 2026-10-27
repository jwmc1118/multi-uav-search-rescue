"""에피소드 실행 + 지표 계산 공용 함수 — rule_evaluation / scale_evaluation / GUI 평가가 같이 쓴다.

    run_episode(sim, policy, seed)   seed 로 맵 / 조난자 / 골든타임 / 시작 위치를 고정 생성하고 정책 샘플링도 같은 seed 로 고정
                                     (정책끼리 같은 상황에서 비교). 에피소드 끝 지표 dict.
    RulePolicy                       그리디 정책을 CheckpointPolicy 와 같은 인터페이스 (step / seed / set_mode) 로
    parse_seeds('0-99' / '0,3,7')    seed 목록
"""
import numpy as np

METRICS = ["rescue_rate", "expire_rate", "discover_rate", "explored", "same_cell", "conflict_cross",
           "escalations", "episode_len"]


class RulePolicy:
    """그리디 정책을 CheckpointPolicy 와 같은 인터페이스로 (step / seed / set_mode) — 비교 평가용."""

    def __init__(self):
        from policies.role_rules import RuleDecider
        self.decider = RuleDecider()
        self.name = "rule-greedy"

    def step(self, sim):
        return sim.step(self.decider)

    def seed(self, seed):
        pass

    def set_mode(self, *a, **k):
        pass


def parse_seeds(text):
    """'0-99' / '0,3,7' / '0-4,10' -> [int]."""
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_episode(sim, policy, seed):
    sim.random_state = np.random.RandomState(seed)   # 맵 / 조난자 / 골든타임 / 시작 위치
    sim.reset()
    policy.seed(seed)                                # 정책 샘플링
    conflict = esc = 0
    done, info = False, {}
    while not done:
        done, info = policy.step(sim)                # 이동은 env 의 BFS 경로 추종
        conflict += int(info["rescue_conflict_cross"])
        esc += int(info["redecide"])
    n_t = max(1, len(sim.targets))
    return {"rescue_rate": info["rescued_total"] / n_t, "expire_rate": info["expired_total"] / n_t,
            "discover_rate": info["discovered_total"] / n_t, "explored": info["explored"],
            "same_cell": info["same_cell_total"], "conflict_cross": conflict, "escalations": esc,
            "episode_len": info["time"]}


# ---------------------------------------------------------------- 상세 실행 (main_rule / main_scale)

def init_signature(sim):
    """reset 직후 초기 상태 지문 — 맵 / depot / 조난자 (위치 · 골든타임 · 마감) / UAV (위치 · 용량 · 센서 · 통신).
    같은 seed 에서 두 정책의 초기 상태가 같은지 확인용."""
    import hashlib
    h = hashlib.md5()
    h.update(np.asarray(sim.obstacles, dtype=np.uint8).tobytes())
    h.update(repr(tuple(sim.depot)).encode())
    h.update(repr([(tuple(t.pos), t.golden_time, t.end_time) for t in sim.targets]).encode())
    h.update(repr([(tuple(a.pos), a.capacity, a.sensor_range, a.comm_range) for a in sim.agents]).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- 슬롯 잘림 (읽기 전용 — env 동작에 영향 없음)

SLOT_KEYS_S = ("role_decisions", "trunc_decisions_S", "dropped_S", "dropped_reachable_S", "dropped_urgent_S", "best_dropped_S")
SLOT_KEYS_K = ("s1_decisions", "trunc_decisions_K", "dropped_K", "dropped_K_cells")


def slot_snapshot(sim):
    """결정 직전 (step 시작 전) UAV 별 S / K 슬롯 잘림. victim_slots / group_clusters 는 직전 step 끝에 만들어지고 결정 중에는
    바뀌지 않으므로 이 시점 값이 결정 때 값과 같다. 반환 [dict] (UAV 순):
      S  n_alloc        할당 가능 후보 수 (자르기 전 = sim.n_alloc)
         dropped        잘려서 victim_slots (rescue 행동 / S 노드 앞쪽) 에 못 들어간 후보 수 = max(0, n_alloc - s_max)
         reachable      잘린 후보 중 d <= τ - t (골든타임 안 도달) 인 수 — 후보 조건 (_assignable) 이 같은 조건이라 항상 = dropped
         urgent         잘린 후보 중 slack (τ - t - d) 이 남은 후보의 최소 slack 보다 작은 (더 급한) 수
         best_dropped   전체 후보에서 EDF (rule_rescue 키: 마감 -> BFS -> id) 로 고를 생존자가 잘린 쪽에 있음
      K  n_blobs, dropped (슬롯 밖 덩어리 수), dropped_cells (그 셀 수 합)"""
    from env.core.targets import deadline
    out = []
    for i in range(sim.n_agents):
        held = sim.held_by_others(i)
        ok = [ti for ti, t in enumerate(sim.targets)
              if t.discovered and sim.knowledge.known_free[i][t.pos] and ti not in held and sim._assignable(i, ti)]
        ok.sort(key=lambda ti: (sim._dist(i, sim.targets[ti].pos), deadline(sim.targets[ti]), ti))
        assert len(ok) == sim.n_alloc[i] and ok[:sim.s_max] == list(sim.victim_slots[i]), "slot_snapshot != env victim_slots"
        kept, cut = ok[:sim.s_max], ok[sim.s_max:]
        slack = {ti: deadline(sim.targets[ti]) - sim.time - sim._dist(i, sim.targets[ti].pos) for ti in ok}
        min_kept = min((slack[ti] for ti in kept), default=None)
        edf = min(ok, key=lambda ti: (deadline(sim.targets[ti]), sim._dist(i, sim.targets[ti].pos), ti)) if ok else None
        g = sim.comp[i]
        blobs = sim.blobs[g] if g < len(getattr(sim, "blobs", [])) else []
        slot_ids = {b.id for b in sim.group_clusters[g]}
        dk = [b for b in blobs if b.id not in slot_ids]
        out.append({"n_alloc": len(ok), "dropped": len(cut),
                    "reachable": sum(1 for ti in cut if 0 <= sim._dist(i, sim.targets[ti].pos) <= deadline(sim.targets[ti]) - sim.time),
                    "urgent": sum(1 for ti in cut if min_kept is not None and slack[ti] < min_kept),
                    "best_dropped": bool(cut) and edf in cut,
                    "n_blobs": len(blobs), "k_dropped": len(dk), "k_dropped_cells": int(sum(b.size for b in dk))})
    return out


def _slot_accumulate(acc, snap, info, sim, decision_log, seed, policy_name):
    for i, sn in enumerate(snap):
        if info["role_valid"][i] > 0:
            acc["role_decisions"] += 1
            acc["trunc_decisions_S"] += int(sn["dropped"] > 0)
            acc["dropped_S"] += sn["dropped"]
            acc["dropped_reachable_S"] += sn["reachable"]
            acc["dropped_urgent_S"] += sn["urgent"]
            acc["best_dropped_S"] += int(sn["best_dropped"])
        if info["explore_valid"][i] > 0:
            acc["s1_decisions"] += 1
            acc["trunc_decisions_K"] += int(sn["k_dropped"] > 0)
            acc["dropped_K"] += sn["k_dropped"]
            acc["dropped_K_cells"] += sn["k_dropped_cells"]
        if decision_log is not None and (info["role_valid"][i] > 0 or info["explore_valid"][i] > 0):
            decision_log.append(dict(policy=policy_name, seed=seed, time=int(info["time"]) - 1, uav=i,
                                     n_targets=len(sim.targets), max_episode_len=int(sim.cfg["max_episode_len"]),
                                     role=bool(info["role_valid"][i] > 0), s1=bool(info["explore_valid"][i] > 0), **sn))


def _slot_metrics(acc):
    """에피소드 합계 + 결정 1 회당 평균 + 잘림이 있었던 결정 비율."""
    m = dict(acc)
    nr, nk = max(1, acc["role_decisions"]), max(1, acc["s1_decisions"])
    m.update({"trunc_frac_S": acc["trunc_decisions_S"] / nr if acc["role_decisions"] else float("nan"),
              "dropped_S_per_dec": acc["dropped_S"] / nr if acc["role_decisions"] else float("nan"),
              "dropped_urgent_S_per_dec": acc["dropped_urgent_S"] / nr if acc["role_decisions"] else float("nan"),
              "best_dropped_frac_S": acc["best_dropped_S"] / nr if acc["role_decisions"] else float("nan"),
              "trunc_frac_K": acc["trunc_decisions_K"] / nk if acc["s1_decisions"] else float("nan"),
              "dropped_K_per_dec": acc["dropped_K"] / nk if acc["s1_decisions"] else float("nan"),
              "dropped_K_cells_per_dec": acc["dropped_K_cells"] / nk if acc["s1_decisions"] else float("nan")})
    return m


# ---------------------------------------------------------------- 진단 (상한 · 통신 그룹 · 만료 원인)

def upper_bounds(sim):
    """reset 직후 sim 의 이론적 상한 (rule_evaluation/upper_bound.py bounds — 같은 계산): ub_loose (depot 에서 골든타임 안에
    도달 가능한 조난자 비율), ub_greedy (전지적 EDF 스케줄 구조 비율)."""
    from env.core.targets import deadline as _deadline
    from rule_evaluation.upper_bound import bounds
    inst = {"obstacles": np.array(sim.obstacles, bool), "depot": tuple(sim.depot),
            "targets": [(tuple(t.pos), int(_deadline(t)), int(t.golden_time)) for t in sim.targets],
            "capacity": int(sim.cfg["capacity"]), "n_agents": sim.n_agents, "t_max": int(sim.cfg["max_episode_len"])}
    b = bounds(inst)
    return {"ub_loose": b["ub_loose"], "ub_greedy": b["ub_greedy"]}


def _assigned_now(sim):
    from env.core.modes import Mode
    return {a.assigned_target for a in sim.agents if a.mode is Mode.RESCUE and a.assigned_target is not None}


def run_episode_detailed(sim, policy, seed, timeseries=True, decision_log=None):
    """run_episode 와 같은 seed 규칙으로 한 에피소드. 반환 (지표 dict, 시계열 [dict] (step 마다), 초기 상태 지문).
    추가 지표: rescue_time_mean (구조 완료 시각 평균), golden_slack_mean ((마감 - 구조 시각) / 골든타임 평균, 구조된 조난자),
    rescued_per_uav, decisions (결정 수 = role + rescue + explore 1 단계), infer_ms_per_decision (모델: 결정이 있던 step 의
    추론 시간 합 / 결정 수, CPU 벽시계).
    슬롯 잘림 (slot_snapshot, 결정 시점 UAV 기준): S = 역할 결정, K = explore 1 단계 결정에서 센다.
      role_decisions, trunc_decisions_S, dropped_S, dropped_reachable_S, dropped_urgent_S, best_dropped_S,
      s1_decisions, trunc_decisions_K, dropped_K, dropped_K_cells + 결정당 평균 (*_per_dec) · 잘림 결정 비율 (trunc_frac_*).
      s_nodes_overflow = env ev_stats survivor_truncated (그룹이 아는 미구조 생존자 전체 > s_max 인 역할 결정 수 — 맡겨짐 /
      도달 불가 생존자까지 포함한 S 노드 넘침, 할당 후보 잘림이 아님), blob_overflow_group_steps (덩어리 > k_max 인 그룹 · step 수).
    decision_log (list) 를 주면 결정마다 slot_snapshot 행을 덧붙인다 (시간대 / 조난자 수별 분석용).
    진단: ub_loose / ub_greedy (upper_bounds), rescue_over_ub = rescue_rate / ub_loose,
      group_size_mean (step 마다 통신 그룹 평균 크기 = N / 그룹 수, 에피소드 평균), n_groups_mean, isolated_frac (혼자인 UAV 비율),
      만료 원인 (N_T 대비): expire_undiscovered_rate (미발견 상태로 만료), expire_unassigned_rate (발견했지만 아무도 맡지 않음),
      expire_assigned_rate (RESCUE UAV 가 맡고 있던 중 만료 — 만료 step 직전 배정 기준, env 가 만료 때 배정을 해제하므로),
      unresolved_end_rate (끝날 때 구조도 만료도 안 된 조난자).
    시계열: step, cum_rescue_rate, cum_discover_rate, explored, role_explore / role_rescue / role_return (UAV 비율)."""
    import time
    from env.core.targets import deadline
    sim.random_state = np.random.RandomState(seed)
    sim.reset()
    policy.seed(seed)
    sig = init_signature(sim)
    n_t = max(1, len(sim.targets))
    conflict = esc = decisions = blob_trunc = 0
    acc = {k: 0 for k in SLOT_KEYS_S + SLOT_KEYS_K}
    pname = getattr(policy, "name", "")
    ub = upper_bounds(sim)
    exp_cause = {"undiscovered": 0, "unassigned": 0, "assigned": 0}
    grp = {"size": 0.0, "n": 0.0, "iso": 0.0, "steps": 0}
    infer_s = 0.0
    done, info, ts = False, {}, []
    while not done:
        snap = slot_snapshot(sim)
        held_before = _assigned_now(sim)
        if hasattr(policy, "act"):                               # 학습 모델: 추론 시간 측정
            t0 = time.perf_counter()
            actions = policy.act(sim)
            dt = time.perf_counter() - t0
            done, info = sim.step_explicit(actions)
        else:
            dt = None
            done, info = policy.step(sim)
        n_dec = int(sum(info["role_valid"]) + sum(info["rescue_valid"]) + sum(info["explore_valid"]))
        if dt is not None and n_dec > 0:
            infer_s += dt
            decisions += n_dec
        elif dt is None:
            decisions += n_dec
        conflict += int(info["rescue_conflict_cross"])
        esc += int(info["redecide"])
        _slot_accumulate(acc, snap, info, sim, decision_log, seed, pname)
        for ti, t in enumerate(sim.targets):
            if t.expired_at == sim.time:
                exp_cause["undiscovered" if not t.discovered else ("assigned" if ti in held_before else "unassigned")] += 1
        sizes = [len(g) for g in sim.groups]
        grp["size"] += sim.n_agents / max(1, len(sizes))
        grp["n"] += len(sizes)
        grp["iso"] += sum(1 for z in sizes if z == 1) / max(1, sim.n_agents)
        grp["steps"] += 1
        blob_trunc += sum(1 for bl in getattr(sim, "blobs", []) if len(bl) > sim.k_max)
        if timeseries:
            roles = np.bincount(np.asarray(info["role"], int), minlength=3) / max(1, len(info["role"]))
            ts.append({"step": int(info["time"]), "cum_rescue_rate": info["rescued_total"] / n_t,
                       "cum_discover_rate": info["discovered_total"] / n_t, "explored": float(info["explored"]),
                       "role_explore": float(roles[0]), "role_rescue": float(roles[1]), "role_return": float(roles[2])})
    rescued = [t for t in sim.targets if t.rescued_at is not None]
    m = {"rescue_rate": info["rescued_total"] / n_t, "expire_rate": info["expired_total"] / n_t,
         "discover_rate": info["discovered_total"] / n_t, "explored": info["explored"],
         "same_cell": info["same_cell_total"], "conflict_cross": conflict, "escalations": esc,
         "episode_len": info["time"], "n_targets": len(sim.targets), "n_agents": sim.n_agents,
         "rescue_time_mean": float(np.mean([t.rescued_at for t in rescued])) if rescued else float("nan"),
         "golden_slack_mean": float(np.mean([(deadline(t) - t.rescued_at) / max(1, t.golden_time) for t in rescued]))
         if rescued else float("nan"),
         "rescued_per_uav": info["rescued_total"] / max(1, sim.n_agents), "decisions": decisions,
         "infer_ms_per_decision": 1000.0 * infer_s / decisions if (hasattr(policy, "act") and decisions) else float("nan"),
         "s_nodes_overflow": int(sim.ev_stats["survivor_truncated"]), "blob_overflow_group_steps": int(blob_trunc),
         **_slot_metrics(acc), **ub,
         "rescue_over_ub": (info["rescued_total"] / n_t) / ub["ub_loose"] if ub["ub_loose"] > 0 else float("nan"),
         "group_size_mean": grp["size"] / max(1, grp["steps"]), "n_groups_mean": grp["n"] / max(1, grp["steps"]),
         "isolated_frac": grp["iso"] / max(1, grp["steps"]),
         "expire_undiscovered_rate": exp_cause["undiscovered"] / n_t, "expire_unassigned_rate": exp_cause["unassigned"] / n_t,
         "expire_assigned_rate": exp_cause["assigned"] / n_t,
         "unresolved_end_rate": sum(1 for t in sim.targets if t.active) / n_t}
    return m, ts, sig
