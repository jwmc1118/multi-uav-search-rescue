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


def run_episode_detailed(sim, policy, seed, timeseries=True):
    """run_episode 와 같은 seed 규칙으로 한 에피소드. 반환 (지표 dict, 시계열 [dict] (step 마다), 초기 상태 지문).
    추가 지표: rescue_time_mean (구조 완료 시각 평균), golden_slack_mean ((마감 - 구조 시각) / 골든타임 평균, 구조된 조난자),
    rescued_per_uav, decisions (결정 수 = role + rescue + explore 1 단계), infer_ms_per_decision (모델: 결정이 있던 step 의
    추론 시간 합 / 결정 수, CPU 벽시계).
    슬롯 잘림 (학습 슬롯 수 유지 -> 큰 N / 맵에서 생길 수 있음): survivor_truncated (그룹이 아는 생존자 > s_max 인 역할 결정 수),
    blob_truncated (frontier 덩어리 > k_max 인 그룹 · step 수).
    시계열: step, cum_rescue_rate, cum_discover_rate, explored, role_explore / role_rescue / role_return (UAV 비율)."""
    import time
    from env.core.targets import deadline
    sim.random_state = np.random.RandomState(seed)
    sim.reset()
    policy.seed(seed)
    sig = init_signature(sim)
    n_t = max(1, len(sim.targets))
    conflict = esc = decisions = blob_trunc = 0
    infer_s = 0.0
    done, info, ts = False, {}, []
    while not done:
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
         "survivor_truncated": int(sim.ev_stats["survivor_truncated"]), "blob_truncated": int(blob_trunc)}
    return m, ts, sig
