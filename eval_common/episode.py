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
