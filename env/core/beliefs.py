"""UAV 별 믿음(belief) — 통신으로만 전달되는 구조 완료 정보.

전역 진실(sim.targets[*].rescued_at)은 보상 계산(학습용, 전지적)에만 쓰고, UAV 의 결정(할당 가능 판정 / 목표 무효)은
이 믿음만 본다.

rescued[i] : UAV i 가 '구조 완료'로 알고 있는 생존자 집합
    알게 되는 경로 — (1) 자기가 구조 (2) 그 생존자 칸을 센서로 직접 봄 (3) n-hop 연결된 UAV 와 병합
만료는 통신 없이 각자 계산 가능하다고 본다 (마감 시각) -> 전역 expired 를 그대로 쓴다.
"""


class BeliefTracker:
    def __init__(self, n_agents):
        self.n = n_agents
        self.reset()

    def reset(self):
        self.rescued = [set() for _ in range(self.n)]

    # ---------------------------------------------------------------- 갱신

    def record_rescue(self, agent_index, target_index):
        self.rescued[agent_index].add(target_index)

    def update(self, sim, visible):
        """센싱/통신 직후 호출. visible[i] = 이번 tick UAV i 센서가 본 셀 집합."""
        reach = sim.knowledge.reach

        # (1) 센서로 직접 본 구조 완료 생존자
        for i in range(self.n):
            for ti, t in enumerate(sim.targets):
                if t.rescued_at is not None and t.pos in visible[i]:
                    self.rescued[i].add(ti)

        # (2) n-hop 병합 (스냅샷 기준 -> 순서 무관)
        rescued_snap = [set(s) for s in self.rescued]
        for i in range(self.n):
            for j in range(self.n):
                if i != j and reach[i, j]:
                    self.rescued[i] |= rescued_snap[j]

    # ---------------------------------------------------------------- 질의

    def is_done(self, agent_index, sim, ti):
        """UAV i 입장에서 생존자가 끝났는가 (구조된 걸 알고 있음 / 만료)."""
        return ti in self.rescued[agent_index] or sim.targets[ti].expired_at is not None
