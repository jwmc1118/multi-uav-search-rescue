"""UAV 별 믿음(belief) — 통신으로만 전달되는 할당/구조 정보.

전역 진실(sim.targets[*].assigned_to / rescued_at)은 보상 계산(학습용, 전지적)에만 쓰고,
UAV 의 결정(후보 / admissible 마스크 / lock 유지)은 이 믿음만 본다.

rescued[i] : UAV i 가 '구조 완료'로 알고 있는 태스크 집합
    알게 되는 경로 — (1) 자기가 구조 (2) 그 태스크 칸을 센서로 직접 봄 (3) n-hop 연결된 UAV 와 병합
claims[i]  : UAV i 가 알고 있는 '다른 UAV 의 lock'  {target: (claimant, 정보 시각)}
    - n-hop 연결된 UAV 의 현재 lock 은 직접 알게 된다 (가장 최신)
    - 연결된 UAV 가 들고 있던 남의 정보도 시각이 더 최신이면 전달받는다 (gossip)
    - 그 태스크가 구조/만료되었다고 알게 되면 지운다
    오래된 정보는 남아 있을 수 있다 (lock 이 풀렸는데 아직 못 만난 경우 = 보수적으로 계속 피함)
만료는 통신 없이 각자 계산 가능하다고 본다 (발견 시각 + 골든타임) -> 전역 expired 를 그대로 쓴다.
"""


class BeliefTracker:
    def __init__(self, n_agents):
        self.n = n_agents
        self.reset()

    def reset(self):
        self.rescued = [set() for _ in range(self.n)]
        self.claims = [dict() for _ in range(self.n)]

    # ---------------------------------------------------------------- 갱신

    def record_rescue(self, agent_index, target_index):
        self.rescued[agent_index].add(target_index)

    def update(self, sim, visible):
        """센싱/통신 직후 호출. visible[i] = 이번 tick UAV i 센서가 본 셀 집합."""
        now = sim.time
        reach = sim.knowledge.reach
        agents = sim.agents

        # (1) 센서로 직접 본 구조 완료 태스크
        for i in range(self.n):
            for ti, t in enumerate(sim.targets):
                if t.rescued_at is not None and t.pos in visible[i]:
                    self.rescued[i].add(ti)

        # (2) n-hop 병합 (스냅샷 기준 -> 순서 무관)
        rescued_snap = [set(s) for s in self.rescued]
        claims_snap = [dict(c) for c in self.claims]
        for i in range(self.n):
            for j in range(self.n):
                if i == j or not reach[i, j]:
                    continue
                self.rescued[i] |= rescued_snap[j]
                for ti, (c, ts) in claims_snap[j].items():          # gossip: 더 최신 정보만
                    if c == i:
                        continue
                    old = self.claims[i].get(ti)
                    if old is None or old[1] < ts:
                        self.claims[i][ti] = (c, ts)

        # (3) 연결된 UAV 의 현재 lock 은 직접 확인 (가장 최신) — 그 UAV 의 옛 claim 은 지움
        for i in range(self.n):
            for j in range(self.n):
                if i == j or not reach[i, j]:
                    continue
                for ti in [ti for ti, (c, _) in self.claims[i].items() if c == j]:
                    del self.claims[i][ti]
                a = agents[j]
                if a.locked and a.assigned_target is not None:
                    self.claims[i][a.assigned_target] = (j, now)
            # 자기 자신에 대한 claim 은 들고 있지 않는다
            for ti in [ti for ti, (c, _) in self.claims[i].items() if c == i]:
                del self.claims[i][ti]

        # (4) 구조 완료 / 만료로 알게 된 태스크의 claim 정리
        for i in range(self.n):
            for ti in list(self.claims[i]):
                t = sim.targets[ti]
                if ti in self.rescued[i] or t.expired_at is not None:
                    del self.claims[i][ti]

    # ---------------------------------------------------------------- 질의

    def is_done(self, agent_index, sim, ti):
        """UAV i 입장에서 태스크가 끝났는가 (구조된 걸 알고 있음 / 만료)."""
        return ti in self.rescued[agent_index] or sim.targets[ti].expired_at is not None

    def claimed_by_other(self, agent_index, ti):
        c = self.claims[agent_index].get(ti)
        return c is not None and c[0] != agent_index
