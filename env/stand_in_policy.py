"""정책 훅 인터페이스 + 임시 대체 정책 (GUI 자동 재생 / 헤드리스 점검용).

RescueSimulation.step() (hook 경로)이 매 tick 아래 2개 훅을 호출한다. 순차 디코딩
(같은 n-hop 그룹에서 앞 순서가 고른 태스크 마스킹)은 환경이 mask 에 반영해서 넘긴다.
학습된 신경망 정책은 이 경로가 아니라 RescueSimulation.step_explicit() 로 들어온다
(training/checkpoint_policy.py). RandomStandInPolicy 는 연구 의사결정 로직이 아니다.
"""
import random


from env.pathfinding import next_step_towards


class AllocationHookPolicy:
    """시뮬레이션이 매 tick 호출하는 결정 훅. 서브클래스가 구현한다."""

    def select_node(self, agent, agent_index, nodes, mask, sim):
        """할당 정책: 슬롯 하나 (0..T-1 태스크, T = explore, T+1 = depot).
        nodes: env.allocation.NodeSet, mask: (T+2,) bool admissible (그룹 충돌 반영)."""
        raise NotImplementedError

    def select_primitive(self, agent, agent_index, goal, sim):
        """Primitive policy: agent.ACTIONS 중 하나. goal 은 목적지 (r, c) 또는 None."""
        raise NotImplementedError


class RandomStandInPolicy(AllocationHookPolicy):
    """임시 대체 정책 — 학습된 정책이 준비되기 전까지만 사용.
    가까운 admissible 태스크 > (용량 0 이면 depot) > explore > depot."""

    def __init__(self, seed=None, rescue_bias=0.7):
        self.rng = random.Random(seed)
        self.rescue_bias = rescue_bias

    def select_node(self, agent, agent_index, nodes, mask, sim):
        T = sim.max_targets
        E, D = T, T + 1
        if nodes.forced_slot is not None:
            return nodes.forced_slot
        tasks = [s for s in range(len(nodes.task_ids)) if mask[s]]
        if tasks and self.rng.random() < self.rescue_bias:
            return min(tasks, key=lambda s: nodes.task_dist[s])
        if agent.capacity == 0:
            return D
        if mask[E]:
            return E
        return D

    def select_primitive(self, agent, agent_index, goal, sim):
        if goal is not None:
            blocked = sim.knowledge.blocked_mask(agent_index)
            step = next_step_towards(agent.pos, goal, blocked, sim.rows, sim.cols)
            if step is not None and step != (0, 0):
                return agent.ACTIONS[agent.MOVES.index(step)]
            if step == (0, 0):
                return "stay"
        return self.rng.choice(agent.valid_actions(sim.obstacles, sim.rows, sim.cols))
