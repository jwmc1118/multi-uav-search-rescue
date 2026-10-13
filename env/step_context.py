"""StepContext — 한 tick 동안 일어난 사건 정보 묶음.

RescueSimulation.step_explicit() / step() 가 채워서 env/rewards.py 의 보상 함수에 넘긴다.
보상 설계에 필요한 이벤트를 여기서 꺼내 쓴다.
"""
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional


@dataclass
class RescueEvent:
    target: int          # sim.targets 인덱스
    agent: int           # 구조한 agent 인덱스
    rescued_at: int
    discovered_at: Optional[int]
    time_taken: Optional[int]   # rescued_at - discovered_at (발견 ~ 구조 소요)


@dataclass
class StepContext:
    time: int
    n_agents: int

    # 할당 결과
    chosen_slot: Dict[int, int] = field(default_factory=dict)      # agent -> 최종 적용된 슬롯
    modes: List = field(default_factory=list)                      # List[Mode] (적용 후)
    assigned_target: List[Optional[int]] = field(default_factory=list)
    free_choice: List[bool] = field(default_factory=list)          # 정책이 실제로 고른 tick (lock 아님)
    double_selected: Set[int] = field(default_factory=set)         # 같은 통신 그룹 안 동시 중복 -> 즉시 양보 (정상이면 0)
    duplicate_pursuit: List[bool] = field(default_factory=list)    # 같은 태스크를 더 가까운 UAV 도 쫓고 있음 / 이미 구조된 태스크를 쫓음 (먼 쪽만 True)
    contact_released: List[int] = field(default_factory=list)      # 통신 연결 순간 충돌 해결로 lock 이 풀린 UAV
    alloc_invalid: Set[int] = field(default_factory=set)           # admissible 밖 슬롯 (대체 적용)
    released_locks: List[int] = field(default_factory=list)        # 도달 불가/마감으로 풀린 lock

    # 구조 / 만료
    rescued_now: List[RescueEvent] = field(default_factory=list)
    expired_now: List[int] = field(default_factory=list)           # target 인덱스

    # 탐색
    newly_explored: List[int] = field(default_factory=list)        # per-agent 이번 tick 개인 지식에 새로 생긴 free 셀 수 (통신 병합 포함)
    newly_sensed: List[float] = field(default_factory=list)        # per-agent 자기 센서로 처음 밝힌 셀 수 |E^i_t| (팀이 몰랐던 셀, 동시 발견은 1/k 분배)
    chosen_mode: List = field(default_factory=list)                # per-agent 이번 tick 고른 노드 종류 (Mode, 적용 직후)
    team_explored_before: float = 0.0                              # 팀 탐색 비율 (|known free| / |C_free|)
    team_explored_after: float = 0.0
    newly_discovered_targets: List[int] = field(default_factory=list)

    # 이동 / depot
    goal: List[Optional[tuple]] = field(default_factory=list)      # per-agent 목적지 (r, c)
    goal_dist_before: List[Optional[int]] = field(default_factory=list)   # 이동 전 목적지까지 BFS 거리
    goal_dist_after: List[Optional[int]] = field(default_factory=list)    # 이동 후
    reached_goal: List[bool] = field(default_factory=list)
    moved: List[bool] = field(default_factory=list)
    blocked_move: List[bool] = field(default_factory=list)         # 막힌 방향으로 이동 시도
    collided: List[bool] = field(default_factory=list)             # 다른 UAV 와 충돌 (같은 칸 / 자리 맞바꿈, depot 제외)
    refilled: Set[int] = field(default_factory=set)
    at_depot: Set[int] = field(default_factory=set)

    # 편의
    capacities: List[int] = field(default_factory=list)
    extras: Dict = field(default_factory=dict)
