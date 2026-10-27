"""StepContext — 한 tick 동안 일어난 사건 정보 묶음.

RoleSimulation._step_role() 이 채운다 (last_ctx). gui_evaluation/alloc_recorder.py 가 이 기록을 읽는다.
"""
from dataclasses import dataclass, field
from typing import List, Set, Optional


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

    # 할당 결과 (이번 tick 결정 직후)
    modes: List = field(default_factory=list)                      # List[Mode] (tick 끝)
    assigned_target: List[Optional[int]] = field(default_factory=list)
    free_choice: List[bool] = field(default_factory=list)          # 이번 tick role 결정을 한 UAV
    chosen_mode: List = field(default_factory=list)                # per-agent 이번 tick 역할 (Mode, 결정 직후)
    chosen_target: List = field(default_factory=list)              # per-agent 이번 tick 배정 생존자 (RESCUE 아니면 None)

    # 구조 / 만료 / 발견
    rescued_now: List[RescueEvent] = field(default_factory=list)
    expired_now: List[int] = field(default_factory=list)           # target 인덱스
    newly_discovered_targets: List[int] = field(default_factory=list)

    # depot
    refilled: Set[int] = field(default_factory=set)                # 이번 tick 재보충한 UAV
