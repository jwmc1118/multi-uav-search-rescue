"""Target(구조자) 정의.

CapAM 논문의 time-bound task에 대응한다. 각 타겟은 골든타임 tau_j를 가지며,
end_time 이후로는 구조 불가(사망 처리 = expired). end_time 인코딩 방식은
docs/research_pipeline_summary.md §9(open item 3) — 여기서는 두 모드를 옵션으로 둔다:

- "from_discovery": 어떤 에이전트든 처음 발견한 시점 + golden_time  (기본)
- "from_start":     에피소드 시작 기준 고정 (= golden_time)
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Target:
    pos: Tuple[int, int]
    golden_time: int

    discovered_at: Optional[int] = None   # 팀 내 누군가 처음 센싱한 tick
    end_time: Optional[int] = None         # 이 시각 이후 미구조 시 expired
    rescued_at: Optional[int] = None
    rescued_by: Optional[int] = None       # agent index
    expired_at: Optional[int] = None

    assigned_to: Optional[int] = None      # 현재 배정된 agent index (지속)
    claimed_by: Optional[int] = None       # 이번 allocation 라운드에서 broadcast 선점한 agent index (임시)

    @property
    def discovered(self) -> bool:
        return self.discovered_at is not None

    @property
    def active(self) -> bool:
        """아직 구조도 만료도 안 된 상태."""
        return self.rescued_at is None and self.expired_at is None

    def remaining(self, now: int) -> Optional[int]:
        """end_time까지 남은 tick. 미발견이면 None."""
        if self.end_time is None:
            return None
        return self.end_time - now

    def reset(self):
        self.discovered_at = None
        self.end_time = None
        self.rescued_at = None
        self.rescued_by = None
        self.expired_at = None
        self.assigned_to = None
        self.claimed_by = None


def deadline(t):
    """조난자의 골든타임 마감 시각 τ_j. from_discovery 모드의 미발견 조난자는 end_time 이 아직 없어 golden_time."""
    return t.end_time if t.end_time is not None else t.golden_time
