"""Target (조난자) 정의 — 골든타임 τ_j (에피소드 시작 기준), end_time = τ_j 이후로는 구조 불가 (만료 = expired)."""
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

    @property
    def discovered(self) -> bool:
        return self.discovered_at is not None

    @property
    def active(self) -> bool:
        """아직 구조도 만료도 안 된 상태."""
        return self.rescued_at is None and self.expired_at is None

    def remaining(self, now: int) -> Optional[int]:
        """end_time 까지 남은 tick (end_time 이 없으면 None) — GUI 표시용."""
        if self.end_time is None:
            return None
        return self.end_time - now


def deadline(t):
    """조난자의 골든타임 마감 시각 τ_j (end_time, reset 전에 GUI 로 추가한 조난자는 golden_time)."""
    return t.end_time if t.end_time is not None else t.golden_time
