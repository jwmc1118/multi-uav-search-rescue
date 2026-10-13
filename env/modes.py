"""UAV 모드 = 통합 할당 정책이 고른 노드의 종류 (role policy 를 대체).

할당 정책 하나가 매 tick 노드 하나를 고른다:
    task j  -> RESCUE  (구조 대상 j 로 이동, 도달 시 구조)
    explore -> EXPLORE (BFS 최근접 frontier 로 이동)
    depot   -> RETURN  (depot 으로 이동, 도달 시 구호품 재보충)
Mode 는 정책 출력이 아니라 그 선택의 결과 상태다 (GUI 표시 / 목적지 / 보상에 사용).

메시지 타입 o^{i'} (one-hot 4, 이웃 context 입력):
    0 explore / 1 return / 2 rescue (tentative) / 3 rescue (locked)
"""
from enum import Enum


class Mode(Enum):
    EXPLORE = 0
    RESCUE = 1
    RETURN = 2


MSG_EXPLORE = 0
MSG_RETURN = 1
MSG_RESCUE_TENTATIVE = 2
MSG_RESCUE_LOCKED = 3
N_MSG_TYPES = 4

N_MODES = 3          # primitive 입력 one-hot (EXPLORE, RESCUE, RETURN) — Mode.value 순서
