"""UAV 역할 (Mode). role 정책이 고른 역할이 곧 UAV 의 현재 상태다 (GUI 표시 / 목적지 / 보상 / 관측 one-hot).

    EXPLORE : explore 할당 (덩어리 -> 구간) 의 목표 셀로 이동
    RESCUE  : 배정 생존자로 이동, 도달 시 구조
    RETURN  : depot 으로 이동, 도달 시 구호품 재보충
Mode.value 가 role 행동 인덱스 / one-hot 순서다.
"""
from enum import Enum


class Mode(Enum):
    EXPLORE = 0
    RESCUE = 1
    RETURN = 2
