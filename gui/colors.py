"""UAV 색 — GUI(맵 / 카드 / 경로)와 분석 그래프(gui_evaluation/alloc_plots.py)가 모두 여기 한 곳의 정의를 쓴다.

배경 의미색 (gui/style.py: frontier 초록 / 미탐색 파랑 / 생존자 노랑 / 장애물 · 회색) 과 겹치지 않게
빨강 · 보라 · 주황 · 청록 · 갈색 · 분홍 · 남색 계열, 옆 번호끼리 명도 차이가 나게 배치했다.
N 이 팔레트보다 크면 순환 (UAV 번호로 구분). pygame / matplotlib 의존 없음 (env 에서도 import 가능).
"""

AGENT_COLORS = ["#D62728",   # 빨강
                "#8E44AD",   # 보라
                "#F39C12",   # 주황
                "#16A5A5",   # 청록
                "#8B5A2B",   # 갈색
                "#E377C2",   # 분홍
                "#2C3E8F"]   # 남색


def agent_hex(i):
    """UAV i 의 색 (#RRGGBB). 팔레트 순환."""
    return AGENT_COLORS[i % len(AGENT_COLORS)]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def agent_rgb(i):
    """UAV i 의 색 (r, g, b) 0..255 — pygame 용."""
    return hex_to_rgb(agent_hex(i))


def lighten(rgb, amount=0.55):
    """흰색 쪽으로 섞은 색 (방문 칸 트레일처럼 넓게 칠하는 곳에 사용)."""
    return tuple(int(v + (255 - v) * amount) for v in rgb)
