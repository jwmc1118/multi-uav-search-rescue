"""GUI 스타일 표 — 화면에 그리는 모든 요소의 색 / 도형 / 의미를 여기 한 곳에서 정의한다.

규칙
  - 화면에 나오는 문자열 (범례 설명 등) 은 영어. 글자 크기 기준 = BASE_FONT (main_gui.py GUI_FONT_SIZE)
  - gui/grid_map.py 의 그리기 코드는 이 표 (STYLE / UI) 만 참조한다 (RGB 숫자 직접 쓰기 금지 — tests/role/test_gui_style.py 가 검사).
  - 새로 그리는 요소는 STYLE 에 등록하고 draw_cell / draw_marker / draw_path 로 그린다. 그리면 Usage 에 기록되고,
    그 패널의 범례 (legend_items) 가 기록된 요소만 표의 순서대로 자동으로 보여 준다.
  - 도형마다 의미 하나:  ● UAV  ▲ 생존자  ★ 이동 목표점  ◆ 구간 대표점 (medoid)  ■ depot  점선 = BFS 경로
  - 칸 채움: 장애물 진회색 / 탐색됨 흰색 / 미탐색 진파랑 / frontier 초록 / 내 구간 = 내 UAV 색 진하게 /
    같은 덩어리 팀원 구간 = 그 팀원 색 연하게 + 팀원 번호
  - UAV 고유색 (gui/colors.py) 은 배경 의미색 (초록 · 파랑 · 노랑 · 회색) 과 겹치지 않는 계열.

Style.color = None 이면 그리는 쪽이 넘긴 UAV 색을 쓴다 (variant: strong / light).
"""
import math
from dataclasses import dataclass

import pygame

from gui.colors import lighten


@dataclass(frozen=True)
class Style:
    shape: str                   # cell / circle / triangle / star / diamond / square / dashed / line / ring / badge / overlay / square_area / square_dashed
    color: tuple = None          # 채움 (None = UAV 색)
    outline: tuple = None        # 테두리 (None = 없음)
    label: str = ""              # 범례 설명 (한국어)
    variant: str = "strong"      # UAV 색 사용 시: strong / light
    alpha: int = 255
    legend: bool = True          # 범례에 넣는가
    hatch: tuple = None          # 칸 위 사선 해칭 색 (None = 없음)
    main_label: str = ""         # 전체 맵 범례 (H) 에서 쓸 설명 (비면 label)


BLACK, WHITE = (0, 0, 0), (255, 255, 255)

# 요소 표 — 순서 = 범례 순서
STYLE = {
    # ---- 칸 채움 ----
    "obstacle":       Style("cell", (64, 64, 68), label="Obstacle"),
    "out_of_map":     Style("cell", (24, 24, 26), label="Out of map", hatch=(95, 95, 100)),
    "free":           Style("cell", (250, 250, 250), outline=(200, 200, 204), label="Free (explored)"),
    "unknown":        Style("cell", (32, 52, 118), label="Unknown"),
    "frontier":       Style("cell", (70, 175, 95), label="Frontier (not in my cluster)", main_label="Frontier"),
    "cluster":        Style("cell", None, label="Cluster (group map: one color per DBSCAN cluster)"),
    "cluster_selected": Style("edge", BLACK, legend=False),     # group map: 선택 UAV 의 덩어리 굵은 테두리
    "my_segment":     Style("cell", None, label="My segment (dark, my color)", variant="strong"),
    "mate_segment":   Style("cell", None, label="Teammate segment in my cluster (light, their color)", variant="light"),
    "seg_other":      Style("cell", None, label="UAV segment (its color + number)", variant="strong"),
    "sensor":         Style("overlay", (255, 255, 255), label="Sensor field of view", alpha=70),
    # ---- 마커 ----
    "uav":            Style("circle", None, outline=BLACK, label="UAV (thick ring = selected)"),
    "uav_selected":   Style("ring", BLACK, legend=False),
    "role_badge":     Style("badge", (20, 20, 20), label="Role badge E / R / D (explore / rescue / return)"),
    "survivor":       Style("triangle", (255, 205, 0), outline=BLACK, label="Survivor"),
    "survivor_hidden": Style("triangle", (255, 205, 0), outline=(90, 90, 90), label="Undiscovered survivor (faded)", alpha=110),
    "survivor_rescued": Style("triangle", WHITE, outline=(120, 120, 120), label="Rescued survivor (white)"),
    "survivor_expired": Style("triangle", (150, 150, 150), outline=BLACK, label="Expired survivor (gray)"),
    "survivor_mine":  Style("ring", None, label="My survivor (ring in my color)"),
    "goal":           Style("star", None, outline=BLACK, label="Goal point"),
    "medoid":         Style("diamond_outline", BLACK, label="Segment medoid"),   # 테두리만 (1 칸 구간 색을 가리지 않게)
    "depot":          Style("square", WHITE, outline=BLACK, label="Depot"),
    "path":           Style("dashed", None, label="BFS path (dashed)"),
    # ---- 범위 오버레이 (V) ----
    "sensor_range":   Style("square_area", None, label="Sensor range (shaded, UAV color)", alpha=60, main_label="Sensor range"),
    "comm_range":     Style("square_dashed", None, label="Comm range (dashed square, UAV color)", main_label="Comm range"),
    # ---- 링크 오버레이 (L) ----
    "assign_link":    Style("line", None, label="Solid line = UAV to assigned survivor"),
    "conflict":       Style("ring", BLACK, label="Thick black ring = duplicate pursuit"),
}

# legend 아이콘의 고정 견본 색 (선택 UAV 와 무관하게 legend 픽셀이 항상 같도록)
LEGEND_SAMPLE_MY = (214, 39, 40)
LEGEND_SAMPLE_MATE = (142, 68, 173)

# group map 덩어리 (DBSCAN cluster) 색 — UAV 색 · frontier 초록 · 미탐색 파랑 · 생존자 노랑 · 회색과 겹치지 않는 파스텔, 순환
CLUSTER_PALETTE = [(253, 203, 160), (214, 190, 236), (250, 170, 160), (232, 214, 170), (240, 180, 210), (190, 200, 250)]


def cluster_color(blob_id):
    return CLUSTER_PALETTE[int(blob_id) % len(CLUSTER_PALETTE)]


# 화면 틀 (범례 대상 아님): 패널 / 카드 / 버튼 / 글자
UI = {
    "bg": (255, 255, 255), "gap": (225, 225, 228), "grid_line": (214, 214, 218),
    "panel": (238, 238, 240), "info": (247, 247, 249), "obs": (240, 242, 245),
    "card": (255, 255, 255), "card_hi": (222, 234, 250), "card_hover": (236, 241, 248), "card_border": (208, 208, 212),
    "button": (70, 130, 180), "button_hover": (95, 155, 205), "button_on": (210, 120, 45), "button_text": (255, 255, 255),
    "text": (25, 25, 25), "subtext": (110, 110, 118), "accent": (40, 100, 200), "warn": (190, 60, 60), "ok": (60, 150, 70),
    "overlay_bg": (250, 250, 250), "overlay_hover": (215, 228, 245), "shade": (0, 0, 0, 120), "frame": (120, 120, 128),
    "supply_on": (86, 156, 255), "supply_off": (200, 200, 205), "badge_text": (255, 255, 255),
    "mode_explore": (86, 156, 255), "mode_rescue": (255, 150, 55), "mode_return": (160, 160, 165),
}

# inspector 패널 (선택 UAV 상세) 기준 글자 크기 — 제목 = BASE_FONT + 4, legend / 본문 = BASE_FONT.
# main_gui.py 의 GUI_FONT_SIZE 로 바꾼다 (GridMap(font_size=...) -> set_base_font)
BASE_FONT = 17
TITLE_DELTA = 4


def set_base_font(size):
    global BASE_FONT
    if size:
        BASE_FONT = int(size)


def font(size, bold=False):
    """화면 글자는 전부 영어 (한글 폰트 불필요)."""
    return pygame.font.SysFont("Times New Roman", size, bold=bold)


def fill_color(key, uav=None):
    s = STYLE[key]
    if s.color is not None:
        return s.color
    rgb = tuple(uav[:3])
    return lighten(rgb, 0.6) if s.variant == "light" else rgb


class Usage:
    """패널별로 실제 그린 요소 기록 -> 범례."""

    def __init__(self):
        self.keys = {}

    def add(self, key, uav=None):
        assert key in STYLE, f"gui/style.py 에 없는 요소: {key}"
        if key not in self.keys:
            self.keys[key] = uav

    def items(self):
        return [(k, self.keys[k]) for k in STYLE if k in self.keys and STYLE[k].legend]


def _rgba(c, a):
    return (*c[:3], a) if len(c) == 3 else c


def draw_cell(surf, key, rect, uav=None, usage=None, text=None, fnt=None):
    s = STYLE[key]
    col = fill_color(key, uav)
    if s.alpha < 255 or s.shape == "overlay":
        tile = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        tile.fill(_rgba(col, s.alpha))
        surf.blit(tile, rect.topleft)
    else:
        surf.fill(col, rect)
    if s.hatch is not None:                                       # 사선 해칭 (/ 방향, 셀 크기에 비례한 간격)
        step = max(3, rect.w // 3)
        for k in range(-rect.h + 1, rect.w, step):                 # 선 위의 점: x - d = k (d = 아래에서 위로)
            x0, x1 = max(0, k), min(rect.w - 1, k + rect.h - 1)
            pygame.draw.line(surf, s.hatch, (rect.left + x0, rect.bottom - 1 - (x0 - k)),
                             (rect.left + x1, rect.bottom - 1 - (x1 - k)), 1)
    if s.outline is not None and rect.w >= 6:
        pygame.draw.rect(surf, s.outline, rect, 1)
    if text and fnt is not None and rect.w >= 8:
        lab = fnt.render(text, True, BLACK)
        surf.blit(lab, lab.get_rect(center=rect.center))
    if usage is not None:
        usage.add(key, uav)


def star_points(cx, cy, outer_r, inner_r, n=5):
    pts, a = [], -math.pi / 2
    for i in range(n * 2):
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        a += math.pi / n
    return pts


def _shape(surf, shape, col, center, r, width=0, alpha=255):
    cx, cy = center
    if alpha < 255:
        layer = pygame.Surface((int(4 * r + 4), int(4 * r + 4)), pygame.SRCALPHA)
        _shape(layer, shape, _rgba(col, alpha), (2 * r + 2, 2 * r + 2), r, width)
        surf.blit(layer, (cx - 2 * r - 2, cy - 2 * r - 2))
        return
    if shape == "circle":
        pygame.draw.circle(surf, col, (int(cx), int(cy)), max(1, int(r)), width)
    elif shape == "triangle":
        pts = [(cx, cy - r), (cx + r * 0.95, cy + r * 0.75), (cx - r * 0.95, cy + r * 0.75)]
        pygame.draw.polygon(surf, col, pts, width)
    elif shape == "star":
        pygame.draw.polygon(surf, col, star_points(cx, cy, r, r * 0.45), width)
    elif shape == "diamond":
        pygame.draw.polygon(surf, col, [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], width)
    elif shape == "diamond_outline":
        pygame.draw.polygon(surf, col, [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], max(2, int(r // 2.5)))
    elif shape in ("square", "square_outline"):
        pygame.draw.rect(surf, col, pygame.Rect(cx - r, cy - r, 2 * r, 2 * r), width)


def draw_marker(surf, key, center, r, uav=None, usage=None, text=None, fnt=None, text_color=BLACK):
    """도형 마커. uav = UAV 색 (Style.color 가 None 인 요소). ring 은 반지름 r 의 굵은 링."""
    s = STYLE[key]
    col = fill_color(key, uav)
    if s.shape == "ring":
        w = max(2, int(r * 0.22))
        pygame.draw.circle(surf, col, (int(center[0]), int(center[1])), int(r), w)
        if key == "uav_selected":                                     # 이중 링
            pygame.draw.circle(surf, col, (int(center[0]), int(center[1])), int(r + w + 2), w)
    elif s.shape == "badge":
        pygame.draw.circle(surf, col, (int(center[0]), int(center[1])), max(5, int(r)))
    elif s.shape in ("square_outline", "diamond_outline"):
        _shape(surf, s.shape, col, center, r, 2)
    elif s.shape == "square_area":                                     # 반투명 채움 + 얇은 테두리 (센서 범위)
        rect = pygame.Rect(int(center[0] - r), int(center[1] - r), int(2 * r), int(2 * r))
        tile = pygame.Surface(rect.size, pygame.SRCALPHA)
        tile.fill(_rgba(col, s.alpha))
        surf.blit(tile, rect.topleft)
        pygame.draw.rect(surf, col, rect, 1)
    elif s.shape == "square_dashed":                                   # 긴 점선 사각형 (통신 범위) — BFS 경로와 달리 검은 테두리 없음
        x0, y0, x1, y1 = center[0] - r, center[1] - r, center[0] + r, center[1] + r
        dash = max(3, int(r / 5))
        for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            n = max(1, int(math.hypot(bx - ax, by - ay) // dash))
            for k in range(0, n, 2):                                   # 짝수 칸만 = 선 / 빈칸 같은 길이
                t0, t1 = k / n, min(1.0, (k + 1) / n)
                pygame.draw.line(surf, col, (ax + (bx - ax) * t0, ay + (by - ay) * t0),
                                 (ax + (bx - ax) * t1, ay + (by - ay) * t1), 3)
    else:
        _shape(surf, s.shape, col, center, r, 0, s.alpha)
        if s.outline is not None:
            _shape(surf, s.shape, s.outline, center, r, max(1, int(r // 6) or 1), 255 if s.alpha == 255 else s.alpha + 60)
    if text and fnt is not None:
        lab = fnt.render(text, True, text_color)
        surf.blit(lab, lab.get_rect(center=(int(center[0]), int(center[1]))))
    if usage is not None:
        usage.add(key, uav)


def draw_edges(surf, key, cells, to_rect, width=2, usage=None):
    """칸 집합의 바깥 테두리 (집합 밖 4-이웃과 맞닿은 변) 를 굵게."""
    col = fill_color(key)
    for (r, c) in cells:
        rect = to_rect((r, c))
        if (r - 1, c) not in cells:
            pygame.draw.line(surf, col, rect.topleft, (rect.right - 1, rect.top), width)
        if (r + 1, c) not in cells:
            pygame.draw.line(surf, col, (rect.left, rect.bottom - 1), (rect.right - 1, rect.bottom - 1), width)
        if (r, c - 1) not in cells:
            pygame.draw.line(surf, col, rect.topleft, (rect.left, rect.bottom - 1), width)
        if (r, c + 1) not in cells:
            pygame.draw.line(surf, col, (rect.right - 1, rect.top), (rect.right - 1, rect.bottom - 1), width)
    if usage is not None:
        usage.add(key)


def draw_path(surf, key, pts, uav, width=2, usage=None, dash=6):
    """점선 (BFS 경로) / 실선 (line). 밝은 색도 보이게 아래에 검은 얇은 테두리."""
    s = STYLE[key]
    col = fill_color(key, uav)
    if len(pts) < 2:
        return
    if s.shape == "line":
        pygame.draw.line(surf, BLACK, pts[0], pts[-1], width + 2)
        pygame.draw.line(surf, col, pts[0], pts[-1], width)
    else:
        on, acc = True, 0.0
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            t = 0.0
            while t < seg:
                step = min(dash - acc, seg - t)
                a = (x0 + (x1 - x0) * t / seg, y0 + (y1 - y0) * t / seg)
                b = (x0 + (x1 - x0) * (t + step) / seg, y0 + (y1 - y0) * (t + step) / seg)
                if on:
                    pygame.draw.line(surf, BLACK, a, b, width + 2)
                    pygame.draw.line(surf, col, a, b, width)
                t += step
                acc += step
                if acc >= dash:
                    acc, on = 0.0, not on
    if usage is not None:
        usage.add(key, uav)


def legend_icon(surf, key, rect, uav=None, fnt=None):
    """범례 아이콘 (그리는 함수와 같은 함수로 그린다)."""
    s = STYLE[key]
    c = rect.center
    r = rect.h // 2 - 1
    uav = uav or LEGEND_SAMPLE_MY
    if key == "cluster":
        draw_cell(surf, key, rect, cluster_color(1), text="C1" if rect.w >= 20 else None, fnt=fnt)
    elif key == "mate_segment":
        draw_cell(surf, key, rect, LEGEND_SAMPLE_MATE, text="1", fnt=fnt)
    elif key == "medoid":
        draw_cell(surf, "my_segment", rect, LEGEND_SAMPLE_MY)
        draw_marker(surf, key, c, max(3, int(rect.h * 0.3)))
    elif s.shape in ("cell", "overlay"):
        if s.shape == "overlay":
            surf.fill(STYLE["unknown"].color, rect)
        draw_cell(surf, key, rect, uav, text="1" if key == "seg_other" else None, fnt=fnt)
    elif s.shape in ("dashed", "line"):
        draw_path(surf, key, [(rect.left, c[1]), (rect.right, c[1])], uav, 2, dash=4)
    elif s.shape == "square_area":
        surf.fill(STYLE["free"].color, rect)
        draw_marker(surf, key, c, r, uav)
    elif s.shape == "square_dashed":
        surf.fill(STYLE["free"].color, rect)
        draw_marker(surf, key, c, r - 1, uav)
    elif key == "survivor_mine":
        draw_marker(surf, "survivor", c, r - 2)
        draw_marker(surf, key, c, r + 1, uav)
    elif key == "uav":
        draw_marker(surf, key, c, r, uav)
        draw_marker(surf, "uav_selected", c, r - 1)
    elif key == "role_badge":
        draw_marker(surf, key, c, r, text="E", fnt=fnt, text_color=WHITE)
    elif key == "depot":
        draw_marker(surf, key, c, r, text="D", fnt=fnt)
    else:
        draw_marker(surf, key, c, r, uav)
