"""Grid map visualization + editor (pygame).

Layout (3 columns):
  [ grid ]  [ info panel: stats / robots / targets ]  [ observation panel: selected robot ]
  [ ------------------------ bottom bar: buttons / speed / golden-time ------------------- ]

env.simulation.RescueSimulation 하나를 받아서 그린다. Text is English, rendered in Times New Roman.
실행은 Exploration_rescue/ 에서  python main.py

Controls
- Space or Play : play / pause          N or Right or Step : single step
- click a robot card (or 1..9)          : select robot -> its local/global observation
  shows in the far-right panel
- P : print selected robot's obs to console        M : toggle policy (stand-in <-> model)
- while paused: left-drag = obstacles, right-click = add/remove target
- Change Map : new random obstacle map + targets   Targets : re-place targets only
- Save/Load : map file   Model : load a trained checkpoint

결정(통합 할당 + primitive)은 기본적으로 env/stand_in_policy.py 의 RandomStandInPolicy(임시 대체물)가,
체크포인트를 불러오면 training/checkpoint_policy.py 를 통해 학습된 정책이 구동한다.
관측 패널의 local 은 로봇 중심 크롭(회전 없음, 북쪽 = 위, 정책 입력),
global 은 저해상도 개별 누적 맵(표시용)이다. 로봇 카드의 배지 = 할당 결과(mode), 자물쇠 표시 = lock.
전체 맵에는 월드 좌표축(env/coords.py: x 오른쪽, y 위쪽, 원점 = 맵 중앙)을 그리고,
화면의 모든 위치 표기는 (x, y) 이다. 하단 바에 마우스가 가리키는 칸의 (x, y) 가 표시된다.
"""
import math
import os
import sys
import glob

import pygame

from env.simulation import RescueSimulation
from env.modes import Mode
from env.knowledge import CH_OBSTACLE, CH_EXPLORED, CH_FRONTIER, CH_TARGET, CH_AGENTS, AGENT_SELF

MAP_DIR = os.path.join("maps", "map")
MAP_TARGET_DIR = os.path.join("maps", "map_target")
# training/train.py 가 체크포인트를 저장하는 곳 (레포 루트/marl_models)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marl_models")

# ---- colors ----
COLOR_BG = (255, 255, 255)
COLOR_GRID_LINE = (208, 208, 208)
COLOR_OBSTACLE = (45, 45, 45)
COLOR_UNKNOWN = (40, 70, 160)
COLOR_FREE = (235, 235, 237)
COLOR_START = (220, 40, 40)
COLOR_DEPOT = (30, 165, 95)
COLOR_TARGET = (255, 200, 0)
COLOR_TARGET_URGENT = (255, 120, 0)
COLOR_TARGET_RESCUED = (120, 200, 120)
COLOR_TARGET_EXPIRED = (200, 70, 70)
COLOR_SELECTED = (255, 255, 255)
COLOR_PANEL_BG = (238, 238, 240)
COLOR_INFO_BG = (247, 247, 249)
COLOR_OBS_BG = (240, 242, 245)
COLOR_GAP_BG = (225, 225, 228)
COLOR_CARD = (255, 255, 255)
COLOR_CARD_HI = (222, 234, 250)
COLOR_CARD_HOVER = (236, 241, 248)
COLOR_CARD_BORDER = (208, 208, 212)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (95, 155, 205)
COLOR_BUTTON_ON = (210, 120, 45)
COLOR_BUTTON_TEXT = (255, 255, 255)
COLOR_TEXT = (25, 25, 25)
COLOR_SUBTEXT = (110, 110, 118)
COLOR_ASSIGN_LINE = (255, 140, 0)
COLOR_FRONTIER = (90, 190, 110)
COLOR_NEIGHBOR = (150, 150, 155)
COLOR_AXIS_X = (215, 45, 45)          # 월드 x 축
COLOR_AXIS_Y = (40, 160, 70)           # 월드 y 축
COLOR_SENSOR = (110, 185, 255, 105)   # 현재 센서 시야 (반투명 옅은 파랑)
COLOR_OVERLAY_BG = (250, 250, 250)
COLOR_OVERLAY_ITEM_HOVER = (215, 228, 245)

MODE_NAME = {Mode.EXPLORE: "EXPLORE", Mode.RESCUE: "RESCUE", Mode.RETURN: "RETURN"}
MODE_COLOR = {Mode.EXPLORE: (86, 156, 255), Mode.RESCUE: (255, 150, 55), Mode.RETURN: (46, 190, 120)}


def _font(size, bold=False):
    return pygame.font.SysFont("Times New Roman", size, bold=bold)


def star_points(cx, cy, outer_r, inner_r, n=5):
    """Vertices of a star polygon centered at (cx, cy)."""
    pts = []
    a = -math.pi / 2
    for i in range(n * 2):
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        a += math.pi / n
    return pts


class Button:
    """Bottom-bar button. toggle=True shows on/off state by color."""

    def __init__(self, rect, text, font, toggle=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.toggle = toggle
        self.on = False

    def draw(self, surface):
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        color = COLOR_BUTTON_ON if (self.toggle and self.on) else \
            (COLOR_BUTTON_HOVER if hover else COLOR_BUTTON)
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        t = self.font.render(self.text, True, COLOR_BUTTON_TEXT)
        surface.blit(t, t.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))


class GridMap:
    def __init__(self, sim: RescueSimulation, cell_size=22, info_panel_width=340,
                 obs_panel_width=330, panel_height=104, extra_height=180, fps=60,
                 model_policy=None):
        self.sim = sim
        self.rows, self.cols = sim.rows, sim.cols
        self.cs = cell_size
        self.panel_height = panel_height
        self.fps = fps
        self.model_policy = model_policy
        sim.keep_map = True      # GUI 의 Reset 은 같은 맵/조난자를 다시 재생 (새 맵은 Change Map)
        self.use_model = model_policy is not None

        self.grid_width = self.cols * self.cs
        self.grid_height = self.rows * self.cs
        self.grid_area_w = max(self.grid_width, 560)
        self.info_x = self.grid_area_w
        self.info_w = info_panel_width
        self.obs_x = self.info_x + info_panel_width
        self.obs_w = obs_panel_width
        self.screen_width = self.obs_x + obs_panel_width
        # 사이드 패널(info/obs) 높이 = 그리드 높이 + extra_height. 하단 바는 그 아래에 붙는다.
        self.content_h = self.grid_height + extra_height
        self.screen_height = self.content_h + panel_height

        self.selected = None
        self.robot_card_rects = []           # [(pygame.Rect, robot_index), ...]
        self.playing = False
        self._frame = 0
        self.speeds = [16, 10, 6, 3, 1]      # auto-step interval in frames; smaller = faster
        self.speed_idx = 2

        self.dragging = False
        self.paint_value = None
        self.last_cell = None

        gt = sim.cfg.get("target_golden_time", [120, 220])
        if isinstance(gt, (int, float)):
            gt = [int(gt), int(gt)]
        self.gt_min, self.gt_max = int(gt[0]), int(gt[1])

        self.status = ""
        self.status_timer = 0
        self.overlay = None                  # None | "load" | "model"
        self.overlay_items = []
        self.overlay_rects = []

        pygame.init()
        pygame.display.set_caption("Rescue Simulation")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.f_title = _font(28, bold=True)
        self.f_head = _font(21, bold=True)
        self.f = _font(19)
        self.f_sm = _font(16)
        self.f_xs = _font(14)

        self._build_buttons()
        self.running = True
        os.makedirs(MAP_DIR, exist_ok=True)

    # ---------- bottom-bar buttons ----------

    def _build_buttons(self):
        self.buttons = {}
        bh = 32
        y1 = self.content_h + 9
        x = 12
        for label, w in [("Save", 62), ("Load", 62), ("Change Map", 112), ("Targets", 84), ("Reset", 66),
                         ("Play", 74), ("Step", 62), ("Model", 70)]:
            self.buttons[label] = Button((x, y1, w, bh), label, self.f_sm,
                                         toggle=(label == "Play"))
            x += w + 7
        x += 6
        self.buttons["spd-"] = Button((x, y1, 28, bh), "-", self.f_sm)
        self._spd_label_x = x + 34
        self.buttons["spd+"] = Button((x + 74, y1, 28, bh), "+", self.f_sm)
        self._status_x = x + 118

        y2 = self.content_h + 52
        self._gt_y = y2
        # golden-time steppers laid out as  min [-] value [+]   max [-] value [+]
        self._gt_min_val_x = 186
        self._gt_max_val_x = 372
        self.buttons["gtmin-"] = Button((150, y2, 26, 26), "-", self.f_sm)
        self.buttons["gtmin+"] = Button((234, y2, 26, 26), "+", self.f_sm)
        self.buttons["gtmax-"] = Button((336, y2, 26, 26), "-", self.f_sm)
        self.buttons["gtmax+"] = Button((420, y2, 26, 26), "+", self.f_sm)
        self._gt_max_label_x = 300

    def _sync_golden_time(self):
        self.gt_min = max(1, self.gt_min)
        self.gt_max = max(self.gt_min, self.gt_max)
        self.sim.cfg["target_golden_time"] = [self.gt_min, self.gt_max]

    def _set_status(self, message, frames=200):
        self.status, self.status_timer = message, frames

    # ---------- coordinates ----------

    def cell_at(self, pos):
        x, y = pos
        if x >= self.grid_width or y >= self.grid_height:
            return None
        col, row = int(x // self.cs), int(y // self.cs)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def cell_rect(self, row, col):
        return pygame.Rect(col * self.cs, row * self.cs, self.cs, self.cs)

    def cell_center(self, row, col):
        return (col * self.cs + self.cs / 2, row * self.cs + self.cs / 2)

    # ---------- simulation control ----------

    def toggle_play(self):
        self.playing = not self.playing
        self.buttons["Play"].on = self.playing
        self.buttons["Play"].text = "Pause" if self.playing else "Play"

    def step_once(self):
        if self.use_model and self.model_policy is not None:
            try:
                allocs, prims = self.model_policy.act(self.sim)
                done, info = self.sim.step_explicit(allocs, prims)
            except Exception as e:
                self._set_status(f"model inference failed -> stand-in: {e}")
                self.use_model = False
                return
        else:
            done, info = self.sim.step()
        if done:
            self._set_status(f"episode done @t{info['time']}  "
                             f"rescued {info['rescued_total']} / expired {info['expired_total']}")
            self.playing = False
            self.buttons["Play"].on = False
            self.buttons["Play"].text = "Play"

    def change_map(self):
        """새 랜덤 장애물 맵 + 조난자 (env/map_gen.py). 장애물 비율 / 조난자 수 범위는 sim.cfg 를 따른다."""
        try:
            self.sim.change_map()
        except RuntimeError as e:
            self._set_status(f"map generation failed: {e}")
            return
        self.playing = False
        self.buttons["Play"].on = False
        self.buttons["Play"].text = "Play"
        ratio = sum(v for row in self.sim.obstacles for v in row) / (self.sim.rows * self.sim.cols)
        self._set_status(f"new map: obstacles {ratio:.0%}, targets {len(self.sim.targets)}")

    def reset_sim(self):
        self.sim.reset()
        self.playing = False
        self.buttons["Play"].on = False
        self.buttons["Play"].text = "Play"
        self._set_status("reset")

    # ---------- save / load / model ----------

    def save_map(self):
        path = self.sim.save_map_file(MAP_DIR, MAP_TARGET_DIR)
        self._set_status(f"saved: {path}")

    def open_overlay(self, kind):
        self.overlay = kind
        self.overlay_rects = []
        if kind == "load":
            items = []
            for d in (MAP_DIR, MAP_TARGET_DIR):
                if os.path.isdir(d):
                    for fn in sorted(os.listdir(d)):
                        if fn.endswith(".json"):
                            items.append((f"{d}/{fn}", os.path.join(d, fn)))
            self.overlay_items = items
        else:
            root = MODELS_DIR
            ckpts = sorted(glob.glob(os.path.join(root, "**", "checkpoint-*"),
                                     recursive=True))
            ckpts = [c for c in ckpts if not c.endswith(".tune_metadata")]
            self.overlay_items = [(os.path.relpath(c, root), c) for c in ckpts][-40:]

    def _pick_overlay(self, pos):
        for rect, path in self.overlay_rects:
            if rect.collidepoint(pos):
                if self.overlay == "load":
                    try:
                        ok = self.sim.load_map_file(path)
                        self._set_status(f"loaded: {path}" if ok else "grid size mismatch")
                    except ValueError as e:          # 예: main 의 depot 이 이 맵에선 장애물 칸
                        self._set_status(f"load failed: {e}")
                    self.reset_sim()
                else:
                    try:
                        from training.checkpoint_policy import CheckpointPolicy
                        self.model_policy = CheckpointPolicy(
                            path, env_config={"map_path": self.sim.cfg.get("map_path")})
                        self.use_model = True
                        self._set_status(f"model loaded: {os.path.basename(path)}")
                    except Exception as e:
                        self._set_status(f"model load failed: {e}")
                self.overlay = None
                return
        self.overlay = None

    # ==================== drawing: grid ====================

    def draw_grid(self):
        surf = self.screen
        surf.fill(COLOR_UNKNOWN, (0, 0, self.grid_width, self.grid_height))
        if self.grid_area_w > self.grid_width:
            surf.fill(COLOR_GAP_BG, (self.grid_width, 0,
                                     self.grid_area_w - self.grid_width, self.grid_height))
        if self.content_h > self.grid_height:
            surf.fill(COLOR_GAP_BG, (0, self.grid_height,
                                     self.grid_area_w, self.content_h - self.grid_height))

        known_free = self.sim.knowledge.team_known_free()
        for r in range(self.rows):
            row = known_free[r]
            for c in range(self.cols):
                if row[c]:
                    pygame.draw.rect(surf, COLOR_BG, self.cell_rect(r, c))

        for a in self.sim.agents:
            for (r, c) in a.visited:
                pygame.draw.rect(surf, a.color, self.cell_rect(r, c))

        dr, dc = self.sim.depot                      # depot = 모든 로봇의 출발 지점
        drect = self.cell_rect(dr, dc)
        pygame.draw.rect(surf, COLOR_DEPOT, drect)
        dlab = self.f_xs.render("D", True, (255, 255, 255))
        surf.blit(dlab, dlab.get_rect(center=drect.center))

        for a in self.sim.agents:
            if not a.comm_range:
                continue
            cx, cy = self.cell_center(*a.pos)
            half = (a.comm_range + 0.5) * self.cs
            pygame.draw.rect(surf, a.color,
                             pygame.Rect(cx - half, cy - half, half * 2, half * 2), width=3)

        for a in self.sim.agents:
            if a.mode is Mode.RESCUE and a.assigned_target is not None:
                t = self.sim.targets[a.assigned_target]
                pygame.draw.line(surf, COLOR_ASSIGN_LINE,
                                 self.cell_center(*a.pos), self.cell_center(*t.pos), 3)

        for r in range(self.rows):
            for c in range(self.cols):
                if self.sim.obstacles[r][c]:
                    pygame.draw.rect(surf, COLOR_OBSTACLE, self.cell_rect(r, c))

        self._draw_sensor_fov()
        self._draw_targets()
        self._draw_grid_lines()
        self._draw_world_axes()
        self._draw_agents()

    def _draw_world_axes(self, tick=10):
        """월드 좌표축: 원점 = 맵 중앙 셀, x 오른쪽(+), y 위쪽(+). tick 칸마다 눈금 숫자."""
        surf, frame = self.screen, self.sim.frame
        ox, oy = self.cell_center(frame.row0, frame.col0)
        w, h = self.grid_width, self.grid_height
        head = max(6, self.cs * 0.45)
        # 축선 + 양의 방향 화살촉
        pygame.draw.line(surf, COLOR_AXIS_X, (0, oy), (w - head, oy), 2)
        pygame.draw.polygon(surf, COLOR_AXIS_X, [(w, oy), (w - head * 1.6, oy - head * 0.7),
                                                 (w - head * 1.6, oy + head * 0.7)])
        pygame.draw.line(surf, COLOR_AXIS_Y, (ox, h), (ox, head), 2)
        pygame.draw.polygon(surf, COLOR_AXIS_Y, [(ox, 0), (ox - head * 0.7, head * 1.6),
                                                 (ox + head * 0.7, head * 1.6)])
        self._axis_label("x", (w - head * 1.2, oy + head + 6), COLOR_AXIS_X)
        self._axis_label("y", (ox + head + 8, head * 1.2), COLOR_AXIS_Y)
        # 눈금
        x0, x1, y0, y1 = frame.extent()
        for xv in range(-(-x0 // tick) * tick, x1 + 1, tick):
            if xv == 0:
                continue
            px = ox + xv * self.cs
            pygame.draw.line(surf, COLOR_AXIS_X, (px, oy - 4), (px, oy + 4), 2)
            self._axis_label(str(xv), (px, oy + 12), COLOR_AXIS_X)
        for yv in range(-(-y0 // tick) * tick, y1 + 1, tick):
            if yv == 0:
                continue
            py = oy - yv * self.cs
            pygame.draw.line(surf, COLOR_AXIS_Y, (ox - 4, py), (ox + 4, py), 2)
            self._axis_label(str(yv), (ox - 16, py), COLOR_AXIS_Y)
        pygame.draw.circle(surf, (255, 255, 255), (int(ox), int(oy)), 4)
        pygame.draw.circle(surf, (20, 20, 20), (int(ox), int(oy)), 4, width=1)
        self._axis_label("O", (ox - 12, oy + 12), (20, 20, 20))

    def _axis_label(self, text, center, color):
        """어두운(unknown) 배경에서도 보이도록 흰 바탕 칩 위에 글자."""
        lab = self.f_xs.render(text, True, color)
        rect = lab.get_rect(center=(int(center[0]), int(center[1])))
        pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(4, 0), border_radius=3)
        self.screen.blit(lab, rect)

    def _draw_sensor_fov(self):
        """각 로봇이 이번 tick 실제로 보고 있는 셀(Robot.sense, 360도 + 장애물 차폐)을
        반투명 옅은 파랑으로 칠한다."""
        cells = set()
        for a in self.sim.agents:
            if a.alive:
                cells |= a.sense(self.sim.obstacles, self.rows, self.cols)
        tile = pygame.Surface((self.cs, self.cs), pygame.SRCALPHA)
        tile.fill(COLOR_SENSOR)
        for (r, c) in cells:
            self.screen.blit(tile, (c * self.cs, r * self.cs))

    def _draw_targets(self):
        surf = self.screen
        now = self.sim.time
        for ti, t in enumerate(self.sim.targets):
            cx, cy = self.cell_center(*t.pos)
            outer = self.cs * 0.46
            if t.rescued_at is not None:
                color = COLOR_TARGET_RESCUED
            elif t.expired_at is not None:
                color = COLOR_TARGET_EXPIRED
            else:
                rem = t.remaining(now)
                color = COLOR_TARGET_URGENT if (rem is not None and rem <= 25) else COLOR_TARGET
            pygame.draw.polygon(surf, color, star_points(cx, cy, outer, outer * 0.45))
            lab = self.f_xs.render(str(ti), True, (20, 20, 20))
            surf.blit(lab, lab.get_rect(center=(cx, cy)))

    def _draw_agents(self):
        surf = self.screen
        for i, a in enumerate(self.sim.agents):
            rect = self.cell_rect(*a.pos)
            outline = COLOR_SELECTED if i == self.selected else COLOR_TEXT
            rad = int(self.cs * 0.46)
            pygame.draw.circle(surf, a.color, rect.center, rad)
            pygame.draw.circle(surf, outline, rect.center, rad, width=3)
            lab = self.f_xs.render(f"R{i}", True, (15, 15, 15))
            surf.blit(lab, lab.get_rect(center=rect.center))

    def _draw_grid_lines(self):
        surf = self.screen
        for c in range(self.cols + 1):
            x = c * self.cs
            pygame.draw.line(surf, COLOR_GRID_LINE, (x, 0), (x, self.grid_height))
        for r in range(self.rows + 1):
            y = r * self.cs
            pygame.draw.line(surf, COLOR_GRID_LINE, (0, y), (self.grid_width, y))

    # ==================== drawing: bottom bar ====================

    def draw_bottom_panel(self):
        surf = self.screen
        pygame.draw.rect(surf, COLOR_PANEL_BG,
                         (0, self.content_h, self.screen_width, self.panel_height))
        for b in self.buttons.values():
            b.draw(surf)

        surf.blit(self.f_sm.render(f"x{self.speeds[self.speed_idx]}", True, COLOR_SUBTEXT),
                  (self._spd_label_x, self.content_h + 16))

        y = self._gt_y + 3
        surf.blit(self.f_sm.render("golden time   min", True, COLOR_SUBTEXT), (12, y))
        surf.blit(self.f_head.render(str(self.gt_min), True, COLOR_TEXT), (self._gt_min_val_x, y - 4))
        surf.blit(self.f_sm.render("max", True, COLOR_SUBTEXT), (self._gt_max_label_x, y))
        surf.blit(self.f_head.render(str(self.gt_max), True, COLOR_TEXT), (self._gt_max_val_x, y - 4))

        cell = self.cell_at(pygame.mouse.get_pos())
        if cell is not None:
            surf.blit(self.f_sm.render(f"cursor (x, y) = {self.sim.frame.to_xy(cell)}", True, COLOR_TEXT),
                      (self.screen_width - 420, y))

        mode = "model" if self.use_model else "stand-in"
        surf.blit(self.f_sm.render(f"policy: {mode}", True,
                                   (40, 120, 200) if self.use_model else COLOR_SUBTEXT),
                  (self.screen_width - 200, y))

        if self.status_timer > 0:
            surf.blit(self.f_sm.render(self.status, True, COLOR_TEXT),
                      (self._status_x, self.content_h + 16))
            self.status_timer -= 1

    # ==================== drawing: info panel ====================

    def draw_info_panel(self):
        surf = self.screen
        x0 = self.info_x
        pygame.draw.rect(surf, COLOR_INFO_BG, (x0, 0, self.info_w, self.content_h))
        pad = 16
        x = x0 + pad
        w = self.info_w - 2 * pad
        y = pad

        info = self.sim._info([], [])
        surf.blit(self.f_title.render(f"t = {self.sim.time}", True, COLOR_TEXT), (x, y))
        y += 38
        surf.blit(self.f_sm.render(
            f"rescued {info['rescued_total']}    expired {info['expired_total']}",
            True, COLOR_SUBTEXT), (x, y))
        y += 22
        surf.blit(self.f_sm.render(
            f"active {info['active']}    found {info['discovered']}",
            True, COLOR_SUBTEXT), (x, y))
        y += 30

        y = self._draw_robots(x, y, w)
        y += 12
        self._draw_target_list(x, y, w)

    def _draw_robots(self, x, y, w):
        surf = self.screen
        surf.blit(self.f_head.render("ROBOTS", True, COLOR_SUBTEXT), (x, y))
        surf.blit(self.f_xs.render("click to inspect", True, COLOR_SUBTEXT),
                  (x + self.f_head.size("ROBOTS")[0] + 12, y + 7))
        y += 32
        self.robot_card_rects = []
        mouse = pygame.mouse.get_pos()
        for i, a in enumerate(self.sim.agents):
            h = 78
            card = pygame.Rect(x, y, w, h)
            self.robot_card_rects.append((card.copy(), i))
            if i == self.selected:
                bg = COLOR_CARD_HI
            elif card.collidepoint(mouse):
                bg = COLOR_CARD_HOVER
            else:
                bg = COLOR_CARD
            pygame.draw.rect(surf, bg, card, border_radius=7)
            pygame.draw.rect(surf, COLOR_CARD_BORDER, card, width=1, border_radius=7)

            pygame.draw.circle(surf, a.color, (x + 18, y + 21), 9)
            surf.blit(self.f.render(f"R{i}", True, COLOR_TEXT), (x + 34, y + 10))

            name = MODE_NAME[a.mode] + (" (lock)" if a.locked else "")
            bw = self.f_xs.size(name)[0] + 16
            badge = pygame.Rect(x + 74, y + 12, bw, 20)
            pygame.draw.rect(surf, MODE_COLOR[a.mode], badge, border_radius=10)
            surf.blit(self.f_xs.render(name, True, (20, 20, 20)), (badge.x + 8, badge.y + 3))

            surf.blit(self.f_sm.render(f"supply {a.capacity}/{a.capacity_max}",
                                       True, COLOR_SUBTEXT), (x + 34, y + 36))
            px = x + 34 + self.f_sm.size("supply 0/0")[0] + 14
            for k in range(a.capacity_max):
                col = (86, 156, 255) if k < a.capacity else (200, 200, 205)
                pygame.draw.rect(surf, col, pygame.Rect(px + k * 15, y + 38, 11, 12),
                                 border_radius=2)

            if a.assigned_target is not None:
                t = self.sim.targets[a.assigned_target]
                txt = f"-> T{a.assigned_target} {self.sim.frame.to_xy(t.pos)}"
            elif a.mode is Mode.RESCUE:
                txt = "-> unassigned"
            elif a.mode is Mode.RETURN:
                txt = f"-> depot {self.sim.frame.to_xy(self.sim.depot)}"
            else:
                txt = "-> frontier" if a.goal else "-> --"
            surf.blit(self.f_sm.render(txt, True, COLOR_SUBTEXT), (x + 34, y + 56))
            y += h + 8
        return y

    def _draw_target_list(self, x, y, w):
        surf = self.screen
        surf.blit(self.f_head.render("TARGETS", True, COLOR_SUBTEXT), (x, y))
        surf.blit(self.f_xs.render("golden time / deadline", True, COLOR_SUBTEXT),
                  (x + self.f_head.size("TARGETS")[0] + 12, y + 7))
        y += 32
        now = self.sim.time
        lh = 30
        for ti, t in enumerate(self.sim.targets):
            if y > self.content_h - lh:
                surf.blit(self.f_sm.render("...", True, COLOR_SUBTEXT), (x, y))
                break
            row = pygame.Rect(x, y, w, lh - 4)
            pygame.draw.rect(surf, COLOR_CARD, row, border_radius=5)
            pygame.draw.rect(surf, COLOR_CARD_BORDER, row, width=1, border_radius=5)

            if t.rescued_at is not None:
                tail, tc = f"RESCUED @t{t.rescued_at} R{t.rescued_by}", (60, 150, 70)
            elif t.expired_at is not None:
                tail, tc = f"EXPIRED @t{t.expired_at}", (190, 60, 60)
            elif t.end_time is not None:
                rem = t.remaining(now)
                tail = f"end t{t.end_time} · left {rem}" + ("" if t.discovered else " · hidden")
                tc = COLOR_TARGET_URGENT if rem <= 25 else COLOR_TEXT
            else:
                # from_discovery: 발견 전에는 시계가 안 돌지만, 골든타임 값 자체는 생성 때 정해져 있다
                tail, tc = f"golden {t.golden_time} · not found", COLOR_SUBTEXT
            surf.blit(self.f_sm.render(f"T{ti}", True, COLOR_TEXT), (x + 8, y + 3))
            surf.blit(self.f_xs.render(str(self.sim.frame.to_xy(t.pos)), True, COLOR_SUBTEXT),
                      (x + 42, y + 6))
            surf.blit(self.f_sm.render(tail, True, tc), (x + 108, y + 3))
            y += lh

    # ==================== drawing: observation panel ====================

    def _render_obs(self, arr, cell_px):
        """Render an observation tensor (H, W, C) as a small image using the map's colors.
        channels: env/knowledge.py (obstacle / explored / frontier / target / agents)."""
        h, w = arr.shape[:2]
        img = pygame.Surface((w * cell_px, h * cell_px))
        img.fill(COLOR_UNKNOWN)
        for r in range(h):
            for c in range(w):
                ag = arr[r, c, CH_AGENTS]
                if ag >= AGENT_SELF:
                    col = COLOR_START
                elif ag > 0:
                    col = COLOR_NEIGHBOR
                elif arr[r, c, CH_TARGET] > 0.5:
                    col = COLOR_TARGET
                elif arr[r, c, CH_OBSTACLE] > 0.5:
                    col = COLOR_OBSTACLE
                elif arr[r, c, CH_FRONTIER] > 0.5:
                    col = COLOR_FRONTIER
                elif arr[r, c, CH_EXPLORED] > 0.5:   # global 은 블록 탐색 비율 (avg pool)
                    col = COLOR_BG
                else:
                    continue
                img.fill(col, (c * cell_px, r * cell_px, cell_px, cell_px))
        return img

    def draw_obs_panel(self):
        surf = self.screen
        x0 = self.obs_x
        pygame.draw.rect(surf, COLOR_OBS_BG, (x0, 0, self.obs_w, self.content_h))
        pygame.draw.line(surf, COLOR_CARD_BORDER, (x0, 0), (x0, self.content_h))
        pad = 16
        x = x0 + pad
        y = self._draw_observation(x, pad, self.obs_w - 2 * pad)
        self._draw_groups(x, y + 24, self.obs_w - 2 * pad)

    def comm_groups(self):
        """raw 통신 인접 행렬의 연결 요소 = 멀티홉으로 이어진 그룹.
        반환: [(robot indices, 그룹 내 최대 hop 수), ...]  (가장 작은 로봇 번호 순)"""
        adj = self.sim.knowledge.compute_adjacency()
        n = len(adj)
        seen = [False] * n
        groups = []
        for s in range(n):
            if seen[s]:
                continue
            seen[s] = True
            comp, stack = [s], [s]
            while stack:
                i = stack.pop()
                for j in range(n):
                    if adj[i, j] and not seen[j]:
                        seen[j] = True
                        comp.append(j)
                        stack.append(j)
            comp.sort()
            # 그룹 지름(최대 최단 hop 수): 각 멤버에서 BFS
            diameter = 0
            for src in comp:
                dist = {src: 0}
                frontier = [src]
                while frontier:
                    nxt = []
                    for i in frontier:
                        for j in comp:
                            if adj[i, j] and j not in dist:
                                dist[j] = dist[i] + 1
                                nxt.append(j)
                    frontier = nxt
                diameter = max(diameter, max(dist.values()))
            groups.append((comp, diameter))
        return groups

    def _draw_groups(self, x, y, w):
        surf = self.screen
        surf.blit(self.f_head.render("GROUPS", True, COLOR_SUBTEXT), (x, y))
        n_hop = self.sim.knowledge.n_hop
        surf.blit(self.f_xs.render(f"multi-hop connected  (n_hop={n_hop})", True, COLOR_SUBTEXT),
                  (x + self.f_head.size("GROUPS")[0] + 12, y + 7))
        y += 32
        lh = 30
        for gi, (members, diameter) in enumerate(self.comm_groups()):
            if y > self.content_h - lh:
                surf.blit(self.f_sm.render("...", True, COLOR_SUBTEXT), (x, y))
                break
            row = pygame.Rect(x, y, w, lh - 4)
            bg = COLOR_CARD_HI if self.selected in members else COLOR_CARD
            pygame.draw.rect(surf, bg, row, border_radius=5)
            pygame.draw.rect(surf, COLOR_CARD_BORDER, row, width=1, border_radius=5)
            surf.blit(self.f_sm.render(f"Group {gi + 1}", True, COLOR_TEXT), (x + 8, y + 3))
            mx = x + 82
            for i in members:
                a = self.sim.agents[i]
                pygame.draw.circle(surf, a.color, (mx + 6, y + 13), 6)
                lab = self.f_sm.render(f"R{i}", True, COLOR_TEXT)
                surf.blit(lab, (mx + 15, y + 3))
                mx += 15 + lab.get_width() + 10
            # 그룹 내 최대 hop 수. n_hop 보다 크면 끝단끼리는 직접 정보가 닿지 않음 -> 강조
            hop_txt = f"{diameter} hop"
            hc = (190, 60, 60) if diameter > n_hop else COLOR_SUBTEXT
            ht = self.f_xs.render(hop_txt, True, hc)
            surf.blit(ht, (row.right - ht.get_width() - 8, y + 6))
            y += lh

    def _draw_observation(self, x, y, avail_w):
        """선택된 로봇의 local/global 관측을 그리고 다음 y 좌표를 반환."""
        surf = self.screen
        if self.selected is None:
            surf.blit(self.f_head.render("OBSERVATION", True, COLOR_SUBTEXT), (x, y))
            surf.blit(self.f_sm.render("click a robot card", True, COLOR_SUBTEXT), (x, y + 34))
            return y + 56
        a = self.sim.agents[self.selected]
        surf.blit(self.f_head.render(f"OBSERVATION  R{self.selected}", True, COLOR_SUBTEXT), (x, y))
        y += 34
        if a.local_window is None or a.global_window is None:
            surf.blit(self.f_sm.render("no data (step once)", True, COLOR_SUBTEXT), (x, y))
            return y + 22

        lw, gw = a.local_window, a.global_window
        # 두 이미지 + 라벨 + 범례가 세로로 다 들어가도록 셀 크기를 맞춘다
        budget = self.grid_height - y - 80        # 범례/라벨용 여유
        lcell = max(2, min(avail_w // lw.shape[1], 15, int(budget * 0.62) // lw.shape[0]))
        gcell = max(2, min(avail_w // gw.shape[1], 18, int(budget * 0.38) // gw.shape[0]))

        surf.blit(self.f_sm.render(f"local  {lw.shape[0]} x {lw.shape[1]}  (robot-centered)",
                                   True, COLOR_SUBTEXT), (x, y))
        y += 22
        limg = self._render_obs(lw, lcell)
        surf.blit(limg, (x, y))
        pygame.draw.rect(surf, (120, 120, 128), (x, y, *limg.get_size()), width=1)
        y += limg.get_height() + 14

        kt = self.sim.knowledge
        mates = [j for j in range(len(self.sim.agents)) if kt.reach[self.selected, j]]
        merged = "  + " + ", ".join(f"R{j}" for j in mates) if mates else "  (alone)"
        surf.blit(self.f_sm.render(f"global  {gw.shape[0]} x {gw.shape[1]}{merged}",
                                   True, COLOR_SUBTEXT), (x, y))
        y += 22
        gimg = self._render_obs(gw, gcell)
        # 지식을 합친(연결된) 팀원 위치를 각자 색으로 표시
        ds = kt.global_downsample
        for j in mates:
            m = self.sim.agents[j]
            center = ((m.pos[1] // ds) * gcell + gcell // 2, (m.pos[0] // ds) * gcell + gcell // 2)
            rad = max(3, gcell // 2 + 1)
            pygame.draw.circle(gimg, m.color, center, rad)
            pygame.draw.circle(gimg, COLOR_TEXT, center, rad, width=1)
        surf.blit(gimg, (x, y))
        pygame.draw.rect(surf, (120, 120, 128), (x, y, *gimg.get_size()), width=1)
        y += gimg.get_height() + 14

        # 범례 (2줄)
        legend = [("obstacle", COLOR_OBSTACLE), ("free", (210, 210, 212)),
                  ("unknown", COLOR_UNKNOWN), ("frontier", COLOR_FRONTIER),
                  ("target", COLOR_TARGET), ("self", COLOR_START), ("neighbor", COLOR_NEIGHBOR)]
        lx = x
        for k, (name, col) in enumerate(legend):
            if k == 4:
                lx = x
                y += 20
            pygame.draw.rect(surf, col, (lx, y + 2, 12, 12))
            surf.blit(self.f_xs.render(name, True, COLOR_SUBTEXT), (lx + 17, y))
            lx += 17 + self.f_xs.size(name)[0] + 14
        return y + 20

    # ==================== overlay ====================

    def draw_overlay(self):
        surf = self.screen
        o = pygame.Rect(self.grid_width // 6, self.screen_height // 8,
                        max(self.grid_area_w, 640) * 2 // 3, self.screen_height * 3 // 4)
        shade = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 120))
        surf.blit(shade, (0, 0))
        pygame.draw.rect(surf, COLOR_OVERLAY_BG, o, border_radius=8)
        pygame.draw.rect(surf, COLOR_TEXT, o, width=2, border_radius=8)
        title = "Load map  (Esc)" if self.overlay == "load" \
            else "Load checkpoint from marl_models/  (Esc)"
        surf.blit(self.f.render(title, True, COLOR_TEXT), (o.x + 16, o.y + 14))

        self.overlay_rects = []
        iy = o.y + 48
        if not self.overlay_items:
            surf.blit(self.f_sm.render("(none)", True, COLOR_SUBTEXT), (o.x + 16, iy))
            return
        for name, path in self.overlay_items:
            ir = pygame.Rect(o.x + 8, iy, o.w - 16, 28)
            if ir.bottom > o.bottom - 8:
                break
            if ir.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(surf, COLOR_OVERLAY_ITEM_HOVER, ir, border_radius=4)
            surf.blit(self.f_sm.render(name[-90:], True, COLOR_TEXT), (ir.x + 8, ir.y + 5))
            self.overlay_rects.append((ir, path))
            iy += 29

    # ==================== events ====================

    def handle_event(self, ev):
        if ev.type == pygame.QUIT:
            self.running = False
            return

        if self.overlay:
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self.overlay = None
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self._pick_overlay(ev.pos)
            return

        b = self.buttons
        if b["Save"].clicked(ev): self.save_map(); return
        if b["Load"].clicked(ev): self.open_overlay("load"); return
        if b["Change Map"].clicked(ev): self.change_map(); return
        if b["Targets"].clicked(ev):
            self.sim.randomize_targets(); self._set_status(f"targets randomized ({len(self.sim.targets)})"); return
        if b["Reset"].clicked(ev): self.reset_sim(); return
        if b["Play"].clicked(ev): self.toggle_play(); return
        if b["Step"].clicked(ev): self.step_once(); return
        if b["Model"].clicked(ev): self.open_overlay("model"); return
        if b["spd-"].clicked(ev): self.speed_idx = max(0, self.speed_idx - 1); return
        if b["spd+"].clicked(ev): self.speed_idx = min(len(self.speeds) - 1, self.speed_idx + 1); return
        if b["gtmin-"].clicked(ev): self.gt_min -= 10; self._sync_golden_time(); return
        if b["gtmin+"].clicked(ev): self.gt_min += 10; self._sync_golden_time(); return
        if b["gtmax-"].clicked(ev): self.gt_max -= 10; self._sync_golden_time(); return
        if b["gtmax+"].clicked(ev): self.gt_max += 10; self._sync_golden_time(); return

        # click a robot card in the info panel -> select / deselect
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for rect, i in self.robot_card_rects:
                if rect.collidepoint(ev.pos):
                    self.selected = None if self.selected == i else i
                    return

        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE: self.toggle_play(); return
            if ev.key in (pygame.K_n, pygame.K_RIGHT): self.step_once(); return
            if ev.key == pygame.K_r: self.reset_sim(); return
            if ev.key == pygame.K_m:
                if self.model_policy is not None:
                    self.use_model = not self.use_model
                    self._set_status(f"policy: {'model' if self.use_model else 'stand-in'}")
                return
            if pygame.K_1 <= ev.key <= pygame.K_9:
                idx = ev.key - pygame.K_1
                if idx < len(self.sim.agents):
                    self.selected = idx
                    self._set_status(f"selected R{idx}")
                return
            if ev.key == pygame.K_p:
                self._print_windows(); return

        if not self.playing:
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                cell = self.cell_at(ev.pos)
                if cell:
                    self.sim.toggle_target(cell)
                return
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                cell = self.cell_at(ev.pos)
                if cell:
                    self.paint_value = not self.sim.obstacles[cell[0]][cell[1]]
                    self.sim.obstacles[cell[0]][cell[1]] = self.paint_value
                    self.dragging = True
                    self.last_cell = cell
                return
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.dragging = False
                self.last_cell = None
                return
            if ev.type == pygame.MOUSEMOTION and self.dragging:
                cell = self.cell_at(ev.pos)
                if cell and cell != self.last_cell:
                    r0, c0 = self.last_cell or cell
                    r1, c1 = cell
                    steps = max(abs(r1 - r0), abs(c1 - c0)) or 1
                    for k in range(steps + 1):
                        rr = round(r0 + (r1 - r0) * k / steps)
                        cc = round(c0 + (c1 - c0) * k / steps)
                        self.sim.obstacles[rr][cc] = self.paint_value
                    self.last_cell = cell

    def _print_windows(self):
        if self.selected is None:
            print("[debug] select a robot first (click a card or press 1-9)")
            return
        a = self.sim.agents[self.selected]
        if a.local_window is None:
            print("[debug] no observation yet")
            return
        print(f"[debug] R{self.selected} (x, y)={self.sim.frame.to_xy(a.pos)} mode={a.mode.name} "
              f"locked={a.locked} capacity={a.capacity}/{a.capacity_max} assigned={a.assigned_target}")
        print("local_window", a.local_window.shape, " global_window", a.global_window.shape)
        ob = self.sim.rl_observation()["agents"][self.selected]
        for k in ("self_feats", "explore_feats", "depot_feats", "node_mask", "free_choice"):
            print(f"{k:14s}", ob[k])
        n = int(ob["task_present"].sum())
        print("task_feats    ", ob["task_feats"][:n])

    # ==================== main loop ====================

    def run(self):
        while self.running:
            for ev in pygame.event.get():
                self.handle_event(ev)

            if self.playing:
                self._frame += 1
                if self._frame % self.speeds[self.speed_idx] == 0:
                    self.step_once()

            self.draw_grid()
            self.draw_info_panel()
            self.draw_obs_panel()
            self.draw_bottom_panel()
            if self.overlay:
                self.draw_overlay()

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()
