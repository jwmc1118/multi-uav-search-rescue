"""Grid map visualization + editor (pygame).

Layout (3 columns):
  [ 전체 맵 ]  [ info panel: stats / robots / targets ]  [ inspector: 선택 UAV 의 local / group 지도 -> legend -> GROUPS ]
  [ ------------------------ bottom bar: buttons / speed / golden-time ------------------- ]

색 / 도형은 전부 gui/style.py 의 스타일 표 (STYLE / UI) 에서 온다 — 여기서 RGB 를 직접 쓰지 않는다.
일반 기호 legend 는 메인 맵 아래 한 줄 바 (H 토글, 기본 표시). inspector 는 local view -> group map (DBSCAN 덩어리 색)
inspector 2 열: 왼쪽 = local view / group map, 오른쪽 = 각 지도 옆에 My cluster segments + Goal / Group clusters + GROUPS.
화면에 나오는 문자열은 전부 영어. inspector 글자 크기 = gui/style.py BASE_FONT (main.py GUI_FONT_SIZE),
inspector 폭 = main.py INSPECTOR_WIDTH — 레이아웃 (_inspector_geometry) 은 창 크기가 바뀔 때만 계산해 캐시하고, 선택 UAV /
위치 / 텍스트와 무관하게 매 프레임 같다. 지도 셀 크기는 패널 폭에 맞추고, 높이가 모자라면 두 지도를 같은 비율로 줄인다.
이벤트 기록 (event_log) 은 계속 쌓지만 화면에는 표시하지 않는다.
  ● UAV (굵은 이중 링 = 선택)  ▲ 생존자  ★ 이동 목표점  ◆ 구간 대표점 (medoid, 개인 뷰)  ■ depot  점선 = BFS 경로
  칸: 장애물 진회색 / 탐색됨 흰색 / 미탐색 진파랑 / frontier 초록 / 내 구간 = 내 UAV 색 진하게 /
      같은 덩어리 팀원 구간 = 그 팀원 색 연하게 + 번호
실행은 레포 루트에서  python main.py

Controls
- Space or Play : play / pause          N or Right or Step : single step
- click a robot card / UAV on the map / 1..9 / Tab : select robot -> 개인 패널, Esc : 해제
- O : print selected robot's obs to console        M : toggle policy (rules -> random -> model)
- while paused: left-drag = obstacles, right-click = add/remove target
- Change Map : new random obstacle map + targets   Targets : re-place targets only
- Save/Load : map file   Model : load a trained checkpoint
- G or Alloc Plots : save allocation analysis figures to figures/<time>_<policy>_t<step>/ and open the folder
- P (B) or Path : BFS 경로   C : 덩어리 · 구간 색칠 (전체 맵)   L : 할당 연결선 · 중복 추적   H : 전체 맵 범례
- V : 센서 범위 (UAV 색 반투명) · 통신 범위 (UAV 색 점선 사각형) — 선택 UAV 가 있으면 그 UAV 만, 없으면 전체

결정은 기본적으로 규칙 정책 (policies/role_rules.py), M 키로 rules -> random -> model. 체크포인트를 불러오면
evaluation/checkpoint_policy.py 를 통해 학습된 정책이 구동한다.
화면의 모든 위치 표기는 월드 좌표 (x, y) 이다 (env/core/coords.py: x 오른쪽, y 위쪽, 원점 = 맵 중앙).
"""
import os
import sys
import glob
import multiprocessing
import queue
import subprocess

import numpy as np
import pygame

from evaluation.alloc_recorder import AllocRecorder
from env.role.simulation import RoleSimulation
from env.core.modes import Mode
from env.core.knowledge import frontier_map
from gui import style
from gui.style import (STYLE, UI, Usage, cluster_color, draw_cell, draw_edges, draw_marker, draw_path, legend_icon,
                       font as style_font)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _open_folder(path):
    """OS 기본 파일 뷰어로 폴더 열기 (비블로킹). 실패해도 무시."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _plot_worker(data, figures_dir, q):
    """별도 프로세스에서 분석 그래프 생성 (GUI 그리기 루프와 GIL 을 다투지 않게). 결과를 큐로 돌려준다."""
    try:
        from evaluation.alloc_plots import save_alloc_plots
        q.put(("ok", save_alloc_plots(data, figures_dir)))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))


MAP_DIR = os.path.join("maps", "map")
MAP_TARGET_DIR = os.path.join("maps", "map_target")
# training/train.py 가 체크포인트를 저장하는 곳 (레포 루트/marl_models)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marl_models")

INSPECTOR_WIDTH = 560          # inspector 패널 기본 폭 (main.py INSPECTOR_WIDTH 로 바꾼다)
MODE_NAME = {Mode.EXPLORE: "EXPLORE", Mode.RESCUE: "RESCUE", Mode.RETURN: "RETURN"}
MODE_UI = {Mode.EXPLORE: "mode_explore", Mode.RESCUE: "mode_rescue", Mode.RETURN: "mode_return"}


def _font(size, bold=False):
    return style_font(size, bold)


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
        color = UI["button_on"] if (self.toggle and self.on) else (UI["button_hover"] if hover else UI["button"])
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        t = self.font.render(self.text, True, UI["button_text"])
        surface.blit(t, t.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))


class GridMap:
    def __init__(self, sim: RoleSimulation, cell_size=22, info_panel_width=340,
                 obs_panel_width=None, panel_height=104, extra_height=180, fps=60, font_size=None,
                 model_policy=None, model_mode="greedy",
                 figures_dir=None, auto_save_plots=False, speed_index=2, model_config=None):
        self.sim = sim
        obs_panel_width = obs_panel_width or INSPECTOR_WIDTH   # inspector 폭 (local / group 지도가 이 폭에 맞춰 커진다)
        style.set_base_font(font_size)                         # inspector 기준 글자 크기 (main.py GUI_FONT_SIZE)
        # 임무 할당 분석 (evaluation/alloc_recorder.py · alloc_plots.py): 기록기를 붙이고, Alloc Plots 버튼 / G 키 / 에피소드 끝(옵션)에 그래프 저장
        self.figures_dir = figures_dir or os.path.join(REPO_ROOT, "figures")
        self.auto_save_plots = auto_save_plots
        self._plot_proc = None
        self._plot_q = None
        # 결정 정책 (M 키로 rules -> random -> model). 모델을 받으면 model 로 시작
        self.role_policy = "model" if model_policy is not None else "rules"
        self.event_log = []                  # 최근 결정 이벤트 (role 모드 우측 패널)
        self.event_log_lines = 8             # 패널에 보이는 이벤트 로그 줄 수 (공간이 남으면 더)
        self.show_clusters = False           # C: 전체 맵에 덩어리 · 구간 색칠 (G-2 규칙, 선택 UAV 시점)
        self.show_links = False              # L: 전체 맵에 UAV -> 생존자 연결선 / 중복 추적
        self.show_ranges = True              # V: 전체 맵에 센서 범위 / 통신 범위 (선택 UAV 가 있으면 그 UAV 만)
        self.show_legend = True              # H: 메인 맵 아래 legend 바 (일반 기호, 기본 표시)
        self.use_main, self.use_personal = Usage(), Usage()   # 이번 프레임에 그린 요소 (범례)
        sim.recorder = AllocRecorder()
        sim.recorder.on_reset(sim)
        self.model_mode = model_mode         # Model 버튼으로 불러올 때 행동 선택 방식 (greedy / sample)
        self.model_config = model_config     # Model 버튼: 체크포인트와 네트워크 구조 비교용 (main 전용 yaml 의 custom_model_config)
        self.rows, self.cols = sim.rows, sim.cols
        self.cs = cell_size
        self.panel_height = panel_height
        self.fps = fps
        self.model_policy = model_policy
        sim.keep_map = True      # GUI 의 Reset 은 같은 맵/조난자를 다시 재생 (새 맵은 Change Map)

        pygame.init()                        # 폰트 먼저 (inspector 텍스트 높이로 창 높이를 정한다)
        self.f_title = _font(28, bold=True)
        self.f_head = _font(21, bold=True)
        self.f = _font(19)
        self.f_sm = _font(16)
        self.f_xs = _font(14)
        self.f_ins_title = style_font(style.BASE_FONT + style.TITLE_DELTA, bold=True)   # inspector 제목
        self.f_ins = style_font(style.BASE_FONT)                                       # inspector legend / 본문
        self.f_legend_bar = style_font(style.BASE_FONT - 1)                            # 메인 맵 legend 바 (H)
        self.f_ins_bold = style_font(style.BASE_FONT, bold=True)                       # inspector 강조 줄
        self.cluster_rows = int((sim.cfg.get("gui") or {}).get("cluster_info_max_rows", 8))   # Cluster info 덩어리 줄 수

        self.grid_width = self.cols * self.cs
        self.grid_height = self.rows * self.cs
        # inspector: 내부 폭을 local (S 칸) / group (cols 칸) 정수 셀에 가장 잘 맞는 값으로 보정 -> 두 지도가 좌우를 꽉 채움
        self.map_w = self._fit_inspector_width(obs_panel_width - 2 * self.INS_PAD)   # 지도 열 폭 (local / group)
        self.text_w = self._measure_text_width()                                      # 오른쪽 설명 열 폭 (최장 문구 기준)
        obs_panel_width = self.INS_PAD + self.map_w + self.INS_COL_GAP + self.text_w + self.INS_PAD
        self.grid_area_w = max(self.grid_width, 560)
        self.info_x = self.grid_area_w
        self.info_w = info_panel_width
        self.obs_x = self.info_x + info_panel_width
        self.obs_w = obs_panel_width
        self.screen_width = self.obs_x + obs_panel_width
        # 사이드 패널(info/obs) 높이 = 그리드 높이 + extra_height. 하단 바는 그 아래에 붙는다.
        self.content_h = self.grid_height + extra_height          # 창 높이는 늘리지 않음 (설명은 지도 오른쪽 열)
        self.screen_height = self.content_h + panel_height

        self.selected = None
        self.show_path = True                # BFS 경로 표시 (Path 버튼 / B 키)
        self.robot_card_rects = []           # [(pygame.Rect, robot_index), ...]
        self.playing = False
        self._frame = 0
        self.speeds = [16, 10, 6, 3, 1]      # auto-step interval in frames; smaller = faster
        self.speed_idx = min(max(0, int(speed_index)), len(self.speeds) - 1)   # 초기 속도 (main 전용 yaml main.speed_index)

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

        pygame.display.set_caption("Rescue Simulation")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self._num_fonts = {}
        self._ins_geom = None                # inspector 고정 레이아웃 캐시 (창 크기 / 글자 크기가 바뀔 때만 재계산)
        self.ins_layout = {}                 # 마지막 inspector 배치 (요소별 Rect, 점검용)

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
                         ("Play", 74), ("Step", 62), ("Model", 70), ("Path", 60), ("Alloc Plots", 104)]:
            self.buttons[label] = Button((x, y1, w, bh), label, self.f_sm,
                                         toggle=label in ("Play", "Path"))
            x += w + 7
        self.buttons["Path"].on = self.show_path
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
        message = str(message)
        if not message.isascii():                    # 화면 글자는 영어만 (한글 예외 메시지 등은 콘솔에 전체 출력)
            print(f"[gui] {message}")
            message = message.encode("ascii", "ignore").decode().split("\n")[0].strip() + "  (details in console)"
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

    def _policy_label(self):
        if self.role_policy == "model" and self.model_policy is not None:
            path = getattr(self.model_policy, "checkpoint_path", "")
            return "model-" + os.path.basename(os.path.dirname(path) or path)   # model-checkpoint_001500
        return self.role_policy

    def step_once(self):
        self.sim.recorder.policy = self._policy_label()
        from policies.role_rules import RandomDecider, RuleDecider
        if not hasattr(self, "_deciders"):
            self._deciders = {"rules": RuleDecider(), "random": RandomDecider()}
        if self.role_policy == "model" and self.model_policy is not None:
            try:
                done, info = self.model_policy.step(self.sim)
            except Exception as e:
                self._set_status(f"model inference failed -> rules: {e}")
                self.role_policy = "rules"
                return
        else:
            done, info = self.sim.step(self._deciders[self.role_policy])
        self._log_events(info)
        if done:
            self._set_status(f"episode done @t{info['time']}  "
                             f"rescued {info['rescued_total']} / expired {info['expired_total']}")
            if self.auto_save_plots:
                self.save_alloc_plots()
            self.playing = False
            self.buttons["Play"].on = False
            self.buttons["Play"].text = "Play"

    def change_map(self):
        """새 랜덤 장애물 맵 + 조난자 (env/core/map_gen.py). 장애물 비율 / 조난자 수 범위는 sim.cfg 를 따른다."""
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
        self.event_log = []
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
                        from evaluation.checkpoint_policy import CheckpointPolicy
                        # GUI sim 의 설정 (main 전용 yaml) 을 그대로 쓴다. 관측 차원 / 네트워크 구조가 다르면 에러
                        policy = CheckpointPolicy(path, env_config=dict(self.sim.cfg), mode=self.model_mode,
                                                  model_config=self.model_config, env_priority=True)
                        self.model_policy = policy
                        self.role_policy = "model"
                        self._set_status(f"model loaded: {os.path.basename(path)}  {policy.describe()}", frames=600)
                    except Exception as e:
                        self._set_status(f"model load failed: {e}")
                self.overlay = None
                return
        self.overlay = None

    # ==================== drawing: 공통 ====================

    def _cell_class(self, i, r, c):
        """UAV i 지식 (그룹 병합) 기준 칸 종류 -> STYLE 키."""
        kt = self.sim.knowledge
        if kt.known_obstacle[i][r, c]:
            return "obstacle"
        if kt.frontier[i][r, c]:
            return "frontier"
        if kt.known_free[i][r, c]:
            return "free"
        return "unknown"

    def _segments_view(self, i):
        """UAV i 시점의 구간 칸 색칠 정보: {cell: (STYLE 키, UAV 색, 번호 표시 여부/번호)} + medoid 목록 + 추정 담당 칸.
        G-2: 내 구간 = 내 색 진하게 / 같은 덩어리 팀원 구간 = 그 팀원 색 연하게 + 셀 하나에 번호."""
        sim = self.sim
        cells, meds = {}, []
        if i is None:
            return cells, meds
        a = sim.agents[i]
        g = sim.comp[i]
        if a.mode is Mode.EXPLORE and a.cluster:
            for c in a.cluster:
                cells[c] = ("my_segment", a.color, None)
            if a.cluster_medoid:
                meds.append(a.cluster_medoid)
        if a.blob is not None:
            for j in sim.assignees(g, a.blob):
                if j == i or not sim.agents[j].cluster:
                    continue
                b = sim.agents[j]
                seg = sorted(b.cluster)
                tag = next((c for c in seg if c != b.cluster_medoid), seg[0])
                for c in seg:
                    cells.setdefault(c, ("mate_segment", b.color, str(j) if c == tag else None))
                if b.cluster_medoid:
                    meds.append(b.cluster_medoid)
        return cells, meds

    def _uav_number(self, j):
        return str(j)

    # ==================== drawing: 전체 맵 ====================

    def draw_grid(self):
        surf, sim = self.screen, self.sim
        self.use_main = Usage()
        u = self.use_main
        surf.fill(UI["gap"], (0, 0, self.grid_area_w, self.content_h))
        kt = sim.knowledge
        team_free = kt.team_known_free()
        team_obs = np.zeros_like(team_free)
        for o in kt.known_obstacle:
            team_obs |= o
        team_front = frontier_map(team_free, team_obs)
        truth = sim.obstacles
        seg_cells = self._main_segments() if self.show_clusters else {}
        for r in range(self.rows):
            for c in range(self.cols):
                rect = self.cell_rect(r, c)
                if truth[r][c]:                                  # 전체 맵 = 진실 (장애물은 몰라도 표시)
                    draw_cell(surf, "obstacle", rect, usage=u)
                elif (r, c) in seg_cells:
                    key, col, lab = seg_cells[(r, c)]
                    draw_cell(surf, key, rect, col, u, lab, self.f_xs)
                elif team_front[r, c]:
                    draw_cell(surf, "frontier", rect, usage=u)
                elif team_free[r, c]:
                    draw_cell(surf, "free", rect, usage=u)
                else:
                    draw_cell(surf, "unknown", rect, usage=u)
        self._draw_grid_lines()
        dr, dc = sim.depot
        draw_marker(surf, "depot", self.cell_center(dr, dc), self.cs * 0.45, usage=u, text="D", fnt=self.f_xs)
        if self.show_ranges:
            self._draw_ranges(u)
        if self.show_links:
            self._draw_links(u)
        self._draw_targets(u)
        if self.show_path:
            for i, a in enumerate(sim.agents):
                self._draw_path_on(surf, i, lambda cell: self.cell_center(*cell), 4 if i == self.selected else 2, u)
        self._draw_agents(u)
        if self.selected is not None:
            a = sim.agents[self.selected]
            if a.goal is not None and a.mode is Mode.EXPLORE:      # RESCUE = 생존자 링, RETURN = depot 이 목표를 나타냄
                draw_marker(surf, "goal", self.cell_center(*a.goal), self.cs * 0.55, a.color, u)
        if self.show_legend:
            self._draw_main_legend()

    def _main_segments(self):
        """C 토글: 선택 UAV 가 있으면 그 시점 (G-2), 없으면 모든 UAV 구간을 각자 색 + 번호로."""
        sim = self.sim
        if self.selected is not None:
            cells, _ = self._segments_view(self.selected)
            mine_blob = sim.agents[self.selected].blob
        else:
            cells, mine_blob = {}, None
        for j, b in enumerate(sim.agents):
            if b.mode is not Mode.EXPLORE or not b.cluster or (j == self.selected) or \
                    (mine_blob is not None and b.blob == mine_blob and sim.comp[j] == sim.comp[self.selected]):
                continue
            seg = sorted(b.cluster)
            tag = next((c for c in seg if c != b.cluster_medoid), seg[0])
            for c in seg:
                cells.setdefault(c, ("seg_other", b.color, str(j) if c == tag else None))
        return cells

    def _draw_path_on(self, surf, i, to_px, width, usage, keep=None, dash=6):
        path = self.sim.planned_path(i)[0]
        if not path or len(path) < 2:
            return
        pts = [to_px(p) for p in path if keep is None or keep(p)]
        if len(pts) > 1:
            draw_path(surf, "path", pts, self.sim.agents[i].color, width, usage, dash=dash)

    def _draw_targets(self, u):
        surf, sim = self.screen, self.sim
        mine = None
        if self.selected is not None:
            mine = sim.agents[self.selected].assigned_target
        for ti, t in enumerate(sim.targets):
            if t.rescued_at is not None:
                key = "survivor_rescued"
            elif t.expired_at is not None:
                key = "survivor_expired"
            elif not t.discovered:
                key = "survivor_hidden"
            else:
                key = "survivor"
            c = self.cell_center(*t.pos)
            draw_marker(surf, key, c, self.cs * 0.48, usage=u)
            if ti == mine and t.active:
                draw_marker(surf, "survivor_mine", c, self.cs * 0.62, sim.agents[self.selected].color, u)

    def _draw_agents(self, u):
        surf = self.screen
        for i, a in enumerate(self.sim.agents):
            c = self.cell_center(*a.pos)
            draw_marker(surf, "uav", c, self.cs * 0.5, a.color, u, self._uav_number(i), self.f_xs, UI["badge_text"])
            if i == self.selected:
                draw_marker(surf, "uav_selected", c, self.cs * 0.62, usage=u)
            letter = {Mode.EXPLORE: "E", Mode.RESCUE: "R", Mode.RETURN: "D"}[a.mode]   # 역할 배지: E / R / D (depot 복귀)
            draw_marker(surf, "role_badge", (c[0] + self.cs * 0.55, c[1] - self.cs * 0.55), max(6, self.cs // 3),
                        usage=u, text=letter, fnt=self.f_xs, text_color=UI["badge_text"])

    def _draw_ranges(self, u):
        """V 토글: 센서 범위 (Chebyshev <= sensor_range, 실제 관측은 장애물에 가려질 수 있음) = UAV 색 반투명 사각형,
        통신 범위 (다른 UAV 가 이 안 = Chebyshev <= comm_range 면 직접 연결 (끊기는 건 + comm_hysteresis 밖),
        env/core/knowledge.py) = UAV 색 점선 사각형.
        선택 UAV 가 있으면 그 UAV 만, 없으면 전체. 맵 셀 영역 밖으로는 그리지 않는다."""
        surf, sim = self.screen, self.sim
        who = [self.selected] if self.selected is not None else range(sim.n_agents)
        surf.set_clip(pygame.Rect(0, 0, self.grid_width, self.grid_height))
        for i in who:
            a = sim.agents[i]
            c = self.cell_center(*a.pos)
            draw_marker(surf, "sensor_range", c, (a.sensor_range + 0.5) * self.cs, a.color, u)
            draw_marker(surf, "comm_range", c, (a.comm_range + 0.5) * self.cs, a.color, u)
        surf.set_clip(None)

    def _draw_links(self, u):
        """L 토글: UAV -> 맡은 생존자 실선, 같은 생존자 중복 추적."""
        surf, sim = self.screen, self.sim
        chasers = {}
        for i, a in enumerate(sim.agents):
            if a.mode is Mode.RESCUE and a.assigned_target is not None:
                t = sim.targets[a.assigned_target]
                draw_path(surf, "assign_link", [self.cell_center(*a.pos), self.cell_center(*t.pos)], a.color, 2, u)
                chasers[a.assigned_target] = chasers.get(a.assigned_target, 0) + 1
        for ti, k in chasers.items():
            if k >= 2:
                draw_marker(surf, "conflict", self.cell_center(*sim.targets[ti].pos), self.cs * 0.9, usage=u)

    MAIN_LEGEND = ("obstacle", "free", "unknown", "frontier", "out_of_map", "survivor", "depot", "uav", "path",
                   "sensor_range", "comm_range")

    def _draw_main_legend(self):
        """메인 맵 아래 (버튼 바 위) 고정 legend 바 — 일반 기호. 맵 셀 영역 아래 빈 공간에 그려 셀을 가리지 않는다 (H 토글)."""
        surf, f = self.screen, self.f_legend_bar
        x0, y = 10, self.grid_height + 8
        lh = f.get_linesize() + 6
        icon = f.get_height()
        x = x0
        for k in self.MAIN_LEGEND:
            label = STYLE[k].main_label or STYLE[k].label
            w = icon + 6 + f.size(label)[0] + 22
            if x + w > self.grid_area_w - 10 and x > x0:
                x, y = x0, y + lh
            legend_icon(surf, k, pygame.Rect(x, y + (lh - icon) // 2, icon, icon), None, f)
            surf.blit(f.render(label, True, UI["text"]), (x + icon + 6, y + (lh - f.get_height()) // 2))
            x += w
        self.legend_bar_rect = pygame.Rect(x0, self.grid_height + 8, self.grid_area_w - 20, y + lh - self.grid_height - 8)

    def _draw_grid_lines(self):
        surf = self.screen
        for c in range(self.cols + 1):
            x = c * self.cs
            pygame.draw.line(surf, UI["grid_line"], (x, 0), (x, self.grid_height))
        for r in range(self.rows + 1):
            y = r * self.cs
            pygame.draw.line(surf, UI["grid_line"], (0, y), (self.grid_width, y))

    # ==================== drawing: bottom bar ====================

    def draw_bottom_panel(self):
        surf = self.screen
        pygame.draw.rect(surf, UI["panel"], (0, self.content_h, self.screen_width, self.panel_height))
        for b in self.buttons.values():
            b.draw(surf)
        surf.blit(self.f_sm.render(f"x{self.speeds[self.speed_idx]}", True, UI["subtext"]),
                  (self._spd_label_x, self.content_h + 16))
        y = self._gt_y + 3
        surf.blit(self.f_sm.render("golden time   min", True, UI["subtext"]), (12, y))
        surf.blit(self.f_head.render(str(self.gt_min), True, UI["text"]), (self._gt_min_val_x, y - 4))
        surf.blit(self.f_sm.render("max", True, UI["subtext"]), (self._gt_max_label_x, y))
        surf.blit(self.f_head.render(str(self.gt_max), True, UI["text"]), (self._gt_max_val_x, y - 4))
        cell = self.cell_at(pygame.mouse.get_pos())
        if cell is not None:
            surf.blit(self.f_sm.render(f"cursor (x, y) = {self.sim.frame.to_xy(cell)}", True, UI["text"]),
                      (self.screen_width - 420, y))
        onoff = lambda v: "on" if v else "off"
        help_txt = (f"[C] segments {onoff(self.show_clusters)}   [L] links {onoff(self.show_links)}   "
                    f"[V] ranges {onoff(self.show_ranges)}   [P] path {onoff(self.show_path)}   "
                    f"[H] legend {onoff(self.show_legend)}"
                    "   [Tab/1-9/click] select  [Esc] deselect  [M] rules/random/model  [G] plots")
        surf.blit(self.f_xs.render(help_txt, True, UI["subtext"]), (self._gt_max_val_x + 110, y + 3))
        surf.blit(self.f_sm.render(f"policy: {self.role_policy}", True, UI["accent"]), (self.screen_width - 200, y))
        if self.status_timer > 0:
            surf.blit(self.f_sm.render(self.status, True, UI["text"]), (self._status_x, self.content_h + 16))
            self.status_timer -= 1

    # ==================== drawing: info panel ====================

    def draw_info_panel(self):
        surf = self.screen
        x0 = self.info_x
        pygame.draw.rect(surf, UI["info"], (x0, 0, self.info_w, self.content_h))
        pad = 16
        x = x0 + pad
        w = self.info_w - 2 * pad
        y = pad
        info = self.sim._info([], [])
        surf.blit(self.f_title.render(f"t = {self.sim.time}", True, UI["text"]), (x, y))
        y += 38
        surf.blit(self.f_sm.render(f"rescued {info['rescued_total']}    expired {info['expired_total']}",
                                   True, UI["subtext"]), (x, y))
        y += 22
        surf.blit(self.f_sm.render(f"active {info['active']}    found {info['discovered_total']}", True, UI["subtext"]), (x, y))
        y += 30
        y = self._draw_robots(x, y, w)
        y += 12
        self._draw_target_list(x, y, w)

    def _draw_robots(self, x, y, w):
        surf = self.screen
        surf.blit(self.f_head.render("ROBOTS", True, UI["subtext"]), (x, y))
        surf.blit(self.f_xs.render("click to inspect", True, UI["subtext"]), (x + self.f_head.size("ROBOTS")[0] + 12, y + 7))
        y += 32
        self.robot_card_rects = []
        mouse = pygame.mouse.get_pos()
        for i, a in enumerate(self.sim.agents):
            h = 78
            card = pygame.Rect(x, y, w, h)
            self.robot_card_rects.append((card.copy(), i))
            bg = "card_hi" if i == self.selected else ("card_hover" if card.collidepoint(mouse) else "card")
            pygame.draw.rect(surf, UI[bg], card, border_radius=7)
            pygame.draw.rect(surf, UI["card_border"], card, width=1, border_radius=7)
            draw_marker(surf, "uav", (x + 18, y + 21), 9, a.color, None, self._uav_number(i), self.f_xs, UI["badge_text"])
            surf.blit(self.f.render(f"R{i}", True, UI["text"]), (x + 34, y + 10))
            name = MODE_NAME[a.mode]
            bw = self.f_xs.size(name)[0] + 16
            badge = pygame.Rect(x + 74, y + 12, bw, 20)
            pygame.draw.rect(surf, UI[MODE_UI[a.mode]], badge, border_radius=10)
            surf.blit(self.f_xs.render(name, True, UI["text"]), (badge.x + 8, badge.y + 3))
            surf.blit(self.f_sm.render(f"supply {a.capacity}/{a.capacity_max}", True, UI["subtext"]), (x + 34, y + 36))
            px = x + 34 + self.f_sm.size("supply 0/0")[0] + 14
            for k in range(a.capacity_max):
                pygame.draw.rect(surf, UI["supply_on"] if k < a.capacity else UI["supply_off"],
                                 pygame.Rect(px + k * 15, y + 38, 11, 12), border_radius=2)
            if a.assigned_target is not None:
                t = self.sim.targets[a.assigned_target]
                txt = f"-> T{a.assigned_target} {self.sim.frame.to_xy(t.pos)}"
            elif a.mode is Mode.RESCUE:
                txt = "-> unassigned"
            elif a.mode is Mode.RETURN:
                txt = f"-> depot {self.sim.frame.to_xy(self.sim.depot)}"
            elif getattr(a, "blob", None) is not None:
                txt = f"-> blob C{a.blob}" + (f", {len(a.cluster)} cells" if a.cluster else "")
            else:
                txt = "-> frontier" if a.goal else "-> --"
            surf.blit(self.f_sm.render(txt, True, UI["subtext"]), (x + 34, y + 56))
            y += h + 8
        return y

    def _draw_target_list(self, x, y, w):
        surf = self.screen
        surf.blit(self.f_head.render("TARGETS", True, UI["subtext"]), (x, y))
        surf.blit(self.f_xs.render("golden time / deadline", True, UI["subtext"]),
                  (x + self.f_head.size("TARGETS")[0] + 12, y + 7))
        y += 32
        now = self.sim.time
        lh = 30
        for ti, t in enumerate(self.sim.targets):
            if y > self.content_h - lh:
                surf.blit(self.f_sm.render("...", True, UI["subtext"]), (x, y))
                break
            row = pygame.Rect(x, y, w, lh - 4)
            pygame.draw.rect(surf, UI["card"], row, border_radius=5)
            pygame.draw.rect(surf, UI["card_border"], row, width=1, border_radius=5)
            if t.rescued_at is not None:
                tail, tc = f"RESCUED @t{t.rescued_at} R{t.rescued_by}", UI["ok"]
            elif t.expired_at is not None:
                tail, tc = f"EXPIRED @t{t.expired_at}", UI["warn"]
            elif t.end_time is not None:
                rem = t.remaining(now)
                tail = f"end t{t.end_time} · left {rem}" + ("" if t.discovered else " · hidden")
                tc = UI["warn"] if rem <= 25 else UI["text"]
            else:
                tail, tc = f"golden {t.golden_time} · not found", UI["subtext"]
            surf.blit(self.f_sm.render(f"T{ti}", True, UI["text"]), (x + 8, y + 3))
            surf.blit(self.f_xs.render(str(self.sim.frame.to_xy(t.pos)), True, UI["subtext"]), (x + 42, y + 6))
            surf.blit(self.f_sm.render(tail, True, tc), (x + 108, y + 3))
            y += lh

    # ==================== drawing: 개인 패널 (선택 UAV) ====================

    INS_PAD, INS_GAP, INS_COL_GAP = 6, 6, 10   # inspector 바깥 여백 / 요소 간격 / 지도 열 - 설명 열 간격

    def _fit_inspector_width(self, w0):
        """지도 열 폭: w0 이상 w0 + 60 px 안에서 local (S 칸) 과 group (cols 칸) 정수 셀이 좌우를 가장 꽉 채우는 폭
        (w % S + w % cols 최소, 동률이면 w0 에 가까운 쪽). 줄이지 않으므로 지도는 설정한 폭보다 작아지지 않는다."""
        S = self.sim.knowledge.local_size
        cands = range(max(2 * S, w0), w0 + 61)
        return min(cands, key=lambda w: (w % S + w % self.cols, abs(w - w0)))

    def _measure_text_width(self):
        """오른쪽 설명 열 폭 = 들어갈 수 있는 가장 긴 문구 (최악의 경우) 의 실제 폰트 폭. 내용이 바뀌어도 레이아웃 고정."""
        n = self.sim.n_agents
        f, fb = self.f_ins, self.f_ins_bold
        sw = f.get_linesize() + 2 - 6 + 8                           # 색 사각형 + 간격
        uavs = ",".join(f"R{i}" for i in range(n))
        cands = [fb.size("My cluster segments")[0], f.size("(k-medoids, local view colors)")[0],
                 fb.size("Group clusters (DBSCAN)")[0],
                 sw + fb.size(f"C999 \u00b7 9999 cells \u00b7 {uavs} (m={n})")[0],
                 sw + fb.size("R99 \u00b7 9999 cells")[0],
                 f.size("Goal: R99 \u2192 (-99, -99) \u00b7 BFS 999 steps")[0],
                 f.size("Rescue target T99 (-99, -99)")[0],
                 f.size("BFS 999 steps \u00b7 deadline left 999")[0],
                 f.size("No explore cluster (role: RESCUE)")[0],
                 2 * f.get_linesize() + f.size("goal point  /  segment medoid")[0] + 20,
                 self.f_ins_title.size(f"GROUPS  (n_hop = {self.sim.knowledge.n_hop})")[0],
                 f.size("[" + " ".join(f"R{i}" for i in range(n)) + "]")[0]]
        return int(max(cands)) + 2

    def _inspector_text_heights(self, W):
        """제목 · 목록 · Goal · GROUPS 높이 (최대 행 수 기준, 내용과 무관). W = 설명 열 폭."""
        title_h = self.f_ins_title.get_linesize() + 2
        lh = self.f_ins.get_linesize() + 2
        n = self.sim.n_agents
        seg_h = (2 + n + 2) * lh                                   # 제목 2 줄 + N 행 + Goal 2 줄
        cl_h = (1 + self.cluster_rows + 2) * lh                    # 제목 + 덩어리 행 + more + noise
        groups_h = max(self._draw_groups_section(0, 0, W, dry=True, groups=[([i], 0) for i in range(n)]),
                       self._draw_groups_section(0, 0, W, dry=True, groups=[(list(range(n)), 0)]))
        return title_h, lh, seg_h, cl_h, groups_h

    def _inspector_geometry(self):
        """inspector 고정 레이아웃 (창 크기 · 글자 크기 · 맵 크기가 바뀔 때만 다시 계산, 캐시). 2 열:
            [ 지도 열 (폭 map_w) ]                  [ 설명 열 (폭 text_w, 최장 문구 기준) ]
              Local view 제목 + local (정사각)        My cluster segments (N 행) + Goal 2 줄   <- local 위쪽에 맞춤
              Group map 제목 + group (종횡비)         Group clusters (max_rows 행 + more + noise) + GROUPS  <- group 위쪽에 맞춤
        두 지도 같은 축척 f = min(1, H / (W (1 + rows / cols))), W = map_w, H = 패널 높이 - 제목 2 개 - 여백.
        f = 1 이면 local 한 변 / group 폭 = 지도 열 폭 (셀 = 정수 픽셀 내림, 남는 픽셀 가운데 정렬)."""
        S = self.sim.knowledge.local_size
        key = (self.obs_x, self.obs_w, self.content_h, style.BASE_FONT, self.rows, self.cols, S, self.sim.n_agents,
               self.cluster_rows, self.map_w, self.text_w)
        g = getattr(self, "_ins_geom", None)
        if g is not None and g["key"] == key:
            return g
        pad, gap = self.INS_PAD, self.INS_GAP
        W, TW = self.map_w, self.text_w
        xm = self.obs_x + pad
        xt = xm + W + self.INS_COL_GAP
        title_h, lh, seg_h, cl_h, groups_h = self._inspector_text_heights(TW)
        H = self.content_h - 2 * pad - 2 * title_h - gap
        f = max(0.0, min(1.0, H / (W * (1 + self.rows / self.cols))))
        lc = max(2, int(f * W + 1e-6) // S)
        gc = max(1, int(f * W + 1e-6) // self.cols)
        y = pad
        rects = {"local_title": pygame.Rect(xm, y, W, title_h)}
        y += title_h
        rects["local"] = pygame.Rect(xm + (W - lc * S) // 2, y, lc * S, lc * S)
        rects["segments"] = pygame.Rect(xt, y, TW, seg_h)
        y += lc * S + gap
        rects["group_title"] = pygame.Rect(xm, y, W, title_h)
        y += title_h
        rects["group"] = pygame.Rect(xm + (W - gc * self.cols) // 2, y, gc * self.cols, gc * self.rows)
        rects["clusters"] = pygame.Rect(xt, y, TW, cl_h)
        gy = y + cl_h + gap
        if gy + groups_h > self.content_h - pad:                    # group 옆에 안 들어가면 segments 아래 (local 옆 여유)
            gy = rects["segments"].bottom + gap
        rects["groups"] = pygame.Rect(xt, gy, TW, groups_h)
        bottom = max(y + gc * self.rows, rects["clusters"].bottom, rects["groups"].bottom)
        self._ins_geom = {"key": key, "lc": lc, "gc": gc, "f": f, "rects": rects, "W": W, "TW": TW, "x0": xm,
                          "pad": pad, "lh": lh, "bottom": bottom}
        return self._ins_geom

    def draw_obs_panel(self):
        """inspector (선택 UAV 상세) — 고정 레이아웃 (_inspector_geometry): local view, group map, Cluster info, GROUPS."""
        surf = self.screen
        pygame.draw.rect(surf, UI["obs"], (self.obs_x, 0, self.obs_w, self.content_h))
        pygame.draw.line(surf, UI["card_border"], (self.obs_x, 0), (self.obs_x, self.content_h))
        geo = self._inspector_geometry()
        R, W = geo["rects"], geo["W"]
        lc, gc = geo["lc"], geo["gc"]
        self.ins_cell_px = (lc, gc)
        self.ins_layout = {k: R[k].copy() for k in ("local", "segments", "group", "clusters", "groups")}
        self.use_personal = Usage()
        i = self.selected
        if i is not None and self.sim.agents[i].local_window is not None:
            a = self.sim.agents[i]
            S = self.sim.knowledge.local_size
            half = S // 2
            limg = self._render_personal(i, a.pos[0] - half, a.pos[1] - half, S, S, lc, True)
            gimg = self._render_personal(i, 0, 0, self.rows, self.cols, gc, False)
            surf.blit(self.f_ins_title.render(f"Local view ({S} x {S}, UAV-centered, north up)", True, UI["text"]),
                      R["local_title"].topleft)
            self._blit_map(limg, R["local"])
            kt = self.sim.knowledge
            members = sorted([i] + [j for j in range(self.sim.n_agents) if kt.reach[i, j]])
            surf.blit(self.f_ins_title.render("Group map (group: " + ", ".join(f"R{j}" for j in members) + ")",
                                              True, UI["text"]), R["group_title"].topleft)
            self._blit_map(gimg, R["group"])
            self.info_swatches = []
            self._draw_segments_block(R["segments"], i, geo["lh"])
            self._draw_clusters_block(R["clusters"], i, geo["lh"])
        else:
            msg = "Select a UAV (click the map or a card, Tab, 1-9)" if i is None else "No data yet (step once)"
            surf.blit(self.f_ins.render(msg, True, UI["subtext"]), R["local_title"].topleft)
            for k in ("local", "group"):                              # 지도 자리 (같은 위치 · 크기)
                pygame.draw.rect(surf, UI["frame"], R[k], width=1)
        self._draw_groups_section(R["groups"].x, R["groups"].y, geo["TW"])

    def _swatch(self, x, y, lh, color=None, key=None, text=None):
        """Cluster info 줄 앞 색 사각형 (group map / local view 에 칠한 것과 같은 색)."""
        sz = lh - 6
        rect = pygame.Rect(x, y + 3, sz, sz)
        draw_cell(self.screen, key or "cluster", rect, color, text=text, fnt=self._num_font(sz))
        pygame.draw.rect(self.screen, UI["frame"], rect, 1)
        self.info_swatches.append((key or "cluster", color, rect))           # 점검용 (색 일치 확인)
        return x + sz + 8

    @staticmethod
    def _cells(n):
        return f"{n} cell" + ("" if n == 1 else "s")

    def _cluster_rows(self, i):
        """선택 UAV 그룹의 덩어리 정보 [(blob, 담당자 목록)] 크기 내림차순 + noise frontier 셀 수."""
        sim = self.sim
        g = sim.comp[i]
        blobs = sorted(sim.blobs[g], key=lambda b: (-b.size, b.id))
        rows = [(b, sim.assignees(g, b.id)) for b in blobs]
        in_blob = set()
        for b in blobs:
            in_blob |= b.cell_set()
        front = sim.knowledge.frontier[i]
        noise = int(front.sum()) - sum(1 for c in in_blob if front[c])
        return rows, max(0, noise)

    def _draw_segments_block(self, rect, i, lh):
        """local view 바로 아래: My cluster segments (k-medoids, local view colors) — 내 덩어리 구간마다
        [담당 UAV 색] R<id> · <셀> cells (내 구간 굵게) + Goal 줄. EXPLORE 가 아니면 'No explore cluster (role: ...)'."""
        surf, sim = self.screen, self.sim
        a = sim.agents[i]
        x, y = rect.x, rect.y
        surf.blit(self.f_ins_bold.render("My cluster segments", True, UI["text"]), (x, y))
        y += lh
        surf.blit(self.f_ins.render("(k-medoids, local view colors)", True, UI["subtext"]), (x, y))
        y += lh
        if a.mode is Mode.EXPLORE and a.blob is not None:
            for j in sim.assignees(sim.comp[i], a.blob):
                b = sim.agents[j]
                key = "my_segment" if j == i else "mate_segment"
                tx = self._swatch(x, y, lh, b.color, key, None if j == i else str(j))
                f = self.f_ins_bold if j == i else self.f_ins
                surf.blit(f.render(f"R{j} \u00b7 {self._cells(len(b.cluster) if b.cluster else 0)}", True, UI["text"]),
                          (tx, y))
                y += lh
        else:
            surf.blit(self.f_ins.render(f"No explore cluster (role: {a.mode.name})", True, UI["subtext"]), (x, y))
        self._draw_goal_line(x, rect.bottom - 2 * lh, lh, i)                 # Goal 2 줄 = 블록 맨 아래 (고정 위치)

    def _draw_clusters_block(self, rect, i, lh):
        """group map 바로 아래: Group clusters (DBSCAN) — 덩어리마다 [색] C<id> · <셀> cells · UAVs: ... (m=...),
        내 덩어리 굵게 + 테두리, 크기 내림차순 상위 cluster_info_max_rows 개 + '... (+k more)', noise 줄."""
        surf, sim = self.screen, self.sim
        a = sim.agents[i]
        x, y, W = rect.x, rect.y, rect.w
        surf.blit(self.f_ins_bold.render("Group clusters (DBSCAN)", True, UI["text"]), (x, y))
        y += lh
        rows, noise = self._cluster_rows(i)
        my_blob = a.blob
        for b, P in rows[:self.cluster_rows]:
            mine = b.id == my_blob
            f = self.f_ins_bold if mine else self.f_ins
            tx = self._swatch(x, y, lh, cluster_color(b.id))
            uavs = ",".join(f"R{j}" for j in P) if P else "no UAV"
            txt = f"C{b.id} \u00b7 {self._cells(b.size)} \u00b7 {uavs}" + (f" (m={len(P)})" if P else "")
            surf.blit(f.render(txt, True, UI["text"]), (tx, y))
            if mine:
                pygame.draw.rect(surf, UI["text"], (x - 3, y, W + 3, lh), 2)
            y += lh
        if len(rows) > self.cluster_rows:
            surf.blit(self.f_ins.render(f"\u2026 (+{len(rows) - self.cluster_rows} more)", True, UI["subtext"]), (x, y))
            y += lh
        if noise:
            tx = self._swatch(x, y, lh, key="frontier")
            surf.blit(self.f_ins.render(f"noise \u00b7 {self._cells(noise)}", True, UI["text"]), (tx, y))

    def _draw_goal_line(self, x, y, lh, i):
        """Goal 2 줄 (설명 열 폭을 줄이려고 나눔).
        EXPLORE: 'Goal: R<id> → (x, y) · BFS <n> steps' / '★ goal point  /  ◆ segment medoid' (지도와 같은 아이콘)
        RESCUE : 'Rescue target T<id> (x, y)' / 'BFS <n> steps · deadline left <s>'
        RETURN : 'Return to depot' / 'BFS <n> steps'"""
        surf, sim = self.screen, self.sim
        a = sim.agents[i]
        bfs = lambda cell: (f"BFS {sim._dist(i, cell)} steps" if sim._dist(i, cell) < 10 ** 9 else "BFS unreachable")
        line = lambda txt, yy, col=UI["text"]: surf.blit(self.f_ins.render(txt, True, col), (x, yy))
        if a.mode is Mode.RESCUE and a.assigned_target is not None:
            t = sim.targets[a.assigned_target]
            left = t.remaining(sim.time)
            line(f"Rescue target T{a.assigned_target} {sim.frame.to_xy(t.pos)}", y)
            line(f"{bfs(t.pos)} \u00b7 deadline left {left if left is not None else '--'}", y + lh)
            return
        if a.mode is Mode.RETURN:
            line("Return to depot", y)
            line(bfs(sim.depot), y + lh)
            return
        if a.goal is None:
            line(f"Goal: R{i} -- (no goal)", y, UI["subtext"])
            return
        line(f"Goal: R{i} \u2192 {sim.frame.to_xy(a.goal)} \u00b7 {bfs(a.goal)}", y)
        y += lh
        r = (lh - 6) / 2
        draw_marker(surf, "goal", (x + r, y + lh / 2), r + 1, a.color)             # 지도와 같은 ★ (내 색)
        cx = x + 2 * r + 6
        lab = self.f_ins.render("goal point  /", True, UI["text"])
        surf.blit(lab, (cx, y))
        cx += lab.get_width() + 6
        sq = pygame.Rect(int(cx), int(y + 3), int(2 * r), int(2 * r))
        draw_cell(surf, "my_segment", sq, a.color)                                 # 지도와 같은 ◆ (구간 위 테두리)
        draw_marker(surf, "medoid", sq.center, max(3, r * 0.6))
        surf.blit(self.f_ins.render("segment medoid", True, UI["text"]), (sq.right + 6, y))

    def _blit_map(self, img, rect):
        self.screen.blit(img, rect.topleft)
        pygame.draw.rect(self.screen, UI["frame"], rect, width=1)

    def _num_font(self, cell_px):
        size = max(10, min(26, int(cell_px * 0.8)))
        if size not in self._num_fonts:
            self._num_fonts[size] = style_font(size, bold=True)
        return self._num_fonts[size]

    def _render_personal(self, i, r0, c0, h, w, cell_px, local):
        """UAV i 지식으로 (r0, c0) 부터 h x w 칸을 그린 이미지 + 오버레이. local = 맵 밖은 장애물로.
        마커 / 선 크기는 셀 크기에 비례 (작은 셀에서도 최소 크기 보장)."""
        sim, u = self.sim, self.use_personal
        img = pygame.Surface((w * cell_px, h * cell_px))
        seg_cells, meds = self._segments_view(i) if local else ({}, [])   # 구간 색 / medoid = local view 만
        blob_of, blob_lab, my_blob = {}, [], None                          # group map = DBSCAN 덩어리 색
        if not local:
            g = sim.comp[i]
            my_blob = sim.agents[i].blob
            for b in sim.blobs[g]:
                for c in b.cell_set():
                    blob_of[c] = b.id
                blob_lab.append(b)
        to_px = lambda cell: ((cell[1] - c0) * cell_px + cell_px / 2, (cell[0] - r0) * cell_px + cell_px / 2)
        inside = lambda cell: r0 <= cell[0] < r0 + h and c0 <= cell[1] < c0 + w
        nf = self._num_font(cell_px)
        small = nf if cell_px >= 10 else None
        sz = lambda f, lo: max(lo, cell_px * f)                     # 셀 비례 크기, 최소 lo px
        for rr in range(h):
            for cc in range(w):
                r, c = r0 + rr, c0 + cc
                rect = pygame.Rect(cc * cell_px, rr * cell_px, cell_px, cell_px)
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    draw_cell(img, "out_of_map", rect, usage=u)              # 맵 밖 (local 창이 경계에 걸칠 때)
                    continue
                kind = self._cell_class(i, r, c)
                if (r, c) in seg_cells and kind == "frontier":
                    key, col, lab = seg_cells[(r, c)]
                    draw_cell(img, key, rect, col, u, lab, small)
                elif (r, c) in blob_of and kind == "frontier":
                    draw_cell(img, "cluster", rect, cluster_color(blob_of[(r, c)]), u)
                else:
                    draw_cell(img, kind, rect, usage=u)
        if blob_lab:
            cell_rect = lambda cell: pygame.Rect((cell[1] - c0) * cell_px, (cell[0] - r0) * cell_px, cell_px, cell_px)
            mine = next((b for b in blob_lab if b.id == my_blob), None)
            if mine is not None:                                     # 선택 UAV 의 덩어리 = 굵은 테두리
                draw_edges(img, "cluster_selected", mine.cell_set(), cell_rect, max(2, cell_px // 3), u)
            lf = self._num_font(max(9, int(cell_px * 1.6)))
            for b in blob_lab:                                       # 덩어리 id 라벨 (대표 위치 = medoid)
                lab = lf.render(f"C{b.id}", True, UI["text"])
                cx, cy = to_px(b.medoid)
                box = lab.get_rect(center=(int(cx), int(cy - cell_px * 1.2)))
                img.blit(lab, box)
        if inside(sim.depot):
            draw_marker(img, "depot", to_px(sim.depot), sz(0.55, 4), usage=u, text="D" if cell_px >= 10 else None, fnt=nf)
        for m in meds:                                               # ◆ = 셀의 60%, 테두리만 (1 칸 구간 색이 보이게)
            if inside(m):
                draw_marker(img, "medoid", to_px(m), max(3, cell_px * 0.3), usage=u)
        a = sim.agents[i]
        kt = sim.knowledge
        for ti, t in enumerate(sim.targets):                         # 내 (그룹) 지식의 생존자 — 구조됨을 아직 모르면 계속 표시
            if not t.discovered or t.expired_at is not None or not kt.known_free[i][t.pos] or not inside(t.pos) \
                    or sim.beliefs.is_done(i, sim, ti):
                continue
            draw_marker(img, "survivor", to_px(t.pos), sz(0.55, 3), usage=u)
            if ti == a.assigned_target:
                draw_marker(img, "survivor_mine", to_px(t.pos), sz(0.8, 5), a.color, u)
        self._draw_path_on(img, i, to_px, max(2, cell_px // 5), u, inside, dash=max(4, int(cell_px * 0.6)))
        group = [i] + [j for j in range(sim.n_agents) if kt.reach[i, j]]
        for j in group:
            b = sim.agents[j]
            if inside(b.pos):
                draw_marker(img, "uav", to_px(b.pos), sz(0.6, 4), b.color, u,
                            self._uav_number(j) if cell_px >= 9 else None, nf, UI["badge_text"])
                if j == i:
                    draw_marker(img, "uav_selected", to_px(b.pos), sz(0.78, 5), usage=u)
        if a.goal is not None and inside(a.goal) and a.mode is Mode.EXPLORE:
            draw_marker(img, "goal", to_px(a.goal), sz(0.75, 4), a.color, u)
        return img

    # ==================== role 모드 (env/role/simulation.py) ====================

    def _log_events(self, info):
        """G-8: Role 이벤트 번호 / Explore 전용 이벤트 (쪼개짐 · 합쳐짐) / 덩어리 대응."""
        sim = self.sim
        names = {Mode.EXPLORE: "EXPLORE", Mode.RESCUE: "RESCUE", Mode.RETURN: "RETURN"}
        for i, ev in enumerate(sim.last_events):
            if not ev:
                continue
            a = sim.agents[i]
            goal = f"T{a.assigned_target}" if a.mode is Mode.RESCUE and a.assigned_target is not None else \
                (f"C{a.blob}" if a.mode is Mode.EXPLORE and a.blob is not None else "--")
            self.event_log.append(f"t{sim.time - 1:>3} R{i} [{','.join(sorted(ev))}] -> {names[a.mode]} {goal}")
        for line in getattr(sim, "step_log", []):
            self.event_log.append(line.replace("->", "→"))
        if info.get("redecide"):
            self.event_log.append(f"t{sim.time - 1:>3} escalation x{info['redecide']}")
        self.event_log = self.event_log[-40:]

    def _draw_groups_section(self, x, y, w, dry=False, groups=None):
        """GROUPS: 제목 + 통신 그룹 [R0 R1 ...] (선택 UAV 의 그룹 강조, 그룹 지름 > n_hop 이면 경고색). 반환 = 높이."""
        surf, f = getattr(self, "screen", None), self.f_ins
        y0 = y
        if not dry:
            surf.blit(self.f_ins_title.render(f"GROUPS  (n_hop = {self.sim.knowledge.n_hop})", True, UI["text"]), (x, y))
        y += self.f_ins_title.get_linesize() + 4
        gx, line = x, f.get_linesize() + 4
        n_hop = self.sim.knowledge.n_hop
        for members, diameter in (groups if groups is not None else self.comm_groups()):
            label = "[" + " ".join(f"R{i}" for i in members) + "]" + (f" {diameter} hop" if diameter > n_hop else "")
            tw = f.size(label)[0]
            if gx + tw > x + w and gx > x:
                gx, y = x, y + line
            if not dry:
                col = UI["accent"] if self.selected in members else (UI["warn"] if diameter > n_hop else UI["text"])
                surf.blit(f.render(label, True, col), (gx, y))
            gx += tw + 14
        return y + line - y0

    # ==================== overlay ====================

    def draw_overlay(self):
        surf = self.screen
        o = pygame.Rect(self.grid_width // 6, self.screen_height // 8,
                        max(self.grid_area_w, 640) * 2 // 3, self.screen_height * 3 // 4)
        shade = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        shade.fill(UI["shade"])
        surf.blit(shade, (0, 0))
        pygame.draw.rect(surf, UI["overlay_bg"], o, border_radius=8)
        pygame.draw.rect(surf, UI["text"], o, width=2, border_radius=8)
        title = "Load map  (Esc)" if self.overlay == "load" else "Load checkpoint from marl_models/  (Esc)"
        surf.blit(self.f.render(title, True, UI["text"]), (o.x + 16, o.y + 14))
        self.overlay_rects = []
        iy = o.y + 48
        if not self.overlay_items:
            surf.blit(self.f_sm.render("(none)", True, UI["subtext"]), (o.x + 16, iy))
            return
        for name, path in self.overlay_items:
            ir = pygame.Rect(o.x + 8, iy, o.w - 16, 28)
            if ir.bottom > o.bottom - 8:
                break
            if ir.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(surf, UI["overlay_hover"], ir, border_radius=4)
            surf.blit(self.f_sm.render(name[-90:], True, UI["text"]), (ir.x + 8, ir.y + 5))
            self.overlay_rects.append((ir, path))
            iy += 29

    def save_alloc_plots(self):
        """지금까지 기록된 에피소드로 분석 그래프 저장 -> 폴더를 OS 뷰어로 연다.
        생성은 별도 프로세스 (GUI 는 계속 동작). 끝나면 poll_alloc_plots() 가 상태 메시지 / 폴더 열기를 처리."""
        if self._plot_proc is not None and self._plot_proc.is_alive():
            self._set_status("alloc plots: still generating ...")
            return
        if self.sim.time == 0:
            self._set_status("alloc plots: nothing recorded yet (step first)")
            return
        data = self.sim.recorder.snapshot(self.sim)          # 사본: 생성 중에 시뮬레이션이 진행돼도 안전
        ctx = multiprocessing.get_context("spawn")           # pygame / SDL 상태를 물려받지 않도록 spawn
        self._plot_q = ctx.Queue()
        self._plot_proc = ctx.Process(target=_plot_worker, args=(data, self.figures_dir, self._plot_q), daemon=True)
        self._plot_proc.start()
        self._set_status(f"alloc plots: generating (t={self.sim.time}) ...", frames=10 ** 6)

    def poll_alloc_plots(self):
        """그래프 생성 결과 확인 (매 프레임). 실패해도 GUI 는 계속 동작."""
        if self._plot_q is None:
            return None
        try:
            kind, msg = self._plot_q.get_nowait()
        except queue.Empty:
            if self._plot_proc is not None and not self._plot_proc.is_alive() and self._plot_proc.exitcode:
                self._set_status(f"alloc plots failed (exit {self._plot_proc.exitcode})", frames=600)
                self._plot_q = None
            return None
        self._plot_q = None
        self._plot_proc.join(timeout=1)
        if kind == "ok":
            self._set_status(f"alloc plots saved: {os.path.relpath(msg, REPO_ROOT)}", frames=600)
            _open_folder(msg)
        else:
            self._set_status(f"alloc plots failed: {msg}", frames=600)
        return kind, msg

    def toggle_path(self):
        self.show_path = not self.show_path
        self.buttons["Path"].on = self.show_path
        self._set_status(f"path {'on' if self.show_path else 'off'}")

    def comm_groups(self):
        """raw 통신 인접 행렬의 연결 요소 = 멀티홉으로 이어진 그룹.
        반환: [(robot indices, 그룹 내 최대 hop 수), ...]  (가장 작은 로봇 번호 순)"""
        adj = self.sim.knowledge.adjacency            # 지금 링크 (hysteresis 포함)
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
        if b["Path"].clicked(ev): self.toggle_path(); return
        if b["Alloc Plots"].clicked(ev): self.save_alloc_plots(); return
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
            # 전체 맵에서 UAV 를 클릭해도 선택 / 해제 (같은 칸에 여러 대면 다음 UAV 로 순환)
            cell = self.cell_at(ev.pos)
            here = [i for i, a in enumerate(self.sim.agents) if cell is not None and a.pos == tuple(cell)]
            if here:
                if self.selected in here:
                    k = here.index(self.selected)
                    self.selected = here[k + 1] if k + 1 < len(here) else None
                else:
                    self.selected = here[0]
                return

        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE: self.toggle_play(); return
            if ev.key in (pygame.K_n, pygame.K_RIGHT): self.step_once(); return
            if ev.key == pygame.K_r: self.reset_sim(); return
            if ev.key == pygame.K_m:                         # rules -> random -> model (불러왔으면) -> rules
                cycle = ["rules", "random"] + (["model"] if self.model_policy is not None else [])
                self.role_policy = cycle[(cycle.index(self.role_policy) + 1) % len(cycle)]
                self._set_status(f"policy: {self.role_policy}")
                return
            if pygame.K_1 <= ev.key <= pygame.K_9:
                idx = ev.key - pygame.K_1
                if idx < len(self.sim.agents):
                    self.selected = idx
                    self._set_status(f"selected R{idx}")
                return
            if ev.key in (pygame.K_p, pygame.K_b):         # P (B 도 같은 동작): BFS 경로 표시
                self.toggle_path(); return
            if ev.key == pygame.K_o:                         # 선택 UAV 관측 콘솔 출력
                self._print_windows(); return
            if ev.key == pygame.K_c:
                self.show_clusters = not self.show_clusters
                self._set_status(f"segments {'on' if self.show_clusters else 'off'}"); return
            if ev.key == pygame.K_h:
                self.show_legend = not self.show_legend
                self._set_status(f"legend {'on' if self.show_legend else 'off'}"); return
            if ev.key == pygame.K_l:
                self.show_links = not self.show_links
                self._set_status(f"assignment links {'on' if self.show_links else 'off'}"); return
            if ev.key == pygame.K_v:
                self.show_ranges = not self.show_ranges
                self._set_status(f"sensor / comm ranges {'on' if self.show_ranges else 'off'}"); return
            if ev.key == pygame.K_TAB:                       # UAV 순환 선택
                n = len(self.sim.agents)
                self.selected = 0 if self.selected is None else (self.selected + 1) % n
                self._set_status(f"selected R{self.selected}"); return
            if ev.key == pygame.K_ESCAPE:
                self.selected = None; return
            if ev.key == pygame.K_g:
                self.save_alloc_plots(); return

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
              f"capacity={a.capacity}/{a.capacity_max} assigned={a.assigned_target}")
        print("local_window", a.local_window.shape)
        ob = self.sim.rl_observation()["agents"][self.selected]
        for k in ("role_self", "role_group", "role_mask", "role_needed", "role_g_res", "role_g_exp",
                  "explore_cur", "explore_kind"):
            print(f"{k:14s}", ob[k])

    # ==================== main loop ====================

    def run(self):
        while self.running:
            for ev in pygame.event.get():
                self.handle_event(ev)

            if self.playing:
                self._frame += 1
                if self._frame % self.speeds[self.speed_idx] == 0:
                    self.step_once()

            self.poll_alloc_plots()
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
