"""role 모드 GUI 점검 (헤드리스) -> results/gui_check/
  1_explore_R*_t*.png   EXPLORE UAV 선택 (같은 덩어리를 여럿이 나눠 맡은 장면)
  2_rescue_R*_t*.png    RESCUE UAV 선택
  small_window.png / large_window.png

    python -m tests.role.gui_check [seed]

점검: 2 열 (왼쪽 local / group, 오른쪽 각 지도 옆에 segments + Goal / clusters + GROUPS), 두 지도 같은 축척 (같은 폭),
      창 높이를 늘리지 않음,
      목록 내용이 바뀌어도 지도 위치 그대로, 목록 색 사각형 = group map 덩어리 색 / local view 구간 색 (화면 픽셀 비교),
      메인 맵 legend 바가 맵 셀 영역 아래.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from env.core.modes import Mode
from env.role.simulation import RoleSimulation
from gui.grid_map import GridMap
from gui.style import cluster_color, fill_color
from training.config_loader import load_config

OUT = "results/gui_check"


def frame(gm, name=None):
    gm.draw_grid(); gm.draw_info_panel(); gm.draw_obs_panel(); gm.draw_bottom_panel()
    if name:
        path = os.path.join(OUT, name)
        pygame.image.save(gm.screen, path)
        return path


def check_layout(gm, tag):
    """2 열 inspector: 요소가 패널 안, 왼쪽 열 local -> group, 오른쪽 열 segments -> clusters / GROUPS 안 겹침,
    설명 블록이 각 지도 옆 (위쪽 맞춤), 두 지도 같은 폭, 창 높이 = 메인 맵 + extra (늘리지 않음), legend 바가 맵 아래."""
    panel = pygame.Rect(gm.obs_x, 0, gm.obs_w, gm.content_h)
    L = gm.ins_layout
    for k in ("local", "segments", "group", "clusters", "groups"):
        assert panel.contains(L[k]), f"{tag}: {k} {L[k]} 가 패널 {panel} 밖 (잘림)"
    assert L["local"].bottom <= L["group"].top
    assert L["local"].right <= L["segments"].left and L["group"].right <= L["clusters"].left, "지도와 설명 열 겹침"
    assert L["segments"].top == L["local"].top and L["clusters"].top == L["group"].top, "설명 블록이 지도 옆이 아님"
    right = sorted((L[k] for k in ("segments", "clusters", "groups")), key=lambda r: r.top)
    for a, b in zip(right[:-1], right[1:]):
        assert a.bottom <= b.top, f"{tag}: 설명 열 {a} 와 {b} 겹침"
    assert abs(L["local"].w - L["group"].w) < max(gm.cols, gm.sim.knowledge.local_size) + 1, (L["local"], L["group"])
    assert gm.content_h == gm.grid_height + 180, f"{tag}: 창 높이가 늘어남"
    bar = gm.legend_bar_rect
    assert bar.top >= gm.grid_height and bar.bottom <= gm.content_h, f"{tag}: legend 바 {bar} 가 맵을 가림 / 잘림"
    return {k: tuple(v) for k, v in L.items()}


def check_colors(gm, i):
    """Cluster info 색 사각형 = 지도에 칠한 색."""
    sim, scr = gm.sim, gm.screen
    G = gm.ins_layout["group"]
    gc = gm.ins_cell_px[1]
    n = 0
    for key, color, rect in gm.info_swatches:
        px = tuple(scr.get_at((rect.x + 2, rect.y + 2)))[:3]
        if key == "cluster":
            assert px == tuple(color), (px, color)
            b = next(b for b in sim.blobs[sim.comp[i]] if cluster_color(b.id) == tuple(color))
            got = [tuple(scr.get_at((G.x + c[1] * gc + gc // 2, G.y + c[0] * gc + gc // 2)))[:3]
                   for c in sorted(b.cell_set()) if sim.knowledge.frontier[i][c]]
            hit = sum(p == tuple(color) for p in got)               # 라벨 / 마커 / 경로가 겹친 셀은 제외하고 대부분 같은 색
            assert hit >= max(1, len(got) // 2), f"group map C{b.id}: {hit}/{len(got)} cells match Cluster info {color}"
            n += 1
        elif key in ("my_segment", "mate_segment"):
            assert px == fill_color(key, color), (key, px)
            n += 1
    return n


def pick(sim, cond):
    return next((i for i, a in enumerate(sim.agents) if cond(i, a)), None)


def make(seed, cell=22, width=None):
    cfg = load_config()["env_config"]
    cfg["seed"] = seed
    cfg["global_downsample"] = 1
    return GridMap(RoleSimulation(cfg), cell_size=cell, obs_panel_width=width)


def main(seed=9):
    os.makedirs(OUT, exist_ok=True)
    gm = make(seed, 21)                                       # main.py GRID_CELL_SIZE
    sim = gm.sim
    ex = None
    for _ in range(120):
        gm.step_once()
        if sim.time >= 8:
            ex = pick(sim, lambda i, a: a.mode is Mode.EXPLORE and a.cluster and len(sim.assignees(sim.comp[i], a.blob)) >= 2)
            if ex is not None:
                break
    assert ex is not None
    gm.selected = ex
    print(frame(gm, f"1_explore_R{ex}_t{sim.time}.png"), f"colors checked {check_colors(gm, ex)}")
    lay = check_layout(gm, "explore")
    for _ in range(300):
        r = pick(sim, lambda i, a: a.mode is Mode.RESCUE and a.assigned_target is not None and not sim.events[i]
                 and not sim.beliefs.is_done(i, sim, a.assigned_target))
        if r is not None:
            break
        gm.step_once()
    gm.selected = r
    print(frame(gm, f"2_rescue_R{r}_t{sim.time}.png"), f"colors checked {check_colors(gm, r)}")
    assert check_layout(gm, "rescue") == lay, "목록 내용이 바뀌었는데 지도 위치가 달라짐"
    geo = gm._inspector_geometry()
    print(f"window {gm.screen_width}x{gm.screen_height}  inspector {gm.obs_w} px = map column {geo['W']} + text column "
          f"{geo['TW']}  f = {geo['f']:.3f}  local cell {geo['lc']} px -> {lay['local'][2]} px,  "
          f"group cell {geo['gc']} px -> {lay['group'][2]} x {lay['group'][3]} px,  bottom slack {gm.content_h - geo['bottom']} px")
    print(f"layout: local {lay['local']} segments {lay['segments']} group {lay['group']} clusters {lay['clusters']} groups {lay['groups']}")
    for name, cell, width in (("small_window.png", 14, 420), ("large_window.png", 26, 760)):
        g = make(seed, cell, width)
        for _ in range(20):
            g.step_once()
        g.selected = 0
        frame(g, name)
        check_layout(g, name)
        gg = g._inspector_geometry()
        print(f"{name}: window {g.screen_width}x{g.screen_height}  f {gg['f']:.3f}  cells local/group {gg['lc']}/{gg['gc']}")
    print("gui_check ok")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
