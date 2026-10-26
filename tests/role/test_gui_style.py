"""GUI 스타일 표 점검 — gui/style.py 에 등록 안 된 색 / 도형을 그리는 코드가 없는지.

    python -m tests.role.test_gui_style

1) 정적: gui/*.py (style.py / colors.py 제외) 에 RGB 숫자 tuple 이 없고, pygame.draw 의 색 인자는 UI[...] (화면 틀) 뿐.
   draw_cell / draw_marker / draw_path / legend 에 넘기는 요소 키는 전부 STYLE 에 있다. 범례 대상은 설명이 있다.
2) 런타임: 모든 토글 (경로 / 구간 C / 연결선 L / 범례 H) 을 켜고 UAV 를 선택해 한 프레임 그린 뒤,
   그려진 요소가 전부 STYLE 에 있고, 메인 맵 legend 바 (일반 기호) 가 맵 셀 영역 아래에 있음 (가리지 않음).
"""
import glob
import os
import re

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gui.style import STYLE, UI

RGB = re.compile(r"\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*(,\s*\d{1,3}\s*)?\)")
KEYED = re.compile(r"draw_(?:cell|marker|path)\(\s*[\w.]+\s*,\s*\"(\w+)\"")
PG_DRAW = re.compile(r"pygame\.draw\.\w+\(\s*[\w.]+\s*,\s*([^,]+?)\s*,")


def check_static():
    files = [f for f in glob.glob("gui/*.py") if os.path.basename(f) not in ("style.py", "colors.py", "__init__.py")]
    bad_rgb, bad_key, bad_draw = [], [], []
    for f in files:
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            code = line.split("#", 1)[0]
            if RGB.search(code):
                bad_rgb.append(f"{f}:{n}: {line.strip()}")
            for k in KEYED.findall(code):
                if k not in STYLE:
                    bad_key.append(f"{f}:{n}: {k}")
            for c in PG_DRAW.findall(code):
                if not (c.startswith("UI[") or c in ("color",)):
                    bad_draw.append(f"{f}:{n}: {c}")
    assert not bad_rgb, "RGB 직접 사용:\n" + "\n".join(bad_rgb)
    assert not bad_key, "STYLE 에 없는 요소:\n" + "\n".join(bad_key)
    assert not bad_draw, "pygame.draw 색이 UI 표가 아님:\n" + "\n".join(bad_draw)
    no_label = [k for k, s in STYLE.items() if s.legend and not s.label]
    assert not no_label, no_label
    print(f"[1] static: {len(files)} gui files — no RGB literals, all drawn keys registered ({len(STYLE)} styles, "
          f"{len(UI)} UI colors)")


def check_runtime():
    from env.core.modes import Mode
    from env.role.simulation import RoleSimulation
    from gui.grid_map import GridMap
    from training.config_loader import load_config
    cfg = load_config()["env_config"]
    cfg["seed"] = 2
    cfg["global_downsample"] = 1
    gm = GridMap(RoleSimulation(cfg), cell_size=14)
    for _ in range(40):
        gm.step_once()
    gm.show_path = gm.show_clusters = gm.show_links = gm.show_legend = True
    sel = next((i for i, a in enumerate(gm.sim.agents) if a.mode is Mode.EXPLORE and a.cluster), 0)
    gm.selected = sel
    gm.draw_grid(); gm.draw_info_panel(); gm.draw_obs_panel(); gm.draw_bottom_panel()
    u = gm.use_main                                       # 전체 맵에 그린 요소 / 메인 맵 legend 바 = 고정 일반 기호
    assert set(u.keys) <= set(STYLE), set(u.keys) - set(STYLE)
    assert all(k in STYLE and STYLE[k].legend for k in gm.MAIN_LEGEND)
    assert gm.legend_bar_rect.top >= gm.grid_height and gm.legend_bar_rect.bottom <= gm.content_h, "legend 바가 맵을 가림"
    print(f"[2] main map drawn {len(u.keys)} elements; legend bar {list(gm.MAIN_LEGEND)} below the map {gm.legend_bar_rect}")
    drawn = set(gm.use_personal.keys)
    assert drawn <= set(STYLE), drawn - set(STYLE)
    print(f"[2] inspector drawn {len(drawn)} elements (no legend in inspector)")

if __name__ == "__main__":
    check_static()
    check_runtime()
    print("test_gui_style: all ok")
