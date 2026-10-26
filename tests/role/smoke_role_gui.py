"""role 모드 GUI 헤드리스 렌더 점검 — 규칙 정책으로 몇 step 진행 후 한 프레임을 PNG 로 저장.

    python -m tests.role.smoke_role_gui [steps]        # tests/out/role_frame.png
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from env.role.simulation import RoleSimulation
from gui.grid_map import GridMap
from training.config_loader import load_config

OUT = "tests/out/role_frame.png"


def main(steps=40, seed=1):
    cfg = load_config()["env_config"]
    cfg["seed"] = seed
    gm = GridMap(RoleSimulation(cfg), cell_size=16)
    assert isinstance(gm.sim, RoleSimulation)
    gm.selected = 0
    for _ in range(steps):
        gm.step_once()
    gm.draw_grid(); gm.draw_info_panel(); gm.draw_obs_panel(); gm.draw_bottom_panel()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pygame.image.save(gm.screen, OUT)
    n_cl = sum(len(c) for c in gm.sim.group_clusters)
    print(f"t={gm.sim.time} clusters {n_cl}, roles {[a.mode.name for a in gm.sim.agents]}, log lines {len(gm.event_log)}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
