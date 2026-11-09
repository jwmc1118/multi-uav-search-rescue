"""GUI Model 창에서 Saved models -> 조합 폴더의 model_vXX 를 골라 실행 (헤드리스, 목록 클릭 경로 그대로).

    python -m tests.role.smoke_role_gui_model <조합 (예: learn_rule_rule)> <model_vXX> [n_agents]   # results/gui_check/4_model_R0.png
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from env.role.simulation import RoleSimulation
from gui.grid_map import GridMap
from training.config_loader import load_config


def main(combo, name, n_agents=5, steps=40):
    cfg = load_config()["env_config"]
    cfg["seed"] = 1
    cfg["global_downsample"] = 1
    cfg["agents"] = [dict(cfg["agents"][0]) for _ in range(n_agents)]
    gm = GridMap(RoleSimulation(cfg), cell_size=22)
    gm.open_overlay("model")                                   # Model 버튼 -> Saved models -> 조합 버튼 -> model_vXX 목록
    gm.sel.update(source="saved", combo=combo)
    gm._fill_model_items()
    entry = next(e for _, e in gm.overlay_items if e.name == name)
    gm.overlay_rects = [(pygame.Rect(0, 0, 10, 10), entry)]
    gm._pick_overlay((1, 1))                                   # 목록 클릭과 같은 경로
    print("status:", gm.status)
    assert gm.current is entry and gm.model_policy is not None, gm.status
    for _ in range(steps):
        gm.step_once()
    gm.selected = 0
    gm.draw_grid(); gm.draw_info_panel(); gm.draw_obs_panel(); gm.draw_bottom_panel()
    os.makedirs("results/gui_check", exist_ok=True)
    out = "results/gui_check/4_model_R0.png"
    pygame.image.save(gm.screen, out)
    print(f"t={gm.sim.time} UAVs {gm.sim.n_agents}  policy {gm.policy_text()}  components {gm.sim.components}")
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
