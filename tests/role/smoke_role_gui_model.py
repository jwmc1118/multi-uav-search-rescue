"""role 모드 GUI 에서 계층 구조 체크포인트 로드 + 실행 (헤드리스). Model 오버레이 클릭 경로 그대로.

    python -m tests.role.smoke_role_gui_model <checkpoint> [n_agents]     # results/gui_check/4_model_R*.png
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from env.role.simulation import RoleSimulation
from gui.grid_map import GridMap
from eval_common.checkpoint_policy import resolve_checkpoint
from training.config_loader import load_config


def main(ckpt, n_agents=5, steps=40):
    cfg = load_config()["env_config"]
    cfg["seed"] = 1
    cfg["global_downsample"] = 1
    cfg["agents"] = [dict(cfg["agents"][0]) for _ in range(n_agents)]
    gm = GridMap(RoleSimulation(cfg), cell_size=22)
    gm.overlay = "model"
    rect = pygame.Rect(0, 0, 10, 10)
    gm.overlay_rects = [(rect, resolve_checkpoint(ckpt))]
    gm._pick_overlay((1, 1))                                   # Model 버튼 -> 목록 클릭과 같은 경로
    print("status:", gm.status)
    assert gm.role_policy == "model" and gm.model_policy is not None
    for _ in range(steps):
        gm.step_once()
    gm.selected = 0
    gm.draw_grid(); gm.draw_info_panel(); gm.draw_obs_panel(); gm.draw_bottom_panel()
    os.makedirs("results/gui_check", exist_ok=True)
    out = "results/gui_check/4_model_R0.png"
    pygame.image.save(gm.screen, out)
    last = gm.model_policy.last[0]
    print(f"t={gm.sim.time} UAVs {gm.sim.n_agents}  R0 role probs {last['role_probs'].round(2)}  "
          f"rescue top {last['rescue_top']}  explore top {last['explore_top']}  V {last['value_role']:+.3f}")
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
