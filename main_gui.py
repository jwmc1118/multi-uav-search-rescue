"""탐색/구조 시뮬레이션 GUI 실행 진입점 (UAV, 고정 고도 2D 격자).

    python main_gui.py                                                            # gui config main: 의 시작 조합 (기본 Rule)
    python main_gui.py --source saved --combo learn_learn_rule --model model_v01   # Saved · best_model/learn_learn_rule/model_v01
    python main_gui.py --source saved --combo learn_rule_rule                      # Saved · 그 조합 폴더의 최신 버전
    python main_gui.py --source training --run learn_learn_rule_20261104_220048 --checkpoint best

- 설정은 gui_evaluation/config/ 의 GUI 전용 yaml (env_gui.yaml + role_gui.yaml + task_gui.yaml, 학습과 같은 순서로 합침) 만
  읽는다. GUI 화면 값과 시작 조합 (source / combo / model / run / checkpoint) 은 env_gui.yaml 의 main: 섹션 (CLI 가 우선).
- 결정 (계층 구조: Role 선택 -> Rescue / Explore 할당 / Return -> BFS 경로 추종, env/role/simulation.py) 은 Model 창에서 고른다
  (gui/model_library.py):
    Source  Saved models (marl_models/best_model) | Training (marl_models/ 아래 run 의 체크포인트)
    Combo   (Saved) rule_rule_rule (모델 없이 규칙 정책) | best_model 아래 조합 폴더 <role>_<rescue>_<explore> (각 rule |
            learn, 모델이 있는 폴더만) -> 그 안의 model_vXX 목록
    Run     (Training) marl_models/<조합>_<시각> run 목록 (조합 이름 표시) -> 체크포인트 (best, checkpoint_* 최신 순)
  조합 (role / rescue / explore: rule | learn) 은 모델 경로 하나로 정해진다 (eval_common/model_loader.py: meta.json /
  run_meta.json -> 폴더 이름 -> params.json). 환경 설정 (맵, UAV 수, cluster, events ...) 은 GUI 전용 yaml 이 우선이고, 관측 차원 / 네트워크
  구조가 지금 코드 · 설정과 맞지 않는 모델은 목록에 "(cannot load)", 통합 task 네트워크 시절 모델은 "(structure mismatch)". 정책 / 모델을 바꾸면 같은 맵 · 조난자로 다시 시작한다.

조작: Space=재생/정지, N 또는 →=한 스텝, 1~9=로봇 선택, O=관측 콘솔 출력, Model 버튼 = 모델 선택 창,
      M=지금 가능한 조합 순환 (rule_rule_rule -> best_model 의 조합 폴더 이름 순, 마지막으로 고른 버전),
      P 또는 B=BFS 경로 표시 토글, V=센서 / 통신 범위 표시 토글, C=덩어리 · 구간 색칠, L=할당 연결선, H=범례,
      G=임무 할당 분석 그래프 저장 (results/gui_evaluation/)
"""
import argparse
import os

from training.config_loader import GUI_FILES, load_config, reward_settings_line

ROOT = os.path.dirname(os.path.abspath(__file__))
START_KEYS = ("source", "combo", "model", "run", "checkpoint")   # 시작 조합 (env_gui.yaml main: · CLI)
# GUI 전용 yaml (gui_evaluation/config/) — 병합 순서는 학습 (training/config_loader.py DEFAULT_FILES) 과 같다


def print_settings(cfg, env, start, mode):
    """시작 시 실제로 쓰는 주요 설정."""
    agents = env.get("agents") or []
    if env.get("map_mode") == "random":
        map_desc = f"random {env.get('rows')} x {env.get('cols')}, obstacle_ratio {env.get('obstacle_ratio')}"
    else:
        map_desc = f"file {env.get('map_path')}"
    cl = env.get("cluster") or {}
    print(f"[main] config: {', '.join(os.path.relpath(p, ROOT) for p in GUI_FILES)}")
    print(f"[main] UAV {len(agents)} 대 {agents[:1]}{' ...' if len(agents) > 1 else ''}, n_hop {env.get('n_hop')}, "
          f"capacity {env.get('capacity')}")
    print(f"[main] map: {map_desc}, depot {env.get('depot')}, targets {env.get('num_targets')} "
          f"(golden {env.get('target_golden_time')}), max_episode_len {env.get('max_episode_len')}")
    print(f"[main] cluster: eps {cl.get('eps')}, min_samples {cl.get('min_samples')}, k_max {cl.get('k_max')}")
    print(f"[main] events: {env.get('events')}")
    print(reward_settings_line(cfg, "main"))
    print(f"[main] start {start} (model_mode {mode})")


def main():
    ap = argparse.ArgumentParser(description="rescue simulation GUI (Model window: Source -> combo / run)",
                                 epilog="examples:\n"
                                        "  python main_gui.py\n"
                                        "  python main_gui.py --source saved --combo learn_learn_rule --model model_v01\n"
                                        "  python main_gui.py --source training --run learn_learn_rule_20261104_220048 --checkpoint best",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("saved", "training"), default=None, help="시작 Source (main.source)")
    ap.add_argument("--combo", default=None,
                    help="Saved 조합 폴더 (best_model/<role>_<rescue>_<explore>, rule_rule_rule = 규칙 정책, main.combo)")
    ap.add_argument("--model", default=None, help="Saved 모델 버전 (예: model_v03, 없으면 그 폴더의 최신, main.model)")
    ap.add_argument("--run", default=None, help="Training run 이름 (marl_models/ 아래, main.run)")
    ap.add_argument("--checkpoint", default=None, help="Training 체크포인트 (best / checkpoint_000420, main.checkpoint)")
    args = ap.parse_args()

    from env.role.simulation import RoleSimulation
    from gui.grid_map import GridMap

    cfg = load_config(GUI_FILES)
    env = cfg["env_config"]
    main_cfg = cfg.get("main") or {}
    model_config = (cfg.get("model") or {}).get("custom_model_config")
    start = {k: getattr(args, k) if getattr(args, k) is not None else main_cfg.get(k) for k in START_KEYS}
    mode = main_cfg.get("model_mode", "greedy")
    print_settings(cfg, env, start, mode)

    GridMap(RoleSimulation(env), cell_size=int(main_cfg.get("cell_size", 21)), model_mode=mode, start=start,
            figures_dir=os.path.join(ROOT, main_cfg.get("figures_dir", os.path.join("results", "gui_evaluation"))),
            auto_save_plots=bool(main_cfg.get("auto_save_plots", False)),
            obs_panel_width=int(main_cfg.get("inspector_width", 560)), font_size=int(main_cfg.get("font_size", 17)),
            fps=int(main_cfg.get("fps", 60)), speed_index=int(main_cfg.get("speed_index", 2)),
            model_config=model_config).run()


if __name__ == "__main__":
    main()
