"""탐색/구조 시뮬레이션 GUI 실행 진입점 (UAV, 고정 고도 2D 격자).

- 설정은 training/config/ 의 main 전용 yaml (env_main.yaml + role_main.yaml + explore_main.yaml + rescue_main.yaml,
  학습과 같은 순서로 합침) 만 읽는다. 학습 (python -m training.train) 은 *_main 이 아닌 원본 yaml 을 읽는다.
  GUI / 체크포인트 같은 main 전용 값은 env_main.yaml 의 main: 섹션.
- 결정 (계층 구조: Role 선택 -> Rescue / Explore 할당 / Return -> BFS 경로 추종, env/role/simulation.py) 은 기본적으로
  규칙 정책 (policies/role_rules.py) 이 구동하고, GUI의 Model 버튼으로 학습된 체크포인트를 불러오면 그 정책으로 굴러간다.
- main.checkpoint 에 경로를 주면 시작할 때 그 모델을 바로 불러온다. 환경 설정 (맵, UAV 수, cluster, events, 보상 ...) 은
  main 전용 yaml 이 우선이고, 관측 차원 / 네트워크 구조를 정하는 값이 체크포인트와 다르면 어떤 키가 다른지 알리고 멈춘다.
  UAV 수 / 센서·통신 범위는 학습 때와 달라도 된다 (모델 파라미터가 N 과 무관).

조작: Space=재생/정지, N 또는 →=한 스텝, 1~9=로봇 선택, O=관측 콘솔 출력, M=정책 토글 (rules -> random -> model),
      P 또는 B=BFS 경로 표시 토글, V=센서 / 통신 범위 표시 토글, C=덩어리 · 구간 색칠, L=할당 연결선, H=범례,
      G=임무 할당 분석 그래프 저장 (figures/)
"""
import os

from env.role.simulation import RoleSimulation
from gui.grid_map import GridMap
from evaluation.checkpoint_policy import CheckpointPolicy, resolve_checkpoint
from training.config_loader import load_config, reward_settings_line

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "training", "config")
# main 전용 yaml — 병합 순서는 학습 (training/config_loader.py DEFAULT_FILES) 과 같다
MAIN_FILES = [os.path.join(CONFIG_DIR, f) for f in ("env_main.yaml", "role_main.yaml", "explore_main.yaml", "rescue_main.yaml")]


def print_settings(cfg, env, main_cfg):
    """시작 시 실제로 쓰는 주요 설정."""
    agents = env.get("agents") or []
    if env.get("map_mode") == "random":
        map_desc = f"random {env.get('rows')} x {env.get('cols')}, obstacle_ratio {env.get('obstacle_ratio')}"
    else:
        map_desc = f"file {env.get('map_path')}"
    cl = env.get("cluster") or {}
    print(f"[main] config: {', '.join(os.path.relpath(p, ROOT) for p in MAIN_FILES)}")
    print(f"[main] UAV {len(agents)} 대 {agents[:1]}{' ...' if len(agents) > 1 else ''}, n_hop {env.get('n_hop')}, "
          f"capacity {env.get('capacity')}")
    print(f"[main] map: {map_desc}, depot {env.get('depot')}, targets {env.get('num_targets')} "
          f"(golden {env.get('target_golden_time')}, {env.get('deadline_mode')}), max_episode_len {env.get('max_episode_len')}")
    print(f"[main] cluster: eps {cl.get('eps')}, min_samples {cl.get('min_samples')}, k_max {cl.get('k_max')}")
    print(f"[main] events: {env.get('events')}")
    print(reward_settings_line(cfg, "main"))
    print(f"[main] policy_mode: {env.get('policy_mode')}, explore_sub_mode: {env.get('explore_sub_mode')}")
    print(f"[main] checkpoint: {main_cfg.get('checkpoint') or 'None (rules)'}  (model_mode {main_cfg.get('model_mode')})")


def main():
    cfg = load_config(MAIN_FILES)
    env = cfg["env_config"]
    main_cfg = cfg.get("main") or {}
    model_config = (cfg.get("model") or {}).get("custom_model_config")
    print_settings(cfg, env, main_cfg)

    model_mode = main_cfg.get("model_mode", "greedy")
    policy = None
    if main_cfg.get("checkpoint"):                           # 체크포인트 -> 학습 모델로 시작 (null 이면 규칙 정책)
        ckpt = resolve_checkpoint(os.path.join(ROOT, main_cfg["checkpoint"]))
        try:
            policy = CheckpointPolicy(ckpt, env_config=env, mode=model_mode, model_config=model_config, env_priority=True)
            print(f"[main] model loaded: {ckpt}  ({policy.describe()})")
        except RuntimeError as e:                            # 관측 차원 / 네트워크 구조가 다른 체크포인트
            raise SystemExit(f"[main] 모델 로드 실패\n{e}")
    GridMap(RoleSimulation(env), cell_size=int(main_cfg.get("cell_size", 21)), model_policy=policy, model_mode=model_mode,
            figures_dir=os.path.join(ROOT, main_cfg.get("figures_dir", "figures")),
            auto_save_plots=bool(main_cfg.get("auto_save_plots", False)),
            obs_panel_width=int(main_cfg.get("inspector_width", 560)), font_size=int(main_cfg.get("font_size", 17)),
            fps=int(main_cfg.get("fps", 60)), speed_index=int(main_cfg.get("speed_index", 2)),
            model_config=model_config).run()


if __name__ == "__main__":
    main()
