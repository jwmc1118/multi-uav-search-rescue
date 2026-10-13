"""탐색/구조 시뮬레이션 GUI 실행 진입점 (UAV, 고정 고도 2D 격자).

- 아래 옵션을 코드에서 직접 바꿔 환경을 구성한다.
- map_path 에 maps/ 아래 JSON 을 주면 그 맵(장애물/타겟)을 불러오고, None 이면 rows x cols 빈 환경.
- 결정(통합 할당: task / explore / depot + primitive)은 기본적으로 env/stand_in_policy.py 의 임시 대체
  정책이 구동하고, GUI의 Model 버튼으로 학습된 체크포인트를 불러오면 그 정책으로 굴러간다.
- 아래 CHECKPOINT 에 경로를 주면 시작할 때 그 모델을 바로 불러온다. 이때 관측/행동 크기를 정하는 값
  (MODEL_SHAPE_KEYS)은 체크포인트의 학습 config(params.json)로 자동으로 덮어쓴다.

조작: Space=재생/정지, N 또는 →=한 스텝, 1~9=로봇 선택, P=관측 콘솔 출력, M=정책 토글
"""
import json
import os

from env.simulation import RescueSimulation
from gui.grid_map import GridMap
from training.checkpoint_policy import CheckpointPolicy, find_params_json, resolve_checkpoint

# 불러올 학습 모델. None = stand-in 정책으로 시작 (GUI Model 버튼으로 나중에 불러와도 됨)
# checkpoint_XXXXXX 폴더나 그 안의 checkpoint-XX 파일 경로 모두 가능. 예)
#   "marl_models/MultiPPO/checkpoints/checkpoint_000200"
#   "marl_models/best_model/2026-10-13_14-30-00/checkpoint_000180"
CHECKPOINT = "marl_models/MultiPPO/checkpoints/checkpoint_000100"
# CHECKPOINT = "marl_models/best_model/2026-10-13_14-30-00/checkpoint_000180"
# CHECKPOINT = None

# 모델 입력/출력 크기를 정하는 env 설정 — 체크포인트를 쓰면 학습 config 값으로 맞춘다
MODEL_SHAPE_KEYS = ["rows", "cols", "agents", "max_targets", "state_size", "local_channels",
                    "global_downsample", "n_hop", "max_episode_len"]


def apply_training_env_config(options, ckpt):
    """체크포인트의 params.json 에서 MODEL_SHAPE_KEYS 를 가져와 options 를 덮어쓴다."""
    pj = find_params_json(ckpt)
    if pj is None:
        raise FileNotFoundError(f"params.json 을 찾지 못함: {ckpt}")
    with open(pj) as f:
        train_env = json.load(f).get("env_config", {})
    for k in MODEL_SHAPE_KEYS:
        if k in train_env and options.get(k) != train_env[k]:
            print(f"[main] {k}: {options.get(k)} -> {train_env[k]} (학습 config 에 맞춤)")
            options[k] = train_env[k]


def main():
    options = {
        # ---- 맵 ----
        "map_mode": "random",      # random: 랜덤 장애물 맵으로 시작 (GUI 'Change Map' 으로 계속 새로 생성)
                                   # file  : map_path 의 맵 (None 이면 빈 환경 — 직접 그리기)
        "map_path": None,          # None 이면 아래 rows/cols 로 빈 환경 생성
        "rows": 40,                # 맵 세로 칸 수 (map_path 없을 때 / random 맵)
        "cols": 60,                # 맵 가로 칸 수 (map_path 없을 때 / random 맵)
        "obstacle_ratio": [0.2, 0.3],   # random 맵 장애물 비율
        "target_mode": "random",   # 조난자 수/위치 랜덤 (depot 에서 도달 가능한 칸만)
        "num_targets": [10, 20],   # 조난자 수 범위

        # ---- 좌표계 (env/coords.py): x = 가로(오른쪽 +), y = 세로(위쪽 +), 원점 (0,0) = 맵 중앙 ----
        #   40 x 60 맵이면 x: -30 ~ 29, y: -19 ~ 20
        "depot": (-30, -19),       # depot = 모든 로봇의 출발/재보충 지점 (x, y). None 이면 맵 파일 값

        # ---- 관측 ----
        "state_size": 21,          # 고해상도 local 관측 창 크기 (로봇 중심, 홀수) — primitive 입력
        "local_channels": ["obstacle", "agents"],  # primitive 입력 채널 (+ frontier / explored / target)
        "global_downsample": 1,    # 저해상도 global 맵 축소 배율 (GUI 표시 / critic state)
        "n_hop": 3,                # 통신 hop 수 (지식 병합 / 이웃 context / 순차 디코딩 그룹)

        # ---- 에피소드 / 구호품 / 골든타임 ----
        "max_episode_len": 400,
        "capacity": 3,             # 로봇 1대의 구호품 용량
        "target_golden_time": [100, 300],   # 타겟 추가 시 이 범위에서 골든타임 랜덤 샘플
        "deadline_mode": "from_discovery",
        "max_targets": 16,         # T: 할당 태스크 슬롯 수 (행동 = T + 2: explore, depot)
        "lock_rescue": True,       # 태스크 선택 = 구조/만료/도달불가까지 유지
        "lock_return": True,       # depot 선택 = 도착까지 유지

        # ---- 로봇 구성 ----
        "agents": [
            {"sensor_range": 4, "comm_range": 8},
            {"sensor_range": 4, "comm_range": 8},
            {"sensor_range": 4, "comm_range": 8},
            {"sensor_range": 4, "comm_range": 8},
        ],
    }

    policy = None
    if CHECKPOINT:
        ckpt = resolve_checkpoint(CHECKPOINT)
        apply_training_env_config(options, ckpt)
        policy = CheckpointPolicy(ckpt, env_config={"map_path": options.get("map_path")})
        print(f"[main] model loaded: {ckpt}")

    sim = RescueSimulation(options)
    GridMap(sim, cell_size=22, model_policy=policy).run()


if __name__ == "__main__":
    main()
