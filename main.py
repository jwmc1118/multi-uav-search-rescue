"""탐색/구조 시뮬레이션 GUI 실행 진입점 (UAV, 고정 고도 2D 격자).

- 아래 옵션을 코드에서 직접 바꿔 환경을 구성한다.
- map_path 에 maps/ 아래 JSON 을 주면 그 맵(장애물/타겟)을 불러오고, None 이면 rows x cols 빈 환경.
- 결정(통합 할당: task / explore / depot + primitive)은 기본적으로 env/stand_in_policy.py 의 임시 대체
  정책이 구동하고, GUI의 Model 버튼으로 학습된 체크포인트를 불러오면 그 정책으로 굴러간다.

조작: Space=재생/정지, N 또는 →=한 스텝, 1~9=로봇 선택, P=관측 콘솔 출력, M=정책 토글
"""
from env.simulation import RescueSimulation
from gui.grid_map import GridMap


def main():
    sim = RescueSimulation({
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
        "target_golden_time": [120, 220],   # 타겟 추가 시 이 범위에서 골든타임 랜덤 샘플
        "deadline_mode": "from_discovery",
        "max_targets": 16,         # T: 할당 태스크 슬롯 수 (행동 = T + 2: explore, depot)
        "lock_rescue": True,       # 태스크 선택 = 구조/만료/도달불가까지 유지
        "lock_return": True,       # depot 선택 = 도착까지 유지

        # ---- 로봇 구성 ----
        "agents": [
            {"sensor_range": 4, "comm_range": 10},
            {"sensor_range": 4, "comm_range": 10},
            {"sensor_range": 4, "comm_range": 10},
        ],
    })

    GridMap(sim, cell_size=22).run()


if __name__ == "__main__":
    main()
