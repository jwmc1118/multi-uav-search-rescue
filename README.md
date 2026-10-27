# Multi-UAV Search & Rescue RL

계층 구조 결정: Role 선택 (Explore / Rescue / Return) → Rescue 할당 (조난자) / Explore (frontier DBSCAN 덩어리 선택 →
같은 덩어리를 m 대가 맡으면 측지 k-medoids 구간 분할 → κ Hungarian 구간 배정) → BFS 경로 추종. Return 목표 = depot.
세 결정 (역할 선택 / 생존자 할당 / frontier 덩어리 선택) 은 각각 독립된 네트워크 (encoder · 이종 그래프 GNN 공유 없음,
attention 집계) 로 학습 (3 헤드 MAPPO) 하거나 규칙 정책으로 고를 수 있다 (yaml `policy_mode`). 정보 구조는 belief:
UAV 는 자기 n-hop 통신 그룹의 병합 지식으로 그룹 그래프를 만들고 GNN 은 그 위에서 로컬로 계산한다.
덩어리 · 구간 생성 (DBSCAN, k-medoids) 과 구간 배정 (Hungarian) 은 env 가 하므로 학습 / 규칙 어느 쪽이든 같다.

참고 논문: `paper/` (저작권 때문에 git 에는 안 올림). `docs/research_pipeline_summary.md`, `docs/Framework.png` 는 초기 3계층 설계 기록.
모든 명령은 **레포 루트에서** 실행한다 (패키지 import 기준점).

## 폴더 구조

```
main.py                         GUI 실행 (CHECKPOINT 로 학습 모델 보기, None 이면 규칙 정책)
env/                            시뮬레이터 — 학습 / 정책과 무관
├── core/                       공통 세계: base_sim (BaseSimulation: 맵 · 조난자 · 센싱 · 구조 · 재보충 · 마감 · critic 관측) ·
│                               agent (UAV) · targets · map_gen (랜덤 맵) · pathfinding (BFS) ·
│                               knowledge (센싱 · n-hop 지도 병합) · beliefs · modes · coords · step_context
└── role/                       simulation (RoleSimulation: 역할 마스크 (부분 귀환) · Role 이벤트 1~6 · 결정 순서 · 보상 · 관측 · 이벤트 로그) ·
                                explore (덩어리 대응 · 쪼개짐 / 합쳐짐 · 구간 rebalance · 2 단계 유지 UAV 매칭 · 노드 특징 / 간선) ·
                                graph_obs (이종 그래프 관측 사양: 노드 U / S / K, 간선 U–U / U–S / U–K / S–S / K–K) ·
                                connectivity (연결성 C = 연결된 쌍 비율 — rescue 보너스 목표 그래프, 실제 위치는 진단 로그) ·
                                clusters (frontier DBSCAN · 측지 k-medoids) · gym_env (RoleEnv)
policies/                       규칙 정책 (학습 없음)
└── role_rules.py               rule_role / rule_rescue (EDF) / rule_explore (덩어리) / rule_explore_sub (구간, κ Hungarian — env 2 단계 고정),
                                RuleDecider, RandomDecider
models/                         신경망 (순수 torch + RLlib 래퍼)
├── capam.py                    CapAM decoder (MHA glimpse + tanh clip pointer) — 생존자 / frontier 할당 네트워크가 각자 사용
├── common/                     attention · encoders (CNN / MLP) · value_norm · critic (CTDE critic, 헤드마다 하나)
└── role/                       gnn (HeteroGNN: 2 층 MLP encoder, 관계별 multi-head attention 집계 | mean_max, 선택적 간선 갱신) ·
                                net (RoleNet: 독립 네트워크 3 개 — 역할 선택 (MLP 분류) / 생존자 할당 (CapAM) / frontier
                                할당 (CapAM, 1 단계만), 순차 디코딩 · 결정마다 그래프 재구성, critic ×3) · rllib_model
training/                       학습
├── train.py                    학습 엔트리포인트
├── config/                     env.yaml (환경 + 학습 공통) · role.yaml · explore.yaml · rescue.yaml (정책별: 학습/규칙 · 이벤트 · 보상 weight · 네트워크)
├── config_loader.py            yaml 합치기 (기본 env + role + explore + rescue, 뒤 파일이 앞 파일 값을 덮어씀)
├── role/                       ppo (RolePPOTrainer: 헤드별 GAE · 마스크 loss) · action_dist
├── callbacks.py                에피소드 지표 (custom_metrics)
├── catalog.py                  RLlib 등록 (env / model / action dist)
└── reward_plot.py              학습 곡선 reward_curve.png
evaluation/                     평가 · 분석 (결과는 results/)
├── checkpoint_policy.py        체크포인트 로드 + 실행 (GUI / 평가 공용)
├── evaluate.py                 체크포인트 평가 (--compare: 학습 vs 규칙)
├── run_rules.py                규칙 정책 200 seed 기준선           -> results/rules/
├── upper_bound.py              이론적 상한 / 실패 원인 분해         -> results/upper_bound/
├── scale_eval.py               UAV 수 N 을 바꿔 평가                -> results/scale_eval/
└── alloc_recorder.py, alloc_plots.py   임무 할당 분석 그래프 (GUI G 키) -> figures/
gui/                            pygame 시각화 · 맵 편집기 (grid_map.py, style.py = 색 · 도형 · 범례 표, colors.py = UAV 색)
tests/                          점검 스크립트 (ray 클러스터 불필요)
└── role/                       test_role_env · test_explore · test_events · test_return_mask · test_gnn ·
                                test_rewards · test_decide_order · test_join_reselect · test_config ·
                                test_gui_style · test_role_rule_equiv · smoke_role_policy · smoke_role_gui(_model) · gui_check · print_obs_spec
maps/                           맵 JSON (map/ 장애물만, map_target/ 조난자 · depot 포함)
marl_models/                    학습 결과 (RolePPO/, best_model/)
results/, figures/              평가 · 분석 출력
```

의존 방향: `env` ← `policies` ← `models` ← `training` ← `evaluation` ← `gui` / `main.py`.
`RoleSimulation` 은 `env/core/base_sim.py` 의 `BaseSimulation` 을 상속한다.

## 자주 쓰는 명령

```bash
# GUI
python main.py

# 학습 — training/config/env.yaml (환경 + 학습 공통) + role.yaml / explore.yaml / rescue.yaml (정책별 설정 · 네트워크) 를 고친 뒤
python -m training.train                                  # 계층 구조 3 헤드 학습 (ablation 은 각 정책 yaml 의 policy_mode)
python -m training.train --smoke --local                  # 1 iteration 점검

# 규칙 정책 기준선 / 평가
python -m rule_evaluation.run_rules                            # 그리디 정책 200 seed
python -m rule_evaluation.evaluate <checkpoint> --compare                            # 학습 정책 vs 규칙
python -m rule_evaluation.upper_bound [--checkpoint <checkpoint>]
python -m scale_evaluation.scale_eval <checkpoint> --n 2,3,4,5,6,8

# 점검
python -m tests.role.test_role_env
python -m tests.role.test_explore                         # explore 2 단계 (구간 분할 / 추적 / rebalance)
python -m tests.role.test_events                          # 이벤트 체계 (Role 1~6, 쪼개짐 / 합쳐짐, 이벤트 5 간격)
python -m tests.role.test_return_mask                     # 부분 귀환 역할 마스크
python -m tests.role.test_gnn                             # 그래프 구성 (ρ / η / ν, ETA σ, 목표 중복) · attention 순열 불변 · 패딩 · 2-hop · 네트워크 독립 / gradient · 연결성 · 보상
python -m tests.role.test_rewards                         # 보상: rescue S - F / U_ε, explore 커버리지 · 옵션 return, 중복 추적, 역할 진동, EXPLORE 마스크, C_pair
python -m tests.role.test_config                          # yaml: 학습용 = *_main key 구조, 깊은 병합, 보상 키
python -m tests.role.test_gui_style                       # GUI 스타일 표 (하드코딩 색 없음, 범례 = 그린 요소)
python -m tests.role.print_obs_spec                       # 관측 명세 (그래프 노드 / 간선 특징 표)
python -m tests.role.smoke_role_policy
```

## 학습 결과 위치

```
marl_models/
├── RolePPO/                           ← 학습 결과. 새로 학습하면 덮어씀
│   ├── checkpoints/checkpoint_XXXXXX/ ← 10 iteration 마다 저장, 전부 유지 (train.py CHECKPOINT_FREQ)
│   ├── checkpoints/best/              ← 고정 seed 평가 (eval.every iteration 마다, 세 head argmax) 기준 best + best_info.json
│   ├── reward_curve.png               ← 보상·에피소드 길이·loss·entropy 곡선 (--plot-every, 기본 10 iteration)
│   ├── progress.csv / result.json / events.out.tfevents.* / params.json
└── best_model/
    └── <학습 시작 시각>_role/            ← 학습마다 새 폴더 (덮어쓰지 않음). 그 학습에서 지금까지 가장 좋은 모델 (checkpoints/best 복사)
        ├── checkpoint_XXXXXX/
        └── params.json / best_info.json (iteration, eval 구조율 · 길이) / reward_curve.png
```

- 체크포인트는 저장 즉시 디스크에 남으므로 학습이 중간에 멈춰도 사라지지 않는다. Ctrl+C 로 멈추면 그 시점도 한 번 더 저장한다.
- 이어서 학습: `python -m training.train --resume <체크포인트 폴더>`
  - `RolePPO/checkpoints/` 안의 것이면 폴더를 유지하고 이어서 쓴다 (그 뒤 iteration 기록은 곡선에서 잘라냄).
  - `best_model/...` 등 다른 곳의 것이면 `RolePPO/` 를 새로 만들어 그 지점부터 학습한다.
- `--resume` 없이 실행하면 `RolePPO/` 는 덮어쓰지만 `best_model/` 은 그대로 남는다.
- 모델 보기: `main.py` 의 `CHECKPOINT` 에 체크포인트 폴더 경로 (`RolePPO/checkpoints/...` 또는 `best_model/..._role/...`).
- best 기준: 고정 seed (run_rules 와 같은 seed 15 개) · 세 head argmax 평가의 구조율 최대, 같으면 에피소드 길이가 짧은 쪽
  (env.yaml `eval.every` / `eval.episodes`). 평가는 driver policy 로 따로 돌려 학습 rollout 에는 영향이 없다.
- `--smoke` 는 `marl_models/RolePPO_smoke/` 에 따로 쓰고 `best_model/` 은 만들지 않는다 (`checkpoints/best/` 는 저장).

## 설치 (새 컴퓨터, Windows + Anaconda 기준)

Python **3.8** 전용이다. gym 0.19 의 메타데이터 버그 때문에 pip / setuptools / wheel 을 먼저 낮춘다.

```powershell
conda create -n exploration_rescue python=3.8 -y
conda activate exploration_rescue
python -m pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40.0" setuptools_scm   # PowerShell 은 < 때문에 따옴표 필수
#   setuptools_scm: --no-build-isolation 일 때 gpustat(ray 의존성) 빌드에 필요

# torch: CPU 권장 (모델이 작아서 GPU 이득이 거의 없고, CUDA 빌드는 워커마다 메모리를 수 GB 씩 먹는다)
pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
#   GPU 를 쓰려면 (RTX 30 시리즈까지): torch==1.8.1+cu111 + yaml num_gpus: 1

pip install -r requirements.txt --no-build-isolation
python -m tests.role.smoke_role_policy                                        # 설치 확인
```

### Windows 에서 자주 나는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `Redis has started but no raylets have registered yet` (raylet.err 에 `node_manager_hostname contains invalid UTF-8`) | **PC 이름에 한글** → 설정 > 시스템 > 정보 > 이 PC 이름 바꾸기 (영문) 후 재부팅 |
| `FileNotFoundError: [WinError 2]` (ray `_autodetect_num_gpus`) | Win11 에 WMIC 없음 → `train.py` 가 GPU 수를 직접 넘기도록 이미 처리됨 |
| `module 'numpy' has no attribute 'object'` | numpy 1.24+ → `pip install numpy==1.23.5` |
| `WinError 1455 페이징 파일이 너무 작습니다` / `LOW MEMORY` | CUDA torch + 워커 많음 → CPU torch 로 바꾸거나 `num_workers` 줄이기 / 가상 메모리 늘리기 |
| `No module named 'training'` | 레포 루트에서 실행하거나 `python training/train.py` (스크립트가 루트를 경로에 추가함) |

### Linux 에서 나는 문제

`include_dashboard=False` 여도 Ray 1.3 의 raylet 은 노드마다 dashboard agent 를 띄운다. 아래 에러는 이 agent 가 죽었다 살아나기를 반복하는 것이라 **학습 자체는 진행되지만** 로그가 계속 쌓인다. 둘 다 `requirements.txt` 에 버전을 고정해 두었다.

| 증상 (`(raylet)` 로그, `The agent on node ... failed`) | 원인 / 해결 |
|---|---|
| `module 'aioredis' has no attribute 'create_redis_pool'` | aioredis 2.x → `pip install aioredis==1.3.1` |
| `socket.gaierror: [Errno -5] No address associated with hostname` (`prometheus_client/exposition.py`) | prometheus_client 0.14+ → `pip install prometheus_client==0.13.1` |
| `CryptographyDeprecationWarning: Python 3.8 is no longer supported` / `Not all Ray CLI dependencies were found` | 경고일 뿐, 무시 |

### 동작 확인된 버전 (2026-10, Linux, Python 3.8.20)

| 패키지 | 버전 | 패키지 | 버전 |
|---|---|---|---|
| ray | 1.3.0 | aiohttp | 3.7.4 |
| gym | 0.19.0 | aioredis | 1.3.1 |
| torch | 1.8.1+cu111 | prometheus_client | 0.13.1 |
| numpy | 1.23.5 | redis | 4.1.4 |
| protobuf | 3.20.3 | pygame | 2.6.1 |
| matplotlib | 3.7.5 | | |
| pip / setuptools / wheel | 24.0 / 65.5.0 / 0.38.4 | | |

`num_workers` 는 (CPU 코어 수 − 4) 정도, 워커 하나당 RAM 약 0.5 GB (CPU torch 기준).

# 환경 상세

## 맵 / 조난자 생성 (`env/core/map_gen.py`)

- `map_mode: random` 이면 **매 에피소드 장애물 맵을 새로 생성**: 랜덤 직사각형(건물/비행금지구역)을 `obstacle_ratio`
  (기본 20 ~ 30%) 까지 놓고 → depot 주변 비움 → depot 에서 flood fill(UAV 이동 규칙과 동일) 로 **갈 수 없는 빈 칸은 장애물로 메움**
  → 비율이 범위 안이면 채택, 아니면 재생성. 따라서 갇힌 영역 / 도달 불가 조난자가 생기지 않는다.
- `target_mode: random` 이면 매 에피소드 조난자 수(`num_targets`, 예: [10, 20]) 와 위치를 새로 뽑는다 (depot 에서 도달 가능한 칸만).
  골든타임은 항상 `target_golden_time` 범위에서 매 에피소드 새로.
- depot: `depot` (월드 x, y) 를 주면 고정, 없으면 맵마다 랜덤.
- 학습 중 맵 크기(`rows`, `cols`)는 고정 (critic state 크기). 테스트는 다른 크기 / UAV 수도 가능 (가중치 크기가 맵과 무관).
- GUI: **Change Map** = 새 랜덤 맵 + 조난자, **Targets** = 조난자만 다시, **Reset** = 같은 맵/조난자 재생.

## 좌표계

| 이름 | 축 | 원점 | 쓰는 곳 |
|---|---|---|---|
| **월드 (x, y)** — `env/core/coords.py` | x 오른쪽 +, y 위쪽 + | 맵 중앙 셀 (`row0 = rows//2, col0 = cols//2`) | 설정(`depot`), GUI, RL 관측 위치값 |
| **상대 (dx, dy)** | 월드와 같은 축 | UAV | 노드 특징 dx, dy / waypoint (÷ 창 반폭) |
| 내부 (row, col) | row 아래 +, col 오른쪽 + | 왼쪽 위 | numpy 인덱스, 맵 JSON |
