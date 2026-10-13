# Exploration_rescue — 통합 할당 기반 Multi-UAV Search & Rescue RL

Role policy 를 없애고 **rescue / explore / return 을 하나의 attention pointer 결정**으로 통합한 2계층 구조
(할당 → primitive). 참고 논문: `paper/` (CapAM, Role-selection, Two-stage — 저작권 때문에 git 에는 안 올림). `docs/research_pipeline_summary.md` 와
`docs/Framework.png` 는 **이전 3계층(role/task/primitive) 설계 기록**이다.

모든 명령은 **`Exploration_rescue/` 폴더에서** 실행한다 (패키지 import 기준점).

## 폴더 구조

| 폴더 | 역할 | 파일 |
|---|---|---|
| `env/` | **환경** (규칙·관측·보상). 학습 프레임워크와 무관 | `simulation.py` 한 tick 진행·관측 / `map_gen.py` 랜덤 맵·조난자 생성 / `allocation.py` 노드 집합·admissible·lock·순차 디코딩 규칙·충돌 해결 / `knowledge.py` 센싱·n-hop 지식 병합·local 크롭 / `pathfinding.py` BFS 거리장·경로·waypoint / `agent.py` UAV (8방향 + stay, 360° 센서) / `modes.py` 모드·메시지 타입 / `targets.py` / `step_context.py` tick 사건 묶음 / `rewards.py` 보상 / `gym_env.py` RLlib gym 래퍼 / `stand_in_policy.py` GUI 용 임시 정책 |
| `models/` | **네트워크** (논문 그림과 1:1) | `alloc_policy.py` Encoder(k-NN CCN + explore/depot) · Context · Decoder(MHA glimpse + pointer) · 순차 디코딩 / `primitive_policy.py` local ⊕ waypoint 채널 CNN / `critic.py` CTDE critic / `rescue_net.py` 전체 조립 (순수 torch) / `rllib_model.py` RLlib 래퍼 / `attention.py` `encoders.py` 공용 블록 |
| `training/` | **학습** (RLlib 1.3.0 PPO) | `multiagent_ppo.py` 헤드별 GAE + `L = L_alloc + L_prim` / `action_dist.py` 2-head 분포 / `sample_batch_keys.py` / `catalog.py` / `train.py` `evaluate.py` `checkpoint_policy.py` / `config/rescue.yaml` |
| `gui/` | pygame 시각화·맵 편집기 | `grid_map.py` |
| `tests/` | 학습 전 점검 | `smoke_test.py` `smoke_loss.py` `smoke_policy.py` |
| `maps/` | 맵 JSON | `map/` (장애물만) · `map_target/` (타겟·depot 포함) |

의존 방향: `env` ← `models` ← `training` ← `gui`/`tests`. 예외: `models/rllib_model.py` 는 컬럼 이름
(`training/sample_batch_keys.py`) 만 가져온다.

## 결정 구조

```
노드 슬롯 A = T + 2 :  [ 태스크 0 .. T-1 | explore (T) | depot (T+1) ]       H^i = [H^i_1; …; H^i_T; F^i_e; F^i_d]

[Encoder]  δ^i_j=[x/W, y/H, τ_j, d^i_j] ─W_node─▶ F ─k-NN(위치) CCN: (F_𝒦 − 1F)W_msg, [F; F̃]W_agg, Σ─▶ H^i_j
           δ^i_e=[T^i/N_T, E^i/|C_free|, d^i_e] ─W_exp─▶ F^i_e        δ^i_d=[x_d/W, y_d/H, d^i_d] ─W_depot─▶ F^i_d
[Context]  ξ^i=[x/W, y/H, c/c_max] ─W_self─▶ A_self ;  Σ_{n-hop 이웃} m^{i'}=[x, y, c, g_x, g_y, o(4)] ─W_nbr─▶ A_nbr
           z^i = [t/t_max, A_self, A_nbr] ─W_q─▶ Q^i
[Decoder]  𝐊=HW_k, 𝐕=HW_v → MHA(Q, 𝐊, 𝐕) = γ^i ;  P = HW_ptr ;  u = C·tanh((γW_glm)Pᵀ/√D) → Mask(𝒜^i) → softmax
[순차 디코딩]  dec_rank 순서로 한 명씩: 같은 n-hop 그룹에서 앞 순서가 고른 태스크 mask,
              이웃 메시지 (g, o) 는 앞 순서 = 이번 선택 / 뒤 순서 = 이전 선택
[Primitive] CNN( local (S,S,C_l) ⊕ waypoint one-hot 채널 ) ⊕ mode(3) ⊕ waypoint(3) [⊕ γ^i: primitive_glimpse] ─▶ 8방향 + stay
[Critic, 학습 전용] 헤드별 CentralCritic( CNN(State) ⊕ CNN(Local) ⊕ extra )
```

- **admissible 𝒜^i** (`env/allocation.py`): 태스크 = 용량>0 · 다른 UAV 에 lock 안 됨 · BFS 도달 가능 · `d^i_j ≤ 잔여 골든타임`
  (옵션 `mask_return_infeasible`). explore = 도달 가능한 frontier 있을 때. **depot 은 항상** → 𝒜^i ≠ ∅.
- **lock**: 태스크 선택 → 구조/만료/도달불가까지 유지 (`lock_rescue`), depot 선택 → 도착까지 유지 (`lock_return`).
  lock 중엔 강제 슬롯 하나만 admissible, 그 tick 의 alloc loss 는 `alloc_valid = 0` 으로 제외.
- **통신 제약 반영 (env/beliefs.py)**: 결정(후보·마스크·lock 유지)은 UAV 별 믿음만 쓴다. "다른 UAV 의 lock" 과
  "구조 완료" 정보는 n-hop 으로 연결될 때만 전달되고(구조 완료는 해당 칸을 직접 봐도 앎), 만료는 각자 계산.
  통신이 안 되는 UAV 끼리는 모른 채 같은 태스크를 lock 할 수 있고, **n-hop 연결되는 순간 BFS 거리가 가까운 쪽만 유지**,
  먼 쪽은 lock 해제 후 다음 tick 재결정(`ctx.contact_released`). 같은 통신 그룹 안 동시 중복은 즉시 해결(`ctx.double_selected`).
- **waypoint**: 선택 노드까지 BFS 경로가 local 창을 벗어나기 직전 셀 (목적지가 창 안이면 목적지).
  환경이 슬롯별로 `node_wp` 로 주고, 모델이 고른 슬롯 것을 2D 채널로 그린다 → 순환 의존 없음.
- BFS 는 에이전트당 tick 1회 (`bfs_field`), 균일 비용 8-연결이라 A* 최단거리와 같다.
- 네트워크 변경: 블록 = `models/<블록>.py`, 연결 = `models/rescue_net.py:forward`,
  크기 = `training/config/rescue.yaml` 의 `custom_model_config`.
- **관측을 바꾸려면 세 곳을 같이**: `env/simulation.py:rl_observation` / `env/gym_env.py` (space) /
  `models/rescue_net.py` (`AGENT_KEYS`, `TEAM_KEYS`). `tests/smoke_test.py` 가 space 적합성을 검사한다.

## 관측

| 키 | shape | 내용 |
|---|---|---|
| `local` | (S,S,C_l) | 자기 중심 크롭, 회전 없음 (북쪽 = 위). 맵 밖 = 장애물. 채널 = `local_channels` (기본 obstacle, agents) |
| `task_feats` / `task_present` / `task_ids` | (T,4) / (T,) / (T,) | δ^i_j, 실제 후보 여부, target index (−1 패딩) |
| `explore_feats` / `depot_feats` / `self_feats` | (3,) 각각 | δ^i_e, δ^i_d, ξ^i |
| `node_mask` / `node_xy` / `node_type` / `node_wp` | (A,) / (A,2) / (A,) / (A,3) | 𝒜^i, 슬롯별 목적지 좌표 (메시지 g), 메시지 타입 o, local waypoint |
| `free_choice` | (1,) | 1 = 정책이 고름, 0 = lock 강제 |
| 팀: `pub` `prev_msg` `reach` `dec_rank` `time` `state` | (N,3) (N,6) (N,N) (N,) (1,) (gh,gw,5) | 이웃 공개 상태 / 이전 결정 메시지 / n-hop 연결 / 디코딩 순서 / t/t_max / critic 전용 |

정규화: 위치 = 월드 좌표 / 맵 크기 (원점 = 맵 중앙, 대략 [−0.5, 0.5]), 시간·거리·잔여 골든타임 = / t_max, 용량 = / c_max.
grid 5채널 = obstacle / explored / frontier / target / agents(자기 1.0, 통신 이웃 0.5). primitive 는 이 중
`local_channels` 로 고른 채널만 받고(yaml / main.py 에서 이름 목록으로 넣고 뺌), waypoint 채널은 항상 붙는다.
critic state 는 5채널 전부.

## 맵 / 조난자 생성 (`env/map_gen.py`)

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
| **월드 (x, y)** — `env/coords.py` | x 오른쪽 +, y 위쪽 + | 맵 중앙 셀 (`row0 = rows//2, col0 = cols//2`) | 설정(`depot`), GUI, RL 관측 위치값 |
| **상대 (dx, dy)** | 월드와 같은 축 | UAV | `node_wp` (÷ 창 반폭) |
| 내부 (row, col) | row 아래 +, col 오른쪽 + | 왼쪽 위 | numpy 인덱스, 맵 JSON |

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
python -m tests.smoke_test                                             # 설치 확인
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

## 실행

```bash
python main.py                                         # GUI (stand-in 정책, Model 버튼으로 체크포인트)
python -m tests.smoke_test                             # 환경/BFS/모델/순차 디코딩/(ray 있으면) 분포
python -m tests.smoke_test --trainer                   # + MultiPPOTrainer 1 iteration
python -m tests.smoke_loss                             # GAE + loss + backward (같은 파라미터 KL = 0 확인)
python -m tests.smoke_policy                           # ray.init 없이 Policy 생성 + learn_on_batch
python -m training.train training/config/rescue.yaml -t 10        # 학습 (10M steps)
python training/train.py --smoke                                  # 직접 실행도 가능 (config 기본값), 1 iteration 점검
python -m training.evaluate <checkpoint> -n 20  [--render]
```

### 학습 결과 위치

```
marl_models/
├── MultiPPO/                          ← 새로 학습하면 덮어씀
│   ├── checkpoints/checkpoint_XXXXXX/ ← 10 iteration 마다 저장, 전부 유지 (train.py CHECKPOINT_FREQ)
│   ├── reward_curve.png               ← 보상·에피소드 길이·loss·entropy 곡선 (--plot-every, 기본 10 iteration)
│   ├── progress.csv / result.json / events.out.tfevents.* / params.json
└── best_model/
    └── <학습 시작 시각>/                ← 학습마다 새 폴더 (덮어쓰지 않음). 그 학습에서 지금까지 가장 좋은 모델
        ├── checkpoint_XXXXXX/
        └── params.json / best_info.json (iteration, reward) / reward_curve.png
```

- 체크포인트는 저장 즉시 디스크에 남으므로 학습이 중간에 멈춰도 사라지지 않는다. Ctrl+C 로 멈추면 그 시점도 한 번 더 저장한다.
- 이어서 학습: `python training/train.py --resume <체크포인트 폴더>`
  - `MultiPPO/checkpoints/` 안의 것이면 폴더를 유지하고 이어서 쓴다 (그 뒤 iteration 기록은 곡선에서 잘라냄).
  - `best_model/...` 등 다른 곳의 것이면 `MultiPPO/` 를 새로 만들어 그 지점부터 학습한다.
- `--resume` 없이 실행하면 `MultiPPO/` 는 덮어쓰지만 `best_model/` 은 그대로 남는다.
- 모델 보기: `main.py` 의 `CHECKPOINT` 에 체크포인트 폴더 경로 (`MultiPPO/checkpoints/...` 또는 `best_model/...`).
- `--smoke` 는 `marl_models/MultiPPO_smoke/` 에 따로 쓰고 best 는 만들지 않는다.

## 보상 (`env/rewards.py`, 계수 = yaml `env_config.reward`)

- 할당 (양(+) 보상은 구조할 때만, 패널티는 팀 공유 + 중복 추적만 개인):
  `r_alloc^i(t) = w_rescue·|R_t|/N_T − (w_undisc/N_T)·Σ_{j∈U_t} min(1, t/max(1, τ_j−δ)) − (w_unres/N_T)·Σ_{j∈F_t} min(1, t/max(1, τ_j))`
  - `R_t` 이번 step 구조, `U_t` 미발견 (구조된 것만 제외, 만료 포함), `F_t` 발견했지만 미구조 (만료 포함)
  - `τ_j` 골든타임 마감 시각 (`end_time`, 없으면 `golden_time`), `δ = discover_margin` (발견 마감을 앞당김)
  - 시간이 지날수록 못 찾은 / 못 구한 조난자 몫 패널티가 커지고 마감 후엔 1 로 포화. 구조하면 그 몫이 즉시 사라짐
  `− w_duplicate·1[중복 추적]` (개인, 매 step, 먼 쪽만: 더 가까운 UAV 도 같은 태스크를 쫓거나 이미 구조된 태스크를 모르고 쫓음 — 보상은 학습용이라 전지적 판정)
- primitive (개인, 모드별): explore = `w_new_cells·|E^i_t|/(2r_s+1)²` (자기 센서로 처음 밝힌 셀, 팀이 몰랐던 셀만) /
  rescue·return = `w_progress·(목적지 BFS 거리 감소)` / 공통 `− w_step − w_blocked·1[막힌 이동] − w_collision·1[UAV 간 충돌: 같은 칸·자리 맞바꿈, depot 제외] (+ w_team·팀 보상)`

## 학습 신호 흐름 (RLlib 1.3.0)

1. `env/simulation.py:_execute` → info 에 `reward_alloc`, `reward_primitive`, `alloc_valid`
2. `training/multiagent_ppo.py:compute_gae_for_sample_batch` → 헤드별 GAE, `alloc_valid` 컬럼
3. `ppo_surrogate_loss` → 헤드 × 에이전트별 PPO loss 합. alloc 헤드의 policy/KL/entropy 는 `alloc_valid` 마스크 평균
4. 학습 forward 는 `SampleBatch.ACTIONS`(폭 2N, 순서 alloc | primitive)의 실제 할당으로 순차 디코딩을 재생 → rollout logit 과 일치
   (RLlib 이 학습 시 ACTIONS 를 numpy 로 넘기므로 `models/rllib_model.py` 가 텐서로 변환 — `tests/smoke_policy.py` 가 회귀 점검)
