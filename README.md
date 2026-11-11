# Multi-UAV Search & Rescue RL

여러 UAV 가 통신 그룹 단위 belief 로 생존자를 탐색 · 구조하는 시뮬레이터와 계층형 MAPPO 학습 코드.
결정은 Role 선택 (Explore / Rescue / Return) → 하위 할당 (Rescue: 생존자 선택 / Explore: frontier 덩어리 선택) → BFS 경로 추종.
같은 덩어리를 여러 대가 맡으면 env 가 측지 k-medoids 로 구간을 나누고 거리 합 최소 Hungarian 으로 배정한다 (학습 / 규칙 공통).

- **세 네트워크** role / rescue / explore — 각자 이종 그래프 GNN (노드 U · S · F, attention 집계) · decoder · critic (V^role / V^res /
  V^exp) 을 가지며 파라미터를 공유하지 않는다 (`models/role/net.py`, 그래프 사양 `env/role/graph_obs.py`).
- **구성요소 모드** 네트워크마다 `rule` (규칙 정책) | `learn` (학습, 저장 모델의 가중치에서 시작 가능).
- **학습 구간** 결정 단위 SMDP GAE. rescue / explore 는 같은 네트워크의 다음 결정 또는 그 역할을 벗어나는 순간까지 (`training/role/ppo.py`).
- 모든 명령은 **레포 루트에서** 실행한다. 참고 논문 `paper/` 는 저작권 때문에 git 에 올리지 않는다.

## 설치

Python **3.8**, Ray / RLlib **1.3.0**, gym 0.19, torch 1.8.1 (아래 '설치 상세' 의 순서대로 — gym 0.19 때문에
pip / setuptools / wheel 을 먼저 낮추고, torch 는 CPU / CUDA 빌드를 골라 따로 설치한 뒤 `pip install -r requirements.txt --no-build-isolation`).

## 폴더 구조

```
main_gui.py           GUI (pygame) — Model 창에서 규칙 정책 / 저장 모델 / 학습 중 체크포인트 선택 (설정 gui_evaluation/config/*_gui.yaml)
main_rule.py          여러 정책 (기본 rule_rule_rule · learn_rule_rule · learn_learn_learn) 같은 seed 비교 + 쌍별 짝 비교 -> results/rule_evaluation/
env/core/             공통 세계: 맵 생성 · 조난자 · 센싱 · n-hop 지도 병합 · belief · BFS · critic 관측
env/role/             RoleSimulation (역할 마스크 · 재결정 이벤트 · 결정 순서 · 보상 · 관측), explore (덩어리 · 구간),
                      clusters (DBSCAN · k-medoids), graph_obs (그래프 사양), gym_env (RoleEnv)
policies/             규칙 정책 (role_rules.py: RuleDecider, 여유 시간 최소 구조, 덩어리 점수 차례) · decision_order (결정 순서, 학습 모델과 공용)
models/role/          gnn (HeteroGNN), net (RoleNet: role / rescue / explore 네트워크 + critic), rllib_model
models/common/        critic (CTDE rich critic) · encoders · value_norm
training/             train.py (엔트리포인트), config/ (env / role / task.yaml), role/ (PPO loss · GAE · action dist),
                      callbacks (지표) · fixed_seed_eval (학습 중 고정 seed 평가) · reward_plot (학습 그래프) · behavior
eval_common/          model_loader (모델 경로 -> 구성요소 · 가중치) · light_policy (ray 없이 추론) · checkpoint_policy ·
                      parallel (seed 병렬) · episode (지표) · stats · paper_style · io
rule_evaluation/      main_rule 설정 · 비교 · 그림, run_rules (규칙 정책 기준선), upper_bound, slot_analysis
gui/, gui_evaluation/ pygame 화면 · 스타일 / GUI 설정 · 임무 할당 분석 그래프
tests/                점검 스크립트 (python -m tests.role.<name>)
maps/                 맵 JSON
marl_models/          학습 run · best 모델 (git 제외 — 학습된 모델은 별도 배포 (링크 TBD)), _archive/ = 보관 (목록에서 무시)
results/              평가 · 분석 출력 (git 제외)
```

## 학습

학습할 구성요소는 `training/config/env.yaml` 의 `components` (명령줄 `--role / --rescue / --explore` · `--<이름>_path` 가 우선):

```yaml
components:                    # 예: learn_learn_rule = role 은 저장 모델의 role 가중치에서 이어 학습, rescue 새로 학습, explore 규칙
  role:    {mode: learn, path: marl_models/best_model/learn_rule_rule/model_v01}   # learn + path = 그 모델의 role 가중치에서 시작
  rescue:  {mode: learn, path: null}                                                # path 없음 = 무작위 초기화
  explore: {mode: rule,  path: null}
```

```bash
python -m training.train                                   # env.yaml + role.yaml + task.yaml 대로
python -m training.train --role learn --role_path marl_models/best_model/learn_rule_rule/model_v01 --rescue learn --explore rule
python -m training.train --smoke --local --run_root /tmp/scratch       # 1 iteration 점검 (run 폴더를 다른 곳에)
python -m training.train --resume marl_models/<run>/checkpoints/checkpoint_000300   # 이어서 학습
```

- `learn` + `path` 는 path 모델의 그 네트워크 가중치만 (접두어 strict) 로드해 시작한다. path 가 없는 경로이거나, 그 모델에서 그 구성요소가
  rule 이거나, 구조가 다르거나, learn 이 하나도 없거나, role 이 rule 인데 rescue / explore 가 rule 이 아니면 시작할 때 바로 멈춘다.
- 네트워크별 lr · PPO clip 은 task.yaml `rescue` / `explore` (role 은 env.yaml `lr` · `clip_param`), `lr_schedule` 은 배율로 모든 네트워크에 적용.
- `--resume`: run_meta.json 이 있는 run 의 체크포인트면 그 run 폴더에 이어 쓰고 그 run 의 구성요소 설정을 쓴다 (폴더 이름과 무관).
- best 기준: `eval.every` iteration 마다 고정 seed (`eval.n_seeds`) greedy 평가의 구조율 최대 (같으면 짧은 에피소드). 학습 중에는 학습 모델만
  평가하고, 기준선 구조율은 `eval.baselines` 에 숫자로 넣어 로그 · 그림에만 쓴다.

## 모델 폴더 이름

조합 이름 = `<role 모드>_<rescue 모드>_<explore 모드>` (예: `learn_rule_rule`, `learn_learn_rule`, `learn_learn_learn`, learn 에 path 가 있어도 learn).

```
marl_models/<조합>_<YYYYMMDD_HHMMSS>[_smoke]/   학습 run: run_meta.json (combo, 구성요소별 {mode, path}), checkpoints/checkpoint_*,
                                                checkpoints/best/, 학습 그래프 fig_*.png, progress.csv, params.json
marl_models/best_model/<조합>/model_vXX/        run 마다 다음 버전으로 승격된 best: checkpoint_*, meta.json, best_info.json, params.json
```

모델 경로 하나로 구성요소를 정한다 (`eval_common/model_loader.py`): meta.json → run_meta.json → 폴더 이름 → params.json.
rescue / explore 분리 전 통합 task 네트워크 시절 모델은 "structure mismatch" 로 표시되고 로드되지 않는다.

## GUI / 평가

```bash
python main_gui.py                                                    # Model 창: Saved models (조합 폴더) / Training (run · 체크포인트)
python main_gui.py --source saved --combo learn_learn_rule --model model_v01
python main_gui.py --source training --run <run 폴더> --checkpoint best

# 같은 seed 짝 비교 (CPU 로 충분: CUDA_VISIBLE_DEVICES="", --workers 기본 = 코어 수 - 2)
python main_rule.py                                                  # rule.yaml compare.policies (기본 3 정책), seed 0-99, 모든 쌍
python main_rule.py --policies rule marl_models/best_model/learn_learn_rule/model_v01 \
                    --labels rule_rule_rule learn_learn_rule             # 정책 목록 바꾸기 (label 생략 = 조합 이름)
python main_rule.py --pairs "learn_learn_learn:rule_rule_rule,learn_learn_learn:learn_rule_rule" --seeds 0-9 --tag quick
python -m rule_evaluation.run_rules                                                                  # 규칙 정책 200 seed
```

스케일 비교 (맵 크기 · UAV 수 · 조난자 수) 는 `rule_evaluation/config/env_rule.yaml` 의 rows · cols · num_targets · max_targets ·
UAV 수 (agents 목록) · target_golden_time 을 바꿔서 main_rule 로 실행한다 (조건마다 `--tag` 를 다르게, max_targets ≥ num_targets 상한 —
작으면 시작할 때 에러). 실제 실행 env 는 시작할 때 콘솔과 meta.json `run_env` 에 출력된다.

## 점검

```bash
python -m tests.role.test_config          # yaml 구조 · 학습용 / GUI / 평가 사본 값 일치
python -m tests.role.test_role_env        # env · 보상 합
python -m tests.role.test_gnn             # 그래프 구성 · attention · 세 네트워크 독립 / gradient
python -m tests.role.test_rewards         # 보상 · SMDP GAE (역할 이탈 시 구간 종료)
python -m tests.role.smoke_role_policy    # 학습 경로 (rollout · 재생 · loss) — 설치 확인용
```
(그 밖의 점검: `tests/role/test_*.py`, `tests/eval/test_parallel_equiv.py`, GUI 헤드리스 `smoke_role_gui*.py` · `gui_check.py`.)

## 재결정 트리거 (env/role/simulation.py — 설정은 role.yaml env_config.events)

| 원인 | 조건 | 재결정 대상 | goal_valid |
|---|---|---|---|
| periodic | t % `period` == 0 (t = 0 포함, 기본 10) — 이벤트와 독립 | alive UAV 전원 (RETURN · 용량 0 포함) | 유지 |
| "1" 생존자 증가 | 그룹 멤버 누구든 자기 직전 belief 보다 아는 생존자가 늘었고 (탐색 발견 / 통신 병합), 그중 미구조 · 미만료 · 그룹이 제때 닿음 · 그룹 RESCUE 가 안 맡음 | 그룹 전원 (역할 무관) | 유지 |
| "2" 구조 완료 | 그 UAV 가 구조 | 그 UAV | 꺼짐 |
| "3" Return 완료 | depot 도착 · 충전 | 그 UAV | 꺼짐 |
| "4" 목표 무효 | RESCUE: conflict / 다른 UAV 구조 / 만료 / 도달 불가, EXPLORE: 덩어리 소멸 / 구간 없음 / exploration complete | 그 UAV | 꺼짐 |

- 재결정하면 역할부터 하위 할당까지 다시 한다 (RESCUE: 자기 현재 목표도 후보, EXPLORE: 지금 덩어리도 후보로 1 단계 → 구간 분할 → 2 단계).
- 같은 생존자 / 덩어리를 다시 고르면 학습 표본이 아니다 (rescue / explore valid 0).
- 규칙 정책: periodic / "1" 은 그룹 그리디 (담당 후보), "2" / "3" / "4" 는 한 대 여유 시간 최소 (slack = τ − t − d 작은 순 → 생존자
  번호, role_sigma / us_slack 과 같은 식 · role_front 를 한 대에 적용한 것과 같은 순서).
- explore 그리디: 1 단계 차례 = 남은 UAV × 닿는 덩어리 쌍 중 점수 (N_m/(n_m+1))/(d+1)^α 최대 (`order_dist_alpha`), 담당 후보가 없으면
  같은 점수 최대 덩어리. 2 단계 = 덩어리 참여자 전원을 구간까지 BFS 거리 합 최소 1:1 Hungarian (goal 우대 없음).

## 보상 (요약, env/role/simulation.py — 설정은 role.yaml / task.yaml 의 env_config.rewards)

| 네트워크 | 보상 (step 갱신 후 상태) |
|---|---|
| role | (\|ΔR_t\| − w_x\|ΔE_t\|)/N_T + (w_d/N_T)·Σ_{k∈ΔF_t} 1[τ_k − t > T_m] − (w_sw/N_T)·1[자발적 역할 전환]. SMDP 할인 γ_r = `role_gamma` |
| rescue (RESCUE 인 step) | −(w_q/T_max)·\|F_t \ (R_t ∪ E_t)\|/N_T − (w_x/N_T)·\|ΔE_t ∩ F_t\| − (w_dup/N_T)·1[할당 충돌] − (w_ts/N_T)·1[자발적 목표 전환] |
| explore (EXPLORE 인 step) | −(w_u/T_max)·\|S \ F_t\|/(N_T − \|E_t \ F_t\|) − (w_ts/N_T)·1[자발적 목표 전환] |

집합은 전역 (S 전체 생존자, F 발견, R 구조, E 만료). RETURN step 의 task 보상은 0. rescue / explore critic 은 학습 전용 전역 입력
`critic_task_global` = [미발견 비율, 대기 비율, t/T_max] 를 받는다 (actor 는 읽지 않음).

## 평가 결과 위치

```
results/
├── rule_evaluation/<YYYYMMDD_HHMMSS>_<tag>/     ← main_rule.py
│   ├── summary.txt                               ← 정책 표 + 짝 비교 표 (콘솔 출력과 같음)
│   ├── per_seed.csv                              ← seed · 정책 별 지표 (seed 순 → 정책 순)
│   ├── summary.csv / pairwise.csv                ← 정책 · 지표별 평균 · 표준편차 · 95% CI / 쌍 · 지표별 차이 · W/T/L · p · Holm p · d_z
│   ├── table_main.tex / table_conflict.tex       ← 논문용 정책 표 + 짝 비교 표 / 할당 충돌 표 + 짝 비교 표
│   ├── progress.csv                              ← 정책 · seed 별 진행률 10 % 구간 누적 발견 · 구조율, 역할 비율, 동시 추격 수
│   ├── figures/fig_compare_*.pdf|png|csv         ← bar (지표 + CI), paired (구조율 · 길이 seed 차이), seed (seed 별 구조율),
│   │                                               cumulative (진행률별 누적 발견 · 구조), roles (정책별 역할 비율),
│   │                                               conflict (동시 추격 · 헛걸음 · 낭비 step), conflict_time (진행률별 동시 추격)
│   └── config.yaml / command.txt / meta.json     ← 실제 설정 사본, 실행 명령, 정책 정보 · 실제 env (run_env 요약) · git commit
└── gui_evaluation/                               ← GUI G 키 임무 할당 그래프
```

- 평가 환경 설정: `rule_evaluation/config/{env,role,task}_rule.yaml` 을 env → role → task 순으로 합친 값 (모든 정책에 같은 env,
  구성요소 모드만 정책마다). 실행 설정 (seed · mode · workers · bootstrap · 그림) 은 `rule.yaml`. 평가 env yaml 은 training/config 와
  같은 값으로 유지 (`tests/role/test_config.py`, 다르게 둘 키는 그 파일의 `ALLOWED` 에 추가). 조건을 바꾸려면 평가 env yaml 을 고친다.
  체크포인트는 가중치와 관측 shape 를 정하는 키에만 쓴다: state_size · local_channels · global_downsample · cluster.k_max 가
  평가 env 와 다르면 체크포인트 값으로 고정해 meta.json `forced_checkpoint_keys` 에 남기고, 모델 파라미터와 무관한 크기 키
  (FREE_KEYS = rows · cols · max_targets · max_episode_len · s_max, `eval_common/checkpoint_policy.py`) 는 평가 yaml 값을 그대로 쓴다
  (s_max 는 모델의 gnn.s_max 도 같은 값으로 맞춰 만들고, 체크포인트와 다르면 run_env 에 "32 (checkpoint 16)" 처럼 표시 — 규칙
  정책의 후보 슬롯 (victim_slots) 도 s_max 개라 규칙 기준선도 같이 바뀐다).
- 같은 seed 에서 모든 정책의 초기 상태 (맵 · depot · 조난자 · 골든타임 · UAV) 가 같은지 확인한다.

결과 열 (per_seed.csv = 에피소드 하나):

| 열 | 뜻 |
|---|---|
| rescue_rate / expire_rate / discover_rate / explored | 구조 · 만료 · 발견 조난자 비율 (N_T 대비), 탐색한 free 셀 비율 |
| rescue_time_mean / golden_slack_mean | 구조 시각 평균, (마감 - 구조 시각) / 골든타임 평균 (구조된 조난자) |
| discover_to_rescue_mean / undiscovered | 구조된 조난자의 발견 → 구조 step 평균, 끝날 때 미발견 생존자 수 |
| conflict_concurrent (_same_group / _cross_group) | 동시 추격: 새 구조 목표 결정 시점에 다른 UAV 가 같은 생존자를 쫓는 중 (전역 기준, 결정당 1 회; 같은 그룹은 마스크상 0) |
| wasted_trips / wasted_steps | 헛걸음: 추격 중 다른 UAV 가 먼저 구조해 끝난 추격 수, 그 추격들 동안 움직인 step 합 |
| conflict_reward / rescue_decisions / *_per_dec | 학습 보상 할당 충돌 (_check_dup) 판정 횟수, 새 구조 목표 결정 수, 결정 1 회당 비율 |
| rescued_per_uav / decisions / infer_ms_per_decision | UAV 당 구조 수, 결정 수, 모델 추론 시간 (CPU, 워커당 스레드 1) |
| role_decisions, trunc_decisions_S, trunc_frac_S | 역할 결정 수, 그중 할당 가능 후보 > s_max 인 결정 수 · 비율 |
| dropped_S (= dropped_reachable_S), dropped_S_per_dec | 잘린 할당 가능 후보 수 합 · 결정당 (후보 조건이 골든타임 안 도달이라 reachable 과 같음) |
| dropped_urgent_S, best_dropped_S | 잘린 후보 중 남은 후보보다 급한 수, 규칙 기준 (여유 시간 최소) 선택이 잘린 쪽에 있던 결정 수 |
| s1_decisions, trunc_decisions_K, trunc_frac_K, dropped_K, dropped_K_cells | explore 1 단계 결정 기준 덩어리 > k_max 잘림 (덩어리 수 · frontier 셀 수) |
| s_nodes_overflow | 그룹이 아는 미구조 생존자 전체 > s_max 인 역할 결정 수 (예전 survivor_truncated, S 노드 넘침) |
| ub_loose / ub_greedy / rescue_over_ub | depot 에서 골든타임 안 도달 가능 비율, 전지적 EDF 스케줄 구조율, 구조율 / ub_loose |
| group_size_mean / n_groups_mean / isolated_frac | 통신 그룹 평균 크기 · 수, 혼자인 UAV 비율 (step 평균) |
| expire_undiscovered / unassigned / assigned_rate | 만료 원인: 미발견 / 발견했지만 미배정 / 맡고 있던 중 (N_T 대비) |
| (main_rule pairwise.csv) pair, mean, ci_lo/hi, win / tie / loss, p_wilcoxon, p_holm, d_z | 쌍 A - B 의 seed 차이 평균 · CI, A 기준 승 / 무 / 패 seed 수, Wilcoxon p (차이 전부 0 이면 1), 같은 지표 안 Holm 보정 p, Cohen's d_z (표준편차 0 이면 빈칸) |

## 설치 상세 (새 컴퓨터, Windows + Anaconda 기준)

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
