# Multi-UAV Search & Rescue RL

계층 구조 결정: Role 선택 (Explore / Rescue / Return) → Rescue 할당 (조난자) / Explore (frontier DBSCAN 덩어리 선택 →
같은 덩어리를 m 대가 맡으면 측지 k-medoids 구간 분할 → Hungarian 구간 배정 (거리 합 최소)) → BFS 경로 추종. Return 목표 = depot.
두 네트워크 — 역할 선택 / task (하위 할당) (생존자 · frontier 덩어리를 한 후보 집합 S ∪ F 에서 고름) — 를 각각 독립된 encoder ·
이종 그래프 GNN (공유 없음, attention 집계) 으로 학습 (role · task 2 head MAPPO) 하거나 규칙 정책으로 고를 수 있다
(yaml `policy_mode: {role, task}`). 정보 구조는 belief: UAV 는 자기 n-hop 통신 그룹의 병합 지식으로 그래프를 만들고 GNN 은
그 위에서 로컬로 계산한다. 그룹 밖 UAV 는 마지막 기록 (중계 포함) 을 메모리 노드로 넣는다 (GNN 입력 전용).
frontier 덩어리 (F 노드) 의 위치 특징 = 결정 UAV 에게 BFS 로 가장 가까운 덩어리 셀 (U–F 거리와 같은 BFS · 셀, 도달 가능 셀이
없으면 셀 평균) — F–F / S–F 간선의 Δp 도 이 점 기준 (사양 `env/role/graph_obs.py`).
그룹 context c_i = [ρ, η, ν, t / T_max] (그룹 미발견 비율, 미탐색 비율, 발견 · 미구조 비율, 에피소드 경과 비율) — 역할 / task decoder 의 ξ_ctx 입력.
덩어리 · 구간 생성 (DBSCAN, k-medoids) 과 구간 배정 (Hungarian) 은 env 가 하므로 학습 / 규칙 어느 쪽이든 같다.

참고 논문: `paper/` (저작권 때문에 git 에는 안 올림). `docs/research_pipeline_summary.md`, `docs/Framework.png` 는 초기 3계층 설계 기록.
모든 명령은 **레포 루트에서** 실행한다 (패키지 import 기준점).

## 폴더 구조

```
main_gui.py                     GUI 실행 (CHECKPOINT 로 학습 모델 보기, None 이면 규칙 정책) — 설정 gui_evaluation/config/*_main.yaml
main_rule.py                    규칙 정책 vs 학습 모델, 같은 seed 몬테카를로 비교 (통계 · 논문용 그림, --run both|rule|learned)  -> results/rule_evaluation/
main_scale.py                   UAV 수 N 과 함께 조난자 · 맵 · 시간 크기 조정, 규칙 vs 학습 (확장성)   -> results/scale_evaluation/
env/                            시뮬레이터 — 학습 / 정책과 무관
├── core/                       공통 세계: base_sim (BaseSimulation: 맵 · 조난자 · 센싱 · 구조 · 재보충 · 마감 · critic 관측) ·
│                               agent (UAV) · targets · map_gen (랜덤 맵) · pathfinding (BFS) ·
│                               knowledge (센싱 · n-hop 지도 병합) · beliefs · modes · coords · step_context
└── role/                       simulation (RoleSimulation: 역할 마스크 (부분 귀환) · 재결정 (주기 + 이벤트 1~4) · 결정 순서 · 보상 · 관측 · 이벤트 로그) ·
                                explore (덩어리 대응 · 쪼개짐 / 합쳐짐 · 구간 rebalance · 2 단계 유지 UAV 매칭 · 노드 특징 / 간선) ·
                                graph_obs (이종 그래프 관측 사양: 노드 U / S / K, 간선 U–U / U–S / U–K / S–S / K–K) ·
                                connectivity (실제 전역 연결성 C = 연결된 쌍 비율 — 평가 · 진단 지표만, 보상 · 특징은 그룹 연결도 ω^g) ·
                                clusters (frontier DBSCAN · 측지 k-medoids) · gym_env (RoleEnv)
policies/                       규칙 정책 (학습 없음)
└── role_rules.py               rule_role / rule_rescue (EDF) / rule_explore (덩어리) / rule_explore_sub (구간, 거리 합 최소 Hungarian — env 2 단계 고정),
                                RuleDecider, RandomDecider
models/                         신경망 (순수 torch + RLlib 래퍼)
├── common/                     encoders (CNN / MLP) · value_norm · critic (CTDE critic, value head 1 개 또는 공유 trunk + 여러 개)
└── role/                       gnn (HeteroGNN: 2 층 MLP encoder, 관계별 multi-head attention 집계, U–S / U–F 간선 갱신) ·
                                net (RoleNet: 독립 네트워크 2 개 — 역할 선택 (ξ_dec(h_i ‖ ξ_ctx(c_i)) -> 3) / task (하위 할당) (S ∪ F
                                pointer attention, 1 단계만), 노드 U (연결 + 메모리) · S · F, 순차 디코딩 · 결정마다 그래프
                                재구성, critic: V^role + 할당 공유 trunk (V^res, V^exp)) · rllib_model
training/                       학습
├── train.py                    학습 엔트리포인트
├── fixed_seed_eval.py          학습 중 고정 seed 평가 (best 체크포인트 선정)
├── config/                     env.yaml (환경 + 학습 공통, memory) · role.yaml (역할: 학습/규칙 · 이벤트 · role 보상 · 역할 네트워크) ·
│                               task.yaml (하위 할당: 학습/규칙 · task 네트워크 · critic · λ_task, rescue 보상, 덩어리 · 1 단계 ·
│                               2 단계 · explore 보상)
├── config_loader.py            yaml 합치기 (기본 env + role + task + explore + rescue, 뒤 파일이 앞 파일 값을 덮어씀)
├── role/                       ppo (RolePPOTrainer: role / task head, 표본 종류 (role / RESCUE / EXPLORE) 별 GAE · 표준화 · 마스크 loss) · action_dist
├── callbacks.py                에피소드 지표 (custom_metrics)
├── catalog.py                  RLlib 등록 (env / model / action dist)
├── reward_plot.py              학습 그래프 6 개 (fig_training_performance / fig_rewards / fig_learning / fig_behavior /
│                               fig_undiscovered + 진단 fig_diagnostics, png + pdf)
└── behavior.py                 고정 seed 평가 에피소드 기록 (진행률 곡선 · 귀환 시작 용량 · 미발견 생존자), 규칙 정책 기록 캐시,
                                짝 비교 bootstrap — 평가 · 그래프 공용
eval_common/                    평가 공용
├── checkpoint_policy.py        체크포인트 로드 + 실행 (GUI / 평가 공용, 다른 N · 맵 크기로 실행)
├── episode.py                  에피소드 실행 · 지표 (METRICS, RulePolicy, run_episode, run_episode_detailed, 초기 상태 지문)
├── stats.py                    평균 · 표준편차, bootstrap CI, 짝 차이 · 승/무/패 · Wilcoxon (+ 최소 p, 효과 크기 d_z)
├── light_policy.py             경량 체크포인트 정책 (ray / trainer 없이 가중치만 torch 로 — 병렬 워커용, CheckpointPolicy 와 같은 결정)
├── parallel.py                 seed 단위 병렬 실행 (spawn, CPU, 워커당 torch 스레드 1, 체크포인트 한 번만 로드)
├── paper_style.py              논문 그림 스타일 (serif, Okabe–Ito, PDF + PNG 300 dpi + 원자료 CSV)
└── io.py                       결과 폴더 (<시각>_<tag>), CSV / JSON, booktabs 표, git commit 기록
gui_evaluation/                 GUI 실행 설정 · 분석
├── config/                     env_main.yaml · role_main.yaml · task_main.yaml (main_gui.py 전용, 학습용과 같은 값 — depot 만 예외)
└── assign_recorder.py, assign_plots.py   임무 할당 분석 그래프 (GUI G 키) -> results/gui_evaluation/
rule_evaluation/                규칙 정책 기준선 · 비교
├── config/rule.yaml            main_rule.py 실행 설정 (seed, 지표, bootstrap, 그림)
├── config/*_rule.yaml          env_rule.yaml · role_rule.yaml · task_rule.yaml (main_rule.py 평가 env — 시뮬레이션 env_config 만, 학습용과 같은 값)
├── compare.py, plots.py        main_rule.py 비교 로직 · 그림 / 표
├── slot_analysis.py            S / K 슬롯 잘림 분석 (규칙 정책, s_max 비교, 조난자 수 · 시간대별)
├── run_rules.py                규칙 정책 200 seed 기준선           -> results/rules/
└── upper_bound.py              이론적 상한 / 실패 원인 분해         -> results/upper_bound/
scale_evaluation/               확장성
├── config/scale.yaml           main_scale.py 실행 설정 (N 목록, 크기 조정 preset (default / fixed), seed, 그림)
├── config/*_scale.yaml         env_scale.yaml · role_scale.yaml · task_scale.yaml (main_scale.py 기준 env — 시뮬레이션 env_config 만, 학습용과 같은 값)
├── scaling.py                  N 별 env_config (r = N / N_ref 지수 조정)
└── compare.py, plots.py        main_scale.py 비교 로직 · 그림 / 표
gui/                            pygame 시각화 · 맵 편집기 (grid_map.py, style.py = 색 · 도형 · 범례 표, colors.py = UAV 색)
tests/                          점검 스크립트 (ray 클러스터 불필요)
└── role/                       test_role_env · test_explore · test_events · test_return_mask · test_gnn ·
                                test_rewards · test_reward_terms · test_task_memory · test_triggers · test_explore_order ·
                                test_decide_order · test_join_reselect · test_config · test_fnode_pos ·
                                test_gui_style · test_role_rule_equiv · smoke_role_policy · smoke_role_gui(_model) · gui_check · print_obs_spec
maps/                           맵 JSON (map/ 장애물만, map_target/ 조난자 · depot 포함)
marl_models/                    학습 결과 (RolePPO/, best_model/)
results/                        평가 · 분석 출력 (아래 '평가 결과 위치')
```

의존 방향: `env` ← `policies` ← `models` ← `training` ← `eval_common` ← `*_evaluation` ← `gui` / `main_*.py`.
`RoleSimulation` 은 `env/core/base_sim.py` 의 `BaseSimulation` 을 상속한다.

## 자주 쓰는 명령

```bash
# GUI
python main_gui.py

# 학습 — training/config/env.yaml (환경 + 학습 공통) + role.yaml (역할) + task.yaml (하위 할당) 을 고친 뒤
python -m training.train                                  # role · task 2 head 학습 (ablation 은 role.yaml / task.yaml 의 policy_mode)
python -m training.train --smoke --local                  # 1 iteration 점검

# 규칙 정책 기준선 / 평가
python -m rule_evaluation.run_rules                            # 그리디 정책 200 seed
python -m rule_evaluation.upper_bound [--checkpoint <checkpoint>]

# 논문용 비교 (GUI 없음, CPU 로 돌려도 됨: CUDA_VISIBLE_DEVICES="")
#   --workers W: seed 단위 병렬 (기본 = CPU 코어 수 - 2, yaml workers). 결과는 workers 수와 무관 (추론 시간 제외)
python main_rule.py --checkpoint <checkpoint> [--seeds 0-99] [--mode greedy|sample] [--tag name] [--workers 16]   # 규칙 + 학습 (짝 비교)
python main_rule.py --run rule [--seeds 0-99]                                       # 규칙 정책만 (체크포인트 불필요)
python main_rule.py --run learned --checkpoint <checkpoint> [--seeds 0-99]          # 학습 모델만
python main_scale.py --checkpoint <checkpoint> [--n 2,3,4,5,6,8,10] [--seeds 0-49] [--preset default|fixed] [--tag name] [--workers 16]
python -m rule_evaluation.slot_analysis [--s_max 16,24,32,48] [--seeds 0-49] [--workers 16]   # 슬롯 잘림 분석 (규칙 정책)
python -m tests.eval.test_parallel_equiv --checkpoint <checkpoint>   # 병렬 = 순차, 경량 정책 = ray 정책

# 점검
python -m tests.role.test_role_env
python -m tests.role.test_explore                         # explore 2 단계 (구간 분할 / 추적 / rebalance)
python -m tests.role.test_fnode_pos                       # F 노드 위치 = 결정 UAV 에게 가장 가까운 덩어리 셀 (동률 · fallback)
python -m tests.role.test_events_targeted                 # 이벤트 체계 (원인별 대상, 쪼개짐 / 합쳐짐, 이벤트 "4" 간격, hysteresis)
python -m tests.role.test_explore_order                   # explore 1 단계 점수 차례 (N/(n+1))/(d+1)^α · fallback · 2 단계 거리 Hungarian
python -m tests.role.test_triggers                        # 주기 재결정 · 이벤트 "1" (탐색 발견 / 통신 병합) · goal_valid · 같은 목표 옵션 이어가기 ·
                                                          #   용량 0 EXPLORE · 규칙 경로 · 예전 트리거 호환 (period 0)
python -m tests.role.test_return_mask                     # 부분 귀환 역할 마스크
python -m tests.role.test_gnn                             # 그래프 구성 (연결 / 메모리 노드, context, ETA σ, 목표 중복) · attention 순열 불변 · 패딩 · 2-hop · 네트워크 독립 / gradient · 연결성 · 보상
python -m tests.role.test_task_memory                    # U 노드 12 · context 4 · S–F kNN 간선, 메모리 노드 (age · 중계 · 마스크 무관), 할당 마스크 (S / F), explore 배율
python -m tests.role.test_rewards                         # 보상: 늦지 않은 발견 (τ - t > T_m 만, 팀 공통), rescue n_suc - F_excess / U_ε,
                                                          #   explore 직접 / 공유 · 옵션 return, 중복 추적
python -m tests.role.test_config                          # yaml: 학습용 = *_main key 구조, 깊은 병합, 보상 키
python -m tests.role.test_gui_style                       # GUI 스타일 표 (하드코딩 색 없음, 범례 = 그린 요소)
python -m tests.role.print_obs_spec                       # 관측 명세 (그래프 노드 / 간선 특징 표)
python -m tests.role.smoke_role_policy
```

## 재결정 트리거 (env/role/simulation.py — 설정은 role.yaml env_config.events)

| 원인 | 조건 | 재결정 대상 | goal_valid |
|---|---|---|---|
| periodic | t % `period` == 0 (t = 0 포함, 기본 10) — 이벤트와 독립 | alive UAV 전원 (RETURN · 용량 0 포함) | 유지 |
| "1" 생존자 증가 | 그룹 멤버 누구든 자기 직전 belief 보다 아는 생존자가 늘었고 (탐색 발견 / 통신 병합), 그중 미구조 · 미만료 · 그룹이 제때 닿음 · 그룹 RESCUE 가 안 맡음 | 그룹 전원 (역할 무관) | 유지 |
| "2" 구조 완료 | 그 UAV 가 구조 | 그 UAV | 꺼짐 |
| "3" Return 완료 | depot 도착 · 충전 | 그 UAV | 꺼짐 |
| "4" 목표 무효 | RESCUE: conflict / 다른 UAV 구조 / 만료 / 도달 불가, EXPLORE: 덩어리 소멸 / 구간 없음 / exploration complete | 그 UAV | 꺼짐 |

- 재결정하면 역할부터 하위 할당까지 다시 한다 (RESCUE: 자기 현재 목표도 후보, EXPLORE: 지금 덩어리도 후보로 1 단계 → 구간 분할 → 2 단계).
- 같은 생존자 / 덩어리를 다시 고르면 학습 표본이 아니고 (rescue / explore valid 0) 옵션이 이어진다.
  다른 목표 · 역할로 바꾸면 열린 rescue 옵션이 그 결정에서 끝나고, 보너스는 `info["reward_rescue_close"]` 로 실려 학습이 그 옵션의 마지막
  step 에 더한다.
- 규칙 정책: periodic / "1" 은 그룹 그리디 (담당 후보), "2" / "3" / "4" 는 한 대 EDF.
- explore 그리디: 1 단계 차례 = 남은 UAV × 닿는 덩어리 쌍 중 점수 (N_m/(n_m+1))/(d+1)^α 최대 (`order_dist_alpha`), 담당 후보가 없으면
  같은 점수 최대 덩어리. 2 단계 = 덩어리 참여자 전원을 구간까지 BFS 거리 합 최소 1:1 Hungarian (goal 우대 없음).

## 보상 (요약, env/role/simulation.py role_rewards — 설정은 role.yaml / task.yaml 의 env_config.rewards)

| head | 보상 (step 갱신 후 상태로 계산) | 끄기 (ablation) |
|---|---|---|
| role | (\|ΔR_t\| − w_x\|ΔE_t\|)/N_T + (w_d/N_T)·Σ_{k∈ΔF_t} 1[τ_k − t > T_m] − (w_sw/N_T)·1[UAV i 자발적 역할 전환] (앞 두 항 팀 공통, 전환 항 개인 — 자발적 = 재결정 원인이 periodic / "1" 뿐 · 이전 역할이 마스크에서 허용 · 역할이 바뀜). 늦지 않은 발견 보상: 발견 시점에 골든타임이 T_m (`discover_margin`) step 넘게 남은 생존자를 처음 발견하면 +w_d/N_T (팀 공통, ΔF_t = 팀 전체 기준 첫 발견, 이후 회수 없음). role SMDP 할인 γ_r = 최상위 `role_gamma` (기본 1.0) | `w_d: 0` (발견 보상), `w_sw: 0` |
| rescue | (n_suc_i − F_excess)/N_T + r_dup_i + bonus_i (개인 + 그룹 F, 옵션 단위, 할인 없음). n_suc_i = UAV i 가 고른 생존자에 시간 안에 닿음 (0/1), F_excess = F − F_min (G_res 전원 같은 값, F_min = max(0, \|U_ε\| − \|G_res\|)), bonus_i = w_b·1[구조 완료로 옵션 끝]·1[F_excess = 0]·((1 − w_c)e^{−d_i} + w_c ω̄^g_i)/N_T, d_i = BFS(i, a_i)/d_max, ω̄^g_i = 옵션 동안 그룹 연결도 ω^g_{i,t} = (\|U_{i,t}\| − 1)/(\|U\| − 1) 평균 (U_{i,t} = belief 를 병합하는 통신 그룹, UAV 노드 특징 omega_group 과 같은 값). 같은 생존자 재선택은 옵션 계속 | — |
| explore | (1 + w_omega ω^g_{i,t})·r_direct + r_share (EXPLORE step 만, 개인 · 자기 belief 기준, 옵션 단위 γ = 최상위 `gamma` 할인 누적, 그룹 연결도 배율은 직접 탐색 항에만). r_direct = (1/\|C_free\|) Σ_{c ∈ V_{i,t} \ C_found,i,t−1} 1/n_c (이번 step 직접 본 새 free 셀, n_c = 같은 그룹 안 동시 관측 수), r_share = w_share·\|ΔC^merge_{i,t}\|/\|C_free\| (통신 병합으로 처음 들어온 free 셀, 그룹이 직접 본 셀 제외) | `w_share: 0` (공유 항), `w_omega: 0` (배율) |

## 학습 결과 위치

```
marl_models/
├── RolePPO/                           ← 학습 결과. 새로 학습하면 덮어씀
│   ├── checkpoints/checkpoint_XXXXXX/ ← 10 iteration 마다 저장, 전부 유지 (train.py CHECKPOINT_FREQ)
│   ├── checkpoints/best/              ← 고정 seed 평가 (eval.every iteration 마다, 두 head argmax) 기준 best + best_info.json
│   ├── fig_training_performance.*     ← 평가 구조율 (± std, rule baseline, best) · 학습 구조 / 만료 / 발견율 · 에피소드 길이 · 구조 시각 ·
│   │                                    learned − rule 짝 비교 (seed 별 구조율 차이, bootstrap 95 % CI) · UAV 당 역할 전환 수 (자발적 / 강제)
│   ├── fig_rewards.*                  ← head 별 행, 보상 항목별 패널 (UAV 당 에피소드 합)
│   ├── fig_learning.*                 ← role / task–rescue / task–explore 의 critic EV (value head 별) · entropy · iteration 당 표본 수
│   ├── fig_diagnostics.*              ← (진단) head 별 approx KL · clip 비율, 결정 시점 메모리 노드 수 · 평균 age
│   ├── fig_behavior.*                 ← 최근 평가의 진행률별 역할 비율 (learned / rule) · 누적 발견 / 구조 · 그룹 연결도 ω^g ·
│   │                                    귀환 시작 시점 남은 용량 분포 (규칙 정책 = 같은 seed), 학습 중 역할 비율
│   ├── fig_undiscovered.*             ← 최근 평가에서 끝까지 발견 못 한 생존자 위치 (seed 누적, learned / rule)
│   │                                    (.png dpi 200 + .pdf, --plot-every 기본 10, EMA 창 env.yaml plot.ema_window 기본 10)
│   ├── eval_latest.json / rule_eval.json ← 최근 평가 기록 / 같은 고정 seed (eval.n_seeds, 기본 20) 의 규칙 정책 기록 캐시
│   ├── progress.csv / result.json / events.out.tfevents.* / params.json
└── best_model/
    └── <학습 시작 시각>_role/            ← 학습마다 새 폴더 (덮어쓰지 않음). 그 학습에서 지금까지 가장 좋은 모델 (checkpoints/best 복사)
        ├── checkpoint_XXXXXX/
        └── params.json / best_info.json (iteration, eval 구조율 · 길이) / fig_*.png
```

- 체크포인트는 저장 즉시 디스크에 남으므로 학습이 중간에 멈춰도 사라지지 않는다. Ctrl+C 로 멈추면 그 시점도 한 번 더 저장한다.
- 이어서 학습: `python -m training.train --resume <체크포인트 폴더>`
  - `RolePPO/checkpoints/` 안의 것이면 폴더를 유지하고 이어서 쓴다 (그 뒤 iteration 기록은 곡선에서 잘라냄).
  - `best_model/...` 등 다른 곳의 것이면 `RolePPO/` 를 새로 만들어 그 지점부터 학습한다.
- `--resume` 없이 실행하면 `RolePPO/` 는 덮어쓰지만 `best_model/` 은 그대로 남는다.
- 모델 보기: `main_gui.py` 의 `CHECKPOINT` 에 체크포인트 폴더 경로 (`RolePPO/checkpoints/...` 또는 `best_model/..._role/...`).
- best 기준: 고정 seed (run_rules 와 같은 seed, eval.n_seeds 개) · 두 head argmax 평가의 구조율 최대, 같으면 에피소드 길이가 짧은 쪽
  (env.yaml `eval.every` / `eval.episodes`). 평가는 driver policy 로 따로 돌려 학습 rollout 에는 영향이 없다.
- `--smoke` 는 `marl_models/RolePPO_smoke/` 에 따로 쓰고 `best_model/` 은 만들지 않는다 (`checkpoints/best/` 는 저장).

## 평가 결과 위치

```
results/
├── rule_evaluation/<YYYYMMDD_HHMMSS>_<tag>_<run>/      ← main_rule.py (run = both / rule / learned)
│   ├── per_episode.csv / timeseries.csv          ← 정책 · seed 별 지표, step 별 누적 구조율 · 발견율 · 탐색 비율 · 역할 비율
│   ├── summary.csv / summary_table.tex           ← 평균 ± 표준편차, 95% bootstrap CI, 짝 차이 (학습 - 규칙) · 승/무/패 · Wilcoxon p
│   ├── figures/fig_*.pdf|png|csv                 ← 지표 CI, 구조율 violin, 짝 산점도, 누적 곡선, 역할 비율, 몬테카를로 수렴
│   └── meta.json                                 ← 실제 병합 설정 · 체크포인트 · git commit
├── scale_evaluation/<YYYYMMDD_HHMMSS>_<tag>_<preset>/   ← main_scale.py
│   ├── scale.csv / per_episode.csv / inference.csv / scale_table.tex
│   ├── figures/fig_*.pdf|png|csv                 ← N 별 비율, 짝 차이, UAV 당 구조 수, 추론 시간
│   └── meta.json                                 ← N 별 실제 env_config · 체크포인트 · git commit
└── gui_evaluation/                               ← GUI G 키 임무 할당 그래프
```

- 평가 환경 설정: 각 평가 폴더의 env yaml 을 env → role → task 순으로 합친 값 — main_rule = `rule_evaluation/config/{env,role,task}_rule.yaml`
  (`--run rule / learned / both` 모두 같은 env), main_scale = `scale_evaluation/config/{env,role,task}_scale.yaml` 위에 N 별 크기 조정
  (`--preset` / `--n`). 실행 설정 (seed · mode · workers · bootstrap · 그림) 은 `rule.yaml` / `scale.yaml`. 평가 env yaml 은 training/config 와
  같은 값으로 유지 (`tests/role/test_config.py`, 다르게 둘 키는 그 파일의 `ALLOWED` 에 추가). 조건을 바꾸려면 평가 env yaml 을 고친다.
  체크포인트는 가중치와 관측 shape 를 정하는 키 (SHAPE_KEYS + cluster.k_max) 에만 쓰고, 평가 env 와 다르면 체크포인트 값으로 고정해
  meta.json `forced_checkpoint_keys` 에 남긴다 (main_scale 은 rows / cols / max_targets / max_episode_len 만 풀어 줌).
- 같은 seed (main_scale 은 같은 (seed, N)) 에서 두 정책의 초기 상태 (맵 · depot · 조난자 · 골든타임 · UAV) 가 같은지 확인한다.

결과 열 (per_episode.csv = 에피소드 하나, summary.csv / scale.csv = 지표마다 정책별 평균 · 표준편차 · CI + 짝 비교):

| 열 | 뜻 |
|---|---|
| rescue_rate / expire_rate / discover_rate / explored | 구조 · 만료 · 발견 조난자 비율 (N_T 대비), 탐색한 free 셀 비율 |
| rescue_time_mean / golden_slack_mean | 구조 시각 평균, (마감 - 구조 시각) / 골든타임 평균 (구조된 조난자) |
| rescued_per_uav / decisions / infer_ms_per_decision | UAV 당 구조 수, 결정 수, 모델 추론 시간 (CPU, 워커당 스레드 1) |
| role_decisions, trunc_decisions_S, trunc_frac_S | 역할 결정 수, 그중 할당 가능 후보 > s_max 인 결정 수 · 비율 |
| dropped_S (= dropped_reachable_S), dropped_S_per_dec | 잘린 할당 가능 후보 수 합 · 결정당 (후보 조건이 골든타임 안 도달이라 reachable 과 같음) |
| dropped_urgent_S, best_dropped_S | 잘린 후보 중 남은 후보보다 급한 수, EDF 선택이 잘린 쪽에 있던 결정 수 |
| s1_decisions, trunc_decisions_K, trunc_frac_K, dropped_K, dropped_K_cells | explore 1 단계 결정 기준 덩어리 > k_max 잘림 (덩어리 수 · frontier 셀 수) |
| s_nodes_overflow | 그룹이 아는 미구조 생존자 전체 > s_max 인 역할 결정 수 (예전 survivor_truncated, S 노드 넘침) |
| ub_loose / ub_greedy / rescue_over_ub | depot 에서 골든타임 안 도달 가능 비율, 전지적 EDF 스케줄 구조율, 구조율 / ub_loose |
| group_size_mean / n_groups_mean / isolated_frac | 통신 그룹 평균 크기 · 수, 혼자인 UAV 비율 (step 평균) |
| expire_undiscovered / unassigned / assigned_rate | 만료 원인: 미발견 / 발견했지만 미배정 / 맡고 있던 중 (N_T 대비) |
| diff_mean, diff_ci_lo/hi, model_win / tie / model_loss | 짝 차이 (학습 - 규칙) 평균 · 95% bootstrap CI, 모델 기준 승 / 무 / 패 비율 |
| d_z, d_z_ci_lo/hi | 효과 크기 Cohen's d_z = 평균 / 표준편차 (짝 차이) · bootstrap CI |
| p_wilcoxon, p_min, n_nonzero, p_note, unanimous | Wilcoxon p, 도달 가능한 최소 p (= 2 / 2^n_nonzero), n < 6 표시, 모든 짝이 한쪽 우세 ("model" / "rule") |

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
