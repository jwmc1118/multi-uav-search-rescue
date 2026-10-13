> **참고:** 이 문서는 이전 3계층(role / task / primitive) 설계 기록이다. 현재 구조(통합 할당 + primitive)는 `README.md` 참고.

# Hierarchical Multi-Robot Reinforcement Learning Framework for Search and Rescue
## 연구 방향 상세 정리 (working title 기준)

> 단일 기종(Ground Vehicle, 다수) 기반. Multicopter/이기종 확장은 현재 범위에서 제외.

---

## 1. 연구 개요

### 1.1 목표
다수의 동일 기종 로봇이 미지의 재난 지역에서 **탐색(explore) → 임무 할당(task allocation) → 원시 행동(primitive action)**의 3단계 계층 구조로 학습하여, 골든타임 내 최대한 많은 구조자를 구조하는 것을 목표로 한다.

### 1.2 기반 논문
| 논문 | 역할 |
|---|---|
| *Autonomous and Adaptive Role Selection* (Zhu & Liu) | Role selection 기본 프레임(explore/cover), role reward $R_t=\alpha R_e+\beta R_c$, loss 분리($L=L_p+L_r$) 구조 |
| *Multi-Robot Environmental Coverage with a Two-Stage Coordination Strategy* (Zhu et al.) | AGNN 기반 n-hop 통신 집계, local perception 중심 설계, 맵 크기 일반화 실험 근거 |
| *Learning to Allocate Time-Bound and Dynamic Tasks to Multiple Robots Using Covariant Attention Neural Networks* (CapAM) | Capacity/depot 제약 하의 attention 기반 task allocation, Query-Key-Value 설계, 목적함수, 마스킹 규칙 |
| 코드베이스 `linaug/Role_selection` | Role/Primitive critic 완전 분리 학습 파이프라인 실제 구현 확인 |

### 1.3 핵심 기여 포인트 (novelty)
CapAM 논문은 모든 태스크(구조자) 위치와 마감시한이 로봇에게 처음부터 알려져 있다고 가정하고, 탐색(exploration) 개념이 없다. 본 연구는 여기에 **① 미지 환경에서의 탐색을 통한 동적 타겟 발견**, **② grid 기반 공간 표현(CNN+AGNN)과 CapAM 스타일 attention 할당의 결합**, **③ 계층적 role selection과 capacity 제약의 통합**을 추가한다.

---

## 2. Role 정의

### 2.1 Role 3종
| Role | 설명 | 실행 가능 조건 |
|---|---|---|
| Explore | Frontier cell로 이동 | 항상 가능 |
| Rescue (=coverage) | Task allocation을 통해 배정된 target cell로 이동, 도달 시 완료 | **잔여 구호품 $c_i>0$일 때만 가능 (하드 마스킹)** |
| Return | Depot(고정 좌표)으로 이동, 도달 시 구호품 완전 재보충 | 항상 가능 (강제 아님, 정책의 전략적 선택) |

- 구호품이 0개여도 explore/return 중 자유롭게 선택 가능. Rescue만 물리적으로 불가능한 행동이므로 **role action space에서 원천 제거**.
- CapAM 논문의 depot 처리 방식(제약 위반 시에만 강제 방문, 그 외엔 depot 선택 확률이 항상 0보다 큼)과 달리, 본 연구는 return을 완전히 학습된 전략적 선택으로 둔다 — 원 논문보다 더 어려운 학습 문제(복귀 타이밍까지 스스로 학습)가 되므로 role reward 설계에 유의.

### 2.2 Role action vs Role state 구분
- **Action** (정책이 직접 샘플링): $\rho_i \in \{0(\text{explore}), 1(\text{rescue 희망})\}$ — return도 포함하면 $\{0,1,2\}$이지만 rescue의 "할당 여부"는 아래처럼 별도 축.
- **State** (환경이 파생 계산, 트리거/로깅용): rescue 의도 시 $\{$미할당$, $할당됨$\}$으로 세분화되어 task allocation 트리거 조건으로 사용.

### 2.3 Role reward
원 논문 방식을 따라:
$$R_t = \alpha R_e + \beta R_c \quad (\alpha+\beta=1)$$
여기에 **return 관련 항**을 추가로 설계 필요 (open item, §8 참고) — 예: 성공적 재보충 후 이어진 rescue 성공 횟수에 대한 지연 보상.

---

## 3. 관측 공간 (Multi-resolution Observation)

### 3.1 좌표계
고정 프레임(allocentric). 각 로봇은 자신만의 개별 누적 global map을 계속 업데이트하며, 팀 전체 맵을 병합하지 않는다 (통신은 AGNN을 통해서만 특징 수준에서 이루어짐).

### 3.2 해상도 이원화
| 맵 | 해상도 | 용도 |
|---|---|---|
| Local perception | 고해상도 | 현재 FOV 순간 관측, Primitive policy의 국소 장애물 회피용 |
| Global map (개별 누적) | 저해상도 (다운샘플링, 배율은 실험으로 결정) | Role policy의 거시적 판단용 |

멀티해상도 설계는 grid map + 로봇 주변 고정 크기 윈도우를 결합한 이 방식은, coarse(전역) 판단과 fine(국소) 실행을 분리하는 표준적인 계층형 내비게이션 패턴과 부합한다. Two-stage 논문의 실험 결과, local observation 정보만으로도 의사결정에 충분하고 global state map은 중복 정보를 가질 수 있음이 확인된 바 있어, **저해상도 축소가 성능에 미치는 영향은 ablation 실험 필요** (open item).

### 3.3 채널 구성
| 채널 | 정보 | 값 | 비고 |
|---|---|---|---|
| 1 | Obstacle | binary | 누적, 미탐색 영역 마스킹 |
| 2 | Explored | binary | 누적, 단조증가 (unexplored는 여집합으로 암묵 표현) |
| 3 | Target/rescuer | binary | 누적, 구조 완료 시 갱신 필요 (broadcast 방식 미정, open item) |
| 4 | Self-position | binary | 순간(instant), 고정 프레임 기준 |

- 4채널은 concat이 아니라 **(4,H,W) 텐서로 stack**하여 하나의 CNN에 통과 (RGB 이미지와 동일 원리).
- 범주형 정보는 binary/one-hot, 연속형 정보(거리·시간)는 스칼라 그대로 사용하는 원칙 적용.

### 3.4 Raw scalar state (CNN을 거치지 않는 정밀 정보)
- 자기 위치 $(x_i,y_i)$: **맵 크기 대비 정규화** $(\hat x_i,\hat y_i)=(x_i/W, y_i/H)$ — zero-shot 일반화(다른 맵 크기)를 위한 필수 처리.
- 잔여 구호품 $c_i$
- CNN의 pooling은 신경망 출력을 이미지 내 작은 국소적 변화에 둔감하게 만드는 것이 목적이라, 위치처럼 "정밀한 산술값"이 중요한 경우 CNN 출력에만 의존하면 정밀도가 손실된다 — 따라서 위치·capacity는 grid 채널과 별개로 raw 값을 유지.

---

## 4. 네트워크 구조 (Ground Vehicle, 단일 기종)

```
Local perception(고해상도) ──CNN──────────────────────────────────┐
                                                                    │
Global map(저해상도, 개별 누적) ──CNN──▶ h_i ──AGNN(N-hop)──▶ h̄_i  │
                                              │                    │
                                              ▼                    │
                                        Role policy ──▶ ρ_i        │
                                                                    │
[Raw data: 자신+N-hop 이웃의 위치·capacity] ──┐                    │
                                              ▼                    │
                        h̄_i ⊕ Raw data ──▶ Task policy ──▶ T_i     │
                        (trigger: role state=1일 때만 실행)          │
                                                                    ▼
                        ρ_i ⊕ T_i ⊕ CNN(local) ⊕ h̄_i ──▶ Primitive policy ──▶ Primitive action
```
(⊕ = concatenation)

각 GV는 독립된 파라미터를 가지며, parameter sharing 여부는 별도 결정 사항(원 논문은 동일 role space 로봇 간 공유).

### 4.1 AGNN (통신/집계)
Two-stage 논문 방식 그대로 채택:
$$\bar H = f(H,S) = \sum_{l=0}^{L} S^l H W_l$$
- $H\in\mathbb{R}^{N\times d}$: 로봇별 CNN 출력 (global map 특징벡터)을 쌓은 행렬
- $S\in\mathbb{R}^{N\times N}$: $S_{ij}=1$ if Manhattan distance $\le r_{comm}$
- $W_l\in\mathbb{R}^{d\times d}$: hop별 학습 가중치
- $\bar H_i=\bar h^i\in\mathbb{R}^d$: 행렬 $\bar H$의 $i$번째 행 = 로봇 $i$의 최종 특징벡터 (Query로 재사용)

### 4.2 Primitive policy 입력 (멀티해상도 결합)
$$\text{input}_i = \text{Concat}(\text{CNN}(\text{local perception}),\ \bar h^i,\ \rho_i,\ T_i)$$
$T_i$는 CNN 등으로 압축하지 않고 **정밀한 좌표값 그대로** 사용 (§3.4 원칙과 동일).

---

## 5. Task Allocation

### 5.1 트리거 조건
1. 자신의 role state가 "rescue, 미할당"이 될 때
2. AGNN으로 연결된 그룹 내에 동일 구조자를 배정받은 에이전트가 존재할 때 (충돌 재트리거)

### 5.2 후보 집합
N-hop 범위 내에서 지금까지 발견된 타겟만 후보 (통신 두절 지역의 타겟은 후보에서 제외). 후보 0개 시 no-op 더미 옵션 자동 선택 (중립 처리, 보상/페널티 없음).

### 5.3 Query / Key / Attention 설계 (CapAM 방식 채택)

**Encoder (Key/Value 생성)** — CapAM의 covariant compositional network 방식을 간소화하여, 각 타겟 $j$의 속성 $\delta_j=[x_j,y_j,\tau_j]$(정규화 좌표 + 골든타임)를 선형변환:
$$F_{\delta_j} = W_d \delta_j^T + b_d,\quad \text{Key}_j = \text{Value}_j = F_{\delta_j}$$

**Context/Query 생성** — CapAM의 context 구성 방식을 그대로 채택:
$$Q_i = W_Q \times \text{Concat}(t,\ Q_{r,i},\ Q_{P_i}) + b_Q$$
$$Q_{r,i} = W_{Q_r}\times[\hat x_i,\hat y_i,c_i] + b_{Q_r} \quad \text{(자기 raw state)}$$
$$Q_{P_i} = \sum_{p\in N_i} W_{P}\times[\hat x_p,\hat y_p,c_p] + b_{P} \quad \text{(이웃 raw state, 순서·개수 무관 합산)}$$
여기에 본 연구에서 추가로 AGNN 출력(장애물/지형 맥락)을 결합:
$$Q_i \leftarrow \text{Concat}(Q_i,\ \bar h^i)$$
- 이웃 raw state를 합산(summation) 처리하는 이유는, 이 연산이 context vector를 로봇 수와 무관하게(agnostic) 만들어주기 때문 (CapAM 논문 근거).
- $\bar h^i$를 별도 추가하는 이유는, CapAM 문제 설정과 달리 본 연구 환경엔 장애물이 존재하여 raw 좌표·거리만으로는 실제 이동 비용(우회 필요 여부)을 반영할 수 없기 때문. Ablation으로 기여도 검증 필요 (open item).

**Attention 계산** (Multi-Head Attention, CapAM/Kool et al. 방식):
$$\text{Attention}(K,V,Q) = \text{softmax}\left(\frac{Q^TK}{\sqrt{d_{embed}}}\right)V^T$$
$$\text{MHA}(K,V,Q) = W_{MHA}\times\text{Concat}(\text{head}_1,...,\text{head}_{h_e}) + b_{MHA}$$
후보 타겟 수만큼 동적 크기의 softmax → 전체 타겟 수를 몰라도 되는 구조 (attention은 입력 개수에 자동으로 맞춰짐).

### 5.4 마스킹 (CapAM 방식)
CapAM은 의사결정 시점에 가능한 모든 태스크 중, 이미 선택되었거나 완료된 태스크, 마감시한이 지난 태스크, 마감 전 도달 불가능한 태스크 등을 확률 0으로 마스킹하며, 이는 마스킹된 노드의 logit을 $-\infty$로 설정하는 방식으로 구현된다. 본 연구에서는 특히 **오토리그레시브 충돌 회피용 마스킹**(이미 다른 로봇이 broadcast로 선점 선언한 타겟)을 이 원칙에 따라 추가.

### 5.5 충돌 해결 — 오토리그레시브 (분산형)
- 결정한 로봇이 즉시 AGNN 통신 채널에 선택 결과를 broadcast
- 같은 순간 트리거된 다른 로봇은 이미 broadcast된 타겟을 마스킹한 상태에서 자기 차례에 결정
- 순서는 우선순위(예: 타겟까지 거리)로 결정 — tie-breaking과 순서 결정을 동시에 해결
- 참고: MAT(Multi-Agent Transformer)의 마스킹된 어텐션 기반 순차 생성, Multi-Agent Pointer Transformer의 이전 결정 참조 구조

### 5.6 보상 (MAGNNET 스타일 + CapAM 목적함수 참고)
CapAM의 목적함수는 임무 완료율(사망하지 않고 마감시한 내 구조된 비율)을 최우선으로 최소화하도록 설계되며, 완료율이 100%가 아닐 때만 이동 거리 항이 작동한다:
$$f_{cost} = \psi - u(\psi)e^{-d_\psi}, \quad \psi=(N-N_{success})/N$$

본 연구의 task-level 보상:
$$r_{task}(i) = -\lambda_1\cdot(\text{구조 완료까지 걸린 시간}) - \lambda_2\cdot\mathbb{1}[\text{동시 중복 선택}] \;(+\lambda_3\cdot r_{idle}\text{, 정규화 방식 보류})$$

---

## 6. Primitive Action

멀티해상도 결합 입력(§4.2)을 받아 실제 grid 이동(상하좌우/정지 등)을 결정. Rescue role일 때는 $T_i$(task policy가 고른 target 좌표), Return일 때는 depot 고정 좌표, Explore일 때는 role policy가 별도로 정한 frontier 좌표를 목적지로 사용 (목적지 산출 로직은 §2.1 role별 정의에 귀속).

---

## 7. 학습 방식

### 7.1 CTDE (Centralized Training, Decentralized Execution)
Actor(정책)는 로봇 자신의 로컬 관측(+AGNN 통신으로 확장된 관측)만으로 독립적으로 액션을 출력하고, Critic(가치함수)은 학습 시에만 여러 로봇의 정보를 함께 보며 실행 시에는 불필요해진다.

### 7.2 계층별 완전 분리 (원 논문 코드 확인됨)
`multiagent_ppo.py` 확인 결과, Role critic과 Primitive critic은 서로 다른 SampleBatch 필드(`REWARDS` vs `REWARDS_ROLE`), 서로 다른 GAE/advantage/value target을 가지며, 최종 loss는 $L=L_p+L_r$로 합쳐져 한 번의 backward로 업데이트된다. **Task policy도 동일한 방식으로 별도 critic/advantage($L_{task}$)를 추가**하여 $L=L_p+L_r+L_{task}$로 확장 예정 (세부 학습 세팅은 open item).

---

## 8. Zero-shot 일반화를 위한 설계 원칙

1. **좌표/거리 정규화**: 맵 크기 대비 비율로 정규화(§3.4) — 다른 맵 크기에도 학습된 위치 관계가 그대로 재사용되도록 함.
2. **CNN 출력 크기 독립성**: Flatten+FC 대신 **Global Average Pooling(GAP)**으로 CNN을 종단하면 맵 크기 변화에도 재학습 없이 forward 가능 (단, 학습 안 해본 스케일에서의 성능은 별도 검증 필요). 혹은 고정 canonical 크기+패딩 방식과 병행 검토.
3. Two-stage 논문은 로봇 주변 고정 크기 local FOV만 사용해서 24×24~70×70까지 재학습 없이 일반화를 검증한 바 있음 — 본 연구의 저해상도 global map 설계가 이 정도 일반화력을 유지하는지 실험 필요.

---

## 9. 아직 결정되지 않은 사항 (Open Items)

1. Return 관련 role reward 세부 설계 (재보충 후 이어진 구조 성공에 대한 지연 보상 형태)
2. $r_{idle}$ 정규화 방식 (비율 vs delta) — 로봇 수<타겟 수 시 영구 미달성 문제 해결 필요
3. 골든타임 $\tau_j$의 정확한 인코딩 방식 (정규화 스케일, 발견 시점 대비 경과 계산 방식)
4. Task policy critic/advantage의 구체적 학습 세팅 (§7.2)
5. 다른 로봇이 구조 완료한 타겟 정보를 개별 누적 맵/task 후보 리스트에서 어떻게 "잊을지"(broadcast 방식)
6. Global map 저해상도 배율 결정 (실험적으로 결정 예정, §3.2)
7. $\bar h^i$(AGNN 장애물 맥락)가 Task policy Query에 실제로 기여하는지 ablation 검증
8. Rescue 완료 시 target map 채널 갱신 로직 및 AGNN을 통한 "타겟 소멸" 정보 전파 방식
