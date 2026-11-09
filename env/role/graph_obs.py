"""이종 그래프 관측 사양 — env (env/role/simulation.py rl_observation) 이 재료를 만들고 모델 (models/role/net.py) 이
결정 UAV 마다 그래프로 조립한다. 세 네트워크 (역할 선택 / rescue / explore) 가 같은 노드 · 관계 구성의 그래프를 각자 encoder · GNN 으로 계산.

정보 구조 (belief): UAV i 는 자기 n-hop 통신 그룹의 병합된 지식으로 그래프를 만들고 GNN 은 그 위에서 로컬로 계산한다
(GNN 층 = 통신 라운드가 아님). 그룹이 모르는 생존자는 그래프에 없다. 그룹 밖 UAV 는 마지막 기록이 있으면 메모리 노드로 들어간다.
노드 특징의 좌표는 절대 정규화 (x / X, y / Y) (frame.normalized), 상대 위치는 간선 특징에만 넣는다.
BFS 거리는 모두 BFS / d_max (도달 불가 1). 같은 이벤트의 순차 결정 중 앞 차례 결정 (역할, 목표) 은 모델이 결정마다 반영한다.
depot 은 노드가 아니다 (U 노드의 depot BFS 특징 · σ 의 depot 경유 ETA 로만 들어간다).

노드 (종류별 특징)
    U (UAV)   x, y, c / c_max, 역할 one-hot (3), 목표 위치 (x_g, y_g) (goal_point, 없으면 현재 위치), depot 까지 BFS,
              그룹 연결도 ω^g = (|U_l| - 1) / (|U| - 1), linked ℓ (1 = 현재 그룹 / 0 = 메모리 노드),
              age a = min(1, (t - t_last) / T_age) (연결 노드 0)                                                     = 12
              U 노드 = 현재 그룹 (linked 1) + 기록이 있는 그룹 밖 UAV (linked 0, 메모리 노드), 상한 = 전체 UAV 수
    S (생존자, s_max)  x, y, 골든타임 경과 비율 min(t / τ, 1)                                                      = 3
    F (frontier 덩어리, k_max)  p^f_m = c* 의 x, y, size_feat(셀 수)                                                = 3
              c* = argmin_{c ∈ 덩어리 m 의 도달 가능 셀} BFS(i → c) — i = 이 그래프를 만드는 결정 UAV, BFS · 셀은 U–F 거리와
              같은 장 (i 의 BFS 장, 추정 담당 셀을 뺀 덩어리 셀). 동률이면 셀 인덱스가 작은 쪽. 도달 가능 셀이 없으면 셀 평균
              (fallback, info fpos_fallback). UAV 마다 자기 기준으로 계산하므로 같은 덩어리라도 UAV 마다 위치가 다르다
그룹 context (노드 아님, 결정하는 UAV i 의 그룹 belief): c_i = [ρ_i, η_i, ν_i, t / T_max]                           = 4
    ρ 미발견 비율 (N_T - |D_i|) / N_T, η 미탐색 비율 (|C_free| - |E_i|) / |C_free|, ν 발견 · 미구조 비율 |N_i^s| / N_T,
    t / T_max 에피소드 경과 비율 (t = 결정 시점 step, T_max = max_episode_len, [0, 1] clip) — 같은 그룹이면 같은 값
    — 역할 / 할당 decoder 입력
간선 (엣지 텐서 E[a, b] 는 a 행 · b 열, 상대 위치 = pos_b - pos_a — 받는 노드 기준 보내는 노드의 위치)
    U–U  Δx, Δy, 거리 / R_c (R_c = 두 UAV 통신 반경 중 작은 쪽, DIST_CLIP 상한), 목표 중복 (같은 생존자 RESCUE 목표 /
         같은 덩어리 슬롯 EXPLORE 배정 — 슬롯 중복은 연결 노드끼리만)                                                = 4
    U–S  BFS(l, s), 골든타임 여유 σ = (τ - t - ETA) / τ ∈ [-1, 1] (ETA = BFS(l, s), 용량 0 이면 BFS(l, depot) + BFS(depot, s),
         v = 1 칸 / step), s 가 l 의 현재 목표                                                                         = 3
    U–F  덩어리의 가장 가까운 셀까지 BFS, l 이 지금 배정된 덩어리 (연결 노드만)                                     = 2
    S–S  kNN (knn_k, k 번째와 동률 포함) Δx, Δy, BFS                                                                = 3
    F–F  kNN Δx, Δy (두 덩어리의 p^f 차이), 가장 가까운 셀끼리 BFS                                                   = 3
    S–F  생존자마다 가장 가까운 덩어리 kNN (같은 규칙) [p_m^f - p_k^s (2, p^f = 위 c*), 생존자에서 덩어리의 가장 가까운 셀까지 BFS]  = 3
메모리 노드의 특징 · U–U / U–S / U–F 간선은 마지막 기록 (위치, 용량, 역할, 목표 생존자 / 목표 위치, depot BFS, ω^g, t_last)
값으로 계산 (σ = 기록된 위치 · 용량, BFS 는 지금 그룹 지도). 마스크 · 결정 순서 · ω^g · 이벤트에는 쓰지 않는다 (보상은 task 할당 충돌 판정에만, env/role/simulation.py _check_dup).
네트워크 (models/role/net.py) — 세 네트워크 모두 노드 U · S · F, 관계 S→U, U→S, F→U, U→F, U→U, S→S, F→F, S→F, F→S (U–S, U–F 간선
층마다 갱신), 파라미터는 네트워크마다 따로
    역할 선택   logits = ξ_dec(h_i ‖ ξ_ctx(c_i)) -> 3
    rescue      후보 = S pointer attention (key 에 U–S 간선)
    explore     후보 = F pointer attention (key 에 U–F 간선). 1 단계만 — 2 단계 구간 배정은 env 의 거리 Hungarian
목표 지점 g_l: RESCUE = 목표 생존자, EXPLORE = 세그먼트 medoid (없으면 goal 셀), RETURN = depot, 없으면 현재 위치
(순차 디코딩 중 이번 이벤트에 1 단계를 고른 UAV 는 고른 덩어리의 p^f = 그 UAV 에게 가장 가까운 셀).
S 노드 순서 (UAV 마다): i 의 할당 가능 후보 (아직 아무도 맡지 않음 · 시간 안에 닿음, BFS 가까운 순) 먼저 -> 나머지 아는 생존자
(그룹이 맡음 / i 는 못 닿음, BFS 순), 최대 s_max. 할당 행동 = S 인덱스 (0 .. s_max - 1) 또는 s_max + F 인덱스.
"""
import math

FU, FS, FF, FC = 12, 3, 3, 4
EUU, EUS, EUF, ESS, EFF, ESF = 4, 3, 2, 3, 3, 3
DIST_CLIP = 4.0                  # U–U 거리 / R_c 상한
SIZE_REF = 100                   # 크기 특징 기준 셀 수 기본값 (explore.size_ref)


def size_feat(n, c_ref=SIZE_REF):
    """덩어리 / 구간 / frontier 셀 수 특징 = min(1, log(1 + n) / log(1 + C_ref)) — GNN F 노드 · critic largest_cluster_size
    공통 (작은 덩어리 차이는 살리고 큰 덩어리는 완만하게)."""
    return min(1.0, math.log1p(max(0.0, float(n))) / math.log1p(float(c_ref)))


# 특징 이름 (tests/role/print_obs_spec.py 와 모델 build 순서가 같다)
U_FEATURES = ("x", "y", "cap", "role_explore", "role_rescue", "role_return", "goal_x", "goal_y", "depot_bfs",
              "omega_group", "linked", "age")
S_FEATURES = ("x", "y", "elapsed")
F_FEATURES = ("x", "y", "size")
CONTEXT_FEATURES = ("rho_undiscovered", "eta_unexplored", "nu_found_unrescued", "time_elapsed")
EDGE_FEATURES = {"UU": ("dx", "dy", "dist_rc", "goal_overlap"), "US": ("bfs", "sigma", "is_target"),
                 "UF": ("bfs", "assigned"), "SS": ("dx", "dy", "bfs"), "FF": ("dx", "dy", "bfs"),
                 "SF": ("dx", "dy", "bfs")}
EDGE_DIMS = {k: len(v) for k, v in EDGE_FEATURES.items()}
assert (len(U_FEATURES), len(S_FEATURES), len(F_FEATURES), len(CONTEXT_FEATURES)) == (FU, FS, FF, FC)
assert tuple(EDGE_DIMS[k] for k in ("UU", "US", "UF", "SS", "FF", "SF")) == (EUU, EUS, EUF, ESS, EFF, ESF)
