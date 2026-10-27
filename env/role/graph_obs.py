"""이종 그래프 관측 사양 — env (env/role/simulation.py rl_observation) 이 재료를 만들고 모델 (models/role/net.py) 이
결정 UAV 마다 그래프로 조립한다. 세 네트워크 (역할 선택 / 생존자 할당 / frontier 할당) 가 각자 encoder · GNN 을 가진다.

정보 구조 (belief): UAV i 는 자기 n-hop 통신 그룹의 병합된 지식으로 그룹 전체 그래프를 만들고 GNN 은 그 위에서 로컬로 계산한다
(GNN 층 = 통신 라운드가 아님). 그룹 밖 UAV / 그룹이 모르는 생존자는 그래프에 없다.
노드 특징의 좌표는 절대 정규화 (x / X, y / Y) (frame.normalized), 상대 위치는 간선 특징에만 넣는다.
BFS 거리는 모두 BFS / d_max (도달 불가 1). 같은 이벤트의 순차 결정 중 앞 차례 결정 (역할, 목표) 은 모델이 결정마다 반영한다.

노드 (종류별 특징)
    U (UAV, 그룹)    x, y, c / c_max, ρ 미발견 비율 (N_T - |D_l|) / N_T, η 미탐색 비율 (|C_free| - |E_l|) / |C_free|,
                     ν 발견 · 미구조 비율 |N_l^s| / N_T, 역할 one-hot (3), 목표 위치 (x_g, y_g) (goal_point, 없으면 현재 위치),
                     depot 까지 BFS                                                                                   = 12
    S (생존자, s_max) x, y, 골든타임 경과 비율 min(t / τ, 1)                                                        = 3
    K (frontier 덩어리, k_max)  셀 중심 x, y, size_feat(셀 수)                                                       = 3
간선 (엣지 텐서 E[a, b] 는 a 행 · b 열, 상대 위치 = pos_b - pos_a — 받는 노드 기준 보내는 노드의 위치)
    U–U  Δx, Δy, 거리 / R_c (R_c = 두 UAV 통신 반경 중 작은 쪽, DIST_CLIP 상한), 목표 중복 (같은 생존자 RESCUE 목표 /
         같은 덩어리 슬롯 EXPLORE 배정)                                                                                = 4
    U–S  BFS(l, s), 골든타임 여유 σ = (τ - t - ETA) / τ ∈ [-1, 1] (ETA = BFS(l, s), 용량 0 이면 BFS(l, depot) + BFS(depot, s),
         v = 1 칸 / step), s 가 l 의 현재 목표                                                                         = 3
    U–K  덩어리의 가장 가까운 셀까지 BFS, l 이 지금 배정된 덩어리                                                     = 2
    S–S  kNN (knn_k, k 번째와 동률 포함) Δx, Δy, BFS                                                                = 3
    K–K  kNN Δx, Δy (덩어리 중심 간), 가장 가까운 셀끼리 BFS                                                         = 3
네트워크별 그래프
    역할 선택      U (그룹 전체) · S, 관계 S→U, U→S, U→U, S→S
    생존자 할당    U (그룹의 RESCUE UAV, 이번 이벤트 앞 차례 포함) · S, 관계 S→U, U→S, U→U, S→S (+ U–S 간선 층마다 갱신)
    frontier 할당  U (그룹의 EXPLORE UAV) · K, 관계 K→U, U→K, U→U, K→K (1 단계만, 2 단계 구간 배정은 env 의 κ Hungarian)
목표 지점 g_l: RESCUE = 목표 생존자, EXPLORE = 세그먼트 medoid (없으면 goal 셀), RETURN = depot, 없으면 현재 위치
(순차 디코딩 중 이번 이벤트에 1 단계를 고른 UAV 는 고른 덩어리 중심).
S 노드 순서 (UAV 마다): i 의 할당 가능 후보 (아직 아무도 맡지 않음 · 시간 안에 닿음, BFS 가까운 순) 먼저 -> 나머지 아는 생존자
(그룹이 맡음 / i 는 못 닿음, BFS 순). rescue 행동 = S 인덱스 (후보 = 앞쪽), 최대 s_max.
"""
import math

FU, FS, FK = 12, 3, 3
EUU, EUS, EUK, ESS, EKK = 4, 3, 2, 3, 3
DIST_CLIP = 4.0                  # U–U 거리 / R_c 상한
SIZE_REF = 100                   # 크기 특징 기준 셀 수 기본값 (explore.size_ref)


def size_feat(n, c_ref=SIZE_REF):
    """덩어리 / 구간 / frontier 셀 수 특징 = min(1, log(1 + n) / log(1 + C_ref)) — GNN K 노드 · critic largest_cluster_size ·
    role g_exp frontier 항 공통 (작은 덩어리 차이는 살리고 큰 덩어리는 완만하게)."""
    return min(1.0, math.log1p(max(0.0, float(n))) / math.log1p(float(c_ref)))


# 특징 이름 (tests/role/print_obs_spec.py 와 모델 build 순서가 같다)
U_FEATURES = ("x", "y", "cap", "rho_undiscovered", "eta_unexplored", "nu_found_unrescued",
              "role_explore", "role_rescue", "role_return", "goal_x", "goal_y", "depot_bfs")
S_FEATURES = ("x", "y", "elapsed")
K_FEATURES = ("x", "y", "size")
EDGE_FEATURES = {"UU": ("dx", "dy", "dist_rc", "goal_overlap"), "US": ("bfs", "sigma", "is_target"),
                 "UK": ("bfs", "assigned"), "SS": ("dx", "dy", "bfs"), "KK": ("dx", "dy", "bfs")}
EDGE_DIMS = {k: len(v) for k, v in EDGE_FEATURES.items()}
assert (len(U_FEATURES), len(S_FEATURES), len(K_FEATURES)) == (FU, FS, FK)
assert (EDGE_DIMS["UU"], EDGE_DIMS["US"], EDGE_DIMS["UK"], EDGE_DIMS["SS"], EDGE_DIMS["KK"]) == (EUU, EUS, EUK, ESS, EKK)
