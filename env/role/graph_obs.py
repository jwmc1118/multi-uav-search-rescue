"""이종 그래프 관측 사양 — env (env/role/simulation.py rl_observation) 이 만들고 모델 (models/role/net.py) 이 그래프로 조립한다.

UAV i 가 결정할 때의 그래프 = i 의 통신 그룹 belief (그룹 밖 UAV / 그룹이 모르는 생존자는 없음).
좌표는 관측에 절대 정규화 좌표 (frame.normalized = (x / W, y / H)) 로 넣고 모델이 i 기준 상대 좌표로 바꾼다.
같은 이벤트의 순차 결정 중 갱신 (결정함, 역할, 목표, 맡겨짐, 배정 수, 목표 기준 연결) 은 모델 / env 가 결정마다 반영한다.

노드 (종류별 특징, 모델이 만드는 값)
    U (UAV, N)        rel_pos(2), cap / cap_max, role one-hot(3), 목표 있음, goal rel_pos(2), 이번 이벤트 결정함, is_self  = 11
    S (생존자, s_max) rel_pos(2), (τ - t) / T, 맡겨짐                                                              = 4
    K (영역, k_max / 세그먼트, n_sub_max)  rel_pos(2) (영역 중심 / medoid), size_feat(크기), 배정 수 / N         = 4
    D (depot, 1)      rel_pos(2)                                                                                    = 2
간선 (양방향 관계는 방향마다 따로 메시지)
    U–S  BFS(l, s) / d_max, (τ_s - t - BFS) / T, 시간 안에 도달 가능, l 이 s 를 목표로 함                      = 4
    U–K  가장 가까운 셀까지 BFS / d_max, l 이 k 에 배정됨, (2 단계) l 의 goal 셀이 k 안                          = 3
    U–U  현재 거리 / R_c, 목표 지점 거리 / R_c, 현재 통신 연결, 목표 기준 연결 (<= R_c)                          = 4
    U–D  BFS / d_max                                                                                              = 1
    S–S  kNN (knn_k) 두 생존자 사이 BFS / d_max                                                                    = 1
    K–K  영역: kNN, 가장 가까운 셀끼리 BFS / d_max / 세그먼트: 전부, frontier 측지 / d_max                      = 1
그래프 밖 role 입력: g_res (2) = [할당 가능 수 / N_T, 최소 slack], g_exp (3) = [덩어리 수 / K_max, size_feat(frontier 셀 수),
ψ^{f,G} = 그룹 belief 발견 생존자 수 / N_T] (env/role/simulation.py ROLE_G_*_FEATURES, 역할 노드 h_RES / h_EXP 에 붙음).
목표 지점 g_l: RESCUE = 목표 생존자, EXPLORE = 세그먼트 medoid (없으면 goal 셀), RETURN = depot, 없으면 현재 위치.
S 노드 순서 (UAV 마다): i 의 할당 가능 후보 (아직 아무도 맡지 않음 · 시간 안에 닿음, BFS 가까운 순) 먼저 -> 나머지 아는 생존자
(그룹이 맡음 / i 는 못 닿음, BFS 순). rescue 행동 = S 인덱스 (후보 = 앞쪽), 최대 s_max.
"""
import math

FU, FS, FK, FD = 11, 4, 4, 2
EUS, EUK, EUU, EUD, ESS, EKK = 4, 3, 4, 1, 1, 1
DIST_CLIP = 4.0                  # U–U 거리 / R_c 상한
SIZE_REF = 100                   # 크기 특징 기준 셀 수 기본값 (explore.size_ref)


def size_feat(n, c_ref=SIZE_REF):
    """덩어리 / 구간 / frontier 셀 수 특징 = min(1, log(1 + n) / log(1 + C_ref)) — GNN K 노드 · critic largest_cluster_size ·
    role g_exp frontier 항 공통 (작은 덩어리 차이는 살리고 큰 덩어리는 완만하게)."""
    return min(1.0, math.log1p(max(0.0, float(n))) / math.log1p(float(c_ref)))

# 특징 이름 (tests/role/print_obs_spec.py 와 모델 build 순서가 같다)
U_FEATURES = ("rel_x", "rel_y", "cap", "role_explore", "role_rescue", "role_return", "has_goal", "goal_rel_x", "goal_rel_y",
              "decided", "is_self")
S_FEATURES = ("rel_x", "rel_y", "time_left", "held")
K_FEATURES = ("rel_x", "rel_y", "size", "n_assigned")
D_FEATURES = ("rel_x", "rel_y")
EDGE_FEATURES = {"US": ("bfs", "slack", "reachable", "targets"), "UK": ("bfs", "assigned", "goal_in"),
                 "UU": ("dist_cur", "dist_goal", "linked", "goal_linked"), "UD": ("bfs",), "SS": ("bfs",), "KK": ("bfs",)}
assert (len(U_FEATURES), len(S_FEATURES), len(K_FEATURES), len(D_FEATURES)) == (FU, FS, FK, FD)
