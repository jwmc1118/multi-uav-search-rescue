"""RoleEnv — RoleSimulation 을 감싸는 gym.Env (학습용, 관측 / 행동 공간 고정).

행동  dict(role=(N,) ∈ {0 EXPLORE, 1 RESCUE, 2 RETURN}, rescue=(N,) ∈ [0, S) (S 노드 = 생존자, S = s_max),
           explore=(N,) ∈ [0, K) (1 단계 덩어리), explore_sub=(N,) ∈ [0, M) (2 단계 구간, M = explore.n_sub_max))
      이번 step 결정이 필요 없는 UAV (role_needed = 0) 의 값과, 고른 역할과 무관한 항목은 무시된다.
      순차 디코딩 (그룹 안 앞 차례 선택 -> 뒤 UAV 그래프 갱신 · 마스크) 은 모델이 policies/decision_order.py 로 차례를 정하며 하고
      env 는 같은 함수 · 같은 관측으로 같은 차례를 재현한다.
보상  스칼라 reward 는 로깅용 합. 헤드별 보상 (reward_role / reward_rescue / reward_explore) 은 info 의 (N,) 배열.

관측 — 이종 그래프 재료 (사양 env/role/graph_obs.py, 모델이 UAV i 기준 그래프로 조립), 명세는 tests/role/print_obs_spec.py
UAV 별 (agents 튜플, 좌표는 절대 정규화 (x / W, y / H), 거리는 / d_max)
    local (S, S, C_l) / role_self (8) / role_group (12)   critic 입력
    role_g_res (2) [할당 가능 수 / N_T, 최소 slack]   role_g_exp (3) [덩어리 수 / K_max, size_feat(frontier 셀 수), 그룹 발견 수 / N_T]
    role_mask (3) / role_needed (1) / explore_kind (1) [2 단계 learned]
    S: s_ids (S,) 생존자 번호 (패딩 -1) / s_pos (S, 2) / s_tau (S,) (τ - t) / T / s_held (S,) 그룹의 다른 UAV 가 맡음 /
       rescue_mask (S,) 할당 가능 후보 / us_d (N, S) BFS(l, s) / us_slack (N, S) (τ - t - BFS) / T / ss_dist (S, S)
    K: k_pos (K, 2) 영역 중심 / k_size (K,) / k_n (K,) 유지 배정 수 / N / uk_d (N, K) / kk_dist (K, K) /
       explore_mask (K,) / explore_cur (1,) 지금 영역 슬롯 / explore_kcap (K,) / explore_size (K,) 셀 수 (결정 순서용)
    세그먼트 표 (덩어리 k 를 kk 구간으로 나눈 결과가 [k, seg_offset(kk) + s], P = Σ_{kk=2..M} kk):
       seg_pos (K, P, 2) medoid / seg_sz (K, P) / useg_d (K, P, N) / useg_goal (K, P, N) l 의 goal 셀이 구간 안 /
       explore_seg_mask (K, P) / explore_seg_size (K, P) 셀 수 / explore_seg_dist (K, P, M) 구간 간 측지 / d_max
팀
    team_ctx (N, 8) [role one-hot, goal 셀 x, y, 결정 대상, 위치 x, y] / team_cap (N,) / team_comm (N,) R_c /
    team_target (N,) 목표 생존자 (-1) / team_slot (N,) 영역 슬롯 (-1) / team_gpos (N, 2) 목표 지점 g_l / team_has_goal (N,) /
    team_depot_d (N,) / depot_pos (2,) / geo (3,) [W, H, d_max] / adj (N, N) 현재 통신 링크 / group (N, N) / reach (N, N) /
    time (1,) / state (gh, gw, 6) (+ 목표 지점 채널) / critic_tasks (T, 9) / critic_task_present (T,) / critic_team (N, 12) /
    critic_global (5,)
"""
import numpy as np
import gym
from gym import spaces

from env.role.simulation import (RoleSimulation, N_ROLES, ROLE_SELF_DIM, ROLE_GROUP_DIM, EXPLORE_KIND_DIM, TEAM_CTX_DIM,
                                 ROLE_G_RES_FEATURES, ROLE_G_EXP_FEATURES, STATE_CHANNELS)


class RoleEnv(gym.Env):
    def __init__(self, config=None):
        self.sim = RoleSimulation(config or {})
        sim = self.sim
        n, S, K, M, P = sim.n_agents, sim.s_max, sim.k_max, sim.n_sub_max, sim.n_seg_slots
        T = sim.max_targets
        s = sim.cfg["state_size"]
        C_l = len(sim.local_ch_idx)
        big = float(sim.rows * sim.cols)
        box = lambda lo, hi, shape: spaces.Box(lo, hi, shape, np.float32)
        agent_space = spaces.Dict({
            "local": box(0.0, 1.0, (s, s, C_l)),
            "role_self": box(-1.0, 1.0, (ROLE_SELF_DIM,)),
            "role_group": box(0.0, 1.0, (ROLE_GROUP_DIM,)),
            "role_g_res": box(-1.0, 1.0, (len(ROLE_G_RES_FEATURES),)),
            "role_g_exp": box(0.0, 1.0, (len(ROLE_G_EXP_FEATURES),)),
            "role_mask": box(0.0, 1.0, (N_ROLES,)),
            "role_needed": box(0.0, 1.0, (1,)),
            "explore_kind": box(0.0, 1.0, (EXPLORE_KIND_DIM,)),
            "s_ids": box(-1.0, np.inf, (S,)),
            "s_pos": box(-1.0, 1.0, (S, 2)),
            "s_tau": box(-1.0, 1.0, (S,)),
            "s_held": box(0.0, 1.0, (S,)),
            "rescue_mask": box(0.0, 1.0, (S,)),
            "us_d": box(0.0, 1.0, (n, S)),
            "us_slack": box(-1.0, 1.0, (n, S)),
            "ss_dist": box(0.0, 1.0, (S, S)),
            "k_pos": box(-1.0, 1.0, (K, 2)),
            "k_size": box(0.0, 1.0, (K,)),
            "k_n": box(0.0, 1.0, (K,)),
            "uk_d": box(0.0, 1.0, (n, K)),
            "kk_dist": box(0.0, 1.0, (K, K)),
            "explore_mask": box(0.0, 1.0, (K,)),
            "explore_cur": box(-1.0, float(K), (1,)),
            "explore_kcap": box(0.0, float(M), (K,)),
            "explore_size": box(0.0, big, (K,)),
            "seg_pos": box(-1.0, 1.0, (K, P, 2)),
            "seg_sz": box(0.0, 1.0, (K, P)),
            "useg_d": box(0.0, 1.0, (K, P, n)),
            "useg_goal": box(0.0, 1.0, (K, P, n)),
            "explore_seg_mask": box(0.0, 1.0, (K, P)),
            "explore_seg_size": box(0.0, big, (K, P)),
            "explore_seg_dist": box(0.0, 1.0, (K, P, M)),
        })
        self.observation_space = spaces.Dict({
            "agents": spaces.Tuple(tuple(agent_space for _ in range(n))),
            "team_ctx": box(-1.0, 1.0, (n, TEAM_CTX_DIM)),
            "team_cap": box(0.0, 1.0, (n,)),
            "team_comm": box(0.0, big, (n,)),
            "team_target": box(-1.0, np.inf, (n,)),
            "team_slot": box(-1.0, float(K), (n,)),
            "team_gpos": box(-1.0, 1.0, (n, 2)),
            "team_has_goal": box(0.0, 1.0, (n,)),
            "team_depot_d": box(0.0, 1.0, (n,)),
            "depot_pos": box(-1.0, 1.0, (2,)),
            "geo": box(0.0, np.inf, (3,)),
            "adj": box(0.0, 1.0, (n, n)),
            "group": box(0.0, 1.0, (n, n)),
            "reach": box(0.0, 1.0, (n, n)),
            "time": box(0.0, 1.0, (1,)),
            "state": box(0.0, 1.0, (sim.gh, sim.gw, STATE_CHANNELS)),
            "critic_tasks": box(-1.0, 1.0, (T, RoleSimulation.CRITIC_TASK_DIM)),
            "critic_task_present": box(0.0, 1.0, (T,)),
            "critic_team": box(-1.0, 1.0, (n, RoleSimulation.CRITIC_TEAM_DIM)),
            "critic_global": box(0.0, 1.0, (RoleSimulation.CRITIC_GLOBAL_DIM,)),
        })
        self.action_space = spaces.Dict({
            "role": spaces.Tuple(tuple(spaces.Discrete(N_ROLES) for _ in range(n))),
            "rescue": spaces.Tuple(tuple(spaces.Discrete(S) for _ in range(n))),
            "explore": spaces.Tuple(tuple(spaces.Discrete(K) for _ in range(n))),
            "explore_sub": spaces.Tuple(tuple(spaces.Discrete(M) for _ in range(n))),
        })
        self.n_agents, self.s_max, self.max_targets, self.k_max, self.n_sub_max = n, S, T, K, M

    def seed(self, seed=None):
        self.sim.random_state = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.sim.reset()
        return self.sim.rl_observation()

    def step(self, action):
        done, info = self.sim.step_explicit({k: np.asarray(action[k]).reshape(-1)
                                             for k in ("role", "rescue", "explore", "explore_sub")})
        scalar = float(np.sum(info["reward_role"]) + np.sum(info["reward_rescue"]) + np.sum(info["reward_explore"]))
        return self.sim.rl_observation(), scalar, done, info
