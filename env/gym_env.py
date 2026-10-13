"""RescueMultiEnv — RescueSimulation 을 감싸는 RLlib 학습용 gym.Env.

Role_selection 의 CoverageEnv 패턴: 팀 전체가 하나의 정책이고, 관측/행동에 per-agent 축(N)이
들어 있으며, 계층별 보상은 스칼라 reward 가 아니라 info 로 나간다
(training/multiagent_ppo.py 가 info 에서 alloc / primitive 보상과 alloc_valid 를 꺼낸다).

노드 슬롯 A = T + 2 : [태스크 0..T-1 | explore (T) | depot (T+1)]

관측 (per agent) — 키 이름은 models/rescue_net.py 가 그대로 읽는다
    local         (S, S, C_l) 자기 중심 크롭, cfg["local_channels"] 만 (회전 없음, 북쪽 = 위) -> Primitive CNN, critic
    task_feats    (T, 4)     δ^i_j = [x/W, y/H, τ_j, d^i_j]                              -> Encoder (CCN)
    task_present  (T,)       실제 후보 = 1 (패딩 = 0)                                     -> k-NN / MHA key 마스크
    task_ids      (T,)       sim.targets index (패딩 -1)                                   -> 순차 디코딩 그룹 충돌
    explore_feats (4,)       δ^i_e = [T^i/N_T, E^i_t/|C_free|, d^i_e, g^i]                -> Encoder (W_exp)
                             g^i = (N_T - D^i)/N_T 못 찾은 조난자 비율 (D^i = i 가 아는 발견 수, 구조·만료 포함)
    depot_feats   (3,)       δ^i_d = [x_d/W, y_d/H, d^i_d]                                -> Encoder (W_depot)
    self_feats    (3,)       ξ^i = [x/W, y/H, c/c_max]                                    -> Context
    node_mask     (A,)       admissible 𝒜^i (lock 이면 강제 슬롯만 1)
    node_xy       (A, 2)     슬롯별 목적지 정규화 좌표 -> 이웃 메시지 g^{i'}
    node_type     (A,)       슬롯별 메시지 타입 o 인덱스 (env/modes.py)
    node_wp       (A, 3)     슬롯별 local waypoint [dx, dy, valid] (÷ 창 반폭)        -> Primitive 2D goal 채널
    free_choice   (1,)       1 = 정책이 고름 / 0 = lock 강제 (alloc loss 마스크)
공용 (팀 테이블 — 모델이 reach 로 n-hop 이웃만 사용)
    pub (N, 3) [x/W, y/H, c/c_max] | prev_msg (N, 6) [g_x, g_y, o(4)] 이전 결정 | reach (N, N)
    dec_rank (N,) 순차 디코딩 순서 | time (1,) t/t_max | state (gh, gw, C) critic 전용
local 채널 = cfg["local_channels"] 순서 (기본 obstacle, agents). waypoint 채널은 모델이 그려 붙인다.
state 채널 = env.knowledge.N_CHANNELS (obstacle / explored / frontier / target / agents) 전부
"""
import numpy as np
import gym
from gym import spaces

from env.simulation import RescueSimulation
from env.agent import Robot
from env.knowledge import N_CHANNELS
from env.modes import N_MSG_TYPES

EXPLORE_DIM = 4                    # δ^i_e = [T^i/N_T, E^i_t/|C_free|, d^i_e, g^i]

N_PRIMITIVE = len(Robot.ACTIONS)   # 9 = 8방향 + stay


class RescueMultiEnv(gym.Env):
    def __init__(self, config=None):
        self.sim = RescueSimulation(config or {})
        n = self.sim.n_agents
        T = self.sim.max_targets
        A = self.sim.n_actions_alloc
        s = self.sim.cfg["state_size"]
        gh, gw = self.sim.gh, self.sim.gw
        C = N_CHANNELS
        C_l = len(self.sim.local_ch_idx)
        inf = np.inf

        agent_space = spaces.Dict({
            "local": spaces.Box(0.0, 1.0, (s, s, C_l), np.float32),
            "task_feats": spaces.Box(-1.0, 1.0, (T, 4), np.float32),
            "task_present": spaces.Box(0.0, 1.0, (T,), np.float32),
            "task_ids": spaces.Box(-1.0, inf, (T,), np.float32),
            "explore_feats": spaces.Box(0.0, 1.0, (EXPLORE_DIM,), np.float32),
            "depot_feats": spaces.Box(-1.0, 1.0, (3,), np.float32),
            "self_feats": spaces.Box(-1.0, 1.0, (3,), np.float32),
            "node_mask": spaces.Box(0.0, 1.0, (A,), np.float32),
            "node_xy": spaces.Box(-1.0, 1.0, (A, 2), np.float32),
            "node_type": spaces.Box(0.0, float(N_MSG_TYPES - 1), (A,), np.float32),
            "node_wp": spaces.Box(-1.0, 1.0, (A, 3), np.float32),
            "free_choice": spaces.Box(0.0, 1.0, (1,), np.float32),
        })
        self.observation_space = spaces.Dict({
            "agents": spaces.Tuple(tuple(agent_space for _ in range(n))),
            "pub": spaces.Box(-1.0, 1.0, (n, 3), np.float32),
            "prev_msg": spaces.Box(-1.0, 1.0, (n, 2 + N_MSG_TYPES), np.float32),
            "reach": spaces.Box(0.0, 1.0, (n, n), np.float32),
            "dec_rank": spaces.Box(0.0, float(n), (n,), np.float32),
            "time": spaces.Box(0.0, 1.0, (1,), np.float32),
            "state": spaces.Box(0.0, 1.0, (gh, gw, C), np.float32),
        })
        # gym.spaces.Dict 는 키를 정렬한다 -> 평탄화 순서 = [alloc | primitive]
        self.action_space = spaces.Dict({
            "alloc": spaces.Tuple(tuple(spaces.Discrete(A) for _ in range(n))),
            "primitive": spaces.Tuple(tuple(spaces.Discrete(N_PRIMITIVE) for _ in range(n))),
        })
        self.n_agents = n
        self.max_targets = T
        self.n_alloc = A

    def seed(self, seed=None):
        self.sim.random_state = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.sim.reset()
        return self.sim.rl_observation()

    def step(self, action):
        done, info = self.sim.step_explicit(
            np.asarray(action["alloc"]).reshape(-1),
            np.asarray(action["primitive"]).reshape(-1),
        )
        # RLlib scalar reward: 로깅용 (실제 학습 신호는 info 의 계층별 보상)
        scalar = float(np.sum(info["reward_primitive"]) + np.sum(info["reward_alloc"]))
        return self.sim.rl_observation(), scalar, done, info
