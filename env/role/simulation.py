"""RoleSimulation — 계층 구조 env:  Role 선택 -> (Rescue 할당 | Explore 할당 2 단계 | Return) -> BFS 경로 추종.

맵 / 조난자 / 센싱·통신 병합 / 믿음 / 구조·재보충·마감 처리는 env/core/base_sim.py BaseSimulation 을 그대로 쓰고,
여기서는 결정 구조와 이동 · 보상 · 관측을 정한다.

이동    매 step 각 UAV 는 목표점까지 자기 (그룹 병합) 지도 BFS 최단경로의 다음 칸으로 1 칸. 알려진 장애물만 막고
        미탐색은 통과 가능으로 가정. 실제 장애물에 막히면 제자리. 목표 없으면 제자리. 같은 칸 허용 (info 에 횟수만).
역할    EXPLORE / RESCUE / RETURN (env.core.modes.Mode). 역할 마스크 (role.return_with_cap: true, 부분 귀환):
            cap = 0                                       RETURN, EXPLORE
            0 < cap < cap_max, 부족분 >= return_min_deficit  RESCUE, EXPLORE, RETURN
            그 외 (부족분 < return_min_deficit, cap = cap_max)  RESCUE, EXPLORE
        RESCUE 는 할당 가능한 생존자가 있을 때만, EXPLORE 는 덩어리가 있을 때만. 전부 막히면 RETURN.
        return_with_cap: false 면 cap = 0 일 때만 RETURN. RETURN = 하위 할당 없이 depot 으로, 도착하면 충전.
Explore 2 단계 (env/role/explore.py)
        1 단계: 덩어리 (frontier DBSCAN, 도달 가능 셀만) 선택 — 그룹 순차, 중복 허용
        2 단계: 최종 담당자 m 대면 kk = min(m, kcap) 구간 (측지 k-medoids). 이번 step 결정하지 않는 담당자는 직전 구간과 헝가리안
                연속 매칭 (그 구간은 마스크), 결정하는 UAV 는 (용량, index) 순으로 남은 구간 선택 (후보 1 개 이하면 정책 생략)
        구간 안 목표점 = 내 구간 frontier 중 BFS 가장 가까운 셀 (규칙). 구간은 매 step rebalance (정책 결정 아님), 비면 소진.

Role 이벤트 1~5 — 대상 UAV 만 역할부터 다시 (role -> EXPLORE 면 explore 할당 / RESCUE 면 rescue 할당 /
RETURN 이면 depot). 대상이 아닌 UAV 는 역할 · 목표 유지 (RESCUE / RETURN 중인 UAV 는 자기 이벤트 3 / 4 / 5 일 때만 재결정)
    1 에피소드 시작                       -> 전체 UAV
    2 새 생존자 발견                      -> 그 그룹에서 현재 역할이 EXPLORE 인 UAV 만. 그룹 belief 에 처음 들어왔고 (직접 관측 /
      통신 병합 무관, 이미 알던 생존자는 제외) 할당 가능 (미구조 · 미만료 · 그룹의 누군가가 t + d <= τ) 하며 그룹의 다른 UAV 가
      맡지 않은 생존자가 있을 때. 같은 step 여러 명이어도 한 번
    3 생존자 구조 완료                    -> 구조한 UAV 만
    4 Return 완료 (depot 도착, 충전)     -> 그 UAV 만
    5 목표 무효 -> 그 UAV 만. RESCUE: conflict (belief 병합으로 같은 생존자를 쫓는 더 가까운 그룹 UAV 를 알게 됨, 먼 쪽만) /
      다른 UAV 가 구조 / 만료 / 도달 불가.  EXPLORE: 내 덩어리에 frontier 없음 (소멸 포함) / 매 step rebalance 후 내 구간이 없는
      상태가 events.no_segment_steps step 연속
    같은 step 에 겹쳐도 UAV 마다 재결정 1 번 (원인은 모두 기록). 결정 순서는 policies/decision_order.py (role -> 1 단계 ->
    2 단계, 한 대가 결정할 때마다 남은 생존자 / 덩어리 / 구간 기준으로 다시 계산, 학습 모델과 같은 함수).
    그룹 구성 변경 / 덩어리 쪼개짐 · 합쳐짐 / timeout 재결정은 없음 (쪼개짐 · 합쳐짐 대응은 현재 플래그 · goal latch 용으로 계산).
Explore goal latch (events.goal_latch): 목표 셀은 관측되어 frontier 가 아니게 됨 / 도달 불가 / 내 구간 밖에
    events.goal_latch_outside_steps step 연속일 때만 바꾼다 (매 step rebalance 로 구간 경계가 움직여도 유지).
통신 hysteresis (events.comm_hysteresis): 링크 생성 거리 <= R_c, 해제 거리 > R_c + h (그룹 / belief / 보상 / 특징 공통).

한 step 처리 순서 (_step_role)
    [직전 step 끝에 준비됨] 지도 갱신 -> 덩어리 생성 / 대응 -> 구간 rebalance -> Role 이벤트 수집 · 결정 대상 (kind)
    1) 결정 순서대로: role -> RESCUE 생존자 / EXPLORE 1 단계
    2) 2 단계 (구간): 이번 step 1 단계를 한 UAV
    3) 이동 (BFS 1 칸) -> 관측 -> 구조 / 만료 -> 덩어리 · 구간 · 이벤트 갱신 -> 보상
보상    role_rewards() (env_config.rewards):
          role    r_role(i, t) = (|R_t| - w_x |X_t|) / N_T - c_osc · 1_osc(i)   (끝날 때 미구조 = 마지막 step 의 X)
                  1_osc: osc_window 안에 역할 A -> B -> A, 두 전환 모두 비강제 (강제 = 이전 역할이 마스크로 막힘 / 귀환 완료 직후)
          rescue  이벤트 끝 그룹 공동 보상 (G_res = 이번 이벤트에 RESCUE 를 고른 그룹 UAV 전원이 같은 값)
                  R_i = (S - F) / N_T + 1[F = 0] [(1 - w_c) e^{-d} + w_c c̄_i] - w_dup · 1_dup  (옵션 단위)
                  σ_u = τ_u - t - min_{G_res} BFS, U_ε = 그룹이 아는 · G_res 밖 UAV 가 맡지 않은 · 0 <= σ_u <= ε 인 생존자,
                  S = G_res 가 할당했고 시간 안에 닿는 수, F = U_ε 중 G_res 가 할당하지 않은 수,
                  d = Σ BFS(l, a_l) / (d_max |G_res|), c̄_i = 결정 t0 ~ t1 (구조 완료 / 이벤트 5 / 끝) 동안 개인 연결도
                  c_{i,t} = (|comp_t(i)| - 1) / (N - 1) 평균 — (S - F) / N_T · 중복은 결정 step, 보너스는 옵션이 끝나는 step 에,
                  1_dup = 같은 생존자를 더 가까운 다른 UAV (그룹 밖 포함) 가 쫓음 (결정당 최대 1 회)
          explore 매 step 내 기여 r = (1 - ψ^f) Δcov^(i) / |free| (1 + w_mult c_{i,t}) (EXPLORE 인 step 만). ψ^f = 팀 전체 발견 / N_T,
                  Δcov^(i) = 팀 전역에서 처음 관측된 free 셀 중 i 가 본 것 (같은 step 에 k 대가 보면 1/k 씩)
        학습 (training/role/ppo.py) 은 결정 단위: role = SMDP (다음 role 결정까지 할인 누적 + 그 가치로 bootstrap),
        rescue = 옵션 (결정 ~ 다음 role / rescue 결정 / 끝) 보상 합 (할인 없음), explore = 옵션 (결정 ~ 다음 explore 결정 /
        역할 변경 / 끝) 할인 누적 — 둘 다 bootstrap 없음 (A = R - V)
마스크  그룹 belief 발견 수 = N_T 면 EXPLORE 금지 (그 순간 그 그룹 EXPLORE UAV 에 이벤트 5: exploration complete)

구동    step(decider)         : decider.role / rescue / explore / explore_sub 를 env 가 결정 순서대로 호출 (규칙 · 랜덤, GUI)
        step_explicit(actions): 모델이 순차 디코딩한 dict(role, rescue, explore, explore_sub) (N,) 를 받는다.
                                마스크 위반 / policy_mode rule 헤드 / 2 단계 hungarian 은 규칙 정책으로 대체.
로깅    이벤트 (step, 종류, 트리거 UAV, 재결정 대상), UAV 별 역할 전환 · 목표 변경, 에피소드 통계 (종류별 횟수, 역할 전환,
        목표 변경, 진동 = rewards.role.osc_window step 안에 목표 A -> B -> A, 부분 귀환, RESCUE 재결정 원인, κ) -> info["event_stats"]
        (끝 step), events.log_dir 가 있으면 에피소드 로그 JSON.
"""
import json
import os

import numpy as np

from env.role.connectivity import pair_connectivity, personal_connectivity
from env.role.graph_obs import size_feat

from env.core.modes import Mode
from env.core.pathfinding import bfs_field, path_to, window_waypoint
from env.core.base_sim import BaseSimulation
from env.core.targets import deadline as _deadline
from env.core.step_context import StepContext
from env.core.agent import Robot
from env.role.explore import ExploreMixin, EXPLORE_DEFAULTS, n_seg_slots, seg_offset
from policies.role_rules import RuleDecider, rule_explore, rule_explore_sub, rule_rescue, rule_role

ROLES = (Mode.EXPLORE, Mode.RESCUE, Mode.RETURN)      # role 행동 인덱스 = Mode.value
N_ROLES = 3
EVENTS = ("1", "2", "3", "4", "5")                    # Role 이벤트 번호 (targeted)
EXPLORE_EVENTS = ("split", "merge")                   # 덩어리 쪼개짐 / 합쳐짐 (재결정 없음, 기록용)
INVALIDATING = {"1", "3", "4", "5"}                   # 당사자 목표 무효 (Context goal_valid)
NEIGHBOR_BINS = (0.25, 0.5, 0.75, 1.0)                # 1-hop 이웃 거리 / R_c 구간 (마지막 = 1 초과, hysteresis)
CAP_STATES = ("zero", "partial", "full")
REWARD_ITEMS = ("role_team", "role_osc", "rescue_sf", "rescue_bonus", "rescue_dup", "explore_cov")   # self.rw / 로그 / 그래프 항목

ROLE_DEFAULTS = {
    "max_redecide": 1,         # 하위 할당 실패 시 RETURN 으로 대체하는 횟수 상한 (점검용)
    "return_with_cap": True,   # 용량이 남아도 RETURN 허용 (부분 귀환). false = 용량 0 일 때만
    "return_min_deficit": 1,   # 부분 귀환 허용 최소 부족분 (cap_max - cap)
}
EVENTS_DEFAULTS = {
    "goal_latch": True,        # explore 목표 셀 유지 (관측 / 도달 불가 / 구간 밖 N step 일 때만 변경)
    "goal_latch_outside_steps": 3,
    "no_segment_steps": 3,     # 구간 없는 상태가 이만큼 연속이면 이벤트 5
    "comm_hysteresis": 2,      # 링크 해제 거리 = R_c + 이 값 (0 = hysteresis 없음)
    "log_events": True,        # 이벤트 / 목표 변경 기록 (info event_stats)
    "log_dir": None,           # 에피소드 로그 JSON 저장 폴더 (None = 저장 안 함)
}
HEADS = ("role", "rescue", "explore")
POLICY_MODE_DEFAULT = {h: "learned" for h in HEADS}   # env_config["policy_mode"]: 헤드별 rule | learned
SUB_MODES = ("learned", "hungarian")                   # env_config["explore_sub_mode"]: 2 단계 정책
CLUSTER_DEFAULTS = {
    "eps": 1.5,                # DBSCAN 이웃 반경 (1.5 = 8-연결)
    "min_samples": 3,          # core 조건 (자기 포함 이웃 수)
    "k_max": 10,               # explore 덩어리 슬롯 수
}
REWARDS_DEFAULTS = {         # env_config.rewards
    "role": {"w_x": 2.0,               # 만료 1 명당 (/ N_T)
             "c_osc": 0.03,            # 역할 진동 (A -> B -> A, 비강제) penalty
             "osc_window": 10,         # 진동 판정 창 W (step) — 역할 진동 penalty 와 목표 진동 로그 공통
             "terminal_unrescued_as_expired": True},   # 끝날 때 미구조 생존자 = 마지막 step 의 만료
    "rescue": {"epsilon_urgent": 5,    # 급한 생존자: 그룹 slack 0 <= σ <= 이 값
               "w_c": 0.5,             # 보너스 (F = 0) = (1 - w_c) e^{-d} + w_c · c̄ (옵션 동안 개인 연결도 평균)
               "w_dup": 0.05},         # 더 가까운 다른 UAV 가 같은 생존자를 쫓음 (결정당 1 회)
    "explore": {"w_mult": 0.2},        # 연결 배율: 새 셀을 밝힐 때만 (1 + w_mult · c_{i,t}) 배
    "connectivity": {"r_c_source": "base"},   # 연결성 C 의 R_c: base = 기본 통신 반경 (hysteresis 없음)
}

# 관측 차원
# critic 입력 (role_self ‖ role_group) — 이름 / 순서 = 관측 인덱스, UAV 의 n-hop 공유 지식 기준
ROLE_SELF_FEATURES = ("x", "y", "capacity", "role_explore", "role_rescue", "role_return", "goal_valid",
                      "since_decision")
ROLE_GROUP_FEATURES = ("group_frac_explore", "group_frac_rescue", "group_frac_return", "assignable_victims",
                       "min_slack", "n_clusters", "group_explored", "undiscovered_g", "time",
                       "nearest_cluster_dist", "largest_cluster_size", "depot_dist")
ROLE_SELF_DIM = len(ROLE_SELF_FEATURES)      # 8
ROLE_GROUP_DIM = len(ROLE_GROUP_FEATURES)    # 12
# ---- Role 노드 g 벡터 (그래프 노드 / 간선은 env/role/graph_obs.py) ----
ROLE_G_RES_FEATURES = ("assignable", "min_slack")              # [할당 가능 생존자 수 / N_T, 최소 slack]
ROLE_G_EXP_FEATURES = ("n_clusters", "frontier_cells", "psi_f_group")   # [덩어리 수 / K_max, size_feat(그룹 frontier 셀 수),
                                                                           #  그룹 belief 발견 생존자 수 / N_T]
EXPLORE_KIND_DIM = 1           # [2 단계 learned]
STATE_CHANNELS = 6             # critic 중앙 state: BaseSimulation 5 채널 + 목표 지점 g_l
TEAM_CTX_DIM = 8               # 팀원 [role one-hot(3), 목표 x, y, 이번 step 결정 필요, 위치 x, y]
CRITIC_TEAM_DIM_ROLE = 12      # 기본 critic_team 8 (BaseSimulation) + [배정 조난자 x, y, 배정 구간 medoid x, y]


def _merge(defaults, given):
    out = dict(defaults)
    out.update(given or {})
    return out


def cap_state(a):
    return "zero" if a.capacity == 0 else ("full" if a.capacity >= a.capacity_max else "partial")


class RoleSimulation(ExploreMixin, BaseSimulation):
    CRITIC_TEAM_DIM = CRITIC_TEAM_DIM_ROLE

    def __init__(self, options=None):
        options = dict(options or {})
        self.role_cfg = _merge(ROLE_DEFAULTS, options.get("role"))
        self.events_cfg = _merge(EVENTS_DEFAULTS, options.get("events"))
        self.cluster_cfg = _merge(CLUSTER_DEFAULTS, options.get("cluster"))
        self.explore_cfg = _merge(EXPLORE_DEFAULTS, options.get("explore"))
        given = options.get("rewards") or {}
        self.rewards_cfg = {h: _merge(REWARDS_DEFAULTS[h], given.get(h)) for h in REWARDS_DEFAULTS}
        if self.rewards_cfg["connectivity"]["r_c_source"] != "base":
            raise ValueError(f"rewards.connectivity.r_c_source={self.rewards_cfg['connectivity']['r_c_source']!r} — base 만 지원")
        self.k_max = int(self.cluster_cfg["k_max"])
        self.n_sub_max = int(self.explore_cfg["n_sub_max"])
        self.n_seg_slots = n_seg_slots(self.n_sub_max)
        if self.explore_cfg["goal_mode"] not in ("nearest", "medoid"):
            raise ValueError(f"explore.goal_mode={self.explore_cfg['goal_mode']!r} — nearest | medoid")
        options["events"] = self.events_cfg                       # BaseSimulation 이 comm_hysteresis 를 읽는다
        self.s_max = int(options.get("s_max") or 16)             # 그래프 S 노드 = rescue 행동 슬롯 수 (model.gnn.s_max 와 같은 값)
        self.policy_mode = _merge(POLICY_MODE_DEFAULT, options.get("policy_mode"))
        bad = {h: m for h, m in self.policy_mode.items() if h not in HEADS or m not in ("rule", "learned")}
        if bad:
            raise ValueError(f"policy_mode {bad} — 헤드 role / rescue / explore, 값 rule | learned")
        self.explore_sub_mode = options.get("explore_sub_mode", "learned")
        if self.explore_sub_mode not in SUB_MODES:
            raise ValueError(f"explore_sub_mode={self.explore_sub_mode!r} — learned | hungarian")
        self._ruled = {h: set() for h in HEADS + ("explore_sub",)}
        self._replaced = {h: set() for h in HEADS + ("explore_sub",)}
        self.episode_tag = None                # 로그 파일 이름 꼬리 (run_rules 가 seed 로 지정)
        super().__init__(options)

    @property
    def sub_learned(self):
        """2 단계를 학습 정책으로 고르는가 (explore 헤드 learned + explore_sub_mode learned)."""
        return self.policy_mode["explore"] == "learned" and self.explore_sub_mode == "learned"

    @property
    def d_max(self):
        """거리 정규화 분모 = 최대 BFS 거리 (env_config.d_max, 없으면 rows + cols)."""
        return float(self.cfg.get("d_max") or (self.rows + self.cols))

    def size_feat(self, n):
        """셀 수 n 의 크기 특징 min(1, log(1 + n) / log(1 + explore.size_ref)) (env/role/graph_obs.py size_feat)."""
        return size_feat(n, float(self.explore_cfg["size_ref"]))

    def _nd(self, d):
        return 1.0 if d is None or d < 0 or d >= 10 ** 9 or not np.isfinite(d) else min(1.0, d / self.d_max)

    def _rel(self, cell, i):
        """cell 의 UAV i 기준 상대 위치 ((x - x_i) / W, (y - y_i) / H) ∈ [-1, 1]."""
        a = np.asarray(self.frame.normalized(cell)) - np.asarray(self.frame.normalized(self.agents[i].pos))
        return np.clip(a, -1.0, 1.0)

    # ================================================================ reset / 준비

    def reset(self, regenerate=True):
        super().reset(regenerate)          # 맵 / 조난자 / 지식 / 믿음 (BaseSimulation) + _prepare (아래에서 덮음)
        for a in self.agents:
            a.mode = Mode.EXPLORE
            a.goal = None
            a.assigned_target = None
            a.last_decision = 0
            a.noseg = 0                       # 구간 없는 상태 연속 step
            a.goal_out = 0                    # 목표 셀이 내 구간 밖인 연속 step (goal latch)
            a.last_dec_t = 0                  # 마지막 role 결정 시각 (결정 간격 통계)
            a.role_since = 0                  # 지금 역할을 시작한 시각 (Context 의 역할 유지 시간)
        self._explore_reset()
        self.same_cell_total = 0
        self.event_totals = {e: 0 for e in EVENTS + EXPLORE_EVENTS}
        self.redecide_total = 0
        self.mask_violation_total = 0
        self.last_event = [(0, ["1"]) for _ in self.agents]   # UAV 별 마지막으로 재결정을 부른 이벤트 (시각, 종류)
        self.last_events = [set() for _ in self.agents]
        self.step_log = []                                    # 이번 step 덩어리 대응 / Explore 이벤트 문장 (GUI 이벤트 로그)
        self.decided_role = [False] * self.n_agents
        self.events = [{"1"} for _ in self.agents]
        self.kind = [None] * self.n_agents
        self._known = [set() for _ in self.agents]
        self._log_reset()
        self._reward_reset()
        self._prepare_role()
        self._explore_after_update([(a.mode, None, None) for a in self.agents], [None] * self.n_agents)
        self._known = [self._known_victims(i) for i in range(self.n_agents)]
        self._set_decide_order()

    def _prepare(self):
        """다음 결정 준비 (BaseSimulation 훅). reset 도중 (role 상태 초기화 전) 호출되면 건너뛴다."""
        if hasattr(self, "events"):
            self._prepare_role()

    def _prepare_role(self):
        """위치 · 지식 · 시간이 바뀐 직후: BFS 필드, 통신 그룹, 할당 가능 생존자 슬롯."""
        n, T = self.n_agents, self.max_targets
        self.fields, self.parents = [], []
        for i, a in enumerate(self.agents):
            d, p = bfs_field(a.pos, self.knowledge.known_obstacle[i], self.rows, self.cols)
            self.fields.append(d)
            self.parents.append(p)
        # 통신 그룹 = reach (n-hop) 연결 성분, 그룹 안 순서 = UAV 번호
        reach = self.knowledge.reach
        comp = [-1] * n
        groups = []
        for s in range(n):
            if comp[s] >= 0:
                continue
            comp[s] = len(groups)
            members, stack = [s], [s]
            while stack:
                u = stack.pop()
                for v in range(n):
                    if reach[u, v] and comp[v] < 0:
                        comp[v] = comp[s]
                        members.append(v)
                        stack.append(v)
            groups.append(sorted(members))
        self.comp, self.groups = comp, groups
        self.group_depot_field = []
        for g in groups:
            self.group_depot_field.append(bfs_field(self.depot, self.knowledge.known_obstacle[g[0]],
                                                    self.rows, self.cols)[0])
        if not hasattr(self, "group_clusters") or len(getattr(self, "group_clusters", [])) != len(groups):
            self.group_clusters = [[] for _ in groups]       # _explore_after_update 가 채운다
        # UAV 별 생존자 슬롯 = 아직 아무도 맡지 않은 할당 가능 생존자 (그룹 지식, BFS 거리 -> 마감 -> id 순 상위 T).
        # rescue 정책 입력 / role 의 Rescue encoder 입력 / g_res / top-m K_feas 가 모두 이 기준
        self.victim_slots, self.n_alloc = [], []
        for i in range(n):
            held = self.held_by_others(i)
            ok = [ti for ti, t in enumerate(self.targets)
                  if t.discovered and self.knowledge.known_free[i][t.pos] and ti not in held and self._assignable(i, ti)]
            ok.sort(key=lambda ti: (self._dist(i, self.targets[ti].pos), _deadline(self.targets[ti]), ti))
            self.n_alloc.append(len(ok))
            ok = ok[:self.s_max]
            self.victim_slots.append(ok)

    def _explore_after_update(self, prev_state, prev_seg, prev_med=None):
        """지도 갱신 직후 explore 파이프라인 (env/role/explore.py). 반환: 구간 소진 / 도달 불가 UAV {i: 이유}."""
        self._seg_cache, self._geo_cache = {}, {}
        self._graph_cache, self._pair_cache, self._match_cache = {}, {}, {}
        self._split_of, self._merge_of = {}, {}
        self._build_blobs()
        self._update_ghosts(prev_state)
        self._relink_and_anchor(prev_seg)
        exhausted = self._rebalance_segments(prev_seg, prev_med or [None] * self.n_agents)
        self._build_slots()
        self._prev_blobs = [self.blobs[self.comp[i]] for i in range(self.n_agents)]
        return exhausted

    def held_by_others(self, i, exclude=()):
        """UAV i 그룹의 다른 UAV (exclude 제외) 가 RESCUE 로 맡고 있는 생존자 집합."""
        return {self.agents[j].assigned_target for j in self.groups[self.comp[i]]
                if j != i and j not in exclude and self.agents[j].mode is Mode.RESCUE
                and self.agents[j].assigned_target is not None}

    def neighbors(self, i):
        """1-hop 직접 연결 UAV (통신 링크, hysteresis 포함)."""
        return [j for j in range(self.n_agents) if self.knowledge.adjacency[i, j]]

    def _dist(self, i, cell):
        d = int(self.fields[i][cell[0], cell[1]])
        return d if d >= 0 else 10 ** 9

    def _assignable(self, i, ti):
        """발견 · 미구조 (내가 아는 한) · 미만료 · d <= τ - t. (같은 step 의 중복 선택은 결정 순서 마스크로 막는다.)"""
        t = self.targets[ti]
        if t.rescued_at is not None or t.expired_at is not None or self.beliefs.is_done(i, self, ti):
            return False
        d = int(self.fields[i][t.pos])
        return 0 <= d <= _deadline(t) - self.time

    def _loses_conflict(self, i, ti, d=None):
        """같은 그룹에 같은 조난자를 쫓는 UAV 중 나보다 가까운 (동률이면 index 작은) UAV 가 있는가 (목표 무효 판정)."""
        pos = self.targets[ti].pos
        d = self._dist(i, pos) if d is None else d
        for j, b in enumerate(self.agents):
            if j == i or b.mode is not Mode.RESCUE or b.assigned_target != ti or self.comp[j] != self.comp[i]:
                continue
            dj = self._dist(j, pos)
            if dj < d or (dj == d and j < i):
                return True
        return False

    def clusters_of(self, i):
        """UAV i 그룹의 덩어리 슬롯 (1 단계 노드)."""
        return self.group_clusters[self.comp[i]]

    def _known_victims(self, i):
        return {ti for ti, t in enumerate(self.targets)
                if t.active and t.discovered and self.knowledge.known_free[i][t.pos] and not self.beliefs.is_done(i, self, ti)}

    def standing(self, j):
        """j 의 현재 배정이 이번 step 에 유효한가 (재결정 대상이 아니거나, 이번 step 이미 결정함). 규칙 정책용."""
        return self.kind[j] is None or j in getattr(self, "_decided_now", ())

    # ================================================================ 마스크

    def role_mask(self, i, banned=(), taken=()):
        """(3,) bool, 인덱스 = Mode.value (EXPLORE, RESCUE, RETURN). taken = 같은 그룹이 이번 step 이미 고른 생존자.
        그룹 belief 발견 수 = N_T (exploration complete) 면 EXPLORE 금지. 전부 막히면 RETURN 만 (fallback).
        self._mask_flags = (EXPLORE 가 exploration complete 로 막힘, RETURN fallback) — 결정 시 로그용."""
        a = self.agents[i]
        m = np.zeros(N_ROLES, bool)
        if a.capacity > 0:
            m[Mode.RESCUE.value] = bool(self.rescue_mask(i, taken).any())
        if a.capacity == 0:
            m[Mode.RETURN.value] = True
        elif self.role_cfg["return_with_cap"] and a.capacity_max - a.capacity >= int(self.role_cfg["return_min_deficit"]):
            m[Mode.RETURN.value] = True                          # 부분 귀환
        complete = self.exploration_complete(self.comp[i])
        m[Mode.EXPLORE.value] = not complete and bool(self.explore_mask(i).any())
        for r in banned:
            m[r] = False
        forced = not m.any()
        if forced:
            m[Mode.RETURN.value] = True
        self._mask_flags = (complete, forced)
        return m

    def rescue_mask(self, i, taken=()):
        m = np.zeros(self.s_max, bool)
        for s, ti in enumerate(self.victim_slots[i]):
            m[s] = ti not in taken
        return m

    def explore_mask(self, i):
        """1 단계 마스크: 추정 담당 셀을 뺀 셀 중 내 (UAV i) 지도로 도달 가능한 셀이 하나라도 있는 덩어리."""
        m = np.zeros(self.k_max, bool)
        f, g = self.fields[i], self.comp[i]
        for k, b in enumerate(self.clusters_of(i)):
            cells = self._blob_cells_for_split(g, b)
            m[k] = bool(cells) and any(f[c] >= 0 for c in cells)
        return m

    # ================================================================ 이벤트 / 결정 대상

    def _set_decide_order(self):
        """이벤트가 있는 UAV 만 결정 대상 (kind role, D_0). 결정 순서는 _decide 가 한 대씩 정할 때마다 policies/decision_order.py 로
        다시 계산한다 (학습 모델과 같은 함수 · 같은 관측 값). 이벤트 원인 기록 (이벤트 2 는 그룹 단위로 _collect_events 에서 기록)."""
        n = self.n_agents
        kind = [None] * n
        order = []
        for g, members in enumerate(self.groups):
            for i in members:
                if self.events[i] and self.agents[i].alive:
                    kind[i] = "role"
                    order.append(i)
                    for e in sorted(self.events[i] - {"2"}):
                        self._log_event(e, i, [i])
        self.kind = kind
        self.decide_order = order                               # D_0 (그룹 순, index 순 — 실제 결정 순서가 아님)
        self._deciding = {i for i in range(n) if kind[i] is not None}
        self._decided_now = set()
        self.last_decision_seq = getattr(self, "decision_seq", [])   # 직전 step 의 실제 결정 순서 (점검 / GUI)
        self.last_s2_blob = getattr(self, "s2_blob", {})
        self.decision_seq = []                                  # 실제 결정 순서 [(단계, UAV)] (role / s1 / s2)
        self.s2_blob = {}                                       # 2 단계 참여자 -> (그룹, 덩어리 id) (결정 시점)
        self.cur_claim, self.cur_xclaim, self.cur_sclaim = {}, {}, {}   # 차례가 될 때의 담당 후보 (생존자 / 덩어리 슬롯 / 구간)
        self.goal_valid = [a.goal is not None and not (self.events[i] & INVALIDATING) for i, a in enumerate(self.agents)]
        self._prev_assign = [(a.mode, a.blob, a.cluster_medoid, a.assigned_target) for a in self.agents]
        self._match_cache = {}                                  # 구간 표 / 현재 플래그는 kind 에 따라 달라진다
        self._seg_cache = {}                                    # 구간 분할 warm start (기존 배정자 goal) 도 kind 에 따라
        self._obs_cache = None                                  # 결정 순서 계산용 관측 (rl_observation 이 채움)

    # ---------------------------------------------------------------- 결정 순서 입력 (관측 값 그대로)

    def _order_obs(self):
        if getattr(self, "_obs_cache", None) is None:
            self.rl_observation()
        return self._obs_cache

    def _order_inputs(self):
        """policies/decision_order.py 입력 (torch, B = 1) — 모델이 받는 관측과 같은 값 (U–S / U–K 간선의 자기 행)."""
        import torch
        ob = self._order_obs()
        A = ob["agents"]
        st = lambda k: torch.as_tensor(np.stack([a[k] for a in A]))[None]
        own = lambda x: torch.stack([x[0, i, i] for i in range(self.n_agents)])[None]   # UAV i 관측의 자기 행
        ids = st("s_ids")
        return {"group": torch.as_tensor(ob["group"] > 0.5)[None], "cap": torch.as_tensor(ob["team_cap"])[None], "ids": ids,
                "valid": (ids >= 0) & (st("rescue_mask") > 0.5), "rd": own(st("us_d")), "slack": own(st("us_slack")),
                "emask": st("explore_mask") > 0.5, "ed": own(st("uk_d")), "esize": st("explore_size"),
                "en": torch.round(st("k_n") * self.n_agents), "useg_d": st("useg_d"), "seg_size": st("explore_seg_size"),
                "ecur": st("explore_cur")[..., 0]}

    def _collect_events(self, prev_pos, rescued_by, exhausted):
        """이벤트 3 / 4 / 5 (그 UAV 만) + 이벤트 2 (그룹 단위 -> 그룹의 EXPLORE UAV). 그룹 belief 에 처음 들어온 생존자만
        (그룹 멤버들의 직전 지식 합집합에 없던 것) — 이미 알던 생존자가 통신 병합으로 다시 들어와도 발생하지 않는다."""
        n = self.n_agents
        ev = [set() for _ in range(n)]
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            if i in rescued_by:
                ev[i].add("3")
            if a.mode is Mode.RETURN and a.pos == self.depot and prev_pos[i] != self.depot:
                ev[i].add("4")
            if a.mode is Mode.EXPLORE and i in exhausted:
                ev[i].add("5")
                self.ev_stats["ev5_causes"]["no_segment"] += 1
            if a.mode is Mode.RESCUE and "3" not in ev[i]:
                ti = a.assigned_target                     # None = 마감으로 배정이 풀림 (BaseSimulation._process_deadlines)
                if ti is None or self._target_invalid(i, ti):
                    ev[i].add("5")
                    self.ev_stats["ev5_causes"]["rescue_invalid"] += 1
            # exploration complete: 그룹 belief 발견 수가 (병합 포함) 처음 N_T 에 도달 -> 그 그룹 EXPLORE UAV 이벤트 5
            if not self._explore_seen_all[i] and self.exploration_complete(self.comp[i]):
                self._explore_seen_all[i] = True
                if a.mode is Mode.EXPLORE and "5" not in ev[i]:
                    ev[i].add("5")
                    self.ev_stats["ev5_causes"]["exploration_complete"] += 1
        known_now = [self._known_victims(i) for i in range(n)]
        for g, members in enumerate(self.groups):
            before = set().union(*[self._known[j] for j in members])
            held = {self.agents[j].assigned_target for j in members if self.agents[j].mode is Mode.RESCUE}
            new = []
            for k in sorted(known_now[members[0]] - before):
                tau = _deadline(self.targets[k])
                if k in held or not any(self._dist(j, self.targets[k].pos) <= tau - self.time for j in members):
                    continue
                new.append(k)
            if not new:
                continue
            explorers = [j for j in members if self.agents[j].alive and self.agents[j].mode is Mode.EXPLORE]
            for j in explorers:
                ev[j].add("2")
            self._log_event("2", None, explorers, group=g)
            for k in new:                                     # 선점 후보: EXPLORE 는 못 가는데 RESCUE 는 갈 수 있는 새 생존자
                tau = _deadline(self.targets[k])
                reach = lambda js: any(self._dist(j, self.targets[k].pos) <= tau - self.time for j in js)
                rescuers = [j for j in members if self.agents[j].mode is Mode.RESCUE]
                if not reach(explorers) and reach(rescuers):
                    self.ev_stats["preempt_candidates"] += 1
        self._known = known_now
        return ev

    def _target_invalid(self, i, ti):
        """6 (RESCUE): 구조됨을 앎 / 만료 / 도달 불가 (d > τ - t) / 같은 조난자를 쫓는 그룹 내 UAV 가 더 가까움 (동률이면 index 큰 쪽)."""
        t = self.targets[ti]
        if t.expired_at is not None or self.beliefs.is_done(i, self, ti):
            return True
        d = int(self.fields[i][t.pos])
        if d < 0 or d > _deadline(t) - self.time:
            return True
        return self._loses_conflict(i, ti, d)

    # ================================================================ 결정

    def _decide(self, role_fn, rescue_fn, explore_fn, sub_fn):
        """한 이벤트의 결정 (*_fn(i, mask) -> int):
        [1] role + rescue: role_front 로 다음 UAV -> 역할 -> RESCUE 면 바로 생존자 (그룹 1:1 마스크). 한 대마다 순서 재계산
        [2] explore 1 단계: EXPLORE 를 고른 UAV 중 신규 (이번에 새로 EXPLORE 이거나 영역이 없음) 만 explore_front 순서로
            덩어리 선택 (실제 선택으로 n_k 갱신). 유지 UAV (직전에도 EXPLORE, 영역 유효) 는 영역 · 구간 그대로, n_k 에 포함
        [3][4] 분할 + 2 단계 (_stage2_all)  [5] rescue 보상 (그룹마다 1 회, [4] 뒤)
        반환: (role_valid, sub_valid {rescue, explore_s1, explore_s2, explore_reselect}, n_redecide, n_violation)."""
        import torch
        from policies.decision_order import explore_front, role_front, role_sigma
        n = self.n_agents
        role_valid = [False] * n
        sub_valid = {"rescue": [False] * n, "explore_s1": [False] * n, "explore_s2": [False] * n,
                     "explore_reselect": [False] * n}
        redecide = violations = 0
        s1, taken = set(), {}
        res_group = {}                                          # 그룹 -> 이번 step RESCUE 로 할당한 UAV (G_res)
        D0 = [i for i in self.decide_order if self.agents[i].alive]
        if D0:
            X = self._order_inputs()
            d0 = torch.zeros(1, n, dtype=torch.bool)
            d0[0, D0] = True
            sig = role_sigma(X["ids"], X["slack"], X["valid"], X["group"], d0)
            # EXPLORE 유지 후보: 이벤트 직전에도 EXPLORE 이고 영역 (덩어리 슬롯) 이 유효 (관측 explore_cur >= 0)
            self._retain_ok = {i for i in D0 if float(X["ecur"][0, i]) >= 0}
            prev_blob = {i: self.agents[i].blob for i in D0}
            rem = d0.clone()
            explorers = []
            while bool(rem.any()):
                tk = torch.zeros_like(X["valid"])
                for i in range(n):
                    tk[0, i] = torch.as_tensor([int(u) in taken.get(self.comp[i], ()) for u in X["ids"][0, i].tolist()])
                sel, claim = role_front(X["group"], rem, X["cap"], X["ids"], X["rd"], sig, X["valid"] & ~tk)
                for i in np.flatnonzero(sel[0].numpy()):
                    c = int(claim[0, i])
                    self.cur_claim[i] = int(X["ids"][0, i, c]) if c >= 0 else None
                    role = self._decide_role(i, role_fn, rescue_fn, taken, res_group, role_valid, sub_valid)
                    violations += role[1]
                    if role[0] is Mode.EXPLORE:
                        explorers.append(i)
                    rem[0, i] = False
                    self.decision_seq.append(("role", int(i)))
            self._log_claim_targets()
            self._res_group = res_group
            # ---- [2] explore 1 단계: 신규 EXPLORE UAV 만 (유지 UAV 는 영역 · 구간 그대로, n_k 에는 기존 배정자로) ----
            retained = [i for i in explorers if i in self._retain_ok]
            fresh = [i for i in explorers if i not in self._retain_ok]
            rem = torch.zeros(1, n, dtype=torch.bool)
            rem[0, fresh] = True
            n_now = X["en"].clone()
            for r in retained:
                for j in self.groups[self.comp[r]]:
                    n_now[0, j, int(X["ecur"][0, r])] += 1
            st = self.ev_stats
            st["ev2_retained"] += sum(1 for i in retained if "2" in self.events[i])
            self._ev2_retained = [i for i in retained if "2" in self.events[i]]
            self._prev_blob = prev_blob
            for g in {self.comp[i] for i in fresh}:
                lead = next(i for i in fresh if self.comp[i] == g)
                st["s1_events"] += 1
                st["s1_participants"] += sum(1 for i in fresh if self.comp[i] == g)
                st["s1_empty_blobs"] += int(sum(1 for k in range(len(self.clusters_of(lead))) if n_now[0, lead, k] == 0))
            while bool(rem.any()):
                sel, claim = explore_front(X["group"], rem, X["emask"], X["ed"], X["esize"], n_now)
                todo = list(np.flatnonzero(sel[0].numpy())) or list(np.flatnonzero(rem[0].numpy()))[:1]
                for i in todo:
                    c = int(claim[0, i]) if bool(sel[0, i]) else -1
                    self.cur_xclaim[i] = c if c >= 0 else None
                    k = self._stage1(i, explore_fn)
                    if k is None:                                    # 마스크상 불가 (안전장치): RETURN
                        redecide += 1
                        self._assign_return(i)
                    else:
                        s1.add(i)
                        st["s1_claim_total"] += 1
                        st["s1_claim_kept"] += int(k == c)
                        for j in self.groups[self.comp[i]]:
                            n_now[0, j, k] += 1
                    rem[0, i] = False
                    self.decision_seq.append(("s1", int(i)))
        for i in s1:
            sub_valid["explore_s1"][i] = True
        s2, reselect = self._stage2_all(s1, sub_fn)
        for i in getattr(self, "_ev2_retained", ()):                  # 유지 UAV 의 영역이 바뀌면 안 됨 (0 이어야 함)
            self.ev_stats["ev2_retained_blob_changed"] += int(self.agents[i].blob != self._prev_blob.get(i))
        self._ev2_retained = []
        for i in s2:
            sub_valid["explore_s2"][i] = True
        sub_valid["explore_reselect"] = [i in reselect for i in range(n)]
        # [5] rescue 보상 — 이벤트의 모든 결정이 끝난 뒤 그룹마다 1 회 ((S - F) / N_T · 중복, 개인 보너스는 옵션 끝 step)
        for g, G in getattr(self, "_res_group", {}).items():         # rescue: (S - F) / N_T + 보너스 + 중복 추적
            self._rescue_event_reward(g, G)
        self._res_group = {}
        self.stats["s1"] += len(s1)
        self.stats["s2"] += len(s2)
        self.stats["reselect"] += len(reselect)
        return role_valid, sub_valid, redecide, violations

    def _decide_role(self, i, role_fn, rescue_fn, taken, res_group, role_valid, sub_valid):
        """UAV i 의 역할 (+ RESCUE 면 생존자). 반환 (역할, 마스크 위반 수)."""
        a = self.agents[i]
        g = self.comp[i]
        violations = 0
        if len(self._group_known(g)) > self.s_max:              # 그룹이 아는 생존자가 S 노드 수를 넘어 잘린 결정
            self.ev_stats["survivor_truncated"] += 1
        if a.mode is Mode.RESCUE:                              # RESCUE UAV 재결정 원인 (3 / 5 만이어야 함)
            for e in self.events[i]:
                self.ev_stats["res_redecide"][e] += 1
        self.ev_stats["dec_intervals"].append(self.time - a.last_dec_t)
        a.last_dec_t = self.time
        mask = self.role_mask(i, (), taken.get(g, ()))
        self.ev_stats["explore_masked_decisions"] += int(self._mask_flags[0])
        self.ev_stats["forced_return_all_masked"] += int(self._mask_flags[1])
        prev = a.mode
        ev2 = "2" in self.events[i]
        r = int(role_fn(i, mask))
        if not (0 <= r < N_ROLES and mask[r]):
            violations += 1
            self._replaced["role"].add(i)
            r = rule_role(self, i, mask)
        role_valid[i] = True
        self._log_role_choice(i, r)
        self._role_transition(i, prev, ROLES[r], forced=not mask[prev.value] or "4" in self.events[i])
        role = ROLES[r]
        if role is Mode.EXPLORE and i not in getattr(self, "_retain_ok", ()):   # 신규: 덩어리는 [2] 1 단계에서
            a.mode, a.assigned_target, a.blob, a.cluster, a.cluster_medoid = Mode.EXPLORE, None, None, None, None
        if role is Mode.RESCUE:
            rm = self.rescue_mask(i, taken.get(g, ()))
            s = int(rescue_fn(i, rm))
            if not (0 <= s < len(rm) and rm[s]):
                violations += 1
                self._replaced["rescue"].add(i)
                s = rule_rescue(self, i, rm)
            ti = self.victim_slots[i][s]
            taken.setdefault(g, set()).add(ti)
            self._assign_rescue(i, ti)
            sub_valid["rescue"][i] = True
            res_group.setdefault(g, []).append(i)
        elif role is Mode.RETURN:
            self._assign_return(i)
        self._decided_now.add(i)
        self._log_claim(i, role, ev2)
        return role, violations

    def _stage1(self, i, explore_fn):
        """1 단계 덩어리 선택. 반환 고른 슬롯 (마스크가 비면 None)."""
        mask = self.explore_mask(i)
        if not mask.any():
            return None
        k = int(explore_fn(i, mask))
        if not (0 <= k < len(mask) and mask[k]):
            self._replaced["explore"].add(i)
            self.mask_violation_step += 1
            k = rule_explore(self, i, mask)
        self._log_blob_choice(i, mask, k)
        b = self.clusters_of(i)[k]
        a = self.agents[i]
        a.mode, a.assigned_target = Mode.EXPLORE, None
        a.blob = b.id
        a.cluster, a.cluster_medoid = None, None                   # 구간은 2 단계에서
        a.last_decision = self.time
        return k

    def blob_dist(self, i, b):
        """UAV i 에서 덩어리 b (추정 담당 셀 제외) 의 가장 가까운 셀까지 BFS (도달 불가 None)."""
        f = self.fields[i]
        d = [int(f[c]) for c in self._blob_cells_for_split(self.comp[i], b) if f[c] >= 0]
        return min(d) if d else None

    def _log_blob_choice(self, i, mask, k):
        """1 단계 선택 통계: (고른 덩어리 BFS + 1) / (가능한 덩어리 중 가장 가까운 BFS + 1)."""
        slots = self.clusters_of(i)
        ds = [self.blob_dist(i, slots[j]) for j in np.flatnonzero(mask)]
        ds = [d for d in ds if d is not None]
        dk = self.blob_dist(i, slots[k])
        if ds and dk is not None:
            self.ev_stats["blob_dist_ratio"].append((dk + 1) / (min(ds) + 1))

    def _stage2_all(self, s1, sub_fn):
        """[3] 분할 + [4] 2 단계. 이번 step 1 단계로 고른 덩어리마다 참여자 P = 이번에 고른 UAV + 그 덩어리의 기존 배정자,
        kk = min(|P|, kcap) 구간 (k-medoids 초기 medoid = 기존 배정자 goal 근처 셀). kk <= 1 이면 덩어리 전체.
        참여자는 seg_front 순서 (남은 가장 큰 구간에 가장 가까운 참여자, 한 대마다 재계산) 로 1:1 마스크 안에서 구간 선택
        (다 차면 전부 다시 열림, 후보 1 개 이하면 정책 호출 없음 = 결정 아님). 기존 배정자는 덩어리 고정 · 구간만 (stage-2-only).
        아무도 새로 고르지 않은 덩어리의 기존 배정자는 참여하지 않는다 (구간 유지는 매 step rebalance 매칭).
        반환 (2 단계를 정책으로 고른 UAV, 그중 stage-2-only 재선택한 기존 배정자)."""
        import torch
        from policies.decision_order import seg_front
        out, reselect = set(), set()
        M = self.n_sub_max
        st = self.ev_stats
        X = self._order_inputs() if s1 else None
        for g, members in enumerate(self.groups):
            by = {}
            for j in members:
                a = self.agents[j]
                if a.mode is Mode.EXPLORE and a.blob is not None:
                    by.setdefault(a.blob, []).append(j)
            for bid, P in sorted(by.items()):
                D = [j for j in P if j in s1]
                b = self.blob_index[g].get(bid)
                if not D or b is None:
                    continue
                kk = min(len(P), self.kcap(g, b))
                keepers = [j for j in P if j not in s1]
                if keepers:
                    st["join_events"] += 1
                if kk <= 1:
                    cells = self._blob_cells_for_split(g, b)
                    for j in D:
                        self._assign_segment(j, cells)
                    continue
                segs = self.segments(g, b, kk)
                slot = next(k for k, c in enumerate(self.clusters_of(P[0])) if c.id == bid)
                off = seg_offset(kk)
                n = self.n_agents
                seg_d = torch.full((1, n, M), float("inf"))
                seg_sz = torch.zeros(1, n, M)
                for j in P:
                    seg_d[0, j, :kk] = X["useg_d"][0, j, slot, off:off + kk, j]
                    seg_sz[0, j, :kk] = X["seg_size"][0, j, slot, off:off + kk]
                same = torch.zeros(1, n, n, dtype=torch.bool)
                for j in P:
                    same[0, j, P] = True
                rem = torch.zeros(1, n, dtype=torch.bool)
                rem[0, P] = True
                used = set()
                self._stage2_ctx = {"members": sorted(P), "segs": segs, "avail": list(range(kk)), "assign": None}
                while bool(rem.any()):
                    mask = np.zeros(M, bool)
                    mask[:kk] = True
                    mask[list(used)] = False
                    if not mask.any():
                        mask[:kk] = True
                    av = torch.as_tensor(mask)[None, None].expand(1, n, M)
                    sel, claim = seg_front(same, rem, av, seg_d, seg_sz)
                    j = int(np.flatnonzero(sel[0].numpy())[0])
                    self.cur_sclaim[j] = int(claim[0, j])
                    if mask.sum() <= 1:
                        s = int(np.flatnonzero(mask)[0])
                    else:
                        s = int(sub_fn(j, mask))
                        if not (0 <= s < M and mask[s]):
                            self._replaced["explore_sub"].add(j)
                            self.mask_violation_step += 1
                            s = rule_explore_sub(self, j, mask)
                        out.add(j)
                        if j in keepers:
                            reselect.add(j)
                    used.add(s)
                    goal0 = self.agents[j].goal
                    self._assign_segment(j, segs[s][0], segs[s][1])
                    if j in keepers:                                # 재선택 통계: goal 구간 유지 / goal 변경
                        st["reselect_uavs"] += 1
                        st["reselect_kept_goal"] += int(goal0 is not None and tuple(goal0) in segs[s][0])
                        st["reselect_goal_change"] += int(self.agents[j].goal != goal0)
                    rem[0, j] = False
                    self.decision_seq.append(("s2", j))
                    self.s2_blob[j] = (g, bid)
                self._stage2_ctx = None
        return out, reselect

    def _assign_segment(self, i, cells, med=None):
        from env.role.clusters import medoid
        a = self.agents[i]
        cells = frozenset(cells)
        a.cluster = cells
        a.cluster_medoid = (tuple(med) if med is not None else medoid(sorted(cells))) if cells else None
        a.goal = self._segment_goal(i, cells, a.goal, a.cluster_medoid) if cells else None
        a.seg_start = self.time

    def _assign_rescue(self, i, ti):
        a = self.agents[i]
        a.mode, a.assigned_target, a.goal = Mode.RESCUE, ti, tuple(self.targets[ti].pos)
        a.cluster = a.cluster_medoid = None
        a.blob = None
        a.last_decision = self.time

    def _assign_return(self, i):
        a = self.agents[i]
        a.mode, a.assigned_target, a.goal = Mode.RETURN, None, tuple(self.depot)
        a.cluster = a.cluster_medoid = None
        a.blob = None
        a.last_decision = self.time

    # ================================================================ 로깅 (A-6, B-8-5)

    def _log_reset(self):
        n = self.n_agents
        self.ev_log, self.uav_log = [], []
        self.ev_stats = {"events": {e: 0 for e in EVENTS + EXPLORE_EVENTS}, "redecide_uavs": 0, "role_switch": 0,
                         "goal_change": 0, "oscillation": 0, "partial_return": 0,
                         "preempt_candidates": 0, "dec_intervals": [], "latch_hold": 0, "link_made": 0, "link_broken": 0,
                         "group_size": [], "nbr_bins": [0] * (len(NEIGHBOR_BINS) + 1),
                         "partial_deficits": [], "rescued_after_partial": [],
                         "role_by_cap": {c: [0, 0, 0] for c in CAP_STATES},
                         "res_redecide": {e: 0 for e in EVENTS}, "blob_dist_ratio": [], "new_cells": [],
                         "join_events": 0, "reselect_uavs": 0, "reselect_kept_goal": 0, "reselect_goal_change": 0,
                         "claim_holders": 0, "claim_not_rescue": 0, "ev2_events": 0, "ev2_redecide": 0, "ev2_rescue": 0,
                         "ev2_rescue_by_claim": 0, "known_to_rescue": [],
                         "s1_events": 0, "s1_empty_blobs": 0, "s1_claim_total": 0, "s1_claim_kept": 0,
                         "s1_participants": 0, "ev2_retained": 0, "ev2_retained_blob_changed": 0,
                         "n_groups": [], "ev2_rescue_nearest": 0, "survivor_truncated": 0,
                         "ev5_causes": {"no_segment": 0, "rescue_invalid": 0, "exploration_complete": 0},
                         "explore_masked_decisions": 0, "forced_return_all_masked": 0}
        self._goal_hist = [[] for _ in range(n)]            # UAV 별 [(목표 시작 시각, 목표)]
        self._partial_pending = [False] * n                 # 부분 귀환 중 (도착하면 그 뒤 구조 수 세기 시작)
        self._after_partial = [None] * n                    # 부분 귀환 충전 후 다음 충전까지 구조 수

    def _log_event(self, e, trigger, targets, group=None):
        self.ev_stats["events"][e] += 1
        if self.events_cfg["log_events"]:
            self.ev_log.append({"t": self.time, "event": e, "trigger": trigger, "group": group,
                                "targets": [int(j) for j in targets]})

    def _log_claim(self, i, role, ev2):
        """담당 후보 통계: 담당 후보가 RESCUE 를 고르지 않음 / 이벤트 2 재결정 UAV 수 · RESCUE 수 (생존자 대상은 _log_claim_targets)."""
        st = self.ev_stats
        if ev2 and self.cur_claim.get(i) is not None:
            st["claim_holders"] += 1
            st["claim_not_rescue"] += int(role is not Mode.RESCUE)
        if ev2:
            self._ev2_now = getattr(self, "_ev2_now", []) + [i]

    def _log_claim_targets(self):
        """이벤트 2 에서 RESCUE 를 고른 UAV 가 그 생존자의 담당 후보였나 (생존자가 정해진 뒤 — top_m 공동 할당 포함)."""
        st = self.ev_stats
        ev2 = getattr(self, "_ev2_now", [])
        if ev2:
            st["ev2_events"] += 1
            st["ev2_redecide"] += len(ev2)
            res = [i for i in ev2 if self.agents[i].mode is Mode.RESCUE and self.agents[i].assigned_target is not None]
            st["ev2_rescue"] += len(res)
            st["ev2_rescue_by_claim"] += sum(1 for i in res if self.cur_claim.get(i) == self.agents[i].assigned_target)
            for i in res:                                          # 그 생존자에 그룹에서 가장 가까운 UAV 였나 (동률 포함)
                pos = self.targets[self.agents[i].assigned_target].pos
                st["ev2_rescue_nearest"] += int(self._dist(i, pos) <= min(self._dist(j, pos) for j in self.groups[self.comp[i]]))
        self._ev2_now = []

    def _log_role_choice(self, i, r):
        a = self.agents[i]
        self.ev_stats["role_by_cap"][cap_state(a)][r] += 1
        if ROLES[r] is Mode.RETURN and 0 < a.capacity < a.capacity_max:         # 부분 귀환 (가득 찬 RETURN = 전부 막힌 fallback)
            self.ev_stats["partial_return"] += 1
            self.ev_stats["partial_deficits"].append(int(a.capacity_max - a.capacity))
            self._partial_pending[i] = True
        elif ROLES[r] is not Mode.RETURN:
            self._partial_pending[i] = False

    def goal_token(self, i):
        a = self.agents[i]
        if a.mode is Mode.RESCUE and a.assigned_target is not None:
            return f"T{a.assigned_target}"
        if a.mode is Mode.EXPLORE and a.blob is not None:
            return f"C{a.blob}"
        if a.mode is Mode.RETURN:
            return "D"
        return "-"

    def _log_decisions(self, prev_role, prev_tok):
        W = int(self.rewards_cfg["role"]["osc_window"])
        for i in sorted(self._decided_now):
            self.ev_stats["redecide_uavs"] += 1
            a = self.agents[i]
            if a.mode is not prev_role[i]:
                self.ev_stats["role_switch"] += 1
                if self.events_cfg["log_events"]:
                    self.uav_log.append({"t": self.time, "uav": i, "type": "role", "old": prev_role[i].name,
                                         "new": a.mode.name})
            tok = self.goal_token(i)
            if tok != prev_tok[i]:
                self.ev_stats["goal_change"] += 1
                h = self._goal_hist[i]
                if len(h) >= 2 and h[-2][1] == tok and self.time - h[-1][0] <= W:
                    self.ev_stats["oscillation"] += 1             # A -> B -> A 가 창 W 안에
                h.append((self.time, tok))
                if self.events_cfg["log_events"]:
                    self.uav_log.append({"t": self.time, "uav": i, "type": "goal", "old": prev_tok[i], "new": tok})

    def _log_after_move(self, rescued, events_after):
        for ev in rescued:                                        # 발견 (그룹에 알려짐) -> 구조 도착 step
            t = self.targets[ev.target]
            if t.discovered_at is not None:
                self.ev_stats["known_to_rescue"].append(self.time - t.discovered_at)
        for ev in rescued:
            if self._after_partial[ev.agent] is not None:
                self._after_partial[ev.agent] += 1
        for i, e in enumerate(events_after):
            if "4" in e:                                          # 충전 완료
                if self._after_partial[i] is not None:
                    self.ev_stats["rescued_after_partial"].append(self._after_partial[i])
                    self._after_partial[i] = None
                if self._partial_pending[i]:
                    self._after_partial[i] = 0
                    self._partial_pending[i] = False

    def event_summary(self):
        """에피소드 통계 (info["event_stats"]): 종류별 횟수, 재결정 · 역할 전환 · 목표 변경 · 진동, 부분 귀환, RESCUE 재결정 원인,
        고른 덩어리 거리 비 (고른 / 가장 가까운, 각 BFS + 1), EXPLORE step 당 새 관측 셀."""
        st = self.ev_stats
        after = st["rescued_after_partial"] + [c for c in self._after_partial if c is not None]
        nb = np.asarray(st["nbr_bins"], float)
        out = {"preempt_candidates": st["preempt_candidates"],
               "dec_interval": float(np.mean(st["dec_intervals"])) if st["dec_intervals"] else float("nan"),
               "latch_hold": st["latch_hold"], "link_made": st["link_made"], "link_broken": st["link_broken"],
               "group_size": float(np.mean(st["group_size"])) if st["group_size"] else float("nan"),
               "n_groups": float(np.mean(st["n_groups"])) if st["n_groups"] else float("nan"),
               "ev2_rescue_nearest": st["ev2_rescue_nearest"] / st["ev2_rescue"] if st["ev2_rescue"] else float("nan"),
               "survivor_truncated": st["survivor_truncated"],
               "ev5_causes": dict(st["ev5_causes"]), "explore_masked_decisions": st["explore_masked_decisions"],
               "forced_return_all_masked": st["forced_return_all_masked"],
               "nbr_dist_hist": (nb / max(1.0, nb.sum())).tolist(),
               "events": dict(st["events"]), "redecide_uavs": st["redecide_uavs"], "role_switch": st["role_switch"],
               "goal_change": st["goal_change"], "oscillation": st["oscillation"],
               "partial_return": st["partial_return"], "partial_deficits": list(st["partial_deficits"]),
               "rescued_after_partial": after,
               "role_by_cap": {c: list(v) for c, v in st["role_by_cap"].items()},
               "rescue_redecide_causes": dict(st["res_redecide"]),
               "blob_dist_ratio": float(np.mean(st["blob_dist_ratio"])) if st["blob_dist_ratio"] else float("nan"),
               "new_cells_per_explore_step": float(np.mean(st["new_cells"])) if st["new_cells"] else float("nan"),
               "join_events": st["join_events"], "reselect_uavs": st["reselect_uavs"],
               "reselect_kept_goal": st["reselect_kept_goal"] / st["reselect_uavs"] if st["reselect_uavs"] else float("nan"),
               "reselect_goal_change": st["reselect_goal_change"],
               "ev2_rescue_by_claim": st["ev2_rescue_by_claim"] / st["ev2_rescue"] if st["ev2_rescue"] else float("nan"),
               "ev2_rescue_other": 1 - st["ev2_rescue_by_claim"] / st["ev2_rescue"] if st["ev2_rescue"] else float("nan"),
               "s1_claim_kept": st["s1_claim_kept"] / st["s1_claim_total"] if st["s1_claim_total"] else float("nan"),
               "s1_empty_blobs": st["s1_empty_blobs"] / st["s1_events"] if st["s1_events"] else float("nan"),
               "s1_participants": st["s1_participants"] / st["s1_events"] if st["s1_events"] else float("nan"),
               "ev2_retained": st["ev2_retained"], "ev2_retained_blob_changed": st["ev2_retained_blob_changed"],
               "claim_not_rescue": st["claim_not_rescue"] / st["claim_holders"] if st["claim_holders"] else float("nan"),
               "ev2_redecide": st["ev2_redecide"] / st["ev2_events"] if st["ev2_events"] else float("nan"),
               "ev2_rescue": st["ev2_rescue"] / st["ev2_events"] if st["ev2_events"] else float("nan"),
               "known_to_rescue": float(np.mean(st["known_to_rescue"])) if st["known_to_rescue"] else float("nan")}
        return out

    def _write_log(self, summary):
        d = self.events_cfg.get("log_dir")
        if not d or not self.events_cfg["log_events"]:
            return
        os.makedirs(d, exist_ok=True)
        tag = self.episode_tag if self.episode_tag is not None else id(self)
        with open(os.path.join(d, f"episode_{tag}.json"), "w") as f:
            json.dump({"events": self.ev_log, "uav": self.uav_log, "stats": summary}, f)

    # ================================================================ step

    def step(self, decider=None):
        """decider: role / rescue / explore (/ explore_sub) (sim, i, mask) 를 가진 객체 (없으면 규칙 정책).
        결정 순서는 _decide 가 정한다."""
        decider = decider or RuleDecider()
        sub = getattr(decider, "explore_sub", None) or (lambda sim, i, m: rule_explore_sub(sim, i, m))
        return self._step_role(lambda i, m: decider.role(self, i, m),
                               lambda i, m: decider.rescue(self, i, m),
                               lambda i, m: decider.explore(self, i, m),
                               lambda i, m: sub(self, i, m))

    def step_explicit(self, actions):
        """actions = dict(role, rescue, explore, explore_sub) 각 (N,) — 결정이 필요 없는 UAV 의 값은 무시.
        모델이 순차 디코딩한 결과를 그대로 적용. policy_mode rule 헤드 / 2 단계 hungarian 은 규칙 정책."""
        arr = {k: np.asarray(actions.get(k, np.zeros(self.n_agents))).reshape(-1)
               for k in ("role", "rescue", "explore", "explore_sub")}
        keys = HEADS + ("explore_sub",)
        first = {h: set() for h in keys}
        self._ruled = {h: set() for h in keys}
        self._replaced = {h: set() for h in keys}

        def pick(kind, rule, learned):
            def f(i, m):
                if not learned or i in first[kind]:
                    self._ruled[kind].add(i)
                    return rule(self, i, m)
                first[kind].add(i)
                return int(arr[kind][i])
            return f
        pm = self.policy_mode
        done, info = self._step_role(pick("role", rule_role, pm["role"] == "learned"),
                                     pick("rescue", rule_rescue, pm["rescue"] == "learned"),
                                     pick("explore", rule_explore, pm["explore"] == "learned"),
                                     pick("explore_sub", rule_explore_sub, self.sub_learned))
        # 학습 마스크: 결정을 실제로 했고 그 행동이 모델 출력 그대로 최종 실행된 경우만 1
        # explore = 1 단계 + 2 단계 하나의 행동 — 이번 step 정책이 한 단계 중 하나라도 대체되면 0
        for h in ("role", "rescue"):
            ok = set(first[h]) - self._ruled[h] - self._replaced[h]
            info[f"{h}_train"] = [float(info[f"{h}_valid"][i] > 0 and i in ok) for i in range(self.n_agents)]
        e1, e2 = info["explore_valid_s1"], info["explore_valid_s2"]
        tr = []
        for i in range(self.n_agents):
            l1 = e1[i] and pm["explore"] == "learned"
            l2 = e2[i] and self.sub_learned
            bad = (i in self._ruled["explore"] and e1[i]) or i in self._replaced["explore"] or \
                (l2 and i in self._replaced["explore_sub"])
            tr.append(1.0 if (l1 or l2) and not bad else 0.0)
        info["explore_train"] = tr
        return done, info

    def _step_role(self, role_fn, rescue_fn, explore_fn, sub_fn):
        n = self.n_agents
        self.mask_violation_step = 0
        events_now = [set(e) for e in self.events]
        for i, e in enumerate(events_now):
            if e:
                self.last_event[i] = (self.time, sorted(e))
        for e in events_now:
            for k in e:
                self.event_totals[k] += 1
        prev_role = [a.mode for a in self.agents]
        prev_tok = [self.goal_token(i) for i in range(n)]

        # ---- 결정 ----
        role_valid, sub_valid, redecide, violations = self._decide(role_fn, rescue_fn, explore_fn, sub_fn)
        violations += self.mask_violation_step
        self.decided_role = role_valid
        self.redecide_total += redecide
        self.mask_violation_total += violations
        self._log_decisions(prev_role, prev_tok)
        chosen_mode = [a.mode for a in self.agents]
        chosen_target = [a.assigned_target for a in self.agents]
        switched = sum(1 for i in range(n) if chosen_mode[i] is not prev_role[i])
        self.stats["role_switch"] += switched
        for i in range(n):
            if chosen_mode[i] is not prev_role[i]:
                self.agents[i].role_since = self.time

        # ---- 이동 (BFS 다음 칸) ----
        idle_no_goal = sum(1 for a in self.agents if a.alive and a.goal is None)   # 결정 후에도 목표가 없는 UAV
        prev_pos = [a.pos for a in self.agents]
        for i, a in enumerate(self.agents):
            if not a.alive or a.goal is None or a.pos == a.goal:
                continue
            path = path_to(self.parents[i], a.pos, a.goal, self.cols)
            if path is None or len(path) < 2:
                continue
            nxt = path[1]
            act = Robot.ACTIONS[Robot.MOVES.index((nxt[0] - a.pos[0], nxt[1] - a.pos[1]))]
            a.move(act, self.obstacles, self.rows, self.cols)   # 미탐색이 실제 장애물이었으면 제자리
        cells = {}
        for a in self.agents:
            if a.pos != self.depot:
                cells[a.pos] = cells.get(a.pos, 0) + 1
        same_cell = sum(1 for v in cells.values() if v >= 2)
        self.same_cell_total += same_cell

        # ---- 시간 진행 + 센싱 / 병합 / 발견 / 구조 / 재보충 / 마감 ----
        ctx = StepContext(time=self.time, n_agents=n)
        was_discovered = [t.discovered for t in self.targets]
        prev_state = [(a.mode, a.blob, a.cluster) for a in self.agents]
        prev_seg = [a.cluster for a in self.agents]
        prev_med = [a.cluster_medoid for a in self.agents]
        self.time += 1
        self.knowledge.update(self.active_target_positions())
        self._update_discovery()
        rescued = self._process_rescues()
        self._process_refill(ctx)
        expired = self._process_deadlines()
        self.beliefs.update(self, self.knowledge.last_visible)
        newly_found = [ti for ti, was in enumerate(was_discovered) if not was and self.targets[ti].discovered]

        # ---- 다음 결정 준비: 덩어리 / 구간 / Role 이벤트 / Explore 이벤트 ----
        self._prepare_role()
        exhausted = self._explore_after_update(prev_state, prev_seg, prev_med)
        rescued_by = {ev.agent for ev in rescued}
        self.last_events = events_now
        self.events = self._collect_events(prev_pos, rescued_by, exhausted)
        self._log_after_move(rescued, self.events)
        self._set_decide_order()
        self._comm_stats()
        self.step_log = [f"t{self.time - 1:>3} blob " + s for s in self.blob_events]

        # ---- 보상 ----
        done = self._is_done()
        rew = self.role_rewards(rescued, expired, done,
                                explore_active=[self.agents[i].alive and chosen_mode[i] is Mode.EXPLORE for i in range(n)],
                                explore_new=[bool(sub_valid["explore_s1"][i] or sub_valid["explore_s2"][i]
                                                  or sub_valid["explore_reselect"][i]) for i in range(n)])

        # ---- info ----
        exp_valid = [bool(sub_valid["explore_s1"][i] or sub_valid["explore_s2"][i]) for i in range(n)]
        in_res = [chosen_mode[i] is Mode.RESCUE and chosen_target[i] is not None for i in range(n)]
        ev_counts = {k: sum(1 for e in events_now if k in e) for k in EVENTS}
        ev_counts["split"] = len(self._split_groups)
        ev_counts["merge"] = len(self._merged)
        m_list = [len(self.assignees(g, b.id)) for g, sl in enumerate(self.group_clusters) for b in sl]
        m_list = [m for m in m_list if m > 0]
        info = self._info(rescued, expired)
        info.update({
            "reward_role": rew["role"], "reward_rescue": rew["rescue"], "reward_explore": rew["explore"],
            "role_valid": [float(v) for v in role_valid],
            "rescue_valid": [float(v) for v in sub_valid["rescue"]],
            "explore_valid": [float(v) for v in exp_valid],
            "explore_valid_s1": [float(v) for v in sub_valid["explore_s1"]],
            "explore_valid_s2": [float(v) for v in sub_valid["explore_s2"]],
            "explore_reselect": [float(v) for v in sub_valid["explore_reselect"]],      # stage-2-only 재선택 (기존 배정자)
            "explore_active": [float(self.agents[i].alive and chosen_mode[i] is Mode.EXPLORE) for i in range(n)],   # 옵션 진행
            "in_rescue_seg": [float(v) for v in in_res],                 # 이번 step RESCUE (배정 생존자 있음)
            "role": [a.mode.value for a in self.agents],
            "events": ev_counts,
            "redecide": redecide, "mask_violations": violations,
            "rescue_conflict_cross": float(self._cross_group_conflict()),
            "group_size_mean": float(np.mean([1 + self.knowledge.reach[i].sum() for i in range(n)])),
            "cap0": sum(1 for a in self.agents if a.capacity == 0),
            "cap0_explore": sum(1 for a in self.agents if a.capacity == 0 and a.mode is Mode.EXPLORE),
            "same_cell_total": self.same_cell_total,
            "idle_no_goal": idle_no_goal,
            "m_mean": float(np.mean(m_list)) if m_list else 0.0,
        })
        if done:                                          # 에피소드 통계 (콜백 / run_rules) — 끝 step 에만 싣는다
            info["explore_stats"] = dict(self.stats, seg_life=list(self.stats["seg_life"]))
            summary = self.event_summary()
            summary["rewards"] = self.reward_summary()
            info["event_stats"] = summary
            self._write_log(summary)
        ctx.chosen_mode, ctx.chosen_target = chosen_mode, chosen_target
        ctx.free_choice = list(role_valid)
        ctx.rescued_now, ctx.expired_now = rescued, expired
        ctx.newly_discovered_targets = newly_found
        ctx.modes = [a.mode for a in self.agents]
        ctx.assigned_target = [a.assigned_target for a in self.agents]
        self.last_ctx = ctx
        if self.recorder is not None:
            self.recorder.on_step(self, ctx, info)
        return done, info

    def _cross_group_conflict(self):
        """서로 다른 통신 그룹의 UAV 가 같은 조난자를 쫓고 있는가 (서로 몰라서 생기는 중복)."""
        by = {}
        for i, a in enumerate(self.agents):
            if a.mode is Mode.RESCUE and a.assigned_target is not None:
                by.setdefault(a.assigned_target, set()).add(self.comp[i])
        return any(len(g) > 1 for g in by.values())

    def map_features(self, i):
        """critic role_group 의 지도 특징 3 개 (UAV i 의 그룹 공유 지도 기준) — 원래 단위 (칸 수). 도달 불가 / 없음 = None."""
        f = self.fields[i]
        cl = self.clusters_of(i)
        meds = [int(f[c.medoid]) for c in cl if int(f[c.medoid]) >= 0]
        dd = int(f[self.depot])
        return {"nearest_cluster_dist": min(meds) if meds else None,
                "largest_cluster_size": max((c.size for c in cl), default=0),
                "depot_dist": dd if dd >= 0 else None}

    # ================================================================ 보상

    def _reward_reset(self):
        n = self.n_agents
        self._role_hist = [[] for _ in range(n)]       # UAV 별 역할 전환 [(시각, 이전, 다음, 강제)]
        self._osc_pen = [0.0] * n                     # 이번 step role 진동 penalty
        self._res_pending = [0.0] * n                 # 이번 step rescue 중복 추적 penalty (-w_dup)
        self._sf_pending = [0.0] * n                  # 이번 step rescue (S - F) / N_T (이벤트 끝, 그룹 공유)
        self._bonus_pending = [0.0] * n               # 이번 step rescue 보너스 (옵션이 끝난 UAV: 1[F = 0] [(1 - w_c) e^{-d} + w_c c̄])
        self._res_opt = [None] * n                    # rescue 옵션: {F, d, csum, n} (결정 ~ 구조 완료 / 이벤트 5 / 끝)
        self._dup_done = [True] * n                   # 지금 rescue 결정에서 중복 penalty 를 이미 받았나 (결정당 1 회)
        self._opt = [None] * n                        # explore 옵션 (로그용): [t0, 누적 보상 (할인 없음)]
        self._explore_seen_all = [False] * n          # 그룹 belief 발견 수가 N_T 에 도달한 것을 이미 겪었나 (이벤트 5)
        self.rw = {k: np.zeros(n) for k in REWARD_ITEMS}
        self.rw_stats = {"osc": 0, "forced": 0, "voluntary": 0, "dup_in": 0, "dup_out": 0,
                         "rescue_events": 0, "res_S": [], "res_F": [], "res_U": [], "res_F_unavoidable": [],
                         "res_bonus": [], "res_sf_term": 0.0, "res_bonus_term": 0.0, "c_bar": [], "conn_mult": [],
                         "c_real": [], "psi_f": [], "explore_term": 0.0, "opt_return": [], "opt_len": [], "res_opt_unclosed": 0}

    def _role_transition(self, i, prev, new, forced):
        """역할 전환 기록 + 진동 판정: osc_window 안에 A -> B -> A 이고 두 전환 모두 비강제면 되돌아가는 결정에 -c_osc."""
        if new is prev:
            return
        st = self.rw_stats
        st["forced" if forced else "voluntary"] += 1
        h = self._role_hist[i]
        h.append((self.time, prev, new, forced))
        if len(h) >= 2:
            t1, a1, b1, f1 = h[-2]
            _, a2, b2, f2 = h[-1]
            if a1 is b2 and b1 is a2 and not f1 and not f2 and self.time - t1 <= int(self.rewards_cfg["role"]["osc_window"]):
                self._osc_pen[i] += float(self.rewards_cfg["role"]["c_osc"])
                st["osc"] += 1

    def real_connectivity(self):
        """실제 연결성 C_t: 팀 전체 활동 UAV (역할 무관, depot 포함) 의 현재 위치로 pair_connectivity (진단 로그 conn/real_mean — 보상에는 쓰지 않음)."""
        G = [l for l, a in enumerate(self.agents) if a.alive]
        return pair_connectivity([self.agents[l].pos for l in G], [self.agents[l].comm_range for l in G])

    def group_discovered(self, g):
        """그룹 g 의 belief 발견 수: 멤버 지도 (known_free) 에 들어온 발견된 생존자 수 (구조 / 만료 포함, step 내 캐시)."""
        key = ("discovered", g)
        if key not in self._pair_cache:
            members = self.groups[g]
            self._pair_cache[key] = sum(1 for ti, t in enumerate(self.targets) if t.discovered and (
                any(self.knowledge.known_free[j][t.pos] for j in members) or any(ti in self.beliefs.rescued[j] for j in members)))
        return self._pair_cache[key]

    def exploration_complete(self, g):
        """그룹 g 가 모든 생존자를 발견했다고 아는가 (belief 발견 수 = N_T) — EXPLORE 마스크 / 이벤트 5."""
        return len(self.targets) > 0 and self.group_discovered(g) >= len(self.targets)

    def rescue_urgent(self, g, G):
        """rescue 보상 입력. G = 이번 이벤트에 RESCUE 를 고른 그룹 UAV (G_res). 반환 (U_ε, σ, d) —
        σ_u = τ_u - t - min_{l ∈ G} BFS(l, u) (이벤트 시점 위치), U_ε = 그룹이 아는 생존자 중 G 밖의 UAV (다른 그룹 포함) 가
        맡지 않았고 0 <= σ_u <= epsilon_urgent 인 것. d[l][u] = BFS 거리 (도달 불가 10**9)."""
        eps = float(self.rewards_cfg["rescue"]["epsilon_urgent"])
        held = {a.assigned_target for j, a in enumerate(self.agents)
                if j not in G and a.mode is Mode.RESCUE and a.assigned_target is not None}
        known = [k for k in self._group_known(g) if k not in held]
        d = {l: {k: self._dist(l, self.targets[k].pos) for k in known} for l in G}
        sig, urgent = {}, []
        for k in known:
            sig[k] = _deadline(self.targets[k]) - self.time - min(d[l][k] for l in G)
            if 0 <= sig[k] <= eps:
                urgent.append(k)
        return urgent, sig, d

    def _rescue_event_reward(self, g, G):
        """이벤트 끝 rescue 보상 (G = G_res): 결정 step 에 (S - F) / N_T (G 전원 같은 값) 와 중복 -w_dup (_check_dup),
        옵션 (결정 ~ 구조 완료 / 이벤트 5 / 끝) 이 끝나는 step 에 개인 보너스 1[F = 0] [(1 - w_c) e^{-d} + w_c c̄_i] (role_rewards).
        S = G 가 할당했고 시간 안에 도착하는 생존자 수, F = U_ε 중 G 누구에게도 할당되지 않은 수,
        d = Σ_{l ∈ G} BFS(l, a_l) / (d_max |G|)."""
        G = [i for i in G if self.agents[i].assigned_target is not None]
        if not G:
            return
        for i in G:
            self._dup_done[i] = False
        n_t = max(1, len(self.targets))
        urgent, sig, _ = self.rescue_urgent(g, G)
        a = {l: self.agents[l].assigned_target for l in G}
        dist = {l: self._dist(l, self.targets[a[l]].pos) for l in G}
        S = sum(1 for l in G if self.time + dist[l] <= _deadline(self.targets[a[l]]))
        F = sum(1 for k in urgent if k not in a.values())
        d = sum(dist[l] for l in G) / (self.d_max * len(G))
        sf = (S - F) / n_t
        for l in G:
            self._sf_pending[l] += sf
            if self._res_opt[l] is not None:                      # 닫히지 않은 옵션 (정상이면 없음) — 보상 없이 버림
                self.rw_stats["res_opt_unclosed"] += 1
            self._res_opt[l] = {"F": F, "d": d, "csum": 0.0, "n": 0}
        st = self.rw_stats
        st["rescue_events"] += 1
        st["res_S"].append(S)
        st["res_F"].append(F)
        st["res_U"].append(len(urgent))
        st["res_F_unavoidable"].append(max(0, len(urgent) - len(G)))
        st["res_bonus"].append(float(F == 0))
        st["res_sf_term"] += sf
        self._check_dup(G)

    def _check_dup(self, uavs=None):
        """중복 추적: 같은 생존자를 목표로 하는 다른 UAV (그룹 밖 포함, 전체 정보) 가 더 가까우면 -w_dup (결정당 1 회)."""
        w = float(self.rewards_cfg["rescue"]["w_dup"])
        for i in (uavs if uavs is not None else range(self.n_agents)):
            a = self.agents[i]
            if self._dup_done[i] or a.mode is not Mode.RESCUE or a.assigned_target is None:
                continue
            ti = a.assigned_target
            di = self._dist(i, self.targets[ti].pos)
            for l, b in enumerate(self.agents):
                if l != i and b.mode is Mode.RESCUE and b.assigned_target == ti and self._dist(l, self.targets[ti].pos) < di:
                    self._res_pending[i] -= w
                    self._dup_done[i] = True
                    self.rw_stats["dup_in" if self.comp[l] == self.comp[i] else "dup_out"] += 1
                    break

    def kappa_cells(self, i, S, ignore=()):
        """κ = (d_{i,S} + 1) / (|S^eff| + 1). d_{i,S} = S 까지 BFS, S^eff = S 중 다른 EXPLORE UAV (env 전체, 그룹 무관,
        ignore 제외) 의 구간에도 있고 그 UAV 의 BFS 가 더 짧은 셀을 뺀 것. 반환 (κ, |S^eff|, |S|) / S 가 비거나 도달 불가면 None."""
        if not S:
            return None
        f = self.fields[i]
        reach = [int(f[c]) for c in S if f[c] >= 0]
        if not reach:
            return None
        others = [j for j, b in enumerate(self.agents) if j != i and j not in ignore and b.alive
                  and b.mode is Mode.EXPLORE and b.cluster]
        eff = 0
        for c in S:
            di = int(f[c]) if f[c] >= 0 else 10 ** 9
            if not any(c in self.agents[j].cluster and 0 <= int(self.fields[j][c]) < di for j in others):
                eff += 1
        return (min(reach) + 1) / (eff + 1), eff, len(S)

    def explore_coverage(self):
        """Δcov^(i) (팀 전역 기준 이번 step 처음 관측된 free 셀을 같은 step 에 본 UAV 수로 나눠 가짐, Σ_i = 처음 관측된 free 셀 수)."""
        new_sets = getattr(self.knowledge, "new_sets", None) or [set() for _ in range(self.n_agents)]
        seen = {}
        for cells in new_sets:
            for c in cells:
                seen[c] = seen.get(c, 0) + 1
        return [sum(1.0 / seen[c] for c in cells if not self.obstacles[c[0]][c[1]]) for cells in new_sets]

    def psi_f(self):
        """ψ^f = 팀 전체에서 지금까지 발견된 생존자 수 / N_T (전역, 학습용 privileged)."""
        return sum(1 for t in self.targets if t.discovered) / max(1, len(self.targets))

    def role_rewards(self, rescued, expired, done=False, explore_active=None, explore_new=None):
        """head 별 보상 (N,) + 항목별 기록 (self.rw).
        role    (|R_t| - w_x |X_t|) / N_T - c_osc 1_osc (매 step, SMDP)
        rescue  결정 step 에 (S - F) / N_T - w_dup, 옵션이 끝나는 step 에 개인 보너스 (학습은 옵션 합, A = R - V)
        explore 매 step 내 기여 r = (1 - ψ^f) Δcov^(i) / |free| (1 + w_mult c_{i,t}) — 이번 step EXPLORE 인 UAV 만 (explore_active),
                학습은 옵션 단위 할인 누적 (training/role/ppo.py). explore_new = 이번 step explore 결정 (옵션 시작, 로그용)."""
        rc, n, n_t = self.rewards_cfg, self.n_agents, max(1, len(self.targets))
        n_x = len(expired)
        if done and rc["role"]["terminal_unrescued_as_expired"]:
            n_x += sum(1 for t in self.targets if t.active)
        team = (len(rescued) - float(rc["role"]["w_x"]) * n_x) / n_t
        self.rw_stats["c_real"].append(self.real_connectivity())          # 진단 로그 (보상 아님)
        role = [team - self._osc_pen[i] for i in range(n)]
        c = self.personal_connectivity()
        self._rescue_options(c, rescued, done)
        rescue = [self._sf_pending[i] + self._bonus_pending[i] + self._res_pending[i] for i in range(n)]
        if explore_active is None:
            explore_active = [a.alive and a.mode is Mode.EXPLORE for a in self.agents]
        psi = self.psi_f()
        self.rw_stats["psi_f"].append(psi)
        cov = self.explore_coverage()
        n_free = float(self.n_free_cells())
        w_mult = float(rc["explore"]["w_mult"])
        explore = [(1.0 - psi) * cov[i] / n_free * (1.0 + w_mult * c[i]) if explore_active[i] else 0.0 for i in range(n)]
        self.rw_stats["conn_mult"] += [1.0 + w_mult * c[i] for i in range(n) if explore_active[i] and cov[i] > 0]
        self.rw_stats["explore_term"] += float(sum(explore))
        self._track_options(explore, explore_active, explore_new or [False] * n, done)
        for i, a in enumerate(self.agents):                       # EXPLORE step 당 새로 관측한 셀 (통계)
            if a.alive and a.mode is Mode.EXPLORE:
                self.ev_stats["new_cells"].append(float(self.knowledge.newly_sensed[i]))
        self.rw["role_team"] += team
        self.rw["role_osc"] -= np.asarray(self._osc_pen)
        self.rw["rescue_sf"] += np.asarray(self._sf_pending)
        self.rw["rescue_bonus"] += np.asarray(self._bonus_pending)
        self.rw["rescue_dup"] += np.asarray(self._res_pending)
        self.rw["explore_cov"] += np.asarray(explore)
        self._osc_pen, self._res_pending = [0.0] * n, [0.0] * n
        self._sf_pending, self._bonus_pending = [0.0] * n, [0.0] * n
        return {"role": role, "rescue": rescue, "explore": explore}

    def personal_connectivity(self):
        """c_{i,t} (N,): 팀 전체 활동 UAV 의 현재 위치로 개인 연결도 (비활동 UAV = 0)."""
        G = [l for l, a in enumerate(self.agents) if a.alive]
        out = [0.0] * self.n_agents
        for l, v in zip(G, personal_connectivity([self.agents[l].pos for l in G], [self.agents[l].comm_range for l in G])):
            out[l] = v
        return out

    def _rescue_options(self, c, rescued, done):
        """열린 rescue 옵션에 이번 step c_{i,t} 를 더하고, 구조 완료 (이번 step 구조) / 이벤트 5 / 에피소드 끝이면 닫아
        이번 step 보상에 개인 보너스 1[F = 0] [(1 - w_c) e^{-d} + w_c c̄_i] 를 싣는다 (다음 step 이 재결정 = 옵션 경계)."""
        w_c = float(self.rewards_cfg["rescue"]["w_c"])
        by = {getattr(ev, "agent", None) for ev in rescued}
        events = getattr(self, "events", None) or [set()] * self.n_agents
        st = self.rw_stats
        for i, o in enumerate(self._res_opt):
            if o is None:
                continue
            o["csum"] += c[i]
            o["n"] += 1
            if i in by or "5" in events[i] or done:
                cbar = o["csum"] / o["n"]
                bonus = ((1 - w_c) * float(np.exp(-o["d"])) + w_c * cbar) if o["F"] == 0 else 0.0
                self._bonus_pending[i] += bonus
                st["c_bar"].append(cbar)
                st["res_bonus_term"] += bonus
                self._res_opt[i] = None

    def _track_options(self, r, active, new, done):
        """explore 옵션 로그 (학습의 R_i 와 같은 경계, 할인 없음): 새 explore 결정이면 이전 옵션을 닫고 새로 시작,
        EXPLORE 가 아니게 되거나 에피소드가 끝나면 닫는다. 기록 = 옵션 보상 합 / 길이 (t1 - t0)."""
        st = self.rw_stats
        for i in range(self.n_agents):
            o = self._opt[i]
            if o is not None and (new[i] or not active[i]):
                st["opt_return"].append(o[1])
                st["opt_len"].append(self.time - 1 - o[0])
                self._opt[i] = o = None
            if new[i]:
                self._opt[i] = o = [self.time - 1, 0.0]
            if o is not None and active[i]:
                o[1] += r[i]
            if done and self._opt[i] is not None:
                st["opt_return"].append(self._opt[i][1])
                st["opt_len"].append(self.time - self._opt[i][0])
                self._opt[i] = None

    def _comm_stats(self):
        """링크 생성 / 해제, 그룹 크기, 1-hop 이웃 거리 / R_c 분포 (매 step)."""
        st, kn = self.ev_stats, self.knowledge
        st["link_made"] += kn.link_made
        st["link_broken"] += kn.link_broken
        st["group_size"].append(float(np.mean([len(self.groups[self.comp[i]]) for i in range(self.n_agents)])))
        st["n_groups"].append(len(self.groups))
        for i, a in enumerate(self.agents):
            for l in self.neighbors(i):
                if l <= i:
                    continue
                d = max(abs(a.pos[0] - self.agents[l].pos[0]), abs(a.pos[1] - self.agents[l].pos[1])) / \
                    min(a.comm_range, self.agents[l].comm_range)
                st["nbr_bins"][int(np.searchsorted(NEIGHBOR_BINS, d, side="left"))] += 1

    def reward_summary(self):
        """에피소드 보상 항목 합 (UAV 평균) + rescue (S / F / U_ε / 보너스) · explore (옵션) · ψ^f 진행 곡선 · 진동 / 중복 통계
        (info["event_stats"]["rewards"])."""
        st = self.rw_stats
        mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
        psi = np.asarray(st["psi_f"], float)
        edges = np.linspace(0, len(psi), 11).astype(int)
        bins = [float(psi[edges[b]:max(edges[b + 1], edges[b] + 1)].mean()) if len(psi) else float("nan") for b in range(10)]
        return {"sum": {k: float(v.mean()) for k, v in self.rw.items()},
                "osc": st["osc"], "forced": st["forced"], "voluntary": st["voluntary"],
                "dup_in": st["dup_in"], "dup_out": st["dup_out"],
                "rescue_events": st["rescue_events"],
                "rescue_S": mean(st["res_S"]), "rescue_F": mean(st["res_F"]), "rescue_U_eps_size": mean(st["res_U"]),
                "rescue_F_unavoidable": mean(st["res_F_unavoidable"]), "rescue_bonus_rate": mean(st["res_bonus"]),
                "rescue_SF_term": st["res_sf_term"], "rescue_bonus_term": st["res_bonus_term"],
                "rescue_c_bar": mean(st["c_bar"]), "explore_conn_mult": mean(st["conn_mult"]),
                "role_rescue_term": float(self.rw["role_team"].mean()),
                "conn_real_mean": mean(st["c_real"]),
                "explore_term": st["explore_term"], "explore_return": mean(st["opt_return"]),
                "explore_option_len": mean(st["opt_len"]), "explore_options": len(st["opt_len"]),
                "psi_f_bins": bins}

    # ================================================================ 관측

    def planned_path(self, i):
        """시각화용: 현재 목표점까지 BFS 경로와 local 창 waypoint (이동에 쓰는 경로와 같다)."""
        a = self.agents[i]
        if a.goal is None:
            return None, None, None
        path = path_to(self.parents[i], a.pos, a.goal, self.cols)
        if path is None:
            return None, None, None
        return path, window_waypoint(path, a.pos, self.knowledge.local_size // 2), a.mode.value

    def _pair_dist(self, g, cells):
        """그룹 g 지도에서 cells 사이 BFS 거리 행렬 (/ d_max, 도달 불가 1) — rescue 간선 / kNN 기준. step 내 캐시."""
        key = (g, tuple(cells))
        if key not in self._pair_cache:
            D = self.bfs_from(g, cells)
            m = np.ones((len(cells), len(cells)), np.float32)
            if D is not None:
                for a_, c in enumerate(cells):
                    v = D[:, c[0], c[1]]
                    m[:, a_] = np.where(np.isfinite(v), np.minimum(1.0, v / self.d_max), 1.0)
            self._pair_cache[key] = m
        return self._pair_cache[key]

    def _group_active(self, g):
        """이번 step 결정이 있는 그룹 (role / s1 / s2) — 간선 · 구간 분할 표는 이 그룹만 계산."""
        return any(self.kind[i] is not None for i in self.groups[g])

    def goal_point(self, l):
        """UAV l 의 목표 지점 g_l (그래프 U 노드 / U–U 간선 / 연결성 C): RESCUE = 목표 생존자, EXPLORE = 세그먼트 medoid
        (없으면 goal 셀), RETURN = depot, 없으면 현재 위치. 반환 (셀, 목표 있음)."""
        a = self.agents[l]
        if a.mode is Mode.RESCUE and a.assigned_target is not None:
            return tuple(self.targets[a.assigned_target].pos), True
        if a.mode is Mode.RETURN:
            return tuple(self.depot), True
        if a.mode is Mode.EXPLORE and a.cluster_medoid is not None:
            return tuple(a.cluster_medoid), True
        if a.mode is Mode.EXPLORE and a.goal is not None:
            return tuple(a.goal), True
        return tuple(a.pos), False

    def s_nodes(self, i):
        """S 노드 (UAV i 그래프의 생존자) = i 의 할당 가능 후보 (victim_slots) 먼저, 그다음 그룹이 아는 나머지 미구조 · 미만료
        생존자 (맡겨짐 / i 는 시간 안에 못 닿음) 를 BFS 순으로, 최대 s_max."""
        cand = list(self.victim_slots[i])
        rest = [ti for ti in self._group_known(self.comp[i]) if ti not in cand]
        rest.sort(key=lambda ti: (self._dist(i, self.targets[ti].pos), ti))
        return (cand + rest)[:self.s_max]

    def _group_known(self, g):
        """그룹 g 가 아는 미구조 · 미만료 생존자 (step 내 캐시)."""
        key = ("known", g)
        if key not in self._pair_cache:
            members = self.groups[g]
            self._pair_cache[key] = sorted(set().union(*[self._known_victims(j) for j in members]))
        return self._pair_cache[key]

    def _cells_dist(self, l, cells):
        """UAV l 에서 cells 의 가장 가까운 셀까지 BFS / d_max (도달 불가 1)."""
        arr = np.asarray(sorted(cells))
        fd = self.fields[l][arr[:, 0], arr[:, 1]]
        fd = fd[fd >= 0]
        return self._nd(int(fd.min()) if len(fd) else None)

    def centralized_state(self):
        """critic 중앙 state = BaseSimulation 5 채널 + 목표 지점 채널 (모든 UAV 의 g_l 위치 = 1)."""
        from env.core.knowledge import _pool
        base = super().centralized_state()
        goal = np.zeros((self.rows, self.cols, 1), np.float32)
        for l in range(self.n_agents):
            c, _ = self.goal_point(l)
            goal[c[0], c[1], 0] = 1.0
        return np.concatenate([base, _pool(goal, self.cfg["global_downsample"])], axis=-1)

    def rl_observation(self):
        """고정 shape 관측 (학습용). 키 / shape 는 env/role/gym_env.py observation space 와 같다. 그래프 사양은 env/role/graph_obs.py."""
        n, S, K, M, P = self.n_agents, self.s_max, self.k_max, self.n_sub_max, self.n_seg_slots
        t_max = float(self.cfg["max_episode_len"])
        n_t = max(1, len(self.targets))
        n_free = self.n_free_cells()
        nd_t = lambda d: 1.0 if d is None or d < 0 or d >= 10 ** 9 else min(1.0, d / t_max)
        active = {g: self._group_active(g) for g in range(len(self.groups))}
        agents_obs = []
        for i, a in enumerate(self.agents):
            g = self.comp[i]
            members = self.groups[g]
            mf = self.map_features(i)
            # ---- critic 입력 (role_self / role_group) ----
            self_f = np.zeros(ROLE_SELF_DIM, np.float32)
            self_f[:3] = [*self.frame.normalized(a.pos), a.capacity / max(1, a.capacity_max)]
            self_f[3 + a.mode.value] = 1.0
            self_f[6] = float(self.goal_valid[i])
            self_f[7] = min(1.0, (self.time - a.last_decision) / t_max)
            frac = [sum(1 for j in members if self.agents[j].mode is r) / len(members) for r in ROLES]
            slack = [(_deadline(self.targets[ti]) - self.time - self._dist(i, self.targets[ti].pos)) / t_max
                     for ti in self.victim_slots[i]]
            d_known = sum(1 for ti, t in enumerate(self.targets)
                          if t.discovered and (self.knowledge.known_free[i][t.pos] or ti in self.beliefs.rescued[i]))
            min_slack = float(np.clip(min(slack), 0, 1)) if slack else 1.0
            explored = float(self.knowledge.known_free[i].sum()) / n_free
            undisc = float(np.clip((len(self.targets) - d_known) / n_t, 0, 1))
            group_f = np.array([*frac, len(self.victim_slots[i]) / n_t, min_slack,
                                len(self.clusters_of(i)) / K, explored, undisc, min(1.0, self.time / t_max),
                                nd_t(mf["nearest_cluster_dist"]), self.size_feat(mf["largest_cluster_size"]),
                                nd_t(mf["depot_dist"])], np.float32)
            # ---- Role g 벡터 ----
            sl = [(_deadline(self.targets[ti]) - self.time - self._dist(i, self.targets[ti].pos)) / t_max
                  for ti in self.victim_slots[i]]
            g_res = np.array([min(1.0, self.n_alloc[i] / n_t), float(np.clip(min(sl), -1, 1)) if sl else 1.0], np.float32)
            g_exp = np.array([len(self.clusters_of(i)) / K, self.size_feat(self.knowledge.frontier[i].sum()),
                              min(1.0, self.group_discovered(g) / n_t)], np.float32)
            # ---- S 노드 / U–S / S–S ----
            s_list = self.s_nodes(i)
            held = self.held_by_others(i)
            s_ids = np.full(S, -1.0, np.float32)
            s_pos = np.zeros((S, 2), np.float32)
            s_tau = np.zeros(S, np.float32)
            s_held = np.zeros(S, np.float32)
            us_d = np.ones((n, S), np.float32)
            us_slack = np.full((n, S), -1.0, np.float32)
            ss_dist = np.ones((S, S), np.float32)
            for s_, ti in enumerate(s_list):
                t = self.targets[ti]
                tau = _deadline(t)
                s_ids[s_] = ti
                s_pos[s_] = self.frame.normalized(t.pos)
                s_tau[s_] = float(np.clip((tau - self.time) / t_max, -1, 1))
                s_held[s_] = float(ti in held)
                for l in members:
                    d = self._dist(l, t.pos)
                    us_d[l, s_] = self._nd(d)
                    if d < 10 ** 9:
                        us_slack[l, s_] = float(np.clip((tau - self.time - d) / t_max, -1, 1))
            if active[g] and len(s_list) > 1:
                ss_dist[:len(s_list), :len(s_list)] = self._pair_dist(g, [self.targets[ti].pos for ti in s_list])
            r_mask = self.rescue_mask(i).astype(np.float32)                 # 후보 = S 앞쪽 (victim_slots)
            # ---- K 노드 (영역) / U–K / K–K, 세그먼트 표 ----
            k_pos = np.zeros((K, 2), np.float32)
            k_size = np.zeros(K, np.float32)
            k_n = np.zeros(K, np.float32)
            e_size = np.zeros(K, np.float32)
            e_kcap = np.zeros(K, np.float32)
            uk_d = np.ones((n, K), np.float32)
            kk_dist = np.ones((K, K), np.float32)
            seg_pos = np.zeros((K, P, 2), np.float32)
            seg_sz = np.zeros((K, P), np.float32)
            seg_size = np.zeros((K, P), np.float32)
            seg_mask = np.zeros((K, P), np.float32)
            seg_dist = np.ones((K, P, M), np.float32)
            useg_d = np.ones((K, P, n), np.float32)
            useg_goal = np.zeros((K, P, n), np.float32)
            slots = self.clusters_of(i)
            for k, b in enumerate(slots):
                cells = b.cell_set()
                k_pos[k] = self.frame.normalized(tuple(np.asarray(sorted(cells), float).mean(0)))
                k_size[k] = self.size_feat(len(cells))
                k_n[k] = self.standing_on(g, b.id, exclude=i) / n
                e_size[k] = len(cells)
                cap = self.kcap(g, b)
                e_kcap[k] = cap
                if not active[g]:
                    continue
                split = self._blob_cells_for_split(g, b)
                for l in members:
                    uk_d[l, k] = self._cells_dist(l, split) if split else 1.0
                for kk in range(2, cap + 1):
                    off = seg_offset(kk)
                    E = self.seg_edges(g, b, kk)
                    for s_, (sc, med) in enumerate(self.segments(g, b, kk)):
                        seg_pos[k, off + s_] = self.frame.normalized(med)
                        seg_sz[k, off + s_] = self.size_feat(len(sc))
                        seg_size[k, off + s_] = len(sc)
                        seg_mask[k, off + s_] = 1.0
                        seg_dist[k, off + s_, :kk] = E[s_]
                        for l in members:
                            useg_d[k, off + s_, l] = self._cells_dist(l, sc)
                            gl = self.agents[l].goal
                            useg_goal[k, off + s_, l] = float(gl is not None and tuple(gl) in sc)
            if active[g] and len(slots) > 1:                      # 덩어리 간 가장 가까운 셀 BFS (kNN · 간선)
                kk_dist = self.blob_edges(g)
            agents_obs.append({
                "local": a.local_window[..., self.local_ch_idx].astype(np.float32),
                "role_self": self_f, "role_group": group_f,
                "role_g_res": g_res, "role_g_exp": g_exp,
                "role_mask": self.role_mask(i).astype(np.float32),
                "role_needed": np.array([float(self.kind[i] == "role")], np.float32),
                "s_ids": s_ids, "s_pos": s_pos, "s_tau": s_tau, "s_held": s_held, "rescue_mask": r_mask,
                "us_d": us_d, "us_slack": us_slack, "ss_dist": ss_dist,
                "k_pos": k_pos, "k_size": k_size, "k_n": k_n, "uk_d": uk_d, "kk_dist": kk_dist,
                "explore_mask": self.explore_mask(i).astype(np.float32),
                "explore_cur": np.array([float(self.slot_of(i))], np.float32),
                "explore_kcap": e_kcap, "explore_size": e_size,
                "explore_kind": np.array([float(self.sub_learned)], np.float32),
                "seg_pos": seg_pos, "seg_sz": seg_sz, "useg_d": useg_d, "useg_goal": useg_goal,
                "explore_seg_mask": seg_mask, "explore_seg_size": seg_size, "explore_seg_dist": seg_dist,
            })
        team = np.zeros((n, TEAM_CTX_DIM), np.float32)
        gpos = np.zeros((n, 2), np.float32)
        has_goal = np.zeros(n, np.float32)
        for j, b in enumerate(self.agents):
            team[j, b.mode.value] = 1.0
            if b.goal is not None:
                team[j, 3:5] = self.frame.normalized(b.goal)
            team[j, 5] = float(self.kind[j] is not None)
            team[j, 6:8] = self.frame.normalized(b.pos)
            c, hg = self.goal_point(j)
            gpos[j], has_goal[j] = self.frame.normalized(c), float(hg)
        group = np.array([[float(self.comp[i] == self.comp[j]) for j in range(n)] for i in range(n)], np.float32)
        out = {
            "agents": tuple(agents_obs),
            "team_ctx": team,
            "team_cap": np.array([b.capacity / max(1, b.capacity_max) for b in self.agents], np.float32),
            "team_comm": np.array([b.comm_range for b in self.agents], np.float32),
            "team_target": np.array([b.assigned_target if b.mode is Mode.RESCUE and b.assigned_target is not None else -1
                                     for b in self.agents], np.float32),
            "team_slot": np.array([self.slot_of(j) for j in range(n)], np.float32),
            "team_gpos": gpos, "team_has_goal": has_goal,
            "team_depot_d": np.array([self._nd(self._dist(j, self.depot)) for j in range(n)], np.float32),
            "depot_pos": np.asarray(self.frame.normalized(self.depot), np.float32),
            "geo": np.array([self.cols, self.rows, self.d_max], np.float32),
            "adj": self.knowledge.adjacency.astype(np.float32),
            "group": group,                                   # 같은 통신 그룹 (n-hop 연결 성분) — 순차 디코딩 마스크 범위
            "reach": self.knowledge.reach.astype(np.float32),
            "time": np.array([min(1.0, self.time / t_max)], np.float32),
            "state": self.centralized_state().astype(np.float32),
            **self.critic_observation(),
        }
        self._obs_cache = out                                 # 같은 상태에서 _decide 가 결정 순서 계산에 쓴다
        return out

    def critic_observation(self):
        out = super().critic_observation()
        team = out["critic_team"]                    # BaseSimulation 이 (N, CRITIC_TEAM_DIM=12) 로 만들고 앞 8 칸을 채움
        base = BaseSimulation.CRITIC_TEAM_DIM         # 8: 그 뒤에 배정 조난자 x, y / 배정 구간 medoid x, y
        for i, a in enumerate(self.agents):
            if a.assigned_target is not None:
                team[i, base:base + 2] = self.frame.normalized(self.targets[a.assigned_target].pos)
            if a.cluster_medoid is not None:
                team[i, base + 2:base + 4] = self.frame.normalized(a.cluster_medoid)
        out["critic_team"] = team
        return out

