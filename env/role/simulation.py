"""RoleSimulation — 계층 구조 env:  Role 선택 -> (Rescue 할당 | Explore 할당 2 단계 | Return) -> BFS 경로 추종.

맵 / 조난자 / 센싱·통신 병합 / 믿음 / 구조·재보충·마감 처리는 env/core/base_sim.py BaseSimulation 을 그대로 쓰고,
여기서는 결정 구조와 이동 · 보상 · 관측을 정한다.

이동    매 step 각 UAV 는 목표점까지 자기 (그룹 병합) 지도 BFS 최단경로의 다음 칸으로 1 칸. 알려진 장애물만 막고
        미탐색은 통과 가능으로 가정. 실제 장애물에 막히면 제자리. 목표 없으면 제자리. 같은 칸 허용 (info 에 횟수만).
역할    EXPLORE / RESCUE / RETURN (env.core.modes.Mode). 역할 마스크 (부분 귀환 포함):
            cap = 0                                       RETURN, EXPLORE
            0 < cap < cap_max, 부족분 >= return_min_deficit  RESCUE, EXPLORE, RETURN
            그 외 (부족분 < return_min_deficit, cap = cap_max)  RESCUE, EXPLORE
        RESCUE 는 할당 가능한 생존자가 있을 때만, EXPLORE 는 덩어리가 있을 때만. 전부 막히면 RETURN.
        RETURN = 하위 할당 없이 depot 으로, 도착하면 충전.
Explore 2 단계 (env/role/explore.py)
        1 단계: 덩어리 (frontier DBSCAN, 도달 가능 셀만) 선택 — 그룹 순차, 중복 허용
        2 단계: 최종 담당자 m 대면 kk = min(m, kcap) 구간 (측지 k-medoids). 이번 step 결정하지 않는 담당자는 직전 구간과 헝가리안
                연속 매칭 (그 구간은 마스크), 결정하는 UAV 는 (용량, index) 순으로 남은 구간 선택 (후보 1 개 이하면 정책 생략)
        구간 안 목표점 = 내 구간 frontier 중 BFS 가장 가까운 셀 (규칙). 구간은 매 step rebalance (정책 결정 아님), 비면 소진.

Role 재결정 원인 — 대상 UAV 만 역할부터 다시 하고 하위 할당까지 다시 한다 (role -> EXPLORE 면 1 단계 덩어리 (지금 덩어리도 후보)
-> 구간 분할 + 2 단계 / RESCUE 면 생존자 (자기 현재 목표도 후보) / RETURN 이면 depot). 대상이 아닌 UAV 는 역할 · 목표 유지.
    periodic  t % events.period == 0 (t = 0 포함)          -> alive UAV 전원 (RETURN 중 · 용량 0 포함). 이벤트와 독립 (시계 리셋 없음)
    "1" 생존자 증가                                        -> 그 그룹 전원 (역할 무관).
        그룹 멤버 누구든 자기 직전 belief 보다 아는 생존자가 늘었고 (탐색 발견 / 통신 병합 모두), 늘어난 생존자 중 미구조 · 미만료 ·
        그룹의 누군가가 τ - t 안에 닿음 · 그룹 RESCUE UAV 가 맡지 않음 인 생존자가 있을 때. 같은 step 여러 명이어도 한 번.
        로그 ev1_explore (그룹 합집합 기준으로도 새로움) / ev1_merge (통신 병합으로만 늘어남).
    "2" 생존자 구조 완료                                   -> 구조한 UAV 만
    "3" Return 완료 (depot 도착, 충전)                    -> 그 UAV 만
    "4" 목표 무효 -> 그 UAV 만. RESCUE: conflict (belief 병합으로 같은 생존자를 쫓는 더 가까운 그룹 UAV 를 알게 됨, 먼 쪽만) /
      다른 UAV 가 구조 / 만료 / 도달 불가.  EXPLORE: 내 덩어리에 frontier 없음 (소멸 포함) / 매 step rebalance 후 내 구간이 없는
      상태가 events.no_segment_steps step 연속 / exploration complete
    같은 step 에 겹쳐도 UAV 마다 재결정 1 번 (원인은 모두 기록). 결정 순서는 policies/decision_order.py (role -> 1 단계 ->
    2 단계, 한 대가 결정할 때마다 남은 생존자 / 덩어리 / 구간 기준으로 다시 계산, 학습 모델과 같은 함수).
    목표 유효 (goal_valid): "2" / "3" / "4" 면 꺼짐, periodic · "1" 은 유지 (관측에 기존 목표가 그대로 보임).
    같은 목표 재선택: RESCUE 유지 + 같은 생존자 / EXPLORE 유지 + 같은 덩어리 면 새 표본이 아니다 (rescue / explore valid 0).
    다른 목표 / 다른 역할이면 새 표본.
Explore goal latch: 목표 셀은 관측되어 frontier 가 아니게 됨 / 도달 불가 / 내 구간 밖에
    events.goal_latch_outside_steps step 연속일 때만 바꾼다 (매 step rebalance 로 구간 경계가 움직여도 유지).
통신 hysteresis (events.comm_hysteresis): 링크 생성 거리 <= R_c, 해제 거리 > R_c + h (그룹 / belief / 보상 / 특징 공통).

한 step 처리 순서 (_step_role)
    [직전 step 끝에 준비됨] 지도 갱신 -> 덩어리 생성 / 대응 -> 구간 rebalance -> 재결정 원인 수집 (다음 step t 기준, 주기 포함)
                          · 결정 대상 (kind) -> 재결정 UAV 가 맡은 생존자를 풀어 생존자 슬롯 다시 만듦
    1) 결정 순서대로: role -> RESCUE 생존자 / EXPLORE 1 단계
    2) 2 단계 (구간): 이번 step 1 단계를 한 UAV
    3) 이동 (BFS 1 칸) -> 관측 -> 구조 / 만료 -> 덩어리 · 구간 · 이벤트 갱신 -> 보상
보상    role_rewards() (env_config.rewards) — step 갱신 (이동 -> 센싱 -> 발견 -> 구조 -> 재보충 -> 만료) 이 끝난 상태로 계산.
        그룹 연결도 ω^g_{i,t} = (|U_{i,t}| - 1) / (|U| - 1), U_{i,t} = belief 를 병합하는 i 의 통신 그룹 (n-hop, comm hysteresis
        포함 = self.groups[self.comp[i]]), |U| = UAV 수 — UAV 노드 특징 omega_group (보상에는 쓰지 않음).
          role    r_role_{i,t} = (|ΔR_t| - w_x |ΔE_t|) / N_T + (w_d / N_T) Σ_{k ∈ ΔF_t} 1[τ_k - t > T_m]
                                 - (w_sw / N_T) 1[i 가 step t 에 자발적 역할 전환]
                  앞 두 항은 팀 공통 (끝날 때 미구조 = 마지막 step 의 E). 늦지 않은 발견 보상: 발견 시점에 골든타임이 T_m
                  (discover_margin) step 넘게 남은 생존자를 처음 발견하면 +w_d / N_T (ΔF_t = 이번 step 팀 전체 기준 첫 발견, 실제 값).
                  자발적 전환 = 재결정 원인이 periodic / "1" 뿐 ("2" / "3" / "4" 가 함께 있으면 아님) · 이전 역할이 이번
                  role_mask 에서 허용 · 새 역할 != 이전 역할 (역할이 같고 목표만 바뀐 경우 아님). t = 0 의 첫 역할 배정은 전환이 아님.
                  penalty 는 전환을 결정한 step 의 role 보상에 들어간다 (그 결정 구간의 return)
          task    r_task_{i,t} = r_role_specific_{i,t} - (w_ts / N_T) 1[자발적 목표 전환_{i,t}] (매 step, UAV 별, rewards.task — task_rewards)
                  모든 집합은 전역 (실제 환경, belief 아님), step t 의 센싱 · 구조 · 만료 처리가 끝난 상태: S = 전체 생존자 (N_T),
                  F_t = 팀 전체가 한 번이라도 직접 관측한 생존자 (Target.discovered), R_t = 구조, E_t = 만료, ΔE_t = 이번 step 만료.
                  자발적 목표 전환 = 재결정에서 같은 역할 (RESCUE -> RESCUE / EXPLORE -> EXPLORE) 을 골랐고 이전 목표가 아직 유효
                  (구조 · 만료 · 도달 불가 · 덩어리 소멸 · 이벤트 "4" 아님) 하며 원인에 "2" / "3" 이 없는데 다른 목표를 고름 (그 결정
                  step 1 회). 덩어리는 계열 (root_blob) 로 비교 — 이전 덩어리에서 쪼개져 나온 덩어리는 같은 목표.
                  EXPLORE 인 step: -(w_u / T_max) |S \ F_t| / (N_T - |E_t \ F_t|)   (분모 0 이면 0, 팀 공통 — UAV 수로 나누지 않음)
                    분자 = 아직 발견 안 된 생존자 (만료 여부 무관), 분모 = 미발견 상태로 만료된 생존자만 뺀 수
                  RESCUE 인 step: -(w_q / T_max) |F_t \ (R_t ∪ E_t)| / N_T - (w_x / N_T) |ΔE_t ∩ F_t| - w_dup 1[할당 충돌]
                    대기 (발견 · 미구조 · 미만료) 와 발견 후 만료는 팀 공통, 할당 충돌 = 같은 생존자를 더 가까운 UAV 가 이미 쫓는데 고름
                    (개인, 결정당 1 회, 범위 = 그룹 + 메모리 노드 기록, w_dup = rewards.task.w_dup / N_T)
                  RETURN 인 step: 0
                  info 에는 역할별로 나눠 싣는다: reward_rescue = RESCUE 인 step 의 r_task, reward_explore = EXPLORE 인 step 의 r_task
        학습 (training/role/ppo.py) 은 결정 단위 SMDP: role = 다음 role 결정까지 γ_r 할인 누적 + 그 가치로 bootstrap,
        rescue / explore = 같은 네트워크의 다음 결정과 그 역할을 벗어나는 step 중 먼저 오는 쪽까지 γ (최상위 gamma) 할인 누적 +
        V^res / V^exp bootstrap
마스크  그룹 belief 발견 수 = N_T 면 EXPLORE 금지 (그 순간 그 그룹 EXPLORE UAV 에 이벤트 "4": exploration complete)

구동    step(decider)         : decider.role / rescue / explore / explore_sub 를 env 가 결정 순서대로 호출 (규칙 · 랜덤, GUI)
        step_explicit(actions): 모델이 순차 디코딩한 dict(role, task) (N,) 를 받는다 (task = task (하위 할당): S 노드 생존자
                                0 .. s_max - 1 / F 노드 덩어리 s_max ..). 마스크 위반 / components 의 rule 구성요소 /
                                2 단계 거리 Hungarian 은 규칙 정책으로 대체. info <role|rescue|explore>_valid / _train = 결정 · 학습 표본.
메모리  memory.enabled: UAV 마다 다른 모든 UAV 의 마지막 기록 (위치, 용량, 역할, 목표 생존자 / 목표 지점, depot BFS, ω^g, t_last).
        reset 때 서로의 초기 상태, 매 step (그룹 갱신 뒤) 현재 그룹 멤버는 지금 상태로, 그룹 밖 UAV 는 멤버들의 기록 중 t_last 가
        가장 최근인 것을 그룹 전원이 공유 (중계). 관측의 메모리 노드 (mem_*, us_d / us_sigma / uk_d 의 그 행) 로만 쓰고 — GNN 입력
        전용 — 마스크 · 결정 순서 · ω^g · 이벤트에는 쓰지 않는다 (보상은 task 할당 충돌 판정에만). 위치 추정 없음 (마지막 위치 그대로).
로깅    이벤트 (step, 종류, 트리거 UAV, 재결정 대상), UAV 별 역할 전환 · 목표 변경, 에피소드 통계 (종류별 횟수, 역할 전환,
        목표 변경, 부분 귀환, 귀환 시작 용량, RESCUE 재결정 원인) -> info["event_stats"]
        (끝 step), events.log_dir 가 있으면 에피소드 로그 JSON.
"""
import json
import os

import numpy as np

from env.role.connectivity import pair_connectivity
from env.role.graph_obs import size_feat

from env.core.modes import Mode
from env.core.pathfinding import bfs_field, path_to, window_waypoint
from env.core.base_sim import BaseSimulation
from env.core.targets import deadline as _deadline
from env.core.step_context import StepContext
from env.core.agent import Robot
from env.role.explore import ExploreMixin, EXPLORE_DEFAULTS
from policies.role_rules import RuleDecider, rule_explore, rule_explore_sub, rule_rescue, rule_role

ROLES = (Mode.EXPLORE, Mode.RESCUE, Mode.RETURN)      # role 행동 인덱스 = Mode.value
N_ROLES = 3
EVENTS = ("periodic", "1", "2", "3", "4")             # Role 재결정 원인 (periodic = t % period == 0, 1 생존자 증가, 2 구조 완료,
                                                      #   3 Return 완료, 4 목표 무효). periodic / 1 = 여러 대 (규칙: 그룹 그리디),
                                                      #   2 / 3 / 4 = 한 대 (EDF)
EXPLORE_EVENTS = ("split", "merge")                   # 덩어리 쪼개짐 / 합쳐짐 (재결정 없음, 기록용)
INVALIDATING = {"2", "3", "4"}                        # 당사자 목표 무효 (Context goal_valid) — periodic · "1" 은 목표 유지
NEIGHBOR_BINS = (0.25, 0.5, 0.75, 1.0)                # 1-hop 이웃 거리 / R_c 구간 (마지막 = 1 초과, hysteresis)
CAP_STATES = ("zero", "partial", "full")
TASK_ITEMS = ("task_undiscovered", "task_wait", "task_found_expire", "task_conflict", "task_switch_rescue", "task_switch_explore")
RESCUE_ITEMS = ("task_wait", "task_found_expire", "task_conflict", "task_switch_rescue")   # RESCUE 인 step 의 task 보상 항목
EXPLORE_ITEMS = ("task_undiscovered", "task_switch_explore")                               # EXPLORE 인 step 의 task 보상 항목
                                                 # task 보상 항목 (EXPLORE 미발견 · RESCUE 대기 · 발견 후 만료 · 할당 충돌 · 목표 전환)
REWARD_ITEMS = ("role_team", "role_discover", "role_switch") + TASK_ITEMS   # self.rw / 로그 / 그래프 항목
VOLUNTARY_CAUSES = {"periodic", "1"}                  # 자발적 역할 전환이 될 수 있는 재결정 원인 (이것뿐일 때만)
DEC_INTERVAL_BINS = (1, 2, 5, 10, 20, 50)             # 결정 간격 분포 구간 (step, 마지막 = 50 초과)

ROLE_DEFAULTS = {
    "max_redecide": 1,         # 하위 할당 실패 시 RETURN 으로 대체하는 횟수 상한 (점검용)
    "return_min_deficit": 1,   # 부분 귀환 (용량이 남아도 RETURN) 허용 최소 부족분 (cap_max - cap)
}
EVENTS_DEFAULTS = {
    "goal_latch_outside_steps": 3,   # explore 목표 셀 유지: 관측 / 도달 불가 / 구간 밖 이만큼 연속일 때만 변경
    "no_segment_steps": 3,     # 구간 없는 상태가 이만큼 연속이면 이벤트 "4" (목표 무효)
    "comm_hysteresis": 2,      # 링크 해제 거리 = R_c + 이 값 (0 = hysteresis 없음)
    "log_events": True,        # 이벤트 / 목표 변경 기록 (info event_stats)
    "log_dir": None,           # 에피소드 로그 JSON 저장 폴더 (None = 저장 안 함)
    "period": 10,              # 주기 재결정: t % period == 0 (t = 0 포함) 이면 alive UAV 전원 (>= 1)
}
HEADS = ("role", "rescue", "explore")                 # env 내부 결정 종류 (task 헤드 = rescue + explore 1 단계)
# env_config["components"]: 구성요소별 모드 {role, rescue, explore: rule | learn} (eval_common/model_loader.py).
# rule 인 구성요소는 env 가 규칙 정책으로 결정, learn 은 모델이 결정 (step_explicit 의 행동)
COMPONENTS_DEFAULT = {"role": "learn", "rescue": "learn", "explore": "learn"}
MEMORY_DEFAULTS = {
    "enabled": True,           # 그룹 밖 UAV 의 마지막 기록을 메모리 노드로 (false = 현재 그룹만)
    "T_age": 50,               # 메모리 노드 age = min(1, (t - t_last) / T_age)
}
CLUSTER_DEFAULTS = {
    "eps": 1.5,                # DBSCAN 이웃 반경 (1.5 = 8-연결)
    "min_samples": 3,          # core 조건 (자기 포함 이웃 수)
    "k_max": 10,               # explore 덩어리 슬롯 수
}
REWARDS_DEFAULTS = {         # env_config.rewards
    "role": {"w_x": 2.0,               # 만료 1 명당 (/ N_T)
             "terminal_unrescued_as_expired": True,    # 끝날 때 미구조 생존자 = 마지막 step 의 만료
             "w_d": 0.2,                       # 늦지 않은 발견 1 명당 (/ N_T, 팀 공통)
             "discover_margin": 10,            # T_m: 발견 시점에 남은 골든타임 τ - t 가 이보다 클 때만 발견 보상 (step)
             "w_sw": 0.1},                     # 자발적 역할 전환 penalty (/ N_T, 전환한 UAV 만)
    "task": {"w_u": 1.0,               # EXPLORE 팀 미발견 패널티 (/ T_max)
             "w_q": 1.0,               # RESCUE 팀 대기 패널티 (/ T_max)
             "w_x": 2.0,               # RESCUE 발견 후 만료 (/ N_T)
             "w_dup": 0.5,             # 할당 충돌 (코드에서 / N_T, 결정당 1 회)
             "w_ts": 0.2},             # 자발적 목표 전환 penalty (/ N_T)
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
TASK_GLOBAL_DIM = 3     # task critic 전역 입력 critic_task_global = [미발견 비율, 대기 비율, t / T_max] (학습 전용)
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
        self.rewards_cfg = {h: _merge(REWARDS_DEFAULTS[h], {k: v for k, v in (given.get(h) or {}).items()
                                                            if k in REWARDS_DEFAULTS[h]})
                            for h in REWARDS_DEFAULTS}     # 없는 키 (예전 config 의 삭제된 항목) 는 무시
        self.period = int(self.events_cfg["period"])
        if self.period < 1:
            raise ValueError(f"events.period={self.period} — 1 이상")
        self.k_max = int(self.cluster_cfg["k_max"])
        self.n_sub_max = int(self.explore_cfg["n_sub_max"])
        options["events"] = self.events_cfg                       # BaseSimulation 이 comm_hysteresis 를 읽는다
        self.s_max = int(options.get("s_max") or 16)             # 그래프 S 노드 = rescue 행동 슬롯 수 (model.gnn.s_max 와 같은 값)
        self.set_components(options.get("components") or COMPONENTS_DEFAULT)
        options["components"] = self.components
        self.memory_cfg = _merge(MEMORY_DEFAULTS, options.get("memory"))
        self._ruled = {h: set() for h in HEADS + ("explore_sub",)}
        self._replaced = {h: set() for h in HEADS + ("explore_sub",)}
        self.episode_tag = None                # 로그 파일 이름 꼬리 (run_rules 가 seed 로 지정)
        super().__init__(options)

    def set_components(self, components):
        """구성요소 모드 설정 (env_config["components"] = {role, rescue, explore: rule | learn}, eval_common/model_loader.py).
        self.head_mode = 결정 종류별 모드 (rule 이 아니면 step_explicit 의 모델 행동을 쓴다)."""
        from eval_common.model_loader import check_modes
        c = check_modes(_merge(COMPONENTS_DEFAULT, components))
        if hasattr(self, "cfg"):
            self.cfg["components"] = c
        self.components = c
        self.head_mode = dict(c)

    @property
    def d_max(self):
        """거리 정규화 분모 = 최대 BFS 거리 (env_config.d_max, 없으면 rows + cols)."""
        return float(self.cfg.get("d_max") or (self.rows + self.cols))

    def size_feat(self, n):
        """셀 수 n 의 크기 특징 min(1, log(1 + n) / log(1 + explore.size_ref)) (env/role/graph_obs.py size_feat)."""
        return size_feat(n, float(self.explore_cfg["size_ref"]))

    def _nd(self, d):
        return 1.0 if d is None or d < 0 or d >= 10 ** 9 or not np.isfinite(d) else min(1.0, d / self.d_max)

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
        self._fpos_last = {}                                  # 마지막 관측의 F 노드 위치 통계 (rl_observation)
        first = "periodic"                                    # t = 0 = 주기 재결정
        self.last_event = [(0, [first]) for _ in self.agents]   # UAV 별 마지막으로 재결정을 부른 이벤트 (시각, 종류)
        self.last_events = [set() for _ in self.agents]
        self.step_log = []                                    # 이번 step 덩어리 대응 / Explore 이벤트 문장 (GUI 이벤트 로그)
        self.decided_role = [False] * self.n_agents
        self.events = [{first} for _ in self.agents]
        self.kind = [None] * self.n_agents
        self._known = [set() for _ in self.agents]
        self._log_reset()
        self._reward_reset()
        self._prepare_role()
        self._explore_after_update([(a.mode, None, None) for a in self.agents], [None] * self.n_agents)
        self._known = [self._known_victims(i) for i in range(self.n_agents)]
        self._set_decide_order()
        self._memory_reset()

    def _prepare(self):
        """다음 결정 준비 (BaseSimulation 훅). reset 도중 (role 상태 초기화 전) 호출되면 건너뛴다."""
        if hasattr(self, "events"):
            self._prepare_role()

    def _prepare_role(self):
        """위치 · 지식 · 시간이 바뀐 직후: BFS 필드, 통신 그룹, 할당 가능 생존자 슬롯."""
        n = self.n_agents
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
        self._build_victim_slots()

    def _build_victim_slots(self, deciding=()):
        """UAV 별 생존자 슬롯 = 그룹의 다른 UAV 가 맡지 않은 할당 가능 생존자 (그룹 지식, BFS 거리 -> 마감 -> id 순 상위 s_max).
        deciding = 이번 step 재결정하는 UAV — 그들이 지금 맡은 생존자는 막지 않는다 (재결정하지 않는 UAV 가 맡은 것만 막음).
        할당 정책 입력 (S 노드 앞쪽 후보) / 결정 순서가 모두 이 기준."""
        self.victim_slots, self.n_assignable = [], []
        self.slot_exclude = set(deciding)                       # 슬롯을 만들 때 풀어 준 UAV (평가 slot_snapshot 이 같은 기준으로 재현)
        for i in range(self.n_agents):
            held = self.held_by_others(i, exclude=deciding)
            ok = [ti for ti, t in enumerate(self.targets)
                  if t.discovered and self.knowledge.known_free[i][t.pos] and ti not in held and self._assignable(i, ti)]
            ok.sort(key=lambda ti: (self._dist(i, self.targets[ti].pos), _deadline(self.targets[ti]), ti))
            self.n_assignable.append(len(ok))
            self.victim_slots.append(ok[:self.s_max])

    def _explore_after_update(self, prev_state, prev_seg, prev_med=None):
        """지도 갱신 직후 explore 파이프라인 (env/role/explore.py). 반환: 구간 소진 / 도달 불가 UAV {i: 이유}."""
        self._seg_cache, self._geo_cache = {}, {}
        self._seed_state = None                                 # rebalance 는 지금 배정자 · goal 로 warm start
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
        elif a.capacity_max - a.capacity >= int(self.role_cfg["return_min_deficit"]):
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
        다시 계산한다 (학습 모델과 같은 함수 · 같은 관측 값). 이벤트 원인 기록 (이벤트 "1" 은 그룹 단위로 _collect_events 에서 기록).
        재결정 UAV 가 지금 맡은 생존자는 생존자 슬롯에서 막지 않도록 슬롯을 다시 만든다."""
        n = self.n_agents
        kind = [None] * n
        order = []
        st = self.ev_stats
        for g, members in enumerate(self.groups):
            for i in members:
                if self.events[i] and self.agents[i].alive:
                    kind[i] = "role"
                    order.append(i)
                    for e in sorted(self.events[i] - {"1"}):
                        self._log_event(e, i, [i])
                    for e in self.events[i]:
                        st["redecide_causes"][e] += 1
                    st["redecide_overlap"] += int(len(self.events[i]) > 1)
        self.kind = kind
        self.decide_order = order                               # D_0 (그룹 순, index 순 — 실제 결정 순서가 아님)
        self._deciding = {i for i in range(n) if kind[i] is not None}
        if self._deciding and any(self.agents[i].mode is Mode.RESCUE for i in self._deciding):
            self._build_victim_slots(self._deciding)
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
        # 이번 이벤트의 구간 분할 warm start 기준 = 결정 전 배정자 · goal (1 단계로 새로 고른 UAV 는 seed 에 넣지 않는다)
        self._seed_state = [(a.mode, a.blob, None if a.goal is None else tuple(a.goal)) for a in self.agents]
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
                "en": torch.round(st("k_n") * self.n_agents), "ecur": st("explore_cur")[..., 0],
                "dmax": torch.as_tensor(ob["geo"][2:3])}

    def _collect_events(self, prev_pos, rescued_by, exhausted):
        """다음 step (self.time) 의 재결정 원인. 한 대: "2" 구조 완료 / "3" Return 완료 / "4" 목표 무효 (그 UAV 만).
        그룹: "1" 생존자 증가 (그룹 멤버 누구든 자기 직전 belief 보다 아는 생존자가 늘었고 그중 미구조 · 미만료 · 그룹의 누군가가
        τ - t 안에 닿음 · 그룹 RESCUE UAV 가 맡지 않음 인 생존자가 있으면 그룹 전원). 주기: t % period == 0 이면 alive UAV 전원
        "periodic"."""
        n = self.n_agents
        ev = [set() for _ in range(n)]
        st = self.ev_stats
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            if i in rescued_by:
                ev[i].add("2")
            if a.mode is Mode.RETURN and a.pos == self.depot and prev_pos[i] != self.depot:
                ev[i].add("3")
            if a.mode is Mode.EXPLORE and i in exhausted:
                ev[i].add("4")
                st["ev4_causes"]["no_segment"] += 1
            if a.mode is Mode.RESCUE and "2" not in ev[i]:
                ti = a.assigned_target                     # None = 마감으로 배정이 풀림 (BaseSimulation._process_deadlines)
                if ti is None or self._target_invalid(i, ti):
                    ev[i].add("4")
                    st["ev4_causes"]["rescue_invalid"] += 1
            # exploration complete: 그룹 belief 발견 수가 (병합 포함) 처음 N_T 에 도달 -> 그 그룹 EXPLORE UAV 이벤트 "4"
            if not self._explore_seen_all[i] and self.exploration_complete(self.comp[i]):
                self._explore_seen_all[i] = True
                if a.mode is Mode.EXPLORE and "4" not in ev[i]:
                    ev[i].add("4")
                    st["ev4_causes"]["exploration_complete"] += 1
        known_now = [self._known_victims(i) for i in range(n)]
        for g, members in enumerate(self.groups):
            union_before = set().union(*[self._known[j] for j in members])
            grown = set().union(*[known_now[j] - self._known[j] for j in members])
            held = {self.agents[j].assigned_target for j in members if self.agents[j].mode is Mode.RESCUE}
            new = []
            for k in sorted(grown):
                tau = _deadline(self.targets[k])
                if k in held or not any(self._dist(j, self.targets[k].pos) <= tau - self.time for j in members):
                    continue
                new.append(k)
            if not new:
                continue
            explorers = [j for j in members if self.agents[j].alive and self.agents[j].mode is Mode.EXPLORE]
            targets = [j for j in members if self.agents[j].alive]
            for j in targets:
                ev[j].add("1")
            self._log_event("1", None, targets, group=g)
            st["ev1_explore" if any(k not in union_before for k in new) else "ev1_merge"] += 1
            for k in new:                                     # 선점 후보: EXPLORE 는 못 가는데 RESCUE 는 갈 수 있는 새 생존자
                tau = _deadline(self.targets[k])
                reach = lambda js: any(self._dist(j, self.targets[k].pos) <= tau - self.time for j in js)
                rescuers = [j for j in members if self.agents[j].mode is Mode.RESCUE]
                if not reach(explorers) and reach(rescuers):
                    st["preempt_candidates"] += 1
        if self.time % self.period == 0:
            for i, a in enumerate(self.agents):
                if a.alive:
                    ev[i].add("periodic")
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
        [1] role + rescue: role_front 로 다음 UAV -> 역할 -> RESCUE 면 바로 생존자 (그룹 1:1 마스크, 재결정하는 UAV 자신의
            현재 목표는 후보에 남음). 한 대마다 순서 재계산
        [2] explore 1 단계: EXPLORE 를 고른 UAV 전원을 explore_front 순서로 덩어리 선택 (지금 덩어리도 후보, 실제 선택으로 n_k
            갱신 — n_k = 재결정하지 않는 기존 배정자 + 이번 이벤트 앞 차례)
        [3][4] 분할 + 2 단계 (_stage2_all)  [5] task 보상 재료: 할당 충돌 (새 RESCUE 표본) · 자발적 목표 전환 (_task_switches)
        같은 목표 재선택: RESCUE 유지 + 같은 생존자 / EXPLORE 유지 + 같은 덩어리 면 새 표본이 아니다 (rescue / explore valid 0).
        반환: (role_valid, sub_valid {rescue, explore_s1 (새 표본), explore_s1_all (1 단계를 한 UAV 전원), explore_s2,
        explore_reselect}, n_redecide, n_violation)."""
        import torch
        from policies.decision_order import explore_front, role_front, role_sigma
        n = self.n_agents
        role_valid = [False] * n
        sub_valid = {"rescue": [False] * n, "explore_s1": [False] * n, "explore_s1_all": [False] * n,
                     "explore_s2": [False] * n, "explore_reselect": [False] * n}
        redecide = violations = 0
        s1, s1_new, taken = set(), set(), {}
        res_group = {}                                          # 그룹 -> 이번 step RESCUE 를 고른 UAV (유지 포함)
        self._prev_goal = {i: (self.agents[i].mode, self.agents[i].assigned_target, self.agents[i].blob)
                           for i in range(n)}               # 결정 전 역할 / 생존자 / 덩어리 (같은 목표 판정)
        self._prev_goal_valid = {i: self._goal_valid(i) for i in range(n)}   # 결정 전 목표가 아직 유효한가 (목표 전환 판정)
        D0 = [i for i in self.decide_order if self.agents[i].alive]
        if D0:
            X = self._order_inputs()
            d0 = torch.zeros(1, n, dtype=torch.bool)
            d0[0, D0] = True
            sig = role_sigma(X["ids"], X["slack"], X["valid"], X["group"], d0)
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
            # ---- [2] explore 1 단계: EXPLORE 를 고른 UAV 전원 (n_k = 재결정하지 않는 기존 배정자 (관측 k_n) + 앞 차례) ----
            rem = torch.zeros(1, n, dtype=torch.bool)
            rem[0, explorers] = True
            n_now = X["en"].clone()
            st = self.ev_stats
            for g in {self.comp[i] for i in explorers}:
                lead = next(i for i in explorers if self.comp[i] == g)
                st["s1_events"] += 1
                st["s1_participants"] += sum(1 for i in explorers if self.comp[i] == g)
                st["s1_empty_blobs"] += int(sum(1 for k in range(len(self.clusters_of(lead))) if n_now[0, lead, k] == 0))
            while bool(rem.any()):
                sel, claim = explore_front(X["group"], rem, X["emask"], X["ed"], X["esize"], n_now,
                                           float(self.explore_cfg["order_dist_alpha"]), X["dmax"])
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
                        pm, _, pb = self._prev_goal[i]
                        if pm is Mode.EXPLORE and pb is not None:              # 직전에도 EXPLORE: 같은 덩어리 재선택 비율
                            st["explore_redecide"] += 1
                            same = self.agents[i].blob == pb
                            st["explore_keep"] += int(same)
                            if not same:
                                s1_new.add(i)
                        else:
                            s1_new.add(i)
                        st["s1_claim_total"] += 1
                        st["s1_claim_kept"] += int(k == c)
                        for j in self.groups[self.comp[i]]:
                            n_now[0, j, k] += 1
                    rem[0, i] = False
                    self.decision_seq.append(("s1", int(i)))
        for i in s1:
            sub_valid["explore_s1_all"][i] = True
        for i in s1_new:
            sub_valid["explore_s1"][i] = True
        self._explore_keep_now = s1 - s1_new
        s2, reselect = self._stage2_all(s1, sub_fn)
        for i in s2:
            sub_valid["explore_s2"][i] = True
        sub_valid["explore_reselect"] = [i in reselect for i in range(n)]
        # [5] task 보상 재료 — 이벤트의 모든 결정이 끝난 뒤: 새 RESCUE 표본의 할당 충돌 (결정당 1 회), 자발적 목표 전환
        for i in range(n):
            if sub_valid["rescue"][i]:
                self._check_dup(i)
        self._task_switches(role_valid)
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
        if a.mode is Mode.RESCUE:                              # RESCUE UAV 재결정 원인
            for e in self.events[i]:
                self.ev_stats["res_redecide"][e] += 1
        self.ev_stats["dec_intervals"].append(self.time - a.last_dec_t)
        a.last_dec_t = self.time
        mask = self.role_mask(i, (), taken.get(g, ()))
        self.ev_stats["explore_masked_decisions"] += int(self._mask_flags[0])
        self.ev_stats["forced_return_all_masked"] += int(self._mask_flags[1])
        prev = a.mode
        ev1 = "1" in self.events[i]
        r = int(role_fn(i, mask))
        if not (0 <= r < N_ROLES and mask[r]):
            violations += 1
            self._replaced["role"].add(i)
            r = rule_role(self, i, mask)
        role_valid[i] = True
        self._log_role_choice(i, r)
        self._role_transition(i, prev, ROLES[r], prev_allowed=bool(mask[prev.value]))
        role = ROLES[r]
        if role is Mode.EXPLORE:
            if a.capacity == 0:
                self.ev_stats["cap0_explore_choice"] += 1
            a.mode, a.assigned_target, a.blob, a.cluster, a.cluster_medoid = Mode.EXPLORE, None, None, None, None   # 덩어리는 [2]
        prev_t = a.assigned_target if prev is Mode.RESCUE else None
        if role is Mode.RESCUE:
            rm = self.rescue_mask(i, taken.get(g, ()))
            s = int(rescue_fn(i, rm))
            if not (0 <= s < len(rm) and rm[s]):
                violations += 1
                self._replaced["rescue"].add(i)
                s = rule_rescue(self, i, rm)
            ti = self.victim_slots[i][s]
            taken.setdefault(g, set()).add(ti)
            keep = False
            if prev_t is not None:                                  # 직전에도 RESCUE: 같은 생존자 재선택 비율
                self.ev_stats["rescue_redecide"] += 1
                self.ev_stats["rescue_keep"] += int(ti == prev_t)
                keep = ti == prev_t
            self._assign_rescue(i, ti)
            if not keep:
                self._log_rescue_choice(i, ti)
            sub_valid["rescue"][i] = not keep
            res_group.setdefault(g, []).append(i)
        elif role is Mode.RETURN:
            self._assign_return(i)
        self._decided_now.add(i)
        self._log_claim(i, role, ev1)
        return role, violations

    def _log_rescue_choice(self, i, ti):
        """평가 기록 (결정에 영향 없음): 새 구조 목표 결정 수와 동시 추격 — 결정 시점에 다른 UAV 가 같은 생존자를 RESCUE 목표로
        쫓고 있음 (전역 기준, 결정당 1 회). 이번 이벤트에서 아직 차례가 안 온 재결정 UAV 의 목표는 풀린 것으로 보고 뺀다 (생존자 슬롯
        _build_victim_slots 와 같은 기준). 쫓는 UAV 중 같은 그룹이 하나라도 있으면 same (마스크상 0 이어야 함), 아니면 cross."""
        st = self.ev_stats
        st["rescue_new_decisions"] += 1
        holders = [j for j, b in enumerate(self.agents)
                   if j != i and b.mode is Mode.RESCUE and b.assigned_target == ti
                   and not (j in self._deciding and j not in self._decided_now)]
        if holders:
            st["concurrent_same" if any(self.comp[j] == self.comp[i] for j in holders) else "concurrent_cross"] += 1

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
                n = self.n_agents
                seg_d = torch.full((1, n, M), float("inf"))
                seg_sz = torch.zeros(1, n, M)
                for j in P:                                         # 구간까지 BFS / d_max, 구간 셀 수 (seg_front 차례)
                    for s_, (sc, _) in enumerate(segs):
                        seg_d[0, j, s_] = float(np.float32(self._cells_dist(j, sc)))
                        seg_sz[0, j, s_] = float(len(sc))
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
        a.goal = self._segment_goal(i, cells, a.goal) if cells else None
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
                         "goal_change": 0, "partial_return": 0,
                         "preempt_candidates": 0, "dec_intervals": [], "latch_hold": 0, "link_made": 0, "link_broken": 0,
                         "group_size": [], "nbr_bins": [0] * (len(NEIGHBOR_BINS) + 1),
                         "partial_deficits": [], "rescued_after_partial": [],
                         "role_by_cap": {c: [0, 0, 0] for c in CAP_STATES},
                         "res_redecide": {e: 0 for e in EVENTS}, "blob_dist_ratio": [],
                         "join_events": 0, "reselect_uavs": 0, "reselect_kept_goal": 0, "reselect_goal_change": 0,
                         "claim_holders": 0, "claim_not_rescue": 0, "ev1_events": 0, "ev1_redecide": 0, "ev1_rescue": 0,
                         "ev1_rescue_by_claim": 0, "known_to_rescue": [],
                         "s1_events": 0, "s1_empty_blobs": 0, "s1_claim_total": 0, "s1_claim_kept": 0,
                         "s1_participants": 0,
                         "n_groups": [], "ev1_rescue_nearest": 0, "survivor_truncated": 0, "fpos_nodes": 0, "fpos_fallback": 0,
                         "ev4_causes": {"no_segment": 0, "rescue_invalid": 0, "exploration_complete": 0},
                         "explore_masked_decisions": 0, "forced_return_all_masked": 0,
                         "redecide_causes": {e: 0 for e in EVENTS}, "redecide_overlap": 0, "ev1_explore": 0, "ev1_merge": 0,
                         "rescue_redecide": 0, "rescue_keep": 0, "explore_redecide": 0, "explore_keep": 0,
                         "cap0_explore_choice": 0,
                         "rescue_new_decisions": 0, "concurrent_same": 0, "concurrent_cross": 0}   # _log_rescue_choice
        self._partial_pending = [False] * n                 # 부분 귀환 중 (도착하면 그 뒤 구조 수 세기 시작)
        self._after_partial = [None] * n                    # 부분 귀환 충전 후 다음 충전까지 구조 수

    def _log_event(self, e, trigger, targets, group=None):
        self.ev_stats["events"][e] += 1
        if self.events_cfg["log_events"]:
            self.ev_log.append({"t": self.time, "event": e, "trigger": trigger, "group": group,
                                "targets": [int(j) for j in targets]})

    def _log_claim(self, i, role, ev1):
        """담당 후보 통계: 담당 후보가 RESCUE 를 고르지 않음 / 이벤트 "1" 재결정 UAV 수 · RESCUE 수 (생존자 대상은 _log_claim_targets)."""
        st = self.ev_stats
        if ev1 and self.cur_claim.get(i) is not None:
            st["claim_holders"] += 1
            st["claim_not_rescue"] += int(role is not Mode.RESCUE)
        if ev1:
            self._ev1_now = getattr(self, "_ev1_now", []) + [i]

    def _log_claim_targets(self):
        """이벤트 "1" 에서 RESCUE 를 고른 UAV 가 그 생존자의 담당 후보였나 (생존자가 정해진 뒤 — top_m 공동 할당 포함)."""
        st = self.ev_stats
        ev1 = getattr(self, "_ev1_now", [])
        if ev1:
            st["ev1_events"] += 1
            st["ev1_redecide"] += len(ev1)
            res = [i for i in ev1 if self.agents[i].mode is Mode.RESCUE and self.agents[i].assigned_target is not None]
            st["ev1_rescue"] += len(res)
            st["ev1_rescue_by_claim"] += sum(1 for i in res if self.cur_claim.get(i) == self.agents[i].assigned_target)
            for i in res:                                          # 그 생존자에 그룹에서 가장 가까운 UAV 였나 (동률 포함)
                pos = self.targets[self.agents[i].assigned_target].pos
                st["ev1_rescue_nearest"] += int(self._dist(i, pos) <= min(self._dist(j, pos) for j in self.groups[self.comp[i]]))
        self._ev1_now = []

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
            if "3" in e:                                          # 충전 완료 (Return 완료)
                if self._after_partial[i] is not None:
                    self.ev_stats["rescued_after_partial"].append(self._after_partial[i])
                    self._after_partial[i] = None
                if self._partial_pending[i]:
                    self._after_partial[i] = 0
                    self._partial_pending[i] = False

    def event_summary(self):
        """에피소드 통계 (info["event_stats"]): 종류별 횟수, 재결정 원인 (UAV 단위, 겹침), 이벤트 "1" 원인 (탐색 발견 / 통신 병합),
        같은 목표 재선택 비율 (rescue / explore keep rate), 재결정 · 역할 전환 · 목표 변경, 결정 간격 분포, 부분 귀환,
        RESCUE 재결정 원인, 용량 0 EXPLORE 선택 수, 고른 덩어리 거리 비 (고른 / 가장 가까운, 각 BFS + 1), EXPLORE step 당 새 관측 셀."""
        st = self.ev_stats
        after = st["rescued_after_partial"] + [c for c in self._after_partial if c is not None]
        nb = np.asarray(st["nbr_bins"], float)
        out = {"preempt_candidates": st["preempt_candidates"],
               "dec_interval": float(np.mean(st["dec_intervals"])) if st["dec_intervals"] else float("nan"),
               "latch_hold": st["latch_hold"], "link_made": st["link_made"], "link_broken": st["link_broken"],
               "group_size": float(np.mean(st["group_size"])) if st["group_size"] else float("nan"),
               "n_groups": float(np.mean(st["n_groups"])) if st["n_groups"] else float("nan"),
               "ev1_rescue_nearest": st["ev1_rescue_nearest"] / st["ev1_rescue"] if st["ev1_rescue"] else float("nan"),
               "survivor_truncated": st["survivor_truncated"],
               "fpos_nodes": st["fpos_nodes"], "fpos_fallback": st["fpos_fallback"],   # F 노드 수 / 셀 평균 fallback 수 (에피소드 합)
               "ev4_causes": dict(st["ev4_causes"]), "explore_masked_decisions": st["explore_masked_decisions"],
               "redecide_causes": dict(st["redecide_causes"]), "redecide_overlap": st["redecide_overlap"],
               "ev1_explore": st["ev1_explore"], "ev1_merge": st["ev1_merge"],
               "rescue_keep_rate": st["rescue_keep"] / st["rescue_redecide"] if st["rescue_redecide"] else float("nan"),
               "explore_keep_rate": st["explore_keep"] / st["explore_redecide"] if st["explore_redecide"] else float("nan"),
               "rescue_keep": st["rescue_keep"], "rescue_redecide": st["rescue_redecide"],
               "explore_keep": st["explore_keep"], "explore_redecide": st["explore_redecide"],
               "cap0_explore_choice": st["cap0_explore_choice"],
               "dec_interval_hist": np.bincount(np.searchsorted(DEC_INTERVAL_BINS, st["dec_intervals"], side="left"),
                                                minlength=len(DEC_INTERVAL_BINS) + 1).tolist(),
               "forced_return_all_masked": st["forced_return_all_masked"],
               "nbr_dist_hist": (nb / max(1.0, nb.sum())).tolist(),
               "events": dict(st["events"]), "redecide_uavs": st["redecide_uavs"], "role_switch": st["role_switch"],
               "goal_change": st["goal_change"],
               "partial_return": st["partial_return"], "partial_deficits": list(st["partial_deficits"]),
               "rescued_after_partial": after,
               "role_by_cap": {c: list(v) for c, v in st["role_by_cap"].items()},
               "rescue_redecide_causes": dict(st["res_redecide"]),
               "blob_dist_ratio": float(np.mean(st["blob_dist_ratio"])) if st["blob_dist_ratio"] else float("nan"),
               "join_events": st["join_events"], "reselect_uavs": st["reselect_uavs"],
               "reselect_kept_goal": st["reselect_kept_goal"] / st["reselect_uavs"] if st["reselect_uavs"] else float("nan"),
               "reselect_goal_change": st["reselect_goal_change"],
               "ev1_rescue_by_claim": st["ev1_rescue_by_claim"] / st["ev1_rescue"] if st["ev1_rescue"] else float("nan"),
               "ev1_rescue_other": 1 - st["ev1_rescue_by_claim"] / st["ev1_rescue"] if st["ev1_rescue"] else float("nan"),
               "s1_claim_kept": st["s1_claim_kept"] / st["s1_claim_total"] if st["s1_claim_total"] else float("nan"),
               "s1_empty_blobs": st["s1_empty_blobs"] / st["s1_events"] if st["s1_events"] else float("nan"),
               "s1_participants": st["s1_participants"] / st["s1_events"] if st["s1_events"] else float("nan"),
               "claim_not_rescue": st["claim_not_rescue"] / st["claim_holders"] if st["claim_holders"] else float("nan"),
               "ev1_redecide": st["ev1_redecide"] / st["ev1_events"] if st["ev1_events"] else float("nan"),
               "ev1_rescue": st["ev1_rescue"] / st["ev1_events"] if st["ev1_events"] else float("nan"),
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
        """actions = dict(role, task) 각 (N,) — task = task (하위 할당) 행동 (0 .. s_max - 1 = S 노드 생존자, s_max .. s_max + k_max - 1
        = F 노드 덩어리). 결정이 필요 없는 UAV 의 값은 무시. 모델이 순차 디코딩한 결과를 그대로 적용. RESCUE 인데 F 를 /
        EXPLORE 인데 S 를 가리키면 마스크 위반 (규칙 정책으로 대체). rule 구성요소 (components) / 2 단계 (거리 Hungarian) 는 규칙 정책.
        info <role|rescue|explore>_train = 결정을 했고 모델 출력 그대로 실행 (같은 목표 재선택은 결정 아님)."""
        n, S = self.n_agents, self.s_max
        role = np.asarray(actions.get("role", np.zeros(n))).reshape(-1)
        task = np.asarray(actions.get("task", np.zeros(n))).reshape(-1).astype(int)
        arr = {"role": role, "rescue": np.where(task < S, task, -1), "explore": np.where(task >= S, task - S, -1)}
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
        pm = self.head_mode
        done, info = self._step_role(pick("role", rule_role, pm["role"] != "rule"),
                                     pick("rescue", rule_rescue, pm["rescue"] != "rule"),
                                     pick("explore", rule_explore, pm["explore"] != "rule"),
                                     pick("explore_sub", rule_explore_sub, False))
        # 학습 마스크: 결정을 실제로 했고 그 행동이 모델 출력 그대로 최종 실행된 경우만 1 (explore = 1 단계만)
        for h in HEADS:
            ok = set(first[h]) - self._ruled[h] - self._replaced[h]
            info[f"{h}_train"] = [float(info[f"{h}_valid"][i] > 0 and i in ok) for i in range(self.n_agents)]
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
        mem_dec = self._memory_stats([i for i in range(n) if self.kind[i] is not None])   # 결정 시점 메모리 노드 (진단)

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
        self._memory_update()
        self.step_log = [f"t{self.time - 1:>3} blob " + s for s in self.blob_events]

        # ---- 보상 ----
        done = self._is_done()
        rew = self.role_rewards(rescued, expired, done, chosen_mode=chosen_mode, newly_found=newly_found)
        keep_e = getattr(self, "_explore_keep_now", set())
        keep_r = {i for i in self._decided_now if chosen_mode[i] is Mode.RESCUE and not sub_valid["rescue"][i]}

        # ---- info ----
        exp_valid = [bool(sub_valid["explore_s1"][i]) for i in range(n)]       # explore 결정 = 1 단계 (2 단계는 Hungarian)
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
            "explore_valid_s1": [float(v) for v in sub_valid["explore_s1_all"]],   # 1 단계를 한 UAV 전원 (같은 덩어리 유지 포함)
            "explore_reselect": [float(v) for v in sub_valid["explore_reselect"]],      # stage-2-only 재선택 (기존 배정자)
            "omega_mean": self.last_omega,                       # UAV 평균 그룹 연결도 ω^g (그래프용)
            "task_switch": [float(v) for v in self._ts_step],    # 이번 step 자발적 목표 전환 (task penalty)
            "mem_nodes_dec": mem_dec[0], "mem_age_dec": mem_dec[1],   # 결정 UAV 그래프의 메모리 노드 수 · 평균 age (없으면 nan)
            "rescue_keep": [float(i in keep_r) for i in range(n)],     # 재결정에서 같은 생존자 유지 (표본 아님)
            "explore_keep": [float(i in keep_e) for i in range(n)],    # 재결정에서 같은 덩어리 유지 (표본 아님)
            "in_rescue_seg": [float(v) for v in in_res],                 # 이번 step RESCUE (배정 생존자 있음)
            "role": [a.mode.value for a in self.agents],
            "events": ev_counts,
            "redecide": redecide, "mask_violations": violations,
            "fpos_nodes": int(getattr(self, "_fpos_last", {}).get("nodes", 0)),        # 이번 step 결정 관측의 F 노드 수 (UAV 합)
            "fpos_fallback": int(getattr(self, "_fpos_last", {}).get("fallback", 0)),  # 그중 도달 가능 셀이 없어 셀 평균을 쓴 수
            "rescue_conflict_cross": float(self._cross_group_conflict()),
            "group_size_mean": float(np.mean([1 + self.knowledge.reach[i].sum() for i in range(n)])),
            "cap0": sum(1 for a in self.agents if a.capacity == 0),
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
        self._sw_pending = [0.0] * n                  # 이번 step 자발적 역할 전환 (role 보상 -w_sw / N_T)
        self._ts_pending = [0.0] * n                  # 이번 step 자발적 목표 전환 (task 보상 -w_ts / N_T)
        self._ts_step = [0.0] * n                     # 직전 보상 step 의 목표 전환 (info task_switch)
        self._dup_pending = [0.0] * n                 # 이번 step 할당 충돌 (task 보상 -w_dup, 결정당 1 회)
        self._explore_seen_all = [False] * n          # 그룹 belief 발견 수가 N_T 에 도달한 것을 이미 겪었나 (이벤트 "4")
        self.last_discover = 0.0                      # 이번 step 늦지 않은 발견 보상 (점검용)
        self.rw = {k: np.zeros(n) for k in REWARD_ITEMS}
        self.rw_stats = {"sw_forced": 0, "sw_voluntary_periodic": 0, "sw_voluntary_ev1": 0,
                         "return_start_cap": [], "dup_in": 0, "dup_out": 0,
                         "discovered": 0, "discovered_timely": 0, "c_real": [],
                         "task_switch": 0, "found_expire": 0}

    def _role_transition(self, i, prev, new, prev_allowed):
        """역할 전환 기록 (t = 0 의 첫 역할 배정은 전환 아님). 자발적 = 재결정 원인이 VOLUNTARY_CAUSES (periodic / "1") 뿐 ·
        이전 역할이 이번 role_mask 에서 허용 — 이번 step role 보상에 -w_sw / N_T (_sw_pending), 원인별 (periodic / "1",
        둘 다면 "1") 수. 나머지는 강제 전환.
        새로 RETURN 을 시작하면 그 시점 남은 용량을 기록 (귀환 시작 용량 분포)."""
        if new is prev or self.time == 0:
            return
        st = self.rw_stats
        if new is Mode.RETURN:
            st["return_start_cap"].append(int(self.agents[i].capacity))
        ev = set(self.events[i])
        forced = not (ev and ev <= VOLUNTARY_CAUSES and prev_allowed)
        if forced:
            st["sw_forced"] += 1
        else:
            st["sw_voluntary_ev1" if "1" in ev else "sw_voluntary_periodic"] += 1
            self._sw_pending[i] = 1.0

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
        """그룹 g 가 모든 생존자를 발견했다고 아는가 (belief 발견 수 = N_T) — EXPLORE 마스크 / 이벤트 "4"."""
        return len(self.targets) > 0 and self.group_discovered(g) >= len(self.targets)

    def _check_dup(self, i):
        """할당 충돌: UAV i 가 고른 생존자를 더 가까운 UAV 가 이미 쫓고 있으면 -w_dup (결정당 1 회, _dup_pending).
        판정 범위 = i 의 그룹 (지금 목표 · 위치) + 메모리 노드 (그룹 밖 UAV 의 마지막 기록: RESCUE 목표 · 위치, 거리는 그룹 지도
        BFS) — 그룹 밖 실제 정보는 쓰지 않는다."""
        a = self.agents[i]
        if a.mode is not Mode.RESCUE or a.assigned_target is None:
            return
        ti = a.assigned_target
        pos = self.targets[ti].pos
        di = self._dist(i, pos)
        g = self.comp[i]
        for l in self.groups[g]:
            b = self.agents[l]
            if l != i and b.mode is Mode.RESCUE and b.assigned_target == ti and self._dist(l, pos) < di:
                self._dup_pending[i] = 1.0
                self.rw_stats["dup_in"] += 1
                return
        for k in self.memory_nodes(i):
            rec = self.memory[i][k]
            if rec["role"] == Mode.RESCUE.value and rec["target"] == ti:
                d = int(self._memory_field(g, rec["pos"])[pos])
                if 0 <= d < di:
                    self._dup_pending[i] = 1.0
                    self.rw_stats["dup_out"] += 1
                    return

    def _goal_valid(self, i):
        """지금 목표가 아직 유효한가 (결정 전, 목표 전환 판정): RESCUE = 생존자가 _target_invalid 가 아님 (구조 · 만료 · 도달
        불가 · 그룹 내 충돌 패배 아님), EXPLORE = 소속 덩어리가 그룹 덩어리 목록에 있음 (소멸 아님). 그 외 False."""
        a = self.agents[i]
        if not a.alive:
            return False
        if a.mode is Mode.RESCUE:
            return a.assigned_target is not None and not self._target_invalid(i, a.assigned_target)
        if a.mode is Mode.EXPLORE:
            return a.blob is not None and a.blob in self.blob_index[self.comp[i]]
        return False

    def _task_switches(self, role_valid):
        """자발적 목표 전환 (task 보상 -w_ts / N_T, 결정 step 1 회): 재결정에서 이전과 같은 역할 (RESCUE -> RESCUE /
        EXPLORE -> EXPLORE) 을 골랐고, 이전 목표가 유효하며 (_goal_valid, 결정 전), 재결정 원인에 "2" · "3" · "4" (목표 완료 ·
        무효) 가 없는데 다른 목표를 고름. 덩어리는 계열 (root_blob: 쪼개져 나온 덩어리 = 같은 목표) 로 비교."""
        for i in range(self.n_agents):
            if not role_valid[i] or not self._prev_goal_valid.get(i) or self.events[i] & INVALIDATING:
                continue
            pm, pt, pb = self._prev_goal[i]
            a = self.agents[i]
            if a.mode is not pm:
                continue
            if pm is Mode.RESCUE:
                changed = a.assigned_target is not None and a.assigned_target != pt
            elif pm is Mode.EXPLORE:
                changed = a.blob is not None and self.root_blob(a.blob) != self.root_blob(pb)
            else:
                changed = False
            if changed:
                self._ts_pending[i] = 1.0
                self.rw_stats["task_switch"] += 1

    def group_connectivity(self):
        """ω^g_{i,t} = (|U_{i,t}| - 1) / (|U| - 1) (N,) — U_{i,t} = i 의 통신 그룹 (belief 병합 범위, n-hop · hysteresis 포함),
        |U| = UAV 수 (1 대면 0). UAV 노드 특징 omega_group · 행동 곡선 공통."""
        n = self.n_agents
        if n <= 1:
            return [0.0] * n
        return [(len(self.groups[self.comp[i]]) - 1) / (n - 1) for i in range(n)]

    def role_rewards(self, rescued, expired, done=False, chosen_mode=None, newly_found=None):
        """네트워크별 보상 (N,) + 항목별 기록 (self.rw). step 갱신 (이동 -> 센싱 -> 발견 -> 구조 -> 재보충 -> 만료) 이 끝난 상태로 계산.
        role    (|ΔR_t| - w_x |ΔE_t|) / N_T + 늦지 않은 발견 보상 (팀 공통, discover_reward, SMDP)
                - (w_sw / N_T) 1[이번 step 자발적 역할 전환] (개인).
        rescue / explore  task_rewards (매 step, UAV 별) 을 그 step 역할로 나눈 것. chosen_mode = 이번 step 역할 (결정 후),
                newly_found = 이번 step 팀 전체 기준으로 처음 발견된 생존자 (role 발견 보상)."""
        rc, n, n_t = self.rewards_cfg, self.n_agents, max(1, len(self.targets))
        n_x = len(expired)
        if done and rc["role"]["terminal_unrescued_as_expired"]:
            n_x += sum(1 for t in self.targets if t.active)
        team = (len(rescued) - float(rc["role"]["w_x"]) * n_x) / n_t
        self.rw_stats["c_real"].append(self.real_connectivity())          # 진단 로그 (보상 아님)
        disc = self.discover_reward(newly_found or [])
        switch = [-float(rc["role"]["w_sw"]) / n_t * self._sw_pending[i] for i in range(n)]
        role = [team + disc + switch[i] for i in range(n)]
        om = self.group_connectivity()
        self.last_omega = float(np.mean(om)) if n else 0.0          # 팀 평균 그룹 연결도 ω^g (info omega_mean)
        terms = self.task_rewards(rescued, expired, chosen_mode or [a.mode for a in self.agents])
        task = [float(sum(terms[k][i] for k in TASK_ITEMS)) for i in range(n)]
        cm = chosen_mode or [a.mode for a in self.agents]
        self.rw["role_team"] += team
        self.rw["role_discover"] += disc
        self.rw["role_switch"] += np.asarray(switch)
        for k in TASK_ITEMS:
            self.rw[k] += terms[k]
        self._ts_step = list(self._ts_pending)
        self._sw_pending, self._ts_pending, self._dup_pending = [0.0] * n, [0.0] * n, [0.0] * n
        return {"role": role,                                        # task 보상은 그 step 역할 (결정 후) 로 나눔 (RETURN = 0)
                "rescue": [task[i] if cm[i] is Mode.RESCUE else 0.0 for i in range(n)],
                "explore": [task[i] if cm[i] is Mode.EXPLORE else 0.0 for i in range(n)]}

    def task_global(self):
        """전역 진행 상태 (실제 환경, step 갱신 후): (미발견 비율 |S \ F| / (N_T - |E \ F|) (분모 0 이면 0, 미발견 만료가 많으면
        1 을 넘을 수 있음 — 보상은 이 값 그대로, critic 입력 critic_task_global 은 [0, 1] 로 자름), 대기 비율 |F \ (R ∪ E)| / N_T,
        발견 후 미구조 만료 판정용 F 여부 목록). task 보상 · task critic 전역 입력 공통."""
        n_t = len(self.targets)
        found = [t.discovered for t in self.targets]
        undisc = sum(1 for f in found if not f)
        lost = sum(1 for t, f in zip(self.targets, found) if not f and t.expired_at is not None)
        wait = sum(1 for t, f in zip(self.targets, found) if f and t.active)
        den = n_t - lost
        return (undisc / den if den > 0 else 0.0), (wait / n_t if n_t else 0.0), found

    def task_rewards(self, rescued, expired, chosen_mode):
        """task 보상 항목별 (N,) — 모든 집합은 전역 (실제 환경), step 갱신 후 상태 (task_global).
        EXPLORE 인 step: -(w_u / T_max) |S \ F_t| / (N_T - |E_t \ F_t|)                     (팀 공통)
        RESCUE 인 step: -(w_q / T_max) |F_t \ (R_t ∪ E_t)| / N_T - (w_x / N_T) |ΔE_t ∩ F_t|   (팀 공통)
                        - w_dup 1[할당 충돌] (개인, w_dup = rewards.task.w_dup / N_T, _check_dup)
        RETURN 인 step: 0. 자발적 목표 전환 -(w_ts / N_T) 은 결정 step 1 회 (_task_switches, 개인) — 같은 역할 재선택이므로 그 step
        역할 (RESCUE / EXPLORE) 로 task_switch_rescue / task_switch_explore 에 나눠 기록."""
        rc = self.rewards_cfg["task"]
        n, n_t = self.n_agents, max(1, len(self.targets))
        t_max = float(self.cfg["max_episode_len"])
        alive = [a.alive for a in self.agents]
        res = np.asarray([alive[i] and chosen_mode[i] is Mode.RESCUE for i in range(n)], float)
        exp_ = np.asarray([alive[i] and chosen_mode[i] is Mode.EXPLORE for i in range(n)], float)
        undisc, wait, found = self.task_global()
        n_fx = sum(1 for k in expired if found[k])                            # 이번 step 만료 중 발견됐던 생존자
        self.rw_stats["found_expire"] += n_fx
        return {"task_undiscovered": -float(rc["w_u"]) / t_max * undisc * exp_,
                "task_wait": -float(rc["w_q"]) / t_max * wait * res,
                "task_found_expire": -float(rc["w_x"]) / n_t * n_fx * res,
                "task_conflict": -float(rc["w_dup"]) / n_t * np.asarray(self._dup_pending) * res,
                "task_switch_rescue": -float(rc["w_ts"]) / n_t * np.asarray(self._ts_pending) * res,
                "task_switch_explore": -float(rc["w_ts"]) / n_t * np.asarray(self._ts_pending) * exp_}

    # ---------------------------------------------------------------- 늦지 않은 발견 보상

    def discover_reward(self, newly_found):
        """늦지 않은 발견 보상 (팀 공통): (w_d / N_T) Σ_{k ∈ ΔF_t} 1[τ_k - t > T_m]. ΔF_t = 이번 step 팀 전체 기준으로 처음
        발견된 생존자 (환경 실제 값), τ_k = 골든타임 마감, t = step 갱신 후 시각, T_m = discover_margin. 이미 만료된 생존자
        (τ_k - t < 0) 는 자동 제외, 이후 구조 · 만료 시 회수 없음. rw_stats discovered / discovered_timely = 첫 발견 수 / 그중 보상 받은 수."""
        rc = self.rewards_cfg["role"]
        new = list(newly_found)
        timely = sum(1 for k in new if _deadline(self.targets[k]) - self.time > float(rc["discover_margin"]))
        self.rw_stats["discovered"] += len(new)
        self.rw_stats["discovered_timely"] += timely
        self.last_discover = float(rc["w_d"]) * timely / max(1, len(self.targets))
        return self.last_discover

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
        """에피소드 보상 항목 합 (UAV 평균) + 발견 보상 · task 통계 (충돌 / 목표 전환 / 만료 / 발견) · UAV 당 역할 전환 수
        (자발적 periodic / "1", 강제) · 귀환 시작 용량 분포 (0 .. c_max 횟수) (info["event_stats"]["rewards"])."""
        st = self.rw_stats
        mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
        n = max(1, self.n_agents)
        cmax = max(a.capacity_max for a in self.agents)
        return {"sum": {k: float(v.mean()) for k, v in self.rw.items()},
                "switch_forced": st["sw_forced"] / n,
                "switch_voluntary_periodic": st["sw_voluntary_periodic"] / n, "switch_voluntary_ev1": st["sw_voluntary_ev1"] / n,
                "return_start_cap_hist": np.bincount(st["return_start_cap"], minlength=cmax + 1)[:cmax + 1].tolist(),
                "dup_in": st["dup_in"], "dup_out": st["dup_out"], "task_switch": st["task_switch"] / n,
                "found_expire": st["found_expire"],
                "role_rescue_term": float(self.rw["role_team"].mean()),
                "role_discover_term": float(self.rw["role_discover"].mean()),
                "discovered": st["discovered"], "discovered_timely": st["discovered_timely"],
                "conn_real_mean": mean(st["c_real"])}

    # ================================================================ 메모리 노드

    def _record(self, k, omega):
        """UAV k 의 지금 상태 기록 (메모리 노드 재료): 위치, 용량, 역할, 목표 생존자 (RESCUE, 없으면 -1), 목표 지점, depot BFS
        (k 의 지도, 도달 불가 None), ω^g, t_last."""
        a = self.agents[k]
        d = self._dist(k, self.depot)
        return {"pos": tuple(a.pos), "cap": int(a.capacity), "role": int(a.mode.value),
                "target": int(a.assigned_target) if a.mode is Mode.RESCUE and a.assigned_target is not None else -1,
                "goal": tuple(self.goal_point(k)[0]), "depot_d": None if d >= 10 ** 9 else int(d),
                "omega": float(omega), "t": int(self.time)}

    def _memory_reset(self):
        """에피소드 시작: 모든 UAV 가 서로의 초기 상태 기록을 가진다."""
        om = self.group_connectivity()
        rec = [self._record(k, om[k]) for k in range(self.n_agents)]
        self.memory = [list(rec) for _ in range(self.n_agents)]

    def _memory_update(self):
        """매 step (그룹 · 상태 갱신 뒤): 현재 그룹 멤버의 기록은 지금 상태로, 그 밖의 UAV 기록은 그룹 멤버들의 기록 중
        t_last 가 가장 최근인 것을 그룹 전원이 공유 (belief 병합과 같은 범위 — 정보 중계). 위치 추정 없음."""
        om = self.group_connectivity()
        for members in self.groups:
            for k in range(self.n_agents):
                best = self._record(k, om[k]) if k in members else max((self.memory[m][k] for m in members),
                                                                        key=lambda r: r["t"])
                for m in members:
                    self.memory[m][k] = best

    def _memory_stats(self, deciders):
        """결정 UAV 들의 그래프 메모리 노드 수 평균, 메모리 노드 평균 age (결정자 없으면 nan, 메모리 노드 없으면 age nan)."""
        if not deciders:
            return float("nan"), float("nan")
        T_age = float(self.memory_cfg["T_age"])
        cnt, ages = [], []
        for i in deciders:
            mem = self.memory_nodes(i)
            cnt.append(len(mem))
            ages += [min(1.0, (self.time - self.memory[i][k]["t"]) / T_age) for k in mem]
        return float(np.mean(cnt)), float(np.mean(ages)) if ages else float("nan")

    def memory_nodes(self, i):
        """UAV i 그래프의 메모리 노드 = 기록이 있는 그룹 밖 UAV (memory.enabled false 면 없음). GNN 입력 전용 —
        마스크 · 결정 순서 · ω^g · 이벤트에는 쓰지 않는다 (보상은 task 할당 충돌 판정 _check_dup 에만)."""
        if not self.memory_cfg["enabled"] or not hasattr(self, "memory"):
            return []
        g = set(self.groups[self.comp[i]])
        return [k for k in range(self.n_agents) if k not in g]

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

    def _memory_field(self, g, pos):
        """그룹 g 지도에서 기록된 위치 pos 로부터의 BFS 장 (메모리 노드의 U–S / U–F 간선, step 내 캐시)."""
        key = ("mem_field", g, tuple(pos))
        if key not in self._pair_cache:
            self._pair_cache[key] = bfs_field(tuple(pos), self.knowledge.known_obstacle[self.groups[g][0]],
                                              self.rows, self.cols)[0]
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

    def eta(self, l, d_ls, d_ds):
        """UAV l 이 생존자 s 에 닿는 예상 시간 (step, v = 1 칸 / step — Robot.MOVES 8 방향 1 칸): 용량이 있으면 BFS(l, s),
        용량 0 이면 depot 경유 BFS(l, depot) + BFS(depot, s) (d_ds = 그룹 depot 필드). 도달 불가면 None."""
        if self.agents[l].capacity > 0:
            return None if d_ls >= 10 ** 9 else float(d_ls)
        d_ld = self._dist(l, self.depot)
        return None if d_ld >= 10 ** 9 or d_ds < 0 else float(d_ld + d_ds)

    def uav_beliefs(self):
        """UAV 노드의 belief 요약 (UAV 마다 자기 n-hop 병합 지식):
        team_rho  ρ = (N_T - |D_l|) / N_T  미발견 생존자 비율 (|D_l| = 발견 · l 의 지도에 있거나 구조됨을 앎)
        team_eta  η = (|C_free| - |E_l|) / |C_free|  미탐색 비율 (|E_l| = l 이 아는 free 셀)
        team_nu   ν = |N_l^s| / N_T  발견했지만 아직 구조되지 않은 생존자 비율 (_known_victims)
        team_time t / T_max  에피소드 경과 비율 (관측 생성 시점 self.time, T_max = max_episode_len, [0, 1] clip — 모든 UAV 같은 값)."""
        n_t = max(1, len(self.targets))
        n_free = float(self.n_free_cells())
        rho, eta, nu = (np.zeros(self.n_agents, np.float32) for _ in range(3))
        for l in range(self.n_agents):
            d_known = sum(1 for ti, t in enumerate(self.targets)
                          if t.discovered and (self.knowledge.known_free[l][t.pos] or ti in self.beliefs.rescued[l]))
            rho[l] = np.clip((len(self.targets) - d_known) / n_t, 0, 1)
            eta[l] = np.clip((n_free - float(self.knowledge.known_free[l].sum())) / n_free, 0, 1)
            nu[l] = min(1.0, len(self._known_victims(l)) / n_t)
        tau = float(np.clip(self.time / float(self.cfg["max_episode_len"]), 0.0, 1.0))
        return {"team_rho": rho, "team_eta": eta, "team_nu": nu, "team_time": np.full(self.n_agents, tau, np.float32)}

    def rl_observation(self):
        """고정 shape 관측 (학습용). 키 / shape 는 env/role/gym_env.py observation space 와 같다. 그래프 사양은 env/role/graph_obs.py."""
        n, S, K = self.n_agents, self.s_max, self.k_max
        t_max = float(self.cfg["max_episode_len"])
        n_t = max(1, len(self.targets))
        n_free = self.n_free_cells()
        nd_t = lambda d: 1.0 if d is None or d < 0 or d >= 10 ** 9 else min(1.0, d / t_max)
        active = {g: self._group_active(g) for g in range(len(self.groups))}
        fpos_all = {}
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
            self_f[7] = min(1.0, (self.time - a.last_decision) / self.period)
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
            # ---- 메모리 노드 (그룹 밖 UAV 의 마지막 기록, i 의 belief) ----
            mem = self.memory_nodes(i)
            m_valid, m_age, m_cap, m_role, m_depot, m_omega = (np.zeros(n, np.float32) for _ in range(6))
            m_pos, m_goal = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
            m_target = np.full(n, -1.0, np.float32)
            T_age = float(self.memory_cfg["T_age"])
            for k in mem:
                rec = self.memory[i][k]
                m_valid[k] = 1.0
                m_age[k] = min(1.0, (self.time - rec["t"]) / T_age)
                m_pos[k], m_goal[k] = self.frame.normalized(rec["pos"]), self.frame.normalized(rec["goal"])
                m_cap[k] = rec["cap"] / max(1, self.agents[k].capacity_max)
                m_role[k], m_target[k] = rec["role"], rec["target"]
                m_depot[k], m_omega[k] = self._nd(rec["depot_d"]), rec["omega"]
            mem_field = {k: self._memory_field(g, self.memory[i][k]["pos"]) for k in mem} if active[g] else {}
            # ---- S 노드 / U–S / S–S ----
            s_list = self.s_nodes(i)
            held = self.held_by_others(i)
            s_ids = np.full(S, -1.0, np.float32)
            s_pos = np.zeros((S, 2), np.float32)
            s_tau = np.zeros(S, np.float32)
            s_held = np.zeros(S, np.float32)
            s_elapsed = np.zeros(S, np.float32)
            us_d = np.ones((n, S), np.float32)
            us_slack = np.full((n, S), -1.0, np.float32)
            us_sigma = np.full((n, S), -1.0, np.float32)
            ss_dist = np.ones((S, S), np.float32)
            dfield = self.group_depot_field[g]
            for s_, ti in enumerate(s_list):
                t = self.targets[ti]
                tau = _deadline(t)
                s_ids[s_] = ti
                s_pos[s_] = self.frame.normalized(t.pos)
                s_tau[s_] = float(np.clip((tau - self.time) / t_max, -1, 1))
                s_held[s_] = float(ti in held)
                s_elapsed[s_] = float(min(1.0, self.time / tau)) if tau > 0 else 1.0
                d_ds = int(dfield[t.pos])                               # depot -> s BFS (그룹 지도, 도달 불가 -1)
                for l in members:
                    d = self._dist(l, t.pos)
                    us_d[l, s_] = self._nd(d)
                    if d < 10 ** 9:
                        us_slack[l, s_] = float(np.clip((tau - self.time - d) / t_max, -1, 1))
                    eta = self.eta(l, d, d_ds)
                    if eta is not None and tau > 0:
                        us_sigma[l, s_] = float(np.clip((tau - self.time - eta) / tau, -1, 1))
                for k, f in mem_field.items():                          # 메모리 노드: 기록된 위치 · 용량 기준
                    rec = self.memory[i][k]
                    d = int(f[t.pos])
                    us_d[k, s_] = self._nd(d)
                    if d < 0:
                        continue
                    eta = float(d) if rec["cap"] > 0 else (None if rec["depot_d"] is None or d_ds < 0
                                                           else float(rec["depot_d"] + d_ds))
                    if eta is not None and tau > 0:
                        us_sigma[k, s_] = float(np.clip((tau - self.time - eta) / tau, -1, 1))
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
            fpos = fpos_all.setdefault(i, [0, 0])                  # [F 노드 수, 도달 가능 셀 없음 -> 셀 평균 fallback 수]
            slots = self.clusters_of(i)
            for k, b in enumerate(slots):
                cells = b.cell_set()
                k_pos[k] = self.frame.normalized(tuple(np.asarray(sorted(cells), float).mean(0)))   # fallback: 셀 평균
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
                c_near = self._nearest_cell(i, split)                 # F 노드 위치 = i 에게 가장 가까운 셀 (U–F 와 같은 BFS · 셀)
                fpos[0] += 1
                if c_near is not None:
                    k_pos[k] = self.frame.normalized(c_near)
                else:
                    fpos[1] += 1
                if split:
                    arr = np.asarray(sorted(split))
                    for l, f in mem_field.items():
                        fd = f[arr[:, 0], arr[:, 1]]
                        fd = fd[fd >= 0]
                        uk_d[l, k] = self._nd(int(fd.min()) if len(fd) else None)
            if active[g] and len(slots) > 1:                      # 덩어리 간 가장 가까운 셀 BFS (kNN · 간선)
                kk_dist = self.blob_edges(g)
            sf_dist = np.ones((S, K), np.float32)                 # 생존자 -> 덩어리의 가장 가까운 셀 BFS (S–F 간선)
            if active[g] and s_list and slots:
                D = self.bfs_from(g, [tuple(self.targets[ti].pos) for ti in s_list])
                for k, b in enumerate(slots):
                    arr = np.asarray(sorted(b.cell_set()))
                    v = D[:, arr[:, 0], arr[:, 1]].min(1)
                    sf_dist[:len(s_list), k] = np.where(np.isfinite(v), np.minimum(1.0, v / self.d_max), 1.0)
            agents_obs.append({
                "local": a.local_window[..., self.local_ch_idx].astype(np.float32),
                "role_self": self_f, "role_group": group_f,
                "role_mask": self.role_mask(i).astype(np.float32),
                "role_needed": np.array([float(self.kind[i] == "role")], np.float32),
                "s_ids": s_ids, "s_pos": s_pos, "s_tau": s_tau, "s_held": s_held, "s_elapsed": s_elapsed,
                "rescue_mask": r_mask, "us_d": us_d, "us_slack": us_slack, "us_sigma": us_sigma, "ss_dist": ss_dist,
                "k_pos": k_pos, "k_size": k_size, "k_n": k_n, "uk_d": uk_d, "kk_dist": kk_dist,
                "explore_mask": self.explore_mask(i).astype(np.float32),
                "explore_cur": np.array([float(self.slot_of(i))], np.float32),
                "explore_kcap": e_kcap, "explore_size": e_size, "sf_dist": sf_dist,
                "mem_valid": m_valid, "mem_age": m_age, "mem_pos": m_pos, "mem_cap": m_cap, "mem_role": m_role,
                "mem_goal": m_goal, "mem_depot_d": m_depot, "mem_omega": m_omega, "mem_target": m_target,
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
            "team_omega": np.array(self.group_connectivity(), np.float32),   # ω^g (U 노드 omega_group, 보상과 같은 값)
            **self.uav_beliefs(),
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
        nodes, fb = sum(v[0] for v in fpos_all.values()), sum(v[1] for v in fpos_all.values())
        st = getattr(self, "ev_stats", None)
        if st is not None and getattr(self, "_fpos_last", {}).get("time") != self.time:   # 같은 상태의 관측은 한 번만 센다
            st["fpos_nodes"] += nodes
            st["fpos_fallback"] += fb
        self._fpos_last = {"time": self.time, "nodes": nodes, "fallback": fb}
        return out

    def _nearest_cell(self, i, cells):
        """F 노드 위치: cells 중 UAV i 의 BFS 장 (self.fields[i], U–F 거리와 같은 장) 으로 가장 가까운 도달 가능 셀
        (동률이면 셀 인덱스 (row, col) 가 작은 쪽). 도달 가능 셀이 없으면 None (호출자가 셀 평균으로 대체)."""
        if not cells:
            return None
        arr = np.asarray(sorted(cells))
        fd = self.fields[i][arr[:, 0], arr[:, 1]]
        ok = np.flatnonzero(fd >= 0)
        if not len(ok):
            return None
        return tuple(int(v) for v in arr[ok[np.argmin(fd[ok])]])

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
        undisc, wait, _ = self.task_global()                  # task critic 전역 입력 (학습 전용, actor 는 읽지 않음)
        g = np.array([undisc, wait, self.time / float(self.cfg["max_episode_len"])], np.float64)
        # 미발견 비율 |S \ F| / (N_T - |E \ F|) 는 미발견 만료가 많으면 1 을 넘는다 (보상은 원래 값) -> critic 입력만 [0, 1]
        out["critic_task_global"] = np.clip(np.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        return out

