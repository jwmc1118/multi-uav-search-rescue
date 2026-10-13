"""보상 함수 — 계층별 (할당 / primitive). 계수는 sim.cfg["reward"] (yaml env_config.reward).

두 함수는 각각 **길이 n_agents 의 float 리스트**를 반환한다.
RescueSimulation._execute() 가 매 tick 호출하고, info["reward_alloc"] / info["reward_primitive"] 로 나가서
training/multiagent_ppo.py 가 헤드별 GAE / advantage / critic 에 쓴다 (L = L_alloc + L_prim).

기본값 (설계 노트 기준, 자유롭게 수정):
  할당 (양(+) 보상은 구조할 때만. 패널티는 팀 공유 = 모든 UAV 동일, 중복 추적만 개인)
    r_alloc^i(t) = w_rescue * |R_t| / N_T
                 - (w_undisc / N_T) * Σ_{j∈U_t} min(1, t / max(1, τ_j - δ))
                 - (w_unres  / N_T) * Σ_{j∈F_t} min(1, t / max(1, τ_j))
                 - w_duplicate * 1[중복 추적: 더 가까운 UAV 도 같은 태스크를 쫓음 / 이미 구조된 태스크를 쫓음]  (먼 쪽만, 매 step)
      N_T = 에피소드 전체 조난자 수, t = 현재 step (시간 진행 후)
      R_t = 이번 step 에 구조된 조난자
      U_t = 아직 발견 안 된 조난자 (구조된 것만 제외, 만료 포함)
      F_t = 발견됐지만 구조 안 된 조난자 (만료 포함)
      τ_j = 골든타임 마감 시각 (end_time, 아직 없으면 golden_time), δ = discover_margin (발견 마감을 δ step 앞당김)
    -> 시간이 지날수록 못 찾은 / 못 구한 조난자 몫 패널티가 커지고, 마감이 지나면 1 로 포화되어 에피소드 끝까지 남는다.
       구조하면 그 사람이 집합에서 빠져 패널티가 즉시 사라진다.
  primitive (개인, 모드별)
    explore         : r_prim = w_new_cells * |E^i_t| / B_e             자기 센서로 처음 밝힌 셀 (Role-selection 논문 R_e 의 개인판)
    rescue / return : r_prim = w_progress * (d_before - d_after)       목적지까지 BFS 거리 감소
    공통            : - w_step - w_blocked * 1[막힌 이동] - w_collision * 1[UAV 간 충돌] (+ w_team * r_alloc_team, 기본 0)

인자:
  sim : RescueSimulation — sim.agents, sim.targets, sim.cfg, ...
  ctx : env.step_context.StepContext — 이번 tick 사건 전부
"""
from env.modes import Mode


def _coef(sim, key, default):
    return float(sim.cfg.get("reward", {}).get(key, default))


def _deadline(t):
    """조난자의 골든타임 마감 시각 τ_j. from_discovery 모드의 미발견 조난자는 end_time 이 아직 없어 golden_time."""
    return t.end_time if t.end_time is not None else t.golden_time


def _team_alloc(sim, ctx):
    """팀 공유 할당 보상: 구조 보상 - 미발견 / 미구조 조난자 시간가중 패널티."""
    n_targets = max(1, len(sim.targets))
    now = sim.time
    margin = _coef(sim, "discover_margin", 10)
    undisc = unres = 0.0
    for t in sim.targets:
        if t.rescued_at is not None:
            continue
        if not t.discovered:                                              # U_t
            undisc += min(1.0, now / max(1.0, _deadline(t) - margin))
        else:                                                             # F_t
            unres += min(1.0, now / max(1.0, _deadline(t)))
    return (_coef(sim, "w_rescue", 1.0) * len(ctx.rescued_now)
            - _coef(sim, "w_undisc", 0.005) * undisc
            - _coef(sim, "w_unres", 0.005) * unres) / n_targets


def alloc_reward(sim, ctx):
    n = ctx.n_agents
    team = _team_alloc(sim, ctx)
    out = [team] * n
    w_dup = _coef(sim, "w_duplicate", 0.001)
    for i, dup in enumerate(ctx.duplicate_pursuit):
        if dup:
            out[i] -= w_dup
    return out


def primitive_reward(sim, ctx):
    """모드별 primitive 보상 (개인).
        explore         : w_new · |E^i_t| / B_e          — 자기 센서로 처음 밝힌 셀 (팀이 몰랐던 셀), B_e = (2 r_s + 1)^2
        rescue / return : w_progress · (d_t - d_{t+1})   — 목적지 BFS 거리 감소
        공통            : - w_step - w_blocked · 1[막힌 이동] - w_collision · 1[UAV 간 충돌]
                          (+ w_team · 팀 할당 보상, 기본 0)
    """
    n = ctx.n_agents
    w_new = _coef(sim, "w_new_cells", 0.1)
    w_prog = _coef(sim, "w_progress", 0.05)
    w_step = _coef(sim, "w_step", 0.001)
    w_blk = _coef(sim, "w_blocked", 0.02)
    w_col = _coef(sim, "w_collision", 1.0)
    w_team = _coef(sim, "w_team", 0.0)
    team = _team_alloc(sim, ctx) if w_team else 0.0
    out = []
    for i in range(n):
        r = -w_step + w_team * team
        if ctx.chosen_mode[i] is Mode.EXPLORE:
            b_e = (2 * sim.agents[i].sensor_range + 1) ** 2
            r += w_new * ctx.newly_sensed[i] / b_e
        else:
            b, a = ctx.goal_dist_before[i], ctx.goal_dist_after[i]
            if b is not None and a is not None:
                r += w_prog * (b - a)
        if ctx.blocked_move[i]:
            r -= w_blk
        if ctx.collided[i]:
            r -= w_col
        out.append(r)
    return out


def compute_rewards(sim, ctx):
    """두 계층 보상을 한 번에. RescueSimulation._execute() 가 호출한다."""
    return {
        "alloc": [float(x) for x in alloc_reward(sim, ctx)],
        "primitive": [float(x) for x in primitive_reward(sim, ctx)],
    }
