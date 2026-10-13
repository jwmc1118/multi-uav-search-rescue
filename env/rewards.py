"""보상 함수 — 계층별 (할당 / primitive). 계수는 sim.cfg["reward"] (yaml env_config.reward).

두 함수는 각각 **길이 n_agents 의 float 리스트**를 반환한다.
RescueSimulation._execute() 가 매 tick 호출하고, info["reward_alloc"] / info["reward_primitive"] 로 나가서
training/multiagent_ppo.py 가 헤드별 GAE / advantage / critic 에 쓴다 (L = L_alloc + L_prim).

기본값 (설계 노트 기준, 자유롭게 수정):
  할당 (팀 공유 + 개인 패널티)
    r_alloc = ( w_rescue * #구조 - w_expire * #만료 ) / N_T          ψ-분해: 구조 1명 = +1/N_T
            + w_explore * (Φ_{t+1} - Φ_t)                              Φ = 팀 탐색 비율 (potential shaping)
            - w_duplicate * 1[중복 추적: 더 가까운 UAV 도 같은 태스크를 쫓음 / 이미 구조된 태스크를 쫓음]  (먼 쪽만, 매 step)
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


def _team_alloc(sim, ctx):
    n_targets = max(1, len(sim.targets))
    r = (_coef(sim, "w_rescue", 1.0) * len(ctx.rescued_now)
         - _coef(sim, "w_expire", 1.0) * len(ctx.expired_now)) / n_targets
    r += _coef(sim, "w_explore", 0.5) * (ctx.team_explored_after - ctx.team_explored_before)
    return r


def alloc_reward(sim, ctx):
    n = ctx.n_agents
    team = _team_alloc(sim, ctx)
    out = [team] * n
    w_dup = _coef(sim, "w_duplicate", 0.003)
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
    w_col = _coef(sim, "w_collision", 0.1)
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
