"""할당 보상 / explore 노드 입력 단위 점검 (ray 불필요).

    python -m tests.smoke_reward

1) 아무도 못 찾은 상태에서 t 가 커질수록 팀 패널티가 단조 증가 (마감 뒤 포화)
2) 한 명 구조 -> 그 step 에 +w_rescue/N_T, 이후 그 사람 몫 패널티 사라짐
3) 팀 패널티는 모든 UAV 동일, duplicate 패널티는 해당 UAV 만
4) explore_feats shape (4,), g^i 가 발견 수에 따라 1 -> 0
5) from_start: 미발견 조난자도 마감(end_time)이 지나면 만료
"""
import numpy as np
import yaml

from env.rewards import _coef, _deadline, alloc_reward
from env.simulation import RescueSimulation
from env.step_context import RescueEvent, StepContext


def make_sim():
    with open("training/config/rescue.yaml") as f:
        cfg = yaml.safe_load(f)["env_config"]
    sim = RescueSimulation(cfg)
    for t in sim.targets:                      # 전부 미발견 상태로 시작
        t.reset()
        if sim.cfg["deadline_mode"] == "from_start":
            t.end_time = t.golden_time
    return sim


def ctx_for(sim, rescued=(), dup=None):
    ctx = StepContext(time=sim.time, n_agents=sim.n_agents)
    ctx.rescued_now = list(rescued)
    ctx.duplicate_pursuit = dup if dup is not None else [False] * sim.n_agents
    return ctx


def team(sim, **kw):
    return alloc_reward(sim, ctx_for(sim, **kw))[0]


def check_monotonic(sim):
    vals = []
    for t in range(0, sim.cfg["max_episode_len"] + 1):
        sim.time = t
        vals.append(team(sim))
    diffs = np.diff(vals)
    assert np.all(diffs <= 1e-12), "패널티가 시간에 따라 줄어들면 안 됨"
    assert vals[0] == 0.0 and vals[1] < 0.0, "t=0 은 0, 이후 음수"
    # 모든 마감 이후엔 포화: 미발견 N_T 명 전부 1 -> -w_undisc
    assert abs(vals[-1] + _coef(sim, "w_undisc", 0.005)) < 1e-9 or \
        max(_deadline(t) for t in sim.targets) - _coef(sim, "discover_margin", 10) > sim.cfg["max_episode_len"]
    print(f"[1] monotonic ok: r(0)={vals[0]:.5f}  r(100)={vals[100]:.5f}  r(end)={vals[-1]:.5f}")


def check_rescue(sim):
    n_t = len(sim.targets)
    w_rescue, w_unres = _coef(sim, "w_rescue", 1.0), _coef(sim, "w_unres", 0.005)
    sim.time = 50
    t0 = sim.targets[0]
    t0.discovered_at = 10
    if t0.end_time is None:
        t0.end_time = 10 + t0.golden_time
    before = team(sim)                                          # 0 번은 F_t 에 들어 있음
    share = w_unres / n_t * min(1.0, sim.time / max(1.0, _deadline(t0)))
    t0.rescued_at = sim.time
    ev = RescueEvent(target=0, agent=0, rescued_at=sim.time, discovered_at=10, time_taken=sim.time - 10)
    at_rescue = team(sim, rescued=[ev])
    assert abs(at_rescue - (before + share + w_rescue / n_t)) < 1e-9, (before, at_rescue, share)
    after = team(sim)                                           # 다음 step (구조 이벤트 없음, 시간 같음)
    assert abs(after - (before + share)) < 1e-9, "구조 후 그 사람 몫 패널티가 사라져야 함"
    print(f"[2] rescue ok: before={before:.5f}  rescue step={at_rescue:.5f} (+{w_rescue / n_t:.4f})  after={after:.5f}")


def check_team_and_duplicate(sim):
    sim.time = 120
    dup = [False] * sim.n_agents
    dup[1] = True
    out = alloc_reward(sim, ctx_for(sim, dup=dup))
    w_dup = _coef(sim, "w_duplicate", 0.001)
    others = [r for i, r in enumerate(out) if i != 1]
    assert all(abs(r - others[0]) < 1e-12 for r in others), "팀 패널티는 모든 UAV 동일"
    assert abs(out[1] - (others[0] - w_dup)) < 1e-12, "duplicate 는 해당 UAV 만"
    print(f"[3] team/duplicate ok: {[round(r, 5) for r in out]}")


def check_explore_g(sim):
    for t in sim.targets:
        t.reset()
    sim.beliefs.reset()
    n_t = len(sim.targets)
    gs = []
    for k in range(n_t + 1):
        feats = sim.rl_observation()["agents"][0]["explore_feats"]
        assert feats.shape == (4,) and feats.dtype == np.float32
        gs.append(float(feats[3]))
        if k < n_t:                         # 조난자 k 를 UAV 0 이 아는 '발견' 으로 만든다 (구조 믿음 경로)
            sim.targets[k].discovered_at = 0
            sim.beliefs.rescued[0].add(k)
    assert abs(gs[0] - 1.0) < 1e-6 and abs(gs[-1]) < 1e-6 and np.all(np.diff(gs) < 0), gs
    print(f"[4] explore_feats (4,) ok, g: {gs[0]:.2f} -> {gs[len(gs) // 2]:.2f} -> {gs[-1]:.2f} ({n_t} targets)")


def check_undiscovered_expire(sim):
    assert sim.cfg["deadline_mode"] == "from_start", "이 점검은 from_start 기준"
    t = min(sim.targets, key=lambda t: t.end_time)
    sim.time = t.end_time - 1
    sim._process_deadlines()
    assert t.expired_at is None and not t.discovered
    sim.time = t.end_time
    expired = sim._process_deadlines()
    assert t.expired_at == t.end_time and sim.targets.index(t) in expired, "미발견 조난자도 마감에 만료돼야 함"
    print(f"[5] undiscovered target expired at t={t.expired_at} (end_time)")


if __name__ == "__main__":
    check_monotonic(make_sim())
    check_rescue(make_sim())
    check_team_and_duplicate(make_sim())
    check_explore_g(make_sim())
    check_undiscovered_expire(make_sim())
    print("smoke_reward: all ok")
