"""(rule, rule, rule) 모드 학습 루프 rollout == rule_evaluation/run_rules.py (default) 결과 점검.

    python -m tests.role.test_role_rule_equiv [episodes]

학습 rollout 경로 (RolePPOTorchPolicy.compute_actions -> RoleEnv.step, policy_mode 전부 rule) 와
규칙 실행 경로 (RoleSimulation.step(RuleDecider()) — run_rules.py default) 를 같은 seed 로 돌려
매 step UAV 위치 / 역할 / 목표와 에피소드 결과(구조 · 만료 · 길이)가 완전히 같은지 확인한다.
"""
import sys

import numpy as np

from env.role.simulation import RoleSimulation
from policies.role_rules import RuleDecider
from tests.role.smoke_role_policy import base_config, make
from training.catalog import register_all


def trace_rules(cfg, seed):
    sim = RoleSimulation(cfg)
    sim.random_state = np.random.RandomState(seed)
    sim.reset()
    out, done = [], False
    while not done:
        done, info = sim.step(RuleDecider())
        out.append([(a.pos, a.mode.value, a.goal) for a in sim.agents])
    return out, (info["rescued_total"], info["expired_total"], sim.time)


def trace_rollout(env, prep, policy, seed):
    env.seed(seed)
    obs = env.reset()
    out, done = [], False
    while not done:
        actions, _, _ = policy.compute_actions(prep.transform(obs)[None], explore=True)
        obs, _, done, info = env.step({k: [int(np.asarray(a)[0]) for a in v] for k, v in actions.items()})
        out.append([(a.pos, a.mode.value, a.goal) for a in env.sim.agents])
        assert sum(info["role_train"]) + sum(info["rescue_train"]) + sum(info["explore_train"]) == 0
    return out, (info["rescued_total"], info["expired_total"], env.sim.time)


def main(episodes=5):
    register_all()
    cfg = base_config(mode=("rule", "rule", "rule"))
    env, prep, policy = make(cfg)
    for s in range(episodes):
        a, ra = trace_rules(cfg["env_config"], s)
        b, rb = trace_rollout(env, prep, policy, s)
        assert ra == rb and a == b, f"seed {s}: rules {ra} vs rollout {rb}"
        print(f"  seed {s}: identical {len(a)} steps, rescued/expired/len {ra}")
    print("test_role_rule_equiv: (rule,rule,rule) rollout == run_rules default")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
