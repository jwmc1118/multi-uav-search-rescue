"""빠른 파이프라인 점검 — 학습을 돌리기 전에 관측/forward/분포가 맞는지 확인.

    python -m tests.smoke_test              (Exploration_rescue/ 에서)
    python -m tests.smoke_test --trainer    # + 실제 MultiPPOTrainer 1 iteration

1) 환경 round-trip + observation_space 적합 (stand-in 정책 수십 step)
2) local 크롭 / 상대좌표 축 + 8방향 이동 / 코너 컷팅 / 360도 센서 차폐
3) BFS 거리 필드 = shortest_path_length, waypoint 가 local 창 안
4) RescueNet forward shape / 순차 디코딩: 같은 n-hop 그룹 중복 선택 0, 학습 재생 logit 일치,
   모델이 고른 슬롯이 환경에서 그대로 적용 (alloc_invalid 0)
5) (ray 있으면) action distribution sample / logp / entropy / kl + RLlib 모델 학습 모드 forward
"""
import argparse

import numpy as np
import torch

from env.gym_env import RescueMultiEnv
from env.agent import Robot
from env.knowledge import CH_OBSTACLE, N_CHANNELS, rel_vector
from env.pathfinding import bfs_field, shortest_path_length
from models.rescue_net import RescueNet, stack_obs

MAP = "maps/map_target/map_20260720_153619.json"


def to_batch(obs):
    """env obs(numpy, 단일) -> 모델 입력용 nested dict of tensors (batch=1)."""
    if isinstance(obs, dict):
        return {k: to_batch(v) for k, v in obs.items()}
    if isinstance(obs, tuple):
        return tuple(to_batch(v) for v in obs)
    return torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)


def check_env(env, steps=40):
    obs = env.reset()
    for _ in range(steps):
        done, info = env.sim.step()
        if done:
            env.sim.reset()
    obs = env.sim.rl_observation()
    assert env.observation_space.contains(obs), "observation_space 불일치"
    print(f"[env] n_agents={env.n_agents} T={env.max_targets} A={env.n_alloc} stand-in {steps} step ok, "
          f"explored={info['explored']:.2f}")


def check_local_frame(env):
    sim = env.sim
    a, kt = sim.agents[0], sim.knowledge
    saved = a.pos
    a.pos = (sim.rows // 2, sim.cols // 2)
    half = sim.cfg["state_size"] // 2
    kt.known_obstacle[0][:] = False
    north = (a.pos[0] - 2, a.pos[1])
    east = (a.pos[0], a.pos[1] + 3)
    kt.known_obstacle[0][north] = True
    kt.known_obstacle[0][east] = True
    kt._build_observation([])
    assert a.local_window[half - 2, half, CH_OBSTACLE] == 1.0, "북쪽 장애물이 크롭 위쪽에 없음"
    assert a.local_window[half, half + 3, CH_OBSTACLE] == 1.0, "동쪽 장애물이 크롭 오른쪽에 없음"
    assert rel_vector(a, north, 1.0) == (0.0, 2.0)               # +y = 위쪽
    assert rel_vector(a, east, 1.0) == (3.0, 0.0)                # +x = 오른쪽
    a.pos = saved
    sim.reset()
    print("[local] 크롭 = 월드 방위 그대로, rel_vector = 월드 (dx, dy) OK")


def check_motion_and_sensing():
    rows = cols = 7
    obst = [[False] * cols for _ in range(rows)]
    r = Robot((3, 3), sensor_range=3)
    for act, d in zip(Robot.ACTIONS, Robot.MOVES):             # 8방향 + stay
        r.pos = (3, 3)
        assert r.move(act, obst, rows, cols) and r.pos == (3 + d[0], 3 + d[1]), act
    r.pos = (3, 3)
    obst[2][3] = obst[3][4] = True
    assert not r.move("NE", obst, rows, cols) and "NE" not in r.valid_actions(obst, rows, cols)
    obst[3][4] = False
    assert r.move("NE", obst, rows, cols) and r.pos == (2, 4)
    obst = [[False] * cols for _ in range(rows)]
    obst[3][4] = True
    r.pos = (3, 3)
    vis = r.sense(obst, rows, cols)
    assert {(0, 3), (6, 3), (3, 0), (0, 0), (6, 6)} <= vis, "전방향이 다 보여야 함"
    assert (3, 4) in vis and (3, 5) not in vis and (3, 6) not in vis, "장애물 뒤는 가려져야 함"
    print("[motion] 8방향 이동 / 코너 컷팅 차단 / 360도 센서 + 차폐 OK")


def check_bfs(env):
    sim = env.sim
    blocked = np.array(sim.obstacles)
    start = sim.depot
    dist, _ = bfs_field(start, blocked, sim.rows, sim.cols)
    rng = np.random.RandomState(0)
    free = np.argwhere(~blocked)
    for k in rng.choice(len(free), 30, replace=False):
        g = tuple(free[k])
        d = shortest_path_length(start, g, blocked.tolist(), sim.rows, sim.cols)
        assert (d if d is not None else -1) == dist[g], (g, d, dist[g])
    for ns in sim.nodes:
        assert np.all(np.abs(ns.waypoint[:, :2]) <= 1.0 + 1e-6)
    print("[bfs] BFS 거리 필드 = 최단거리, waypoint 는 local 창 안 OK")


def check_model(env, steps=40):
    N, T, A = env.n_agents, env.max_targets, env.n_alloc
    net = RescueNet(N, T, env.sim.cfg["state_size"], len(env.sim.local_ch_idx), N_CHANNELS)
    print("[model] params: " + ", ".join(
        f"{n}={sum(p.numel() for p in m.parameters()) / 1e3:.0f}k" for n, m in net.named_children()))
    obs = env.reset()
    dup = invalid = 0
    max_err = 0.0
    for _ in range(steps):
        ob = stack_obs(to_batch(obs))
        with torch.no_grad():
            out = net(ob)
            out2 = net(ob, taken_alloc=out["allocs"])
        max_err = max(max_err, float((out["alloc_logits"] - out2["alloc_logits"]).abs().max()))
        al = out["allocs"][0].numpy()
        picks = [int(obs["agents"][i]["task_ids"][al[i]]) if al[i] < T else None for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if picks[i] is not None and picks[i] == picks[j] and obs["reach"][i, j] > 0:
                    dup += 1
        prim = torch.distributions.Categorical(logits=out["prim_logits"][0]).sample().numpy()
        obs, _, done, info = env.step({"alloc": al, "primitive": prim})
        invalid += info["alloc_invalid"]
        if done:
            obs = env.reset()
    assert tuple(out["alloc_logits"].shape) == (1, N, A) and tuple(out["prim_logits"].shape) == (1, N, 9)
    assert dup == 0 and invalid == 0 and max_err < 1e-5, (dup, invalid, max_err)
    print(f"[model] forward ok, 그룹 중복 0, alloc_invalid 0, 재생 logit 오차 {max_err:.1e}")
    return obs


def check_rllib(env, obs):
    try:
        from gym.spaces import Box
        from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
        from ray.rllib.models.torch.torch_action_dist import TorchCategorical
        from ray.rllib.policy.sample_batch import SampleBatch
        from models.rllib_model import RescueModel
        from models.primitive_policy import N_ACTIONS
        from training.action_dist import HierarchicalMultiActionDistribution
    except ImportError as e:
        print(f"[rllib] 건너뜀 (ray 없음: {e})")
        return
    N, A = env.n_agents, env.n_alloc
    prep = DictFlatteningPreprocessor(env.observation_space)
    obs_space = Box(-np.inf, np.inf, (prep.size,), np.float32)
    obs_space.original_space = env.observation_space
    num_outputs = N * A + N * N_ACTIONS
    model = RescueModel(obs_space, env.action_space, num_outputs, {"custom_model_config": {}}, "smoke")
    out, _ = model({"obs": to_batch(obs)}, [], None)
    assert tuple(out.shape) == (1, num_outputs + N)
    dist = HierarchicalMultiActionDistribution(
        out, model, child_distributions=[TorchCategorical] * (2 * N),
        input_lens=[A] * N + [N_ACTIONS] * N, action_space=env.action_space)
    sample = dist.sample()
    flat = torch.cat([torch.stack(list(sample[k]), -1).float() for k in ("alloc", "primitive")], -1)
    lp, ent, kl = dist.logp(flat), dist.entropy(), dist.kl(dist)
    out2, _ = model({"obs": to_batch(obs), SampleBatch.ACTIONS: flat}, [], None)
    assert torch.allclose(out[:, :N * A], out2[:, :N * A], atol=1e-5), "학습 모드 재생 logit 불일치"
    print(f"[rllib] model out {tuple(out.shape)}  logp {tuple(lp['alloc'].shape)}/{tuple(lp['primitive'].shape)}"
          f"  entropy ok  kl ok  학습 모드 재생 일치")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", action="store_true", help="실제 MultiPPOTrainer 1 iter 시도")
    args = ap.parse_args()

    env = RescueMultiEnv({"map_path": MAP, "max_targets": 16, "n_hop": 2})
    check_env(env)
    check_local_frame(env)
    check_motion_and_sensing()
    check_bfs(env)
    obs = check_model(env)
    check_rllib(env, obs)
    print("\nOK")

    if args.trainer:
        print("\n--- MultiPPOTrainer 1 iteration ---")
        import ray
        import yaml
        from training.catalog import register_all
        from training.multiagent_ppo import MultiPPOTrainer
        with open("training/config/rescue.yaml", "rb") as f:
            tcfg = yaml.safe_load(f)
        tcfg.update(num_workers=0, num_envs_per_worker=1, train_batch_size=400,
                    rollout_fragment_length=200, sgd_minibatch_size=200)
        ray.init(local_mode=True, ignore_reinit_error=True, num_gpus=0, include_dashboard=False)
        register_all()
        res = MultiPPOTrainer(config=tcfg).train()
        print("iter ok. episode_reward_mean =", res.get("episode_reward_mean"))


if __name__ == "__main__":
    main()
