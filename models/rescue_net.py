"""RescueNet — 통합 할당 정책 + primitive 정책 + critic 2개 (순수 torch, RLlib 비의존).

    [Allocation]  Encoder(태스크 k-NN CCN + explore + depot) / Context(자기 + n-hop 이웃) / Decoder(MHA glimpse
                  + pointer) — 순차 디코딩 ─▶ 슬롯 s^i ∈ {태스크 0..T-1, explore T, depot T+1}
    [Primitive]   CNN(local ⊕ waypoint 채널) ⊕ mode ⊕ waypoint ⊕ γ^i ─▶ a^i (8방향 + stay)
    [Critic]      헤드별 CentralCritic( CNN(State) ⊕ CNN(Local) ⊕ extra )   — 학습 전용

models/rllib_model.py 가 이 모듈을 RLlib TorchModelV2 로 감싼다. tests/smoke_test.py 는 RLlib 없이 이걸 직접 쓴다.
"""
import copy

import torch
import torch.nn as nn

from env.modes import N_MODES, N_MSG_TYPES
from models.alloc_policy import AllocationPolicy, SELF_DIM
from models.critic import CentralCritic
from models.primitive_policy import PrimitivePolicy, slot_mode_onehot, N_ACTIONS, WP_DIM

DEFAULT_MODEL_OPTIONS = {
    "embed_dim": 128,            # D
    "n_heads": 8,                # n_h  (D_h = D / n_h)
    "knn_k": 4,                  # k   (k_j = min(k, T^i - 1))
    "clip": 10.0,                # C   (pointer tanh clipping)
    "local_features": 128,       # primitive local CNN 출력
    "cnn_filters": [[16, 3, 2], [32, 3, 2], [64, 3, 1]],
    "primitive_hidden": [128, 64],
    "primitive_glimpse": "none",     # primitive 입력에 γ^i: 'none'(기본, 미사용) / 'detach' / 'grad'
    "critic_state_features": 128,
    "critic_local_features": 64,
    "critic_hidden": [128, 64],
    "activation": "relu",
}

AGENT_KEYS = ("local", "task_feats", "task_present", "task_ids", "explore_feats", "depot_feats",
              "self_feats", "node_mask", "node_xy", "node_type", "node_wp", "free_choice")
TEAM_KEYS = ("pub", "prev_msg", "reach", "dec_rank", "time", "state")


def stack_obs(obs):
    """RLlib 관측 dict (agents = tuple of per-agent dict) -> (B,N,...) 텐서 dict."""
    ag = obs["agents"]
    n = len(ag)
    out = {k: torch.stack([ag[i][k] for i in range(n)], dim=1).float() for k in AGENT_KEYS}
    for k in TEAM_KEYS:
        out[k] = obs[k].float()
    return out


class RescueNet(nn.Module):
    def __init__(self, n_agents, n_tasks, local_size, local_channels, state_channels, options=None):
        """local_channels: 관측 local 채널 수 (cfg["local_channels"] 개수, waypoint 제외)
        state_channels: critic 중앙집중 state 채널 수 (env.knowledge.N_CHANNELS)"""
        super().__init__()
        cfg = copy.deepcopy(DEFAULT_MODEL_OPTIONS)
        cfg.update(options or {})
        self.cfg = cfg
        self.N, self.T, self.A = n_agents, n_tasks, n_tasks + 2
        act = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}[cfg["activation"]]
        filt = tuple(tuple(f) for f in cfg["cnn_filters"])
        D = cfg["embed_dim"]

        # ---- actors (실행 시 사용) ----
        self.alloc = AllocationPolicy(n_tasks, N_MSG_TYPES, D, cfg["n_heads"], cfg["knn_k"], cfg["clip"])
        self.primitive = PrimitivePolicy(local_channels, local_size, cfg["local_features"], D, filt,
                                         tuple(cfg["primitive_hidden"]), act, cfg["primitive_glimpse"])
        # ---- critics (학습 전용, actor 와 인코더 비공유) ----
        kw = dict(state_channels=state_channels, local_channels=local_channels, local_hw=(local_size, local_size),
                  state_dim=cfg["critic_state_features"], local_dim=cfg["critic_local_features"],
                  filters=filt, hidden=tuple(cfg["critic_hidden"]), activation=act)
        self.alloc_critic = CentralCritic(extra_dim=SELF_DIM + 1, **kw)
        self.primitive_critic = CentralCritic(extra_dim=SELF_DIM + 1 + N_MODES + WP_DIM, **kw)

    def forward(self, ob, taken_alloc=None, greedy=False):
        """ob: stack_obs() 결과. taken_alloc (B,N): 학습 시 실제 할당 행동.
        반환 dict: alloc_logits (B,N,A), allocs (B,N), prim_logits (B,N,9), values {alloc, primitive} (B,N)."""
        B, N = ob["self_feats"].shape[:2]
        logits_a, allocs, gamma = self.alloc(ob, taken=taken_alloc, greedy=greedy)

        mode_oh = slot_mode_onehot(allocs, self.T)
        wp = torch.gather(ob["node_wp"], 2, allocs[..., None, None].expand(B, N, 1, WP_DIM)).squeeze(2)
        logits_p = self.primitive(ob["local"], mode_oh, wp, gamma)

        t = ob["time"][:, None, :].expand(B, N, 1)
        base = torch.cat([ob["self_feats"], t], dim=-1)
        values = {
            "alloc": self.alloc_critic(ob["state"], ob["local"], base),
            "primitive": self.primitive_critic(ob["state"], ob["local"],
                                               torch.cat([base, mode_oh, wp], dim=-1)),
        }
        return {"alloc_logits": logits_a, "allocs": allocs, "prim_logits": logits_p,
                "gamma": gamma, "values": values}
