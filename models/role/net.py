"""RoleNet — 계층 구조 env (env/role/simulation.py) 용 Role / Rescue / Explore actor + 헤드별 critic (순수 torch).

독립된 네트워크 3 개 (encoder · GNN 파라미터 공유 없음, models/role/gnn.py HeteroGNN, 그래프 사양 env/role/graph_obs.py):
[역할 선택]     노드 U (그룹 전체) · S, 관계 S→U, U→S, U→U, S→S
                z_i = MLP([h_i^L ‖ h_i^0]) ∈ R^3 (h_i^0 = encoder 출력 skip) -> role_mask -> softmax
[생존자 할당]   노드 U (그룹의 RESCUE UAV — 이번 이벤트 앞 차례 포함) · S, 관계 S→U, U→S, U→U, S→S, U–S 간선 층마다 갱신
                CapAM decoder: query = W_c [h_i^L ‖ t / T], 후보 j 의 F_j = Linear([h_{s_j}^L ‖ e_{i,j}^L])
                마스크 = rescue_mask (할당 가능 후보) & 같은 그룹 앞 차례가 고른 생존자 제외
[frontier 할당] 노드 U (그룹의 EXPLORE UAV) · K (덩어리 슬롯), 관계 K→U, U→K, U→U, K→K (간선 갱신 없음)
                CapAM decoder: query = W_c [h_i^L ‖ t / T], F_j = Linear([h_{k_j}^L ‖ e_{i,j}]), 마스크 = explore_mask
                (같은 덩어리 중복 선택 허용). 1 단계만 — 같은 덩어리를 고른 UAV 들의 구간 배정은 env 의 κ Hungarian.
[Critic x3]     models/common/critic.py RichCentralCritic (헤드별 독립, actor 와 인코더 비공유) + ValueNorm x3
                extra_i = [role_self, role_group] (+ rescue: 고른 S 특징 / explore: 고른 K 특징)

순차 디코딩 — env _decide 와 같은 흐름 · 같은 차례 함수 (policies/decision_order.py, 한 대가 결정할 때마다 다시 계산):
  [1] role_front 차례 -> 역할 선택 네트워크 -> RESCUE 면 바로 생존자 할당 네트워크 (역할 반영 후 그래프를 따로 계산)
  [2] 신규 EXPLORE UAV (영역 없음) 를 explore_front 차례로 -> frontier 할당 네트워크. 영역 유지 UAV (explore_cur >= 0) 는
      영역 그대로, 차례 계산의 배정 수에 기존 배정자로 포함
  결정마다 역할 · 목표 (RESCUE = 생존자, RETURN = depot, EXPLORE 1 단계 = 고른 덩어리 중심) 를 팀 상태에 반영하고 다음 차례
  그래프를 다시 만든다. 학습 재생 (taken = SampleBatch.ACTIONS) 도 같은 루프 -> rollout 과 같은 logits.
모든 파라미터는 N (UAV 수) 과 무관.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from env.role.graph_obs import DIST_CLIP, EUK, EUS, EUU, EKK, ESS, FK, FS, FU
from policies.decision_order import explore_front, role_front, role_sigma
from models.capam import MASK_VALUE, CapAMDecoder
from models.role.gnn import AGGRS, HeteroGNN, mlp2
from models.common.critic import RichCentralCritic, CRITIC_TASK_DIM, CRITIC_GLOBAL_DIM
from models.common.value_norm import ValueNorm

ROLE_EXPLORE, ROLE_RESCUE, ROLE_RETURN = 0, 1, 2
N_ROLES = 3
ROLE_HEADS = ("role", "rescue", "explore")
RES_FEAT_DIM = 6          # critic 용 고른 S 특징: rel_pos(2), (τ - t) / T, 맡겨짐, BFS / d_max, slack
EXP_FEAT_DIM = 5          # critic 용 고른 K 특징: rel_pos(2), 크기, 배정 수 / N, BFS / d_max

GNN_DEFAULTS = {"embed_dim": 128, "n_layers": 2, "knn_k": 6, "s_max": 16, "aggr": "attention", "attn_heads": 4}
# head 별 기본값 (custom_model_config.role / rescue / explore 로 덮어씀)
HEAD_MODEL_DEFAULTS = {
    "n_heads": 8,                # CapAM decoder MHA 헤드 수 (rescue / explore)
    "clip": 10.0,                # pointer tanh clip
    "entropy_coeff": 0.01,       # 이 헤드 엔트로피 보너스 계수
    "value_norm": True,          # 이 헤드 critic ValueNorm
    "critic": {"cnn_filters": [[16, 3, 2], [32, 3, 2], [64, 3, 1]], "state_features": 128, "local_features": 64,
               "hidden": [128, 64], "set_dim": 64, "set_heads": 4},
}
# 없어진 옵션 — 주면 에러 (이전 구조 config / 체크포인트 판별)
REMOVED_HEAD_KEYS = ("role_encoder", "team_heads", "arch", "mlp_hidden", "embed_dim", "knn_k")
REMOVED_TOP_KEYS = ("share_context", "env_features")
LOSS_DEFAULTS = {"lambda_role": 1.0, "lambda_exp": 1.0, "lambda_res": 1.0}  # L = Σ λ_h L_h
DEFAULT_ROLE_MODEL_OPTIONS = {"activation": "relu"}

# 간선 BASE 방향 (a 행, b 열) 과 네트워크별 관계 (이름, src, dst, 간선 키)
BASE = {"UU": ("U", "U"), "US": ("U", "S"), "SS": ("S", "S"), "UK": ("U", "K"), "KK": ("K", "K")}
REL_US = (("S2U", "S", "U", "US"), ("U2S", "U", "S", "US"), ("U2U", "U", "U", "UU"), ("S2S", "S", "S", "SS"))
REL_UK = (("K2U", "K", "U", "UK"), ("U2K", "U", "K", "UK"), ("U2U", "U", "U", "UU"), ("K2K", "K", "K", "KK"))


def resolve_model_options(options):
    """custom_model_config -> {'activation', 'gnn': {...}, 'role': {...}, 'rescue': {...}, 'explore': {...},
    'entropy_coeff_heads': {...}, 'loss': {...}}."""
    options = dict(options or {})
    bad = [k for k in REMOVED_TOP_KEYS if k in options]
    for h in ROLE_HEADS:
        bad += [f"{h}.{k}" for k in REMOVED_HEAD_KEYS if k in (options.get(h) or {})]
    if bad:
        raise ValueError(f"model {bad} 는 제거된 옵션 (네트워크 3 개는 model.gnn 설정을 함께 쓴다) — 이전 구조의 "
                         f"config / 체크포인트입니다. 지금 구조와 호환되지 않으니 새로 학습하세요")
    out = copy.deepcopy(DEFAULT_ROLE_MODEL_OPTIONS)
    out.update({k: options[k] for k in DEFAULT_ROLE_MODEL_OPTIONS if k in options})
    out["gnn"] = dict(GNN_DEFAULTS, **(options.get("gnn") or {}))
    if out["gnn"]["aggr"] not in AGGRS:
        raise ValueError(f"gnn.aggr={out['gnn']['aggr']!r} — {' | '.join(AGGRS)}")
    for h in ROLE_HEADS:
        hc = copy.deepcopy(HEAD_MODEL_DEFAULTS)
        given = copy.deepcopy(options.get(h) or {})
        hc["critic"].update(given.pop("critic", {}) or {})
        hc.update(given)
        out[h] = hc
    out["entropy_coeff_heads"] = {h: float(out[h]["entropy_coeff"]) for h in ROLE_HEADS}   # loss 가 읽는다
    out["loss"] = dict(LOSS_DEFAULTS, **(options.get("loss") or {}))
    return out


AGENT_KEYS = ("local", "role_self", "role_group", "role_mask", "role_needed",
              "s_ids", "s_pos", "s_tau", "s_held", "s_elapsed", "rescue_mask", "us_d", "us_slack", "us_sigma", "ss_dist",
              "k_pos", "k_size", "k_n", "uk_d", "kk_dist", "explore_mask", "explore_cur", "explore_kcap", "explore_size")
TEAM_KEYS = ("team_ctx", "team_cap", "team_comm", "team_target", "team_slot", "team_gpos", "team_depot_d", "depot_pos",
             "team_rho", "team_eta", "team_nu", "geo", "group", "time", "state",
             "critic_tasks", "critic_task_present", "critic_team", "critic_global")


def stack_role_obs(obs):
    """RLlib / env 관측 dict (agents = 튜플) -> (B, N, ...) 텐서 dict."""
    ag = obs["agents"]
    n = len(ag)
    out = {k: torch.stack([ag[i][k] for i in range(n)], dim=1).float() for k in AGENT_KEYS}
    for k in TEAM_KEYS:
        out[k] = obs[k].float()
    return out


def _masked(logits, mask):
    """마스크 적용. 전부 막힌 행은 0 번만 열어 (결정 없는 UAV 의 dummy 행동) NaN 을 막는다 — loss 에서 제외된다."""
    m = mask > 0.5
    m = m | (~m.any(dim=-1, keepdim=True) & (torch.arange(m.shape[-1], device=m.device) == 0))
    return logits.masked_fill(~m, MASK_VALUE)


def knn_adj(d, mask, k):
    """d (G,n,n) 거리, mask (G,n) -> adj (G,n,n) bool: 행 v 가 받는 이웃 = 유효 노드 (자기 제외) 중 거리 <= k 번째로 가까운
    이웃의 거리 — k 번째와 동률인 이웃은 모두 포함 (이웃 수가 k 를 넘을 수 있음, 노드 순서와 무관)."""
    G, n, _ = d.shape
    kk = min(int(k), n - 1)
    if kk <= 0:
        return torch.zeros(G, n, n, dtype=torch.bool, device=d.device)
    eye = torch.eye(n, dtype=torch.bool, device=d.device)[None]
    dd = d.masked_fill(eye | ~mask[:, None, :], 1e9)
    kth = torch.topk(dd, kk, dim=-1, largest=False).values[..., -1:]     # k 번째 거리 (유효 이웃이 k 개 미만이면 1e9)
    return (dd <= kth) & (dd < 1e8) & mask[:, :, None]


def own(x):
    """(B,N,N,...) 의 [b, i, i, ...] (UAV i 관측의 자기 행) -> (B,N,...)."""
    return torch.diagonal(x, dim1=1, dim2=2).movedim(-1, 1)


class RoleNetwork(nn.Module):
    """역할 선택: HeteroGNN (U, S) + MLP 분류 z_i = MLP([h_i^L ‖ h_i^0])."""

    def __init__(self, D, g):
        super().__init__()
        self.gnn = HeteroGNN({"U": FU, "S": FS}, {"UU": EUU, "US": EUS, "SS": ESS}, BASE, REL_US, D, g["n_layers"],
                             g["aggr"], g["attn_heads"])
        self.dec = mlp2(2 * D, D, N_ROLES)

    def forward(self, gr):
        h, h0, _ = self.gnn(gr["x"], gr["mask"], gr["edge"], gr["adj"])
        r = torch.arange(gr["ii"].shape[0], device=gr["ii"].device)
        return self.dec(torch.cat([h["U"][r, gr["ii"]], h0["U"][r, gr["ii"]]], -1)), h


class AssignNetwork(nn.Module):
    """생존자 / frontier 할당: HeteroGNN (U, 후보 종류 C) + CapAM decoder (query = [h_i^L ‖ t], F_j = Linear([h_j ‖ e_ij]))."""

    def __init__(self, D, g, cand, node_dims, edge_dims, relations, edge_key, n_heads, clip, edge_update=None):
        super().__init__()
        self.cand, self.edge_key = cand, edge_key
        self.gnn = HeteroGNN(node_dims, edge_dims, BASE, relations, D, g["n_layers"], g["aggr"], g["attn_heads"],
                             edge_update=edge_update)
        self.proj = nn.Linear(2 * D, D)
        self.dec = CapAMDecoder(D, D + 1, n_heads, clip)            # w_q 가 W_c [h_i ‖ t / T]

    def forward(self, gr):
        h, _, e = self.gnn(gr["x"], gr["mask"], gr["edge"], gr["adj"])
        r = torch.arange(gr["ii"].shape[0], device=gr["ii"].device)
        q = torch.cat([h["U"][r, gr["ii"]], gr["t"]], -1)
        F = self.proj(torch.cat([h[self.cand], e[self.edge_key][r, gr["ii"]]], -1))
        return self.dec(q, F, gr["mask"][self.cand]), h


class RoleNet(nn.Module):
    def __init__(self, n_agents, s_max, k_max, local_size, local_channels, state_channels, dims, options=None):
        """dims: {role_self, role_group, critic_task, critic_team, critic_global}."""
        super().__init__()
        cfg = resolve_model_options(options)
        self.cfg = cfg
        self.N, self.T, self.K = n_agents, s_max, k_max
        if int(cfg["gnn"]["s_max"]) != s_max:
            raise RuntimeError(f"gnn.s_max={cfg['gnn']['s_max']} 와 관측 S 노드 수 {s_max} 가 다름 — env_config.s_max 를 맞추세요")
        act = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}[cfg["activation"]]
        g = cfg["gnn"]
        D = int(g["embed_dim"])
        self.D, self.knn_k = D, int(g["knn_k"])
        hr, hs, he = cfg["role"], cfg["rescue"], cfg["explore"]
        self.role_net = RoleNetwork(D, g)
        self.rescue_net = AssignNetwork(D, g, "S", {"U": FU, "S": FS}, {"UU": EUU, "US": EUS, "SS": ESS}, REL_US, "US",
                                        hs["n_heads"], hs["clip"], edge_update="US")
        self.explore_net = AssignNetwork(D, g, "K", {"U": FU, "K": FK}, {"UU": EUU, "UK": EUK, "KK": EKK}, REL_UK, "UK",
                                         he["n_heads"], he["clip"])
        role_ctx = dims["role_self"] + dims["role_group"]

        def critic(hc, extra):
            c = hc["critic"]
            return RichCentralCritic(extra_dim=extra, state_channels=state_channels, local_channels=local_channels,
                                     local_hw=(local_size, local_size), state_dim=c["state_features"],
                                     local_dim=c["local_features"], filters=tuple(tuple(f) for f in c["cnn_filters"]),
                                     hidden=tuple(c["hidden"]), activation=act, set_dim=c["set_dim"],
                                     set_heads=c["set_heads"], task_dim=dims.get("critic_task", CRITIC_TASK_DIM),
                                     team_dim=dims["critic_team"], global_dim=dims.get("critic_global", CRITIC_GLOBAL_DIM))
        self.critic_role = critic(hr, role_ctx)
        self.critic_rescue = critic(hs, role_ctx + RES_FEAT_DIM)
        self.critic_explore = critic(he, role_ctx + EXP_FEAT_DIM)
        self.debug_trace = None              # 리스트를 넣으면 순차 디코딩 차례별 그래프 / 임베딩을 기록 (테스트용)
        self.timing = None                   # dict 를 넣으면 GNN 호출 수 / 결정 수 / 시간을 더한다 (G 로그)
        for h in ROLE_HEADS:                 # 헤드별 ValueNorm (그 헤드 value_norm: true 일 때만)
            if cfg[h]["value_norm"]:
                setattr(self, f"value_norm_{h}", ValueNorm())

    def head_networks(self):
        """{헤드: 그 헤드의 actor 네트워크} — 세 네트워크는 파라미터를 공유하지 않는다."""
        return {"role": self.role_net, "rescue": self.rescue_net, "explore": self.explore_net}

    def encoder_parameters(self):
        """{헤드: 그 네트워크의 encoder + GNN 파라미터} (gradient norm 측정용)."""
        return {h: list(n.gnn.parameters()) for h, n in self.head_networks().items()}

    def value_norms(self):
        """{헤드: ValueNorm 또는 None}."""
        return {h: getattr(self, f"value_norm_{h}", None) for h in ROLE_HEADS}

    def module_groups(self):
        """파라미터 집계용: 네트워크별 GNN (encoder 포함) / decoder, critic."""
        out = {}
        for h, n in self.head_networks().items():
            out[f"{h}.gnn"] = [n.gnn]
            out[f"{h}.decoder"] = [m for m in n.children() if m is not n.gnn]
        for h in ROLE_HEADS:
            out[f"critic.{h}"] = [getattr(self, f"critic_{h}")]
        return out

    # ------------------------------------------------------------ forward

    def forward(self, ob, taken=None, greedy=False):
        """ob: stack_role_obs() 결과. taken: {'role', 'rescue', 'explore'} (B,N) long — 학습 재생 시 실제 행동.
        반환: logits {head: (B,N,K_h)} (결정 시점 마스크 적용), actions {head: (B,N)}, stage {'s1', 'rank'},
        values / values_norm {head: (B,N)}."""
        B, N = ob["role_self"].shape[:2]
        T, K, dev = self.T, self.K, ob["role_self"].device
        role_ctx = torch.cat([ob["role_self"], ob["role_group"]], dim=-1)
        cur_role = ob["team_ctx"][..., 0:3].argmax(-1)                       # 현재 역할
        group = ob["group"] > 0.5
        eye = torch.eye(N, dtype=torch.bool, device=dev)[None]
        needed = ob["role_needed"][..., 0] > 0.5                            # D_0 (이번 step role 재결정)
        r_mask0, res_mask0, exp_mask = ob["role_mask"] > 0.5, ob["rescue_mask"] > 0.5, ob["explore_mask"] > 0.5
        ids = ob["s_ids"]
        s_valid = ids >= 0
        pos = ob["team_ctx"][..., 6:8]                                      # 절대 정규화 위치
        scale = ob["geo"][:, 0:2]                                           # (cols, rows) — 정규화 좌표 -> 셀
        comm = ob["team_comm"]
        rc = torch.minimum(comm[:, :, None], comm[:, None, :]).clamp(min=1e-6)   # 쌍별 R_c
        cap = ob["team_cap"]
        rd, rsl, ed = own(ob["us_d"]), own(ob["us_slack"]), own(ob["uk_d"])
        ar_k = torch.arange(K, device=dev)

        # 순차 디코딩 상태 (팀 테이블 — 절대 좌표)
        role_act = cur_role.clone()
        res_act = torch.zeros(B, N, dtype=torch.long, device=dev)
        exp_act = torch.zeros(B, N, dtype=torch.long, device=dev)
        s1 = torch.zeros(B, N, dtype=torch.bool, device=dev)
        done = torch.zeros(B, N, dtype=torch.bool, device=dev)
        goal = ob["team_gpos"].clone()                                      # 목표 지점 g_l (없으면 현재 위치)
        tgt = ob["team_target"].clone()                                     # RESCUE 목표 생존자 id (-1 없음)
        slot_now = ob["team_slot"].round().long()                           # EXPLORE 영역 슬롯 (-1 없음)
        picked = torch.full((B, N), -1.0, device=dev)                       # 이번 step 고른 생존자 id

        role_logits = _masked(torch.zeros(B, N, N_ROLES, device=dev), r_mask0.float())
        res_logits = _masked(torch.zeros(B, N, T, device=dev), res_mask0.float())
        exp_logits = _masked(torch.zeros(B, N, K, device=dev), exp_mask.float())

        def choose(logits, head, bi, ii):
            if taken is not None:
                return taken[head][bi, ii].long().clamp(0, logits.shape[-1] - 1)
            if greedy:
                return logits.argmax(-1)
            return torch.distributions.Categorical(logits=logits.detach()).sample()

        def taken_victims():                                                # (B,N,T) 같은 그룹이 이번 step 고른 생존자
            same = (ids[:, :, None, :] == picked[:, None, :, None]) & (picked[:, None, :, None] >= 0)
            return (same & group[..., None] & done[:, None, :, None]).any(dim=2)

        def build(kind, bi, ii):
            """kind 'role' | 'rescue' | 'explore' 네트워크의 그래프 ((bi, ii) 차례 UAV 들, G 개)."""
            G = bi.shape[0]
            A = lambda k: ob[k][bi, ii]
            p = pos[bi]                                                     # (G,N,2)
            ra = role_act[bi]
            mU = group[bi, ii]                                              # 그룹 밖 UAV 없음
            if kind == "rescue":
                mU = mU & (ra == ROLE_RESCUE)
            elif kind == "explore":
                mU = mU & (ra == ROLE_EXPLORE)
            xU = torch.cat([p, cap[bi][..., None], ob["team_rho"][bi][..., None], ob["team_eta"][bi][..., None],
                            ob["team_nu"][bi][..., None], Fn.one_hot(ra, N_ROLES).float(), goal[bi],
                            ob["team_depot_d"][bi][..., None]], -1)
            dp = p[:, None, :, :] - p[:, :, None, :]                        # [a, b] = p_b - p_a
            dist = (dp * scale[bi][:, None, None, :]).norm(dim=-1) / rc[bi]
            tg, sl = tgt[bi], slot_now[bi]
            overlap = ((tg[:, :, None] == tg[:, None, :]) & (tg[:, :, None] >= 0)) | \
                      ((sl[:, :, None] == sl[:, None, :]) & (sl[:, :, None] >= 0))
            x, mask = {"U": xU}, {"U": mU}
            edge = {"UU": torch.cat([dp, dist.clamp(max=DIST_CLIP)[..., None], overlap.float()[..., None]], -1)}
            adj = {"UU": mU[:, :, None] & mU[:, None, :] & ~eye}
            if kind in ("role", "rescue"):
                sid = A("s_ids")
                mS = sid >= 0
                sp = A("s_pos")
                x["S"], mask["S"] = torch.cat([sp, A("s_elapsed")[..., None]], -1), mS
                is_t = (tg[:, :, None] == sid[:, None, :]) & mS[:, None, :]
                edge["US"] = torch.stack([A("us_d"), A("us_sigma"), is_t.float()], -1)
                adj["US"] = mU[:, :, None] & mS[:, None, :]
                edge["SS"] = torch.cat([sp[:, None, :, :] - sp[:, :, None, :], A("ss_dist")[..., None]], -1)
                adj["SS"] = knn_adj(A("ss_dist"), mS, self.knn_k)
            else:
                mK = A("explore_kcap") > 0.5
                kp = A("k_pos")
                x["K"], mask["K"] = torch.cat([kp, A("k_size")[..., None]], -1), mK
                asg = (sl[:, :, None] == ar_k) & (sl[:, :, None] >= 0)
                edge["UK"] = torch.stack([A("uk_d"), asg.float()], -1)
                adj["UK"] = mU[:, :, None] & mK[:, None, :]
                edge["KK"] = torch.cat([kp[:, None, :, :] - kp[:, :, None, :], A("kk_dist")[..., None]], -1)
                adj["KK"] = knn_adj(A("kk_dist"), mK, self.knn_k)
            return {"bi": bi, "ii": ii, "t": ob["time"][bi], "x": x, "mask": mask, "edge": edge, "adj": adj}

        def run(net, gr):
            if self.timing is not None:
                import time
                t0 = time.perf_counter()
                out = net(gr)
                self.timing["gnn_s"] = self.timing.get("gnn_s", 0.0) + time.perf_counter() - t0
                self.timing["gnn_calls"] = self.timing.get("gnn_calls", 0) + 1
                self.timing["gnn_rows"] = self.timing.get("gnn_rows", 0) + int(gr["ii"].shape[0])
                return out
            return net(gr)

        def scatter(base, bi, ii, val):
            return base.index_put((bi, ii), val)

        def trace(stage, sel, gr, h):
            if self.debug_trace is not None:
                self.debug_trace.append({"stage": stage, "sel": sel.clone(), "graph": gr, "h": h})

        rank = {k: torch.zeros(B, N, dtype=torch.long, device=dev) for k in ("role", "s1")}   # 실제 차례 (1, 2, ...)
        # ---- [1] role_front 차례 -> 역할 선택 -> RESCUE 면 바로 생존자 할당 (한 대마다 차례 재계산) ----
        valid = s_valid & res_mask0
        sig = role_sigma(ids, rsl, valid, group, needed)
        rem = needed.clone()
        for it in range(N):
            if not bool(rem.any()):
                break
            res_mask = res_mask0 & ~taken_victims()
            sel, _ = role_front(group, rem, cap, ids, rd, sig, valid & res_mask)
            bi, ii = sel.nonzero(as_tuple=True)
            rank["role"] = torch.where(sel, torch.full_like(rank["role"], it + 1), rank["role"])
            r_mask = r_mask0.clone()
            r_mask[..., ROLE_RESCUE] &= res_mask.any(-1)                    # 남은 생존자 없으면 RESCUE 금지
            r_mask[..., ROLE_RETURN] |= ~r_mask.any(-1)
            gr = build("role", bi, ii)
            lg_raw, h = run(self.role_net, gr)
            lg = _masked(lg_raw, r_mask[bi, ii].float())
            ra = choose(lg, "role", bi, ii)
            trace("role", sel, gr, h)
            role_logits = scatter(role_logits, bi, ii, lg)
            role_act = scatter(role_act, bi, ii, ra)
            tgt = scatter(tgt, bi, ii, torch.full_like(ii, -1, dtype=tgt.dtype))
            ne = ra != ROLE_EXPLORE
            slot_now = scatter(slot_now, bi[ne], ii[ne], torch.full_like(ii[ne], -1))
            ret = ra == ROLE_RETURN
            goal = scatter(goal, bi[ret], ii[ret], ob["depot_pos"][bi[ret]])        # RETURN 목표 = depot
            on = ra == ROLE_RESCUE
            if bool(on.any()):                                              # 생존자 할당 네트워크 (역할 반영 후 따로 계산)
                b2, i2 = bi[on], ii[on]
                gr2 = build("rescue", b2, i2)
                rl_raw, h2 = run(self.rescue_net, gr2)
                rm = res_mask[b2, i2]
                rl = _masked(rl_raw, rm.float())
                rs = choose(rl, "rescue", b2, i2)
                trace("rescue", sel, gr2, h2)
                res_logits = scatter(res_logits, b2, i2, rl)
                res_act = scatter(res_act, b2, i2, rs)
                rsc = rs.clamp(0, T - 1)
                vid = ids[b2, i2, rsc]
                picked = scatter(picked, b2, i2, vid)
                tgt = scatter(tgt, b2, i2, vid)
                goal = scatter(goal, b2, i2, ob["s_pos"][b2, i2, rsc])
            done = done | sel
            rem = rem & ~sel

        # ---- [2] explore 1 단계: 신규 EXPLORE UAV 만 explore_front 차례 (실제 선택으로 n_k 갱신, 한 대마다 재계산) ----
        #      유지 UAV (직전에도 EXPLORE, 영역 유효 = explore_cur >= 0) 는 영역 그대로, n_k 에 기존 배정자로
        cur = ob["explore_cur"][..., 0].round().long()
        retained = needed & (role_act == ROLE_EXPLORE) & (cur >= 0)
        rem = needed & (role_act == ROLE_EXPLORE) & ~retained
        ret_hot = Fn.one_hot(cur.clamp(0, K - 1), K).float() * retained[..., None].float()   # [b,j,k]
        n_now = torch.round(ob["k_n"] * N) + torch.einsum("bij,bjk->bik", group.float(), ret_hot)
        for it in range(N):
            if not bool(rem.any()):
                break
            sel, _ = explore_front(group, rem, exp_mask, ed, ob["explore_size"], n_now)
            bi, ii = sel.nonzero(as_tuple=True)
            rank["s1"] = torch.where(sel, torch.full_like(rank["s1"], it + 1), rank["s1"])
            gr = build("explore", bi, ii)
            el_raw, h = run(self.explore_net, gr)
            el = _masked(el_raw, exp_mask[bi, ii].float())
            ex = choose(el, "explore", bi, ii)
            trace("s1", sel, gr, h)
            exp_logits = scatter(exp_logits, bi, ii, el)
            exp_act = scatter(exp_act, bi, ii, ex)
            s1 = s1 | sel
            slot_now = scatter(slot_now, bi, ii, ex)
            inc = Fn.one_hot(slot_now.clamp(0, K - 1), K).float() * sel[..., None].float()
            n_now = n_now + torch.einsum("bij,bjk->bik", group.float(), inc)
            goal = scatter(goal, bi, ii, ob["k_pos"][bi, ii, ex.clamp(0, K - 1)])
            rem = rem & ~sel
        exp_act = torch.where(s1, exp_act, cur.clamp(min=0))

        if taken is not None:
            role_act, res_act, exp_act = taken["role"].long(), taken["rescue"].long(), taken["explore"].long()

        # ---- critics ----
        gat = lambda x, a: torch.gather(x, 2, a[..., None]).squeeze(-1)
        ra_, ea_ = res_act.clamp(0, T - 1), exp_act.clamp(0, K - 1)
        res_feat = torch.cat([torch.gather(ob["s_pos"], 2, ra_[..., None, None].expand(B, N, 1, 2)).squeeze(2) - pos,
                              torch.stack([gat(ob["s_tau"], ra_), gat(ob["s_held"], ra_), gat(rd, ra_), gat(rsl, ra_)], -1)], -1)
        exp_feat = torch.cat([torch.gather(ob["k_pos"], 2, ea_[..., None, None].expand(B, N, 1, 2)).squeeze(2) - pos,
                              torch.stack([gat(ob["k_size"], ea_), gat(ob["k_n"], ea_), gat(ed, ea_)], -1)], -1)
        team_act = torch.cat([Fn.one_hot(role_act.clamp(0, N_ROLES - 1), N_ROLES).float(), goal], dim=-1).detach()
        ckw = dict(self_feats=ob["role_self"][..., :3], tasks=ob["critic_tasks"], task_present=ob["critic_task_present"],
                   team=ob["critic_team"], team_act=team_act, glob=ob["critic_global"])
        raw = {
            "role": self.critic_role(ob["state"], ob["local"], role_ctx, **ckw),
            "rescue": self.critic_rescue(ob["state"], ob["local"], torch.cat([role_ctx, res_feat], -1), **ckw),
            "explore": self.critic_explore(ob["state"], ob["local"], torch.cat([role_ctx, exp_feat], -1), **ckw),
        }
        vn = self.value_norms()
        values = {h: (vn[h].denormalize(v) if vn[h] is not None else v) for h, v in raw.items()}
        return {"logits": {"role": role_logits, "rescue": res_logits, "explore": exp_logits},
                "actions": {"role": role_act, "rescue": res_act, "explore": exp_act},
                "stage": {"s1": s1, "rank": rank},
                "values": values, "values_norm": raw}
