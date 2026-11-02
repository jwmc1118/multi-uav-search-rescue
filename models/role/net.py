"""RoleNet — 계층 구조 env (env/role/simulation.py) 용 Role / task (하위 할당) actor + critic (순수 torch).

독립된 네트워크 2 개 (encoder · GNN 파라미터 공유 없음, models/role/gnn.py HeteroGNN, 그래프 사양 env/role/graph_obs.py).
두 네트워크의 그래프는 같은 구성:
    노드 U (현재 그룹 = 연결 노드 + 그룹 밖 UAV 의 마지막 기록 = 메모리 노드, 역할 무관) · S (생존자) · F (frontier 덩어리)
    관계 S→U, U→S, F→U, U→F, U→U, S→S, F→F, S→F, F→S, U–S / U–F 간선 층마다 갱신
    encoder = 노드 종류별 · 간선 종류별 2 층 MLP (ReLU) -> D, 그룹 context c_i = [ρ_i, η_i, ν_i, t / T_max] 는 ξ_ctx (2 층 MLP) -> D
[역할 선택]   logits = ξ^role_dec( h_i^(L) ‖ ξ_ctx(c_i) ) ∈ R^3 -> role_mask -> softmax
[task (하위 할당)]   후보 = S ∪ F (행동 = S 인덱스 / s_max + F 인덱스), pointer attention
              q_i = W_Q [h_i^(L) ‖ r_i ‖ ξ_ctx(c_i)] (r_i = 방금 고른 역할 one-hot), k_j = W_K [h_j^(L) ‖ e_ij^(L)]
              (j ∈ S: U–S 간선, j ∈ F: U–F 간선), u_ij = C tanh(q_iᵀ k_j / √D), π(j | i) = softmax_{S ∪ F}(u_ij + M_ij)
              M: RESCUE -> rescue_mask 생존자만 (F 전부 차단, 같은 그룹 앞 차례가 고른 생존자 제외), EXPLORE -> explore_mask
              덩어리만 (S 전부 차단, 같은 덩어리 중복 허용). 1 단계만 — 2 단계 구간 배정은 env 의 거리 Hungarian.
[Critic]      models/common/critic.py RichCentralCritic (actor 와 인코더 비공유)
              role : V^role (extra = [role_self, role_group]) + ValueNorm
              할당 : 공유 trunk + value head 2 개 (V^res, V^exp) + ValueNorm 2 개 (역할별)
                     extra = [role_self, role_group] ‖ 고른 S 특징 (6) ‖ 고른 F 특징 (5) (rescue / explore critic 입력의 합집합)

순차 디코딩 — env _decide 와 같은 흐름 · 같은 차례 함수 (policies/decision_order.py, 한 대가 결정할 때마다 다시 계산):
  [1] role_front 차례 -> 역할 선택 네트워크 -> RESCUE 면 바로 task 네트워크 (r = RESCUE, 역할 반영 후 그래프를 따로 계산)
  [2] EXPLORE 를 고른 재결정 UAV 전원을 explore_front 차례로 -> task 네트워크 (r = EXPLORE, 지금 덩어리도 후보). 차례 계산의
      배정 수 n_k = 재결정하지 않는 기존 배정자 (관측 k_n) + 이번 이벤트 앞 차례
  결정마다 역할 · 목표 (RESCUE = 생존자, RETURN = depot, EXPLORE 1 단계 = 고른 덩어리의 F 노드 위치 p^f) 를 연결 노드 상태에 반영하고 다음
  차례 그래프를 다시 만든다 (메모리 노드는 마지막 기록 그대로). 학습 재생 (taken = SampleBatch.ACTIONS) 도 같은 루프.
모든 파라미터는 N (UAV 수) 과 무관.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from env.role.graph_obs import DIST_CLIP, EFF, ESF, ESS, EUF, EUS, EUU, FC, FF, FS, FU
from policies.decision_order import explore_front, role_front, role_sigma
from models.role.gnn import HeteroGNN, mlp2
from models.common.critic import RichCentralCritic, CRITIC_TASK_DIM, CRITIC_GLOBAL_DIM
from models.common.value_norm import ValueNorm

MASK_VALUE = -1e9          # 마스크된 후보 / 역할의 logit
ROLE_EXPLORE, ROLE_RESCUE, ROLE_RETURN = 0, 1, 2
N_ROLES = 3
POLICY_HEADS = ("role", "task")                 # actor 네트워크 (학습 head)
VALUE_HEADS = ("role", "rescue", "explore")      # value head (rescue / explore = task critic 의 역할별 head)
RES_FEAT_DIM = 6          # critic 용 고른 S 특징: rel_pos(2), (τ - t) / T, 맡겨짐, BFS / d_max, slack
EXP_FEAT_DIM = 5          # critic 용 고른 F 특징: rel_pos(2), 크기, 배정 수 / N, BFS / d_max

GNN_DEFAULTS = {"embed_dim": 128, "n_layers": 2, "knn_k": 6, "s_max": 16, "attn_heads": 4}
CRITIC_DEFAULTS = {"cnn_filters": [[16, 3, 2], [32, 3, 2], [64, 3, 1]], "state_features": 128, "local_features": 64,
                   "hidden": [128, 64], "set_dim": 64, "set_heads": 4}
# 네트워크별 기본값 (custom_model_config.role / task 으로 덮어씀)
NET_DEFAULTS = {
    "role": {"entropy_coeff": 0.01, "value_norm": True, "critic": CRITIC_DEFAULTS},
    "task": {"clip": 10.0,                      # pointer tanh clip C
              "entropy_coeff_rescue": 0.01,      # RESCUE 표본 엔트로피 보너스 계수
              "entropy_coeff_explore": 0.01,     # EXPLORE 표본 엔트로피 보너스 계수
              "value_norm": True,                # 역할별 ValueNorm (V^res, V^exp)
              "critic": CRITIC_DEFAULTS},
}
LOSS_DEFAULTS = {"lambda_role": 1.0, "lambda_task": 1.0}  # L = λ_role L^role + λ_task L^task
DEFAULT_ROLE_MODEL_OPTIONS = {"activation": "relu",
                              "order_dist_alpha": 1.0}   # explore 1 단계 차례의 α — env_config.explore.order_dist_alpha 와 같아야
                                                         #   함 (training/config_loader.py 가 복사, 파라미터 아님)

# 간선 BASE 방향 (a 행, b 열) 과 관계 (이름, src, dst, 간선 키) — 두 네트워크 공통
BASE = {"UU": ("U", "U"), "US": ("U", "S"), "UF": ("U", "F"), "SS": ("S", "S"), "FF": ("F", "F"), "SF": ("S", "F")}
RELATIONS = (("S2U", "S", "U", "US"), ("U2S", "U", "S", "US"), ("F2U", "F", "U", "UF"), ("U2F", "U", "F", "UF"),
             ("U2U", "U", "U", "UU"), ("S2S", "S", "S", "SS"), ("F2F", "F", "F", "FF"),
             ("S2F", "S", "F", "SF"), ("F2S", "F", "S", "SF"))
NODE_DIMS = {"U": FU, "S": FS, "F": FF}
EDGE_DIMS = {"UU": EUU, "US": EUS, "UF": EUF, "SS": ESS, "FF": EFF, "SF": ESF}
EDGE_UPDATE = ("US", "UF")


def resolve_model_options(options):
    """custom_model_config -> {'activation', 'order_dist_alpha', 'gnn': {...}, 'role': {...}, 'task': {...}, 'loss': {...}}.
    모르는 키 (예전 config 에 남은 삭제된 항목) 는 무시."""
    options = dict(options or {})
    out = copy.deepcopy(DEFAULT_ROLE_MODEL_OPTIONS)
    out.update({k: options[k] for k in DEFAULT_ROLE_MODEL_OPTIONS if k in options})
    out["gnn"] = dict(GNN_DEFAULTS, **{k: v for k, v in (options.get("gnn") or {}).items() if k in GNN_DEFAULTS})
    for h in POLICY_HEADS:
        hc = copy.deepcopy(NET_DEFAULTS[h])
        given = copy.deepcopy(options.get(h) or {})
        hc["critic"].update({k: v for k, v in (given.pop("critic", {}) or {}).items() if k in CRITIC_DEFAULTS})
        hc.update({k: v for k, v in given.items() if k in NET_DEFAULTS[h]})
        out[h] = hc
    out["loss"] = dict(LOSS_DEFAULTS, **{k: v for k, v in (options.get("loss") or {}).items() if k in LOSS_DEFAULTS})
    return out


AGENT_KEYS = ("local", "role_self", "role_group", "role_mask", "role_needed",
              "s_ids", "s_pos", "s_tau", "s_held", "s_elapsed", "rescue_mask", "us_d", "us_slack", "us_sigma", "ss_dist",
              "k_pos", "k_size", "k_n", "uk_d", "kk_dist", "explore_mask", "explore_cur", "explore_kcap", "explore_size",
              "sf_dist", "mem_valid", "mem_age", "mem_pos", "mem_cap", "mem_role", "mem_goal", "mem_depot_d", "mem_omega",
              "mem_target")
TEAM_KEYS = ("team_ctx", "team_cap", "team_comm", "team_target", "team_slot", "team_gpos", "team_depot_d", "depot_pos",
             "team_rho", "team_eta", "team_nu", "team_time", "team_omega", "geo", "group", "time", "state",
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


def knn_adj(d, mask, k, mask_col=None):
    """d (G,n,m) 거리, mask (G,n) 행 노드 유효, mask_col (G,m) 열 노드 유효 (없으면 정사각 · 자기 제외) -> adj (G,n,m) bool:
    행 v 가 받는 이웃 = 유효 열 노드 중 거리 <= k 번째로 가까운 이웃의 거리 — k 번째와 동률인 이웃은 모두 포함
    (이웃 수가 k 를 넘을 수 있음, 노드 순서와 무관)."""
    G, n, m = d.shape
    square = mask_col is None
    mask_col = mask if square else mask_col
    kk = min(int(k), m - 1 if square else m)
    if kk <= 0:
        return torch.zeros(G, n, m, dtype=torch.bool, device=d.device)
    bad = ~mask_col[:, None, :].expand(G, n, m)
    if square:
        bad = bad | torch.eye(n, dtype=torch.bool, device=d.device)[None]
    dd = d.masked_fill(bad, 1e9)
    kth = torch.topk(dd, kk, dim=-1, largest=False).values[..., -1:]     # k 번째 거리 (유효 이웃이 k 개 미만이면 1e9)
    return (dd <= kth) & (dd < 1e8) & mask[:, :, None]


def own(x):
    """(B,N,N,...) 의 [b, i, i, ...] (UAV i 관측의 자기 행) -> (B,N,...)."""
    return torch.diagonal(x, dim1=1, dim2=2).movedim(-1, 1)


class GraphNetwork(nn.Module):
    """공통 블록: HeteroGNN (노드 U · S · F, RELATIONS, U–S / U–F 간선 갱신) + 그룹 context encoder ξ_ctx."""

    def __init__(self, D, g):
        super().__init__()
        self.gnn = HeteroGNN(NODE_DIMS, EDGE_DIMS, BASE, RELATIONS, D, g["n_layers"], g["attn_heads"], edge_update=EDGE_UPDATE)
        self.ctx = mlp2(FC, D, D)

    def encode(self, gr):
        h, _, e = self.gnn(gr["x"], gr["mask"], gr["edge"], gr["adj"])
        r = torch.arange(gr["ii"].shape[0], device=gr["ii"].device)
        return h, e, r, self.ctx(gr["ctx"])


class RoleNetwork(GraphNetwork):
    """역할 선택: logits = ξ_dec(h_i^(L) ‖ ξ_ctx(c_i)) ∈ R^3."""

    def __init__(self, D, g):
        super().__init__(D, g)
        self.dec = mlp2(2 * D, D, N_ROLES)

    def forward(self, gr):
        h, _, r, c = self.encode(gr)
        return self.dec(torch.cat([h["U"][r, gr["ii"]], c], -1)), h


class TaskNetwork(GraphNetwork):
    """task (하위 할당): 후보 S ∪ F pointer attention u_ij = C tanh(q_iᵀ k_j / √D), q_i = W_Q [h_i ‖ r_i ‖ ξ_ctx(c_i)],
    k_j = W_K [h_j ‖ e_ij] (j ∈ S: U–S 간선, j ∈ F: U–F 간선). 마스크는 호출부."""

    def __init__(self, D, g, clip):
        super().__init__(D, g)
        self.D, self.clip = D, float(clip)
        self.w_q = nn.Linear(2 * D + N_ROLES, D)
        self.w_k = nn.Linear(2 * D, D)

    def forward(self, gr):
        h, e, r, c = self.encode(gr)
        ii = gr["ii"]
        q = self.w_q(torch.cat([h["U"][r, ii], gr["r"], c], -1))
        cand = torch.cat([torch.cat([h["S"], e["US"][r, ii]], -1), torch.cat([h["F"], e["UF"][r, ii]], -1)], 1)
        u = (q.unsqueeze(1) * self.w_k(cand)).sum(-1) / math.sqrt(self.D)
        return self.clip * torch.tanh(u), h


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
        self.role_net = RoleNetwork(D, g)
        self.task_net = TaskNetwork(D, g, cfg["task"]["clip"])
        role_ctx = dims["role_self"] + dims["role_group"]

        def critic(hc, extra, n_out=1):
            c = hc["critic"]
            return RichCentralCritic(extra_dim=extra, state_channels=state_channels, local_channels=local_channels,
                                     local_hw=(local_size, local_size), state_dim=c["state_features"],
                                     local_dim=c["local_features"], filters=tuple(tuple(f) for f in c["cnn_filters"]),
                                     hidden=tuple(c["hidden"]), activation=act, set_dim=c["set_dim"],
                                     set_heads=c["set_heads"], task_dim=dims.get("critic_task", CRITIC_TASK_DIM),
                                     team_dim=dims["critic_team"], global_dim=dims.get("critic_global", CRITIC_GLOBAL_DIM),
                                     n_out=n_out)
        self.critic_role = critic(cfg["role"], role_ctx)
        self.critic_task = critic(cfg["task"], role_ctx + RES_FEAT_DIM + EXP_FEAT_DIM, n_out=2)   # [V^res, V^exp]
        self.debug_trace = None              # 리스트를 넣으면 순차 디코딩 차례별 그래프 / 임베딩을 기록 (테스트용)
        self.timing = None                   # dict 를 넣으면 GNN 호출 수 / 결정 수 / 시간을 더한다 (G 로그)
        if cfg["role"]["value_norm"]:
            self.value_norm_role = ValueNorm()
        if cfg["task"]["value_norm"]:       # 역할별 ValueNorm 2 개
            self.value_norm_rescue = ValueNorm()
            self.value_norm_explore = ValueNorm()

    def head_networks(self):
        """{head: 그 head 의 actor 네트워크} — 두 네트워크는 파라미터를 공유하지 않는다."""
        return {"role": self.role_net, "task": self.task_net}

    def encoder_parameters(self):
        """{head: 그 네트워크의 encoder + GNN 파라미터} (gradient norm 측정용)."""
        return {h: list(n.gnn.parameters()) for h, n in self.head_networks().items()}

    def value_norms(self):
        """{value head: ValueNorm 또는 None}."""
        return {h: getattr(self, f"value_norm_{h}", None) for h in VALUE_HEADS}

    def entropy_coeffs(self):
        """{value head (= 표본 종류): 엔트로피 계수} — role / RESCUE 표본 / EXPLORE 표본."""
        a = self.cfg["task"]
        return {"role": float(self.cfg["role"]["entropy_coeff"]), "rescue": float(a["entropy_coeff_rescue"]),
                "explore": float(a["entropy_coeff_explore"])}

    def module_groups(self):
        """파라미터 집계용: 네트워크별 GNN (encoder 포함) / decoder (context encoder 포함), critic."""
        out = {}
        for h, n in self.head_networks().items():
            out[f"{h}.gnn"] = [n.gnn]
            out[f"{h}.decoder"] = [m for m in n.children() if m is not n.gnn]
        out["critic.role"] = [self.critic_role]
        out["critic.task"] = [self.critic_task]
        return out

    # ------------------------------------------------------------ forward

    def forward(self, ob, taken=None, greedy=False):
        """ob: stack_role_obs() 결과. taken: {'role', 'task'} (B,N) long — 학습 재생 시 실제 행동.
        반환: logits {role (B,N,3), task (B,N,T+K)} (결정 시점 마스크 적용), actions {role, task} (B,N),
        stage {'s1', 'rescue', 'rank'}, values / values_norm {role, rescue, explore} (B,N)."""
        B, N = ob["role_self"].shape[:2]
        T, K, dev = self.T, self.K, ob["role_self"].device
        role_ctx = torch.cat([ob["role_self"], ob["role_group"]], dim=-1)
        ctx = torch.stack([ob["team_rho"], ob["team_eta"], ob["team_nu"], ob["team_time"]], -1)   # 그룹 context c_i (B,N,FC=4)
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
        no_s = torch.zeros(B, N, T, dtype=torch.bool, device=dev)
        no_f = torch.zeros(B, N, K, dtype=torch.bool, device=dev)

        # 순차 디코딩 상태 (팀 테이블 — 절대 좌표, 연결 노드)
        role_act = cur_role.clone()
        cur = ob["explore_cur"][..., 0].round().long()
        task_act = torch.where(cur >= 0, T + cur, torch.zeros_like(cur))  # 결정 없는 UAV: 지금 덩어리 (없으면 0, dummy)
        s1 = torch.zeros(B, N, dtype=torch.bool, device=dev)
        resc = torch.zeros(B, N, dtype=torch.bool, device=dev)
        done = torch.zeros(B, N, dtype=torch.bool, device=dev)
        goal = ob["team_gpos"].clone()                                      # 목표 지점 g_l (없으면 현재 위치)
        tgt = ob["team_target"].clone()                                     # RESCUE 목표 생존자 id (-1 없음)
        slot_now = ob["team_slot"].round().long()                           # EXPLORE 영역 슬롯 (-1 없음)
        picked = torch.full((B, N), -1.0, device=dev)                       # 이번 step 고른 생존자 id

        role_logits = _masked(torch.zeros(B, N, N_ROLES, device=dev), r_mask0.float())
        task_logits = _masked(torch.zeros(B, N, T + K, device=dev), torch.cat([res_mask0, exp_mask], -1).float())

        def choose(logits, head, bi, ii):
            if taken is not None:
                return taken[head][bi, ii].long().clamp(0, logits.shape[-1] - 1)
            if greedy:
                return logits.argmax(-1)
            return torch.distributions.Categorical(logits=logits.detach()).sample()

        def taken_victims():                                                # (B,N,T) 같은 그룹이 이번 step 고른 생존자
            same = (ids[:, :, None, :] == picked[:, None, :, None]) & (picked[:, None, :, None] >= 0)
            return (same & group[..., None] & done[:, None, :, None]).any(dim=2)

        def build(bi, ii, r=None):
            """(bi, ii) 차례 UAV 들 (G 개) 의 그래프 — U (연결 + 메모리) · S · F. r = task 네트워크의 역할 one-hot."""
            A = lambda k: ob[k][bi, ii]
            link = group[bi, ii]                                            # 연결 노드 (현재 그룹)
            memn = (A("mem_valid") > 0.5) & ~link                           # 메모리 노드 (그룹 밖 마지막 기록)
            mU = link | memn
            lk = link[..., None]
            p = torch.where(lk, pos[bi], A("mem_pos"))                      # (G,N,2)
            ra = torch.where(link, role_act[bi], A("mem_role").round().long().clamp(0, N_ROLES - 1))
            tg = torch.where(link, tgt[bi], A("mem_target"))
            sl = torch.where(link, slot_now[bi], torch.full_like(slot_now[bi], -1))   # 슬롯 (연결 노드만)
            f = lambda live, mem: torch.where(link, live, mem)[..., None]
            xU = torch.cat([p, f(cap[bi], A("mem_cap")), Fn.one_hot(ra, N_ROLES).float(),
                            torch.where(lk, goal[bi], A("mem_goal")), f(ob["team_depot_d"][bi], A("mem_depot_d")),
                            f(ob["team_omega"][bi], A("mem_omega")), link.float()[..., None],
                            torch.where(link, torch.zeros_like(A("mem_age")), A("mem_age"))[..., None]], -1)
            dp = p[:, None, :, :] - p[:, :, None, :]                        # [a, b] = p_b - p_a
            dist = (dp * scale[bi][:, None, None, :]).norm(dim=-1) / rc[bi]
            overlap = ((tg[:, :, None] == tg[:, None, :]) & (tg[:, :, None] >= 0)) | \
                      ((sl[:, :, None] == sl[:, None, :]) & (sl[:, :, None] >= 0))
            sid = A("s_ids")
            mS = sid >= 0
            sp = A("s_pos")
            mF = A("explore_kcap") > 0.5
            kp = A("k_pos")
            is_t = (tg[:, :, None] == sid[:, None, :]) & mS[:, None, :]
            asg = (sl[:, :, None] == ar_k) & (sl[:, :, None] >= 0)
            x = {"U": xU, "S": torch.cat([sp, A("s_elapsed")[..., None]], -1), "F": torch.cat([kp, A("k_size")[..., None]], -1)}
            mask = {"U": mU, "S": mS, "F": mF}
            edge = {"UU": torch.cat([dp, dist.clamp(max=DIST_CLIP)[..., None], overlap.float()[..., None]], -1),
                    "US": torch.stack([A("us_d"), A("us_sigma"), is_t.float()], -1),
                    "UF": torch.stack([A("uk_d"), asg.float()], -1),
                    "SS": torch.cat([sp[:, None, :, :] - sp[:, :, None, :], A("ss_dist")[..., None]], -1),
                    "FF": torch.cat([kp[:, None, :, :] - kp[:, :, None, :], A("kk_dist")[..., None]], -1),
                    "SF": torch.cat([kp[:, None, :, :] - sp[:, :, None, :], A("sf_dist")[..., None]], -1)}
            adj = {"UU": mU[:, :, None] & mU[:, None, :] & ~eye, "US": mU[:, :, None] & mS[:, None, :],
                   "UF": mU[:, :, None] & mF[:, None, :], "SS": knn_adj(A("ss_dist"), mS, self.knn_k),
                   "FF": knn_adj(A("kk_dist"), mF, self.knn_k), "SF": knn_adj(A("sf_dist"), mS, self.knn_k, mF)}
            gr = {"bi": bi, "ii": ii, "ctx": ctx[bi, ii], "x": x, "mask": mask, "edge": edge, "adj": adj}
            if r is not None:
                gr["r"] = Fn.one_hot(torch.full_like(ii, r), N_ROLES).float()
            return gr

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

        def trace(stage, sel, gr, h, lg=None):
            if self.debug_trace is not None:
                self.debug_trace.append({"stage": stage, "sel": sel.clone(), "graph": gr, "h": h, "logits": lg})

        rank = {k: torch.zeros(B, N, dtype=torch.long, device=dev) for k in ("role", "s1")}   # 실제 차례 (1, 2, ...)
        # ---- [1] role_front 차례 -> 역할 선택 -> RESCUE 면 바로 할당 (한 대마다 차례 재계산) ----
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
            gr = build(bi, ii)
            lg_raw, h = run(self.role_net, gr)
            lg = _masked(lg_raw, r_mask[bi, ii].float())
            ra = choose(lg, "role", bi, ii)
            trace("role", sel, gr, h, lg)
            role_logits = scatter(role_logits, bi, ii, lg)
            role_act = scatter(role_act, bi, ii, ra)
            tgt = scatter(tgt, bi, ii, torch.full_like(ii, -1, dtype=tgt.dtype))
            ne = ra != ROLE_EXPLORE
            slot_now = scatter(slot_now, bi[ne], ii[ne], torch.full_like(ii[ne], -1))
            ret = ra == ROLE_RETURN
            goal = scatter(goal, bi[ret], ii[ret], ob["depot_pos"][bi[ret]])        # RETURN 목표 = depot
            on = ra == ROLE_RESCUE
            if bool(on.any()):                                              # task 네트워크 (r = RESCUE, 역할 반영 후 따로 계산)
                b2, i2 = bi[on], ii[on]
                gr2 = build(b2, i2, ROLE_RESCUE)
                al_raw, h2 = run(self.task_net, gr2)
                al = _masked(al_raw, torch.cat([res_mask[b2, i2], no_f[b2, i2]], -1).float())   # F 전부 차단
                rs = choose(al, "task", b2, i2)
                trace("rescue", sel, gr2, h2, al)
                task_logits = scatter(task_logits, b2, i2, al)
                task_act = scatter(task_act, b2, i2, rs)
                resc = scatter(resc, b2, i2, torch.ones_like(i2, dtype=torch.bool))
                rsc = rs.clamp(0, T - 1)
                vid = ids[b2, i2, rsc]
                picked = scatter(picked, b2, i2, vid)
                tgt = scatter(tgt, b2, i2, vid)
                goal = scatter(goal, b2, i2, ob["s_pos"][b2, i2, rsc])
            done = done | sel
            rem = rem & ~sel

        # ---- [2] explore 1 단계: EXPLORE 를 고른 재결정 UAV 전원 explore_front 차례 (실제 선택으로 n_k 갱신, 한 대마다 재계산) ----
        rem = needed & (role_act == ROLE_EXPLORE)
        n_now = torch.round(ob["k_n"] * N)
        for it in range(N):
            if not bool(rem.any()):
                break
            sel, _ = explore_front(group, rem, exp_mask, ed, ob["explore_size"], n_now,
                                   float(self.cfg["order_dist_alpha"]), ob["geo"][:, 2])
            bi, ii = sel.nonzero(as_tuple=True)
            rank["s1"] = torch.where(sel, torch.full_like(rank["s1"], it + 1), rank["s1"])
            gr = build(bi, ii, ROLE_EXPLORE)
            al_raw, h = run(self.task_net, gr)
            al = _masked(al_raw, torch.cat([no_s[bi, ii], exp_mask[bi, ii]], -1).float())   # S 전부 차단
            a = choose(al, "task", bi, ii)
            trace("s1", sel, gr, h, al)
            ex = (a - T).clamp(0, K - 1)
            task_logits = scatter(task_logits, bi, ii, al)
            task_act = scatter(task_act, bi, ii, a)
            s1 = s1 | sel
            slot_now = scatter(slot_now, bi, ii, ex)
            inc = Fn.one_hot(slot_now.clamp(0, K - 1), K).float() * sel[..., None].float()
            n_now = n_now + torch.einsum("bij,bjk->bik", group.float(), inc)
            goal = scatter(goal, bi, ii, ob["k_pos"][bi, ii, ex])
            rem = rem & ~sel

        if taken is not None:
            role_act, task_act = taken["role"].long(), taken["task"].long()

        # ---- critics ----
        gat = lambda x, a: torch.gather(x, 2, a[..., None]).squeeze(-1)
        ra_ = torch.where(task_act < T, task_act, torch.zeros_like(task_act)).clamp(0, T - 1)
        ea_ = torch.where(task_act >= T, task_act - T, cur.clamp(min=0)).clamp(0, K - 1)
        res_feat = torch.cat([torch.gather(ob["s_pos"], 2, ra_[..., None, None].expand(B, N, 1, 2)).squeeze(2) - pos,
                              torch.stack([gat(ob["s_tau"], ra_), gat(ob["s_held"], ra_), gat(rd, ra_), gat(rsl, ra_)], -1)], -1)
        exp_feat = torch.cat([torch.gather(ob["k_pos"], 2, ea_[..., None, None].expand(B, N, 1, 2)).squeeze(2) - pos,
                              torch.stack([gat(ob["k_size"], ea_), gat(ob["k_n"], ea_), gat(ed, ea_)], -1)], -1)
        team_act = torch.cat([Fn.one_hot(role_act.clamp(0, N_ROLES - 1), N_ROLES).float(), goal], dim=-1).detach()
        ckw = dict(self_feats=ob["role_self"][..., :3], tasks=ob["critic_tasks"], task_present=ob["critic_task_present"],
                   team=ob["critic_team"], team_act=team_act, glob=ob["critic_global"])
        va = self.critic_task(ob["state"], ob["local"], torch.cat([role_ctx, res_feat, exp_feat], -1), **ckw)
        raw = {"role": self.critic_role(ob["state"], ob["local"], role_ctx, **ckw),
               "rescue": va[..., 0], "explore": va[..., 1]}
        vn = self.value_norms()
        values = {h: (vn[h].denormalize(v) if vn[h] is not None else v) for h, v in raw.items()}
        return {"logits": {"role": role_logits, "task": task_logits},
                "actions": {"role": role_act, "task": task_act},
                "stage": {"s1": s1, "rescue": resc, "rank": rank},
                "values": values, "values_norm": raw}
