"""RoleNet — 계층 구조 env (env/role/simulation.py) 용 Role / Rescue / Explore actor + 헤드별 critic (순수 torch).

[Encoder]  이종 그래프 GNN 하나 (models/role/gnn.py) 를 세 head 가 공유한다 (detach 없음 — 세 head loss gradient 가 모두 GNN 으로).
           결정하는 UAV i 마다 그래프 = i 의 통신 그룹 belief (사양 env/role/graph_obs.py). 관측의 절대 정규화 좌표를 i 기준 상대
           좌표로 바꾸고, 같은 이벤트 앞 차례 결정 (결정함, 역할, 목표, 맡겨짐, 배정 수, 목표 기준 연결) 을 반영해 결정마다 다시 만든다.
           GNN 은 그 차례에 결정하는 UAV 만 계산한다 (gather). 출력 h_self, S / K / D 임베딩.
[Query]    q_ctx = W_c [ h_self ‖ t / T ]                     (head 마다 W_c)
[Rescue]   후보 = S 노드 중 할당 가능 (rescue_mask, 같은 그룹 앞 차례가 고른 생존자 제외)
             Q = W_q [ q_ctx ‖ mean(후보 S) ]
[Explore]  1 단계 (영역 K):     Q1 = W_q [ q_ctx ‖ mean K    ‖ 0   ‖ e_stage1 ]
           2 단계 (구간 K_seg): 고른 영역의 구간으로 그래프를 다시 만들어 GNN 재계산
                                Q2 = W_q [ q_ctx ‖ mean Kseg ‖ h_c ‖ e_stage2 | e_reselect ]   (h_c = 1 단계 그래프의 고른 영역 임베딩)
             log π(c, s) = log π1(c) + log π2(s | c). 2 단계 후보가 1 개 이하면 생략 (log π2 = 0)
             합류 재선택 (stage-2-only): 기존 배정자가 있는 영역에 새 UAV 가 들어오면 기존 배정자도 구간만 다시 고른다 (e_reselect)
[Role]     역할 노드 3 개 (EXPLORE, RESCUE, RETURN)
             h_RES = W_r [ mean ‖ max 할당 가능 S ‖ g_res ],  h_EXP = W_e [ mean ‖ max 고를 수 있는 K ‖ g_exp ],  h_RET = W_t h_D
             Q = W_q [ q_ctx ‖ mean(H) ]
[Decoder]  head 마다 MHA glimpse + tanh clip pointer (models/capam.py CapAMDecoder) -> 마스크 -> softmax
[Critic x3] models/common/critic.py RichCentralCritic (헤드별 독립, actor 와 인코더 비공유) + ValueNorm x3
             extra_i = [role_self, role_group] (+ rescue: 고른 S 특징 / explore: 고른 K 특징 + 재선택 플래그)

순차 디코딩 — env _decide 와 같은 흐름 · 같은 차례 함수 (policies/decision_order.py, 한 대가 결정할 때마다 다시 계산):
  [1] role + rescue: role_front 차례 -> 역할 -> RESCUE 면 바로 생존자 (같은 그래프 계산 하나로 두 결정).
  [2] 1 단계: EXPLORE 를 고른 신규 UAV (영역 없음) 만 explore_front 차례. 영역 유지 UAV (explore_cur >= 0) 는 그대로 두고
      배정 수에 기존 배정자로 포함.
  [4] 2 단계: 이번 step 고른 영역마다 참여자 = 고른 UAV + 기존 배정자 (stage-2-only), kk = min(m, kcap) >= 2 면
      구간 = seg_*[f, offset(kk) : +kk]. seg_front 차례, 앞 차례가 고른 구간은 마스크 (다 차면 다시 열림).
학습 재생 (taken = SampleBatch.ACTIONS) 도 같은 루프 -> rollout 과 같은 logits. 모든 파라미터는 N (UAV 수) 과 무관.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from env.role.graph_obs import DIST_CLIP
from policies.decision_order import explore_front, role_front, role_sigma, seg_front
from models.capam import MASK_VALUE, CapAMDecoder
from models.role.gnn import HeteroGNN, masked_mean_max
from models.common.critic import RichCentralCritic, CRITIC_TASK_DIM, CRITIC_GLOBAL_DIM
from models.common.value_norm import ValueNorm

ROLE_EXPLORE, ROLE_RESCUE, ROLE_RETURN = 0, 1, 2
N_ROLES = 3
ROLE_HEADS = ("role", "rescue", "explore")
RES_FEAT_DIM = 6          # critic 용 고른 S 특징: rel_pos(2), (τ - t) / T, 맡겨짐, BFS / d_max, slack
EXP_FEAT_DIM = 5          # critic 용 고른 K 특징: rel_pos(2), 크기, 배정 수 / N, BFS / d_max

GNN_DEFAULTS = {"embed_dim": 128, "n_layers": 2, "knn_k": 6, "s_max": 16, "aggr": "mean_max"}
# head 별 기본값 (custom_model_config.role / rescue / explore 로 덮어씀)
HEAD_MODEL_DEFAULTS = {
    "n_heads": 8,                # decoder MHA 헤드 수
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


def resolve_model_options(options):
    """custom_model_config -> {'activation', 'gnn': {...}, 'role': {...}, 'rescue': {...}, 'explore': {...},
    'entropy_coeff_heads': {...}, 'loss': {...}}."""
    options = dict(options or {})
    bad = [k for k in REMOVED_TOP_KEYS if k in options]
    for h in ROLE_HEADS:
        bad += [f"{h}.{k}" for k in REMOVED_HEAD_KEYS if k in (options.get(h) or {})]
    if bad:
        raise ValueError(f"model {bad} 는 제거된 옵션 (encoder 는 세 head 공유 이종 그래프 GNN, 차원은 model.gnn) — 이전 구조의 "
                         f"config / 체크포인트입니다. 지금 구조와 호환되지 않으니 새로 학습하세요")
    out = copy.deepcopy(DEFAULT_ROLE_MODEL_OPTIONS)
    out.update({k: options[k] for k in DEFAULT_ROLE_MODEL_OPTIONS if k in options})
    out["gnn"] = dict(GNN_DEFAULTS, **(options.get("gnn") or {}))
    if out["gnn"]["aggr"] != "mean_max":
        raise ValueError(f"gnn.aggr={out['gnn']['aggr']!r} — mean_max 만 지원")
    for h in ROLE_HEADS:
        hc = copy.deepcopy(HEAD_MODEL_DEFAULTS)
        given = copy.deepcopy(options.get(h) or {})
        hc["critic"].update(given.pop("critic", {}) or {})
        hc.update(given)
        out[h] = hc
    out["entropy_coeff_heads"] = {h: float(out[h]["entropy_coeff"]) for h in ROLE_HEADS}   # loss 가 읽는다
    out["loss"] = dict(LOSS_DEFAULTS, **(options.get("loss") or {}))
    return out


AGENT_KEYS = ("local", "role_self", "role_group", "role_g_res", "role_g_exp", "role_mask", "role_needed",
              "s_ids", "s_pos", "s_tau", "s_held", "rescue_mask", "us_d", "us_slack", "ss_dist",
              "k_pos", "k_size", "k_n", "uk_d", "kk_dist", "explore_mask", "explore_cur", "explore_kcap",
              "explore_size", "explore_kind", "seg_pos", "seg_sz", "useg_d", "useg_goal",
              "explore_seg_mask", "explore_seg_size", "explore_seg_dist")
TEAM_KEYS = ("team_ctx", "team_cap", "team_comm", "team_target", "team_slot", "team_gpos", "team_has_goal",
             "team_depot_d", "depot_pos", "geo", "adj", "group", "reach", "time", "state",
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


def _mean(H, valid):
    v = valid.unsqueeze(-1).float()
    return (H * v).sum(-2) / v.sum(-2).clamp(min=1.0)


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


class RoleNet(nn.Module):
    def __init__(self, n_agents, s_max, k_max, local_size, local_channels, state_channels, dims, options=None, n_sub=4):
        """dims: {role_self, role_group, g_res, g_exp, critic_task, critic_team, critic_global}. n_sub = 2 단계 구간 슬롯 수."""
        super().__init__()
        cfg = resolve_model_options(options)
        self.cfg = cfg
        self.N, self.T, self.K, self.M = n_agents, s_max, k_max, n_sub
        if int(cfg["gnn"]["s_max"]) != s_max:
            raise RuntimeError(f"gnn.s_max={cfg['gnn']['s_max']} 와 관측 S 노드 수 {s_max} 가 다름 — env_config.s_max 를 맞추세요")
        act = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}[cfg["activation"]]
        D = int(cfg["gnn"]["embed_dim"])
        self.D, self.knn_k = D, int(cfg["gnn"]["knn_k"])
        self.gnn = HeteroGNN(D, cfg["gnn"]["n_layers"])
        hr, hs, he = cfg["role"], cfg["rescue"], cfg["explore"]
        self.q_ctx = nn.ModuleDict({h: nn.Linear(D + 1, D) for h in ROLE_HEADS})         # W_c (head 마다)
        self.dec = nn.ModuleDict({"role": CapAMDecoder(D, 2 * D, hr["n_heads"], hr["clip"]),
                                  "rescue": CapAMDecoder(D, 2 * D, hs["n_heads"], hs["clip"]),
                                  "explore": CapAMDecoder(D, 4 * D, he["n_heads"], he["clip"])})
        self.stage = nn.Embedding(3, D)                       # e_stage1, e_stage2, e_reselect (합류 stage-2-only)
        self.w_r = nn.Linear(2 * D + dims["g_res"], D)        # W_r
        self.w_e = nn.Linear(2 * D + dims["g_exp"], D)        # W_e
        self.w_t = nn.Linear(D, D)                            # W_t
        self.act = act()
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
        self.critic_explore = critic(he, role_ctx + EXP_FEAT_DIM + 1)     # + stage-2-only (재선택) 플래그
        self.debug_trace = None              # 리스트를 넣으면 순차 디코딩 차례별 그래프 / 임베딩을 기록 (테스트용)
        self.timing = None                   # dict 를 넣으면 GNN 호출 수 / 결정 수 / 시간을 더한다 (G 로그)
        for h in ROLE_HEADS:                 # 헤드별 ValueNorm (그 헤드 value_norm: true 일 때만)
            if cfg[h]["value_norm"]:
                setattr(self, f"value_norm_{h}", ValueNorm())

    def shared_encoder_parameters(self):
        """세 head 가 함께 학습하는 encoder (GNN) 파라미터 (gradient norm 측정용)."""
        return list(self.gnn.parameters())

    def value_norms(self):
        """{헤드: ValueNorm 또는 None}."""
        return {h: getattr(self, f"value_norm_{h}", None) for h in ROLE_HEADS}

    def module_groups(self):
        """파라미터 집계용: 공유 GNN / head 별 query · decoder / critic."""
        out = {"shared.gnn": [self.gnn],
               "role.decoder": [self.q_ctx["role"], self.dec["role"], self.w_r, self.w_e, self.w_t],
               "rescue.decoder": [self.q_ctx["rescue"], self.dec["rescue"]],
               "explore.decoder": [self.q_ctx["explore"], self.dec["explore"], self.stage]}
        for h in ROLE_HEADS:
            out[f"critic.{h}"] = [getattr(self, f"critic_{h}")]
        return out

    # ------------------------------------------------------------ 그래프

    def encode(self, g):
        """g = build 결과 -> h {종류: (G,n,D)}."""
        return self.gnn(g["x"], g["mask"], g["edge"], g["adj"])

    def query(self, head, h, g):
        h_self = h["U"][torch.arange(h["U"].shape[0], device=h["U"].device), g["ii"]]
        return self.q_ctx[head](torch.cat([h_self, g["t"]], -1))

    # ------------------------------------------------------------ forward

    def forward(self, ob, taken=None, greedy=False):
        """ob: stack_role_obs() 결과. taken: {'role', 'rescue', 'explore', 'explore_sub'} (B,N) long — 학습 재생 시 실제 행동.
        반환: logits {head: (B,N,K_h)} (결정 시점 마스크 적용, explore_sub 포함), actions {head: (B,N)},
        stage {'s1', 's2', 's2d', ...}, values / values_norm {head: (B,N)}."""
        B, N = ob["role_self"].shape[:2]
        T, K, M, dev = self.T, self.K, self.M, ob["role_self"].device
        D = self.D
        role_ctx = torch.cat([ob["role_self"], ob["role_group"]], dim=-1)
        cur_role = ob["team_ctx"][..., 0:3].argmax(-1)                       # 현재 역할
        group = ob["group"] > 0.5
        eye = torch.eye(N, dtype=torch.bool, device=dev)[None]
        needed = ob["role_needed"][..., 0] > 0.5                            # D_0 (이번 step role 재결정)
        sub_learned = ob["explore_kind"][..., 0] > 0.5                      # 2 단계 learned
        r_mask0, res_mask0, exp_mask = ob["role_mask"] > 0.5, ob["rescue_mask"] > 0.5, ob["explore_mask"] > 0.5
        ids = ob["s_ids"]
        s_valid = ids >= 0
        k_valid = ob["explore_kcap"] > 0.5                                  # 셀이 있는 영역 슬롯
        pos = ob["team_ctx"][..., 6:8]                                      # 절대 정규화 위치
        scale = ob["geo"][:, 0:2]                                           # (cols, rows) — 정규화 좌표 -> 셀
        comm = ob["team_comm"]
        rc = torch.minimum(comm[:, :, None], comm[:, None, :]).clamp(min=1e-6)   # 쌍별 R_c
        cap = ob["team_cap"]
        rd, rsl, ed = own(ob["us_d"]), own(ob["us_slack"]), own(ob["uk_d"])
        useg_own = own(ob["useg_d"].movedim(-1, 2))                         # (B,N,K,P): i 관측의 i 행
        ar_k = torch.arange(K, device=dev)

        # 순차 디코딩 상태 (팀 테이블 — 절대 좌표)
        role_act = cur_role.clone()
        res_act = torch.zeros(B, N, dtype=torch.long, device=dev)
        exp_act = torch.zeros(B, N, dtype=torch.long, device=dev)
        s1 = torch.zeros(B, N, dtype=torch.bool, device=dev)
        done = torch.zeros(B, N, dtype=torch.bool, device=dev)
        goal = ob["team_gpos"].clone()                                      # 목표 지점 g_l
        has_goal = ob["team_has_goal"] > 0.5
        tgt = ob["team_target"].clone()                                     # RESCUE 목표 생존자 id (-1 없음)
        slot_now = ob["team_slot"].round().long()                           # EXPLORE 영역 슬롯 (-1 없음)
        picked = torch.full((B, N), -1.0, device=dev)                       # 이번 step 고른 생존자 id
        exp_pick = torch.full((B, N), -1, dtype=torch.long, device=dev)     # 이번 step 1 단계로 고른 영역 슬롯
        keep_blob = torch.full((B, N), -1, dtype=torch.long, device=dev)    # 영역을 유지한 EXPLORE UAV 의 영역 슬롯

        role_logits = _masked(torch.zeros(B, N, N_ROLES, device=dev), r_mask0.float())
        res_logits = _masked(torch.zeros(B, N, T, device=dev), res_mask0.float())
        exp_logits = _masked(torch.zeros(B, N, K, device=dev), exp_mask.float())

        def choose(logits, head, bi, ii):
            if taken is not None:
                return taken[head][bi, ii].long().clamp(0, logits.shape[-1] - 1)
            if greedy:
                return logits.argmax(-1)
            return torch.distributions.Categorical(logits=logits.detach()).sample()

        def mates_done():                                                   # [b,i,j] 같은 그룹 앞 차례가 결정함 (j != i)
            return group & ~eye & done[:, None, :]

        def taken_victims():                                                # (B,N,T) 같은 그룹이 이번 step 고른 생존자
            same = (ids[:, :, None, :] == picked[:, None, :, None]) & (picked[:, None, :, None] >= 0)
            return (same & group[..., None] & done[:, None, :, None]).any(dim=2)

        def n_extra():                                                      # (B,N,K) 같은 그룹 앞 차례가 이번 step 고른 / 유지한 수
            pick = torch.where(keep_blob >= 0, keep_blob, exp_pick)
            hit = (pick[:, None, :, None] == ar_k) & mates_done()[..., None]
            return hit.float().sum(2)

        def build(bi, ii, seg=None):
            """(bi, ii) 차례 UAV 들의 그래프 (G 개). seg = 2 단계 구간 그래프 {f, idx, mask, n, sub, done2, same}."""
            G = bi.shape[0]
            ar = torch.arange(G, device=dev)
            A = lambda k: ob[k][bi, ii]
            p = pos[bi]                                                     # (G,N,2)
            pi = p[ar, ii]
            mU = group[bi, ii]                                              # 그룹 밖 UAV 없음
            gl = goal[bi]
            hg = has_goal[bi].float()
            dn = done[bi]
            xU = torch.cat([p - pi[:, None], cap[bi][..., None], Fn.one_hot(role_act[bi], N_ROLES).float(), hg[..., None],
                            (gl - pi[:, None]) * hg[..., None], dn.float()[..., None],
                            (torch.arange(N, device=dev)[None] == ii[:, None]).float()[..., None]], -1)
            sc = scale[bi][:, None, None, :]
            rcg = rc[bi]
            d_cur = ((p[:, :, None] - p[:, None]) * sc).norm(dim=-1) / rcg
            d_goal = ((gl[:, :, None] - gl[:, None]) * sc).norm(dim=-1) / rcg
            eUU = torch.stack([d_cur.clamp(max=DIST_CLIP), d_goal.clamp(max=DIST_CLIP), ob["adj"][bi],
                               (d_goal <= 1.0).float()], -1)
            aUU = mU[:, :, None] & mU[:, None, :] & ~eye
            # S
            sid = A("s_ids")
            mS = sid >= 0
            held = (A("s_held") > 0.5) | taken_victims()[bi, ii]
            xS = torch.cat([A("s_pos") - pi[:, None], A("s_tau")[..., None], held.float()[..., None]], -1)
            sl = A("us_slack")
            eUS = torch.stack([A("us_d"), sl, (sl >= 0).float(),
                               ((tgt[bi][:, :, None] == sid[:, None, :]) & mS[:, None, :]).float()], -1)
            aUS = mU[:, :, None] & mS[:, None, :]
            aSS = knn_adj(A("ss_dist"), mS, self.knn_k)
            # K (1 단계: 영역 / 2 단계: 고른 영역의 구간)
            if seg is None:
                mK = A("explore_kcap") > 0.5
                xK = torch.cat([A("k_pos") - pi[:, None], A("k_size")[..., None],
                                (A("k_n") + n_extra()[bi, ii] / N)[..., None]], -1)
                asg = (slot_now[bi][:, :, None] == ar_k) & (slot_now[bi][:, :, None] >= 0)
                eUK = torch.stack([A("uk_d"), asg.float(), torch.zeros_like(asg, dtype=torch.float)], -1)
                eKK = A("kk_dist")
                aKK = knn_adj(eKK, mK, self.knn_k)
            else:
                f, idx, mK = seg["f"], seg["idx"], seg["mask"]
                gk = lambda k: ob[k][bi, ii, f]                             # (G,P,...)
                gi = lambda x: torch.gather(x, 1, idx.view(G, M, *([1] * (x.dim() - 2))).expand(G, M, *x.shape[2:]))
                xK = torch.cat([gi(gk("seg_pos")) - pi[:, None], gi(gk("seg_sz"))[..., None],
                                (seg["n"] / N)[..., None]], -1)
                asg = seg["asg"]                                            # (G,N,M) l 이 이번 이벤트에 구간 s 를 고름
                gin = torch.where(seg["done2"][..., None], asg, gi(gk("useg_goal")).transpose(1, 2) > 0.5)
                eUK = torch.stack([gi(gk("useg_d")).transpose(1, 2), asg.float(), gin.float()], -1)
                eKK = gi(gk("explore_seg_dist"))[..., :M]
                aKK = mK[:, :, None] & mK[:, None, :] & ~torch.eye(M, dtype=torch.bool, device=dev)[None]
            aUK = mU[:, :, None] & mK[:, None, :]
            xD = (ob["depot_pos"][bi] - pi)[:, None]
            mD = torch.ones(G, 1, dtype=torch.bool, device=dev)
            eUD = ob["team_depot_d"][bi][:, :, None, None]
            aUD = mU[:, :, None]
            return {"bi": bi, "ii": ii, "t": ob["time"][bi],
                    "x": {"U": xU, "S": xS, "K": xK, "D": xD}, "mask": {"U": mU, "S": mS, "K": mK, "D": mD},
                    "edge": {"US": eUS, "UK": eUK, "UU": eUU, "UD": eUD, "SS": A("ss_dist")[..., None],
                             "KK": eKK[..., None]},
                    "adj": {"US": aUS, "UK": aUK, "UU": aUU, "UD": aUD, "SS": aSS, "KK": aKK}}

        def run_gnn(g):
            if self.timing is not None:
                import time
                t0 = time.perf_counter()
                h = self.encode(g)
                self.timing["gnn_s"] = self.timing.get("gnn_s", 0.0) + time.perf_counter() - t0
                self.timing["gnn_calls"] = self.timing.get("gnn_calls", 0) + 1
                self.timing["gnn_rows"] = self.timing.get("gnn_rows", 0) + int(g["ii"].shape[0])
                return h
            return self.encode(g)

        def scatter(base, bi, ii, val):
            return base.index_put((bi, ii), val)

        rank = {k: torch.zeros(B, N, dtype=torch.long, device=dev) for k in ("role", "s1", "s2")}   # 실제 차례 (1, 2, ...)
        # ---- [1] role + rescue: decision_order.role_front 로 그룹마다 다음 차례 (한 대마다 재계산) ----
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
            g = build(bi, ii)
            h = run_gnn(g)
            rm = res_mask[bi, ii]
            q = self.query("role", h, g)
            h_res = self.act(self.w_r(torch.cat([masked_mean_max(h["S"][:, None], rm[:, None])[:, 0],
                                                 ob["role_g_res"][bi, ii]], -1)))
            h_exp = self.act(self.w_e(torch.cat([masked_mean_max(h["K"][:, None], exp_mask[bi, ii][:, None])[:, 0],
                                                 ob["role_g_exp"][bi, ii]], -1)))
            h_ret = self.w_t(h["D"][:, 0])
            H = torch.stack([h_exp, h_res, h_ret], dim=1)                  # (G,3,D)
            ones3 = torch.ones(H.shape[:2], dtype=torch.bool, device=dev)
            lg = _masked(self.dec["role"](torch.cat([q, H.mean(1)], -1), H, ones3), r_mask[bi, ii].float())
            ra = choose(lg, "role", bi, ii)
            qr = self.query("rescue", h, g)
            rl = _masked(self.dec["rescue"](torch.cat([qr, _mean(h["S"], rm)], -1), h["S"], g["mask"]["S"]), rm.float())
            rs = choose(rl, "rescue", bi, ii)
            if self.debug_trace is not None:
                self.debug_trace.append({"stage": "role", "sel": sel.clone(), "graph": g, "h": h, "res_mask": res_mask.clone()})
            on = ra == ROLE_RESCUE
            role_logits = scatter(role_logits, bi, ii, lg)
            role_act = scatter(role_act, bi, ii, ra)
            res_logits = scatter(res_logits, bi[on], ii[on], rl[on])
            res_act = scatter(res_act, bi[on], ii[on], rs[on])
            rsc = rs.clamp(0, T - 1)
            vid = ids[bi, ii, rsc]
            picked = scatter(picked, bi[on], ii[on], vid[on])
            tgt = scatter(tgt, bi, ii, torch.where(on, vid, torch.full_like(vid, -1.0)))
            slot_now = scatter(slot_now, bi[ra != ROLE_EXPLORE], ii[ra != ROLE_EXPLORE],
                               torch.full_like(ii[ra != ROLE_EXPLORE], -1))
            ng = torch.where(on[:, None], ob["s_pos"][bi, ii, rsc],
                             torch.where((ra == ROLE_RETURN)[:, None], ob["depot_pos"][bi], goal[bi, ii]))
            goal = scatter(goal, bi, ii, ng)
            has_goal = scatter(has_goal, bi, ii, has_goal[bi, ii] | (ra != ROLE_EXPLORE))
            done = done | sel
            rem = rem & ~sel

        # ---- [2] explore 1 단계: 신규 EXPLORE UAV 만 explore_front 차례 (실제 선택으로 n_k 갱신, 한 대마다 재계산) ----
        #      유지 UAV (직전에도 EXPLORE, 영역 유효 = explore_cur >= 0) 는 영역 · 구간 그대로, n_k 에 기존 배정자로
        cur = ob["explore_cur"][..., 0].round().long()
        retained = needed & (role_act == ROLE_EXPLORE) & (cur >= 0)
        rem = needed & (role_act == ROLE_EXPLORE) & ~retained
        ret_hot = Fn.one_hot(cur.clamp(0, K - 1), K).float() * retained[..., None].float()   # [b,j,k]
        n_now = torch.round(ob["k_n"] * N) + torch.einsum("bij,bjk->bik", group.float(), ret_hot)
        keep_blob = torch.where(retained, cur, keep_blob)
        e1 = self.stage.weight[0]
        for it in range(N):
            if not bool(rem.any()):
                break
            sel, _ = explore_front(group, rem, exp_mask, ed, ob["explore_size"], n_now)
            bi, ii = sel.nonzero(as_tuple=True)
            rank["s1"] = torch.where(sel, torch.full_like(rank["s1"], it + 1), rank["s1"])
            g = build(bi, ii)
            h = run_gnn(g)
            q = self.query("explore", h, g)
            mK = g["mask"]["K"]
            hk = _mean(h["K"], mK)
            el = _masked(self.dec["explore"](torch.cat([q, hk, torch.zeros_like(hk), e1.expand_as(hk)], -1), h["K"], mK),
                         exp_mask[bi, ii].float())
            ex = choose(el, "explore", bi, ii)
            if self.debug_trace is not None:
                self.debug_trace.append({"stage": "s1", "sel": sel.clone(), "graph": g, "h": h})
            exp_logits = scatter(exp_logits, bi, ii, el)
            exp_act = scatter(exp_act, bi, ii, ex)
            s1 = s1 | sel
            exp_pick = scatter(exp_pick, bi, ii, ex)
            slot_now = scatter(slot_now, bi, ii, ex)
            inc = Fn.one_hot(torch.where(sel, exp_pick, torch.zeros_like(exp_pick)).clamp(0, K - 1), K).float() \
                * sel[..., None].float()
            n_now = n_now + torch.einsum("bij,bjk->bik", group.float(), inc)
            goal = scatter(goal, bi, ii, ob["k_pos"][bi, ii, ex.clamp(0, K - 1)])
            has_goal = scatter(has_goal, bi, ii, torch.ones_like(ii, dtype=torch.bool))
            rem = rem & ~sel

        # ---- [3][4] 분할 + 2 단계 (env _stage2_all 과 같은 규칙) ----
        keeper = (role_act == ROLE_EXPLORE) & (cur >= 0) & ~s1               # 기존 배정자 (결정하지 않은 UAV + 영역 유지 UAV)
        final = torch.where(s1, exp_act, torch.where(keeper, cur, torch.full_like(cur, -1)))
        same_f = group & (final[:, :, None] == final[:, None, :]) & (final[:, :, None] >= 0)        # [b,i,j]
        join = (same_f & s1[:, None, :]).any(-1) & (final >= 0)
        reselect = keeper & join
        m = same_f.sum(-1)
        fk = final.clamp(0, K - 1)
        kcap = torch.gather(ob["explore_kcap"], 2, fk[..., None]).squeeze(-1).round().long()
        kk = torch.minimum(m, kcap)
        part = (s1 | reselect) & join & (kk >= 2)
        P = ob["seg_pos"].shape[3]
        off = (kk.clamp(min=2) * (kk.clamp(min=2) - 1)) // 2 - 1            # offset(kk) = kk (kk - 1) / 2 - 1
        slot = torch.arange(M, device=dev)
        idx = (off[..., None] + slot).clamp(0, P - 1)                        # (B,N,M)
        g2 = lambda key: torch.gather(torch.gather(ob[key], 2, fk[..., None, None].expand(B, N, 1, P)).squeeze(2), 2, idx)
        mask2 = (g2("explore_seg_mask") > 0.5) & (slot < kk[..., None])
        size2 = g2("explore_seg_size")
        d2 = torch.gather(torch.gather(useg_own, 2, fk[..., None, None].expand(B, N, 1, P)).squeeze(2), 2, idx)
        same_p = same_f & part[:, None, :] & part[:, :, None]
        sub_logits = _masked(torch.zeros(B, N, M, device=dev), mask2.float())
        sub_act = torch.zeros(B, N, dtype=torch.long, device=dev)
        done2 = torch.zeros(B, N, dtype=torch.bool, device=dev)
        s2d = torch.zeros(B, N, dtype=torch.bool, device=dev)
        hc = torch.zeros(B, N, D, device=dev)
        if bool(part.any()):                                                 # h_c = 1 단계 최종 그래프의 고른 영역 임베딩
            bi, ii = part.nonzero(as_tuple=True)
            g = build(bi, ii)
            h = run_gnn(g)
            hc = scatter(hc, bi, ii, h["K"][torch.arange(bi.shape[0], device=dev), fk[bi, ii]])
        rem = part.clone()
        for it in range(N):
            if not bool(rem.any()):
                break
            picks = (same_f & done2[:, None, :])[..., None] & (sub_act[:, None, :, None] == slot)   # [b,i,j,s] 앞 차례 선택
            tk = picks.any(2)
            m2 = mask2 & ~tk
            m2 = torch.where(m2.any(-1, keepdim=True), m2, mask2)
            sel, _ = seg_front(same_p, rem, m2, d2, size2)
            bi, ii = sel.nonzero(as_tuple=True)
            rank["s2"] = torch.where(sel, torch.full_like(rank["s2"], it + 1), rank["s2"])
            cnt = m2.sum(-1)
            seg = {"f": fk[bi, ii], "idx": idx[bi, ii], "mask": mask2[bi, ii], "n": picks.float().sum(2)[bi, ii],
                   "asg": picks[bi, ii], "done2": done2[bi] & same_f[bi, ii]}          # asg (G,N,M): 팀원 j 가 구간 s 를 고름
            g = build(bi, ii, seg)
            h = run_gnn(g)
            q = self.query("explore", h, g)
            mK = g["mask"]["K"]
            e2 = torch.where(reselect[bi, ii][:, None], self.stage.weight[2].expand(bi.shape[0], D),
                             self.stage.weight[1].expand(bi.shape[0], D))
            lg = _masked(self.dec["explore"](torch.cat([q, _mean(h["K"], mK), hc[bi, ii], e2], -1), h["K"], mK),
                         m2[bi, ii].float())
            c2 = cnt[bi, ii] >= 2
            sa = torch.where(c2, choose(lg, "explore_sub", bi, ii), m2[bi, ii].float().argmax(-1))
            if self.debug_trace is not None:
                self.debug_trace.append({"stage": "s2", "sel": sel.clone(), "graph": g, "h": h})
            sub_logits = scatter(sub_logits, bi[c2], ii[c2], lg[c2])
            sub_act = scatter(sub_act, bi, ii, sa)
            goal = scatter(goal, bi, ii, ob["seg_pos"][bi, ii, fk[bi, ii], idx[bi, ii, sa.clamp(0, M - 1)]])
            s2d = scatter(s2d, bi[c2], ii[c2], torch.ones_like(ii[c2], dtype=torch.bool))
            done2 = done2 | sel
            rem = rem & ~sel
        s2 = s2d & sub_learned
        rs_only = reselect & s2                                               # stage-2-only sample (s1 = 0, s2 = 1)
        exp_act = torch.where(s1, exp_act, fk)

        if taken is not None:
            role_act, res_act = taken["role"].long(), taken["rescue"].long()
            exp_act, sub_act = taken["explore"].long(), taken["explore_sub"].long()

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
            "explore": self.critic_explore(ob["state"], ob["local"],
                                           torch.cat([role_ctx, exp_feat, rs_only.float()[..., None]], -1), **ckw),
        }
        vn = self.value_norms()
        values = {h: (vn[h].denormalize(v) if vn[h] is not None else v) for h, v in raw.items()}
        return {"logits": {"role": role_logits, "rescue": res_logits, "explore": exp_logits, "explore_sub": sub_logits},
                "actions": {"role": role_act, "rescue": res_act, "explore": exp_act, "explore_sub": sub_act},
                "stage": {"s1": s1, "s2": s2, "s2d": s2d, "part": part, "final": final, "m": m, "kk": kk,
                          "reselect": rs_only, "join": join, "rank": rank},
                "values": values, "values_norm": raw}
