"""학습된 체크포인트를 GUI/평가에서 굴리기 위한 래퍼.

gui/grid_map.py (Model 버튼), main.py, evaluation/evaluate.py 가 .step(sim) 으로 쓴다: 모델이 그룹 순서대로 순차 디코딩한
dict(role, rescue, explore, explore_sub) 를 sim.step_explicit() 로 그대로 넣는다.
ray 가 없거나 복원 실패 시 예외를 던지므로 호출부에서 잡아 상태메시지로 처리한다.

행동 선택 방식 mode: "greedy" (기본, 세 헤드 argmax) / "sample" (학습 때와 같은 확률 정책). 이동은 항상 BFS 경로.

실행 환경 설정 두 가지:
    env_priority=True  (main.py / GUI Model 버튼): 호출자 env_config (main 전용 yaml) 를 그대로 쓴다. 관측 차원 · 네트워크
                       구조를 정하는 값 (structure_diff) 이 체크포인트와 다르면 어떤 키가 다른지 에러. 입력 정규화 분모
                       (NORM_KEYS) 가 다르면 경고만 (self.warnings).
    env_priority=False (evaluation 스크립트): 체크포인트 env_config 를 기본으로 호출자 값으로 덮어쓰되 SHAPE_KEYS 는 체크포인트 값.

학습 때와 다른 UAV 수 (N) 로 실행:
    모델 파라미터는 N 과 무관하다 (모든 텐서 shape 가 N 독립 — 에이전트 축은 per-agent 공유 모듈 / attention / pooling).
    그래서 체크포인트 config 로 원래 trainer 를 복원해 가중치를 꺼낸 뒤, 실행용 env_config (호출자 UAV 구성) 로
    새 trainer 를 만들어 가중치를 옮겨 싣는다. actor (ACTOR_PREFIXES) 는 strict 로 전부 일치해야 하고,
    실행에 안 쓰는 critic / value_norm 은 shape 가 다르면 건너뛴다.
"""
import glob
import os
import random

import numpy as np


# 관측 / 모델 shape (또는 관측 정규화 분모) 를 정하는 env 설정 — 실행 시에도 체크포인트 값을 따라야 한다
SHAPE_KEYS = ("max_targets", "s_max", "state_size", "local_channels", "global_downsample", "rows", "cols", "max_episode_len")
# 입력 정규화 분모 — 달라도 실행은 되지만 학습 때와 입력 스케일이 달라진다 (env_priority 에서 경고)
NORM_KEYS = ("max_episode_len", "d_max", "rows", "cols", "explore.size_ref")
ACTOR_PREFIXES = ("net.gnn.", "net.q_ctx.", "net.dec.", "net.stage.", "net.w_r.", "net.w_e.", "net.w_t.")
MODES = ("greedy", "sample")


def is_role_config(train_cfg):
    """계층 구조 (Role / Rescue / Explore) 체크포인트인가 — 예전 구조 (env: rescue) 체크포인트는 불러올 수 없다."""
    return train_cfg.get("env") == "rescue_role"


def runtime_env_config(train_env, caller_env=None):
    """실행용 env_config = 체크포인트 env_config 를 기본으로 호출자 값으로 덮어쓰되 SHAPE_KEYS 는 체크포인트 값.
    반환: (env_config, 경고 문자열 목록)."""
    cfg = dict(train_env)
    cfg.update(caller_env or {})
    warnings = []
    for k in SHAPE_KEYS:
        if k in train_env and cfg.get(k) != train_env[k]:
            warnings.append(f"{k}: {cfg.get(k)} -> {train_env[k]} (checkpoint)")
            cfg[k] = train_env[k]
    # 계층 구조: explore 노드 슬롯 수 (cluster.k_max) 도 shape 를 정한다
    tk = (train_env.get("cluster") or {}).get("k_max")
    if tk is not None and (cfg.get("cluster") or {}).get("k_max", tk) != tk:
        warnings.append(f"cluster.k_max: {cfg['cluster'].get('k_max')} -> {tk} (checkpoint)")
    if tk is not None:
        cfg["cluster"] = dict(cfg.get("cluster") or {}, k_max=tk)
    return cfg, warnings


def _get(cfg, key, default=None):
    """'a.b' 키 값 (없으면 default)."""
    for part in key.split(".")[:-1]:
        cfg = cfg.get(part) or {}
    return cfg.get(key.split(".")[-1], default)


def structure_diff(train_cfg, env_config, model_config=None):
    """관측 차원 · 네트워크 구조를 정하는 값 중 체크포인트와 실행 설정이 다른 것 [(키, 체크포인트 값, 실행 값)].
    env: max_targets, s_max, cluster.k_max, explore.n_sub_max.
    model (model_config 를 주면): activation, gnn (embed_dim / n_layers / knn_k / s_max / aggr), 헤드별 n_heads."""
    from env.role.explore import EXPLORE_DEFAULTS
    from env.role.simulation import CLUSTER_DEFAULTS
    from models.role.net import ROLE_HEADS, resolve_model_options
    te = train_cfg.get("env_config") or {}
    out = []

    def chk(key, a, b):
        if a != b:
            out.append((key, a, b))
    chk("max_targets", te.get("max_targets"), env_config.get("max_targets"))
    chk("s_max", te.get("s_max"), env_config.get("s_max"))
    chk("cluster.k_max", _get(te, "cluster.k_max", CLUSTER_DEFAULTS["k_max"]),
        _get(env_config, "cluster.k_max", CLUSTER_DEFAULTS["k_max"]))
    chk("explore.n_sub_max", _get(te, "explore.n_sub_max", EXPLORE_DEFAULTS["n_sub_max"]),
        _get(env_config, "explore.n_sub_max", EXPLORE_DEFAULTS["n_sub_max"]))
    if model_config is not None:
        ma = resolve_model_options((train_cfg.get("model") or {}).get("custom_model_config"))
        mb = resolve_model_options(model_config)
        chk("model.activation", ma["activation"], mb["activation"])
        for k in ma["gnn"]:
            chk(f"model.gnn.{k}", ma["gnn"][k], mb["gnn"].get(k))
        for h in ROLE_HEADS:
            chk(f"model.{h}.n_heads", ma[h]["n_heads"], mb[h]["n_heads"])
    return out


def norm_diff(train_env, env_config):
    """NORM_KEYS 중 다른 것 ['키: 체크포인트 -> 실행']."""
    return [f"{k}: {_get(train_env, k)} -> {_get(env_config, k)}" for k in NORM_KEYS
            if _get(train_env, k) != _get(env_config, k)]


def with_n_agents(env_config, n):
    """UAV 수만 n 으로 바꾼 env_config (센서 / 통신 범위는 첫 UAV 값 복제). n=None 이면 그대로."""
    cfg = dict(env_config)
    if n is not None:
        cfg["agents"] = [dict(cfg["agents"][0]) for _ in range(int(n))]
    return cfg


def restore_or_explain(trainer, checkpoint_path):
    """trainer.restore + 관측 / 모델 구조 불일치면 읽기 쉬운 에러. 예: 관측 / encoder 구조 (이종 그래프 GNN 도입 등) 가
    바뀐 뒤 이전 체크포인트를 불러오면 torch 의 size mismatch / missing key 또는 옵티마이저
    상태 크기 불일치가 나는데, 어떤 파라미터가 몇 -> 몇인지 (알 수 있으면) 와 함께 호환되지 않음을 알린다."""
    import re
    try:
        trainer.restore(checkpoint_path)
    except (RuntimeError, ValueError) as e:
        msg = str(e)
        found = re.findall(r"size mismatch for ([\w.]+): copying a param with shape torch.Size\(\[([\d, ]*)\]\) "
                           r"from checkpoint, the shape in current model is torch.Size\(\[([\d, ]*)\]\)", msg)
        structural = found or any(k in msg for k in ("Missing key", "Unexpected key", "doesn't match the size of optimizer"))
        if not structural:
            raise
        lines = "\n".join(f"    {k}: checkpoint [{a}] -> current [{b}]" for k, a, b in found[:6])
        more = f"\n    ... (+{len(found) - 6})" if len(found) > 6 else ""
        detail = (f" ({len(found)} 개 파라미터):\n{lines}{more}" if found else f": {msg.splitlines()[0][:200]}")
        raise RuntimeError(
            f"체크포인트와 현재 모델의 구조 / 입력 차원이 다릅니다{detail}\n체크포인트: {checkpoint_path}\n"
            "관측 / 네트워크 구성이 바뀐 뒤에 저장된 체크포인트가 아니면 로드할 수 없습니다 — 새로 학습하세요.\n"
            f"{RECENT_CHANGES}") from None


RECENT_CHANGES = ("    (최근 변경: 보상 재설계 (explore 커버리지 옵션 / rescue S - F / role 구조 · 만료 + 진동), 그룹 기준 EXPLORE 마스크,\n"
                  "     role g_exp 2 -> 3 (그룹 발견 비율 ψ^{f,G}); 크기 특징 로그 정규화 (size_ref);\n"
                  "     그 전: encoder = 세 head 공유 이종 그래프 GNN (CapAM encoder / Context / 특징 레지스트리 node_feats 제거,\n"
                  "     관측 = 그래프 노드 / 간선 표, rescue 행동 = S 노드 s_max), rescue / explore 보상에 연결성 C;\n"
                  "     그 전: 결정 순서를 모델이 계산 (관측 dec_rank / explore_seg_taken 제거, explore_size /\n"
                  "     explore_seg_size 추가, team_ctx 9 -> 8);\n"
                  "     그 전: 최종 입력 — 생존자 노드 4 [rel_pos, bfs_dist, slack], explore 노드 6 [rel_pos, bfs_dist, size,\n"
                  "     n_assigned, current_flag], Context ξ 9 (group_size / mean_neighbor_dist 제거), explore_kind 3 -> 1,\n"
                  "     role encoder 항상 공유 (role_encoder 옵션 제거), h_RET 선형 출력;\n"
                  "     그 전: 노드 특징 레지스트리 node_feats, encoder 간선 특징, explore stage 임베딩, role g_ret 제거)")


def check_compatible(train_cfg, checkpoint_path):
    """체크포인트 config 가 지금 코드의 모델 옵션으로 해석되는지 (제거된 옵션이면 읽기 쉬운 에러)."""
    from models.role.net import resolve_model_options
    try:
        resolve_model_options((train_cfg.get("model") or {}).get("custom_model_config"))
    except ValueError as e:
        raise RuntimeError(f"예전 구조의 체크포인트라 지금 코드와 호환되지 않습니다 — 새로 학습하세요.\n체크포인트: "
                           f"{checkpoint_path}\n원인: {e}\n{RECENT_CHANGES}") from None


def load_train_config(checkpoint_path):
    """체크포인트의 학습 config (params.json)."""
    import json
    pj = find_params_json(checkpoint_path)
    if pj is None:
        raise FileNotFoundError(f"params.json 을 찾지 못함: {checkpoint_path}")
    with open(pj) as f:
        return json.load(f)


def resolve_checkpoint(path):
    """checkpoint_XXXXXX 폴더면 안의 checkpoint-XX 파일 경로로 바꾼다."""
    if os.path.isdir(path):
        files = [f for f in glob.glob(os.path.join(path, "checkpoint-*"))
                 if not f.endswith(".tune_metadata")]
        if not files:
            raise FileNotFoundError(f"체크포인트 파일이 없음: {path}")
        path = files[0]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"체크포인트가 없음: {path}")
    return path


def find_params_json(checkpoint_path):
    """체크포인트의 학습 config(params.json) 위치. 저장 구조에 따라 한두 단계 위에 있다.
    marl_models/RolePPO/checkpoints/checkpoint_X/checkpoint-X  -> marl_models/RolePPO/params.json
    marl_models/best_model/<날짜>/checkpoint_X/checkpoint-X    -> marl_models/best_model/<날짜>/params.json"""
    d = os.path.dirname(os.path.abspath(checkpoint_path))
    for _ in range(3):
        d = os.path.dirname(d)
        pj = os.path.join(d, "params.json")
        if os.path.isfile(pj):
            return pj
    return None


def read_weights(checkpoint_path):
    """RLlib 체크포인트 파일에서 policy 가중치 {이름: numpy} 만 읽는다 (trainer 복원 없이)."""
    import pickle
    with open(checkpoint_path, "rb") as f:
        state = pickle.loads(pickle.load(f)["worker"])["state"]["default_policy"]
    return {k: v for k, v in state.items() if k.startswith("net.")}


def critic_only_mismatch(target, weights, actor_prefixes=ACTOR_PREFIXES):
    """actor 파라미터 이름 · shape 가 모두 같고 다른 건 critic / value_norm 뿐인가."""
    a_t = {k: tuple(v.shape) for k, v in target.items() if k.startswith(actor_prefixes)}
    a_w = {k: tuple(np.shape(v)) for k, v in weights.items() if k.startswith(actor_prefixes)}
    return a_t == a_w


def load_weights(model, weights, actor_prefixes=ACTOR_PREFIXES):
    """체크포인트 가중치(state_dict)를 다른 UAV 수로 만든 model 에 싣는다.
    actor (ACTOR_PREFIXES) 는 이름·shape 가 전부 일치해야 하며 (불일치 시 어떤 파라미터인지 에러),
    critic / value_norm 은 shape 가 다르거나 없으면 건너뛴다 (실행에 안 씀). 반환: 건너뛴 키 목록."""
    import torch
    target = model.state_dict()
    actor_t = {k for k in target if k.startswith(actor_prefixes)}
    actor_w = {k for k in weights if k.startswith(actor_prefixes)}
    if actor_t != actor_w:
        raise RuntimeError(f"actor 파라미터 이름 불일치: 모델에만 {sorted(actor_t - actor_w)[:5]}, "
                           f"체크포인트에만 {sorted(actor_w - actor_t)[:5]}")
    bad = [(k, tuple(weights[k].shape), tuple(target[k].shape)) for k in actor_t
           if tuple(weights[k].shape) != tuple(target[k].shape)]
    if bad:
        raise RuntimeError("actor 파라미터 shape 불일치 (체크포인트 -> 실행 모델): "
                           + ", ".join(f"{k} {a}->{b}" for k, a, b in bad[:5]))
    load, skipped = {}, []
    for k, v in weights.items():
        if k in target and tuple(v.shape) == tuple(target[k].shape):
            load[k] = torch.as_tensor(np.asarray(v.cpu() if hasattr(v, "cpu") else v)).to(target[k].device)
        else:
            skipped.append(k)
    missing = [k for k in target if k not in load]
    model.load_state_dict({**target, **load}, strict=True)
    return sorted(set(skipped) | set(missing))


class CheckpointPolicy:
    MODES = MODES

    def __init__(self, checkpoint_path, env_config=None, mode="greedy", seed=None, model_config=None, env_priority=False):
        """env_config: 실행 환경 설정 (호출자 sim 의 UAV 구성 / 맵 / 조난자 ...). UAV 수가 학습과 달라도 된다
        (self.trained_n -> self.running_n). env_priority=True 면 env_config 를 그대로 쓰고 구조가 다르면 에러
        (model_config = 호출자 custom_model_config, 주면 네트워크 구조도 비교), False 면 SHAPE_KEYS 를 체크포인트 값으로
        강제 (내역은 self.warnings)."""
        import ray

        from training.catalog import register_all

        train_cfg = load_train_config(checkpoint_path)
        if not is_role_config(train_cfg):
            raise RuntimeError(f"예전 구조 (env: {train_cfg.get('env')!r}) 체크포인트는 지원하지 않음 — RolePPO 체크포인트를 쓸 것: "
                               f"{checkpoint_path}")
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True, num_gpus=0,
                     include_dashboard=False)
        register_all()

        check_compatible(train_cfg, checkpoint_path)
        train_env = train_cfg.get("env_config", {})
        if env_priority:
            diff = structure_diff(train_cfg, env_config or {}, model_config)
            if diff:
                raise RuntimeError("체크포인트와 관측 차원 / 네트워크 구조가 다른 설정 (main 전용 yaml 을 체크포인트에 맞추거나 "
                                   "새로 학습하세요):\n" + "\n".join(f"    {k}: checkpoint {a} vs main {b}" for k, a, b in diff)
                                   + f"\n체크포인트: {checkpoint_path}")
            run_env, self.warnings = dict(env_config), norm_diff(train_env, env_config)
        else:
            run_env, self.warnings = runtime_env_config(train_env, env_config)
        self.env_priority = env_priority
        self.trained_n = len(train_env.get("agents", []))
        self.running_n = len(run_env.get("agents", []))
        self.env_config = run_env

        # 1) 체크포인트 config 그대로 원래 trainer 복원 -> 가중치. critic 입력 차원만 바뀐 체크포인트 (예: critic_team 열 정리 전)
        #    는 복원이 실패하므로 체크포인트 파일에서 가중치만 읽는다 (actor 는 아래 load_weights 가 전부 일치하는지 확인)
        src = self._make_trainer(train_cfg, train_env)
        try:
            restore_or_explain(src, checkpoint_path)
            weights = src.get_policy().model.state_dict()
        except RuntimeError as e:
            weights = read_weights(checkpoint_path)
            if not critic_only_mismatch(src.get_policy().model.state_dict(), weights):
                src.stop()
                raise e
            print("[checkpoint] critic 입력 차원이 지금 코드와 달라 actor 가중치만 불러옴 (실행에는 critic 을 쓰지 않음)")
        # 2) 실행용 env_config 로 trainer 를 만들고 가중치 이식 (UAV 수가 같아도 같은 경로 — 결과는 원래 trainer 와 동일)
        self.trainer = self._make_trainer(train_cfg, run_env)
        self.skipped = load_weights(self.trainer.get_policy().model, weights)
        self.last = None                                # 마지막 결정의 UAV 별 모델 출력 (GUI 표시용)
        src.stop()
        self.checkpoint_path = checkpoint_path
        self.set_mode(mode)
        if seed is not None:
            self.seed(seed)

    @staticmethod
    def _make_trainer(train_cfg, env_config):
        import copy
        from training.catalog import ROLE_ENV_NAME
        from training.role.ppo import RolePPOTrainer
        cfg = copy.deepcopy(train_cfg)
        cfg.pop("callbacks", None)     # params.json 에는 클래스가 문자열로 저장됨 -> 추론엔 필요 없으니 기본값 사용
        cfg["num_workers"] = 0
        cfg["num_envs_per_worker"] = 1
        cfg["explore"] = False
        cfg["num_gpus"] = 0            # 추론은 CPU (학습 때 num_gpus: 1 이어도)
        cfg["env_config"] = copy.deepcopy(env_config)
        # RLlib 1.3 TorchPolicy 는 num_gpus 와 상관없이 CUDA 가 보이면 모델을 GPU 에 올린다 -> 학습이 GPU 를 쓰는 중이면
        # CUDA out of memory. 추론 trainer 를 만드는 동안만 CUDA 를 안 보이게 해서 CPU 에 올린다
        import torch
        orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            return RolePPOTrainer(config=cfg, env=ROLE_ENV_NAME)
        finally:
            torch.cuda.is_available = orig

    def describe(self):
        """상태바 / 로그용: 'trained N=4 -> running N=5' (+ 강제된 설정 / 입력 정규화 분모 차이)."""
        msg = f"trained N={self.trained_n} -> running N={self.running_n}"
        if self.warnings:
            msg += ("  [input scale differs: " if self.env_priority else "  [forced: ") + "; ".join(self.warnings) + "]"
        return msg

    def set_mode(self, mode="greedy"):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} — {MODES}")
        self.mode = mode
        self.trainer.get_policy().model.greedy = mode == "greedy"     # 세 헤드 모두 greedy / 샘플링 (이동은 BFS)

    @staticmethod
    def seed(seed):
        """샘플링 재현용 (torch / numpy / random)."""
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def step(self, sim):
        """sim 을 이 정책으로 한 step 진행. 반환 (done, info)."""
        return sim.step_explicit(self.act(sim))

    def act(self, sim):
        """계층 구조: 모델이 그룹 순서대로 순차 디코딩한 dict(role, rescue, explore, explore_sub). UAV 별 출력은 self.last 에.
        self.last[i] = {role_probs (3,), rescue_top / explore_top / sub_top [(슬롯, 확률)] 상위 3, value_role,
                        s1 / s2 (이번 step explore 1 / 2 단계 결정 여부)}."""
        import torch
        obs = sim.rl_observation()
        action, _, extra = self.trainer.compute_action(obs, explore=True, full_fetch=True)
        out = {h: np.asarray(action[h]).reshape(-1) for h in ("role", "rescue", "explore", "explore_sub")}
        model = self.trainer.get_policy().model
        lo = model.last_out
        if lo is not None:
            with torch.no_grad():
                pr = {h: torch.softmax(lo["logits"][h][0], -1).cpu().numpy()
                      for h in ("role", "rescue", "explore", "explore_sub")}
                vr = lo["values"]["role"][0].cpu().numpy()
                st = {k: lo["stage"][k][0].cpu().numpy() for k in ("s1", "s2")}
            top = lambda p: [(int(k), float(p[k])) for k in np.argsort(-p)[:3] if p[k] > 1e-4]
            self.last = [{"role_probs": pr["role"][i], "rescue_top": top(pr["rescue"][i]),
                          "explore_top": top(pr["explore"][i]), "sub_top": top(pr["explore_sub"][i]),
                          "s1": bool(st["s1"][i]), "s2": bool(st["s2"][i]),
                          "value_role": float(vr[i])} for i in range(sim.n_agents)]
        return out
