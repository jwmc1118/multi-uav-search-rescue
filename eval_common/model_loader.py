"""모델 경로 하나로 구성요소별 모드 + 가중치를 정하는 공통 로더 — 학습 (구성요소 path 로드) · GUI · main_rule · main_scale 이
모두 이 함수로 모델을 읽는다.

구성요소 role / rescue / explore, 모드:
    rule   규칙 정책 (네트워크 호출 · 학습 없음)
    learn  학습 (path 가 있으면 그 가중치에서 시작) — 저장된 모델에서는 그 run 이 학습한 구성요소

    spec = load_model(path)        path = best 모델 폴더 (best_model/<조합>/model_vXX), 학습 체크포인트 폴더
                                   (<run>/checkpoints/checkpoint_XXXXXX 또는 checkpoints/best) 또는 checkpoint-* 파일
    spec.components                {"role" | "rescue" | "explore": {"mode": rule | learn, "path": 시작 가중치 모델 | None}}
    spec.modes                     {구성요소: 모드}
    spec.weights                   {파라미터 이름: numpy}
    spec.combo                     조합 이름 "<role 모드>_<rescue 모드>_<explore 모드>" (예: learn_learn_rule)
    spec.describe()                "role learn(from best_model/learn_rule_rule/model_v01) · rescue learn · explore rule"
    inspect_model(path)            가중치 없이 같은 판정 (목록 · 요약용)
    env_with_components(env, m)    env_config 에 구성요소 모드 ({구성요소: 모드}) 를 넣은 사본 (env 가 rule 인 결정을 규칙 정책으로)
    combo_name(modes)              {구성요소: 모드} -> 조합 이름 (run 폴더 marl_models/<조합>_<시각>, best_model/<조합>/model_vXX)
    combo_of_name(name)            폴더 이름 (<조합> 또는 <조합>_<YYYYMMDD_HHMMSS>[_smoke]) -> {구성요소: 모드} 또는 None
    load_prefixes(model, weights, prefixes)  접두어로 고른 파라미터를 strict 로드 (이름 · shape 불일치는 에러)
    actor_prefixes(modes)          실행에 쓰는 actor 네트워크 접두어 (rule 이 아닌 구성요소)
네트워크 단위 (접두어 PREFIXES): role · rescue · explore — 로드 · 고정 · 시작 가중치 모두.
구성요소 판정 순서: 모델 폴더의 meta.json -> run 폴더의 run_meta.json -> 폴더 이름 (best_model/<조합>/model_vXX,
marl_models/<조합>_<시각>) -> 학습 config (params.json) 에 저장된 구성요소 (env_config.components, 통합 task 네트워크 시절 모델은
env_config.policy_mode {role, task: learned | rule}) -> 전부 learn.
rescue / explore 를 하나로 쓰던 예전 통합 task 네트워크 (net.task_net.*) 로 학습된 모델은 rescue / explore 가 rule 이 아니면
로드 불가 (UnifiedTaskModel, GUI 목록에서는 "structure mismatch"). rescue / explore 가 rule 인 모델 (예: learn_rule_rule/model_v01) 은
role 부분만 쓰므로 로드된다.
"""
import json
import os
import re

from eval_common.checkpoint_policy import find_params_json, read_weights, resolve_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPONENT_KEYS = ("role", "rescue", "explore")
MODES = ("rule", "learn")
COMBO_RE = re.compile(r"^(rule|learn)_(rule|learn)_(rule|learn)(?:_\d{8}_\d{6})?(?:_smoke)?$")
ROLE_PREFIXES = ("net.role_net.", "net.critic_role.", "net.value_norm_role.")    # role 부분 (actor · critic · ValueNorm)
RESCUE_PREFIXES = ("net.rescue_net.", "net.critic_rescue.", "net.value_norm_rescue.")      # rescue 네트워크
EXPLORE_PREFIXES = ("net.explore_net.", "net.critic_explore.", "net.value_norm_explore.")   # explore 네트워크
PREFIXES = {"role": ROLE_PREFIXES, "rescue": RESCUE_PREFIXES, "explore": EXPLORE_PREFIXES}   # 구성요소 -> 접두어
ACTOR = {"role": "net.role_net.", "rescue": "net.rescue_net.", "explore": "net.explore_net."}
UNIFIED_TASK = "net.task_net."                                                   # 예전 통합 task 네트워크 (로드 불가)
META, RUN_META = "meta.json", "run_meta.json"
_POLICY_MODE = {"rule": "rule", "learned": "learn"}                            # params.json env_config.policy_mode 값 (통합 task 시절)


class UnifiedTaskModel(RuntimeError):
    """rescue / explore 분리 전 통합 task 네트워크로 학습된 모델 (rescue / explore 가 rule 이 아님)."""


def check_modes(m):
    """{구성요소: 모드} 검사 (키 = COMPONENT_KEYS, 값 = MODES)."""
    m = dict(m)
    if set(m) != set(COMPONENT_KEYS) or any(v not in MODES for v in m.values()):
        raise ValueError(f"components {m} — 키 {COMPONENT_KEYS}, 값 {MODES}")
    return m


def modes_of(components):
    """{구성요소: {mode, path}} 또는 {구성요소: 모드} -> {구성요소: 모드}."""
    return check_modes({k: (v["mode"] if isinstance(v, dict) else v) for k, v in components.items()})


def combo_name(modes):
    """{구성요소: 모드} (또는 {구성요소: {mode, path}}) -> "<role>_<rescue>_<explore>" (learn 에 path 가 있어도 learn)."""
    m = modes_of(modes)
    return "_".join(m[k] for k in COMPONENT_KEYS)


def combo_of_name(name):
    """폴더 이름 -> {구성요소: 모드} (조합 이름 · <조합>_<YYYYMMDD_HHMMSS>[_smoke] 가 아니면 None)."""
    mt = COMBO_RE.match(name or "")
    return dict(zip(COMPONENT_KEYS, mt.groups())) if mt else None


def combo_of_path(path, levels=4):
    """path 와 그 위 폴더들 (levels 단계) 중 처음 나오는 조합 이름 폴더의 {구성요소: 모드} (없으면 None)."""
    d = os.path.abspath(path)
    for _ in range(levels):
        m = combo_of_name(os.path.basename(d))
        if m:
            return m
        d = os.path.dirname(d)
    return None


def modes_from_params(train_cfg):
    """학습 config (params.json) 에 저장된 구성요소 모드 — env_config.components 또는 통합 task 시절의
    env_config.policy_mode {role, task: learned | rule}. 판정 못 하면 None."""
    env = (train_cfg or {}).get("env_config") or {}
    c = env.get("components")
    if isinstance(c, dict) and set(c) == set(COMPONENT_KEYS):
        return modes_of(c)
    pm = env.get("policy_mode")
    if isinstance(pm, dict) and pm.get("role") in _POLICY_MODE and pm.get("task") in _POLICY_MODE:
        return {"role": _POLICY_MODE[pm["role"]], "rescue": _POLICY_MODE[pm["task"]], "explore": _POLICY_MODE[pm["task"]]}
    return None


def describe_components(components):
    """{구성요소: {mode, path}} -> "role learn(from best_model/learn_rule_rule/model_v01) · rescue learn · explore rule"
    (learn 인데 path 에서 시작했으면 "learn(from ...)")."""
    out = []
    for k in COMPONENT_KEYS:
        c = components[k]
        p = c.get("path")
        tag = f"(from {p.replace('marl_models/', '')})" if p and c["mode"] == "learn" else ""
        out.append(f"{k} {c['mode']}{tag}")
    return " · ".join(out)


def components_from_meta(meta, path=None, train_cfg=None):
    """meta.json / run_meta.json -> {구성요소: {mode, path}}. 구성요소 정보가 없으면 path 의 폴더 이름
    (combo_of_path) -> train_cfg (params.json) 에 저장된 구성요소 -> 전부 learn."""
    comp = (meta or {}).get("components")
    if isinstance(comp, dict) and set(comp) == set(COMPONENT_KEYS) and all(isinstance(v, dict) for v in comp.values()):
        out = {k: {"mode": comp[k]["mode"], "path": comp[k].get("path")} for k in COMPONENT_KEYS}
    else:
        m = (combo_of_path(path) if path else None) or modes_from_params(train_cfg) or {k: "learn" for k in COMPONENT_KEYS}
        out = {k: {"mode": m[k], "path": None} for k in COMPONENT_KEYS}
    check_modes({k: v["mode"] for k, v in out.items()})
    return out


def actor_prefixes(modes):
    """실행에 쓰는 actor 접두어 (rule 이 아닌 구성요소) — 가중치 로드는 이 부분만 strict."""
    return tuple(ACTOR[k] for k in COMPONENT_KEYS if modes[k] != "rule")


def env_with_components(env_config, modes):
    """env_config 사본에 구성요소 모드 ({구성요소: 모드}, env/role/simulation.py components) 를 넣는다."""
    return dict(env_config, components=check_modes(modes))


def rel(path):
    """레포 기준 상대 경로 (레포 밖이면 절대 경로)."""
    p = os.path.abspath(path)
    return os.path.relpath(p, REPO_ROOT) if p.startswith(REPO_ROOT + os.sep) else p


def abs_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_checkpoint(path):
    """모델 경로 -> checkpoint-* 파일. 폴더면 best_info.json 의 checkpoint, 없으면 가장 큰 checkpoint_XXXXXX, 없으면 그 폴더."""
    path = abs_path(path)
    if os.path.isdir(path) and not os.path.basename(path).startswith("checkpoint_"):
        info = _read_json(os.path.join(path, "best_info.json")) or {}
        named = os.path.join(path, str(info.get("checkpoint") or ""))
        dirs = sorted((d for d in os.listdir(path) if d.startswith("checkpoint_") and os.path.isdir(os.path.join(path, d))),
                      key=lambda d: int(re.sub(r"\D", "", d) or 0))
        if info.get("checkpoint") and os.path.isdir(named):
            path = named
        elif dirs:
            path = os.path.join(path, dirs[-1])
    return resolve_checkpoint(path)


def find_meta(checkpoint):
    """checkpoint-* 파일에서 위로 올라가며 meta.json (모델 폴더) -> run_meta.json (run 폴더). 반환 (meta, 종류, 폴더) 또는
    (None, None, None)."""
    d = os.path.dirname(os.path.abspath(checkpoint))
    for name in (META, RUN_META):
        cur = d
        for _ in range(4):
            m = _read_json(os.path.join(cur, name))
            if m is not None:
                return m, name, cur
            cur = os.path.dirname(cur)
    return None, None, None


def iteration_of(checkpoint):
    m = re.search(r"checkpoint[-_]0*(\d+)", os.path.basename(os.path.dirname(checkpoint)) or "")
    return int(m.group(1)) if m else None


class ModelSpec:
    def __init__(self, path, checkpoint, components, meta, meta_kind, weights, train_cfg):
        self.path, self.checkpoint, self.components = path, checkpoint, components
        self.modes = {k: v["mode"] for k, v in components.items()}
        self.combo = combo_name(self.modes)
        self.meta, self.meta_kind, self.weights, self.train_cfg = meta, meta_kind, weights, train_cfg
        self.iteration = iteration_of(checkpoint)

    def describe(self):
        return describe_components(self.components)


def inspect_model(path, with_weights=False, use=COMPONENT_KEYS):
    """모델 경로 -> ModelSpec. with_weights=False 면 가중치를 읽지 않는다 (조합 요약 · 목록용). 구성요소: meta.json -> run_meta.json
    -> 폴더 이름 -> params.json (components_from_meta). use = 쓸 구성요소 (학습의 path 로드는 그 구성요소 하나) — 그중 rescue / explore 가
    rule 이 아닌데 통합 task 네트워크 구조면 UnifiedTaskModel (role 만 쓰면 role 부분은 구조가 같으므로 통과)."""
    ck = find_checkpoint(path)
    meta, kind, _ = find_meta(ck)
    pj = find_params_json(ck)
    train_cfg = _read_json(pj) if pj else None
    if train_cfg is None:
        raise FileNotFoundError(f"params.json (학습 config) 을 찾지 못함: {ck}")
    comp = components_from_meta(meta, ck, train_cfg)
    cmc = (train_cfg.get("model") or {}).get("custom_model_config") or {}
    task_net = any(comp[k]["mode"] != "rule" for k in ("rescue", "explore") if k in use)
    if task_net and not {"rescue", "explore"} <= set(cmc):
        raise UnifiedTaskModel(f"structure mismatch: {rel(path)} 은 예전 통합 task 네트워크 (rescue / explore 분리 전) 로 학습된 "
                               f"모델 — 지금 코드 (rescue / explore 네트워크 분리) 로는 로드할 수 없습니다 (새로 학습하세요)")
    weights = None
    if with_weights:
        weights = read_weights(ck)
        if task_net and any(k.startswith(UNIFIED_TASK) for k in weights):
            raise UnifiedTaskModel(f"structure mismatch: {rel(path)} 의 가중치가 예전 통합 task 네트워크 ({UNIFIED_TASK}*) — "
                                   f"rescue / explore 분리 구조로는 로드할 수 없습니다 (새로 학습하세요)")
    return ModelSpec(rel(path), ck, comp, meta, kind, weights, train_cfg)


def load_model(path, use=COMPONENT_KEYS):
    """모델 경로 -> ModelSpec (구성요소 모드 · 가중치 · 학습 config). use 는 inspect_model 과 같다."""
    return inspect_model(path, with_weights=True, use=use)


def _pick(weights, prefixes):
    return {k: v for k, v in weights.items() if k.startswith(tuple(prefixes))}


def same_names_shapes(target, source, label):
    """target / source ({이름: 텐서 | 배열}) 의 이름 집합 · shape 가 같은지 (다르면 RuntimeError)."""
    ta = {k: tuple(getattr(v, "shape", ())) for k, v in target.items()}
    sb = {k: tuple(getattr(v, "shape", ())) for k, v in source.items()}
    if set(ta) != set(sb):
        raise RuntimeError(f"{label}: 파라미터 이름 불일치 — 대상에만 {sorted(set(ta) - set(sb))[:5]}, "
                           f"원본에만 {sorted(set(sb) - set(ta))[:5]}")
    bad = [(k, sb[k], ta[k]) for k in ta if ta[k] != sb[k]]
    if bad:
        raise RuntimeError(f"{label}: shape 불일치 " + ", ".join(f"{k} {s} -> {t}" for k, s, t in bad[:5]))


def load_prefixes(model, weights, prefixes, label="weights"):
    """torch 모델에 weights 중 prefixes 부분을 strict 로드 (모델의 같은 부분과 이름 · shape 가 전부 같아야 함)."""
    import numpy as np
    import torch
    target = model.state_dict()
    tsub, wsub = _pick(target, prefixes), _pick(weights, prefixes)
    same_names_shapes(tsub, wsub, label)
    new = dict(target)
    for k, v in wsub.items():
        new[k] = torch.as_tensor(np.asarray(v.cpu() if hasattr(v, "cpu") else v)).to(target[k].device).to(target[k].dtype)
    model.load_state_dict(new, strict=True)
    return sorted(wsub)
