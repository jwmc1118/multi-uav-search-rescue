"""config yaml 로더 — training/config/ 의 env.yaml (환경 + 학습 공통) + role.yaml + explore.yaml + rescue.yaml (정책별:
학습 / 규칙, 이벤트 · 결정 규칙, 보상 weight, 네트워크) 를 합친다.

    load_config()                              # 기본 네 파일
    load_config([..., "my.yaml"])              # 파일 목록 (뒤 파일이 앞 파일 값을 덮어씀)

합치는 규칙: 앞 파일부터 차례로, dict 는 키별로 재귀 병합, 그 외 (list / 숫자 / 문자열) 는 뒤의 값으로 교체.
합친 뒤 model.custom_model_config.gnn.s_max (그래프 S 노드 수 = rescue 행동 슬롯 수) 를 env_config.s_max 로 넣는다 (한 곳에서만 설정).
"""
import copy
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_FILES = [os.path.join(CONFIG_DIR, f) for f in ("env.yaml", "role.yaml", "explore.yaml", "rescue.yaml")]


def merge(base, over):
    """over 를 base 위에 덮어쓴 새 dict (dict 는 재귀, 나머지는 교체)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def reward_settings_line(cfg, tag="train"):
    """실제 적용되는 보상 설정 한 줄 (env 기본값 REWARDS_DEFAULTS 위에 yaml env_config.rewards, gamma = 최상위 config) —
    train.py / main.py 시작 출력 (설정 반영 확인용)."""
    from env.role.simulation import REWARDS_DEFAULTS
    env = cfg.get("env_config") or {}
    given = env.get("rewards") or {}
    rw = {h: dict(REWARDS_DEFAULTS[h], **(given.get(h) or {})) for h in REWARDS_DEFAULTS}
    r, s = rw["role"], rw["rescue"]
    return (f"[{tag}] rewards: role w_x={r['w_x']} c_osc={r['c_osc']}, "
            f"rescue eps={s['epsilon_urgent']} w_c={s['w_c']} w_dup={s['w_dup']}, "
            f"explore coverage (1-psi_f) own x (1 + w_mult c) w_mult={rw['explore']['w_mult']}, "
            f"gamma={cfg.get('gamma')}, r_c_source={rw['connectivity']['r_c_source']}")


def load_config(paths=None):
    """paths (기본 DEFAULT_FILES) 의 yaml 을 차례로 합친 config."""
    if isinstance(paths, str):
        paths = [paths]
    cfg = {}
    for p in paths or DEFAULT_FILES:
        with open(p, "rb") as f:
            cfg = merge(cfg, yaml.safe_load(f) or {})
    s_max = (((cfg.get("model") or {}).get("custom_model_config") or {}).get("gnn") or {}).get("s_max")
    if s_max is not None and isinstance(cfg.get("env_config"), dict):
        cfg["env_config"]["s_max"] = int(s_max)
    return cfg
