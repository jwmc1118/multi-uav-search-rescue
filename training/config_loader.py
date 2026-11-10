"""config yaml 로더 — training/config/ 의 env.yaml (환경 + 학습 공통) + role.yaml (역할: 학습 / 규칙, 이벤트, role 보상,
역할 선택 네트워크) + task.yaml (하위 할당: 학습 / 규칙, rescue / explore 네트워크 · critic · loss, task 보상, 덩어리 · 1 단계 ·
2 단계) 를 차례로 합친다.

    load_config()                              # 기본 세 파일
    load_config([..., "my.yaml"])              # 파일 목록 (뒤 파일이 앞 파일 값을 덮어씀)

합치는 규칙: 앞 파일부터 차례로, dict 는 키별로 재귀 병합, 그 외 (list / 숫자 / 문자열) 는 뒤의 값으로 교체.
합친 뒤 model.custom_model_config.gnn.s_max (그래프 S 노드 수 = 할당 행동의 생존자 슬롯 수) 를 env_config.s_max 로 넣는다 (한 곳에서만 설정).
최상위 role_gamma (role SMDP 할인 γ_r) 는 model.custom_model_config.role_gamma 로 복사한다 (RLlib 이 모르는 최상위 키라
train.py 가 빼므로, training/role/ppo.py 가 policy config 의 이 값으로 role advantage 를 계산). 최상위 gamma 는 task SMDP 할인 · RLlib 설정.
env_config.explore.order_dist_alpha (explore 1 단계 차례의 α) 는 model.custom_model_config.order_dist_alpha 로 복사한다
(env · 그리디 · 모델 순차 디코딩이 같은 α 를 쓰게).
"""
import copy
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_FILES = [os.path.join(CONFIG_DIR, f) for f in ("env.yaml", "role.yaml", "task.yaml")]
# main_gui.py 전용 yaml (학습용과 key 구조가 같음, 병합 순서도 같음) — gui_evaluation/config/
GUI_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui_evaluation", "config")
GUI_FILES = [os.path.join(GUI_CONFIG_DIR, f) for f in ("env_gui.yaml", "role_gui.yaml", "task_gui.yaml")]
# main_rule.py 평가 환경 yaml (학습용과 key 구조 · 병합 순서 같음) — rule_evaluation/config/
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULE_CONFIG_DIR = os.path.join(_REPO, "rule_evaluation", "config")
RULE_FILES = [os.path.join(RULE_CONFIG_DIR, f) for f in ("env_rule.yaml", "role_rule.yaml", "task_rule.yaml")]


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
    """실제 적용되는 보상 설정 한 줄 (env 기본값 REWARDS_DEFAULTS 위에 yaml env_config.rewards, role_gamma = role SMDP 할인,
    γ = 최상위 gamma) — train.py / main_gui.py 시작 출력 (설정 반영 확인용)."""
    from env.role.simulation import EVENTS_DEFAULTS, REWARDS_DEFAULTS
    env = cfg.get("env_config") or {}
    given = env.get("rewards") or {}
    rw = {h: dict(REWARDS_DEFAULTS[h], **{k: v for k, v in (given.get(h) or {}).items() if k in REWARDS_DEFAULTS[h]})
          for h in REWARDS_DEFAULTS}
    r, k = rw["role"], rw["task"]
    ev = dict(EVENTS_DEFAULTS, **(env.get("events") or {}))
    return (f"[{tag}] rewards: role w_x={r['w_x']} timely discovery w_d={r['w_d']} T_m={r['discover_margin']} "
            f"w_sw={r['w_sw']}, task w_u={k['w_u']} w_q={k['w_q']} w_x={k['w_x']} w_dup={k['w_dup']}/N_T w_ts={k['w_ts']}, "
            + (f"gamma={cfg.get('gamma')} role_gamma={cfg.get('role_gamma')}; " if "gamma" in cfg else "")   # GUI 설정에는 없음
            + f"events period={ev['period']}")


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
    if isinstance(cfg.get("env_config"), dict):
        sync_order_alpha(cfg["env_config"], cfg.setdefault("model", {}).setdefault("custom_model_config", {}))
    if cfg.get("role_gamma") is not None:
        cfg.setdefault("model", {}).setdefault("custom_model_config", {})["role_gamma"] = float(cfg["role_gamma"])
    check_task_rewards(cfg)
    return cfg


def check_task_rewards(cfg):
    """task 보상 w_x <= w_q 이면 경고 1 줄 (생존자를 만료시켜 대기 패널티를 없애는 해킹 방지 조건 w_x > w_q)."""
    tk = (((cfg.get("env_config") or {}).get("rewards") or {}).get("task") or {})
    if "w_x" in tk and "w_q" in tk and float(tk["w_x"]) <= float(tk["w_q"]):
        print(f"[config] warning: rewards.task w_x={tk['w_x']} <= w_q={tk['w_q']} — 만료로 대기 패널티를 피하는 쪽이 유리해질 수 있음")


def order_alpha(env_config):
    """env_config.explore.order_dist_alpha (없으면 env 기본값)."""
    from env.role.explore import EXPLORE_DEFAULTS
    return float(((env_config or {}).get("explore") or {}).get("order_dist_alpha", EXPLORE_DEFAULTS["order_dist_alpha"]))


def sync_order_alpha(env_config, model_config):
    """모델 순차 디코딩의 α = env 의 α (model_config = custom_model_config, 제자리 수정)."""
    model_config["order_dist_alpha"] = order_alpha(env_config)
    return model_config

