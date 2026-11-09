"""설정 yaml 점검 (ray 불필요).

    python -m tests.role.test_config

[1] GUI 사본 (gui_evaluation *_gui.yaml) 은 GUI 가 쓰는 최상위 키 (model, env_config, main) 만 두고, 학습용 yaml (env / role /
    task) 의 그 키들과 파일마다 key 집합이 같다 — 단 GUI 사본에만 main.* (화면 설정 · 시작 조합). 학습 전용 (PPO · 병렬 수집 ·
    eval · plot · 학습 모드) 은 GUI 사본에 없어야 한다.
    평가 env (rule_evaluation *_rule.yaml, scale_evaluation *_scale.yaml) 는 시뮬레이션 env_config 만 두므로, 세 파일을 합친
    env_config 를 학습용 합친 env_config 에서 뺀 부분 (OMIT: gui.*) 을 제외하고 비교 — key 집합 · 값 같음 (보상 포함).
    값이 다른 key 를 출력하고, 허용 목록 (ALLOWED, 사본별 — 지금은 모두 GUI 관련: gui.*, depot) 밖의 값
    차이가 있으면 실패. 다르게 둘 키는 ALLOWED 에 추가
[2] load_config 는 dict 를 key 단위로 깊게 합친다 (role.yaml 과 task.yaml 의 env_config.rewards 가 함께 남음), list / 값은 교체
[3] 보상 설정 키: role w_x / terminal_unrescued_as_expired / w_d / discover_margin / w_sw (role.yaml), task w_u / w_q / w_x /
    w_dup / w_ts (task.yaml), events period (env.yaml 에는 rewards 없음). role_gamma 는
    load_config 가 model.custom_model_config.role_gamma 로 넣는다 (role 보상 키는 정확히 이 다섯 개)
[4] 구조: 학습 구성요소 키 (components) 는 학습용 env.yaml 에만, policy_mode 없음, 네트워크 설정은 role.yaml / task.yaml 에만, env.yaml env_config.memory,
    task 보상 값 = 코드 기본값 (w_u 1 · w_q 1 · w_x 2 · w_dup 0.5 · w_ts 0.2), eval every 25 · n_seeds 10
"""
import os

import yaml

from training.config_loader import (CONFIG_DIR, GUI_CONFIG_DIR, GUI_FILES, RULE_CONFIG_DIR, RULE_FILES, SCALE_CONFIG_DIR,
                                    SCALE_FILES, load_config, merge, reward_settings_line)

BASES = ("env", "role", "task")
COPIES = {"_gui": GUI_CONFIG_DIR, "_rule": RULE_CONFIG_DIR, "_scale": SCALE_CONFIG_DIR}   # 학습용 yaml 의 사본 (접미사: 폴더)
EVAL_FILES = {"_rule": RULE_FILES, "_scale": SCALE_FILES}       # 평가 env (env_config 만, 합쳐서 비교)
OMIT = ("gui.",)                                                 # 평가 env 에서 뺀 학습용 env_config key (GUI)
GUI_KEYS = {"model", "env_config", "main"}                     # GUI 사본에 두는 최상위 key (나머지는 학습 전용)
GUI_ONLY = ("main.",)                                           # GUI 사본에만 있는 key (main_gui.py 화면 설정)
ALLOWED = {"_gui": ("env_config.gui.", "env_config.depot"),      # 사본별 값 차이 허용 key (접두사, GUI · 실행 관련)
           "_rule": ("gui.", "depot"), "_scale": ("gui.", "depot")}       # — 다르게 둘 key 는 여기에 추가


def leaves(d, p=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(leaves(v, p + (k,)))
        else:
            out[".".join(p + (k,))] = v
    return out


def read(name):
    """학습용 yaml 은 training/config/, 사본 (*_gui / *_rule / *_scale.yaml) 은 COPIES 의 폴더."""
    d = next((v for k, v in COPIES.items() if name.endswith(f"{k}.yaml")), CONFIG_DIR)
    with open(os.path.join(d, name)) as f:
        return yaml.safe_load(f) or {}


def _compare(name, a, m, allowed, bad_all):
    only_a, only_m = sorted(set(a) - set(m)), sorted(set(m) - set(a))
    assert not only_a and not only_m, f"{name}: key 집합이 다름 — 학습용에만 {only_a}, 사본에만 {only_m}"
    diff = sorted(k for k in a if a[k] != m[k])
    bad_all += [f"{name}: {k}" for k in diff if not k.startswith(allowed)]
    print(f"[1] {name}: {len(a)} keys identical set; value differs: "
          + (", ".join(f"{k} ({a[k]!r} -> {m[k]!r})" for k in diff) or "none"))


def check_main_sync():
    bad_all = []
    for b in BASES:
        gui = read(f"{b}_gui.yaml")
        assert set(gui) <= GUI_KEYS, f"{b}_gui.yaml: GUI 가 쓰지 않는 최상위 key {sorted(set(gui) - GUI_KEYS)}"
        train = {k: v for k, v in leaves(read(f"{b}.yaml")).items() if k.split(".")[0] in GUI_KEYS}
        _compare(f"{b}.yaml vs {b}_gui.yaml", train, {k: v for k, v in leaves(gui).items() if not k.startswith(GUI_ONLY)},
                 ALLOWED["_gui"], bad_all)
    train = {k: v for k, v in leaves(load_config()["env_config"]).items() if not k.startswith(OMIT)}
    for suf, files in EVAL_FILES.items():
        cfg = load_config(files)
        assert set(cfg) <= {"env_config", "model"} and set(cfg.get("model") or {}) <= {"custom_model_config"}, \
            f"{suf}: env_config 밖의 설정 {sorted(cfg)}"       # model.custom_model_config.order_dist_alpha 는 load_config 가 넣음
        _compare(f"merged env_config: training vs *{suf}.yaml", train, leaves(cfg["env_config"]), ALLOWED[suf], bad_all)
    assert not bad_all, f"허용 목록 밖의 값 차이 {bad_all}"


def check_deep_merge():
    m = merge({"a": {"x": 1, "y": {"p": 1, "q": 2}}, "l": [1, 2]}, {"a": {"y": {"q": 3}, "z": 4}, "l": [9]})
    assert m == {"a": {"x": 1, "y": {"p": 1, "q": 3}, "z": 4}, "l": [9]}, m
    cfg = load_config()
    rw = cfg["env_config"]["rewards"]
    assert set(rw) == {"role", "task"}
    assert set(rw["role"]) == {"w_x", "terminal_unrescued_as_expired", "w_d", "discover_margin", "w_sw"}
    assert set(rw["task"]) == {"w_u", "w_q", "w_x", "w_dup", "w_ts"}
    ev = cfg["env_config"]["events"]
    assert ev["period"] == 10 and not {"new_survivor_targets", "option_continue_on_same_goal", "goal_latch"} & set(ev)
    assert cfg["env_config"]["s_max"] == cfg["model"]["custom_model_config"]["gnn"]["s_max"]
    assert cfg["model"]["custom_model_config"]["role_gamma"] == cfg["role_gamma"] == 1.0   # role SMDP γ_r (gamma 0.99 와 별개)
    print("[2] load_config deep-merges dicts key by key (env_config.rewards: role / task from head yamls; "
          "role_gamma -> model.custom_model_config.role_gamma), lists / scalars replaced")


def check_reward_keys():
    for suffix in ("", "_gui"):
        env, role, task = (read(f"{b}{suffix}.yaml").get("env_config") or {} for b in BASES)
        assert not env.get("rewards"), env.get("rewards")
        rr = role["rewards"]["role"]
        assert set(task["rewards"]) == {"task"}
        assert {"cluster", "explore"} <= set(task) and "return_with_cap" not in role["role"]
        assert "policy_mode" not in task and "policy_mode" not in role, suffix   # 구성요소 모드는 components / 모델 경로가 정함
        assert set(rr) == {"w_x", "terminal_unrescued_as_expired", "w_d", "discover_margin", "w_sw"}, sorted(rr)
        assert "goal_mode" not in task["explore"] and "deadline_mode" not in env
    cfg = load_config()
    g = cfg["model"]["custom_model_config"]["gnn"]
    assert "aggr" not in g and "attn_heads" in g
    print("[3] " + reward_settings_line(cfg, "train"))


def check_restored_values():
    """구조 / 값: config 파일 = env / role / task, 학습 구성요소 키 (components {role, rescue, explore: {mode, path}}, 학습용에만),
    GUI 시작 조합 키, memory 설정, 보상 값."""
    assert sorted(f for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml")) == ["env.yaml", "role.yaml", "task.yaml"]
    assert sorted(f for f in os.listdir(GUI_CONFIG_DIR) if f.endswith(".yaml")) == \
        ["env_gui.yaml", "role_gui.yaml", "task_gui.yaml"]
    assert sorted(os.listdir(RULE_CONFIG_DIR)) == ["env_rule.yaml", "role_rule.yaml", "rule.yaml", "task_rule.yaml"]
    assert sorted(os.listdir(SCALE_CONFIG_DIR)) == ["env_scale.yaml", "role_scale.yaml", "scale.yaml", "task_scale.yaml"]
    for files in (None, GUI_FILES):
        cfg = load_config(files) if files else load_config()
        e = cfg["env_config"]
        assert "policy_mode" not in e and "components" not in e        # 구성요소 모드는 학습 (components) / 모델 경로가 정함
        if files is None:
            from eval_common.model_loader import COMPONENT_KEYS, MODES, combo_name
            comp = cfg["components"]
            assert set(comp) == set(COMPONENT_KEYS) and all(set(c) == {"mode", "path"} and c["mode"] in MODES
                                                            for c in comp.values()), comp
            combo_name(comp)
        else:
            assert "components" not in cfg, "GUI 사본에 학습 구성요소 키"
            assert {"source", "combo", "model", "run", "checkpoint"} <= set(cfg["main"]), cfg["main"]
        mc = cfg["model"]["custom_model_config"]
        assert {"role", "rescue", "explore", "loss", "gnn"} <= set(mc) and "task" not in mc
        assert set(mc["loss"]) == {"lambda_role", "lambda_rescue", "lambda_explore"}
        for k in ("rescue", "explore"):
            assert {"lr", "clip_param", "entropy_coeff", "pointer_clip", "value_norm", "critic"} == set(mc[k]), mc[k]
        assert e["memory"] == {"enabled": True, "T_age": 50}
        assert e["rewards"]["task"] == {"w_u": 1.0, "w_q": 1.0, "w_x": 2.0, "w_dup": 0.5, "w_ts": 0.2}, e["rewards"]["task"]
        if files is None:
            assert set(cfg["eval"]) == {"every", "n_seeds", "baselines"} and cfg["eval"]["every"] == 25 and cfg["eval"]["n_seeds"] == 10, cfg["eval"]
            from training.behavior import BASELINES
            assert set(cfg["eval"]["baselines"]) == set(BASELINES), cfg["eval"]["baselines"]
        else:
            assert not {"eval", "plot", "gamma", "role_gamma", "lr", "num_workers"} & set(cfg), "GUI 사본에 학습 전용 키"
    from env.role.simulation import REWARDS_DEFAULTS
    assert REWARDS_DEFAULTS["task"] == {"w_u": 1.0, "w_q": 1.0, "w_x": 2.0, "w_dup": 0.5, "w_ts": 0.2} and set(REWARDS_DEFAULTS) == {"role", "task"}
    print(f"[4] structure: components {load_config()['components']}, files env / role / task, memory {{enabled, T_age 50}}; task rewards {e['rewards']['task']}, eval every 25 / n_seeds 10 (GUI 사본에는 없음); "
          f"code defaults match")


if __name__ == "__main__":
    check_main_sync()
    check_deep_merge()
    check_reward_keys()
    check_restored_values()
    print("test_config: all ok")
