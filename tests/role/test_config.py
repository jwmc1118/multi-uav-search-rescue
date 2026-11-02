"""설정 yaml 점검 (ray 불필요).

    python -m tests.role.test_config

[1] 학습용 yaml (env / role / task) 과 GUI 사본 (gui_evaluation *_main.yaml) 은 파일마다 key 집합이 같다.
    평가 env (rule_evaluation *_rule.yaml, scale_evaluation *_scale.yaml) 는 시뮬레이션 env_config 만 두므로, 세 파일을 합친
    env_config 를 학습용 합친 env_config 에서 뺀 부분 (OMIT: gui.*, rewards.*) 을 제외하고 비교 — key 집합 같음.
    값이 다른 key 를 출력하고, 허용 목록 (ALLOWED, 사본별 — 지금은 모두 GUI · 실행 관련: main.*, gui.*, depot) 밖의 값
    차이가 있으면 실패. 다르게 둘 키는 ALLOWED 에 추가
[2] load_config 는 dict 를 key 단위로 깊게 합친다 (role.yaml 과 task.yaml 의 env_config.rewards 가 함께 남음), list / 값은 교체
[3] 보상 설정 키: role w_x / terminal_unrescued_as_expired / w_d / discover_margin / w_sw (role.yaml), rescue epsilon_urgent /
    w_b / w_c / w_dup · explore w_share / w_omega (task.yaml), events period (env.yaml 에는 rewards 없음). role_gamma 는
    load_config 가 model.custom_model_config.role_gamma 로 넣는다 (role 보상 키는 정확히 이 다섯 개)
[4] 구조: policy_mode 키 = {role, task} (값 rule | learned), 네트워크 설정은 role.yaml / task.yaml 에만, env.yaml env_config.memory,
    값 w_share 0.5 · w_omega 0.3 · w_dup 0.05 · eval.n_seeds 20 (코드 기본값 w_sw 0.1)
"""
import os

import yaml

from training.config_loader import (CONFIG_DIR, MAIN_CONFIG_DIR, MAIN_FILES, RULE_CONFIG_DIR, RULE_FILES, SCALE_CONFIG_DIR,
                                    SCALE_FILES, load_config, merge, reward_settings_line)

BASES = ("env", "role", "task")
COPIES = {"_main": MAIN_CONFIG_DIR, "_rule": RULE_CONFIG_DIR, "_scale": SCALE_CONFIG_DIR}   # 학습용 yaml 의 사본 (접미사: 폴더)
EVAL_FILES = {"_rule": RULE_FILES, "_scale": SCALE_FILES}       # 평가 env (env_config 만, 합쳐서 비교)
OMIT = ("gui.", "rewards.")                                      # 평가 env 에서 뺀 학습용 env_config key (GUI · 보상)
ALLOWED = {"_main": ("main.", "env_config.gui.", "env_config.depot"),   # 사본별 값 차이 허용 key (접두사, GUI · 실행 관련)
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
    """학습용 yaml 은 training/config/, 사본 (*_main / *_rule / *_scale.yaml) 은 COPIES 의 폴더."""
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
        _compare(f"{b}.yaml vs {b}_main.yaml", leaves(read(f"{b}.yaml")), leaves(read(f"{b}_main.yaml")),
                 ALLOWED["_main"], bad_all)
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
    assert set(rw) == {"role", "rescue", "explore"}
    assert set(rw["role"]) == {"w_x", "terminal_unrescued_as_expired", "w_d", "discover_margin", "w_sw"}
    assert set(rw["explore"]) == {"w_share", "w_omega"}
    assert set(rw["rescue"]) == {"epsilon_urgent", "w_b", "w_c", "w_dup"}
    ev = cfg["env_config"]["events"]
    assert ev["period"] == 10 and not {"new_survivor_targets", "option_continue_on_same_goal", "goal_latch"} & set(ev)
    assert cfg["env_config"]["s_max"] == cfg["model"]["custom_model_config"]["gnn"]["s_max"]
    assert cfg["model"]["custom_model_config"]["role_gamma"] == cfg["role_gamma"] == 1.0   # role SMDP γ_r (gamma 0.99 와 별개)
    main = load_config(MAIN_FILES)
    assert main["model"]["custom_model_config"]["role_gamma"] == main["role_gamma"]
    print("[2] load_config deep-merges dicts key by key (env_config.rewards: role / rescue / explore from head yamls; "
          "role_gamma -> model.custom_model_config.role_gamma), lists / scalars replaced")


def check_reward_keys():
    for suffix in ("", "_main"):
        env, role, task = (read(f"{b}{suffix}.yaml").get("env_config") or {} for b in BASES)
        assert not env.get("rewards"), env.get("rewards")
        rr = role["rewards"]["role"]
        assert set(task["rewards"]) == {"rescue", "explore"} and set(task["rewards"]["explore"]) == {"w_share", "w_omega"}
        assert set(task["rewards"]["rescue"]) == {"epsilon_urgent", "w_b", "w_c", "w_dup"}
        assert {"cluster", "explore", "policy_mode"} <= set(task) and "return_with_cap" not in role["role"]
        assert set(rr) == {"w_x", "terminal_unrescued_as_expired", "w_d", "discover_margin", "w_sw"}, sorted(rr)
        assert "goal_mode" not in task["explore"] and "deadline_mode" not in env
    cfg = load_config()
    g = cfg["model"]["custom_model_config"]["gnn"]
    assert "aggr" not in g and "attn_heads" in g
    print("[3] " + reward_settings_line(cfg, "train"))


def check_restored_values():
    """구조 / 값: config 파일 = env / role / task, policy_mode 키 {role, task}, memory 설정, 보상 값."""
    assert sorted(f for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml")) == ["env.yaml", "role.yaml", "task.yaml"]
    assert sorted(f for f in os.listdir(MAIN_CONFIG_DIR) if f.endswith(".yaml")) == \
        ["env_main.yaml", "role_main.yaml", "task_main.yaml"]
    assert sorted(os.listdir(RULE_CONFIG_DIR)) == ["env_rule.yaml", "role_rule.yaml", "rule.yaml", "task_rule.yaml"]
    assert sorted(os.listdir(SCALE_CONFIG_DIR)) == ["env_scale.yaml", "role_scale.yaml", "scale.yaml", "task_scale.yaml"]
    for files in (None, MAIN_FILES):
        cfg = load_config(files) if files else load_config()
        e = cfg["env_config"]
        assert set(e["policy_mode"]) == {"role", "task"} and set(e["policy_mode"].values()) <= {"rule", "learned"}
        mc = cfg["model"]["custom_model_config"]
        assert {"role", "task", "loss", "gnn"} <= set(mc) and not {"rescue", "explore"} & set(mc)
        assert set(mc["loss"]) == {"lambda_role", "lambda_task"}
        assert e["memory"] == {"enabled": True, "T_age": 50}
        assert e["rewards"]["explore"]["w_share"] == 0.5 and e["rewards"]["explore"]["w_omega"] == 0.3
        assert e["rewards"]["rescue"]["w_dup"] == 0.05
        assert cfg["eval"]["n_seeds"] == 20 and "episodes" not in cfg["eval"]
    from env.role.simulation import REWARDS_DEFAULTS
    assert REWARDS_DEFAULTS["explore"] == {"w_share": 0.5, "w_omega": 0.3} and REWARDS_DEFAULTS["role"]["w_sw"] == 0.1
    assert REWARDS_DEFAULTS["rescue"]["w_dup"] == 0.05
    print(f"[4] structure: policy_mode {load_config()['env_config']['policy_mode']} (keys role / task), files env / role / task, memory {{enabled, T_age 50}}; w_share 0.5, w_omega 0.3, w_dup 0.05, eval.n_seeds 20; "
          f"code defaults match")


if __name__ == "__main__":
    check_main_sync()
    check_deep_merge()
    check_reward_keys()
    check_restored_values()
    print("test_config: all ok")
