"""설정 yaml 점검 (ray 불필요).

    python -m tests.role.test_config

[1] 학습용 yaml (env / role / explore / rescue) 과 *_main.yaml 의 key 집합이 같다. 값이 다른 key 를 출력하고,
    허용 목록 (GUI · 실행 관련: main.*, env_config.gui.*, env_config.depot) 밖의 값 차이가 있으면 실패
[2] load_config 는 dict 를 key 단위로 깊게 합친다 (env.yaml 과 head yaml 의 env_config.rewards 가 함께 남음), list / 값은 교체
[3] 보상 설정: 예전 키 (rewards.role.lambda_conn / gamma / w_conn, rescue delta, events.osc_window) 없음,
    role discover_mode / w_d / discover_feasible_only, rescue epsilon_urgent / w_c / w_dup / f_mode, explore w_mult / psi_scale,
    connectivity r_c_source 가 각 파일에 있음. shaping_gamma 는 yaml 이 아니라 load_config 가 최상위 gamma 에서 넣는다
"""
import os

import yaml

from training.config_loader import CONFIG_DIR, MAIN_CONFIG_DIR, MAIN_FILES, load_config, merge, reward_settings_line

BASES = ("env", "role", "explore", "rescue")
ALLOWED = ("main.", "env_config.gui.", "env_config.depot")       # 값 차이 허용 (GUI · 실행 관련)


def leaves(d, p=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(leaves(v, p + (k,)))
        else:
            out[".".join(p + (k,))] = v
    return out


def read(name):
    """학습용 yaml 은 training/config/, *_main.yaml 은 gui_evaluation/config/."""
    with open(os.path.join(MAIN_CONFIG_DIR if name.endswith("_main.yaml") else CONFIG_DIR, name)) as f:
        return yaml.safe_load(f) or {}


def check_main_sync():
    for b in BASES:
        a, m = leaves(read(f"{b}.yaml")), leaves(read(f"{b}_main.yaml"))
        only_a, only_m = sorted(set(a) - set(m)), sorted(set(m) - set(a))
        assert not only_a and not only_m, f"{b}: key 집합이 다름 — 학습용에만 {only_a}, main 에만 {only_m}"
        diff = sorted(k for k in a if a[k] != m[k])
        bad = [k for k in diff if not k.startswith(ALLOWED)]
        print(f"[1] {b}.yaml vs {b}_main.yaml: {len(a)} keys identical set; value differs: "
              + (", ".join(f"{k} ({a[k]!r} -> {m[k]!r})" for k in diff) or "none"))
        assert not bad, f"{b}: 허용 목록 밖의 값 차이 {bad}"


def check_deep_merge():
    m = merge({"a": {"x": 1, "y": {"p": 1, "q": 2}}, "l": [1, 2]}, {"a": {"y": {"q": 3}, "z": 4}, "l": [9]})
    assert m == {"a": {"x": 1, "y": {"p": 1, "q": 3}, "z": 4}, "l": [9]}, m
    cfg = load_config()
    rw = cfg["env_config"]["rewards"]
    assert rw["connectivity"]["r_c_source"] == "base"                 # env.yaml
    assert set(rw["role"]) == {"w_x", "c_osc", "osc_window", "terminal_unrescued_as_expired", "discover_mode", "w_d",
                               "discover_feasible_only", "shaping_gamma"}   # role.yaml + shaping_gamma (config_loader)
    assert set(rw["explore"]) == {"w_mult", "psi_scale"} and set(rw["rescue"]) == {"epsilon_urgent", "w_c", "w_dup", "f_mode"}
    assert cfg["env_config"]["s_max"] == cfg["model"]["custom_model_config"]["gnn"]["s_max"]
    assert rw["role"]["shaping_gamma"] == cfg["gamma"]                  # 발견 shaping γ = 학습 gamma
    main = load_config(MAIN_FILES)
    assert main["env_config"]["rewards"]["role"]["shaping_gamma"] == main["gamma"]
    print("[2] load_config deep-merges dicts key by key (env_config.rewards: connectivity from env.yaml + role / "
          "rescue from head yamls), lists / scalars replaced")


def check_reward_keys():
    for suffix in ("", "_main"):
        env, role, exp, res = (read(f"{b}{suffix}.yaml").get("env_config") or {} for b in BASES)
        assert set((env.get("rewards") or {})) == {"connectivity"}, env.get("rewards")
        rr = role["rewards"]["role"]
        assert not {"lambda_conn", "gamma", "w_conn"} & set(rr)
        assert "osc_window" not in role["events"] and "osc_window" in rr
        assert set(exp["rewards"]["explore"]) == {"w_mult", "psi_scale"} and "explore_sub_mode" not in exp
        assert set(res["rewards"]["rescue"]) == {"epsilon_urgent", "w_c", "w_dup", "f_mode"}
        assert {"discover_mode", "w_d", "discover_feasible_only"} <= set(rr) and "shaping_gamma" not in rr   # γ 는 yaml 이 아니라 gamma 에서
    cfg = load_config()
    g = cfg["model"]["custom_model_config"]["gnn"]
    assert g["aggr"] in ("attention", "mean_max") and "attn_heads" in g
    print("[3] " + reward_settings_line(cfg, "train"))


if __name__ == "__main__":
    check_main_sync()
    check_deep_merge()
    check_reward_keys()
    print("test_config: all ok")
