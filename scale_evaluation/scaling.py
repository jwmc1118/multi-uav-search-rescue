"""N 별 환경 설정 (main_scale.py) — 체크포인트 기준 env_config 를 r = N / N_ref 로 크기 조정.

    scaled_env_config(base, n, n_ref, rule)   rule = {survivors, map, golden_time, max_episode_len: 지수 e 또는 None}
        값 <- round(값 x r^e) (None = 그대로). num_targets / target_golden_time 은 int 또는 [min, max] 모두 처리.
        UAV 는 첫 UAV 설정 복제 (센서 / 통신 범위 고정), max_targets = max(기준값, 조난자 수 상한),
        d_max = None (새 맵 rows + cols — 입력 정규화도 새 맵 기준).
    describe(env)   표 / 그림 보조 라벨용 {rows, cols, survivors 'a–b', golden 'a–b', max_episode_len}
    FREE_KEYS       CheckpointPolicy 가 체크포인트 값으로 고정하지 않을 키 (모델 파라미터와 무관한 크기 값)
"""
import copy

FREE_KEYS = ("rows", "cols", "max_targets", "max_episode_len")
ITEMS = ("survivors", "map", "golden_time", "max_episode_len")


def _scale(v, f, lo=1):
    if isinstance(v, (list, tuple)):
        return [max(lo, int(round(x * f))) for x in v]
    return max(lo, int(round(v * f)))


def _hi(v):
    return max(v) if isinstance(v, (list, tuple)) else v


def scaled_env_config(base, n, n_ref, rule):
    unknown = set(rule or {}) - set(ITEMS)
    if unknown:
        raise ValueError(f"scale preset 에 모르는 항목 {sorted(unknown)} — {ITEMS}")
    env = copy.deepcopy(base)
    r = float(n) / float(n_ref)
    f = {k: (None if (rule or {}).get(k) is None else r ** float(rule[k])) for k in ITEMS}
    env["agents"] = [dict(env["agents"][0]) for _ in range(int(n))]
    if f["survivors"] is not None:
        env["num_targets"] = _scale(env["num_targets"], f["survivors"])
    if f["map"] is not None:
        env["rows"], env["cols"] = _scale(env["rows"], f["map"], 8), _scale(env["cols"], f["map"], 8)
    if f["golden_time"] is not None:
        env["target_golden_time"] = _scale(env["target_golden_time"], f["golden_time"])
    if f["max_episode_len"] is not None:
        env["max_episode_len"] = _scale(env["max_episode_len"], f["max_episode_len"])
    env["max_targets"] = max(int(base.get("max_targets") or 0), int(_hi(env["num_targets"])))
    env["d_max"] = None
    return env


def _rng(v):
    return f"{v[0]}–{v[1]}" if isinstance(v, (list, tuple)) and v[0] != v[1] else str(_hi(v))


def describe(env):
    return {"rows": int(env["rows"]), "cols": int(env["cols"]), "survivors": _rng(env["num_targets"]),
            "golden": _rng(env["target_golden_time"]), "max_episode_len": int(env["max_episode_len"])}
