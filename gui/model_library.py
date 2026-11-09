"""GUI 모델 선택 창 (Source -> 조합 / Run) 의 목록 — 모델 경로 하나로 조합을 정하는 건 eval_common/model_loader.py.

    Saved models   marl_models/best_model/<조합>/model_vXX — 조합 폴더를 그대로 버튼으로 (맨 앞에 모델 없이 도는 rule_rule_rule),
                   고르면 그 안의 model_vXX 목록 (iter, eval rescue, 구성요소 요약)
    Training       marl_models/ 아래 checkpoints/ 가 있는 run 폴더 (best_model · _archive 제외) — 조합 이름 표시 (run_meta.json -> 폴더 이름 ->
                   params.json, eval_common/model_loader.py components_from_meta), 고르면 체크포인트 = best + checkpoint_XXXXXX
                   (최신 순, TRAIN_MAX 개)

    saved_combos()                      -> 모델이 있는 best_model 하위 조합 폴더 이름 (이름 순)
    saved_models(combo)                 -> [ModelEntry] (이름 순), 폴더가 없거나 비면 []
    training_runs()                     -> [RunEntry] (최근 순)
    run_checkpoints(run)                -> [ModelEntry] (best 먼저 + 최근 순)
    check_entry(entry, env, model_config)  지금 코드 · GUI 설정으로 로드 가능한지 (이유 문자열 또는 None, 같은 학습 run 은 한 번)
    summary(entry)                      current 줄 ("Saved · learn_learn_rule · model_v01" / "Training · <run> · checkpoint_000300")
rescue / explore 분리 전 통합 task 네트워크로 학습된 모델 (rescue / explore 가 rule 아님) 은 "(structure mismatch)" 로 표시되고
선택되지 않는다.
"""
import glob
import json
import os
import re

from eval_common.model_loader import (RUN_META, combo_name, components_from_meta, find_checkpoint, inspect_model,
                                      iteration_of)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(REPO_ROOT, "marl_models")
MODEL_ROOT = os.path.join(RUN_ROOT, "best_model")
SKIP_DIRS = ("best_model", "_archive")                           # marl_models/ 아래 run 이 아닌 폴더 (_archive = 보관, 목록에서 무시)
TRAIN_MAX = 30                                                   # run 마다 보일 최근 체크포인트 수 (best 제외)
RULE_COMBO = "rule_rule_rule"                                     # 모델 없이 규칙 정책 (Saved 의 첫 버튼)
_RUN_CHECK = {}                                                  # (params.json, mtime) -> 로드 불가 이유


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class ModelEntry:
    """목록 한 줄: Saved 의 model_vXX 또는 Training 의 체크포인트."""

    def __init__(self, source, group, name, path):
        self.source, self.group, self.name, self.path = source, group, name, path   # group = 조합 폴더 또는 run 이름
        self.checkpoint = self.spec = None
        self.iteration = self.rescue = None
        self.error = None                # 로드 불가 이유 (None = 아직 모름 / 가능)
        self.checked = False
        info = _read_json(os.path.join(path, "best_info.json")) or {}
        try:
            self.checkpoint = find_checkpoint(path)              # 구조 불일치여도 목록에는 보이게 (선택만 막음)
            self.iteration = info.get("training_iteration", iteration_of(self.checkpoint))
            self.spec = inspect_model(path)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            self.error = str(e).split("\n")[0][:200]
        self.rescue = info.get("eval_rescue_rate")

    def label(self):
        info = []
        if self.iteration is not None:
            info.append(f"iter {self.iteration}")
        if self.rescue is not None:
            info.append(f"eval rescue {self.rescue:.3f}")
        if self.spec is not None:
            info.append(self.spec.describe())
        s = self.name + (f"   ({', '.join(info)})" if info else "")
        if not self.error:
            return s
        return s + ("   (structure mismatch)" if self.error.startswith("structure mismatch") else "   (cannot load)")

    def key(self):
        """정책 캐시 키: 체크포인트 경로 + 수정 시각 (best/ 는 갱신되면 같은 경로에 새 체크포인트)."""
        try:
            return self.checkpoint, os.path.getmtime(self.checkpoint)
        except (OSError, TypeError):
            return self.checkpoint, None


class RunEntry:
    def __init__(self, path):
        self.path, self.name = path, os.path.basename(path)
        meta = _read_json(os.path.join(path, RUN_META))
        self.combo = combo_name(components_from_meta(meta, path, _read_json(os.path.join(path, "params.json"))))
        self.has_meta = meta is not None

    def label(self):
        return f"{self.name}   ({self.combo}{'' if self.has_meta else ', no run_meta'})"


def saved_combos():
    """모델 (하위 폴더) 이 하나라도 있는 best_model 하위 조합 폴더 이름 (이름 순)."""
    if not os.path.isdir(MODEL_ROOT):
        return []
    return sorted(c for c in os.listdir(MODEL_ROOT) if os.path.isdir(os.path.join(MODEL_ROOT, c))
                  and any(os.path.isdir(os.path.join(MODEL_ROOT, c, n)) for n in os.listdir(os.path.join(MODEL_ROOT, c))))


def saved_models(combo):
    base = os.path.join(MODEL_ROOT, combo)
    if not os.path.isdir(base):
        return []
    names = sorted(n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n)))
    return [ModelEntry("saved", combo, n, os.path.join(base, n)) for n in names]


def training_runs():
    """marl_models/ 아래 checkpoints/ 에 체크포인트가 있는 run (best_model · _archive 제외), 최근 수정 순."""
    runs = []
    for d in glob.glob(os.path.join(RUN_ROOT, "*")):
        if os.path.basename(d) in SKIP_DIRS or not os.path.isdir(os.path.join(d, "checkpoints")):
            continue
        if glob.glob(os.path.join(d, "checkpoints", "checkpoint_*")) or os.path.isdir(os.path.join(d, "checkpoints", "best")):
            runs.append(d)
    runs.sort(key=os.path.getmtime, reverse=True)
    return [RunEntry(d) for d in runs]


def run_checkpoints(run):
    ck = os.path.join(run.path, "checkpoints")
    out = []
    best = os.path.join(ck, "best")
    if os.path.isdir(best):
        e = ModelEntry("training", run.name, "best", best)
        if e.checkpoint is not None:
            e.name = f"best ({os.path.basename(os.path.dirname(e.checkpoint))})"
            out.append(e)
    num = lambda d: int(re.sub(r"\D", "", os.path.basename(d)) or 0)
    dirs = sorted((d for d in glob.glob(os.path.join(ck, "checkpoint_*")) if os.path.isdir(d)), key=num, reverse=True)
    for d in dirs[:TRAIN_MAX]:
        e = ModelEntry("training", run.name, os.path.basename(d), d)
        if e.checkpoint is not None:                             # 저장 중인 폴더 (파일 아직 없음) 는 건너뜀
            out.append(e)
    return out


def check_entry(entry, env, model_config=None):
    """로드 가능 여부 (가능하면 None, 아니면 이유). 학습 config 와 GUI 설정의 관측 차원 · 네트워크 구조 키를 비교하고
    (structure_diff), 맞으면 지금 코드로 actor 를 만들어 가중치 shape 까지 확인 (eval_common/light_policy.py, ray 없음).
    같은 학습 run (같은 params.json) 의 체크포인트는 구조가 같으므로 한 번만 확인한다."""
    if entry.checked:
        return entry.error
    entry.checked = True
    if entry.error:
        return entry.error
    from eval_common.checkpoint_policy import SHAPE_KEYS, find_params_json, structure_diff
    pj = find_params_json(entry.checkpoint)
    run = (pj, os.path.getmtime(pj), entry.spec.describe())
    if run not in _RUN_CHECK:
        diff = structure_diff(entry.spec.train_cfg, env, model_config)
        err = None
        if diff:
            err = "differs from current config: " + ", ".join(f"{k} checkpoint {a} vs gui {b}" for k, a, b in diff)
        else:
            try:
                from eval_common.light_policy import LightPolicy
                LightPolicy(entry.path).prepare(dict(env), free_keys=SHAPE_KEYS)
            except Exception as e:                               # 관측 차원 / actor 구조가 지금 코드와 다름
                err = str(e).split("\n")[0][:300]
        _RUN_CHECK[run] = err
    entry.error = _RUN_CHECK[run]
    return entry.error


def summary(entry):
    """current 줄: "Saved · learn_learn_rule · model_v01" / "Training · learn_learn_rule_20261104_220048 · checkpoint_000300"."""
    if entry.source == "training":
        return f"Training · {entry.group} · {os.path.basename(os.path.dirname(entry.checkpoint))}"
    return f"Saved · {entry.group} · {entry.name}"
