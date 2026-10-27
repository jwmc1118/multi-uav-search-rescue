"""평가 결과 저장 — 결과 폴더 (results/<kind>/<YYYYMMDD_HHMMSS>_<tag>/), CSV / JSON, LaTeX (booktabs) 표, git commit 해시."""
import csv
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_run_dir(kind, tag):
    """results/<kind>/<YYYYMMDD_HHMMSS>_<tag>/ 를 만들고 경로를 돌려준다."""
    d = os.path.join(ROOT, "results", kind, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    os.makedirs(d, exist_ok=True)
    return d


def git_info():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def write_csv(path, rows, fields=None):
    """rows = [dict]. fields 를 안 주면 모든 행의 키 (처음 나온 순서)."""
    if not rows:
        open(path, "w").close()
        return path
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item") and callable(o.item):
        return o.item()
    if isinstance(o, float) and o != o:
        return None
    return o


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)
    return path


def latex_table(path, header, rows, col_align=None, caption=None, label=None):
    """booktabs 표. header = [열 이름], rows = [[셀 문자열]] (굵게는 호출부가 \\textbf{...})."""
    col_align = col_align or ("l" + "r" * (len(header) - 1))
    lines = ["\\begin{table}[t]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += [f"\\begin{{tabular}}{{{col_align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
