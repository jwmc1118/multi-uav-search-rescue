"""논문용 그림 스타일 (IEEE) — rule / scale 평가 그림이 같이 쓴다.

    apply_style()                 serif (Times 계열 -> Nimbus Roman -> STIX), mathtext stix, 축 라벨 8.5 pt, 눈금 7.5 pt,
                                  위 · 오른쪽 테두리 없음, 얇은 격자
    figure(width="single"|"double", height)   단단 3.5 in / 양단 7.16 in
    POLICY_STYLE                  정책 색 (Okabe–Ito 기반, 규칙 = 회색, 모델 = 파랑) + 마커 — 흑백 인쇄에서도 구분
    save(fig, path_no_ext, data_rows)   PDF (벡터) + PNG (300 dpi) + 원자료 CSV (같은 이름)
그림 안에는 제목을 넣지 않는다 (캡션은 논문에서).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_common.io import write_csv  # noqa: E402

WIDTH = {"single": 3.5, "double": 7.16}
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F"}
POLICY_STYLE = {
    "rule": {"color": OKABE_ITO["gray"], "marker": "s", "linestyle": "--", "label": "Rule (greedy)"},
    "model": {"color": OKABE_ITO["blue"], "marker": "o", "linestyle": "-", "label": "Learned"},
}
ROLE_COLORS = {"explore": OKABE_ITO["skyblue"], "rescue": OKABE_ITO["orange"], "return": OKABE_ITO["gray"]}


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.4, "grid.linestyle": "-",
        "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 2,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(width="single", height=2.3, ncols=1, nrows=1, **kw):
    apply_style()
    kw.setdefault("constrained_layout", True)                # 패널 라벨 · 눈금이 겹치지 않게
    return plt.subplots(nrows, ncols, figsize=(WIDTH[width], height), **kw)


def save(fig, path_no_ext, data_rows=None, dpi=300):
    """PDF + PNG + (원자료가 있으면) 같은 이름의 CSV. 반환: 저장한 파일 목록."""
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    out = [path_no_ext + ".pdf", path_no_ext + ".png"]
    fig.savefig(out[0])
    fig.savefig(out[1], dpi=dpi)
    plt.close(fig)
    if data_rows is not None:
        out.append(write_csv(path_no_ext + ".csv", data_rows))
    return out
