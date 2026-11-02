"""explore 1 단계 차례 (점수 (N_m / (n_m + 1)) / (d_jm + 1)^α) · fallback · 2 단계 거리 Hungarian 점검 — ray 불필요.

    python -m tests.role.test_explore_order

 [1] explore_front 단위: UAV 1 이 큰 덩어리 A (100 칸) 를 맡은 상태에서 UAV 2 차례, A 거리 10 · 작은 빈 덩어리 B (6 칸) 거리 5 ->
     α = 1 이면 A (50 / 11 > 6 / 6, 빈 덩어리 우선 없음). 같은 덩어리면 가장 가까운 UAV 가 차례. α 를 키우면 가까운 덩어리 쪽으로.
     동률은 큰 덩어리 -> 슬롯 순 -> 거리 -> UAV 번호
 [2] env 장면 (그리디): UAV 옆 8 칸 덩어리 vs 먼 30 칸 덩어리 -> α 0 이면 큰 덩어리, α 1 이면 가까운 덩어리
 [3] rule_explore fallback (담당 후보 없음): 점수 최대 덩어리 (α 1 -> 먼 큰 덩어리, α 2 -> 가까운 작은 덩어리)
 [4] 2 단계: 비용 행렬 = 구간까지 BFS 최소 거리, 배정 = 총 거리 합 최소 (전수 비교), 기존 goal 이 든 구간 우대 없음
"""
import itertools
import os

import numpy as np
import torch

from env.core.modes import Mode
from policies.decision_order import explore_front
from tests.role.test_events_targeted import fresh
from tests.role.test_explore import line, place, set_front, tick

E = Mode.EXPLORE.value
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def front(alpha, d_cells, size, n, dmax=100.0):
    """UAV 1 대 (또는 여러 대, 한 그룹) · 덩어리 K 개 텐서로 explore_front."""
    d = torch.as_tensor(np.asarray(d_cells, np.float32) / dmax)[None]              # (1, N, K)
    N, K = d.shape[1:]
    sz = torch.as_tensor(np.tile(np.asarray(size, np.float32), (N, 1)))[None]
    nn_ = torch.as_tensor(np.tile(np.asarray(n, np.float32), (N, 1)))[None]
    group = torch.ones(1, N, N, dtype=torch.bool)
    rem = torch.ones(1, N, dtype=torch.bool)
    emask = torch.ones(1, N, K, dtype=torch.bool)
    sel, claim = explore_front(group, rem, emask, d, sz, nn_, alpha, torch.tensor([dmax]))
    j = int(np.flatnonzero(sel[0].numpy())[0])
    return j, int(claim[0, j])


def check_front_unit():
    # spec: UAV 1 이 A (100 칸) 를 맡음 (n_A = 1), UAV 2 차례: A 거리 10, B (6 칸, 빈 덩어리) 거리 5
    assert front(1.0, [[10, 5]], [100, 6], [1, 0])[1] == 0                       # A: 50 / 11 = 4.55 > B: 6 / 6 = 1
    # 같은 덩어리 하나, UAV 2 대 -> 가까운 UAV (거리 3) 가 차례
    assert front(1.0, [[7], [3]], [20], [0]) == (1, 0)
    # α 를 키우면 가까운 쪽: 덩어리 0 (30 칸, 40) / 1 (10 칸, 2)
    got = {a: front(a, [[40, 2]], [30, 10], [0, 0])[1] for a in (0.0, 0.2, 1.0, 2.0)}
    assert got == {0.0: 0, 0.2: 0, 1.0: 1, 2.0: 1}, got
    # 동률: 점수 같으면 큰 덩어리 (덩어리 0: 20 칸 / n 1 -> 10, 덩어리 1: 10 칸 / n 0 -> 10, 거리 같음)
    assert front(1.0, [[4, 4]], [20, 10], [1, 0])[1] == 0
    # 점수 · 크기 같음 -> 슬롯 순, 같은 덩어리 같은 거리 -> 번호 작은 UAV
    assert front(1.0, [[3, 3]], [10, 10], [0, 0]) == (0, 0)
    assert front(1.0, [[5, 9], [5, 9]], [10, 10], [0, 0]) == (0, 0)
    print(f"[1] explore_front: A (100 cells, held by 1, d 10) vs empty B (6 cells, d 5) -> A (50/11 > 6/6); same blob -> nearest "
          f"UAV; claimed blob by α {got} (larger α -> nearer); ties -> larger blob, slot, distance, index")


def two_blob_sim(alpha, near=8):
    sim = fresh(1, explore={"order_dist_alpha": alpha})
    sim.time = 0
    place(sim, [(10, 3)])
    front_mask = line(10, 5, 4 + near) | line(27, 5, 34)              # 가까운 작은 덩어리 (near 칸) / 먼 큰 덩어리 (30 칸)
    set_front(sim, front_mask)
    tick(sim)
    return sim


def blob_of(sim, i):
    b = sim.blob_index[sim.comp[i]][sim.agents[i].blob]
    return "near" if len(b.cell_set()) < 30 else "far"


def check_env_scene():
    from tests.role.test_decide_order import run_decide
    out = {}
    for a in (0.0, 1.0):
        sim = two_blob_sim(a)
        _, s1 = run_decide(sim, role={0: E}, ev="periodic")
        out[a] = (blob_of(sim, 0), s1)
    assert out[0.0][0] == "far" and out[1.0][0] == "near", out
    print(f"[2] env greedy scene (UAV next to an 8-cell blob, 30-cell blob far away): α 0 -> {out[0.0][0]} (large) blob, "
          f"α 1 -> {out[1.0][0]} blob")


def check_fallback():
    from policies.role_rules import rule_explore
    res, scores = {}, {}
    for a in (1.0, 2.0):
        sim = two_blob_sim(a, near=3)                                 # 가까운 3 칸 (d 2) / 먼 30 칸 (d 17)
        sim.agents[0].mode = Mode.EXPLORE
        sim.cur_xclaim = {}
        slots = sim.clusters_of(0)
        k = rule_explore(sim, 0, sim.explore_mask(0))
        res[a] = "near" if len(slots[k].cell_set()) == 3 else "far"
        sc = {("near" if len(b.cell_set()) == 3 else "far"): len(b.cell_set()) / (sim.blob_dist(0, b) + 1) ** a for b in slots}
        scores[a] = {k_: round(v, 3) for k_, v in sc.items()}
        assert res[a] == max(sc, key=sc.get), (a, res, sc)
    assert res == {1.0: "far", 2.0: "near"}, res
    print(f"[3] rule_explore fallback (no claim): best score blob -> α 1 {res[1.0]} {scores[1.0]}, α 2 {res[2.0]} {scores[2.0]}")


def check_stage2_distance():
    from policies.role_rules import distance_hungarian, seg_distance
    sim = fresh(3)
    sim.time = 0
    place(sim, [(14, 4), (14, 20), (14, 34)])
    set_front(sim, line(15, 5, 34))
    tick(sim)
    cells = sorted(sim.clusters_of(0)[0].cell_set(), key=lambda c: c[1])
    segs = [(frozenset(cells[0:10]), None), (frozenset(cells[10:20]), None), (frozenset(cells[20:30]), None)]
    members = [0, 1, 2]
    D = np.array([[seg_distance(sim, i, segs[s][0]) for s in range(3)] for i in members], float)
    for i in members:                                                  # BFS 최소 거리와 같은지
        for s in range(3):
            assert D[i, s] == min(int(sim.fields[i][c]) for c in segs[s][0])
    best = min(itertools.permutations(range(3)), key=lambda p: sum(D[i, p[i]] for i in members))
    got = distance_hungarian(sim, members, segs)
    assert sum(D[i, got[i]] for i in members) == sum(D[i, best[i]] for i in members), (got, best)
    # goal 우대 없음: R0 (14, 15) · R1 (14, 16), 구간 왼쪽 (열 5-19) / 오른쪽 (20-34) -> 거리로 R0 왼쪽 · R1 오른쪽 (1 + 4).
    # R0 의 goal 을 오른쪽 구간에 두어도 배정이 그대로 (goal 구간 우대 없음)
    sim2 = fresh(2)
    sim2.time = 0
    place(sim2, [(14, 15), (14, 16)])
    set_front(sim2, line(15, 5, 34))
    tick(sim2)
    c2 = sorted(sim2.clusters_of(0)[0].cell_set(), key=lambda c: c[1])
    segs2 = [(frozenset(c2[:15]), None), (frozenset(c2[15:]), None)]
    base = distance_hungarian(sim2, [0, 1], segs2)
    sim2.agents[0].goal = (15, 30)
    got2 = distance_hungarian(sim2, [0, 1], segs2)
    assert base == got2 == {0: 0, 1: 1}, (base, got2)
    print(f"[4] stage 2: cost = BFS min distance {D.astype(int).tolist()}, assignment {got} = min total "
          f"{int(sum(D[i, got[i]] for i in members))}; own goal in the other segment -> assignment unchanged {got2} (no goal bonus)")


if __name__ == "__main__":
    check_front_unit()
    check_env_scene()
    check_fallback()
    check_stage2_distance()
    print("test_explore_order: all ok")
