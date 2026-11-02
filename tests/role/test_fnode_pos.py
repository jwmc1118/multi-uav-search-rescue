"""F 노드 위치 특징 점검 (ray 불필요).

    python -m tests.role.test_fnode_pos

[1] 빈 맵 직선 frontier 덩어리 하나를 두 UAV 가 같은 그룹에서 볼 때, 각 UAV 관측의 k_pos = 자기에게 BFS 로 가장 가까운 셀
    (양 끝 셀이 서로 다름), 동률이면 셀 인덱스 (row, col) 가 작은 쪽. fallback (도달 가능 셀 없음) 0
"""
import numpy as np

from tests.role.test_explore import blank_sim, line, place, set_front, tick


def check_nearest_cell():
    sim = blank_sim(2)
    place(sim, [(15, 2), (15, 37)])                       # 덩어리 = 15 행 5..34 열, 두 UAV 가 양 끝 바깥
    set_front(sim, line(15, 5, 34))
    tick(sim)
    obs = sim.rl_observation()
    got = [tuple(obs["agents"][i]["k_pos"][0]) for i in range(2)]
    want = [tuple(np.float32(sim.frame.normalized(c))) for c in ((15, 5), (15, 34))]
    assert got == want and got[0] != got[1], (got, want)
    assert sim._fpos_last["fallback"] == 0 and sim._fpos_last["nodes"] == 2, sim._fpos_last

    place(sim, [(10, 3), (15, 37)])                       # (15, 5..8) 모두 BFS 5 (8-연결) -> 인덱스가 작은 (15, 5)
    tick(sim)
    assert sim._dist(0, (15, 5)) == sim._dist(0, (15, 8)) == 5
    assert sim._nearest_cell(0, sim._blob_cells_for_split(sim.comp[0], sim.clusters_of(0)[0])) == (15, 5)
    assert sim._nearest_cell(0, []) is None
    print(f"[1] same blob, two UAVs: F position R0 {(15, 5)} / R1 {(15, 34)} (own nearest cell, normalized {got}); "
          f"tie (BFS 5 for cols 5..8) -> smallest index (15, 5); fallback 0")


if __name__ == "__main__":
    check_nearest_cell()
    print("test_fnode_pos: all ok")
