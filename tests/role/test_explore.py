"""explore 2 단계 단위 점검 (ray 불필요) — 빈 맵에 frontier / 통신 연결을 직접 놓고 env 파이프라인을 호출.
(이벤트 체계 시나리오는 tests/role/test_events_targeted.py)

    python -m tests.role.test_explore

 [1] 구간 분할: L 자 덩어리 k=2 -> 가로 / 세로 두 연속 구간 (측지 거리), 결정성 (입력 순서 무관), medoid 는 실제 frontier 셀,
     도달 불가 frontier 제외 (A-1)
 [7] 매 step rebalance: 재분할 + 매칭 -> 같은 쪽 유지 · 크기 균형 · 연결 성분 1 개, |C| < m 이면 남는 UAV 이벤트 5
 [8] 이동 목표점 동률: BFS 거리가 같으면 유클리드 거리가 짧은 셀 (오른쪽 같은 행 vs 오른쪽 위 대각선 -> 같은 행), 그다음 (행, 열)
"""
import numpy as np

from env.core.modes import Mode
from env.role.clusters import frontier_clusters, partition, geodesic_matrix
from env.role.simulation import RoleSimulation
from policies.role_rules import rule_explore_sub
from training.config_loader import load_config

ROWS, COLS = 30, 40


def blank_sim(n=3, **over):
    cfg = load_config()["env_config"]
    cfg.update(map_mode="file", map_path=None, rows=ROWS, cols=COLS, num_targets=[1, 1], target_mode="random",
               agents=[dict(cfg["agents"][0]) for _ in range(n)])
    for k, v in over.items():                                            # 예: role={...}, events={...}
        cfg[k] = dict(cfg.get(k) or {}, **v)
    sim = RoleSimulation(cfg)
    sim.random_state = np.random.RandomState(0)
    sim.reset()
    for t in sim.targets:
        t.discovered_at = None
    K = sim.knowledge
    for i in range(n):
        K.known_free[i][:] = True
        K.known_obstacle[i][:] = False
    connect(sim)
    for a in sim.agents:
        a.mode, a.blob, a.cluster, a.goal = Mode.EXPLORE, None, None, None
    sim._known = [set() for _ in sim.agents]
    sim._members_prev = [frozenset(range(n)) for _ in range(n)]
    return sim


def park(sim, i):
    """시나리오와 무관한 UAV: 용량 0 + RETURN (재결정돼도 RETURN 유지)."""
    sim.agents[i].mode, sim.agents[i].capacity = Mode.RETURN, 0


def connect(sim, groups=None):
    n = sim.n_agents
    groups = groups or [list(range(n))]
    R = np.zeros((n, n), bool)
    for g in groups:
        for i in g:
            for j in g:
                R[i, j] = i != j
    sim.knowledge.reach = R


def set_front(sim, mask):
    for i in range(sim.n_agents):
        sim.knowledge.frontier[i] = mask.copy()


def line(r, c0, c1):
    m = np.zeros((ROWS, COLS), bool)
    m[r, c0:c1 + 1] = True
    return m


def tick(sim):
    """지도 갱신 직후 처리 (덩어리 / 대응 / 추정 담당 / 구간 추적 / 슬롯). 반환: 소진 {i: 이유}."""
    prev_state = [(a.mode, a.blob, a.cluster) for a in sim.agents]
    prev_seg = [a.cluster for a in sim.agents]
    sim._prepare_role()
    return sim._explore_after_update(prev_state, prev_seg)


def decide(sim, parties=(), role=None, explore=None, ev="2"):
    """parties 에 Role 이벤트 ev 를 주고 결정. role / explore: {i: 값} (없으면 지금 역할 / 슬롯 0). 반환: (2 단계 호출 순서, info)."""
    role, explore = role or {}, explore or {}
    sim.events = [{ev} if i in parties else set() for i in range(sim.n_agents)]
    sim._set_decide_order()
    sim.mask_violation_step = 0
    calls = []

    def sub(i, m):                                                      # 2 단계는 그리디 (κ Hungarian)
        calls.append(i)
        return rule_explore_sub(sim, i, m)
    rv, sv, _, viol = sim._decide(lambda i, m: role.get(i, sim.agents[i].mode.value),   # 지정 없으면 지금 역할
                                  lambda i, m: int(np.flatnonzero(m)[0]),
                                  lambda i, m: explore.get(i, 0), sub)
    assert viol == 0 and sim.mask_violation_step == 0
    return calls, sv


def check_partition():
    L = [(10, c) for c in range(0, 20)] + [(r, 0) for r in range(11, 30)]          # L 자 (꺾이는 칸 = (10, 0))
    segs = partition(L, 2)
    assert len(segs) == 2
    frac = lambda cells, f: sum(1 for c in cells if f(c)) / len(cells)
    rowseg = [s for s in segs if frac(s[0], lambda c: c[0] == 10) >= 0.8]          # 꺾이는 칸 근처 몇 칸은 어느 쪽이든
    colseg = [s for s in segs if frac(s[0], lambda c: c[1] == 0 and c[0] > 10) >= 0.8]
    assert rowseg and colseg, [(len(s[0]), s[1]) for s in segs]
    for cells, med in segs:
        assert med in cells, "medoid 는 실제 frontier 셀"
        D = geodesic_matrix(sorted(cells))
        assert np.isfinite(D).all(), "구간은 측지적으로 연속"
    rng = np.random.RandomState(0)
    shuffled = [L[k] for k in rng.permutation(len(L))]
    assert partition(shuffled, 2) == segs and partition(L, 2) == segs, "결정적이어야 함"
    front = line(5, 2, 20)
    reach = np.ones((ROWS, COLS), bool)
    reach[5, 15:21] = False                                              # 오른쪽 끝 6 칸 도달 불가
    bl = frontier_clusters(front, reachable=reach)
    assert len(bl) == 1 and bl[0].size == 13 and all(c[1] < 15 for c in bl[0].cell_set())
    print(f"[1] L blob k=2 -> segments {sorted(len(s[0]) for s in segs)} (row arm / column arm, geodesic-contiguous), "
          f"deterministic, medoids on frontier; unreachable frontier excluded (19 -> 13 cells)")


def setup_line(sim, agents=(0,), front=None):
    """직선 frontier 덩어리 하나에 agents 를 차례로 배정 (1 단계)."""
    set_front(sim, line(15, 5, 34) if front is None else front)
    tick(sim)
    calls = []
    for i in agents:
        c, _ = decide(sim, parties=[i])
        calls.append(c)
        sim.time += 3
        tick(sim)
    return calls


def place(sim, pos):
    for i, p in enumerate(pos):
        sim.agents[i].pos = p


def check_rebalance():
    """매 step rebalance: 덩어리 전체를 담당자 수로 재분할 (warm start + 헝가리안) -> 같은 쪽 유지, 크기 균형,
    연결 성분 1 개. 셀이 담당자보다 적으면 남는 UAV 는 덩어리에서 빠짐 (이벤트 6), 덩어리가 사라지면 전원 이벤트 6."""
    from scipy.ndimage import label
    sim = blank_sim(4)
    place(sim, [(14, 3), (14, 20), (14, 36), (25, 20)])
    park(sim, 3)
    setup_line(sim, agents=(0, 1, 2))
    col = lambda i: np.mean([c[1] for c in sim.agents[i].cluster])
    order0 = sorted(range(3), key=col)
    comps = lambda cells: label(np.isin(np.arange(COLS), [c[1] for c in cells])[None, :] if len({c[0] for c in cells}) == 1
                                else np.zeros((1, 1)), structure=np.ones((3, 3)))[1]
    for shift in (16, 17, 18):                                           # frontier 가 계속 물러남 (셀은 모두 새것)
        set_front(sim, line(shift, 5, 34))
        ex = tick(sim)
        assert not ex
        sizes = [len(sim.agents[i].cluster) for i in range(3)]
        assert sorted(range(3), key=col) == order0, "구간 <-> UAV 매칭이 뒤바뀜"
        assert np.std(sizes) / np.mean(sizes) <= 0.25 and sum(sizes) == 30, sizes
        assert all(comps(sim.agents[i].cluster) == 1 for i in range(3))
        assert all(sim.agents[i].goal in sim.agents[i].cluster for i in range(3))
    set_front(sim, line(18, 5, 6))                                       # 셀 2 개 < 담당자 3
    for _ in range(int(sim.events_cfg["no_segment_steps"])):             # 구간 없는 상태가 no_segment_steps 연속이면
        ex = tick(sim)
    ev = sim._collect_events([a.pos for a in sim.agents], set(), ex)
    lost = [i for i in range(3) if ex.get(i) == "no segment"]
    assert len(lost) == 1 and "5" in ev[lost[0]], (ex, ev)
    set_front(sim, np.zeros((ROWS, COLS), bool))                         # 덩어리 소멸
    ex = tick(sim)
    assert all(i in ex for i in range(3) if i not in lost)
    print(f"[7] rebalance: frontier retreat x3 -> same left/middle/right order, sizes {sizes} (cv <= 0.25), 1 component each; "
          f"|C| < m for 3 steps -> R{lost[0]} event 5; blob gone -> all event 5")


def check_goal_tie():
    sim = blank_sim(2)
    place(sim, [(15, 10), (25, 20)])
    park(sim, 1)
    tick(sim)
    same_row, diag = (15, 14), (13, 14)                                 # 둘 다 BFS 4 (8-연결), 유클리드 4 vs sqrt(20)
    assert sim._dist(0, same_row) == sim._dist(0, diag) == 4
    g = sim._segment_goal(0, frozenset([same_row, diag]))
    assert g == same_row, g
    up, down = (11, 14), (19, 14)                                       # BFS · 유클리드 모두 같음 -> (행, 열) 사전순
    assert sim._segment_goal(0, frozenset([down, up])) == up
    print(f"[8] goal tie: same-row {same_row} chosen over diagonal {diag} (both BFS 4); full tie -> lexicographic {up}")


if __name__ == "__main__":
    check_partition()
    check_rebalance()
    check_goal_tie()
    print("test_explore: all ok")
