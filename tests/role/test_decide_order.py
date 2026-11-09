"""결정 순서 (policies/decision_order.py) + 그리디 배정 + 학습 경로 점검.

    python -m tests.role.test_decide_order

role_order (B-1)
 [1] 생존자 A (급함) 근처 R1, R2 / 생존자 B 근처 R4 / 멀리 R0, R3 -> 차례 R1 -> R4 -> 나머지 (가까운 생존자 거리 -> 용량 -> 번호)
 [2] R1 이 A, B 모두에 가장 가까우면 A 가 R1, B 는 그다음으로 가까운 UAV
 [3] 거리 동률 -> 용량 많은 UAV, 용량 동률 -> 번호 작은 UAV / 용량 0 UAV 는 담당 후보가 아니고 맨 뒤
 [4] 생존자 5 명, UAV 2 대 -> 가장 급한 두 생존자만 담당 후보
 [5] σ 는 이벤트 시작 D_0 기준 고정: 가까운 UAV 가 먼저 빠져도 생존자 순서가 바뀌지 않음
 [6] 학습 경로: 맨 앞 UAV 가 EXPLORE / 다른 생존자를 고르면 비어 있는 생존자의 가장 가까운 UAV 가 바로 다음 차례
 [7] 그리디: [1] 배치에서 R1 -> A, R4 -> B RESCUE, 나머지 EXPLORE / GUI 장면 (R0 멀리, R4 바로 옆) 은 R4 가 RESCUE
explore_order (B-3, 점수 (N_m / (n_m + 1)) / (d + 1)^α 최대 쌍, α = 1 — 단위 점검은 tests/role/test_explore_order.py)
 [8] 덩어리 (10, 9, 8) · UAV 5 대 / (50, 6, 4) · 5 대 / 4 개 · 2 대 / 먼 2 셀 덩어리: env 그리디의 1 단계 차례 · 덩어리가 점수 규칙을
     따로 계산한 결과와 같다 (한 대마다 n 갱신)
 [9] 기존 배정자가 있는 덩어리는 n 에 포함 (재결정하지 않는 기존 배정자)
[10] 학습 경로: 맨 앞 UAV 가 담당 후보 대신 다른 덩어리를 고르면 실제 선택으로 n_m 이 갱신되어 다음 차례가 바뀜
재결정 (EXPLORE 를 고른 재결정 UAV 는 전원 1 단계, 같은 덩어리 재선택 = 새 표본 아님)
[12] 이벤트 "1" 에서 R1 -> RESCUE, R2 / R3 EXPLORE: 1 단계에 참여해 같은 덩어리를 다시 고름 -> explore_valid 0 (표본 아님),
     explore keep 2 / 2
[13] R1 이 빠진 영역은 rebalance 로 다시 나뉘고 남은 UAV 는 자기 goal 쪽 구간 유지
[14] 구조를 마치고 EXPLORE 로 돌아온 UAV 는 1 단계 참여
[15] 새 UAV 가 R1~R3 의 덩어리에 합류하고 R1~R3 도 같은 step 에 재결정 -> 넷 다 1 단계 · 2 단계, 새 표본은 R0 만
학습 모델
[11] 정책 rollout 에서 모델이 디코딩한 차례 (role / 1 단계 / 2 단계) == env 가 재현한 차례 (모든 step), 앞 차례 결정이 Context 에
"""
import numpy as np

from env.core.modes import Mode
from policies.role_rules import RuleDecider
from tests.role.test_events_targeted import add_victim, fresh
from tests.role.test_explore import line, place, set_front, tick, ROWS, COLS

E, R, D = Mode.EXPLORE.value, Mode.RESCUE.value, Mode.RETURN.value


def run_decide(sim, role=None, rescue=None, explore=None, ev="1", parties=None):
    """parties (기본 전원) 에 이벤트 ev, role / rescue / explore = {UAV: 값} 또는 None (None 이면 그리디).
    rescue 값은 생존자 번호 (그 UAV 슬롯으로 바꿈). 반환 (role 차례, s1 차례)."""
    g = RuleDecider()
    parties = range(sim.n_agents) if parties is None else parties
    sim.events = [{ev} if i in parties else set() for i in range(sim.n_agents)]
    sim._set_decide_order()
    sim.mask_violation_step = 0

    def role_fn(i, m):
        r = (role or {}).get(i)
        return r if r is not None and m[r] else g.role(sim, i, m)

    def rescue_fn(i, m):
        u = (rescue or {}).get(i)
        if u is not None and u in sim.victim_slots[i] and m[sim.victim_slots[i].index(u)]:
            return sim.victim_slots[i].index(u)
        return g.rescue(sim, i, m)

    def explore_fn(i, m):
        k = (explore or {}).get(i)
        return k if k is not None and m[k] else g.explore(sim, i, m)
    _, sv, _, viol = sim._decide(role_fn, rescue_fn, explore_fn, lambda i, m: g.explore_sub(sim, i, m))
    assert viol == 0 and sim.mask_violation_step == 0
    sim._last_sv = sv
    return [j for st, j in sim.decision_seq if st == "role"], [j for st, j in sim.decision_seq if st == "s1"]


def scene1():
    sim = fresh(5)
    sim.time = 0
    place(sim, [(15, 20), (5, 8), (5, 10), (20, 5), (25, 32)])      # R0 멀리, R1 / R2 는 A 근처, R3 멀리, R4 는 B 근처
    set_front(sim, line(28, 10, 30))
    a = add_victim(sim, (5, 5), deadline=20)                          # A: σ = 20 - 3 = 17 (급함)
    b = add_victim(sim, (25, 35), deadline=100)                       # B: σ = 97
    tick(sim)
    return sim, a, b


def check_role_basic():
    sim, a, b = scene1()
    order, _ = run_decide(sim)
    assert order == [1, 4, 0, 2, 3] and sim.cur_claim[1] == a and sim.cur_claim[4] == b, (order, sim.cur_claim)
    assert all(sim.cur_claim.get(i) is None for i in (0, 2, 3))
    print(f"[1] order {order}: R1 -> A, R4 -> B, then rest (no free survivor left -> capacity -> index)")


def check_shared_nearest():
    sim = fresh(5)
    sim.time = 0
    place(sim, [(25, 5), (5, 10), (5, 22), (25, 20), (25, 35)])
    set_front(sim, line(28, 10, 30))
    a = add_victim(sim, (5, 5), deadline=20)                          # R1 d 5
    b = add_victim(sim, (5, 15), deadline=100)                        # R1 d 5, R2 d 7
    tick(sim)
    order, _ = run_decide(sim)
    assert order[:2] == [1, 2] and sim.cur_claim[1] == a and sim.cur_claim[2] == b, (order, sim.cur_claim)
    print(f"[2] R1 nearest to both: A (urgent) -> R1, B -> next nearest R2; order {order}")


def check_ties_and_zero_cap():
    sim = fresh(4)
    sim.time = 0
    place(sim, [(5, 9), (5, 1), (5, 6), (25, 35)])                    # A 까지 R0 d 4, R1 d 4, R2 d 1 (용량 0)
    set_front(sim, line(28, 10, 30))
    a = add_victim(sim, (5, 5), deadline=40)
    tick(sim)
    sim.agents[2].capacity = 0
    sim.agents[0].capacity, sim.agents[1].capacity = 2, 3
    order, _ = run_decide(sim)
    assert order[0] == 1 and order[-1] == 2 and sim.cur_claim[1] == a and sim.cur_claim.get(2) is None, (order, sim.cur_claim)
    sim = fresh(4)
    sim.time = 0
    place(sim, [(5, 9), (5, 1), (5, 6), (25, 35)])
    set_front(sim, line(28, 10, 30))
    a = add_victim(sim, (5, 5), deadline=40)
    tick(sim)
    sim.agents[2].capacity = 0
    sim.agents[0].capacity, sim.agents[1].capacity = 2, 2
    order2, _ = run_decide(sim)
    assert order2[0] == 0 and order2[-1] == 2, order2
    print(f"[3] tie d 4: capacity 3 > 2 -> R1 first {order}; same capacity -> lower index R0 {order2}; capacity-0 R2 "
          f"(closest, d 1) never a claim, last")


def check_more_victims():
    sim = fresh(2)
    sim.time = 0
    place(sim, [(10, 10), (20, 30)])
    set_front(sim, line(28, 10, 30))
    vs = [add_victim(sim, p, deadline=dl) for p, dl in (((10, 14), 40), ((12, 12), 30), ((20, 26), 35), ((5, 5), 90),
                                                        ((25, 35), 80))]
    tick(sim)
    sig = {v: min(dl - sim._dist(i, sim.targets[v].pos) for i in range(2)) for v, dl in zip(vs, (40, 30, 35, 90, 80))}
    urgent = sorted(vs, key=lambda v: (sig[v], v))[:2]
    run_decide(sim)
    assert sorted(sim.cur_claim.values()) == sorted(urgent), (sim.cur_claim, urgent, sig)
    print(f"[4] 5 survivors, 2 UAVs: claims {sim.cur_claim} = two most urgent (σ {[sig[v] for v in urgent]})")


def fixed_sigma_scene():
    sim = fresh(5)
    sim.time = 0
    place(sim, [(2, 35), (10, 10), (10, 20), (25, 10), (25, 30)])
    set_front(sim, line(28, 12, 20))
    a = add_victim(sim, (10, 13), deadline=13)                        # R1 d 3 -> σ 10, R2 d 7
    c = add_victim(sim, (10, 7), deadline=30)                         # R1 d 3 -> σ 27 (R1 이 빠지면 R2 d 13 -> 17)
    b = add_victim(sim, (25, 33), deadline=23)                        # R4 d 3 -> σ 20
    tick(sim)
    return sim, a, b, c


def check_fixed_sigma():
    sim, a, b, c = fixed_sigma_scene()
    order, _ = run_decide(sim, role={1: E, 2: E})                      # A 의 담당 R1, 그다음 R2 가 EXPLORE
    assert order[:3] == [1, 2, 4], order                               # σ 고정: B (20) 가 C (27) 보다 먼저 -> R4
    print(f"[5] σ fixed at event start: R1 and R2 (A's nearest) pick EXPLORE -> next is R4 for B (σ 20) "
          f"not C (recomputed σ would be 17): order {order}")


def check_learned_path():
    sim, a, b = scene1()
    order, _ = run_decide(sim, role={1: E})
    assert order[:2] == [1, 2] and sim.cur_claim[2] == a, (order, sim.cur_claim)
    sim, a, b = scene1()
    order2, _ = run_decide(sim, role={1: R}, rescue={1: b})            # R1 이 B 를 고름 -> A 는 비어 있음
    assert order2[:2] == [1, 2] and sim.cur_claim[2] == a and sim.agents[1].assigned_target == b, (order2, sim.cur_claim)
    print(f"[6] learned path: front R1 picks EXPLORE -> A's next nearest R2 is next {order[:2]}; R1 rescues B instead -> "
          f"R2 next for A {order2[:2]}")


def check_greedy_assign():
    sim, a, b = scene1()
    run_decide(sim)
    modes = [x.mode for x in sim.agents]
    assert modes[1] is Mode.RESCUE and sim.agents[1].assigned_target == a
    assert modes[4] is Mode.RESCUE and sim.agents[4].assigned_target == b
    assert all(modes[i] is Mode.EXPLORE for i in (0, 2, 3)), modes
    st = sim.ev_stats
    assert st["ev1_rescue"] == 2 and st["ev1_rescue_by_claim"] == 2 and st["claim_not_rescue"] == 0
    sim = fresh(5)                                                    # GUI 장면: R0 멀리 (번호 작음), R4 바로 옆
    sim.time = 0
    place(sim, [(5, 5), (28, 2), (28, 4), (28, 6), (20, 30)])
    set_front(sim, line(10, 10, 30))
    v = add_victim(sim, (20, 31), deadline=150)
    tick(sim)
    order, _ = run_decide(sim)
    assert sim.agents[4].mode is Mode.RESCUE and sim.agents[4].assigned_target == v and sim.agents[0].mode is Mode.EXPLORE
    print(f"[7] greedy: R1 -> A, R4 -> B RESCUE, R0 / R2 / R3 EXPLORE (claim match 2/2); GUI scene: R4 (adjacent) RESCUE, "
          f"far R0 EXPLORE (order {order})")


ALPHA = 1.0


def blob_scene(sizes, n_uav, uav_pos, far=None):
    """덩어리 = 서로 떨어진 가로 줄 (크기 sizes). 반환 sim, 줄 첫 셀 목록. 1 단계 차례 α = ALPHA."""
    sim = fresh(n_uav, explore={"order_dist_alpha": ALPHA})
    sim.time = 0
    place(sim, uav_pos)
    front = np.zeros((ROWS, COLS), bool)
    starts = []
    rows = [3, 9, 15, 21, 27]
    for k, s in enumerate(sizes):
        r = rows[k]
        c0 = 2 if far is None or k != far else COLS - 1 - s
        if s > 30:                                                    # 큰 덩어리는 두 줄
            front |= line(r, 2, 31) | line(r + 1, 2, 2 + s - 31)
        else:
            front |= line(r, c0, c0 + s - 1)
        starts.append((r, c0))
    set_front(sim, front)
    tick(sim)
    return sim, starts


def blob_of(sim, i, starts):
    cells = sim.blob_index[sim.comp[i]][sim.agents[i].blob].cell_set()
    return next(k for k, st in enumerate(starts) if st in cells)


def expected_turns(sim, uavs, forced=None):
    """점수 규칙을 따로 계산: 남은 UAV x 닿는 덩어리 쌍 중 (N / (n + 1)) / (d + 1)^α 최대 (동률 큰 덩어리 -> 슬롯 -> 거리 -> 번호),
    고른 덩어리 n += 1 (n 초기값 = 재결정하지 않는 기존 배정자). forced = {차례 번호: 슬롯} (학습 경로처럼 다른 덩어리를 고름).
    반환 [(UAV, 슬롯)]."""
    g = sim.comp[uavs[0]]
    slots = sim.group_clusters[g]
    n = {b.id: sum(1 for j in sim.groups[g] if j not in uavs and sim.agents[j].mode is Mode.EXPLORE and sim.agents[j].blob == b.id)
         for b in slots}
    rem, out = list(uavs), []
    while rem:
        best = None
        for j in rem:
            m = sim.explore_mask(j)
            for k, b in enumerate(slots):
                if not m[k]:
                    continue
                d = sim.blob_dist(j, b)
                size = len(b.cell_set())
                score = (size / (n[b.id] + 1)) / (d + 1) ** ALPHA
                key = (-score, -size, k, d, j)
                if best is None or key < best[0]:
                    best = (key, j, k)
        _, j, k = best
        k = (forced or {}).get(len(out), k)
        out.append((j, k))
        n[slots[k].id] += 1
        rem.remove(j)
    return out


def actual_turns(sim):
    slots = {b.id: k for k, b in enumerate(sim.group_clusters[0])}
    return [(j, slots[sim.agents[j].blob]) for s, j in sim.decision_seq if s == "s1"]


def check_explore_order():
    pos5 = [(5, 20), (11, 20), (17, 20), (23, 20), (13, 30)]
    out = []
    for sizes, n_uav, pos, far in (([10, 9, 8], 5, pos5, None), ([50, 6, 4], 5, pos5, None),
                                   ([10, 9, 8, 7], 2, [(5, 20), (20, 20)], None), ([10, 2], 2, [(3, 5), (5, 6)], 1)):
        sim, st = blob_scene(sizes, n_uav, pos, far=far)
        want = expected_turns(sim, list(range(n_uav)))
        run_decide(sim, role={i: E for i in range(n_uav)}, ev="periodic")
        got = actual_turns(sim)
        assert got == want, (sizes, got, want)
        out.append((sizes, np.bincount([k for _, k in got], minlength=len(sizes)).tolist()))
    print(f"[8] stage-1 turns follow the score rule (N/(n+1))/(d+1)^α (α {ALPHA}, n updated per turn) — UAVs per blob: "
          + "; ".join(f"{sz} -> {c}" for sz, c in out))


def check_existing_assignee():
    pos = [(5, 20), (11, 20), (17, 20), (23, 20)]
    sim, st = blob_scene([10, 9, 8], 4, pos)
    c1 = next(b.id for b in sim.clusters_of(0) if st[0] in b.cell_set())
    sim.agents[0].mode, sim.agents[0].blob = Mode.EXPLORE, c1          # R0 이 C1 의 기존 배정자 (재결정 안 함)
    tick(sim)
    want = expected_turns(sim, [1, 2, 3])
    run_decide(sim, role={1: E, 2: E, 3: E}, parties=[1, 2, 3])
    got = actual_turns(sim)
    assert got == want, (got, want)
    sim2, st2 = blob_scene([10, 9, 8], 4, pos)                         # 기존 배정자 없음 (비교)
    free = expected_turns(sim2, [1, 2, 3])
    assert want != free, "기존 배정자가 n 에 들어가면 차례가 달라져야 하는 장면"
    print(f"[9] existing assignee R0 on C1 counted in n_C1: turns {got} (without it {free})")


def check_explore_learned():
    pos5 = [(5, 20), (11, 20), (17, 20), (23, 20), (13, 30)]
    sim, st = blob_scene([10, 9, 8], 5, pos5)
    first_free = expected_turns(sim, list(range(5)))
    alt = next(k for k in range(3) if k != first_free[0][1])            # 첫 차례 UAV 는 담당 후보 대신 다른 덩어리
    want = expected_turns(sim, list(range(5)), forced={0: alt})
    first = None

    def pick(i, m):
        nonlocal first
        if first is None:
            first = i
            return alt
        return None
    g = RuleDecider()
    sim.events = [{"periodic"} for _ in range(5)]
    sim._set_decide_order()
    sim.mask_violation_step = 0
    sim._decide(lambda i, m: E, lambda i, m: g.rescue(sim, i, m),
                lambda i, m: pick(i, m) if first is None else g.explore(sim, i, m),
                lambda i, m: g.explore_sub(sim, i, m))
    got = actual_turns(sim)
    assert got == want and got[1:] != first_free[1:], (got, want, first_free)
    print(f"[10] learned path: first UAV picks slot {alt} instead of claimed {first_free[0][1]} -> n updated by the real choice, "
          f"later turns change: {got} (greedy would be {first_free})")


def retain_scene():
    """R0 은 멀리 (RETURN), R1 / R2 / R3 가 한 줄 덩어리를 나눠 탐색 중, R1 바로 옆에 새 생존자."""
    sim = fresh(4)
    sim.time = 0
    place(sim, [(28, 2), (14, 5), (14, 20), (14, 34)])
    sim.agents[0].mode, sim.agents[0].capacity = Mode.RETURN, 0
    set_front(sim, line(15, 5, 34))
    tick(sim)
    bid = sim.clusters_of(1)[0].id
    for i in (1, 2, 3):
        sim.agents[i].mode, sim.agents[i].blob = Mode.EXPLORE, bid
    tick(sim)
    return sim, bid


def check_retain():
    sim, bid = retain_scene()
    v = add_victim(sim, (14, 2), deadline=200)
    tick(sim)
    order, s1 = run_decide(sim, parties=[1, 2, 3])                     # 이벤트 "1" (재결정 대상 R1~R3)
    assert sim.agents[1].mode is Mode.RESCUE and sim.agents[1].assigned_target == v, order
    sv = sim._last_sv
    assert sorted(s1) == [2, 3] and all(sim.agents[i].blob == bid for i in (2, 3)), (s1, [sim.agents[i].blob for i in (2, 3)])
    assert [sv["explore_s1"][i] for i in (2, 3)] == [False, False] and [sv["explore_s1_all"][i] for i in (2, 3)] == [True, True]
    assert sim.ev_stats["explore_keep"] == 2 and sim.ev_stats["explore_redecide"] == 2
    print(f"[12] event 1: R1 -> RESCUE T{v}, R2 / R3 re-decide EXPLORE: stage 1 (s1 {s1}), same blob again -> explore_valid 0 "
          f"(option continues), keep 2 / 2")
    g2, g3 = sim.agents[2].goal, sim.agents[3].goal
    n_before = (len(sim.agents[2].cluster), len(sim.agents[3].cluster))
    sim.agents[1].pos = (14, 3)
    tick(sim)                                                          # rebalance: 담당 2 대로 다시 나눔
    a2, a3 = sim.agents[2], sim.agents[3]
    near = lambda g, cells: min(max(abs(g[0] - c[0]), abs(g[1] - c[1])) for c in cells)
    assert a2.goal == g2 and a3.goal == g3                             # goal latch: 목표 셀 그대로
    assert max(c[1] for c in a2.cluster) < min(c[1] for c in a3.cluster), "R2 / R3 좌우가 바뀜"
    assert near(g2, a2.cluster) <= 1 and near(g3, a3.cluster) == 0, (g2, g3)   # 자기 goal 쪽 구간 (경계 1 칸 이내)
    assert not (a2.cluster & a3.cluster) and len(a2.cluster) + len(a3.cluster) == 30
    print(f"[13] R1 left: rebalance re-splits the blob into 2 ({n_before} -> {(len(a2.cluster), len(a3.cluster))} cells), "
          f"R2 / R3 keep their sides (goal-side segments, goals unchanged by latch)")
    return sim, bid


def check_back_to_explore():
    sim, bid = retain_scene()
    v = add_victim(sim, (14, 2), deadline=200)
    tick(sim)
    run_decide(sim, parties=[1, 2, 3])
    a1 = sim.agents[1]
    a1.pos = sim.targets[v].pos                                        # 구조 완료
    sim.targets[v].rescued_at = sim.time
    a1.assigned_target = None
    tick(sim)
    _, s1 = run_decide(sim, role={1: E}, ev="2", parties=[1])
    assert s1 == [1] and sim.agents[1].blob is not None, s1
    print(f"[14] R1 back to EXPLORE after its rescue (event 2) -> joins stage 1 (s1 {s1}), blob C{sim.agents[1].blob}")


def check_retained_join():
    sim, bid = retain_scene()
    sim.agents[0].capacity, sim.agents[0].mode = 3, Mode.RETURN       # R0 이 귀환을 마치고 EXPLORE 로 (신규)
    sim.agents[0].pos = (16, 10)
    tick(sim)
    order, s1 = run_decide(sim, role={0: E, 1: E, 2: E, 3: E}, parties=[0, 1, 2, 3], ev="1")   # R1~R3 도 같은 이벤트에 재결정
    s2 = [j for st, j in sim.decision_seq if st == "s2"]
    sv = sim._last_sv
    assert sorted(s1) == [0, 1, 2, 3] and all(sim.agents[i].blob == bid for i in range(4)), (s1, sim.agents[0].blob)
    assert sorted(s2) == [0, 1, 2, 3], s2                              # 넷 다 2 단계 참여
    assert sv["explore_s1"] == [True, False, False, False], sv["explore_s1"]   # 새 표본은 새로 EXPLORE 인 R0 만
    assert sim.ev_stats["reselect_uavs"] == 0                          # 1 단계를 안 한 기존 배정자 없음
    print(f"[15] new UAV R0 joins the blob of R1~R3 which re-decide in the same step -> all four in stage 1 and stage 2 "
          f"(order {s2}); only R0 is a new explore sample (R1~R3 same blob -> option continues)")


def check_model_order(steps=150):
    """학습 모델 rollout: 모델이 디코딩한 차례 (stage rank) == env 의 decision_seq (그룹별 상대 순서)."""
    from tests.role.smoke_role_policy import base_config, make
    from training.catalog import register_all
    register_all()
    env, prep, policy = make(base_config(mode=("learn", "learn")))
    N = env.n_agents
    n_cmp = 0
    for seed in range(3):
        env.seed(seed)
        obs = env.reset()
        for t in range(steps):
            ob = prep.transform(obs)
            actions, _, _ = policy.compute_actions(ob[None], explore=True)
            rank = {k: v[0].cpu().numpy() for k, v in policy.model.last_out["stage"]["rank"].items()}
            act = {k: [int(np.asarray(a)[0]) for a in v] for k, v in actions.items()}
            obs, _, done, info = env.step(act)
            sim = env.sim
            assert info["mask_violations"] == 0
            for stage in ("role", "s1"):                                 # 2 단계는 env 의 Hungarian (모델 결정 아님)
                seq = [j for s, j in sim.last_decision_seq if s == stage]
                model = [j for j in range(N) if rank[stage][j] > 0]
                assert sorted(seq) == sorted(model), (t, stage, seq, rank[stage])
                # env 는 iteration 마다 (그룹별 차례 하나씩) 처리 -> env 순서의 rank 가 감소하면 안 됨
                assert [rank[stage][j] for j in seq] == sorted(rank[stage][j] for j in seq), (t, stage, seq, rank[stage])
                n_cmp += len(seq)
            if done:
                break
    assert n_cmp > 0
    print(f"[11] learned rollouts: model decoding order (role / stage 1) == env decision order at every step "
          f"({n_cmp} decisions compared), no mask violations")


if __name__ == "__main__":
    check_role_basic()
    check_shared_nearest()
    check_ties_and_zero_cap()
    check_more_victims()
    check_fixed_sigma()
    check_learned_path()
    check_greedy_assign()
    check_explore_order()
    check_existing_assignee()
    check_explore_learned()
    check_retain()
    check_back_to_explore()
    check_retained_join()
    check_model_order()
    print("test_decide_order: all ok")
