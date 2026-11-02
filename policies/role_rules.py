"""계층 구조 env (env/role/simulation.py) 의 그리디 정책 — baseline / 디버깅 / policy_mode 의 rule 헤드 / 마스크 위반 대체.

결정 순서는 env 가 policies/decision_order.py 로 정하고, 차례가 된 UAV 의 담당 후보를 sim.cur_claim (생존자) /
sim.cur_xclaim (덩어리 슬롯) 에 넣어 둔다. 그리디는 담당 후보를 그대로 고르므로 순서가 '한 번 계산한 전체 순서' 와 같다.

    rule_role    : 주기 · 이벤트 "1" (여러 대 재결정, 그룹 그리디): 담당 후보 생존자가 있으면 RESCUE, 없으면 EXPLORE
                   (용량 0 이면 RETURN)
                   이벤트 "2" / "3" / "4" (한 대): 용량 0 -> RETURN, 할당 가능 생존자 있음 -> RESCUE, 없으면 EXPLORE
    rule_rescue  : 주기 · "1" 은 담당 후보 생존자, 그 외 EDF (마감 τ 빠른 순 -> BFS 거리 -> 번호)
    rule_explore : 1 단계 — 담당 후보 덩어리 (explore_front: 남은 UAV x 닿는 덩어리 쌍 중 점수 (N_m / (n_m + 1)) / (d + 1)^α
                   최대, 한 대마다 n_m 갱신). 담당 후보가 없거나 마스크 밖이면 같은 점수 최대 덩어리
    rule_explore_sub : 2 단계 — 덩어리 참여자 전원을 1:1 Hungarian (비용 = 구간까지 BFS 최소 거리, 합 최소, goal 우대 없음)
    RuleDecider  : 위 규칙 묶음 (그리디 정책 하나)
    RandomDecider: 마스크 안에서 균등 무작위 (비교 / 디버깅)

decider 인터페이스: role(sim, i, mask) / rescue(sim, i, mask) / explore(sim, i, mask) / explore_sub(sim, i, mask) -> 슬롯 인덱스.
RoleSimulation.step(decider) 에 넘기거나, step_explicit 이 rule 헤드 / 마스크 위반 대체에 이 규칙을 쓴다.
기준선: python -m rule_evaluation.run_rules
"""
import numpy as np

from env.core.modes import Mode
from env.core.targets import deadline as _deadline


def multi_event(sim, i):
    """여러 대 재결정 (주기 "periodic" / 이벤트 "1") 로만 재결정하는가 — 이벤트 "2" / "3" / "4" 가 섞이면 한 대 규칙.
    (번호는 env/role/simulation.py EVENTS)"""
    ev = sim.events[i] if i < len(getattr(sim, "events", ())) else set()
    return bool(ev) and not (ev & {"2", "3", "4"})


def rule_role(sim, i, mask):
    """여러 대 재결정 (주기 / 이벤트 "1"): 담당 후보 생존자 -> RESCUE, 없으면 EXPLORE (용량 0 이면 RETURN).
    한 대 재결정 (이벤트 "2" / "3" / "4"): 용량 0 -> RETURN, 할당 가능 생존자 있음 -> RESCUE, 없으면 EXPLORE.
    막히면 마스크 안의 다음 역할."""
    a = sim.agents[i]
    if a.capacity == 0 and mask[Mode.RETURN.value]:
        return Mode.RETURN.value
    if multi_event(sim, i):
        u = getattr(sim, "cur_claim", {}).get(i)
        rescue = u is not None and u in sim.victim_slots[i]
    else:
        rescue = True
    prefer = [Mode.RESCUE, Mode.EXPLORE] if rescue else [Mode.EXPLORE, Mode.RESCUE]
    for m in prefer + [Mode.RETURN]:
        if mask[m.value]:
            return m.value
    return int(np.flatnonzero(mask)[0])


def rule_rescue(sim, i, mask):
    """이벤트 1 / 2: 담당 후보 생존자 (마스크 안이면). 그 외 EDF: 마감 τ 빠른 순 -> BFS 거리 -> 생존자 번호."""
    if multi_event(sim, i):
        u = getattr(sim, "cur_claim", {}).get(i)
        if u is not None and u in sim.victim_slots[i]:
            s = sim.victim_slots[i].index(u)
            if mask[s]:
                return s
    best = None
    for s, ti in enumerate(sim.victim_slots[i]):
        if not mask[s]:
            continue
        key = (_deadline(sim.targets[ti]), sim._dist(i, sim.targets[ti].pos), ti)
        if best is None or key < best[0]:
            best = (key, s)
    return best[1]


def rule_explore(sim, i, mask):
    """1 단계: 담당 후보 덩어리 (sim.cur_xclaim). 없거나 마스크 밖이면 explore_front 와 같은 점수 (N_m / (n_m + 1)) / (d_im + 1)^α
    최대 덩어리 (같으면 큰 덩어리 -> 슬롯 순 -> 거리). α = explore.order_dist_alpha, n_m = 그룹의 다른 UAV 중 지금 그 덩어리를
    맡은 수 (재결정하지 않는 기존 배정자 + 이번 이벤트 앞 차례), d = BFS 칸 수 (도달 불가 덩어리는 맨 뒤)."""
    k = getattr(sim, "cur_xclaim", {}).get(i)
    if k is not None and 0 <= k < len(mask) and mask[k]:
        return int(k)
    alpha = float(getattr(sim, "explore_cfg", {}).get("order_dist_alpha", 1.0))
    group = sim.groups[sim.comp[i]]
    best = None
    for k, cl in enumerate(sim.clusters_of(i)):
        if k >= len(mask) or not mask[k]:
            continue
        d = sim.blob_dist(i, cl)
        size = len(cl.cell_set())
        n = sum(1 for j in group if j != i and sim.agents[j].mode is Mode.EXPLORE and sim.agents[j].blob == cl.id)
        score = -1.0 if d is None else (size / (n + 1)) / (d + 1) ** alpha
        key = (-score, -size, k, 10 ** 9 if d is None else d)
        if best is None or key < best[0]:
            best = (key, k)
    return best[1]


def seg_distance(sim, i, cells):
    """UAV i 에서 구간 셀까지 BFS 최소 거리 (UAV 지도 BFS, 도달 불가 None)."""
    f = sim.fields[i]
    reach = [int(f[c]) for c in cells if f[c] >= 0]
    return min(reach) if reach else None


def distance_hungarian(sim, members, segs, avail=None):
    """2 단계: 거리 합 최소 1:1 배정. 비용 = UAV i 에서 구간 S 셀까지 BFS 최소 거리 d_iS (seg_distance, goal 우대 없음),
    도달 불가 구간은 1e6. 구간보다 참여자가 많으면 남은 참여자끼리 전체 구간에서 한 번 더. 반환 {UAV: 구간 index}."""
    from scipy.optimize import linear_sum_assignment
    out, todo = {}, list(members)
    choices = list(range(len(segs))) if avail is None else list(avail)
    while todo:
        if not choices:
            choices = list(range(len(segs)))
        C = np.zeros((len(todo), len(choices)))
        for r, i in enumerate(todo):
            for c, s in enumerate(choices):
                d = seg_distance(sim, i, segs[s][0])
                C[r, c] = float(d) if d is not None else 1e6
        rows, cols = linear_sum_assignment(C)
        for r, c in zip(rows, cols):
            out[todo[r]] = int(choices[c])
        todo = [i for i in todo if i not in out]
        choices = list(range(len(segs)))
    return out


def rule_explore_sub(sim, i, mask):
    """2 단계 규칙. env 의 2 단계 (RoleSimulation._stage2_all) 가 sim._stage2_ctx = {members, segs, avail} 를 채운 뒤
    참여자 차례마다 호출한다 — 첫 호출에서 참여자 전원을 distance_hungarian (거리 합 최소) 으로 한 번 배정하고, 이후 그 배정을 돌려준다
    (순서와 무관). 배정 구간이 마스크되면 (구간보다 참여자가 많음) 가장 가까운 구간."""
    ctx = getattr(sim, "_stage2_ctx", None)
    if ctx is not None and i in ctx["members"]:
        if ctx["assign"] is None:
            ctx["assign"] = distance_hungarian(sim, ctx["members"], ctx["segs"], ctx.get("avail"))
        s = ctx["assign"].get(i)
        if s is not None and s < len(mask) and mask[s]:
            return s
        segs = ctx["segs"]
        best = None
        for s in np.flatnonzero(mask):
            if s >= len(segs):
                continue
            arr = np.asarray(segs[s][0])
            fd = sim.fields[i][arr[:, 0], arr[:, 1]]
            fd = fd[fd >= 0]
            d = int(fd.min()) if len(fd) else 10 ** 9
            if best is None or (d, s) < best:
                best = (d, int(s))
        if best is not None:
            return best[1]
    return int(np.flatnonzero(mask)[0])


class RuleDecider:
    """그리디 정책 (role + rescue: 담당 후보 / EDF, explore 1 단계: 점수 (N / (n + 1)) / (d + 1)^α 차례, 2 단계: 거리 Hungarian)."""

    def role(self, sim, i, mask):
        return rule_role(sim, i, mask)

    def rescue(self, sim, i, mask):
        return rule_rescue(sim, i, mask)

    def explore(self, sim, i, mask):
        return rule_explore(sim, i, mask)

    def explore_sub(self, sim, i, mask):
        return rule_explore_sub(sim, i, mask)


class RandomDecider:
    """마스크 안에서 균등 무작위 (디버깅 / 비교용)."""

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def _pick(self, mask):
        return int(self.rng.choice(np.flatnonzero(mask)))

    def role(self, sim, i, mask):
        return self._pick(mask)

    def rescue(self, sim, i, mask):
        return self._pick(mask)

    def explore(self, sim, i, mask):
        return self._pick(mask)

    def explore_sub(self, sim, i, mask):
        return self._pick(mask)
