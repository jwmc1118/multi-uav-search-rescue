"""통합 할당 메커니즘 — 태스크(구조) / explore / depot 노드를 한 번의 pointer 결정으로 고른다.

에이전트 i 의 노드 집합 (H^i 의 행 순서와 같다, models/alloc_policy.py):

    slot 0 .. T-1  : 후보 태스크 (발견 + active + i 의 지식에 있음, BFS 거리순 상위 T 개, 패딩)
    slot T  (= E)  : explore  (목적지 = BFS 최근접 frontier)
    slot T+1 (= D) : depot    (목적지 = depot, 도달 시 재보충)

여기서 구현하는 것은 '환경 메커니즘'뿐이다 (어떤 노드를 고를지는 학습 대상):
- prepare()        : 에이전트당 BFS 1회 -> 노드 특징(d^i_j, d^i_e, d^i_d), admissible 마스크 𝒜^i,
                     primitive waypoint, lock 강제 슬롯, 디코딩 우선순위
- decode_order()   : 순차(오토리그레시브) 디코딩 순서. 강제(lock) 에이전트 먼저, 나머지는
                     admissible 태스크까지 최소 BFS 거리 오름차순 (tie = index)
- group_conflicts(): 같은 n-hop 그룹에서 앞 순서가 이미 고른 태스크 슬롯 (𝒮^{<i}) — 모델이
                     디코딩 루프에서 같은 규칙을 텐서로 적용한다 (models/alloc_policy.py)
- apply()          : 최종 선택 적용. 같은 n-hop 그룹(통신 연결) 안에서 같은 태스크가 겹치면 즉시 BFS 거리가
                     가까운 쪽이 갖고 나머지는 explore(불가하면 depot) (정책이 정상이면 순차 디코딩으로 0).
                     통신이 안 되는 그룹끼리는 서로 모른 채 같은 태스크를 lock 할 수 있다.
- resolve_contacts(): 같은 태스크를 lock 한 UAV 들이 n-hop 연결되는 순간 가까운 쪽이 유지하고
                     먼 쪽은 lock 해제 -> 다음 tick 에 정책이 재결정 (ctx.contact_released)

후보 / admissible / lock 유지 판단은 UAV 별 믿음(env/beliefs.py)만 쓴다:
  "다른 UAV 가 lock 함" = 통신으로 알게 된 claim, "끝난 태스크" = 구조를 알고 있음 또는 만료.

admissible 규칙 (𝒜^i):
    태스크 j : capacity > 0, (내가 아는 한) 다른 UAV 가 lock 하지 않음, BFS 도달 가능, d^i_j <= 잔여 골든타임
               (+ 옵션 mask_return_infeasible: t + d^i_j + d_{j->depot} <= t_max)
    explore  : 도달 가능한 frontier 가 있을 때
    depot    : 항상 (𝒜^i 가 비지 않음을 보장)
    lock     : 강제 슬롯 하나만 admissible (정책 loss 에서 제외 — free_choice = 0)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from env.modes import Mode, MSG_EXPLORE, MSG_RETURN, MSG_RESCUE_TENTATIVE, MSG_RESCUE_LOCKED
from env.pathfinding import bfs_field, path_to, window_waypoint, shortest_path_length


@dataclass
class NodeSet:
    """에이전트 i 의 이번 tick 노드 집합. 길이 A = T + 2 배열들은 슬롯 순서 그대로."""
    task_ids: List[int]                    # 슬롯 -> sim.targets index (len <= T)
    task_dist: List[int]                   # 슬롯별 BFS 거리 (-1 = 도달 불가)
    n_known: int                           # T^i: 알고 있는 발견 + active 태스크 수 (자르기 전)
    frontier: Optional[Tuple[int, int]]    # explore 목적지
    d_explore: int                         # -1 = frontier 없음 / 도달 불가
    d_depot: int
    mask: np.ndarray                       # (A,) bool admissible (그룹 내 순차 디코딩 충돌은 제외)
    goal: List[Optional[Tuple[int, int]]]  # (A,) 슬롯별 목적지 셀
    waypoint: np.ndarray                   # (A, 3) [dx, dy, valid] — local 창 안 경로 waypoint, ÷ half
    forced_slot: Optional[int] = None      # lock 이면 강제 슬롯
    priority: float = float("inf")
    dist_field: Optional[np.ndarray] = field(default=None, repr=False)


class Allocator:
    def __init__(self, max_targets, lock_rescue=True, lock_return=True,
                 mask_return_infeasible=False):
        self.T = int(max_targets)
        self.E = self.T                 # explore 슬롯
        self.D = self.T + 1             # depot 슬롯
        self.A = self.T + 2
        self.lock_rescue = lock_rescue
        self.lock_return = lock_return
        self.mask_return_infeasible = mask_return_infeasible

    # ------------------------------------------------------------ helpers

    def task_msg_type(self):
        return MSG_RESCUE_LOCKED if self.lock_rescue else MSG_RESCUE_TENTATIVE

    def slot_msg_type(self, slot):
        if slot == self.E:
            return MSG_EXPLORE
        if slot == self.D:
            return MSG_RETURN
        return self.task_msg_type()


    def release(self, sim, agent_index):
        """agent 의 rescue 배정/lock 해제."""
        a = sim.agents[agent_index]
        ti = a.assigned_target
        if ti is not None and sim.targets[ti].assigned_to == agent_index:
            sim.targets[ti].assigned_to = None
        a.assigned_target = None
        a.locked = False

    # ------------------------------------------------------------ prepare

    def _validate_locks(self, sim, fields):
        """도달 불가 / 마감 초과가 된 rescue lock 은 풀어 준다. 반환: 해제된 에이전트 index 목록."""
        released = []
        for i, a in enumerate(sim.agents):
            if not (a.locked and a.mode is Mode.RESCUE and a.assigned_target is not None):
                continue
            t = sim.targets[a.assigned_target]
            d = int(fields[i][t.pos])
            rem = t.remaining(sim.time)
            done = sim.beliefs.is_done(i, sim, a.assigned_target)
            if done or d < 0 or (rem is not None and d > rem) or a.capacity <= 0:
                self.release(sim, i)
                released.append(i)
        return released

    def prepare(self, sim):
        """모든 에이전트의 NodeSet 계산. sim.time / 지식 / 위치가 갱신된 직후 호출."""
        T, E, D, A = self.T, self.E, self.D, self.A
        half = sim.knowledge.local_size // 2
        fields, parents = [], []
        for i, a in enumerate(sim.agents):
            dist, parent = bfs_field(a.pos, sim.knowledge.known_obstacle[i], sim.rows, sim.cols)
            fields.append(dist)
            parents.append(parent)
        released = self._validate_locks(sim, fields)

        depot_field = None
        if self.mask_return_infeasible:
            depot_field = [bfs_field(sim.depot, sim.knowledge.known_obstacle[i], sim.rows, sim.cols)[0]
                           for i in range(len(sim.agents))]
        t_max = sim.cfg["max_episode_len"]

        out = []
        for i, a in enumerate(sim.agents):
            dist, parent = fields[i], parents[i]
            known_free = sim.knowledge.known_free[i]

            # ---- 후보 태스크: 거리순, lock 된 자기 타겟은 맨 앞 ----
            cand = [ti for ti, t in enumerate(sim.targets)
                    if t.discovered and known_free[t.pos] and not sim.beliefs.is_done(i, sim, ti)]
            n_known = len(cand)

            def key(ti):
                d = int(dist[sim.targets[ti].pos])
                return (0 if ti == a.assigned_target and a.locked else 1,
                        d if d >= 0 else 10 ** 9, ti)
            cand = sorted(cand, key=key)[:T]
            task_dist = [int(dist[sim.targets[ti].pos]) for ti in cand]

            # ---- frontier (BFS 최근접, 가능하면 자기 칸 제외) ----
            fr_cells = np.argwhere(sim.knowledge.frontier[i])
            frontier, d_e = None, -1
            if len(fr_cells):
                fd = dist[fr_cells[:, 0], fr_cells[:, 1]]
                ok = fd > 0
                if not ok.any():
                    ok = fd >= 0
                if ok.any():
                    k = int(np.argmin(np.where(ok, fd, 10 ** 9)))
                    frontier, d_e = tuple(int(v) for v in fr_cells[k]), int(fd[k])
            d_d = int(dist[sim.depot])

            # ---- admissible ----
            mask = np.zeros(A, bool)
            for s, ti in enumerate(cand):
                t = sim.targets[ti]
                d = task_dist[s]
                rem = t.remaining(sim.time)
                ok = (a.capacity > 0 and d >= 0 and not sim.beliefs.claimed_by_other(i, ti)
                      and (rem is None or d <= rem))
                if ok and depot_field is not None:
                    back = int(depot_field[i][t.pos])
                    ok = back >= 0 and sim.time + d + back <= t_max
                mask[s] = ok
            mask[E] = frontier is not None
            mask[D] = True

            # ---- 목적지 / waypoint ----
            goal = [sim.targets[ti].pos for ti in cand] + [None] * (T - len(cand)) + [frontier, sim.depot]
            wp = np.zeros((A, 3), np.float32)
            for s, g in enumerate(goal):
                if g is None or dist[g] < 0:
                    continue
                p = path_to(parent, a.pos, g, sim.cols)
                w = window_waypoint(p, a.pos, half)
                wp[s] = ((w[1] - a.pos[1]) / max(1, half), (a.pos[0] - w[0]) / max(1, half), 1.0)

            # ---- lock 강제 ----
            forced = None
            if a.locked and a.mode is Mode.RESCUE and a.assigned_target in cand:
                forced = cand.index(a.assigned_target)
            elif a.locked and a.mode is Mode.RETURN:
                forced = D
            if forced is not None:
                mask[:] = False
                mask[forced] = True

            # ---- 디코딩 우선순위 ----
            if forced is not None:
                prio = -1.0
            else:
                ds = [task_dist[s] for s in range(len(cand)) if mask[s]]
                prio = float(min(ds)) if ds else float("inf")

            out.append(NodeSet(task_ids=cand, task_dist=task_dist, n_known=n_known,
                               frontier=frontier, d_explore=d_e, d_depot=d_d, mask=mask,
                               goal=goal, waypoint=wp, forced_slot=forced, priority=prio,
                               dist_field=dist))
        return out, released

    # ------------------------------------------------------------ 순차 디코딩

    @staticmethod
    def decode_order(nodes):
        """우선순위 오름차순 에이전트 index 목록 (tie = index)."""
        return sorted(range(len(nodes)), key=lambda i: (nodes[i].priority, i))

    def group_conflicts(self, i, picks, nodes, reach):
        """picks: {j: target index} (이미 디코딩된 에이전트). i 와 n-hop 연결된 j 가 고른 태스크의
        슬롯 = True 인 (A,) bool."""
        out = np.zeros(self.A, bool)
        taken = {ti for j, ti in picks.items() if ti is not None and j != i and reach[i, j]}
        for s, ti in enumerate(nodes[i].task_ids):
            if ti in taken:
                out[s] = True
        return out

    # ------------------------------------------------------------ 적용

    @staticmethod
    def _components(members, reach):
        """members 를 reach(n-hop 연결) 기준 연결 요소로 나눈다."""
        left, comps = list(members), []
        while left:
            comp, stack = [left[0]], [left[0]]
            left = left[1:]
            while stack:
                u = stack.pop()
                for v in [v for v in left if reach[u, v]]:
                    left.remove(v)
                    comp.append(v)
                    stack.append(v)
            comps.append(comp)
        return comps

    def resolve_contacts(self, sim, ctx=None):
        """같은 태스크를 lock 한 UAV 들이 n-hop 연결되면 BFS 거리가 가까운 쪽만 유지.
        먼 쪽은 lock 해제 -> 다음 tick 자유 선택. 반환: 해제된 UAV index 목록."""
        reach = sim.knowledge.reach
        by_target = {}
        for i, a in enumerate(sim.agents):
            if a.alive and a.mode is Mode.RESCUE and a.assigned_target is not None:
                by_target.setdefault(a.assigned_target, []).append(i)
        released = []
        for ti, members in by_target.items():
            if len(members) < 2:
                continue
            pos = sim.targets[ti].pos

            def dist(k):
                d = shortest_path_length(sim.agents[k].pos, pos, sim.knowledge.known_obstacle[k],
                                         sim.rows, sim.cols)
                return (d if d is not None else 10 ** 9, k)
            for comp in self._components(members, reach):
                if len(comp) < 2:
                    continue
                winner = min(comp, key=dist)
                for k in comp:
                    if k != winner:
                        self.release(sim, k)
                        if sim.targets[ti].assigned_to is None and sim.agents[winner].locked:
                            sim.targets[ti].assigned_to = winner
                        released.append(k)
        if ctx is not None:
            ctx.contact_released.extend(released)
        return released

    def _fallback(self, ns):
        return self.E if ns.frontier is not None else self.D

    def apply(self, sim, nodes, allocs, ctx):
        """allocs: (n,) 슬롯 정수. 에이전트 상태(mode / 목적지 / lock / 메시지)를 갱신한다."""
        T, E, D = self.T, self.E, self.D
        n = len(sim.agents)
        slots = [None] * n
        for i, a in enumerate(sim.agents):
            if not a.alive:
                continue
            ns = nodes[i]
            s = int(allocs[i])
            if not (0 <= s < self.A) or not ns.mask[s]:
                ctx.alloc_invalid.add(i)
                s = ns.forced_slot if ns.forced_slot is not None else self._fallback(ns)
            slots[i] = s

        # 같은 n-hop 그룹(서로 통신 중) 안에서 같은 태스크가 겹치면 즉시 해결: BFS 거리가 가까운 쪽.
        # 통신이 안 되는 UAV 끼리는 서로 모르므로 그대로 둔다 (만나는 순간 resolve_contacts 가 해결).
        reach = sim.knowledge.reach
        by_target = {}
        for i, s in enumerate(slots):
            if s is not None and s < T:
                by_target.setdefault(nodes[i].task_ids[s], []).append(i)
        for ti, members in by_target.items():
            if len(members) < 2:
                continue
            key = lambda k: (0 if nodes[k].forced_slot is not None else 1, nodes[k].task_dist[slots[k]], k)
            for comp in self._components(members, reach):
                if len(comp) < 2:
                    continue
                winner = min(comp, key=key)
                for k in comp:
                    if k != winner:
                        ctx.double_selected.add(k)
                        slots[k] = self._fallback(nodes[k])

        for i, a in enumerate(sim.agents):
            s = slots[i]
            if s is None:
                a.goal = None
                continue
            ns = nodes[i]
            ctx.chosen_slot[i] = s
            if s < T:
                ti = ns.task_ids[s]
                if a.assigned_target is not None and a.assigned_target != ti:
                    self.release(sim, i)
                a.mode = Mode.RESCUE
                a.assigned_target = ti
                a.locked = self.lock_rescue
                if self.lock_rescue:
                    sim.targets[ti].assigned_to = i        # 전역 기록 (GUI / 디버그용 — 결정에는 안 씀)
            else:
                if a.assigned_target is not None:
                    self.release(sim, i)
                if s == E:
                    a.mode = Mode.EXPLORE
                    a.locked = False
                else:
                    a.mode = Mode.RETURN
                    a.locked = self.lock_return and a.pos != sim.depot
            a.goal = ns.goal[s]
            a.msg_type = self.slot_msg_type(s)
            a.msg_goal = sim.frame.normalized(a.goal) if a.goal is not None else sim.frame.normalized(a.pos)
        return slots
