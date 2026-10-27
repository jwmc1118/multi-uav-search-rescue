"""Explore 파이프라인 (RoleSimulation 의 mixin) — 덩어리 생성 / 대응 / 매 step 구간 rebalance / 구간 분할 표 / 노드 특징.

용어
    덩어리 (blob, C_k)  : 그룹 공유 지도의 frontier DBSCAN 클러스터 (A-1). step 간 id 가 이어진다 (A-2).
    구간 (segment)      : 같은 덩어리를 m 대가 맡을 때 측지 k-medoids 로 나눈 조각 (A-5). a.cluster = 내 구간 셀.
    m_k                 : 같은 그룹 안에서 덩어리 C_k 를 맡은 UAV 수 (A-4)
    추정 담당 (ghost)    : 같은 덩어리를 맡던 팀원이 통신 범위 밖으로 나간 경우, 그의 마지막 구간을 그룹 지식에 남겨 둔 것
                          (재분할하지 않음, 다시 만나거나 그 구간이 소진되면 해제)

한 step 끝 (지도 갱신 직후) 의 처리 = _explore_after_update():
    A-1 덩어리 생성 (그룹마다 1 회, reachable frontier 만) -> A-2 대응 (유지 / 쪼개짐 1->n / 합쳐짐 n->1 / 소멸)
    -> 추정 담당 갱신 -> UAV 소속 덩어리 이어 붙이기 (쪼개짐 / 합쳐짐은 기록만, 재결정 없음) -> 매 step 구간 rebalance
       (정책 결정 아님, 소진 / 도달 불가 = Role 이벤트 5) -> 덩어리 슬롯 (크기 내림차순 K_max, 담당 중인 덩어리는 항상 포함)
    쪼개짐 (1 -> n) : 내 구간 셀이 가장 많이 남은 조각으로 소속 이동 / 합쳐짐 (n -> 1) : 새 덩어리로 / 소멸 : Role 이벤트 5
2 단계 (결정 시점, RoleSimulation._stage2_all): 이번 step 1 단계로 고른 덩어리 c 의 참여자 P_c = 고른 UAV + 기존 배정자,
    kk = min(|P_c|, kcap) 구간 (k-medoids 초기 medoid = 기존 배정자 각자의 goal 셀에 가장 가까운 frontier 셀).
    참여자는 policies/decision_order.py seg_front 순서로 1:1 마스크 안에서 구간 선택 (기존 배정자는 구간만 = stage-2-only).
매 step rebalance 의 연속 매칭 비용 (_assignee_cost): 내 goal 셀이 든 구간 0, 없으면 내 위치에서 구간의 가장 가까운 셀까지 BFS.
UAV 별 상태 (Robot 에 붙임)
    blob         : 소속 덩어리 id (None = 없음)          cluster / cluster_medoid : 내 구간 셀 (frozenset) / 대표점
    seg_start    : 지금 구간을 받은 시각 (구간 소진까지 step 통계)
"""
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment

from env.core.modes import Mode
from env.role.clusters import frontier_clusters, geodesic_matrix, partition, medoid

EXPLORE_DEFAULTS = {
    "n_sub_max": 5,             # 2 단계 구간 슬롯 최대 수 (구간 수 상한, 기본 = UAV 수)
    "size_ref": 100,            # 크기 특징 min(1, log(1 + |C|) / log(1 + size_ref)) (env/role/graph_obs.py size_feat)
    "match_dilate": 1,          # A-2 대응: 이전 덩어리 셀을 이만큼 (Chebyshev) 넓혀 겹침 판정 (frontier 1 칸 후퇴 흡수)
    "track_margin": 2,          # 소속 덩어리가 사라졌을 때 직전 구간에서 r_s + track_margin 안의 새 덩어리로 이어 붙임
    "kmedoids_iters": 10,       # k-medoids 반복 상한
    "goal_mode": "nearest",     # nearest = 내 구간 frontier 중 BFS 가장 가까운 셀 / medoid = 내 구간 medoid
}


def n_seg_slots(n_sub_max):
    """구간 분할 표의 슬롯 수 P = Σ_{kk=2..M} kk (kk = 구간 수)."""
    return sum(range(2, int(n_sub_max) + 1))


def seg_offset(kk):
    """구간 수 kk 인 분할이 표에서 시작하는 위치 (kk >= 2)."""
    return sum(range(2, kk))


class ExploreMixin:
    # ================================================================ 초기화

    def _explore_reset(self):
        self.blob_seq = 0
        self.ghosts = [dict() for _ in self.agents]          # 관측 UAV i -> {팀원 j: 마지막 구간 셀}
        self._prev_members = [frozenset([i]) for i in range(self.n_agents)]
        self._prev_blobs = [[] for _ in self.agents]          # UAV 별: 직전 step 그 그룹의 덩어리 목록
        self.blob_events = []                                 # 이번 step 덩어리 대응 결과 (split / merge 문장)
        self._split_groups = set()                            # 이번 step 담당 덩어리가 쪼개진 그룹
        self._merged = set()                                  # 이번 step 담당 덩어리가 합쳐진 (그룹, 새 덩어리 id)
        self.stats = {"s1": 0, "s2": 0, "reselect": 0, "split": 0, "merge": 0, "ghost_drop": 0, "seg_life": [], "role_switch": 0}
        for a in self.agents:
            a.blob = None
            a.cluster = None
            a.cluster_medoid = None
            a.seg_start = 0

    def _new_blob_id(self):
        self.blob_seq += 1
        return self.blob_seq

    # ================================================================ A-1 / A-2

    def _build_blobs(self):
        """그룹마다 덩어리 생성 + 이전 step 과 대응 (id 이어 붙이기). self.blobs[g] = 전체 (크기 내림차순)."""
        cc = self.cluster_cfg
        self.blobs, self.blob_index = [], []
        self.blob_events = []
        for g, members in enumerate(self.groups):
            lead = members[0]
            reach = np.zeros((self.rows, self.cols), bool)
            for j in members:
                reach |= self.fields[j] >= 0
            bl = frontier_clusters(self.knowledge.frontier[lead], cc["eps"], cc["min_samples"], reachable=reach)
            self._match_blobs(g, bl)
            self.blobs.append(bl)
            self.blob_index.append({b.id: b for b in bl})

    def _match_blobs(self, g, new):
        """A-2: 이전 step 덩어리 (그룹 멤버들이 알던 것) 와 이번 덩어리를 frontier 셀 겹침 (match_dilate 만큼 넓힘) 으로 대응."""
        prev = {}
        for j in self.groups[g]:
            for b in self._prev_blobs[j]:
                prev[b.id] = b
        dil = int(self.explore_cfg["match_dilate"])
        owner = {}
        for pid, b in prev.items():
            for r, c in b.cell_set():
                for dr in range(-dil, dil + 1):
                    for dc in range(-dil, dil + 1):
                        owner.setdefault((r + dr, c + dc), set()).add(pid)
        links = []
        for q in new:
            ps = set()
            for cell in q.cell_set():
                ps |= owner.get(cell, set())
            links.append(ps)
        # 이분 그래프 연결 성분 (prev id <-> new index)
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for qi, ps in enumerate(links):
            find(("q", qi))
            for p in ps:
                parent[find(("p", p))] = find(("q", qi))
        comps = {}
        for qi in range(len(new)):
            comps.setdefault(find(("q", qi)), ([], set()))[0].append(qi)
        for qi, ps in enumerate(links):
            comps[find(("q", qi))][1].update(ps)
        self._split_of, self._merge_of = getattr(self, "_split_of", {}), getattr(self, "_merge_of", {})
        for qs, ps in comps.values():
            if not ps:
                for qi in qs:
                    new[qi].id, new[qi].origin, new[qi].parents = self._new_blob_id(), "new", ()
            elif len(ps) == 1 and len(qs) == 1:
                new[qs[0]].id, new[qs[0]].origin, new[qs[0]].parents = next(iter(ps)), "keep", tuple(ps)
            elif len(ps) == 1:                                           # 쪼개짐 1 -> n
                p = next(iter(ps))
                ids = []
                for qi in qs:
                    new[qi].id, new[qi].origin, new[qi].parents = self._new_blob_id(), "split", (p,)
                    ids.append(new[qi].id)
                self._split_of[(g, p)] = ids
                self.blob_events.append(f"C{p} -> " + ", ".join(f"C{k}" for k in ids))
            elif len(qs) == 1:                                           # 합쳐짐 n -> 1
                q = new[qs[0]]
                q.id, q.origin, q.parents = self._new_blob_id(), "merge", tuple(sorted(ps))
                for p in ps:
                    self._merge_of[(g, p)] = q.id
                self.blob_events.append(" + ".join(f"C{p}" for p in sorted(ps)) + f" -> C{q.id}")
            else:                                                        # n -> m: 쪼개짐으로 처리 (1 단계부터)
                for qi in qs:
                    new[qi].id, new[qi].origin, new[qi].parents = self._new_blob_id(), "split", tuple(sorted(links[qi]))
                    for p in links[qi]:
                        self._split_of.setdefault((g, p), []).append(new[qi].id)

    # ================================================================ 추정 담당 / 소속 / 매 step rebalance

    def _update_ghosts(self, prev_state):
        """통신 이탈한 같은 덩어리 팀원의 마지막 구간을 추정 담당으로 남기고, 재합류 / 소진되면 해제."""
        for i in range(self.n_agents):
            now = frozenset(self.groups[self.comp[i]])
            gone = self._prev_members[i] - now
            mode_i, blob_i = prev_state[i][0], prev_state[i][1]
            for j in gone:
                mode_j, blob_j, seg_j = prev_state[j]
                if mode_i is Mode.EXPLORE and mode_j is Mode.EXPLORE and blob_i is not None and blob_i == blob_j and seg_j:
                    if j not in self.ghosts[i]:
                        self.stats["ghost_drop"] += 1
                    self.ghosts[i][j] = frozenset(seg_j)
            for j in list(self.ghosts[i]):
                if j in now:
                    del self.ghosts[i][j]                               # 다시 만남
                    continue
                cells = frozenset(c for c in self.ghosts[i][j] if self.knowledge.frontier[i][c])
                if cells:
                    self.ghosts[i][j] = cells
                else:
                    del self.ghosts[i][j]                               # 그 구간 소진
            self._prev_members[i] = now

    def group_ghosts(self, g):
        """그룹 지식의 추정 담당 {팀원 j: 셀} (멤버들이 아는 것의 합, 그룹 안에 있는 j 는 제외)."""
        out = {}
        members = set(self.groups[g])
        for i in members:
            for j, cells in self.ghosts[i].items():
                if j not in members:
                    out[j] = out.get(j, frozenset()) | cells
        return out

    def _ghost_cells(self, g, blob):
        cs = blob.cell_set()
        return frozenset().union(*[c & cs for c in self.group_ghosts(g).values()])

    def _ghosts_on(self, g, blob):
        cs = blob.cell_set()
        return {j for j, cells in self.group_ghosts(g).items() if cells & cs}

    def _relink_and_anchor(self, prev_seg):
        """UAV 소속 덩어리를 이번 step id 로 이어 붙인다 (A-2 대응). 합쳐짐 -> 새 덩어리로 (2 단계 재결정 대상 기록),
        쪼개짐 -> 내 구간이 들어 있는 조각으로 (그룹의 EXPLORE UAV 1 단계 재결정 기록), 소멸 -> 직전 구간 근처 새 덩어리가
        있으면 이어 붙이고 없으면 구간 rebalance 에서 소진 (Role 이벤트 6)."""
        self._split_groups, self._merged = set(), set()
        for i, a in enumerate(self.agents):
            if a.mode is not Mode.EXPLORE or a.blob is None:
                continue
            g = self.comp[i]
            idx = self.blob_index[g]
            p = a.blob
            if p in idx:
                continue
            if (g, p) in self._merge_of:
                a.blob = self._merge_of[(g, p)]
                self._merged.add((g, a.blob))
                continue
            if (g, p) in self._split_of:
                pieces = [idx[k] for k in self._split_of[(g, p)] if k in idx]
                self._reanchor(i, pieces, prev_seg[i])
                self._split_groups.add(g)
                continue
            # 소멸: 직전 구간 근처 (r_s + track_margin) 의 새 덩어리가 있으면 이어 붙인다 (frontier 가 크게 이동한 경우)
            seg = prev_seg[i]
            if seg:
                near = self._near_mask(seg, a.sensor_range + int(self.explore_cfg["track_margin"]))
                for b in self.blobs[g]:
                    if b.origin == "new" and any(near[c] for c in b.cell_set()):
                        b.id, b.origin, b.parents = p, "keep", (p,)
                        self.blob_index[g] = {bb.id: bb for bb in self.blobs[g]}
                        break
        self.stats["split"] += len(self._split_groups)
        self.stats["merge"] += len(self._merged)

    def _reanchor(self, i, pieces, seg):
        """내 구간 셀이 가장 많이 남은 조각 (동률이면 목표점이 든 조각 -> BFS 가까운 쪽) 으로 소속 이동."""
        a = self.agents[i]
        if not pieces:
            return
        best = None
        for b in pieces:
            cs = b.cell_set()
            has_goal = a.goal is not None and a.goal in cs
            overlap = len(cs & seg) if seg else 0
            d = min(self._dist(i, c) for c in cs)
            key = (-overlap, not has_goal, d, b.id)
            if best is None or key < best[0]:
                best = (key, b)
        a.blob = best[1].id

    def _near_mask(self, cells, radius):
        m = np.zeros((self.rows, self.cols), bool)
        for r, c in cells:
            m[r, c] = True
        if radius <= 0:
            return m
        return binary_dilation(m, structure=np.ones((3, 3), bool), iterations=int(radius))

    def _rebalance_segments(self, prev_seg, prev_med):
        """매 step 구간 rebalance (정책 호출 없음):
        소속 덩어리는 이전 step 에서 이어 받고 (A-2 대응), 담당자 m >= 2 인 덩어리는 현재 frontier 셀 전체 (추정 담당 셀 제외)
        를 k = min(m, |C|) 로 측지 k-medoids 재분할 — 초기 medoid = 각 UAV 의 goal 셀에 측지 거리 가장 가까운 현재 셀 (goal 이
        없으면 직전 구간 medoid). 새 구간 <-> UAV = _assignee_cost (goal 셀이 든 구간 0, 아니면 위치에서 구간까지 BFS) 합 최소
        매칭 (헝가리안). 구간마다 연결 성분 1 개. m = 1 이면 덩어리 전체. 구간을 못 받은 UAV (|C| < m) 는 덩어리에서 빠지고 Role 이벤트 6.
        반환: 구간을 잃은 UAV {i: 이유}."""
        from env.role.clusters import kmedoids, contiguous_labels, balance_labels
        exhausted = {}
        iters = int(self.explore_cfg["kmedoids_iters"])
        for g, members in enumerate(self.groups):
            by_blob = {}
            for i in members:
                a = self.agents[i]
                if a.mode is Mode.EXPLORE and a.blob is not None:
                    by_blob.setdefault(a.blob, []).append(i)
            for bid, P in sorted(by_blob.items()):
                b = self.blob_index[g].get(bid)
                cells = self._blob_cells_for_split(g, b) if b is not None else []
                if not cells:
                    for i in P:
                        seg = prev_seg[i] or frozenset()
                        exhausted[i] = "unreachable" if any(self.knowledge.frontier[i][c] for c in seg) else "exhausted"
                    continue
                if len(P) == 1:
                    self._set_segment(P[0], cells, medoid(cells))
                    continue
                key = (g, bid)
                if key not in self._geo_cache:
                    self._geo_cache[key] = geodesic_matrix(cells)
                D = self._geo_cache[key]
                index = {c: n for n, c in enumerate(cells)}
                warm = self._goal_seeds(g, cells, P, fallback=prev_med)       # warm start: 각자의 goal 셀에 가장 가까운 셀
                k = min(len(P), len(cells))
                labels, meds = kmedoids(D, k, iters, init=[index[c] for c in warm.values()])
                labels = contiguous_labels(cells, labels, meds)
                labels = balance_labels(cells, labels, meds, D)
                segs = [([cells[n] for n in np.flatnonzero(labels == j)], cells[meds[j]]) for j in range(len(meds))]
                segs = [s for s in segs if s[0]]
                cost = np.array([[self._assignee_cost(i, sc) for sc, _ in segs] for i in P])
                rows, cols = linear_sum_assignment(cost)
                got = {P[r]: segs[c] for r, c in zip(rows, cols)}
                for i in P:
                    if i in got:
                        self._set_segment(i, got[i][0], got[i][1])
                    else:                                          # |C| < m: 구간 없음 -> no_segment_steps 연속이면 이벤트 5
                        a = self.agents[i]
                        a.noseg = getattr(a, "noseg", 0) + 1
                        a.cluster, a.cluster_medoid = None, None
                        a.goal = self._latch_goal(i, frozenset())
                        if a.noseg >= int(self.events_cfg["no_segment_steps"]):
                            exhausted[i] = "no segment"
        for i, why in exhausted.items():
            a = self.agents[i]
            if a.cluster is not None:
                self.stats["seg_life"].append(self.time - a.seg_start)
            a.cluster = a.cluster_medoid = None
            a.goal = None
            a.blob = None
        return exhausted

    def _nearest_index(self, g, cell, cells, taken):
        """cell 에서 측지 (그룹 지도 BFS) 거리 가장 가까운 cells 의 index (taken 제외, 동률은 좌표 순)."""
        from env.core.pathfinding import bfs_field
        lead = self.groups[g][0]
        d = bfs_field(cell, self.knowledge.known_obstacle[lead], self.rows, self.cols)[0]
        best = None
        for n, c in enumerate(cells):
            if n in taken:
                continue
            v = int(d[c])
            v = v if v >= 0 else 10 ** 6 + max(abs(c[0] - cell[0]), abs(c[1] - cell[1]))
            if best is None or (v, c) < best[0]:
                best = ((v, c), n)
        return best[1] if best else None

    def _set_segment(self, i, cells, med):
        """매 step rebalance 결과 반영 (정책 결정 아님) — 목표 셀은 goal latch 규칙으로."""
        a = self.agents[i]
        a.cluster = frozenset(cells)
        a.cluster_medoid = tuple(med)
        a.noseg = 0
        a.goal = self._latch_goal(i, a.cluster)

    def _latch_goal(self, i, cells):
        """goal latch (events.goal_latch): 지금 목표 셀은 (관측되어 frontier 가 아님 / 도달 불가 / 내 구간 밖
        goal_latch_outside_steps step 연속) 일 때만 바꾼다. 바꿀 때는 _segment_goal (내 구간에서 BFS 가장 가까운 셀)."""
        a = self.agents[i]
        cur = a.goal
        if not self.events_cfg["goal_latch"] or cur is None:
            a.goal_out = 0
            return self._segment_goal(i, cells, cur, a.cluster_medoid) if cells else None
        if not (self.knowledge.frontier[i][cur] and self._dist(i, cur) < 10 ** 9):
            a.goal_out = 0
            return self._segment_goal(i, cells, None, a.cluster_medoid) if cells else None
        if cur in cells:
            a.goal_out = 0
            return cur
        a.goal_out = getattr(a, "goal_out", 0) + 1
        if a.goal_out >= int(self.events_cfg["goal_latch_outside_steps"]):
            a.goal_out = 0
            return self._segment_goal(i, cells, None, a.cluster_medoid) if cells else None
        self.ev_stats["latch_hold"] += 1                           # 구간 경계가 움직였지만 목표 유지
        return cur

    def _segment_goal(self, i, cells, cur=None, med=None):
        """이동 목표점. goal_mode nearest: 현재 목표점이 아직 내 구간의 frontier 면 유지, 아니면 구간 안에서 BFS 가장
        가까운 셀 — BFS 동률이면 UAV 위치와 유클리드 거리가 짧은 셀, 그래도 동률이면 (행, 열) 사전순.
        (예: 오른쪽 같은 행 셀과 오른쪽 위 대각선 셀이 둘 다 BFS 4 면 같은 행 셀.)
        goal_mode medoid: 구간 medoid (도달 불가면 nearest 규칙)."""
        if self.explore_cfg["goal_mode"] == "medoid":
            med = med or self.agents[i].cluster_medoid
            if med is not None and med in cells and self._dist(i, med) < 10 ** 9:
                return med
        if cur is not None and cur in cells and self.knowledge.frontier[i][cur] and self._dist(i, cur) < 10 ** 9:
            return cur
        pr, pc = self.agents[i].pos
        best = None
        for c in cells:
            d = int(self.fields[i][c])
            if d < 0:
                continue
            key = (d, (c[0] - pr) ** 2 + (c[1] - pc) ** 2, c)       # BFS -> 유클리드 (제곱) -> (행, 열)
            if best is None or key < best:
                best = key
        return best[2] if best else None

    # ================================================================ 슬롯 / X1·X2·X4 판정

    def _build_slots(self):
        """그룹별 덩어리 슬롯: 크기 내림차순 상위 K_max, 그룹 안에서 담당 중인 덩어리는 항상 포함."""
        self.group_clusters = []
        for g, bl in enumerate(self.blobs):
            held = {self.agents[j].blob for j in self.groups[g]
                    if self.agents[j].mode is Mode.EXPLORE and self.agents[j].blob is not None}
            slots = list(bl[:self.k_max])
            extra = [b for b in bl[self.k_max:] if b.id in held]
            for b in extra:
                for k in range(len(slots) - 1, -1, -1):
                    if slots[k].id not in held:
                        slots[k] = b
                        break
            slots.sort(key=lambda b: (-b.size, b.medoid))
            self.group_clusters.append(slots)

    def slot_of(self, i):
        a = self.agents[i]
        if a.mode is not Mode.EXPLORE or a.blob is None:
            return -1
        for k, b in enumerate(self.clusters_of(i)):
            if b.id == a.blob:
                return k
        return -1

    def assignees(self, g, blob_id):
        return [j for j in self.groups[g] if self.agents[j].mode is Mode.EXPLORE and self.agents[j].blob == blob_id]

    # ================================================================ 구간 분할 표 (A-5) / 2 단계 노드

    def _blob_cells_for_split(self, g, b):
        return sorted(b.cell_set() - self._ghost_cells(g, b))

    def kcap(self, g, b):
        """덩어리 b 의 구간 수 상한 = min(n_sub_max, |C'|) (C' = 추정 담당 셀을 뺀 셀)."""
        return min(int(self.explore_cfg["n_sub_max"]), len(self._blob_cells_for_split(g, b)))

    def segments(self, g, b, kk):
        """덩어리 b 를 kk 구간으로 (step 내 캐시, 결정적). [(cells tuple, medoid)]. k-medoids 초기 medoid = 유지 UAV 각자의
        goal 셀에 가장 가까운 frontier 셀 (기존 배정자 쪽 구간이 그 UAV 주변에 생기도록)."""
        key = (g, b.id, kk)
        if key not in self._seg_cache:
            cells = self._blob_cells_for_split(g, b)
            dkey = (g, b.id)
            if dkey not in self._geo_cache:
                self._geo_cache[dkey] = geodesic_matrix(cells) if len(cells) > 1 else None
            D = self._geo_cache[dkey]
            seeds = list(self._goal_seeds(g, cells, *self._seed_keepers(g, b)).values()) if kk >= 2 else []
            segs = partition(cells, kk, D, int(self.explore_cfg["kmedoids_iters"]), init=seeds) if kk >= 2 \
                else [(tuple(cells), medoid(cells))]
            self._seg_cache[key] = segs
        return self._seg_cache[key]

    # ---------------------------------------------------------------- 그룹 지도 BFS (scipy csgraph, step 내 캐시)

    def _grid_graph(self, g):
        """그룹 g 공유 지도 (리드 UAV 지식의 알려진 장애물만 막음, 미탐색 통과 — bfs_field 와 같은 규칙) 의 8-연결 그래프.
        대각선은 양옆 두 칸이 모두 막혔을 때만 금지 (코너 컷팅 규칙 동일)."""
        if g in self._graph_cache:
            return self._graph_cache[g]
        from scipy.sparse import csr_matrix
        R, C = self.rows, self.cols
        bl = self.knowledge.known_obstacle[self.groups[g][0]]
        free = ~bl
        idx = np.arange(R * C).reshape(R, C)
        rows, cols = [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r0, r1 = 0, R - dr
            c0, c1 = max(0, -dc), C - max(0, dc)
            ok = free[r0:r1, c0:c1] & free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            if dr and dc:
                ok &= ~(bl[r0 + dr:r1 + dr, c0:c1] & bl[r0:r1, c0 + dc:c1 + dc])
            a, b = idx[r0:r1, c0:c1][ok], idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][ok]
            rows += [a, b]
            cols += [b, a]
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        G = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(R * C, R * C))
        self._graph_cache[g] = G
        return G

    def bfs_from(self, g, cells, min_only=False):
        """그룹 g 지도에서 cells 각각 (또는 min_only: 가장 가까운 것) 으로부터의 BFS 거리. 도달 불가 = inf.
        반환 (len(cells), rows, cols) / min_only (rows, cols)."""
        from scipy.sparse.csgraph import dijkstra
        src = [r * self.cols + c for r, c in cells]
        if not src:
            return None
        d = dijkstra(self._grid_graph(g), indices=src, unweighted=True, min_only=min_only)
        return d.reshape((self.rows, self.cols) if min_only else (len(src), self.rows, self.cols))

    # ================================================================ 2 단계: 유지 UAV 매칭 / 현재 플래그

    def _assignee_cost(self, l, cells):
        """기존 배정자 l 의 구간 매칭 비용: l 의 goal 셀 (goal latch 셀) 이 구간에 있으면 0, 아니면 l 의 위치에서 구간의 가장
        가까운 셀까지 BFS (도달 불가 10^6)."""
        a = self.agents[l]
        if a.goal is not None and tuple(a.goal) in cells:
            return 0.0
        arr = np.asarray(sorted(cells))
        fd = self.fields[l][arr[:, 0], arr[:, 1]]
        fd = fd[fd >= 0]
        return float(fd.min()) if len(fd) else 10.0 ** 6

    def _keepers(self, g, b):
        """덩어리 b 의 지금 배정자 (EXPLORE, 소속 b) — 구간 분할 warm start (각자의 goal 근처 셀) 기준."""
        return [j for j in self.groups[g] if self.agents[j].mode is Mode.EXPLORE and self.agents[j].blob == b.id]

    def _seed_keepers(self, g, b):
        """구간 분할 warm start 의 (배정자, {배정자: goal}). 결정 이벤트 중이면 결정 전 스냅샷 (sim._seed_state) — 이번 이벤트에
        1 단계로 새로 고른 UAV 는 seed 에 넣지 않는다. 아니면 지금 배정자 · goal."""
        snap = getattr(self, "_seed_state", None)
        if snap is None:
            return self._keepers(g, b), None
        uavs = [j for j in self.groups[g] if snap[j][0] is Mode.EXPLORE and snap[j][1] == b.id]
        return uavs, {j: snap[j][2] for j in uavs}

    def _goal_seeds(self, g, cells, uavs, goals=None, fallback=None):
        """{UAV: cells 중 그 UAV goal 셀에 측지 (그룹 지도 BFS) 거리 가장 가까운 셀} — k-medoids warm start 용, 셀 중복 없음.
        goals 를 주면 그 goal (스냅샷) 을, 아니면 지금 goal 을 쓴다. goal 이 없으면 fallback[UAV] (예: 직전 구간 medoid) 기준,
        그것도 없으면 빠진다."""
        index = {c: n for n, c in enumerate(cells)}
        out, taken = {}, set()
        for l in uavs:
            ref = goals[l] if goals is not None else self.agents[l].goal
            if ref is None and fallback is not None:
                ref = fallback[l]
            if ref is None:
                continue
            ref = tuple(ref)
            n0 = index.get(ref)
            if n0 is None or n0 in taken:
                n0 = self._nearest_index(g, ref, cells, taken)
            if n0 is not None:
                out[l] = cells[n0]
                taken.add(n0)
        return out

    def goal_segment(self, i, segs):
        """2 단계 현재 플래그: UAV i 의 현재 goal 셀이 든 구간 index (없으면 None — 새로 합류하는 UAV 는 항상 None)."""
        goal = self.agents[i].goal
        if goal is None:
            return None
        goal = tuple(goal)
        return next((s_ for s_, (cells, _) in enumerate(segs) if goal in cells), None)

    # ================================================================ 그래프 K 노드 / 간선 (env/role/graph_obs.py)

    def standing_on(self, g, bid, exclude=None):
        """1 단계 n_k: 덩어리 bid 에 배정이 유지되는 같은 그룹 UAV (이번 step 결정하지 않는 UAV) 수."""
        return sum(1 for j in self.groups[g] if j != exclude and self.agents[j].mode is Mode.EXPLORE
                   and self.agents[j].blob == bid and self.kind[j] is None)

    def blob_edges(self, g):
        """1 단계 간선: 슬롯 덩어리끼리 가장 가까운 셀 사이 BFS / d_max (K, K), 도달 불가 / 패딩 1. step 내 캐시."""
        key = ("blob_edges", g)
        if key not in self._match_cache:
            slots = self.group_clusters[g]
            K = self.k_max
            out = np.ones((K, K), np.float32)
            fields = [self.bfs_from(g, sorted(b.cell_set()), min_only=True) for b in slots]
            for a_, fa in enumerate(fields):
                for b_, bb in enumerate(slots):
                    if a_ == b_:
                        out[a_, b_] = 0.0
                        continue
                    arr = np.asarray(sorted(bb.cell_set()))
                    d = float(np.min(fa[arr[:, 0], arr[:, 1]]))
                    out[a_, b_] = min(1.0, d / self.d_max) if np.isfinite(d) else 1.0
            self._match_cache[key] = out
        return self._match_cache[key]

    def seg_edges(self, g, b, kk):
        """2 단계 간선: kk 구간 medoid 사이 frontier 측지 거리 / d_max (k-medoids 와 같은 행렬), 끊기면 1. (kk, kk)."""
        segs = self.segments(g, b, kk)
        cells = self._blob_cells_for_split(g, b)
        D = self._geo_cache.get((g, b.id))
        index = {c: n for n, c in enumerate(cells)}
        out = np.ones((kk, kk), np.float32)
        for a_, (_, ma) in enumerate(segs):
            for b_, (_, mb) in enumerate(segs):
                d = D[index[ma], index[mb]] if D is not None and ma in index and mb in index else np.inf
                out[a_, b_] = 0.0 if a_ == b_ else (min(1.0, d / self.d_max) if np.isfinite(d) else 1.0)
        return out
