"""결정 순서 — 한 이벤트 안에서 그룹마다 '다음에 결정할 UAV' 와 그 담당 후보를 정한다 (env / 그리디 / 학습 모델 공통).

세 함수 모두 torch 텐서 (B, N, ...) 를 받아 그룹마다 맨 앞 UAV 하나를 고른다. 학습 모델 (models/role/net.py) 은 한 대가 결정할
때마다 실제 선택을 반영해 다시 부르고, env (env/role/simulation.py _decide) 도 같은 관측 값으로 같은 함수를 부른다
(그리디는 담당 후보를 그대로 고르므로 결과가 '한 번 계산한 전체 순서' 와 같다). 입력은 전부 관측 값 그대로
(거리 = bfs_dist 특징 d / d_max, 여유 = slack 특징, 크기 = explore_size / explore_seg_size) 라서 env 와 모델의 동률 판정이 같다.

role_front (B-1)    U = 그룹이 아직 맡지 않은 할당 가능 생존자 (UAV 별 rescue 슬롯), D = 남은 재결정 UAV
    생존자를 σ_u = max_{i ∈ D_0} slack_i(u) (= (τ_u - t - min_i d_iu) / T, 이벤트 시작 D_0 기준 고정) 오름차순 -> 생존자 번호,
    가장 급한 생존자부터 '닿는 (그 UAV 슬롯에 있음) 용량 > 0 UAV' 중 (거리, -용량, 번호) 최소가 다음 차례 (담당 후보 = 그 생존자).
    닿는 UAV 가 없으면 (용량 0 여부, U 중 가장 가까운 생존자 거리 (없으면 ∞), -용량, 번호) 최소.
explore_front (B-3) K = 그룹 덩어리 슬롯 (크기 내림차순), n_k = 배정 수 (기존 + 이번 이벤트 앞 결정)
    n_k = 0 인 덩어리를 슬롯 순으로, 닿는 (explore_mask) UAV 중 (거리, 번호) 최소가 다음 차례 (담당 후보 = 그 덩어리).
    n_k = 0 인 덩어리에 닿는 UAV 가 없으면 |C_k| / (n_k + 1) 최대 (같으면 큰 덩어리, 슬롯 순) 덩어리에 가장 가까운 UAV.
seg_front (B-5)     덩어리마다 참여자 P, 남은 구간을 크기 내림차순 -> 구간 번호, 그 구간에 가장 가까운 참여자 (거리, 번호).
"""
import torch

INF = float("inf")


def _filter(C, key):
    """C (bool) 중 key 최소인 것만 남김 (마지막 축들 전체에서). key 는 C 로 broadcast."""
    k = torch.where(C, key.expand_as(C).float(), torch.full_like(C, INF, dtype=torch.float32))
    m = k.flatten(2).min(-1).values
    return C & (k == m.view(*m.shape, *([1] * (C.dim() - 2))))


def _lexmin(C, keys):
    for key in keys:
        C = _filter(C, key)
    return C


def _per_group(C, group, rem):
    """C [b,i,j,...] (i = 그룹 기준 UAV) -> sel (B,N): 그 그룹에서 고른 j, 고른 위치 C 의 j 행 (B,N,...)."""
    pick = C.flatten(3).any(-1) if C.dim() > 3 else C                 # [b,i,j]
    sel = (pick & group).any(1) & rem
    row = (C & group.view(*group.shape, *([1] * (C.dim() - 3)))).any(1)   # [b,j,...]
    return sel, row


def role_sigma(ids, slack, valid, group, d0):
    """σ (B,N,T): UAV i 슬롯 s 의 생존자 u 에 대해 max_{j ∈ D_0 ∩ 그룹(i), u ∈ 슬롯 j} slack_j(u). 이벤트 시작에 한 번."""
    same = (ids[:, :, :, None, None] == ids[:, None, None, :, :]) & valid[:, :, :, None, None] & valid[:, None, None, :, :]
    same = same & (group & d0[:, None, :])[:, :, None, :, None]
    s = torch.where(same, slack[:, None, None, :, :].expand_as(same), torch.full_like(same, -INF, dtype=torch.float32))
    sig = s.flatten(3).max(-1).values
    return torch.where(torch.isfinite(sig), sig, torch.full_like(sig, INF))


def role_front(group, rem, cap, ids, d, sig, avail):
    """반환 (sel (B,N) bool — 그룹마다 다음 차례 UAV, claim (B,N) long — 그 UAV 의 담당 후보 슬롯 (없으면 -1))."""
    B, N, T = ids.shape
    dev = ids.device
    idx = torch.arange(N, device=dev, dtype=torch.float32)
    pair = (rem & (cap > 0))[:, :, None] & avail                                         # [b,j,s]
    C = group[:, :, :, None] & pair[:, None, :, :]                                       # [b,i,j,s]
    C = _lexmin(C, [sig[:, None], ids[:, None], d[:, None], -cap[:, None, :, None], idx.view(1, 1, N, 1)])
    has = C.flatten(2).any(-1)                                                           # [b,i]
    near = torch.where(avail, d, torch.full_like(d, INF)).min(-1).values                 # [b,j]
    F = group & rem[:, None, :] & ~has[:, :, None]                                       # [b,i,j]
    F = _lexmin(F, [(cap <= 0).float()[:, None], near[:, None], -cap[:, None], idx.view(1, 1, N)])
    sel_c, row = _per_group(C, group, rem)
    sel_f, _ = _per_group(F, group, rem)
    claim = torch.where(sel_c, row.float().argmax(-1), torch.full((B, N), -1, device=dev, dtype=torch.long))
    return sel_c | sel_f, claim


def explore_front(group, rem, emask, d, size, n):
    """반환 (sel (B,N), claim (B,N) 덩어리 슬롯 (없으면 -1)). n / size 는 기준 UAV i (남은 결정 UAV) 의 행을 쓴다
    (결정 UAV 끼리는 같은 값: 기존 배정자 수 + 이번 이벤트 앞 차례 선택)."""
    B, N, K = d.shape
    dev = d.device
    idx = torch.arange(N, device=dev, dtype=torch.float32)
    kk = torch.arange(K, device=dev, dtype=torch.float32)
    pair = rem[:, :, None] & emask                                                       # [b,j,k]
    # 기준 행 i 는 남은 결정 UAV 만 (n_assigned 는 자기 자신을 빼고 세므로, 결정하지 않는 기존 배정자의 행은 다르다)
    C = (group & rem[:, :, None])[:, :, :, None] & pair[:, None, :, :]                   # [b,i,j,k]
    C1 = _lexmin(C & (n[:, :, None, :] == 0), [kk.view(1, 1, 1, K), d[:, None], idx.view(1, 1, N, 1)])
    has1 = C1.flatten(2).any(-1)
    score = size / (n + 1)
    C2 = C & ~has1[:, :, None, None]
    C2 = _lexmin(C2, [-score[:, :, None, :], -size[:, :, None, :], kk.view(1, 1, 1, K), d[:, None], idx.view(1, 1, N, 1)])
    C = C1 | C2
    sel, row = _per_group(C, group, rem)
    claim = torch.where(sel, row.float().argmax(-1), torch.full((B, N), -1, device=dev, dtype=torch.long))
    return sel, claim


def seg_front(same, rem, avail, d, size):
    """같은 덩어리 참여자 그룹 same (B,N,N), 남은 참여자 rem, 고를 수 있는 구간 avail (B,N,M) (참여자마다 같은 덩어리 기준),
    d / size (B,N,M). 반환 (sel, claim 구간)."""
    B, N, M = d.shape
    dev = d.device
    idx = torch.arange(N, device=dev, dtype=torch.float32)
    ss = torch.arange(M, device=dev, dtype=torch.float32)
    C = same[:, :, :, None] & (rem[:, :, None] & avail)[:, None, :, :]                  # [b,i,j,s]
    C = _lexmin(C, [-size[:, :, None, :], ss.view(1, 1, 1, M), d[:, None], idx.view(1, 1, N, 1)])
    sel, row = _per_group(C, same, rem)
    claim = torch.where(sel, row.float().argmax(-1), torch.full((B, N), -1, device=dev, dtype=torch.long))
    return sel, claim
