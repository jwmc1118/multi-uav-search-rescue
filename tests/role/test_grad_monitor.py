"""공유 encoder gradient 불균형 경고 (training/callbacks.py EncGradMonitor) 합성 점검 (ray 학습 불필요).

    python -m tests.role.test_grad_monitor

 [1] 단발 스파이크 (1 iteration 만 한 head 가 100 배) -> 경고 없음
 [2] 10 배 넘는 불균형이 계속되면 EMA 비율이 10 을 넘은 뒤 10 iteration 연속일 때 처음 경고, 그 전에는 없음
 [3] 불균형이 풀리면 연속 수가 0 으로 돌아감 (다시 10 iteration 이 필요)
 [4] 경고 문자열에 그 기간 head 별 EV / entropy 시작 -> 끝
"""
from training.callbacks import EncGradMonitor

BAL = {"role": 1.0, "exp": 1.5, "res": 2.0}
EV = {"role": 0.5, "explore": 0.4, "rescue": 0.3}
ENT = {"role": 0.6, "explore": 1.0, "rescue": 0.3}


def run(seq):
    m = EncGradMonitor()
    out = []
    for k, norms in enumerate(seq):
        _, ratio, warn = m.update(norms, {h: v + 0.01 * k for h, v in EV.items()}, {h: v - 0.01 * k for h, v in ENT.items()})
        out.append((ratio, m.streak, warn))
    return out


def check_spike():
    seq = [BAL] * 20 + [{"role": 0.01, "exp": 1.5, "res": 2.0}] + [BAL] * 20      # role 만 1 iteration 1/100
    out = run(seq)
    assert not any(w for _, _, w in out), "스파이크에 경고"
    print(f"[1] single spike (role x1/100 for 1 iteration): max EMA ratio {max(r for r, _, _ in out):.2f}, "
          f"max streak {max(s for _, s, _ in out)} -> no warning")


def check_sustained():
    imb = {"role": 0.05, "exp": 2.0, "res": 2.5}                                     # 50 배
    out = run([BAL] * 5 + [imb] * 60)
    first_over = next(k for k, (r, _, _) in enumerate(out) if r >= 10)
    warns = [k for k, (_, _, w) in enumerate(out) if w]
    assert warns and warns[0] == first_over + 9, (first_over, warns)
    assert all(out[k][1] < 10 for k in range(warns[0])), "10 iteration 전에 경고"
    print(f"[2] sustained 50x imbalance from iter 5: EMA ratio >= 10 at iter {first_over}, first warning at iter {warns[0]} "
          f"(10 consecutive), repeats at {warns[1:3]}")
    return out[warns[0]][2]


def check_reset():
    imb = {"role": 0.05, "exp": 2.0, "res": 2.5}
    out = run([BAL] * 5 + [imb] * 22 + [BAL] * 30 + [imb] * 22)
    peak = max(s for _, s, _ in out[:27])
    assert 0 < peak < 10 and out[56][1] == 0 and not any(w for _, _, w in out), (peak, out[56])
    print(f"[3] imbalance 22 iterations (streak reaches {peak}), recovery 30 -> streak 0, imbalance 22 again -> "
          f"no warning (streak restarts)")


if __name__ == "__main__":
    check_spike()
    w = check_sustained()
    check_reset()
    assert "EV" in w and "entropy" in w and "->" in w
    print("[4] warning text:\n" + w)
    print("test_grad_monitor: all ok")
