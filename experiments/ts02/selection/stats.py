"""D2-2: kiểm định (C5) — bootstrap cặp theo cụm mutant, Holm."""
import random


def bootstrap_diff(cost_a, cost_b, cost_base, n=10_000, seed=0):
    """Hiệu tiết kiệm A − B = (Σb − Σa)/Σbase, mỗi phần tử một mutant. Trả (lo, hi) 95% và p hai phía."""
    rng, k = random.Random(seed), len(cost_a)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        diffs.append((sum(cost_b[i] for i in idx) - sum(cost_a[i] for i in idx)) / sum(cost_base[i] for i in idx))
    diffs.sort()
    p = min(1.0, 2 * min(sum(x <= 0 for x in diffs), sum(x >= 0 for x in diffs)) / n)
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1], p


def recall_ci(caught, n=10_000, seed=0):
    """caught: 0/1 cho từng mutant fail. Khoảng tin cậy bootstrap 95% của recall theo mutant (#41)."""
    rng, k = random.Random(seed), len(caught)
    rs = sorted(sum(caught[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return rs[int(0.025 * n)], rs[int(0.975 * n) - 1]


def bootstrap_recall_diff(caught_a, caught_b, n=10_000, seed=0):
    """#48: hiệu recall theo mutant A − B, bootstrap cặp trên các mutant fail. Trả (lo, hi) 95%."""
    rng, k = random.Random(seed), len(caught_a)
    ds = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        ds.append(sum(caught_a[i] - caught_b[i] for i in idx) / k)
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out
