"""Plan 2026-10-07 (#51) DoD 1: cận trên oracle trên dữ liệu tính tay của test_eval."""
import itertools
import math

from evaluate import Decision, metrics
from oracle import gap, o_mut, o_test, upper
from test_eval import D, MUT, TESTS

IDS = [m["mutant_id"] for m in MUT]
BASE = 440.0  # B0′: (40 + 100) × 2 + (60 + 100)


def test_o_test_runs_exactly_failing_tests():
    r = metrics(o_test(D, IDS), D)
    assert (r["recall_mutant"], r["recall_test"]) == (1.0, 1.0)
    assert math.isclose(r["savings"], 1 - (137 + 133 + 60) / BASE)  # m1 {a, cạnh} 7 s, m2 {b} 3 s, m3 bỏ e2e


def test_o_mut_runs_cheapest_failing_test():
    dec = o_mut(D, IDS)
    assert dec["m1"].tests == {TESTS[1]} and not dec["m3"].e2e
    r = metrics(dec, D)
    assert r["recall_mutant"] == 1.0
    assert math.isclose(r["savings"], 1 - (132 + 133 + 60) / BASE)
    assert r["savings"] >= metrics(o_test(D, IDS), D)["savings"]


def test_upper_by_hand_and_monotone():
    assert math.isclose(upper(D, IDS, 1.0), metrics(o_mut(D, IDS), D)["savings"])
    assert math.isclose(upper(D, IDS, 0.5), 1 - (132 + 40 + 60) / BASE)  # bắt m1 (rẻ hơn), bỏ m2
    assert math.isclose(upper(D, IDS, 0.0), 1 - (40 + 40 + 60) / BASE)
    us = [upper(D, IDS, r / 100) for r in range(101)]
    assert all(a >= b for a, b in zip(us, us[1:]))


def test_upper_matches_brute_force_over_all_decisions():
    """Vét cạn: mỗi mutant bỏ e2e hoặc chạy một tập con bất kỳ của 4 test (17³ quyết định)."""
    opts = [None] + [set(c) for n in range(len(TESTS) + 1) for c in itertools.combinations(TESTS, n)]
    svc = {m["mutant_id"]: m["service"] for m in MUT}
    best = {}
    for combo in itertools.product(opts, repeat=len(IDS)):
        dec = {i: Decision({svc[i]}, t is not None, t or set()) for i, t in zip(IDS, combo)}
        r = metrics(dec, D)
        best[r["recall_mutant"]] = max(best.get(r["recall_mutant"], -1), r["savings"])
    for r in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert math.isclose(upper(D, IDS, r), max(s for rr, s in best.items() if rr >= r - 1e-12))


def test_gap_uses_exact_recall_not_grid():
    rows = [{"method": "x", "op": "t95", "savings_mean": 0.1, "recall_mutant_mean": 0.5},
            {"method": "y", "op": "t95", "savings_mean": 0.2, "recall_mutant_mean": 0.51}]
    g = gap(D, IDS, rows, "main")
    assert math.isclose(g[0]["upper"], upper(D, IDS, 0.5)) and math.isclose(g[0]["gap"], g[0]["upper"] - 0.1)
    assert math.isclose(g[1]["upper"], upper(D, IDS, 1.0))  # 0,51 × 2 mutant fail → phải bắt cả 2
    assert g[0]["split"] == "main"
