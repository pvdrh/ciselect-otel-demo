import math
import random

import pytest

from data import COLLECTOR, Data
from evaluate import Decision, b0prime, cost, metrics
from stats import bootstrap_diff, bootstrap_recall_diff, holm, recall_ci
from threshold import pick_threshold, select

# 3 mutant × 4 test, tính tay. Thời lượng (lần chạy sạch): t_a 2 s, t_b 3 s, t_e 5 s; collector là W.
TESTS = [COLLECTOR, "test_logs::test_service_has_logs[a]", "test_logs::test_service_has_logs[b]",
         "test_traces_edges::test_service_edge_exists[a->b]"]
C, TA, TB, TE = TESTS
LAB = {("m1", TA): 1, ("m1", TE): 1, ("m2", TB): 1}  # m1 fail ở a và cạnh; m2 fail ở b (xuyên service, S = a); m3 không fail
MUT = [{"mutant_id": "m1", "service": "a", "has_fail": 1}, {"mutant_id": "m2", "service": "a", "has_fail": 1},
       {"mutant_id": "m3", "service": "b", "has_fail": 0}]
D = Data(MUT, TESTS, [{"mutant_id": m["mutant_id"], "test_id": t, "label": LAB.get((m["mutant_id"], t), 0)}
                      for m in MUT for t in TESTS],
         {TA: 2.0, TB: 3.0, TE: 5.0}, W=20.0, t_start=70.0, build_nocache={"a": 40.0, "b": 60.0})


def test_cost_c0_with_39():
    assert cost(Decision({"a"}, False, set()), D) == 40.0
    # e2e: t_start + W + Σ dur, collector không tính lại (#39)
    assert cost(Decision({"a"}, True, {C, TA}), D) == 40.0 + 70.0 + 20.0 + 2.0
    assert cost(b0prime(MUT[2], D), D) == 60.0 + 70.0 + 20.0 + 10.0


def test_metrics_by_hand():
    dec = {"m1": Decision({"a"}, True, {C, TA}),   # bắt m1
           "m2": Decision({"a"}, True, {C, TA}),   # sót m2 (fail ở b) → sót lỗi xuyên service
           "m3": Decision({"b"}, False, set())}
    r = metrics(dec, D)
    base = (40 + 100) * 2 + (60 + 100)  # B0′: build S + t_start + W + 10 s
    got = (40 + 92) * 2 + 60
    assert r["recall_mutant"] == 0.5
    assert math.isclose(r["savings"], 1 - got / base)
    assert math.isclose(r["recall_test"], 1 / 3)
    assert r["cross_missed"] == 1.0  # m2 là mutant xuyên service duy nhất; m1 có nhãn trên cạnh a->b (chạm a) nên không tính
    assert math.isclose(r["savings_ii"], 100 / base)  # m3 bỏ e2e
    assert math.isclose(r["savings_iii"], 2 * 8 / base)  # m1, m2 bỏ TB và TE


def test_pick_threshold_tightest_reaching_recall():
    # điểm cao nhất của mỗi mutant fail trên các test mang nhãn fail: 0,9 / 0,6 / 0,3 / 0,2
    best = {"x1": 0.9, "x2": 0.6, "x3": 0.3, "x4": 0.2}
    assert pick_threshold(best, 0.5) == 0.6
    assert pick_threshold(best, 0.75) == 0.3
    assert pick_threshold(best, 0.95) == 0.2  # ceil(0,95 · 4) = 4
    assert pick_threshold({}, 0.95) == -math.inf  # không có mutant fail: chạy hết


def test_select_runs_e2e_only_when_a_test_passes_threshold():
    assert select({TA: 0.4, TB: 0.7}, 0.5) == ({TB}, True)
    assert select({TA: 0.4, TB: 0.3}, 0.5) == (set(), False)


def test_holm_by_hand():
    assert holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_bootstrap_diff_detects_sign():
    rng = random.Random(0)
    base = [100.0] * 50
    a = [50 + rng.random() for _ in range(50)]  # A rẻ hơn B rõ rệt
    b = [80 + rng.random() for _ in range(50)]
    lo, hi, p = bootstrap_diff(a, b, base, n=2000, seed=0)
    assert 0.25 < lo < hi < 0.35 and p < 0.01
    lo, hi, p = bootstrap_diff(a, a, base, n=2000, seed=0)
    assert lo == hi == 0.0 and p == 1.0


def test_recall_ci_upper_bound():
    lo, hi = recall_ci([1] * 95 + [0] * 5, n=2000, seed=0)
    assert lo < 0.95 < hi


def test_pick_threshold_zero_target_is_tightest():
    """Review D2-7 #2: mục tiêu recall 0 → không cần bắt mutant nào → ngưỡng +∞ (không chọn test)."""
    assert pick_threshold({"x1": 0.9, "x2": 0.2}, 0.0) == math.inf


def test_bootstrap_recall_diff_paired():
    """#48: hiệu recall A − B, bootstrap cặp trên cùng mutant fail."""
    a, b = [1] * 90 + [0] * 10, [1] * 70 + [0] * 30  # A bắt mọi mutant B bắt, thêm 20
    lo, hi = bootstrap_recall_diff(a, b, n=2000, seed=0)
    assert 0.1 < lo < 0.2 < hi < 0.3
    assert bootstrap_recall_diff(a, a, n=500, seed=0) == (0.0, 0.0)  # cặp: cùng dữ liệu → hiệu đúng 0
