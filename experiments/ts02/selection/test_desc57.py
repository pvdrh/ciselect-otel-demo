"""#57/#58 (plan 2026-10-08-phan-tich-o4-o5.md bản 3): test trên dữ liệu tính tay cho desc57."""
import math
import random

import pandas as pd
import pytest

from data import COLLECTOR, Data
from desc57 import agree, boot_cells, cell_counts, cross_only, groups, in_ci, miss_split, o5_pairs
from evaluate import Decision
from stats import bootstrap_diff


def test_groups_by_mechanism():
    rows = [{"mutant_id": "a", "type": "M3", "service": "cart", "site_id": "M3:cart:OTEL_EXPORTER_OTLP_ENDPOINT:host"},
            {"mutant_id": "b", "type": "M3", "service": "fe", "site_id": "M3:fe:OTEL_COLLECTOR_HOST:host"},
            {"mutant_id": "c", "type": "M3", "service": "fe", "site_id": "M3:fe:FRONTEND_HOST:host"},
            {"mutant_id": "d", "type": "M2", "service": "load-generator", "site_id": "M2:load-generator:src/x.py:3"},
            {"mutant_id": "e", "type": "M3", "service": "load-generator",
             "site_id": "M3:load-generator:OTEL_EXPORTER_OTLP_ENDPOINT:port"}]
    tel, lg = groups(rows, labeled={"a", "b", "c", "d", "e"})
    assert tel == {"a", "b", "e"} and lg == {"d", "e"}
    assert groups(rows, labeled={"a", "d"}) == ({"a"}, {"d"})  # chỉ mutant có nhãn


def _per(rows):
    cols = ["seed", "method", "op", "mutant_id", "type", "has_fail", "e2e", "caught"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def test_miss_split_counts_and_pooled_share():
    per = _per([(0, "b3", "t95", "m1", "M2", 1, 0, 0),  # sót (ii)
                (0, "b3", "t95", "m2", "M2", 1, 1, 0),  # sót (iii)
                (0, "b3", "t95", "m3", "M2", 1, 1, 1),
                (0, "b3", "t95", "m4", "M0", 0, 0, 0),  # không có fail: không tính
                (1, "b3", "t95", "m1", "M2", 1, 1, 1), (1, "b3", "t95", "m2", "M2", 1, 1, 1),
                (1, "b3", "t95", "m3", "M2", 1, 1, 1)])
    r = miss_split(per).set_index(["method", "op"]).loc[("b3", "t95")]
    assert (r.n_fail, r.miss, r.miss_ii, r.miss_iii) == (6, 2, 1, 1)
    assert r.share_iii == 0.5 and r.n_seed_no_miss == 1  # seed 1 không sót: bỏ khỏi trung bình theo seed
    assert math.isclose(r.miss_mean, 1.0) and math.isclose(r.share_iii_seed_mean, 0.5)


def test_miss_split_rejects_caught_without_e2e():
    with pytest.raises(AssertionError):
        miss_split(_per([(0, "b3", "t95", "m1", "M2", 1, 0, 1)]))


def test_cross_only_needs_every_fail_outside_service():
    tests = [COLLECTOR, "test_traces::test_service_has_traces[a]", "test_logs::test_service_has_logs[b]",
             "test_traces_edges::test_service_edge_exists[a->b]"]
    lab = lambda m, fails: [{"mutant_id": m, "test_id": t, "label": int(t in fails)} for t in tests]  # noqa: E731
    labels = (lab("x", {tests[2]}) + lab("y", {tests[1], tests[2]}) + lab("z", {tests[0]}) + lab("w", {tests[3]})
              + lab("v", set()))
    ms = [{"mutant_id": m, "service": "a"} for m in "xyzwv"]
    assert cross_only(Data(ms, tests, labels, {}, 0.0, 0.0, {})) == {"x"}  # collector, cạnh a->b đều chạm a


def test_boot_cells_resamples_whole_cells():
    # ô A: 1 mutant, A tiết kiệm hơn B 1 đơn vị trên base 1; ô B: 3 mutant không chênh
    ca, cb, base = [0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]
    cell = ["A", "B", "B", "B"]
    (lo, hi), (rlo, rhi), m, *_ = boot_cells(ca, cb, base, [0] * 4, [0] * 4, [0] * 4, cell, n=4000, seed=0)
    assert (lo, hi) == (0.0, 1.0)  # chỉ có thể là 0 (BB), 0,25 (AB), 1 (AA)
    assert m == 0 and math.isnan(rlo) and math.isnan(rhi)  # không có mutant fail: không có khoảng recall


def test_boot_cells_constant_difference_and_recall():
    ca, cb, base = [1, 1, 2, 2], [2, 2, 4, 4], [10, 10, 20, 20]
    (lo, hi), (rlo, rhi), m, *_ = boot_cells(ca, cb, base, [1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], ["A", "A", "B", "B"],
                                      n=500, seed=0)
    assert math.isclose(lo, 0.1) and math.isclose(hi, 0.1) and (rlo, rhi) == (1.0, 1.0) and m == 500


def test_boot_cells_singletons_match_bootstrap_diff_bit_for_bit():
    # #61 E-4: mỗi mutant một cụm thì cùng chuỗi Random(0).randrange(k), cùng thứ tự cộng với stats.bootstrap_diff
    rng = random.Random(1)
    ca = [rng.uniform(0, 100) for _ in range(40)]
    cb = [x + rng.uniform(-30, 31) for x in ca]
    base = [rng.uniform(100, 200) for _ in range(40)]
    ci, _, _, p, q, _ = boot_cells(ca, cb, base, [0] * 40, [0] * 40, [0] * 40, list(range(40)), n=2000, seed=0)
    lo, hi, p0 = bootstrap_diff(ca, cb, base, n=2000, seed=0)
    assert ci == (lo, hi) and p == p0 and 0 < p < 1 and p == min(1.0, 2 * q)


def test_boot_cells_tail_fractions():
    # tiết kiệm không chênh: mọi d = 0 nên q = 1, p = 1; recall: d_r = 0 chỉ ở lần lấy (B, B), khoảng 1/4
    _, _, n_r, p, q, q_r = boot_cells([1, 1], [1, 1], [1, 1], [1, 0], [0, 0], [1, 1], ["A", "B"], n=4000, seed=0)
    assert (p, q, n_r) == (1.0, 1.0, 4000) and abs(q_r - 0.25) < 0.03
    *_, q_r = boot_cells([0], [1], [1], [0], [0], [0], ["A"], n=10)
    assert math.isnan(q_r)  # không lần lấy nào có mutant fail


def test_o5_pairs_has_no_p_column():
    # #59 giữ "không p" cho O5; boot_cells trả thêm p nhưng nơi gọi trong O5 bỏ đi
    rng, rows = random.Random(0), []
    for seed in (0, 1):
        for method in ("lgbm", "b3k", "b3"):
            for op in ("t95", "t99"):
                for i in range(4):
                    rows.append({"seed": seed, "method": method, "op": op, "mutant_id": f"m{i}", "cost": rng.uniform(0, 5),
                                 "caught": rng.randint(0, 1), "has_fail": i % 2, "cost_b0prime": 10.0})
    out = o5_pairs(pd.DataFrame(rows), {f"m{i}": ("s", ("code",)) if i < 2 else ("t", ("code",)) for i in range(4)}, 1.0)
    assert len(out) == 4 and not {"p", "p_holm", "cell_p"} & set(out.columns)
    assert {"cell_ci_lo", "cell_ci_hi", "cell_r_ci_lo", "cell_r_ci_hi"} <= set(out.columns)


def test_in_ci_needs_same_sign_and_inside():
    assert in_ci(0.05, 0.04, 0.06, 0.05) and not in_ci(0.07, 0.04, 0.06, 0.05)
    assert not in_ci(-0.001, -0.01, 0.01, 0.002)  # trong khoảng nhưng đổi dấu


def test_agree_ignores_tests_when_both_skip_e2e():
    skip_a, skip_b = Decision({"s"}, False, {"t1"}), Decision({"s"}, False, set())
    assert agree(skip_a, skip_b) == (True, True)
    assert agree(Decision({"s"}, True, {"t1"}), Decision({"s"}, True, {"t1", "t2"})) == (True, False)
    assert agree(Decision({"s"}, True, {"t1"}), skip_b) == (False, False)


def test_cell_counts_outer_and_small_cells():
    ms = [{"mutant_id": f"m{i}", "service": "a", "record": {"kinds": ("code",) if i < 3 else ("config",)}}
          for i in range(5)]
    splits = [(["m0", "m1", "m3"], ["m2", "m4"]), (["m0", "m1", "m2"], ["m3", "m4"])]
    # split 1: m2 (code, n_k = 2) và m4 (config, n_k = 1) là ô nhỏ; split 2: m3, m4 ở ô config rỗng
    assert cell_counts(ms, splits) == {"empty": 2, "small": 2}
