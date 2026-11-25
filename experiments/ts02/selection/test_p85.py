"""Plan gộp 2026-10-14 (#86): định nghĩa của P1, P4–P7, P9 kiểm trên dữ liệu tính tay."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p85 import be_vs_ref, labels_rule, near_dup_in_train, pair_groups, random_addition, step_break_even

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_labels import build_labels  # noqa: E402


def test_pair_groups_deltas_share_and_seconds():
    bg = pd.DataFrame([{"method": "lgbm", "op": "t95", "group": g, "contrib_mean": v}
                       for g, v in (("M0", 0.03), ("im_lang", 0.03), ("co_fail", 0.02))]
                      + [{"method": "b3", "op": "t95", "group": g, "contrib_mean": v}
                         for g, v in (("M0", 0.0), ("im_lang", 0.01), ("co_fail", 0.01))])
    mt = pd.DataFrame([{"pair": "lgbm[t95] vs b3[t95]", "d_savings": 0.06}])
    r = pair_groups(bg, mt, [("b3", "t95", "b3", "t95")], b0p_s=200.0).iloc[0]
    assert (r.d_M0, r.d_im_lang, r.d_co_fail) == pytest.approx((0.03, 0.02, 0.01))
    assert r.share_M0_im == pytest.approx(0.05 / 0.06) and r.s_co_fail == pytest.approx(2.0)
    with pytest.raises(AssertionError):  # tổng theo nhóm phải bằng hiệu tiết kiệm của cặp
        pair_groups(bg, mt.assign(d_savings=0.07), [("b3", "t95", "b3", "t95")], b0p_s=200.0)


def _table():
    row = lambda run, mid, test, out, status="ok": {"run_id": run, "mutant_id": mid, "test_id": test,  # noqa: E731
                                                     "outcome": out, "status": status}
    return [row("a1", "m", "A", "fail"), row("a1", "m", "B", "fail"), row("a2", "m", "A", "pass"), row("a2", "m", "B", "fail"),
            row("b1", "n", "A", "fail"), row("b1", "n", "B", "pass"), row("b2", "n", "A", "pass"), row("b2", "n", "B", "pass"),
            row("c1", "clean", "A", "pass"), row("c1", "clean", "B", "pass")]


def test_labels_rule_all_matches_build_labels_and_any_keeps_one_run_fails():
    unhealthy = {"b2": {"cart"}}  # n sửa "pay": lượt b2 bị loại vì service khác không healthy
    svc = {"m": "pay", "n": "pay"}
    all_ = labels_rule(_table(), unhealthy, svc, ["A", "B"], all)
    ref, _ = build_labels(_table(), unhealthy, svc)
    assert all_ == {(r["mutant_id"], r["test_id"]): r["label"] for r in ref}
    assert all_ == {("m", "A"): 0, ("m", "B"): 1, ("n", "A"): 1, ("n", "B"): 0}  # n chỉ còn một lượt hợp lệ
    assert labels_rule(_table(), unhealthy, svc, ["A", "B"], any) == {("m", "A"): 1, ("m", "B"): 1, ("n", "A"): 1,
                                                                      ("n", "B"): 0}


def test_near_dup_in_train_needs_same_key_same_failset_and_other_fold():
    key = {"a": "k", "b": "k", "c": "k", "d": "j"}
    fails = {"a": frozenset("X"), "b": frozenset("X"), "c": frozenset("Y"), "d": frozenset("X")}
    fold = {(0, "a"): 0, (0, "b"): 1, (0, "c"): 1, (0, "d"): 0, (1, "a"): 0, (1, "b"): 0, (1, "c"): 1, (1, "d"): 1}
    n = near_dup_in_train(key, fails, fold)
    assert n == {(0, "a"): 1, (0, "b"): 1, (0, "c"): 0, (0, "d"): 0, (1, "a"): 0, (1, "b"): 0, (1, "c"): 0, (1, "d"): 0}


def test_step_break_even_by_hand_and_no_extra_catch():
    ds, db, be = step_break_even(0.089, 0.046, 40, 16, b0p_s=193.0, n=353)
    assert ds == pytest.approx(0.043 * 193.0) and db == pytest.approx(4.8) and be == pytest.approx(0.043 * 193.0 * 353 / 4.8)
    assert math.isnan(step_break_even(0.05, 0.04, 10, 10, b0p_s=193.0, n=353)[2])


def test_be_vs_ref_by_hand():
    # 4 mutant, 2 có fail; X rẻ hơn mốc 10 s ở mọi mutant và sót 1/2 mutant có fail; π = 0,5
    ref = pd.DataFrame({"cost": [20.0] * 4, "caught": [1.0, 1.0, 0.0, 0.0], "has_fail": [1, 1, 0, 0]})
    x = ref.assign(cost=10.0, caught=[1.0, 0.0, 0.0, 0.0])
    case, be, lo, hi = be_vs_ref(x, ref, pi=0.5, m_lo=0.25, m_hi=0.75)
    assert case == "đánh đổi" and be == pytest.approx(10 / (0.5 * 0.5))
    assert lo == pytest.approx(10 / (0.5 * 0.75)) and hi == pytest.approx(10 / (0.5 * 0.25))


def test_random_addition_keeps_t1_adds_exactly_the_b2_count_from_outside_t1():
    t1, t2, pool = {"a", "b"}, {"a", "b", "c", "d"}, ["a", "b", "c", "d", "e", "f"]
    got = random_addition(t1, t2, pool, np.random.default_rng(0))
    assert t1 <= got and len(got) == 4 and got - t1 <= {"c", "d", "e", "f"}
    assert got == random_addition(t1, t2, pool, np.random.default_rng(0))
    assert random_addition(t1, t1, pool, np.random.default_rng(0)) == t1
