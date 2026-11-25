import pandas as pd
import pytest

from summarize import main_tests, pooled


def _per(rows):
    cols = ["seed", "method", "op", "mutant_id", "has_fail", "caught", "cost", "cost_b0prime", "in_scope", "type", "service"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


# 2 seed × 2 mutant (m1 fail, m2 không); B0′ = 100 mỗi mutant.
ROWS = [(s, "lgbm", "t95", "m1", 1, c, 60.0, 100.0, True, "M2", "cart") for s, c in ((0, 1), (1, 0))] + \
       [(s, "lgbm", "t95", "m2", 0, 0, 40.0, 100.0, True, "M0", "cart") for s in (0, 1)] + \
       [(s, "b3", "t95", m, f, f, 90.0, 100.0, True, t, "cart") for s in (0, 1) for m, f, t in (("m1", 1, "M2"), ("m2", 0, "M0"))]


def test_pooled_by_hand():
    p = pooled(_per(ROWS)).set_index(["method", "op"])
    assert p.loc[("lgbm", "t95"), "savings_mean"] == pytest.approx(0.5)  # 1 − 100/200 mỗi seed
    assert p.loc[("lgbm", "t95"), "recall_mutant_mean"] == pytest.approx(0.5)  # seed 0: 1, seed 1: 0
    assert p.loc[("b3", "t95"), "savings_mean"] == pytest.approx(0.1)


def test_main_tests_average_seeds_per_mutant():
    """#49 Q1: trung bình chi phí và tỉ lệ bắt theo mutant qua seed, rồi bootstrap một lần."""
    t = main_tests(_per(ROWS), pairs=[("b3", "t95", "b3", "t95")], n=500).iloc[0]
    assert t["d_savings"] == pytest.approx(0.4)  # (90+90 − 60−40)/200
    assert t["d_recall"] == pytest.approx(-0.5)  # m1: lgbm bắt 0,5 lần, b3 bắt 1
    assert not t["pareto"] and t["p_holm"] == t["p"]


def test_by_seed_point_diffs():
    """#49 Q1 độ nhạy: hiệu điểm theo từng seed."""
    from summarize import by_seed
    t = by_seed(_per(ROWS), pairs=[("b3", "t95", "b3", "t95")]).set_index("seed")
    assert t.loc[0, "d_savings"] == pytest.approx(0.4) and t.loc[0, "d_recall"] == pytest.approx(0.0)
    assert t.loc[1, "d_recall"] == pytest.approx(-1.0)


def test_main_tests_and_by_seed_take_left_method():
    """#51 R3: vế trái không gắn cứng lgbm."""
    from summarize import by_seed
    t = main_tests(_per(ROWS), pairs=[("lgbm", "t95", "lgbm", "t95")], n=500, left="b3").iloc[0]
    assert t["pair"] == "b3[t95] vs lgbm[t95]" and t["d_savings"] == pytest.approx(-0.4)
    assert t["savings_b3"] == pytest.approx(0.1)
    s = by_seed(_per(ROWS), pairs=[("lgbm", "t95", "lgbm", "t95")], left="b3").set_index("seed")
    assert s.loc[1, "d_recall"] == pytest.approx(1.0)
