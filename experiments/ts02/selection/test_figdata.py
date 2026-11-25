import math

import pytest

from evaluate import b0prime, cost
from figdata import curve, importance
from test_run import TINY, _data


def test_curve_endpoints():
    """τ = −∞: chọn hết = B0′ (tiết kiệm 0, recall 1). τ = +∞: không chạy e2e (recall 0, chỉ còn build S)."""
    d = _data()
    ids = [m["mutant_id"] for m in d.mutants]
    sc = {(i, t): 0.5 for i in ids for t in d.tests}
    pts = curve(d, sc, ids, [-math.inf, math.inf])
    assert pts[0] == pytest.approx((-math.inf, 0.0, 1.0))
    base = sum(cost(b0prime(m, d), d) for m in d.mutants)
    assert pts[1][1:] == pytest.approx((1 - sum(d.build_nocache[m["service"]] for m in d.mutants) / base, 0.0))


def test_importance_sums_to_one_and_drops_fail_rate():
    d = _data()
    ids = [m["mutant_id"] for m in d.mutants]
    imp = importance(d, "lgbm_nofr", TINY["lgbm"][0], ids, seed=0)
    assert "fail_rate" not in imp and sum(imp.values()) == pytest.approx(1.0)


def test_curve_matches_metrics():
    import random

    from evaluate import metrics
    from run import _decide
    d, rng = _data(), random.Random(1)
    ids = [m["mutant_id"] for m in d.mutants]
    sc = {(i, t): rng.random() for i in ids for t in d.tests}
    for tau in (0.3, 0.7, 0.95):
        m = metrics(_decide(d, sc, ids, tau), d)
        assert curve(d, sc, ids, [tau])[0][1:] == pytest.approx((m["savings"], m["recall_mutant"]))
