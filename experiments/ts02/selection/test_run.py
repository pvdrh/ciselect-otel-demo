from collections import Counter

import math

from run import METHODS, diag, run_seed
from test_models import mk

TINY = {"b2": [(1, "callers"), (None, "undirected")], "b3": [1.0],
        "lgbm": [{"num_leaves": 15, "learning_rate": 0.1, "n_estimators": 20, "min_child_samples": 5}],
        "lr": [{"C": 1.0}]}
TINY["lgbm_nofr"] = TINY["lgbm"]


def _data():
    d = mk(40)
    for m in d.mutants:  # has_fail và type cho phân tầng C4
        m["has_fail"] = int(any(r["label"] for r in d.labels if r["mutant_id"] == m["mutant_id"]))
        m["type"] = "M2" if m["has_fail"] else "M0"
    return d


def test_run_seed_covers_every_mutant_once_per_method_and_op():
    d = _data()
    rows, per = run_seed(d, seed=0, grids=TINY, k=5)
    ops = Counter((r["method"], r["op"]) for r in per)
    assert set(m for m, _ in ops) == set(METHODS)
    assert all(n == len(d.mutants) for n in ops.values())  # mỗi mutant đúng một fold test
    for (method, op), _ in ops.items():
        ids = [r["mutant_id"] for r in per if (r["method"], r["op"]) == (method, op)]
        assert len(set(ids)) == len(ids)
    assert {"t95", "t99", "match_b1", "match_b1h", "match_b2"} <= {r["op"] for r in per if r["method"] == "lgbm"}
    b0 = [r for r in per if r["method"] == "b0"]
    assert all(r["caught"] for r in b0 if r["has_fail"])
    assert {r["fold"] for r in rows} == set(range(5))


def test_run_seed_deterministic():
    d = _data()
    (r1, p1), (r2, p2) = run_seed(d, 0, TINY, 5), run_seed(d, 0, TINY, 5)
    strip = lambda rows: [{k: v for k, v in r.items() if not k.startswith("time_")} for r in rows]  # noqa: E731
    assert p1 == p2 and strip(r1) == strip(r2)


def test_diag_by_hand():
    """#47 Q3: PR-AUC (average precision) và ROC-AUC tính tay."""
    sc = {("m", "a"): 0.9, ("m", "b"): 0.8, ("m", "c"): 0.1}
    pr, roc = diag(sc, {("m", "a"), ("m", "c")})
    assert math.isclose(pr, (1 + 2 / 3) / 2) and math.isclose(roc, 0.5)
    assert all(math.isnan(x) for x in diag(sc, set()))  # một lớp: không định nghĩa


def test_rows_have_diag_only_for_scored_methods():
    d = _data()
    rows, _ = run_seed(d, seed=0, grids=TINY, k=5)
    for r in rows:
        if r["method"] in ("b3", "lr", "lgbm", "lgbm_nofr"):
            assert math.isnan(r["pr_auc"]) or 0.0 <= r["pr_auc"] <= 1.0
        else:
            assert r["pr_auc"] is None and r["roc_auc"] is None


def test_run_seed_methods_subset_keeps_baselines_unchanged():
    """#51: run_seed(methods=("rf",)) chạy baseline như cũ cộng rf; mặc định giữ bộ D2."""
    d = _data()
    rows, _ = run_seed(d, 0, TINY | {"rf": [{"n_estimators": 10}]}, 5, methods=("rf",))
    assert {r["method"] for r in rows} == {"b0", "b1", "b1h", "b2", "rf"}
    strip = lambda rs: [{k: v for k, v in r.items() if not k.startswith("time_")} for r in rs]  # noqa: E731
    base = [r for r in run_seed(d, 0, TINY, 5)[0] if r["method"] in ("b0", "b1", "b1h", "b2")]
    assert strip([r for r in rows if r["method"] != "rf"]) == strip(base)


def test_rf_grid_has_32_distinct_configs():
    import json
    from run import GRIDS
    g = GRIDS["rf"]
    assert len(g) == len({json.dumps(c, sort_keys=True) for c in g}) == 32
    assert {c["class_weight"] for c in g} == {None, "balanced"} and {c["max_features"] for c in g} == {"sqrt", 0.5}


def test_b3k_equals_b3_when_every_service_has_one_file_kind():
    d = _data()  # mk: mọi mutant có kinds = ("code",)
    grids = dict(TINY, b3k=TINY["b3"])
    _, p3 = run_seed(d, 0, grids, 5, methods=("b3",))
    _, pk = run_seed(d, 0, grids, 5, methods=("b3k",))
    key = lambda r: (r["fold"], r["op"], r["mutant_id"])  # noqa: E731
    a = {key(r): (r["cost"], r["caught"], r["e2e"]) for r in p3 if r["method"] == "b3"}
    b = {key(r): (r["cost"], r["caught"], r["e2e"]) for r in pk if r["method"] == "b3k"}
    assert a and a == b
