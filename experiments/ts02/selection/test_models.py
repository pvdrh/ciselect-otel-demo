import copy
import math

import numpy as np

from data import COLLECTOR, Data
from features import FEATURES, distances, features
from models import fit_predict

EDGES = {("fe", "ck"), ("ck", "pay"), ("ck", "cart")}  # fe gọi ck, ck gọi pay và cart
SVCS = ["fe", "ck", "pay", "cart"]
TP = "test_traces::test_service_has_traces[pay]"
TE = "test_traces_edges::test_service_edge_exists[fe->ck]"
TESTS = [COLLECTOR, TP, TE]


def mk(n=12):
    ms = [{"mutant_id": f"m{i}", "type": "M2", "service": SVCS[i % 4], "has_fail": 0,
           "record": {"service": SVCS[i % 4], "kinds": ("code",), "n_files": 1, "n_lines": 2, "in_scope": True,
                      "comment_only": False}} for i in range(n)]
    labels = [{"mutant_id": m["mutant_id"], "test_id": t, "label": int(t == TP and m["service"] in {"pay", "ck"})}
              for m in ms for t in TESTS]
    return Data(ms, TESTS, labels, {TP: 1.0, TE: 2.0}, 10.0, 70.0, {s: 1.0 for s in SVCS}, EDGES)


def test_distances_directions():
    # từ S tới service của test: ngược chiều gọi (bên gọi S), xuôi chiều gọi (S gọi), vô hướng
    assert distances(EDGES, "pay", {"ck"}) == (1, math.inf, 1)
    assert distances(EDGES, "pay", {"fe"}) == (2, math.inf, 2)
    assert distances(EDGES, "fe", {"pay"}) == (math.inf, 2, 2)
    assert distances(EDGES, "cart", {"pay"}) == (math.inf, math.inf, 2)
    assert distances(EDGES, "pay", {"pay", "ck"}) == (0, 0, 0)


def test_feature_matrix_shape_and_no_mutant_type():
    d = mk()
    X, keys = features(d, [m["mutant_id"] for m in d.mutants[:8]], [m["mutant_id"] for m in d.mutants[8:]])
    assert X.shape == (4 * len(TESTS), len(FEATURES))
    assert keys[0] == ("m8", COLLECTOR)
    assert not any("M0" in f or "type" in f for f in FEATURES)  # C6: không dùng nhãn loại mutant
    assert np.isfinite(X).all()


def test_no_leakage_from_test_fold_labels():
    d = mk()
    train, test = [m["mutant_id"] for m in d.mutants[:8]], [m["mutant_id"] for m in d.mutants[8:]]
    X1, _ = features(d, train, test)
    d2 = copy.deepcopy(d)
    for r in d2.labels:
        if r["mutant_id"] in test:
            r["label"] = 1 - r["label"]
    X2, _ = features(d2, train, test)
    assert (X1 == X2).all()


def test_models_deterministic_and_learn_signal():
    d = mk(40)
    train, test = [m["mutant_id"] for m in d.mutants[:32]], [m["mutant_id"] for m in d.mutants[32:]]
    for kind, params in [("lgbm", {"num_leaves": 15, "learning_rate": 0.1, "n_estimators": 100, "min_child_samples": 10}),
                         ("lr", {"C": 1.0})]:
        s1 = fit_predict(kind, params, d, train, test, seed=0)
        s2 = fit_predict(kind, params, d, train, test, seed=0)
        assert s1 == s2
        lab = {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
        pos = [v for k, v in s1.items() if lab[k]]
        neg = [v for k, v in s1.items() if not lab[k]]
        assert min(pos) > max(neg)  # tín hiệu tách biệt hoàn toàn trong dữ liệu giả


def test_train_rows_fail_rate_is_out_of_fold():
    """Review D2-7 #1: dòng train không được mã hoá chính nhãn của nó (target leakage)."""
    import zlib
    d = mk(40)
    train = [m["mutant_id"] for m in d.mutants]
    X, keys = features(d, train, train)
    col = FEATURES.index("fail_rate")
    svc = {m["mutant_id"]: m["service"] for m in d.mutants}
    lab = {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
    for (mid, t), x in zip(keys, X[:, col]):
        f = zlib.crc32(mid.encode()) % 5
        others = [o for o in train if svc[o] == svc[mid] and zlib.crc32(o.encode()) % 5 != f]
        assert x == (sum(lab[o, t] for o in others) + 1.0) / (len(others) + 2.0)


def test_features_can_drop_fail_rate():
    """#47 Q1: phân tích phụ B4 không có fail_rate — bỏ đúng cột đó, các cột khác giữ nguyên."""
    d = mk()
    tr, te = [m["mutant_id"] for m in d.mutants[:8]], [m["mutant_id"] for m in d.mutants[8:]]
    X, _ = features(d, tr, te)
    Xn, _ = features(d, tr, te, drop=("fail_rate",))
    col = FEATURES.index("fail_rate")
    assert Xn.shape == (X.shape[0], X.shape[1] - 1)
    assert (Xn == np.delete(X, col, axis=1)).all()


def test_class_weight_reaches_model():
    """#47 Q2: class_weight trong grid đổi điểm của cả LR và LGBM."""
    d = mk(40)
    for r in d.labels:  # làm mất cân bằng hơn: chỉ giữ dương ở pay
        if d.mutants[int(r["mutant_id"][1:])]["service"] != "pay":
            r["label"] = 0
    train, test = [m["mutant_id"] for m in d.mutants[:32]], [m["mutant_id"] for m in d.mutants[32:]]
    for kind, params in [("lr", {"C": 1.0}),
                         ("lgbm", {"num_leaves": 15, "learning_rate": 0.1, "n_estimators": 20, "min_child_samples": 5})]:
        a = fit_predict(kind, {**params, "class_weight": None}, d, train, test, seed=0)
        b = fit_predict(kind, {**params, "class_weight": "balanced"}, d, train, test, seed=0)
        assert a != b, kind


def test_nofr_ignores_fail_rate():
    """#47 Q1: lgbm_nofr chạy được và trả điểm hợp lệ; việc bỏ đúng cột kiểm ở test_features_can_drop_fail_rate."""
    d = mk(40)
    train, test = [m["mutant_id"] for m in d.mutants[:32]], [m["mutant_id"] for m in d.mutants[32:]]
    params = {"num_leaves": 15, "learning_rate": 0.1, "n_estimators": 20, "min_child_samples": 5}
    s = fit_predict("lgbm_nofr", params, d, train, test, seed=0)
    assert len(s) == len(test) * len(d.tests) and all(0.0 <= v <= 1.0 for v in s.values())


def test_rf_kind_builds_random_forest():
    """#51 R2: thiếu nhánh "rf" thì _model âm thầm trả LogisticRegression."""
    from sklearn.ensemble import RandomForestClassifier
    from models import N_JOBS, _model
    m = _model("rf", {"n_estimators": 10, "max_depth": None}, 3)
    assert isinstance(m, RandomForestClassifier) and (m.n_jobs, m.random_state, m.n_estimators) == (N_JOBS, 3, 10)
    d, ids = mk(), [f"m{i}" for i in range(12)]
    assert fit_predict("rf", {"n_estimators": 10}, d, ids[:8], ids[8:], 0) == \
        fit_predict("rf", {"n_estimators": 10}, d, ids[:8], ids[8:], 0)
