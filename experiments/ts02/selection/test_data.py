from pathlib import Path

import pytest

from data import ALL, folds, load, loso_folds, meta_of
from diff import parse
from record import change_record
from scope import Scope

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "results" / "ts02_pilot" / "20261001-134531_seed1"
BUILD = ROOT / "results" / "build_nocache" / "20261001-122853_seed0"
DEMO = ROOT / "data" / "raw" / "opentelemetry-demo"
need_pilot = pytest.mark.skipif(not (PILOT.exists() and DEMO.exists()), reason="cần pilot D1-5 và clone demo")


def test_test_meta_follows_c1():
    assert meta_of("test_logs::test_service_has_logs[ad]") == {"services": {"ad"}, "kind": "service", "signal": "logs"}
    assert meta_of("test_metrics::test_service_has_metrics[email]")["signal"] == "metrics"
    assert meta_of("test_traces_edges::test_service_edge_exists[frontend->checkout]") == {
        "services": {"frontend", "checkout"}, "kind": "edge", "signal": "traces"}
    assert meta_of("test_collector::test_collector_healthy") == {"services": ALL, "kind": "collector", "signal": "none"}


def test_change_record_kinds():
    allow = {"files": {"src/checkout/main.go": {"service": "checkout", "generated": False},
                       "src/checkout/genproto/oteldemo/demo.pb.go": {"service": "checkout", "generated": True}}}
    code = parse("diff --git a/src/checkout/main.go b/src/checkout/main.go\n@@ -1 +1 @@\n-a\n+b\n")
    assert change_record(code, Scope(True, "checkout", "ok", 1, 2), allow, "checkout") == {
        "service": "checkout", "in_scope": True, "kinds": ("code",), "n_files": 1, "n_lines": 2,
        "comment_only": False}
    proto = parse("diff --git a/pb/demo.proto b/pb/demo.proto\n@@ -1 +1 @@\n-a\n+b\n"
                  "diff --git a/src/checkout/genproto/oteldemo/demo.pb.go b/src/checkout/genproto/oteldemo/demo.pb.go\n@@ -1 +1 @@\n-a\n+b\n")
    assert change_record(proto, Scope(True, "checkout", "ok", 1, 2), allow, "checkout")["kinds"] == ("proto",)
    # proto-only: ngoài phạm vi; D2 vẫn dùng service do mô phỏng quy định (#40)
    r = change_record(proto[:1], Scope(False, None, "proto_without_stub"), allow, "payment")
    assert (r["service"], r["in_scope"], r["kinds"], r["n_files"], r["n_lines"]) == ("payment", False, ("proto",), 1, 2)
    # review D2-7 #3: luật quy sai service thì không tính vào tập T (#40)
    assert change_record(code, Scope(True, "cart", "ok", 1, 2), allow, "checkout")["in_scope"] is False
    comp = parse("diff --git a/compose.yaml b/compose.yaml\n@@ -1 +1 @@\n-a\n+b\n")
    assert change_record(comp, Scope(True, "cart", "ok", 1, 2), allow, "cart")["kinds"] == ("config",)


def _mutants(n_per_class=6):
    return [{"mutant_id": f"m{t}{f}{i}", "type": t, "has_fail": f}
            for t in ["M0", "M1"] for f in [0, 1] for i in range(n_per_class)]


def test_folds_partition_and_stratify():
    ms = _mutants()
    fs = folds(ms, seed=0)
    assert len(fs) == 5
    tests = [set(te) for _, te in fs]
    assert set().union(*tests) == {m["mutant_id"] for m in ms}
    assert sum(len(t) for t in tests) == len(ms)  # mỗi mutant đúng 1 fold test
    for tr, te in fs:
        assert not set(tr) & set(te)
        assert {(m["type"], m["has_fail"]) for m in ms if m["mutant_id"] in te} == {("M0", 0), ("M0", 1), ("M1", 0), ("M1", 1)}


def test_folds_seeded():
    ms = _mutants()
    assert folds(ms, 0) == folds(ms, 0)
    assert folds(ms, 0) != folds(ms, 1)


def test_loso_one_service_per_test_fold():
    ms = [{"mutant_id": f"{s}{i}", "service": s} for s in ["cart", "ad", "email"] for i in range(3)]
    fs = loso_folds(ms)
    assert sorted(svc for svc, _, _ in fs) == ["ad", "cart", "email"]
    for svc, tr, te in fs:
        assert {m["service"] for m in ms if m["mutant_id"] in te} == {svc}
        assert not set(tr) & set(te)


@need_pilot
def test_load_pilot():
    d = load(PILOT, BUILD, DEMO)
    st = {m["mutant_id"]: m for m in d.mutants}
    assert len(d.mutants) == 28  # 30 − 2 build_error
    assert all(m["status"] in {"ok", "unhealthy_mutated"} for m in d.mutants)
    assert len(d.tests) == 55 and all(t in d.dur for t in d.tests if t != "test_collector::test_collector_healthy")
    assert len(d.labels) == 28 * 55
    assert {(r["mutant_id"], r["test_id"]) for r in d.labels} == {(m, t) for m in st for t in d.tests}
    for m in d.mutants:
        assert m["has_fail"] == int(any(r["label"] for r in d.labels if r["mutant_id"] == m["mutant_id"]))
        assert m["record"]["service"] == m["service"]
        if m["record"]["in_scope"]:
            assert m["record"]["kinds"]
    # chi phí (#39): W = trung vị collector trên lần chạy sạch, collector không nằm trong dur
    assert d.W > 0 and "test_collector::test_collector_healthy" not in d.dur
    assert d.t_start > 0 and set(d.build_nocache) >= {m["service"] for m in d.mutants}
    # đồ thị B2 (C3): cạnh env dựng từ compose.yaml + .env của base 3.1.0; V1 đếm được 21 cạnh
    assert len(d.edges) == 21 and ("checkout", "cart") in d.edges
