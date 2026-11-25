from baselines import b0, b1, b1h, b2, b3_scores, b3_table, b3k_scores, reach
from comments import comment_only
from data import COLLECTOR, Data
from diff import parse

# Đồ thị 4+1 service tự dựng: s→t nghĩa là s gọi t.
EDGES = {("fp", "fe"), ("fe", "cart"), ("fe", "ck"), ("ck", "cart"), ("ck", "pay")}
T = {f"test_traces::test_service_has_traces[{s}]": s for s in ["fp", "fe", "cart", "ck", "pay"]}
EDGE = "test_traces_edges::test_service_edge_exists[ck->pay]"
TESTS = [COLLECTOR, *T, EDGE]
D = Data([], TESTS, [], {}, 0.0, 0.0, {s: 1.0 for s in ["fp", "fe", "cart", "ck", "pay"]})


def _tests_of(*svcs):
    return {t for t, s in T.items() if s in svcs}


def test_reach_callers_and_undirected():
    assert reach(EDGES, "cart", 1, "callers") == {"fe", "ck"}
    assert reach(EDGES, "cart", 2, "callers") == {"fe", "ck", "fp"}
    assert reach(EDGES, "pay", 1, "callers") == {"ck"}
    assert reach(EDGES, "pay", 2, "callers") == {"ck", "fe"}
    assert reach(EDGES, "pay", 2, "undirected") == {"ck", "fe", "cart"}
    assert reach(EDGES, "pay", None, "callers") == {"ck", "fe", "fp"}  # k = ∞


def test_b0_builds_and_runs_everything():
    dec = b0({"service": "pay"}, D)
    assert dec.build == set(D.build_nocache) and dec.e2e and dec.tests == set(TESTS)


def test_b1_tests_touching_s_plus_collector():
    dec = b1({"service": "pay"}, D)
    assert (dec.build, dec.e2e) == ({"pay"}, True)
    assert dec.tests == _tests_of("pay") | {COLLECTOR, EDGE}


def test_b1h_skips_e2e_only_for_comment_only():
    assert b1h({"service": "pay", "record": {"comment_only": True}}, D).e2e is False
    assert b1h({"service": "pay", "record": {"comment_only": False}}, D).tests == b1({"service": "pay"}, D).tests


def test_b2_closure():
    dec = b2({"service": "pay"}, D, EDGES, 1, "callers")
    assert dec.tests == _tests_of("pay", "ck") | {COLLECTOR, EDGE}


def p(path, minus, plus):
    return parse(f"diff --git a/{path} b/{path}\n@@ -1 +1 @@\n-{minus}\n+{plus}\n")


def test_comment_only():
    assert comment_only(p("a.go", "	// Parse x", "	// Parse x (note)"))
    assert comment_only(p("a.py", "x = 1  # c", "x = 1  # c2"))
    assert comment_only(p("a.go", "x := 1", "x  :=  1"))  # chỉ khoảng trắng
    assert not comment_only(p("a.go", 'u := "http://a"', 'u := "http://b"'))  # // trong string không phải comment
    assert not comment_only(p("a.py", "s = '#a'", "s = '#b'"))
    assert not comment_only(p("a.go", 'u := "https://n.example/a"', 'u := "https://n.example/b"'))  # string có chữ n
    assert not comment_only(p("a.js", r'x = "a\"//" + 1', r'x = "a\"//" + 2'))  # dấu nháy được escape trong string
    assert not comment_only(p("a.go", "x := 1", "x := 2"))
    assert not comment_only(p("a.go", "	 * old", "	 * new"))  # dòng giữa block comment: không chắc → chạy e2e
    assert not comment_only(p("a.xyz", "a", "a"))  # ngôn ngữ lạ: không kết luận
    assert not comment_only([])


def test_comment_only_rejects_comment_syntax_that_changes_behaviour():
    """#54: các trường hợp trước đây trả True sai (review plan #53)."""
    raw = lambda path, body: parse(f"diff --git a/{path} b/{path}\n{body}")  # noqa: E731
    assert not comment_only(raw("a.go", "@@ -2,0 +3 @@\n+/*\n@@ -5,0 +7 @@\n+*/\n"))  # bọc khối code bằng /* */
    assert not comment_only(raw("a.go", "@@ -3 +2,0 @@\n-/*\n@@ -7 +5,0 @@\n-*/\n"))  # gỡ /* */: bật lại code
    assert not comment_only(raw("a.js", "@@ -1 +1,2 @@\n-  return {\n+  return\n+  {\n"))  # tách dòng: ASI
    assert not comment_only(p("a.go", "//go:build linux", "//go:build windows"))
    assert not comment_only(p("run.sh", "#!/bin/sh", "#!/bin/bash"))
    assert not comment_only(p("a.php", "#[Route('/a')]", "#[Route('/b')]"))  # attribute PHP 8, không phải comment
    assert not comment_only(p("a.rb", "# frozen_string_literal: true", "# frozen_string_literal: false"))
    assert not comment_only(p("a.py", "        return x", "    return x"))  # thụt lề đổi luồng chạy
    assert not comment_only(p("c.yaml", "  key: 1", "key: 1"))


def test_comment_only_still_accepts_plain_comment_edits():
    """#54: sửa lỗi không được làm mất các trường hợp đúng (gồm kiểu M0 của D1)."""
    assert comment_only(p("a.py", "    # note", "    # note (note)"))
    assert comment_only(p("c.yaml", "# a", "# b"))
    assert comment_only(p("a.go", "x := 1 /* a */", "x := 1 /* b */"))
    assert comment_only(p("a.php", "# a", "# b"))


def test_comment_only_keeps_other_directives():
    """#54: directive cùng loại với //go:build và #! (trình biên dịch/thông dịch đọc chúng)."""
    assert not comment_only(p("a.go", "// +build linux", "// +build windows"))
    assert not comment_only(p("a.ts", "// @ts-ignore", "// @ts-nocheck"))
    assert not comment_only(p("a.ts", '/// <reference types="a" />', '/// <reference types="b" />'))
    assert not comment_only(p("a.py", "# -*- coding: utf-8 -*-", "# -*- coding: latin-1 -*-"))
    assert comment_only(p("a.go", "// Copyright The OpenTelemetry Authors", "// Copyright The OpenTelemetry Authors (note)"))


def test_b3_uses_train_only_with_laplace():
    labels = [{"mutant_id": "m1", "test_id": "t1", "label": 1}, {"mutant_id": "m1", "test_id": "t2", "label": 0},
              {"mutant_id": "m2", "test_id": "t1", "label": 0}, {"mutant_id": "m2", "test_id": "t2", "label": 0},
              {"mutant_id": "m3", "test_id": "t1", "label": 1}]  # m3 thuộc fold test
    svc = {"m1": "a", "m2": "a", "m3": "a"}
    tab = b3_table(["m1", "m2"], labels, svc, a=1)
    assert b3_scores(tab, "a", ["t1", "t2"], a=1) == {"t1": (1 + 1) / (2 + 2), "t2": (0 + 1) / (2 + 2)}
    assert b3_scores(tab, "unseen", ["t1"], a=1) == {"t1": 0.5}  # service chưa có trong train: prior

# #57 O5: B3k = B3 theo khoá (S, loại file), prior là tỉ lệ của S không gồm ô.
B3K_LAB = [{"mutant_id": m, "test_id": "t1", "label": y} for m, y in
           [("c1", 1), ("c2", 1), ("c3", 1), ("c4", 1), ("g1", 0), ("x1", 1), ("x2", 0)]]
B3K_SVC = {"c1": "a", "c2": "a", "c3": "a", "c4": "a", "g1": "a", "x1": "b", "x2": "b"}
B3K_KEY = {m: (s, ("config",) if m == "g1" else ("code",)) for m, s in B3K_SVC.items()}


def _b3k(train, s, k, a=1.0):
    ts, tk = b3_table(train, B3K_LAB, B3K_SVC, a), b3_table(train, B3K_LAB, B3K_KEY, a)
    return b3k_scores(ts, tk, s, k, ["t1"], a)["t1"]


def test_b3k_empty_cell_equals_b3_of_service():
    train = ["c1", "c2", "c3", "c4"]  # ô (a, config) rỗng ở train
    assert _b3k(train, "a", ("a", ("config",))) == b3_scores(b3_table(train, B3K_LAB, B3K_SVC, 1), "a", ["t1"], 1)["t1"]


def test_b3k_single_kind_service_equals_b3():
    train = ["x1", "x2"]  # service b chỉ có một loại file
    for a in (0.5, 1.0, 2.0):
        assert _b3k(train, "b", ("b", ("code",)), a) == b3_scores(b3_table(train, B3K_LAB, B3K_SVC, a), "b", ["t1"], a)["t1"]


def test_b3k_small_cell_shrinks_to_rate_of_other_kinds():
    train = ["c1", "c2", "c3", "c4", "g1"]
    p_other = (4 + 1) / (4 + 2)  # ô (a, code): 4/4 fail
    assert abs(_b3k(train, "a", ("a", ("config",))) - (0 + 2 * p_other) / (1 + 2)) < 1e-12  # ô (a, config): 0/1
    assert abs(_b3k(train, "a", ("a", ("code",))) - (4 + 2 * (0 + 1) / (1 + 2)) / (4 + 2)) < 1e-12


def test_b3k_unseen_service_is_prior():
    assert _b3k(["c1"], "z", ("z", ("code",))) == 0.5
