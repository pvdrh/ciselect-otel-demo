"""#51 DoD 2: kiểm trùng dòng baseline với run D2 (R1: chỉ cho phép lệch số thực ở mức làm tròn)."""
from run_extra import LABELS, same_rows

K = ("fold", "method", "op")
A = [{"fold": "0", "method": "b1", "op": "natural", "savings": "0.1", "config": "null", "time_fit_s": "0.0"}]


def test_same_rows_tolerates_float_rounding_and_time():
    assert same_rows(A, [dict(A[0], savings="0.10000000000000002", time_fit_s="9.9")], K) == []


def test_same_rows_reports_real_differences():
    assert same_rows(A, [dict(A[0], savings="0.1001")], K)
    assert same_rows(A, [dict(A[0], config='[1, "callers"]')], K)
    assert same_rows(A, [], K)  # thiếu dòng
    assert same_rows(A, A + [dict(A[0], fold="1")], K)  # thừa dòng


def test_labels_keep_rf_provenance_and_add_b3k():
    assert LABELS["rf"] == ("plans/2026-10-07-d2-rf-oracle.md", "#51 (thăm dò, thêm sau khi đã thấy kết quả D2)")
    plan, dec = LABELS["b3k"]
    assert plan == "plans/2026-10-08-phan-tich-o4-o5.md" and dec.startswith("#57")
