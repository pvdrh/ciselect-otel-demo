import csv

import pytest

from cls_report import rebuild, report
from run import _write, run_seed
from test_run import TINY, _data


def _roundtrip(tmp_path, rows):
    """table.csv đọc lại từ đĩa: mọi giá trị là chuỗi, như khi chạy thật."""
    _write(tmp_path / "table.csv", rows)
    return list(csv.DictReader(open(tmp_path / "table.csv", encoding="utf-8")))


def test_rebuild_reproduces_table(tmp_path):
    d = _data()
    rows, _ = run_seed(d, seed=0, grids=TINY, k=5)
    decs = rebuild(d, _roundtrip(tmp_path, rows), seed=0)  # tự raise nếu tiết kiệm/recall lệch table
    assert len(decs) == len({(r["method"], r["op"]) for r in rows})
    assert all(len(v) == len(d.mutants) for v in decs.values())


def test_rebuild_rejects_tampered_tau(tmp_path):
    d = _data()
    rows, _ = run_seed(d, seed=0, grids=TINY, k=5)
    rows = _roundtrip(tmp_path, rows)
    r = next(r for r in rows if r["method"] == "lgbm" and r["op"] == "t95")
    r["savings"] = str(float(r["savings"]) + 0.01)
    with pytest.raises(AssertionError):
        rebuild(d, rows, seed=0)


def test_report_class1_recall_is_pooled_recall_test(tmp_path):
    d = _data()
    rows, _ = run_seed(d, seed=0, grids=TINY, k=5)
    out = report(d, rebuild(d, _roundtrip(tmp_path, rows), seed=0))
    lab = {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
    n_pos = sum(lab.values())
    for r in out:
        assert r["tp"] + r["fn"] == n_pos and r["tp"] + r["fp"] + r["tn"] + r["fn"] == len(lab)
        if r["method"] == "b0":  # B0 chọn mọi test: recall lớp 1 = 1
            assert r["recall_1"] == 1.0 and r["fn"] == 0
