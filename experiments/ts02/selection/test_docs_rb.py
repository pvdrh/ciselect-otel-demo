"""Q1 (a) của #88: commit chỉ sửa tài liệu tính theo mốc ở R_b."""
import pandas as pd
import pytest

from docs_rb import summary, with_docs

Q = {"300": 12 / 300, "104": 12 / 104}


def _net():
    rows = []
    for lop, nets in [(300, [-40.0, 20.0]), (104, [50.0, 60.0])]:
        for moc in ("R_a", "R_b"):
            for fb in ("F_tool", "F_ci"):
                for mode, v in zip(("mac_dinh", "b0p"), nets):
                    rows.append({"lop": lop, "moc": moc, "fallback": fb, "mode": mode, "c_job": "c2",
                                 "net": v - (1000 if moc == "R_a" else 0)})
    df = pd.DataFrame(rows)
    return df.assign(doc=df.net.map(lambda v: "thiệt" if v < 0 else "không xác định dấu"))


def test_with_docs_adds_q_times_b0_cost_only_at_rb():
    r = with_docs(_net(), Q, 1375.968)
    assert (r.moc == "R_b").all() and len(r) == 8
    x = r[(r.lop == 300) & (r["mode"] == "mac_dinh")].iloc[0]
    assert x.docs_term == pytest.approx(12 / 300 * 1375.968)
    assert x.net_docs == pytest.approx(-40 + 55.03872)
    assert (x.doc, x.doc_docs) == ("thiệt", "không xác định dấu")


def test_with_docs_zero_share_keeps_net():
    r = with_docs(_net(), {"300": 0.0, "104": 0.0}, 1375.968)
    assert (r.net_docs == r.net).all() and (r.doc_docs == r.doc).all()


def test_summary_counts_cells_per_class_and_mode_group():
    s = summary(with_docs(_net(), Q, 1375.968)).set_index(["lop", "nhom"])
    a = s.loc[(300, "4 mode")]
    assert (a.n, a.thiet, a.thiet_docs, a.net_min) == (1, 1, 0, -40.0)
    assert s.loc[(104, "b0p")].thiet == 0
