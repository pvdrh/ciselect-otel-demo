"""Q1 (a) của #88: hàng độ nhạy cho cận trên hiệu quả ròng ở mốc cổng e2e của CI gốc (R_b).

Commit chỉ sửa tài liệu được tính theo mốc thay cho chênh 0 của #53: mốc build mọi service và chạy mọi test (chi phí
của B0), còn công cụ không build và không chạy e2e; mỗi commit một lần chạy (k = 1). Chi phí job select đã trừ cho mọi
commit trong net của #53, nên phần cộng thêm là q·C_B0. Hậu xử lý net_effect.csv (#53) và p3_net_b0p.csv (#86);
không p, không Holm.

Chạy: python docs_rb.py RESULTS_DIR OUT_DIR
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from extra_desc import SCOPE
from run_nhom import provenance
from summarize import _load

NET = "ts02_extra_desc/20261007-182150_seed0-4/net_effect.csv"
B0P = "ts02_p85/20261015-092354_seed0-4/p3_net_b0p.csv"
THIET = "thiệt"


def with_docs(net, q, c_b0):
    """Dòng R_b của net; q: lớp → tỉ lệ commit chỉ sửa tài liệu; c_b0: chi phí B0 (giây)."""
    r = net[net.moc == "R_b"].copy()
    r["q_docs"] = r.lop.astype(str).map(q)
    assert r.q_docs.notna().all()
    r["docs_term"] = r.q_docs * c_b0
    r["net_docs"] = r.net + r.docs_term
    r["doc_docs"] = np.where(r.net_docs < 0, THIET, "không xác định dấu")
    return r


def summary(r):
    a, b = (r[r.fallback == f].drop(columns="fallback").reset_index(drop=True) for f in ("F_ci", "F_tool"))
    assert a.equals(b)  # ở R_b không có phạt, hai cách về B0 cho cùng số
    a = a.assign(nhom=np.where(a["mode"] == "b0p", "b0p", "4 mode"))
    return a.groupby(["lop", "nhom"]).agg(
        n=("net", "size"), net_min=("net", "min"), net_max=("net", "max"), thiet=("net", lambda x: int((x < 0).sum())),
        net_docs_min=("net_docs", "min"), net_docs_max=("net_docs", "max"),
        thiet_docs=("net_docs", lambda x: int((x < 0).sum()))).reset_index()


def main(results, out):
    res, out = Path(results), Path(out)
    net = pd.concat([pd.read_csv(res / NET), pd.read_csv(res / B0P)], ignore_index=True)
    sc = json.loads((res / SCOPE).read_text(encoding="utf-8"))
    q = {"300": sc["reason"]["docs_only"] / sc["n"], "104": sc["reason_human"]["docs_only"] / (sc["n"] - sc["bot"])}
    assert q == {"300": 12 / 300, "104": 12 / 104}
    per, src = _load(str(res / "ts02_eval/*_seed[0-4]"), "per_mutant.csv")
    c = per[per.method == "b0"].cost
    assert len(src) == 5 and np.allclose(c, c.iloc[0], rtol=0, atol=1e-9)  # B0 build mọi service, chạy mọi test: hằng
    r = with_docs(net, q, float(c.iloc[0]))
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "net_docs.csv", index=False)
    summary(r).to_csv(out / "summary.csv", index=False)
    (out / "params.json").write_text(json.dumps({
        "plan": "docs/2026-10-15-phan-bien-plan-n/tong-hop.md mục 4.1, Q1 (a)", "decision": "#88",
        "prereg": "docs/preregistration.md C6, Bổ sung sau khoá (#88)", "net_effect": str(res / NET),
        "net_b0p": str(res / B0P), "scope": str(res / SCOPE), "eval": src, "q_docs": q, "c_b0": float(c.iloc[0]), "k": 1,
        "rule": "net_docs = net + q·C_B0, chỉ ở R_b; < 0 ghi 'thiệt', không âm ghi 'không xác định dấu'", **provenance(out)},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\npandas {pd.__version__}\nnumpy {np.__version__}\n"
                                 f"{platform.platform()}\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:3])
