"""#48: classification report mức dòng (mutant, test) cho một run D2-8 — mô tả, bổ sung sau khi xem seed 0.

- Dự đoán dương = test được chọn chạy (có e2e và test nằm trong tập chọn); nhãn = test fail (C2).
- Quyết định dựng lại từ `config` + `tau` trong table.csv, không tune lại; tiết kiệm/recall phải khớp table.csv.
- Gộp các fold test của seed, mỗi phương pháp × điểm vận hành một dòng.

Chạy: python cls_report.py RUN_DIR BUILD_NOCACHE_DIR DEMO EVAL_DIR OUT_DIR [DECISION]
"""
import csv
import json
import math
import sys
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from baselines import b0, b1, b1h, b2
from data import folds, load, loso_folds
from evaluate import metrics
from run import _by_id, _decide, _scores, _write

RULES = {"b0": b0, "b1": b1, "b1h": b1h}


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def rebuild(d, rows, seed, loso=False):
    """rows: table.csv đã đọc (chuỗi) → {(method, op): {mutant_id: Decision}}."""
    splits = [(tr, te) for _, tr, te in loso_folds(d.mutants)] if loso else folds(d.mutants, seed, 5)
    ms, cache, out = _by_id(d), {}, {}
    for r in rows:
        method, train, test = r["method"], *splits[int(r["fold"])]
        cfg = json.loads(r["config"])
        if method in RULES:
            dec = {i: RULES[method](ms[i], d) for i in test}
        elif method == "b2":
            dec = {i: b2(ms[i], d, d.edges, *cfg) for i in test}
        else:
            key = (r["fold"], method)
            if key not in cache:
                cache[key] = _scores(method, cfg, d, train, test, seed)
            dec = _decide(d, cache[key], test, float(r["tau"]))
        m = metrics(dec, d)
        for k in ("savings", "recall_mutant", "recall_test"):
            assert _same(m[k], float(r[k])), (method, r["op"], r["fold"], k, m[k], r[k])
        out.setdefault((method, r["op"]), {}).update(dec)
    return out


def report(d, decs):
    lab = {(r["mutant_id"], r["test_id"]): int(r["label"]) for r in d.labels}
    keys = sorted(lab)
    out = []
    for (method, op), dec in decs.items():
        y = [lab[k] for k in keys]
        p = [int(dec[mid].e2e and t in dec[mid].tests) for mid, t in keys]
        rep = classification_report(y, p, labels=[0, 1], output_dict=True, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        row = {"method": method, "op": op, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        for c, name in (("0", "0"), ("1", "1"), ("macro avg", "macro"), ("weighted avg", "weighted")):
            row |= {f"{s}_{name}": rep[c][s] for s in ("precision", "recall", "f1-score")}
        row |= {"support_0": rep["0"]["support"], "support_1": rep["1"]["support"], "accuracy": rep["accuracy"]}
        out.append({k.replace("f1-score", "f1"): v for k, v in row.items()})
    return out


def main(run_dir, build_dir, demo, eval_dir, out, decision="#48"):
    eval_dir, out = Path(eval_dir), Path(out)
    params = json.loads((eval_dir / "params.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(eval_dir / "table.csv", encoding="utf-8")))
    d = load(run_dir, build_dir, demo)
    decs = rebuild(d, rows, params["seed"], params["split"] == "loso")
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "cls_report.csv", report(d, decs))
    (out / "params.json").write_text(json.dumps({"decision": decision, "source_eval": str(eval_dir), "data": str(run_dir),
                                                 "seed": params["seed"], "split": params["split"],
                                                 "check": "savings/recall_mutant/recall_test khớp table.csv"},
                                                indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:7])
