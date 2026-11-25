"""Huấn luyện thư mục mô hình cho ciselect: hai mô hình, ngưỡng, bảng B3, tham số B2 và chế độ mặc định.

- Cấu hình: chọn bằng kiểm định chéo 4 fold trên toàn bộ mutant, seed 0 (`run._tune_scored`). B2 chọn bằng `run._tune_b2`.
- Ngưỡng: lấy trên điểm out-of-fold của phép chia chính (K = 5, seed 0) với cấu hình đã chọn. Không dùng LOSO,
  không dùng điểm in-sample.
- Mô hình cuối: fit trên toàn bộ mutant. Không dùng mô hình này để báo hiệu năng, số hiệu năng lấy từ phần đánh giá.
Đầu vào: thư mục run sinh bộ mutant, run đo build không cache, clone demo, thư mục tổng hợp kết quả đánh giá.
Đầu ra: OUT_DIR gồm model_lgbm.txt, lr.pkl, threshold.json, default_mode.json và các tệp JSON khác mà cli.py đọc.

Chạy: python tool/train_release.py RUN_DIR BUILD_NOCACHE_DIR DEMO SUMMARY_DIR OUT_DIR
"""
import argparse
import json
import pickle
import platform
from pathlib import Path

import numpy as np

import _paths  # noqa: F401,E402
from allowlist import build  # noqa: E402
from baselines import b3_table  # noqa: E402
from data import Data, folds, load  # noqa: E402
from default_mode import apply_rule41  # noqa: E402
from evaluate import metrics  # noqa: E402
from features import FEATURES, features  # noqa: E402
from models import _model  # noqa: E402
from run import GRIDS, _best, _decide, _scores, _tune_b2, _tune_scored  # noqa: E402
from threshold import pick_threshold  # noqa: E402

SCORED = ("b3", "lr", "lgbm")  # lgbm_nofr chỉ là phân tích phụ ngoài họ kiểm định Holm, không đưa vào thư mục mô hình


def fit_release(d, seed=0, grids=GRIDS, targets=None):
    targets = targets or {"t95": 0.95, "t99": 0.99}
    ids, lab = [m["mutant_id"] for m in d.mutants], {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
    rel = {"config": {"b2": _tune_b2(d, ids, grids["b2"])}, "threshold": {}, "models": {}}
    for m in SCORED:
        cfg, _ = _tune_scored(m, d, ids, seed, grids[m])
        oof = {}
        for tr, te in folds(d.mutants, seed, 5):
            oof |= _scores(m, cfg, d, tr, te, seed)
        best = _best(d, oof, ids)
        rel["config"][m] = cfg
        rel["threshold"][m] = {}
        for k, r in targets.items():
            tau = pick_threshold(best, r)
            rel["threshold"][m][k] = {"target": r, "tau": tau,
                                      "oof_recall": metrics(_decide(d, oof, ids, tau), d)["recall_mutant"]}
        if m != "b3":
            X, keys = features(d, ids, ids)
            rel["models"][m] = _model(m, cfg, seed).fit(X, np.array([lab[k] for k in keys]))
    rel["b3"] = b3_table(ids, d.labels, {m["mutant_id"]: m["service"] for m in d.mutants}, rel["config"]["b3"])
    return rel


def write_release(out, d, rel, allow, b2cfg, rule):
    import lightgbm
    import sklearn
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w = lambda name, obj: (out / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")  # noqa: E731
    rel["models"]["lgbm"].booster_.save_model(str(out / "model_lgbm.txt"))
    (out / "lr.pkl").write_bytes(pickle.dumps(rel["models"]["lr"]))
    t = rel["b3"]
    w("b3.json", {"a": rel["config"]["b3"], "n": t["n"], "fails": [[s, tt, v] for (s, tt), v in t["fails"].items()]})
    w("threshold.json", rel["threshold"])
    w("train.json", {"mutants": [{"mutant_id": m["mutant_id"], "service": m["service"], "record": m["record"]} for m in d.mutants],
                     "labels": [[r["mutant_id"], r["test_id"], r["label"]] for r in d.labels],
                     "tests": d.tests, "dur": d.dur, "W": d.W, "t_start": d.t_start})
    w("allowlist.json", allow)
    w("catalog.json", {"tests": d.tests, "services": list(d.build_nocache)})
    w("b2.json", b2cfg)
    w("features.json", {"features": FEATURES, "lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__,
                        "numpy": np.__version__})
    w("default_mode.json", rule)
    w("params.json", {"plan": "plans/2026-10-05-d3-t5-mo-hinh-giao-di.md", "seed": 0, "config": rel["config"],
                      "threshold_source": "OOF C4 K=5 seed 0", "n_mutants": len(d.mutants),
                      "note": "không dùng để báo hiệu năng; hiệu năng lấy từ D2"})
    (out / "env.txt").write_text(f"Python {platform.python_version()}\nlightgbm {lightgbm.__version__}\n"
                                 f"scikit-learn {sklearn.__version__}\nnumpy {np.__version__}\n{platform.platform()}\n",
                                 encoding="utf-8")


def load_models(md):
    from lightgbm import Booster
    return Booster(model_file=str(Path(md) / "model_lgbm.txt")), pickle.loads((Path(md) / "lr.pkl").read_bytes())


def train_data(md):
    """train.json → Data (đầu vào features/b3); record.kinds về tuple như change_record."""
    t = json.loads((Path(md) / "train.json").read_text(encoding="utf-8"))
    ms = [dict(m, record=dict(m["record"], kinds=tuple(m["record"]["kinds"]))) for m in t["mutants"]]
    labels = [{"mutant_id": a, "test_id": b, "label": c} for a, b, c in t["labels"]]
    return Data(ms, t["tests"], labels, t["dur"], t["W"], t["t_start"], {})


def main(run_dir, build_dir, demo, summary, out):
    import pandas as pd
    d = load(run_dir, build_dir, demo)
    s = Path(summary)
    rule = apply_rule41(pd.read_csv(s / "table.csv"), pd.read_csv(s / "main_tests.csv"))
    rule["summary"] = str(s)
    targets = {"t95": 0.95, "t99": 0.99} | ({"r_star": rule["recall_target"]} if rule["recall_target"] else {})
    rel = fit_release(d, 0, GRIDS, targets)
    k, h = rel["config"]["b2"]
    write_release(out, d, rel, build(Path(run_dir) / "mutants"), {"k": k, "direction": h}, rule)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="thư mục run sinh bộ mutant (có mutants/ và mutant_status.csv)")
    ap.add_argument("build_dir", help="thư mục run đo thời gian build không cache")
    ap.add_argument("demo", help="clone OpenTelemetry Demo có tag 3.1.0")
    ap.add_argument("summary", help="thư mục tổng hợp kết quả đánh giá, dùng để chọn chế độ mặc định")
    ap.add_argument("out", help="thư mục ghi mô hình")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(**vars(parse()))
