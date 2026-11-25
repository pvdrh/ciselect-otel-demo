"""ciselect: đọc diff giữa hai commit, in JSON gồm service cần build lại, có chạy e2e không và test e2e nào.

Công cụ không có thuật toán riêng. Luật phạm vi, bản ghi thay đổi, các chiến lược, đặc trưng và quyết định đều import
từ experiments/ts02/selection, tức đúng code dùng trong phần đánh giá.
Đầu vào: repo git, hai commit, chế độ, thư mục mô hình.
Đầu ra: một dòng JSON ra stdout. Lệnh git lỗi thì in lỗi ra stderr và thoát mã 2.
Chạy: python tool/ciselect/cli.py --repo P --base R1 --head R2 --mode {b1,b1h,b2,b3,lr,lgbm,b0p,default} --model-dir D
      [--op {t95,t99,r_star}] [--max-lines N]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # thư mục tool/
import _paths  # noqa: F401,E402
from baselines import b0, b1, b1h, b2, b3_scores  # noqa: E402
from compose_owner import owner_by_line  # noqa: E402
from data import Data, _show, base_edges, k8s_base_edges  # noqa: E402
from diff import changes  # noqa: E402
from evaluate import Decision, b0prime  # noqa: E402
from features import features  # noqa: E402
from record import change_record  # noqa: E402
from run import _decide  # noqa: E402
from scope import scope  # noqa: E402

RULE_MODES, SCORED = ("b1", "b1h", "b2", "b0p"), ("b3", "lr", "lgbm")
MODES = RULE_MODES + SCORED + ("default",)
NEW = "__change__"


def load_release(md):
    """Nạp thư mục mô hình do train_release.py ghi: tập train tối giản, hai mô hình, bảng B3, ngưỡng."""
    from train_release import load_models, train_data
    md = Path(md)
    lgbm, lr = load_models(md)
    b3 = json.loads((md / "b3.json").read_text(encoding="utf-8"))
    return {"train": train_data(md), "lgbm": lgbm, "lr": lr,
            "b3": {"a": b3["a"], "n": b3["n"], "fails": {(s, t): v for s, t, v in b3["fails"]}},
            "threshold": json.loads((md / "threshold.json").read_text(encoding="utf-8"))}


def _with_new(rel, service, record, edges):
    t = rel["train"]
    new = {"mutant_id": NEW, "service": service, "record": record}
    return Data(t.mutants + [new], t.tests, t.labels, t.dur, t.W, t.t_start, {}, edges)


def rows(rel, service, record, edges):
    """Ma trận đặc trưng của thay đổi mới: gọi nguyên features() của phần đánh giá, train là cả tập train trong thư mục mô hình."""
    d = _with_new(rel, service, record, edges)
    return features(d, [m["mutant_id"] for m in rel["train"].mutants], [NEW])[0]


def scored_decision(rel, mode, op, service, record, edges):
    d = _with_new(rel, service, record, edges)
    if mode == "b3":
        sc = {(NEW, t): v for t, v in b3_scores(rel["b3"], service, d.tests, rel["b3"]["a"]).items()}
    else:
        X = rows(rel, service, record, edges)
        p = rel["lgbm"].predict(X) if mode == "lgbm" else rel["lr"].predict_proba(X)[:, 1]
        sc = {(NEW, t): float(v) for t, v in zip(d.tests, p)}
    if op not in rel["threshold"][mode]:
        sys.exit(f"ciselect: threshold.json không có điểm vận hành {op} cho mode {mode}")
    return _decide(d, sc, [NEW], rel["threshold"][mode][op]["tau"])[NEW]


def decide(changes, mode, assets, owner, edges, max_lines=None, op="t95"):
    sc = scope(changes, assets["allow"], owner, max_lines)
    cat = assets["catalog"]
    d = Data([], cat["tests"], [], {}, 0.0, 0.0, dict.fromkeys(cat["services"]), edges, cat.get("test_services"))
    if sc.reason == "docs_only":
        dec = Decision(set(), False, set())
    elif not sc.in_scope:
        dec = b0(None, d)
    else:
        rec = change_record(changes, sc, assets["allow"], sc.service)
        m = {"service": sc.service, "record": rec}
        if mode in SCORED:
            dec = scored_decision(assets["release"](), mode, op, sc.service, rec, edges)
        else:
            dec = {"b1": lambda: b1(m, d), "b1h": lambda: b1h(m, d),
                   "b0p": lambda: b0prime(m, d),  # B0′: build service bị đổi và chạy mọi test
                   "b2": lambda: b2(m, d, edges, assets["b2"]["k"], assets["b2"]["direction"])}[mode]()
    out = {"mode": mode, "in_scope": sc.in_scope, "service": sc.service, "reason": sc.reason,
           "build": sorted(dec.build), "e2e": dec.e2e, "tests": sorted(dec.tests)}
    return out | ({"op": op} if mode in SCORED else {})


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="ciselect", description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="repo git chứa cả hai commit")
    ap.add_argument("--base", required=True, help="commit base, thường là base của pull request")
    ap.add_argument("--head", required=True, help="commit cần chọn build và test")
    ap.add_argument("--mode", required=True, choices=MODES,
                    help="chiến lược chọn; default đọc default_mode.json trong --model-dir")
    ap.add_argument("--model-dir", required=True, type=Path, help="thư mục mô hình, ví dụ tool/model")
    ap.add_argument("--op", choices=("t95", "t99", "r_star"), default="t95",
                    help="ngưỡng của các chế độ dựa trên điểm (b3, lr, lgbm)")
    ap.add_argument("--max-lines", type=int,
                    help="diff đổi nhiều dòng hơn số này thì chạy toàn bộ; bỏ trống là không giới hạn")
    return ap.parse_args(argv)


def main(argv):
    a = parse(argv)
    mode, op = a.mode, a.op
    if mode == "default":  # default_mode.json do train_release.py ghi khi huấn luyện
        rule = json.loads((a.model_dir / "default_mode.json").read_text(encoding="utf-8"))
        mode, op = rule["mode"], ("r_star" if rule["step"] == 1 else "t95")
    assets = {k: json.loads((a.model_dir / f"{f}.json").read_text(encoding="utf-8"))
              for k, f in [("allow", "allowlist"), ("catalog", "catalog"), ("b2", "b2")]}
    if mode in SCORED:
        assets["release"] = lambda: load_release(a.model_dir)  # chỉ nạp mô hình khi diff trong phạm vi, diff ngoài phạm vi trả lời ngay
    g = assets["catalog"].get("graph")  # {"k8s": thư mục manifest}: repo khác OTel Demo, đồ thị đọc từ manifest Kubernetes
    if g and mode in SCORED:
        sys.exit(f"ciselect: mode {mode} cần mô hình huấn luyện trên repo này; repo chưa có nhãn chỉ chạy b0p, b1, b1h, b2")
    try:
        owner = [] if g else owner_by_line(_show(a.repo, a.base, "compose.yaml"))  # đọc compose.yaml ở commit base vì số dòng trong diff đánh theo base
        edges = None if mode not in ("b2",) + SCORED else (  # đồ thị cũng đọc ở commit base
            k8s_base_edges(a.repo, a.base, g["k8s"]) if g else base_edges(a.repo, a.base))
        out = decide(changes(a.repo, a.base, a.head), mode, assets, owner, edges, a.max_lines, op)
    except subprocess.CalledProcessError as e:
        print(f"ciselect: git error: {e.stderr or e}", file=sys.stderr)
        return 2
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
