"""D2-1: nạp bộ dữ liệu ts02 và chia fold (tiền đăng ký C1, C4; chi phí theo C0 sửa bởi #39).

Dùng: d = load(RUN_DIR, BUILD_NOCACHE_DIR, DEMO_CLONE)
"""
import csv
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from allowlist import build
from compose_owner import owner_by_line
from diff import parse
from record import change_record
from scope import scope

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graph"))
import yaml  # noqa: E402
from extract import extract_edges, k8s_edges  # noqa: E402

ALL = frozenset({"*"})  # test chạm mọi service (collector, C1)
COLLECTOR = "test_collector::test_collector_healthy"
UNITS = {"ok", "unhealthy_mutated"}  # C5: bỏ infra_error, build_error


def meta_of(test_id):
    module, _, rest = test_id.partition("::")
    param = rest[rest.find("[") + 1:-1] if "[" in rest else ""
    if module == "test_collector":
        return {"services": ALL, "kind": "collector", "signal": "none"}
    if module == "test_traces_edges":
        return {"services": set(param.split("->")), "kind": "edge", "signal": "traces"}
    return {"services": {param}, "kind": "service", "signal": module.removeprefix("test_")}


def touches(meta, service):
    return meta["services"] is ALL or service in meta["services"]


@dataclass
class Data:
    mutants: list  # dict: mutant_id, type, service, status, has_fail, record
    tests: list
    labels: list  # dict: mutant_id, test_id, label
    dur: dict  # test → trung vị thời lượng trên lần chạy sạch (không gồm collector)
    W: float  # #39: trung vị thời lượng collector = warmup
    t_start: float
    build_nocache: dict
    edges: set = None  # C3: cạnh env (s gọi t) của base
    test_services: dict = None  # plan 2026-11-21: test → service, cho repo mà ID test không ghi service


def _read(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def base_edges(demo, tag):
    """C3 + #40: đồ thị dựng từ compose.yaml và .env của base, không từ bản đã đột biến."""
    with tempfile.TemporaryDirectory() as tmp:
        for f in ["compose.yaml", ".env"]:
            Path(tmp, f).write_text(_show(demo, tag, f), encoding="utf-8")
        return extract_edges(Path(tmp, "compose.yaml"), Path(tmp, ".env"))["env"]


def k8s_base_edges(repo, ref, d):
    """Plan 2026-11-21: đồ thị dựng từ manifest k8s trong thư mục d của base (repo khác OTel Demo)."""
    names = subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", f"{ref}:{d}"], capture_output=True,
                           text=True, encoding="utf-8", check=True).stdout.split()
    return k8s_edges([x for n in names if n.endswith((".yaml", ".yml"))
                      for x in yaml.safe_load_all(_show(repo, ref, f"{d}/{n}"))])


def _show(demo, ref, path):
    return subprocess.run(["git", "-C", str(demo), "show", f"{ref}:{path}"], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def load(run_dir, build_dir, demo, tag="3.1.0"):
    run_dir = Path(run_dir)
    status = {r["mutant_id"]: r["status"] for r in _read(run_dir / "mutant_status.csv")}
    allow = build(run_dir / "mutants")
    own = owner_by_line(_show(demo, tag, "compose.yaml"))
    labels = [dict(r, label=int(r["label"])) for r in _read(run_dir / "labels.csv") if status.get(r["mutant_id"]) in UNITS]
    fail = defaultdict(int)
    for r in labels:
        fail[r["mutant_id"]] |= r["label"]
    mutants = []
    for m in _read(run_dir / "mutants" / "mutants.csv"):
        if status.get(m["mutant_id"]) not in UNITS:
            continue
        ch = parse((run_dir / "mutants" / "patches" / f"{m['mutant_id']}.patch").read_text(encoding="utf-8"))
        mutants.append({"mutant_id": m["mutant_id"], "type": m["type"], "service": m["service"],
                        "status": status[m["mutant_id"]], "has_fail": fail[m["mutant_id"]],
                        "record": change_record(ch, scope(ch, allow, own), allow, m["service"])})
    tests = sorted({r["test_id"] for r in labels})

    by_test, starts = defaultdict(list), {}
    for r in _read(run_dir / "table.csv"):
        if r["mutant_id"] == "clean" and r["status"] == "ok" and r["test_id"] in tests:
            by_test[r["test_id"]].append(float(r["duration_s"]))
            starts[r["run_id"]] = float(r["t_start"])
    dur = {t: statistics.median(v) for t, v in by_test.items() if t != COLLECTOR}
    builds = defaultdict(list)
    for r in _read(Path(build_dir) / "table.csv"):
        if r["returncode"] == "0":
            builds[r["service"]].append(float(r["seconds"]))
    return Data(mutants, tests, labels, dur, statistics.median(by_test[COLLECTOR]), statistics.median(starts.values()),
                {s: statistics.median(v) for s, v in builds.items()}, base_edges(demo, tag))


def folds(mutants, seed, k=5):
    """C4 chia chính: K = 5 theo mutant (mỗi mutant là một nhóm), phân tầng theo (loại, có fail).

    G2a (#88): mutant có trường "group" thì chia theo nhóm: đổi nhãn nhóm bằng hoán vị theo seed trên các khoá đã sắp xếp,
    rồi StratifiedGroupKFold(shuffle=False), vì shuffle=True của scikit-learn 1.7.1 không phân tầng."""
    ids = [m["mutant_id"] for m in mutants]
    strata = [f"{m['type']}|{m['has_fail']}" for m in mutants]
    if "group" not in mutants[0]:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [([ids[i] for i in tr], [ids[i] for i in te]) for tr, te in skf.split(ids, strata)]
    keys = [m["group"] for m in mutants]
    uniq = sorted(set(keys))
    lab = dict(zip(uniq, np.random.RandomState(seed).permutation(len(uniq))))
    out = []
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=False).split(ids, strata, [lab[x] for x in keys]):
        assert not {keys[i] for i in tr} & {keys[i] for i in te}  # không nhóm nào ở cả train và test
        out.append(([ids[i] for i in tr], [ids[i] for i in te]))
    return out


def loso_folds(mutants):
    """C4 sensitivity: leave-one-service-out."""
    svcs = sorted({m["service"] for m in mutants})
    return [(s, [m["mutant_id"] for m in mutants if m["service"] != s], [m["mutant_id"] for m in mutants if m["service"] == s])
            for s in svcs]
