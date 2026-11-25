"""Danh sách file cho phép theo service, dựng từ patch D1-7 (#40); đánh dấu stub sinh tự động (#42)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

from diff import parse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mutants"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graph"))
from sites import SERVICES  # noqa: E402

PROTO = "pb/demo.proto"
SPECIAL = {PROTO, "compose.yaml"}


def build(mutants_dir):
    mutants_dir = Path(mutants_dir)
    handlers = {f for _, (_, files, _, _) in SERVICES.items() for f in files}
    svc, without_proto = {}, defaultdict(bool)
    for m in csv.DictReader(open(mutants_dir / "mutants.csv", encoding="utf-8")):
        paths = [c.path for c in parse((mutants_dir / "patches" / f"{m['mutant_id']}.patch").read_text(encoding="utf-8"))]
        for p in paths:
            if p in SPECIAL:
                continue
            svc[p] = m["service"]
            without_proto[p] |= PROTO not in paths
    return {"files": {p: {"service": s, "generated": not without_proto[p] and p not in handlers} for p, s in svc.items()}}
