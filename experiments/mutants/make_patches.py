"""D1-2/D1-3: chọn mẫu N site theo seed (sample.py, tỉ lệ #29) rồi sinh patch cho từng mutant trên bản clone sạch.

M1 sửa pb/demo.proto của service có stub sinh sẵn (regen = go/python/ts/cpp) thì sinh lại stub cho đúng service đó
bằng script của demo (`docker-gen-proto.sh <lang> <svc>`), và đưa stub mới vào patch.

Chạy: python make_patches.py DEMO_CLONE SITES_CSV OUT_DIR N SEED → OUT_DIR/patches/*.patch + OUT_DIR/mutants.csv
Tập H (plan 2026-11-21 bước 1): python make_patches.py heldout DEMO_CLONE SITES_CSV D1_MUTANTS_CSV OUT_DIR SEED
"""
import csv
import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path

from sample import sample

PROPS = {"M0": 0.2, "M1": 0.3, "M2": 0.3, "M3": 0.2}  # decisions #29
REGEN_FN = {"go": "go", "python": "python", "ts": "ts", "cpp": "cpp"}


def docker_regen(demo, lang, service):
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}  # Git Bash đổi /build/... thành đường dẫn Windows
    r = subprocess.run(["sh", "./docker-gen-proto.sh", REGEN_FN[lang], service], cwd=demo, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=1800)
    assert r.returncode == 0, f"regen {lang} {service} lỗi:\n{r.stderr[-2000:]}"


def git(demo, *a):
    return subprocess.run(["git", *a], cwd=demo, check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def make_patch(demo, site, regen=docker_regen):
    demo = Path(demo)
    p = demo / site["file"]
    with open(p, encoding="utf-8", newline="") as fh:  # giữ nguyên CRLF/LF của file
        text = fh.read()
    if site["all"] == "True":
        assert site["old"] in text, f"{site['file']}: không thấy {site['old']}"
        text = text.replace(site["old"], site["new"])
    else:
        lines = text.splitlines(keepends=True)
        i = int(site["line"]) - 1
        body = lines[i].rstrip("\r\n")
        assert body == site["old"], f"{site['file']}:{site['line']} không khớp site (bản clone lệch?)"
        lines[i] = site["new"] + lines[i][len(body):]
        text = "".join(lines)
    try:
        p.write_text(text, encoding="utf-8", newline="")
        if site.get("regen"):
            regen(demo, site["regen"], site["service"])
        git(demo, "add", "-A")
        patch = git(demo, "diff", "--cached", "--binary")
        stat = git(demo, "diff", "--cached", "--numstat").splitlines()
    finally:
        git(demo, "reset", "-q", "--hard")
        git(demo, "clean", "-qfd")
    return {"patch": patch, "files_changed": len(stat),
            "lines_changed": sum(int(a) + int(b) for a, b, _ in (s.split("\t") for s in stat) if a != "-"),
            "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest()}


def main(demo, sites_csv, out, n, seed):
    n, seed = int(n), int(seed)
    assert git(demo, "status", "--porcelain") == "", "bản clone phải sạch"
    sites = list(csv.DictReader(open(sites_csv, encoding="utf-8")))
    chosen = sample(sites, n, PROPS, seed)
    random.Random(seed).shuffle(chosen)  # thứ tự chạy xáo theo seed (D1 §2)
    _write(demo, chosen, out, seed, "s")


def _write(demo, chosen, out, seed, prefix):
    out = Path(out)
    (out / "patches").mkdir(parents=True, exist_ok=True)
    cols = ["mutant_id", "seed", "order", "type", "service", "language", "site_id", "build",
            "files_changed", "lines_changed", "patch_sha256"]
    with (out / "mutants.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k, s in enumerate(chosen):
            mid = f"{prefix}{seed}-{k:04d}"
            r = make_patch(demo, s)
            (out / "patches" / f"{mid}.patch").write_text(r["patch"], encoding="utf-8", newline="")
            w.writerow({"mutant_id": mid, "seed": seed, "order": k, "type": s["type"], "service": s["service"],
                        "language": s["language"], "site_id": s["site_id"], "build": int(s["file"] != "compose.yaml"),
                        **{c: r[c] for c in ["files_changed", "lines_changed", "patch_sha256"]}})
            f.flush()
            print(mid, s["site_id"], r["files_changed"], flush=True)


def heldout(sites, used, seed=0):
    """Plan 2026-11-21 bước 1: site D1 chưa dùng; xáo theo seed, site khác M0 đứng trước (chạy trước ở bước 4b)."""
    out = sorted((s for s in sites if s["site_id"] not in used), key=lambda s: s["site_id"])
    random.Random(seed).shuffle(out)
    return sorted(out, key=lambda s: s["type"] == "M0")


def main_heldout(demo, sites_csv, d1_mutants_csv, out, seed=0):
    assert git(demo, "status", "--porcelain") == "", "bản clone phải sạch"
    used = {m["site_id"] for m in csv.DictReader(open(d1_mutants_csv, encoding="utf-8"))}
    _write(demo, heldout(list(csv.DictReader(open(sites_csv, encoding="utf-8"))), used, int(seed)), out, int(seed), "h")


if __name__ == "__main__":
    if sys.argv[1] == "heldout":
        main_heldout(*sys.argv[2:7])
    else:
        main(*sys.argv[1:6])
