"""Chạy workflow bằng `act` ở chế độ self-hosted trên một mutant, giữ nguyên image phát hành có sẵn trên máy.

`docker compose build` ghi đè tag phát hành `ghcr.io/open-telemetry/demo:3.1.0-<svc>`, và containerd image store
xoá luôn image cũ. Vì vậy script gắn tag dự phòng `…-cibak` trước khi chạy để giữ image. Chạy xong, script tag lại
từ bản dự phòng, gỡ tag dự phòng và kiểm image ID như trước.
Đầu vào: mã mutant, thư mục run sinh bộ mutant, thư mục mô hình, clone demo.
Đầu ra: OUT_DIR gồm act.log, artifact của các job và params.json.
Chạy: python tool/ci/run_act.py MUTANT_ID RUN_DIR MODEL_DIR DEMO OUT_DIR [MODE]
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # thư mục tool/
import _paths  # noqa: F401,E402
from make_repo import make_repo  # noqa: E402

ACT = shutil.which("act") or str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
                                 / "nektos.act_Microsoft.Winget.Source_8wekyb3d8bbwe" / "act.exe")
PREFIX = "ghcr.io/open-telemetry/demo:3.1.0-"
MIN_FREE_GB = 8


def backup(tag):
    return tag + "-cibak"


def images():
    r = subprocess.run(["docker", "images", "--no-trunc", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}"],
                       capture_output=True, text=True, check=True)
    return dict(x.split() for x in r.stdout.splitlines() if x.startswith(PREFIX) and not x.split()[0].endswith("-cibak"))


def restore(before, now, tag):
    changed = [t for t, i in before.items() if now.get(t) != i]
    for t in changed:
        tag(backup(t), t)
    return changed


def act_cmd(base, mode, art, env_file):
    # env_file rỗng: không để act nạp .env của demo vào env của job (Git Bash đổi /var/... thành đường dẫn Windows)
    return [ACT, "workflow_dispatch", "-W", ".github/workflows/ciselect.yml", "-P", "ubuntu-latest=-self-hosted",
            "--env-file", str(env_file), "--artifact-server-path", str(art),
            "--input", f"base={base}", "--input", "head=HEAD", "--input", f"mode={mode}"]


def _docker(*a):
    subprocess.run(["docker", *a], check=True, capture_output=True)


def main(mutant_id, run_dir, model_dir, demo, out, mode="default", plan="plans/2026-10-06-d3-t7-workflow-act.md"):
    free = shutil.disk_usage("C:/").free / 1e9
    if free < MIN_FREE_GB:
        sys.exit(f"đĩa C: còn {free:.1f} GB < {MIN_FREE_GB} GB: dừng")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    before = images()
    for t in before:
        _docker("tag", t, backup(t))
    try:  # trả lại image cả khi bị ngắt hay make_repo lỗi
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = Path(tmp) / "repo"
            base = make_repo(repo, mutant_id, run_dir, model_dir, demo)
            t0 = time.time()
            (Path(tmp) / "empty.env").write_text("", encoding="utf-8")
            r = subprocess.run(act_cmd(base, mode, out / "artifacts", Path(tmp) / "empty.env"),
                               cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")
            wall = time.time() - t0
    finally:
        changed = restore(before, images(), lambda src, dst: _docker("tag", src, dst))
        now = images()
        after = {t: i for t, i in now.items() if t in before}  # tag mới (pull image chưa có) không phải lỗi trả
        if after == before:
            for t in before:
                _docker("rmi", backup(t))  # chỉ gỡ tag: tag gốc vẫn trỏ cùng image
        (out / "images.json").write_text(json.dumps({"before": before, "after": after, "retagged": changed,
                                                     "new": sorted(set(now) - set(before))}, indent=1), encoding="utf-8")
    (out / "act.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    (out / "params.json").write_text(json.dumps({
        "plan": plan, "mutant_id": mutant_id, "mode": mode, "base_commit": base,
        "act": subprocess.run([ACT, "--version"], capture_output=True, text=True).stdout.strip(),
        "act_args": "workflow_dispatch -W .github/workflows/ciselect.yml -P ubuntu-latest=-self-hosted --env-file <rỗng>", "exit_code": r.returncode,
        "wall_s": round(wall, 1)}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "env.txt").write_text(f"Python {platform.python_version()}\n{platform.platform()}\n", encoding="utf-8")
    if after != before:
        sys.exit("image phát hành chưa trả về đúng ID: xem images.json")
    print(mutant_id, "exit", r.returncode, f"{wall:.0f}s")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mutant_id", help="mã mutant, ví dụ s0-0013")
    ap.add_argument("run_dir", help="thư mục run sinh bộ mutant (có mutants/ và mutant_status.csv)")
    ap.add_argument("model_dir", help="thư mục mô hình, ví dụ tool/model")
    ap.add_argument("demo", help="clone OpenTelemetry Demo có tag 3.1.0")
    ap.add_argument("out", help="thư mục ghi kết quả của lượt chạy")
    ap.add_argument("mode", nargs="?", default='default', help="chế độ của ciselect")
    ap.add_argument("plan", nargs="?", default='plans/2026-10-06-d3-t7-workflow-act.md', help="đường dẫn plan ghi vào params.json")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(**vars(parse()))
