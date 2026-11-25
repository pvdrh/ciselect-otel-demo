"""Spike B (plan 2026-09-30-b): chạy một lần demo (sạch hoặc có mutant) và ghi kết quả từng test.

Dùng:
    python run_spike.py --out RESULT_DIR --run-id clean-1
    python run_spike.py --out RESULT_DIR --run-id M1-01 --patch mutants/M1-01.patch --service cart
"""
import argparse
import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from junit import parse_junit

HERE = Path(__file__).resolve().parent
DEMO = HERE.parents[1] / "data" / "raw" / "opentelemetry-demo"
TAG = "3.1.0"
COMPOSE = ["docker", "compose", "--env-file", ".env",
           "-f", "compose.yaml", "-f", "compose.observability.yaml", "-f", "compose.extras.yaml"]
ENV = {**os.environ, "DEMO_VERSION": TAG}
FIELDS = ["run_id", "mutant_id", "service", "status", "t_build", "t_start", "t_test_tel", "t_test_cy",
          "t_total", "peak_mem_mb", "oracle", "test_id", "outcome", "duration_s"]


def sh(cmd, timeout, **kw):
    return subprocess.run(cmd, cwd=DEMO, env=ENV, timeout=timeout, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def mem_sampler(stop, peak):
    # Tổng RAM các container, lấy mẫu mỗi 5 s. ponytail: đọc `docker stats`, đủ cho spike.
    units = {"B": 1e-6, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
    while not stop.is_set():
        r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        total = 0.0
        for line in r.stdout.splitlines():
            used = line.split("/")[0].strip()
            num = used.rstrip("abcdefghijklmnopqrstuvwxyzKMGiB")
            unit = used[len(num):]
            try:
                total += float(num) * units.get(unit, 0)
            except ValueError:  # container đang dừng: "--"
                pass
        peak[0] = max(peak[0], total)
        stop.wait(5)


def telemetry_cmd(raw, env_file, warmup, poll_timeout, probe, tests=None, probe_timeout="120", tee=False):
    """Lệnh chạy image test telemetry; tests: file node id (plan T6), không có thì chạy đủ như D1.
    probe_timeout, tee: plan 2026-11-09 (#120), khối probe như CI gốc; tee in dòng của fixture cả khi probe đạt."""
    ids = Path(tests).read_text(encoding="utf-8").split() if tests else []
    return ["docker", "run", "--rm", "--network", "opentelemetry-demo", "--env-file", str(env_file), "-v", f"{raw}:/out",
            "-e", "TEST_SCOPE=minimal", "-e", f"WARMUP_SECONDS={warmup}",
            "-e", f"POLL_TIMEOUT={poll_timeout}", "-e", f"WARMUP_PROBE_ENABLED={probe}",
            "-e", "WARMUP_PROBE_CHECKOUTS=5", "-e", f"WARMUP_PROBE_TIMEOUT={probe_timeout}",
            "opentelemetry-demo-telemetry-tests", "--tb=short", "-q"] + (["--capture=tee-sys"] if tee else []) + [
            "--junitxml=/out/telemetry.xml"] + ids


def restore_cmds(image, old_id, new_id):
    """Plan 2026-11-21 bước 0b: containerd image store xoá image không còn tag khi build ghi đè tag, nên trả bằng ID
    (#120) hỏng ở #122. Trả từ tag dự phòng -cibak như tool/ci/run_act.py, gỡ tag dự phòng, xoá image mutant."""
    bak = image + "-cibak"
    return [["docker", "tag", bak, image], ["docker", "rmi", bak]] + ([] if old_id == new_id else [["docker", "rmi", new_id]])


def image_id(image):
    r = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", image], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def service_image(service):
    return json.loads(sh(COMPOSE + ["config", "--format", "json"], 120, check=True).stdout)["services"][service]["image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--patch")
    ap.add_argument("--service", default="")
    ap.add_argument("--poll-timeout", default="180")
    ap.add_argument("--warmup", default="240")
    ap.add_argument("--no-cypress", action="store_true")
    ap.add_argument("--probe", default="true", help="false: tắt checkout probe của warmup (probe làm mọi test fail khi checkout hỏng)")
    ap.add_argument("--no-build", action="store_true", help="mutant chỉ sửa compose.yaml (M3)")
    ap.add_argument("--compose-extra", help="file compose override thêm vào (decisions #31)")
    ap.add_argument("--tests", help="file node id pytest, mỗi dòng một test (plan T6); bỏ trống = chạy đủ")
    ap.add_argument("--probe-timeout", default="120", help="WARMUP_PROBE_TIMEOUT (#120: 180 như CI gốc)")
    ap.add_argument("--tee-sys", action="store_true", help="pytest --capture=tee-sys như CI gốc (#120)")
    a = ap.parse_args()
    if a.compose_extra:
        COMPOSE.extend(["-f", str(Path(a.compose_extra).resolve())])

    out = Path(a.out).resolve()
    raw = out / "raw" / a.run_id
    raw.mkdir(parents=True, exist_ok=True)
    mutant_id = Path(a.patch).stem if a.patch else "clean"
    t = {"t_build": 0.0, "t_start": 0.0, "t_test_tel": 0.0, "t_test_cy": 0.0}
    status = "ok"
    t0 = time.time()
    peak = [0.0]
    stop = threading.Event()
    sampler = threading.Thread(target=mem_sampler, args=(stop, peak), daemon=True)
    build = bool(a.patch and not a.no_build)
    img = service_image(a.service) if build else ""
    if build and image_id(img + "-cibak"):  # lượt trước chết giữa chừng: trả tag từ bản dự phòng, không gắn đè
        subprocess.run(["docker", "tag", img + "-cibak", img], check=True, capture_output=True)
    old = image_id(img) if build else ""
    restore_ok = not build or bool(old)  # không có ID image phát hành thì không khôi phục được: không chạy
    rmi_ok = True  # xoá được image mutant; hỏng chỉ để lại image dangling, không làm bẩn lượt sau
    if not restore_ok:
        status = "restore_error"

    try:
        if a.patch and status == "ok":
            sh(["git", "apply", str(Path(a.patch).resolve())], 60, check=True)
        if build and status == "ok":
            subprocess.run(["docker", "tag", img, img + "-cibak"], check=True, capture_output=True)
            s = time.time()
            r = sh(COMPOSE + ["build", a.service], 3600)
            t["t_build"] = time.time() - s
            (raw / "build.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            if r.returncode != 0:
                status = "build_error"
        if status == "ok":
            sampler.start()
            s = time.time()
            r = sh(COMPOSE + ["up", "-d", "--force-recreate", "--remove-orphans", "--wait", "--wait-timeout", "600"], 900)
            t["t_start"] = time.time() - s
            (raw / "up.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            if r.returncode != 0:
                # Service bị đột biến không healthy thì tính là fail thật; chỉ gắn infra_error khi không có mutant (D1 §Review).
                status = "unhealthy_mutated" if a.patch else "infra_error"
            s = time.time()
            r = subprocess.run(telemetry_cmd(raw, DEMO / ".env", a.warmup, a.poll_timeout, a.probe, a.tests,
                                             probe_timeout=a.probe_timeout, tee=a.tee_sys),
                               cwd=DEMO, env=ENV, timeout=7200, capture_output=True, text=True, encoding="utf-8", errors="replace")
            t["t_test_tel"] = time.time() - s
            (raw / "telemetry.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            if not a.no_cypress:
                s = time.time()
                r = sh(COMPOSE + ["-f", "compose.tests.yaml", "run", "--rm", "-v", f"{raw}:/out",
                                  # ruling spike B1: test gọi /feature qua proxy + render đầu chậm → baseUrl proxy, timeout 15 s
                                  "-e", "CYPRESS_baseUrl=http://frontend-proxy:8080", "-e", "CYPRESS_defaultCommandTimeout=15000",
                                  "frontendTests",
                                  "--reporter", "junit", "--reporter-options", "mochaFile=/out/cy-[hash].xml"], 3600)
                t["t_test_cy"] = time.time() - s
                (raw / "cypress.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    finally:
        stop.set()
        try:
            if status != "ok" or a.patch:
                r = sh(COMPOSE + ["ps", "-a"], 120)
                (raw / "ps.log").write_text(r.stdout + r.stderr, encoding="utf-8")
                r = sh(COMPOSE + ["logs", "--no-color", "--tail", "200"], 300)
                (raw / "containers.log").write_text(r.stdout + r.stderr, encoding="utf-8")
            sh(COMPOSE + ["down", "-v", "--remove-orphans"], 600)  # -v: không xoá thì mỗi lần dựng để lại khoảng 70 MB volume ẩn danh
            if a.patch:
                sh(["git", "checkout", "--", "."], 60)
        finally:
            if build and old:
                # build đè tag 3.1.0-<svc> bằng image mutant → trả tag từ tag dự phòng -cibak rồi kiểm lại (bước 0b).
                # #120 thay lệnh pull cũ: pull không kiểm mã trả về, hỏng thì image mutant giữ tag cho các lượt sau.
                # Đặt trong finally để vẫn chạy khi down/checkout hết giờ (code-review 2026-11-10).
                rc = [subprocess.run(c, capture_output=True).returncode for c in restore_cmds(img, old, image_id(img))]
                restore_ok = image_id(img) == old
                rmi_ok = all(x == 0 for x in rc[1:])

    rows = parse_junit(raw / "telemetry.xml", "telemetry")
    for x in sorted(raw.glob("cy-*.xml")):
        rows += parse_junit(x, "cypress")
    base = {"run_id": a.run_id, "mutant_id": mutant_id, "service": a.service, "status": status,
            **{k: round(v, 1) for k, v in t.items()}, "t_total": round(time.time() - t0, 1),
            "peak_mem_mb": round(peak[0])}
    table = out / "table.csv"
    new = not table.exists()
    with table.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for oracle, test_id, outcome, dur in rows or [("", "", "", 0.0)]:
            w.writerow({**base, "oracle": oracle, "test_id": test_id, "outcome": outcome, "duration_s": dur})
    print(json.dumps({**base, "n_tests": len(rows), "n_fail": sum(r[2] == "fail" for r in rows), "restore_ok": restore_ok,
                      "rmi_ok": rmi_ok}))


if __name__ == "__main__":
    main()
