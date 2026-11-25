"""Job `e2e` của workflow: kéo image phát hành, build lại service được chọn, dựng hệ, chạy test đã chọn, gỡ hệ.

Lệnh compose và tham số test giống bộ chạy dùng khi sinh bộ dữ liệu (`experiments/spike/run_spike.py`).
Đầu vào: chuỗi JSON danh sách service cần build và chuỗi JSON danh sách test, lấy từ output của job select.
Đầu ra: OUT_DIR gồm tests.txt và kết quả bài kiểm thử dạng junit.
Chạy (từ gốc repo demo): python tool/ci/e2e.py BUILD_JSON TESTS_JSON OUT_DIR
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPIKE = ROOT / "experiments" / "spike"
sys.path.insert(0, str(SPIKE))
from run_spike import COMPOSE, telemetry_cmd  # noqa: E402

BASE = COMPOSE + ["-f", str(SPIKE / "compose.memfix.yaml")]
ENV = {**os.environ, "DEMO_VERSION": "3.1.0"}


def node_id(test_id):
    mod, rest = test_id.split("::", 1)
    return f"{mod}.py::{rest}"


def commands(services, tests_file, out):
    return [BASE + ["pull", "-q"],
            BASE + ["build"] + list(services),
            BASE + ["up", "-d", "--force-recreate", "--remove-orphans", "--wait", "--wait-timeout", "600"],
            telemetry_cmd(Path(out).resolve(), Path(".env").resolve(), "240", "60", "false", tests_file),
            BASE + ["down", "-v", "--remove-orphans"]]


def main(build_json, tests_json, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tf = out / "tests.txt"
    tf.write_text("\n".join(node_id(t) for t in json.loads(tests_json)) + "\n", encoding="utf-8")
    *steps, down = commands(json.loads(build_json), tf, out)
    try:
        for c in steps:
            print("+", " ".join(map(str, c)), flush=True)
            r = subprocess.run(c, env=ENV)
            if r.returncode and c is not steps[-1]:  # build/up hỏng: dừng; test fail: vẫn gỡ stack, báo mã thoát
                sys.exit(r.returncode)
    finally:
        subprocess.run(down, env=ENV)
    sys.exit(r.returncode)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("build_json", help="chuỗi JSON: danh sách service cần build lại")
    ap.add_argument("tests_json", help="chuỗi JSON: danh sách test e2e cần chạy")
    ap.add_argument("out", help="thư mục ghi kết quả junit")
    return ap.parse_args(argv)


if __name__ == "__main__":
    # stdout chuyển hướng trên Windows dùng cp1252, không in được tiếng Việt
    if (getattr(sys.stdout, "encoding", None) or "").lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main(**vars(parse()))
