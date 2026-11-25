"""Job `select` của workflow: kiểm phiên bản thư viện, chạy ciselect, ghi output cho GitHub Actions.

Phiên bản lightgbm, scikit-learn, numpy phải trùng với features.json của thư mục mô hình.
Đầu ra: OUT_JSON chứa quyết định. Ba khoá build, e2e, tests ghi thêm vào $GITHUB_OUTPUT.

Chạy: python tool/ci/select_step.py BASE HEAD MODE MODEL_DIR OUT_JSON
"""
import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # thư mục tool/
import _paths  # noqa: F401,E402

CLI = Path(__file__).resolve().parents[1] / "ciselect" / "cli.py"


def check_versions(features, installed):
    bad = {k: (features[k], installed.get(k)) for k in ("lightgbm", "scikit-learn", "numpy") if installed.get(k) != features[k]}
    if bad:
        sys.exit(f"phiên bản thư viện lệch mô hình giao đi (cần, đang có): {bad}")


def outputs(o):
    return f"build={json.dumps(o['build'])}\ne2e={str(o['e2e']).lower()}\ntests={json.dumps(o['tests'])}\n"


def main(base, head, mode, model_dir, out_json):
    feat = json.loads((Path(model_dir) / "features.json").read_text(encoding="utf-8"))
    check_versions(feat, {k: version(k) for k in ("lightgbm", "scikit-learn", "numpy")})
    r = subprocess.run([sys.executable, str(CLI), "--repo", ".", "--base", base, "--head", head, "--mode", mode,
                        "--model-dir", str(model_dir)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ciselect lỗi (mã {r.returncode}):\n{r.stderr}")
    o = json.loads(r.stdout)
    Path(out_json).write_text(json.dumps(o, indent=1), encoding="utf-8")
    print(json.dumps(o))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(outputs(o))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="commit base")
    ap.add_argument("head", help="commit head")
    ap.add_argument("mode", help="chế độ của ciselect")
    ap.add_argument("model_dir", help="thư mục mô hình, ví dụ tool/model")
    ap.add_argument("out_json", help="tệp JSON ghi quyết định")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(**vars(parse()))
