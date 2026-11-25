"""Dựng repo để chạy workflow bằng `act`: demo 3.1.0, commit "tool" (công cụ, mô hình, workflow), trên cùng là commit mutant.

Base của ciselect là commit "tool", nên diff base..HEAD chỉ chứa thay đổi của mutant. Không ghi vào DEMO.
"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPY = ["tool/_paths.py", "tool/ciselect", "tool/ci", "tool/train_release.py", "tool/requirements.txt", "experiments/ts02/selection",
        "experiments/mutants", "experiments/graph",
        "experiments/spike/run_spike.py", "experiments/spike/junit.py", "experiments/spike/compose.memfix.yaml"]


def _git(r, *a):
    return subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True, text=True).stdout.strip()


def clone(demo, out):
    """clone --shared (chỉ đọc object của DEMO), không checkout, tắt autocrlf, có danh tính để commit."""
    subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "-q", "--shared", "--no-checkout", str(demo), str(out)],
                   check=True, capture_output=True)
    for k, v in [("core.autocrlf", "false"), ("user.name", "ci"), ("user.email", "ci@local")]:
        _git(out, "config", k, v)


def apply_commit(repo, patch, msg, tag=None):
    """Áp patch rồi commit; có tag thì về tag trước (bỏ mutant cũ)."""
    if tag:
        _git(repo, "checkout", "-q", "-f", "--detach", tag)
        _git(repo, "clean", "-fdq")
    _git(repo, "apply", str(Path(patch).resolve()))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)


def make_repo(out, mutant_id, run_dir, model_dir, demo, tag="3.1.0"):
    out = Path(out)
    clone(demo, out)
    _git(out, "checkout", "-q", "-b", "ci", tag)
    ign = shutil.ignore_patterns("__pycache__", "test_*.py", "conftest.py")
    for p in COPY:
        src, dst = ROOT / p, out / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, ignore=ign) if src.is_dir() else shutil.copy2(src, dst)
    shutil.copytree(model_dir, out / "model")
    (out / ".github" / "workflows").mkdir(parents=True, exist_ok=True)  # demo có sẵn workflow riêng: act chạy với -W
    shutil.copy2(ROOT / "tool" / "ci" / "ciselect.yml", out / ".github" / "workflows" / "ciselect.yml")
    _git(out, "add", "-A")
    _git(out, "commit", "-qm", "ci: thêm ciselect, mô hình giao đi, workflow")
    base = _git(out, "rev-parse", "HEAD")
    apply_commit(out, Path(run_dir) / "mutants" / "patches" / f"{mutant_id}.patch", f"mutant {mutant_id}")
    return base
