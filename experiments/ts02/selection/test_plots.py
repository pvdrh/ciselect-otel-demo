"""K3 (#85): plots.py chỉ đọc CSV đã ghi; test kiểm đường bao của H1 và việc mọi hình được tạo, không kiểm điểm ảnh."""
from pathlib import Path

import pandas as pd
import pytest

import plots

ROOT = Path(__file__).resolve().parents[3]
SUMM = ROOT / "results" / "ts02_summary" / "20261005-195113_seed0-4"
FIGD = ROOT / "results" / "ts02_figdata" / "20261005-192342_seed0-4"
ORAC = ROOT / "results" / "ts02_oracle" / "20261007-181629_seed0"


def test_envelope_takes_best_savings_among_thresholds_reaching_recall():
    c = pd.DataFrame({"seed": [0, 0, 0, 1, 1], "recall_mutant": [1.0, 0.9, 0.7, 1.0, 0.8],
                      "savings": [0.0, 0.1, 0.3, 0.05, 0.2]})
    assert plots.envelope(c, [0.7, 0.9, 1.0]).tolist() == [[0.3, 0.1, 0.0], [0.2, 0.05, 0.05]]


@pytest.mark.skipif(not (SUMM.exists() and FIGD.exists() and ORAC.exists()), reason="thiếu run D2")
def test_main_writes_every_figure_from_saved_runs(tmp_path):
    plots.main(SUMM, FIGD, tmp_path, ORAC)
    for name in ("h1_tiet_kiem_theo_recall", "h2_sot_xuyen_service", "h3_theo_loai_mutation", "h4_gain_lightgbm",
                 "h5_duong_U_r"):
        for ext in ("png", "svg"):
            assert (tmp_path / f"{name}.{ext}").stat().st_size > 0
    assert (tmp_path / "params.json").exists()
