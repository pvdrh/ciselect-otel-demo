"""D2-9: vẽ H1–H4 từ summary/ và figdata/, H5 (đường U(r), #51) từ oracle/ (chỉ đọc CSV đã ghi, không tính lại số).

Chạy: python plots.py SUMMARY_DIR FIGDATA_DIR OUT_DIR [ORACLE_DIR]
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Màu cố định theo thực thể (dataviz palette mặc định, đã chạy validate_palette.js — light: pass, contrast WARN → nhãn trực tiếp).
STYLE = {"lgbm": ("#2a78d6", "o", "B4-LGBM"), "lgbm_nofr": ("#eb6834", "s", "B4-LGBM không fail_rate"),
         "lr": ("#1baf7a", "^", "B4-LR"), "b3": ("#eda100", "D", "B3"), "b1": ("#e87ba4", "v", "B1"),
         "b1h": ("#008300", "P", "B1h"), "b2": ("#4a3aa7", "X", "B2"), "b0p": ("#e34948", "*", "B0′")}
# Mã hoá phụ cho bản in đen trắng (outline mục 11): kiểu nét ở H1, hoa văn ở H3.
DASH = {"b3": "-", "lr": "--", "lgbm": "-.", "lgbm_nofr": ":"}
HATCH = {"b1": "", "b1h": "//", "b2": "\\\\", "b3": "xx", "lr": "..", "lgbm": "--", "lgbm_nofr": "++"}
MAIN = [("b1", "natural"), ("b1h", "natural"), ("b2", "natural"), ("b3", "t95"), ("lr", "t95"), ("lgbm", "t95"),
        ("lgbm_nofr", "t95")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def _comma(fig, axes="xy"):
    """Dấu phẩy thập phân, chỉ trên trục số (axes)."""
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        for a in axes:
            getattr(ax, f"{a}axis").set_major_formatter(f)


def _save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def envelope(c, grid):
    """Mỗi seed: tiết kiệm lớn nhất trong các τ có recall ≥ r (đường bao, ngưỡng nhìn trên fold test — chỉ mô tả)."""
    out = []
    for _, g in c.groupby("seed"):
        out.append([g.savings[g.recall_mutant >= r - 1e-12].max() for r in grid])
    return np.array(out)


def h1(summ, fig_d, out):
    c, t = pd.read_csv(fig_d / "h1_curves.csv"), pd.read_csv(summ / "table.csv").set_index(["method", "op"])
    grid = np.linspace(0.6, 1.0, 81)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in ("b3", "lr", "lgbm", "lgbm_nofr"):
        col, mk, lab = STYLE[m]
        e = envelope(c[c.method == m], grid)
        ax.fill_between(grid, e.min(0), e.max(0), color=col, alpha=0.15, linewidth=0)
        ax.plot(grid, e.mean(0), color=col, linewidth=2, linestyle=DASH[m], marker=mk, markevery=10,
                markersize=5, label=lab)
    for m in ("b1", "b1h", "b2"):
        col, mk, lab = STYLE[m]
        r = t.loc[(m, "natural")]
        ax.errorbar(r.recall_mutant_mean, r.savings_mean, xerr=r.recall_mutant_std, yerr=r.savings_std, fmt=mk,
                    color=col, markersize=8, markeredgecolor="white", markeredgewidth=1.5, label=lab, zorder=5)
        ax.annotate(lab, (r.recall_mutant_mean, r.savings_mean), xytext=(6, 4), textcoords="offset points", color=INK)
    col, mk, lab = STYLE["b0p"]
    ax.plot(1.0, 0.0, mk, color=col, markersize=11, markeredgecolor="white", markeredgewidth=1.5, label=lab, zorder=5)
    ax.annotate("B0′ (mốc)", (1.0, 0.0), xytext=(-10, -14), textcoords="offset points", color=INK, ha="right")
    ax.set_ylim(bottom=-0.035)
    ax.axvline(0.95, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.947, ax.get_ylim()[1] * 0.97, "recall 0,95", color=MUTED, va="top", ha="right")
    ax.set_xlabel("Recall theo mutant (fold kiểm tra)")
    ax.set_ylabel("Tiết kiệm so với B0′")
    ax.legend(loc="lower left", fontsize=8)
    _comma(fig)
    _save(fig, out, "h1_tiet_kiem_theo_recall")


def h2(summ, out):
    t = pd.read_csv(summ / "table.csv").set_index(["method", "op"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = [STYLE[m][2] + ("" if op == "natural" else f" ({op})") for m, op in MAIN]
    y = np.arange(len(MAIN))[::-1]
    for yi, (m, op) in zip(y, MAIN):
        r = t.loc[(m, op)]
        ax.barh(yi, r.cross_missed_mean, xerr=r.cross_missed_std, color=STYLE[m][0], height=0.6,
                error_kw={"ecolor": MUTED, "elinewidth": 1})
        ax.text(r.cross_missed_mean + r.cross_missed_std + 0.005, yi, f"{r.cross_missed_mean:.3f}".replace(".", ","),
                va="center", color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Tỉ lệ mutant có lỗi xuyên dịch vụ bị sót (trung bình ± độ lệch chuẩn, 5 seed)")
    ax.grid(axis="y", visible=False)
    _comma(fig, "x")
    _save(fig, out, "h2_sot_xuyen_service")


def h3(summ, out):
    b = pd.read_csv(summ / "by_type.csv").set_index(["method", "op", "type"])
    types = ["M0", "M1", "M2", "M3"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    w = 0.8 / len(MAIN)
    for k, (m, op) in enumerate(MAIN):
        col, _, lab = STYLE[m]
        for ax, metric in zip(axes, ("recall_mutant", "savings")):
            ts = types[1:] if metric == "recall_mutant" else types  # M0 không có fail
            x = np.array([types.index(tt) for tt in ts]) + (k - len(MAIN) / 2 + 0.5) * w
            ax.bar(x, [b.loc[(m, op, tt), f"{metric}_mean"] for tt in ts], width=w * 0.9, color=col,
                   hatch=HATCH[m], edgecolor="white", linewidth=0,
                   label=lab + ("" if op == "natural" else f" ({op})"))
    axes[0].set_title("Recall theo mutant", color=INK, fontsize=9)
    axes[1].set_title("Tiết kiệm so với B0′", color=INK, fontsize=9)
    for ax in axes:
        ax.set_xticks(range(4), types)
        ax.grid(axis="x", visible=False)
    axes[0].set_xlim(0.5, 3.5)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    _comma(fig, "y")
    _save(fig, out, "h3_theo_loai_mutation")


def h4(fig_d, out, top=15):
    g = pd.read_csv(fig_d / "h4_gain.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2))
    for ax, m in zip(axes, ("lgbm", "lgbm_nofr")):
        s = g[g.method == m].groupby("feature").gain.agg(["mean", "std"]).sort_values("mean").tail(top)
        ax.barh(range(len(s)), s["mean"], xerr=s["std"], color=STYLE[m][0], height=0.6,
                error_kw={"ecolor": MUTED, "elinewidth": 1})
        ax.set_yticks(range(len(s)), s.index, fontsize=7.5)
        ax.set_title(STYLE[m][2], color=INK, fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("Gain chuẩn hoá (trung bình ± sd, 5 fold × 5 seed)", fontsize=9)
    _comma(fig, "x")
    _save(fig, out, "h4_gain_lightgbm")


H5_POINTS = [("b1", "natural"), ("b1h", "natural"), ("b2", "natural"), ("b3", "t95"), ("lr", "t95"),
             ("lgbm", "t95"), ("rf", "t95")]  # đúng các dòng của Bảng 4.10


def h5(oracle, out):
    """Đường U(r) (cận trên lý tưởng theo recall) và các phương pháp là điểm, phép chia chính (#51, thăm dò)."""
    u, g = pd.read_csv(oracle / "upper_curve.csv"), pd.read_csv(oracle / "gap.csv")
    g = g[g.split == "main"].set_index(["method", "op"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.step(u.r, u.upper, where="post", color=INK, linewidth=1.8, label="U(r): cận trên lý tưởng")
    style = dict(STYLE, rf=("#7a7a7a", "h", "RF"))
    for m, op in H5_POINTS:
        col, mk, lab = style[m]
        r = g.loc[(m, op)]
        ax.vlines(r.recall_mutant, r.savings, r.upper, color=MUTED, linestyle=":", linewidth=1)
        ax.plot(r.recall_mutant, r.savings, mk, color=col, markersize=8, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5, label=lab + ("" if op == "natural" else f" ({op})"))
    ax.set_xlim(0.7, 1.005)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Recall theo mutant (fold kiểm tra)")
    ax.set_ylabel("Tiết kiệm so với B0′")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    _comma(fig)
    _save(fig, out, "h5_duong_U_r")


def main(summ, fig_d, out, oracle=None):
    summ, fig_d, out = Path(summ), Path(fig_d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    h1(summ, fig_d, out)
    h2(summ, out)
    h3(summ, out)
    h4(fig_d, out)
    if oracle:
        h5(Path(oracle), out)
    (out / "params.json").write_text(f'{{"summary": "{summ.as_posix()}", "figdata": "{fig_d.as_posix()}", '
                                     f'"oracle": "{Path(oracle).as_posix() if oracle else ""}"}}\n', encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:5])
