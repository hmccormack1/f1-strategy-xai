"""Static PNG figures. Blue = XGBoost/SHAP, orange = EBM, gray = reference."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import FEATURE_COLS, SHORT_NAMES

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def importance_comparison(imp_a, imp_b, label_a, label_b, title, path, color_b=ORANGE, top_n=12):
    order = pd.Series(imp_a).sort_values().index[-top_n:]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(order) + 1.3))
    for i, f in enumerate(order):
        ax.plot([imp_a[f], imp_b[f]], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter([imp_a[f] for f in order], y, s=60, color=BLUE, label=label_a, zorder=2, edgecolor=SURFACE, lw=1.5)
    ax.scatter([imp_b[f] for f in order], y, s=60, color=color_b, label=label_b, zorder=2, edgecolor=SURFACE, lw=1.5)
    ax.set_yticks(y, [SHORT_NAMES[f] for f in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |contribution| (log-odds of pitting)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, path)


def shape_comparison(curves, path, title="How each feature pushes the pit prediction"):
    """curves: {feature: (shap_df, ebm_df)} with columns x, c."""
    feats = list(curves)
    n = len(feats)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3.2 * rows), squeeze=False)
    for ax, f in zip(axes.flat, feats):
        s, e = curves[f]
        ax.axhline(0, color=GRAY, lw=1)
        ax.plot(s["x"], s["c"], color=BLUE, marker="o", ms=4, label="XGBoost + SHAP")
        ax.plot(e["x"], e["c"], color=ORANGE, marker="o", ms=4, label="EBM (glass box)")
        ax.set_title(SHORT_NAMES[f], fontsize=12)
        ax.set_ylabel("Contribution (log-odds)")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    _save(fig, path)


def stability_bars(mean, std, path, title="SHAP importance across 5 retrainings"):
    order = pd.Series(mean).sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh([SHORT_NAMES[f] for f in order], [mean[f] for f in order], height=0.6,
            xerr=[std[f] for f in order], color=BLUE, error_kw=dict(ecolor=INK2, capsize=3, lw=1))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP| (log-odds); error bar = ±1 SD across seeds")
    ax.set_title(title)
    _save(fig, path)


def deletion_plot(result, path, title="Faithfulness: removing the 'most important' features"):
    colors = {"SHAP ranking": BLUE, "EBM ranking (applied to XGBoost)": ORANGE, "Random ranking": GRAY}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, ys in result["curves"].items():
        ls = "--" if name.startswith("Random") else "-"
        ax.plot(result["k"], ys, color=colors[name], marker="o", ms=5, ls=ls, label=name)
    ax.set_xlabel("Number of top-ranked features removed")
    ax.set_ylabel("Avg. predicted pit probability")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def performance_bars(metrics, seasons, path, metric="pr_auc", title="Pit prediction quality (PR-AUC)"):
    names = [("baseline", "Tyre-age rule", GRAY), ("xgboost", "XGBoost", BLUE), ("ebm", "EBM", ORANGE)]
    x = np.arange(len(seasons))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (key, label, color) in enumerate(names):
        vals = [metrics[key][s][metric] for s in seasons]
        bars = ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=2, color=INK2, fontsize=9)
    rates = [metrics["xgboost"][s]["pit_rate"] for s in seasons]
    for xi, r in zip(x, rates):
        ax.hlines(r, xi - 1.5 * w, xi + 1.5 * w, color=INK2, lw=1, ls=":")
    ax.set_xticks(x, [f"{s} season" + (" (shift test)" if i else " (test)") for i, s in enumerate(seasons)])
    ax.set_ylabel("PR-AUC (dotted = random guessing)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=3)
    _save(fig, path)


def race_timeline(df, probs, stops, threshold, path, title, highlight_lap=None):
    """df: one driver's laps (LapNumber, safety_car, vsc); probs aligned; stops: list of in-laps.
    If path is None, the figure is returned instead of saved (used by the app)."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    if highlight_lap is not None:
        ax.axvline(highlight_lap, color=INK2, lw=1, ls=":")
    laps = df["LapNumber"].values
    for lap, sc, vsc in zip(laps, df["safety_car"], df["vsc"]):
        if sc or vsc:
            ax.axvspan(lap - 0.5, lap + 0.5, color="#eda100", alpha=0.18, lw=0)
    ax.plot(laps, probs, color=BLUE, label="Predicted P(pit in next 3 laps)")
    ax.axhline(threshold, color=GRAY, lw=1, ls="--", label="Decision threshold")
    for i, s in enumerate(stops):
        ax.axvline(s, color=ORANGE, lw=2, label="Actual pit stop" if i == 0 else None)
    ax.set_xlabel("Lap  (shaded = safety car / VSC)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    if path is None:
        fig.tight_layout()
        return fig
    _save(fig, path)


def lap_explanation(names, shap_vals, ebm_vals, title="Why? Contribution of each factor this lap"):
    """Side-by-side horizontal bars of SHAP (XGBoost) and EBM contributions for one lap."""
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.2))
    ax.axvline(0, color=GRAY, lw=1)
    ax.barh(y - 0.2, shap_vals, height=0.38, color=BLUE, label="XGBoost + SHAP")
    ax.barh(y + 0.2, ebm_vals, height=0.38, color=ORANGE, label="EBM")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("← against pitting      contribution (log-odds)      toward pitting →")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def iterations_plot(result, path, title="Explanations led to a better model"):
    names = list(result)
    means = [result[n]["pr_auc_mean"] for n in names]
    sds = [result[n]["pr_auc_sd"] for n in names]
    colors = [BLUE if n.startswith("v3") else (GRAY if "control" in n else "#9dbfea") for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=sds, height=0.6, color=colors, error_kw=dict(ecolor=INK2, capsize=3, lw=1))
    for yi, m, sd in zip(y, means, sds):
        ax.text(m + sd + 0.006, yi, f"{m:.3f}", va="center", color=INK2, fontsize=10)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(means) * 1.18)
    ax.set_xlabel("PR-AUC on unseen test season (mean ± SD over 5 seeds)")
    ax.set_title(title)
    _save(fig, path)


def case_study_panel(d, probs, stops, threshold, focus_lap, names, shap_vals, ebm_vals, title, subtitle, path):
    """Two-row figure: race timeline (top) and the explanation at the focus lap (bottom)."""
    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.5)
    ax = fig.add_subplot(gs[0])
    laps = d["LapNumber"].values
    for lap, c in zip(laps, (d["safety_car"] | d["vsc"] | d["sc_upcoming"])):
        if c:
            ax.axvspan(lap - 0.5, lap + 0.5, color="#eda100", alpha=0.18, lw=0)
    ax.plot(laps, probs, color=BLUE, label="Model: P(pit in next 3 laps)")
    ax.axhline(threshold, color=GRAY, lw=1, ls="--", label="Decision threshold")
    for i, s in enumerate(stops):
        ax.axvline(s, color=ORANGE, lw=2, label="Actual pit stop" if i == 0 else None)
    ax.axvline(focus_lap, color=INK, lw=1, ls=":", label=f"Explained lap ({focus_lap})")
    ax.set_ylim(0, 1.32)  # headroom so the legend sits above the curve
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Lap  (shaded = safety car / VSC)")
    ax.set_ylabel("Probability")
    fig.suptitle(title, x=0.125, y=0.97, ha="left", fontweight="bold", fontsize=14)
    ax.set_title(subtitle, fontsize=10.5, fontweight="normal", color=INK2)
    ax.legend(loc="upper left", fontsize=8.5, ncol=4)

    bx = fig.add_subplot(gs[1])
    y = np.arange(len(names))
    bx.axvline(0, color=GRAY, lw=1)
    bx.barh(y - 0.2, shap_vals, height=0.38, color=BLUE, label="XGBoost + SHAP")
    bx.barh(y + 0.2, ebm_vals, height=0.38, color=ORANGE, label="EBM")
    bx.set_yticks(y, names)
    bx.invert_yaxis()
    bx.grid(axis="y", visible=False)
    bx.set_xlabel("← against pitting      contribution (log-odds)      toward pitting →")
    bx.set_title(f"Why the model said what it did on lap {focus_lap}", fontsize=12)
    bx.legend(loc="lower right", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
