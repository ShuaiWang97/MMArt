"""
plot_results.py

Generates publication-quality figures for the PolyArt perspective
complementarity experiment (Phase 3 results).

Three figures:
  Fig 1 — Grouped bar chart: all 9 conditions × 3 metrics (main result)
  Fig 2 — Heatmap: perspective × metric alignment (singles only)
  Fig 3 — Leave-one-out delta: marginal contribution of each perspective

Usage:
  python scripts/plot_results.py

Output:
  output/figures/fig1_conditions_bar.pdf  (+ .png)
  output/figures/fig2_alignment_heatmap.pdf
  output/figures/fig3_loo_delta.pdf
"""

import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parent.parent
RESULTS    = REPO_ROOT / "output" / "phase3_results"
OUT_DIR    = REPO_ROOT / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ACM MM column width ≈ 3.33 in; full page ≈ 7.0 in
FULL_W, HALF_W = 7.0, 3.4
DPI = 300

# Palette — colorblind-safe
C_FLUX  = "#2171B5"   # blue  — flux2_klein
C_QWEN  = "#CB181D"   # red   — qwen_image
C_LIGHT = 0.45        # alpha for error bars / hatching

CONDITION_LABELS = {
    "N": "N", "F": "F", "E": "E", "H": "H",
    "NFE": "NFE", "NFH": "NFH", "NEH": "NEH", "FEH": "FEH",
    "NFEH": "NFEH",
}

METRIC_LABELS = {
    "clip_sim":      "CLIP Similarity\n(Style / Semantic)",
    "dino_sim":      "DINOv2 Similarity\n(Composition)",
    "emotion_agree": "Emotion Agreement\n(Affective Tone)",
}

PERSPECTIVE_LABELS = {"N": "Narrative", "F": "Formal",
                       "E": "Emotional", "H": "Historical"}

ALL_CONDITIONS = ["N", "F", "E", "H", "NFE", "NFH", "NEH", "FEH", "NFEH"]
SINGLES        = ["N", "F", "E", "H"]
LOO            = ["NFE", "NFH", "NEH", "FEH"]
METRICS        = ["clip_sim", "dino_sim", "emotion_agree"]
MODELS         = ["flux2_klein", "qwen_image"]
MODEL_LABELS   = {"flux2_klein": "FLUX.2-Klein", "qwen_image": "Qwen-Image"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_summary():
    path = RESULTS / "results_summary.json"
    return json.load(open(path))

def mean_std(summary, cond, model, metric):
    entry = summary.get(cond, {}).get(model, {}).get(metric, {})
    return entry.get("mean", 0.0), entry.get("std", 0.0)

# ---------------------------------------------------------------------------
# Figure 1 — Grouped bar chart (main result)
# ---------------------------------------------------------------------------

def fig1_conditions_bar(summary):
    n_conds   = len(ALL_CONDITIONS)
    n_metrics = len(METRICS)
    n_models  = len(MODELS)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    fig.subplots_adjust(wspace=0.32, bottom=0.32, top=0.86)

    bar_w   = 0.35
    offsets = [-bar_w / 2, bar_w / 2]
    colors  = [C_FLUX, C_QWEN]
    hatches = ["", "//"]

    x = np.arange(n_conds)

    for ax, metric in zip(axes, METRICS):
        for i, (model, color, hatch, offset) in enumerate(
            zip(MODELS, colors, hatches, offsets)
        ):
            means = [mean_std(summary, c, model, metric)[0] for c in ALL_CONDITIONS]
            stds  = [mean_std(summary, c, model, metric)[1] for c in ALL_CONDITIONS]

            bars = ax.bar(
                x + offset, means,
                width=bar_w,
                color=color, alpha=0.85,
                hatch=hatch, edgecolor="white", linewidth=0.4,
                yerr=stds, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "grey"},
                zorder=3,
            )

        # Shade regions
        ax.axvspan(-0.5, 3.5, color="#f0f0f0", zorder=0, label="Singles")
        ax.axvspan(3.5, 7.5, color="#e8f0ff", zorder=0, label="Leave-one-out")
        ax.axvspan(7.5, 8.5, color="#fff0e0", zorder=0, label="Full (NFEH)")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in ALL_CONDITIONS],
            fontsize=7.5, ha="right", rotation=45
        )
        # Group bracket annotations
        y_ann = -0.18
        for mid, label in [(1.5, "Singles"), (5.5, "Leave-one-out"), (8, "Full")]:
            ax.annotate(label, xy=(mid, y_ann), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=6.5, color="#444444",
                        style="italic")
        ax.set_title(METRIC_LABELS[metric], fontsize=8, pad=4)
        ax.set_ylabel("Score", fontsize=7.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Legend — models only; groups annotated directly on axes
    patches = [
        mpatches.Patch(color=C_FLUX, label="FLUX.2-Klein"),
        mpatches.Patch(color=C_QWEN, hatch="//", label="Qwen-Image"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Reconstruction Fidelity Across Perspective Conditions",
        fontsize=9, fontweight="bold", y=0.97
    )

    for ext in ["pdf", "png"]:
        p = OUT_DIR / f"fig1_conditions_bar.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1 → {OUT_DIR}/fig1_conditions_bar.[pdf|png]")


# ---------------------------------------------------------------------------
# Figure 2 — Heatmap: perspective × metric alignment (singles only)
# ---------------------------------------------------------------------------

def fig2_alignment_heatmap(summary):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.72, 2.4))
    fig.subplots_adjust(wspace=0.45, bottom=0.15, top=0.85)

    metric_short = ["CLIP\n(Style)", "DINO\n(Composition)", "Emotion\n(Affective)"]

    for ax, model in zip(axes, MODELS):
        matrix = np.zeros((len(SINGLES), len(METRICS)))
        for i, cond in enumerate(SINGLES):
            for j, metric in enumerate(METRICS):
                matrix[i, j] = mean_std(summary, cond, model, metric)[0]

        # Normalize each column (metric) to [0,1] so relative ranking is visible
        col_min = matrix.min(axis=0, keepdims=True)
        col_max = matrix.max(axis=0, keepdims=True)
        norm    = (matrix - col_min) / (col_max - col_min + 1e-8)

        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        # Annotate with raw scores
        for i in range(len(SINGLES)):
            for j in range(len(METRICS)):
                ax.text(j, i, f"{matrix[i,j]:.3f}",
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="white" if norm[i, j] > 0.6 else "black")

        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(metric_short, fontsize=8)
        ax.set_yticks(range(len(SINGLES)))
        ax.set_yticklabels(
            [f"{c} ({PERSPECTIVE_LABELS[c]})" for c in SINGLES],
            fontsize=8
        )
        ax.set_title(MODEL_LABELS[model], fontsize=8.5, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Perspective–Metric Alignment (Singles Only)\n"
        "Darker = higher score within metric column",
        fontsize=8.5, fontweight="bold", y=1.02
    )

    for ext in ["pdf", "png"]:
        p = OUT_DIR / f"fig2_alignment_heatmap.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2 → {OUT_DIR}/fig2_alignment_heatmap.[pdf|png]")


# ---------------------------------------------------------------------------
# Figure 3 — Leave-one-out delta (marginal contribution)
# ---------------------------------------------------------------------------

def fig3_loo_delta(summary):
    # Delta = NFEH score − LOO score (positive = removing that perspective hurts)
    # LOO mapping: remove P → condition without P
    removed_to_cond = {"N": "FEH", "F": "NEH", "E": "NFH", "H": "NFE"}

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.2))
    fig.subplots_adjust(wspace=0.4, bottom=0.28, top=0.82)

    bar_w   = 0.3
    offsets = [-bar_w / 2, bar_w / 2]
    colors  = [C_FLUX, C_QWEN]
    hatches = ["", "//"]
    x       = np.arange(len(SINGLES))

    for ax, metric in zip(axes, METRICS):
        for model, color, hatch, offset in zip(MODELS, colors, hatches, offsets):
            nfeh_score = mean_std(summary, "NFEH", model, metric)[0]
            deltas = []
            for p in SINGLES:
                loo_cond  = removed_to_cond[p]
                loo_score = mean_std(summary, loo_cond, model, metric)[0]
                deltas.append(nfeh_score - loo_score)

            bar_colors = [color if d >= 0 else "#999999" for d in deltas]
            ax.bar(
                x + offset, deltas,
                width=bar_w,
                color=bar_colors, alpha=0.85,
                hatch=hatch, edgecolor="white", linewidth=0.4,
                zorder=3,
            )

        ax.axhline(0, color="black", linewidth=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"−{p} ({PERSPECTIVE_LABELS[p][:4]})" for p in SINGLES],
            fontsize=7.5, ha="right", rotation=30
        )
        ax.set_title(METRIC_LABELS[metric], fontsize=8, pad=4)
        ax.set_ylabel("NFEH − LOO (↑ = more important)", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    patches = [
        mpatches.Patch(color=C_FLUX, label="FLUX.2-Klein"),
        mpatches.Patch(color=C_QWEN, hatch="//", label="Qwen-Image"),
        mpatches.Patch(color="#999999", label="Removing this perspective slightly helps"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Marginal Contribution of Each Perspective (NFEH − Leave-One-Out)\n"
        "Positive = removing this perspective hurts reconstruction",
        fontsize=8.5, fontweight="bold", y=0.97
    )

    for ext in ["pdf", "png"]:
        p = OUT_DIR / f"fig3_loo_delta.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3 → {OUT_DIR}/fig3_loo_delta.[pdf|png]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading results ...")
    summary = load_summary()

    print("Generating figures ...")
    fig1_conditions_bar(summary)
    fig2_alignment_heatmap(summary)
    fig3_loo_delta(summary)

    print(f"\nAll figures saved to {OUT_DIR}/")
