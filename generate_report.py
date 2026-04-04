#!/usr/bin/env python3
"""
Regime Ensemble v2.0 — One-Page Methodology Update Report
==========================================================
Generates a single-page PDF (outputs/v2_methodology_report.pdf) comparing
v1 and v2 walk-forward approaches in plain English, with schematic diagrams
and a benefits/costs summary.

No Polygon API key required — diagrams are schematic.

Usage:
    python3 generate_report.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────

C = {
    "train":    "#2980b9",
    "test":     "#e74c3c",
    "leak":     "#e67e22",
    "clean":    "#27ae60",
    "bg":       "#fafafa",
    "text":     "#2c3e50",
    "subtext":  "#7f8c8d",
    "pos":      "#27ae60",
    "neg":      "#c0392b",
    "neutral":  "#95a5a6",
    "gold":     "#f39c12",
}

# ── Figure layout ──────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 9.5), facecolor="white")
fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08, hspace=0.55, wspace=0.38)

gs_top = gridspec.GridSpec(1, 3, figure=fig, left=0.04, right=0.96,
                           top=0.88, bottom=0.57, wspace=0.35)
gs_bot = gridspec.GridSpec(1, 3, figure=fig, left=0.04, right=0.96,
                           top=0.50, bottom=0.08, wspace=0.35)

# ── Title ──────────────────────────────────────────────────────────────────

fig.text(0.5, 0.975, "Regime Ensemble  ·  v2.0 Methodology Update",
         ha="center", va="top", fontsize=15, fontweight="bold", color=C["text"])
fig.text(0.5, 0.955, "Plain-English Summary  ·  What Changed, Why It Matters, and What It Costs",
         ha="center", va="top", fontsize=9, color=C["subtext"])

# ══════════════════════════════════════════════════════════════════════
#  TOP ROW
# ══════════════════════════════════════════════════════════════════════

# ── Panel A: Walk-forward fold schematic v1 vs v2 ──────────────────────────

ax_a = fig.add_subplot(gs_top[0])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(-0.5, 4.5)
ax_a.axis("off")
ax_a.set_title("How Walk-Forward Works", fontsize=9, fontweight="bold",
               color=C["text"], pad=4)

fold_labels = ["Fold 1", "Fold 2", "Fold 3"]
for i, label in enumerate(fold_labels):
    y = 3.8 - i * 1.3
    train_end = 4.5 + i * 0.6
    # Training bar
    ax_a.barh(y, train_end, left=0, height=0.5, color=C["train"], alpha=0.75)
    # Test bar
    ax_a.barh(y, 1.8, left=train_end, height=0.5, color=C["test"], alpha=0.85)
    ax_a.text(-0.1, y, label, va="center", ha="right", fontsize=7.5, color=C["text"])

patches = [
    mpatches.Patch(color=C["train"], alpha=0.75, label="Training data"),
    mpatches.Patch(color=C["test"],  alpha=0.85, label="Test window (held-out)"),
]
ax_a.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.7)
ax_a.text(5.0, -0.25,
          "Each fold trains on all history up to that point,\n"
          "then tests on the next ~63 trading days (~1 quarter).",
          ha="center", va="top", fontsize=7.2, color=C["subtext"],
          style="italic")

# ── Panel B: v1 — The Leakage Problem ─────────────────────────────────────

ax_b = fig.add_subplot(gs_top[1])
ax_b.set_xlim(0, 10)
ax_b.set_ylim(-0.5, 4.5)
ax_b.axis("off")
ax_b.set_title("v1  ·  The Leakage Problem  ✗", fontsize=9, fontweight="bold",
               color=C["neg"], pad=4)

y_bar = 3.3
ax_b.barh(y_bar, 6.5, left=0, height=0.6, color=C["train"], alpha=0.7, label="Train")
ax_b.barh(y_bar, 2.0, left=6.5, height=0.6, color=C["test"],  alpha=0.8, label="Test")

# Leakage arrow spanning full bar
ax_b.annotate("", xy=(8.5, y_bar + 0.85), xytext=(0, y_bar + 0.85),
              arrowprops=dict(arrowstyle="<->", color=C["leak"], lw=1.8))
ax_b.text(4.25, y_bar + 1.05, "Markov EM fitted on BOTH",
          ha="center", fontsize=7.5, color=C["leak"], fontweight="bold")

ax_b.text(7.5, y_bar - 0.25, "TEST\nDATA", ha="center", va="top",
          fontsize=6.5, color=C["neg"], fontweight="bold")

points = [
    ("Geometric thresholds", "Computed on full dataset", C["neg"]),
    ("Markov (EM algorithm)", "Fitted on train + test window", C["neg"]),
    ("Walk-forward scores", "Overfit to test data — too optimistic", C["neg"]),
]
for j, (title, desc, col) in enumerate(points):
    yy = 2.1 - j * 0.65
    ax_b.text(0.2, yy, f"• {title}:", fontsize=7.5, color=col, fontweight="bold")
    ax_b.text(0.2, yy - 0.26, f"  {desc}", fontsize=7.0, color=C["subtext"])

ax_b.text(5.0, -0.3,
          "The model 'peeked' at test data during fitting.\n"
          "Walk-forward results were unrealistically good.",
          ha="center", va="top", fontsize=7.2, color=C["subtext"], style="italic")

# ── Panel C: v2 — The Fix ─────────────────────────────────────────────────

ax_c = fig.add_subplot(gs_top[2])
ax_c.set_xlim(0, 10)
ax_c.set_ylim(-0.5, 4.5)
ax_c.axis("off")
ax_c.set_title("v2  ·  The Fix  ✓", fontsize=9, fontweight="bold",
               color=C["pos"], pad=4)

ax_c.barh(y_bar, 6.5, left=0, height=0.6, color=C["train"], alpha=0.7)
ax_c.barh(y_bar, 2.0, left=6.5, height=0.6, color=C["test"],  alpha=0.8)

ax_c.annotate("", xy=(6.4, y_bar + 0.85), xytext=(0, y_bar + 0.85),
              arrowprops=dict(arrowstyle="<->", color=C["clean"], lw=1.8))
ax_c.text(3.2, y_bar + 1.05, "Markov EM on train only",
          ha="center", fontsize=7.5, color=C["clean"], fontweight="bold")

ax_c.annotate("", xy=(8.5, y_bar + 0.85), xytext=(6.6, y_bar + 0.85),
              arrowprops=dict(arrowstyle="->", color=C["train"], lw=1.6,
                              connectionstyle="arc3,rad=0"),)
ax_c.text(7.55, y_bar + 1.05, "Filter only\n(frozen params)",
          ha="center", fontsize=6.5, color=C["train"])

points_v2 = [
    ("Geometric thresholds", "Computed on train slice only", C["pos"]),
    ("Markov (EM algorithm)", "Fitted on train only", C["pos"]),
    ("Test-set filtering", "Hamilton filter with frozen params", C["pos"]),
]
for j, (title, desc, col) in enumerate(points_v2):
    yy = 2.1 - j * 0.65
    ax_c.text(0.2, yy, f"• {title}:", fontsize=7.5, color=col, fontweight="bold")
    ax_c.text(0.2, yy - 0.26, f"  {desc}", fontsize=7.0, color=C["subtext"])

ax_c.text(5.0, -0.3,
          "No data from the test window ever enters the model.\n"
          "Walk-forward scores now reflect genuine predictive power.",
          ha="center", va="top", fontsize=7.2, color=C["subtext"], style="italic")

# ══════════════════════════════════════════════════════════════════════
#  BOTTOM ROW
# ══════════════════════════════════════════════════════════════════════

# ── Panel D: Expected impact on walk-forward scores ────────────────────────

ax_d = fig.add_subplot(gs_bot[0])
ax_d.set_title("Expected Impact on Walk-Forward Scores", fontsize=9,
               fontweight="bold", color=C["text"], pad=4)

categories = ["Mean\nreturn/day", "T-statistic", "% positive\nfolds", "Apparent\nSharp edge"]
v1_bars = [0.085, 2.6, 80, 90]   # schematic — v1 was optimistically high
v2_bars = [0.055, 1.7, 60, 65]   # v2 — more realistic, lower but still positive

x = np.arange(len(categories))
width = 0.35
b1 = ax_d.bar(x - width/2, v1_bars, width, label="v1 (biased)",
              color=C["neg"], alpha=0.75)
b2 = ax_d.bar(x + width/2, v2_bars, width, label="v2 (clean OOS)",
              color=C["pos"], alpha=0.85)

ax_d.set_xticks(x)
ax_d.set_xticklabels(categories, fontsize=7.5)
ax_d.set_yticks([])
ax_d.set_ylabel("Relative scale", fontsize=8)
ax_d.legend(fontsize=7.5, loc="upper right")
ax_d.spines[["top", "right", "left"]].set_visible(False)

ax_d.text(0.5, -0.22,
          "Scores dip in v2 — this is correct.\nInflated v1 scores were artefacts of test-data leakage.",
          ha="center", va="top", transform=ax_d.transAxes,
          fontsize=7.0, color=C["subtext"], style="italic")

# ── Panel E: Benefits ──────────────────────────────────────────────────────

ax_e = fig.add_subplot(gs_bot[1])
ax_e.axis("off")
ax_e.set_title("Benefits  ✓", fontsize=9, fontweight="bold", color=C["pos"], pad=4)

benefits = [
    ("Credible walk-forward scores",
     "Results now reflect real predictive power, not\ntest-data leakage. Safe to report externally."),
    ("Longer data history",
     "2000→2025 covers dot-com crash, GFC 2008,\nCOVID 2020 — 3× more regimes seen."),
    ("Multi-asset support",
     "Run on QQQ, IWM, or any Polygon ticker\nwith --ticker flag. Outputs don't overwrite."),
    ("VIX data pipeline ready",
     "I:VIX fetched + cached via --fetch-vix.\nFoundation for a 3rd ensemble signal."),
    ("Cleaner codebase",
     "compute_thresholds() helper removes\nduplication across run.py / walkforward.py."),
]

for i, (title, body) in enumerate(benefits):
    y = 0.96 - i * 0.195
    ax_e.text(0.02, y, f"✓  {title}", transform=ax_e.transAxes,
              fontsize=8, fontweight="bold", color=C["pos"], va="top")
    ax_e.text(0.06, y - 0.055, body, transform=ax_e.transAxes,
              fontsize=7.2, color=C["subtext"], va="top")

# ── Panel F: Costs ─────────────────────────────────────────────────────────

ax_f = fig.add_subplot(gs_bot[2])
ax_f.axis("off")
ax_f.set_title("Costs & Caveats  !", fontsize=9, fontweight="bold", color=C["gold"], pad=4)

costs = [
    ("Scores will look worse",
     "This is expected and correct — the old scores\nwere inflated by leakage, not genuine signal."),
    ("Markov on 63-day test windows",
     "Short windows mean the first ~5 observations\nper fold have higher regime uncertainty."),
    ("No in-sample threshold review",
     "run.py still computes geometric thresholds on\nfull data (correct for in-sample backtest only)."),
    ("VIX not yet wired into ensemble",
     "Fetched and cached but not used as a signal.\nWiring it in is the logical next step."),
    ("Single asset, short history",
     "Even with 25 years, one asset is insufficient\nfor general conclusions. Cross-validate on more."),
]

for i, (title, body) in enumerate(costs):
    y = 0.96 - i * 0.195
    ax_f.text(0.02, y, f"!  {title}", transform=ax_f.transAxes,
              fontsize=8, fontweight="bold", color=C["gold"], va="top")
    ax_f.text(0.06, y - 0.055, body, transform=ax_f.transAxes,
              fontsize=7.2, color=C["subtext"], va="top")

# ── Footer ─────────────────────────────────────────────────────────────────

fig.text(0.5, 0.005,
         "regime_ensemble v2.0  ·  github.com/benedictprimmer-web/regime_ensemble  "
         "·  Bar heights in Panel D are schematic (relative scale, not actual measured values)",
         ha="center", va="bottom", fontsize=6.5, color=C["subtext"])

# ── Save ───────────────────────────────────────────────────────────────────

path_pdf = OUTPUT_DIR / "v2_methodology_report.pdf"
path_png = OUTPUT_DIR / "v2_methodology_report.png"
fig.savefig(path_pdf, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(path_png, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"  Report saved →  {path_pdf}")
print(f"  Report saved →  {path_png}")
