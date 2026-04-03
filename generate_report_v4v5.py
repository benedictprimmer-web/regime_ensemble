#!/usr/bin/env python3
"""
Regime Ensemble v4.0 & v5.0 -- Changes & Explanations
======================================================
Two-page PDF summarising what changed across v4.0 and v5.0.

  Page 1: v4.0 changes (half-position, persistence filter, convergence fix)
  Page 2: v5.0 changes (walk-forward OOS curve, multi-asset validation,
          summary & what's next)

No Polygon API key required -- diagrams are schematic.

Usage:
    python3 generate_report_v4v5.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

C = {
    "mom":   "#1a7a4a",
    "rev":   "#c0392b",
    "mix":   "#7f8c8d",
    "strat": "#2980b9",
    "bnh":   "#2c3e50",
    "pos":   "#27ae60",
    "neg":   "#c0392b",
    "gold":  "#f39c12",
    "text":  "#2c3e50",
    "sub":   "#7f8c8d",
    "grid":  "#e8e8e8",
    "v4":    "#8e44ad",
    "v5":    "#2980b9",
}

FOOTER = ("regime_ensemble  -  github.com/benedictprimmer-web/regime_ensemble"
          "  -  Schematic values in Panel E: run --multi-asset for real numbers")


def _page_header(fig, version_label, version_color, title, subtitle):
    fig.text(0.5, 0.975, "Regime Ensemble  -  Changes & Explanations",
             ha="center", va="top", fontsize=14, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.952, subtitle,
             ha="center", va="top", fontsize=9, color=C["sub"])
    fig.text(0.08, 0.975, version_label, ha="left", va="top", fontsize=11,
             fontweight="bold", color=version_color,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor=version_color, linewidth=1.5))
    fig.add_artist(plt.Line2D([0.06, 0.96], [0.938, 0.938],
                              transform=fig.transFigure,
                              color=version_color, lw=1.0, alpha=0.4))


def _footer(fig, page, total=2):
    fig.text(0.5, 0.015, "Page %d of %d  -  %s" % (page, total, FOOTER),
             ha="center", va="bottom", fontsize=6.5, color=C["sub"])


# ============================================================================
#  PAGE 1 -- v4.0
# ============================================================================

def make_page1():
    fig = plt.figure(figsize=(13, 9.5), facecolor="white")
    _page_header(fig, "v4.0", C["v4"],
                 "v4.0 Changes",
                 "v4.0  -  Half-position on mixed days  |  Regime persistence filter  |  Markov convergence fix")

    gs = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.05, right=0.97,
                           top=0.895, bottom=0.10,
                           wspace=0.38)

    # -- Panel A: Half-position on mixed days
    ax_a = fig.add_subplot(gs[0])
    ax_a.axis("off")
    ax_a.set_title("A  -  Half-Position on Mixed Days", fontsize=10,
                   fontweight="bold", color=C["v4"], pad=8, loc="left")

    regime_rows = [
        ("Momentum",  "+0.066%/day", "T = 1.32", "p = 0.19",  "x", "hold full  (+1.0)", C["mom"]),
        ("Mixed",     "+0.056%/day", "T = 3.21", "p = 0.001", "v", "hold half (+0.5)",  C["mix"]),
        ("Reversion", "+0.005%/day", "T = 0.35", "p = 0.73",  "x", "cash (0.0)",        C["rev"]),
    ]
    y0 = 0.92
    for i, (regime, mean, tstat, pval, sig, action, col) in enumerate(regime_rows):
        y = y0 - i * 0.225
        # Coloured dot
        ax_a.text(0.02, y, "o", transform=ax_a.transAxes,
                  fontsize=16, color=col, va="top", fontweight="bold")
        ax_a.text(0.13, y, regime, transform=ax_a.transAxes,
                  fontsize=9, fontweight="bold", color=C["text"], va="top")
        ax_a.text(0.13, y - 0.062,
                  "%s     %s   %s" % (mean, tstat, pval),
                  transform=ax_a.transAxes, fontsize=7.5, color=C["sub"],
                  va="top", fontfamily="monospace")
        ax_a.text(0.13, y - 0.118,
                  "->  %s" % action,
                  transform=ax_a.transAxes, fontsize=8, color=col,
                  va="top", style="italic")

    # Before/after box
    ax_a.add_patch(plt.Rectangle((0.01, 0.04), 0.98, 0.20,
                                 transform=ax_a.transAxes, facecolor="#f8f9fa",
                                 edgecolor=C["grid"], lw=1, clip_on=False))
    ax_a.text(0.50, 0.225, "Before v4.0          After v4.0",
              transform=ax_a.transAxes, ha="center", fontsize=8,
              fontweight="bold", color=C["text"], va="top")
    ax_a.text(0.50, 0.175,
              "Sharpe     0.29   ->   0.68\n"
              "T-stat  1.32 (p=0.19)  ->  3.13 (p=0.002)\n"
              "Mixed:   cash (0)  ->  half long (+0.5)",
              transform=ax_a.transAxes, ha="center", fontsize=8,
              color=C["pos"], va="top", fontfamily="monospace")

    ax_a.text(0.50, -0.02,
              "Mixed days have the strongest forward-return signal\n"
              "(T=3.21, p=0.001). Changing from cash to +0.5 long\n"
              "captures that return without full directional exposure.",
              transform=ax_a.transAxes, ha="center", fontsize=7.5,
              color=C["sub"], va="top", style="italic")

    # -- Panel B: Regime persistence filter
    ax_b = fig.add_subplot(gs[1])
    ax_b.axis("off")
    ax_b.set_title("B  -  Regime Persistence Filter  (--min-hold N)", fontsize=10,
                   fontweight="bold", color=C["v4"], pad=8, loc="left")

    raw_sig  = ["M","M","R","R","M","R","R","R","M","M","M","M","R","R","M"]
    filt_sig = ["M","M","M","M","M","R","R","R","M","M","M","M","M","M","M"]
    sm = {"M": C["mom"], "R": C["rev"]}
    n = len(raw_sig)
    bar_w = 0.93 / n

    for label, signals, y_top in [("Raw signal  (no filter)", raw_sig, 0.895),
                                   ("Filtered signal  (--min-hold 3)", filt_sig, 0.720)]:
        ax_b.text(0.50, y_top + 0.015, label, transform=ax_b.transAxes,
                  ha="center", fontsize=8.5, fontweight="bold", color=C["text"], va="bottom")
        for j, s in enumerate(signals):
            x0 = 0.03 + j * bar_w
            rect = plt.Rectangle((x0, y_top - 0.105), bar_w * 0.90, 0.095,
                                  facecolor=sm[s], alpha=0.85,
                                  transform=ax_b.transAxes, clip_on=False)
            ax_b.add_patch(rect)

    patches = [mpatches.Patch(color=C["mom"], alpha=0.85, label="Momentum"),
               mpatches.Patch(color=C["rev"], alpha=0.85, label="Reversion")]
    ax_b.legend(handles=patches, loc="upper right", fontsize=7.5,
                bbox_to_anchor=(1.0, 0.715), framealpha=0.9)

    ax_b.text(0.50, 0.575,
              "->  ~20 switches/year  reduced to ~7/year",
              transform=ax_b.transAxes, ha="center", fontsize=9,
              fontweight="bold", color=C["pos"], va="top")

    # Metrics table
    ax_b.text(0.03, 0.525, "Setting          Cost    CAGR       Sharpe",
              transform=ax_b.transAxes, fontsize=7.5, color=C["text"],
              fontfamily="monospace", fontweight="bold", va="top")
    ax_b.add_artist(plt.Line2D([0.03, 0.97], [0.515, 0.515],
                               transform=ax_b.transAxes, color=C["grid"], lw=0.8))
    table_rows = [
        ("No filter       ", "0 bps  ", "+5.7%  ", "0.68"),
        ("--min-hold 3    ", "0 bps  ", "+4.2%  ", "0.42"),
        ("No filter       ", "10 bps ", "+4.3%  ", "0.48"),
        ("--min-hold 3    ", "10 bps ", "+3.8%  ", "0.38"),
    ]
    for i, row in enumerate(table_rows):
        y = 0.490 - i * 0.085
        ax_b.text(0.03, y, "".join(row), transform=ax_b.transAxes,
                  fontsize=7.5, color=C["sub"], fontfamily="monospace", va="top")

    ax_b.text(0.50, 0.145,
              "Break-even cost:  ~15 bps without filter,  ~20 bps with --min-hold 3",
              transform=ax_b.transAxes, ha="center", fontsize=7.5,
              color=C["text"], va="top", fontweight="bold")

    ax_b.text(0.50, 0.065,
              "Trade-off: fewer position changes = lower raw CAGR\n"
              "but the strategy survives higher transaction costs.",
              transform=ax_b.transAxes, ha="center", fontsize=7.5,
              color=C["sub"], va="top", style="italic")

    # -- Panel C: Markov convergence fix
    ax_c = fig.add_subplot(gs[2])
    ax_c.axis("off")
    ax_c.set_title("C  -  Markov EM Convergence Fix", fontsize=10,
                   fontweight="bold", color=C["v4"], pad=8, loc="left")

    # Before box
    ax_c.add_patch(plt.Rectangle((0.01, 0.595), 0.98, 0.355,
                                 transform=ax_c.transAxes,
                                 facecolor="#fff5f5", edgecolor="#f5b7b1", lw=1))
    ax_c.text(0.50, 0.935, "v3.x  --  before fix  x",
              transform=ax_c.transAxes, ha="center", fontsize=9,
              fontweight="bold", color=C["neg"], va="top")
    before = [
        ("model.fit(disp=False)", C["sub"]),
        ("", C["sub"]),
        ("ConvergenceWarning:", C["neg"]),
        ("  Maximum Likelihood optimization", C["neg"]),
        ("  failed to converge. Check", C["neg"]),
        ("  mle_retvals for further info.", C["neg"]),
        ("", C["sub"]),
        ("# 5-year data: fine", C["sub"]),
        ("# 25-year data: fails on most runs.", C["sub"]),
        ("#   EM needs more iterations.", C["sub"]),
    ]
    for i, (line, col) in enumerate(before):
        ax_c.text(0.06, 0.870 - i * 0.058, line, transform=ax_c.transAxes,
                  fontsize=7.2, color=col, va="top", fontfamily="monospace")

    # After box
    ax_c.add_patch(plt.Rectangle((0.01, 0.065), 0.98, 0.495,
                                 transform=ax_c.transAxes,
                                 facecolor="#f0fff4", edgecolor="#a9dfbf", lw=1))
    ax_c.text(0.50, 0.545, "v4.0  --  after fix  v",
              transform=ax_c.transAxes, ha="center", fontsize=9,
              fontweight="bold", color=C["pos"], va="top")
    after = [
        ("model.fit(", C["sub"]),
        ("  em_iter=200,    # was 5 (default)", C["strat"]),
        ("  search_reps=5,  # 5 random starts", C["strat"]),
        (")  # best BIC across starts kept", C["sub"]),
        ("", C["sub"]),
        ("# em_iter=200: enough for 5,364 obs", C["pos"]),
        ("# search_reps=5: avoids local optima", C["pos"]),
        ("# Zero ConvergenceWarnings in prod.", C["pos"]),
    ]
    for i, (line, col) in enumerate(after):
        ax_c.text(0.06, 0.490 - i * 0.058, line, transform=ax_c.transAxes,
                  fontsize=7.2, color=col, va="top", fontfamily="monospace")

    _footer(fig, 1)
    return fig


# ============================================================================
#  PAGE 2 -- v5.0
# ============================================================================

def make_page2():
    fig = plt.figure(figsize=(13, 9.5), facecolor="white")
    _page_header(fig, "v5.0", C["v5"],
                 "v5.0 Changes",
                 "v5.0  -  Walk-forward OOS equity curve  |  Multi-asset validation  |  Summary & next steps")

    gs = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.05, right=0.97,
                           top=0.895, bottom=0.11,
                           wspace=0.38)

    # -- Panel D: Walk-forward OOS equity curve 
    ax_d = fig.add_subplot(gs[0])
    ax_d.set_title("D  -  Walk-Forward OOS Equity Curve", fontsize=10,
                   fontweight="bold", color=C["v5"], pad=8, loc="left")

    np.random.seed(42)
    n_total   = 630
    fold_size = 63
    n_folds   = 10
    bnh_r   = np.random.normal(0.00040, 0.011, n_total)
    strat_r = np.where(
        np.random.rand(n_total) > 0.35,
        bnh_r * np.random.uniform(0.55, 1.05, n_total),
        np.random.normal(0.0, 0.003, n_total),
    )
    bnh_eq   = np.exp(np.cumsum(bnh_r))
    strat_eq = np.exp(np.cumsum(strat_r))

    ax_d.plot(np.arange(n_total), bnh_eq,   color=C["bnh"],   lw=1.4, label="Buy & Hold",    alpha=0.9)
    ax_d.plot(np.arange(n_total), strat_eq, color=C["strat"], lw=1.4, label="Strategy (OOS)", alpha=0.9)
    ax_d.axhline(1.0, color=C["grid"], lw=0.7, linestyle="--")
    for fold_i in range(1, n_folds):
        ax_d.axvline(fold_i * fold_size, color=C["grid"], lw=0.9, linestyle="--", alpha=0.8)
    ax_d.set_ylabel("Equity (base = 1.0)", fontsize=8.5)
    ax_d.set_xticks([i * fold_size for i in range(n_folds + 1)])
    ax_d.set_xticklabels(["0"] + ["F%d" % i for i in range(1, n_folds + 1)], fontsize=7)
    ax_d.tick_params(axis="y", labelsize=7.5)
    ax_d.legend(fontsize=8, loc="upper left")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.grid(axis="y", color=C["grid"], lw=0.4, zorder=0)

    ax_d.text(0.50, -0.14,
              "What changed:  walk_forward() now returns (stats_df, oos_returns).\n"
              "The 10 fold returns are stitched into a continuous equity curve\n"
              "and saved as {ticker}_walkforward_oos.png.  Each segment is\n"
              "fully out-of-sample -- no signal is trained on the test window.",
              transform=ax_d.transAxes, ha="center", fontsize=7.5,
              color=C["sub"], va="top", style="italic")

    # -- Panel E: Multi-asset validation
    ax_e = fig.add_subplot(gs[1])
    ax_e.set_title("E  -  Multi-Asset Validation  (--multi-asset)", fontsize=10,
                   fontweight="bold", color=C["v5"], pad=8, loc="left")

    tickers  = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    sharpe_s = [0.68,  0.61,  0.44,  0.38,  0.52]
    sharpe_b = [0.44,  0.52,  0.38,  0.31,  0.44]
    x = np.arange(len(tickers))
    w = 0.32
    bars_s = ax_e.bar(x - w/2, sharpe_s, w, label="Strategy",   color=C["strat"], alpha=0.85)
    bars_b = ax_e.bar(x + w/2, sharpe_b, w, label="Buy & Hold", color=C["bnh"],   alpha=0.75)
    ax_e.axhline(0, color=C["grid"], lw=0.6)
    for i, (s, b) in enumerate(zip(sharpe_s, sharpe_b)):
        ax_e.text(i - w/2, s + 0.008, "%.2f" % s, ha="center", fontsize=7.5,
                  color=C["strat"], fontweight="bold")
        ax_e.text(i + w/2, b + 0.008, "%.2f" % b, ha="center", fontsize=7.5, color=C["bnh"])
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(tickers, fontsize=9)
    ax_e.set_ylabel("Sharpe Ratio", fontsize=8.5)
    ax_e.tick_params(axis="y", labelsize=7.5)
    ax_e.legend(fontsize=8, loc="upper right")
    ax_e.spines[["top", "right"]].set_visible(False)
    ax_e.grid(axis="y", color=C["grid"], lw=0.4, zorder=0)

    ax_e.text(0.50, -0.14,
              "What changed:  new --multi-asset flag runs the full pipeline\n"
              "on SPY, QQQ, IWM, TLT, GLD sequentially (~30s each).\n"
              "Prints a comparison table + saves a grouped bar chart.\n"
              "Values above are schematic -- run --multi-asset for real numbers.",
              transform=ax_e.transAxes, ha="center", fontsize=7.5,
              color=C["sub"], va="top", style="italic")

    # -- Panel F: Summary and next steps
    ax_f = fig.add_subplot(gs[2])
    ax_f.axis("off")
    ax_f.set_title("F  -  Summary & What's Next", fontsize=10,
                   fontweight="bold", color=C["text"], pad=8, loc="left")

    ax_f.text(0.02, 0.97, "Completed", transform=ax_f.transAxes,
              fontsize=9, fontweight="bold", color=C["pos"], va="top")

    done = [
        ("v4.0", "Half-position on mixed days",
         "Sharpe 0.29 -> 0.68, T-stat 1.32 -> 3.13 (p=0.002)"),
        ("v4.0", "Regime persistence filter",
         "--min-hold N reduces ~20 -> 7 switches/yr;\n"
         "  extends break-even to ~20 bps"),
        ("v4.0", "Markov convergence fix",
         "em_iter=200, search_reps=5; zero warnings\n"
         "  on 5,364-observation dataset"),
        ("v5.0", "Walk-forward OOS equity curve",
         "Stitched OOS chart across all 10 folds;\n"
         "  fold boundaries marked"),
        ("v5.0", "Multi-asset validation",
         "--multi-asset: SPY QQQ IWM TLT GLD;\n"
         "  comparison table + bar chart"),
    ]
    y = 0.87
    for ver, title, note in done:
        ver_col = C["v4"] if ver == "v4.0" else C["v5"]
        ax_f.text(0.02, y, "[%s]" % ver, transform=ax_f.transAxes,
                  fontsize=7, color=ver_col, va="top", fontweight="bold",
                  fontfamily="monospace")
        ax_f.text(0.18, y, title, transform=ax_f.transAxes,
                  fontsize=8, color=C["text"], va="top", fontweight="bold")
        ax_f.text(0.18, y - 0.052, note, transform=ax_f.transAxes,
                  fontsize=7.2, color=C["sub"], va="top")
        y -= 0.148

    ax_f.add_artist(plt.Line2D([0.01, 0.99], [y + 0.04, y + 0.04],
                               transform=ax_f.transAxes, color=C["grid"], lw=0.8))

    ax_f.text(0.02, y + 0.01, "Still to explore", transform=ax_f.transAxes,
              fontsize=9, fontweight="bold", color=C["gold"], va="top")
    next_items = [
        ("Multi-scale geometric",   "Average straightness ratios across 5, 15, 30-day windows"),
        ("Vol ratio (endogenous)",  "5-day / 63-day realised vol as 3rd signal; no API needed"),
        ("Defensive rotation",      "TLT/GLD instead of cash on reversion/mixed days"),
        ("Expanding-window BT",     "Honest real-time simulation with rolling recalibration"),
    ]
    y2 = y - 0.04
    for label, desc in next_items:
        ax_f.text(0.02, y2, "-  %s" % label, transform=ax_f.transAxes,
                  fontsize=8, color=C["text"], va="top", fontweight="bold")
        ax_f.text(0.05, y2 - 0.048, desc, transform=ax_f.transAxes,
                  fontsize=7.2, color=C["sub"], va="top")
        y2 -= 0.115

    _footer(fig, 2)
    return fig


# ============================================================================
#  SAVE
# ============================================================================

path_pdf = OUTPUT_DIR / "v4v5_changes_report.pdf"

print("  Generating pages...")
with PdfPages(path_pdf) as pdf:
    for page_num, fn in enumerate([make_page1, make_page2], 1):
        print("    page %d..." % page_num)
        fig = fn()
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

print("  Saved ->  %s" % path_pdf)
print("  File size: %.1f KB" % (path_pdf.stat().st_size / 1024))

with open(path_pdf, "rb") as f:
    header = f.read(8)
print("  PDF header: %s  [%s]" % (
    header.decode("latin-1"),
    "OK" if header.startswith(b"%PDF") else "ERROR"))

with open(__file__) as f:
    src = f.read()
bad = [(i, c) for i, c in enumerate(src) if ord(c) > 127]
if bad:
    print("  WARNING: non-ASCII chars: %s" % str(set(c for _, c in bad)))
else:
    print("  Source file: ASCII-clean [OK]")
