#!/usr/bin/env python3
"""
Regime Ensemble v4.0 & v5.0 — Changes & Explanations
======================================================
One-page schematic report summarising what changed across v4.0 and v5.0,
why each change was made, and what effect it had.

No Polygon API key required — diagrams are schematic.

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
import numpy as np

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

C = {
    "mom":     "#1a7a4a",
    "rev":     "#c0392b",
    "mix":     "#7f8c8d",
    "strat":   "#2980b9",
    "bnh":     "#2c3e50",
    "pos":     "#27ae60",
    "neg":     "#c0392b",
    "warn":    "#e67e22",
    "gold":    "#f39c12",
    "text":    "#2c3e50",
    "sub":     "#7f8c8d",
    "bg":      "#fafafa",
    "v4":      "#8e44ad",
    "v5":      "#2980b9",
}

# ── Figure ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10), facecolor="white")

# Header
fig.text(0.5, 0.979, "Regime Ensemble  ·  v4.0 & v5.0  ·  Changes & Explanations",
         ha="center", va="top", fontsize=15, fontweight="bold", color=C["text"])
fig.text(0.5, 0.958, "What changed, why it matters, and what each improvement adds",
         ha="center", va="top", fontsize=9, color=C["sub"])

# Divider between v4 and v5 rows
fig.text(0.5, 0.565, "─" * 115, ha="center", va="center", fontsize=7, color="#bdc3c7")
fig.text(0.02, 0.565, "v5.0", ha="left", va="center", fontsize=8,
         fontweight="bold", color=C["v5"],
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C["v5"], linewidth=1.2))
fig.text(0.02, 0.945, "v4.0", ha="left", va="center", fontsize=8,
         fontweight="bold", color=C["v4"],
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C["v4"], linewidth=1.2))

gs_top = gridspec.GridSpec(1, 3, figure=fig, left=0.05, right=0.97,
                           top=0.935, bottom=0.575, wspace=0.32)
gs_bot = gridspec.GridSpec(1, 3, figure=fig, left=0.05, right=0.97,
                           top=0.545, bottom=0.055, wspace=0.32)

# ══════════════════════════════════════════════════════════════════════
#  TOP ROW  —  v4.0
# ══════════════════════════════════════════════════════════════════════

# ── Panel A: Half-position on mixed days ──────────────────────────────────

ax_a = fig.add_subplot(gs_top[0])
ax_a.axis("off")
ax_a.set_title("A  ·  Half-Position on Mixed Days", fontsize=9,
               fontweight="bold", color=C["v4"], pad=5)

# Regime stats table (schematic, real numbers from v4.0 run)
rows = [
    ("Momentum",  "+0.066%/day", "T = 1.32", "p = 0.19", "✗", "→  hold full (+1.0)"),
    ("Mixed",     "+0.056%/day", "T = 3.21", "p = 0.001","✓", "→  hold half (+0.5)"),
    ("Reversion", "+0.005%/day", "T = 0.35", "p = 0.73", "✗", "→  cash  (0.0)"),
]
col_colors = [C["mom"], C["mix"], C["rev"]]
y0 = 0.93
for i, (regime, mean, tstat, pval, sig, action) in enumerate(rows):
    y = y0 - i * 0.20
    ax_a.text(0.02, y, f"●", transform=ax_a.transAxes,
              fontsize=11, color=col_colors[i], va="top")
    ax_a.text(0.10, y, regime, transform=ax_a.transAxes,
              fontsize=8, fontweight="bold", color=C["text"], va="top")
    ax_a.text(0.10, y - 0.065, f"{mean}   {tstat}   {pval}  {sig}",
              transform=ax_a.transAxes, fontsize=7.2, color=C["sub"], va="top",
              fontfamily="monospace")
    ax_a.text(0.10, y - 0.120, action, transform=ax_a.transAxes,
              fontsize=7.5, color=col_colors[i], va="top", style="italic")

# Before/after metrics
ax_a.text(0.5, 0.22,
          "Before v4.0                After v4.0",
          transform=ax_a.transAxes, ha="center", fontsize=7.5,
          fontweight="bold", color=C["text"], va="top")
ax_a.text(0.5, 0.12,
          "Sharpe  0.29   →   0.68\n"
          "T-stat  1.32 (p=0.19)  →  3.13 (p=0.002)\n"
          "Mixed days: cash  →  +0.5 long",
          transform=ax_a.transAxes, ha="center", fontsize=7.5,
          color=C["pos"], va="top", fontfamily="monospace")

ax_a.text(0.5, -0.04,
          "Mixed days have the strongest signal (T=3.21) — changing\n"
          "them from cash to half-long captures that return.",
          transform=ax_a.transAxes, ha="center", fontsize=7.0,
          color=C["sub"], va="top", style="italic")

# ── Panel B: Regime persistence filter ────────────────────────────────────

ax_b = fig.add_subplot(gs_top[1])
ax_b.axis("off")
ax_b.set_title("B  ·  Regime Persistence Filter  (--min-hold N)", fontsize=9,
               fontweight="bold", color=C["v4"], pad=5)

# Signal trace diagram
def draw_signal_row(ax, y, label, signals, colors_map, fontsize=7.5):
    ax.text(0.01, y + 0.01, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", color=C["text"], va="bottom")
    n = len(signals)
    for j, s in enumerate(signals):
        x0 = 0.01 + j * (0.98 / n)
        x1 = 0.01 + (j + 1) * (0.98 / n)
        ax.axhspan(y - 0.09, y, xmin=x0, xmax=x1,
                   facecolor=colors_map[s], alpha=0.75, transform=ax.transAxes)
    return

raw_sig  = ["M","M","R","R","M","R","R","R","M","M","M","M","R","R","M"]
filt_sig = ["M","M","M","M","M","R","R","R","M","M","M","M","M","M","M"]
sm = {"M": C["mom"], "R": C["rev"]}

ax_b.text(0.50, 0.97, "Raw signal  (without filter)", transform=ax_b.transAxes,
          ha="center", fontsize=8, fontweight="bold", color=C["text"], va="top")

n = len(raw_sig)
bar_w = 0.96 / n
for j, s in enumerate(raw_sig):
    x0 = 0.02 + j * bar_w
    rect = plt.Rectangle((x0, 0.82), bar_w * 0.92, 0.10,
                          facecolor=sm[s], alpha=0.8,
                          transform=ax_b.transAxes, clip_on=False)
    ax_b.add_patch(rect)

ax_b.text(0.50, 0.77, "Filtered signal  (--min-hold 3)", transform=ax_b.transAxes,
          ha="center", fontsize=8, fontweight="bold", color=C["text"], va="top")

for j, s in enumerate(filt_sig):
    x0 = 0.02 + j * bar_w
    rect = plt.Rectangle((x0, 0.62), bar_w * 0.92, 0.10,
                          facecolor=sm[s], alpha=0.8,
                          transform=ax_b.transAxes, clip_on=False)
    ax_b.add_patch(rect)

patches = [mpatches.Patch(color=C["mom"], alpha=0.8, label="Momentum"),
           mpatches.Patch(color=C["rev"], alpha=0.8, label="Reversion")]
ax_b.legend(handles=patches, loc="upper right", fontsize=6.5,
            bbox_to_anchor=(1.0, 0.56), framealpha=0.7)

ax_b.text(0.5, 0.53, "→  ~20 switches/year  reduced to ~7/year",
          transform=ax_b.transAxes, ha="center", fontsize=8,
          fontweight="bold", color=C["pos"], va="top")

metrics = [
    ("No filter",     "0bps", "+5.7% CAGR", "0.68 Sharpe", "~20 switches/yr"),
    ("--min-hold 3",  "0bps", "+4.2% CAGR", "0.42 Sharpe", "~7 switches/yr"),
    ("No filter",    "10bps", "+4.3% CAGR", "0.48 Sharpe", "break-even ~15bps"),
    ("--min-hold 3", "10bps", "+3.8% CAGR", "0.38 Sharpe", "break-even ~20bps"),
]
y_start = 0.40
ax_b.text(0.01, y_start + 0.05, "  Setting      Cost   CAGR      Sharpe",
          transform=ax_b.transAxes, fontsize=6.8, color=C["text"],
          fontfamily="monospace", fontweight="bold", va="top")
for i, (setting, cost, cagr, sharpe, note) in enumerate(metrics):
    y = y_start - i * 0.09
    ax_b.text(0.01, y, f"  {setting:<13} {cost:<6} {cagr:<10} {sharpe}",
              transform=ax_b.transAxes, fontsize=6.8, color=C["sub"],
              fontfamily="monospace", va="top")

ax_b.text(0.5, -0.04,
          "Trade-off: fewer switches = lower raw return, but\n"
          "strategy survives higher transaction costs.",
          transform=ax_b.transAxes, ha="center", fontsize=7.0,
          color=C["sub"], va="top", style="italic")

# ── Panel C: Markov convergence fix ──────────────────────────────────────

ax_c = fig.add_subplot(gs_top[2])
ax_c.axis("off")
ax_c.set_title("C  ·  Markov EM Convergence Fix", fontsize=9,
               fontweight="bold", color=C["v4"], pad=5)

ax_c.text(0.5, 0.97, "v3.x  (before fix)  ✗", transform=ax_c.transAxes,
          ha="center", fontsize=8, fontweight="bold", color=C["neg"], va="top")

before_lines = [
    "model.fit(disp=False)",
    "",
    "ConvergenceWarning:",
    "  Maximum Likelihood optimization",
    "  failed to converge. Check",
    "  mle_retvals for further info.",
    "",
    "  → 5-year data: fine",
    "  → 25-year data: warnings on",
    "    most runs. EM needs more",
    "    iterations at this scale.",
]
for i, line in enumerate(before_lines):
    color = C["neg"] if "Warning" in line or "failed" in line or "Check" in line else C["sub"]
    ax_c.text(0.05, 0.88 - i * 0.055, line, transform=ax_c.transAxes,
              fontsize=6.8, color=color, va="top", fontfamily="monospace")

ax_c.text(0.5, 0.32, "v4.0  (after fix)  ✓", transform=ax_c.transAxes,
          ha="center", fontsize=8, fontweight="bold", color=C["pos"], va="top")

after_lines = [
    "model.fit(",
    "  em_iter=200,   # was 5 (default)",
    "  search_reps=5, # 5 random starts",
    ")                # best BIC selected",
    "",
    "  → em_iter=200: enough iterations",
    "    for 5,364-observation dataset",
    "  → search_reps=5: avoids local",
    "    optima in the EM landscape",
    "  → Zero warnings in production",
]
for i, line in enumerate(after_lines):
    color = C["pos"] if "→" in line and "Zero" in line else (
            C["strat"] if line.startswith("  #") or "em_iter" in line or "search_reps" in line else C["sub"])
    ax_c.text(0.05, 0.25 - i * 0.055, line, transform=ax_c.transAxes,
              fontsize=6.8, color=color, va="top", fontfamily="monospace")

ax_c.text(0.5, -0.04,
          "More EM iterations + multiple random starts = reliable\n"
          "convergence on long datasets. Best BIC is kept.",
          transform=ax_c.transAxes, ha="center", fontsize=7.0,
          color=C["sub"], va="top", style="italic")

# ══════════════════════════════════════════════════════════════════════
#  BOTTOM ROW  —  v5.0
# ══════════════════════════════════════════════════════════════════════

# ── Panel D: Walk-forward OOS equity curve ────────────────────────────────

ax_d = fig.add_subplot(gs_bot[0])
ax_d.set_title("D  ·  Walk-Forward OOS Equity Curve  (--walkforward)", fontsize=9,
               fontweight="bold", color=C["v5"], pad=5)

np.random.seed(42)
n_total = 630
fold_size = 63
n_folds = 10

bnh_r   = np.random.normal(0.00035, 0.011, n_total)
strat_r = np.where(
    np.random.rand(n_total) > 0.35,
    bnh_r * np.random.uniform(0.6, 1.1, n_total),
    np.random.normal(0.0, 0.004, n_total),
)

bnh_eq   = np.exp(np.cumsum(bnh_r))
strat_eq = np.exp(np.cumsum(strat_r))

x = np.arange(n_total)
ax_d.plot(x, bnh_eq,   color=C["bnh"],   linewidth=1.3, label="Buy & Hold", alpha=0.9)
ax_d.plot(x, strat_eq, color=C["strat"], linewidth=1.3, label="Strategy (OOS)", alpha=0.9)
ax_d.axhline(1.0, color="#bdc3c7", linewidth=0.5, linestyle="--")

for fold_i in range(1, n_folds):
    ax_d.axvline(fold_i * fold_size, color="#bdc3c7", linewidth=0.7,
                 linestyle="--", alpha=0.7)

ax_d.set_ylabel("Equity (base=1)", fontsize=8)
ax_d.set_xticks([i * fold_size for i in range(n_folds + 1)])
ax_d.set_xticklabels([f"F{i}" if i > 0 else "Start" for i in range(n_folds + 1)],
                     fontsize=6.5)
ax_d.tick_params(axis="y", labelsize=7)
ax_d.legend(fontsize=7.5, loc="upper left")
ax_d.spines[["top", "right"]].set_visible(False)

ax_d.text(0.5, -0.18,
          "Schematic only — actual chart saved as {ticker}_walkforward_oos.png.\n"
          "Each segment is one out-of-sample fold (~63 days). No look-ahead.",
          transform=ax_d.transAxes, ha="center", fontsize=6.8,
          color=C["sub"], va="top", style="italic")

# ── Panel E: Multi-asset validation ──────────────────────────────────────

ax_e = fig.add_subplot(gs_bot[1])
ax_e.set_title("E  ·  Multi-Asset Validation  (--multi-asset)", fontsize=9,
               fontweight="bold", color=C["v5"], pad=5)

tickers   = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
sharpe_s  = [0.68,  0.61,  0.44,  0.38,  0.52]
sharpe_b  = [0.44,  0.52,  0.38,  0.31,  0.44]

x = np.arange(len(tickers))
w = 0.35
ax_e.bar(x - w/2, sharpe_s, w, label="Strategy",   color=C["strat"], alpha=0.85)
ax_e.bar(x + w/2, sharpe_b, w, label="Buy & Hold", color=C["bnh"],   alpha=0.75)
ax_e.axhline(0, color="#bdc3c7", linewidth=0.5)

for i, (s, b) in enumerate(zip(sharpe_s, sharpe_b)):
    ax_e.text(i - w/2, s + 0.01, f"{s:.2f}", ha="center", fontsize=6.5,
              color=C["strat"], fontweight="bold")
    ax_e.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=6.5, color=C["bnh"])

ax_e.set_xticks(x)
ax_e.set_xticklabels(tickers, fontsize=8)
ax_e.set_ylabel("Sharpe Ratio", fontsize=8)
ax_e.tick_params(axis="y", labelsize=7)
ax_e.legend(fontsize=7.5, loc="upper right")
ax_e.spines[["top", "right"]].set_visible(False)

ax_e.text(0.5, -0.18,
          "Schematic values — run python run.py --multi-asset --skip-bic for real numbers.\n"
          "Question: does the Sharpe improvement hold across asset classes?",
          transform=ax_e.transAxes, ha="center", fontsize=6.8,
          color=C["sub"], va="top", style="italic")

# ── Panel F: Summary / what's next ───────────────────────────────────────

ax_f = fig.add_subplot(gs_bot[2])
ax_f.axis("off")
ax_f.set_title("F  ·  Where Things Stand", fontsize=9,
               fontweight="bold", color=C["text"], pad=5)

done = [
    ("v4.0", "Half-position on mixed days",     "Sharpe 0.29 → 0.68, p=0.002"),
    ("v4.0", "Regime persistence filter",        "--min-hold N, ~20 → 7 switches/yr"),
    ("v4.0", "Markov convergence fix",           "em_iter=200, search_reps=5"),
    ("v5.0", "Walk-forward OOS equity curve",    "Stitched chart, fold boundaries"),
    ("v5.0", "Multi-asset validation",           "--multi-asset: SPY QQQ IWM TLT GLD"),
]

next_items = [
    "Multi-scale geometric (windows 5/15/30)",
    "Vol ratio as 3rd signal (endogenous)",
    "Defensive rotation (TLT/GLD vs cash)",
    "Expanding-window honest backtest",
]

ax_f.text(0.02, 0.97, "Completed", transform=ax_f.transAxes,
          fontsize=8, fontweight="bold", color=C["pos"], va="top")
for i, (ver, title, note) in enumerate(done):
    y = 0.88 - i * 0.135
    ver_col = C["v4"] if ver == "v4.0" else C["v5"]
    ax_f.text(0.02, y, f"[{ver}]", transform=ax_f.transAxes,
              fontsize=6.5, color=ver_col, va="top", fontweight="bold",
              fontfamily="monospace")
    ax_f.text(0.16, y, title, transform=ax_f.transAxes,
              fontsize=7.5, color=C["text"], va="top", fontweight="bold")
    ax_f.text(0.16, y - 0.060, note, transform=ax_f.transAxes,
              fontsize=7.0, color=C["sub"], va="top")

ax_f.text(0.02, 0.24, "Still to explore", transform=ax_f.transAxes,
          fontsize=8, fontweight="bold", color=C["gold"], va="top")
for i, item in enumerate(next_items):
    ax_f.text(0.02, 0.15 - i * 0.075, f"·  {item}", transform=ax_f.transAxes,
              fontsize=7.2, color=C["sub"], va="top")

# ── Footer ─────────────────────────────────────────────────────────────────

fig.text(0.5, 0.012,
         "regime_ensemble  ·  github.com/benedictprimmer-web/regime_ensemble  "
         "·  Sharpe values in Panel E are schematic — run --multi-asset for real numbers",
         ha="center", va="bottom", fontsize=6.5, color=C["sub"])

# ── Save ───────────────────────────────────────────────────────────────────

path_pdf = OUTPUT_DIR / "v4v5_changes_report.pdf"
path_png = OUTPUT_DIR / "v4v5_changes_report.png"
fig.savefig(path_pdf, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(path_png, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"  Saved →  {path_pdf}")
print(f"  Saved →  {path_png}")
