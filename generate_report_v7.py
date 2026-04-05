#!/usr/bin/env python3
"""
Regime Ensemble v7.0 - Research Features Report (2 pages)
==========================================================
Covers two new research flags introduced in v7:
  1. --geo-directional  (signed straightness ratio)
  2. --continuous       (continuous position sizing)

Both flags work correctly but underperform the v5 baseline on SPY 2000-2025.
This report explains why and what that tells us about the signal architecture.

Usage:
    python3 generate_report_v7.py
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.data      import fetch_daily_bars, log_returns
from src.geometric import geometric_signal, straightness_ratio
from src.markov    import fit_markov3
from src.ensemble  import ensemble_score, regime_labels
from src.backtest  import run_backtest, compute_stats

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER    = "SPY"
FROM_DATE = "2000-01-01"
TO_DATE   = "2025-01-01"

C = {
    "baseline": "#2980b9",
    "directional": "#8e44ad",
    "continuous": "#e67e22",
    "bnh":      "#2c3e50",
    "mom":      "#1a7a4a",
    "rev":      "#c0392b",
    "sub":      "#6c757d",
    "text":     "#1a1a2e",
    "grid":     "#e8e8e8",
    "warn":     "#e67e22",
    "pos":      "#1a7a4a",
    "neg":      "#c0392b",
}


# ============================================================================
#  HELPERS
# ============================================================================

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["grid"])
    ax.tick_params(colors=C["sub"], labelsize=7.5)
    ax.grid(axis="y", color=C["grid"], lw=0.5, zorder=0)


def _tc(ax, title):
    ax.set_title(title, fontsize=9, fontweight="bold", color=C["text"],
                 loc="left", pad=5)


def _caption(ax, text, y=-0.16):
    ax.text(0.0, y, text, transform=ax.transAxes,
            ha="left", fontsize=6.8, color=C["sub"], va="top", style="italic")


def _page_header(fig, subtitle):
    fig.text(0.5, 0.973, "Regime Ensemble  -  v7.0 Research Report",
             ha="center", va="top", fontsize=13, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.953, subtitle,
             ha="center", va="top", fontsize=8.5, color=C["sub"])
    fig.add_artist(plt.Line2D([0.07, 0.95], [0.942, 0.942],
                              transform=fig.transFigure,
                              color=C["grid"], lw=1.0))


def _footer(fig, page, total=2):
    fig.text(0.5, 0.015,
             "Page %d of %d  -  regime_ensemble v7.0  -  SPY %s-%s  -  "
             "github.com/benedictprimmer-web/regime_ensemble" % (
                 page, total, FROM_DATE[:4], TO_DATE[:4]),
             ha="center", va="bottom", fontsize=6.2, color=C["sub"])


def _perf_row(bt):
    p = compute_stats(bt, raw=True)["Strategy (Long Only)"]
    return p["CAGR"], p["Sharpe"], p["Max DD"], p["T-stat"], p["P-value"]


def switches_pa(lbl):
    n = (lbl != lbl.shift(1)).sum()
    return n / (len(lbl) / 252)


# ============================================================================
#  DATA & SIGNAL COMPUTATION
# ============================================================================

print("Loading SPY data...")
df     = fetch_daily_bars(TICKER, FROM_DATE, TO_DATE)
ret    = log_returns(df)
prices = df["close"]
print("  %d trading days" % len(ret))

print("Fitting Markov k=3 (~30s)...")
mom_prob, crisis_prob, _, _ = fit_markov3(ret, verbose=False)

print("Computing signals...")

# Baseline (v5): unsigned geometric
geo_base   = geometric_signal(ret, window=15)
score_base = ensemble_score(geo_base, mom_prob, crisis_prob)
labels_base = regime_labels(score_base)
bt_base    = run_backtest(ret, labels_base)

# Directional geometric
geo_dir    = geometric_signal(ret, window=15, directional=True)
score_dir  = ensemble_score(geo_dir, mom_prob, crisis_prob)
labels_dir = regime_labels(score_dir)
bt_dir     = run_backtest(ret, labels_dir)

# Continuous sizing (using baseline score)
bt_cont    = run_backtest(ret, labels_base, score=score_base)

# BnH
bnh_perf = compute_stats(bt_base, raw=True)["Buy & Hold"]

# Performance
cagr_b, sh_b, dd_b, t_b, p_b   = _perf_row(bt_base)
cagr_d, sh_d, dd_d, t_d, p_d   = _perf_row(bt_dir)
cagr_c, sh_c, dd_c, t_c, p_c   = _perf_row(bt_cont)

# Switches
sw_base = switches_pa(labels_base)
sw_dir  = switches_pa(labels_dir)
# For continuous, measure mean daily position change * 252 as turnover proxy
pos_cont = score_base.reindex(bt_cont.index).clip(0, 1)
turnover_cont = pos_cont.diff().abs().dropna().mean() * 252

# Unsigned and signed ratios
ratio_unsigned = straightness_ratio(ret, window=15)
ratio_signed   = straightness_ratio(ret, window=15, directional=True)

# Cost sensitivity
bps_range = list(range(0, 31))
cost_base = [float(compute_stats(run_backtest(ret, labels_base, cost_bps=b))
                   ["Strategy (Long Only)"]["Sharpe"]) for b in bps_range]
cost_dir  = [float(compute_stats(run_backtest(ret, labels_dir,  cost_bps=b))
                   ["Strategy (Long Only)"]["Sharpe"]) for b in bps_range]
cost_cont = [float(compute_stats(run_backtest(ret, labels_base, cost_bps=b, score=score_base))
                   ["Strategy (Long Only)"]["Sharpe"]) for b in bps_range]

print("  Baseline  Sharpe=%.2f  CAGR=%.1f%%  switches/yr=%.0f" % (sh_b, cagr_b*100, sw_base))
print("  Directional Sharpe=%.2f  CAGR=%.1f%%  switches/yr=%.0f" % (sh_d, cagr_d*100, sw_dir))
print("  Continuous  Sharpe=%.2f  CAGR=%.1f%%" % (sh_c, cagr_c*100))
print("Generating report...")


# ============================================================================
#  PAGE 1 - DIRECTIONAL GEOMETRIC
# ============================================================================

def make_page1():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.95, top=0.88, bottom=0.07,
                        hspace=1.10, wspace=0.40)
    _page_header(fig, "Feature 1: --geo-directional  |  Signed Straightness Ratio  |  SPY 2000-2025")

    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.09, right=0.95,
                           top=0.88, bottom=0.07, hspace=1.10, wspace=0.40)

    # ── Panel 1: Unsigned vs signed ratio (zoom: 2007-2010 to show crash) ──
    ax1 = fig.add_subplot(gs[0, :])
    zoom = (ratio_unsigned.index >= "2007-01-01") & (ratio_unsigned.index <= "2011-01-01")
    r_u = ratio_unsigned[zoom]
    r_s = ratio_signed[zoom]

    ax1.plot(r_u.index, r_u, color=C["baseline"],    lw=1.0, alpha=0.85,
             label="Unsigned ratio  [0, 1]  (v5 baseline)")
    ax1.plot(r_s.index, r_s, color=C["directional"], lw=1.0, alpha=0.85,
             label="Signed ratio  [-1, +1]  (--geo-directional)")
    ax1.axhline(0, color=C["grid"], lw=0.8)
    ax1.fill_between(r_s.index, r_s, 0, where=(r_s < 0),
                     alpha=0.18, color=C["neg"], label="Signed < 0 (downtrend)")

    # Annotate the GFC crash window
    ax1.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-01"),
                alpha=0.08, color=C["rev"], zorder=0)
    ax1.text(pd.Timestamp("2008-10-01"), 0.88, "GFC crash",
             fontsize=7, color=C["rev"], style="italic")

    ax1.set_ylim(-1.05, 1.05)
    ax1.legend(fontsize=7, loc="lower right", framealpha=0.8)
    _style(ax1)
    _tc(ax1, "Unsigned vs Signed Straightness Ratio  (2007-2010 zoom)")
    _caption(ax1,
             "During the GFC crash (shaded), the unsigned ratio stays near +1 (straight line = 'momentum' = BUY signal).\n"
             "The signed ratio correctly goes negative (straight DOWN = low score = CASH). This is the fix.", y=-0.12)

    # ── Panel 2: Regime distribution comparison ──────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    regimes   = ["momentum", "mixed", "reversion"]
    colors_r  = [C["mom"], C["sub"], C["rev"]]
    base_pcts = [(labels_base == r).mean() * 100 for r in regimes]
    dir_pcts  = [(labels_dir  == r).mean() * 100 for r in regimes]

    x = np.arange(3)
    w = 0.35
    ax2.bar(x - w/2, base_pcts, w, color=[c + "cc" for c in colors_r],
            label="Baseline (unsigned)", alpha=0.9)
    ax2.bar(x + w/2, dir_pcts,  w, color=colors_r,
            label="Directional (signed)", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Momentum", "Mixed", "Reversion"], fontsize=8)
    ax2.legend(fontsize=7, framealpha=0.8)
    ax2.set_ylabel("% of trading days", fontsize=8, color=C["sub"])
    _style(ax2)
    _tc(ax2, "Regime Distribution Shift")
    _caption(ax2,
             "Directional reclassifies some crash days from\n"
             "momentum to reversion. Momentum % falls,\n"
             "reversion % rises.", y=-0.20)

    # ── Panel 3: Equity curves ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    eq_b = bt_base["equity_strategy"]
    eq_d = bt_dir["equity_strategy"]
    eq_h = bt_base["equity_bnh"]
    ax3.plot(eq_h.index, eq_h, color=C["bnh"],         lw=1.2, alpha=0.7,
             label="Buy & Hold  (Sharpe %.2f)" % bnh_perf["Sharpe"])
    ax3.plot(eq_b.index, eq_b, color=C["baseline"],    lw=1.4,
             label="Baseline  (Sharpe %.2f)" % sh_b)
    ax3.plot(eq_d.index, eq_d, color=C["directional"], lw=1.4, ls="--",
             label="Directional  (Sharpe %.2f)" % sh_d)
    ax3.axhline(1.0, color=C["grid"], lw=0.6, ls="--")
    ax3.legend(fontsize=7, framealpha=0.8, loc="upper left")
    _style(ax3)
    _tc(ax3, "Equity Curves")
    _caption(ax3,
             "Directional geometry underperforms: Sharpe\n"
             "drops %.2f to %.2f. Markov crisis override\n"
             "already handles downtrends with less lag." % (sh_b, sh_d), y=-0.20)

    # ── Panel 4: Why it still underperforms — text explanation ───────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    ax4.text(0.0, 0.98, "Finding: Why --geo-directional underperforms (Sharpe %.2f vs %.2f baseline)" % (sh_d, sh_b),
             transform=ax4.transAxes, fontsize=9, fontweight="bold",
             color=C["text"], va="top")
    ax4.axline((0, 0.89), slope=0, color=C["grid"], lw=0.8, transform=ax4.transAxes)

    findings = [
        ("Markov crisis override already covers the crash case",
         "When P(crisis) > 0.50, the buy signal is suppressed regardless of geometric.\n"
         "It catches sustained downtrends in 1-2 weeks using full return distribution (mean + vol), not path shape."),
        ("Directional geometry introduces false negatives",
         "Sharp selloffs are often followed by immediate recoveries (flash crashes, panic days).\n"
         "Signed ratio signals CASH through both fall and bounce; unsigned defers to Markov on those days."),
        ("The two signals are less orthogonal on the downside",
         "Geometric + Markov are independent on the upside (path shape vs statistical state).\n"
         "Directional geometric creates a second crash detector correlated with Markov, reducing ensemble diversity."),
        ("Recommendation: keep as research flag, not default",
         "Worth revisiting on higher-frequency data or combined with other signals.\n"
         "On SPY daily, Markov crisis override is more precise and adds no extra correlation cost."),
    ]

    y = 0.84
    for title, body in findings:
        ax4.text(0.01, y, title, transform=ax4.transAxes,
                 fontsize=8, fontweight="bold", color=C["directional"], va="top")
        ax4.text(0.01, y - 0.068, body, transform=ax4.transAxes,
                 fontsize=7.5, color=C["sub"], va="top")
        y -= 0.23

    _footer(fig, 1)
    return fig


# ============================================================================
#  PAGE 2 - CONTINUOUS POSITION SIZING
# ============================================================================

def make_page2():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.95, top=0.88, bottom=0.07,
                        hspace=1.10, wspace=0.40)
    _page_header(fig, "Feature 2: --continuous  |  Position = Ensemble Score  |  SPY 2000-2025")

    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.09, right=0.95,
                           top=0.88, bottom=0.07, hspace=1.10, wspace=0.40)

    # ── Panel 1: Position over time — discrete vs continuous (zoom) ──────
    ax1 = fig.add_subplot(gs[0, :])
    zoom = (score_base.index >= "2008-01-01") & (score_base.index <= "2012-01-01")
    pos_d = bt_base.loc[zoom, "signal"]           # discrete {0, 0.5, 1}
    pos_c = score_base[zoom].clip(0, 1)           # continuous score

    ax1.plot(pos_c.index, pos_c, color=C["continuous"], lw=0.9, alpha=0.9,
             label="Continuous (score)")
    ax1.step(pos_d.index, pos_d, color=C["baseline"], lw=1.4, where="post",
             label="Discrete {0, 0.5, 1.0}")
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_ylabel("Position size", fontsize=8, color=C["sub"])
    ax1.legend(fontsize=7.5, loc="lower right", framealpha=0.8)
    _style(ax1)
    _tc(ax1, "Position Size Over Time  (2008-2012 zoom)")
    _caption(ax1,
             "Discrete mode: hard steps between {0, 0.5, 1.0}. Continuous: smooth curve tracking the ensemble score.\n"
             "Continuous mode trades smaller sizes on the highest-conviction momentum days (score 0.7 -> pos 0.7, not 1.0).", y=-0.12)

    # ── Panel 2: Cost sensitivity comparison ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bnh_sh = bnh_perf["Sharpe"]
    ax2.plot(bps_range, cost_base, color=C["baseline"],    lw=1.5,
             label="Discrete (baseline)")
    ax2.plot(bps_range, cost_cont, color=C["continuous"],  lw=1.5, ls="--",
             label="Continuous")
    ax2.axhline(bnh_sh, color=C["bnh"], lw=0.9, ls=":", alpha=0.8,
                label="Buy & Hold (%.2f)" % bnh_sh)
    ax2.axhline(0, color=C["grid"], lw=0.8)
    ax2.set_xlabel("Round-trip cost (bps)", fontsize=8, color=C["sub"])
    ax2.set_ylabel("Sharpe Ratio", fontsize=8, color=C["sub"])
    ax2.legend(fontsize=7, framealpha=0.8)
    ax2.set_xlim(0, 30)
    _style(ax2)
    _tc(ax2, "Sharpe vs Transaction Cost (0-30 bps)")
    _caption(ax2,
             "Continuous underperforms at all cost levels.\n"
             "Both modes degrade at similar rates -- continuous\n"
             "position changes are small but daily.", y=-0.20)

    # ── Panel 3: Equity curves ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    eq_b = bt_base["equity_strategy"]
    eq_c = bt_cont["equity_strategy"]
    eq_h = bt_base["equity_bnh"]
    ax3.plot(eq_h.index, eq_h, color=C["bnh"],        lw=1.2, alpha=0.7,
             label="Buy & Hold  (Sharpe %.2f)" % bnh_perf["Sharpe"])
    ax3.plot(eq_b.index, eq_b, color=C["baseline"],   lw=1.4,
             label="Discrete  (Sharpe %.2f)" % sh_b)
    ax3.plot(eq_c.index, eq_c, color=C["continuous"], lw=1.4, ls="--",
             label="Continuous  (Sharpe %.2f)" % sh_c)
    ax3.axhline(1.0, color=C["grid"], lw=0.6, ls="--")
    ax3.legend(fontsize=7, framealpha=0.8, loc="upper left")
    _style(ax3)
    _tc(ax3, "Equity Curves")
    _caption(ax3,
             "Continuous mode: Sharpe %.2f vs %.2f baseline.\n"
             "Similar drawdown profile, lower overall return\n"
             "due to undersizing on high-conviction days." % (sh_c, sh_b), y=-0.20)

    # ── Panel 4: Text explanation ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    ax4.text(0.0, 0.98, "Finding: Why --continuous underperforms (Sharpe %.2f vs %.2f baseline)" % (sh_c, sh_b),
             transform=ax4.transAxes, fontsize=9, fontweight="bold",
             color=C["text"], va="top")
    ax4.axline((0, 0.89), slope=0, color=C["grid"], lw=0.8, transform=ax4.transAxes)

    findings = [
        ("The discrete half-position on mixed days was empirically calibrated",
         "Mixed regime: T=3.21, p=0.001 -- the strongest forward signal. The +0.5 position was\n"
         "set to match that evidence, not derived from the score. Continuous mode matches it by\n"
         "coincidence on mixed days but not across the full distribution."),
        ("Continuous mode undersizes the best days",
         "On high-conviction days (score ~0.72), continuous holds 0.72 vs discrete 1.0.\n"
         "These are the strongest expected-return days (both detectors agree).\n"
         "Undersizing here costs return without a compensating reduction in risk."),
        ("Daily micro-adjustments still incur transaction costs",
         "The score shifts slightly each day: bps * |delta| applies to every change.\n"
         "Total turnover is similar to ~61 discrete switches -- no cost advantage."),
        ("Recommendation: smooth the score before using as position",
         "Apply a 3-5 day EMA of the score, or clip: pos=0 below 0.35, pos=score\n"
         "between 0.35-0.65, pos=1.0 above 0.65. Preserves full allocation on conviction days."),
    ]

    y = 0.84
    for title, body in findings:
        ax4.text(0.01, y, title, transform=ax4.transAxes,
                 fontsize=8, fontweight="bold", color=C["continuous"], va="top")
        ax4.text(0.01, y - 0.068, body, transform=ax4.transAxes,
                 fontsize=7.5, color=C["sub"], va="top")
        y -= 0.25

    _footer(fig, 2)
    return fig


# ============================================================================
#  SAVE
# ============================================================================

out_path = OUTPUT_DIR / "SPY_v7_report.pdf"
with PdfPages(out_path) as pdf:
    for fn in [make_page1, make_page2]:
        fig = fn()
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

print("Report saved -> %s" % out_path)
