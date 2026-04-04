#!/usr/bin/env python3
"""
Regime Ensemble v6.0 - Technical Quant Report (3 pages)
========================================================
A standalone technical document covering the three v6.0 additions:
  1. Vol ratio dampening    (--vol-signal)
  2. Multi-scale geometric  (--multi-scale)
  3. Expanding-window BT    (--expanding)

All charts use real SPY 2000-2025 data from cache.
The expanding-window computation takes ~5-10 minutes.

Usage:
    python3 generate_report_v6.py
"""

import sys
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from src.data      import fetch_daily_bars, log_returns
from src.geometric import (geometric_signal, straightness_ratio,
                           multi_scale_ratio, MULTI_WINDOWS)
from src.markov    import fit_markov3
from src.ensemble  import ensemble_score, regime_labels, vol_ratio
from src.backtest  import run_backtest, compute_stats, regime_return_stats
from src.expanding import expanding_backtest

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKER    = "SPY"
FROM_DATE = "2000-01-01"
TO_DATE   = "2025-01-01"

C = {
    "mom":     "#1a7a4a",
    "rev":     "#c0392b",
    "mix":     "#7f8c8d",
    "strat":   "#2980b9",
    "bnh":     "#2c3e50",
    "v6":      "#8e44ad",
    "pos":     "#27ae60",
    "neg":     "#c0392b",
    "warn":    "#e67e22",
    "text":    "#1a1a2e",
    "sub":     "#6c757d",
    "grid":    "#e8e8e8",
    "bg":      "#ffffff",
}


# ============================================================================
#  HELPERS
# ============================================================================

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["grid"])
    ax.tick_params(colors=C["sub"], labelsize=7.5)
    ax.yaxis.label.set_color(C["sub"])
    ax.xaxis.label.set_color(C["sub"])
    ax.grid(axis="y", color=C["grid"], lw=0.5, zorder=0)


def _tc(ax, title, caption=None):
    """Set left-aligned title; place caption below x-axis if given."""
    ax.set_title(title, fontsize=9, fontweight="bold", color=C["text"],
                 loc="left", pad=5)
    if caption:
        ax.text(0.5, -0.17, caption, transform=ax.transAxes,
                ha="center", fontsize=6.8, color=C["sub"],
                va="top", style="italic", wrap=True)


def _page_header(fig, subtitle):
    fig.text(0.5, 0.973, "Regime Ensemble  -  v6.0 Technical Report",
             ha="center", va="top", fontsize=13, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.953, subtitle,
             ha="center", va="top", fontsize=8.5, color=C["sub"])
    fig.add_artist(plt.Line2D([0.07, 0.95], [0.942, 0.942],
                              transform=fig.transFigure,
                              color=C["grid"], lw=1.0))


def _footer(fig, page, total=3):
    fig.text(0.5, 0.015,
             "Page %d of %d  -  regime_ensemble v6.0  -  SPY %s-%s  -  "
             "github.com/benedictprimmer-web/regime_ensemble" % (
                 page, total, FROM_DATE[:4], TO_DATE[:4]),
             ha="center", va="bottom", fontsize=6.2, color=C["sub"])


def _regime_dist(labels):
    return {r: (labels == r).mean() * 100
            for r in ["momentum", "mixed", "reversion"]}


def _perf_row(bt):
    p = compute_stats(bt, raw=True)
    s = p["Strategy (Long Only)"]
    return s["CAGR"], s["Sharpe"], s["Max DD"], s["T-stat"], s["P-value"]


# ============================================================================
#  DATA LOAD AND SIGNAL COMPUTATION
# ============================================================================

print("Loading SPY data...")
df     = fetch_daily_bars(TICKER, FROM_DATE, TO_DATE)
ret    = log_returns(df)
prices = df["close"]
print("  %d trading days" % len(ret))

print("Computing vol ratio...")
vr = vol_ratio(ret)

print("Computing geometric signals...")
ratio_5d  = straightness_ratio(ret, window=5)
ratio_15d = straightness_ratio(ret, window=15)
ratio_30d = straightness_ratio(ret, window=30)
ratio_ms  = multi_scale_ratio(ret, windows=MULTI_WINDOWS)

geo_single = geometric_signal(ret, window=15)
geo_multi  = geometric_signal(ret, windows=MULTI_WINDOWS)

print("Fitting Markov k=3 (~30s)...")
mom_prob, crisis_prob, choppy_prob, trans_info = fit_markov3(ret, verbose=False)

print("Computing ensemble variants...")
# v5 baseline: single-scale geometric, no vol dampening
score_v5    = ensemble_score(geo_single, mom_prob, crisis_prob)
labels_v5   = regime_labels(score_v5)
bt_v5       = run_backtest(ret, labels_v5)

# v6 multi-scale only
score_v6ms  = ensemble_score(geo_multi, mom_prob, crisis_prob)
labels_v6ms = regime_labels(score_v6ms)
bt_v6ms     = run_backtest(ret, labels_v6ms)

# v6 full: multi-scale + vol dampening
score_v6    = ensemble_score(geo_multi, mom_prob, crisis_prob, vol_ratio_series=vr)
labels_v6   = regime_labels(score_v6)
bt_v6       = run_backtest(ret, labels_v6)

# Performance numbers
cagr_v5,  sh_v5,  dd_v5,  t_v5,  p_v5  = _perf_row(bt_v5)
cagr_v6ms, sh_v6ms, dd_v6ms, t_v6ms, p_v6ms = _perf_row(bt_v6ms)
cagr_v6,  sh_v6,  dd_v6,  t_v6,  p_v6  = _perf_row(bt_v6)
bnh_perf = compute_stats(bt_v6, raw=True)["Buy & Hold"]

# Vol ratio stats
vr_clean    = vr.dropna()
vr_pct_damp = (vr_clean > 1.5).mean() * 100
vr_pct_full = (vr_clean >= 2.0).mean() * 100
vr_mean     = vr_clean.mean()

# Regime distributions
dist_v5  = _regime_dist(labels_v5)
dist_v6ms = _regime_dist(labels_v6ms)
dist_v6  = _regime_dist(labels_v6)

# Forward return stats for v6
fwd_ret    = ret.shift(-1).rename("log_return")
aligned_v6 = ret.reindex(labels_v6.index)
frd_v6     = regime_return_stats(fwd_ret, labels_v6)

# Cost sensitivity
bps_range  = np.arange(0, 31, 1)
cost_v5    = []
cost_v6    = []
for bps in bps_range:
    cost_v5.append(float(compute_stats(
        run_backtest(ret, labels_v5, cost_bps=int(bps)))["Strategy (Long Only)"]["Sharpe"]))
    cost_v6.append(float(compute_stats(
        run_backtest(ret, labels_v6, cost_bps=int(bps)))["Strategy (Long Only)"]["Sharpe"]))

# Number of regime switches per year
def switches_pa(lbl):
    n = (lbl != lbl.shift(1)).sum()
    return n / (len(lbl) / 252)

sw_v5  = switches_pa(labels_v5)
sw_v6ms = switches_pa(labels_v6ms)
sw_v6  = switches_pa(labels_v6)

print("  v5  Sharpe=%.2f  CAGR=%.1f%%  MaxDD=%.1f%%  switches/yr=%.0f" % (
    sh_v5, cagr_v5*100, dd_v5*100, sw_v5))
print("  v6ms Sharpe=%.2f  CAGR=%.1f%%  MaxDD=%.1f%%  switches/yr=%.0f" % (
    sh_v6ms, cagr_v6ms*100, dd_v6ms*100, sw_v6ms))
print("  v6  Sharpe=%.2f  CAGR=%.1f%%  MaxDD=%.1f%%  switches/yr=%.0f" % (
    sh_v6, cagr_v6*100, dd_v6*100, sw_v6))

print("Running expanding-window backtest (~5-10 mins)...")
exp_bt  = expanding_backtest(ret, multi_scale=True, verbose=True)
exp_perf = compute_stats(exp_bt, raw=True)
exp_s   = exp_perf["Strategy (Long Only)"]
exp_b   = exp_perf["Buy & Hold"]
exp_sharpe = exp_s["Sharpe"]
optimism   = sh_v6 - exp_sharpe
print("  Expanding Sharpe=%.2f  Optimism bias=%.2f" % (exp_sharpe, optimism))


# ============================================================================
#  PAGE 1 - NEW SIGNALS
# ============================================================================

def make_page1():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    _page_header(fig, "Page 1 of 3  -  New Signals: Vol Ratio Dampening & Multi-Scale Geometric")

    # Two rows: top = full-width vol ratio; bottom = 2 panels
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.09, right=0.95,
                           top=0.880, bottom=0.13,
                           height_ratios=[1.1, 1.0],
                           hspace=0.72, wspace=0.38)

    # -- Panel 1: Vol ratio time series --------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    vr_plot = vr.dropna()

    # Shade turbulence zones (ratio > 1.5)
    ax1.fill_between(vr_plot.index, 1.5, vr_plot,
                     where=(vr_plot >= 1.5),
                     color=C["rev"], alpha=0.25,
                     label="Partial dampening (ratio 1.5-2.0)")
    ax1.fill_between(vr_plot.index, 2.0, vr_plot,
                     where=(vr_plot >= 2.0),
                     color=C["rev"], alpha=0.45,
                     label="Full suppression (ratio >= 2.0)")
    ax1.plot(vr_plot.index, vr_plot, color=C["sub"], lw=0.7, alpha=0.8)
    ax1.axhline(1.0, color=C["grid"],    lw=1.0, ls="--", label="Neutral (ratio = 1.0)")
    ax1.axhline(1.5, color=C["warn"],    lw=0.9, ls=":",  alpha=0.7)
    ax1.axhline(2.0, color=C["rev"],     lw=0.9, ls=":",  alpha=0.7,
                label="Full suppress threshold (2.0)")
    ax1.set_ylabel("Vol ratio (5d ann. / 63d ann.)", fontsize=8)
    ax1.set_ylim(0, min(vr_plot.quantile(0.995), 3.5))
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.legend(fontsize=7, loc="upper right", framealpha=0.85)
    _style(ax1)
    _tc(ax1, "Realised Vol Ratio: 5-Day / 63-Day Annualised  (SPY 2000-2025)",
        "Active dampening on %.1f%% of trading days (ratio > 1.5).  Full suppression on %.1f%% "
        "(ratio >= 2.0).  Mean ratio = %.2f.  Orthogonal to Markov crisis override." % (
            vr_pct_damp, vr_pct_full, vr_mean))
    fig.autofmt_xdate(rotation=25, ha="right")

    # -- Panel 2: Multi-scale geometric - 3 windows + average ---------------
    ax2 = fig.add_subplot(gs[1, 0])
    # Compute 2-year rolling mean for readability
    w = 126
    ax2.plot(ratio_5d.rolling(w).mean().dropna().index,
             ratio_5d.rolling(w).mean().dropna(),
             color="#3498db", lw=1.0, alpha=0.75, label="5-day (126d MA)")
    ax2.plot(ratio_15d.rolling(w).mean().dropna().index,
             ratio_15d.rolling(w).mean().dropna(),
             color="#e67e22", lw=1.0, alpha=0.75, label="15-day (126d MA)")
    ax2.plot(ratio_30d.rolling(w).mean().dropna().index,
             ratio_30d.rolling(w).mean().dropna(),
             color="#9b59b6", lw=1.0, alpha=0.75, label="30-day (126d MA)")
    ax2.plot(ratio_ms.rolling(w).mean().dropna().index,
             ratio_ms.rolling(w).mean().dropna(),
             color=C["text"], lw=1.8, label="Average (multi-scale)")
    ax2.set_ylabel("Straightness ratio", fontsize=8)
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.legend(fontsize=7, loc="lower right", framealpha=0.85)
    _style(ax2)
    _tc(ax2, "Multi-Scale Geometric: Window Comparison",
        "126-day MA shown for clarity. 5-day window is most reactive; "
        "30-day captures sustained trends. Averaging reduces single-window noise.")
    fig.autofmt_xdate(rotation=25, ha="right")

    # -- Panel 3: Regime distribution comparison -----------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    regimes = ["momentum", "mixed", "reversion"]
    labels_names = ["v5 (single-scale)", "v6 multi-scale", "v6 full (+vol)"]
    dists = [dist_v5, dist_v6ms, dist_v6]
    colors_bar = [C["mom"], C["mix"], C["rev"]]
    x = np.arange(len(regimes))
    w_bar = 0.22

    for i, (dist, lname) in enumerate(zip(dists, labels_names)):
        vals = [dist[r] for r in regimes]
        bars = ax3.bar(x + (i - 1) * w_bar, vals, w_bar,
                       label=lname, alpha=0.82)
        # Colour bars by regime
        for bar, col in zip(bars, colors_bar):
            bar.set_facecolor(col)
            bar.set_alpha(0.45 + i * 0.15)

    ax3.set_xticks(x)
    ax3.set_xticklabels(["Momentum", "Mixed", "Reversion"], fontsize=8)
    ax3.set_ylabel("% of trading days", fontsize=8)
    ax3.tick_params(axis="y", labelsize=7.5)
    ax3.legend(fontsize=7, loc="upper right", framealpha=0.85)
    _style(ax3)
    _tc(ax3, "Regime Distribution: v5 vs v6 Variants",
        "Vol dampening reduces effective momentum allocation on turbulent days. "
        "Multi-scale geometric produces near-identical distribution to single-scale "
        "(signal shape is preserved; noise is reduced).")

    _footer(fig, 1)
    return fig


# ============================================================================
#  PAGE 2 - PERFORMANCE ANALYSIS
# ============================================================================

def make_page2():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    _page_header(fig, "Page 2 of 3  -  Performance Analysis: v5 Baseline vs v6 Variants")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.10, right=0.95,
                           top=0.880, bottom=0.06,
                           height_ratios=[1.4, 1.1, 0.9],
                           hspace=1.05, wspace=0.40)

    # -- Panel 1: Equity curves (full width) ---------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(bt_v5.index, bt_v5["equity_bnh"],
             color=C["bnh"], lw=1.5, alpha=0.9,
             label="Buy & Hold  (Sharpe %.2f)" % bnh_perf["Sharpe"])
    ax1.plot(bt_v5.index, bt_v5["equity_strategy"],
             color=C["strat"], lw=1.3, ls="--", alpha=0.85,
             label="v5 baseline  (Sharpe %.2f, CAGR %+.1f%%)" % (sh_v5, cagr_v5*100))
    ax1.plot(bt_v6ms.index, bt_v6ms["equity_strategy"],
             color=C["warn"], lw=1.3, ls="-.", alpha=0.85,
             label="v6 multi-scale only  (Sharpe %.2f)" % sh_v6ms)
    ax1.plot(bt_v6.index, bt_v6["equity_strategy"],
             color=C["v6"], lw=1.8,
             label="v6 full (+vol ratio)  (Sharpe %.2f, CAGR %+.1f%%)" % (sh_v6, cagr_v6*100))
    ax1.axhline(1.0, color=C["grid"], lw=0.7, ls="--")
    ax1.set_ylabel("Portfolio value (base = 1.0)", fontsize=8)
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.88)
    _style(ax1)
    _tc(ax1, "Equity Curves: v5 Baseline vs v6 Variants  (SPY 2000-2025, 0 bps, 1-day lag)")
    fig.autofmt_xdate(rotation=25, ha="right")

    # -- Panel 2: Forward returns by regime (v6 full) -------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    regime_order = ["momentum", "mixed", "reversion"]
    cols_bar = [C["mom"], C["mix"], C["rev"]]
    means, cis, ns, tstats, pvals = [], [], [], [], []
    fwd_aligned = ret.shift(-1).rename("log_return")
    labels_v6_fwd = labels_v6.reindex(fwd_aligned.dropna().index)
    for regime in regime_order:
        r  = fwd_aligned.dropna()[labels_v6_fwd == regime].dropna()
        n  = len(r)
        m  = r.mean() * 100
        se = r.sem() * 100
        ci = stats.t.ppf(0.975, df=n - 1) * se if n > 1 else 0
        t, p = stats.ttest_1samp(r, 0) if n > 5 else (np.nan, np.nan)
        means.append(m); cis.append(ci); ns.append(n)
        tstats.append(t); pvals.append(p)

    ax2.bar(regime_order, means, color=cols_bar, alpha=0.82, width=0.5, zorder=3)
    ax2.errorbar(regime_order, means, yerr=cis, fmt="none",
                 color=C["text"], capsize=5, lw=1.5, zorder=4)
    ax2.axhline(0, color=C["sub"], lw=0.8)
    for i, (m, ci, n, t, p) in enumerate(zip(means, cis, ns, tstats, pvals)):
        if not np.isnan(t):
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
            offset = (ci + 0.003) * (1 if m >= 0 else -1)
            ax2.text(i, m + offset,
                     "t=%.2f\np=%.3f %s" % (t, p, sig),
                     ha="center", va="bottom" if m >= 0 else "top",
                     fontsize=6.5, color=C["text"])
        ax2.text(i, -0.050, "n=%d" % n,
                 ha="center", fontsize=6.5, color=C["sub"])
    ax2.set_ylim(-0.09, max(max(means) + max(cis) + 0.06, 0.20))
    ax2.set_ylabel("Mean next-day return (%)", fontsize=8)
    _style(ax2)
    _tc(ax2, "Next-Day Returns by Regime (v6 Full)",
        "Error bars: 95% CI.  ** p<0.05  * p<0.10  ns = not significant.\n"
        "Vol dampening preserves regime signal integrity.")

    # -- Panel 3: Cost sensitivity v5 vs v6 ----------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    be_v5 = int(bps_range[np.argmin(np.abs(cost_v5))])
    be_v6 = int(bps_range[np.argmin(np.abs(cost_v6))])
    ax3.plot(bps_range, cost_v5, color=C["strat"], lw=1.8, ls="--",
             label="v5 baseline  (break-even ~%d bps)" % be_v5)
    ax3.plot(bps_range, cost_v6, color=C["v6"],   lw=1.8,
             label="v6 full       (break-even ~%d bps)" % be_v6)
    ax3.fill_between(bps_range, cost_v6, cost_v5,
                     where=(np.array(cost_v6) >= np.array(cost_v5)),
                     color=C["pos"], alpha=0.12, label="v6 advantage")
    ax3.fill_between(bps_range, cost_v6, cost_v5,
                     where=(np.array(cost_v6) < np.array(cost_v5)),
                     color=C["rev"], alpha=0.12, label="v6 disadvantage")
    ax3.axhline(0, color=C["sub"], lw=0.8)
    ax3.axhline(bnh_perf["Sharpe"], color=C["bnh"], lw=0.9, ls=":",
                label="B&H Sharpe (%.2f)" % bnh_perf["Sharpe"])
    ax3.set_xlabel("Round-trip transaction cost (bps)", fontsize=8)
    ax3.set_ylabel("Strategy Sharpe Ratio", fontsize=8)
    ax3.legend(fontsize=6.8, loc="upper right", framealpha=0.88)
    _style(ax3)
    _tc(ax3, "Sharpe vs Transaction Cost (0-30 bps)")
    ax3.text(0.5, -0.25, "v5: ~%.0f switches/yr.  v6: ~%.0f switches/yr.  "
             "Fewer switches extend the profitable cost range." % (sw_v5, sw_v6),
             transform=ax3.transAxes, ha="center", fontsize=6.8,
             color=C["sub"], va="top", style="italic")

    # -- Panel 4: Performance attribution table (full width) -----------------
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    _tc(ax4, "Performance Attribution  (SPY 2000-2025, 0 bps, 1-day lag)")

    rows_data = [
        ("CAGR",      "%+.1f%%" % (bnh_perf["CAGR"]*100),
                      "%+.1f%%" % (cagr_v5*100),
                      "%+.1f%%" % (cagr_v6ms*100),
                      "%+.1f%%" % (cagr_v6*100)),
        ("Sharpe",    "%.2f" % bnh_perf["Sharpe"],
                      "%.2f" % sh_v5,
                      "%.2f" % sh_v6ms,
                      "%.2f" % sh_v6),
        ("Max DD",    "%.1f%%" % (bnh_perf["Max DD"]*100),
                      "%.1f%%" % (dd_v5*100),
                      "%.1f%%" % (dd_v6ms*100),
                      "%.1f%%" % (dd_v6*100)),
        ("T-stat",    "-",
                      "%.2f (p=%.3f)" % (t_v5, p_v5),
                      "%.2f (p=%.3f)" % (t_v6ms, p_v6ms),
                      "%.2f (p=%.3f)" % (t_v6, p_v6)),
        ("Switches/yr", "-",
                      "%.0f" % sw_v5,
                      "%.0f" % sw_v6ms,
                      "%.0f" % sw_v6),
    ]
    headers = ["Metric", "Buy & Hold", "v5 baseline", "v6 multi-scale", "v6 full (+vol)"]
    col_x   = [0.00, 0.22, 0.42, 0.60, 0.79]
    col_colors = [C["text"], C["bnh"], C["strat"], C["warn"], C["v6"]]

    for j, (h, cx, cc) in enumerate(zip(headers, col_x, col_colors)):
        ax4.text(cx, 0.95, h, transform=ax4.transAxes,
                 fontsize=8.5, fontweight="bold", color=cc, va="top")
    ax4.plot([0, 1], [0.88, 0.88], transform=ax4.transAxes,
             color=C["grid"], lw=0.8, clip_on=False)

    for i, row in enumerate(rows_data):
        y = 0.75 - i * 0.15
        bg = "#f8f9fa" if i % 2 == 0 else "white"
        ax4.add_patch(plt.Rectangle((0, y - 0.05), 1, 0.16,
                      transform=ax4.transAxes, color=bg, zorder=0))
        for j, (val, cx, cc) in enumerate(zip(row, col_x, col_colors)):
            ax4.text(cx, y, val, transform=ax4.transAxes,
                     fontsize=8, color=cc if j > 0 else C["text"],
                     fontweight="bold" if j > 0 else "normal", va="top")

    _footer(fig, 2)
    return fig


# ============================================================================
#  PAGE 3 - EXPANDING-WINDOW HONEST VALIDATION
# ============================================================================

def make_page3():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    _page_header(fig, "Page 3 of 3  -  Expanding-Window Honest Validation & v1-v6 Summary")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.10, right=0.95,
                           top=0.880, bottom=0.06,
                           height_ratios=[1.4, 1.1, 0.9],
                           hspace=1.05, wspace=0.40)

    # -- Panel 1: Expanding-window equity curve (full width) ------------------
    ax1 = fig.add_subplot(gs[0, :])
    exp_eq_s  = np.exp(exp_bt["strategy_return"].cumsum())
    exp_eq_b  = np.exp(exp_bt["bnh_return"].cumsum())

    # In-sample equity curve (v6 full) aligned to same period
    in_eq_s = bt_v6["equity_strategy"].reindex(exp_bt.index)
    in_eq_b = bt_v6["equity_bnh"].reindex(exp_bt.index)

    ax1.plot(exp_eq_b.index, exp_eq_b,
             color=C["bnh"],  lw=1.5, alpha=0.9,
             label="Buy & Hold  (Sharpe %.2f)" % exp_b["Sharpe"])
    ax1.plot(in_eq_s.index,  in_eq_s,
             color=C["v6"],   lw=1.2, ls="--", alpha=0.7,
             label="v6 in-sample  (Sharpe %.2f, uses full-history thresholds)" % sh_v6)
    ax1.plot(exp_eq_s.index, exp_eq_s,
             color=C["v6"],   lw=1.9,
             label="v6 expanding-window OOS  (Sharpe %.2f, annual refit)" % exp_sharpe)
    ax1.axhline(1.0, color=C["grid"], lw=0.7, ls="--")
    ax1.set_ylabel("Portfolio value (base = 1.0)", fontsize=8)
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.88)
    _style(ax1)
    _tc(ax1,
        "Expanding-Window OOS vs In-Sample  (v6 multi-scale + vol ratio, 0 bps)",
        "Expanding: geometric thresholds and Markov EM refitted annually on all prior data. "
        "Test period begins after 2-year minimum training window (~2002). "
        "Optimism bias = in-sample Sharpe (%.2f) - expanding Sharpe (%.2f) = %+.2f." % (
            sh_v6, exp_sharpe, optimism))
    fig.autofmt_xdate(rotation=25, ha="right")

    # -- Panel 2: Drawdown comparison -----------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    exp_dd  = exp_eq_s / exp_eq_s.cummax() - 1
    in_dd   = in_eq_s  / in_eq_s.cummax()  - 1
    bnh_dd  = exp_eq_b / exp_eq_b.cummax() - 1
    ax2.fill_between(bnh_dd.dropna().index, bnh_dd.dropna(), 0,
                     alpha=0.25, color=C["bnh"],
                     label="Buy & Hold  max %.1f%%" % (bnh_dd.min()*100))
    ax2.fill_between(exp_dd.dropna().index, exp_dd.dropna(), 0,
                     alpha=0.55, color=C["v6"],
                     label="Expanding OOS  max %.1f%%" % (exp_dd.min()*100))
    ax2.plot(in_dd.dropna().index, in_dd.dropna(),
             color=C["v6"], lw=0.8, ls="--", alpha=0.5,
             label="In-sample  max %.1f%%" % (in_dd.min()*100))
    ax2.set_ylabel("Drawdown from peak", fontsize=8)
    ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=7, loc="lower left", framealpha=0.85)
    _style(ax2)
    _tc(ax2, "Drawdown: Expanding OOS vs In-Sample")
    fig.autofmt_xdate(rotation=25, ha="right")

    # -- Panel 3: Rolling Sharpe comparison -----------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    roll_win = 252
    in_ret_aligned  = bt_v6["strategy_return"].reindex(exp_bt.index).dropna()
    exp_ret_aligned = exp_bt["strategy_return"].dropna()

    def rolling_sharpe(r, w=roll_win):
        return (r.rolling(w).mean() * 252) / (r.rolling(w).std() * np.sqrt(252))

    rs_in  = rolling_sharpe(in_ret_aligned)
    rs_exp = rolling_sharpe(exp_ret_aligned)
    rs_bnh = rolling_sharpe(exp_bt["bnh_return"].dropna())

    ax3.plot(rs_bnh.dropna().index, rs_bnh.dropna(),
             color=C["bnh"],  lw=1.2, alpha=0.8, label="Buy & Hold")
    ax3.plot(rs_in.dropna().index,  rs_in.dropna(),
             color=C["v6"],   lw=1.0, ls="--", alpha=0.6, label="In-sample")
    ax3.plot(rs_exp.dropna().index, rs_exp.dropna(),
             color=C["v6"],   lw=1.6, label="Expanding OOS")
    ax3.axhline(0, color=C["sub"], lw=0.8)
    ax3.set_ylabel("Sharpe (252-day rolling)", fontsize=8)
    ax3.tick_params(axis="x", labelsize=7)
    ax3.legend(fontsize=7, loc="lower right", framealpha=0.85)
    _style(ax3)
    _tc(ax3, "Rolling 252-Day Sharpe: In-Sample vs Expanding OOS",
        "In-sample uses full-history thresholds throughout. "
        "Expanding OOS refits annually. Convergence of the two lines "
        "indicates low optimism bias in a given period.")
    fig.autofmt_xdate(rotation=25, ha="right")

    # -- Panel 4: Version history summary table -------------------------------
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    _tc(ax4, "Version History  -  Key Changes and Impact")

    version_rows = [
        ("v1.0", "Geometric + Markov k=3 ensemble",                "SPY 2022-2025",  "--",   "--",   "--"),
        ("v2.0", "Walk-forward leakage fix; 2000-2025 data",        "SPY 2000-2025",  "--",   "--",   "--"),
        ("v3.0", "BIC extended; VIX pipeline",                      "SPY 2000-2025",  "+1.8%","0.29","p=0.19"),
        ("v4.0", "Half-position mixed; persistence filter",         "SPY 2000-2025",  "+5.7%","0.68","p=0.002"),
        ("v5.0", "Walk-forward OOS curve; multi-asset validation",  "SPY+4 assets",   "+5.7%","0.68","p=0.002"),
        ("v6.0", "Vol ratio; multi-scale geo; expanding-window BT", "SPY 2000-2025",
         "%+.1f%%" % (cagr_v6*100), "%.2f" % sh_v6, "p=%.3f" % p_v6),
    ]
    hdrs = ["Version", "Key Change", "Data", "CAGR", "Sharpe", "T-stat sig."]
    col_x2 = [0.00, 0.10, 0.55, 0.70, 0.79, 0.89]

    for j, (h, cx) in enumerate(zip(hdrs, col_x2)):
        ax4.text(cx, 0.95, h, transform=ax4.transAxes,
                 fontsize=8, fontweight="bold", color=C["text"], va="top")
    ax4.plot([0, 1], [0.87, 0.87], transform=ax4.transAxes,
             color=C["grid"], lw=0.8, clip_on=False)

    for i, row in enumerate(version_rows):
        y   = 0.74 - i * 0.125
        bg  = "#f8f9fa" if i % 2 == 0 else "white"
        vc  = C["v6"] if row[0] == "v6.0" else (C["strat"] if row[0] in ("v4.0","v5.0") else C["sub"])
        ax4.add_patch(plt.Rectangle((0, y - 0.04), 1, 0.125,
                      transform=ax4.transAxes, color=bg, zorder=0))
        for j, (val, cx) in enumerate(zip(row, col_x2)):
            ax4.text(cx, y, val, transform=ax4.transAxes,
                     fontsize=7.5,
                     color=vc if j == 0 else C["text"],
                     fontweight="bold" if j == 0 else "normal",
                     va="top")

    _footer(fig, 3)
    return fig


# ============================================================================
#  SAVE
# ============================================================================

out_path = OUTPUT_DIR / "SPY_v6_report.pdf"

print("\nGenerating pages...")
with PdfPages(out_path) as pdf:
    for page_num, fn in enumerate([make_page1, make_page2, make_page3], 1):
        print("  page %d..." % page_num)
        f = fn()
        pdf.savefig(f, bbox_inches="tight", facecolor="white")
        plt.close(f)

print("Report saved -> %s" % out_path)
print("File size: %.1f KB" % (out_path.stat().st_size / 1024))

with open(out_path, "rb") as f:
    hdr = f.read(8)
print("PDF header: %s  [%s]" % (
    hdr.decode("latin-1"),
    "OK" if hdr.startswith(b"%PDF") else "ERROR"))

with open(__file__) as f:
    src = f.read()
bad = [(i, c) for i, c in enumerate(src) if ord(c) > 127]
if bad:
    print("WARNING: non-ASCII chars: %s" % str(set(c for _, c in bad)))
else:
    print("Source file: ASCII-clean [OK]")
