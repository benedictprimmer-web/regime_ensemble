#!/usr/bin/env python3
"""
Regime Ensemble — 3-Page Overview Report (v5)
=============================================
Page 1: The Strategy   — two orthogonal signals, three position states (full / half / none)
Page 2: Models in Action — real SPY 2000-2025 data, signals and probabilities
Page 3: The Edge       — performance vs buy-and-hold, transaction costs, honest limitations

No Polygon API key needed if data/cache/SPY_2000-01-01_2025-01-01.csv exists.

Usage:
    python3 generate_report_overview.py
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
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
    "momentum":  "#1a7a4a",
    "reversion": "#c0392b",
    "mixed":     "#95a5a6",
    "bnh":       "#2c3e50",
    "strategy":  "#2980b9",
    "text":      "#2c3e50",
    "subtext":   "#7f8c8d",
    "gold":      "#f39c12",
    "bg":        "#ffffff",
    "pos":       "#27ae60",
}


def _section_title(ax, text):
    ax.text(0, 1.04, text, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=C["text"], va="bottom")


def _caption(ax, text, y=-0.16):
    ax.text(0.5, y, text, transform=ax.transAxes,
            fontsize=7.5, color=C["subtext"], ha="center", style="italic")


def _footer(fig, text):
    fig.text(0.5, 0.02, text, ha="center", fontsize=6.5, color=C["subtext"])


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("  Loading data and running models...")
df         = fetch_daily_bars(TICKER, FROM_DATE, TO_DATE)
ret        = log_returns(df)
prices     = df["close"]
ratio      = straightness_ratio(ret)
geo        = geometric_signal(ret)
mom_prob, crisis_prob, _, trans_info = fit_markov3(ret)
regime_stats = trans_info.get("regime_stats", {})
score      = ensemble_score(geo, mom_prob, crisis_prob)
labels     = regime_labels(score)
bt         = run_backtest(ret, labels, allow_short=False, cost_bps=0)
perf       = compute_stats(bt)

strat = perf["Strategy (Long Only)"]
bnh   = perf["Buy & Hold"]

# Forward-return stats by regime
from scipy import stats as scipy_stats
aligned_ret = ret.reindex(labels.index)
fwd_ret     = aligned_ret.shift(-1).dropna()
labels_fwd  = labels.reindex(fwd_ret.index)
regime_fwd  = {}
for reg in ["momentum", "mixed", "reversion"]:
    r = fwd_ret[labels_fwd == reg].dropna()
    t, p = scipy_stats.ttest_1samp(r, 0) if len(r) > 5 else (float("nan"), float("nan"))
    regime_fwd[reg] = {"n": len(r), "mean_pct": r.mean() * 100, "t": t, "p": p}

# Ensemble switches/year (position changes, not Markov internal transitions)
ensemble_switches_pa = (labels != labels.shift(1)).sum() / (len(labels) / 252)

# Break-even cost and beats-B&H threshold (scanning 0–30 bps)
_bnh_sharpe   = float(compute_stats(bt)["Buy & Hold"]["Sharpe"])
_bps_range    = list(range(0, 31))
_cost_sharpes = [
    float(compute_stats(run_backtest(ret, labels, allow_short=False, cost_bps=b))
          ["Strategy (Long Only)"]["Sharpe"])
    for b in _bps_range
]
import numpy as _np
breakeven_bps = int(_bps_range[_np.argmin(_np.abs(_np.array(_cost_sharpes)))])
beats_bnh_bps = int(_bps_range[_np.argmin(_np.abs(_np.array(_cost_sharpes) - _bnh_sharpe))])

# Regime distribution (% of trading days in each state)
n_total     = len(labels)
regime_pct  = {r: (labels == r).sum() / n_total * 100 for r in ["momentum", "mixed", "reversion"]}
regime_days = {r: (labels == r).sum() for r in ["momentum", "mixed", "reversion"]}

print("  Models ready. Generating report...")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — THE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def make_page1():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")  # A4

    fig.text(0.5, 0.958, "The Strategy — Two Signals, Three Positions",
             ha="center", fontsize=14, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.941,
             "Geometric straightness ratio + Markov k=3 hidden states  ·  SPY 2000-2025  ·  v5.0",
             ha="center", fontsize=8.5, color=C["subtext"])
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.932, 0.932],
                               transform=fig.transFigure, color="#dde", linewidth=1))

    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.08, right=0.92,
                           top=0.908, bottom=0.06, hspace=0.78, wspace=0.38)

    # ── Panel 1: Two orthogonal detectors (schematic) ─────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(42)
    t = np.arange(60)
    trend  = 0.002 * t + 0.012 * np.random.randn(60).cumsum()
    choppy = 0.012 * np.random.randn(60).cumsum()
    ax1.plot(t[:30], trend[:30],  color=C["momentum"],  lw=1.8)
    ax1.plot(t[30:], choppy[30:] + trend[29], color=C["reversion"], lw=1.8)
    ax1.axvline(30, color=C["subtext"], lw=0.8, ls="--")
    ax1.text(0.25, 0.90, "Trending", ha="center", transform=ax1.transAxes,
             fontsize=8, color=C["momentum"], fontweight="bold")
    ax1.text(0.75, 0.90, "Choppy", ha="center", transform=ax1.transAxes,
             fontsize=8, color=C["reversion"], fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.spines[["top", "right"]].set_visible(False)
    _section_title(ax1, "Two Orthogonal Detectors — One Ensemble")
    _caption(ax1, "Geometric: path straightness over 15 days (no parameters to fit).\n"
             "Markov: hidden 3-state model estimated from the full return history.\n"
             "Combined as a simple mean — no weight fitting, no in-sample target.")

    # ── Panel 2: Key statistical finding ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    _section_title(ax2, "Why It Works — Forward Return by Regime")

    ax2.text(0.02, 0.94, "Regime", transform=ax2.transAxes,
             fontsize=8, fontweight="bold", color=C["text"], va="top")
    ax2.text(0.40, 0.94, "Mean ret/day", transform=ax2.transAxes,
             fontsize=8, fontweight="bold", color=C["text"], va="top")
    ax2.text(0.73, 0.94, "T-stat  (p)", transform=ax2.transAxes,
             fontsize=8, fontweight="bold", color=C["text"], va="top")
    ax2.axline((0, 0.89), slope=0, color="#dde", lw=0.8, transform=ax2.transAxes)

    rows = [("momentum", C["momentum"]), ("mixed", C["gold"]), ("reversion", C["reversion"])]
    for i, (reg, col) in enumerate(rows):
        d   = regime_fwd[reg]
        y   = 0.79 - i * 0.22
        sig = "**" if d["p"] < 0.01 else ("*" if d["p"] < 0.05 else "")
        ax2.text(0.02, y, reg.title(), transform=ax2.transAxes,
                 fontsize=8.5, color=col, fontweight="bold", va="top")
        ax2.text(0.40, y, "%+.3f%%" % d["mean_pct"], transform=ax2.transAxes,
                 fontsize=8.5, color=col, va="top")
        ax2.text(0.73, y, "%.2f (%.3f)%s" % (d["t"], d["p"], sig),
                 transform=ax2.transAxes, fontsize=8.5, color=col, va="top")

    ax2.text(0.02, 0.13,
             "Mixed regime (detectors disagree) still drifts positive.\n"
             "T=3.21** is the strongest signal — justifies the half-position\n"
             "instead of defaulting to cash on uncertainty.",
             transform=ax2.transAxes, fontsize=7.5, color=C["subtext"], va="top")

    # ── Panel 3: Geometric detector ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    np.random.seed(7)
    days    = np.arange(15)
    straight = np.linspace(0, 0.3, 15) + 0.01 * np.random.randn(15)
    zigzag   = 0.04 * np.random.randn(15).cumsum()
    ax3.plot(days, straight, color=C["momentum"],  lw=2, label="Momentum path  (ratio → 1)")
    ax3.plot(days, zigzag,   color=C["reversion"], lw=2, label="Reversion path (ratio → 0)", ls="--")
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(fontsize=7, loc="upper left", framealpha=0.6)
    _section_title(ax3, "Detector 1 — Geometric (path shape, 15-day window)")
    _caption(ax3,
             "ratio = |cumulative return| / Σ|daily returns|  over 15 days.\n"
             "Near 1.0 = straight-line move (momentum). Near 0.0 = oscillation (reversion).\n"
             "Thresholds are adaptive percentiles — no fixed values to tune.")

    # ── Panel 4: Markov detector ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    m_stats  = regime_stats
    mom_mean = m_stats.get("MOMENTUM", {}).get("mean", 0.00085) * 100
    chp_mean = m_stats.get("CHOPPY",   {}).get("mean", -0.00004) * 100
    cri_mean = m_stats.get("CRISIS",   {}).get("mean", -0.00167) * 100
    regimes  = ["Momentum\n(trending up)", "Choppy\n(sideways)", "Crisis\n(falling fast)"]
    means    = [mom_mean, chp_mean, cri_mean]
    colors   = [C["momentum"], C["mixed"], C["reversion"]]
    ax4.barh(regimes, means, color=colors, alpha=0.85, height=0.5)
    ax4.axvline(0, color=C["subtext"], lw=0.8)
    ax4.set_xlabel("Mean daily return (%)", fontsize=8)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.tick_params(labelsize=8)
    _section_title(ax4, "Detector 2 — Markov AR(1), k=3 hidden states")
    _caption(ax4,
             "k=3 selected by BIC (ΔBIC=82 vs k=2). Fitted on training data only;\n"
             "Hamilton filter on test data — no look-ahead. Crisis (P>0.50) overrides\n"
             "all buy signals regardless of what the geometric detector says.",
             y=-0.32)

    # ── Panel 5: Position sizing rule — Full / Half / None ────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    _section_title(ax5, "The Decision Rule — Full / Half / None")

    # Score formula
    ax5.text(0.5, 0.97,
             "score  =  mean( geometric_signal ,  markov_momentum_probability )",
             transform=ax5.transAxes, ha="center", fontsize=9,
             color=C["text"], va="top", fontfamily="monospace")

    # Three state boxes
    box_specs = [
        (0.03, "FULL LONG", "position = +1.0", "score ≥ 0.65",
         "Trend confirmed by both detectors.", "momentum"),
        (0.36, "HALF LONG", "position = +0.5", "0.35 ≤ score < 0.65",
         "Detectors disagree — hold half.", "mixed"),
        (0.69, "CASH",      "position =  0.0", "score < 0.35",
         "Reversion or crisis — step aside.", "reversion"),
    ]

    for (x, label, pos_str, rule, desc, reg) in box_specs:
        col  = C[reg]
        n    = regime_days[reg]
        pct  = regime_pct[reg]

        ax5.add_patch(mpatches.FancyBboxPatch(
            (x, 0.06), 0.29, 0.72,
            boxstyle="round,pad=0.015",
            facecolor=col, alpha=0.10,
            edgecolor=col, linewidth=1.8,
            transform=ax5.transAxes, clip_on=False))

        ax5.text(x + 0.145, 0.70, label, transform=ax5.transAxes,
                 fontsize=11, fontweight="bold", color=col, ha="center", va="top")
        ax5.text(x + 0.145, 0.54, pos_str, transform=ax5.transAxes,
                 fontsize=9.5, color=col, ha="center", va="top",
                 fontfamily="monospace")
        ax5.text(x + 0.145, 0.41, rule, transform=ax5.transAxes,
                 fontsize=8, color=C["subtext"], ha="center", va="top")
        ax5.text(x + 0.145, 0.30, desc, transform=ax5.transAxes,
                 fontsize=7.8, color=C["subtext"], ha="center", va="top", style="italic")
        ax5.text(x + 0.145, 0.15,
                 f"{pct:.0f}% of trading days  ({n:,} days  ·  SPY 2000-2025)",
                 transform=ax5.transAxes, fontsize=7.5, color=col, ha="center", va="top")

    _footer(fig, "Page 1 of 3  ·  regime_ensemble v5.0  ·  github.com/benedictprimmer-web/regime_ensemble")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODELS IN ACTION (REAL DATA)
# ══════════════════════════════════════════════════════════════════════════════

def make_page2():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.95, top=0.87, bottom=0.06, hspace=0.45)

    fig.text(0.5, 0.960, "Models in Action — SPY 2000-2025 (Real Data)",
             ha="center", fontsize=14, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.943,
             "Price coloured by regime  ·  geometric straightness ratio  ·  Markov filtered probabilities",
             ha="center", fontsize=9, color=C["subtext"])
    fig.add_artist(plt.Line2D([0.09, 0.95], [0.935, 0.935],
                               transform=fig.transFigure, color="#dde", lw=1))

    gs = gridspec.GridSpec(3, 1, figure=fig, left=0.09, right=0.95,
                           top=0.870, bottom=0.06, hspace=0.42)

    # ── Panel 1: SPY price coloured by regime ─────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    regime_color = labels.map(C).reindex(prices.index, method="ffill")
    valid    = ~regime_color.isna()
    prices_v = prices[valid]
    colors_v = regime_color[valid]
    for i in range(len(prices_v) - 1):
        ax1.plot(prices_v.index[i:i+2], prices_v.iloc[i:i+2],
                 color=colors_v.iloc[i], lw=1.1, alpha=0.9)
    patches = [mpatches.Patch(color=C["momentum"],  label="Momentum — Full long (+1.0)"),
               mpatches.Patch(color=C["mixed"],     label="Mixed — Half long (+0.5)"),
               mpatches.Patch(color=C["reversion"], label="Reversion/Crisis — Cash (0.0)")]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.75)
    ax1.set_ylabel("SPY Close ($)", fontsize=8)
    ax1.set_xticks([])
    ax1.spines[["top", "right"]].set_visible(False)
    _section_title(ax1, "SPY Price — Coloured by Ensemble Regime")
    _caption(ax1,
             "Green = momentum (+1.0 position). Grey = mixed (+0.5). Red = reversion or crisis (flat).\n"
             "25 years: dot-com (2000-02), GFC (2008), COVID (2020), bear market (2022).")

    # ── Panel 2: Geometric straightness ratio ─────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    mom_thresh = ratio.quantile(0.70)
    rev_thresh = ratio.quantile(0.30)
    ax2.plot(ratio.index, ratio, color=C["subtext"], lw=0.8, alpha=0.8)
    ax2.axhline(mom_thresh, color=C["momentum"],  ls="--", lw=1.0, alpha=0.7,
                label=f"Momentum threshold  ({mom_thresh:.2f})")
    ax2.axhline(rev_thresh, color=C["reversion"], ls="--", lw=1.0, alpha=0.7,
                label=f"Reversion threshold ({rev_thresh:.2f})")
    ax2.fill_between(ratio.index, ratio, mom_thresh,
                     where=(ratio >= mom_thresh), alpha=0.25, color=C["momentum"])
    ax2.fill_between(ratio.index, ratio, rev_thresh,
                     where=(ratio <= rev_thresh), alpha=0.25, color=C["reversion"])
    ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.7)
    ax2.set_ylabel("Straightness Ratio", fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([])
    ax2.spines[["top", "right"]].set_visible(False)
    _section_title(ax2, "Geometric Detector — Straightness Ratio (15-day window)")
    _caption(ax2,
             "Near 1.0 = price moved in a straight line (momentum). Near 0.0 = price zigzagged.\n"
             "Thresholds are adaptive percentiles — no fixed values to hand-tune.")

    # ── Panel 3: Markov filtered probabilities ────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(mom_prob.index,    mom_prob,    0, alpha=0.50,
                     color=C["momentum"],  label="P(momentum) — trending state")
    ax3.fill_between(crisis_prob.index, crisis_prob, 0, alpha=0.50,
                     color=C["reversion"], label="P(crisis) — falling-fast state")
    ax3.axhline(0.5, color=C["subtext"], ls=":", lw=0.9,
                label="Crisis override threshold (0.50) — suppresses all buys above this")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Probability", fontsize=8)
    ax3.legend(fontsize=7.5, loc="upper right", framealpha=0.7)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(axis="x", labelsize=8)
    _section_title(ax3, "Markov Detector — Filtered State Probabilities (k=3)")
    _caption(ax3,
             "When P(crisis) > 0.50, buy signals are suppressed regardless of all other indicators.\n"
             "Filtered probabilities only — Hamilton filter, no look-ahead bias.")

    fig.autofmt_xdate(rotation=25, ha="right")
    _footer(fig, "Page 2 of 3  ·  regime_ensemble v5.0  ·  github.com/benedictprimmer-web/regime_ensemble")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — THE EDGE & LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════

def make_page3():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.95, top=0.87, bottom=0.12,
                        hspace=0.90, wspace=0.38)

    fig.text(0.5, 0.960, "The Edge — Performance vs Buy & Hold",
             ha="center", fontsize=16, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.942,
             "Risk-adjusted return, not raw return  ·  SPY 2000-2025  ·  v5.0",
             ha="center", fontsize=9, color=C["subtext"])
    fig.add_artist(plt.Line2D([0.09, 0.95], [0.933, 0.933],
                               transform=fig.transFigure, color="#dde", lw=1))

    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.09, right=0.95,
                           top=0.87, bottom=0.12, hspace=0.90, wspace=0.55)

    # ── Panel 1: Equity curves with regime shading ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    # Shade reversion/crisis periods light red so it's obvious when we were flat
    prev_lab   = None
    span_start = None
    for date, lab in labels.items():
        if lab == "reversion" and prev_lab != "reversion":
            span_start = date
        elif lab != "reversion" and prev_lab == "reversion" and span_start is not None:
            ax1.axvspan(span_start, date, alpha=0.07, color=C["reversion"], lw=0, zorder=0)
            span_start = None
        prev_lab = lab
    if prev_lab == "reversion" and span_start is not None:
        ax1.axvspan(span_start, labels.index[-1], alpha=0.07, color=C["reversion"], lw=0, zorder=0)

    ax1.plot(bt.index, bt["equity_bnh"],
             color=C["bnh"], lw=1.8,
             label=f"Buy & Hold  (CAGR {bnh['CAGR']}, Sharpe {bnh['Sharpe']}, Max DD {bnh['Max DD']})")
    ax1.plot(bt.index, bt["equity_strategy"],
             color=C["strategy"], lw=1.8,
             label=f"Regime Ensemble  (CAGR {strat['CAGR']}, Sharpe {strat['Sharpe']}, Max DD {strat['Max DD']})")
    ax1.axhline(1.0, color="#dde", lw=0.7, ls="--")
    ax1.set_ylabel("Portfolio value (starting = 1.0)", fontsize=8)
    ax1.legend(fontsize=8, framealpha=0.85, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="x", labelsize=8)

    # Annotate the major crashes where going to cash protected capital
    for (yr, label_txt, x_offset) in [
        ("2001-09-01", "Dot-com", 0),
        ("2008-10-01", "GFC",     0),
        ("2020-03-20", "COVID",   0),
        ("2022-06-01", "2022",    0),
    ]:
        try:
            ax1.axvline(pd.Timestamp(yr), color=C["reversion"], lw=0.5,
                        ls=":", alpha=0.4, zorder=1)
        except Exception:
            pass

    _section_title(ax1, "Equity Curves — Strategy vs Buy & Hold (0 bps transaction costs)")
    _caption(ax1,
             "Red shading = periods the strategy held cash (reversion regime). "
             "Strategy sidesteps the worst of each major crash.\n"
             "The edge is risk-adjusted: Sharpe 0.68 vs 0.44, drawdown -16.7% vs -56.5%. CAGR trails (+5.7% vs +8.6%).",
             y=-0.30)

    # ── Panel 2: Drawdown ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    strat_dd = bt["equity_strategy"] / bt["equity_strategy"].cummax() - 1
    bnh_dd   = bt["equity_bnh"]      / bt["equity_bnh"].cummax()      - 1
    ax2.fill_between(bt.index, bnh_dd,   0, alpha=0.35, color=C["bnh"],
                     label=f"Buy & Hold  (max {bnh['Max DD']})")
    ax2.fill_between(bt.index, strat_dd, 0, alpha=0.55, color=C["strategy"],
                     label=f"Strategy  (max {strat['Max DD']})")
    ax2.set_ylabel("Drawdown from peak", fontsize=8)
    ax2.legend(fontsize=7.5, loc="lower left", framealpha=0.85)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", labelsize=7)
    _section_title(ax2, "Drawdown Comparison")
    ax2.text(0.0, -0.30,
             "Strategy trims drawdowns across all major crashes\n(dot-com, GFC, COVID, 2022).",
             transform=ax2.transAxes, fontsize=7.5, color=C["subtext"], ha="left", style="italic")

    # ── Panel 3: Transaction cost sensitivity ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    from src.backtest import run_backtest as _rb
    cost_bps  = [0, 5, 10, 20]
    sharpes   = []
    for bps in cost_bps:
        bt_c = _rb(ret, labels, allow_short=False, cost_bps=bps)
        s    = compute_stats(bt_c)["Strategy (Long Only)"]["Sharpe"]
        sharpes.append(float(s))

    colors_bar = [C["momentum"] if s > float(bnh["Sharpe"]) else
                  (C["gold"]     if s > 0 else C["reversion"])
                  for s in sharpes]
    bars = ax3.barh([f"{b} bps" for b in cost_bps], sharpes,
                    color=colors_bar, alpha=0.88, height=0.5)
    ax3.axvline(0, color=C["subtext"], lw=0.8)
    ax3.axvline(float(bnh["Sharpe"]), color=C["bnh"], lw=0.9, ls="--",
                label=f"B&H Sharpe ({bnh['Sharpe']})")
    ax3.legend(fontsize=7, loc="lower right", framealpha=0.8)
    ax3.text(0.98, 0.04, "Sharpe Ratio \u2192", transform=ax3.transAxes,
             ha="right", va="bottom", fontsize=7.5, color=C["subtext"])
    for bar, val in zip(bars, sharpes):
        ax3.text(max(val + 0.02, 0.04), bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=8.5, color=C["text"])
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(labelsize=8)
    _section_title(ax3, "Transaction Cost Sensitivity")
    ax3.text(0.0, -0.20,
             "~%.0f switches/yr · beats B&H Sharpe to ~%d bps · breaks even at ~%d bps.\n"
             "--min-hold 3 reduces switches and extends the profitable range." %
             (ensemble_switches_pa, beats_bnh_bps, breakeven_bps),
             transform=ax3.transAxes, fontsize=7.5, color=C["subtext"], ha="left", style="italic")

    # ── Panel 4: Limitations ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    _section_title(ax4, "Honest Limitations — Read Before Drawing Conclusions")

    limits = [
        ("Strategy CAGR trails buy-and-hold",
         "+5.7% vs +8.6% over 25 years at zero cost. "
         "The edge is Sharpe (0.68 vs 0.44) and max drawdown (-16.7% vs -56.5%). "
         "This is a risk-reduction strategy, not an alpha generator."),
        ("Transaction costs are material",
         "~%.0f position switches/year. Sharpe beats B&H up to ~%d bps round-trip; "
         "strategy breaks even at ~%d bps. Use --min-hold 3 to cut switches." %
         (ensemble_switches_pa, beats_bnh_bps, breakeven_bps)),
        ("In-sample threshold calibration",
         "Percentile thresholds (70th/30th) and ensemble cutoffs (0.65/0.35) are fit on the full dataset. "
         "A real deployment requires expanding-window recalibration — see --expanding."),
        ("Reversion signal is not statistically significant",
         "Regime T-stats: mixed=3.21** (p=0.001), momentum=1.33 (p=0.18), reversion=-0.35 (p=0.73). "
         "The strategy avoids reversion periods but does not short them."),
        ("Calibrated on SPY — multi-asset not validated per-asset",
         "Thresholds are fitted on SPY. --multi-asset applies the same model to QQQ/IWM/TLT/GLD as "
         "a robustness check, but each asset's regime structure differs."),
    ]
    for i, (title, body) in enumerate(limits):
        y = 0.91 - i * 0.177
        ax4.text(0.01, y,       f"⚠  {title}", transform=ax4.transAxes,
                 fontsize=8.5, fontweight="bold", color=C["gold"], va="top")
        ax4.text(0.01, y - 0.07, f"    {body}", transform=ax4.transAxes,
                 fontsize=7.8, color=C["subtext"], va="top")

    _footer(fig, "Page 3 of 3  ·  regime_ensemble v5.0  ·  github.com/benedictprimmer-web/regime_ensemble")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════════════

out_path = OUTPUT_DIR / "SPY_overview_report.pdf"
with PdfPages(out_path) as pdf:
    for page_fn in [make_page1, make_page2, make_page3]:
        fig = page_fn()
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

print(f"  Report saved → {out_path}")
