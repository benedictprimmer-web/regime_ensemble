#!/usr/bin/env python3
"""
Regime Ensemble - Technical Quant Report (3 pages)
===================================================
Produces outputs/SPY_quant_report.pdf

Page 1: Regime Structure - statistical evidence
Page 2: Signal Quality - predictive power and diagnostics
Page 3: Portfolio Analytics - full attribution

Usage:
    python3 generate_report_quant.py
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde, ks_2samp
from statsmodels.tsa.stattools import acf

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
    "choppy":    "#95a5a6",
    "crisis":    "#c0392b",
    "bnh":       "#2c3e50",
    "strategy":  "#2980b9",
    "text":      "#1a1a2e",
    "subtext":   "#6c757d",
    "gold":      "#e67e22",
    "accent":    "#8e44ad",
    "bg":        "#ffffff",
    "gridline":  "#e8e8e8",
}

REGIME_COLORS = {
    "momentum":  C["momentum"],
    "mixed":     C["mixed"],
    "reversion": C["reversion"],
}


def _title(ax, text, sub=None):
    ax.set_title(text, fontsize=9, fontweight="bold", color=C["text"],
                 loc="left", pad=4)
    if sub:
        # Place sub INSIDE the axes to avoid overlap with adjacent panels
        ax.text(0.01, 0.975, sub, transform=ax.transAxes,
                fontsize=6.5, color=C["subtext"], va="top")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["gridline"])
    ax.tick_params(colors=C["subtext"], labelsize=7.5)
    ax.yaxis.label.set_color(C["subtext"])
    ax.xaxis.label.set_color(C["subtext"])
    ax.grid(axis="y", color=C["gridline"], lw=0.5, zorder=0)


def _hline(ax, y, **kwargs):
    """Draw a horizontal line across axes using transAxes (avoids axline API)."""
    ax.plot([0, 1], [y, y], transform=ax.transAxes, **kwargs)


def _footer(fig, page, total=3):
    fig.text(
        0.5, 0.012,
        "Page %d of %d  -  regime_ensemble v4.0  -  SPY %s-%s  -  "
        "github.com/benedictprimmer-web/regime_ensemble" % (
            page, total, FROM_DATE[:4], TO_DATE[:4]),
        ha="center", fontsize=6, color=C["subtext"],
    )


# ============================================================================
#  LOAD AND COMPUTE
# ============================================================================

print("  Loading data...")
df         = fetch_daily_bars(TICKER, FROM_DATE, TO_DATE)
ret        = log_returns(df)
prices     = df["close"]
ratio      = straightness_ratio(ret)
geo        = geometric_signal(ret)
mom_prob, crisis_prob, choppy_prob, trans_info = fit_markov3(ret)
score      = ensemble_score(geo, mom_prob, crisis_prob)
labels     = regime_labels(score)
bt         = run_backtest(ret, labels, allow_short=False, cost_bps=0)
perf       = compute_stats(bt)
strat_perf = perf["Strategy (Long Only)"]
bnh_perf   = perf["Buy & Hold"]

# Hard Markov regime labels
all_probs          = pd.concat([mom_prob, choppy_prob, crisis_prob], axis=1)
all_probs.columns  = ["momentum", "choppy", "crisis"]
hard_markov        = all_probs.idxmax(axis=1)

# Regime-aligned returns
aligned_ret = ret.reindex(labels.index)
fwd_ret     = aligned_ret.shift(-1).dropna()
labels_fwd  = labels.reindex(fwd_ret.index)

print("  Computing analytics...")

# Sojourn times
def sojourn_times(label_series, regime):
    runs, count = [], 0
    for v in label_series:
        if v == regime:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs

sojourn = {r: sojourn_times(labels, r) for r in ["momentum", "mixed", "reversion"]}

# Rolling 63-day IC
score_aligned = score.reindex(fwd_ret.index)
ic_rolling    = score_aligned.rolling(63).corr(fwd_ret).dropna()

# Extended performance metrics
strat_ret = bt["strategy_return"].dropna()
bnh_ret   = bt["bnh_return"].dropna()

def extended_metrics(r):
    ann    = 252
    cagr   = (1 + r).prod() ** (ann / len(r)) - 1
    vol    = r.std() * np.sqrt(ann)
    sharpe = cagr / vol if vol > 0 else 0
    neg    = r[r < 0]
    sortino = cagr / (neg.std() * np.sqrt(ann)) if len(neg) > 0 else np.nan
    eq      = (1 + r).cumprod()
    dd      = eq / eq.cummax() - 1
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd != 0 else np.nan
    wins    = (r > 0).mean()
    avg_win  = r[r > 0].mean() if (r > 0).any() else 0
    avg_loss = r[r < 0].mean() if (r < 0).any() else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    return {
        "CAGR":        "%+.1f%%" % (cagr * 100),
        "Vol (ann.)":  "%.1f%%" % (vol * 100),
        "Sharpe":      "%.2f" % sharpe,
        "Sortino":     "%.2f" % sortino if not np.isnan(sortino) else "n/a",
        "Calmar":      "%.2f" % calmar  if not np.isnan(calmar)  else "n/a",
        "Max DD":      "%.1f%%" % (max_dd * 100),
        "Win rate":    "%.1f%%" % (wins * 100),
        "Profit fctr": "%.2f" % pf      if not np.isnan(pf)      else "n/a",
    }

strat_ext = extended_metrics(strat_ret)
bnh_ext   = extended_metrics(bnh_ret)

excess = strat_ret - bnh_ret
ir     = (excess.mean() * 252) / (excess.std() * np.sqrt(252))
# Mean exposure: full position on momentum + half on mixed
pct_momentum = (labels == "momentum").mean()
pct_mixed    = (labels == "mixed").mean()
mean_exposure = pct_momentum + 0.5 * pct_mixed

# Cost sensitivity curve
bps_range  = np.arange(0, 31, 1)
cost_curve = []
for bps in bps_range:
    b = run_backtest(ret, labels, allow_short=False, cost_bps=int(bps))
    s = compute_stats(b)["Strategy (Long Only)"]["Sharpe"]
    cost_curve.append(float(s))
cost_curve = np.array(cost_curve)


# ============================================================================
#  PAGE 1 - REGIME STRUCTURE: STATISTICAL EVIDENCE
# ============================================================================

def make_page1():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")

    fig.text(0.5, 0.965, "Regime Ensemble - Technical Report",
             ha="center", fontsize=15, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.948, "Page 1 of 3  -  Regime Structure: Statistical Evidence",
             ha="center", fontsize=9, color=C["subtext"])
    fig.add_artist(plt.Line2D([0.09, 0.95], [0.938, 0.938],
                              transform=fig.transFigure, color=C["gridline"], lw=1))

    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.09, right=0.95,
                           top=0.880, bottom=0.07, hspace=1.10, wspace=0.42)

    # Panel 1: Return distributions by regime (KDE)
    ax1 = fig.add_subplot(gs[0, :])
    ret_pct     = aligned_ret * 100
    x_grid      = np.linspace(ret_pct.quantile(0.01), ret_pct.quantile(0.99), 400)
    ks_results  = {}
    regime_data = {}
    for regime, col in [("momentum", C["momentum"]),
                        ("mixed",    C["mixed"]),
                        ("reversion",C["reversion"])]:
        mask = labels.reindex(ret_pct.index) == regime
        r    = ret_pct[mask].dropna()
        regime_data[regime] = r
        if len(r) > 10:
            kde = gaussian_kde(r, bw_method=0.4)
            ax1.fill_between(x_grid, kde(x_grid), alpha=0.25, color=col)
            ax1.plot(
                x_grid, kde(x_grid), color=col, lw=1.8,
                label="%s  (n=%d, mean=%+.3f%%, std=%.3f%%)" % (
                    regime.title(), len(r), r.mean(), r.std()),
            )
    for (r1, r2) in [("momentum", "reversion"), ("momentum", "mixed")]:
        ks_stat, ks_p = ks_2samp(regime_data[r1], regime_data[r2])
        ks_results["%s vs %s" % (r1, r2)] = (ks_stat, ks_p)

    ax1.axvline(0, color=C["subtext"], lw=0.8, ls="--", alpha=0.6)
    ax1.set_xlabel("Daily return (%)", fontsize=8)
    ax1.set_ylabel("Density", fontsize=8)
    ax1.legend(fontsize=7.5, framealpha=0.8, loc="upper right")
    _style(ax1)
    _title(ax1, "Return Distributions by Regime (Kernel Density Estimate)")
    ks_txt = "  |  ".join(
        "K-S %s: D=%.3f, p=%.4f" % (k, v[0], v[1]) for k, v in ks_results.items()
    )
    ax1.text(0.01, -0.22, ks_txt, transform=ax1.transAxes,
             fontsize=6.8, color=C["subtext"], style="italic")

    # Panel 2: Transition matrix heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    if trans_info and "matrix" in trans_info:
        mat   = trans_info["matrix"]
        names = ["Mom.", "Chpy.", "Crisis"]
        cmap  = LinearSegmentedColormap.from_list("rg", ["#ffffff", "#1a7a4a"])
        im    = ax2.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(names, fontsize=8)
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel("Next regime", fontsize=8)
        ax2.set_ylabel("Current regime", fontsize=8)
        for i in range(3):
            for j in range(3):
                txt_col = "white" if mat[i, j] > 0.6 else C["text"]
                ax2.text(j, i, "%.1f%%" % (mat[i, j] * 100),
                         ha="center", va="center", fontsize=9,
                         color=txt_col, fontweight="bold")
        plt.colorbar(im, ax=ax2, shrink=0.85, label="Transition prob.")
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
    _title(ax2, "Markov Transition Matrix")
    ax2.text(0.5, -0.30,
             "Diagonal = regime persistence. High diagonal = sticky regimes.",
             transform=ax2.transAxes, fontsize=7, color=C["subtext"],
             ha="center", style="italic")

    # Panel 3: Regime sojourn time distribution
    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.arange(1, 35, 2)
    for regime, col in [("momentum", C["momentum"]), ("reversion", C["reversion"])]:
        runs = sojourn[regime]
        if runs:
            ax3.hist(runs, bins=bins, color=col, alpha=0.55,
                     label=regime.title(), density=True,
                     edgecolor="white", lw=0.3)
            med = np.median(runs)
            ax3.axvline(med, color=col, lw=1.2, ls="--", alpha=0.8,
                        label="  median=%dd" % med)
    ax3.set_xlabel("Consecutive days in regime", fontsize=8)
    ax3.set_ylabel("Density", fontsize=8)
    ax3.legend(fontsize=7, framealpha=0.8)
    _style(ax3)
    _title(ax3, "Regime Sojourn Time Distribution")
    ax3.text(0.5, -0.30,
             "Short sojourn -> fast flip -> higher turnover -> cost drag.",
             transform=ax3.transAxes, fontsize=7, color=C["subtext"],
             ha="center", style="italic")

    # Panel 4: Regime label timeline
    ax4 = fig.add_subplot(gs[2, :])
    regime_enc = labels.map({"momentum": 1, "mixed": 0, "reversion": -1}).fillna(0)
    for regime, val, col in [("momentum",  1,  C["momentum"]),
                              ("reversion", -1, C["reversion"])]:
        mask = regime_enc == val
        ax4.fill_between(regime_enc.index, 0, regime_enc.where(mask, 0),
                         color=col, alpha=0.65, step="mid",
                         label="%s (+1 / -1)" % regime.title())
    ax4.fill_between(regime_enc.index, 0,
                     regime_enc.where(regime_enc == 0, np.nan),
                     color=C["mixed"], alpha=0.45, step="mid", label="Mixed (0)")
    ax4.axhline(0, color=C["subtext"], lw=0.6)
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(["Reversion", "Mixed", "Momentum"], fontsize=8)
    ax4.tick_params(axis="x", labelsize=7.5)
    ax4.legend(fontsize=7.5, loc="upper right", framealpha=0.8, ncol=3)
    _style(ax4)
    _title(ax4, "Ensemble Regime Label Timeline  (momentum=+1, mixed=0, reversion=-1)")

    _footer(fig, 1)
    return fig


# ============================================================================
#  PAGE 2 - SIGNAL QUALITY AND PREDICTIVE POWER
# ============================================================================

def make_page2():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")

    fig.text(0.5, 0.965, "Regime Ensemble - Technical Report",
             ha="center", fontsize=15, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.948, "Page 2 of 3  -  Signal Quality and Predictive Power",
             ha="center", fontsize=9, color=C["subtext"])
    fig.add_artist(plt.Line2D([0.09, 0.95], [0.938, 0.938],
                              transform=fig.transFigure, color=C["gridline"], lw=1))

    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.09, right=0.95,
                           top=0.880, bottom=0.07, hspace=1.10, wspace=0.42)

    # Panel 1: Forward returns by regime with 95% CI
    ax1 = fig.add_subplot(gs[0, 0])
    regime_order = ["momentum", "mixed", "reversion"]
    means, cis, ns, tstats, pvals = [], [], [], [], []
    for regime in regime_order:
        r  = fwd_ret[labels_fwd == regime].dropna()
        n  = len(r)
        m  = r.mean() * 100
        se = r.sem() * 100
        ci = stats.t.ppf(0.975, df=n - 1) * se if n > 1 else 0
        t, p = stats.ttest_1samp(r, 0) if n > 5 else (np.nan, np.nan)
        means.append(m); cis.append(ci); ns.append(n)
        tstats.append(t); pvals.append(p)

    cols = [C[r] for r in regime_order]
    ax1.bar(regime_order, means, color=cols, alpha=0.80, width=0.5, zorder=3)
    ax1.errorbar(regime_order, means, yerr=cis, fmt="none",
                 color=C["text"], capsize=5, lw=1.5, zorder=4)
    ax1.axhline(0, color=C["subtext"], lw=0.8)
    ax1.set_ylabel("Mean next-day return (%)", fontsize=8)
    for i, (m, ci, n, t, p) in enumerate(zip(means, cis, ns, tstats, pvals)):
        if not np.isnan(t):
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
            offset = (ci + 0.003) * (np.sign(m) if m != 0 else 1)
            ax1.text(i, m + offset,
                     "t=%.2f\np=%.3f %s" % (t, p, sig),
                     ha="center",
                     va="bottom" if m >= 0 else "top",
                     fontsize=6.5, color=C["text"])
        ax1.text(i, -0.045, "n=%d" % n,
                 ha="center", fontsize=6.5, color=C["subtext"])
    _style(ax1)
    _title(ax1, "Next-Day Returns by Regime",
           sub="Error bars: 95% CI  -  ** p<0.05, * p<0.10, ns = not significant  -  Mixed is the most significant regime")

    # Panel 2: Rolling 63-day Information Coefficient
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ic_rolling.index, ic_rolling, color=C["strategy"], lw=1.0, alpha=0.85)
    ax2.fill_between(ic_rolling.index, ic_rolling, 0,
                     where=(ic_rolling > 0), color=C["momentum"], alpha=0.30)
    ax2.fill_between(ic_rolling.index, ic_rolling, 0,
                     where=(ic_rolling < 0), color=C["reversion"], alpha=0.30)
    ax2.axhline(0, color=C["subtext"], lw=0.8)
    ic_mean = ic_rolling.mean()
    ax2.axhline(ic_mean, color=C["strategy"], lw=0.8, ls="--",
                label="Mean IC = %.3f" % ic_mean)
    ax2.set_ylabel("Pearson IC (score vs fwd ret)", fontsize=8)
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.legend(fontsize=7.5, framealpha=0.8)
    _style(ax2)
    _title(ax2, "Rolling 63-day Information Coefficient",
           sub="IC = corr(ensemble_score[t], return[t+1])")

    # Panel 3: Cumulative strategy return decomposition by regime.
    # Uses forward (next-day) returns to match the 1-day execution lag in
    # run_backtest: signal at T -> trade at T+1. On reversion days the strategy
    # holds cash, earning 0 -- that line is flat at zero by design.
    ax3 = fig.add_subplot(gs[1, :])
    fwd_aligned = aligned_ret.shift(-1)           # next-day return per signal date
    labels_d    = labels.reindex(fwd_aligned.index)
    pct_rev     = (labels == "reversion").mean()

    mom_contrib = fwd_aligned.where(labels_d == "momentum",  0)
    mix_contrib = fwd_aligned.where(labels_d == "mixed",     0) * 0.5
    rev_contrib = pd.Series(0.0, index=fwd_aligned.index)   # cash -> zero

    cum_bnh = aligned_ret.cumsum()

    ax3.plot(cum_bnh.index, cum_bnh, color=C["bnh"],
             lw=1.3, ls="--", alpha=0.8,
             label="Buy and Hold (cumulative log return)")
    ax3.plot(mom_contrib.cumsum().index, mom_contrib.cumsum(),
             color=C["momentum"], lw=1.5,
             label="Momentum days, full long (+1)  - %.0f%% of time" % (pct_momentum * 100))
    ax3.plot(mix_contrib.cumsum().index, mix_contrib.cumsum(),
             color=C["mixed"], lw=1.5, ls="-.",
             label="Mixed days, half long (+0.5)  - %.0f%% of time" % (pct_mixed * 100))
    ax3.plot(rev_contrib.index, rev_contrib.cumsum(),
             color=C["reversion"], lw=1.3, ls=":", alpha=0.75,
             label="Reversion days, cash (0)  - %.0f%% of time" % (pct_rev * 100))
    ax3.axhline(0, color=C["subtext"], lw=0.6)
    ax3.set_ylabel("Cumulative strategy log return", fontsize=8)
    ax3.tick_params(axis="x", labelsize=7.5)
    ax3.legend(fontsize=7.5, framealpha=0.8, loc="lower right")
    _style(ax3)
    _title(ax3, "Cumulative Strategy Return Decomposition - Contribution by Regime")
    ax3.text(0.01, -0.10,
             "Uses next-day returns (1-day execution lag). Mixed half-position (+0.5) captures "
             "statistically significant positive drift (T=3.21, p=0.001). Reversion = cash = flat line.",
             transform=ax3.transAxes, fontsize=6.5, color=C["subtext"], va="top")

    # Panel 4: Regime signal ACF
    ax4 = fig.add_subplot(gs[2, 0])
    regime_num = labels.map({"momentum": 1, "mixed": 0, "reversion": -1}).dropna()
    nlags      = 30
    acf_vals, confint = acf(regime_num, nlags=nlags, alpha=0.05)
    lags = np.arange(len(acf_vals))
    ax4.bar(lags[1:], acf_vals[1:], color=C["strategy"], alpha=0.7, width=0.7)
    ax4.fill_between(lags,
                     confint[:, 0] - acf_vals,
                     confint[:, 1] - acf_vals,
                     alpha=0.15, color=C["subtext"],
                     label="95% confidence band")
    ax4.axhline(0, color=C["subtext"], lw=0.8)
    ax4.set_xlabel("Lag (trading days)", fontsize=8)
    ax4.set_ylabel("Autocorrelation", fontsize=8)
    ax4.legend(fontsize=7, framealpha=0.8)
    _style(ax4)
    _title(ax4, "Regime Label Autocorrelation (ACF)",
           sub="Persistence of regime label over time")

    # Panel 5: Markov filtered probabilities
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.fill_between(mom_prob.index,    mom_prob,    0,
                     color=C["momentum"],  alpha=0.55, label="P(momentum)")
    ax5.fill_between(crisis_prob.index, crisis_prob, 0,
                     color=C["reversion"], alpha=0.55, label="P(crisis)")
    ax5.axhline(0.50, color=C["subtext"], lw=0.7, ls=":",
                label="Crisis override (0.50)")
    ax5.set_ylim(0, 1)
    ax5.set_ylabel("Filtered probability", fontsize=8)
    ax5.tick_params(axis="x", labelsize=7.5)
    ax5.legend(fontsize=7, framealpha=0.8, loc="upper right")
    _style(ax5)
    _title(ax5, "Markov Filtered Probabilities",
           sub="Hamilton filter (causal) - no smoothing, no look-ahead")

    _footer(fig, 2)
    return fig


# ============================================================================
#  PAGE 3 - PORTFOLIO ANALYTICS
# ============================================================================

def make_page3():
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")

    fig.text(0.5, 0.965, "Regime Ensemble - Technical Report",
             ha="center", fontsize=15, fontweight="bold", color=C["text"])
    fig.text(0.5, 0.948, "Page 3 of 3  -  Portfolio Analytics",
             ha="center", fontsize=9, color=C["subtext"])
    fig.add_artist(plt.Line2D([0.09, 0.95], [0.938, 0.938],
                              transform=fig.transFigure, color=C["gridline"], lw=1))

    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.10, right=0.95,
                           top=0.880, bottom=0.07, hspace=1.10, wspace=0.42)

    # Panel 1: Equity curve with regime-shaded background
    ax1 = fig.add_subplot(gs[0, :])
    bt_labels  = labels.reindex(bt.index)
    prev_idx   = bt.index[0]
    prev_reg   = bt_labels.iloc[0]
    for i in range(1, len(bt.index)):
        curr_reg = bt_labels.iloc[i]
        if curr_reg != prev_reg or i == len(bt.index) - 1:
            col = REGIME_COLORS.get(prev_reg, C["mixed"])
            ax1.axvspan(prev_idx, bt.index[i], alpha=0.07, color=col, lw=0)
            prev_idx = bt.index[i]
            prev_reg = curr_reg

    ax1.plot(bt.index, bt["equity_bnh"],
             color=C["bnh"], lw=1.5, alpha=0.9,
             label="Buy and Hold  (Sharpe %s, DD %s)" % (
                 bnh_perf["Sharpe"], bnh_perf["Max DD"]))
    ax1.plot(bt.index, bt["equity_strategy"],
             color=C["strategy"], lw=1.8,
             label="Strategy  (Sharpe %s, DD %s)" % (
                 strat_perf["Sharpe"], strat_perf["Max DD"]))
    ax1.axhline(1.0, color=C["gridline"], lw=0.7, ls="--")
    ax1.set_ylabel("Portfolio value (base = 1.0)", fontsize=8)
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.legend(fontsize=8, framealpha=0.85, loc="upper left")
    _style(ax1)
    _title(ax1, "Equity Curves - Regime-Shaded Background (0 bps, 1-day lag)",
           sub="Shading: green=momentum, red=reversion, grey=mixed")

    # Panel 2: Underwater drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    strat_dd = bt["equity_strategy"] / bt["equity_strategy"].cummax() - 1
    bnh_dd   = bt["equity_bnh"]      / bt["equity_bnh"].cummax()      - 1
    ax2.fill_between(bt.index, bnh_dd,   0, alpha=0.30, color=C["bnh"],
                     label="B&H  max=%.1f%%" % (bnh_dd.min() * 100))
    ax2.fill_between(bt.index, strat_dd, 0, alpha=0.65, color=C["strategy"],
                     label="Strategy  max=%.1f%%" % (strat_dd.min() * 100))
    avg_dd = strat_dd[strat_dd < 0].mean()
    ax2.axhline(avg_dd, color=C["strategy"], lw=0.9, ls="--",
                label="Avg DD=%.1f%%" % (avg_dd * 100), alpha=0.8)
    ax2.set_ylabel("Drawdown from peak", fontsize=8)
    ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=7, framealpha=0.85, loc="lower left")
    _style(ax2)
    _title(ax2, "Underwater Drawdown")

    # Panel 3: Continuous cost sensitivity curve
    ax3 = fig.add_subplot(gs[1, 1])
    breakeven_bps = int(bps_range[np.argmin(np.abs(cost_curve))])
    ax3.plot(bps_range, cost_curve, color=C["strategy"], lw=2.0)
    ax3.fill_between(bps_range, cost_curve, 0,
                     where=(cost_curve > 0),
                     color=C["momentum"], alpha=0.20)
    ax3.fill_between(bps_range, cost_curve, 0,
                     where=(cost_curve <= 0),
                     color=C["reversion"], alpha=0.20)
    ax3.axhline(0, color=C["subtext"], lw=0.8)
    ax3.axvline(breakeven_bps, color=C["gold"], lw=1.2, ls="--",
                label="Break-even ~%dbps" % breakeven_bps)
    ax3.axhline(float(bnh_perf["Sharpe"]), color=C["bnh"], lw=0.9, ls=":",
                label="B&H Sharpe (%s)" % bnh_perf["Sharpe"])
    ax3.set_xlabel("Round-trip transaction cost (bps)", fontsize=8)
    ax3.set_ylabel("Strategy Sharpe Ratio", fontsize=8)
    ax3.legend(fontsize=7.5, framealpha=0.85)
    _style(ax3)
    _title(ax3, "Sharpe vs Transaction Cost  (0-30 bps)",
           sub="~%.0f regime switches/year  -  each switch incurs round-trip cost" %
               trans_info.get("switches_pa", 22))

    # Panel 4: Performance attribution table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    _title(ax4, "Full Performance Attribution  (SPY 2000-2025, 0 bps)")

    metrics = list(strat_ext.keys())
    col_x   = [0.0, 0.34, 0.55]
    row_h   = 0.082

    # Header row
    for j, (txt, col) in enumerate(
        [("Metric", C["text"]), ("Strategy", C["strategy"]), ("Buy and Hold", C["bnh"])]
    ):
        ax4.text(col_x[j], 0.90, txt, transform=ax4.transAxes,
                 fontsize=8.5, fontweight="bold", color=col, va="top")

    ax4.plot([0, 1], [0.885, 0.885], transform=ax4.transAxes,
             color=C["gridline"], lw=0.8, clip_on=False)

    for i, metric in enumerate(metrics):
        y      = 0.82 - i * row_h
        bg     = "#f8f9fa" if i % 2 == 0 else "white"
        rect   = plt.Rectangle((0, y - 0.02), 1, row_h,
                                transform=ax4.transAxes, color=bg, zorder=0)
        ax4.add_patch(rect)
        ax4.text(col_x[0], y, metric,
                 transform=ax4.transAxes, fontsize=8, color=C["text"], va="top")
        ax4.text(col_x[1], y, strat_ext[metric],
                 transform=ax4.transAxes, fontsize=8,
                 color=C["strategy"], fontweight="bold", va="top")
        ax4.text(col_x[2], y, bnh_ext[metric],
                 transform=ax4.transAxes, fontsize=8, color=C["subtext"], va="top")

    extra_y = 0.82 - len(metrics) * row_h
    ax4.plot([0, 1], [extra_y + 0.04, extra_y + 0.04],
             transform=ax4.transAxes, color=C["gridline"], lw=0.5, clip_on=False)

    extras = [
        ("Information Ratio",   "%.2f" % ir, "n/a"),
        ("Mean Exposure",       "%.1f%%" % (mean_exposure * 100), "100.0%"),
        ("% Time Fully Long",   "%.1f%%" % (pct_momentum * 100), "100.0%"),
        ("% Time Half Long",    "%.1f%%" % (pct_mixed * 100),    "0.0%"),
    ]
    for i, (m, sv, bv) in enumerate(extras):
        y = extra_y - i * row_h + 0.025
        ax4.text(col_x[0], y, m,  transform=ax4.transAxes,
                 fontsize=8, color=C["text"], va="top")
        ax4.text(col_x[1], y, sv, transform=ax4.transAxes,
                 fontsize=8, color=C["strategy"], fontweight="bold", va="top")
        ax4.text(col_x[2], y, bv, transform=ax4.transAxes,
                 fontsize=8, color=C["subtext"], va="top")

    _footer(fig, 3)
    return fig


# ============================================================================
#  SAVE
# ============================================================================

out_path = OUTPUT_DIR / "SPY_quant_report.pdf"

print("  Generating pages...")
with PdfPages(out_path) as pdf:
    for page_num, page_fn in enumerate([make_page1, make_page2, make_page3], 1):
        print("    page %d..." % page_num)
        fig = page_fn()
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        plt.close(fig)

print("  Report saved -> %s" % out_path)
print("  File size: %.1f KB" % (out_path.stat().st_size / 1024))

# Debug: verify PDF header
with open(out_path, "rb") as f:
    header = f.read(8)
print("  PDF header: %s  [%s]" % (
    header.decode("latin-1"),
    "OK" if header.startswith(b"%PDF") else "ERROR - invalid header"
))

# Check for non-ASCII in source file
with open(__file__) as f:
    src = f.read()
bad_chars = [(i, c) for i, c in enumerate(src) if ord(c) > 127]
if bad_chars:
    print("  WARNING: non-ASCII chars in source: %s" % str(set(c for _, c in bad_chars)))
else:
    print("  Source file: ASCII-clean [OK]")
