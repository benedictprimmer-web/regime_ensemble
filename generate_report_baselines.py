#!/usr/bin/env python3
"""
Baseline Comparison Report — Three-Model Test Ladder
=====================================================
Compares three progressively richer regime strategies out-of-sample:

    Model 1 — Purely simple:    200d MA + 30d slope + VIX bucket
    Model 2 — Simple + stress:  Model 1 + rvol overlay + drawdown + volume
    Model 3 — HMM ensemble:     Current full system (Geometric + Markov k=3)

Paul's hypothesis: Model 1 will be surprisingly strong, Model 2 may be
the sweet spot, Model 3 may add only modest improvement.

Usage:
    python generate_report_baselines.py
    python generate_report_baselines.py --ticker SPY --from 2000-01-01 --to 2025-01-01
    python generate_report_baselines.py --cost-bps 5
    python generate_report_baselines.py --no-vix   # skip VIX fetch (Model 1 ignores VIX)

Outputs (saved to outputs/):
    {ticker}_{from_year}_{to_year}_baseline_comparison.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.data      import fetch_daily_bars, log_returns, vix_levels
from src.baselines import model1_signal, model2_signal
from src.geometric import geometric_signal, compute_thresholds
from src.markov    import fit_markov3
from src.ensemble  import ensemble_score, regime_labels
from src.backtest  import run_backtest, compute_stats

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Colour palette — consistent with run.py
COLORS = {
    "bnh":     "#2c3e50",
    "model1":  "#8e44ad",
    "model2":  "#e67e22",
    "model3":  "#2980b9",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _regime_from_score(score: pd.Series) -> pd.Series:
    """Convert a continuous [0,1] score to a log_return-compatible DataFrame for run_backtest."""
    return score


def _format_stats(raw: dict) -> dict:
    """Format raw stats dict for printing."""
    s = raw
    return {
        "CAGR":    f"{s['CAGR']*100:+.1f}%",
        "Sharpe":  f"{s['Sharpe']:.2f}",
        "Max DD":  f"{s['Max DD']*100:.1f}%",
        "T-stat":  f"{s['T-stat']:.2f}",
        "P-value": f"{s['P-value']:.3f}",
    }


def _print_table(results: list[dict]) -> None:
    """Print a side-by-side comparison table."""
    metrics = ["CAGR", "Sharpe", "Max DD", "T-stat", "P-value"]
    col_w   = 14
    label_w = 28

    header = f"{'Strategy':<{label_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print()
    print("=" * (label_w + col_w * len(metrics)))
    print("  Baseline Comparison — Full-Sample (in-sample overview)")
    print("=" * (label_w + col_w * len(metrics)))
    print(header)
    print("-" * (label_w + col_w * len(metrics)))
    for row in results:
        label  = row["label"]
        values = "".join(f"{row[m]:>{col_w}}" for m in metrics)
        print(f"{label:<{label_w}}{values}")
    print("=" * (label_w + col_w * len(metrics)))
    print()


# ── Chart ──────────────────────────────────────────────────────────────────────

def plot_comparison(
    bt_bnh: pd.DataFrame,
    bt_m1: pd.DataFrame,
    bt_m2: pd.DataFrame,
    bt_m3: pd.DataFrame,
    ticker: str,
    from_date: str,
    to_date: str,
    run_prefix: str,
    cost_bps: float,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    def _eq(bt):
        return np.exp(bt["strategy_return"].cumsum())

    eq_bnh = np.exp(bt_bnh["bnh_return"].cumsum())
    eq_m1  = _eq(bt_m1)
    eq_m2  = _eq(bt_m2)
    eq_m3  = _eq(bt_m3)

    ax1.plot(eq_bnh.index, eq_bnh, color=COLORS["bnh"],    linewidth=1.5, label="Buy & Hold",            alpha=0.8)
    ax1.plot(eq_m1.index,  eq_m1,  color=COLORS["model1"], linewidth=1.5, label="Model 1: Simple rule",  alpha=0.9)
    ax1.plot(eq_m2.index,  eq_m2,  color=COLORS["model2"], linewidth=1.5, label="Model 2: + Stress",     alpha=0.9)
    ax1.plot(eq_m3.index,  eq_m3,  color=COLORS["model3"], linewidth=1.5, label="Model 3: HMM Ensemble", alpha=0.9)
    ax1.axhline(1.0, color="#bdc3c7", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Equity (base = 1.0)", fontsize=9)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_title(
        f"Three-Model Baseline Comparison — {ticker} {from_date[:4]}–{to_date[:4]}  "
        f"(cost = {cost_bps:.0f} bps)",
        fontsize=11, pad=8,
    )

    def _dd(eq):
        return eq / eq.cummax() - 1

    ax2.fill_between(eq_m3.index,  _dd(eq_m3),  0, alpha=0.35, color=COLORS["model3"],  label="Model 3")
    ax2.fill_between(eq_m2.index,  _dd(eq_m2),  0, alpha=0.35, color=COLORS["model2"],  label="Model 2")
    ax2.fill_between(eq_m1.index,  _dd(eq_m1),  0, alpha=0.35, color=COLORS["model1"],  label="Model 1")
    ax2.fill_between(eq_bnh.index, _dd(eq_bnh), 0, alpha=0.20, color=COLORS["bnh"],     label="Buy & Hold")
    ax2.set_ylabel("Drawdown", fontsize=9)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.tick_params(axis="x", labelsize=8)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    path = OUTPUT_DIR / f"{run_prefix}_baseline_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Sharpe comparison chart ───────────────────────────────────────────────────

def _rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Annualised rolling Sharpe ratio."""
    mu  = returns.rolling(window).mean() * 252
    sig = returns.rolling(window).std() * np.sqrt(252)
    return (mu / sig.replace(0, np.nan)).rename(returns.name)


def plot_sharpe_comparison(
    bt_bnh: pd.DataFrame,
    bt_m1: pd.DataFrame,
    bt_m2: pd.DataFrame,
    bt_m3: pd.DataFrame,
    ticker: str,
    from_date: str,
    to_date: str,
    run_prefix: str,
    cost_bps: float,
) -> None:
    labels  = ["Buy & Hold", "Model 1\nSimple rule", "Model 2\n+ Stress", "Model 3\nHMM Ensemble"]
    cols    = ["bnh_return", "strategy_return", "strategy_return", "strategy_return"]
    bts     = [bt_bnh,        bt_m1,              bt_m2,              bt_m3]
    colors  = [COLORS["bnh"], COLORS["model1"],   COLORS["model2"],   COLORS["model3"]]

    sharpes = []
    for bt, col in zip(bts, cols):
        r = bt[col].dropna()
        s = (r.mean() * 252) / (r.std() * np.sqrt(252))
        sharpes.append(s)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    # Left: bar chart of overall Sharpe
    bars = ax1.bar(labels, sharpes, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    ax1.axhline(0, color="#bdc3c7", linewidth=0.8)
    ax1.set_ylabel("Annualised Sharpe Ratio", fontsize=9)
    ax1.set_title("Overall Sharpe Ratio", fontsize=10, pad=8)
    ax1.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, sharpes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax1.set_ylim(bottom=min(0, min(sharpes)) - 0.05,
                 top=max(sharpes) + 0.15)

    # Right: rolling 252-day Sharpe over time
    rs_bnh = _rolling_sharpe(bt_bnh["bnh_return"].rename("Buy & Hold"))
    rs_m1  = _rolling_sharpe(bt_m1["strategy_return"].rename("Model 1"))
    rs_m2  = _rolling_sharpe(bt_m2["strategy_return"].rename("Model 2"))
    rs_m3  = _rolling_sharpe(bt_m3["strategy_return"].rename("Model 3"))

    ax2.plot(rs_bnh.index, rs_bnh, color=COLORS["bnh"],    linewidth=1.3, alpha=0.7, label="Buy & Hold")
    ax2.plot(rs_m1.index,  rs_m1,  color=COLORS["model1"], linewidth=1.3, alpha=0.85, label="Model 1: Simple rule")
    ax2.plot(rs_m2.index,  rs_m2,  color=COLORS["model2"], linewidth=1.3, alpha=0.85, label="Model 2: + Stress")
    ax2.plot(rs_m3.index,  rs_m3,  color=COLORS["model3"], linewidth=1.3, alpha=0.85, label="Model 3: HMM Ensemble")
    ax2.axhline(0, color="#bdc3c7", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Rolling 1-Year Sharpe", fontsize=9)
    ax2.set_title("Rolling Annual Sharpe Ratio (252-day window)", fontsize=10, pad=8)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        f"Sharpe Ratio Comparison — {ticker} {from_date[:4]}–{to_date[:4]}  (cost = {cost_bps:.0f} bps)",
        fontsize=11, y=1.01,
    )
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()

    path = OUTPUT_DIR / f"{run_prefix}_sharpe_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Crisis vs calm analysis ───────────────────────────────────────────────────

def _crisis_mask(prices: pd.Series, drawdown_threshold: float = -0.15) -> pd.Series:
    """
    Boolean mask: True on days when price is >threshold below its 252-day high.
    Default -0.15 captures sustained drawdown periods (2002, 2008, 2020 etc.)
    without flagging every minor dip.
    """
    peak = prices.rolling(252, min_periods=63).max()
    dd   = prices / peak - 1
    return (dd < drawdown_threshold).rename("in_crisis")


def plot_crisis_vs_calm(
    bt_bnh: pd.DataFrame,
    bt_m1: pd.DataFrame,
    bt_m2: pd.DataFrame,
    bt_m3: pd.DataFrame,
    prices: pd.Series,
    ticker: str,
    from_date: str,
    to_date: str,
    run_prefix: str,
    cost_bps: float,
    drawdown_threshold: float = -0.15,
) -> None:
    """
    Two-panel chart:
        Top:    Cumulative alpha (strategy equity / B&H equity) for each model.
                Rising = outperforming B&H; flat = matching; falling = underperforming.
                Shaded regions = crisis periods (drawdown > threshold).
        Bottom: Bar chart of Sharpe ratio split into crisis vs calm periods,
                showing where each model earns its edge.
    """
    crisis = _crisis_mask(prices, drawdown_threshold).reindex(bt_bnh.index, method="ffill").fillna(False)

    models = [
        ("Model 1", bt_m1, COLORS["model1"]),
        ("Model 2", bt_m2, COLORS["model2"]),
        ("Model 3", bt_m3, COLORS["model3"]),
    ]

    # ── Cumulative alpha ──────────────────────────────────────────────
    bnh_eq = np.exp(bt_bnh["bnh_return"].cumsum())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=False,
    )

    for label, bt, color in models:
        strat_eq = np.exp(bt["strategy_return"].cumsum())
        alpha    = (strat_eq / bnh_eq.reindex(strat_eq.index)).dropna()
        ax1.plot(alpha.index, alpha, color=color, linewidth=1.4, label=label, alpha=0.9)

    ax1.axhline(1.0, color="#bdc3c7", linewidth=0.9, linestyle="--", label="Parity with B&H")

    # Shade crisis periods
    crisis_aligned = crisis.reindex(bnh_eq.index, method="ffill").fillna(False)
    in_crisis = False
    start = None
    for date, flag in crisis_aligned.items():
        if flag and not in_crisis:
            start = date
            in_crisis = True
        elif not flag and in_crisis:
            ax1.axvspan(start, date, alpha=0.10, color="#c0392b", zorder=0)
            in_crisis = False
    if in_crisis:
        ax1.axvspan(start, crisis_aligned.index[-1], alpha=0.10, color="#c0392b", zorder=0)

    ax1.set_ylabel("Strategy equity ÷ B&H equity", fontsize=9)
    ax1.set_title(
        f"Cumulative Alpha vs Buy & Hold — {ticker} {from_date[:4]}–{to_date[:4]}\n"
        f"Shaded = crisis periods (drawdown > {abs(drawdown_threshold)*100:.0f}%)",
        fontsize=10, pad=8,
    )
    ax1.legend(fontsize=8, loc="upper left")
    ax1.tick_params(axis="x", labelsize=8)

    # ── Sharpe split: crisis vs calm ──────────────────────────────────
    bar_labels  = []
    crisis_sharpes = []
    calm_sharpes   = []

    bnh_ret = bt_bnh["bnh_return"]
    bnh_crisis_r = bnh_ret[crisis.reindex(bnh_ret.index, fill_value=False)].dropna()
    bnh_calm_r   = bnh_ret[~crisis.reindex(bnh_ret.index, fill_value=False)].dropna()

    def _sharpe(r):
        if len(r) < 20 or r.std() == 0:
            return np.nan
        return (r.mean() * 252) / (r.std() * np.sqrt(252))

    # B&H reference bars
    bar_labels.append("Buy &\nHold")
    crisis_sharpes.append(_sharpe(bnh_crisis_r))
    calm_sharpes.append(_sharpe(bnh_calm_r))

    for label, bt, _ in models:
        r         = bt["strategy_return"]
        cr        = crisis.reindex(r.index, fill_value=False)
        crisis_r  = r[cr].dropna()
        calm_r    = r[~cr].dropna()
        bar_labels.append(label.replace(" ", "\n", 1))
        crisis_sharpes.append(_sharpe(crisis_r))
        calm_sharpes.append(_sharpe(calm_r))

    x      = np.arange(len(bar_labels))
    width  = 0.35
    bars_c = ax2.bar(x - width/2, crisis_sharpes, width, label="Crisis periods",
                     color="#c0392b", alpha=0.75, edgecolor="white")
    bars_n = ax2.bar(x + width/2, calm_sharpes,   width, label="Calm periods",
                     color="#27ae60", alpha=0.75, edgecolor="white")

    ax2.axhline(0, color="#bdc3c7", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, fontsize=8)
    ax2.set_ylabel("Annualised Sharpe", fontsize=9)
    ax2.set_title(
        f"Sharpe ratio: crisis vs calm  "
        f"(crisis = drawdown > {abs(drawdown_threshold)*100:.0f}%,  "
        f"{crisis_aligned.sum()} days = {crisis_aligned.mean()*100:.0f}% of sample)",
        fontsize=9, pad=6,
    )
    ax2.legend(fontsize=8)

    for bar in list(bars_c) + list(bars_n):
        h = bar.get_height()
        if not np.isnan(h):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.03 if h >= 0 else -0.10),
                f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout(pad=2.0)

    path = OUTPUT_DIR / f"{run_prefix}_crisis_vs_calm.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Capture ratio analysis ────────────────────────────────────────────────────

def _capture_stats(strat_ret: pd.Series, bnh_ret: pd.Series) -> dict:
    """
    Compute upside/downside capture ratios and alpha attribution.

    Upside capture:   mean(strategy | bnh > 0) / mean(bnh | bnh > 0)
    Downside capture: mean(strategy | bnh < 0) / mean(bnh | bnh < 0)
    Values < 1.0 on downside = better crisis protection.
    Values > 1.0 on upside   = captures full bull market.

    Alpha attribution: sum of (strategy - bnh) in up days vs down days,
    expressed as annualised bps.
    """
    aligned = pd.concat([strat_ret.rename("s"), bnh_ret.rename("b")], axis=1).dropna()
    up   = aligned[aligned["b"] > 0]
    down = aligned[aligned["b"] < 0]

    def _cap(subset, col):
        bnh_mean = subset["b"].mean()
        return subset[col].mean() / bnh_mean if bnh_mean != 0 else np.nan

    upside_cap   = _cap(up,   "s")
    downside_cap = _cap(down, "s")

    # Alpha attribution in annualised bps
    ann_factor   = 252 * 10_000   # convert daily log-return to annual bps
    alpha_up     = (aligned.loc[up.index,   "s"] - aligned.loc[up.index,   "b"]).sum() / len(aligned) * ann_factor
    alpha_down   = (aligned.loc[down.index, "s"] - aligned.loc[down.index, "b"]).sum() / len(aligned) * ann_factor
    alpha_total  = alpha_up + alpha_down

    return {
        "upside_cap":   upside_cap,
        "downside_cap": downside_cap,
        "alpha_up_bps":    alpha_up,
        "alpha_down_bps":  alpha_down,
        "alpha_total_bps": alpha_total,
        "n_up":   len(up),
        "n_down": len(down),
    }


def plot_capture_analysis(
    bt_bnh: pd.DataFrame,
    bt_m1: pd.DataFrame,
    bt_m2: pd.DataFrame,
    bt_m3: pd.DataFrame,
    ticker: str,
    from_date: str,
    to_date: str,
    run_prefix: str,
    cost_bps: float,
) -> None:
    bnh_ret = bt_bnh["bnh_return"]
    models  = [
        ("Model 1", bt_m1, COLORS["model1"]),
        ("Model 2", bt_m2, COLORS["model2"]),
        ("Model 3", bt_m3, COLORS["model3"]),
    ]

    stats = {label: _capture_stats(bt["strategy_return"], bnh_ret)
             for label, bt, _ in models}

    # ── Print table ───────────────────────────────────────────────────
    print()
    print("  Capture Ratio & Alpha Attribution")
    print(f"  {'':18s}  {'Upside cap':>12}  {'Downside cap':>13}  {'Alpha (up)':>12}  {'Alpha (dn)':>12}  {'Alpha total':>12}")
    print("  " + "-" * 80)
    for label, s in stats.items():
        print(
            f"  {label:<18}  {s['upside_cap']:>11.1%}  {s['downside_cap']:>12.1%}  "
            f"{s['alpha_up_bps']:>+11.0f}  {s['alpha_down_bps']:>+11.0f}  {s['alpha_total_bps']:>+11.0f}"
        )
        print(f"  {'':18s}  (n up={s['n_up']}, n down={s['n_down']})")
    print()

    # ── Chart ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_scatter  = fig.add_subplot(gs[0, 0])   # upside vs downside scatter
    ax_asym     = fig.add_subplot(gs[0, 1])   # asymmetry ratio bars
    ax_rolling  = fig.add_subplot(gs[1, :])   # rolling capture ratio over time

    bar_labels = [m[0] for m in models]
    colors     = [m[2] for m in models]

    up_caps   = [stats[l]["upside_cap"]   for l in bar_labels]
    down_caps = [stats[l]["downside_cap"] for l in bar_labels]
    asym      = [stats[l]["downside_cap"] / stats[l]["upside_cap"] for l in bar_labels]

    # ── Panel 1: Scatter — upside vs downside capture ─────────────────
    # Diagonal = perfect symmetry. Ideal zone = top-left (high up, low down).
    lim = max(max(up_caps), max(down_caps)) + 0.08
    ax_scatter.plot([0, lim], [0, lim], color="#bdc3c7", linewidth=1.2,
                    linestyle="--", zorder=0, label="Symmetric line\n(no asymmetry)")

    # Shade the "ideal" zone: upside > downside
    ax_scatter.fill_between([0, lim], [0, 0], [0, lim],
                             alpha=0.06, color="#27ae60", zorder=0)
    ax_scatter.fill_between([0, lim], [0, lim], [lim, lim],
                             alpha=0.06, color="#c0392b", zorder=0)

    # B&H reference point
    ax_scatter.scatter([1.0], [1.0], color=COLORS["bnh"], s=120, zorder=5,
                       marker="s", label="Buy & Hold (100%, 100%)")

    for label, uc, dc, color in zip(bar_labels, up_caps, down_caps, colors):
        ax_scatter.scatter(uc, dc, color=color, s=180, zorder=6, marker="o")
        ax_scatter.annotate(
            label, (uc, dc),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8.5, color=color, fontweight="bold",
        )

    ax_scatter.set_xlabel("Upside capture (% of B&H gains captured)", fontsize=9)
    ax_scatter.set_ylabel("Downside capture (% of B&H losses taken)", fontsize=9)
    ax_scatter.set_title("Upside vs Downside Capture\nIdeal = below diagonal (protect more than you sacrifice)", fontsize=9, pad=8)
    ax_scatter.set_xlim(0, lim)
    ax_scatter.set_ylim(0, lim)
    ax_scatter.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_scatter.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_scatter.legend(fontsize=7.5, loc="upper left")

    ax_scatter.text(lim * 0.55, lim * 0.08, "IDEAL ZONE\n(better downside protection)", fontsize=7,
                    color="#27ae60", alpha=0.7, style="italic")
    ax_scatter.text(lim * 0.05, lim * 0.88, "WORSE ZONE\n(more downside than upside)", fontsize=7,
                    color="#c0392b", alpha=0.7, style="italic")

    # ── Panel 2: Asymmetry ratio bars ─────────────────────────────────
    # downside_cap / upside_cap: 1.0 = symmetric, <1.0 = good asymmetry
    x = np.arange(len(bar_labels))
    bars = ax_asym.bar(x, asym, color=colors, width=0.5, edgecolor="white", alpha=0.85)
    ax_asym.axhline(1.0, color="#c0392b", linewidth=1.5, linestyle="--",
                    label="1.0 = no asymmetry (symmetric reducer)")
    ax_asym.axhline(0.7, color="#27ae60", linewidth=1.0, linestyle=":",
                    alpha=0.7, label="0.7 = target (30% better downside)")
    ax_asym.set_xticks(x)
    ax_asym.set_xticklabels(bar_labels, fontsize=9)
    ax_asym.set_ylabel("Downside capture ÷ Upside capture", fontsize=9)
    ax_asym.set_title("Asymmetry Ratio\n< 1.0 = genuine crisis protection, = 1.0 = pure exposure reducer", fontsize=9, pad=8)
    ax_asym.legend(fontsize=7.5)
    ax_asym.set_ylim(0, max(asym) + 0.15)
    for bar, val in zip(bars, asym):
        ax_asym.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── Panel 3: Rolling 63-day capture ratio over time ───────────────
    # Shows whether asymmetry improves during different market conditions
    window = 63
    for label, bt, color in models:
        r_s = bt["strategy_return"]
        r_b = bnh_ret.reindex(r_s.index)

        def _rolling_capture_ratio(strat, bnh, w):
            up_strat = strat.where(bnh > 0).rolling(w, min_periods=20).mean()
            up_bnh   = bnh.where(bnh > 0).rolling(w, min_periods=20).mean()
            dn_strat = strat.where(bnh < 0).rolling(w, min_periods=20).mean()
            dn_bnh   = bnh.where(bnh < 0).rolling(w, min_periods=20).mean()
            up_cap   = (up_strat / up_bnh.replace(0, np.nan)).clip(0, 2)
            dn_cap   = (dn_strat / dn_bnh.replace(0, np.nan)).clip(0, 2)
            return (dn_cap / up_cap.replace(0, np.nan)).clip(0, 2)

        ratio_ts = _rolling_capture_ratio(r_s, r_b, window)
        ax_rolling.plot(ratio_ts.index, ratio_ts, color=color, linewidth=1.1,
                        alpha=0.8, label=label)

    ax_rolling.axhline(1.0, color="#c0392b", linewidth=1.2, linestyle="--",
                       alpha=0.7, label="1.0 = no asymmetry")
    ax_rolling.axhline(0.7, color="#27ae60", linewidth=1.0, linestyle=":",
                       alpha=0.6, label="0.7 = target")
    ax_rolling.set_ylabel("Rolling asymmetry ratio (dn cap ÷ up cap)", fontsize=9)
    ax_rolling.set_title(
        f"Rolling {window}-day Asymmetry Ratio over time  "
        f"— below 1.0 = model selectively protecting on down periods",
        fontsize=9, pad=8,
    )
    ax_rolling.legend(fontsize=8, loc="upper right")
    ax_rolling.tick_params(axis="x", labelsize=8)
    ax_rolling.set_ylim(0, 2.2)
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.suptitle(
        f"Capture Ratios & Alpha Attribution — {ticker} {from_date[:4]}–{to_date[:4]}  (cost = {cost_bps:.0f} bps)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    path = OUTPUT_DIR / f"{run_prefix}_capture_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Three-model baseline comparison report")
    parser.add_argument("--ticker",    default="SPY",        help="Ticker symbol (default SPY)")
    parser.add_argument("--from",      default="2000-01-01", dest="from_date", help="Start date")
    parser.add_argument("--to",        default="2025-01-01", dest="to_date",   help="End date")
    parser.add_argument("--cost-bps",  type=float, default=0, help="Round-trip transaction cost in bps (default 0)")
    parser.add_argument("--no-vix",    action="store_true",   help="Skip VIX fetch; Model 1 uses no VIX adjustment")
    args = parser.parse_args()

    ticker     = args.ticker
    from_date  = args.from_date
    to_date    = args.to_date
    cost_bps   = args.cost_bps
    run_prefix = f"{ticker}_{from_date[:4]}_{to_date[:4]}"

    print(f"\n{'='*60}")
    print(f"  Regime Ensemble — Baseline Comparison")
    print(f"  Ticker: {ticker}  |  {from_date} → {to_date}  |  cost: {cost_bps:.0f} bps")
    print(f"{'='*60}\n")

    # ── Fetch data ─────────────────────────────────────────────────────
    print("Fetching price data...")
    bars    = fetch_daily_bars(ticker, from_date, to_date)
    prices  = bars["close"].rename("close")
    returns = log_returns(bars)
    volume  = bars["volume"] if "volume" in bars.columns else None

    vix = None
    if not args.no_vix:
        print("Fetching VIX...")
        try:
            vix_bars = fetch_daily_bars("I:VIX", from_date, to_date)
            vix = vix_levels(vix_bars)
            print(f"  VIX fetched: {len(vix)} days")
        except Exception as e:
            print(f"  VIX fetch failed ({e}) — proceeding without VIX")
            vix = None

    # Align all series to the returns index
    prices  = prices.reindex(returns.index)
    if vix is not None:
        vix = vix.reindex(returns.index, method="ffill")
    if volume is not None:
        volume = volume.reindex(returns.index)

    # ── Model 1: Simple rule-based ─────────────────────────────────────
    print("\nBuilding Model 1 (simple rule: 200d MA + 30d slope + VIX)...")
    score_m1 = model1_signal(prices, vix=vix)

    # ── Model 2: Simple + stress overlays ──────────────────────────────
    print("Building Model 2 (simple + rvol/drawdown/volume overlays)...")
    score_m2 = model2_signal(prices, returns, vix=vix, volume=volume)

    # ── Model 3: Full HMM ensemble ─────────────────────────────────────
    print("\nFitting Model 3 (HMM ensemble — this may take ~30s)...")
    mom_prob, crisis_prob, choppy_prob, _ = fit_markov3(returns, verbose=True)

    geo = geometric_signal(returns)
    score_m3 = ensemble_score(geo, mom_prob, crisis_prob)
    labels_m3 = regime_labels(score_m3)

    # ── Backtests ──────────────────────────────────────────────────────
    print(f"\nRunning backtests (cost = {cost_bps:.0f} bps)...")

    # Buy & Hold (need a dummy regime that's always momentum)
    dummy_regime = pd.Series("momentum", index=returns.index, name="regime")
    bt_bnh = run_backtest(returns.rename("log_return"), dummy_regime, cost_bps=0)

    bt_m1 = run_backtest(returns.rename("log_return"), labels_m3,   # regime ignored — score used
                         cost_bps=cost_bps, score=score_m1.rename("signal"))
    bt_m2 = run_backtest(returns.rename("log_return"), labels_m3,
                         cost_bps=cost_bps, score=score_m2.rename("signal"))
    bt_m3 = run_backtest(returns.rename("log_return"), labels_m3,
                         cost_bps=cost_bps, score=score_m3.rename("signal"))

    # ── Stats table ────────────────────────────────────────────────────
    def _strat_stats(bt):
        from src.backtest import compute_stats as _cs
        return _cs(bt, raw=True)["Strategy (Long Only)"]

    rows = [
        {"label": "Buy & Hold",           **_format_stats({
            "CAGR":    np.exp(bt_bnh["bnh_return"].sum()) ** (252 / len(bt_bnh)) - 1,
            "Sharpe":  bt_bnh["bnh_return"].mean() * 252 / (bt_bnh["bnh_return"].std() * np.sqrt(252)),
            "Max DD":  (np.exp(bt_bnh["bnh_return"].cumsum()) / np.exp(bt_bnh["bnh_return"].cumsum()).cummax() - 1).min(),
            "T-stat":  0.0, "P-value": 1.0,
        })},
        {"label": "Model 1: Simple rule", **_format_stats(_strat_stats(bt_m1))},
        {"label": "Model 2: + Stress",    **_format_stats(_strat_stats(bt_m2))},
        {"label": "Model 3: HMM Ensemble",**_format_stats(_strat_stats(bt_m3))},
    ]

    # Fix B&H row properly using compute_stats
    bnh_raw = compute_stats(bt_bnh, raw=True)["Buy & Hold"]
    rows[0] = {"label": "Buy & Hold", **_format_stats(bnh_raw)}

    _print_table(rows)

    # ── Signal coverage summary ────────────────────────────────────────
    print("Signal coverage (% of days in each position):")
    for label, score in [("Model 1", score_m1), ("Model 2", score_m2), ("Model 3", score_m3)]:
        full  = (score >= 0.9).mean() * 100
        half  = ((score > 0.1) & (score < 0.9)).mean() * 100
        cash  = (score <= 0.1).mean() * 100
        print(f"  {label:<12}  full={full:.0f}%  mixed={half:.0f}%  cash={cash:.0f}%")
    print()

    # ── Chart ──────────────────────────────────────────────────────────
    print("Plotting equity curves...")
    plot_comparison(bt_bnh, bt_m1, bt_m2, bt_m3,
                    ticker, from_date, to_date, run_prefix, cost_bps)

    print("Plotting Sharpe comparison...")
    plot_sharpe_comparison(bt_bnh, bt_m1, bt_m2, bt_m3,
                           ticker, from_date, to_date, run_prefix, cost_bps)

    print("Plotting crisis vs calm analysis...")
    plot_crisis_vs_calm(bt_bnh, bt_m1, bt_m2, bt_m3, prices,
                        ticker, from_date, to_date, run_prefix, cost_bps)

    print("Plotting capture ratio analysis...")
    plot_capture_analysis(bt_bnh, bt_m1, bt_m2, bt_m3,
                          ticker, from_date, to_date, run_prefix, cost_bps)

    print("\nDone.")


if __name__ == "__main__":
    main()
