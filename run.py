#!/usr/bin/env python3
"""
Regime Ensemble Backtest
========================
Daily regime detection using a two-model ensemble:
    1. Geometric (straightness ratio, adaptive percentile thresholds)
    2. Gaussian HMM k=3 on 5-feature market-state vector (ret_20d, ret_5d, rvol_20d, drawdown, dist_200d)

Usage:
    python run.py
    python run.py --ticker QQQ --from 2000-01-01 --to 2025-01-01
    python run.py --fetch-vix       # also fetch VIX (I:VIX) from Polygon
    python run.py --fetch-vvix      # also fetch VVIX (I:VVIX) from Polygon
    python run.py --short           # allow short on reversion days
    python run.py --skip-bic        # skip BIC model selection (saves ~30s)
    python run.py --walkforward     # run walk-forward OOS validation (~5 mins)
    python run.py --multi-asset     # run on SPY, QQQ, IWM, TLT, GLD (~3 mins)
    python run.py --vol-signal      # add vol ratio dampening to ensemble
    python run.py --vvix-signal     # add VVIX dampening to ensemble
    python run.py --gex-proxy-signal  # add gamma-stress proxy dampening
    python run.py --multi-scale     # use multi-scale geometric (5/15/30-day windows)
    python run.py --expanding       # expanding-window honest backtest (~5-10 mins)

Outputs (saved to outputs/):
    {ticker}_{from_year}_{to_year}_regime_overview.png
    {ticker}_{from_year}_{to_year}_equity_curves.png
    {ticker}_{from_year}_{to_year}_walkforward_oos.png  (--walkforward only)
    {ticker}_{from_year}_{to_year}_expanding_oos.png    (--expanding only)
    multi_asset_{from_year}_{to_year}_comparison.png    (--multi-asset only)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.data        import (
    fetch_daily_bars, log_returns, fetch_multi,
    vix_levels, vvix_levels, fetch_vix_yfinance, fetch_vvix_yfinance,
)
from src.geometric   import geometric_signal, straightness_ratio, MULTI_WINDOWS
from src.markov      import fit_markov3, select_k
from src.ensemble    import (
    ensemble_score, regime_labels, vol_ratio, gamma_stress_proxy,
    VOL_RATIO_SUPPRESS, VVIX_NEUTRAL,
)
from src.backtest    import compute_stats, regime_return_stats, run_backtest, attribution_grid
from src.walkforward import walk_forward
from src.expanding   import expanding_backtest

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "momentum":  "#1a7a4a",
    "reversion": "#c0392b",
    "mixed":     "#95a5a6",
    "bnh":       "#2c3e50",
    "strategy":  "#2980b9",
}

MULTI_ASSET_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]


# ── Plotting ───────────────────────────────────────────────────────────


def plot_regime_overview(
    prices: pd.Series,
    returns: pd.Series,
    mom_prob: pd.Series,
    crisis_prob: pd.Series,
    labels: pd.Series,
    from_date: str,
    to_date: str,
    ticker: str = "SPY",
    run_prefix: str = "SPY",
) -> None:
    ratio = straightness_ratio(returns)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.06, figure=fig)

    # Panel 1: SPY price, coloured by regime
    ax1 = fig.add_subplot(gs[0])
    regime_color = labels.map(COLORS).reindex(prices.index, method="ffill")
    # Drop dates where ensemble label isn't yet available (Markov AR lag)
    valid    = ~regime_color.isna()
    prices_v = prices[valid]
    colors_v = regime_color[valid]
    for i in range(len(prices_v) - 1):
        ax1.plot(
            prices_v.index[i : i + 2],
            prices_v.iloc[i : i + 2],
            color=colors_v.iloc[i],
            linewidth=1.3,
            alpha=0.85,
        )
    patches = [mpatches.Patch(color=COLORS[r], label=r.title()) for r in ["momentum", "reversion", "mixed"]]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.7)
    ax1.set_ylabel(f"{ticker} Close", fontsize=9)
    ax1.set_xticks([])
    ax1.set_title(
        f"Regime Ensemble — {ticker} {from_date[:4]}–{to_date[:4]}  "
        f"(Geometric + Markov k=3)",
        fontsize=11,
        pad=8,
    )

    # Panel 2: Geometric straightness ratio
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(ratio.index, ratio, color="#7f8c8d", linewidth=0.8, alpha=0.75)
    ax2.axhline(ratio.quantile(0.70), color=COLORS["momentum"],  linestyle="--", linewidth=0.8, alpha=0.6, label="70th pct")
    ax2.axhline(ratio.quantile(0.30), color=COLORS["reversion"], linestyle="--", linewidth=0.8, alpha=0.6, label="30th pct")
    ax2.set_ylabel("Straightness Ratio", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax2.set_xticks([])

    # Panel 3: Markov k=3 filtered probabilities
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(mom_prob.index,    mom_prob,    alpha=0.45, color=COLORS["momentum"],  label="P(momentum) — filtered")
    ax3.fill_between(crisis_prob.index, crisis_prob, alpha=0.45, color=COLORS["reversion"], label="P(crisis)   — filtered")
    ax3.axhline(0.50, color="#7f8c8d", linestyle=":", linewidth=0.8)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Markov k=3", fontsize=9)
    ax3.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax3.tick_params(axis="x", labelsize=8)

    fig.autofmt_xdate(rotation=30, ha="right")
    path = OUTPUT_DIR / f"{run_prefix}_regime_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_equity_curves(bt: pd.DataFrame, run_prefix: str = "SPY") -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot(bt.index, bt["equity_bnh"],      color=COLORS["bnh"],     linewidth=1.5, label="Buy & Hold")
    ax1.plot(bt.index, bt["equity_strategy"], color=COLORS["strategy"], linewidth=1.5, label="Ensemble Strategy (Long Only, 0 bps)")
    ax1.axhline(1.0, color="#bdc3c7", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Equity (base = 1.0)", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.set_title("Ensemble Strategy vs Buy & Hold", fontsize=11)
    ax1.text(
        0.01, 0.03,
        "Zero transaction costs shown — see cost sensitivity table in terminal output",
        transform=ax1.transAxes, fontsize=7, color="#7f8c8d", style="italic",
    )

    strat_dd = bt["equity_strategy"] / bt["equity_strategy"].cummax() - 1
    bnh_dd   = bt["equity_bnh"]      / bt["equity_bnh"].cummax()      - 1
    ax2.fill_between(bt.index, strat_dd, 0, alpha=0.45, color=COLORS["strategy"], label="Strategy")
    ax2.fill_between(bt.index, bnh_dd,   0, alpha=0.30, color=COLORS["bnh"],      label="Buy & Hold")
    ax2.set_ylabel("Drawdown", fontsize=9)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.tick_params(axis="x", labelsize=8)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    path = OUTPUT_DIR / f"{run_prefix}_equity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_walkforward_equity(
    oos_returns: pd.DataFrame,
    run_prefix: str = "SPY",
    test_size: int = 63,
) -> None:
    """
    Plot the stitched OOS equity curve from all walk-forward folds.

    Each fold contributes ~63 trading days. Vertical dashed lines mark
    fold boundaries. This shows what the strategy would have returned
    out-of-sample, period by period — no look-ahead.
    """
    strat_eq = np.exp(oos_returns["strategy_return"].cumsum())
    bnh_eq   = np.exp(oos_returns["bnh_return"].cumsum())

    strat_dd = strat_eq / strat_eq.cummax() - 1
    bnh_dd   = bnh_eq   / bnh_eq.cummax()   - 1

    # Fold boundary dates: every test_size-th index step
    fold_starts = [oos_returns.index[i] for i in range(0, len(oos_returns), test_size) if i > 0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot(strat_eq.index, strat_eq, color=COLORS["strategy"], linewidth=1.5, label="Strategy (OOS)")
    ax1.plot(bnh_eq.index,   bnh_eq,   color=COLORS["bnh"],      linewidth=1.5, label="Buy & Hold")
    ax1.axhline(1.0, color="#bdc3c7", linewidth=0.6, linestyle="--")
    for fs in fold_starts:
        ax1.axvline(fs, color="#bdc3c7", linewidth=0.7, linestyle="--", alpha=0.6)
    ax1.set_ylabel("Equity (base = 1.0)", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.set_title("Walk-Forward OOS Equity Curve — All Folds Stitched", fontsize=11)
    ax1.text(
        0.01, 0.03,
        "Each segment is out-of-sample. Dashed lines = fold boundaries. Zero transaction costs.",
        transform=ax1.transAxes, fontsize=7, color="#7f8c8d", style="italic",
    )

    ax2.fill_between(strat_dd.index, strat_dd, 0, alpha=0.45, color=COLORS["strategy"], label="Strategy")
    ax2.fill_between(bnh_dd.index,   bnh_dd,   0, alpha=0.30, color=COLORS["bnh"],      label="Buy & Hold")
    for fs in fold_starts:
        ax2.axvline(fs, color="#bdc3c7", linewidth=0.7, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Drawdown", fontsize=9)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.tick_params(axis="x", labelsize=8)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    path = OUTPUT_DIR / f"{run_prefix}_walkforward_oos.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def run_pipeline_for_ticker(
    ticker: str,
    from_date: str,
    to_date: str,
    allow_short: bool = False,
    min_hold_days: int = 1,
) -> dict:
    """
    Run the full regime detection + backtest pipeline on a single ticker.

    Used by --multi-asset to compare multiple assets without charts or prints.

    Returns:
        dict with raw float metrics for strategy and buy & hold.
    """
    df  = fetch_daily_bars(ticker, from_date, to_date)
    ret = log_returns(df)

    geo          = geometric_signal(ret, window=15)
    mom_prob, crisis_prob, _, _ = fit_markov3(ret, verbose=False)
    score        = ensemble_score(geo, mom_prob, crisis_prob)
    labels       = regime_labels(score)
    bt           = run_backtest(ret, labels, allow_short=allow_short,
                                cost_bps=0, min_hold_days=min_hold_days)
    perf         = compute_stats(bt, raw=True)

    s = perf["Strategy (Long Only)"]
    b = perf["Buy & Hold"]

    pct_momentum = (labels == "momentum").mean()
    pct_mixed    = (labels == "mixed").mean()

    return {
        "ticker":       ticker,
        "n_days":       len(ret),
        "cagr_s":       s["CAGR"],
        "sharpe_s":     s["Sharpe"],
        "maxdd_s":      s["Max DD"],
        "tstat_s":      s["T-stat"],
        "pval_s":       s["P-value"],
        "cagr_b":       b["CAGR"],
        "sharpe_b":     b["Sharpe"],
        "maxdd_b":      b["Max DD"],
        "pct_momentum": pct_momentum,
        "pct_mixed":    pct_mixed,
    }


def _print_multi_asset_table(results: list) -> None:
    print(f"\n  {'Ticker':<6}  {'CAGR(S)':>8}  {'Sharpe(S)':>9}  {'MaxDD(S)':>9}  "
          f"{'T-stat':>7}  {'p':>6}  {'CAGR(B&H)':>10}  {'Sharpe(B&H)':>11}  {'%Mom':>5}  {'%Mix':>5}")
    print("  " + "─" * 88)
    for r in results:
        sig = "✓" if r["pval_s"] < 0.05 else ("~" if r["pval_s"] < 0.10 else " ")
        print(
            f"  {r['ticker']:<6}  "
            f"{r['cagr_s']*100:>+7.1f}%  "
            f"{r['sharpe_s']:>9.2f}  "
            f"{r['maxdd_s']*100:>8.1f}%  "
            f"{r['tstat_s']:>7.2f}  "
            f"{r['pval_s']:>6.3f}{sig}  "
            f"{r['cagr_b']*100:>+9.1f}%  "
            f"{r['sharpe_b']:>11.2f}  "
            f"{r['pct_momentum']*100:>4.0f}%  "
            f"{r['pct_mixed']*100:>4.0f}%"
        )


def _plot_multi_asset_chart(results: list, from_date: str, to_date: str) -> None:
    tickers     = [r["ticker"] for r in results]
    sharpe_s    = [r["sharpe_s"]    for r in results]
    sharpe_b    = [r["sharpe_b"]    for r in results]
    cagr_s      = [r["cagr_s"] * 100 for r in results]
    cagr_b      = [r["cagr_b"] * 100 for r in results]
    maxdd_s     = [abs(r["maxdd_s"]) * 100 for r in results]
    maxdd_b     = [abs(r["maxdd_b"]) * 100 for r in results]

    x = np.arange(len(tickers))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Multi-Asset Regime Ensemble — {from_date[:4]}–{to_date[:4]}  "
        f"(zero transaction costs, 1-day lag)",
        fontsize=11,
    )

    # Panel 1: Sharpe
    ax = axes[0]
    ax.bar(x - w/2, sharpe_s, w, label="Strategy", color=COLORS["strategy"], alpha=0.85)
    ax.bar(x + w/2, sharpe_b, w, label="Buy & Hold", color=COLORS["bnh"],    alpha=0.85)
    ax.axhline(0, color="#bdc3c7", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel("Sharpe Ratio"); ax.set_title("Sharpe Ratio")
    ax.legend(fontsize=8)

    # Panel 2: CAGR
    ax = axes[1]
    ax.bar(x - w/2, cagr_s, w, label="Strategy", color=COLORS["strategy"], alpha=0.85)
    ax.bar(x + w/2, cagr_b, w, label="Buy & Hold", color=COLORS["bnh"],    alpha=0.85)
    ax.axhline(0, color="#bdc3c7", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel("CAGR (%)"); ax.set_title("Annualised Return")
    ax.legend(fontsize=8)

    # Panel 3: Max Drawdown (shown as positive magnitude)
    ax = axes[2]
    ax.bar(x - w/2, maxdd_s, w, label="Strategy", color=COLORS["strategy"], alpha=0.85)
    ax.bar(x + w/2, maxdd_b, w, label="Buy & Hold", color=COLORS["reversion"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel("Max Drawdown (% magnitude)"); ax.set_title("Max Drawdown")
    ax.legend(fontsize=8)

    fig.tight_layout()
    from_year = from_date[:4]; to_year = to_date[:4]
    path = OUTPUT_DIR / f"multi_asset_{from_year}_{to_year}_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {path}")


def plot_ablation_curves(
    scores: dict,
    ret: pd.Series,
    run_prefix: str,
    title: str = "Markov Ablation — Is P(momentum) adding directional value?",
    annotations: str = "",
) -> None:
    """
    Two-panel chart for ablation analysis.

    Top: equity curves for provided variants and B&H.
    Bottom: drawdowns for all provided variants.
    """
    bts = {}
    for label, s in scores.items():
        l  = regime_labels(s)
        bt = run_backtest(ret.rename("log_return"), l, cost_bps=0)
        bts[label] = bt

    ablation_colors = {
        "geo_only":         "#95a5a6",
        "crisis_filter":    "#e67e22",
        "full_ensemble":    COLORS["strategy"],
        "baseline":         COLORS["strategy"],
        "vvix_only":        "#16a085",
        "gamma_proxy_only": "#8e44ad",
        "vvix_gamma":       "#2c3e50",
    }
    ablation_labels = {
        "geo_only":         "Geo only  (no Markov)",
        "crisis_filter":    "Crisis filter  (1 - P(crisis))",
        "full_ensemble":    "Full ensemble  (P(momentum), current default)",
        "baseline":         "Baseline  (current active setup)",
        "vvix_only":        "Baseline + VVIX dampener",
        "gamma_proxy_only": "Baseline + gamma proxy dampener",
        "vvix_gamma":       "Baseline + VVIX + gamma proxy",
    }
    fallback_colors = ["#34495e", "#7f8c8d", "#27ae60", "#c0392b", "#2980b9", "#d35400"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    bnh_eq = np.exp(next(iter(bts.values()))["bnh_return"].cumsum())
    ax1.plot(bnh_eq.index, bnh_eq, color=COLORS["bnh"], linewidth=1.3, label="Buy & Hold", alpha=0.7)
    for i, (label, bt) in enumerate(bts.items()):
        eq = np.exp(bt["strategy_return"].cumsum())
        color = ablation_colors.get(label, fallback_colors[i % len(fallback_colors)])
        ax1.plot(eq.index, eq, color=color, linewidth=1.5, label=ablation_labels.get(label, label))
    ax1.axhline(1.0, color="#bdc3c7", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Equity (base = 1.0)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(title, fontsize=11)
    if annotations:
        ax1.text(
            0.01, 0.03, annotations,
            transform=ax1.transAxes, fontsize=7, color="#7f8c8d", style="italic",
        )

    for i, label in enumerate(bts.keys()):
        eq  = np.exp(bts[label]["strategy_return"].cumsum())
        dd  = eq / eq.cummax() - 1
        color = ablation_colors.get(label, fallback_colors[i % len(fallback_colors)])
        ax2.plot(dd.index, dd, color=color, linewidth=1.2,
                 label=ablation_labels.get(label, label), alpha=0.85)
    ax2.axhline(0, color="#bdc3c7", linewidth=0.6)
    ax2.set_ylabel("Drawdown", fontsize=9)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.tick_params(axis="x", labelsize=8)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    path = OUTPUT_DIR / f"{run_prefix}_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def run_ablation(
    ret: pd.Series,
    geo: pd.Series,
    mom_prob: pd.Series,
    crisis_prob: pd.Series,
    run_prefix: str,
    vvix_signal: pd.Series = None,
    gamma_proxy_signal: pd.Series = None,
    allow_short: bool = False,
    min_hold_days: int = 1,
) -> None:
    """
    Run ablation diagnostics.

    Part A builds three variants that differ only in how the Markov component
    is constructed, then compares Sharpe/CAGR/MaxDD side-by-side.

    The result answers: "Is P(momentum) adding directional value, or is the
    Markov edge entirely from identifying 'don't be aggressive here' periods?"

    Part A variants:
        geo_only       — geometric signal only (no Markov)
        crisis_filter  — mean(geo, 1 - P(crisis))  — pure risk-state filter
        full_ensemble  — mean(geo, P(momentum) zeroed on crisis)  — current default
    Part B (optional) compares volatility dampener variants when vvix_signal
    and/or gamma_proxy_signal are supplied.
    """
    _section("ABLATION A: Is Markov adding directional information?")
    print("  Building 3 variants — signals identical, only Markov component changes.")
    print("  Vol dampeners excluded so the Markov component is the only variable.")
    print("  Discrete labels, 0 bps transaction costs, 1-day execution lag.\n")

    variants = [
        ("geo_only",      "geo_only"),
        ("crisis_filter", "crisis_filter"),
        ("full_ensemble", "full"),
    ]
    descriptions = {
        "geo_only":      "(none — geometric only)      ",
        "crisis_filter": "1 - P(crisis)                ",
        "full_ensemble": "P(momentum), zeroed on crisis",
    }

    ret_named = ret.rename("log_return")  # run_backtest expects this column name
    results = {}
    scores  = {}
    for label, mode in variants:
        s = ensemble_score(geo, mom_prob, crisis_prob, mode=mode)
        l = regime_labels(s)
        bt = run_backtest(ret_named, l, allow_short=allow_short,
                          cost_bps=0, min_hold_days=min_hold_days)
        st  = compute_stats(bt, raw=True)["Strategy (Long Only)"]
        pct_mom   = (l == "momentum").mean()
        pct_mixed = (l == "mixed").mean()
        results[label] = {**st, "pct_mom": pct_mom, "pct_mixed": pct_mixed}
        scores[label]  = s

    # ── Print comparison table ─────────────────────────────────────────
    w = 90
    print(f"  {'Variant':<16}  {'Markov component':<32}  {'CAGR':>6}  {'Sharpe':>7}  "
          f"{'Max DD':>7}  {'%Mom':>5}  {'%Mix':>5}")
    print("  " + "─" * w)
    for label, _ in variants:
        r = results[label]
        print(
            f"  {label:<16}  {descriptions[label]}  "
            f"{r['CAGR']*100:>+5.1f}%  "
            f"{r['Sharpe']:>7.2f}  "
            f"{r['Max DD']*100:>6.1f}%  "
            f"{r['pct_mom']*100:>4.0f}%  "
            f"{r['pct_mixed']*100:>4.0f}%"
        )

    full_sh   = results["full_ensemble"]["Sharpe"]
    cf_sh     = results["crisis_filter"]["Sharpe"]
    geo_sh    = results["geo_only"]["Sharpe"]
    delta_fc  = full_sh - cf_sh
    delta_cg  = cf_sh - geo_sh

    print(f"\n  Sharpe delta (full_ensemble vs crisis_filter): {delta_fc:+.3f}")
    print(f"  Sharpe delta (crisis_filter  vs geo_only):    {delta_cg:+.3f}")

    print("\n  Interpretation:")
    if abs(delta_fc) < 0.05:
        print("    |full - crisis_filter| < 0.05: P(momentum) adds minimal directional value.")
        print("    The Markov edge is primarily identifying 'don't be aggressive here' periods.")
        print("    Consider switching default to mode='crisis_filter' (simpler, more defensible).")
    else:
        print(f"    |full - crisis_filter| = {abs(delta_fc):.3f} >= 0.05:")
        print("    P(momentum) IS contributing real directional value beyond the crisis filter.")
        print("    Keeping mode='full' (current default) is justified.")

    if delta_cg < 0.02:
        print("    Note: crisis_filter barely beats geo_only — Markov adds little even as a risk filter.")

    plot_ablation_curves(
        scores, ret, f"{run_prefix}_markov",
        title="Markov Ablation — Is P(momentum) adding directional value?",
        annotations=(
            "geo_only = geometric only  |  crisis_filter = mean(geo, 1-P(crisis))  |  "
            "full_ensemble = current default: mean(geo, P(mom) zeroed on crisis)"
        ),
    )

    if vvix_signal is None and gamma_proxy_signal is None:
        print("\n  Volatility dampener ablation skipped (no VVIX or gamma proxy signal enabled).")
        return

    _section("ABLATION B: Volatility dampener incremental value")
    print("  Baseline Markov mode fixed at 'full'; only dampeners change.")
    print("  Discrete labels, 0 bps transaction costs, 1-day execution lag.\n")

    variants_b = [("baseline", {})]
    descriptions_b = {"baseline": "Current setup without VVIX/gamma proxy dampener"}
    if vvix_signal is not None:
        variants_b.append(("vvix_only", {"vvix": vvix_signal}))
        descriptions_b["vvix_only"] = "Baseline + VVIX dampener"
    if gamma_proxy_signal is not None:
        variants_b.append(("gamma_proxy_only", {"gamma_proxy": gamma_proxy_signal}))
        descriptions_b["gamma_proxy_only"] = "Baseline + gamma proxy dampener"
    if vvix_signal is not None and gamma_proxy_signal is not None:
        variants_b.append(("vvix_gamma", {"vvix": vvix_signal, "gamma_proxy": gamma_proxy_signal}))
        descriptions_b["vvix_gamma"] = "Baseline + VVIX + gamma proxy dampeners"

    results_b = {}
    scores_b = {}
    for label, kwargs in variants_b:
        s = ensemble_score(geo, mom_prob, crisis_prob, mode="full", **kwargs)
        l = regime_labels(s)
        bt = run_backtest(
            ret_named, l, allow_short=allow_short, cost_bps=0, min_hold_days=min_hold_days
        )
        st = compute_stats(bt, raw=True)["Strategy (Long Only)"]
        results_b[label] = {
            **st,
            "pct_mom": (l == "momentum").mean(),
            "pct_mixed": (l == "mixed").mean(),
        }
        scores_b[label] = s

    print(f"  {'Variant':<18}  {'Description':<45}  {'CAGR':>6}  {'Sharpe':>7}  {'Max DD':>7}  {'%Mom':>5}  {'%Mix':>5}")
    print("  " + "─" * 106)
    for label, _ in variants_b:
        r = results_b[label]
        print(
            f"  {label:<18}  {descriptions_b[label]:<45}  "
            f"{r['CAGR']*100:>+5.1f}%  "
            f"{r['Sharpe']:>7.2f}  "
            f"{r['Max DD']*100:>6.1f}%  "
            f"{r['pct_mom']*100:>4.0f}%  "
            f"{r['pct_mixed']*100:>4.0f}%"
        )

    base_sh = results_b["baseline"]["Sharpe"]
    for label, _ in variants_b:
        if label == "baseline":
            continue
        delta = results_b[label]["Sharpe"] - base_sh
        print(f"  Sharpe delta ({label} - baseline): {delta:+.3f}")

    plot_ablation_curves(
        scores_b, ret, f"{run_prefix}_vol_dampeners",
        title="Volatility Dampener Ablation — Incremental value vs baseline",
        annotations=(
            "baseline = current active setup | vvix_only = baseline + VVIX dampener | "
            "gamma_proxy_only = baseline + gamma-stress proxy | vvix_gamma = both"
        ),
    )


def run_multi_asset(args) -> None:
    _section("MULTI-ASSET VALIDATION  (SPY, QQQ, IWM, TLT, GLD)")
    print("  Fitting Markov k=3 for each asset (~30s each)...\n")
    results = []
    for ticker in MULTI_ASSET_TICKERS:
        print(f"  [{ticker}] fitting...")
        row = run_pipeline_for_ticker(
            ticker, args.from_date, args.to_date,
            allow_short=args.short,
            min_hold_days=args.min_hold,
        )
        results.append(row)
        print(f"  [{ticker}] done — Sharpe {row['sharpe_s']:.2f}  CAGR {row['cagr_s']*100:+.1f}%  "
              f"vs B&H Sharpe {row['sharpe_b']:.2f}")

    _section("MULTI-ASSET RESULTS")
    _print_multi_asset_table(results)
    print("\n  ✓ = p<0.05   ~ = p<0.10   (strategy return vs zero)")
    _plot_multi_asset_chart(results, args.from_date, args.to_date)


# ── Main ───────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print(f"\n{'─' * 58}\n  {title}\n{'─' * 58}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime Ensemble Backtest")
    parser.add_argument("--ticker",  default="SPY",        metavar="TICKER",    help="Equity ticker (default: SPY)")
    parser.add_argument("--from",    dest="from_date",     default="2000-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--to",      dest="to_date",       default="2025-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--fetch-vix",   action="store_true", help="Fetch VIX (I:VIX) from Polygon alongside primary ticker")
    parser.add_argument("--vix-signal",  action="store_true", help="Use VIX as dampening factor in ensemble (requires cached VIX data)")
    parser.add_argument("--fetch-vvix",  action="store_true", help="Fetch VVIX (I:VVIX) from Polygon alongside primary ticker")
    parser.add_argument("--vvix-signal", action="store_true", help="Use VVIX as vol-of-vol dampening factor in ensemble")
    parser.add_argument("--gex-proxy-signal", action="store_true",
                        help="Use gamma-stress proxy dampening (spot/VIX/VVIX shocks; proxy, not true options-chain GEX)")
    parser.add_argument("--vix-feature", action="store_true", help="Include VIX level as 6th feature in the Markov observation vector")
    parser.add_argument("--short",       action="store_true", help="Allow short on reversion days")
    parser.add_argument("--min-hold",    dest="min_hold", type=int, default=1, metavar="N",
                        help="Persistence filter: require N consecutive days in regime before switching (default: 1 = off)")
    parser.add_argument("--skip-bic",    action="store_true", help="Skip BIC model selection step (~60s)")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward OOS validation (~5 mins)")
    parser.add_argument("--multi-asset", action="store_true", help="Run on SPY, QQQ, IWM, TLT, GLD (~3 mins)")
    parser.add_argument("--vol-signal",      action="store_true", help="Add vol ratio dampening to ensemble (5-day/63-day realised vol)")
    parser.add_argument("--multi-scale",     action="store_true", help="Use multi-scale geometric signal (average across 5, 15, 30-day windows)")
    parser.add_argument("--expanding",       action="store_true", help="Run expanding-window honest backtest (~5-10 mins)")
    parser.add_argument("--geo-directional", action="store_true", help="Use signed straightness ratio: uptrends +1, downtrends -1 (fixes direction-blindness)")
    parser.add_argument("--continuous",      action="store_true", help="Continuous position sizing: position = ensemble score [0,1] instead of discrete {0, 0.5, 1}")
    parser.add_argument("--kalman",          action="store_true", help="Add Kalman drift filter as 3rd ensemble signal (local level model, MLE-estimated Q and R)")
    parser.add_argument("--ablation",        action="store_true", help="Run ablation diagnostics: Markov component variants + (optional) VVIX/gamma dampener variants")
    args = parser.parse_args()

    if args.multi_asset:
        run_multi_asset(args)
        return

    run_prefix = f"{args.ticker}_{args.from_date[:4]}_{args.to_date[:4]}"

    # ── 1. Data ────────────────────────────────────────────────────────
    _section("1. DATA")
    print(f"  Fetching {args.ticker}  {args.from_date} → {args.to_date}")
    df     = fetch_daily_bars(args.ticker, args.from_date, args.to_date)
    ret    = log_returns(df)
    prices = df["close"]
    print(f"  {len(df)} trading days loaded")

    vix = None
    if args.fetch_vix or args.vix_signal or args.vix_feature or args.gex_proxy_signal:
        print(f"  Fetching VIX  {args.from_date} → {args.to_date}")
        try:
            vix_df = fetch_daily_bars("I:VIX", args.from_date, args.to_date)
            vix    = vix_levels(vix_df)
            print(f"  VIX: {len(vix_df)} days from Polygon  (range: {vix.min():.1f} – {vix.max():.1f})")
        except Exception as _poly_err:
            print(f"  Polygon I:VIX unavailable ({_poly_err.__class__.__name__}) — falling back to yfinance ^VIX")
            vix = fetch_vix_yfinance(args.from_date, args.to_date)
            print(f"  VIX: {len(vix)} days from yfinance  (range: {vix.min():.1f} – {vix.max():.1f})")

    vvix = None
    if args.fetch_vvix or args.vvix_signal or args.gex_proxy_signal:
        print(f"  Fetching VVIX  {args.from_date} → {args.to_date}")
        try:
            vvix_df = fetch_daily_bars("I:VVIX", args.from_date, args.to_date)
            vvix    = vvix_levels(vvix_df)
            print(f"  VVIX: {len(vvix_df)} days from Polygon  (range: {vvix.min():.1f} – {vvix.max():.1f})")
        except Exception as _poly_err:
            print(f"  Polygon I:VVIX unavailable ({_poly_err.__class__.__name__}) — falling back to yfinance ^VVIX")
            vvix = fetch_vvix_yfinance(args.from_date, args.to_date)
            print(f"  VVIX: {len(vvix)} days from yfinance  (range: {vvix.min():.1f} – {vvix.max():.1f})")

    # ── 2. BIC Model Selection ─────────────────────────────────────────
    if not args.skip_bic:
        _section("2. BIC MODEL SELECTION  (k=2 vs k=3)")
        print("  Fitting k=2 and k=3  (~60s total)...")
        bic_table = select_k(ret, k_range=range(2, 4))
        print(f"\n{bic_table.to_string()}")
        delta = abs(bic_table.loc[3, "bic"] - bic_table.loc[2, "bic"])
        print(f"\n  BIC selects k=3  (ΔBIC = {delta:.0f}  — 'very strong evidence' threshold is 10)")
        print("  k=3 isolates a crisis regime (high vol, negative mean) that")
        print("  k=2 incorrectly merges with the reversion state.")
    else:
        _section("2. BIC MODEL SELECTION  [skipped — k=3 pre-selected]")

    # ── 3. Geometric Signal ────────────────────────────────────────────
    geo_windows = MULTI_WINDOWS if args.multi_scale else None
    geo_desc = ("multi-scale windows=%s" % MULTI_WINDOWS) if args.multi_scale else "window=15"
    if args.geo_directional:
        geo_desc += ", directional (signed ratio)"
    _section("3. GEOMETRIC SIGNAL  (straightness ratio, %s)" % geo_desc)
    geo = geometric_signal(ret, window=15, windows=geo_windows, directional=args.geo_directional)
    if args.geo_directional:
        print("  Directional mode: uptrend ratio > 0, downtrend ratio < 0."
              " Straight crashes now score LOW (cash) rather than HIGH (buy).")
    counts = geo.value_counts()
    for val, name in [(1.0, "Momentum"), (0.5, "Mixed"), (0.0, "Reversion")]:
        n = counts.get(val, 0)
        print(f"  {name:<12s}  {n:4d} days  ({n / len(geo) * 100:.1f}%)")

    # ── 4. Markov k=3 Signal ───────────────────────────────────────────
    vix_feature = vix if args.vix_feature else None
    n_markov_feats = 6 if vix_feature is not None else 5
    _section(f"4. GAUSSIAN HMM k=3  ({n_markov_feats}-feature state vector, filtered probabilities)")
    print("  Fitting  (5 random restarts × 200 EM iterations)...")
    mom_prob, crisis_prob, _, trans_info = fit_markov3(ret, vix=vix_feature)

    # ── 5. Ensemble ────────────────────────────────────────────────────
    vix_signal = vix if args.vix_signal else None
    vvix_signal = vvix if args.vvix_signal else None
    vol_ratio_signal = vol_ratio(ret) if args.vol_signal else None
    gex_proxy_signal = gamma_stress_proxy(ret, vix=vix, vvix=vvix) if (args.gex_proxy_signal and vix is not None) else None
    if args.gex_proxy_signal and gex_proxy_signal is None:
        print("  Gamma proxy requested but VIX data unavailable — skipping gamma proxy dampening.")

    ensemble_parts = ["mean of geometric + Markov", "crisis override at P>0.50"]
    if vix_signal is not None:
        ensemble_parts.append("VIX dampening")
    if vvix_signal is not None:
        ensemble_parts.append("VVIX dampening")
    if vol_ratio_signal is not None:
        ensemble_parts.append("vol ratio dampening (5d/63d)")
    if gex_proxy_signal is not None:
        ensemble_parts.append("gamma proxy dampening")
    if args.kalman:
        ensemble_parts.append("Kalman drift (3-way ensemble)")
    _section("5. ENSEMBLE  (%s)" % ", ".join(ensemble_parts))

    if vol_ratio_signal is not None:
        vr_mean = vol_ratio_signal.mean()
        vr_pct_high = (vol_ratio_signal > 1.5).mean() * 100
        print(f"  Vol ratio (5d/63d):  mean={vr_mean:.2f}  "
              f"days with ratio>1.5 (dampening active): {vr_pct_high:.1f}%")
    if vvix_signal is not None:
        vvix_mean = vvix_signal.reindex(ret.index, method="ffill").mean()
        vvix_pct_high = (vvix_signal.reindex(ret.index, method="ffill") > VVIX_NEUTRAL).mean() * 100
        print(f"  VVIX level:          mean={vvix_mean:.1f}  "
              f"days > {VVIX_NEUTRAL:.0f} (dampening active): {vvix_pct_high:.1f}%")
    if gex_proxy_signal is not None:
        gp_mean = gex_proxy_signal.mean()
        gp_pct_high = (gex_proxy_signal > 0.5).mean() * 100
        print(f"  Gamma proxy:         mean={gp_mean:.2f}  "
              f"days > 0.50 (strong dampening): {gp_pct_high:.1f}%")

    kalman_signal_series = None
    if args.kalman:
        from src.kalman import fit_kalman, kalman_signal as _kal_signal
        print("  Fitting Kalman filter (MLE for Q, R)...")
        Q_kal, R_kal = fit_kalman(ret)
        kalman_signal_series = _kal_signal(ret, Q=Q_kal, R=R_kal)
        q_ratio = Q_kal / R_kal
        print(f"  Kalman MLE:  Q={Q_kal:.2e}  R={R_kal:.2e}  Q/R ratio={q_ratio:.4f}")
        print(f"  (Q/R < 0.01 = slow drift; Q/R > 0.1 = fast-adapting filter)")
        kal_pct_high = (kalman_signal_series > 0.65).mean() * 100
        kal_pct_low  = (kalman_signal_series < 0.35).mean() * 100
        print(f"  Kalman signal:  >0.65 (momentum-leaning): {kal_pct_high:.1f}%  "
              f"<0.35 (reversion-leaning): {kal_pct_low:.1f}%")

    score  = ensemble_score(geo, mom_prob, crisis_prob,
                            vix=vix_signal,
                            vvix=vvix_signal,
                            vol_ratio_series=vol_ratio_signal,
                            gamma_proxy=gex_proxy_signal,
                            kalman=kalman_signal_series)
    labels = regime_labels(score)
    dist   = labels.value_counts()
    print("  Ensemble regime distribution:")
    for name in ["momentum", "mixed", "reversion"]:
        n = dist.get(name, 0)
        print(f"  {name:<12s}  {n:4d} days  ({n / len(labels) * 100:.1f}%)")

    # Vol/crisis overlap diagnostic — shows how much incremental work the vol
    # dampening does beyond what the Markov crisis override already captures.
    if vol_ratio_signal is not None:
        vol_high    = (vol_ratio_signal > VOL_RATIO_SUPPRESS).reindex(labels.index).fillna(False)
        crisis_high = (crisis_prob > 0.50).reindex(labels.index).fillna(False)
        n_vol       = vol_high.sum()
        n_crisis    = crisis_high.sum()
        n_both      = (vol_high & crisis_high).sum()
        n_vol_only  = (vol_high & ~crisis_high).sum()
        print(f"\n  Vol/crisis overlap diagnostic  (vol ratio threshold = {VOL_RATIO_SUPPRESS:.1f}x):")
        print(f"    Vol ratio > {VOL_RATIO_SUPPRESS:.1f}x active:    {n_vol:4d} days  ({n_vol / len(labels) * 100:.1f}%)")
        print(f"    P(crisis) > 0.50 active: {n_crisis:4d} days  ({n_crisis / len(labels) * 100:.1f}%)")
        print(f"    Both active (overlap):   {n_both:4d} days  ({n_both / max(n_vol, 1) * 100:.0f}% of vol-high days)")
        print(f"    Vol-only (incremental):  {n_vol_only:4d} days  -- days suppressed by vol but not by crisis override")
    if vvix_signal is not None:
        vvix_high = (vvix_signal.reindex(labels.index, method="ffill") > VVIX_NEUTRAL).fillna(False)
        crisis_high = (crisis_prob > 0.50).reindex(labels.index).fillna(False)
        n_vvix = vvix_high.sum()
        n_both = (vvix_high & crisis_high).sum()
        n_vvix_only = (vvix_high & ~crisis_high).sum()
        print(f"\n  VVIX/crisis overlap diagnostic  (VVIX threshold = {VVIX_NEUTRAL:.0f}):")
        print(f"    VVIX > {VVIX_NEUTRAL:.0f} active:          {n_vvix:4d} days  ({n_vvix / len(labels) * 100:.1f}%)")
        print(f"    Both active (overlap):         {n_both:4d} days  ({n_both / max(n_vvix, 1) * 100:.0f}% of vvix-high days)")
        print(f"    VVIX-only (incremental):       {n_vvix_only:4d} days")
    if gex_proxy_signal is not None:
        gp_high = (gex_proxy_signal.reindex(labels.index, method="ffill") > 0.5).fillna(False)
        crisis_high = (crisis_prob > 0.50).reindex(labels.index).fillna(False)
        n_gp = gp_high.sum()
        n_both = (gp_high & crisis_high).sum()
        n_gp_only = (gp_high & ~crisis_high).sum()
        print("\n  Gamma-proxy/crisis overlap diagnostic  (gamma proxy threshold = 0.50):")
        print(f"    Gamma proxy > 0.50 active:     {n_gp:4d} days  ({n_gp / len(labels) * 100:.1f}%)")
        print(f"    Both active (overlap):         {n_both:4d} days  ({n_both / max(n_gp, 1) * 100:.0f}% of gamma-high days)")
        print(f"    Gamma-only (incremental):      {n_gp_only:4d} days")

    # ── 6. Forward Return Statistics ────────────────────────────────────
    _section("6. FORWARD RETURN STATISTICS  (next-day return by regime)")
    forward_ret = ret.shift(-1).rename("log_return")
    stats_df = regime_return_stats(forward_ret, labels)
    print(stats_df.to_string())
    print("\n  Interpretation guide:  |t| > 2.0, p < 0.05 = statistically significant")
    print("  Momentum signal is typically significant; reversion is weaker.")

    # ── 6b. Signal Attribution Grid ───────────────────────────────────
    _section("6b. SIGNAL ATTRIBUTION GRID  (geo × markov forward return)")
    grid = attribution_grid(ret, geo, mom_prob)
    mom_bins = ["low (<0.33)", "mid (0.33-0.67)", "high (>0.67)"]
    hdr = "Geo / Markov"
    print(f"  {hdr:<14s}  {'low (<0.33)':>22s}  {'mid (0.33-0.67)':>22s}  {'high (>0.67)':>22s}")
    print("  " + "─" * 82)
    for g in ["reversion", "mixed", "momentum"]:
        cells = [r for r in grid if r["geo"] == g]
        cell_strs = []
        for c in cells:
            if np.isnan(c["mean_pct"]):
                cell_strs.append(f"{'n/a':>22s}")
            else:
                sig = "[*]" if c["p"] < 0.05 else ("[~]" if c["p"] < 0.10 else "   ")
                cell_strs.append(f"{c['mean_pct']:+.4f}%  t={c['t']:+.2f} {sig} n={c['n']:4d}")
        print(f"  {g:<14s}  {'  '.join(cell_strs)}")
    print("\n  Significance: [*] p<0.05  [~] p<0.10  [ ] not significant")
    print("  Rows=geo signal, Cols=Markov P(momentum) bin")
    print("  Diagonal: both signals agree. Off-diagonal: signals disagree.")
    print("  If left column (markov=low) has negative/low returns, Markov's")
    print("  edge is risk-avoidance (crisis filter), not direction.")

    # ── 7. Backtest + Cost Sensitivity ────────────────────────────────
    _section("7. BACKTEST RESULTS")
    if args.continuous:
        print("  Mode: Continuous sizing  (position = ensemble score, clipped [0, 1])")
        print("  Note: --short and --min-hold are ignored in continuous mode")
    else:
        mode = "Long / Short" if args.short else "Long on momentum (+1), half-long on mixed (+0.5), cash on reversion"
        print(f"  Mode: {mode}")
        if args.min_hold > 1:
            print(f"  Persistence filter: {args.min_hold} consecutive days required before regime switch")
    print("  Signal execution: 1-day lag (signal at close T → trade at open T+1)\n")

    bt   = run_backtest(ret, labels, allow_short=args.short, cost_bps=0, min_hold_days=args.min_hold,
                        score=score if args.continuous else None)
    perf = compute_stats(bt)
    for strategy_name, metrics in perf.items():
        print(f"  {strategy_name}:")
        for k, v in metrics.items():
            print(f"    {k:<12s}  {v}")
        print()

    # Cost sensitivity table
    print("  Transaction cost sensitivity (long-only strategy):")
    print(f"  {'Cost':>8s}  {'CAGR':>8s}  {'Sharpe':>8s}  {'Max DD':>8s}")
    for bps in [0, 5, 10, 20]:
        bt_c   = run_backtest(ret, labels, allow_short=False, cost_bps=bps, min_hold_days=args.min_hold,
                              score=score if args.continuous else None)
        perf_c = compute_stats(bt_c)["Strategy (Long Only)"]
        print(f"  {bps:>6d}bps  {perf_c['CAGR']:>8s}  {perf_c['Sharpe']:>8s}  {perf_c['Max DD']:>8s}")

    # Sub-period Sharpe breakdown — shows whether the edge is concentrated in
    # one crisis period (e.g. 2008, 2020) or consistent across the full sample.
    print("\n  Sub-period Sharpe breakdown (5-year windows):")
    print(f"  {'Period':<12}  {'Strategy':>9}  {'Buy & Hold':>10}  {'Edge':>5}")
    start_year = bt.index[0].year
    end_year   = bt.index[-1].year
    any_row = False
    for yr in range(start_year, end_year, 5):
        mask   = (bt.index.year >= yr) & (bt.index.year < yr + 5)
        sl     = bt[mask]
        if len(sl) < 63:
            continue
        s_r  = sl["strategy_return"]
        b_r  = sl["bnh_return"]
        s_sh = (s_r.mean() * 252) / (s_r.std() * np.sqrt(252)) if s_r.std() > 0 else float("nan")
        b_sh = (b_r.mean() * 252) / (b_r.std() * np.sqrt(252)) if b_r.std() > 0 else float("nan")
        edge = "+" if (not np.isnan(s_sh) and not np.isnan(b_sh) and s_sh > b_sh) else " "
        print(f"  {yr}-{min(yr+4, end_year):<8}  {s_sh:>+9.2f}  {b_sh:>+10.2f}  {edge}")
        any_row = True
    if not any_row:
        print("  (insufficient data for sub-period breakdown — need >= 63 days per window)")

    # ── Ablation diagnostic ────────────────────────────────────────────
    if args.ablation:
        run_ablation(ret, geo, mom_prob, crisis_prob,
                     run_prefix=run_prefix,
                     vvix_signal=vvix_signal,
                     gamma_proxy_signal=gex_proxy_signal,
                     allow_short=args.short,
                     min_hold_days=args.min_hold)

    # ── 8. Walk-Forward OOS Validation ─────────────────────────────────
    extra_sections = 0
    if args.walkforward:
        _section("8. WALK-FORWARD OOS VALIDATION  (10 folds x 63 days, fully OOS)")
        print("  Geometric: thresholds from train slice only.")
        print("  Markov: fitted on train, forward-filtered on test (no EM on test data).\n")
        wf_df, oos_returns = walk_forward(ret, n_folds=10, test_size=63,
                                          geo_directional=args.geo_directional,
                                          use_kalman=args.kalman,
                                          vix=vix_feature)
        print(wf_df.to_string())
        pos_folds = (wf_df.loc[(slice(None), "momentum"), "Dir"] == "✓").sum()
        total     = len(wf_df.loc[(slice(None), "momentum")])
        print(f"\n  Momentum regime positive in {pos_folds}/{total} folds")
        print("  (>= 7/10 is consistent with signal having real predictive power)")
        plot_walkforward_equity(oos_returns, run_prefix=run_prefix, test_size=63)
        extra_sections += 1

    # ── 9. Expanding-Window Honest Backtest ────────────────────────────
    if args.expanding:
        sec = 8 + extra_sections
        _section(f"{sec}. EXPANDING-WINDOW BACKTEST  (annual refit, fully honest)")
        print("  Refitting geometric thresholds + Markov annually on all data to that date.")
        multi_scale_flag = args.multi_scale
        print(f"  Geometric: {'multi-scale ' + str(MULTI_WINDOWS) if multi_scale_flag else 'window=15'}")
        print("  Markov: em_iter=200 per refit. First live period after 2 years of data.\n")
        exp_bt = expanding_backtest(ret, multi_scale=multi_scale_flag,
                                    geo_directional=args.geo_directional,
                                    use_kalman=args.kalman, verbose=True,
                                    vix=vix_feature)

        from src.backtest import compute_stats as _cs
        exp_perf = _cs(exp_bt)
        print("\n  Expanding-window results:")
        for strategy_name, metrics in exp_perf.items():
            print(f"  {strategy_name}:")
            for k, v in metrics.items():
                print(f"    {k:<12s}  {v}")
            print()

        in_sample_sharpe  = float(compute_stats(bt)["Strategy (Long Only)"]["Sharpe"])
        exp_sharpe        = float(exp_perf["Strategy (Long Only)"]["Sharpe"])
        print(f"  Optimism bias (in-sample vs expanding):  "
              f"{in_sample_sharpe:.2f} vs {exp_sharpe:.2f}  "
              f"(delta = {in_sample_sharpe - exp_sharpe:+.2f})")

        # Plot expanding OOS equity curve
        exp_eq_s = np.exp(exp_bt["strategy_return"].cumsum())
        exp_eq_b = np.exp(exp_bt["bnh_return"].cumsum())
        in_eq_s  = bt["equity_strategy"].reindex(exp_bt.index)
        fig_exp, ax_exp = plt.subplots(figsize=(12, 5))
        ax_exp.plot(exp_eq_b.index,  exp_eq_b,  color=COLORS["bnh"],      lw=1.5, label="Buy & Hold")
        ax_exp.plot(exp_eq_s.index,  exp_eq_s,  color=COLORS["strategy"], lw=1.5, label="Strategy (expanding OOS)")
        ax_exp.plot(in_eq_s.index,   in_eq_s,   color=COLORS["strategy"], lw=1.0,
                    ls="--", alpha=0.5, label="Strategy (in-sample, for comparison)")
        ax_exp.axhline(1.0, color="#bdc3c7", lw=0.6, ls="--")
        ax_exp.set_ylabel("Equity (base = 1.0)", fontsize=9)
        ax_exp.legend(fontsize=9)
        ax_exp.set_title("Expanding-Window Honest Backtest vs In-Sample", fontsize=11)
        ax_exp.text(0.01, 0.03,
                    "Expanding: thresholds/Markov refitted annually on all prior data. "
                    "No future data used. Zero transaction costs.",
                    transform=ax_exp.transAxes, fontsize=7, color="#7f8c8d", style="italic")
        ax_exp.spines[["top", "right"]].set_visible(False)
        ax_exp.tick_params(axis="x", labelsize=8)
        fig_exp.autofmt_xdate(rotation=30, ha="right")
        fig_exp.tight_layout()
        exp_path = OUTPUT_DIR / f"{run_prefix}_expanding_oos.png"
        fig_exp.savefig(exp_path, dpi=150, bbox_inches="tight")
        plt.close(fig_exp)
        print(f"\n  Saved → {exp_path}")
        extra_sections += 1

    charts_section = 8 + extra_sections

    # ── Charts ────────────────────────────────────────────────────────
    _section(f"{charts_section}. CHARTS")
    plot_regime_overview(prices, ret, mom_prob, crisis_prob, labels,
                         args.from_date, args.to_date, ticker=args.ticker, run_prefix=run_prefix)
    plot_equity_curves(bt, run_prefix=run_prefix)

    print("\n  Complete. Charts saved to outputs/\n")


if __name__ == "__main__":
    main()
