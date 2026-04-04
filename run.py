#!/usr/bin/env python3
"""
Regime Ensemble Backtest
========================
Daily regime detection using a two-model ensemble:
    1. Geometric (straightness ratio, adaptive percentile thresholds)
    2. Markov Switching AR(1), k=3 selected by BIC (ΔBIC=82 over k=2)

Usage:
    python run.py
    python run.py --ticker QQQ --from 2000-01-01 --to 2025-01-01
    python run.py --fetch-vix       # also fetch VIX (I:VIX) from Polygon
    python run.py --short           # allow short on reversion days
    python run.py --skip-bic        # skip BIC model selection (saves ~30s)
    python run.py --walkforward     # run walk-forward OOS validation (~5 mins)
    python run.py --multi-asset     # run on SPY, QQQ, IWM, TLT, GLD (~3 mins)
    python run.py --vol-signal      # add vol ratio dampening to ensemble
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
from src.data        import fetch_daily_bars, log_returns, fetch_multi, vix_levels
from src.geometric   import geometric_signal, straightness_ratio, MULTI_WINDOWS
from src.markov      import fit_markov3, select_k
from src.ensemble    import ensemble_score, regime_labels, vol_ratio
from src.backtest    import compute_stats, regime_return_stats, run_backtest
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
    parser.add_argument("--short",       action="store_true", help="Allow short on reversion days")
    parser.add_argument("--min-hold",    dest="min_hold", type=int, default=1, metavar="N",
                        help="Persistence filter: require N consecutive days in regime before switching (default: 1 = off)")
    parser.add_argument("--skip-bic",    action="store_true", help="Skip BIC model selection step (~60s)")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward OOS validation (~5 mins)")
    parser.add_argument("--multi-asset", action="store_true", help="Run on SPY, QQQ, IWM, TLT, GLD (~3 mins)")
    parser.add_argument("--vol-signal",  action="store_true", help="Add vol ratio dampening to ensemble (5-day/63-day realised vol)")
    parser.add_argument("--multi-scale", action="store_true", help="Use multi-scale geometric signal (average across 5, 15, 30-day windows)")
    parser.add_argument("--expanding",   action="store_true", help="Run expanding-window honest backtest (~5-10 mins)")
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
    if args.fetch_vix or args.vix_signal:
        print(f"  Fetching VIX (I:VIX)  {args.from_date} → {args.to_date}")
        vix_df = fetch_daily_bars("I:VIX", args.from_date, args.to_date)
        vix    = vix_levels(vix_df)
        print(f"  VIX: {len(vix_df)} days loaded  (range: {vix.min():.1f} – {vix.max():.1f})")

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
    _section("3. GEOMETRIC SIGNAL  (straightness ratio, %s)" % geo_desc)
    geo = geometric_signal(ret, window=15, windows=geo_windows)
    counts = geo.value_counts()
    for val, name in [(1.0, "Momentum"), (0.5, "Mixed"), (0.0, "Reversion")]:
        n = counts.get(val, 0)
        print(f"  {name:<12s}  {n:4d} days  ({n / len(geo) * 100:.1f}%)")

    # ── 4. Markov k=3 Signal ───────────────────────────────────────────
    _section("4. MARKOV k=3 SIGNAL  (filtered probabilities + transition analysis)")
    print("  Fitting  (~30s)...")
    mom_prob, crisis_prob, _, trans_info = fit_markov3(ret)

    # ── 5. Ensemble ────────────────────────────────────────────────────
    vix_signal = vix if args.vix_signal else None
    vol_ratio_signal = vol_ratio(ret) if args.vol_signal else None

    ensemble_parts = ["mean of geometric + Markov", "crisis override at P>0.50"]
    if vix_signal is not None:
        ensemble_parts.append("VIX dampening")
    if vol_ratio_signal is not None:
        ensemble_parts.append("vol ratio dampening (5d/63d)")
    _section("5. ENSEMBLE  (%s)" % ", ".join(ensemble_parts))

    if vol_ratio_signal is not None:
        vr_mean = vol_ratio_signal.mean()
        vr_pct_high = (vol_ratio_signal > 1.5).mean() * 100
        print(f"  Vol ratio (5d/63d):  mean={vr_mean:.2f}  "
              f"days with ratio>1.5 (dampening active): {vr_pct_high:.1f}%")

    score  = ensemble_score(geo, mom_prob, crisis_prob,
                            vix=vix_signal,
                            vol_ratio_series=vol_ratio_signal)
    labels = regime_labels(score)
    dist   = labels.value_counts()
    print("  Ensemble regime distribution:")
    for name in ["momentum", "mixed", "reversion"]:
        n = dist.get(name, 0)
        print(f"  {name:<12s}  {n:4d} days  ({n / len(labels) * 100:.1f}%)")

    # Vol/crisis overlap diagnostic — shows how much incremental work the vol
    # dampening does beyond what the Markov crisis override already captures.
    if vol_ratio_signal is not None:
        from src.ensemble import VOL_RATIO_SUPPRESS
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

    # ── 6. Forward Return Statistics ────────────────────────────────────
    _section("6. FORWARD RETURN STATISTICS  (next-day return by regime)")
    forward_ret = ret.shift(-1).rename("log_return")
    stats_df = regime_return_stats(forward_ret, labels)
    print(stats_df.to_string())
    print("\n  Interpretation guide:  |t| > 2.0, p < 0.05 = statistically significant")
    print("  Momentum signal is typically significant; reversion is weaker.")

    # ── 7. Backtest + Cost Sensitivity ────────────────────────────────
    _section("7. BACKTEST RESULTS")
    mode = "Long / Short" if args.short else "Long on momentum (+1), half-long on mixed (+0.5), cash on reversion"
    print(f"  Mode: {mode}")
    if args.min_hold > 1:
        print(f"  Persistence filter: {args.min_hold} consecutive days required before regime switch")
    print("  Signal execution: 1-day lag (signal at close T → trade at open T+1)\n")

    bt   = run_backtest(ret, labels, allow_short=args.short, cost_bps=0, min_hold_days=args.min_hold)
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
        bt_c   = run_backtest(ret, labels, allow_short=False, cost_bps=bps, min_hold_days=args.min_hold)
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

    # ── 8. Walk-Forward OOS Validation ─────────────────────────────────
    extra_sections = 0
    if args.walkforward:
        _section("8. WALK-FORWARD OOS VALIDATION  (10 folds x 63 days, fully OOS)")
        print("  Geometric: thresholds from train slice only.")
        print("  Markov: fitted on train, forward-filtered on test (no EM on test data).\n")
        wf_df, oos_returns = walk_forward(ret, n_folds=10, test_size=63)
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
        exp_bt = expanding_backtest(ret, multi_scale=multi_scale_flag, verbose=True)

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
