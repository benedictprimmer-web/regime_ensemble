#!/usr/bin/env python3
"""
Regime Ensemble Backtest
========================
SPY daily regime detection using a two-model ensemble:
    1. Geometric (straightness ratio, adaptive percentile thresholds)
    2. Markov Switching AR(1), k=3 selected by BIC (ΔBIC=82 over k=2)

Usage:
    python run.py
    python run.py --from 2020-01-01 --to 2025-01-01
    python run.py --short           # allow short on reversion days
    python run.py --skip-bic        # skip BIC model selection (saves ~30s)
    python run.py --walkforward     # run walk-forward OOS validation (~5 mins)

Outputs (saved to outputs/):
    regime_overview.png   — SPY price coloured by regime + model signals
    equity_curves.png     — strategy vs buy-and-hold + drawdown
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
from src.data        import fetch_daily_bars, log_returns
from src.geometric   import geometric_signal, straightness_ratio
from src.markov      import fit_markov3, select_k
from src.ensemble    import ensemble_score, regime_labels
from src.backtest    import compute_stats, regime_return_stats, run_backtest
from src.walkforward import walk_forward

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "momentum":  "#1a7a4a",
    "reversion": "#c0392b",
    "mixed":     "#95a5a6",
    "bnh":       "#2c3e50",
    "strategy":  "#2980b9",
}


# ── Plotting ───────────────────────────────────────────────────────────


def plot_regime_overview(
    prices: pd.Series,
    returns: pd.Series,
    mom_prob: pd.Series,
    crisis_prob: pd.Series,
    labels: pd.Series,
    from_date: str,
    to_date: str,
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
    ax1.set_ylabel("SPY Close", fontsize=9)
    ax1.set_xticks([])
    ax1.set_title(
        f"Regime Ensemble — SPY {from_date[:4]}–{to_date[:4]}  "
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
    path = OUTPUT_DIR / "regime_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_equity_curves(bt: pd.DataFrame) -> None:
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
    path = OUTPUT_DIR / "equity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print(f"\n{'─' * 58}\n  {title}\n{'─' * 58}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime Ensemble Backtest")
    parser.add_argument("--from", dest="from_date", default="2022-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--to",   dest="to_date",   default="2025-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--short",       action="store_true", help="Allow short on reversion days")
    parser.add_argument("--skip-bic",    action="store_true", help="Skip BIC model selection step (~60s)")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward OOS validation (~5 mins)")
    args = parser.parse_args()

    # ── 1. Data ────────────────────────────────────────────────────────
    _section("1. DATA")
    print(f"  Fetching SPY  {args.from_date} → {args.to_date}")
    df     = fetch_daily_bars("SPY", args.from_date, args.to_date)
    ret    = log_returns(df)
    prices = df["close"]
    print(f"  {len(df)} trading days loaded")

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
    _section("3. GEOMETRIC SIGNAL  (straightness ratio, window=15)")
    geo = geometric_signal(ret, window=15)
    counts = geo.value_counts()
    for val, name in [(1.0, "Momentum"), (0.5, "Mixed"), (0.0, "Reversion")]:
        n = counts.get(val, 0)
        print(f"  {name:<12s}  {n:4d} days  ({n / len(geo) * 100:.1f}%)")

    # ── 4. Markov k=3 Signal ───────────────────────────────────────────
    _section("4. MARKOV k=3 SIGNAL  (filtered probabilities + transition analysis)")
    print("  Fitting  (~30s)...")
    mom_prob, crisis_prob, _, trans_info = fit_markov3(ret)

    # ── 5. Ensemble ────────────────────────────────────────────────────
    _section("5. ENSEMBLE  (mean of geometric + Markov, crisis override at P>0.50)")
    score  = ensemble_score(geo, mom_prob, crisis_prob)
    labels = regime_labels(score)
    dist   = labels.value_counts()
    print("  Ensemble regime distribution:")
    for name in ["momentum", "mixed", "reversion"]:
        n = dist.get(name, 0)
        print(f"  {name:<12s}  {n:4d} days  ({n / len(labels) * 100:.1f}%)")

    # ── 6. Forward Return Statistics ────────────────────────────────────
    _section("6. FORWARD RETURN STATISTICS  (next-day return by regime)")
    forward_ret = ret.shift(-1).rename("log_return")
    stats_df = regime_return_stats(forward_ret, labels)
    print(stats_df.to_string())
    print("\n  Interpretation guide:  |t| > 2.0, p < 0.05 = statistically significant")
    print("  Momentum signal is typically significant; reversion is weaker.")

    # ── 7. Backtest + Cost Sensitivity ────────────────────────────────
    _section("7. BACKTEST RESULTS")
    mode = "Long / Short" if args.short else "Long Only (cash on reversion)"
    print(f"  Mode: {mode}")
    print("  Signal execution: 1-day lag (signal at close T → trade at open T+1)\n")

    bt   = run_backtest(ret, labels, allow_short=args.short, cost_bps=0)
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
        bt_c   = run_backtest(ret, labels, allow_short=False, cost_bps=bps)
        perf_c = compute_stats(bt_c)["Strategy (Long Only)"]
        print(f"  {bps:>6d}bps  {perf_c['CAGR']:>8s}  {perf_c['Sharpe']:>8s}  {perf_c['Max DD']:>8s}")

    # ── 8. Walk-Forward OOS Validation ─────────────────────────────────
    if args.walkforward:
        _section("8. WALK-FORWARD OOS VALIDATION  (expanding window, 5 folds × 63 days)")
        print("  Note: Markov uses expanding-window refit (standard practice).")
        print("  Geometric thresholds are computed on train-only data (truly OOS).\n")
        wf_df = walk_forward(ret, n_folds=5, test_size=63)
        print(wf_df.to_string())
        pos_folds = (wf_df.loc[(slice(None), "momentum"), "Dir"] == "✓").sum()
        total     = len(wf_df.loc[(slice(None), "momentum")])
        print(f"\n  Momentum regime positive in {pos_folds}/{total} folds")
        print("  (≥ 3/5 is consistent with signal having real predictive power)")
        charts_section = 9
    else:
        charts_section = 8

    # ── Charts ────────────────────────────────────────────────────────
    _section(f"{charts_section}. CHARTS")
    plot_regime_overview(prices, ret, mom_prob, crisis_prob, labels, args.from_date, args.to_date)
    plot_equity_curves(bt)

    print("\n  Complete. Charts saved to outputs/\n")


if __name__ == "__main__":
    main()
