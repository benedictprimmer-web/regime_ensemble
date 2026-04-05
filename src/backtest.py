"""
Regime-following backtest and performance statistics.

Signal execution rules (daily, no leverage):
    regime = "momentum"  →  full long   (+1.0)
    regime = "mixed"     →  half long   (+0.5)  -- statistically significant
    regime = "reversion" →  cash        ( 0.0)  [long-only variant]
                         →  short       (-1.0)  [long/short, --short flag]

Continuous sizing mode (score parameter):
    When a `score` Series is passed to run_backtest(), discrete regime labels
    are ignored and position = score (clipped to [0, 1]). This eliminates
    hard threshold transitions and reduces the ~61 label switches/year.
    Transaction costs are lower because position changes are gradual.

"Mixed" gets a half-position because it has the strongest forward-return
signal (T=3.21, p=0.001 on 2000-2025 data). The original cash position
was leaving statistically significant return on the table.

Execution assumption: signal known at close of day T, trade executes at
open of day T+1. Implemented via signal.shift(1).

Transaction cost model: round-trip cost applied proportional to position change.
    |delta_position| = 0.5 for 0 <-> 0.5 or 0.5 <-> 1.0: cost_bps * 0.5
    |delta_position| = 1.0 for 0 <-> 1.0: cost_bps
    |delta_position| = 2.0 for +1 <-> -1: 2 * cost_bps
"""

import numpy as np
import pandas as pd
from scipy import stats


def _apply_persistence_filter(regime: pd.Series, min_hold_days: int) -> pd.Series:
    """
    Require a regime to hold for min_hold_days consecutive days before
    the position switches. Reduces whipsawing and transaction costs.

    Example (min_hold_days=3):
        raw:      mom mom rev rev rev mom mom ...
        filtered: mom mom mom mom rev rev rev ...
                              ^^^--- switch delayed until 3rd consecutive day
    """
    filtered = regime.copy()
    held = regime.iloc[0]
    candidate = regime.iloc[0]
    streak = 1
    for i in range(1, len(regime)):
        sig = regime.iloc[i]
        if sig == candidate:
            streak += 1
        else:
            candidate = sig
            streak = 1
        if streak >= min_hold_days:
            held = candidate
        filtered.iloc[i] = held
    return filtered


def run_backtest(
    returns: pd.Series,
    regime: pd.Series,
    allow_short: bool = False,
    cost_bps: float = 0,
    min_hold_days: int = 1,
    score: pd.Series = None,
) -> pd.DataFrame:
    """
    Run regime-following backtest.

    Args:
        returns       : daily log returns (named "log_return")
        regime        : regime labels ("momentum" / "reversion" / "mixed")
        allow_short   : if True, go short (-1) on reversion days
        cost_bps      : round-trip transaction cost per switch in basis points
        min_hold_days : persistence filter -- regime must hold this many
                        consecutive days before position changes (default=1,
                        i.e. no filtering). Use 3-5 to reduce turnover.
        score         : optional continuous ensemble score [0, 1]. When
                        provided, position = score directly (clipped to [0,1]),
                        bypassing discrete regime labels. Reduces label
                        switches and transaction costs at any cost level.

    Returns:
        DataFrame with columns:
            log_return, signal, strategy_return, bnh_return,
            equity_strategy, equity_bnh
    """
    if score is not None:
        # Continuous sizing: position = ensemble score, no discrete labels
        aligned = pd.concat([returns.rename("log_return"), score], axis=1).dropna()
        ret    = aligned["log_return"]
        signal = aligned[score.name].clip(0, 1).rename("signal")
    else:
        aligned = pd.concat([returns, regime], axis=1).dropna()
        ret = aligned["log_return"]
        reg = aligned["regime"]

        if min_hold_days > 1:
            reg = _apply_persistence_filter(reg, min_hold_days)

        signal = pd.Series(0.0, index=reg.index, name="signal")
        signal[reg == "momentum"] = 1.0
        signal[reg == "mixed"]    = 0.5
        if allow_short:
            signal[reg == "reversion"] = -1.0

    # 1-day execution lag: signal at T → position at T+1
    gross_ret = (signal.shift(1) * ret)

    # Transaction costs: proportional to position change magnitude
    # |Δposition| = 1 for 0↔1, 1 for 0↔-1, 2 for +1↔-1
    switch_cost = signal.diff().abs() * (cost_bps / 10_000)
    strategy_ret = (gross_ret - switch_cost.shift(1)).rename("strategy_return")

    equity_strat = np.exp(strategy_ret.cumsum()).rename("equity_strategy")
    equity_bnh   = np.exp(ret.cumsum()).rename("equity_bnh")

    return pd.concat(
        [ret, signal, strategy_ret, ret.rename("bnh_return"), equity_strat, equity_bnh],
        axis=1,
    )


def compute_stats(bt: pd.DataFrame, raw: bool = False) -> dict:
    """
    Compute annualised return, Sharpe, max drawdown, and t-stat.

    Args:
        raw : if True, return raw floats instead of formatted strings.
              Useful for programmatic comparison (e.g. multi-asset table).

    Returns dict: {"Strategy": {...}, "Buy & Hold": {...}}
    """
    def _stats(col: str) -> dict:
        r   = bt[col].dropna()
        n   = len(r)
        ann = 252
        cagr    = np.exp(r.sum()) ** (ann / n) - 1
        sharpe  = (r.mean() * ann) / (r.std() * np.sqrt(ann)) if r.std() > 0 else np.nan
        eq      = np.exp(r.cumsum())
        max_dd  = (eq / eq.cummax() - 1).min()
        t, p    = stats.ttest_1samp(r, 0)
        if raw:
            return {"CAGR": cagr, "Sharpe": sharpe, "Max DD": max_dd, "T-stat": t, "P-value": p}
        return {
            "CAGR":     f"{cagr*100:+.1f}%",
            "Sharpe":   f"{sharpe:.2f}",
            "Max DD":   f"{max_dd*100:.1f}%",
            "T-stat":   f"{t:.2f}",
            "P-value":  f"{p:.3f}",
        }

    return {
        "Strategy (Long Only)": _stats("strategy_return"),
        "Buy & Hold":           _stats("bnh_return"),
    }


def attribution_grid(
    ret: pd.Series,
    geo: pd.Series,
    markov_mom: pd.Series,
) -> list:
    """
    3x3 forward-return grid: geo signal level × Markov P(momentum) bin.

    Rows: geometric signal ∈ {0.0 reversion, 0.5 mixed, 1.0 momentum}
    Cols: markov_mom ∈ {low <0.33, mid 0.33–0.67, high >0.67}

    Each cell: (mean_fwd_return_pct, t_stat, p_value, n_days).

    Returns a list of dicts, one per non-empty cell, suitable for printing.
    """
    df = pd.DataFrame({
        "geo": geo,
        "mom": markov_mom,
        "fwd": ret.shift(-1),
    }).dropna()

    geo_labels = {0.0: "reversion", 0.5: "mixed", 1.0: "momentum"}
    mom_bins   = [0.0, 0.333, 0.667, 1.001]
    mom_labels = ["low (<0.33)", "mid (0.33-0.67)", "high (>0.67)"]

    df["geo_lbl"] = df["geo"].map(geo_labels)
    df["mom_lbl"] = pd.cut(df["mom"], bins=mom_bins, labels=mom_labels, include_lowest=True)

    rows = []
    for g in ["reversion", "mixed", "momentum"]:
        for m in mom_labels:
            cell = df[(df["geo_lbl"] == g) & (df["mom_lbl"] == m)]["fwd"]
            n = len(cell)
            if n < 5:
                rows.append({"geo": g, "mom_bin": m, "mean_pct": np.nan, "t": np.nan, "p": np.nan, "n": n})
            else:
                t, p = stats.ttest_1samp(cell, 0)
                rows.append({"geo": g, "mom_bin": m, "mean_pct": cell.mean() * 100, "t": t, "p": p, "n": n})
    return rows


def regime_return_stats(forward_returns: pd.Series, regime: pd.Series) -> pd.DataFrame:
    """
    Mean next-day return, t-stat, and p-value by ensemble regime label.

    This is the primary statistical test of predictive power.
    A signal with no edge produces t-stats near zero across all regimes.

    Args:
        forward_returns : ret.shift(-1) — what happens the day after
        regime          : regime labels at signal time
    """
    aligned = pd.concat([forward_returns, regime], axis=1).dropna()
    rows = []
    for label in ["momentum", "mixed", "reversion"]:
        r = aligned.loc[aligned["regime"] == label, aligned.columns[0]]
        if len(r) < 10:
            continue
        t, p = stats.ttest_1samp(r, 0)
        rows.append({
            "Regime":      label,
            "N days":      len(r),
            "Mean %/day":  f"{r.mean()*100:+.4f}%",
            "T-stat":      f"{t:.2f}",
            "P-value":     f"{p:.3f}",
            "Sig":         "✓" if p < 0.05 else ("~" if p < 0.10 else "✗"),
        })
    return pd.DataFrame(rows).set_index("Regime")
