"""
Regime-following backtest and performance statistics.

Signal execution rules (daily, no leverage):
    regime = "momentum"  →  long  (+1)
    regime = "reversion" →  flat  ( 0)  [long-only variant]
                         →  short (-1)  [long/short, --short flag]
    regime = "mixed"     →  cash  ( 0)

Execution assumption: signal known at close of day T, trade executes at
open of day T+1. Implemented via signal.shift(1).

No transaction costs, no slippage, no borrow costs for shorts.
See README limitations for why the reported numbers overstate real performance.
"""

import numpy as np
import pandas as pd
from scipy import stats


def run_backtest(
    returns: pd.Series,
    regime: pd.Series,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    Run regime-following backtest.

    Args:
        returns     : daily log returns (named "log_return")
        regime      : regime labels ("momentum" / "reversion" / "mixed")
        allow_short : if True, go short (-1) on reversion days

    Returns:
        DataFrame with columns:
            log_return, signal, strategy_return, bnh_return,
            equity_strategy, equity_bnh
    """
    aligned = pd.concat([returns, regime], axis=1).dropna()
    ret = aligned["log_return"]
    reg = aligned["regime"]

    signal = pd.Series(0.0, index=reg.index, name="signal")
    signal[reg == "momentum"] = 1.0
    if allow_short:
        signal[reg == "reversion"] = -1.0

    # 1-day execution lag: signal at T → position at T+1
    strategy_ret = (signal.shift(1) * ret).rename("strategy_return")

    equity_strat = np.exp(strategy_ret.cumsum()).rename("equity_strategy")
    equity_bnh   = np.exp(ret.cumsum()).rename("equity_bnh")

    return pd.concat(
        [ret, signal, strategy_ret, ret.rename("bnh_return"), equity_strat, equity_bnh],
        axis=1,
    )


def compute_stats(bt: pd.DataFrame) -> dict:
    """
    Compute annualised return, Sharpe, max drawdown, and t-stat.

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
