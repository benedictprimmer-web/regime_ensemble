"""
Geometric regime detection via the straightness ratio.

Core idea: measure how "straight" the price path was over the last N days.

    ratio(t) = |sum(r_{t-N+1} .. r_t)| / sum(|r_{t-N+1}| .. |r_t|)

    ratio → 1.0  →  price moved in a straight line  →  MOMENTUM
    ratio → 0.0  →  price zigzagged back to start    →  REVERSION

Thresholds are percentile-based (adaptive):
    top 30%    of the ratio distribution → momentum
    bottom 30% of the ratio distribution → reversion

Fixed thresholds (e.g. ratio > 0.7) break at longer windows where noise
accumulates and the ratio rarely exceeds 0.5 even in genuine trends.
Adaptive thresholds are robust to window length and volatility regime.

When `mom_thresh`/`rev_thresh` are supplied to `geometric_signal()` from
a training slice via `compute_thresholds()`, there is no look-ahead.
For the full in-sample run (run.py), thresholds are computed on the full
history — this is fine for the in-sample backtest but not for walk-forward.
"""

import numpy as np
import pandas as pd

WINDOW          = 15   # 3 calendar weeks — captures short-term momentum
MOMENTUM_PCT    = 70   # top 30% of distribution → momentum
REVERSION_PCT   = 30   # bottom 30% of distribution → reversion


def straightness_ratio(returns: pd.Series, window: int = WINDOW) -> pd.Series:
    """
    Rolling straightness ratio over `window` bars.

    Returns:
        Series of values in [0, 1] aligned to the returns index.
    """
    net   = returns.rolling(window).sum().abs()
    total = returns.abs().rolling(window).sum()
    ratio = (net / total.replace(0, np.nan)).fillna(0)
    return ratio.rename("straightness")


def compute_thresholds(
    returns: pd.Series,
    window: int          = WINDOW,
    momentum_pct: float  = MOMENTUM_PCT,
    reversion_pct: float = REVERSION_PCT,
) -> tuple:
    """
    Compute straightness ratio percentile thresholds from a data slice.

    Call this on the training slice, then pass the results to
    `geometric_signal()` to avoid look-ahead bias in walk-forward validation.

    Returns:
        (mom_thresh, rev_thresh) — float pair
    """
    ratio = straightness_ratio(returns, window)
    return ratio.quantile(momentum_pct / 100), ratio.quantile(reversion_pct / 100)


def geometric_signal(
    returns: pd.Series,
    window: int          = WINDOW,
    momentum_pct: float  = MOMENTUM_PCT,
    reversion_pct: float = REVERSION_PCT,
    mom_thresh: float    = None,
    rev_thresh: float    = None,
) -> pd.Series:
    """
    Geometric regime signal as a 0/0.5/1 float.

        1.0  momentum  (top momentum_pct% of ratio distribution)
        0.5  mixed
        0.0  reversion (bottom reversion_pct% of ratio distribution)

    Pass `mom_thresh` and `rev_thresh` (from `compute_thresholds()` on a
    training slice) to avoid look-ahead bias in walk-forward validation.
    When omitted, thresholds are computed on the full `returns` series.

    Returns:
        Series aligned to returns.index.
    """
    ratio = straightness_ratio(returns, window)
    if mom_thresh is None:
        mom_thresh = ratio.quantile(momentum_pct / 100)
    if rev_thresh is None:
        rev_thresh = ratio.quantile(reversion_pct / 100)

    signal = pd.Series(0.5, index=returns.index, name="geo_signal")
    signal[ratio >= mom_thresh] = 1.0
    signal[ratio <= rev_thresh] = 0.0
    return signal
