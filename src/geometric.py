"""
Geometric regime detection via the straightness ratio.

Core idea: measure how "straight" the price path was over the last N days.

    ratio(t) = |sum(r_{t-N+1} .. r_t)| / sum(|r_{t-N+1}| .. |r_t|)

    ratio → 1.0  →  price moved in a straight line  →  MOMENTUM
    ratio → 0.0  →  price zigzagged back to start    →  REVERSION

Directional variant (directional=True):

    ratio(t) = sum(r_{t-N+1} .. r_t) / sum(|r_{t-N+1}| .. |r_t|)

    ratio → +1.0  →  straight UP    →  momentum (buy)
    ratio →  0.0  →  choppy          →  reversion (cash)
    ratio → -1.0  →  straight DOWN   →  reversion (cash)

The directional variant fixes the standard version's direction-blindness: a
clean crash scores +1.0 in the standard ratio (triggering a buy signal) but
−1.0 directionally (correctly triggering cash). Markov's crisis override
still protects against sustained crashes, but the directional signal acts
faster — it responds on the first window after the move begins, not after
Markov's EM has accumulated enough evidence.

Thresholds are percentile-based (adaptive):
    top 30%    of the ratio distribution → momentum
    bottom 30% of the ratio distribution → reversion

When `mom_thresh`/`rev_thresh` are supplied to `geometric_signal()` from
a training slice via `compute_thresholds()`, there is no look-ahead.
For the full in-sample run (run.py), thresholds are computed on the full
history — this is fine for the in-sample backtest but not for walk-forward.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

WINDOW          = 15   # 3 calendar weeks — captures short-term momentum
MOMENTUM_PCT    = 70   # top 30% of distribution → momentum
REVERSION_PCT   = 30   # bottom 30% of distribution → reversion
MULTI_WINDOWS   = [5, 15, 30]   # default multi-scale windows (1wk, 3wk, 6wk)


def straightness_ratio(
    returns: pd.Series,
    window: int = WINDOW,
    directional: bool = False,
) -> pd.Series:
    """
    Rolling straightness ratio over `window` bars.

    Args:
        directional : if False (default), ratio is in [0, 1] — direction-blind.
                      if True, ratio is in [-1, 1]: positive = uptrend,
                      negative = downtrend. Fixes direction-blindness where
                      straight crashes score identically to straight rallies.

    Returns:
        Series aligned to the returns index.
    """
    net   = returns.rolling(window).sum()
    total = returns.abs().rolling(window).sum()
    if not directional:
        net = net.abs()
    ratio = (net / total.replace(0, np.nan)).fillna(0)
    return ratio.rename("straightness")


def multi_scale_ratio(
    returns: pd.Series,
    windows: List[int] = MULTI_WINDOWS,
    directional: bool = False,
) -> pd.Series:
    """
    Average straightness ratio across multiple windows.

    Combining short (5-day), medium (15-day), and long (30-day) windows
    reduces single-window noise: fast windows catch sharp reversals,
    slow windows capture sustained trends. The average is smoother and
    more robust across volatility regimes than any single window.

    Returns:
        Series of averaged ratios, aligned to returns.index.
    """
    ratios = [straightness_ratio(returns, w, directional=directional) for w in windows]
    return pd.concat(ratios, axis=1).mean(axis=1).rename("straightness_multi")


def compute_thresholds(
    returns: pd.Series,
    window: int          = WINDOW,
    momentum_pct: float  = MOMENTUM_PCT,
    reversion_pct: float = REVERSION_PCT,
    windows: Optional[List[int]] = None,
    directional: bool = False,
) -> tuple:
    """
    Compute straightness ratio percentile thresholds from a data slice.

    Call this on the training slice, then pass the results to
    `geometric_signal()` to avoid look-ahead bias in walk-forward validation.

    Args:
        windows     : if provided, use multi-scale averaged ratio (overrides window)
        directional : pass through to straightness_ratio()

    Returns:
        (mom_thresh, rev_thresh) — float pair
    """
    if windows is not None:
        ratio = multi_scale_ratio(returns, windows, directional=directional)
    else:
        ratio = straightness_ratio(returns, window, directional=directional)
    return ratio.quantile(momentum_pct / 100), ratio.quantile(reversion_pct / 100)


def geometric_signal(
    returns: pd.Series,
    window: int                   = WINDOW,
    momentum_pct: float           = MOMENTUM_PCT,
    reversion_pct: float          = REVERSION_PCT,
    mom_thresh: float             = None,
    rev_thresh: float             = None,
    windows: Optional[List[int]]  = None,
    directional: bool             = False,
) -> pd.Series:
    """
    Geometric regime signal as a 0/0.5/1 float.

        1.0  momentum  (top momentum_pct% of ratio distribution)
        0.5  mixed
        0.0  reversion (bottom reversion_pct% of ratio distribution)

    Args:
        windows     : if provided, use multi-scale averaged ratio across these
                      windows (e.g. [5, 15, 30]) instead of single `window`.
                      Overrides the `window` parameter.
        mom_thresh  : pre-computed threshold (from compute_thresholds on train
                      slice) — avoids look-ahead in walk-forward validation.
        rev_thresh  : pre-computed threshold (same)
        directional : if True, use signed ratio so straight-down moves score
                      low (cash) rather than high (momentum). The 70th/30th
                      percentile thresholds are re-computed on the signed
                      distribution and naturally separate uptrends, choppy,
                      and downtrends.

    Returns:
        Series aligned to returns.index.
    """
    if windows is not None:
        ratio = multi_scale_ratio(returns, windows, directional=directional)
    else:
        ratio = straightness_ratio(returns, window, directional=directional)

    if mom_thresh is None:
        mom_thresh = ratio.quantile(momentum_pct / 100)
    if rev_thresh is None:
        rev_thresh = ratio.quantile(reversion_pct / 100)

    signal = pd.Series(0.5, index=returns.index, name="geo_signal")
    signal[ratio >= mom_thresh] = 1.0
    signal[ratio <= rev_thresh] = 0.0
    return signal
