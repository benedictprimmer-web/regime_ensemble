"""
Three-model test ladder for baseline comparison.

Paul's suggested research design:
    Model 1 — Purely simple (rule-based)
        price vs 200d MA, 30d MA slope, VIX bucket
        Tests whether explicit structural rules capture the edge without any ML.

    Model 2 — Simple plus stress
        Model 1 + realised vol overlay + drawdown filter + volume participation
        Tests whether adding observable stress variables improves over pure trend.

    Model 3 — Hidden-state on state variables
        The existing HMM ensemble (Geometric + Markov k=3 on 5-feature vector).
        See src/markov.py and src/ensemble.py. Not implemented here.

Each function returns a pd.Series of floats in [0, 1] aligned to the input
index, compatible with run_backtest(score=...) for continuous sizing.

Signal values:
    1.0  →  full long (momentum)
    0.75 →  mostly long
    0.5  →  half long (mixed / uncertain)
    0.0  →  cash (defensive)
"""

import numpy as np
import pandas as pd


# ── Helpers ────────────────────────────────────────────────────────────────────

def _above_200d(prices: pd.Series) -> pd.Series:
    """True where price is above its 200-day simple moving average."""
    ma200 = prices.rolling(200, min_periods=50).mean()
    return (prices > ma200).rename("above_200d")


def _30d_slope_positive(prices: pd.Series) -> pd.Series:
    """True where the 30-day MA slope is positive (today's MA > yesterday's MA)."""
    ma30 = prices.rolling(30, min_periods=10).mean()
    return (ma30 > ma30.shift(1)).rename("slope_30d_pos")


def _vix_bucket(vix: pd.Series) -> pd.Series:
    """
    Classify VIX level into three buckets:
        'low'  → VIX < 20   (calm, supportive)
        'mid'  → 20 ≤ VIX < 30  (elevated, caution)
        'high' → VIX ≥ 30   (fear, reduce)
    """
    bucket = pd.Series("mid", index=vix.index, name="vix_bucket")
    bucket[vix < 20]  = "low"
    bucket[vix >= 30] = "high"
    return bucket


def _rvol_stressed(returns: pd.Series,
                   short_window: int = 20,
                   long_window: int = 63,
                   multiplier: float = 1.5) -> pd.Series:
    """
    True where short-term realised vol exceeds multiplier × long-term realised vol.
    Flags turbulent / crisis-like volatility regimes.
    """
    ann = np.sqrt(252)
    short_rvol = returns.rolling(short_window, min_periods=10).std() * ann
    long_rvol  = returns.rolling(long_window,  min_periods=30).std() * ann
    ratio = short_rvol / long_rvol.replace(0, np.nan)
    return (ratio > multiplier).fillna(False).rename("rvol_stressed")


def _drawdown_exceeded(prices: pd.Series, threshold: float = -0.10) -> pd.Series:
    """
    True where drawdown from the 252-day rolling high exceeds threshold.
    Default -0.10 means 10% drawdown from peak.
    """
    peak = prices.rolling(252, min_periods=63).max()
    dd   = prices / peak - 1
    return (dd < threshold).rename("dd_exceeded")


def _volume_below_avg(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    True where today's volume is below the rolling average.
    Indicates low participation; trend moves on low volume are less reliable.
    Returns all-False if volume is None or all-zero.
    """
    if volume is None or (volume == 0).all():
        return pd.Series(False, index=volume.index if volume is not None else pd.Index([]))
    avg = volume.rolling(window, min_periods=5).mean()
    return (volume < avg).rename("vol_below_avg")


# ── Model 1: Purely simple rule-based ─────────────────────────────────────────

def model1_signal(
    prices: pd.Series,
    vix: pd.Series = None,
) -> pd.Series:
    """
    Model 1: pure structural rule — 200d MA + 30d slope + VIX bucket.

    Signal mapping:
        Below 200d MA                           → 0.0  (defensive)
        Above 200d, 30d slope negative          → 0.5  (mixed / pullback)
        Above 200d, 30d slope positive, VIX high→ 0.5  (stressed bull)
        Above 200d, 30d slope positive, VIX mid → 0.75 (cautious momentum)
        Above 200d, 30d slope positive, VIX low → 1.0  (full momentum)
        Above 200d, 30d slope positive, no VIX  → 1.0  (VIX not available)

    Args:
        prices : daily close price series (not returns)
        vix    : optional VIX daily close levels; if None, VIX bucket = 'low'
                 for all dates (effectively ignores VIX adjustment)

    Returns:
        pd.Series of floats in {0.0, 0.5, 0.75, 1.0}
    """
    above200 = _above_200d(prices)
    slope_pos = _30d_slope_positive(prices)

    if vix is not None:
        vix_aligned = vix.reindex(prices.index, method="ffill")
        bucket = _vix_bucket(vix_aligned)
    else:
        bucket = pd.Series("low", index=prices.index, name="vix_bucket")

    signal = pd.Series(0.0, index=prices.index, name="model1_signal")

    in_trend = above200
    signal[in_trend & slope_pos & (bucket == "low")]  = 1.0
    signal[in_trend & slope_pos & (bucket == "mid")]  = 0.75
    signal[in_trend & slope_pos & (bucket == "high")] = 0.5
    signal[in_trend & ~slope_pos]                     = 0.5

    return signal


# ── Model 2: Simple plus stress overlays ──────────────────────────────────────

def model2_signal(
    prices: pd.Series,
    returns: pd.Series,
    vix: pd.Series = None,
    volume: pd.Series = None,
    rvol_multiplier: float = 1.5,
    drawdown_threshold: float = -0.10,
) -> pd.Series:
    """
    Model 2: Model 1 + stress overlays (rvol, drawdown, volume participation).

    Overlays applied sequentially on top of Model 1 signal:
        1. If rvol stressed (short > 1.5× long):  → 0.0 (capital preservation)
        2. If drawdown > threshold (default 10%):  → 0.0 (capital preservation)
        3. If volume below 20d avg:               → halve signal (low conviction)

    The rvol and drawdown rules are hard overrides because they represent
    conditions where equity risk is materially elevated regardless of trend.
    The volume rule is a soft reduction because low-volume trends can persist;
    it reduces position but does not exit entirely.

    Args:
        prices             : daily close price series
        returns            : daily log returns
        vix                : optional VIX daily close levels
        volume             : optional daily volume series; if None, volume
                             participation overlay is skipped
        rvol_multiplier    : threshold for short/long vol ratio (default 1.5)
        drawdown_threshold : max drawdown from 252d high (default -0.10)

    Returns:
        pd.Series of floats in [0, 1]
    """
    base = model1_signal(prices, vix=vix)

    rvol_flag = _rvol_stressed(returns, multiplier=rvol_multiplier)
    dd_flag   = _drawdown_exceeded(prices, threshold=drawdown_threshold)

    signal = base.copy().rename("model2_signal")

    # Hard overrides: zero out on stress
    stress = rvol_flag | dd_flag
    signal[stress] = 0.0

    # Soft reduction: low volume halves signal
    if volume is not None:
        vol_flag = _volume_below_avg(volume).reindex(signal.index, fill_value=False)
        signal[~stress & vol_flag] *= 0.5

    return signal
