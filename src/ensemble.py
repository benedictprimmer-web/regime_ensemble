"""
Two-model ensemble: Geometric + Markov k=3, with optional dampening signals.

Ensemble score = mean( geo_signal, markov_signal_crisis_adjusted )

    geo_signal      : {0.0, 0.5, 1.0}  hard label -> float
    markov_signal   : P(momentum)       0-1 continuous (filtered)
    crisis override : if P(crisis) > 0.50, set markov_signal = 0
                      regardless of P(momentum) -- capital preservation first
    VIX dampening   : if vix is provided, scale markov_adj by a fear factor
                      vix_factor = 1 - clip((vix - 20) / 20, 0, 1)
                      VIX <= 20 -> no change; VIX 30 -> halved; VIX >= 40 -> zero
    Vol ratio       : if vol_ratio is provided, scale markov_adj by a turbulence
                      factor derived from realised vol:
                      vol_factor = 1 - clip((ratio - 1.0) / 1.0, 0, 1)
                      ratio <= 1.0 -> no dampening; ratio = 2.0 -> fully suppressed
                      (ratio = short_vol / long_vol, e.g. 5-day ann. / 63-day ann.)

Score -> regime label:
    >= 0.65  ->  momentum   (both models leaning momentum)
    <= 0.35  ->  reversion  (both models leaning reversion)
    0.35-0.65 -> mixed       (models disagree or both uncertain)

Why equal weighting?
    Fitting weights to historical returns would constitute in-sample
    optimisation. Equal weights are more conservative and defensible.
    The ensemble value comes from combining orthogonal signals
    (short-term path shape vs long-term statistical state), not from
    fitting weights.

Why continuous dampening rather than a hard override?
    Both VIX and vol ratio are turbulence indicators orthogonal to the
    Markov crisis state (which is backward-looking). Continuous dampening
    avoids a cliff-edge at a single threshold and keeps the existing score
    calibration intact on normal days (the majority of trading days).
"""

import numpy as np
import pandas as pd
from typing import Optional

MOMENTUM_THRESHOLD  = 0.65
REVERSION_THRESHOLD = 0.35
CRISIS_THRESHOLD    = 0.50
VIX_NEUTRAL         = 20.0   # VIX at or below this: no dampening
VIX_FULL_SUPPRESS   = 40.0   # VIX at or above this: momentum fully suppressed
VOL_RATIO_NEUTRAL   = 1.0    # short/long vol at or below this: no dampening
VOL_RATIO_SUPPRESS  = 2.0    # short/long vol at or above this: momentum fully suppressed


def vol_ratio(returns: pd.Series,
              short_window: int = 5,
              long_window: int = 63) -> pd.Series:
    """
    Compute rolling realised vol ratio: short-term ann. vol / long-term ann. vol.

    Values > 1.0 mean short-term vol is elevated relative to baseline.
    Values >> 2.0 indicate turbulent, crisis-like conditions.

    Args:
        returns      : daily log returns
        short_window : short rolling window in days (default 5 = 1 week)
        long_window  : long rolling window in days  (default 63 = 1 quarter)

    Returns:
        Series of ratios, aligned to returns.index.
    """
    ann = np.sqrt(252)
    short_vol = returns.rolling(short_window).std() * ann
    long_vol  = returns.rolling(long_window).std()  * ann
    return (short_vol / long_vol.replace(0, np.nan)).rename("vol_ratio")


def ensemble_score(
    geo: pd.Series,
    markov_mom: pd.Series,
    markov_crisis: pd.Series,
    vix: Optional[pd.Series] = None,
    vol_ratio_series: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute ensemble score as mean of geometric and crisis-adjusted Markov.

    Dampening is applied to the Markov component before averaging.
    Multiple dampening signals are applied multiplicatively (independent).

    Args:
        geo              : geometric signal {0.0, 0.5, 1.0}
        markov_mom       : Markov P(momentum), 0-1
        markov_crisis    : Markov P(crisis),   0-1
        vix              : optional VIX level series (requires Polygon paid plan)
        vol_ratio_series : optional realised vol ratio (short_vol / long_vol);
                           use vol_ratio() to compute from returns

    Returns:
        Series of scores in [0, 1].
    """
    markov_adj = markov_mom.copy()
    markov_adj[markov_crisis > CRISIS_THRESHOLD] = 0.0

    if vix is not None:
        vix_aligned = vix.reindex(markov_adj.index, method="ffill")
        vix_factor = 1.0 - np.clip(
            (vix_aligned - VIX_NEUTRAL) / (VIX_FULL_SUPPRESS - VIX_NEUTRAL),
            0.0, 1.0,
        )
        markov_adj = markov_adj * vix_factor

    if vol_ratio_series is not None:
        vr_aligned = vol_ratio_series.reindex(markov_adj.index, method="ffill")
        vr_factor = 1.0 - np.clip(
            (vr_aligned - VOL_RATIO_NEUTRAL) / (VOL_RATIO_SUPPRESS - VOL_RATIO_NEUTRAL),
            0.0, 1.0,
        )
        markov_adj = markov_adj * vr_factor

    # Markov loses one observation (AR lag) -- dropna aligns on common index
    aligned = pd.concat([geo, markov_adj], axis=1).dropna()
    return aligned.mean(axis=1).rename("ensemble_score")


def regime_labels(score: pd.Series) -> pd.Series:
    """Map ensemble scores to regime labels."""
    labels = pd.Series("mixed", index=score.index, name="regime")
    labels[score >= MOMENTUM_THRESHOLD]  = "momentum"
    labels[score <= REVERSION_THRESHOLD] = "reversion"
    return labels
