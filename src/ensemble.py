"""
Two-model ensemble: Geometric + Markov k=3, with optional VIX dampening.

Ensemble score = mean( geo_signal, markov_signal_crisis_adjusted )

    geo_signal      : {0.0, 0.5, 1.0}  hard label -> float
    markov_signal   : P(momentum)       0-1 continuous (filtered)
    crisis override : if P(crisis) > 0.50, set markov_signal = 0
                      regardless of P(momentum) -- capital preservation first
    VIX dampening   : if vix is provided, scale markov_adj by a fear factor
                      vix_factor = 1 - clip((vix - 20) / 20, 0, 1)
                      VIX <= 20 -> no change; VIX 30 -> halved; VIX >= 40 -> zero

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

Why VIX dampening rather than a hard override?
    VIX is forward-looking implied vol -- orthogonal to the Markov crisis
    state which is backward-looking. A continuous dampening factor avoids
    a cliff-edge at a single threshold and keeps the existing score
    calibration intact (VIX < 20, the majority of trading days, leaves
    the signal unchanged).
"""

import numpy as np
import pandas as pd
from typing import Optional

MOMENTUM_THRESHOLD  = 0.65
REVERSION_THRESHOLD = 0.35
CRISIS_THRESHOLD    = 0.50
VIX_NEUTRAL         = 20.0   # VIX at or below this: no dampening
VIX_FULL_SUPPRESS   = 40.0   # VIX at or above this: momentum fully suppressed


def ensemble_score(
    geo: pd.Series,
    markov_mom: pd.Series,
    markov_crisis: pd.Series,
    vix: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute ensemble score as mean of geometric and crisis-adjusted Markov.

    Args:
        geo           : geometric signal {0.0, 0.5, 1.0}
        markov_mom    : Markov P(momentum), 0-1
        markov_crisis : Markov P(crisis),   0-1
        vix           : optional VIX level series; if provided, dampens the
                        momentum signal in high-fear environments

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

    # Markov loses one observation (AR lag) -- dropna aligns on common index
    aligned = pd.concat([geo, markov_adj], axis=1).dropna()
    return aligned.mean(axis=1).rename("ensemble_score")


def regime_labels(score: pd.Series) -> pd.Series:
    """Map ensemble scores to regime labels."""
    labels = pd.Series("mixed", index=score.index, name="regime")
    labels[score >= MOMENTUM_THRESHOLD]  = "momentum"
    labels[score <= REVERSION_THRESHOLD] = "reversion"
    return labels
