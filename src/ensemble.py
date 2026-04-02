"""
Two-model ensemble: Geometric + Markov k=3.

Ensemble score = mean( geo_signal, markov_signal_crisis_adjusted )

    geo_signal      : {0.0, 0.5, 1.0}  hard label → float
    markov_signal   : P(momentum)       0–1 continuous (filtered)
    crisis override : if P(crisis) > 0.50, set markov_signal = 0
                      regardless of P(momentum) — capital preservation first

Score → regime label:
    >= 0.65  →  momentum   (both models leaning momentum)
    <= 0.35  →  reversion  (both models leaning reversion)
    0.35–0.65 → mixed       (models disagree or both uncertain)

Why equal weighting?
    Fitting weights to historical returns would constitute in-sample
    optimisation. Equal weights are more conservative and defensible.
    The ensemble value comes from combining orthogonal signals
    (short-term path shape vs long-term statistical state), not from
    fitting weights.
"""

import pandas as pd

MOMENTUM_THRESHOLD  = 0.65
REVERSION_THRESHOLD = 0.35
CRISIS_THRESHOLD    = 0.50


def ensemble_score(
    geo: pd.Series,
    markov_mom: pd.Series,
    markov_crisis: pd.Series,
) -> pd.Series:
    """
    Compute ensemble score as mean of geometric and crisis-adjusted Markov.

    Args:
        geo           : geometric signal {0.0, 0.5, 1.0}
        markov_mom    : Markov P(momentum), 0–1
        markov_crisis : Markov P(crisis),   0–1

    Returns:
        Series of scores in [0, 1].
    """
    markov_adj = markov_mom.copy()
    markov_adj[markov_crisis > CRISIS_THRESHOLD] = 0.0

    # Markov loses one observation (AR lag) — dropna aligns on common index
    aligned = pd.concat([geo, markov_adj], axis=1).dropna()
    return aligned.mean(axis=1).rename("ensemble_score")


def regime_labels(score: pd.Series) -> pd.Series:
    """Map ensemble scores to regime labels."""
    labels = pd.Series("mixed", index=score.index, name="regime")
    labels[score >= MOMENTUM_THRESHOLD]  = "momentum"
    labels[score <= REVERSION_THRESHOLD] = "reversion"
    return labels
