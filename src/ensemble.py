"""
Ensemble: Geometric + Markov k=3 + optional Kalman, with optional dampening.

Base ensemble (2-signal):
    score = mean( geo_signal, markov_signal_crisis_adjusted )

With Kalman (3-signal, --kalman flag):
    score = mean( geo_signal, markov_signal_crisis_adjusted, kalman_signal_crisis_adjusted )

    geo_signal      : {0.0, 0.5, 1.0}  hard label -> float
    markov_signal   : P(momentum)       0-1 continuous (filtered)
    kalman_signal   : norm.cdf(μ_t / sqrt(P_t + R))  0-1 continuous drift probability
    crisis override : if P(crisis) > 0.50, set markov AND kalman signal = 0
                      regardless of other indicators -- capital preservation first
    VIX dampening   : if vix is provided, scale markov_adj by a fear factor
                      vix_factor = 1 - clip((vix - 20) / 20, 0, 1)
                      VIX <= 20 -> no change; VIX 30 -> halved; VIX >= 40 -> zero
    VVIX dampening  : if vvix is provided, scale markov_adj by a vol-of-vol
                      factor:
                      vvix_factor = 1 - clip((vvix - 90) / 50, 0, 1)
                      VVIX <= 90 -> no change; VVIX 115 -> halved; VVIX >= 140 -> zero
    Vol ratio       : if vol_ratio is provided, scale markov_adj by a turbulence
                      factor derived from realised vol:
                      vol_factor = 1 - clip((ratio - 1.0) / 1.0, 0, 1)
                      ratio <= 1.0 -> no dampening; ratio = 2.0 -> fully suppressed
                      (ratio = short_vol / long_vol, e.g. 5-day ann. / 63-day ann.)
    Gamma proxy     : if gamma_proxy is provided, scale markov_adj by
                      (1 - gamma_proxy), where gamma_proxy in [0,1] captures
                      spot/vol stress that approximates negative-gamma risk.

Score -> regime label:
    >= 0.65  ->  momentum   (models leaning momentum)
    <= 0.35  ->  reversion  (models leaning reversion)
    0.35-0.65 -> mixed       (models disagree or both uncertain)

Why equal weighting?
    Fitting weights to historical returns would constitute in-sample
    optimisation. Equal weights are more conservative and defensible.
    The ensemble value comes from combining orthogonal signals
    (short-term path shape vs long-term statistical state vs continuous drift),
    not from fitting weights.

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
VVIX_NEUTRAL        = 90.0   # VVIX at or below this: no dampening
VVIX_FULL_SUPPRESS  = 140.0  # VVIX at or above this: momentum fully suppressed
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


def gamma_stress_proxy(
    returns: pd.Series,
    vix: pd.Series,
    vvix: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Build a daily gamma-stress proxy in [0,1] from spot/vol shock behavior.

    This is not true dealer GEX from options OI. It is a causal proxy:
      - downside spot shock (large negative return vs rolling vol)
      - VIX jump shock (daily pct change)
      - optional VVIX jump shock (daily pct change)

    Higher values imply higher stress where trend-following exposure should be
    dampened.
    """
    returns = pd.to_numeric(returns, errors="coerce")
    spot_sigma = returns.rolling(20, min_periods=10).std().replace(0, np.nan)
    spot_down = (-returns / spot_sigma).clip(lower=0.0)
    spot_norm = (spot_down / 3.0).clip(0.0, 1.0)

    vix_aligned = pd.to_numeric(vix.reindex(returns.index, method="ffill"), errors="coerce")
    vix_jump = vix_aligned.pct_change(fill_method=None).clip(lower=0.0)
    vix_norm = (vix_jump / 0.08).clip(0.0, 1.0)

    if vvix is not None:
        vvix_aligned = pd.to_numeric(vvix.reindex(returns.index, method="ffill"), errors="coerce")
        vvix_jump = vvix_aligned.pct_change(fill_method=None).clip(lower=0.0)
        vvix_norm = (vvix_jump / 0.10).clip(0.0, 1.0)
        raw = 0.50 * spot_norm + 0.30 * vix_norm + 0.20 * vvix_norm
    else:
        raw = 0.60 * spot_norm + 0.40 * vix_norm

    out = raw.rolling(3, min_periods=1).mean().clip(0.0, 1.0)
    return pd.Series(out, index=returns.index, name="gamma_proxy")


def _build_markov_component(
    markov_mom: pd.Series,
    markov_crisis: pd.Series,
    mode: str,
) -> Optional[pd.Series]:
    """
    Build the Markov component of the ensemble score.

    mode='full'          → P(momentum) zeroed where P(crisis) > 0.50 (default).
                           Treats Markov as both a directional and risk-state signal.

    mode='crisis_filter' → 1 - P(crisis). Pure risk-state filter: when crisis
                           probability is high the component approaches 0 (cash),
                           when low it approaches 1 (full long). Ignores P(momentum)
                           entirely. Use if the ablation shows mom_prob adds nothing
                           beyond crisis_prob.

    mode='geo_only'      → None. Excludes the Markov component; score = geo only.
                           The lowest baseline for ablation comparison.

    Returns pd.Series or None. None signals ensemble_score to skip Markov entirely.
    """
    if mode == "geo_only":
        return None
    if mode == "crisis_filter":
        return (1.0 - markov_crisis).rename("markov_adj")
    # mode == 'full' (default)
    adj = markov_mom.copy()
    adj[markov_crisis > CRISIS_THRESHOLD] = 0.0
    return adj.rename("markov_adj")


def ensemble_score(
    geo: pd.Series,
    markov_mom: pd.Series,
    markov_crisis: pd.Series,
    vix: Optional[pd.Series] = None,
    vvix: Optional[pd.Series] = None,
    vol_ratio_series: Optional[pd.Series] = None,
    gamma_proxy: Optional[pd.Series] = None,
    kalman: Optional[pd.Series] = None,
    mode: str = "full",
) -> pd.Series:
    """
    Compute ensemble score as equal-weighted mean of component signals.

    2-signal (base):  mean(geo, markov_adj)
    3-signal (Kalman): mean(geo, markov_adj, kalman_adj)

    Dampening is applied to the Markov component before averaging.
    The crisis override (P(crisis) > 0.50) zeroes both Markov and Kalman
    when mode='full'. Multiple dampening signals are applied multiplicatively.

    Args:
        geo              : geometric signal {0.0, 0.5, 1.0}
        markov_mom       : Markov P(momentum), 0-1
        markov_crisis    : Markov P(crisis),   0-1
        vix              : optional VIX level series (requires Polygon paid plan)
        vvix             : optional VVIX level series (vol-of-vol index)
        vol_ratio_series : optional realised vol ratio (short_vol / long_vol);
                           use vol_ratio() to compute from returns
        gamma_proxy      : optional gamma-stress proxy in [0,1] from
                           gamma_stress_proxy(); higher => stronger dampening
        kalman           : optional Kalman drift signal in [0, 1];
                           from kalman_signal() in src.kalman
        mode             : how the Markov component is built — see _build_markov_component.
                           'full' (default) preserves existing behaviour exactly.

    Returns:
        Series of scores in [0, 1].
    """
    markov_adj = _build_markov_component(markov_mom, markov_crisis, mode)

    if markov_adj is not None:
        if vix is not None:
            vix_aligned = vix.reindex(markov_adj.index, method="ffill")
            vix_factor = 1.0 - np.clip(
                (vix_aligned - VIX_NEUTRAL) / (VIX_FULL_SUPPRESS - VIX_NEUTRAL),
                0.0, 1.0,
            )
            markov_adj = markov_adj * vix_factor

        if vvix is not None:
            vvix_aligned = vvix.reindex(markov_adj.index, method="ffill")
            vvix_factor = 1.0 - np.clip(
                (vvix_aligned - VVIX_NEUTRAL) / (VVIX_FULL_SUPPRESS - VVIX_NEUTRAL),
                0.0, 1.0,
            )
            markov_adj = markov_adj * vvix_factor

        if vol_ratio_series is not None:
            vr_aligned = vol_ratio_series.reindex(markov_adj.index, method="ffill")
            vr_factor = 1.0 - np.clip(
                (vr_aligned - VOL_RATIO_NEUTRAL) / (VOL_RATIO_SUPPRESS - VOL_RATIO_NEUTRAL),
                0.0, 1.0,
            )
            markov_adj = markov_adj * vr_factor

        if gamma_proxy is not None:
            gp_aligned = gamma_proxy.reindex(markov_adj.index, method="ffill").clip(0.0, 1.0)
            markov_adj = markov_adj * (1.0 - gp_aligned)

    components = [geo] if markov_adj is None else [geo, markov_adj]

    if kalman is not None:
        ref_idx    = markov_adj.index if markov_adj is not None else geo.index
        kalman_adj = kalman.reindex(ref_idx, fill_value=np.nan).copy()
        kalman_adj[markov_crisis.reindex(kalman_adj.index, fill_value=0.0) > CRISIS_THRESHOLD] = 0.0
        components.append(kalman_adj)

    aligned = pd.concat(components, axis=1).dropna()
    return aligned.mean(axis=1).rename("ensemble_score")


def regime_labels(score: pd.Series) -> pd.Series:
    """Map ensemble scores to regime labels."""
    labels = pd.Series("mixed", index=score.index, name="regime")
    labels[score >= MOMENTUM_THRESHOLD]  = "momentum"
    labels[score <= REVERSION_THRESHOLD] = "reversion"
    return labels
