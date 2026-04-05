"""
Local-level Kalman filter for market drift estimation.

Model:
    State:       μ_t = μ_{t-1} + η_t,  η_t ~ N(0, Q)  (random walk drift)
    Observation: r_t = μ_t  + ε_t,     ε_t ~ N(0, R)  (noisy observation)

Q controls how fast the drift estimate can change (process noise).
R controls how noisy individual daily returns are (observation noise).

Both Q and R are estimated by maximum likelihood on the training data —
only 2 parameters, compared to 15+ for Markov AR(1) k=3.
Very low overfitting risk: 2 parameters on 6,000+ observations gives
vastly more degrees of freedom than the Markov model.

Signal construction:
    kalman_signal_t = norm.cdf(μ_t / sqrt(P_t + R))

This is the probability that the next return is positive given the
current drift estimate μ_t and combined uncertainty sqrt(P_t + R).

    μ_t > 0, low uncertainty → signal near 1.0  (confident momentum)
    μ_t < 0, low uncertainty → signal near 0.0  (confident reversion)
    μ_t ≈ 0  or  high uncertainty → signal ≈ 0.5 (uncertain / mixed)

Why Kalman addresses the key weakness (lagging crisis detection):
    The Markov EM algorithm is a batch procedure: it fits regime
    parameters on the full training set. Crisis detection is inherently
    backward-looking — it requires enough crisis-like observations to
    accumulate before the filtered probability P(crisis) crosses 0.5.
    In practice this takes roughly 10-15 trading days.

    The Kalman filter updates at every observation with
    uncertainty-weighted evidence. A step-change in drift (sudden drop
    at the onset of a crash) is detectable within 1-3 observations.

    The two approaches are complementary:
        Kalman — fast, continuous drift tracking; no regime structure
        Markov  — slow, regime-structured; catches sustained states
        Geometric — path shape; orthogonal to both

    Equal-weighted 3-way ensemble: mean(geo, Markov_adj, Kalman_adj)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Optional, Tuple


def _run_filter(
    returns_arr: np.ndarray,
    Q: float,
    R: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core Kalman filter recursion for the local level model.

    Predict step:  μ_{t|t-1} = μ_{t-1|t-1}
                   P_{t|t-1} = P_{t-1|t-1} + Q

    Update step:   innovation  = r_t - μ_{t|t-1}
                   S_t         = P_{t|t-1} + R           (innovation variance)
                   K_t         = P_{t|t-1} / S_t         (Kalman gain)
                   μ_{t|t}     = μ_{t|t-1} + K_t * innovation
                   P_{t|t}     = (1 - K_t) * P_{t|t-1}

    Args:
        returns_arr : 1-D array of daily log returns
        Q           : process noise variance (drift change variance)
        R           : observation noise variance (return noise)

    Returns:
        mu_filtered  : filtered state means  E[μ_t | r_1, …, r_t]
        var_filtered : filtered state variances Var[μ_t | r_1, …, r_t]
        innovations  : prediction errors r_t - E[r_t | r_1, …, r_{t-1}]
        innov_var    : innovation variances P_{t|t-1} + R
    """
    n   = len(returns_arr)
    mu  = np.zeros(n)
    P   = np.zeros(n)
    inn = np.zeros(n)
    S   = np.zeros(n)

    # Diffuse prior: zero expected drift, high variance
    mu_t = 0.0
    P_t  = R * 10.0

    for t in range(n):
        # Predict
        mu_pred = mu_t
        P_pred  = P_t + Q

        # Innovation
        innov_t = returns_arr[t] - mu_pred
        S_t     = P_pred + R

        # Update
        K    = P_pred / S_t
        mu_t = mu_pred + K * innov_t
        P_t  = (1.0 - K) * P_pred

        mu[t]  = mu_t
        P[t]   = P_t
        inn[t] = innov_t
        S[t]   = S_t

    return mu, P, inn, S


def _neg_log_likelihood(log_params: np.ndarray, returns_arr: np.ndarray) -> float:
    """Negative log-likelihood for Nelder-Mead optimisation (log-space parameterisation)."""
    Q = np.exp(log_params[0])
    R = np.exp(log_params[1])

    _, _, inn, S = _run_filter(returns_arr, Q, R)

    # Skip first observation (dominated by diffuse prior)
    inn = inn[1:]
    S   = S[1:]

    if np.any(S <= 0):
        return 1e10

    ll = -0.5 * np.sum(np.log(2.0 * np.pi * S) + inn ** 2 / S)
    return -ll


def fit_kalman(returns: pd.Series) -> Tuple[float, float]:
    """
    Estimate Q and R by maximum likelihood on the provided return series.

    Starting point: R ≈ var(returns), Q ≈ 0.01 × R (drift changes slowly).
    Optimisation is in log-space to enforce positivity.

    Args:
        returns : daily log return series

    Returns:
        (Q, R) : MLE estimates of process noise and observation noise variances
    """
    arr = returns.dropna().values
    R0  = float(np.var(arr))
    Q0  = 0.01 * R0
    x0  = np.array([np.log(Q0), np.log(R0)])

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(arr,),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-9},
    )

    Q_hat = float(np.exp(result.x[0]))
    R_hat = float(np.exp(result.x[1]))
    return Q_hat, R_hat


def kalman_signal(
    returns: pd.Series,
    Q: Optional[float] = None,
    R: Optional[float] = None,
) -> pd.Series:
    """
    Compute Kalman drift signal: probability that next return is positive.

        signal_t = norm.cdf(μ_t / sqrt(P_t + R))

    This maps the filtered drift estimate onto [0, 1]:
        positive drift + low uncertainty → near 1.0
        negative drift + low uncertainty → near 0.0
        zero drift or high uncertainty   → near 0.5

    If Q and R are not provided they are estimated via MLE (in-sample use).
    For out-of-sample use, pass Q and R estimated on the training slice only.

    Args:
        returns : daily log return series
        Q       : process noise variance (None → estimated by MLE)
        R       : observation noise variance (None → estimated by MLE)

    Returns:
        pd.Series in [0, 1], same index as returns (after dropna).
    """
    clean = returns.dropna()
    arr   = clean.values

    if Q is None or R is None:
        Q, R = fit_kalman(clean)

    mu, P, _, _ = _run_filter(arr, Q, R)
    signal = norm.cdf(mu / np.sqrt(P + R))

    return pd.Series(signal, index=clean.index, name="kalman_signal")
