"""
Markov Switching Autoregression regime detection.

Model: MarkovAutoregression(k=3, AR order=1)

BIC model selection result (SPY 2022-2025, ~750 observations):
    k=2: BIC = -7,761   (baseline)
    k=3: BIC = -7,843   (ΔBIC = 82 over k=2)

ΔBIC > 10 is "very strong evidence" by standard criteria. The third
regime captures a distinct crisis state (mean ≈ -0.62%/day, vol ≈ 69%
annualised) that k=2 blends with the reversion state, understating
downside risk.

IMPORTANT — filtered vs smoothed probabilities:
    Smoothed probabilities use data from both past AND future to estimate
    the regime at time T. They look clean on charts but constitute
    look-ahead bias. This implementation uses FILTERED probabilities only:
    the estimate at time T uses only data up to T.

    Rule: smoothed probabilities are never used in statistical tests
    or equity curves. They are labelled explicitly if plotted.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

AR_ORDER = 1


def select_k(returns: pd.Series, k_range: range = range(2, 4)) -> pd.DataFrame:
    """
    Fit Markov AR(1) for each k and return AIC/BIC comparison.

    Used once for model selection. Prints progress — each fit takes ~30s.

    Returns:
        DataFrame indexed by k with columns: log_likelihood, aic, bic
    """
    rows = []
    for k in k_range:
        print(f"    Fitting k={k}...")
        model = sm.tsa.MarkovAutoregression(
            returns.dropna(),
            k_regimes=k,
            order=AR_ORDER,
            switching_ar=True,
            switching_variance=True,
        )
        res = model.fit(disp=False)
        rows.append({"k": k, "log_likelihood": round(res.llf, 1), "aic": round(res.aic, 1), "bic": round(res.bic, 1)})
    return pd.DataFrame(rows).set_index("k")


def fit_markov3(returns: pd.Series):
    """
    Fit Markov AR(1) with k=3 regimes and return FILTERED probabilities.

    Regime labelling by mean daily return:
        momentum → regime with highest mean
        crisis   → regime with lowest mean  (most negative)
        choppy   → the remainder

    Returns:
        mom_prob    : P(momentum), 0–1 continuous, DatetimeIndex
        crisis_prob : P(crisis),   0–1 continuous, DatetimeIndex
        choppy_prob : P(choppy),   0–1 continuous, DatetimeIndex
    """
    ret = returns.dropna()
    model = sm.tsa.MarkovAutoregression(
        ret,
        k_regimes=3,
        order=AR_ORDER,
        switching_ar=True,
        switching_variance=True,
    )
    results = model.fit(disp=False)

    # ── Label regimes by mean return ──────────────────────────────────
    p      = results.params
    consts = [p.get(f"const[{i}]", 0) for i in range(3)]
    sig2   = [p.get(f"sigma2[{i}]", np.nan) for i in range(3)]

    mom_idx    = int(np.argmax(consts))
    crisis_idx = int(np.argmin(consts))
    choppy_idx = [i for i in range(3) if i not in (mom_idx, crisis_idx)][0]

    # ── Extract FILTERED probabilities ────────────────────────────────
    # filtered_marginal_probabilities is either a DataFrame or ndarray
    # depending on statsmodels version.
    filt = results.filtered_marginal_probabilities
    if hasattr(filt, "iloc"):
        idx   = filt.index
        probs = filt.values
    else:
        probs = np.array(filt)
        # AR(1) drops the first observation — align index from the tail
        idx = ret.index[len(ret) - len(probs):]

    mom_prob    = pd.Series(probs[:, mom_idx],    index=idx, name="markov_mom")
    crisis_prob = pd.Series(probs[:, crisis_idx], index=idx, name="markov_crisis")
    choppy_prob = pd.Series(probs[:, choppy_idx], index=idx, name="markov_choppy")

    # ── Print regime characteristics (empirical weighted stats) ───────
    ret_arr = ret.values[-len(probs):]
    print("\n  Markov k=3 regime characteristics:")
    for name, ri in [("MOMENTUM", mom_idx), ("CHOPPY", choppy_idx), ("CRISIS", crisis_idx)]:
        w      = probs[:, ri]
        w_norm = w / w.sum() if w.sum() > 0 else w
        mean_r = float(np.dot(w_norm, ret_arr))
        vol    = float(np.sqrt(np.dot(w_norm, (ret_arr - mean_r) ** 2)) * np.sqrt(252))
        print(f"    {name:10s}  mean = {mean_r*100:+.3f}%/day   vol = {vol*100:.0f}% ann.")

    return mom_prob, crisis_prob, choppy_prob
