"""
Markov Switching Autoregression regime detection.

Model: MarkovAutoregression(k=3, AR order=1)

BIC model selection result (SPY 2000-2025, ~5364 observations):
    k=2: BIC = baseline
    k=3: BIC lower (ΔBIC > 10 = "very strong evidence")

The third regime captures a distinct crisis state (high vol, negative
mean) that k=2 blends with the reversion state, understating downside risk.

Convergence: em_iter=200 with search_reps=5 random starts. The best
fit (lowest BIC across starts) is selected. This eliminates the
ConvergenceWarning that appeared with the default em_iter=5 on 25-year data.

IMPORTANT -- filtered vs smoothed probabilities:
    Smoothed probabilities use data from both past AND future to estimate
    the regime at time T. They look clean on charts but constitute
    look-ahead bias. This implementation uses FILTERED probabilities only:
    the estimate at time T uses only data up to T.

    Rule: smoothed probabilities are never used in statistical tests
    or equity curves. They are labelled explicitly if plotted.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import EstimationWarning

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


def fit_markov3(returns: pd.Series, verbose: bool = True):
    """
    Fit Markov AR(1) with k=3 regimes and return FILTERED probabilities
    plus transition analysis.

    Regime labelling by mean daily return:
        momentum → regime with highest mean
        crisis   → regime with lowest mean  (most negative)
        choppy   → the remainder

    Returns:
        mom_prob    : P(momentum), 0–1 continuous, DatetimeIndex
        crisis_prob : P(crisis),   0–1 continuous, DatetimeIndex
        choppy_prob : P(choppy),   0–1 continuous, DatetimeIndex
        trans_info  : dict with transition matrix and expected durations
    """
    ret = returns.dropna()
    model = sm.tsa.MarkovAutoregression(
        ret,
        k_regimes=3,
        order=AR_ORDER,
        switching_ar=True,
        switching_variance=True,
    )
    # em_iter=200 + search_reps=5 random starts: take best (lowest BIC).
    # Eliminates ConvergenceWarning on long datasets (25 years needs more
    # EM iterations). EstimationWarning (probability re-scaling in some
    # random starts) is suppressed -- it is numerical noise, not a failure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EstimationWarning)
        results = model.fit(disp=False, em_iter=200, search_reps=5)

    # ── Label regimes by mean return ──────────────────────────────────
    p      = results.params
    consts = [p.get(f"const[{i}]", 0) for i in range(3)]

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

    # ── Regime characteristics (empirical weighted stats) ─────────────
    ret_arr = ret.values[-len(probs):]
    regime_stats = {}
    for name, ri in [("MOMENTUM", mom_idx), ("CHOPPY", choppy_idx), ("CRISIS", crisis_idx)]:
        w      = probs[:, ri]
        w_norm = w / w.sum() if w.sum() > 0 else w
        mean_r = float(np.dot(w_norm, ret_arr))
        vol    = float(np.sqrt(np.dot(w_norm, (ret_arr - mean_r) ** 2)) * np.sqrt(252))
        regime_stats[name] = {"mean": mean_r, "vol": vol}

    if verbose:
        print("\n  Markov k=3 regime characteristics:")
        for name, s in regime_stats.items():
            print(f"    {name:10s}  mean = {s['mean']*100:+.3f}%/day   vol = {s['vol']*100:.0f}% ann.")

    # ── Transition matrix ─────────────────────────────────────────────
    # regime_transition shape: (k, k) where [i, j] = P(next=i | current=j)
    # (statsmodels convention: column = current regime, row = next regime)
    trans_info = {}
    try:
        trans_raw = np.array(results.regime_transition)

        # Handle 3D shape (AR models can return (k, k, 1) or (k, k))
        if trans_raw.ndim == 3:
            trans_raw = trans_raw[:, :, 0]

        # Reorder so row=current, column=next (intuitive reading direction)
        # statsmodels: trans[j, i] = P(next=j | current=i) → transpose to [i, j]
        trans_matrix = trans_raw.T  # shape (k, k), [current, next]

        # Map internal regime indices to names
        idx_to_name = {mom_idx: "MOMENTUM", choppy_idx: "CHOPPY", crisis_idx: "CRISIS"}
        ordered = [mom_idx, choppy_idx, crisis_idx]
        names   = [idx_to_name[i] for i in ordered]

        # Extract stay-in-regime probabilities (diagonal in ordered matrix)
        ordered_trans = trans_matrix[np.ix_(ordered, ordered)]
        expected_dur  = {names[i]: 1.0 / (1.0 - ordered_trans[i, i])
                         for i in range(3) if ordered_trans[i, i] < 1.0}

        # Count observed regime switches per year (from hard labels)
        hard_labels  = pd.Series(
            [idx_to_name[np.argmax(probs[t])] for t in range(len(probs))],
            index=idx,
        )
        n_switches    = (hard_labels != hard_labels.shift(1)).sum()
        years         = len(probs) / 252
        switches_pa   = n_switches / years if years > 0 else 0

        trans_info = {
            "matrix":       ordered_trans,
            "names":        names,
            "expected_dur": expected_dur,
            "switches_pa":  switches_pa,
            "regime_stats": regime_stats,
        }

        if verbose:
            print("\n  Transition matrix  (row = current regime, col = next regime):")
            header = f"    {'':12s}" + "".join(f"{n:>12s}" for n in names)
            print(header)
            for i, row_name in enumerate(names):
                row = "".join(f"{ordered_trans[i, j]:>12.1%}" for j in range(3))
                print(f"    {row_name:12s}{row}")
            print("\n  Expected regime duration:")
            for name, dur in expected_dur.items():
                print(f"    {name:12s}  {dur:.1f} trading days  (~{dur/21:.1f} months)")
            print(f"\n  Regime switches per year: {switches_pa:.0f}")

    except Exception as e:
        if verbose:
            print(f"\n  [transition matrix unavailable: {e}]")

    return mom_prob, crisis_prob, choppy_prob, trans_info
