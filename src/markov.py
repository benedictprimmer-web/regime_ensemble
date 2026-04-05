"""
Multivariate Hidden Markov Model regime detection.

Model: GaussianHMM(n_components=3, covariance_type='diag')

Observation vector (5 features, z-scored on train data):
    ret_20d   -- 20-day cumulative log return       (medium-term direction)
    ret_5d    -- 5-day cumulative log return         (short-term direction)
    rvol_20d  -- 20-day realised vol (annualised)    (stress / volatility level)
    drawdown  -- distance from 252-day high          (loss from recent peak)
    dist_200d -- price / 200-day MA - 1             (structural trend position)

Why this beats AR(1) on daily returns:
    AR(1) asks "are daily returns autocorrelated one step ahead?" — for SPY,
    that autocorrelation is typically 0.01–0.03 (near noise). The model ends
    up inferring regime almost entirely from variance, not from any directional
    structure. By contrast, this feature vector directly measures the quantities
    that define market state: trend persistence (ret_20d), short-term momentum
    (ret_5d), volatility regime (rvol_20d), stress depth (drawdown), and
    structural trend position (dist_200d). The hidden states become economically
    interpretable rather than statistical artefacts.

State labelling rule (mechanical, enforced at every refit):
    MOMENTUM → state with highest mean ret_20d
    CRISIS   → state with lowest  mean ret_20d
    CHOPPY   → the remaining state

    Sorting by ret_20d (not eyeballing charts) ensures that regime labels are
    consistent across walk-forward refits and cannot swap meaning between folds.

Filtered probabilities:
    The forward algorithm is implemented manually so that P(state_t | data_1..t)
    uses only past data at each time step — no look-ahead bias. hmmlearn's
    predict_proba / score_samples use the forward-backward algorithm (smoothed),
    which is look-ahead and must not be used in equity curves or t-tests.

Walk-forward use:
    fit_and_filter_markov(train_ret, test_ret) fits the HMM + computes the
    z-score scaler on train data, then applies both frozen to the test slice.
    The test features are constructed using the last 250 days of train as
    rolling-window warm-up — no EM or scaler fitting on test data.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM


# ── Constants ─────────────────────────────────────────────────────────────────

N_STATES      = 3
N_ITER        = 200   # EM iterations per random start
N_RESTARTS    = 5     # independent random starts; best log-likelihood is kept
FEATURE_NAMES = ["ret_20d", "ret_5d", "rvol_20d", "drawdown", "dist_200d"]


# ── Feature construction ──────────────────────────────────────────────────────

def _build_features(returns: pd.Series, warmup: pd.Series = None) -> pd.DataFrame:
    """
    Build the 5-feature observation matrix from a daily log-return series.

    Args:
        returns : daily log returns (the period you want features for)
        warmup  : optional history to prepend for rolling-window initialisation.
                  Typically the last 250 days of the training slice. The output
                  is sliced back to `returns.index` before returning.

    Returns:
        DataFrame with columns matching FEATURE_NAMES, indexed by returns.index.
        Rows where any feature is NaN are kept — callers must dropna() before
        passing to the HMM (and reindex back afterwards).
    """
    if warmup is not None:
        combined = pd.concat([warmup, returns])
    else:
        combined = returns

    # Reconstruct a price series from cumulative log returns (base = 1.0).
    # Relative measures (drawdown, dist_200d) are unaffected by the base level.
    price = np.exp(combined.cumsum())

    feats = pd.DataFrame({
        "ret_20d":  combined.rolling(20,  min_periods=10).sum(),
        "ret_5d":   combined.rolling(5,   min_periods=3).sum(),
        "rvol_20d": combined.rolling(20,  min_periods=10).std() * np.sqrt(252),
        "drawdown": price / price.rolling(252, min_periods=63).max() - 1,
        "dist_200d": price / price.rolling(200, min_periods=50).mean() - 1,
    }, index=combined.index)

    if warmup is not None:
        return feats.loc[returns.index]
    return feats


def _scale(feats: pd.DataFrame, mu: pd.Series, sigma: pd.Series) -> np.ndarray:
    """Z-score feats using pre-computed train statistics."""
    return ((feats - mu) / sigma).values


# ── State labelling ───────────────────────────────────────────────────────────

def _label_states(model: GaussianHMM) -> tuple:
    """
    Mechanically assign MOMENTUM / CRISIS indices from model means.

    Sorts hidden states by their mean of ret_20d (feature index 0, which has
    the clearest economic interpretation). The state with the highest mean
    ret_20d is MOMENTUM; the lowest is CRISIS; the middle is CHOPPY.

    Returns:
        (mom_idx, crisis_idx, choppy_idx)  — internal HMM state indices
    """
    means_20d = model.means_[:, 0]          # ret_20d mean per state
    order      = np.argsort(means_20d)      # ascending: crisis → choppy → momentum
    crisis_idx = int(order[0])
    choppy_idx = int(order[1])
    mom_idx    = int(order[2])
    return mom_idx, crisis_idx, choppy_idx


# ── Forward filter ────────────────────────────────────────────────────────────

def _forward_filter(model: GaussianHMM, obs: np.ndarray) -> np.ndarray:
    """
    Causal forward pass: P(state_t | obs_1 ... obs_t) for each t.

    Uses only the forward algorithm (no backward smoothing), so the probability
    at t depends only on past and present data — no look-ahead bias.

    Args:
        model : fitted GaussianHMM with covariance_type='diag'
        obs   : scaled observation array, shape (n_obs, n_features)

    Returns:
        alpha : filtered state probabilities, shape (n_obs, n_states)
    """
    n_obs, _ = obs.shape
    k = model.n_components

    # Emission log-probabilities under each state's Gaussian
    log_emit = np.zeros((n_obs, k))
    for j in range(k):
        cov = np.diag(model.covars_[j])      # diag covariance → full matrix
        log_emit[:, j] = multivariate_normal.logpdf(obs, mean=model.means_[j], cov=cov)

    alpha = np.zeros((n_obs, k))

    # t = 0: initialise from start probabilities
    e0 = np.exp(log_emit[0] - log_emit[0].max())  # subtract max for numerical stability
    alpha[0] = model.startprob_ * e0
    s = alpha[0].sum()
    alpha[0] = alpha[0] / s if s > 0 else np.ones(k) / k

    # t > 0: predict then update
    for t in range(1, n_obs):
        predict = alpha[t - 1] @ model.transmat_
        e = np.exp(log_emit[t] - log_emit[t].max())
        alpha[t] = predict * e
        s = alpha[t].sum()
        alpha[t] = alpha[t] / s if s > 0 else np.ones(k) / k

    return alpha


# ── HMM fitting with restarts ─────────────────────────────────────────────────

def _fit_hmm(X: np.ndarray, n_restarts: int = N_RESTARTS) -> GaussianHMM:
    """
    Fit GaussianHMM(k=3, diag) with multiple random starts; return best fit
    by training log-likelihood.

    Args:
        X          : scaled observation array, shape (n_obs, n_features), no NaNs
        n_restarts : number of independent EM runs

    Returns:
        best GaussianHMM instance
    """
    best_model  = None
    best_score  = -np.inf

    for seed in range(n_restarts):
        model = GaussianHMM(
            n_components    = N_STATES,
            covariance_type = "diag",
            n_iter          = N_ITER,
            random_state    = seed,
            tol             = 1e-4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)

        try:
            ll = model.score(X)          # mean log-likelihood per observation
        except Exception:
            continue

        if ll > best_score:
            best_score = ll
            best_model = model

    if best_model is None:
        raise RuntimeError("All HMM restarts failed to converge.")

    return best_model


# ── BIC calculation ───────────────────────────────────────────────────────────

def _bic(model: GaussianHMM, X: np.ndarray) -> float:
    """
    BIC = -2 * total_log_likelihood + n_params * log(n_samples)

    For GaussianHMM with diag covariance, n_features = d, n_components = k:
        n_params = (k-1)           # startprob
                 + k*(k-1)        # transmat (rows sum to 1)
                 + k*d            # means
                 + k*d            # diag variances
                 = k^2 - 1 + 2*k*d
    """
    n, d = X.shape
    k    = model.n_components
    n_params = k**2 - 1 + 2 * k * d
    total_ll = model.score(X) * n
    return -2 * total_ll + n_params * np.log(n)


# ── Public API ────────────────────────────────────────────────────────────────

def select_k(returns: pd.Series, k_range: range = range(2, 5)) -> pd.DataFrame:
    """
    Fit multivariate GaussianHMM for each k and compare by AIC / BIC.

    Used once for model selection. Each fit is fast (seconds, not minutes).

    Returns:
        DataFrame indexed by k with columns: log_likelihood, aic, bic
    """
    feats = _build_features(returns).dropna()
    mu    = feats.mean()
    sigma = feats.std().replace(0, 1)
    X     = _scale(feats, mu, sigma)

    rows = []
    for k in k_range:
        print(f"    Fitting k={k}...")
        best_ll = -np.inf
        best_m  = None
        for seed in range(N_RESTARTS):
            m = GaussianHMM(n_components=k, covariance_type="diag",
                            n_iter=N_ITER, random_state=seed, tol=1e-4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X)
            try:
                ll = m.score(X)
            except Exception:
                continue
            if ll > best_ll:
                best_ll = ll
                best_m  = m

        n     = len(X)
        d     = X.shape[1]
        n_params = k**2 - 1 + 2 * k * d
        total_ll = best_ll * n
        aic   = -2 * total_ll + 2 * n_params
        bic   = -2 * total_ll + n_params * np.log(n)
        rows.append({"k": k, "log_likelihood": round(total_ll, 1),
                     "aic": round(aic, 1), "bic": round(bic, 1)})

    return pd.DataFrame(rows).set_index("k")


def fit_markov3(returns: pd.Series, verbose: bool = True):
    """
    Fit multivariate HMM (k=3) on returns and return FILTERED probabilities.

    State labelling is mechanical at each call:
        MOMENTUM → state with highest mean ret_20d
        CRISIS   → state with lowest  mean ret_20d
        CHOPPY   → remainder

    Returns:
        mom_prob    : P(momentum), pd.Series on returns.index
        crisis_prob : P(crisis),   pd.Series on returns.index
        choppy_prob : P(choppy),   pd.Series on returns.index
        trans_info  : dict with transition matrix, expected durations, regime stats
    """
    # Build and clean features
    feats      = _build_features(returns)
    feats_clean = feats.dropna()
    idx_clean  = feats_clean.index

    if len(feats_clean) < 50:
        raise ValueError(f"Too few valid observations ({len(feats_clean)}) to fit HMM.")

    # Z-score on the full sample (in-sample use — appropriate here)
    mu    = feats_clean.mean()
    sigma = feats_clean.std().replace(0, 1)
    X     = _scale(feats_clean, mu, sigma)

    # Fit
    model = _fit_hmm(X)

    # Mechanical state labelling
    mom_idx, crisis_idx, choppy_idx = _label_states(model)

    # Causal forward filter (no look-ahead)
    filt_probs = _forward_filter(model, X)    # shape (n_clean, 3)

    mom_prob    = pd.Series(filt_probs[:, mom_idx],    index=idx_clean, name="markov_mom")
    crisis_prob = pd.Series(filt_probs[:, crisis_idx], index=idx_clean, name="markov_crisis")
    choppy_prob = pd.Series(filt_probs[:, choppy_idx], index=idx_clean, name="markov_choppy")

    # Reindex to original returns.index (NaN for warm-up rows)
    mom_prob    = mom_prob.reindex(returns.index)
    crisis_prob = crisis_prob.reindex(returns.index)
    choppy_prob = choppy_prob.reindex(returns.index)

    # ── Empirical regime characteristics ──────────────────────────────
    ret_arr    = returns.reindex(idx_clean).values
    regime_stats = {}
    for name, ri in [("MOMENTUM", mom_idx), ("CHOPPY", choppy_idx), ("CRISIS", crisis_idx)]:
        w      = filt_probs[:, ri]
        w_norm = w / w.sum() if w.sum() > 0 else w
        mean_r = float(np.dot(w_norm, ret_arr))
        vol    = float(np.sqrt(np.dot(w_norm, (ret_arr - mean_r) ** 2)) * np.sqrt(252))
        regime_stats[name] = {"mean": mean_r, "vol": vol}

    # Sanity checks
    if regime_stats["MOMENTUM"]["mean"] < 0:
        warnings.warn(
            "MOMENTUM regime has negative mean return — EM may have found a degenerate "
            "solution. Consider re-running; check feature scaling.",
            RuntimeWarning, stacklevel=2,
        )
    if regime_stats["CRISIS"]["mean"] > 0:
        warnings.warn(
            "CRISIS regime has positive mean return — state ordering may be unstable. "
            "Check feature scaling.",
            RuntimeWarning, stacklevel=2,
        )

    if verbose:
        print("\n  Markov k=3 regime characteristics (5-feature multivariate HMM):")
        for name, s in regime_stats.items():
            print(f"    {name:10s}  mean = {s['mean']*100:+.3f}%/day   vol = {s['vol']*100:.0f}% ann.")

    # ── Transition matrix ──────────────────────────────────────────────
    # hmmlearn: transmat_[i, j] = P(next=j | current=i) — row=current, col=next
    trans_info = {}
    try:
        trans_matrix = model.transmat_   # shape (k, k)

        idx_to_name = {mom_idx: "MOMENTUM", choppy_idx: "CHOPPY", crisis_idx: "CRISIS"}
        ordered     = [mom_idx, choppy_idx, crisis_idx]
        names       = [idx_to_name[i] for i in ordered]

        ordered_trans = trans_matrix[np.ix_(ordered, ordered)]
        expected_dur  = {
            names[i]: 1.0 / (1.0 - ordered_trans[i, i])
            for i in range(3) if ordered_trans[i, i] < 1.0
        }

        hard_labels = pd.Series(
            [idx_to_name[np.argmax(filt_probs[t])] for t in range(len(filt_probs))],
            index=idx_clean,
        )
        n_switches  = (hard_labels != hard_labels.shift(1)).sum()
        years       = len(filt_probs) / 252
        switches_pa = n_switches / years if years > 0 else 0

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


def fit_and_filter_markov(
    train_ret: pd.Series,
    test_ret: pd.Series,
) -> tuple:
    """
    Fit multivariate HMM on train_ret, forward-filter on test_ret.

    Used by walk-forward and expanding-window validation. All fitting
    (HMM parameters, z-score scaler) is done on train data only.
    The last 250 days of train are prepended to test_ret as rolling-window
    warm-up when constructing test features — this uses only past data.

    State labelling is re-derived mechanically from the trained model at each
    call, so it cannot drift or swap across folds.

    Args:
        train_ret : training slice daily log returns
        test_ret  : test slice daily log returns

    Returns:
        (mom_prob, crisis_prob) as pd.Series aligned to test_ret.index
        (NaN where test features could not be computed).
    """
    # ── Train: features, scaler, HMM ──────────────────────────────────
    train_feats = _build_features(train_ret).dropna()
    if len(train_feats) < 50:
        raise ValueError(f"Insufficient clean train observations: {len(train_feats)}")

    mu    = train_feats.mean()
    sigma = train_feats.std().replace(0, 1)
    X_train = _scale(train_feats, mu, sigma)

    model = _fit_hmm(X_train)

    # Mechanical state labelling — consistent across every refit
    mom_idx, crisis_idx, _ = _label_states(model)

    # ── Test: features with warm-up history, same scaler ──────────────
    warmup     = train_ret.iloc[-250:]   # last ~1yr of train for rolling init
    test_feats = _build_features(test_ret, warmup=warmup)
    test_clean = test_feats.dropna()

    if len(test_clean) == 0:
        raise ValueError("No valid test observations after dropping NaN features.")

    X_test = _scale(test_clean, mu, sigma)

    # ── Causal forward filter with frozen parameters ───────────────────
    filt_probs = _forward_filter(model, X_test)   # shape (n_test_clean, 3)

    mom_prob    = pd.Series(filt_probs[:, mom_idx],    index=test_clean.index)
    crisis_prob = pd.Series(filt_probs[:, crisis_idx], index=test_clean.index)

    # Reindex to full test_ret.index (NaN for any warm-up rows still missing)
    mom_prob    = mom_prob.reindex(test_ret.index)
    crisis_prob = crisis_prob.reindex(test_ret.index)

    return mom_prob, crisis_prob
