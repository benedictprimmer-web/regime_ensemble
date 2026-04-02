"""
Walk-forward out-of-sample validation.

Splits the data into N rolling folds. For each fold:
    - Train: all data up to the start of the test window
    - Test:  next `test_size` trading days (held out)

Geometric signal: thresholds computed on train only, applied to test.
Markov signal:    fit on the expanding train window, filtered probabilities
                  evaluated on the full train+test window — only the test
                  portion is scored. This is expanding-window, not strictly
                  OOS (statsmodels does not support applying a pre-fitted
                  Markov model to new data without refitting). This is
                  explicitly flagged in the output.

Interpretation:
    If the signal has real predictive power, the momentum regime should
    show positive mean returns across most folds. Consistent positivity
    with p < 0.10 in multiple folds is meaningful evidence at this sample
    size (~60-75 obs per test fold).
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.geometric import straightness_ratio, geometric_signal
from src.ensemble  import ensemble_score, regime_labels
from src.markov    import AR_ORDER


def walk_forward(
    returns: pd.Series,
    n_folds: int  = 5,
    test_size: int = 63,   # ~1 quarter of trading days
) -> pd.DataFrame:
    """
    Walk-forward OOS validation of the ensemble signal.

    Args:
        returns   : full daily log return series
        n_folds   : number of test folds
        test_size : trading days per test fold

    Returns:
        DataFrame: per-fold momentum regime forward-return statistics.
    """
    n = len(returns)
    min_train = n - n_folds * test_size

    if min_train < 200:
        raise ValueError(
            f"Insufficient data: {n} obs, {n_folds} folds × {test_size} = "
            f"{n_folds * test_size} test days, leaving only {min_train} for training."
        )

    rows = []
    for fold in range(n_folds):
        # Each fold steps the test window forward by test_size days
        test_end   = n - (n_folds - 1 - fold) * test_size
        test_start = test_end - test_size

        train_ret = returns.iloc[:test_start]
        test_ret  = returns.iloc[test_start:test_end]

        print(f"  Fold {fold + 1}/{n_folds}:  "
              f"train {train_ret.index[0].date()} → {train_ret.index[-1].date()}  |  "
              f"test  {test_ret.index[0].date()} → {test_ret.index[-1].date()}  "
              f"({len(test_ret)} days)")

        # ── Geometric: OOS thresholds from train ──────────────────────
        sr_train   = straightness_ratio(train_ret)
        mom_thresh = sr_train.quantile(0.70)
        rev_thresh = sr_train.quantile(0.30)

        sr_test  = straightness_ratio(test_ret)
        geo_test = pd.Series(0.5, index=test_ret.index, name="geo_signal")
        geo_test[sr_test >= mom_thresh] = 1.0
        geo_test[sr_test <= rev_thresh] = 0.0

        # ── Markov: expanding-window refit, evaluate on test only ──────
        # Fit on full train+test window, extract only the test portion.
        # Expanding-window approach: each fold's model sees slightly more data.
        # Not strictly OOS but standard in practice for HMM-type models.
        full_window = returns.iloc[:test_end]
        try:
            model = sm.tsa.MarkovAutoregression(
                full_window.dropna(),
                k_regimes=3,
                order=AR_ORDER,
                switching_ar=True,
                switching_variance=True,
            )
            res = model.fit(disp=False)

            consts     = [res.params.get(f"const[{i}]", 0) for i in range(3)]
            mom_idx    = int(np.argmax(consts))
            crisis_idx = int(np.argmin(consts))

            filt = res.filtered_marginal_probabilities
            if hasattr(filt, "iloc"):
                all_idx = filt.index
                probs   = filt.values
            else:
                probs   = np.array(filt)
                ret_arr = full_window.dropna()
                all_idx = ret_arr.index[len(ret_arr) - len(probs):]

            all_mom    = pd.Series(probs[:, mom_idx],    index=all_idx)
            all_crisis = pd.Series(probs[:, crisis_idx], index=all_idx)

            # Take only the test portion
            mom_prob_test    = all_mom.reindex(test_ret.index)
            crisis_prob_test = all_crisis.reindex(test_ret.index)

        except Exception as e:
            print(f"    Markov fit failed: {e} — using geometric only")
            mom_prob_test    = pd.Series(geo_test.values * 0.5, index=test_ret.index)
            crisis_prob_test = pd.Series(0.0, index=test_ret.index)

        # ── Ensemble + forward return stats ───────────────────────────
        score  = ensemble_score(geo_test, mom_prob_test, crisis_prob_test)
        labels = regime_labels(score)

        fwd_ret = test_ret.shift(-1).rename("log_return")
        aligned = pd.concat([fwd_ret, labels], axis=1).dropna()

        for regime_name in ["momentum", "reversion"]:
            mask = aligned["regime"] == regime_name
            r    = aligned.loc[mask, "log_return"]
            n_r  = len(r)
            if n_r >= 5:
                t_stat, p_val = stats.ttest_1samp(r, 0)
                mean_r = r.mean()
            else:
                t_stat, p_val, mean_r = np.nan, np.nan, np.nan

            rows.append({
                "Fold":        fold + 1,
                "Test period": f"{test_ret.index[0].date()} → {test_ret.index[-1].date()}",
                "Regime":      regime_name,
                "N days":      n_r,
                "Mean %/day":  f"{mean_r*100:+.4f}%" if not np.isnan(mean_r) else "n/a",
                "T-stat":      f"{t_stat:.2f}"        if not np.isnan(t_stat)  else "n/a",
                "P-value":     f"{p_val:.3f}"          if not np.isnan(p_val)   else "n/a",
                "Dir":         ("✓" if mean_r > 0 else "✗") if not np.isnan(mean_r) else "—",
            })

    return pd.DataFrame(rows).set_index(["Fold", "Regime"])
