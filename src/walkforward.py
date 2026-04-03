"""
Walk-forward out-of-sample validation.

Splits the data into N rolling folds. For each fold:
    - Train: all data up to the start of the test window
    - Test:  next `test_size` trading days (held out)

Geometric signal: thresholds computed on train only via `compute_thresholds()`,
                  then applied to the test slice — fully OOS.

Markov signal:    fit on train-only data, then forward-filtered on the test
                  slice using `.filter(res_train.params)` (statsmodels 0.14+).
                  This freezes the EM-estimated parameters at train time and
                  applies only the Hamilton filter to the test observations —
                  fully OOS with no look-ahead bias.

Interpretation:
    If the signal has real predictive power, the momentum regime should
    show positive mean returns across most folds. Consistent positivity
    with p < 0.10 in multiple folds is meaningful evidence at this sample
    size (~60-75 obs per test fold).
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import EstimationWarning

from src.geometric import straightness_ratio, geometric_signal, compute_thresholds
from src.ensemble  import ensemble_score, regime_labels
from src.markov    import AR_ORDER
from src.backtest  import run_backtest


def walk_forward(
    returns: pd.Series,
    n_folds: int  = 5,
    test_size: int = 63,   # ~1 quarter of trading days
) -> tuple:
    """
    Walk-forward OOS validation of the ensemble signal.

    Args:
        returns   : full daily log return series
        n_folds   : number of test folds
        test_size : trading days per test fold

    Returns:
        (stats_df, oos_returns) where:
            stats_df    : per-fold momentum/reversion forward-return statistics
            oos_returns : continuous OOS daily returns DataFrame with columns
                          strategy_return and bnh_return (all folds concatenated)
    """
    n = len(returns)
    min_train = n - n_folds * test_size

    if min_train < 200:
        raise ValueError(
            f"Insufficient data: {n} obs, {n_folds} folds × {test_size} = "
            f"{n_folds * test_size} test days, leaving only {min_train} for training."
        )

    rows = []
    oos_pieces = []
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
        mom_thresh, rev_thresh = compute_thresholds(train_ret)
        geo_test = geometric_signal(test_ret, mom_thresh=mom_thresh, rev_thresh=rev_thresh)

        # ── Markov: train-only fit, forward-filter on test ─────────────
        # Fit the Markov model on train data only. Then construct a new
        # model instance over the test slice and call .filter(train_params)
        # to run the Hamilton filter with frozen parameters — no EM on test.
        try:
            model_train = sm.tsa.MarkovAutoregression(
                train_ret.dropna(),
                k_regimes=3,
                order=AR_ORDER,
                switching_ar=True,
                switching_variance=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EstimationWarning)
                res_train = model_train.fit(disp=False, em_iter=200)

            consts     = [res_train.params.get(f"const[{i}]", 0) for i in range(3)]
            mom_idx    = int(np.argmax(consts))
            crisis_idx = int(np.argmin(consts))

            # Forward-filter test slice with frozen train parameters
            model_test = sm.tsa.MarkovAutoregression(
                test_ret.dropna(),
                k_regimes=3,
                order=AR_ORDER,
                switching_ar=True,
                switching_variance=True,
            )
            res_test = model_test.filter(res_train.params)

            filt = res_test.filtered_marginal_probabilities
            if hasattr(filt, "iloc"):
                test_idx = filt.index
                probs    = filt.values
            else:
                probs    = np.array(filt)
                ret_arr  = test_ret.dropna()
                test_idx = ret_arr.index[len(ret_arr) - len(probs):]

            mom_prob_test    = pd.Series(probs[:, mom_idx],    index=test_idx)
            crisis_prob_test = pd.Series(probs[:, crisis_idx], index=test_idx)

            mom_prob_test    = mom_prob_test.reindex(test_ret.index)
            crisis_prob_test = crisis_prob_test.reindex(test_ret.index)

        except Exception as e:
            print(f"    Markov fit failed: {e} — using geometric only")
            mom_prob_test    = pd.Series(geo_test.values * 0.5, index=test_ret.index)
            crisis_prob_test = pd.Series(0.0, index=test_ret.index)

        # ── Ensemble + forward return stats ───────────────────────────
        score  = ensemble_score(geo_test, mom_prob_test, crisis_prob_test)
        labels = regime_labels(score)

        bt_fold = run_backtest(test_ret, labels, allow_short=False, cost_bps=0)
        oos_pieces.append(bt_fold[["strategy_return", "bnh_return"]])

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

    stats_df    = pd.DataFrame(rows).set_index(["Fold", "Regime"])
    oos_returns = pd.concat(oos_pieces).sort_index().dropna()
    return stats_df, oos_returns
