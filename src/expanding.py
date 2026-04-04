"""
Expanding-window honest backtest.

The standard run.py backtest uses in-sample thresholds throughout — the
geometric percentile thresholds and Markov parameters are fitted on the
full 2000-2025 dataset. This is correct for a research report but overstates
what you would have achieved in real-time, because you cannot know the final
distribution of ratios at the start of the sample.

This module implements a proper expanding-window simulation:
    - At each refit date, geometric thresholds are computed on ALL data up
      to (but not including) that date.
    - The Markov model is refitted on all data up to that date.
    - Labels for the NEXT period are generated using those frozen parameters
      (Hamilton filter only — no EM on future data).
    - Periods step annually (252 trading days) to keep runtime manageable.
      Typical runtime: ~5-10 minutes for 25 years.

This answers: "what would the strategy have actually returned if run live,
re-fitting each year as more data accumulated?"

Comparison to in-sample backtest:
    Optimism bias = in-sample Sharpe - expanding-window Sharpe
    A large gap means the thresholds/model are overfit to the full history.
    A small gap (< 0.1 Sharpe) suggests the signal is robust.
"""

import numpy as np
import pandas as pd

from src.geometric import geometric_signal, compute_thresholds, MULTI_WINDOWS
from src.ensemble  import ensemble_score, regime_labels
from src.markov    import fit_and_filter_markov
from src.backtest  import run_backtest


def expanding_backtest(
    returns: pd.Series,
    min_train_days: int = 504,    # 2 years minimum before first live period
    refit_freq: int     = 252,    # refit annually (1 trading year)
    multi_scale: bool   = False,  # use multi-scale geometric if True
    geo_directional: bool = False,
    verbose: bool       = True,
) -> pd.DataFrame:
    """
    Run an expanding-window backtest: refit model annually, trade the next year.

    At each step:
      1. Geometric thresholds computed on train slice only.
      2. Markov model fitted on train slice only (EM).
      3. Markov Hamilton filter applied to test slice with frozen params.
      4. Ensemble labels generated for test slice.
      5. Strategy returns recorded for test slice (1-day execution lag).

    Args:
        returns        : full daily log return series
        min_train_days : minimum days before first refit (default ~2 years)
        refit_freq     : trading days between refits (default 252 = annual)
        multi_scale    : if True, use multi-scale geometric (windows 5/15/30)
        verbose        : print progress per step

    Returns:
        DataFrame with columns: strategy_return, bnh_return
        Covers the test periods only (excludes the initial training window).
    """
    n = len(returns)
    geo_windows = MULTI_WINDOWS if multi_scale else None

    pieces = []
    step = 0

    for train_end in range(min_train_days, n, refit_freq):
        test_end = min(train_end + refit_freq, n)
        train_ret = returns.iloc[:train_end]
        test_ret  = returns.iloc[train_end:test_end]

        if len(test_ret) < 5:
            break

        step += 1
        if verbose:
            print("  Step %d:  train %s -> %s  |  test %s -> %s  (%d days)" % (
                step,
                train_ret.index[0].date(), train_ret.index[-1].date(),
                test_ret.index[0].date(),  test_ret.index[-1].date(),
                len(test_ret),
            ))

        # -- Geometric: thresholds from train only ---------------------------
        mom_thresh, rev_thresh = compute_thresholds(
            train_ret, windows=geo_windows, directional=geo_directional
        )
        geo_test = geometric_signal(
            test_ret,
            mom_thresh=mom_thresh,
            rev_thresh=rev_thresh,
            windows=geo_windows,
            directional=geo_directional,
        )

        # -- Markov: fit on train, forward-filter test -----------------------
        try:
            mom_prob_test, crisis_prob_test = fit_and_filter_markov(train_ret, test_ret)
        except Exception as e:
            if verbose:
                print("    Markov fit failed: %s -- using geometric only" % e)
            mom_prob_test    = pd.Series(geo_test.values * 0.5, index=test_ret.index)
            crisis_prob_test = pd.Series(0.0, index=test_ret.index)

        # -- Ensemble + backtest ---------------------------------------------
        score  = ensemble_score(geo_test, mom_prob_test, crisis_prob_test)
        labels = regime_labels(score)
        bt     = run_backtest(test_ret, labels, allow_short=False, cost_bps=0)
        pieces.append(bt[["strategy_return", "bnh_return"]])

    if not pieces:
        raise ValueError("No test periods generated. Increase data length or reduce min_train_days.")

    return pd.concat(pieces).sort_index().dropna()
