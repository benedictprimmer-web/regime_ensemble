# Regime Ensemble

Equity market regime detection on SPY daily bars using a two-model ensemble:

1. **Geometric** — straightness ratio (path shape over a 15-day window)
2. **Markov Switching AR(1), k=3** — probabilistic hidden-state model, regime count selected by BIC

The ensemble combines both signals with equal weights and a crisis override. Outputs a daily regime label (momentum / reversion / mixed) and runs a simple backtest against buy-and-hold.

---

## Models

### 1. Geometric — Straightness Ratio

Measures how straight the price path was over the last N days:

```
ratio(t) = |sum(r_{t-N+1} .. r_t)| / sum(|r_{t-N+1}| .. |r_t|)
```

- `ratio → 1.0` — price moved in a straight line → **MOMENTUM**
- `ratio → 0.0` — price zigzagged back to origin → **REVERSION**

**Thresholds** are percentile-based (adaptive): top 30% of the ratio distribution = momentum, bottom 30% = reversion. Fixed thresholds break at longer windows where noise accumulates and the ratio rarely exceeds 0.5 even in genuine trends.

No parameters to estimate. Zero fitting cost. No look-ahead in the signal itself (caveat: quantile thresholds are computed in-sample — see limitations).

---

### 2. Markov Switching AR(1), k=3

A Hidden Markov Model where returns follow a different AR(1) process in each regime. The EM algorithm simultaneously fits:
- Regime means and volatilities
- AR(1) coefficients per regime
- Transition matrix (probability of staying in or switching regimes)

**Why k=3?** BIC model selection on 2022–2025 SPY data:

| k | BIC      | ΔBIC vs k=2 |
|---|----------|-------------|
| 2 | −7,761   | —           |
| 3 | **−7,843** | **82** ✓  |

ΔBIC > 10 is "very strong evidence" by standard criteria. The third regime captures a distinct **crisis state** (mean ≈ −0.62%/day, vol ≈ 69% annualised) that k=2 conflates with the reversion state, understating downside risk.

**Filtered probabilities only.** Smoothed probabilities use future data and constitute look-ahead bias. All statistical tests and equity curves use filtered probabilities exclusively. The transition matrix shows regime persistence of ~92% (expected duration ~12 days) — regimes are sticky, as expected.

---

### 3. Ensemble

```
ensemble_score = mean( geo_signal, markov_crisis_adjusted )
```

Where:
- `geo_signal` ∈ {0.0, 0.5, 1.0} (hard label converted to float)
- `markov_crisis_adjusted` = P(momentum); set to 0 if P(crisis) > 0.50
- Crisis override: when Markov flags crisis, suppress the long signal regardless of other votes

Score thresholds:
- `≥ 0.65` → **momentum**
- `≤ 0.35` → **reversion**
- `0.35–0.65` → **mixed** (cash)

Equal weighting is deliberate. Fitting weights to historical returns would constitute in-sample optimisation. The ensemble's value comes from combining orthogonal signals — short-term path shape (geometric) vs long-term statistical state (Markov) — not from weight fitting.

---

## Results (SPY 2022–2025)

### Markov k=3 regime characteristics

| Regime | Mean return | Vol (ann.) |
|--------|-------------|------------|
| MOMENTUM | +0.142%/day | 10% |
| CHOPPY   | −0.029%/day | 23% |
| CRISIS   | −0.199%/day | 16% |

### Transition matrix (row = current, col = next regime)

|          | MOMENTUM | CHOPPY | CRISIS |
|----------|----------|--------|--------|
| MOMENTUM | 73.6%    | 0.5%   | 25.9%  |
| CHOPPY   | 0.5%     | 99.5%  | 0.0%   |
| CRISIS   | 100.0%   | 0.0%   | 0.0%   |

Expected durations: MOMENTUM ~4 days, CHOPPY ~200 days (~9 months). The CHOPPY regime is highly sticky — once in it, the model rarely exits. The CRISIS regime is transient (~1 day expected duration), acting more as a downside flag than a persistent state.

### Forward return by regime (next-day return when ensemble = X)

| Regime | N | Mean %/day | T-stat | P-value |
|--------|---|-----------|--------|---------|
| Momentum  | 245 | +0.0906% | 2.13 | 0.034 ✓ |
| Mixed     | 206 | +0.1228% | 1.62 | 0.108 ✗ |
| Reversion | 299 | −0.0839% | −1.06 | 0.290 ✗ |

The momentum signal is statistically significant (p < 0.05). The reversion signal is not — the model reliably identifies conditions where returns are likely to be positive, but does not reliably identify when they will be negative. This is a common finding; identifying trending conditions is easier than identifying mean-reverting ones.

### Backtest (long-only, 1-day execution lag)

| Strategy | CAGR | Sharpe | Max Drawdown |
|----------|------|--------|--------------|
| Ensemble (long only) | +7.8% | 1.23 | −4.2% |
| Buy & Hold | +7.1% | 0.39 | −24.3% |

The edge is primarily in **risk reduction** (max DD −4.2% vs −24.3%), not return maximisation. The strategy spends ~40% of the time in cash (reversion/mixed), which caps upside but substantially cuts drawdown.

### Transaction cost sensitivity

| Cost | CAGR | Sharpe | Max DD |
|------|------|--------|--------|
| 0 bps | +7.8% | 1.23 | −4.2% |
| 5 bps | +5.5% | 0.89 | −5.0% |
| 10 bps | +3.4% | 0.55 | −5.7% |
| 20 bps | −0.9% | −0.15 | −8.3% |

At realistic round-trip costs (5–10 bps), the Sharpe halves. At 20 bps the strategy is unprofitable. With ~22 regime switches per year, transaction drag is material.

---

## Limitations

These are not afterthoughts. They are the primary reasons why reported performance should not be extrapolated.

**1. Short sample (~750 observations)**
Three years is insufficient to validate across full economic cycles. The 2022–2025 window spans one specific macro transition (low-rates bear → high-rates recovery). Performance in 2008, 2013, or 2020 is untested. Standard practice is 10+ years minimum.

**2. Transaction costs are material**
At 5bps round-trip, Sharpe drops from 1.23 to 0.89. At 10bps, to 0.55. At 20bps, the strategy is unprofitable. With ~22 regime switches per year, costs are not negligible. Run `python run.py --skip-bic` to see the full cost sensitivity table.

**3. In-sample threshold calibration**
Percentile thresholds (top/bottom 30%) and ensemble cutoffs (0.65/0.35) were determined by observing the full dataset. In production these require expanding-window estimation — at time T, you can only use data up to T.

**4. EM fitting on the full window**
Even with filtered probabilities, the EM algorithm is fit on the entire 2022–2025 period. A genuinely real-time system would require periodic re-fitting on expanding or rolling windows, with associated instability at refitting points.

**5. Reversion signal lacks statistical power**
The t-stat on the reversion regime is typically ~0.8–1.1 (p > 0.3). There is no statistical justification for shorting on reversion days. The `--short` flag is provided for research purposes only.

**6. EM local optima**
The Markov EM algorithm is sensitive to initialisation. Results may vary slightly across runs. Key findings (regime ordering, crisis regime characteristics) are stable; probability values at individual dates may shift.

**7. Single asset, single period**
Regime structure on SPY 2022–2025 may not generalise to other assets, markets, or time periods. Cross-validation across assets and out-of-sample periods is necessary before drawing general conclusions.

---

## Setup

```bash
git clone <repo>
cd regime-ensemble
pip install -r requirements.txt
cp .env.example .env
# add your Polygon API key to .env
```

## Usage

```bash
# Default: SPY 2022-01-01 to 2025-01-01
python run.py

# Custom date range
python run.py --from 2020-01-01 --to 2025-01-01

# Skip BIC model selection (saves ~30s, uses k=3 directly)
python run.py --skip-bic

# Walk-forward OOS validation (~5 mins, 5 folds × 63 days)
python run.py --skip-bic --walkforward

# Allow short on reversion days (see limitations — signal is weak)
python run.py --short
```

Outputs saved to `outputs/`:
- `regime_overview.png` — SPY price coloured by regime + model signals below
- `equity_curves.png` — strategy vs buy-and-hold + drawdown panel

---

## Project structure

```
regime-ensemble/
├── run.py               — entry point
├── src/
│   ├── data.py          — Polygon fetcher, CSV cache
│   ├── geometric.py     — straightness ratio, adaptive thresholds
│   ├── markov.py        — Markov k=3, BIC selection, filtered probabilities, transition matrix
│   ├── ensemble.py      — combine signals, crisis override
│   ├── backtest.py      — backtest engine, cost sensitivity, performance statistics
│   └── walkforward.py   — walk-forward OOS validation
├── data/cache/          — cached CSV files (gitignored)
└── outputs/             — saved charts (gitignored)
```
