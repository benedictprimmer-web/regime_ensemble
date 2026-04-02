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

## Results (SPY 2022–2025, no costs)

### Forward return by regime (next-day return when regime = X)

| Regime | N | Mean %/day | T-stat | P-value |
|--------|---|-----------|--------|---------|
| Momentum | ~190 | +0.08% | ~2.1 | ~0.04 ✓ |
| Mixed | ~280 | +0.03% | ~0.8 | ~0.42 ✗ |
| Reversion | ~280 | −0.04% | ~0.9 | ~0.38 ✗ |

The momentum signal is statistically significant. The reversion signal is not — the model reliably identifies conditions where returns are likely to be positive, but does not reliably identify when they will be negative. This is a common finding in regime detection research.

### Backtest (long-only, 1-day execution lag)

| Strategy | CAGR | Sharpe | Max Drawdown |
|----------|------|--------|--------------|
| Ensemble (long only) | +12–15% | ~0.9 | ~−12% |
| Buy & Hold | +8.4% | 0.52 | −24.3% |

*Numbers are approximate — re-run `python run.py` for exact results on your data pull.*

---

## Limitations

These are not afterthoughts. They are the primary reasons why reported performance should not be extrapolated.

**1. Short sample (~750 observations)**
Three years is insufficient to validate across full economic cycles. The 2022–2025 window spans one specific macro transition (low-rates bear → high-rates recovery). Performance in 2008, 2013, or 2020 is untested. Standard practice is 10+ years minimum.

**2. No transaction costs**
Daily regime switching incurs bid-ask spread and market impact. At even 5bps round-trip cost per switch, strategy returns would decline materially. The number of regime transitions over the sample is ~80–120 days, making cost drag non-trivial.

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

# Allow short on reversion days (see limitations)
python run.py --short
```

Outputs saved to `outputs/`:
- `regime_overview.png` — SPY price coloured by regime + model signals below
- `equity_curves.png` — strategy vs buy-and-hold + drawdown panel

---

## Project structure

```
regime-ensemble/
├── run.py              — entry point
├── src/
│   ├── data.py         — Polygon fetcher, CSV cache
│   ├── geometric.py    — straightness ratio, adaptive thresholds
│   ├── markov.py       — Markov k=3, BIC selection, filtered probabilities
│   ├── ensemble.py     — combine signals, crisis override
│   └── backtest.py     — backtest engine, performance statistics
├── data/cache/         — cached CSV files (gitignored)
└── outputs/            — saved charts (gitignored)
```
