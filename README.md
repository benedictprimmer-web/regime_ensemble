# Regime Ensemble

Detects daily equity market regimes (momentum / reversion / mixed) using a two-model ensemble, then backtests a simple long/cash strategy against buy-and-hold. The primary goal is **risk reduction**, not return maximisation.

![SPY price coloured by detected regime](docs/regime_overview.png)

---

## Documents

| Document | Description |
|---|---|
| [3-Page Report (PDF)](docs/SPY_3page_report.pdf) | Plain-English overview — price, signals, equity curves, limitations (2000–2025) |
| [Technical Quant Report (PDF)](docs/SPY_quant_report.pdf) | Regime KDE distributions, K-S tests, transition matrix, rolling IC, full attribution |
| [v2.0 Methodology Update (PDF)](docs/v2_methodology_report.pdf) | What changed in v2: walk-forward leakage fix, extended data pipeline |

---

## Results at a Glance

SPY · 2000–2025 · zero transaction costs · 1-day execution lag

| Metric | Strategy | Buy & Hold |
|---|---|---|
| CAGR | +5.7% | **+8.6%** |
| Sharpe Ratio | **0.68** | 0.44 |
| Max Drawdown | **−16.7%** | −56.5% |
| T-stat (p-value) | 3.13 (p=0.002) | 2.02 (p=0.043) |

> **Strategy = full long on momentum, half-long on mixed, cash on reversion.** Sharpe beats buy-and-hold (0.68 vs 0.44) with dramatically lower drawdown (−16.7% vs −56.5%). Remains profitable at 10bps round-trip (Sharpe 0.28). At 20bps it breaks even. See the reports below for the full cost sensitivity table.

---

## How It Works

### 1 · Geometric Detector — Path Shape

Measures how straight the last 15 days of price movement was:

```
ratio(t) = |cumulative return over window| / sum of |daily returns|
```

- **Near 1.0** → price moved in a straight line → momentum
- **Near 0.0** → price zigzagged back to start → mean-reversion

Thresholds are adaptive percentiles (top/bottom 30% of the distribution), not fixed values. No parameters to estimate — zero fitting cost, fast.

### 2 · Markov Switching AR(1) — Hidden State

A statistical model that learns three hidden market states from daily return patterns. k=3 is selected by BIC (ΔBIC = 82 over k=2):

| Regime | Mean return | Vol (ann.) |
|---|---|---|
| Momentum | +0.142%/day | 10% |
| Choppy | −0.029%/day | 23% |
| Crisis | −0.199%/day | 16% |

Uses **filtered probabilities only** — no look-ahead bias. When P(crisis) > 0.50, the buy signal is suppressed regardless of other indicators.

### 3 · Ensemble

```
score = mean( geometric_signal, markov_momentum_probability )
```

- Score ≥ 0.65 → **momentum** (hold)
- Score ≤ 0.35 → **reversion** (cash)
- 0.35 – 0.65  → **mixed** (cash)

Equal weighting is deliberate — fitting weights to historical returns would be in-sample optimisation. The value comes from combining two orthogonal signals: short-term path shape vs long-term statistical state.

---

## Setup

```bash
git clone https://github.com/benedictprimmer-web/regime_ensemble.git
cd regime_ensemble
pip install -r requirements.txt
cp .env.example .env
# Add your Polygon.io API key to .env
```

---

## Usage

```bash
# Default: SPY 2000-01-01 to 2025-01-01
python3 run.py --skip-bic

# Custom ticker and date range
python3 run.py --ticker QQQ --from 2010-01-01 --to 2025-01-01 --skip-bic

# Fetch VIX data (requires Polygon paid plan) and use as dampening signal
python3 run.py --fetch-vix --vix-signal --skip-bic

# Walk-forward out-of-sample validation (~10 mins, 10 folds x 63 days)
python3 run.py --skip-bic --walkforward

# Allow short on reversion days (signal is weak -- see limitations)
python3 run.py --short --skip-bic

# Generate the 3-page PDF report (uses cached data, no API key needed once cached)
python3 generate_report_3page.py

# Generate the 1-page v2 methodology update
python3 generate_report.py
```

Outputs are saved to `outputs/` and prefixed with `{ticker}_{from}_{to}_`.

---

## Limitations

These are not afterthoughts — they are the primary reasons results should not be extrapolated.

1. **Strategy underperforms B&H on raw CAGR** — +5.7% vs +8.6% over 25 years. The edge is a better Sharpe (0.68 vs 0.44) and dramatically lower drawdown. The signal is statistically significant (T=3.13, p=0.002).
2. **Transaction costs are material** — ~20 regime switches/year means costs compound. Strategy is profitable to ~15bps round-trip; breaks even around 20bps.
3. **In-sample threshold calibration** — percentile thresholds and ensemble cutoffs were tuned on the full dataset. Real-time use requires expanding-window recalibration.
4. **Reversion signal is not significant** — reversion p=0.73 over 25 years. The `--short` flag exists for research only.
5. **Single asset** — SPY only. Regime structure may not generalise to other assets or markets.

---

## Project Structure

```
regime_ensemble/
├── run.py                    -- main entry point (backtest + charts)
├── generate_report_3page.py  -- 3-page plain-English PDF report (2000-2025)
├── generate_report_quant.py  -- 3-page technical quant PDF report (2000-2025)
├── generate_report.py        -- 1-page v2 methodology update PDF
├── requirements.txt
├── .env.example              -- copy to .env, add POLYGON_API_KEY
├── src/
│   ├── data.py               -- Polygon.io fetcher, CSV cache, multi-ticker support
│   ├── geometric.py          -- straightness ratio, adaptive thresholds
│   ├── markov.py             -- Markov k=3, BIC selection, filtered probabilities
│   ├── ensemble.py           -- combine signals, crisis override
│   ├── backtest.py           -- backtest engine, cost sensitivity, performance stats
│   └── walkforward.py        -- walk-forward OOS validation (v2: fully OOS)
├── docs/
│   ├── SPY_3page_report.pdf  -- 3-page overview report
│   ├── v2_methodology_report.pdf -- v2 methodology update
│   ├── regime_overview.png   -- SPY price coloured by regime
│   └── equity_curves.png     -- strategy vs buy-and-hold
└── data/cache/               -- cached CSV files (gitignored)
```

---

## Changelog

### v3.0
- **Extended data to 2000–2025** — all reports and report generators now use the full 25-year history covering dot-com crash, GFC 2008, COVID 2020, and 2022 bear market.
- **Half-position on mixed days** — "mixed" regime (when the two detectors disagree) has the strongest forward-return signal (T=3.21, p=0.001). Changed from cash to 0.5× long; Sharpe improves from 0.29 to 0.68, T-stat from 1.32 to 3.13.
- **VIX signal integration** — `--vix-signal` flag adds VIX as a continuous dampening factor on the momentum signal (VIX ≤ 20 = no effect; VIX 30 = 50% dampening; VIX ≥ 40 = fully suppressed). Orthogonal to the existing Markov crisis override.
- **Walk-forward extended to 10 folds** — `--walkforward` now runs 10 × 63-day folds (was 5), covering multiple distinct market cycles for more robust OOS validation.

### v2.0
- **Fixed walk-forward leakage** — Markov EM now fitted on train-only data; test slice is forward-filtered with frozen parameters via `.filter(params)`. Geometric thresholds use the new `compute_thresholds()` helper for clean train/test isolation.
- **Extended data pipeline** — default date range 2000-2025, `--ticker` flag for any Polygon ticker, `--fetch-vix` for `I:VIX` data, output files prefixed to avoid overwrites.
- **`docs/` folder** — PDFs and charts committed to the repo for direct GitHub viewing.
- **Report generators** — `generate_report_3page.py` and `generate_report.py` for reproducible PDF outputs.

### v1.0
- Initial release: geometric + Markov k=3 ensemble on SPY 2022-2025, walk-forward validation, transaction cost sensitivity.
