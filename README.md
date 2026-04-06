# Regime Ensemble

Detects daily equity market regimes using a two-model ensemble and backtests a **full / half / none** long strategy against buy-and-hold. The primary objective is **risk reduction**, not return maximisation.

| Strategy vs Buy & Hold | SPY price coloured by regime |
|---|---|
| ![Strategy vs Buy & Hold equity curves](docs/equity_curves.png) | ![SPY price coloured by detected regime](docs/regime_overview.png) |

---

## The Strategy

Two orthogonal signals are combined into a single ensemble score:

```
score = mean( geometric_signal , markov_momentum_probability )
```

The score drives three discrete position states:

| Score | Position | Regime | Rationale |
|---|---|---|---|
| ≥ 0.65 | **+1.0** (full long) | Momentum | Both detectors agree — trend confirmed |
| 0.35 – 0.65 | **+0.5** (half long) | Mixed | Detectors disagree — still drifts positive (T=3.21, p=0.001) |
| < 0.35 | **0.0** (cash) | Reversion / Crisis | Reversion likely, or Markov P(crisis) > 0.50 |

Equal weighting between the two signals is deliberate — fitting weights to historical returns would be in-sample optimisation. The value comes from combining two genuinely orthogonal signals: short-term path shape vs long-term statistical state.

---

## Results at a Glance

SPY · 2000–2025 · zero transaction costs · 1-day execution lag · **v5 core strategy**

| Metric | Strategy | Buy & Hold |
|---|---|---|
| CAGR | +5.7% | **+8.6%** |
| Sharpe Ratio | **0.68** | 0.44 |
| Max Drawdown | **-16.7%** | -56.5% |
| T-stat (p-value) | 3.13 (p=0.002) | 2.02 (p=0.043) |

The edge is **risk-adjusted return**, not raw return. The strategy sacrifices CAGR to stay flat during reversion and crisis regimes — the result is a Sharpe 55% higher than buy-and-hold and drawdowns that are 3.4× smaller.

> **These numbers are in-sample** — the Markov model and geometric thresholds are fitted on the full 2000–2025 dataset. For a honest estimate of live performance, run `python3 run.py --expanding --skip-bic` (annual refit, fully out-of-sample). The observed optimism bias is small (~0.03 Sharpe points, per v6 analysis), but the expanding-window number is the one to quote.

**Transaction costs matter.** ~61 position switches/year means costs compound quickly. Sharpe beats B&H to ~12 bps round-trip; strategy breaks even at ~17 bps. Use `--min-hold 3` to cut switches and extend the profitable range.

---

## How It Works

### Signal 1 — Geometric Detector (path shape)

Measures how straight the last 15 days of price movement was:

```
ratio(t) = |cumulative return over window| / Σ|daily returns|
```

- **Near 1.0** → price moved in a straight line → momentum signal
- **Near 0.0** → price zigzagged back to start → reversion signal

Thresholds are adaptive percentiles (top/bottom 30% of the ratio distribution), not fixed values. Zero parameters to estimate — no fitting cost, no look-ahead.

### Signal 2 — Gaussian HMM k=3 (hidden market state)

A 3-state hidden Markov model fit on a 5-feature observation vector that directly measures market structure rather than inferring it from daily return autocorrelation:

| Feature | What it measures |
|---|---|
| `ret_20d` | 20-day cumulative return — medium-term direction |
| `ret_5d` | 5-day cumulative return — short-term momentum |
| `rvol_20d` | 20-day realised vol (annualised) — stress level |
| `drawdown` | Distance from 252-day high — peak-to-trough loss |
| `dist_200d` | Price / 200-day MA − 1 — structural trend position |

State labels are assigned mechanically at every refit: highest mean `ret_20d` = Momentum, lowest = Crisis, middle = Choppy. Uses **filtered (causal) probabilities only** — no look-ahead bias. When P(crisis) > 0.50, the buy signal is suppressed regardless of all other indicators.

### The Key Statistical Finding

The mixed regime — when the two detectors disagree — still has a statistically significant positive forward return (T=3.21, p=0.001). This is what justifies holding a half-position rather than defaulting to cash on uncertainty.

| Regime | Mean return/day | T-stat | p-value |
|---|---|---|---|
| Mixed | +0.05% | 3.21 | **0.001** |
| Momentum | +0.06% | 1.33 | 0.18 |
| Reversion | -0.01% | -0.35 | 0.73 |

---

## Reports

Start with the overview. The technical and research reports build on it.

| Document | Script | Description |
|---|---|---|
| [Overview (3 pages)](docs/SPY_overview_report.pdf) | `generate_report_overview.py` | **Start here.** Strategy, signals in action, performance vs B&H, limitations |
| [Technical Quant (3 pages)](docs/SPY_quant_report.pdf) | `generate_report_quant.py` | KDE distributions, K-S tests, transition matrix, rolling IC, full attribution |
| [v6 Technical (3 pages)](docs/SPY_v6_report.pdf) | `generate_report_v6.py` | Vol-ratio dampening, multi-scale geometric, expanding-window OOS |
| [v7 Research (2 pages)](docs/SPY_v7_research_report.pdf) | `generate_report_v7.py` | Directional geometric and continuous sizing — both underperform, with analysis of why |
| [v8 Research (2 pages)](docs/SPY_v8_report.pdf) | `generate_report_v8.py` | Kalman drift filter — Q≈0 finding, why daily Kalman underperforms |
| [v9 Volatility Research (3 pages)](docs/SPY_v9_vol_report copy.pdf) | `generate_report_v9.py` | VIX/VVIX/gamma-proxy extensions and ablations; current active research direction |
| [Paul Refinements v2](docs/Paul_refinement_pdf_V2.pdf) | `generate_report_paul_refinements_v2.py` | Cleaned presentation + diagnostics for communicating model behavior and decisions |
| [v4/v5 Changes](docs/v4v5_changes_report.pdf) | `generate_report_v4v5.py` | Half-position mixed regime, persistence filter, walk-forward OOS |

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
# Standard backtest (SPY 2000-2025)
python3 run.py --skip-bic

# Custom ticker and date range
python3 run.py --ticker QQQ --from 2010-01-01 --to 2025-01-01 --skip-bic
```

```bash
# Generate reports (uses cached data — no API key needed once cached)
python3 generate_report_overview.py   # 3-page overview: strategy, signals, results
python3 generate_report_quant.py      # full quant deep-dive
python3 generate_report_v6.py         # v6 technical report
python3 generate_report_v7.py         # v7 research: directional geo + continuous sizing
python3 generate_report_v8.py         # v8 research: Kalman filter
```

```bash
# Validation
python3 run.py --walkforward --skip-bic     # walk-forward OOS (10 folds × 63 days)
python3 run.py --multi-asset --skip-bic     # SPY, QQQ, IWM, TLT, GLD comparison
python3 run.py --expanding   --skip-bic     # expanding-window honest backtest (~5-10 min)
```

```bash
# v6 additions (on top of v5 baseline)
python3 run.py --vol-signal --multi-scale --skip-bic   # vol dampening + multi-scale geo
```

```bash
# Volatility expansion (VVIX + gamma-stress proxy; proxy != true options-chain GEX)
python3 run.py --vvix-signal --skip-bic                              # add VVIX dampening
python3 run.py --gex-proxy-signal --skip-bic                         # add gamma-stress proxy dampening
python3 run.py --vvix-signal --gex-proxy-signal --ablation --skip-bic  # baseline vs vvix/proxy variants
```

```bash
# Research flags (v7 and v8 — both underperform the v5 baseline)
python3 run.py --geo-directional --skip-bic   # signed straightness ratio (v7)
python3 run.py --continuous      --skip-bic   # continuous position sizing (v7)
python3 run.py --kalman          --skip-bic   # Kalman drift filter (v8, Q≈0)
python3 run.py --short           --skip-bic   # allow short on reversion (signal not significant)
python3 run.py --min-hold 3      --skip-bic   # persistence filter: 3-day minimum hold
```

Outputs are saved to `outputs/` prefixed with `{ticker}_{from}_{to}_`.

---

## Limitations

These are not afterthoughts — they are the primary reasons results should not be extrapolated.

1. **Strategy underperforms B&H on raw CAGR** — +5.7% vs +8.6% over 25 years. The edge is a better Sharpe (0.68 vs 0.44) and dramatically lower drawdown. The signal is statistically significant (T=3.13, p=0.002).
2. **Transaction costs are material** — ~61 position switches/year means costs compound. Sharpe beats B&H to ~12 bps round-trip; strategy breaks even at ~17 bps.
3. **In-sample threshold calibration** — percentile thresholds and ensemble cutoffs were tuned on the full dataset. Real-time use requires expanding-window recalibration (see `--expanding`).
4. **Reversion signal is not significant** — reversion p=0.73 over 25 years. The `--short` flag exists for research only.
5. **Calibrated on SPY** — thresholds and Markov parameters are fitted on SPY. The `--multi-asset` flag applies the same model to QQQ, IWM, TLT, GLD as a validation check, but each asset's regime structure differs.
6. **Gamma proxy is not true dealer GEX** — `--gex-proxy-signal` uses a causal spot/vol shock proxy (returns + VIX + optional VVIX), not full options-chain open-interest based gamma exposure.

---

## Versioning

The core strategy is **v5**. Versions 6, 7, and 8 are research iterations that extend v5 — v6 adds useful enhancements, v7 and v8 are honest investigations that underperform.

### Current focus
- We are now prioritizing the **volatility extension track** (v9 branch work): VIX/VVIX conditioning plus gamma-stress proxy dampening.
- v7 and v8 remain valuable negative results and should be kept as documented research trials.
- Paul refinement reports are treated as communication/analysis improvements around the same project, not a separate strategy fork.

### v9.0 (in progress)
- **VVIX dampening** (`--vvix-signal`) — adds a vol-of-vol stress suppressor on the Markov momentum component.
- **Gamma-stress proxy dampening** (`--gex-proxy-signal`) — adds a causal proxy built from downside spot shock + VIX jump (+ optional VVIX jump). This is a proxy and intentionally not labeled as true dealer GEX.
- **Extended ablation** (`--ablation` with `--vvix-signal` and/or `--gex-proxy-signal`) — compares baseline vs dampener variants to isolate incremental value.

### v8.0
- **Kalman drift filter** (`--kalman`) — local-level Kalman model estimated by MLE on innovation likelihood. Only 2 parameters (Q, R). Signal = norm.cdf(μ_t / √(P_t + R)) added as a 3rd ensemble component.
- **Result: Sharpe 0.60 vs 0.68 baseline** — MLE correctly estimates Q ≈ 0 (daily drift is undetectable against return noise). The signal stays near 0.5 and dilutes the existing ensemble. The underperformance is itself evidence against overfitting: an overfit model would have tuned Q to match historical patterns.
- See the [v8 Research Report](docs/SPY_v8_report.pdf) for Q/R parameter stability, signal distributions, and full results.

### v7.0
- **Directional geometric** (`--geo-directional`) — signed straightness ratio in [−1, +1]. Fixes direction-blindness where crashes are labelled momentum. **Result: Sharpe 0.32** — the Markov crisis override already handles downtrends; directional geometry creates a correlated second crash detector that reduces diversity without adding information.
- **Continuous position sizing** (`--continuous`) — position = ensemble score [0,1] directly, bypassing discrete {0, 0.5, 1} labels. **Result: Sharpe 0.56** — the empirically calibrated discrete half-position (T=3.21, p=0.001) outperforms raw score; high-conviction days are underweighted.
- See the [v7 Research Report](docs/SPY_v7_research_report.pdf) for charts and full analysis.

### v6.0
- **Vol ratio dampening** (`--vol-signal`) — 5-day / 63-day realised vol ratio suppresses Markov momentum signal when short-term vol exceeds 2× baseline. Active ~11% of days. Endogenous, orthogonal to existing signals.
- **Multi-scale geometric** (`--multi-scale`) — averages straightness ratio across 5, 15, and 30-day windows before thresholding. More robust across volatility regimes.
- **Expanding-window honest backtest** (`--expanding`) — refits geometric thresholds and Markov model annually on all data up to that date. Optimism bias vs in-sample = 0.03 Sharpe points (small gap).

### v5.0
- Walk-forward OOS equity curve (`--walkforward`) — 10-fold stitched continuous OOS curve.
- Multi-asset validation (`--multi-asset`) — SPY, QQQ, IWM, TLT, GLD.

### v4.0
- Half-position on mixed days — mixed regime T=3.21, p=0.001. Changed from cash to +0.5 long. Sharpe improved 0.29 → 0.68; strategy T-stat 1.32 (p=0.19) → 3.13 (p=0.002).
- Regime persistence filter (`--min-hold N`) — N consecutive days required before position switches.

### v3.0
- Extended data to 2000-2025 — covers dot-com, GFC 2008, COVID 2020, 2022 bear market.

### v2.0
- Fixed walk-forward leakage — Markov EM fitted on train-only data; test slice forward-filtered with frozen parameters.

### v1.0
- Initial release: geometric + Markov k=3 ensemble on SPY 2022-2025.

---

## Project Structure

```
regime_ensemble/
├── run.py                           -- main entry point: backtest + charts
│
├── generate_report_overview.py      -- 3-page overview: strategy, signals, results  ← start here
├── generate_report_quant.py         -- 3-page technical quant deep-dive
├── generate_report_v6.py            -- v6 technical: vol dampening, multi-scale, expanding window
├── generate_report_v7.py            -- v7 research: directional geo + continuous sizing
├── generate_report_v8.py            -- v8 research: Kalman drift filter
├── generate_report_v4v5.py          -- v4/v5 methodology changes (historical reference)
├── generate_report_v2_historical.py -- v2 one-page methodology update (historical reference)
│
├── requirements.txt
├── .env.example                     -- copy to .env, add POLYGON_API_KEY
│
├── src/
│   ├── data.py          -- Polygon.io fetcher, CSV cache, multi-ticker support
│   ├── geometric.py     -- straightness ratio, adaptive thresholds, multi-scale
│   ├── markov.py        -- Markov k=3, BIC selection, filtered probabilities
│   ├── ensemble.py      -- combine signals, vol dampening, crisis override
│   ├── backtest.py      -- backtest engine, cost sensitivity, performance stats
│   ├── kalman.py        -- local-level Kalman filter (v8)
│   ├── walkforward.py   -- walk-forward OOS validation (10 × 63-day folds)
│   └── expanding.py     -- expanding-window honest backtest (annual refit)
│
├── docs/
│   ├── SPY_overview_report.pdf      -- 3-page overview (v5)
│   ├── SPY_quant_report.pdf         -- full technical quant report (v5)
│   ├── SPY_v6_report.pdf            -- v6 features technical report
│   ├── SPY_v7_research_report.pdf   -- v7 research report
│   ├── SPY_v8_report.pdf            -- v8 research report
│   ├── v4v5_changes_report.pdf      -- v4/v5 methodology changes
│   ├── equity_curves.png            -- strategy vs buy-and-hold
│   └── regime_overview.png          -- SPY price coloured by regime
│
└── data/cache/                      -- cached CSV files (gitignored)
```
