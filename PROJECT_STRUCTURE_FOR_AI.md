# Project Structure For AI Agents

## Canonical Repo
- Path: `/Users/benrimmer/regime-ensemble`
- Remote: `git@github.com:benedictprimmer-web/regime_ensemble.git`
- Main stable branch: `main`
- Active research branch: `feat/vix-volatility-research`

## What Goes Where

### Main Baseline Regime-Ensemble Track
- Purpose: core ensemble model and historical report workflow.
- Typical branch: `main`.
- Key files:
  - `run.py`
  - `src/markov.py`
  - `src/geometric.py`
  - `src/ensemble.py`
  - standard report scripts `generate_report_*`

### Volatility Extension Track (VIX/VVIX/GEX proxy)
- Purpose: volatility-aware extensions to the ensemble.
- Branch home: `feat/vix-volatility-research`.
- Key additions:
  - VIX handling (`--fetch-vix`, `--vix-signal`, `--vix-feature`)
  - VVIX handling (`--fetch-vvix`, `--vvix-signal`)
  - gamma-stress proxy (`--gex-proxy-signal`)
- Primary script and report naming are kept stable:
  - `generate_report_v9.py`
  - `docs/SPY_v9_vol_report copy.pdf`

## Other Local Project (Not Same Repo)
- Path: `/Users/benrimmer/level2`
- This is a separate local research codebase lineage, not the same git repo clone.
- `Paul_X_post` artifacts are a separate stream and are intentionally ignored for this repo organization.

## What To Publish
- Publish strategy/code/report artifacts from this repo (`run.py`, `src/*`, `generate_report_*.py`, `docs/*.pdf`, `README.md`).
- Do not publish assistant memory/tooling folders (`.claude/`) or unrelated side-project artifacts.

## Regime vs Ensemble Terminology
- `regime`: infer market state labels (momentum/reversion/mixed).
- `ensemble`: combine multiple regime detectors into one final signal.
