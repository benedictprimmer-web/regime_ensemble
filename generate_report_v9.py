#!/usr/bin/env python3
"""
Regime Ensemble v9.0 - VVIX and Gamma-Proxy Report (3 pages)
==============================================================
Documents the volatility expansion:
    1) VVIX dampening
    2) Gamma-stress proxy dampening (proxy, not true options-chain GEX)

Usage:
    python3 generate_report_v9.py
    python3 generate_report_v9.py --ticker SPY --from 2000-01-01 --to 2025-01-01
"""

import argparse
import sys
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.data import (
    fetch_daily_bars,
    fetch_vix_yfinance,
    fetch_vvix_yfinance,
    log_returns,
    vix_levels,
    vvix_levels,
)
from src.geometric import geometric_signal
from src.markov import fit_markov3
from src.ensemble import (
    ensemble_score,
    gamma_stress_proxy,
    regime_labels,
    VVIX_NEUTRAL,
)
from src.backtest import run_backtest, compute_stats

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

C = {
    "base": "#2980b9",
    "vvix": "#16a085",
    "gamma": "#8e44ad",
    "both": "#2c3e50",
    "bnh": "#7f8c8d",
    "pos": "#1a7a4a",
    "neg": "#c0392b",
    "txt": "#1a1a2e",
    "sub": "#6c757d",
    "grid": "#e6e6e6",
}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["grid"])
    ax.tick_params(colors=C["sub"], labelsize=8)
    ax.grid(axis="y", color=C["grid"], lw=0.6, zorder=0)


def _title(ax, t):
    ax.set_title(t, fontsize=10, fontweight="bold", color=C["txt"], loc="left", pad=6)


def _page_header(fig, subtitle):
    fig.text(0.5, 0.975, "Regime Ensemble  -  v9.0 Volatility Expansion",
             ha="center", va="top", fontsize=13, fontweight="bold", color=C["txt"])
    fig.text(0.5, 0.955, subtitle,
             ha="center", va="top", fontsize=9, color=C["sub"])
    fig.add_artist(plt.Line2D([0.07, 0.95], [0.945, 0.945],
                              transform=fig.transFigure, color=C["grid"], lw=1.0))


def _footer(fig, page, total, ticker, from_date, to_date):
    fig.text(
        0.5,
        0.015,
        f"Page {page} of {total}  -  {ticker} {from_date[:4]}-{to_date[:4]}  -  "
        "VVIX + gamma-proxy report",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=C["sub"],
    )


def _fetch_vix(from_date: str, to_date: str) -> pd.Series:
    try:
        vix_df = fetch_daily_bars("I:VIX", from_date, to_date)
        return vix_levels(vix_df)
    except Exception:
        return fetch_vix_yfinance(from_date, to_date)


def _fetch_vvix(from_date: str, to_date: str) -> pd.Series:
    try:
        vvix_df = fetch_daily_bars("I:VVIX", from_date, to_date)
        return vvix_levels(vvix_df)
    except Exception:
        return fetch_vvix_yfinance(from_date, to_date)


def _perf(bt: pd.DataFrame) -> dict:
    return compute_stats(bt, raw=True)["Strategy (Long Only)"]


def _to_clean_series(x, name: str, index: pd.Index = None) -> pd.Series:
    """
    Coerce x into a numeric pd.Series with an optional target index.
    Handles the common yfinance edge-case where a one-column DataFrame is returned.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            s = pd.Series(dtype=float, name=name)
        else:
            s = x.iloc[:, 0]
    elif isinstance(x, pd.Series):
        s = x
    else:
        s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = name
    if index is not None:
        s = s.reindex(index, method="ffill")
    return s


def _build_summary_table(variants: dict) -> pd.DataFrame:
    rows = []
    for name, bt in variants.items():
        p = _perf(bt)
        rows.append({
            "Variant": name,
            "CAGR": p["CAGR"] * 100,
            "Sharpe": p["Sharpe"],
            "Max DD": p["Max DD"] * 100,
            "T-stat": p["T-stat"],
            "P-value": p["P-value"],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate v9 VVIX/Gamma proxy report")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--from", dest="from_date", default="2000-01-01")
    parser.add_argument("--to", dest="to_date", default="2025-01-01")
    args = parser.parse_args()

    ticker = args.ticker
    from_date = args.from_date
    to_date = args.to_date

    print(f"Loading {ticker}...")
    bars = fetch_daily_bars(ticker, from_date, to_date)
    ret = log_returns(bars)

    print("Loading VIX/VVIX...")
    vix = _to_clean_series(_fetch_vix(from_date, to_date), "vix", index=ret.index)
    vvix = _to_clean_series(_fetch_vvix(from_date, to_date), "vvix", index=ret.index)
    vvix_available = vvix.notna().sum() >= 200
    vvix_for_model = vvix if vvix_available else None

    print("Fitting core signals...")
    geo = geometric_signal(ret, window=15)
    mom, crisis, _, _ = fit_markov3(ret, verbose=False)

    # Variants
    gp = gamma_stress_proxy(ret, vix=vix, vvix=vvix_for_model)
    score_base = ensemble_score(geo, mom, crisis)
    score_vvix = ensemble_score(geo, mom, crisis, vvix=vvix_for_model) if vvix_for_model is not None else score_base.copy()
    score_gamma = ensemble_score(geo, mom, crisis, gamma_proxy=gp)
    score_both = (
        ensemble_score(geo, mom, crisis, vvix=vvix_for_model, gamma_proxy=gp)
        if vvix_for_model is not None else score_gamma.copy()
    )

    labels_base = regime_labels(score_base)
    labels_vvix = regime_labels(score_vvix)
    labels_gamma = regime_labels(score_gamma)
    labels_both = regime_labels(score_both)

    bt_base = run_backtest(ret, labels_base, cost_bps=0)
    bt_vvix = run_backtest(ret, labels_vvix, cost_bps=0)
    bt_gamma = run_backtest(ret, labels_gamma, cost_bps=0)
    bt_both = run_backtest(ret, labels_both, cost_bps=0)

    variants_bt = {
        "Baseline": bt_base,
        "+ VVIX": bt_vvix,
        "+ Gamma Proxy": bt_gamma,
        "+ VVIX + Gamma": bt_both,
    }
    summary = _build_summary_table(variants_bt)

    bnh = compute_stats(bt_base, raw=True)["Buy & Hold"]

    # Shared aligned diagnostic frame
    diag = pd.concat(
        [
            ret.rename("ret"),
            vix.rename("vix"),
            vvix.rename("vvix"),
            gp.reindex(ret.index, method="ffill").rename("gamma_proxy"),
            crisis.reindex(ret.index).rename("crisis_prob"),
        ],
        axis=1,
    ).dropna()

    out_pdf = DOCS_DIR / f"{ticker}_v9_vol_report.pdf"
    print(f"Writing {out_pdf} ...")

    with PdfPages(out_pdf) as pdf:
        # ------------------------------------------------------------------
        # PAGE 1: WHAT WAS ADDED + SIGNAL BEHAVIOR
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        _page_header(fig, f"{ticker} {from_date} to {to_date}  |  New additions: VVIX dampener + gamma-stress proxy")
        gs = gridspec.GridSpec(3, 1, figure=fig, left=0.09, right=0.95, top=0.90, bottom=0.08, hspace=0.58)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(diag.index, diag["vix"], lw=1.0, color=C["base"], label="VIX")
        ax1.plot(diag.index, diag["vvix"], lw=1.0, color=C["vvix"], label="VVIX")
        ax1.axhline(VVIX_NEUTRAL, color=C["vvix"], ls="--", lw=0.8, alpha=0.8, label=f"VVIX neutral={VVIX_NEUTRAL:.0f}")
        _title(ax1, "VIX and VVIX Regime Context")
        ax1.set_ylabel("Index level")
        ax1.legend(fontsize=8, ncol=3, loc="upper left")
        _style(ax1)

        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.plot(diag.index, diag["gamma_proxy"], lw=1.1, color=C["gamma"], label="Gamma-stress proxy")
        ax2.fill_between(diag.index, diag["gamma_proxy"], 0, color=C["gamma"], alpha=0.15)
        ax2.axhline(0.5, color=C["sub"], ls=":", lw=0.9, label="High stress > 0.50")
        _title(ax2, "Gamma-Stress Proxy (Causal Spot/Vol Shock Composite)")
        ax2.set_ylabel("Proxy [0,1]")
        ax2.set_ylim(0, 1.02)
        ax2.legend(fontsize=8, loc="upper left")
        _style(ax2)

        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        eq_base = np.exp(bt_base["strategy_return"].cumsum())
        eq_both = np.exp(bt_both["strategy_return"].cumsum())
        eq_bnh = np.exp(bt_base["bnh_return"].cumsum())
        ax3.plot(eq_bnh.index, eq_bnh, color=C["bnh"], lw=1.0, alpha=0.8, label="Buy & Hold")
        ax3.plot(eq_base.index, eq_base, color=C["base"], lw=1.2, label="Baseline")
        ax3.plot(eq_both.index, eq_both, color=C["both"], lw=1.2, label="+ VVIX + Gamma")
        _title(ax3, "High-Level Equity Impact")
        ax3.set_ylabel("Equity (base=1)")
        ax3.legend(fontsize=8, loc="upper left")
        _style(ax3)

        fig.text(
            0.09,
            0.035,
            "Interpretation: VVIX adds a vol-of-vol brake, while gamma-proxy suppresses momentum during sharp downside + vol-jump days.\n"
            "This is intentionally a proxy-based risk overlay (not options-chain dealer GEX), designed to reduce tail exposure without changing core regime logic.\n"
            + ("VVIX data quality: sufficient for full variant testing." if vvix_available else "VVIX data quality: limited in this run; VVIX variants collapse to baseline behavior."),
            fontsize=8,
            color=C["sub"],
        )
        _footer(fig, page=1, total=3, ticker=ticker, from_date=from_date, to_date=to_date)
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------
        # PAGE 2: ABLATION PERFORMANCE + TABLE
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        _page_header(fig, "Ablation: Baseline vs VVIX/Gamma Variants (0 bps costs)")
        gs = gridspec.GridSpec(3, 2, figure=fig, left=0.09, right=0.95, top=0.88, bottom=0.08, hspace=0.55, wspace=0.30)

        ax1 = fig.add_subplot(gs[0:2, :])
        for name, bt, col in [
            ("Baseline", bt_base, C["base"]),
            ("+ VVIX", bt_vvix, C["vvix"]),
            ("+ Gamma Proxy", bt_gamma, C["gamma"]),
            ("+ VVIX + Gamma", bt_both, C["both"]),
        ]:
            eq = np.exp(bt["strategy_return"].cumsum())
            ax1.plot(eq.index, eq, lw=1.2, label=name, color=col)
        ax1.plot(eq_bnh.index, eq_bnh, lw=1.0, color=C["bnh"], ls="--", label="Buy & Hold")
        _title(ax1, "Equity Curves")
        ax1.set_ylabel("Equity (base=1)")
        ax1.legend(fontsize=8, ncol=3, loc="upper left")
        _style(ax1)

        ax2 = fig.add_subplot(gs[2, 0])
        names = summary["Variant"].tolist()
        sharpes = summary["Sharpe"].values
        bars = ax2.bar(names, sharpes, color=[C["base"], C["vvix"], C["gamma"], C["both"]])
        ax2.axhline(float(bnh["Sharpe"]), color=C["bnh"], lw=1.0, ls="--", label=f"B&H Sharpe={bnh['Sharpe']:.2f}")
        for b, s in zip(bars, sharpes):
            ax2.text(b.get_x() + b.get_width() / 2, s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
        _title(ax2, "Sharpe Comparison")
        ax2.tick_params(axis="x", rotation=18, labelsize=7)
        ax2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.26))
        _style(ax2)

        ax3 = fig.add_subplot(gs[2, 1])
        ax3.axis("off")
        table_lines = ["Performance Summary (Strategy):", ""]
        for _, r in summary.iterrows():
            table_lines.append(
                f"{r['Variant']:<15}  Sharpe {r['Sharpe']:+.2f}  CAGR {r['CAGR']:+.1f}%  MaxDD {r['Max DD']:.1f}%"
            )
        table_lines += [
            "",
            f"Buy & Hold: Sharpe {bnh['Sharpe']:+.2f}, CAGR {bnh['CAGR']*100:+.1f}%, MaxDD {bnh['Max DD']*100:.1f}%",
            "",
            "Notes:",
            "- VVIX acts as a steady stress suppressor when vol-of-vol is elevated.",
            "- Gamma proxy reacts faster to downside spot+vol shock clustering.",
            "- Combined variant is the strictest risk posture.",
        ]
        wrapped = []
        for ln in table_lines:
            if ln.startswith("- "):
                wrapped.extend(textwrap.wrap(ln, width=52, break_long_words=False, break_on_hyphens=False))
            else:
                wrapped.append(ln)
        ax3.text(0.0, 1.0, "\n".join(wrapped), va="top", ha="left", fontsize=7.6, color=C["txt"], family="monospace")

        _footer(fig, page=2, total=3, ticker=ticker, from_date=from_date, to_date=to_date)
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------------
        # PAGE 3: STRESS DIAGNOSTICS + COST SENSITIVITY
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        _page_header(fig, "Stress Diagnostics and Transaction-Cost Robustness")
        gs = gridspec.GridSpec(3, 1, figure=fig, left=0.22, right=0.95, top=0.90, bottom=0.08, hspace=0.55)

        ax1 = fig.add_subplot(gs[0, 0])
        crisis_high = (diag["crisis_prob"] > 0.5)
        vvix_high = (diag["vvix"] > VVIX_NEUTRAL)
        gp_high = (diag["gamma_proxy"] > 0.5)

        metrics = {
            "VVIX high days": 100 * vvix_high.mean(),
            "Gamma high days": 100 * gp_high.mean(),
            "Crisis days": 100 * crisis_high.mean(),
            "VVIX∩Crisis of VVIX": 100 * (vvix_high & crisis_high).sum() / max(vvix_high.sum(), 1),
            "Gamma∩Crisis of Gamma": 100 * (gp_high & crisis_high).sum() / max(gp_high.sum(), 1),
            "Gamma-only days": 100 * (gp_high & ~crisis_high).mean(),
        }

        ax1.barh(list(metrics.keys()), list(metrics.values()), color=[C["vvix"], C["gamma"], C["neg"], C["vvix"], C["gamma"], C["both"]])
        for i, v in enumerate(metrics.values()):
            ax1.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8)
        _title(ax1, "Overlap Diagnostics")
        ax1.set_xlabel("Percent")
        ax1.tick_params(axis="y", labelsize=7.2)
        _style(ax1)

        ax2 = fig.add_subplot(gs[1, 0])
        cost_bps = list(range(0, 26, 1))

        def _sharpe_for(bt_labels):
            vals = []
            for b in cost_bps:
                bt = run_backtest(ret, bt_labels, cost_bps=b)
                vals.append(float(compute_stats(bt, raw=True)["Strategy (Long Only)"]["Sharpe"]))
            return vals

        sh_base = _sharpe_for(labels_base)
        sh_vvix = _sharpe_for(labels_vvix)
        sh_gamma = _sharpe_for(labels_gamma)
        sh_both = _sharpe_for(labels_both)

        ax2.plot(cost_bps, sh_base, lw=1.2, color=C["base"], label="Baseline")
        ax2.plot(cost_bps, sh_vvix, lw=1.2, color=C["vvix"], label="+ VVIX")
        ax2.plot(cost_bps, sh_gamma, lw=1.2, color=C["gamma"], label="+ Gamma Proxy")
        ax2.plot(cost_bps, sh_both, lw=1.2, color=C["both"], label="+ VVIX + Gamma")
        ax2.axhline(float(bnh["Sharpe"]), color=C["bnh"], ls="--", lw=1.0, label="B&H Sharpe")
        _title(ax2, "Sharpe vs Transaction Cost")
        ax2.set_xlabel("Round-trip cost (bps)")
        ax2.set_ylabel("Sharpe")
        ax2.legend(fontsize=8, ncol=3, loc="upper right")
        _style(ax2)

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.axis("off")
        delta_sh_vvix = float(summary.loc[summary["Variant"] == "+ VVIX", "Sharpe"].iloc[0] - summary.loc[summary["Variant"] == "Baseline", "Sharpe"].iloc[0])
        delta_sh_gamma = float(summary.loc[summary["Variant"] == "+ Gamma Proxy", "Sharpe"].iloc[0] - summary.loc[summary["Variant"] == "Baseline", "Sharpe"].iloc[0])
        delta_sh_both = float(summary.loc[summary["Variant"] == "+ VVIX + Gamma", "Sharpe"].iloc[0] - summary.loc[summary["Variant"] == "Baseline", "Sharpe"].iloc[0])

        final_text = (
            "Implementation Summary\n"
            "\n"
            "1. VVIX dampener: multiplies Markov momentum component by a vol-of-vol stress factor.\n"
            "2. Gamma-stress proxy: causal [0,1] proxy from downside return shock + VIX jump + optional VVIX jump.\n"
            "3. Both overlays are optional and leave baseline behavior unchanged when disabled.\n"
            "\n"
            "Observed deltas vs baseline (this sample):\n"
            f"- Sharpe (+VVIX): {delta_sh_vvix:+.3f}\n"
            f"- Sharpe (+Gamma proxy): {delta_sh_gamma:+.3f}\n"
            f"- Sharpe (+VVIX+Gamma): {delta_sh_both:+.3f}\n"
            "\n"
            "Caveat: Gamma proxy is not true dealer GEX from options OI and should be treated\n"
            "as a robust stress overlay, not as exact dealer-position inference."
        )
        ax3.text(0.0, 1.0, final_text, va="top", ha="left", fontsize=8.6, color=C["txt"])

        _footer(fig, page=3, total=3, ticker=ticker, from_date=from_date, to_date=to_date)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Done. Saved report -> {out_pdf}")


if __name__ == "__main__":
    main()
