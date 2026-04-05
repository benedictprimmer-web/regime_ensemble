#!/usr/bin/env python3
"""
Paul Refinement Report v2
=========================
Documents the ablation results and signal attribution analysis from
running `python3 run.py --skip-bic --ablation`.

Answers Paul's core question: Is Markov adding directional value, or is it
purely a risk-state filter? The answer: it is primarily a risk-state filter.
The geometric signal carries the directional edge.

Usage:
    python3 generate_report_paul_refinements_v2.py

Output:
    docs/Paul_refinement_pdf_V2.pdf
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)
OUTPUT   = DOCS_DIR / "Paul_refinement_pdf_V2.pdf"

C = {
    "momentum":  "#1a7a4a",
    "crisis":    "#c0392b",
    "mixed":     "#7f8c8d",
    "blue":      "#2980b9",
    "dark":      "#2c3e50",
    "bg":        "#f8f9fa",
    "border":    "#dee2e6",
    "orange":    "#e67e22",
    "amber":     "#f39c12",
    "light_red": "#fadbd8",
    "light_grn": "#d5f5e3",
    "light_blu": "#d6eaf8",
    "light_org": "#fdebd0",
}

TITLE_FONT  = {"fontsize": 16, "fontweight": "bold", "color": C["dark"]}
HEAD_FONT   = {"fontsize": 11, "fontweight": "bold", "color": C["dark"]}
SUB_FONT    = {"fontsize": 9,  "fontweight": "bold", "color": C["dark"]}
BODY_FONT   = {"fontsize": 8.5, "color": "#2c3e50"}
MONO_FONT   = {"fontsize": 7.5, "color": "#2c3e50", "fontfamily": "monospace"}
LABEL_FONT  = {"fontsize": 8,  "color": "#7f8c8d", "style": "italic"}
WARN_FONT   = {"fontsize": 8,  "color": C["crisis"], "fontweight": "bold"}
OK_FONT     = {"fontsize": 8,  "color": C["momentum"], "fontweight": "bold"}


def _box(ax, x, y, w, h, color, alpha=0.12, radius=0.01):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor="none", alpha=alpha,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)


def _hline(ax, y, color=C["border"], lw=0.8):
    ax.plot([0, 1], [y, y], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=False)


def _footer(ax, page, total=3):
    ax.text(0.5, 0.012, f"Regime Ensemble — Paul Refinement v2  ·  Page {page} of {total}",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=7, color="#aaaaaa")


# ── Actual ablation numbers (from python3 run.py --skip-bic --ablation) ──────
ABLATION = [
    # (variant_name, markov_desc, cagr, sharpe, max_dd, pct_mom, pct_mix)
    ("geo_only",       "(none — geometric only)",        "+5.7%", "0.47", "-25.3%", "30%", "40%"),
    ("crisis_filter",  "1 - P(crisis)",                  "+7.1%", "0.66", "-21.9%", "55%", "28%"),
    ("full_ensemble",  "P(momentum), zeroed on crisis",  "+4.8%", "0.65", "-14.6%", "30%", "22%"),
]
SHARPE_DELTA_FULL_VS_CRISIS = -0.010
SHARPE_DELTA_CRISIS_VS_GEO  = +0.190

# ── Attribution grid (actual numbers from Section 6b output) ─────────────────
# Format: (geo_label, mom_bin, mean_pct, t_stat, p_sig, n)
# p_sig: "*" = p<0.05, "~" = p<0.10, "" = not significant
GRID = [
    # geo=reversion row
    ("reversion", "low\n(<0.33)",     "+0.054%", "+1.37", "",  1140),
    ("reversion", "mid\n(0.33-0.67)", "+0.073%", "+0.48", "",    38),
    ("reversion", "high\n(>0.67)",    "+0.032%", "+0.92", "",   391),
    # geo=mixed row
    ("mixed",     "low\n(<0.33)",     "-0.051%", "-1.37", "",  1401),
    ("mixed",     "mid\n(0.33-0.67)", "+0.031%", "+0.19", "",    23),
    ("mixed",     "high\n(>0.67)",    "+0.020%", "+0.85", "",   705),
    # geo=momentum row
    ("momentum",  "low\n(<0.33)",     "+0.203%", "+3.32", "*",  720),
    ("momentum",  "mid\n(0.33-0.67)", "-0.157%", "-1.06", "",    17),
    ("momentum",  "high\n(>0.67)",    "+0.007%", "+0.36", "",   865),
]


def make_page1(pdf):
    """Ablation results — actual numbers + decision."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    _box(ax, 0.03, 0.91, 0.94, 0.07, C["dark"], alpha=0.08)
    ax.text(0.5, 0.965, "Regime Ensemble — Paul Refinement v2",
            ha="center", va="center", transform=ax.transAxes, **TITLE_FONT)
    ax.text(0.5, 0.926, "Ablation Results + Signal Attribution  (SPY 2000–2025, 0 bps, 1-day lag)",
            ha="center", va="center", transform=ax.transAxes, **LABEL_FONT)

    # Paul's question
    y = 0.88
    ax.text(0.05, y, "Paul's Question", va="top", **HEAD_FONT)
    y -= 0.018
    _box(ax, 0.03, y - 0.060, 0.94, 0.063, C["amber"], alpha=0.10)
    ax.text(0.06, y - 0.010,
            "'Is the Markov model really adding directional information, or is it mostly identifying",
            va="top", **BODY_FONT)
    y -= 0.016
    ax.text(0.06, y - 0.010,
            " don't be aggressive here periods?' — The --ablation flag now answers this directly.",
            va="top", **BODY_FONT)
    y -= 0.048

    # Ablation table
    _hline(ax, y + 0.005)
    ax.text(0.05, y, "Ablation Results  (python3 run.py --skip-bic --ablation)", va="top", **HEAD_FONT)
    y -= 0.025

    # Table header
    _box(ax, 0.03, y - 0.190, 0.94, 0.200, C["dark"], alpha=0.04)
    col_x = [0.05, 0.25, 0.42, 0.52, 0.62, 0.73, 0.83]
    headers = ["Variant", "Markov component", "CAGR", "Sharpe", "Max DD", "%Mom", "%Mix"]
    for h, x in zip(headers, col_x):
        ax.text(x, y - 0.008, h, va="top", fontsize=8, fontweight="bold", color=C["dark"])
    y -= 0.022
    ax.plot([0.04, 0.96], [y + 0.002, y + 0.002], color=C["border"], linewidth=0.6,
            transform=ax.transAxes, clip_on=False)

    row_colors = [C["mixed"], C["orange"], C["blue"]]
    for i, (variant, markov_desc, cagr, sharpe, max_dd, pct_mom, pct_mix) in enumerate(ABLATION):
        is_default = (variant == "full_ensemble")
        row_alpha  = 0.12 if is_default else 0.06
        _box(ax, 0.03, y - 0.038, 0.94, 0.040, row_colors[i], alpha=row_alpha)
        suffix = "  <- current default" if is_default else ""
        ax.text(col_x[0], y - 0.010, variant + suffix, va="top",
                fontsize=8, fontfamily="monospace",
                fontweight="bold" if is_default else "normal", color=C["dark"])
        ax.text(col_x[1], y - 0.010, markov_desc, va="top", **BODY_FONT)
        ax.text(col_x[2], y - 0.010, cagr,    va="top", **BODY_FONT)
        ax.text(col_x[3], y - 0.010, sharpe,  va="top", **BODY_FONT)
        ax.text(col_x[4], y - 0.010, max_dd,  va="top", **BODY_FONT)
        ax.text(col_x[5], y - 0.010, pct_mom, va="top", **BODY_FONT)
        ax.text(col_x[6], y - 0.010, pct_mix, va="top", **BODY_FONT)
        y -= 0.045

    y -= 0.010

    # Sharpe deltas
    ax.text(0.06, y,
            f"Sharpe delta (full_ensemble vs crisis_filter):  {SHARPE_DELTA_FULL_VS_CRISIS:+.3f}",
            va="top", **MONO_FONT)
    y -= 0.015
    ax.text(0.06, y,
            f"Sharpe delta (crisis_filter  vs geo_only):     {SHARPE_DELTA_CRISIS_VS_GEO:+.3f}",
            va="top", **MONO_FONT)
    y -= 0.030

    # Interpretation
    _hline(ax, y + 0.005)
    ax.text(0.05, y, "Interpretation", va="top", **HEAD_FONT)
    y -= 0.022

    _box(ax, 0.03, y - 0.145, 0.94, 0.148, C["blue"], alpha=0.06)
    interp_lines = [
        ("1.", "Geometric signal does the directional work.  The +0.19 Sharpe lift from geo_only → crisis_filter"),
        ("",   "   shows that adding a Markov risk filter (any Markov filter) is the main gain."),
        ("2.", "P(momentum) does NOT add additional directional value.  Sharpe delta full vs crisis_filter = −0.010"),
        ("",   "   (below the 0.05 threshold). This confirms Paul's hypothesis: the Markov edge is the risk filter."),
        ("3.", "full_ensemble has substantially lower drawdown: −14.6% vs −21.9% for crisis_filter.  The reason:"),
        ("",   "   full_ensemble routes only 30% of days to momentum vs 55% for crisis_filter. 1−P(crisis) is high"),
        ("",   "   most of the time (crisis periods are rare), so crisis_filter is a less conservative strategy."),
        ("4.", "Decision: keep full_ensemble as default.  The Sharpe difference is negligible (0.010), but the"),
        ("",   "   Max DD advantage (−14.6% vs −21.9%) is material and consistent with the stated objective of"),
        ("",   "   risk reduction. P(momentum) = crisis_prob; it contributes through risk control, not direction."),
    ]
    for bullet, line in interp_lines:
        bx = 0.06 if bullet else 0.08
        ax.text(bx, y - 0.010, bullet, va="top", fontsize=8.5, fontweight="bold", color=C["blue"])
        ax.text(0.09, y - 0.010, line,  va="top", **BODY_FONT)
        y -= 0.015

    _footer(ax, 1)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_page2(pdf):
    """Signal attribution grid — visual 3x3 table + interpretation."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    _box(ax, 0.03, 0.94, 0.94, 0.04, C["dark"], alpha=0.06)
    ax.text(0.5, 0.962, "Signal Attribution Grid  (geo x Markov forward return)",
            ha="center", va="center", transform=ax.transAxes, **TITLE_FONT)

    y = 0.915
    ax.text(0.05, y, "What the grid measures", va="top", **HEAD_FONT)
    y -= 0.018
    desc_lines = [
        "Each cell shows the mean next-day forward return for days where (geometric signal level) AND (Markov P(momentum)",
        "bin) both hold simultaneously. This isolates which combination of signals drives returns — and reveals whether",
        "Markov's contribution is directional (top-right high-confidence cells) or risk-avoidance (left column suppression).",
    ]
    for line in desc_lines:
        ax.text(0.06, y, line, va="top", **BODY_FONT)
        y -= 0.015
    y -= 0.010

    # Grid legend
    ax.text(0.05, y, "Legend:", va="top", fontsize=8, fontweight="bold", color=C["dark"])
    ax.text(0.14, y, "Cell content:  mean fwd return (%/day) | t-stat | significance | N days", va="top", **BODY_FONT)
    y -= 0.015
    ax.text(0.14, y, "[*] = p < 0.05 (significant)     [~] = p < 0.10     (blank) = not significant", va="top", **BODY_FONT)
    y -= 0.025

    # Draw 3x3 grid
    geo_rows   = ["reversion", "mixed", "momentum"]
    mom_cols   = ["low\n(<0.33)", "mid\n(0.33-0.67)", "high\n(>0.67)"]
    col_labels = ["Markov P(mom)\nlow (<0.33)", "Markov P(mom)\nmid (0.33-0.67)", "Markov P(mom)\nhigh (>0.67)"]

    grid_top = y
    cell_w   = 0.28
    cell_h   = 0.105
    margin_l = 0.14
    col_x    = [margin_l + i * (cell_w + 0.01) for i in range(3)]

    # Column headers
    for i, (col_l, cx) in enumerate(zip(col_labels, col_x)):
        _box(ax, cx, grid_top - 0.035, cell_w, 0.038, C["blue"], alpha=0.10)
        for j, part in enumerate(col_l.split("\n")):
            ax.text(cx + cell_w / 2, grid_top - 0.008 - j * 0.013, part,
                    ha="center", va="top", fontsize=7.5, fontweight="bold", color=C["dark"])
    y = grid_top - 0.038

    # Row label column header
    ax.text(0.02, grid_top - 0.015, "Geo\nsignal",
            ha="center", va="top", fontsize=7.5, fontweight="bold", color=C["dark"])

    # Grid cells
    geo_colors = {"reversion": C["crisis"], "mixed": C["mixed"], "momentum": C["momentum"]}
    for row_i, geo in enumerate(geo_rows):
        row_y = y - row_i * (cell_h + 0.008)
        # Row label
        _box(ax, 0.01, row_y - cell_h, 0.10, cell_h, geo_colors[geo], alpha=0.15)
        ax.text(0.06, row_y - cell_h / 2, geo,
                ha="center", va="center", fontsize=8, fontweight="bold", color=geo_colors[geo])

        for col_i, col_l in enumerate(mom_cols):
            cx   = col_x[col_i]
            cell = next(r for r in GRID if r[0] == geo and r[1] == col_l)
            mean_pct, t_stat, sig, n = cell[2], cell[3], cell[4], cell[5]

            # Cell colour: green if mean positive + sig, red if negative, neutral otherwise
            if sig == "*":
                cell_color = C["momentum"]; cell_alpha = 0.18
            elif mean_pct.startswith("-"):
                cell_color = C["crisis"]; cell_alpha = 0.08
            else:
                cell_color = C["dark"]; cell_alpha = 0.04

            _box(ax, cx, row_y - cell_h, cell_w, cell_h, cell_color, alpha=cell_alpha)

            sig_str = f"  [{sig}]" if sig else ""
            ax.text(cx + cell_w / 2, row_y - 0.018,
                    f"{mean_pct}/day{sig_str}",
                    ha="center", va="top", fontsize=8.5,
                    fontweight="bold" if sig == "*" else "normal",
                    color=geo_colors[geo] if sig == "*" else C["dark"])
            ax.text(cx + cell_w / 2, row_y - 0.035,
                    f"t = {t_stat}",
                    ha="center", va="top", **BODY_FONT)
            ax.text(cx + cell_w / 2, row_y - 0.050,
                    f"n = {n:,}",
                    ha="center", va="top", fontsize=7.5, color=C["mixed"])

    y = y - 3 * (cell_h + 0.008) - 0.020

    # Key finding callout
    _hline(ax, y + 0.005)
    ax.text(0.05, y, "Key Finding", va="top", **HEAD_FONT)
    y -= 0.022

    _box(ax, 0.03, y - 0.148, 0.94, 0.150, C["momentum"], alpha=0.07)
    finding_lines = [
        ("Only one cell is statistically significant:",
         "geo=momentum, markov=low  (mean +0.203%/day, t=+3.32, p<0.05, n=720)"),
        ("", ""),
        ("What this means — geometric does direction, Markov does risk-state:",
         ""),
        ("  a.", "When the geometric signal says momentum (straight-line price), returns are strongly positive"),
        ("     ", "    (+0.20%/day) EVEN WHEN Markov P(momentum) is low. The directional edge belongs to geo."),
        ("  b.", "When geo=momentum AND Markov P(momentum) is high (n=865), the mean return is near zero"),
        ("     ", "    (+0.007%/day, t=0.36). P(momentum) adds no extra directional information."),
        ("  c.", "The left column (markov=low) pattern shows Markov's real role: routing reversion+low-markov"),
        ("     ", "    days to cash. Mixed+low days have slightly negative returns — correctly suppressed."),
        ("  d.", "Conclusion: P(momentum) functions as a conservative risk-state filter, not a direction"),
        ("     ", "    predictor. The full_ensemble default is kept to preserve the Max DD advantage (−14.6%)."),
    ]
    for label, line in finding_lines:
        if not label and not line:
            y -= 0.005
            continue
        bx = 0.06
        lx = 0.06 if not label.startswith("  ") else 0.09
        if label and not label.startswith("  "):
            ax.text(bx, y - 0.010, label, va="top", fontsize=8.5, fontweight="bold", color=C["momentum"])
            if line:
                ax.text(0.06, y - 0.023, line, va="top", **MONO_FONT)
                y -= 0.015
        else:
            ax.text(lx, y - 0.010, label, va="top", fontsize=8, fontweight="bold", color=C["dark"])
            ax.text(lx + 0.04 if label.strip() else 0.10, y - 0.010, line, va="top", **BODY_FONT)
        y -= 0.015

    _footer(ax, 2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_page3(pdf):
    """Summary: all Paul refinements, current state, next steps."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    _box(ax, 0.03, 0.94, 0.94, 0.04, C["dark"], alpha=0.06)
    ax.text(0.5, 0.962, "Complete Paul Refinement Summary  (v1 + v2)",
            ha="center", va="center", transform=ax.transAxes, **TITLE_FONT)

    y = 0.915
    ax.text(0.05, y, "All Changes Made in Response to Paul's Critique", va="top", **HEAD_FONT)
    y -= 0.022

    changes = [
        ("1", C["blue"],
         "In-sample transparency (v1)",
         "Added explicit blockquote to README results table: numbers are in-sample.",
         "Points to --expanding for honest OOS estimate (optimism bias ~0.03 Sharpe)."),
        ("2", C["momentum"],
         "Markov observation vector upgraded (v1)",
         "Replaced AR(1)/statsmodels with 5-feature GaussianHMM/hmmlearn.",
         "States now represent market structure: direction, vol, drawdown, trend position."),
        ("3", C["orange"],
         "Mechanical state labelling (v1)",
         "_label_states() sorts all three states by mean ret_20d at every refit.",
         "Prevents label swaps across walk-forward folds; consistent semantics always."),
        ("4", C["amber"],
         "Ablation diagnostic (v1)",
         "--ablation flag: compares geo_only / crisis_filter / full_ensemble.",
         "Built the infrastructure to answer Paul's directional-value question."),
        ("5", C["blue"],
         "Ablation results interpreted (v2)",
         "Ran ablation: Sharpe delta full vs crisis_filter = -0.010 (< 0.05 threshold).",
         "Decision: keep full_ensemble for Max DD advantage (-14.6% vs -21.9%)."),
        ("6", C["momentum"],
         "Signal attribution grid (v2)",
         "New attribution_grid() in backtest.py; printed in Section 6b of run.py.",
         "3x3 geo x Markov table reveals where each signal's edge comes from."),
    ]
    for num, color, title, line1, line2 in changes:
        _box(ax, 0.03, y - 0.060, 0.94, 0.062, color, alpha=0.07)
        ax.text(0.06, y - 0.012, f"Change {num}:  {title}", va="top", **SUB_FONT)
        ax.text(0.07, y - 0.028, line1, va="top", **BODY_FONT)
        ax.text(0.07, y - 0.043, line2, va="top", **BODY_FONT)
        y -= 0.070

    # Current state
    _hline(ax, y + 0.005)
    ax.text(0.05, y, "Current Model State", va="top", **HEAD_FONT)
    y -= 0.022

    _box(ax, 0.03, y - 0.095, 0.94, 0.097, C["dark"], alpha=0.04)
    state_rows = [
        ("Signal 1",      "Geometric straightness ratio, 15-day window, adaptive 30th/70th percentile thresholds"),
        ("Signal 2",      "5-feature GaussianHMM k=3 (ret_20d, ret_5d, rvol_20d, drawdown, dist_200d)"),
        ("Ensemble",      "mean(geo, markov_adj) where markov_adj = P(momentum), zeroed when P(crisis) > 0.50"),
        ("Ensemble role", "Geometric drives direction. Markov is confirmed as a risk-state filter, not direction."),
        ("State labels",  "Mechanically assigned at every refit by mean ret_20d — no label swaps across folds"),
        ("Causal filter", "Manual alpha-recursion (forward only). No backward smoothing. No look-ahead bias."),
    ]
    for key, val in state_rows:
        ax.text(0.06, y - 0.010, key + ":", va="top", fontsize=8, fontweight="bold", color=C["dark"])
        ax.text(0.22, y - 0.010, val, va="top", **BODY_FONT)
        y -= 0.016
    y -= 0.012

    # Files changed
    _hline(ax, y + 0.005)
    ax.text(0.05, y, "Files Changed (v1 + v2)", va="top", **HEAD_FONT)
    y -= 0.020

    file_changes = [
        ("src/markov.py",    "v1: Complete rewrite. AR(1)/statsmodels -> GaussianHMM/hmmlearn."),
        ("src/backtest.py",  "v2: New attribution_grid() function (3x3 geo x markov forward-return table)."),
        ("src/ensemble.py",  "v1: _build_markov_component() + mode param. Backward compatible."),
        ("run.py",           "v1+v2: --ablation flag, run_ablation(), plot_ablation_curves(), Section 6b grid."),
        ("README.md",        "v1: In-sample transparency note. Signal 2 description updated to 5-feature HMM."),
        ("requirements.txt", "v1: Added hmmlearn>=0.3"),
        ("generate_report_paul_refinements_v1.py", "v1: 3-page PDF documenting the three code changes."),
        ("generate_report_paul_refinements_v2.py", "v2: This document — ablation results + attribution grid."),
    ]
    _box(ax, 0.03, y - len(file_changes) * 0.016 - 0.008, 0.94,
         len(file_changes) * 0.016 + 0.010, C["dark"], alpha=0.04)
    for fname, desc in file_changes:
        ax.text(0.06, y, fname, va="top", fontsize=8, fontfamily="monospace",
                fontweight="bold", color=C["blue"])
        ax.text(0.38, y, desc, va="top", **BODY_FONT)
        y -= 0.016

    _footer(ax, 3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Generating {OUTPUT} ...")
    with PdfPages(OUTPUT) as pdf:
        make_page1(pdf)
        make_page2(pdf)
        make_page3(pdf)

        d = pdf.infodict()
        d["Title"]   = "Regime Ensemble — Paul Refinement v2"
        d["Author"]  = "Ben Rimmer"
        d["Subject"] = "Ablation results + signal attribution grid"

    print(f"Saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
