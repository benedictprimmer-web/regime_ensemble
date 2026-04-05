#!/usr/bin/env python3
"""
Paul Refinement Report v1
=========================
Documents the three code changes made in response to Paul's external critique
of the regime ensemble model. Produces a 3-page PDF.

Usage:
    python3 generate_report_paul_refinements_v1.py

Output:
    docs/Paul_refinement_pdf_V1.pdf
"""

from pathlib import Path
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)
OUTPUT   = DOCS_DIR / "Paul_refinement_pdf_V1.pdf"

# ── Colour palette (matches the main report style) ────────────────────────────
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
    ax.plot([0, 1], [y, y], color=color, linewidth=lw, transform=ax.transAxes, clip_on=False)


def _footer(ax, page, total=3):
    ax.text(0.5, 0.012, f"Regime Ensemble — Paul Refinement v1  ·  Page {page} of {total}",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=7, color="#aaaaaa")


def make_page1(pdf):
    """Overview + Change 1: in-sample transparency."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Header ────────────────────────────────────────────────────────
    _box(ax, 0.03, 0.91, 0.94, 0.07, C["dark"], alpha=0.08)
    ax.text(0.5, 0.965, "Regime Ensemble — Paul Refinement v1",
            ha="center", va="center", transform=ax.transAxes, **TITLE_FONT)
    ax.text(0.5, 0.926, "Three targeted improvements in response to external quant critique",
            ha="center", va="center", transform=ax.transAxes, **LABEL_FONT)

    # ── Critique summary box ───────────────────────────────────────────
    y = 0.88
    ax.text(0.05, y, "Critique Summary  (Paul, via James)", va="top", **HEAD_FONT)
    y -= 0.025
    _box(ax, 0.03, y - 0.095, 0.94, 0.10, C["amber"], alpha=0.10)
    critique_lines = [
        "1.  Research design leakage — thresholds and Markov model fit on full dataset; reported Sharpe is at least partly in-sample.",
        "2.  AR(1) on daily returns too narrow — for equities, daily return autocorrelation ≈ 0.01–0.03 (near noise). The model",
        "    infers regime almost entirely from variance clustering, not directional persistence.",
        "3.  State labelling not enforced mechanically — comparisons across time periods are shaky without a consistent mapping rule.",
        "4.  No ablation to show whether Markov adds directional value or is simply a crisis-avoidance filter.",
    ]
    for i, line in enumerate(critique_lines):
        ax.text(0.06, y - 0.012 - i * 0.018, line, va="top", **BODY_FONT)

    # ── Changes overview ───────────────────────────────────────────────
    y = 0.755
    ax.text(0.05, y, "Three Changes Made", va="top", **HEAD_FONT)
    y -= 0.020
    for num, title, desc, color in [
        ("1", "In-sample transparency (README + docs)",
         "Added explicit note to headline results table: numbers are in-sample. Points to --expanding for honest OOS estimate.", C["blue"]),
        ("2", "Markov observation vector upgraded",
         "Replaced AR(1) on daily returns with a 5-feature multivariate GaussianHMM. States now represent market structure\n"
         "directly: medium/short-term direction, volatility level, drawdown depth, and structural trend position.", C["momentum"]),
        ("3", "Mechanical state labelling + ablation diagnostic",
         "States labelled at every refit by sorting mean(ret_20d) — no post-hoc eyeballing. New --ablation flag compares\n"
         "three ensemble variants to answer: is Markov adding directional value or purely risk-state filtering?", C["orange"]),
    ]:
        _box(ax, 0.03, y - 0.075, 0.94, 0.078, color, alpha=0.08)
        ax.text(0.06, y - 0.012, f"Change {num}:  {title}", va="top", **SUB_FONT)
        for j, dline in enumerate(desc.split("\n")):
            ax.text(0.07, y - 0.030 - j * 0.017, dline, va="top", **BODY_FONT)
        y -= 0.095

    # ── Change 1 detail ───────────────────────────────────────────────
    y = 0.445
    _hline(ax, y + 0.01)
    ax.text(0.05, y, "Change 1 — In-Sample Transparency", va="top", **HEAD_FONT)
    y -= 0.022

    ax.text(0.06, y, "Problem:", va="top", **WARN_FONT)
    ax.text(0.14, y,
            "The Markov model and geometric percentile thresholds are fitted on the full 2000–2025 dataset.",
            va="top", **BODY_FONT)
    y -= 0.018
    ax.text(0.14, y,
            "Headline results (Sharpe 0.68, Max DD −16.7%) are therefore partly in-sample.",
            va="top", **BODY_FONT)
    y -= 0.025

    ax.text(0.06, y, "Fix:", va="top", **OK_FONT)
    ax.text(0.14, y,
            "Added a blockquote directly under the results table in README.md:",
            va="top", **BODY_FONT)
    y -= 0.022

    _box(ax, 0.05, y - 0.052, 0.90, 0.055, C["amber"], alpha=0.10)
    readme_text = (
        "> These numbers are in-sample — the Markov model and geometric thresholds are\n"
        "> fitted on the full 2000–2025 dataset. For a honest estimate of live performance,\n"
        "> run python3 run.py --expanding --skip-bic (annual refit, fully OOS).\n"
        "> Observed optimism bias: ~0.03 Sharpe points (per v6 analysis)."
    )
    for j, line in enumerate(readme_text.split("\n")):
        ax.text(0.07, y - 0.010 - j * 0.013, line, va="top", **MONO_FONT)
    y -= 0.065

    ax.text(0.06, y,
            "The walk-forward (--walkforward) and expanding-window (--expanding) infrastructure already existed;",
            va="top", **BODY_FONT)
    y -= 0.015
    ax.text(0.06, y,
            "this change ensures results are framed honestly in the main README rather than discovered only in flags.",
            va="top", **BODY_FONT)

    _footer(ax, 1)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_page2(pdf):
    """Change 2: Markov observation vector."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Header ────────────────────────────────────────────────────────
    _box(ax, 0.03, 0.94, 0.94, 0.04, C["dark"], alpha=0.06)
    ax.text(0.5, 0.962, "Change 2 — Markov Observation Vector Upgraded",
            ha="center", va="center", transform=ax.transAxes, **TITLE_FONT)

    # ── Problem ───────────────────────────────────────────────────────
    y = 0.915
    ax.text(0.05, y, "Problem — AR(1) on daily returns", va="top", **WARN_FONT)
    y -= 0.022
    problem_lines = [
        "For SPY, daily return autocorrelation ≈ 0.01–0.03 (near noise). AR(1) asks 'are returns serially correlated",
        "one step ahead?' — a question that contains almost no information at daily frequency for large-cap equities.",
        "In practice, the three hidden states the model learned were almost entirely variance clusters:",
        "   MOMENTUM ≈ low variance period   |   CHOPPY ≈ moderate variance   |   CRISIS ≈ high variance",
        "The AR(1) coefficient contributed almost nothing beyond the switching variance — a simpler GARCH-style",
        "model would have done the same work with fewer parameters and no hidden-state framing.",
    ]
    for line in problem_lines:
        ax.text(0.06, y, line, va="top", **BODY_FONT)
        y -= 0.015
    y -= 0.005

    # ── Fix ───────────────────────────────────────────────────────────
    ax.text(0.05, y, "Fix — 5-feature multivariate GaussianHMM", va="top", **OK_FONT)
    y -= 0.022
    ax.text(0.06, y,
            "Replaced statsmodels MarkovAutoregression with hmmlearn GaussianHMM (diag covariance, k=3).",
            va="top", **BODY_FONT)
    y -= 0.015
    ax.text(0.06, y,
            "The hidden states now explain a multivariate observation vector that directly measures market structure:",
            va="top", **BODY_FONT)
    y -= 0.025

    # Feature table
    _box(ax, 0.05, y - 0.115, 0.90, 0.118, C["blue"], alpha=0.06)
    col_x = [0.07, 0.22, 0.60]
    headers = ["Feature", "Window", "What it captures"]
    for i, (hdr, x) in enumerate(zip(headers, col_x)):
        ax.text(x, y - 0.008, hdr, va="top", fontsize=8, fontweight="bold", color=C["dark"])
    y -= 0.022
    features = [
        ("ret_20d",   "20-day cumulative return",   "Medium-term direction — the primary state-labelling signal"),
        ("ret_5d",    "5-day  cumulative return",   "Short-term momentum — catches fast reversals"),
        ("rvol_20d",  "20-day realised vol (ann.)", "Volatility level / stress state"),
        ("drawdown",  "Distance from 252d high",    "Loss depth from recent peak"),
        ("dist_200d", "Price / 200d MA  − 1",       "Structural trend position (above/below long-run trend)"),
    ]
    for feat, window, desc in features:
        ax.text(col_x[0], y, feat,   va="top", **MONO_FONT)
        ax.text(col_x[1], y, window, va="top", **BODY_FONT)
        ax.text(col_x[2], y, desc,   va="top", **BODY_FONT)
        y -= 0.018
    y -= 0.010

    ax.text(0.06, y,
            "Features are z-scored using training-data statistics before fitting — no test-data leakage into the scaler.",
            va="top", **BODY_FONT)
    y -= 0.020

    # ── Key design details ────────────────────────────────────────────
    ax.text(0.05, y, "Key design details", va="top", **SUB_FONT)
    y -= 0.022
    design_rows = [
        ("Library",          "hmmlearn.hmm.GaussianHMM  (replaces statsmodels MarkovAutoregression)"),
        ("Covariance",       "Diagonal — stable with 5 features; avoids near-singular full covariance in short folds"),
        ("Random restarts",  "5 independent starts, best log-likelihood selected — eliminates degenerate EM solutions"),
        ("EM iterations",    "200 per restart (same as previous AR(1) setting)"),
        ("Filtered probs",   "Manual forward pass (α-recursion only). hmmlearn's predict_proba uses forward-backward"),
        ("",                 "(smoothed — look-ahead bias). Causal filtering is implemented separately in _forward_filter()"),
        ("Walk-forward",     "fit_and_filter_markov(train, test): HMM + scaler fitted on train only; last 250 days of"),
        ("",                 "train prepended to test for rolling-window warm-up. No EM on test data."),
    ]
    _box(ax, 0.05, y - len(design_rows) * 0.017 - 0.010, 0.90,
         len(design_rows) * 0.017 + 0.012, C["dark"], alpha=0.04)
    for key, val in design_rows:
        if key:
            ax.text(0.07, y,    key + ":", va="top", fontsize=8, fontweight="bold", color=C["dark"])
        ax.text(0.225, y, val, va="top", **BODY_FONT)
        y -= 0.017
    y -= 0.015

    # ── Before/After code comparison ───────────────────────────────────
    ax.text(0.05, y, "Before / After — core observation", va="top", **SUB_FONT)
    y -= 0.022

    before_code = [
        "# BEFORE  (AR(1) on daily log returns)",
        "model = sm.tsa.MarkovAutoregression(",
        "    returns, k_regimes=3, order=1,",
        "    switching_ar=True,",
        "    switching_variance=True,",
        ")",
    ]
    after_code = [
        "# AFTER  (5-feature GaussianHMM)",
        "features = DataFrame({",
        "    'ret_20d':  returns.rolling(20).sum(),",
        "    'ret_5d':   returns.rolling(5).sum(),",
        "    'rvol_20d': returns.rolling(20).std() * sqrt(252),",
        "    'drawdown': price / price.rolling(252).max() - 1,",
        "    'dist_200d':price / price.rolling(200).mean() - 1,",
        "})",
        "model = GaussianHMM(n_components=3,",
        "    covariance_type='diag', n_iter=200)",
        "model.fit(scale(features))",
    ]
    n_rows = max(len(before_code), len(after_code))
    row_h  = 0.014
    box_h  = n_rows * row_h + 0.020

    _box(ax, 0.03, y - box_h, 0.455, box_h, C["crisis"],    alpha=0.06)
    _box(ax, 0.51, y - box_h, 0.455, box_h, C["momentum"],  alpha=0.06)

    for j, line in enumerate(before_code):
        ax.text(0.04, y - 0.010 - j * row_h, line, va="top", **MONO_FONT)
    for j, line in enumerate(after_code):
        ax.text(0.52, y - 0.010 - j * row_h, line, va="top", **MONO_FONT)

    _footer(ax, 2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_page3(pdf):
    """Change 3: Mechanical state labelling + Ablation diagnostic."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Header ────────────────────────────────────────────────────────
    _box(ax, 0.03, 0.94, 0.94, 0.04, C["dark"], alpha=0.06)
    ax.text(0.5, 0.962,
            "Change 3 — Mechanical State Labelling + Ablation Diagnostic",
            ha="center", va="center", transform=ax.transAxes, **TITLE_FONT)

    # ── State labelling ───────────────────────────────────────────────
    y = 0.915
    ax.text(0.05, y, "3a — Mechanical state labelling at every refit", va="top", **HEAD_FONT)
    y -= 0.022

    ax.text(0.06, y, "Problem:", va="top", **WARN_FONT)
    ax.text(0.14, y,
            "Without an explicit mapping rule, hidden states can swap labels across walk-forward refits.",
            va="top", **BODY_FONT)
    y -= 0.015
    ax.text(0.14, y,
            "Calling the state with highest mean 'momentum' in one fold and a different state in the next makes",
            va="top", **BODY_FONT)
    y -= 0.015
    ax.text(0.14, y,
            "cross-fold comparisons meaningless — the 'momentum' label may refer to different behaviour each time.",
            va="top", **BODY_FONT)
    y -= 0.022

    ax.text(0.06, y, "Fix:", va="top", **OK_FONT)
    ax.text(0.14, y, "_label_states() sorts all three states by their mean ret_20d at every call:", va="top", **BODY_FONT)
    y -= 0.022

    _box(ax, 0.05, y - 0.065, 0.90, 0.068, C["momentum"], alpha=0.06)
    label_code = [
        "def _label_states(model):",
        "    means_20d  = model.means_[:, 0]         # mean ret_20d per state",
        "    order      = np.argsort(means_20d)       # ascending: crisis → choppy → momentum",
        "    crisis_idx = int(order[0])",
        "    choppy_idx = int(order[1])",
        "    mom_idx    = int(order[2])",
        "    return mom_idx, crisis_idx, choppy_idx   # same semantics at every refit",
    ]
    for j, line in enumerate(label_code):
        ax.text(0.06, y - 0.010 - j * 0.012, line, va="top", **MONO_FONT)
    y -= 0.100

    ax.text(0.06, y,
            "This applies in fit_markov3() (full-sample), fit_and_filter_markov() (walk-forward), and",
            va="top", **BODY_FONT)
    y -= 0.014
    ax.text(0.06, y,
            "expanding_backtest() — state semantics are consistent across all contexts.",
            va="top", **BODY_FONT)
    y -= 0.030

    # ── Ablation ─────────────────────────────────────────────────────
    _hline(ax, y + 0.008)
    ax.text(0.05, y, "3b — Ablation diagnostic  (--ablation flag)", va="top", **HEAD_FONT)
    y -= 0.022

    ax.text(0.06, y,
            "Paul's core question: 'Is the Markov model really adding directional information, or is it mostly",
            va="top", **BODY_FONT)
    y -= 0.015
    ax.text(0.06, y,
            "identifying don't be aggressive here periods?'  A new --ablation flag answers this directly.",
            va="top", **BODY_FONT)
    y -= 0.025

    ax.text(0.06, y, "Three ensemble variants — only the Markov component differs:", va="top", **SUB_FONT)
    y -= 0.020

    _box(ax, 0.05, y - 0.098, 0.90, 0.100, C["dark"], alpha=0.04)
    variants = [
        ("geo_only",      "(none — geometric only)            ",
         "score = geo_signal                                  Baseline: no Markov at all"),
        ("crisis_filter", "1 − P(crisis)                      ",
         "score = mean(geo, 1 − crisis_prob)                  Markov as pure risk-state filter"),
        ("full_ensemble", "P(momentum), zeroed on crisis       ",
         "score = mean(geo, mom_prob × [crisis<0.5])          Current default"),
    ]
    ax.text(0.07, y - 0.008, "Variant", va="top", fontsize=8, fontweight="bold", color=C["dark"])
    ax.text(0.22, y - 0.008, "Markov component", va="top", fontsize=8, fontweight="bold", color=C["dark"])
    ax.text(0.55, y - 0.008, "Formula / Rationale", va="top", fontsize=8, fontweight="bold", color=C["dark"])
    y -= 0.022
    for name, component, formula in variants:
        ax.text(0.07, y, name,      va="top", fontsize=8, fontfamily="monospace", color=C["dark"])
        ax.text(0.22, y, component, va="top", **BODY_FONT)
        ax.text(0.55, y, formula,   va="top", **MONO_FONT)
        y -= 0.025
    y -= 0.010

    ax.text(0.06, y,
            "If |Sharpe(full_ensemble) − Sharpe(crisis_filter)| < 0.05:",
            va="top", **BODY_FONT)
    y -= 0.015
    ax.text(0.08, y,
            "→  P(momentum) adds minimal directional value. The Markov edge is the risk-state filter.",
            va="top", **BODY_FONT)
    y -= 0.015
    ax.text(0.08, y,
            "→  Switch default to mode='crisis_filter' (simpler, more defensible).",
            va="top", **BODY_FONT)
    y -= 0.025

    ax.text(0.06, y,
            "If delta ≥ 0.05:  P(momentum) IS contributing real directional value. Full ensemble justified.",
            va="top", **BODY_FONT)
    y -= 0.030

    # ── Sample output box ─────────────────────────────────────────────
    ax.text(0.05, y, "Sample ablation output  (python3 run.py --skip-bic --ablation)", va="top", **SUB_FONT)
    y -= 0.022

    _box(ax, 0.05, y - 0.110, 0.90, 0.112, C["dark"], alpha=0.05)
    sample_output = [
        "  Variant          Markov component                    CAGR   Sharpe   Max DD   %Mom   %Mix",
        "  ────────────────────────────────────────────────────────────────────────────────────────────",
        "  geo_only          (none — geometric only)            +X.X%    X.XX   -XX.X%    XX%    XX%",
        "  crisis_filter     1 - P(crisis)                      +X.X%    X.XX   -XX.X%    XX%    XX%",
        "  full_ensemble     P(momentum), zeroed on crisis      +X.X%    X.XX   -XX.X%    XX%    XX%",
        "",
        "  Sharpe delta (full_ensemble vs crisis_filter):  +X.XX",
        "  Sharpe delta (crisis_filter  vs geo_only):      +X.XX",
        "",
        "  Interpretation:",
        "    [auto-generated based on delta threshold of 0.05]",
    ]
    for j, line in enumerate(sample_output):
        ax.text(0.06, y - 0.010 - j * 0.010, line, va="top", **MONO_FONT)
    y -= 0.125

    # ── Files changed summary ─────────────────────────────────────────
    _hline(ax, y + 0.005)
    ax.text(0.05, y, "Files changed", va="top", **HEAD_FONT)
    y -= 0.020

    file_changes = [
        ("src/markov.py",    "Complete rewrite. AR(1)/statsmodels → GaussianHMM/hmmlearn. _build_features(),"),
        ("",                 "_forward_filter(), _label_states() helpers. Same public API preserved."),
        ("src/ensemble.py",  "_build_markov_component() helper + mode='full'|'crisis_filter'|'geo_only' param."),
        ("",                 "Backward compatible: calling with no mode= returns bit-for-bit identical output."),
        ("run.py",           "--ablation flag, run_ablation(), plot_ablation_curves(). Section headers updated."),
        ("README.md",        "In-sample note added. Signal 2 description updated to 5-feature HMM."),
        ("requirements.txt", "Added hmmlearn>=0.3"),
    ]
    _box(ax, 0.05, y - len(file_changes) * 0.016 - 0.008, 0.90,
         len(file_changes) * 0.016 + 0.010, C["dark"], alpha=0.04)
    for fname, desc in file_changes:
        if fname:
            ax.text(0.07, y, fname, va="top", fontsize=8, fontfamily="monospace",
                    fontweight="bold", color=C["blue"])
        ax.text(0.26, y, desc, va="top", **BODY_FONT)
        y -= 0.016

    _footer(ax, 3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Generating {OUTPUT} ...")
    with PdfPages(OUTPUT) as pdf:
        make_page3(pdf)   # page 1 of 3
        make_page1(pdf)   # page 2 of 3
        make_page2(pdf)   # page 3 of 3

        d = pdf.infodict()
        d["Title"]   = "Regime Ensemble — Paul Refinement v1"
        d["Author"]  = "Ben Rimmer"
        d["Subject"] = "Three code improvements in response to external quant critique"

    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
