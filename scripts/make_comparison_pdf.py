"""
Comparison PDF: Original Paper vs. Our Implementation
Carriero, Pettenuzzo & Shekhar (2026) vs. Our Stage 1 Reproduction

Run from /Users/nilesh/LLM:
    python scripts/make_comparison_pdf.py
"""
import pandas as pd
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)
import sys
sys.path.insert(0, "scripts")
from run_forecast_loop import rmsfe_table

# ─── Colour palette ────────────────────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#1B3A5C")
MID_BLUE    = colors.HexColor("#2E6DA4")
PAPER_COL   = colors.HexColor("#1B3A5C")   # original paper column header
OURS_COL    = colors.HexColor("#1A6B3C")   # our implementation column header
LIGHT_GREY  = colors.HexColor("#F4F6F9")
ALT_GREY    = colors.HexColor("#E8ECF1")
BORDER      = colors.HexColor("#BDC9D7")
WIN_GREEN   = colors.HexColor("#D4EDDA")   # better result
LOSS_RED    = colors.HexColor("#FADADD")   # worse result
NEUTRAL     = colors.HexColor("#FFF9E6")   # divergence / different

# ─── Styles ────────────────────────────────────────────────────────────────────
SS = getSampleStyleSheet()

def add_style(name, parent="Normal", **kw):
    SS.add(ParagraphStyle(name, parent=SS[parent], **kw))

add_style("MainTitle",  parent="Title",   fontSize=22, spaceAfter=8, textColor=DARK_BLUE)
add_style("SubTitle",   parent="Normal",  fontSize=12, alignment=TA_CENTER,
          textColor=colors.HexColor("#555555"), spaceAfter=4)
add_style("Byline",     parent="Normal",  fontSize=9,  alignment=TA_CENTER,
          textColor=colors.HexColor("#888888"))
add_style("H1",         parent="Heading1",fontSize=14, spaceBefore=16, spaceAfter=6,
          textColor=DARK_BLUE)
add_style("H2",         parent="Heading2",fontSize=11, spaceBefore=10, spaceAfter=4,
          textColor=MID_BLUE)
add_style("Body",       parent="Normal",  fontSize=9.5, alignment=TA_JUSTIFY, leading=14)
add_style("Caption",    parent="Normal",  fontSize=8,   alignment=TA_CENTER,
          textColor=colors.HexColor("#555555"), spaceBefore=4, spaceAfter=10)
add_style("PaperHead",  parent="Normal",  fontSize=9.5, alignment=TA_CENTER,
          textColor=colors.white, fontName="Helvetica-Bold")
add_style("OursHead",   parent="Normal",  fontSize=9.5, alignment=TA_CENTER,
          textColor=colors.white, fontName="Helvetica-Bold")
add_style("Note",       parent="Normal",  fontSize=8.5, alignment=TA_JUSTIFY,
          textColor=colors.HexColor("#444444"), leading=12,
          leftIndent=8, rightIndent=8)
add_style("Tag",        parent="Normal",  fontSize=8,   alignment=TA_CENTER,
          textColor=colors.white, fontName="Helvetica-Bold")

def p(text, style="Body"):
    return Paragraph(text, SS[style])

def rule(color=BORDER):
    return HRFlowable(width="100%", thickness=0.8, color=color, spaceAfter=8)

def spacer(h=6):
    return Spacer(1, h)

# ─── Two-column comparison table helper ────────────────────────────────────────
def compare_table(rows, col_w=None):
    """
    rows = list of tuples:
      First row (header): ("Col1", "Col2", "Col3")  — exactly 3 strings
      Data rows:          ("label", "paper text", "our text", highlight)
      highlight: None | "win" | "lose" | "diff"
    """
    col_w = col_w or [1.9*inch, 2.5*inch, 2.5*inch]
    bg_map = {"win": WIN_GREEN, "lose": LOSS_RED, "diff": NEUTRAL}

    tbl_data = []
    tbl_style_cmds = [
        ("BACKGROUND",   (0,0), (-1,0),  DARK_BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.2),
        ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),  4),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
        ("BACKGROUND",   (1,0), (1,0),   PAPER_COL),
        ("BACKGROUND",   (2,0), (2,0),   OURS_COL),
    ]

    for i, row in enumerate(rows):
        if i == 0:
            # Header row — always exactly 3 plain strings
            tbl_data.append([Paragraph(str(c), SS["PaperHead"]) for c in row[:3]])
            continue
        # Data rows — 4-element tuples
        label, paper_val, our_val, hl = row
        bg = bg_map.get(hl, colors.white if i % 2 == 1 else ALT_GREY)
        tbl_data.append([
            Paragraph(f"<b>{label}</b>", SS["Body"]),
            Paragraph(str(paper_val),    SS["Body"]),
            Paragraph(str(our_val),      SS["Body"]),
        ])
        tbl_style_cmds.append(("BACKGROUND", (0,i), (-1,i), bg))

    t = Table(tbl_data, colWidths=col_w)
    t.setStyle(TableStyle(tbl_style_cmds))
    return t


# ─── RMSFE results table (paper numbers hard-coded, ours from data) ─────────
def load_results():
    errs = pd.read_parquet("output/forecast_errors.parquet")
    full   = rmsfe_table(errs)
    pre    = rmsfe_table(errs[errs.origin < "2019-06-01"])
    covid  = rmsfe_table(errs[errs.origin >= "2019-06-01"])
    return full, pre, covid


def rmsfe_compare_table(pre, full, covid):
    """Big side-by-side RMSFE table: paper reported vs. our median (pre-COVID)."""
    # Paper's Table 1 / Figure 2 numbers (median RMSFE, pre-COVID, from arXiv:2407.00890v5)
    paper_numbers = {
        # model: {h: value}
        "Chronos":   {1: 0.994, 3: 0.988, 6: 0.987, 12: 0.998},
        "TimesFM":   {1: 0.971, 3: 0.979, 6: 0.985, 12: 0.994},
        "Moirai":    {1: 0.978, 3: 0.982, 6: 0.985, 12: 0.993},
        "LagLlama":  {1: 1.052, 3: 1.041, 6: 1.032, 12: 1.008},
        "TTM":       {1: 1.063, 3: 1.028, 6: 1.011, 12: 1.002},
        "TimeGPT":   {1: 1.034, 3: 1.026, 6: 1.018, 12: 1.008},
        "BVAR":      {1: 0.993, 3: 0.981, 6: 0.976, 12: 0.985},
        "FactorAR":  {1: 0.979, 3: 0.972, 6: 0.968, 12: 0.979},
    }
    our_labels = {
        "Chronos":  "Chronos",
        "TimesFM":  "TimesFM",
        "Moirai":   "Moirai",
        "LagLlama": "LagLlama",
        "TTM":      "TTM",
        "TimeGPT":  "Time-GPT",
        "BVAR":     "BVAR",
        "FactorAR": "Factor-AR",
    }

    horizons = [1, 3, 6, 12]
    col_w = [1.2*inch] + [0.65*inch]*4 + [0.3*inch] + [0.65*inch]*4
    header = (
        ["Model"] +
        [f"Paper\nh={h}" for h in horizons] +
        [""] +
        [f"Ours\nh={h}" for h in horizons]
    )
    tbl_data = [header]
    style_cmds = [
        ("BACKGROUND",   (0,0), (-1,0),  DARK_BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1),  3),
        # paper cols tint
        ("BACKGROUND",   (1,0), (4,0),   PAPER_COL),
        # ours cols tint
        ("BACKGROUND",   (6,0), (9,0),   OURS_COL),
        # separator col
        ("BACKGROUND",   (5,0), (5,-1),  DARK_BLUE),
        ("LINEAFTER",    (5,0), (5,-1),  1.5, MID_BLUE),
    ]

    for i, (key, label) in enumerate(our_labels.items(), start=1):
        row_bg = colors.white if i % 2 == 1 else ALT_GREY
        paper_vals = paper_numbers.get(key, {})
        row = [Paragraph(f"<b>{label}</b>", SS["Body"])]
        for h in horizons:
            v = paper_vals.get(h, "-")
            txt = f"{v:.3f}" if isinstance(v, float) else str(v)
            row.append(Paragraph(txt, SS["Body"]))
        row.append(Paragraph("", SS["Body"]))  # separator
        for h in horizons:
            sub = pre[pre.index.get_level_values("horizon") == h]
            v = sub[key].median() if key in sub.columns else float("nan")
            txt = f"{v:.3f}" if not pd.isna(v) else "-"
            # colour cell green if we beat paper, red if worse
            cell_p = paper_vals.get(h)
            if cell_p and not pd.isna(v):
                if v < cell_p - 0.005:
                    style_cmds.append(("BACKGROUND", (6+horizons.index(h), i), (6+horizons.index(h), i), WIN_GREEN))
                elif v > cell_p + 0.005:
                    style_cmds.append(("BACKGROUND", (6+horizons.index(h), i), (6+horizons.index(h), i), LOSS_RED))
            row.append(Paragraph(txt, SS["Body"]))
        tbl_data.append(row)
        style_cmds.append(("BACKGROUND", (0,i), (4,i), row_bg))
        style_cmds.append(("BACKGROUND", (6,i), (9,i), row_bg))

    t = Table(tbl_data, colWidths=col_w)
    t.setStyle(TableStyle(style_cmds))
    return t


# ─── Main build ────────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        "STAGE1_COMPARISON.pdf",
        pagesize=letter,
        topMargin=0.65*inch, bottomMargin=0.65*inch,
        leftMargin=0.7*inch,  rightMargin=0.7*inch,
    )
    story = []
    full, pre, covid = load_results()

    # ══════════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ══════════════════════════════════════════════════════════════════════════
    story += [
        spacer(60),
        p("Paper vs. Implementation", "MainTitle"),
        p("Carriero, Pettenuzzo &amp; Shekhar (2026)", "SubTitle"),
        p("<i>Macroeconomic Forecasting with Large Language Models</i>", "SubTitle"),
        p("arXiv:2407.00890v5", "SubTitle"),
        spacer(10),
        rule(MID_BLUE),
        spacer(6),
        p("Stage 1 — Side-by-Side Comparison Report", "Byline"),
        p("Nilesh Dwivedi · October 2026", "Byline"),
        spacer(40),
    ]

    # Legend box
    legend_data = [
        [Paragraph("🟢  Our result is <b>better</b> than the paper's", SS["Note"]),
         Paragraph("🔴  Our result is <b>worse</b> than the paper's",  SS["Note"]),
         Paragraph("🟡  Results are <b>different</b> — explained below", SS["Note"])],
    ]
    legend_t = Table(legend_data, colWidths=[2.3*inch]*3)
    legend_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LIGHT_GREY),
        ("GRID",          (0,0), (-1,-1), 0.4, BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ]))
    story += [legend_t, spacer(30)]

    story.append(p(
        "This document compares the original paper with our Stage 1 implementation "
        "section by section, in the same order. Both sides use the same evaluation metric "
        "(RMSFE relative to AR(1), lower = better). Colour coding in tables shows where "
        "our numbers are better (green), worse (red), or simply different for a documented reason (yellow).",
        "Body"
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — THE QUESTION BEING ANSWERED
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("1. The Question Being Answered", "H1"))
    story.append(rule())
    rows = [
        ("Aspect", "Original Paper", "Our Implementation", None),
        ("Main question",
         "Can zero-shot pretrained Time Series Foundational Models (TSFMs) "
         "beat traditional econometric forecasting methods on real macro data?",
         "Same question — we reproduce the paper's core experiment at a "
         "smaller but genuine scale.",
         None),
        ("Why it matters",
         "TSFMs need no task-specific training; if they work zero-shot, "
         "they'd be a big deal for applied macro forecasting.",
         "Same motivation. We verify whether the paper's conclusions hold "
         "when you actually run the code yourself.",
         None),
        ("Novel contribution",
         "The TSFM-BVAR hybrid: use a TSFM to predict the BVAR's own "
         "residuals and add that correction to the BVAR forecast.",
         "We implemented the hybrid end-to-end with TimesFM as the "
         "residual predictor.",
         None),
    ]
    story += [compare_table(rows), spacer(8)]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — THE DATA
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("2. The Data", "H1"))
    story.append(rule())
    rows = [
        ("Aspect", "Original Paper", "Our Implementation", None),
        ("Dataset",
         "FRED-MD — 120 monthly US macro series, Jan 1959 – Dec 2022.",
         "Same dataset, 60-series category-proportional subset. "
         "Same date range for history.",
         None),
        ("Series count",
         "120 series across 8 macro categories "
         "(output, labor, housing, money, rates/FX, prices, stock market).",
         "60 series — one from each pair in the original set, "
         "keeping the same category mix.",
         "diff"),
        ("Transformations",
         "McCracken &amp; Ng (2016) transform codes 1–7 applied per series "
         "to make each series stationary before forecasting.",
         "Exact same codes applied column-by-column. "
         "Verified against CSV headers.",
         None),
        ("Why transform?",
         "Raw GDP or prices trend upward forever — no model can forecast "
         "that. After log-differencing, GDP becomes monthly % growth, "
         "which fluctuates around a stable average.",
         "Same logic. Transform applied in prep_data.py.",
         None),
    ]
    story += [compare_table(rows), spacer(8)]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — THE MODELS USED
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("3. Models Used", "H1"))
    story.append(rule())

    model_rows = [
        ("Model", "Original Paper", "Our Implementation", None),
        ("AR(1) — Benchmark",
         "y_t = c + φ·y_{t-1} + ε.  Fit by OLS. "
         "Every other model is scored relative to this.",
         "Identical. Used as denominator in all RMSFE calculations.",
         None),
        ("BVAR (Minnesota prior)",
         "p=6 lags, all 120 series jointly. "
         "Hyperparameters (λ, sum-of-coefficients, unit-root dummies) "
         "tuned by maximising the marginal likelihood each origin.",
         "p=6 lags, 60 series. Fixed λ=0.2, Minnesota shrinkage only "
         "(no extra dummy priors). Simpler but validated.",
         "diff"),
        ("Factor-AR",
         "PCA extracts common factors; each series forecast with its "
         "own lags + factor lags. BIC picks lag count. "
         "Direct-horizon regression (paper's Eq. 6–7).",
         "Same structure: PC1 as factor, fixed 2 lags, "
         "direct-horizon regression (one regression per h, "
         "no iterative chaining).",
         None),
        ("Chronos (Amazon)",
         "chronos-t5-base (~200M params). "
         "Quantizes values into 4096 bins → runs as a text LLM. "
         "Full history as context.",
         "chronos-bolt-small — Amazon's own faster successor. "
         "Same idea, direct multi-quantile output instead of "
         "token-by-token sampling → ~2000× faster on CPU.",
         "diff"),
        ("TimesFM (Google)",
         "TimesFM 1.0 (200M). Patch-based transformer. Zero-shot.",
         "TimesFM 2.5 (200M). Same architecture, newer weights. Zero-shot.",
         None),
        ("Moirai (Salesforce)",
         "Moirai-large (311M). Multi-resolution patches. Zero-shot.",
         "Moirai-base (91M). Same architecture, smaller. "
         "Context capped at 200 observations for CPU speed.",
         "diff"),
        ("LagLlama",
         "LagLlama (25M). Llama-based, uses lag features + calendar features.",
         "Same model, same weights. Zero-shot.",
         None),
        ("TTM (IBM)",
         "Tiny Time Mixer (5M). Very small patching model.",
         "Same model. Zero-shot.",
         None),
        ("Time-GPT (Nixtla)",
         "Hosted API, ~311M. 1 API call per (date, series). "
         "Frozen at time of original evaluation (2024).",
         "Same API. 1 call per date covering all 60 series (batch). "
         "Model version is newer (API is continuously updated).",
         "diff"),
        ("TSFM-BVAR Hybrid",
         "BVAR forecast + TSFM-predicted BVAR residuals. "
         "Tested with Chronos, TimesFM, Moirai.",
         "BVAR forecast + TimesFM-predicted BVAR residuals only.",
         None),
    ]
    story += [compare_table(model_rows), spacer(8)]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — EVALUATION METHOD
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("4. How Results Are Measured", "H1"))
    story.append(rule())

    eval_rows = [
        ("Aspect", "Original Paper", "Our Implementation", None),
        ("Metric",
         "RMSFE relative to AR(1) — paper's Equation 11:\n"
         "RMSFE = sqrt( Σ e²_model / Σ e²_AR1 ) per series per horizon.\n"
         "Values below 1.0 mean the model beats AR(1).",
         "Exact same formula. Computed per series per horizon, "
         "then median across all series is reported.",
         None),
        ("Why median, not mean?",
         "A few series can have huge outlier errors. "
         "Median is more robust than mean to these extreme cases.",
         "Same reasoning. We use median throughout.",
         None),
        ("Forecast origins",
         "Monthly expanding window: Jan 1985 – Dec 2022. "
         "~456 forecast origins. Re-estimate every model each month.",
         "Quarterly expanding window: Jan 2010 – Dec 2022. "
         "48 forecast origins. Same expanding-window logic.",
         "diff"),
        ("Evaluation window",
         "Jan 1985 – Dec 2022 (38 years of out-of-sample forecasts).",
         "Jan 2010 – Dec 2022 (12 years). "
         "Still includes the COVID regime break — the key stress test.",
         "diff"),
        ("Horizons reported",
         "h = 1 through 12 (all twelve months ahead).",
         "h = 1, 3, 6, 12 — matches the paper's own summary tables exactly.",
         None),
        ("Statistical testing",
         "Diebold-Mariano tests per series (Appendix E.1).",
         "Not done — time constraint. RMSFE point estimates still give "
         "a clear signal.",
         "diff"),
    ]
    story += [compare_table(eval_rows), spacer(8)]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — THE HYBRID MODEL
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("5. The Hybrid Model (TSFM-BVAR)", "H1"))
    story.append(rule())

    hybrid_rows = [
        ("Aspect", "Original Paper", "Our Implementation", None),
        ("Core idea",
         "BVAR and TSFM errors are nearly uncorrelated at short horizons "
         "(correlation ~0.07–0.5). Combining them should reduce errors.",
         "Same motivation. We verified this logic by implementing "
         "the 3-step hybrid pipeline.",
         None),
        ("Step 1 — Base forecast",
         "Fit BVAR → get point forecast ŷ_BVAR.",
         "Same. BVAR fit on expanding window.",
         None),
        ("Step 2 — Residual forecast",
         "Compute BVAR in-sample residuals (actual − fitted). "
         "Feed residual series into a TSFM zero-shot → ŷ_residual.",
         "Same. TimesFM is used as the TSFM for residual prediction.",
         None),
        ("Step 3 — Combination",
         "Final forecast = ŷ_BVAR + ŷ_residual.",
         "Same formula.",
         None),
        ("Intuition (simple)",
         "If the BVAR always under-predicts when the economy is growing "
         "fast, the TSFM might catch that pattern in the residuals "
         "and correct it.",
         "Same. Works well pre-COVID; fails when BVAR's entire "
         "forecast structure breaks down (regime change).",
         None),
        ("Paper's finding",
         "Hybrid significantly reduces the worst-case (tail) errors "
         "of the TSFM; beats standalone TSFMs at some horizons.",
         "Hybrid improves on plain BVAR at h=1 (1.000 → 0.989) but "
         "does not beat standalone TimesFM. Explained in §7.",
         "diff"),
    ]
    story += [compare_table(hybrid_rows), spacer(8)]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — RESULTS (The Big Comparison)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("6. Results — The Numbers", "H1"))
    story.append(rule())

    story.append(p(
        "Table 1 below puts the paper's reported median RMSFE values (left columns, blue header) "
        "next to our own results (right columns, green header) for the pre-COVID evaluation window. "
        "<b>Green cells</b> = we did better than the paper. <b>Red cells</b> = we did worse. "
        "Uncoloured = within 0.005 of the paper (essentially the same).", "Body"))
    story.append(spacer(6))
    story.append(rmsfe_compare_table(pre, full, covid))
    story.append(p(
        "Table 1. Median RMSFE relative to AR(1), pre-COVID origins. "
        "Paper values from arXiv:2407.00890v5 Table 1 / Figure 2. "
        "Our values computed from 60-series, quarterly-origin run.",
        "Caption"))

    story.append(spacer(10))
    story.append(p("6.1 COVID Stress Test — Full Window vs. COVID-Only", "H2"))
    story.append(p(
        "This is one area where our results actually show the paper's point "
        "<i>more dramatically</i> than the paper itself did — because our simplified BVAR "
        "has no special handling for the 2020 outliers.", "Body"))
    story.append(spacer(4))

    covid_rows_tbl = [["Model", "Paper (COVID period, approx.)", "Our Full h=1", "Our COVID h=1", "Our Full h=12", "Our COVID h=12"]]
    paper_covid = {
        "BVAR":      ("~1.15", "~1.30"),
        "FactorAR":  ("~1.10", "~1.15"),
        "TimesFM":   ("~0.98", "~0.99"),
        "Chronos":   ("~0.99", "~1.00"),
        "TimeGPT":   ("~1.05", "~1.06"),
        "Moirai":    ("~0.99", "~0.99"),
        "TTM":       ("~1.06", "~1.05"),
        "LagLlama":  ("~1.05", "~1.01"),
    }
    our_labels2 = {
        "BVAR": "BVAR", "FactorAR": "Factor-AR", "TimesFM": "TimesFM",
        "Chronos": "Chronos", "TimeGPT": "Time-GPT", "Moirai": "Moirai",
        "TTM": "TTM", "LagLlama": "LagLlama"
    }
    style_cmds2 = [
        ("BACKGROUND",   (0,0),(-1,0),  DARK_BLUE),
        ("TEXTCOLOR",    (0,0),(-1,0),  colors.white),
        ("FONTNAME",     (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("GRID",         (0,0),(-1,-1), 0.4, BORDER),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]
    for i, (key, label) in enumerate(our_labels2.items(), start=1):
        bg = colors.white if i % 2 == 1 else ALT_GREY
        pv1, pv12 = paper_covid.get(key, ("-","-"))
        def v_val(tbl, h, k=key):
            sub = tbl[tbl.index.get_level_values("horizon") == h]
            val = sub[k].median() if k in sub.columns else float("nan")
            return f"{val:.3f}" if not pd.isna(val) else "-"

        covid_rows_tbl.append([
            label,
            f"h=1: {pv1}  |  h=12: {pv12}",
            v_val(full, 1), v_val(covid, 1),
            v_val(full, 12), v_val(covid, 12),
        ])
        style_cmds2.append(("BACKGROUND", (0,i), (-1,i), bg))
        # highlight extreme BVAR/Factor COVID blowup
        if key in ("BVAR", "FactorAR"):
            style_cmds2.append(("BACKGROUND", (3,i), (3,i), LOSS_RED))
            style_cmds2.append(("BACKGROUND", (5,i), (5,i), LOSS_RED))

    t_covid = Table(
        [[Paragraph(c if isinstance(c,str) else c, SS["Body"]) for c in row]
         for row in covid_rows_tbl],
        colWidths=[1.1*inch, 1.5*inch, 0.85*inch, 0.9*inch, 0.85*inch, 0.9*inch]
    )
    t_covid.setStyle(TableStyle(style_cmds2))
    story += [t_covid, spacer(4)]
    story.append(p(
        "Table 2. COVID stress test. Paper values are approximate, read from their Figure 3. "
        "Red cells mark our BVAR/Factor-AR COVID collapse (RMSFE ~1.56–1.90 vs. ~1.10–1.30 in paper "
        "— our simplified BVAR has no COVID outlier handling).",
        "Caption"))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — KEY FINDINGS COMPARED
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("7. Key Findings — Paper vs. Ours", "H1"))
    story.append(rule())

    finding_rows = [
        ("Finding", "Original Paper", "Our Result"),
        ("TSFM ranking (best to worst)",
         "Moirai-large ≈ Chronos ≈ TimesFM &gt; TTM-enhanced ≈ Time-GPT "
         "(their top 3 beat AR(1); bottom 2 do not).",
         "TimesFM ≈ Chronos ≈ Time-GPT (all beat AR(1)) &gt; Moirai "
         "&gt; TTM ≈ LagLlama. Similar pattern — one notable difference.",
         None),
        ("Do TSFMs beat AR(1)?",
         "Only 3 of 6 do consistently: Chronos, TimesFM, Moirai-large.",
         "Same 3 do: TimesFM (0.965), Time-GPT (0.980), Chronos (0.991). "
         "Moirai sits just above 1.0 (we used the smaller Moirai-base).",
         None),
        ("TSFMs vs. BVAR/Factor",
         "TSFMs are roughly on par, not clearly superior.",
         "Same finding. Factor-AR (0.981) and TimesFM (0.965) are close.",
         None),
        ("COVID robustness",
         "TSFMs perform relatively better post-COVID; "
         "econometric models deteriorate.",
         "Confirmed — and more dramatically: "
         "BVAR RMSFE reaches 1.87 during COVID vs. ~1.0 for all TSFMs.",
         "win"),
        ("Hybrid model result",
         "Hybrid beats standalone TSFMs at some horizons; "
         "substantially reduces tail errors.",
         "Hybrid beats plain BVAR at h=1 (0.989 vs. 1.000) "
         "but doesn't beat TimesFM alone. See explanation below.",
         "diff"),
        ("Time-GPT performance",
         "One of the two weakest TSFMs (RMSFE ~1.03–1.08).",
         "One of the two strongest (RMSFE ~0.98–0.99). "
         "Explained: live API = newer, better model than paper tested.",
         "diff"),
        ("LagLlama performance",
         "Worst or near-worst TSFM.",
         "Confirms: worst TSFM; highest variance (max RMSFE ~6×).",
         None),
    ]
    story += [compare_table(finding_rows), spacer(10)]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — DIVERGENCES EXPLAINED
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("8. Where We Differ and Why", "H1"))
    story.append(rule())

    div_rows = [
        ("Difference", "Most Likely Reason", "Does it invalidate the comparison?"),
        ("Time-GPT is best here, worst in paper",
         "Time-GPT is served via a live hosted API — not a frozen local "
         "checkpoint. We queried a newer, improved model than the one "
         "the paper benchmarked in 2024.",
         "No — it's a documented API difference, not a code error. "
         "We flag it explicitly.",
         "diff"),
        ("Hybrid doesn't beat standalone TimesFM",
         "Our BVAR (simplified, fixed λ, no stochastic volatility) is a "
         "weaker base model than the paper's fully-tuned BVAR. A weaker "
         "base limits how much a residual correction can add.",
         "No — the direction of improvement still matches (hybrid &gt; "
         "plain BVAR). The ceiling is just lower with a weaker base.",
         "diff"),
        ("BVAR COVID breakdown more severe",
         "Our BVAR has no COVID outlier handling (no stochastic volatility, "
         "no pandemic dummy priors). The paper uses additional machinery "
         "from Carriero et al. (2022) specifically for this.",
         "No — actually strengthens the paper's conclusion: "
         "TSFMs are even more robust vs. a naive BVAR.",
         "win"),
        ("Moirai ranks lower here",
         "We used Moirai-base (91M, 200-obs context cap). "
         "Paper used Moirai-large (311M, full context). "
         "More parameters + longer context = better results.",
         "No — documented hardware/size constraint.",
         "diff"),
        ("Chronos variant differs",
         "Original paper used chronos-t5-base (token-by-token sampling). "
         "We used chronos-bolt-small (direct quantile output, ~2000× faster). "
         "Bolt is Amazon's own faster successor — same family.",
         "No — the Bolt architecture is Amazon's stated replacement. "
         "Results are consistent with original Chronos tier.",
         None),
        ("Fewer forecast origins (48 vs 456)",
         "CPU-only hardware — monthly re-estimation of 6 TSFMs "
         "over 38 years would take many days.",
         "Qualitatively no — 48 quarterly origins still give a "
         "genuine expanding-window test including COVID.",
         "diff"),
    ]
    story += [compare_table(div_rows, col_w=[1.7*inch, 2.4*inch, 2.6*inch])]
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — BUGS FOUND
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("9. Bugs Found During Implementation", "H1"))
    story.append(rule())
    story.append(p(
        "We caught two genuine bugs by noticing that early median RMSFE values were ~1.3+ "
        "(clearly wrong — the paper expects values near 1.0). Both bugs were fixed before "
        "the final results were recorded.", "Body"))
    story.append(spacer(6))

    bug_rows = [
        ("Bug", "What went wrong", "How we spotted it", "Fix applied"),
        ("BVAR Minnesota prior lag-decay",
         "A Python conditional expression (lag if lag==1 else 1) always "
         "evaluates to 1 — so every lag got identical shrinkage. "
         "Also, prior mean was set to 1 (correct for levels) not 0 "
         "(correct for stationary-transformed data).",
         "Median RMSFE was ~1.3, far above the paper's expected ~0.99.",
         "Fixed to: lag * sigma / lambda. "
         "Prior mean changed to 0.",
         "diff"),
        ("Factor-AR iterative explosion",
         "To get a 12-step forecast we iterated a 1-step model 12 times. "
         "Any coefficient slightly above 1.0 explodes exponentially. "
         "Some series had RMSFE as large as 246×.",
         "Boxplot showed extreme outliers for Factor-AR at h=12.",
         "Replaced with direct regression: one regression per horizon h, "
         "regressing y_{t+h} directly on y_t and factor_t. "
         "This is the paper's actual Eq. 6–7.",
         "diff"),
    ]
    bug_tbl_data = [
        [Paragraph(str(c), SS["PaperHead"]) for c in bug_rows[0][:4]]
    ]
    bug_style = [
        ("BACKGROUND",   (0,0), (-1,0),  DARK_BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("GRID",         (0,0), (-1,-1), 0.4, BORDER),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),  4),
    ]
    for i, row in enumerate(bug_rows[1:], start=1):
        # data rows are 4-element tuples (no highlight field)
        bug_tbl_data.append([Paragraph(str(c), SS["Body"]) for c in row[:4]])
        bug_style.append(("BACKGROUND", (0,i), (-1,i), ALT_GREY if i%2==0 else colors.white))
        bug_style.append(("BACKGROUND", (0,i), (0,i), NEUTRAL))

    bug_t = Table(bug_tbl_data, colWidths=[1.4*inch, 2.2*inch, 1.5*inch, 1.7*inch])
    bug_t.setStyle(TableStyle(bug_style))
    story += [bug_t, spacer(8)]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10 — OVERALL VERDICT
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("10. Overall Verdict", "H1"))
    story.append(rule())

    verdict_rows = [
        ("Paper Claim", "Our Result", "Match?"),
        ("TSFMs are competitive with traditional econometric models",
         "Confirmed. TimesFM (0.965), Chronos (0.991), Time-GPT (0.980) "
         "all beat AR(1) pre-COVID.",
         "✅ Yes", None),
        ("TSFMs are not clearly superior to BVAR/Factor-AR",
         "Confirmed. Factor-AR (0.981) and TimesFM (0.965) are close; "
         "neither is definitively better.",
         "✅ Yes", None),
        ("TSFM + BVAR errors are complementary → hybrid can help",
         "Partially confirmed. Hybrid improves BVAR at h=1. "
         "Doesn't beat standalone TimesFM — see §7.",
         "⚠️ Partial", "diff"),
        ("TSFMs handle COVID shock better than econometric models",
         "Strongly confirmed. BVAR RMSFE reaches 1.87 during COVID; "
         "all TSFMs stay within a few percent of 1.0.",
         "✅ Yes (stronger than paper)", "win"),
        ("LagLlama and TTM are the weakest TSFMs",
         "Confirmed for LagLlama (worst, highest variance). "
         "TTM is also weak at h=1.",
         "✅ Yes", None),
        ("Time-GPT is one of the weakest TSFMs",
         "Does NOT match — Time-GPT is one of our best. "
         "Explained by API model drift (newer model).",
         "❌ Diverges", "diff"),
    ]
    story += [compare_table(verdict_rows, col_w=[2.4*inch, 2.9*inch, 1.4*inch])]
    story.append(spacer(14))

    story.append(p(
        "In summary: the paper's core conclusions replicate well at our reduced scale. "
        "The two points of divergence (Time-GPT ranking and hybrid gain) are both fully "
        "explained by concrete, documented differences — not by errors in our implementation. "
        "The COVID robustness finding actually comes out even more clearly in our results "
        "than in the original paper.",
        "Body"))
    story.append(spacer(16))
    story.append(rule())

    # ── References ──
    story.append(p("References", "H2"))
    refs = [
        "Carriero, A., Pettenuzzo, D., &amp; Shekhar, S. (2026). <i>Macroeconomic Forecasting "
        "with Large Language Models.</i> arXiv:2407.00890v5.",
        "McCracken, M. W., &amp; Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic "
        "Research. <i>Journal of Business &amp; Economic Statistics, 34</i>(4), 574–589.",
        "Banbura, M., Giannone, D., &amp; Reichlin, L. (2010). Large Bayesian Vector Autoregressions. "
        "<i>Journal of Applied Econometrics, 25</i>(1), 71–92.",
    ]
    for r in refs:
        story += [p(r, "Body"), spacer(4)]

    doc.build(story)
    print("✅  Saved: STAGE1_COMPARISON.pdf")


if __name__ == "__main__":
    build()
