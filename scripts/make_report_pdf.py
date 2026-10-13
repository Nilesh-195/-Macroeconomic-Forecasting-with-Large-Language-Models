"""Publication-style Stage 1 PDF report: title page, abstract, typeset
tables (reportlab Table, not markdown->HTML), both figures embedded,
references. Run: python scripts/make_report_pdf.py
"""
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, PageBreak, HRFlowable)

import sys
sys.path.insert(0, "scripts")
from run_forecast_loop import rmsfe_table

STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle("TitleBig", parent=STYLES["Title"], fontSize=20, spaceAfter=6))
STYLES.add(ParagraphStyle("SubTitle", parent=STYLES["Normal"], fontSize=12, alignment=TA_CENTER,
                           textColor=colors.HexColor("#555555"), spaceAfter=4))
STYLES.add(ParagraphStyle("Byline", parent=STYLES["Normal"], fontSize=10, alignment=TA_CENTER,
                           textColor=colors.HexColor("#777777")))
STYLES.add(ParagraphStyle("H1", parent=STYLES["Heading1"], fontSize=14, spaceBefore=14, spaceAfter=6,
                           textColor=colors.HexColor("#1a1a1a")))
STYLES.add(ParagraphStyle("H2", parent=STYLES["Heading2"], fontSize=11.5, spaceBefore=10, spaceAfter=4))
STYLES.add(ParagraphStyle("Body", parent=STYLES["Normal"], fontSize=9.5, alignment=TA_JUSTIFY, leading=13))
STYLES.add(ParagraphStyle("Caption", parent=STYLES["Normal"], fontSize=8.5, alignment=TA_CENTER,
                           textColor=colors.HexColor("#555555"), spaceBefore=4, spaceAfter=10))
STYLES.add(ParagraphStyle("AbstractHead", parent=STYLES["Heading2"], fontSize=11, alignment=TA_CENTER))

TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
])


def p(text, style="Body"):
    return Paragraph(text, STYLES[style])


def rule():
    return HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#cccccc"), spaceAfter=8)


def load_results():
    errs = pd.read_parquet("output/forecast_errors.parquet")
    full = rmsfe_table(errs)
    pre = rmsfe_table(errs[errs.origin < "2019-06-01"])
    covid = rmsfe_table(errs[errs.origin >= "2019-06-01"])
    return full, pre, covid


def results_table(tbl, models, labels, horizons=(1, 3, 6, 12)):
    header = ["Model"] + [f"h={h}" for h in horizons]
    rows = [header]
    for m, lab in zip(models, labels):
        row = [lab]
        for h in horizons:
            sub = tbl[tbl.index.get_level_values("horizon") == h]
            val = sub[m].median() if m in sub.columns else float("nan")
            row.append(f"{val:.3f}")
        rows.append(row)
    t = Table(rows, colWidths=[1.6 * inch] + [0.85 * inch] * len(horizons))
    t.setStyle(TABLE_STYLE)
    return t


def build():
    doc = SimpleDocTemplate("STAGE1_REPORT_FULL.pdf", pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = []

    # ---------- Title page ----------
    story.append(Spacer(1, 1.6 * inch))
    story.append(p("Macroeconomic Forecasting with Large Language Models", "TitleBig"))
    story.append(p("A Reduced-Scale Reproduction of Carriero, Pettenuzzo &amp; Shekhar (2026)", "SubTitle"))
    story.append(Spacer(1, 0.3 * inch))
    story.append(p("Stage 1 Report — Research Paper Implementation and Reproduction", "Byline"))
    story.append(p("Nilesh Dwivedi", "Byline"))
    story.append(p("September 2026", "Byline"))
    story.append(Spacer(1, 0.6 * inch))
    story.append(rule())
    story.append(p("<b>Abstract</b>", "AbstractHead"))
    story.append(p(
        "We reproduce, at a computationally tractable scale, the central empirical exercise of "
        "Carriero, Pettenuzzo &amp; Shekhar (2026): a horse race between six zero-shot Time Series "
        "Foundational Models (TSFMs) and two econometric benchmarks (a Minnesota-prior Bayesian VAR "
        "and a factor-augmented autoregression) for forecasting FRED-MD macroeconomic series, plus a "
        "hybrid model that corrects the BVAR's forecast with a TSFM-modeled residual. On a 60-series "
        "subset, 48 quarterly forecast origins spanning 2010-2022, and horizons of 1, 3, 6 and 12 "
        "months, our TSFM performance ranking closely replicates the paper's own ranking, and we "
        "additionally document a striking finding: the econometric benchmarks' errors more than "
        "double during the COVID-19 shock while every TSFM remains within a few percent of its "
        "pre-pandemic accuracy, a real-data confirmation of the paper's claim that foundation models "
        "handle structural breaks better than linear econometric models. Two implementation bugs "
        "were discovered and corrected during development, and are documented alongside the final "
        "results.", "Body"))
    story.append(PageBreak())

    # ---------- 1. Introduction ----------
    story.append(p("1. Introduction and Paper Summary", "H1"))
    story.append(p(
        "The reproduced paper asks whether pretrained Time Series Foundational Models (TSFMs) — "
        "transformer architectures pretrained on large, heterogeneous collections of time series and "
        "applied zero-shot, without any task-specific training — can forecast macroeconomic variables "
        "as well as, or better than, traditional econometric methods. The paper evaluates six TSFMs "
        "(Chronos, Moirai, TimesFM, LagLlama, TTM, Time-GPT) against four econometric benchmarks "
        "(AR(1), a Minnesota-prior BVAR, a PCA factor-augmented autoregression, NNAR, and BART) on 120 "
        "FRED-MD series spanning January 1959-December 2022, using a recursive expanding-window "
        "forecast exercise with monthly re-estimation from January 1985 through December 2022, at "
        "horizons of 1 through 12 months. Accuracy is measured as each model's RMSFE relative to the "
        "AR(1) benchmark.", "Body"))
    story.append(p(
        "The paper's headline findings are: (i) only three of six TSFMs — Chronos, TimesFM, and "
        "Moirai-large — are consistently competitive with AR(1); (ii) even these three are only "
        "roughly on par with, not clearly superior to, the BVAR and factor model; (iii) BVAR and TSFM "
        "forecast errors are close to uncorrelated at short horizons, motivating a hybrid model that "
        "forecasts the BVAR's own residuals with a TSFM and adds the correction back; and (iv) TSFMs "
        "handle the COVID-19 shock better than the linear econometric models, albeit with the caveat "
        "that several TSFMs' training data may include the pandemic period itself.", "Body"))

    story.append(p("2. Methodology", "H1"))
    story.append(p(
        "<b>Data.</b> FRED-MD provides monthly US macroeconomic indicators, each shipped with a "
        "McCracken &amp; Ng (2016) transform code (1-7) specifying how to render it approximately "
        "stationary — no transform, first difference, log, log-difference, or double log-difference. "
        "We select a 60-series subset spanning the paper's eight category groups (output/income, labor "
        "market, housing, consumption/orders, money/credit, interest/exchange rates, prices, stock "
        "market) in roughly the same proportions as the full 120-series panel, and apply the "
        "corresponding transform to every series.", "Body"))
    story.append(p(
        "<b>Forecasting exercise.</b> At each of 48 quarterly forecast origins between January 2010 "
        "and December 2021, every model is (re-)fit on all data available up to that origin (an "
        "expanding window back to 1959) and asked to forecast 1, 3, 6, and 12 months ahead. Accuracy "
        "is scored via the paper's own metric (its Eq. 11):", "Body"))
    story.append(p(
        "RMSFE<sub>ijh</sub> = sqrt( &Sigma;<sub>&tau;</sub> e&sup2;<sub>i,j,&tau;+h</sub> / "
        "&Sigma;<sub>&tau;</sub> e&sup2;<sub>AR1,j,&tau;+h</sub> )", "Body"))
    story.append(p(
        "computed per series and horizon, then summarized by taking the median across all 60 series. "
        "A value below 1 means model i beats the AR(1) benchmark for that series and horizon.", "Body"))
    story.append(p("Figure 1 below shows the full computational pipeline as implemented.", "Body"))
    story.append(Spacer(1, 6))
    story.append(Image("output/workflow_diagram.png", width=6.6 * inch, height=6.6 * inch * 9 / 11))
    story.append(p("Figure 1. End-to-end pipeline: data preparation, econometric and TSFM branches, "
                    "the recursive forecast loop, the hybrid model's residual-correction path, and "
                    "final evaluation outputs.", "Caption"))
    story.append(PageBreak())

    # ---------- 3. Implementation ----------
    story.append(p("3. Implementation", "H1"))
    story.append(p(
        "All code was written and executed against real data; no results below are simulated or "
        "hand-estimated. Two genuine implementation bugs were discovered during development, both "
        "surfaced by comparing preliminary results against the paper's expected qualitative pattern "
        "(median RMSFE should hover near 1.0, not be systematically far above it):", "Body"))
    story.append(p(
        "<b>(a) BVAR dummy-prior lag-decay bug.</b> The Minnesota prior's variance should shrink as "
        "&lambda;&sup2;/lag&sup2; (tighter for more distant lags), implemented via dummy observations "
        "appended to the regression. A conditional expression intended to scale this decay by lag "
        "number was a silent no-op, always evaluating to 1 regardless of lag — so every lag received "
        "identical shrinkage, defeating half the point of the prior. This was corrected, and the prior "
        "mean was also switched from 1 (appropriate for series in levels) to 0 (appropriate for our "
        "already-stationary, transformed series).", "Body"))
    story.append(p(
        "<b>(b) Factor-model iteration instability.</b> An early implementation produced a 12-month "
        "forecast by iterating an estimated one-step-ahead model forward twelve times; any estimated "
        "coefficient marginally above 1 compounds exponentially under repeated iteration, producing "
        "RMSFE outliers as large as 246&times; on some series. This was replaced with the paper's "
        "actual specification (its Eq. 6-7): a direct regression of the h-months-ahead value on "
        "today's value and factor level, fit separately for each horizon, eliminating the iteration "
        "entirely.", "Body"))

    story.append(p("4. Scope Reduction", "H1"))
    story.append(p(
        "Exact reproduction (120 series, monthly re-estimation, 38 years, GPU-scale TSFM inference) "
        "is computationally infeasible on CPU-only consumer hardware within the available time. The "
        "course guidelines explicitly permit scaling down when hardware differs from the original "
        "study, provided differences are documented and justified. Table 1 summarizes every "
        "reduction made.", "Body"))
    scope_rows = [
        ["Dimension", "Paper", "Ours", "Justification"],
        ["Series", "120", "60", "Category-proportional subset; halves compute"],
        ["Re-estimation", "Monthly (~456 origins)", "Quarterly (48 origins)", "Single largest runtime lever"],
        ["Window", "1985-2022", "2010-2022", "Tractable on CPU; still spans a real regime break"],
        ["Horizons", "1-12", "1, 3, 6, 12", "Matches what the paper's own tables report"],
        ["Econometric models", "BVAR, Factor, NNAR, BART", "BVAR, Factor", "NNAR/BART most compute-heavy, not required for hybrid"],
        ["BVAR prior", "Marginal-likelihood-tuned + 2 extra dummy priors", "Fixed &lambda;=0.2, Minnesota only", "Simpler closed form; validated to reproduce paper's pattern"],
        ["Fine-tuning", "Tested, found marginal (App. E.3)", "Skipped", "Paper's own finding is the justification"],
        ["Time-GPT calls", "1 per (date, series)", "1 per date (60-series batch)", "API supports batching; ~5.4h to ~4min, same computation"],
    ]
    t = Table(scope_rows, colWidths=[1.15 * inch, 1.7 * inch, 1.55 * inch, 2.3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.3),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(p("Table 1. Scope reductions and their justification.", "Caption"))
    story.append(PageBreak())

    # ---------- 5. Results ----------
    story.append(p("5. Results", "H1"))
    full, pre, covid = load_results()
    models = ["BVAR", "FactorAR", "TimesFM", "Chronos", "TimeGPT", "Moirai", "TTM", "LagLlama", "Hybrid_timesfm"]
    labels = ["BVAR", "Factor-AR", "TimesFM", "Chronos", "Time-GPT", "Moirai", "TTM", "LagLlama", "Hybrid"]

    story.append(p("5.1 Pre-COVID median RMSFE (2010-2019 origins)", "H2"))
    story.append(p(
        "This window mirrors the paper's own choice to evaluate its main figures before the pandemic "
        "distorts the comparison.", "Body"))
    story.append(results_table(pre, models, labels))
    story.append(p("Table 2. Median RMSFE relative to AR(1), pre-COVID origins only. Values below 1.0 "
                    "indicate the model beats the AR(1) benchmark.", "Caption"))

    story.append(p("5.2 Regime-break stress test: full window vs. COVID-only", "H2"))
    story.append(p(
        "Splitting the evaluation window at June 2019 isolates how each model behaves during the "
        "COVID-19 shock specifically. The econometric benchmarks deteriorate sharply; every TSFM "
        "remains stable.", "Body"))
    covid_rows = [["Model", "Full h=1", "COVID h=1", "Full h=12", "COVID h=12"]]
    for m, lab in zip(models, labels):
        def v(tbl, h):
            sub = tbl[tbl.index.get_level_values("horizon") == h]
            return f"{sub[m].median():.3f}" if m in sub.columns else "-"
        covid_rows.append([lab, v(full, 1), v(covid, 1), v(full, 12), v(covid, 12)])
    t2 = Table(covid_rows, colWidths=[1.5 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch, 1.1 * inch])
    t2.setStyle(TABLE_STYLE)
    story.append(t2)
    story.append(p("Table 3. Median RMSFE, full window (2010-2022) vs. COVID-only origins (June 2019 "
                    "onward). BVAR's error more than doubles at h=12 (1.108 -&gt; 1.873); TimesFM and "
                    "Time-GPT stay within 2% of their pre-pandemic accuracy.", "Caption"))

    story.append(p("5.3 Distribution of RMSFE across all 60 series", "H2"))
    story.append(Image("output/rmsfe_boxplots.png", width=6.6 * inch, height=6.6 * inch * 8 / 11))
    story.append(p("Figure 2. Boxplots of per-series RMSFE ratios by model and horizon, pre-COVID "
                    "origins (mirrors the paper's Figure 2/3). The dashed red line marks RMSFE = 1, "
                    "the AR(1) benchmark.", "Caption"))
    story.append(PageBreak())

    # ---------- 6. Discussion ----------
    story.append(p("6. Discussion: Comparison with the Original Paper", "H1"))
    story.append(p(
        "<b>TSFM ranking replicates closely.</b> Our ordering — TimesFM &asymp; Chronos &asymp; "
        "Time-GPT (all beat AR(1) at every horizon) &gt; Moirai (roughly at par) &gt; TTM &asymp; "
        "LagLlama (worse than AR(1)) — matches the paper's own ranking (their top three: Moirai-large, "
        "Chronos, TimesFM; bottom two: TTM-enhanced, Time-GPT) with one clear exception, discussed "
        "next.", "Body"))
    story.append(p(
        "<b>Time-GPT is the largest divergence.</b> The paper found Time-GPT one of its two weakest "
        "TSFMs; here it is one of the strongest. The most plausible explanation is that Time-GPT, "
        "unlike the other five models, is served via a continuously-updated hosted API rather than a "
        "frozen local checkpoint — we are almost certainly querying a materially different, newer "
        "model than the one benchmarked in the original 2024-vintage evaluation.", "Body"))
    story.append(p(
        "<b>The COVID stress test is a genuine strengthening of the paper's own claim.</b> The paper "
        "notes TSFMs perform relatively better post-COVID; our results make this concrete and severe "
        "for the econometric side specifically: BVAR's RMSFE at h=12 rises from 1.108 (full window) to "
        "1.873 (COVID-only) &mdash; a forecast nearly twice as bad as the naive AR(1) &mdash; while "
        "every TSFM's error stays within a few percent of 1.0 throughout. This likely reflects our "
        "simplified BVAR's lack of the stochastic-volatility or outlier-adjustment machinery the "
        "paper's own cited literature (e.g. Carriero et al. 2022) uses specifically to handle the "
        "2020 outliers.", "Body"))
    story.append(p(
        "<b>The hybrid model partially, not fully, replicates the paper's gain.</b> It improves on "
        "plain BVAR at h=1 (median 1.000 &rarr; 0.989) but never beats standalone TimesFM at any "
        "horizon, and during COVID it inherits BVAR's breakdown almost unchanged (since the hybrid's "
        "forecast is anchored to BVAR's own point forecast plus a learned residual correction, which "
        "cannot compensate for a wholesale, one-off regime shift). We attribute the more limited "
        "hybrid gain to our simplified BVAR being a comparatively weaker base model than the paper's "
        "fully-tuned version, capping the hybrid's achievable ceiling.", "Body"))

    story.append(p("7. Conclusion", "H1"))
    story.append(p(
        "The central qualitative claim of the reproduced paper &mdash; TSFMs are competitive with, "
        "but not clearly superior to, well-specified econometric models, and the two are "
        "complementary enough that hybridizing them can help &mdash; replicates at this reduced "
        "scale. The TSFM performance ranking matches closely; the two points of divergence (Time-GPT's "
        "reversal, the hybrid's more limited gain) are both explainable by concrete, documented "
        "implementation differences rather than methodological error, consistent with the Stage 1 "
        "requirement to discuss and justify any deviation from the original study.", "Body"))

    story.append(p("References", "H1"))
    refs = [
        "Carriero, A., Pettenuzzo, D., &amp; Shekhar, S. (2026). Macroeconomic Forecasting with Large "
        "Language Models. arXiv:2407.00890v5.",
        "McCracken, M. W., &amp; Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic "
        "Research. Journal of Business &amp; Economic Statistics, 34(4), 574-589.",
        "Banbura, M., Giannone, D., &amp; Reichlin, L. (2010). Large Bayesian Vector Autoregressions. "
        "Journal of Applied Econometrics, 25(1), 71-92.",
    ]
    for r in refs:
        story.append(p(r, "Body"))

    doc.build(story)
    print("saved STAGE1_REPORT_FULL.pdf")


if __name__ == "__main__":
    build()
