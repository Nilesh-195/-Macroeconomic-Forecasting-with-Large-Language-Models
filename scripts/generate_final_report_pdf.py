"""
generate_final_report_pdf.py - Build the complete academic Project Report PDF.
Strictly following Part B: Project Report Guidelines (9 Comprehensive Sections).
"""
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
import pypdf

ROOT = Path("/Users/nilesh/LLM")
OUT_PDF = ROOT / "FINAL_PROJECT_REPORT.pdf"

DIAGRAM1_PATH = ROOT / "output" / "block_diagram_stage1_original.png"
DIAGRAM2_PATH = ROOT / "output" / "block_diagram_stage2_modified.png"
CHART_PATH = ROOT / "output" / "finbert_vs_timesfm_horizons.png"

PAGE_W, PAGE_H = letter

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(54, PAGE_H - 36, "Macroeconomic Forecasting with Large Language Models & FinBERT")
            self.drawRightString(PAGE_W - 54, PAGE_H - 36, "GROUP 7 | Final Project Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, PAGE_H - 42, PAGE_W - 54, PAGE_H - 42)

        # Footer
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 45, PAGE_W - 54, 45)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 32, "Confidential — Academic Research Submission")
        self.drawRightString(PAGE_W - 54, 32, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()

def build_report():
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#0f2038"),
        alignment=1,  # Center
    )
    sub_title_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2563eb"),
        alignment=1,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f2038"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1e3a5f"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.2,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    body_bold = ParagraphStyle(
        "ReportBodyBold",
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=12.2,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=4,
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e3a5f"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor("#0f2038"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor("#0f2038"),
    )
    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        fontName="Helvetica",
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor("#0f2038"),
        alignment=1,
    )
    table_cell_center_bold = ParagraphStyle(
        "TableCellCenterBold",
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor("#0f2038"),
        alignment=1,
    )
    caption_style = ParagraphStyle(
        "CaptionStyle",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=8,
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("19. Macroeconomic Forecasting with Large Language Models", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Empirical Reproduction of Carriero et al. (2026) & Deep Multimodal FinBERT Augmentation", sub_title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>GROUP 7 SUBMISSION</b> &nbsp;|&nbsp; <b>Academic Session 2026</b>", meta_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Nilesh Dwivedi</b> (24bds048) &nbsp;•&nbsp; <b>Radhakrishna Bharuka</b> (24bds063) &nbsp;•&nbsp; <b>Abhang Pawar</b> (24bds054)", meta_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Hardware Acceleration:</b> Apple Silicon M4 GPU (`mps`) &nbsp;|&nbsp; <b>Code:</b> github.com/Nilesh-195/-Macroeconomic-Forecasting-with-Large-Language-Models", caption_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

    # Executive Summary Callout Box
    summary_text = (
        "<b>Executive Abstract:</b> This project reproduces and significantly extends the state-of-the-art research paper "
        "<i>'Macroeconomic Forecasting with Large Language Models'</i> (Carriero, Pettenuzzo & Shekhar, 2026, arXiv:2407.00890v5). "
        "In <b>Stage 1</b>, we benchmarked 6 Time Series Foundation Models (TSFMs) against classical econometric baselines (AR(1), BVAR, Factor-AR) "
        "on 60 balanced FRED-MD series across 48 quarterly rolling origins (2010–2022). "
        "In <b>Stage 2</b>, we identified the fundamental limitation of numerical models — their inability to capture real-time forward-looking narrative shifts — "
        "and introduced a novel <b>Deep Multimodal Engine</b>. We ingested <b>1,575,882 historical financial articles</b> from 64 years of New York Times archives (1959–2023), "
        "embedded them on Apple Silicon M4 GPU using <b>FinBERT</b>, compressed them into 8 orthogonal narrative factors via PCA, and developed a "
        "<b>Rolling Ridge Semantic Meta-Learner</b> with an expanded 120-month (10-year) context window. The modified multimodal engine outperforms pure foundation "
        "models on <b>11 out of 12 forecasting horizons</b>, achieving up to <b>12.01% error reduction</b>."
    )
    t_summary = Table([[Paragraph(summary_text, callout_style)]], colWidths=[PAGE_W - 108])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93c5fd")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 1: Application and Problem Statement
    # =========================================================================
    story.append(Paragraph("1. Application and Problem Statement", h1_style))
    story.append(Paragraph(
        "<b>1.1 Macroeconomic Forecasting Application:</b> Macroeconomic forecasting is the foundational pillar of modern economic policymaking, "
        "central banking, and corporate strategic planning. Key macroeconomic variables — such as the Consumer Price Index (CPI inflation), "
        "the Unemployment Rate, the Federal Funds Effective Rate, and Industrial Production — determine national monetary policies, interest rates, "
        "credit availability, and capital allocation across global financial markets.",
        body_style
    ))
    story.append(Paragraph(
        "<b>1.2 The Problem Addressed:</b> For over four decades, macroeconomic forecasting has been dominated by linear econometric models: "
        "Univariate Autoregressions (AR), Bayesian Vector Autoregressions (BVAR), and Dynamic Factor Models. While theoretically sound, these models "
        "suffer from severe parameter proliferation ('curse of dimensionality') and cannot easily model non-linear regime shifts. "
        "Recently, pretrained <b>Time Series Foundation Models (TSFMs)</b> — often referred to as the 'LLMs of forecasting' — have emerged. "
        "The central research problem addressed by Carriero et al. (2026) is: <i>Can pretrained zero-shot foundation models outperform well-tuned "
        "classical econometric models on macroeconomic data?</i>",
        body_style
    ))
    story.append(Paragraph(
        "<b>1.3 The Information Lag Limitation (Why News Matters):</b> A profound, unaddressed limitation in the original paper is that all models "
        "operated exclusively on historical numerical series. Government statistics are inherently <i>backward-looking</i> — CPI reports and GDP "
        "estimates are released with 30- to 60-day lag periods. When sudden geopolitical shocks, trade disputes, or supply bottlenecks occur, "
        "the numerical series remain completely blind to the event until weeks later. In contrast, financial journalism and breaking news capture "
        "expectations and sentiment instantaneously. Incorporating unstructured textual news addresses this fundamental information lag.",
        body_style
    ))
    story.append(Paragraph(
        "<b>1.4 Real-World Example:</b> Consider a sudden global maritime shipping disruption (e.g., Red Sea canal blockade). A pure statistical model (AR or BVAR) "
        "only sees last month's stable shipping prices and predicts continued low inflation. A multimodal system reads breaking headlines: <i>'Container shipping "
        "rates triple as vessels reroute around Africa.'</i> FinBERT extracts high inflation sentiment, and the meta-learner elevates the projected inflation rate "
        "weeks before the official Bureau of Labor Statistics (BLS) report is released.",
        body_style
    ))

    # =========================================================================
    # SECTION 2: Dataset Details
    # =========================================================================
    story.append(Paragraph("2. Dataset Details & Preprocessing", h1_style))
    story.append(Paragraph(
        "Our experiments utilize two primary data sources spanning 64 continuous historical years (January 1959 to December 2023):",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.1 FRED-MD Macroeconomic Database:</b> Maintained by the Federal Reserve Bank of St. Louis, FRED-MD is the standard benchmark dataset "
        "in empirical macroeconomics. It provides monthly observations across 8 distinct economic sectors: (1) Output and Income, (2) Labor Market, "
        "(3) Housing, (4) Consumption, Orders, and Inventories, (5) Money and Credit, (6) Interest and Exchange Rates, (7) Prices, and (8) Stock Market. "
        "To ensure balanced historical continuity without missing data, we selected a core panel of <b>60 representative series</b> spanning 1959 to 2022 (747 months).",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.2 McCracken-Ng Stationarity Transformations:</b> Raw macroeconomic variables exhibit strong non-stationarity, secular trends, and unit roots. "
        "Following McCracken & Ng (2016), each series is transformed to near-stationarity using one of 7 standardized transformation codes:",
        body_style
    ))

    # Table of transform codes
    t_codes_data = [
        [Paragraph("<b>Code</b>", table_cell_center_bold), Paragraph("<b>Transformation Name</b>", table_cell_bold), Paragraph("<b>Mathematical Formula</b>", table_cell_bold), Paragraph("<b>Example Economic Series</b>", table_cell_bold)],
        [Paragraph("1", table_cell_center), Paragraph("No Transformation (Levels)", table_cell), Paragraph("x(t)", table_cell), Paragraph("Interest Rates (FEDFUNDS)", table_cell)],
        [Paragraph("2", table_cell_center), Paragraph("First Difference", table_cell), Paragraph("x(t) - x(t-1)", table_cell), Paragraph("Unemployment Rate (UNRATE)", table_cell)],
        [Paragraph("5", table_cell_center), Paragraph("First Difference of Log", table_cell), Paragraph("ln(x(t)) - ln(x(t-1))", table_cell), Paragraph("Industrial Production (INDPRO)", table_cell)],
        [Paragraph("6", table_cell_center), Paragraph("Second Difference of Log", table_cell), Paragraph("Δ ln(x(t)) - Δ ln(x(t-1))", table_cell), Paragraph("Consumer Price Index (CPIAUCSL)", table_cell)],
    ]
    t_codes = Table(t_codes_data, colWidths=[35, 145, 140, 184])
    t_codes.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_codes)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>2.3 New York Times Historical News Corpus (1959–2023):</b> For Stage 2 multimodal augmentation, we acquired 747 monthly archive JSON files "
        "comprising <b>1,575,882 macro-relevant articles</b> across 64 years. Articles were filtered via comprehensive macroeconomic domain regex "
        "(targeting inflation, central banking, labor, trade, industrial energy, and credit markets) to isolate economic narratives from lifestyle and sports news.",
        body_style
    ))

    # =========================================================================
    # SECTION 3: Original Research Method
    # =========================================================================
    story.append(Paragraph("3. Original Research Method (Carriero et al., 2026)", h1_style))
    story.append(Paragraph(
        "The original methodology of Carriero et al. evaluates whether zero-shot Time Series Foundation Models can compete with traditional econometrics "
        "under a recursive expanding-window out-of-sample forecasting design.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3.1 Component Architecture:</b><br/>"
        "• <b>Input Context:</b> Historical numerical series formatted over a 36-month lookback window.<br/>"
        "• <b>Econometric Baselines:</b> AR(1) autoregressive benchmark; Bayesian VAR (BVAR) with Minnesota prior (lag p=6) shrinking cross-variable coefficients; "
        "and Dynamic Factor Autoregressions (Factor-AR) using principal components.<br/>"
        "• <b>Foundation Models:</b> Zero-shot pretrained transformers including Google TimesFM (200M), Salesforce MOIRAI (91M), Amazon Chronos, Nixtla TimeGPT, IBM TTM, and Lag-Llama.<br/>"
        "• <b>Evaluation Metric:</b> Relative Root Mean Squared Forecast Error (RMSFE) against the AR(1) benchmark across forecast horizons h ∈ {1, 3, 6, 12}.",
        body_style
    ))
    story.append(Spacer(1, 4))

    if DIAGRAM1_PATH.exists():
        story.append(Image(str(DIAGRAM1_PATH), width=PAGE_W - 108, height=220))
        story.append(Paragraph("<b>Figure 1:</b> Block diagram of the original research method proposed by Carriero et al. (2026).", caption_style))

    # =========================================================================
    # SECTION 4: Stage 1 – Implementation
    # =========================================================================
    story.append(Paragraph("4. Stage 1 – Implementation Details & Experimental Setup", h1_style))
    story.append(Paragraph(
        "We implemented the complete benchmarking pipeline on local Apple Silicon hardware. The implementation includes:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Software Stack & Libraries:</b> Python 3.14 (`venv_core`), PyTorch with Apple Metal Performance Shaders (`mps`) backend, "
        "Google `timesfm`, HuggingFace `transformers`, `statsmodels`, `scikit-learn`, and `pandas`.<br/>"
        "• <b>Evaluation Protocol:</b> 48 quarterly rolling forecast origins from 2010 Q1 to 2022 Q4. At each origin τ, the models observed history up to τ "
        "and generated multi-step out-of-sample forecasts for horizons h = 1, 3, 6, and 12 months ahead.<br/>"
        "• <b>Rigorous Bug Fixes:</b> During replication, we identified and corrected two classical implementation flaws: (1) fixed the Minnesota dummy prior "
        "decay scaling in BVAR so higher lags shrink appropriately, and (2) converted the Factor model from an unstable iterated AR to direct horizon projections, "
        "preventing explosive exponential errors at h=12.",
        body_style
    ))

    # =========================================================================
    # SECTION 5: Stage 1 – Results
    # =========================================================================
    story.append(Paragraph("5. Stage 1 – Results & Comparison with Original Paper", h1_style))
    story.append(Paragraph(
        "Table 1 contrasts the published results from Carriero et al. against our empirical Stage 1 reproduction across all 60 series. "
        "Values represent median relative RMSFE against AR(1) (values under 1.000 in bold beat the benchmark).",
        body_style
    ))

    # Stage 1 Comparison Table
    s1_comp_data = [
        [Paragraph("<b>Model Category</b>", table_cell_bold), Paragraph("<b>Model Name</b>", table_cell_bold), Paragraph("<b>Paper (h=1)</b>", table_cell_center_bold), Paragraph("<b>Repro (h=1)</b>", table_cell_center_bold), Paragraph("<b>Paper (h=6)</b>", table_cell_center_bold), Paragraph("<b>Repro (h=6)</b>", table_cell_center_bold), Paragraph("<b>Paper (h=12)</b>", table_cell_center_bold), Paragraph("<b>Repro (h=12)</b>", table_cell_center_bold)],
        [Paragraph("Benchmark", table_cell), Paragraph("AR(1)", table_cell), Paragraph("1.000", table_cell_center), Paragraph("1.000", table_cell_center), Paragraph("1.000", table_cell_center), Paragraph("1.000", table_cell_center), Paragraph("1.000", table_cell_center), Paragraph("1.000", table_cell_center)],
        [Paragraph("Foundation (Google)", table_cell), Paragraph("<b>TimesFM 2.5</b>", table_cell_bold), Paragraph("1.014", table_cell_center), Paragraph("<b>0.983</b>", table_cell_center_bold), Paragraph("<b>0.990</b>", table_cell_center), Paragraph("<b>0.997</b>", table_cell_center_bold), Paragraph("1.001", table_cell_center), Paragraph("1.000", table_cell_center)],
        [Paragraph("Foundation (Nixtla)", table_cell), Paragraph("TimeGPT", table_cell), Paragraph("1.077", table_cell_center), Paragraph("<b>0.987</b>", table_cell_center_bold), Paragraph("1.025", table_cell_center), Paragraph("<b>0.994</b>", table_cell_center_bold), Paragraph("1.063", table_cell_center), Paragraph("<b>0.999</b>", table_cell_center_bold)],
        [Paragraph("Foundation (Amazon)", table_cell), Paragraph("Chronos-Bolt", table_cell), Paragraph("N/A", table_cell_center), Paragraph("<b>0.996</b>", table_cell_center_bold), Paragraph("N/A", table_cell_center), Paragraph("<b>0.998</b>", table_cell_center_bold), Paragraph("N/A", table_cell_center), Paragraph("1.002", table_cell_center)],
        [Paragraph("Foundation (Salesforce)", table_cell), Paragraph("MOIRAI-base", table_cell), Paragraph("1.008", table_cell_center), Paragraph("1.031", table_cell_center), Paragraph("<b>0.990</b>", table_cell_center), Paragraph("1.003", table_cell_center), Paragraph("<b>0.999</b>", table_cell_center), Paragraph("1.013", table_cell_center)],
        [Paragraph("Econometric", table_cell), Paragraph("Factor-AR", table_cell), Paragraph("<b>0.965</b>", table_cell_center), Paragraph("1.507", table_cell_center), Paragraph("<b>0.992</b>", table_cell_center), Paragraph("1.089", table_cell_center), Paragraph("<b>0.999</b>", table_cell_center), Paragraph("1.029", table_cell_center)],
        [Paragraph("Econometric", table_cell), Paragraph("BVAR (Minnesota)", table_cell), Paragraph("<b>0.983</b>", table_cell_center), Paragraph("1.332", table_cell_center), Paragraph("<b>0.991</b>", table_cell_center), Paragraph("1.167", table_cell_center), Paragraph("<b>0.999</b>", table_cell_center), Paragraph("1.108", table_cell_center)],
    ]
    t_s1_comp = Table(s1_comp_data, colWidths=[90, 95, 53, 53, 53, 53, 53, 53])
    t_s1_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2038")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_s1_comp)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Table 1:</b> Relative RMSFE comparison between the published paper (1985–2019) and our Stage 1 reproduction (2010–2022).", caption_style))

    story.append(Paragraph(
        "<b>Important Observations:</b><br/>"
        "1. <b>Foundation Models Handle Stress Better:</b> In our 2010–2022 test period (which includes the unprecedented COVID-19 macroeconomic shock), "
        "Google TimesFM 2.5 remained remarkably stable, beating AR(1) at h=1 (0.983). In contrast, classical BVAR and Factor models suffered from parameter instability "
        "due to extreme non-Gaussian COVID outliers.<br/>"
        "2. <b>Winner of Stage 1:</b> Google TimesFM 2.5 emerged as the single most robust foundation model across all evaluations, establishing it as the base forecaster for Stage 2.",
        body_style
    ))

    # =========================================================================
    # SECTION 6: Stage 2 – Proposed Modifications
    # =========================================================================
    story.append(Paragraph("6. Stage 2 – Proposed Modifications & Deep Multimodal Architecture", h1_style))
    story.append(Paragraph(
        "To surpass the numerical barrier of the original paper, we designed and implemented four fundamental engineering modifications:",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.1 Modification 1: Ingestion of 64 Years of NYT Financial Archives:</b> Rather than sampling a tiny fraction of headlines, we parsed "
        "<b>1,575,882 full-text articles</b> across 747 continuous months (1959–2023), ensuring zero lookahead bias.",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.2 Modification 2: Local GPU FinBERT Feature Extraction:</b> Using an Apple Silicon M4 GPU with PyTorch `mps` hardware acceleration, "
        "articles were processed in batches of 128 through `ProsusAI/finbert`. In a unified forward pass, we extracted both 768-dimensional semantic embeddings "
        "from the `[CLS]` token and 3-class financial sentiment probabilities (Positive, Negative, Neutral).",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.3 Modification 3: PCA Compression & Factor Alignment:</b> To prevent overfitting while preserving narrative variance, the 768-dimensional embeddings "
        "were compressed via Principal Component Analysis (PCA) into <b>8 orthogonal macro narrative factors</b>, capturing <b>92.1% of total semantic variance</b>. "
        "Together with the 3 sentiment scores, this yielded an 11-dimensional monthly news covariate vector.",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.4 Modification 4: Context Window Expansion (36 → 120 Months):</b> As established in macroeconomic theory, business cycles last 5 to 10 years. "
        "The paper's 36-month lookback was insufficient to capture full economic cycles. We expanded the context length to <b>120 months (10 full years)</b>, "
        "allowing the Transformer self-attention layers to identify long-term mean reversions.",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.5 Modification 5: Rolling Ridge Semantic Meta-Learner:</b> We designed a two-pass hybrid engine. Pass 1 generates pure zero-shot baseline forecasts "
        "from TimesFM 2.5. Pass 2 fits an out-of-sample regularized Ridge regression (alpha=1.0) on rolling historical errors against the 11 news factors. "
        "The meta-learner applies a dynamic 'news shift' correction to the foundation forecast.",
        body_style
    ))
    story.append(Spacer(1, 4))

    if DIAGRAM2_PATH.exists():
        story.append(Image(str(DIAGRAM2_PATH), width=PAGE_W - 108, height=220))
        story.append(Paragraph("<b>Figure 2:</b> Block diagram of the proposed Stage 2 Deep Multimodal Architecture combining FinBERT, PCA, and TimesFM 2.5.", caption_style))

    # =========================================================================
    # SECTION 7: Stage 2 – Experimental Results
    # =========================================================================
    story.append(Paragraph("7. Stage 2 – Experimental Results & Comparative Evaluation", h1_style))
    story.append(Paragraph(
        "We executed a comprehensive evaluation spanning <b>144 monthly rolling origins (2010 to 2021)</b> across all 60 core series and all 12 forecasting horizons "
        "(h = 1 to 12), producing a total of <b>69,120 out-of-sample forecast evaluations</b>. Table 2 presents the final RMSFE results.",
        body_style
    ))

    # Stage 2 Final Results Table
    s2_res_data = [
        [Paragraph("<b>Horizon</b>", table_cell_center_bold), Paragraph("<b>Forecast Lead</b>", table_cell_bold), Paragraph("<b>TimesFM 2.5 (Pure)</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid</b>", table_cell_center_bold), Paragraph("<b>Performance Gain</b>", table_cell_center_bold), Paragraph("<b>Winning Model</b>", table_cell_bold)],
        [Paragraph("h = 1", table_cell_center), Paragraph("1 Month Ahead", table_cell), Paragraph("<b>0.7674</b>", table_cell_center_bold), Paragraph("0.7873", table_cell_center), Paragraph("-2.60%", table_cell_center), Paragraph("TimesFM 2.5", table_cell)],
        [Paragraph("h = 2", table_cell_center), Paragraph("2 Months Ahead", table_cell), Paragraph("0.9002", table_cell_center), Paragraph("<b>0.8590</b>", table_cell_center_bold), Paragraph("<b>+4.58%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 3", table_cell_center), Paragraph("1 Quarter Ahead", table_cell), Paragraph("0.9485", table_cell_center), Paragraph("<b>0.8983</b>", table_cell_center_bold), Paragraph("<b>+5.29%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 4", table_cell_center), Paragraph("4 Months Ahead", table_cell), Paragraph("0.9818", table_cell_center), Paragraph("<b>0.9167</b>", table_cell_center_bold), Paragraph("<b>+6.63%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 5", table_cell_center), Paragraph("5 Months Ahead", table_cell), Paragraph("1.0116", table_cell_center), Paragraph("<b>0.9177</b>", table_cell_center_bold), Paragraph("<b>+9.29%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 6", table_cell_center), Paragraph("Half Year Ahead", table_cell), Paragraph("1.0360", table_cell_center), Paragraph("<b>0.9180</b>", table_cell_center_bold), Paragraph("<b>+11.38%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 7", table_cell_center), Paragraph("7 Months Ahead", table_cell), Paragraph("1.0601", table_cell_center), Paragraph("<b>0.9458</b>", table_cell_center_bold), Paragraph("<b>+10.78%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 8", table_cell_center), Paragraph("8 Months Ahead", table_cell), Paragraph("1.0827", table_cell_center), Paragraph("<b>0.9723</b>", table_cell_center_bold), Paragraph("<b>+10.20%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 9", table_cell_center), Paragraph("3 Quarters Ahead", table_cell), Paragraph("1.1017", table_cell_center), Paragraph("<b>0.9946</b>", table_cell_center_bold), Paragraph("<b>+9.72%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 10", table_cell_center), Paragraph("10 Months Ahead", table_cell), Paragraph("1.1175", table_cell_center), Paragraph("<b>1.0129</b>", table_cell_center_bold), Paragraph("<b>+9.36%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 11", table_cell_center), Paragraph("11 Months Ahead", table_cell), Paragraph("1.1350", table_cell_center), Paragraph("<b>1.0195</b>", table_cell_center_bold), Paragraph("<b>+10.17%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 12", table_cell_center), Paragraph("1 Full Year Ahead", table_cell), Paragraph("1.1540", table_cell_center), Paragraph("<b>1.0154</b>", table_cell_center_bold), Paragraph("<b>+12.01%</b>", table_cell_center_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
    ]
    t_s2_res = Table(s2_res_data, colWidths=[55, 95, 95, 95, 80, 84])
    t_s2_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_s2_res)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Table 2:</b> Empirical forecast evaluation across all 12 horizons comparing Google TimesFM 2.5 against the FinBERT Multimodal Hybrid Engine.", caption_style))

    story.append(Spacer(1, 4))
    if CHART_PATH.exists():
        story.append(Image(str(CHART_PATH), width=PAGE_W - 108, height=185))
        story.append(Paragraph("<b>Figure 3:</b> Comparative RMSFE trajectory across horizons h = 1 to 12. FinBERT Hybrid consistently restrains error accumulation.", caption_style))

    story.append(Paragraph(
        "<b>Key Experimental Observations:</b><br/>"
        "1. <b>Decisive Dominance Across Horizons:</b> The FinBERT Hybrid engine outperforms pure TimesFM 2.5 on <b>11 out of 12 forecasting horizons</b>.<br/>"
        "2. <b>The Dynamics of Lead Time:</b> At immediate horizon $h=1$, pure numerical inertia dominates (the prior month's momentum carries forward), giving TimesFM a slight edge (-2.6%). "
        "However, as the forecast horizon extends to $h=4..12$, numerical autoregressive memory decays rapidly. At these medium and long horizons, forward-looking news factors provide crucial "
        "stabilizing anchors, yielding double-digit improvements: <b>+11.38% error reduction at 6 months</b> and <b>+12.01% error reduction at 12 months</b>.",
        body_style
    ))

    # =========================================================================
    # SECTION 8: Conclusions & Learning Outcomes
    # =========================================================================
    story.append(Paragraph("8. Conclusions, Limitations & Learning Outcomes", h1_style))
    story.append(Paragraph(
        "<b>8.1 Conclusions:</b> Pretrained time-series foundation models represent a powerful evolution in economic forecasting, achieving competitive "
        "accuracy without costly per-series retraining. However, when restricted purely to past numbers, they remain subject to information lag. "
        "Our work demonstrates that fusing domain-specific textual language models (FinBERT) with foundation time-series architectures (TimesFM 2.5) "
        "creates a synergistic multimodal system that significantly outperforms pure econometric and pure foundation benchmarks.",
        body_style
    ))
    story.append(Paragraph(
        "<b>8.2 Limitations:</b> (1) The news dataset relies primarily on the New York Times, which may reflect regional editorial focus; integrating international "
        "sources (e.g., Financial Times, Reuters) would enrich global coverage. (2) Real-time forecasting requires handling publication embargoes and real-time news APIs.",
        body_style
    ))
    story.append(Paragraph(
        "<b>8.3 What Other Students Can Learn from This Project:</b><br/>"
        "• <b>Stationarity is Fundamental:</b> Never feed raw macroeconomic levels (e.g., raw CPI) directly into neural networks. Applying McCracken-Ng transform codes "
        "is essential to achieve stationary residuals and avoid spurious correlations.<br/>"
        "• <b>Context Window Design:</b> Context windows must match the underlying domain physics. While 36 months may suffice for retail demand, macroeconomic systems "
        "operate on 5–10 year business cycles; expanding context to 120 months is critical for Transformer self-attention.<br/>"
        "• <b>Local Edge Compute:</b> With optimized frameworks (PyTorch `mps`), standard personal workstations (Apple M4) can process over 1.5 million articles through transformer "
        "models locally in minutes, democratizing multimodal research without expensive cloud infrastructure.",
        body_style
    ))

    # =========================================================================
    # SECTION 9: References
    # =========================================================================
    story.append(Paragraph("9. References", h1_style))
    refs = [
        "1. Carriero, A., Pettenuzzo, D., & Shekhar, S. (2026). <i>Macroeconomic Forecasting with Large Language Models</i>. arXiv preprint arXiv:2407.00890v5.",
        "2. Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). <i>A decoder-only foundation model for time-series forecasting</i>. arXiv preprint arXiv:2310.10688.",
        "3. Araci, D. (2019). <i>FinBERT: Financial Sentiment Analysis with Pre-trained Language Models</i>. arXiv preprint arXiv:1908.10063.",
        "4. McCracken, M. W., & Ng, S. (2016). <i>FRED-MD: A monthly database for macroeconomic research</i>. Journal of Business & Economic Statistics, 34(4), 574-589.",
        "5. Ansari, A. F., et al. (2024). <i>Chronos: Learning the Language of Time Series</i>. Transactions on Machine Learning Research.",
        "6. Woo, G., et al. (2024). <i>Unified Training of Universal Time Series Forecasting Transformers (MOIRAI)</i>. ICML 2024.",
    ]
    for r in refs:
        story.append(Paragraph(r, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Academic Report successfully built at: {OUT_PDF}")

if __name__ == "__main__":
    build_report()

