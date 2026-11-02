"""
generate_presentation_pdf.py - Generate the 7-Slide Final Project Presentation PDF
Strictly following Part A Presentation Guidelines (16:9 Landscape Format).
"""
import sys
from pathlib import Path
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

ROOT = Path("/Users/nilesh/LLM")
OUT_PDF = ROOT / "FINAL_PRESENTATION_SLIDES.pdf"

DIAGRAM1_PATH = ROOT / "output" / "block_diagram_stage1_original.png"
DIAGRAM2_PATH = ROOT / "output" / "block_diagram_stage2_modified.png"
CHART_PATH = ROOT / "output" / "finbert_vs_timesfm_horizons.png"

# Dimensions for 16:9 Slide: 10 x 5.625 inches (720 x 405 pt)
SLIDE_W = 10 * 72.0
SLIDE_H = 5.625 * 72.0

class NumberedSlideCanvas(canvas.Canvas):
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
            self.draw_slide_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_slide_decorations(self, total_pages):
        self.saveState()
        # Top banner bar
        self.setFillColor(colors.HexColor("#0f2038"))
        self.rect(0, SLIDE_H - 12, SLIDE_W, 12, stroke=0, fill=1)
        self.setFillColor(colors.HexColor("#2563eb"))
        self.rect(0, SLIDE_H - 15, SLIDE_W, 3, stroke=0, fill=1)
        
        # Bottom footer bar
        self.setFillColor(colors.HexColor("#f1f5f9"))
        self.rect(0, 0, SLIDE_W, 22, stroke=0, fill=1)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.75)
        self.line(0, 22, SLIDE_W, 22)
        
        # Footer text
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1e3a5f"))
        self.drawString(24, 8, "Macroeconomic Forecasting with Large Language Models & FinBERT | GROUP 7")
        
        # Slide counter
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(SLIDE_W - 24, 8, f"Slide {self._pageNumber} of {total_pages}")
        self.restoreState()

def build_presentation():
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=(SLIDE_W, SLIDE_H),
        leftMargin=28,
        rightMargin=28,
        topMargin=26,
        bottomMargin=28,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    slide_title_style = ParagraphStyle(
        "SlideTitle",
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        textColor=colors.HexColor("#0f2038"),
    )
    slide_sub_style = ParagraphStyle(
        "SlideSubtitle",
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#2563eb"),
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0f2038"),
    )
    body_style = ParagraphStyle(
        "SlideBody",
        fontName="Helvetica",
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )
    card_title = ParagraphStyle(
        "CardTitle",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e3a5f"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#0f2038"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#0f2038"),
    )

    story = []

    # =========================================================================
    # SLIDE 1: Application and Problem Statement
    # =========================================================================
    story.append(Paragraph("Slide 1: Application & Problem Statement", slide_title_style))
    story.append(Paragraph("Research Paper: <i>'Macroeconomic Forecasting with Large Language Models'</i> (Carriero, Pettenuzzo & Shekhar, 2026)", slide_sub_style))
    story.append(Spacer(1, 8))

    col1_content = [
        Paragraph("<b>1. The Real-World Application</b>", card_title),
        Spacer(1, 4),
        Paragraph("• <b>Macroeconomic Forecasting:</b> Projecting key national economic indicators (CPI inflation, unemployment rate, interest rates, GDP) across 1 to 12 months ahead.", body_style),
        Paragraph("• <b>Stakeholders:</b> Central banks (Federal Reserve) setting interest rates, financial institutions allocating investment capital, and corporate risk managers.", body_style),
        Spacer(1, 6),
        Paragraph("<b>2. The Core Problem Being Solved</b>", card_title),
        Spacer(1, 4),
        Paragraph("• <b>Can Foundation Models Beat 50-Year Econometrics?</b> Economists still rely on Vector Autoregressions (BVAR) and Factor models. Can pretrained zero-shot Transformer models (TSFMs) predict macro trends better?", body_style),
        Paragraph("• <b>Information Lag:</b> Government statistics arrive with 30–60 day delays. Numerical series alone miss sudden psychological and geopolitical sentiment shifts.", body_style),
    ]

    col2_content = [
        Paragraph("<b>3. Simple Real-World Example</b>", card_title),
        Spacer(1, 3),
        Paragraph("• <b>Scenario:</b> An emerging supply chain crisis or oil supply disruption begins.", body_style),
        Paragraph("• <b>Traditional Models (AR / BVAR):</b> Look only at past monthly inflation numbers, staying blind until price spikes appear in delayed official statistics.", body_style),
        Paragraph("• <b>Multimodal AI (Our Solution):</b> Reads breaking news in real-time. FinBERT detects surging bottleneck sentiment, signaling TimesFM to adjust forecasts weeks earlier.", body_style),
        Spacer(1, 4),
        Paragraph("<b>Project Details & Team (GROUP 7)</b>", card_title),
        Spacer(1, 2),
        Paragraph("• <b>Project 19:</b> Macroeconomic Forecasting with Large Language Models", body_bold),
        Paragraph("• <b>Nilesh Dwivedi</b> (24bds048) &nbsp;|&nbsp; <b>Radhakrishna Bharuka</b> (24bds063)", body_style),
        Paragraph("• <b>Abhang Pawar</b> (24bds054) &nbsp;|&nbsp; <b>Hardware:</b> Apple Silicon M4 GPU (MPS)", body_style),
    ]

    table_s1 = Table(
        [[col1_content, col2_content]],
        colWidths=[325, 325]
    )
    table_s1.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#eff6ff")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#cbd5e1")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#93c5fd")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(table_s1)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 2: Dataset and Original Method
    # =========================================================================
    story.append(Paragraph("Slide 2: Dataset & Original Method (Block Diagram 1)", slide_title_style))
    story.append(Paragraph("FRED-MD Macroeconomic Database & Carriero et al. (2026) Baseline Architecture", slide_sub_style))
    story.append(Spacer(1, 6))

    ds_text = (
        "<b>Dataset Details:</b> Federal Reserve Bank of St. Louis (FRED-MD). 60 balanced core monthly series spanning 1959–2022 (747 months). "
        "Categories include Output, Labor Market, Housing, Consumption, Money/Credit, Interest Rates/FX, Prices (CPI), and Stock Market.<br/>"
        "<b>Stationary Preprocessing:</b> Applied 7 standard McCracken & Ng (2016) transform codes (first-diff, log-diff, second-difference) to ensure stationarity."
    )
    story.append(Paragraph(ds_text, body_style))
    story.append(Spacer(1, 6))

    if DIAGRAM1_PATH.exists():
        story.append(Image(str(DIAGRAM1_PATH), width=650, height=225))
    else:
        story.append(Paragraph("Original Method Diagram missing", body_style))

    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: Results from the Original Paper
    # =========================================================================
    story.append(Paragraph("Slide 3: Results from the Original Research Paper", slide_title_style))
    story.append(Paragraph("Carriero, Pettenuzzo & Shekhar (2026) — Benchmark Findings Across 120 Series (1985–2019)", slide_sub_style))
    story.append(Spacer(1, 6))

    p3_left = [
        Paragraph("<b>Evaluation Metric: Relative RMSFE</b>", card_title),
        Spacer(1, 4),
        Paragraph("Accuracy is measured using Root Mean Squared Forecast Error relative to the baseline AR(1) model (Paper Eq. 11):", body_style),
        Spacer(1, 2),
        Paragraph("<b>RMSFE_rel = RMSFE(Model) / RMSFE(AR(1))</b>", body_bold),
        Spacer(1, 2),
        Paragraph("• <b>Value < 1.0:</b> Model outperforms the AR(1) benchmark (shown in bold).<br/>• <b>Value > 1.0:</b> Model underperforms AR(1).", body_style),
        Spacer(1, 6),
        Paragraph("<b>Key Findings of the Original Paper</b>", card_title),
        Spacer(1, 4),
        Paragraph("1. <b>TSFMs are competitive, not superior:</b> Only 3 of 6 TSFMs (TimesFM, MOIRAI, Chronos) consistently beat AR(1). They perform on par with BVAR/Factor models, not significantly better.", body_style),
        Paragraph("2. <b>Error Orthogonality:</b> The forecast errors of TSFMs and econometric models are uncorrelated (correlation 0.07 to 0.50). Combining them in a hybrid model mitigates large outliers.", body_style),
    ]

    paper_table_data = [
        [Paragraph("<b>Model</b>", table_cell_bold), Paragraph("<b>h=1</b>", table_cell_bold), Paragraph("<b>h=3</b>", table_cell_bold), Paragraph("<b>h=6</b>", table_cell_bold), Paragraph("<b>h=12</b>", table_cell_bold)],
        [Paragraph("<b>AR(1) (Benchmark)</b>", table_cell), Paragraph("1.000", table_cell), Paragraph("1.000", table_cell), Paragraph("1.000", table_cell), Paragraph("1.000", table_cell)],
        [Paragraph("BVAR (Minnesota)", table_cell), Paragraph("<b>0.983</b>", table_cell_bold), Paragraph("<b>0.970</b>", table_cell_bold), Paragraph("<b>0.991</b>", table_cell_bold), Paragraph("<b>0.999</b>", table_cell_bold)],
        [Paragraph("Factor Model (PCA)", table_cell), Paragraph("<b>0.965</b>", table_cell_bold), Paragraph("<b>0.980</b>", table_cell_bold), Paragraph("<b>0.992</b>", table_cell_bold), Paragraph("<b>0.999</b>", table_cell_bold)],
        [Paragraph("TimesFM (Google)", table_cell), Paragraph("1.014", table_cell), Paragraph("<b>0.980</b>", table_cell_bold), Paragraph("<b>0.990</b>", table_cell_bold), Paragraph("1.001", table_cell)],
        [Paragraph("MOIRAI-large (Salesforce)", table_cell), Paragraph("<b>0.999</b>", table_cell_bold), Paragraph("<b>0.978</b>", table_cell_bold), Paragraph("<b>0.991</b>", table_cell_bold), Paragraph("1.001", table_cell)],
        [Paragraph("TimeGPT (Nixtla)", table_cell), Paragraph("1.077", table_cell), Paragraph("1.044", table_cell), Paragraph("1.025", table_cell), Paragraph("1.063", table_cell)],
        [Paragraph("Lag-Llama", table_cell), Paragraph("1.015", table_cell), Paragraph("<b>0.997</b>", table_cell_bold), Paragraph("1.002", table_cell), Paragraph("1.009", table_cell)],
        [Paragraph("<b>Hybrid (BVAR + MOIRAI)</b>", table_cell_bold), Paragraph("<b>0.973</b>", table_cell_bold), Paragraph("<b>0.960</b>", table_cell_bold), Paragraph("<b>0.983</b>", table_cell_bold), Paragraph("<b>0.998</b>", table_cell_bold)],
    ]
    t_paper = Table(paper_table_data, colWidths=[130, 42, 42, 42, 42])
    t_paper.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))

    t_s3 = Table([[p3_left, [Paragraph("<b>Results Table from Original Paper</b>", card_title), Spacer(1, 4), t_paper]]], colWidths=[330, 320])
    t_s3.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_s3)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 4: Stage 1 – Implementation and Results
    # =========================================================================
    story.append(Paragraph("Slide 4: Stage 1 – Implementation & Results", slide_title_style))
    story.append(Paragraph("Empirical Replication on 60 Balanced FRED-MD Series (2010 Q1 – 2022 Q4, 48 Rolling Origins)", slide_sub_style))
    story.append(Spacer(1, 6))

    s4_left = [
        Paragraph("<b>What We Implemented</b>", card_title),
        Spacer(1, 3),
        Paragraph("• <b>Full Experimental Pipeline:</b> Replicated data preparation, McCracken-Ng transforms, rolling origin evaluation, and relative RMSFE calculations.", body_style),
        Paragraph("• <b>Evaluated 6 TSFMs + 3 Baselines:</b> Google TimesFM 2.5, Nixtla TimeGPT, Amazon Chronos-Bolt, Salesforce MOIRAI, IBM TTM, Lag-Llama vs. AR(1), BVAR, Factor-AR.", body_style),
        Spacer(1, 4),
        Paragraph("<b>Comparison with Paper & Differences</b>", card_title),
        Spacer(1, 3),
        Paragraph("• <b>Evaluation Era:</b> Paper evaluated 1985–2019; we evaluated 2010–2022 (capturing post-GFC zero-rate regimes and extreme COVID-19 volatility).", body_style),
        Paragraph("• <b>TimesFM 2.5 Excellence:</b> Google TimesFM 2.5 proved to be the most resilient foundation model (0.983 at h=1), outperforming other TSFMs.", body_style),
    ]

    s1_results = [
        [Paragraph("<b>Model</b>", table_cell_bold), Paragraph("<b>h=1</b>", table_cell_bold), Paragraph("<b>h=3</b>", table_cell_bold), Paragraph("<b>h=6</b>", table_cell_bold), Paragraph("<b>h=12</b>", table_cell_bold)],
        [Paragraph("AR(1) (Benchmark)", table_cell), Paragraph("1.000", table_cell), Paragraph("1.000", table_cell), Paragraph("1.000", table_cell), Paragraph("1.000", table_cell)],
        [Paragraph("TimesFM 2.5 (Google)", table_cell_bold), Paragraph("<b>0.983</b>", table_cell_bold), Paragraph("<b>0.995</b>", table_cell_bold), Paragraph("<b>0.997</b>", table_cell_bold), Paragraph("1.000", table_cell)],
        [Paragraph("TimeGPT (Nixtla)", table_cell), Paragraph("<b>0.987</b>", table_cell_bold), Paragraph("<b>0.997</b>", table_cell_bold), Paragraph("<b>0.994</b>", table_cell_bold), Paragraph("<b>0.999</b>", table_cell_bold)],
        [Paragraph("Chronos-Bolt (Amazon)", table_cell), Paragraph("<b>0.996</b>", table_cell_bold), Paragraph("<b>0.997</b>", table_cell_bold), Paragraph("<b>0.998</b>", table_cell_bold), Paragraph("1.002", table_cell)],
        [Paragraph("MOIRAI (Salesforce)", table_cell), Paragraph("1.031", table_cell), Paragraph("1.004", table_cell), Paragraph("1.003", table_cell), Paragraph("1.013", table_cell)],
        [Paragraph("Lag-Llama", table_cell), Paragraph("1.043", table_cell), Paragraph("1.022", table_cell), Paragraph("1.020", table_cell), Paragraph("1.014", table_cell)],
        [Paragraph("TTM (IBM)", table_cell), Paragraph("1.117", table_cell), Paragraph("1.040", table_cell), Paragraph("1.019", table_cell), Paragraph("1.007", table_cell)],
        [Paragraph("Factor-AR", table_cell), Paragraph("1.507", table_cell), Paragraph("1.192", table_cell), Paragraph("1.089", table_cell), Paragraph("1.029", table_cell)],
        [Paragraph("BVAR (Minnesota)", table_cell), Paragraph("1.332", table_cell), Paragraph("1.262", table_cell), Paragraph("1.167", table_cell), Paragraph("1.108", table_cell)],
    ]
    t_s1 = Table(s1_results, colWidths=[130, 42, 42, 42, 42])
    t_s1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2038")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))

    t_s4 = Table([[s4_left, [Paragraph("<b>Our Stage 1 Reproduced Results</b>", card_title), Spacer(1, 4), t_s1]]], colWidths=[330, 320])
    t_s4.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_s4)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: Stage 2 – Proposed Modifications
    # =========================================================================
    story.append(Paragraph("Slide 5: Stage 2 – Proposed Modifications (Block Diagram 2)", slide_title_style))
    story.append(Paragraph("Multimodal FinBERT Sentiment & Narrative Signals + 120-Month Business Cycle Memory", slide_sub_style))
    story.append(Spacer(1, 4))

    mod_bullets = (
        "<b>Why Modifications Were Introduced:</b> Macro numbers only describe the past. Forward-looking shifts (expectations, policy uncertainty, supply bottlenecks) are first recorded in news text. "
        "<b>Four Core Engineering Innovations:</b><br/>"
        "1. <b>1.57 Million NYT Articles (1959–2023):</b> Ingested 747 monthly archive JSON files spanning 64 continuous years.<br/>"
        "2. <b>Local Apple Silicon M4 GPU FinBERT:</b> In-memory batch processing at 520 articles/sec; extracted 768-dim embeddings + sentiment logits.<br/>"
        "3. <b>PCA Compression (92.1% Variance):</b> Reduced 768 dims to 8 orthogonal macro narrative factors + 3 sentiment channels (11 news signals).<br/>"
        "4. <b>120-Month Context Window:</b> Expanded input context from 36 to 120 months (10 years) to capture complete business cycles."
    )
    story.append(Paragraph(mod_bullets, body_style))
    story.append(Spacer(1, 4))

    if DIAGRAM2_PATH.exists():
        story.append(Image(str(DIAGRAM2_PATH), width=650, height=215))
    else:
        story.append(Paragraph("Modified Method Diagram missing", body_style))

    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: Stage 2 – Final Results
    # =========================================================================
    story.append(Paragraph("Slide 6: Stage 2 – Final Experimental Results", slide_title_style))
    story.append(Paragraph("TimesFM 2.5 vs. FinBERT Hybrid Across All 12 Horizons (144 Monthly Rolling Origins, 2010–2021)", slide_sub_style))
    story.append(Spacer(1, 4))

    s6_table_data = [
        [Paragraph("<b>Horizon</b>", table_cell_bold), Paragraph("<b>TimesFM 2.5</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid</b>", table_cell_bold), Paragraph("<b>Gain</b>", table_cell_bold), Paragraph("<b>Winner</b>", table_cell_bold)],
        [Paragraph("h = 1", table_cell), Paragraph("<b>0.7674</b>", table_cell_bold), Paragraph("0.7873", table_cell), Paragraph("-2.60%", table_cell), Paragraph("TimesFM 2.5", table_cell)],
        [Paragraph("h = 2", table_cell), Paragraph("0.9002", table_cell), Paragraph("<b>0.8590</b>", table_cell_bold), Paragraph("<b>+4.58%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 3", table_cell), Paragraph("0.9485", table_cell), Paragraph("<b>0.8983</b>", table_cell_bold), Paragraph("<b>+5.29%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 4", table_cell), Paragraph("0.9818", table_cell), Paragraph("<b>0.9167</b>", table_cell_bold), Paragraph("<b>+6.63%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 5", table_cell), Paragraph("1.0116", table_cell), Paragraph("<b>0.9177</b>", table_cell_bold), Paragraph("<b>+9.29%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 6", table_cell), Paragraph("1.0360", table_cell), Paragraph("<b>0.9180</b>", table_cell_bold), Paragraph("<b>+11.38%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 7", table_cell), Paragraph("1.0601", table_cell), Paragraph("<b>0.9458</b>", table_cell_bold), Paragraph("<b>+10.78%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 8", table_cell), Paragraph("1.0827", table_cell), Paragraph("<b>0.9723</b>", table_cell_bold), Paragraph("<b>+10.20%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 9", table_cell), Paragraph("1.1017", table_cell), Paragraph("<b>0.9946</b>", table_cell_bold), Paragraph("<b>+9.72%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 10", table_cell), Paragraph("1.1175", table_cell), Paragraph("<b>1.0129</b>", table_cell_bold), Paragraph("<b>+9.36%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 11", table_cell), Paragraph("1.1350", table_cell), Paragraph("<b>1.0195</b>", table_cell_bold), Paragraph("<b>+10.17%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
        [Paragraph("h = 12", table_cell), Paragraph("1.1540", table_cell), Paragraph("<b>1.0154</b>", table_cell_bold), Paragraph("<b>+12.01%</b>", table_cell_bold), Paragraph("<b>FinBERT Hybrid ✅</b>", table_cell_bold)],
    ]
    t_s6 = Table(s6_table_data, colWidths=[55, 65, 75, 50, 85])
    t_s6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a5f")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('ALIGN', (1,0), (3,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
    ]))

    s6_chart = [
        Paragraph("<b>Key Empirical Findings</b>", card_title),
        Spacer(1, 3),
        Paragraph("• <b>FinBERT Hybrid Wins 11 of 12 Horizons:</b> Massive gains at medium and long horizons (+11.38% at h=6, +12.01% at h=12).", body_style),
        Paragraph("• <b>Why News Matters Over Time:</b> At $h=1$, inertia dominates (past month predicts next). At $h=6$ to $12$, numerical momentum decays, and forward-looking news sentiment drives the trajectory.", body_style),
        Spacer(1, 4),
    ]
    if CHART_PATH.exists():
        s6_chart.append(Image(str(CHART_PATH), width=310, height=155))

    t_s6_container = Table([[t_s6, s6_chart]], colWidths=[335, 315])
    t_s6_container.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_s6_container)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 7: Conclusions and Key Learning
    # =========================================================================
    story.append(Paragraph("Slide 7: Conclusions & Key Learning", slide_title_style))
    story.append(Paragraph("Summary of Implementation, Proposed Modifications, and Takeaways for Students", slide_sub_style))
    story.append(Spacer(1, 8))

    c1 = [
        Paragraph("<b>1. What We Learned from the Paper</b>", card_title),
        Spacer(1, 4),
        Paragraph("• Time Series Foundation Models (TSFMs) are genuine competitors to econometric models, but they are not silver bullets when trained purely on past numbers.", body_style),
        Paragraph("• BVAR and TSFM errors are largely orthogonal, proving that neural models and statistical models capture distinct economic phenomena.", body_style),
        Spacer(1, 8),
        Paragraph("<b>2. What Our Modifications Achieved</b>", card_title),
        Spacer(1, 4),
        Paragraph("• <b>Multimodal Superiority:</b> Fusing text embeddings (FinBERT) with foundation time-series forecasts reduced forecasting error by up to <b>12.01%</b>.", body_style),
        Paragraph("• <b>Context Window Matters:</b> Expanding from 36 to 120 months allows Transformers to observe complete boom-and-bust business cycles.", body_style),
    ]

    c2 = [
        Paragraph("<b>3. What Other Students Can Learn</b>", card_title),
        Spacer(1, 4),
        Paragraph("• <b>Data Transformation is Critical:</b> Macro data is non-stationary. Applying McCracken-Ng transforms (second differences for CPI) is non-negotiable for stable convergence.", body_style),
        Paragraph("• <b>Local GPU Hardware is Sufficient:</b> Using PyTorch MPS on Apple Silicon M4 enabled processing 1.57 million articles in ~40 minutes without expensive cloud servers.", body_style),
        Paragraph("• <b>Two-Pass Architecture:</b> Separating foundation inference (TimesFM) from semantic alignment (Ridge Meta-Learner) creates an efficient, modular, and interpretable forecasting pipeline.", body_style),
        Spacer(1, 8),
        Paragraph("<b>4. Reproducibility & Open Source</b>", card_title),
        Spacer(1, 4),
        Paragraph("• Full code, instructions, and reports: <b>github.com/Nilesh-195/-Macroeconomic-Forecasting-with-Large-Language-Models</b>", body_style),
    ]

    t_s7 = Table([[c1, c2]], colWidths=[325, 325])
    t_s7.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#eff6ff")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#cbd5e1")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#93c5fd")),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_s7)

    doc.build(story, canvasmaker=NumberedSlideCanvas)
    print(f"Presentation successfully built at: {OUT_PDF}")

if __name__ == "__main__":
    build_presentation()

