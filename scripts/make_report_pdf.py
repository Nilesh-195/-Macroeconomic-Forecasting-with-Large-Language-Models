"""
make_report_pdf.py  –  Render STAGE1_REPORT.md to STAGE1_REPORT_FULL.pdf
using ReportLab with proper table support.
"""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MD_FILE = ROOT / "STAGE1_REPORT.md"
PDF_FILE = ROOT / "STAGE1_REPORT_FULL.pdf"

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────
NAVY = colors.HexColor("#1a2e4a")
STEEL = colors.HexColor("#2d5f8a")
LIGHT_BLUE = colors.HexColor("#dce8f5")
ALT_ROW = colors.HexColor("#f4f8fc")
HEADER_BG = colors.HexColor("#1e3a5f")
SECTION_BG = colors.HexColor("#e8f0f8")
CODE_BG = colors.HexColor("#f5f5f5")
BOLD_GREEN = colors.HexColor("#0a6e2e")  # values < 1.000
WHITE = colors.white
BLACK = colors.black
DIVIDER = colors.HexColor("#b0c8e0")

# ──────────────────────────────────────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────────────────────────────────────
SS = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "title",
    parent=SS["Title"],
    fontSize=20,
    textColor=NAVY,
    spaceAfter=6,
    leading=24,
)
SUBTITLE_STYLE = ParagraphStyle(
    "subtitle",
    parent=SS["Normal"],
    fontSize=10,
    textColor=STEEL,
    spaceAfter=14,
    leading=14,
)
H2_STYLE = ParagraphStyle(
    "h2",
    parent=SS["Heading2"],
    fontSize=13,
    textColor=WHITE,
    backColor=HEADER_BG,
    spaceBefore=14,
    spaceAfter=6,
    leading=18,
    leftIndent=-6,
    rightIndent=-6,
    borderPad=4,
)
H3_STYLE = ParagraphStyle(
    "h3",
    parent=SS["Heading3"],
    fontSize=11,
    textColor=NAVY,
    spaceBefore=10,
    spaceAfter=4,
    leading=14,
    borderPad=2,
)
BODY_STYLE = ParagraphStyle(
    "body",
    parent=SS["Normal"],
    fontSize=9.5,
    leading=14,
    spaceAfter=6,
    textColor=BLACK,
)
BULLET_STYLE = ParagraphStyle(
    "bullet",
    parent=BODY_STYLE,
    leftIndent=14,
    bulletIndent=4,
    spaceAfter=3,
)
CODE_STYLE = ParagraphStyle(
    "code",
    parent=SS["Code"],
    fontSize=8,
    leading=11,
    backColor=CODE_BG,
    leftIndent=10,
    rightIndent=10,
    spaceAfter=8,
    spaceBefore=4,
    fontName="Courier",
)
CAPTION_STYLE = ParagraphStyle(
    "caption",
    parent=BODY_STYLE,
    fontSize=8.5,
    textColor=STEEL,
    alignment=1,  # centre
    spaceAfter=4,
    spaceBefore=2,
)
ITALIC_STYLE = ParagraphStyle(
    "italic_note",
    parent=BODY_STYLE,
    fontSize=8.5,
    textColor=colors.HexColor("#444444"),
    spaceAfter=6,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def inline_fmt(text: str) -> str:
    """Convert **bold**, *italic*, and `code` to ReportLab XML safely."""
    # First protect backtick spans (code), then bold, then italic
    # We use a two-pass approach: extract segments, format, rejoin

    # Step 1: Split on backtick code spans first to avoid mangling them
    parts = re.split(r"(`[^`]+`)", text)
    result = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            inner = part[1:-1]
            # Escape XML in code spans
            inner = inner.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            result.append(f"<font name='Courier' size='8.5'>{inner}</font>")
        else:
            # Escape XML special chars in plain text parts
            part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Bold before italic to avoid overlapping tags
            part = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", part)
            part = re.sub(r"\*(.+?)\*", r"<i>\1</i>", part)
            result.append(part)
    return "".join(result)


def make_table(rows: list[list[str]], has_header: bool = True) -> Table:
    """Render a list of string rows as a styled ReportLab Table."""
    # Convert inline markdown in every cell
    fmt_rows = []
    for i, row in enumerate(rows):
        fmt_row = []
        for cell in row:
            cell = cell.strip()
            style = BODY_STYLE if i > 0 else ParagraphStyle(
                "th", parent=BODY_STYLE, textColor=WHITE, fontName="Helvetica-Bold", fontSize=9
            )
            fmt_row.append(Paragraph(inline_fmt(cell), style))
        fmt_rows.append(fmt_row)

    col_count = len(rows[0])
    avail_width = A4[0] - 4 * cm  # page minus margins
    col_width = avail_width / col_count

    # Wider first column for model/aspect tables
    if col_count in (2, 3, 5):
        col_widths = [avail_width * 0.38] + [(avail_width * 0.62) / (col_count - 1)] * (col_count - 1)
    else:
        col_widths = [col_width] * col_count

    tbl = Table(fmt_rows, colWidths=col_widths, repeatRows=1 if has_header else 0)

    style_cmds = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        # Body rows
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        # Alternating row colours
        *[
            ("BACKGROUND", (0, r), (-1, r), ALT_ROW if r % 2 == 0 else WHITE)
            for r in range(1, len(rows))
        ],
        # Section sub-header rows (cells that span / are "bold section label rows")
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, DIVIDER),
        ("ROWBACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.8, STEEL),
    ]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ──────────────────────────────────────────────────────────────────────────────
# Markdown parser  →  ReportLab flowables
# ──────────────────────────────────────────────────────────────────────────────

def parse_markdown(text: str):
    lines = text.splitlines()
    flowables = []
    i = 0

    def flush_paragraph(buf):
        content = " ".join(buf).strip()
        if content.startswith("*") and content.endswith("*") and content.count("*") == 2:
            flowables.append(Paragraph(inline_fmt(content[1:-1]), ITALIC_STYLE))
        else:
            flowables.append(Paragraph(inline_fmt(content), BODY_STYLE))

    while i < len(lines):
        line = lines[i]

        # ── Blank line ──────────────────────────────────────────────────────
        if line.strip() == "":
            i += 1
            continue

        # ── Horizontal rule ─────────────────────────────────────────────────
        if re.match(r"^-{3,}$", line.strip()):
            flowables.append(HRFlowable(width="100%", thickness=0.6, color=DIVIDER, spaceAfter=4))
            i += 1
            continue

        # ── H1 ─────────────────────────────────────────────────────────────
        if line.startswith("# ") and not line.startswith("## "):
            flowables.append(Paragraph(inline_fmt(line[2:]), TITLE_STYLE))
            i += 1
            continue

        # ── H2 ─────────────────────────────────────────────────────────────
        if line.startswith("## "):
            flowables.append(Spacer(1, 6))
            flowables.append(Paragraph("  " + inline_fmt(line[3:]), H2_STYLE))
            i += 1
            continue

        # ── H3 ─────────────────────────────────────────────────────────────
        if line.startswith("### "):
            flowables.append(Paragraph(inline_fmt(line[4:]), H3_STYLE))
            i += 1
            continue

        # ── Code block ──────────────────────────────────────────────────────
        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # closing ```
            code_text = "\n".join(code_lines)
            flowables.append(Preformatted(code_text, CODE_STYLE))
            continue

        # ── Markdown table ───────────────────────────────────────────────────
        if "|" in line and line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            # Drop separator row(s)  (lines like |:---|:---:|)
            rows = []
            for tl in table_lines:
                if re.match(r"^\|[\s\-:|]+\|[\s\-:|]*$", tl.strip()):
                    continue
                cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                flowables.append(Spacer(1, 4))
                flowables.append(make_table(rows))
                flowables.append(Spacer(1, 6))
            continue

        # ── Bullet list ──────────────────────────────────────────────────────
        if re.match(r"^[-*]\s", line):
            content = line[2:].strip()
            flowables.append(Paragraph("• " + inline_fmt(content), BULLET_STYLE))
            i += 1
            continue

        # ── Numbered list ────────────────────────────────────────────────────
        m = re.match(r"^(\d+)\.\s(.*)", line)
        if m:
            content = m.group(2).strip()
            flowables.append(Paragraph(f"{m.group(1)}. " + inline_fmt(content), BULLET_STYLE))
            i += 1
            continue

        # ── Bold paragraph (starts with **) ──────────────────────────────────
        if line.startswith("**") or line.startswith("*"):
            flowables.append(Paragraph(inline_fmt(line), BODY_STYLE))
            i += 1
            continue

        # ── Caption-like lines (Table N. / Figure N.) ──────────────────────
        if re.match(r"^\*\*(Table|Figure)\s+\d+", line):
            flowables.append(Paragraph(inline_fmt(line), CAPTION_STYLE))
            i += 1
            continue

        # ── Regular paragraph (may span multiple lines) ──────────────────────
        buf = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") \
              and not lines[i].startswith("```") and "|" not in lines[i] \
              and not re.match(r"^[-*]\s", lines[i]) and not re.match(r"^-{3,}$", lines[i].strip()):
            buf.append(lines[i])
            i += 1
        flush_paragraph(buf)

    return flowables


# ──────────────────────────────────────────────────────────────────────────────
# Page template with header/footer
# ──────────────────────────────────────────────────────────────────────────────

def build_pdf(md_text: str, out_path: Path):
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2 * cm,
    )

    def header_footer(canvas, doc):
        canvas.saveState()
        w, h = A4
        # Header bar
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 1.5 * cm, w, 1.5 * cm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(2 * cm, h - 1 * cm, "Stage 1 Report — Macroeconomic Forecasting with LLMs")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 2 * cm, h - 1 * cm, f"Page {doc.page}")
        # Footer
        canvas.setFillColor(DIVIDER)
        canvas.rect(0, 0, w, 0.8 * cm, fill=1, stroke=0)
        canvas.setFillColor(STEEL)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(2 * cm, 0.28 * cm, "Carriero, Pettenuzzo & Shekhar (2026)  ·  arXiv:2407.00890v5")
        canvas.restoreState()

    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="main",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=header_footer)])

    flowables = parse_markdown(md_text)
    doc.build(flowables)
    print(f"PDF written to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    md_text = MD_FILE.read_text(encoding="utf-8")
    build_pdf(md_text, PDF_FILE)
