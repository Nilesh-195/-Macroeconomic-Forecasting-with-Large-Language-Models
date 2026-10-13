"""Convert a markdown report to PDF. Pure-Python (markdown + xhtml2pdf),
no system deps (LaTeX/wkhtmltopdf) needed.
"""
import sys
import markdown
from xhtml2pdf import pisa

CSS = """
<style>
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.4; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 14pt; margin-top: 18px; color: #222; }
h3 { font-size: 12pt; margin-top: 14px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9pt; }
th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; }
th { background-color: #eee; }
code { background-color: #f2f2f2; padding: 1px 3px; font-family: Courier; }
pre { background-color: #f2f2f2; padding: 8px; font-family: Courier; font-size: 8pt; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 10px; color: #555; }
</style>
"""


def convert(src_path, dst_path):
    with open(src_path) as f:
        md_text = f.read()
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html = f"<html><head>{CSS}</head><body>{body}</body></html>"
    with open(dst_path, "wb") as out:
        result = pisa.CreatePDF(html, dest=out)
    if result.err:
        raise SystemExit(f"PDF conversion failed for {src_path}")
    print(f"saved {dst_path}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
