"""Render submission.md to submission.pdf in the sprint template's layout.

Run from the repo root:  uv run --with weasyprint --with markdown python scripts/build_pdf.py
The title, author block, and abstract are lifted from the markdown itself, so the PDF
regenerates faithfully after any edit to submission.md.
"""

import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "submission.md"
OUT = ROOT / "submission.pdf"

text = SRC.read_text(encoding="utf-8")
text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

title = re.search(r"^# (.+)$", text, flags=re.M).group(1).strip()
authors = re.search(r"^\*\*Authors:\*\* (.+)$", text, flags=re.M).group(1).strip()
abstract = text.split("## Abstract", 1)[1].split("## 1.", 1)[0].strip()
body_md = "## 1." + text.split("## 1.", 1)[1]

md = markdown.Markdown(extensions=["tables", "attr_list"])
abstract_html = md.convert(abstract)
md.reset()
body_html = md.convert(body_md)

def author_cell(entry):
    entry = entry.strip()
    m = re.match(r"(.+?)\s*\((.+?)\)$", entry)
    name, affil = (m.group(1), m.group(2)) if m else (entry, "Independent")
    return f'<div class="author"><div class="name">{name}</div><div class="affil">{affil}</div></div>'

author_cells = "".join(author_cell(a) for a in authors.split(","))

html = f"""
<html><head><meta charset="utf-8"><style>
@page {{
  size: A4; margin: 2.2cm 2.4cm;
  @bottom-center {{ content: counter(page); font-family: Georgia, serif; font-size: 9pt; color: #444; }}
}}
body {{ font-family: Georgia, 'Century Schoolbook L', serif; font-size: 10pt; line-height: 1.42; color: #111; }}
.toprule {{ border-top: 2.4pt solid #111; margin: 0 0 26pt 0; }}
h1.title {{ text-align: center; font-size: 17pt; letter-spacing: 0.4pt; margin: 0 0 6pt 0; }}
h1.title sup {{ font-size: 10pt; }}
.titlerule {{ border-top: 0.8pt solid #111; margin: 14pt 0 22pt 0; }}
.authors {{ display: flex; justify-content: center; gap: 48pt; margin-bottom: 14pt; }}
.author {{ text-align: center; }}
.author .name {{ font-size: 11pt; }}
.author .affil {{ font-size: 10pt; color: #333; }}
.with {{ text-align: center; margin: 4pt 0 18pt 0; font-size: 10.5pt; }}
.with b {{ display: block; }}
h2.abs {{ text-align: center; font-size: 12pt; margin: 6pt 0 8pt 0; }}
.abstract {{ margin: 0 42pt 10pt 42pt; text-align: justify; font-size: 9.5pt; }}
.footnote {{ font-size: 8.5pt; border-top: 0.6pt solid #777; padding-top: 3pt; margin-top: 24pt; width: 55%; }}
h2 {{ font-size: 13pt; margin: 16pt 0 6pt 0; page-break-after: avoid; }}
h3 {{ font-size: 11pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }}
h4 {{ font-size: 10pt; margin: 10pt 0 3pt 0; page-break-after: avoid; }}
p {{ margin: 0 0 7pt 0; text-align: justify; }}
ul, ol {{ margin: 0 0 7pt 0; padding-left: 20pt; }}
li {{ margin-bottom: 3pt; text-align: justify; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; }}
table {{ border-collapse: collapse; margin: 8pt auto; font-size: 9pt; font-variant-numeric: tabular-nums; }}
th, td {{ border-top: 0.6pt solid #999; border-bottom: 0.6pt solid #999; padding: 3pt 9pt; }}
thead th {{ border-top: 1.1pt solid #111; border-bottom: 0.7pt solid #111; }}
tbody tr:last-child td {{ border-bottom: 1.1pt solid #111; }}
img {{ display: block; max-width: 88%; margin: 10pt auto 4pt auto; }}
img + em, p > em:only-child {{ display: block; }}
em {{ }}
p em:only-child {{ font-size: 8.8pt; text-align: justify; }}
a {{ color: #111; text-decoration: none; }}
</style></head><body>
<div class="toprule"></div>
<h1 class="title">{title}<sup>1</sup></h1>
<div class="titlerule"></div>
<div class="authors">{author_cells}</div>
<div class="with"><b>With</b>Apart Research</div>
<h2 class="abs">Abstract</h2>
<div class="abstract">{abstract_html}</div>
<div class="footnote"><sup>1</sup> Research conducted at the Digital Minds Research Sprint, August 2026.</div>
<div style="page-break-after: always;"></div>
{body_html}
</body></html>
"""

html = re.sub(r"10\^(-?\d+)", lambda m: f"10<sup>{m.group(1).replace('-', '−')}</sup>", html)

HTML(string=html, base_url=str(ROOT)).write_pdf(OUT)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
