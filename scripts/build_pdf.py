"""Render submission.md to submission.pdf in the sprint template's layout.

Run from the repo root:  uv run --with weasyprint --with markdown python scripts/build_pdf.py
The title, author block, and abstract are lifted from the markdown itself, so the PDF
regenerates faithfully after any edit to submission.md.

Layout follows the official sprint template: Old Standard TT throughout (vendored in
fonts/, OFL), a title page with thick rule / title / thin rule / author grid / "With
Apart Research" / centered Abstract and a sprint footnote pinned to the bottom of page 1.
"""

import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "submission.md"
OUT = ROOT / "submission.pdf"

# Page geometry (A4). The title page pins its footnote to the bottom of the text block,
# so the content height has to be stated explicitly rather than inherited from `height:100%`.
PAGE_H_PT = 841.89
MARGIN_TOP_PT = 56.69  # 2.0cm
MARGIN_BOTTOM_PT = 56.69
TITLEPAGE_H_PT = PAGE_H_PT - MARGIN_TOP_PT - MARGIN_BOTTOM_PT

text = SRC.read_text(encoding="utf-8")
text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

title = re.search(r"^# (.+)$", text, flags=re.M).group(1).strip()
authors = re.search(r"^\*\*Authors:\*\* (.+)$", text, flags=re.M).group(1).strip()
abstract = text.split("## Abstract", 1)[1].split("## 1.", 1)[0].strip()
body_md = "## 1." + text.split("## 1.", 1)[1]

# Smart quotes only: dashes and ellipses are left exactly as written so that no
# character carrying meaning (ranges, minus signs, the human TODO) is rewritten.
md = markdown.Markdown(
    extensions=["tables", "attr_list", "smarty"],
    extension_configs={"smarty": {"smart_dashes": False, "smart_ellipses": False}},
)
abstract_html = md.convert(abstract)
md.reset()
body_html = md.convert(body_md)
md.reset()
title_html = re.sub(r"</?p>", "", md.convert(title)).strip()


def author_cell(entry):
    entry = entry.strip()
    m = re.match(r"(.+?)\s*\((.+?)\)$", entry)
    name, affil = (m.group(1), m.group(2)) if m else (entry, "Independent")
    return f'<div class="author"><div class="name">{name}</div><div class="affil">{affil}</div></div>'


author_cells = "".join(author_cell(a) for a in authors.split(","))


def wrap_figures(html):
    """Bind each image to the italic caption that follows it.

    Markdown emits the caption either inside the image's own paragraph (caption on the
    next source line) or as a sibling paragraph (blank line between them); both forms
    let a page break fall between figure and caption. Wrapping the pair in one <figure>
    lets `break-inside: avoid` hold them together.
    """
    with_caption = re.compile(
        r"<p>\s*(<img\b[^>]*>)\s*(?:</p>\s*<p>\s*)?<em>(.*?)</em>\s*</p>", re.DOTALL
    )
    html = with_caption.sub(
        lambda m: f'<figure class="fig">{m.group(1)}'
        f"<figcaption>{m.group(2)}</figcaption></figure>",
        html,
    )
    # Any image left without a caption still gets the no-split wrapper.
    lone = re.compile(r"<p>\s*(<img\b[^>]*>)\s*</p>", re.DOTALL)
    return lone.sub(lambda m: f'<figure class="fig">{m.group(1)}</figure>', html)


body_html = wrap_figures(body_html)

FONT_STACK = "'Old Standard TT', Georgia, 'Century Schoolbook L', serif"

html = f"""
<html><head><meta charset="utf-8"><title>{title}</title><style>
@font-face {{
  font-family: 'Old Standard TT';
  src: url('fonts/OldStandard-Regular.ttf') format('truetype');
  font-weight: normal; font-style: normal;
}}
@font-face {{
  font-family: 'Old Standard TT';
  src: url('fonts/OldStandard-Bold.ttf') format('truetype');
  font-weight: bold; font-style: normal;
}}
@font-face {{
  font-family: 'Old Standard TT';
  src: url('fonts/OldStandard-Italic.ttf') format('truetype');
  font-weight: normal; font-style: italic;
}}
@page {{
  size: A4; margin: 2.0cm 2.4cm;
  @bottom-center {{ content: counter(page); font-family: {FONT_STACK}; font-size: 9pt; color: #444; }}
}}
body {{ font-family: {FONT_STACK}; font-size: 10pt; line-height: 1.42; color: #111;
        orphans: 2; widows: 2; }}

/* --- title page ------------------------------------------------------------ */
.titlepage {{ position: relative; height: {TITLEPAGE_H_PT:.2f}pt; }}
.toprule {{ border-top: 2.4pt solid #111; margin: 0 0 26pt 0; }}
h1.title {{ text-align: center; font-size: 18pt; font-weight: bold; letter-spacing: 0.3pt;
            line-height: 1.24; margin: 0 0 6pt 0; }}
h1.title sup {{ font-size: 9pt; }}
.titlerule {{ border-top: 0.8pt solid #111; margin: 14pt 0 24pt 0; }}
.authors {{ display: flex; justify-content: center; flex-wrap: wrap; gap: 16pt 52pt;
            margin-bottom: 16pt; }}
.author {{ text-align: center; }}
.author .name {{ font-size: 11.5pt; }}
.author .affil {{ font-size: 10pt; color: #333; }}
.with {{ text-align: center; margin: 6pt 0 20pt 0; font-size: 10.5pt; line-height: 1.5; }}
.with b {{ display: block; }}
h2.abs {{ text-align: center; font-size: 12.5pt; margin: 8pt 0 8pt 0; }}
.abstract {{ margin: 0 40pt; text-align: justify; font-size: 9.8pt; line-height: 1.42; }}
.abstract p {{ margin: 0 0 6pt 0; }}
.footnote {{ position: absolute; left: 0; bottom: 0; width: 66%;
             font-size: 8.6pt; line-height: 1.35;
             border-top: 0.6pt solid #777; padding-top: 4pt; }}

/* --- body ------------------------------------------------------------------ */
h2 {{ font-size: 13pt; margin: 17pt 0 6pt 0; page-break-after: avoid; break-after: avoid;
      page-break-inside: avoid; break-inside: avoid; }}
h3 {{ font-size: 11.2pt; margin: 13pt 0 4pt 0; page-break-after: avoid; break-after: avoid;
      page-break-inside: avoid; break-inside: avoid; }}
h4 {{ font-size: 10.3pt; margin: 11pt 0 3pt 0; page-break-after: avoid; break-after: avoid;
      page-break-inside: avoid; break-inside: avoid; }}
p {{ margin: 0 0 6pt 0; text-align: justify; overflow-wrap: break-word; }}
ul, ol {{ margin: 0 0 6pt 0; padding-left: 20pt; }}
li {{ margin-bottom: 2.5pt; text-align: justify; overflow-wrap: break-word;
      page-break-inside: avoid; break-inside: avoid; }}
sup {{ font-size: 0.72em; line-height: 0; vertical-align: super; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }}

/* --- tables ---------------------------------------------------------------- */
table {{ border-collapse: collapse; margin: 9pt auto 10pt auto; font-size: 9.3pt;
         font-variant-numeric: tabular-nums;
         page-break-inside: avoid; break-inside: avoid; }}
th, td {{ border-top: 0.6pt solid #999; border-bottom: 0.6pt solid #999; padding: 3pt 9pt; }}
thead {{ display: table-header-group; }}
thead th {{ border-top: 1.1pt solid #111; border-bottom: 0.7pt solid #111; }}
tbody tr:last-child td {{ border-bottom: 1.1pt solid #111; }}
tr {{ page-break-inside: avoid; break-inside: avoid; }}

/* --- figures --------------------------------------------------------------- */
figure.fig {{ margin: 9pt 0 9pt 0; text-align: center;
              page-break-inside: avoid; break-inside: avoid; }}
/* the width also guards any image wrap_figures() did not match */
img {{ display: block; width: 76%; margin: 8pt auto 6pt auto; }}
img[src*="fig3"], img[src*="fig4"] {{ width: 88%; }}
figure.fig img {{ margin: 0 auto 6pt auto; }}
figcaption {{ font-size: 9pt; font-style: italic; line-height: 1.38; text-align: justify; }}
a {{ color: #111; text-decoration: none; }}
</style></head><body>
<div class="titlepage">
  <div class="toprule"></div>
  <h1 class="title">{title_html}<sup>1</sup></h1>
  <div class="titlerule"></div>
  <div class="authors">{author_cells}</div>
  <div class="with"><b>With</b>Apart Research</div>
  <h2 class="abs">Abstract</h2>
  <div class="abstract">{abstract_html}</div>
  <div class="footnote"><sup>1</sup> Research conducted at the Digital Minds Research
  Sprint, August 2026.</div>
</div>
<div style="page-break-after: always;"></div>
{body_html}
</body></html>
"""

html = re.sub(r"10\^(-?\d+)", lambda m: f"10<sup>{m.group(1).replace('-', '−')}</sup>", html)

HTML(string=html, base_url=str(ROOT)).write_pdf(OUT)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
