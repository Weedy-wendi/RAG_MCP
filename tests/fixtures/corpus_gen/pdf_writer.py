"""Render a :class:`~.facts.DocPlan` to a native PDF.

Supports the layout families the corpus needs:

* **single column** -- statements, statutes, manuals;
* **two column** -- arXiv-style papers, via ``BaseDocTemplate`` + two ``Frame``s;
* **three column** -- magazine reports;
* **vertical** -- characters top-to-bottom, columns right-to-left (raw canvas).

Plus the page furniture: title page, table of contents, running header/footer,
diagonal transliterated watermark, PDF outline bookmarks, and the five table
difficulty classes (simple / merged / multi-page / borderless / multi-header).

Determinism
-----------
``reportlab.rl_config.invariant`` is enabled and every document template is built
with ``invariant=1``, which pins the creation date and internal document id so
that two runs with the same seed produce byte-identical PDFs.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.art import ArtSpec, draw as draw_art
from tests.fixtures.corpus_gen.facts import (
    AdBlock,
    Block,
    Bullets,
    CodeBlock,
    DocPlan,
    FigureBlock,
    FootnoteBlock,
    FormulaBlock,
    Heading,
    PageBreak as PlanPageBreak,
    Para,
    ReferenceBlock,
    Rule,
    TableBlock,
)
from tests.fixtures.corpus_gen.fonts import FontSet, register_fonts
from tests.fixtures.corpus_gen.fsutil import ensure_dir

# Determinism: pin dates and document ids.
try:
    from reportlab import rl_config

    rl_config.invariant = 1
except Exception:  # pragma: no cover - older reportlab
    pass

PAGE_SIZE = A4
MARGIN_X = 18 * mm
MARGIN_TOP = 20 * mm
MARGIN_BOTTOM = 18 * mm

HEADER_EVERY_N_PAGES_MIN = 3

#: Colour used for the translucent watermark.
WATERMARK_FILL = colors.Color(0.45, 0.45, 0.55, alpha=0.13)

#: Marker placed in every watermark.  Verification checks this marker and the
#: document id separately: a diagonal watermark climbs across the page, and an
#: extractor may emit its halves in either order (and may separate the glyphs of
#: a rotated run with spaces), so requiring one contiguous token is not reliable.
WATERMARK_MARKER = "CONFIDENTIAL"


def watermark_for(doc_id: str) -> str:
    """The watermark string for a document, e.g. ``FIN-01-CONFIDENTIAL``."""
    return f"{doc_id}-{WATERMARK_MARKER}"


@dataclass
class RenderStats:
    """Facts about a rendered PDF, recorded into the manifest."""

    pages: int = 0
    figures: int = 0
    tables: int = 0
    bookmarks: int = 0
    watermark_text: str = ""
    header_text: str = ""
    footer_pattern: str = ""
    bytes_written: int = 0
    vertical: bool = False
    layout: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": self.pages,
            "figures": self.figures,
            "tables": self.tables,
            "bookmarks": self.bookmarks,
            "watermark_text": self.watermark_text,
            "header_text": self.header_text,
            "footer_pattern": self.footer_pattern,
            "bytes_written": self.bytes_written,
            "vertical": self.vertical,
            "layout": self.layout,
        }


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def _body_font_size(layout: str) -> float:
    return {S.TWO_COL: 8.6, S.THREE_COL: 8.0, S.VERTICAL: 11.0}.get(layout, 9.8)


def build_styles(spec: S.DocSpec, fonts: FontSet) -> Dict[str, ParagraphStyle]:
    """Build the paragraph styles for one document."""
    base = getSampleStyleSheet()
    size = _body_font_size(spec.layout)
    leading = size * 1.45
    cjk = fonts.cjk

    def style(name: str, **kw: Any) -> ParagraphStyle:
        params = {
            "name": name,
            "fontName": cjk,
            "fontSize": size,
            "leading": leading,
            "alignment": TA_JUSTIFY,
            "spaceAfter": size * 0.45,
            "textColor": colors.HexColor("#1f2933"),
            "wordWrap": "CJK",
        }
        params.update(kw)
        return ParagraphStyle(**params)

    return {
        "title": style(
            "CorpusTitle", fontSize=size * 2.2, leading=size * 2.6,
            alignment=TA_CENTER, spaceAfter=size, fontName=fonts.cjk_bold,
        ),
        "subtitle": style(
            "CorpusSubtitle", fontSize=size * 1.25, leading=size * 1.7,
            alignment=TA_CENTER, textColor=colors.HexColor("#52606d"),
            spaceAfter=size * 1.4,
        ),
        "h1": style(
            "CorpusH1", fontSize=size * 1.45, leading=size * 1.9, spaceBefore=size * 0.8,
            spaceAfter=size * 0.55, fontName=fonts.cjk_bold, alignment=TA_LEFT,
        ),
        "h2": style(
            "CorpusH2", fontSize=size * 1.2, leading=size * 1.6, spaceBefore=size * 0.6,
            spaceAfter=size * 0.45, fontName=fonts.cjk_bold, alignment=TA_LEFT,
        ),
        "h3": style(
            "CorpusH3", fontSize=size * 1.05, leading=size * 1.5, spaceBefore=size * 0.5,
            spaceAfter=size * 0.4, fontName=fonts.cjk_bold, alignment=TA_LEFT,
        ),
        "h4": style(
            "CorpusH4", fontSize=size, leading=size * 1.4, spaceBefore=size * 0.4,
            spaceAfter=size * 0.35, fontName=fonts.cjk_bold, alignment=TA_LEFT,
        ),
        "h5": style(
            "CorpusH5", fontSize=size * 0.95, leading=size * 1.35, spaceBefore=size * 0.35,
            spaceAfter=size * 0.3, fontName=fonts.cjk_bold, alignment=TA_LEFT,
        ),
        "body": style("CorpusBody"),
        "bullet": style("CorpusBullet", leftIndent=size * 1.3, spaceAfter=size * 0.3),
        "caption": style(
            "CorpusCaption", fontSize=size * 0.85, leading=size * 1.2,
            alignment=TA_CENTER, textColor=colors.HexColor("#52606d"),
            spaceBefore=size * 0.25, spaceAfter=size * 0.6,
        ),
        "code": style(
            "CorpusCode", fontName=fonts.mono, fontSize=size * 0.82,
            leading=size * 1.12, alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"), spaceAfter=0,
        ),
        "formula": style(
            "CorpusFormula", fontSize=size * 1.15, leading=size * 1.6,
            alignment=TA_CENTER, spaceBefore=size * 0.4, spaceAfter=size * 0.6,
        ),
        "footnote": style(
            "CorpusFootnote", fontSize=size * 0.76, leading=size * 1.05,
            textColor=colors.HexColor("#52606d"),
        ),
        "reference": style(
            "CorpusReference", fontSize=size * 0.78, leading=size * 1.1,
            leftIndent=size * 0.9, firstLineIndent=-size * 0.9, spaceAfter=size * 0.15,
        ),
        "ad": style(
            "CorpusAd", fontSize=size * 0.9, leading=size * 1.25,
            fontName=fonts.cjk_bold, textColor=colors.HexColor("#7a4b00"),
        ),
        "table": style(
            "CorpusTableCell", fontSize=max(5.6, size * 0.72),
            leading=max(6.4, size * 0.95), alignment=TA_LEFT, spaceAfter=0,
        ),
    }


HEADING_LEVELS = {"CorpusH1": 0, "CorpusH2": 1, "CorpusH3": 2, "CorpusH4": 3, "CorpusH5": 4}


# ---------------------------------------------------------------------------
# Document template with bookmarks
# ---------------------------------------------------------------------------


class CorpusDocTemplate(BaseDocTemplate):
    """``BaseDocTemplate`` that can emit an outline entry for every heading.

    ``_enable_bookmarks`` is consumed *before* ``super().__init__`` so that the
    private keyword never reaches reportlab, which would reject it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._enable_bookmarks = bool(kwargs.pop("_enable_bookmarks", False))
        self.bookmark_count = 0
        super().__init__(*args, **kwargs)

    def afterFlowable(self, flowable: Any) -> None:  # noqa: D102 - reportlab hook
        if not self._enable_bookmarks:
            return
        if not isinstance(flowable, Paragraph):
            return
        name = getattr(flowable.style, "name", "")
        level = HEADING_LEVELS.get(name)
        if level is None:
            return
        text = flowable.getPlainText()
        if not text.strip():
            return
        key = f"bm{self.bookmark_count}"
        self.bookmark_count += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text[:90], key, level=level, closed=(level > 0))


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------


@dataclass
class Furniture:
    """Header / footer / watermark settings for one document."""

    header: str = ""
    footer: str = ""
    watermark: str = ""
    show: bool = True


def _make_page_hooks(furniture: Furniture, fonts: FontSet):
    """Return ``(on_first, on_later)`` callbacks drawing header/footer/watermark."""

    def _draw(canv: rl_canvas.Canvas, doc: Any) -> None:
        canv.saveState()
        width, height = PAGE_SIZE

        if furniture.watermark:
            canv.saveState()
            canv.setFillColor(WATERMARK_FILL)
            canv.setFont(fonts.cjk_bold, 46)
            canv.translate(width / 2.0, height / 2.0)
            canv.rotate(38)
            canv.drawCentredString(0, 0, furniture.watermark)
            canv.restoreState()

        if furniture.show:
            canv.setFont(fonts.cjk, 7.4)
            canv.setFillColor(colors.HexColor("#7b8794"))
            if furniture.header:
                canv.drawString(MARGIN_X, height - MARGIN_TOP + 8, furniture.header)
                canv.setStrokeColor(colors.HexColor("#cbd2d9"))
                canv.setLineWidth(0.4)
                canv.line(MARGIN_X, height - MARGIN_TOP + 5,
                          width - MARGIN_X, height - MARGIN_TOP + 5)
            if furniture.footer:
                canv.drawCentredString(width / 2.0, MARGIN_BOTTOM - 10,
                                       furniture.footer.replace("{page}", str(doc.page)))
        canv.restoreState()

    return _draw, _draw


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

#: Cell spans per style, as ``((col0,row0),(col1,row1))`` over two header rows.
TABLE_SPANS: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {
    S.TABLE_MERGED: [((0, 0), (0, 1)), ((1, 0), (2, 0)), ((3, 0), (4, 0))],
    S.TABLE_MULTIHEADER: [((1, 0), (2, 0)), ((3, 0), (4, 0))],
}

TABLE_WIDTH_FRACTIONS: Dict[str, Tuple[float, ...]] = {
    S.TABLE_SIMPLE: (0.34, 0.22, 0.22, 0.22),
    S.TABLE_MERGED: (0.24, 0.19, 0.19, 0.19, 0.19),
    S.TABLE_MULTIHEADER: (0.24, 0.19, 0.19, 0.19, 0.19),
    S.TABLE_MULTIPAGE: (0.08, 0.42, 0.14, 0.18, 0.18),
    S.TABLE_BORDERLESS: (0.18, 0.20, 0.22, 0.40),
    S.TABLE_INFO: (0.28, 0.72),
}

HEADER_ROWS: Dict[str, int] = {
    S.TABLE_SIMPLE: 1,
    S.TABLE_MERGED: 2,
    S.TABLE_MULTIHEADER: 2,
    S.TABLE_MULTIPAGE: 1,
    S.TABLE_BORDERLESS: 1,
    S.TABLE_INFO: 0,
}

#: Styles drawn without any grid lines.
_NO_GRID_STYLES = (S.TABLE_BORDERLESS, S.TABLE_INFO)


def build_table_flowable(
    block: TableBlock,
    styles: Dict[str, ParagraphStyle],
    avail_width: float,
    seed: int,
) -> Table:
    """Convert a :class:`TableBlock` into a styled reportlab ``Table``."""
    cell_style = styles["table"]
    n_cols = max(len(row) for row in block.rows)
    header_rows = HEADER_ROWS.get(block.style, 1)

    data: List[List[Any]] = []
    for r, row in enumerate(block.rows):
        out_row: List[Any] = []
        for c in range(n_cols):
            raw = row[c] if c < len(row) else ""
            if r < header_rows and str(raw).strip():
                out_row.append(Paragraph(f"<b>{_esc(raw)}</b>", cell_style))
            else:
                out_row.append(Paragraph(_esc(raw), cell_style))
        data.append(out_row)

    fractions = TABLE_WIDTH_FRACTIONS.get(block.style, TABLE_WIDTH_FRACTIONS[S.TABLE_SIMPLE])
    if len(fractions) != n_cols:
        fractions = tuple(1.0 / n_cols for _ in range(n_cols))
    col_widths = [avail_width * f for f in fractions]

    repeat_rows = 1 if block.style == S.TABLE_MULTIPAGE else 0
    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=repeat_rows,
        splitByRow=1,
        hAlign="LEFT",
    )

    cmds: List[Tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header_rows > 0:
        cmds.append(("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e4e7eb")))

    if block.style in _NO_GRID_STYLES:
        # Structure carried purely by position -- no rules at all.
        if header_rows > 0:
            cmds.append(("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.5,
                         colors.HexColor("#9aa5b1")))
    else:
        cmds.append(("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")))
        if header_rows > 1:
            cmds.append(("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6,
                         colors.HexColor("#616e7c")))
        if block.style == S.TABLE_SIMPLE:
            cmds.append(("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
                         [colors.white, colors.HexColor("#f5f7fa")]))

    for start, end in TABLE_SPANS.get(block.style, []):
        cmds.append(("SPAN", start, end))

    table.setStyle(TableStyle(cmds))
    return table


def _esc(value: Any) -> str:
    """Escape text for reportlab's mini-HTML parser."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Block -> flowable
# ---------------------------------------------------------------------------


def block_to_flowables(
    block: Block,
    spec: S.DocSpec,
    styles: Dict[str, ParagraphStyle],
    fonts: FontSet,
    avail_width: float,
    seed: int,
    counters: Dict[str, int],
) -> List[Any]:
    """Convert one plan block into reportlab flowables."""
    if isinstance(block, Heading):
        style_name = {0: "title", 1: "h1", 2: "h2", 3: "h3", 4: "h4"}.get(block.level, "h5")
        if block.level == 0:
            return [Spacer(1, 34 * mm), Paragraph(_esc(block.text), styles["title"])]
        return [Paragraph(_esc(block.text), styles[style_name])]

    if isinstance(block, Para):
        style = styles.get(block.style, styles["body"])
        return [Paragraph(_esc(block.text), style)]

    if isinstance(block, Bullets):
        items = [
            ListItem(Paragraph(_esc(item), styles["bullet"]), leftIndent=10)
            for item in block.items
        ]
        flowable = ListFlowable(
            items,
            bulletType="1" if block.ordered else "bullet",
            start=1 if block.ordered else None,
            leftIndent=12,
        )
        return [flowable, Spacer(1, 4)]

    if isinstance(block, CodeBlock):
        rows: List[Any] = []
        if block.caption:
            rows.append(Paragraph(_esc(block.caption), styles["caption"]))
        for line in block.lines or [""]:
            rows.append(Paragraph(_esc(line) or "&nbsp;", styles["code"]))
        inner = Table([[r] for r in rows], colWidths=[avail_width])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        return [inner, Spacer(1, 5)]

    if isinstance(block, TableBlock):
        counters["table"] += 1
        flow: List[Any] = []
        if block.caption:
            flow.append(Paragraph(_esc(block.caption), styles["caption"]))
        flow.append(build_table_flowable(block, styles, avail_width, seed))
        flow.append(Spacer(1, 6))
        return flow

    if isinstance(block, FigureBlock):
        counters["figure"] += 1
        rng = random.Random(f"{seed}:{spec.doc_id}:{block.fig_id}")
        spec_art = ArtSpec(
            kind=block.art,
            title=block.art_kwargs.get("title", ""),
            labels=list(block.art_kwargs.get("labels", [])),
            values=list(block.art_kwargs.get("values", [])),
            body=list(block.art_kwargs.get("body", [])),
        )
        png = draw_art(spec_art, rng)
        target_w = avail_width * block.width_ratio
        try:
            iw, ih = ImageReader(io.BytesIO(png)).getSize()
            target_h = target_w * (ih / float(iw)) if iw else target_w * 0.55
        except Exception:  # noqa: BLE001
            target_h = target_w * 0.55
        flow = [Image(io.BytesIO(png), width=target_w, height=target_h)]
        if block.caption:
            flow.append(Paragraph(_esc(block.caption), styles["caption"]))
        flow.append(Spacer(1, 4))
        return flow

    if isinstance(block, FormulaBlock):
        return [Paragraph(_esc(block.text), styles["formula"])]

    if isinstance(block, FootnoteBlock):
        flow = [Spacer(1, 3)]
        for item in block.items:
            flow.append(Paragraph(_esc(item), styles["footnote"]))
        flow.append(Spacer(1, 3))
        return flow

    if isinstance(block, ReferenceBlock):
        return [Paragraph(_esc(item), styles["reference"]) for item in block.items]

    if isinstance(block, AdBlock):
        rows: List[Any] = [Paragraph(_esc(block.headline), styles["ad"])]
        rows.extend(Paragraph(_esc(line), styles["footnote"]) for line in block.lines)
        inner = Table([[r] for r in rows], colWidths=[avail_width])
        inner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8e6")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e0a800")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return [inner, Spacer(1, 7)]

    if isinstance(block, Rule):
        return [Spacer(1, 4)]

    if isinstance(block, PlanPageBreak):
        return [PageBreak()]

    return []


# ---------------------------------------------------------------------------
# Vertical rendering
# ---------------------------------------------------------------------------


def render_vertical(
    plan: DocPlan,
    spec: S.DocSpec,
    out_path: Path,
    fonts: FontSet,
    furniture: Furniture,
) -> RenderStats:
    """Render a vertical (top-to-bottom, right-to-left) document.

    Blocks are grouped by page breaks and each group is laid out on exactly one
    page, so the page count matches ``spec.pages`` by construction.
    """
    stats = RenderStats(layout=spec.layout, vertical=True)
    stats.watermark_text = furniture.watermark
    stats.header_text = furniture.header
    stats.footer_pattern = furniture.footer

    groups = _split_on_pagebreaks(plan.body)
    canv = rl_canvas.Canvas(str(out_path), pagesize=PAGE_SIZE, invariant=1)
    canv.setTitle(spec.title)
    canv.setAuthor("Corpus Generator")
    canv.setSubject(spec.category)

    width, height = PAGE_SIZE
    char_size = 11.0
    line_gap = char_size * 1.28
    col_gap = char_size * 2.1
    top = height - MARGIN_TOP - 6
    bottom = MARGIN_BOTTOM + 6
    right = width - MARGIN_X
    left = MARGIN_X

    for page_index, group in enumerate(groups, start=1):
        if page_index > 1:
            canv.showPage()

        if furniture.watermark:
            canv.saveState()
            canv.setFillColor(WATERMARK_FILL)
            canv.setFont(fonts.cjk_bold, 46)
            canv.translate(width / 2.0, height / 2.0)
            canv.rotate(38)
            canv.drawCentredString(0, 0, furniture.watermark)
            canv.restoreState()

        if furniture.show:
            canv.setFont(fonts.cjk, 7.4)
            canv.setFillColor(colors.HexColor("#7b8794"))
            canv.drawString(MARGIN_X, height - MARGIN_TOP + 8, furniture.header)
            canv.drawCentredString(width / 2.0, MARGIN_BOTTOM - 10,
                                   furniture.footer.replace("{page}", str(page_index)))

        texts: List[Tuple[str, str]] = []
        for block in group:
            if isinstance(block, Heading):
                texts.append((block.text, "bold"))
            elif isinstance(block, Para):
                texts.append((block.text, "body"))
            elif isinstance(block, Bullets):
                texts.extend((f"・{item}", "body") for item in block.items)
            elif isinstance(block, (FootnoteBlock, ReferenceBlock)):
                texts.extend((item, "small") for item in block.items)
            elif isinstance(block, FormulaBlock):
                texts.append((block.text, "body"))
            elif isinstance(block, CodeBlock):
                texts.extend((line, "small") for line in block.lines)
            elif isinstance(block, AdBlock):
                texts.append((block.headline, "bold"))
                texts.extend((line, "small") for line in block.lines)
            elif isinstance(block, TableBlock):
                for row in block.rows:
                    texts.append((" ｜ ".join(str(c) for c in row if str(c).strip()), "small"))
                if block.caption:
                    texts.append((block.caption, "small"))
            elif isinstance(block, FigureBlock):
                stats.figures += 1
                # Figures are drawn as a reserved block; vertical pages have room.
                texts.append((block.caption, "small"))

        col_x = right
        cursor_y = top
        for text, kind in texts:
            size = {"bold": char_size * 1.18, "body": char_size, "small": char_size * 0.84}[kind]
            font = fonts.cjk_bold if kind == "bold" else fonts.cjk
            gap = size * 1.3
            i = 0
            while i < len(text):
                if cursor_y - gap < bottom:
                    col_x -= col_gap + size
                    cursor_y = top
                    if col_x < left:
                        break
                ch = text[i]
                canv.setFont(font, size)
                canv.setFillColor(colors.HexColor("#1f2933"))
                canv.drawString(col_x, cursor_y - size, ch)
                cursor_y -= gap
                i += 1
            cursor_y -= gap * 0.55
            if col_x < left:
                break

    canv.save()
    stats.pages = max(1, len(groups))
    stats.bytes_written = out_path.stat().st_size
    return stats


def _split_on_pagebreaks(blocks: Sequence[Block]) -> List[List[Block]]:
    """Group blocks into pages, dropping empty trailing groups."""
    groups: List[List[Block]] = [[]]
    for block in blocks:
        if isinstance(block, PlanPageBreak):
            groups.append([])
        else:
            groups[-1].append(block)
    return [g for g in groups if g] or [[]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def render_pdf(
    plan: DocPlan,
    spec: S.DocSpec,
    out_path: Path,
    seed: int = 1337,
) -> RenderStats:
    """Render *plan* to *out_path* and return statistics about the result."""
    fonts = register_fonts()
    ensure_dir(Path(out_path).parent)

    version = f"v{1 + (spec.seq % 4)}.{spec.seq % 10}"
    furniture = Furniture(
        header=f"{spec.doc_id} · {spec.title[:60]} · {version}",
        footer=f"{spec.doc_id} | 第 {{page}} 页 | {version} | 内部资料 请勿外传"
        if spec.language != S.LANG_EN
        else f"{spec.doc_id} | page {{page}} | {version} | internal use only",
        # A single whitespace-free token, so a rotated run cannot be split by the
        # extractor inserting spaces between glyphs.
        watermark=watermark_for(spec.doc_id) if spec.watermark else "",
        show=spec.pages >= HEADER_EVERY_N_PAGES_MIN,
    )

    if spec.layout == S.VERTICAL:
        return render_vertical(plan, spec, Path(out_path), fonts, furniture)

    styles = build_styles(spec, fonts)
    # ``BaseDocTemplate`` takes page hooks on the PageTemplate, not on build().
    _, on_page = _make_page_hooks(furniture, fonts)

    columns = spec.columns
    if columns == 1:
        frame = Frame(
            MARGIN_X, MARGIN_BOTTOM,
            PAGE_SIZE[0] - 2 * MARGIN_X,
            PAGE_SIZE[1] - MARGIN_TOP - MARGIN_BOTTOM,
            id="main", showBoundary=0,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        frames = [frame]
    else:
        gap = 7 * mm
        usable = PAGE_SIZE[0] - 2 * MARGIN_X
        frame_w = (usable - gap * (columns - 1)) / columns
        frame_h = PAGE_SIZE[1] - MARGIN_TOP - MARGIN_BOTTOM
        frames = [
            Frame(
                MARGIN_X + i * (frame_w + gap), MARGIN_BOTTOM, frame_w, frame_h,
                id=f"col{i}", showBoundary=0,
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
            )
            for i in range(columns)
        ]

    doc = CorpusDocTemplate(
        str(out_path),
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title=spec.title,
        author="Corpus Generator",
        subject=S.CATEGORY_NAMES[spec.category],
        invariant=1,
        _enable_bookmarks=spec.bookmarks,
    )
    doc.addPageTemplates([PageTemplate(id="page", frames=frames, onPage=on_page)])

    avail_width = frames[0].width
    counters = {"table": 0, "figure": 0}

    story: List[Any] = []
    for block in plan.body:
        story.extend(
            block_to_flowables(block, spec, styles, fonts, avail_width, seed, counters)
        )
    # A trailing page break would emit an extra blank page.
    while story and isinstance(story[-1], PageBreak):
        story.pop()
    if not story:
        story = [Paragraph("(empty)", styles["body"])]

    doc.build(story)

    stats = RenderStats(
        pages=doc.page,
        figures=counters["figure"],
        tables=counters["table"],
        bookmarks=doc.bookmark_count,
        watermark_text=furniture.watermark,
        header_text=furniture.header,
        footer_pattern=furniture.footer,
        bytes_written=Path(out_path).stat().st_size,
        vertical=False,
        layout=spec.layout,
    )
    return stats
