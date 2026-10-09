"""Deterministic vector-art renderer built on Pillow.

Produces the raster images embedded in the corpus: flow charts, bar/pie/line
charts, mock photographs, UI screenshots and formula images.

Everything is drawn at 2x and downsampled, which gives cheap anti-aliasing.
All randomness comes from a caller-supplied :class:`random.Random`, so a given
``(spec, seed)`` pair always yields byte-identical PNG output.

Every function returns PNG **bytes**; ``pdf_writer`` wraps them in a reportlab
``Image`` flowable.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tests.fixtures.corpus_gen.fonts import CJK_CANDIDATES

SCALE = 2  # supersampling factor

PALETTE: Tuple[str, ...] = (
    "#2c6fbb", "#d9534f", "#5cb85c", "#f0ad4e",
    "#8e6fbf", "#17a2b8", "#e377c2", "#7f7f7f",
)
INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#d7dbe0"
PAPER = "#ffffff"

_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}
_CJK_TTF: Optional[Path] = None


def _cjk_ttf() -> Optional[Path]:
    """First existing CJK TrueType file, or ``None``."""
    global _CJK_TTF
    if _CJK_TTF is None:
        for path, _index in CJK_CANDIDATES:
            if Path(path).exists():
                _CJK_TTF = Path(path)
                break
    return _CJK_TTF


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """A TrueType font at *size*, falling back to Pillow's bitmap default."""
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    ttf = _cjk_ttf()
    font: ImageFont.ImageFont
    if ttf is None:
        font = ImageFont.load_default()
    else:
        # .ttc files need an explicit index; PIL accepts one via ``index``.
        try:
            font = ImageFont.truetype(str(ttf), size, index=0)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font  # type: ignore[assignment]
    return font


@dataclass
class ArtSpec:
    """Description of one image to draw."""

    kind: str = "flow"
    title: str = ""
    labels: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    body: List[str] = field(default_factory=list)
    width: int = 900
    height: int = 540
    accent: str = PALETTE[0]


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def _canvas(spec: ArtSpec, background: str = PAPER) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (spec.width * SCALE, spec.height * SCALE), background)
    return img, ImageDraw.Draw(img)


def _finish(img: Image.Image, spec: ArtSpec, frame: bool = True) -> bytes:
    if frame:
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [1 * SCALE, 1 * SCALE, (spec.width - 1) * SCALE, (spec.height - 1) * SCALE],
            outline="#c3c9d0",
            width=SCALE,
        )
    img = img.resize((spec.width, spec.height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _text_center(draw: ImageDraw.ImageDraw, box: Sequence[int], text: str, size: int, fill: str = INK, bold: bool = False) -> None:
    font = _font(size * SCALE, bold)
    x0, y0, x1, y1 = box
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
    except Exception:  # noqa: BLE001 - bitmap fallback
        tw, th = draw.textlength(text, font=font), size * SCALE
    draw.text(
        ((x0 + x1 - tw) / 2 - left, (y0 + y1 - th) / 2 - top),
        text,
        font=font,
        fill=fill,
    )


def _title_bar(draw: ImageDraw.ImageDraw, spec: ArtSpec) -> int:
    """Draw the chart title; return the y offset content should start at."""
    if not spec.title:
        return int(24 * SCALE)
    _text_center(draw, (0, int(14 * SCALE), spec.width * SCALE, int(50 * SCALE)), spec.title, 17, INK, bold=True)
    return int(64 * SCALE)


# ---------------------------------------------------------------------------
# chart kinds
# ---------------------------------------------------------------------------


def _flow(spec: ArtSpec, rng: random.Random) -> bytes:
    img, draw = _canvas(spec)
    top = _title_bar(draw, spec)
    labels = spec.labels or ["输入", "处理", "输出"]

    cols = 3 if len(labels) > 4 else 2
    rows = max(1, math.ceil(len(labels) / cols))
    box_w = int(spec.width * SCALE * 0.72 / cols)
    box_h = int((spec.height * SCALE - top - 30 * SCALE) / max(rows, 1) * 0.62)
    gap_x = int((spec.width * SCALE - cols * box_w) / (cols + 1))
    gap_y = int((spec.height * SCALE - top - 20 * SCALE - rows * box_h) / max(rows, 1))

    centres: List[Tuple[int, int]] = []
    for i, label in enumerate(labels):
        r, c = divmod(i, cols)
        x0 = gap_x + c * (box_w + gap_x)
        y0 = top + 10 * SCALE + r * (box_h + gap_y)
        colour = PALETTE[i % len(PALETTE)]
        draw.rounded_rectangle(
            [x0, y0, x0 + box_w, y0 + box_h], radius=8 * SCALE,
            fill=colour, outline="#ffffff", width=2 * SCALE,
        )
        _text_center(draw, (x0, y0, x0 + box_w, y0 + box_h), label, 14, "#ffffff", bold=True)
        centres.append((x0 + box_w // 2, y0 + box_h // 2))

    # arrows between consecutive boxes
    for (x0, y0), (x1, y1) in zip(centres, centres[1:]):
        if x1 > x0 and abs(y1 - y0) < box_h:  # same row: horizontal arrow
            start, end = (x0 + box_w // 2, y0), (x1 - box_w // 2, y1)
        else:  # next row: elbow
            start, end = (x0, y0 + box_h // 2), (x1, y1 - box_h // 2)
        draw.line([start, end], fill=MUTED, width=2 * SCALE)
        _arrow_head(draw, start, end)
    return _finish(img, spec)


def _arrow_head(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], size: int = 9) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    s = size * SCALE
    pts = [
        end,
        (end[0] - s * math.cos(angle - 0.4), end[1] - s * math.sin(angle - 0.4)),
        (end[0] - s * math.cos(angle + 0.4), end[1] - s * math.sin(angle + 0.4)),
    ]
    draw.polygon(pts, fill=MUTED)


def _axes(draw: ImageDraw.ImageDraw, spec: ArtSpec, top: int, pad_left: int = 70, pad_bottom: int = 60) -> Tuple[int, int, int, int]:
    x0 = pad_left * SCALE
    y0 = top
    x1 = spec.width * SCALE - 30 * SCALE
    y1 = spec.height * SCALE - pad_bottom * SCALE
    for i in range(5):  # horizontal grid lines
        y = y0 + (y1 - y0) * i // 4
        draw.line([(x0, y), (x1, y)], fill=GRID, width=SCALE)
    draw.line([(x0, y0), (x0, y1)], fill=INK, width=SCALE)
    draw.line([(x0, y1), (x1, y1)], fill=INK, width=SCALE)
    return x0, y0, x1, y1


def _bar(spec: ArtSpec, rng: random.Random) -> bytes:
    img, draw = _canvas(spec)
    top = _title_bar(draw, spec)
    labels = spec.labels or [f"项{i}" for i in range(1, 6)]
    values = spec.values or [rng.uniform(10, 100) for _ in labels]
    x0, y0, x1, y1 = _axes(draw, spec, top)

    n = len(values)
    slot = (x1 - x0) / n
    bar_w = slot * 0.56
    peak = max(values) or 1.0
    for i, (label, value) in enumerate(zip(labels, values)):
        bx0 = x0 + slot * i + (slot - bar_w) / 2
        bh = (y1 - y0) * (value / peak) * 0.92
        colour = PALETTE[i % len(PALETTE)]
        draw.rectangle([bx0, y1 - bh, bx0 + bar_w, y1], fill=colour)
        _text_center(draw, (int(bx0), int(y1 - bh - 26 * SCALE), int(bx0 + bar_w), int(y1 - bh - 2 * SCALE)),
                     _fmt(value), 12, INK, bold=True)
        _text_center(draw, (int(bx0 - 6 * SCALE), int(y1 + 4 * SCALE), int(bx0 + bar_w + 6 * SCALE), int(y1 + 34 * SCALE)),
                     label, 11, MUTED)
    return _finish(img, spec)


def _pie(spec: ArtSpec, rng: random.Random) -> bytes:
    img, draw = _canvas(spec)
    top = _title_bar(draw, spec)
    labels = spec.labels or [f"类别{i}" for i in range(1, 5)]
    values = spec.values or [rng.uniform(10, 40) for _ in labels]
    total = sum(values) or 1.0

    radius = int(min(spec.width * SCALE * 0.30, (spec.height * SCALE - top) * 0.40))
    cx = int(spec.width * SCALE * 0.34)
    cy = top + int((spec.height * SCALE - top) * 0.48)
    box = [cx - radius, cy - radius, cx + radius, cy + radius]

    start = -90.0
    for i, value in enumerate(values):
        extent = 360.0 * value / total
        draw.pieslice(box, start, start + extent, fill=PALETTE[i % len(PALETTE)], outline="#ffffff", width=2 * SCALE)
        start += extent

    legend_x = int(spec.width * SCALE * 0.66)
    legend_y = top + int(10 * SCALE)
    for i, (label, value) in enumerate(zip(labels, values)):
        y = legend_y + i * 30 * SCALE
        draw.rectangle([legend_x, y, legend_x + 16 * SCALE, y + 16 * SCALE], fill=PALETTE[i % len(PALETTE)])
        draw.text((legend_x + 24 * SCALE, y), f"{label}  {value / total * 100:.1f}%",
                  font=_font(13 * SCALE), fill=INK)
    return _finish(img, spec)


def _line(spec: ArtSpec, rng: random.Random) -> bytes:
    img, draw = _canvas(spec)
    top = _title_bar(draw, spec)
    labels = spec.labels or [str(i) for i in range(1, 9)]
    values = spec.values or [rng.uniform(20, 100) for _ in labels]
    x0, y0, x1, y1 = _axes(draw, spec, top)

    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(values)
    pts = [
        (
            x0 + int((x1 - x0) * (i / max(n - 1, 1))),
            y1 - int((y1 - y0) * ((v - lo) / span) * 0.86) - 8 * SCALE,
        )
        for i, v in enumerate(values)
    ]
    draw.line(pts, fill=spec.accent, width=3 * SCALE, joint="curve")
    for (px, py), value in zip(pts, values):
        draw.ellipse([px - 5 * SCALE, py - 5 * SCALE, px + 5 * SCALE, py + 5 * SCALE],
                     fill="#ffffff", outline=spec.accent, width=2 * SCALE)
    for i, label in enumerate(labels):
        px = x0 + int((x1 - x0) * (i / max(n - 1, 1)))
        _text_center(draw, (px - 30 * SCALE, y1 + 4 * SCALE, px + 30 * SCALE, y1 + 32 * SCALE), label, 11, MUTED)
    return _finish(img, spec, frame=False)


def _photo(spec: ArtSpec, rng: random.Random) -> bytes:
    """An abstract photograph: gradient sky, horizon band, soft shapes, vignette."""
    w, h = spec.width * SCALE, spec.height * SCALE
    img = Image.new("RGB", (w, h), "#9fb8cf")
    draw = ImageDraw.Draw(img)

    horizon = int(h * rng.uniform(0.55, 0.70))
    for y in range(0, horizon):  # sky gradient
        t = y / max(horizon, 1)
        draw.line([(0, y), (w, y)], fill=(
            int(150 + 60 * t), int(180 + 40 * t), int(210 + 25 * t)))
    for y in range(horizon, h):  # ground gradient
        t = (y - horizon) / max(h - horizon, 1)
        draw.line([(0, y), (w, y)], fill=(
            int(92 + 40 * t), int(112 + 34 * t), int(84 + 26 * t)))

    for _ in range(rng.randint(3, 6)):  # soft blobs
        cx, cy = rng.randint(0, w), rng.randint(0, horizon)
        r = rng.randint(int(w * 0.06), int(w * 0.18))
        shade = rng.randint(200, 245)
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=(shade, shade, shade))

    for _ in range(rng.randint(2, 5)):  # foreground poles / structures
        px = rng.randint(0, w)
        pw = rng.randint(int(w * 0.01), int(w * 0.035))
        ph = rng.randint(int(h * 0.18), int(h * 0.42))
        draw.rectangle([px, horizon - ph, px + pw, horizon + rng.randint(0, 30 * SCALE)], fill="#3f4a3a")

    img = img.filter(ImageFilter.GaussianBlur(0.6 * SCALE))  # depth of field
    return _finish(img, spec, frame=False)


def _screenshot(spec: ArtSpec, rng: random.Random) -> bytes:
    img, draw = _canvas(spec, background="#f4f6f8")
    title_h = int(30 * SCALE)
    draw.rectangle([0, 0, spec.width * SCALE, title_h], fill="#dfe3e8")
    for i, colour in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        cx = int((16 + i * 18) * SCALE)
        cy = title_h // 2
        draw.ellipse([cx - 6 * SCALE, cy - 6 * SCALE, cx + 6 * SCALE, cy + 6 * SCALE], fill=colour)
    if spec.title:
        draw.text((int(86 * SCALE), int(9 * SCALE)), spec.title, font=_font(12 * SCALE), fill=MUTED)

    side_w = int(spec.width * SCALE * 0.24)
    draw.rectangle([0, title_h, side_w, spec.height * SCALE], fill="#e8ecf1")
    for i in range(6):
        y = title_h + int((18 + i * 30) * SCALE)
        draw.rectangle([int(14 * SCALE), y, int((14 + rng.randint(60, 130)) * SCALE), y + int(12 * SCALE)],
                       fill="#c7ced6")

    lines = spec.body or ["GET /v1/items", "200 OK", "{ \"count\": 3 }"]
    for i, line in enumerate(lines[:12]):
        y = title_h + int((22 + i * 26) * SCALE)
        colour = "#0b7285" if line.strip().startswith(("GET", "POST", "PUT", "DELETE")) else INK
        draw.text((side_w + int(20 * SCALE), y), line, font=_font(12 * SCALE), fill=colour)
    return _finish(img, spec, frame=False)


def _formula(spec: ArtSpec, rng: random.Random) -> bytes:
    """A formula rendered as an image -- deliberately NOT text-extractable."""
    img, draw = _canvas(spec, background="#fbfbfd")
    formula = spec.body[0] if spec.body else "Attention(Q,K,V) = softmax(QKᵀ/√d_k)V"
    _text_center(draw, (0, 0, spec.width * SCALE, spec.height * SCALE), formula, 26, "#111827", bold=False)
    return _finish(img, spec, frame=True)


_DRAWERS = {
    "flow": _flow,
    "bar": _bar,
    "pie": _pie,
    "line": _line,
    "photo": _photo,
    "screenshot": _screenshot,
    "formula": _formula,
}

ART_KINDS: Tuple[str, ...] = tuple(_DRAWERS)


def draw(spec: ArtSpec, rng: random.Random) -> bytes:
    """Render *spec* to PNG bytes.  Raises ``KeyError`` for an unknown kind."""
    if spec.kind not in _DRAWERS:
        raise KeyError(f"unknown art kind {spec.kind!r}; expected one of {ART_KINDS}")
    return _DRAWERS[spec.kind](spec, rng)


def _fmt(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}"
