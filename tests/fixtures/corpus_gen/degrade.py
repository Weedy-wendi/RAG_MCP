"""Turn a native PDF into a "dirty" one: scanned image-only, or OCR'd.

Two degradations, both produced by rasterising the native page and rebuilding a
PDF from the resulting image:

``scanned``
    Image-only PDF.  There is **no text layer**, so ``markitdown`` extracts
    nothing and the ingestion pipeline is expected to fail the document
    (``DocumentChunker`` raises on empty text).  This is the intended coverage of
    "the system has no OCR".

``ocr``
    The same degraded page image, plus an **invisible** text layer (alpha 0) that
    mimics a real OCR'd PDF: the text is extractable but carries OCR-style
    recognition errors.  Line positions come from ``pdfplumber`` bounding boxes
    measured on the *native* source, so the invisible text sits where the visible
    glyphs are.

Memory is bounded: pages are rasterised, degraded and written one at a time
rather than materialising every page image at once (a 50-page scan would
otherwise hold ~300 MB of bitmaps).
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pypdfium2 as pdfium
import pdfplumber
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.fsutil import ensure_dir

#: Character confusions a real OCR engine makes, Latin and CJK.
OCR_CONFUSIONS: Tuple[Tuple[str, str], ...] = (
    ("0", "O"), ("1", "l"), ("5", "S"), ("8", "B"), ("2", "Z"),
    ("6", "G"), ("9", "g"), ("rn", "m"), ("cl", "d"), ("vv", "w"),
    ("己", "已"), ("未", "末"), ("日", "曰"), ("土", "士"), ("人", "入"),
    ("0", "〇"), ("千", "干"), ("戌", "戍"), ("m", "rn"),
)


@dataclass
class DegradeResult:
    """What was done to produce a dirty PDF."""

    pages: int
    profile: str
    dpi: int
    mode: str
    ocr_error_rate: float = 0.0
    bytes_written: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "pages": self.pages,
            "scan_profile": self.profile,
            "scan_dpi": self.dpi,
            "mode": self.mode,
            "ocr_error_rate": round(self.ocr_error_rate, 4),
            "bytes_written": self.bytes_written,
        }


# ---------------------------------------------------------------------------
# Per-page degradation
# ---------------------------------------------------------------------------


def _paper_tint(img: Image.Image, strength: float) -> Image.Image:
    """Warm, slightly darkened 'aged paper' cast."""
    arr = np.asarray(img, dtype=np.float32)
    arr *= strength
    arr[..., 0] += 3.0   # warm red
    arr[..., 1] += 1.5
    arr[..., 2] -= 2.0   # cool blue pulled down
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _add_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    if sigma <= 0:
        return img
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), "RGB")


def _add_speckles(img: Image.Image, rng: random.Random, count: int) -> Image.Image:
    """Scanner dust: small dark specks and short hairline strokes."""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for _ in range(count):
        x = rng.randrange(width)
        y = rng.randrange(height)
        radius = rng.choice((1, 1, 2))
        shade = rng.randint(40, 120)
        draw.ellipse([x, y, x + radius, y + radius], fill=(shade, shade, shade))
    for _ in range(max(1, count // 6)):
        x = rng.randrange(width)
        y = rng.randrange(height)
        length = rng.randint(8, 40)
        draw.line([(x, y), (x + length, y + rng.randint(-3, 3))],
                  fill=(90, 90, 90), width=1)
    return img


def _add_margin_shadow(img: Image.Image) -> Image.Image:
    """Darken the binding edge, as a flatbed scan of a bound volume does."""
    arr = np.asarray(img, dtype=np.float32)
    height, width, _ = arr.shape
    band = max(6, width // 26)
    ramp = np.linspace(0.72, 1.0, band, dtype=np.float32)
    arr[:, :band, :] *= ramp[None, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _jpeg_round_trip(img: Image.Image, quality: int) -> Image.Image:
    if quality >= 96:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_page(img: Image.Image, spec: S.DocSpec, page_index: int) -> Image.Image:
    """Apply the document's degradation profile to one rasterised page."""
    rng = random.Random(f"{spec.doc_id}:{spec.scan_profile}:{page_index}")
    np_rng = np.random.default_rng(abs(hash((spec.doc_id, page_index))) % (2**32))

    aged = spec.scan_profile in ("aged", "noisy", "lowq_jpeg")
    out = _paper_tint(img, 0.95 if aged else 0.97)

    if spec.scan_noise > 0:
        out = _add_noise(out, spec.scan_noise, np_rng)

    if spec.scan_profile in ("aged", "noisy", "lowq_jpeg", "blur"):
        out = _add_speckles(out, rng, count=90 if aged else 40)

    if spec.scan_blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=spec.scan_blur))

    if abs(spec.scan_skew) > 0.01:
        out = out.rotate(
            spec.scan_skew, resample=Image.BICUBIC, expand=True, fillcolor=(248, 246, 242)
        )

    if spec.scan_profile in ("aged", "lowq_jpeg", "clean"):
        out = _add_margin_shadow(out)

    out = _jpeg_round_trip(out, spec.scan_jpeg)
    return out


# ---------------------------------------------------------------------------
# OCR error model
# ---------------------------------------------------------------------------


def corrupt_ocr_text(text: str, rate: float, rng: random.Random) -> str:
    """Introduce OCR-style recognition errors at roughly *rate* per character."""
    if not text or rate <= 0:
        return text
    out: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        roll = rng.random()

        if roll < rate * 0.35:
            # drop a character
            i += 1
            continue
        if roll < rate * 0.75:
            # substitute using a plausible confusion
            pair = rng.choice([p for p in OCR_CONFUSIONS if p[0][0] == ch] or list(OCR_CONFUSIONS))
            out.append(pair[1])
            i += len(pair[0])
            continue
        if roll < rate and i + 1 < len(text) and not text[i + 1].isspace() and not ch.isspace():
            # transpose two adjacent characters
            out.append(text[i + 1])
            out.append(ch)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def ocr_error_rate(spec: S.DocSpec) -> float:
    """Per-character error rate implied by the document's scan profile."""
    return {
        "clean": 0.010,
        "blur": 0.030,
        "skew": 0.025,
        "noisy": 0.060,
        "lowq_jpeg": 0.085,
        "aged": 0.045,
    }.get(spec.scan_profile, 0.03)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _page_lines(source_pdf: Path) -> List[List[Dict[str, object]]]:
    """Per-page text lines with bounding boxes, from the native PDF."""
    pages: List[List[Dict[str, object]]] = []
    with pdfplumber.open(str(source_pdf)) as pdf:
        for page in pdf.pages:
            try:
                lines = page.extract_text_lines()
            except Exception:  # noqa: BLE001
                lines = []
            pages.append(list(lines))
    return pages


def make_scanned(source_pdf: Path, out_pdf: Path, spec: S.DocSpec) -> DegradeResult:
    """Write an image-only PDF (no text layer) from *source_pdf*."""
    ensure_dir(Path(out_pdf).parent)
    doc = pdfium.PdfDocument(str(source_pdf))
    canv = rl_canvas.Canvas(str(out_pdf), pagesize=A4, invariant=1)
    scale = spec.scan_dpi / 72.0
    pages = 0
    try:
        for index in range(len(doc)):
            bitmap = doc[index].render(scale=scale)
            try:
                img = bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
            img = degrade_page(img, spec, index)
            _draw_page_image(canv, img)
            canv.showPage()
            pages += 1
    finally:
        doc.close()
    canv.save()
    return DegradeResult(
        pages=pages, profile=spec.scan_profile, dpi=spec.scan_dpi,
        mode="scanned", bytes_written=Path(out_pdf).stat().st_size,
    )


def make_ocr(source_pdf: Path, out_pdf: Path, spec: S.DocSpec) -> DegradeResult:
    """Write a scanned-looking PDF with an invisible, OCR-corrupted text layer."""
    ensure_dir(Path(out_pdf).parent)
    doc = pdfium.PdfDocument(str(source_pdf))
    lines_by_page = _page_lines(source_pdf)
    canv = rl_canvas.Canvas(str(out_pdf), pagesize=A4, invariant=1)
    scale = spec.scan_dpi / 72.0
    rate = ocr_error_rate(spec)
    pages = 0
    try:
        for index in range(len(doc)):
            bitmap = doc[index].render(scale=scale)
            try:
                img = bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
            img = degrade_page(img, spec, index)
            _draw_page_image(canv, img)
            _draw_invisible_text(canv, lines_by_page[index] if index < len(lines_by_page) else [],
                                 spec, index, rate)
            canv.showPage()
            pages += 1
    finally:
        doc.close()
    canv.save()
    return DegradeResult(
        pages=pages, profile=spec.scan_profile, dpi=spec.scan_dpi,
        mode="ocr", ocr_error_rate=rate, bytes_written=Path(out_pdf).stat().st_size,
    )


def _draw_page_image(canv: rl_canvas.Canvas, img: Image.Image) -> None:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    width, height = A4
    canv.drawImage(ImageReader(buf), 0, 0, width=width, height=height, anchor="sw")


def _draw_invisible_text(
    canv: rl_canvas.Canvas,
    lines: Sequence[Dict[str, object]],
    spec: S.DocSpec,
    page_index: int,
    rate: float,
) -> None:
    """Draw the OCR text layer with zero alpha so it is extractable but unseen."""
    rng = random.Random(f"{spec.doc_id}:ocr:{page_index}")
    _, page_height = A4
    canv.saveState()
    try:
        canv.setFillAlpha(0)
    except Exception:  # noqa: BLE001 - older reportlab: fall back to white text
        canv.setFillColorRGB(1, 1, 1)
    from tests.fixtures.corpus_gen.fonts import register_fonts

    fonts = register_fonts()

    for line in lines:
        text = str(line.get("text", "") or "")
        if not text.strip():
            continue
        top = float(line.get("top", 0.0) or 0.0)
        bottom = float(line.get("bottom", top + 8.0) or (top + 8.0))
        x0 = float(line.get("x0", 0.0) or 0.0)
        height = max(4.0, bottom - top)
        size = min(13.0, max(4.0, height * 0.82))
        corrupted = corrupt_ocr_text(text, rate, rng)
        canv.setFont(fonts.cjk, size)
        canv.drawString(x0, page_height - bottom, corrupted)

    canv.restoreState()
