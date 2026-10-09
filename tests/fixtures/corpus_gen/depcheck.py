"""Dependency guard and runtime API probe for the corpus generator.

Two modes:

``python tests/fixtures/corpus_gen/depcheck.py``
    Assert every required third-party package is importable.  Exits 2 with
    the exact ``pip install`` command when something is missing.

``python tests/fixtures/corpus_gen/depcheck.py --probe``
    Actually exercise every library call the generator relies on, so that an
    upstream API rename is caught here instead of halfway through a
    300-second corpus build.  Exits 1 if any probe fails.

The probe is deliberately explicit about call shapes: this project pins
``reportlab>=4.0`` and ``pymupdf>=1.24`` but neither was installed when the
generator was written, so every call is verified rather than assumed.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.corpus_gen.fsutil import scratch_dir  # noqa: E402

#: Scratch root for probe artefacts.  Never use ``tempfile`` here: its
#: ``0o700`` directories are unwritable on this platform (see fsutil).
PROBE_ROOT = Path(os.environ.get("CORPUS_PROBE_DIR", str(_REPO_ROOT / ".probe-tmp")))

# package import name -> (pip requirement, why we need it)
REQUIRED: dict[str, str] = {
    "reportlab": "reportlab>=4.0",
    "fitz": "pymupdf>=1.24",
    "PIL": "pillow",
    "pypdfium2": "pypdfium2",
    "pdfplumber": "pdfplumber",
    "markitdown": "markitdown[pdf]",
}

PIP_COMMAND = (
    'pip install "reportlab>=4.0" "pymupdf>=1.24" '
    '"pillow" "pypdfium2" "pdfplumber" "markitdown[pdf]"'
)


def missing_packages() -> List[str]:
    """Return the import names of required packages that are unavailable."""
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def check_requirements() -> int:
    """Print a dependency report.  Returns an exit code."""
    missing = missing_packages()
    if not missing:
        return 0

    print("[FAIL] Missing required packages for the corpus generator:", file=sys.stderr)
    for name in missing:
        print(f"  - {name}  (install: {REQUIRED[name]})", file=sys.stderr)
    print(f"\nInstall them with:\n  {PIP_COMMAND}\n", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# API probe
# ---------------------------------------------------------------------------


def _probe_pypdfium2() -> str:
    """pypdfium2 must rasterise a page to a PIL image."""
    import pypdfium2 as pdfium

    # Render an existing PDF that we know exists in the repo when possible.
    tmp = scratch_dir(PROBE_ROOT, "probe-")
    pdf_path = tmp / "tiny.pdf"
    _write_tiny_pdf(pdf_path)

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        if len(doc) != 1:
            raise AssertionError(f"expected 1 page, got {len(doc)}")
        page = doc[0]
        render = getattr(page, "render", None) or getattr(page, "render_topil", None)
        if render is None:
            raise AssertionError("page has neither render() nor render_topil()")
        bitmap = render(scale=150 / 72)
        to_pil = getattr(bitmap, "to_pil", None)
        if to_pil is None:
            raise AssertionError("PdfBitmap has no to_pil()")
        img = to_pil()
        if img.size[0] < 100 or img.size[1] < 100:
            raise AssertionError(f"unexpected raster size {img.size}")
        version = (
            getattr(pdfium, "__version__", None)
            or getattr(pdfium, "V_PYPDFIUM2", None)
            or getattr(pdfium, "V_LIBPDFIUM", None)
            or "unknown"
        )
        return (
            f"pypdfium2 {version} render()={hasattr(page, 'render')} "
            f"raster={img.size} mode={img.mode}"
        )
    finally:
        doc.close()


def _probe_reportlab_canvas() -> str:
    """Alpha / text-render-mode / invariant support used by watermark + OCR layers."""
    import reportlab
    from reportlab.pdfgen import canvas

    names = [n for n in ("setFillAlpha", "setStrokeAlpha", "setTextRenderMode", "bookmarkPage", "addOutlineEntry")]
    present = {n: hasattr(canvas.Canvas, n) for n in names}

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(200, 200))
    # These are the exact calls the generator makes.
    if present["setFillAlpha"]:
        c.setFillAlpha(0.0)
    if present["setTextRenderMode"]:
        c.setTextRenderMode(3)
    c.drawString(10, 10, "probe")
    c.save()

    import reportlab.rl_config as rl_config

    invariant = hasattr(rl_config, "invariant")
    return f"reportlab {reportlab.Version} {present} rl_config.invariant={invariant}"


def _probe_reportlab_layout() -> str:
    """BaseDocTemplate + multiple Frames + Table SPAN/repeatRows."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet

    tmp = scratch_dir(PROBE_ROOT, "probe-")
    out = tmp / "layout.pdf"

    styles = getSampleStyleSheet()
    frame_w = A4[0] / 2 - 60
    frames = [
        Frame(40, 50, frame_w, A4[1] - 100, id="left", showBoundary=0),
        Frame(A4[0] / 2 + 20, 50, frame_w, A4[1] - 100, id="right", showBoundary=0),
    ]
    doc = BaseDocTemplate(str(out), pagesize=A4, invariant=1)
    doc.addPageTemplates([PageTemplate(id="two", frames=frames)])

    data = [["h1", "h2", "h3"], ["a", "b", "c"], ["d", "e", "f"]]
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("GRID", (0, 0), (-1, -1), 0.5, None)]))

    story = [Paragraph("probe " * 40, styles["BodyText"]), Spacer(1, 12), table]
    doc.build(story)

    import pypdfium2 as pdfium

    d = pdfium.PdfDocument(str(out))
    pages = len(d)
    d.close()
    if pages < 1:
        raise AssertionError("layout PDF produced no pages")
    return f"BaseDocTemplate+2 Frames ok, {pages} page(s), Table SPAN+repeatRows ok"


def _probe_pdfplumber() -> str:
    """pdfplumber must expose per-line bounding boxes for the OCR text layer."""
    import pdfplumber

    tmp = scratch_dir(PROBE_ROOT, "probe-")
    pdf_path = tmp / "lines.pdf"
    _write_tiny_pdf(pdf_path, lines=["First line here", "Second line here"])

    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[0]
        has_lines = hasattr(page, "extract_text_lines")
        lines = page.extract_text_lines() if has_lines else []
        keys = sorted(lines[0].keys()) if lines else []
    if not has_lines:
        raise AssertionError("pdfplumber page has no extract_text_lines()")
    if not lines:
        raise AssertionError("extract_text_lines() returned nothing for a text PDF")
    for required in ("text", "top", "bottom", "x0", "x1"):
        if required not in lines[0]:
            raise AssertionError(f"extract_text_lines() missing key {required!r}: {keys}")
    return f"pdfplumber {pdfplumber.__version__} lines={len(lines)} keys={keys}"


def _probe_pymupdf() -> str:
    """PyMuPDF must find embedded raster images (activates PdfLoader image path)."""
    import fitz
    from PIL import Image

    tmp = scratch_dir(PROBE_ROOT, "probe-")
    png = tmp / "img.png"
    Image.new("RGB", (80, 60), (200, 30, 30)).save(png)
    pdf_path = tmp / "withimg.pdf"
    _write_tiny_pdf(pdf_path, image_path=png)

    doc = fitz.open(str(pdf_path))
    try:
        images = doc[0].get_images(full=True)
        count = len(images)
        if count < 1:
            raise AssertionError("embedded image was not found by get_images()")
        xref = images[0][0]
        base = doc.extract_image(xref)
        if not base.get("image"):
            raise AssertionError("extract_image() returned no bytes")
        return f"pymupdf {fitz.__doc__.strip().splitlines()[0]} images={count} ext={base['ext']}"
    finally:
        doc.close()


def _probe_markitdown() -> str:
    """markitdown is the production extraction path; anchors are validated with it."""
    from markitdown import MarkItDown

    tmp = scratch_dir(PROBE_ROOT, "probe-")
    pdf_path = tmp / "md.pdf"
    _write_tiny_pdf(pdf_path, lines=["AnchorText Alpha", "AnchorText Beta"])

    md = MarkItDown()
    result = md.convert(str(pdf_path))
    text = result.text_content if hasattr(result, "text_content") else str(result)
    if "AnchorText" not in text:
        raise AssertionError(f"markitdown did not recover the text: {text!r}")
    return f"markitdown ok, extracted {len(text)} chars"


def _probe_reportlab_fonts() -> str:
    """A CJK TrueType font must register, otherwise every Chinese doc is tofu."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    found = [p for p in candidates if Path(p).exists()]
    if not found:
        raise AssertionError(f"no CJK font among {candidates}")
    pdfmetrics.registerFont(TTFont("ProbeCJK", found[0]))
    width = pdfmetrics.stringWidth("中文测试", "ProbeCJK", 12)
    if width <= 0:
        raise AssertionError("CJK font reported zero width for Chinese text")
    return f"CJK font {found[0]} width('中文测试',12)={width:.2f}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_tiny_pdf(path: Path, lines: List[str] | None = None, image_path: Path | None = None) -> None:
    """Write a minimal PDF used only by the probes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    lines = lines or ["Probe document line one", "Probe document line two"]
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 700
    for line in lines:
        c.drawString(60, y, line)
        y -= 20
    if image_path is not None:
        c.drawImage(str(image_path), 60, 400, width=120, height=90)
    c.save()


PROBES: List[Tuple[str, Callable[[], str]]] = [
    ("reportlab canvas API", _probe_reportlab_canvas),
    ("reportlab layout", _probe_reportlab_layout),
    ("reportlab CJK fonts", _probe_reportlab_fonts),
    ("pypdfium2 rasterise", _probe_pypdfium2),
    ("pdfplumber text lines", _probe_pdfplumber),
    ("pymupdf image extraction", _probe_pymupdf),
    ("markitdown extraction", _probe_markitdown),
]


def run_probes() -> int:
    """Run every API probe.  Returns an exit code."""
    code = check_requirements()
    if code != 0:
        return code

    failures = 0
    print("API probe")
    print("=" * 68)
    for label, fn in PROBES:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - probe reports anything
            failures += 1
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"  [ OK ] {label}")
            print(f"         {detail}")
    print("=" * 68)
    if failures:
        print(f"{failures} probe(s) failed.")
        return 1
    print(f"All {len(PROBES)} probes passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Corpus generator dependency guard / API probe.")
    parser.add_argument("--probe", action="store_true", help="Also exercise the library calls the generator uses.")
    args = parser.parse_args()

    if args.probe:
        return run_probes()
    return check_requirements()


if __name__ == "__main__":
    sys.exit(main())
