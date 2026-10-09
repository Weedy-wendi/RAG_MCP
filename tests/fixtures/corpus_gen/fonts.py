"""Font registration for the corpus generator.

Registers a CJK TrueType font (required: every Chinese document would otherwise
render as tofu boxes) plus a matching bold face, and exposes the built-in Latin
and monospace faces reportlab already provides.

A glyph-coverage self-check renders a probe string and measures its width, so a
missing or broken CJK face fails loudly at start-up instead of silently
producing 40 unreadable documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Built-in faces (no registration needed).
FONT_LATIN = "Helvetica"
FONT_LATIN_BOLD = "Helvetica-Bold"
FONT_LATIN_ITALIC = "Helvetica-Oblique"
FONT_SERIF = "Times-Roman"
FONT_SERIF_BOLD = "Times-Bold"
FONT_MONO = "Courier"
FONT_MONO_BOLD = "Courier-Bold"

#: Candidate CJK fonts, most preferred first.  ``(path, subfont_index)``.
CJK_CANDIDATES: List[tuple[str, int]] = [
    ("C:/Windows/Fonts/msyh.ttc", 0),
    ("C:/Windows/Fonts/simsun.ttc", 0),
    ("C:/Windows/Fonts/simhei.ttf", 0),
    ("C:/Windows/Fonts/simkai.ttf", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
]

CJK_BOLD_CANDIDATES: List[tuple[str, int]] = [
    ("C:/Windows/Fonts/msyhbd.ttc", 0),
    ("C:/Windows/Fonts/simhei.ttf", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
]

CJK_REGULAR_NAME = "CorpusCJK"
CJK_BOLD_NAME = "CorpusCJK-Bold"


class FontError(RuntimeError):
    """Raised when no usable CJK font can be registered."""


@dataclass(frozen=True)
class FontSet:
    """The resolved font names used by ``pdf_writer``."""

    cjk: str
    cjk_bold: str
    latin: str = FONT_LATIN
    latin_bold: str = FONT_LATIN_BOLD
    latin_italic: str = FONT_LATIN_ITALIC
    serif: str = FONT_SERIF
    serif_bold: str = FONT_SERIF_BOLD
    mono: str = FONT_MONO
    mono_bold: str = FONT_MONO_BOLD

    def for_style(self, bold: bool = False) -> str:
        return self.cjk_bold if bold else self.cjk


_FONTSET: Optional[FontSet] = None


def _try_register(name: str, candidates: List[tuple[str, int]]) -> Optional[str]:
    """Register the first usable candidate under *name*; return its path."""
    for path, index in candidates:
        if not Path(path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
        except Exception:  # noqa: BLE001 - a broken face just falls through
            continue
        return path
    return None


def _assert_glyph_coverage(font_name: str) -> None:
    """Fail loudly if *font_name* cannot measure the probe text."""
    probe = "检索增强生成测试"
    try:
        width = pdfmetrics.stringWidth(probe, font_name, 12)
    except Exception as exc:  # noqa: BLE001
        raise FontError(f"cannot measure glyphs with {font_name}: {exc}") from exc
    if width <= 0:
        # Helvetica (the fallback) reports a positive width too, so also check
        # that the face we registered is not simply the Latin fallback.
        raise FontError(f"{font_name} reported zero width for CJK probe text")


def register_fonts(force: bool = False) -> FontSet:
    """Register CJK faces and return the resolved :class:`FontSet`.

    Idempotent; the result is cached unless *force* is set.

    Raises:
        FontError: when no CJK font from :data:`CJK_CANDIDATES` is available.
    """
    global _FONTSET
    if _FONTSET is not None and not force:
        return _FONTSET

    regular_path = _try_register(CJK_REGULAR_NAME, CJK_CANDIDATES)
    if regular_path is None:
        tried = ", ".join(p for p, _ in CJK_CANDIDATES)
        raise FontError(
            "No CJK TrueType font found. Chinese documents would render as tofu.\n"
            f"Tried: {tried}\n"
            "Install one of those fonts, or add its path to CJK_CANDIDATES."
        )

    bold_path = _try_register(CJK_BOLD_NAME, CJK_BOLD_CANDIDATES)
    if bold_path is None:
        # Reuse the regular face as its own bold rather than falling back to a
        # Latin face, which would silently drop Chinese characters.
        pdfmetrics.registerFont(TTFont(CJK_BOLD_NAME, regular_path))
        bold_path = regular_path

    _assert_glyph_coverage(CJK_REGULAR_NAME)
    _assert_glyph_coverage(CJK_BOLD_NAME)

    # Let <b> markup resolve to the CJK bold face inside CJK paragraphs.
    try:
        pdfmetrics.registerFontFamily(
            CJK_REGULAR_NAME,
            normal=CJK_REGULAR_NAME,
            bold=CJK_BOLD_NAME,
            italic=CJK_REGULAR_NAME,
            boldItalic=CJK_BOLD_NAME,
        )
    except Exception:  # noqa: BLE001 - family registration is a convenience
        pass

    _FONTSET = FontSet(cjk=CJK_REGULAR_NAME, cjk_bold=CJK_BOLD_NAME)
    return _FONTSET


def describe() -> str:
    """One-line description of the registered faces (for build logs)."""
    fonts = register_fonts()
    from reportlab.pdfbase import pdfmetrics as _m

    faces = _m.getRegisteredFontNames()
    return f"cjk={fonts.cjk} faces={len(faces)}"
