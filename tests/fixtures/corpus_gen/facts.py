"""Document model: the single source of truth for rendering **and** the answer bank.

Architecture
------------
A content module returns a :class:`DocPlan`.  It contains

* ``body``  -- an ordered list of :class:`Block` objects, rendered by ``pdf_writer``;
* ``facts`` -- the :class:`Fact` objects whose ``anchor`` text is written into
  that body;
* ``refusals`` -- questions that the document deliberately does **not** answer.

``qa.py`` builds the answer bank **only** from ``facts``/``refusals``.  Because the
question and the sentence it asks about are authored together and rendered from
the same object, an answer can never drift away from the corpus.  ``verify_corpus``
then re-checks every anchor against the text that ``markitdown`` extracts — the
production code path — so a claim is only accepted if it survives real extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Fact kinds  (mirrors the five question categories required by the corpus spec)
# ---------------------------------------------------------------------------

FACTUAL = "factual"
TABLE = "table"
CROSS_PAGE = "cross_page"
CROSS_DOC = "cross_doc"
FIGURE = "figure"
REFUSAL = "refusal"

QUESTION_KINDS: Tuple[str, ...] = (FACTUAL, TABLE, CROSS_PAGE, CROSS_DOC, FIGURE, REFUSAL)

#: Table styles that count as a difficulty witness.  Kept as a literal tuple so
#: this module does not depend on :mod:`spec`; ``spec.TABLE_DIFFICULTY_STYLES``
#: must stay in sync and a unit test asserts that it does.
_DIFFICULTY_TABLE_STYLES: Tuple[str, ...] = (
    "simple", "merged", "multipage", "borderless", "multiheader",
)

#: Kinds that require a second anchor in the same document.
_NEEDS_SECONDARY = (CROSS_PAGE,)


# ---------------------------------------------------------------------------
# Blocks -- the renderable intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """Base class for everything ``pdf_writer`` can render."""

    kind: str = "block"


@dataclass
class Heading(Block):
    text: str = ""
    level: int = 1
    kind: str = "heading"


@dataclass
class Para(Block):
    """A paragraph.  ``text`` may contain reportlab inline markup."""

    text: str = ""
    style: str = "body"
    kind: str = "para"


@dataclass
class Bullets(Block):
    items: List[str] = field(default_factory=list)
    ordered: bool = False
    kind: str = "bullets"


@dataclass
class CodeBlock(Block):
    lines: List[str] = field(default_factory=list)
    caption: Optional[str] = None
    kind: str = "code"


@dataclass
class TableBlock(Block):
    """A table.

    ``style`` selects the difficulty class the corpus matrix asks for:

    ``simple``      plain grid, single header row
    ``merged``      vertically/horizontally spanned header + data cells
    ``multipage``   long table with a repeating header row (spans page breaks)
    ``borderless``  no grid lines at all (layout carries the structure)
    ``multiheader`` two-row grouped header built from spans
    """

    rows: List[Sequence[str]] = field(default_factory=list)
    caption: Optional[str] = None
    style: str = "simple"
    col_widths: Optional[List[float]] = None
    kind: str = "table"


@dataclass
class FigureBlock(Block):
    """An embedded raster image plus its caption line.

    The caption is real text, so it survives extraction and can be asked about;
    the image itself exercises the PyMuPDF / ImageStorage path.
    """

    fig_id: str = ""
    caption: str = ""
    art: str = "flow"  # flow | bar | pie | line | photo | formula | screenshot
    width_ratio: float = 0.86
    #: Extra keyword arguments forwarded to ``art.ArtSpec`` (labels, values, ...).
    art_kwargs: Dict[str, Any] = field(default_factory=dict)
    kind: str = "figure"


@dataclass
class FormulaBlock(Block):
    """A formula rendered as extractable Unicode text."""

    text: str = ""
    display: bool = True
    kind: str = "formula"


@dataclass
class FootnoteBlock(Block):
    items: List[str] = field(default_factory=list)
    kind: str = "footnotes"


@dataclass
class ReferenceBlock(Block):
    items: List[str] = field(default_factory=list)
    kind: str = "references"


@dataclass
class AdBlock(Block):
    """A mock advertisement: pure noise the retriever must not surface."""

    headline: str = ""
    lines: List[str] = field(default_factory=list)
    kind: str = "ad"


@dataclass
class PageBreak(Block):
    kind: str = "pagebreak"


@dataclass
class Rule(Block):
    kind: str = "rule"


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    """One answerable question plus the evidence that must be written to the PDF.

    Attributes:
        fact_id: Stable id, ``<DOC>-F<nn>``.
        kind: One of :data:`QUESTION_KINDS` (excluding ``refusal``).
        question: The question text for the answer bank.
        answer: The reference answer.
        anchor: A substring that **must** appear in the extracted text of this
            document.  This is the self-validation hook.
        secondary_anchor: A second required substring (``cross_page``).
        secondary_doc: For ``cross_doc``, the other document that must also
            contain :attr:`secondary_anchor`.
        answer_type: ``text`` | ``numeric`` | ``date`` | ``list`` | ``boolean``.
        section: Human-readable section reference for the answer bank.
        page_hint: Rough page number, for human navigation only.
        figure_id: Figure this question is about (``figure`` kind only).
        difficulty: ``easy`` | ``medium`` | ``hard``.
        requires_ocr: True when the owning document has no clean text layer, so
            the anchor is only expected to match fuzzily.
    """

    fact_id: str
    kind: str
    question: str
    answer: str
    anchor: str
    secondary_anchor: Optional[str] = None
    secondary_doc: Optional[str] = None
    answer_type: str = "text"
    section: str = ""
    page_hint: int = 0
    figure_id: str = ""
    difficulty: str = "medium"
    requires_ocr: bool = False

    def __post_init__(self) -> None:
        if self.kind not in QUESTION_KINDS or self.kind == REFUSAL:
            raise ValueError(f"{self.fact_id}: invalid fact kind {self.kind!r}")
        if not self.anchor:
            raise ValueError(f"{self.fact_id}: anchor must be non-empty")
        if self.kind in _NEEDS_SECONDARY and not self.secondary_anchor:
            raise ValueError(f"{self.fact_id}: kind={self.kind} requires secondary_anchor")
        if self.kind == CROSS_DOC and not self.secondary_doc:
            raise ValueError(f"{self.fact_id}: kind={CROSS_DOC} requires secondary_doc")
        if self.kind == FIGURE and not self.figure_id:
            raise ValueError(f"{self.fact_id}: kind={FIGURE} requires figure_id")

    @property
    def required_anchors(self) -> Tuple[str, ...]:
        """Every substring that must appear in this document's extracted text."""
        if self.secondary_anchor and self.secondary_doc is None:
            return (self.anchor, self.secondary_anchor)
        return (self.anchor,)

    def to_qa_record(self, doc_id: str, qa_index: int, expected_sources: Sequence[str]) -> Dict[str, Any]:
        """Serialise to the answer-bank record consumed by tooling and humans."""
        secondary_doc = self.secondary_doc or ""
        return {
            "qid": f"{doc_id}-Q{qa_index:02d}",
            "doc_id": doc_id,
            "fact_id": self.fact_id,
            "category": self.kind,
            "question": self.question,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "evidence": {
                "section": self.section,
                "page_hint": self.page_hint,
                "anchor_text": self.anchor,
                "secondary_anchor_text": self.secondary_anchor,
            },
            "expected_sources": list(expected_sources),
            "cross_doc_refs": [secondary_doc] if secondary_doc else [],
            "expect_no_answer": False,
            "reject_terms": [],
            "requires_ocr": self.requires_ocr,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class RefusalQuestion:
    """A question the corpus deliberately cannot answer.

    ``reject_terms`` are the entities the question asks about.  The verifier
    asserts they appear in **no** document of the whole corpus, which is what
    makes "the system should decline" a checkable claim rather than a hope.
    """

    fact_id: str
    question: str
    reject_terms: Tuple[str, ...]
    reason: str
    difficulty: str = "medium"

    def __post_init__(self) -> None:
        if not self.reject_terms:
            raise ValueError(f"{self.fact_id}: refusal needs at least one reject term")

    def to_qa_record(self, doc_id: str, qa_index: int, expected_sources: Sequence[str]) -> Dict[str, Any]:
        return {
            "qid": f"{doc_id}-Q{qa_index:02d}",
            "doc_id": doc_id,
            "fact_id": self.fact_id,
            "category": REFUSAL,
            "question": self.question,
            "answer": "文档中未提及（应拒答）",
            "answer_type": "refusal",
            "evidence": {
                "section": "",
                "page_hint": 0,
                "anchor_text": None,
                "secondary_anchor_text": None,
            },
            "expected_sources": list(expected_sources),
            "cross_doc_refs": [],
            "expect_no_answer": True,
            "reject_terms": list(self.reject_terms),
            "requires_ocr": False,
            "difficulty": self.difficulty,
        }


# ---------------------------------------------------------------------------
# Document plan
# ---------------------------------------------------------------------------


@dataclass
class DocPlan:
    """Everything needed to render one document and to build its answer bank."""

    doc_id: str
    category: str
    title: str
    subtitle: str = ""
    body: List[Block] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    refusals: List[RefusalQuestion] = field(default_factory=list)
    #: Optional per-document overrides recorded into the manifest.
    notes: Dict[str, Any] = field(default_factory=dict)

    def add(self, *blocks: Block) -> "DocPlan":
        """Append renderable blocks."""
        self.body.extend(blocks)
        return self

    def add_facts(self, *facts: Fact) -> "DocPlan":
        """Register answerable facts."""
        self.facts.extend(facts)
        return self

    def add_refusals(self, *refusals: RefusalQuestion) -> "DocPlan":
        """Register unanswerable questions."""
        self.refusals.extend(refusals)
        return self

    # -- introspection used by coverage / verification ---------------------

    @property
    def question_count(self) -> int:
        return len(self.facts) + len(self.refusals)

    def kind_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {kind: 0 for kind in QUESTION_KINDS}
        for fact in self.facts:
            counts[fact.kind] += 1
        counts[REFUSAL] = len(self.refusals)
        return counts

    def figure_count(self) -> int:
        return sum(1 for block in self.body if isinstance(block, FigureBlock))

    def table_count(self) -> int:
        return sum(1 for block in self.body if isinstance(block, TableBlock))

    def table_styles(self) -> Dict[str, int]:
        """Count genuine table-difficulty witnesses.

        The cover-page metadata table uses the ``info`` style and is excluded, so
        a document is never credited with a hard table it does not really have.
        """
        counts: Dict[str, int] = {}
        for block in self.body:
            if isinstance(block, TableBlock) and block.style in _DIFFICULTY_TABLE_STYLES:
                counts[block.style] = counts.get(block.style, 0) + 1
        return counts

    def heading_count(self) -> int:
        return sum(1 for block in self.body if isinstance(block, Heading))

    def ad_count(self) -> int:
        return sum(1 for block in self.body if isinstance(block, AdBlock))

    def footnote_count(self) -> int:
        return sum(len(block.items) for block in self.body if isinstance(block, FootnoteBlock))

    def reference_count(self) -> int:
        return sum(len(block.items) for block in self.body if isinstance(block, ReferenceBlock))

    def formula_counts(self) -> Tuple[int, int]:
        """Return ``(text_formulas, formula_images)``."""
        text_formulas = sum(1 for b in self.body if isinstance(b, FormulaBlock))
        image_formulas = sum(
            1 for b in self.body if isinstance(b, FigureBlock) and b.art == "formula"
        )
        return text_formulas, image_formulas

    def has_toc(self) -> bool:
        return any(isinstance(b, Heading) and b.text.strip().lower() in {"目录", "contents", "table of contents"}
                   for b in self.body)


def flatten_text(plan: "DocPlan") -> str:
    """Concatenate every piece of text a plan will render.

    Used by the fast unit tests to prove that a question's anchor really was
    written into the document, without rendering a PDF.  Table cells are joined
    in cell order, matching how an extractor walks a table.
    """
    parts: List[str] = []
    for block in plan.body:
        if isinstance(block, Heading):
            parts.append(block.text)
        elif isinstance(block, Para):
            parts.append(block.text)
        elif isinstance(block, Bullets):
            parts.extend(block.items)
        elif isinstance(block, CodeBlock):
            if block.caption:
                parts.append(block.caption)
            parts.extend(block.lines)
        elif isinstance(block, TableBlock):
            if block.caption:
                parts.append(block.caption)
            for row in block.rows:
                parts.extend(str(cell) for cell in row)
        elif isinstance(block, FigureBlock):
            parts.append(block.caption)
        elif isinstance(block, FormulaBlock):
            parts.append(block.text)
        elif isinstance(block, (FootnoteBlock, ReferenceBlock)):
            parts.extend(block.items)
        elif isinstance(block, AdBlock):
            parts.append(block.headline)
            parts.extend(block.lines)
    return "\n".join(parts)
