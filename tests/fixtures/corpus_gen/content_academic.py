"""Academic paper renderer (arXiv / PubMed Central / CNKI imitation).

Papers are the formula- and figure-heaviest family: architecture diagrams, result
curves, numbered equations, dense footnote apparatus and a reference list.  A
subset also renders a formula **as an image**, which is deliberately not
text-extractable and therefore separates "the text layer contains the formula"
from "the formula is only pixels".
"""

from __future__ import annotations

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.content_common import Profile, build_document

#: Two-column arXiv-style flow with heavy formula and citation apparatus.
PROFILE = Profile(
    figure_arts=("flow", "line", "line", "flow"),
    formula_text=6,
    code_blocks=0,
    footnote_every=3,
    references=True,
    extra_tables_per_12_pages=0,
    body_sentences=4,
    difficulty="hard",
)


def build(spec: S.DocSpec, seed: int, fill: float | None = None):
    """Build the :class:`~.facts.DocPlan` for one academic paper."""
    return build_document(spec, seed, PROFILE, fill=fill)
