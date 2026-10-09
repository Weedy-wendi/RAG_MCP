"""Financial report renderer (SEC 10-K / annual report / prospectus / research note).

This family carries the highest table density: merged-cell segment tables,
multi-page quota/notes tables with repeating headers, grouped multi-level headers
and borderless responsibility matrices.  Every document also mixes pie, bar and
line charts, and the prose quotes high-precision figures so that numeric fidelity
is testable.  Two documents form a corrected-results version pair.
"""

from __future__ import annotations

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.content_common import Profile, build_document

#: Statement-heavy flow with dense tables and footnotes.
PROFILE = Profile(
    figure_arts=("pie", "bar", "line"),
    formula_text=1,
    code_blocks=0,
    footnote_every=2,
    references=False,
    extra_tables_per_12_pages=2,
    body_sentences=3,
    difficulty="hard",
)


def build(spec: S.DocSpec, seed: int, fill: float | None = None):
    """Build the :class:`~.facts.DocPlan` for one financial report."""
    return build_document(spec, seed, PROFILE, fill=fill)
