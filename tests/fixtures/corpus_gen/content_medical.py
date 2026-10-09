"""Medical and life-science renderer (WHO guideline / FDA & NMPA label / review).

Distinctive features: specialist terminology, dosage tables with merged cells that
span page breaks, figure/photo mixed layout (anatomy and treatment-flow diagrams),
and bilingual Chinese/English labels.  Two documents form a guideline version pair
(v1 / v2) so that "which recommendation changed?" is answerable only across
documents.
"""

from __future__ import annotations

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.content_common import Profile, build_document

#: Guideline flow: dosage tables, treatment diagrams, bilingual labels.
PROFILE = Profile(
    figure_arts=("flow", "photo", "bar"),
    formula_text=2,
    code_blocks=0,
    footnote_every=3,
    references=False,
    extra_tables_per_12_pages=1,
    body_sentences=3,
    difficulty="hard",
)


def build(spec: S.DocSpec, seed: int, fill: float | None = None):
    """Build the :class:`~.facts.DocPlan` for one medical document."""
    return build_document(spec, seed, PROFILE, fill=fill)
