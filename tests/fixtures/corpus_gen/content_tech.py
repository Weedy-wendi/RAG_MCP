"""Technical manual renderer (GitHub Docs / vendor manual / API reference).

Distinctive features: a multi-level heading tree, parameter tables, fenced code
blocks, console screenshots, and version-difference sections.  Two documents in
this family form a version pair (v2 / v2.1) whose API differences are the subject
of cross-document questions; one document is 220 pages of procedurally unique API
reference so the corpus covers the "200+ pages" bracket.
"""

from __future__ import annotations

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.content_common import Profile, build_document

#: Reference-manual flow: code, screenshots, parameter tables.
PROFILE = Profile(
    figure_arts=("screenshot", "flow", "bar", "screenshot"),
    formula_text=1,
    code_blocks=8,
    footnote_every=4,
    references=False,
    extra_tables_per_12_pages=1,
    body_sentences=3,
    difficulty="medium",
)


def build(spec: S.DocSpec, seed: int, fill: float | None = None):
    """Build the :class:`~.facts.DocPlan` for one technical manual."""
    return build_document(spec, seed, PROFILE, fill=fill)
