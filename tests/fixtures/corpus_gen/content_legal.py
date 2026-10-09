"""Legal and policy renderer (statute / contract / policy white paper).

Distinctive features: numbered articles with nested clause references, long
dependent sentences, cross-references of the form "见第 12 条第 3 款", scanned
statutes with no text layer, an OCR'd bilingual regulation, and one vertical
(columns right-to-left) reproduction of a historical provision.  Legal documents
deliberately contain almost no figures, which is what produces the corpus's
"no figure" coverage.
"""

from __future__ import annotations

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.content_common import Profile, build_document

#: Clause-dense flow: long sentences, few figures, occasional simple table.
PROFILE = Profile(
    figure_arts=("flow",),
    formula_text=0,
    code_blocks=0,
    footnote_every=6,
    references=True,
    extra_tables_per_12_pages=0,
    body_sentences=4,
    difficulty="hard",
)


def build(spec: S.DocSpec, seed: int, fill: float | None = None):
    """Build the :class:`~.facts.DocPlan` for one legal or policy document."""
    return build_document(spec, seed, PROFILE, fill=fill)
