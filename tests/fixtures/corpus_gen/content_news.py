"""News / magazine renderer (long-form report / magazine / encyclopedia entry).

Distinctive features: three-column magazine flow, photo-led layout with captions,
mock advertisement blocks that the retriever must not surface, and unstructured
narrative prose.  One document is a vertical (right-to-left) scanned newspaper
page, which combines the hardest layout with the hardest text-extraction case.
"""

from __future__ import annotations

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.content_common import Profile, build_document

#: Narrative flow: long prose, photo-led figures, advert noise.
PROFILE = Profile(
    figure_arts=("photo", "bar", "photo", "pie"),
    formula_text=0,
    code_blocks=0,
    footnote_every=5,
    references=False,
    extra_tables_per_12_pages=1,
    body_sentences=5,
    difficulty="easy",
)


def build(spec: S.DocSpec, seed: int, fill: float | None = None):
    """Build the :class:`~.facts.DocPlan` for one news or magazine document."""
    return build_document(spec, seed, PROFILE, fill=fill)
