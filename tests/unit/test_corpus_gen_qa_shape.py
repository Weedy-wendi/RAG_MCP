"""Answer-bank shape tests, and anchor self-validation without rendering.

The strongest check here is :func:`test_every_anchor_occurs_in_plan_text`: every
question's claimed evidence must literally occur in the text the plan will
render.  That is the pre-render equivalent of the check ``verify_corpus.py``
performs against the extracted PDF text, so an anchor typo is caught in
milliseconds rather than after rendering 100 documents.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.corpus_gen import coverage  # noqa: E402
from tests.fixtures.corpus_gen import spec as S  # noqa: E402
from tests.fixtures.corpus_gen.facts import (  # noqa: E402
    QUESTION_KINDS,
    REFUSAL,
    flatten_text,
)

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub("", text or "")


@pytest.fixture(scope="module")
def checked() -> coverage.PlanCoverage:
    return coverage.plan_coverage()


@pytest.fixture(scope="module")
def plan_text(checked: coverage.PlanCoverage) -> dict:
    return {doc_id: normalize(flatten_text(plan)) for doc_id, plan in checked.plans.items()}


def test_question_kinds_are_all_valid(checked: coverage.PlanCoverage) -> None:
    for doc_id, plan in checked.plans.items():
        for fact in plan.facts:
            assert fact.kind in QUESTION_KINDS, (doc_id, fact.fact_id, fact.kind)
            assert fact.kind != REFUSAL
            assert fact.question.strip(), (doc_id, fact.fact_id)
            assert fact.answer is not None


def test_fact_ids_are_unique_per_document(checked: coverage.PlanCoverage) -> None:
    for doc_id, plan in checked.plans.items():
        ids = [f.fact_id for f in plan.facts] + [r.fact_id for r in plan.refusals]
        assert len(ids) == len(set(ids)), f"{doc_id}: duplicate fact ids"


def test_every_anchor_occurs_in_plan_text(checked: coverage.PlanCoverage, plan_text: dict) -> None:
    """Each anchor must be a real substring of what the document will render."""
    problems = []
    for doc_id, plan in checked.plans.items():
        text = plan_text[doc_id]
        for fact in plan.facts:
            for anchor in (*fact.required_anchors, fact.secondary_anchor):
                if not anchor:
                    continue
                if normalize(anchor) not in text:
                    problems.append(f"{doc_id}/{fact.fact_id}: anchor missing from plan: {anchor!r}")
    assert not problems, "\n".join(problems[:40])


def test_evidence_anchors_are_unique_within_a_document(checked: coverage.PlanCoverage) -> None:
    """Two evidence-anchored questions must not share the same evidence token.

    Only ``factual``/``table``/``figure`` questions are checked: a cross-page or
    cross-document question legitimately *reuses* an existing anchor as one of the
    two pieces of evidence it compares.
    """
    problems = []
    for doc_id, plan in checked.plans.items():
        seen: dict = {}
        for fact in plan.facts:
            if fact.kind not in ("factual", "table", "figure"):
                continue
            key = normalize(fact.anchor)
            if key in seen:
                problems.append(f"{doc_id}: {fact.fact_id} reuses anchor from {seen[key]}")
            seen[key] = fact.fact_id
    assert not problems, "\n".join(problems[:40])


def test_cross_page_questions_have_two_anchors_on_different_pages(
    checked: coverage.PlanCoverage,
) -> None:
    for doc_id, plan in checked.plans.items():
        for fact in plan.facts:
            if fact.kind != "cross_page":
                continue
            assert fact.secondary_anchor, f"{doc_id}/{fact.fact_id}: no secondary anchor"
            assert fact.page_hint != 0
            # the two anchors must be different text, otherwise it is not cross-page
            assert normalize(fact.anchor) != normalize(fact.secondary_anchor), fact.fact_id


def test_cross_document_questions_reference_a_real_pair(checked: coverage.PlanCoverage) -> None:
    seen = 0
    for doc_id, plan in checked.plans.items():
        for fact in plan.facts:
            if fact.kind != "cross_doc":
                continue
            seen += 1
            assert fact.secondary_doc, f"{doc_id}/{fact.fact_id}: no partner document"
            partner = checked.specs.get(fact.secondary_doc)
            assert partner is not None, f"{doc_id}: unknown partner {fact.secondary_doc}"
            # the pair relation must be symmetric
            assert partner.version_pair == doc_id
    assert seen == 12, f"expected one cross-document question per pair member, got {seen}"


def test_figure_questions_reference_a_rendered_figure(checked: coverage.PlanCoverage) -> None:
    for doc_id, plan in checked.plans.items():
        figure_ids = {b.fig_id for b in plan.body if hasattr(b, "fig_id")}
        for fact in plan.facts:
            if fact.kind != "figure":
                continue
            assert fact.figure_id, f"{doc_id}/{fact.fact_id}: no figure id"
            assert fact.figure_id in figure_ids, f"{doc_id}: unknown figure {fact.figure_id}"


def test_refusal_terms_do_not_occur_anywhere_in_the_corpus(
    checked: coverage.PlanCoverage, plan_text: dict
) -> None:
    """A refusal question is only valid if its subject exists nowhere.

    This is the pre-render form of the corpus-wide refusal check, so a reject
    term that collides with generated prose is caught immediately.
    """
    corpus = "\n".join(plan_text.values())
    problems = []
    for doc_id, plan in checked.plans.items():
        for refusal in plan.refusals:
            for term in refusal.reject_terms:
                if normalize(term) in corpus:
                    problems.append(f"{doc_id}/{refusal.fact_id}: {term!r} occurs in the corpus")
    assert not problems, "\n".join(problems[:40])


def test_refusal_questions_are_unique_per_document(checked: coverage.PlanCoverage) -> None:
    for doc_id, plan in checked.plans.items():
        questions = [r.question for r in plan.refusals]
        assert len(questions) == len(set(questions)), f"{doc_id}: duplicate refusal questions"


def test_qa_records_serialise_without_loss(checked: coverage.PlanCoverage) -> None:
    from tests.fixtures.corpus_gen import qa

    for doc_id, plan in checked.plans.items():
        spec = checked.specs[doc_id]
        doc_qa = qa.build_doc_qa(spec, plan, spec.relpath())
        payload = doc_qa.payload
        assert payload["question_count"] == plan.question_count
        assert len(payload["questions"]) == plan.question_count
        qids = [q["qid"] for q in payload["questions"]]
        assert len(qids) == len(set(qids)), f"{doc_id}: duplicate qids"
        assert all(q["qid"].startswith(doc_id) for q in payload["questions"])
        assert payload["expected_ingestion"]["expect"] in ("success", "failure")


def test_expected_ingestion_matches_extractability(checked: coverage.PlanCoverage) -> None:
    from tests.fixtures.corpus_gen import qa

    for doc_id, plan in checked.plans.items():
        spec = checked.specs[doc_id]
        expectation = qa.expected_ingestion(spec, plan)
        if spec.is_scanned:
            assert expectation["expect"] == "failure"
            assert "no text content" in expectation["error_contains"]
        else:
            assert expectation["expect"] == "success"
            low, high = expectation["chunk_count_range"]
            assert 0 < low <= high, (doc_id, low, high)
