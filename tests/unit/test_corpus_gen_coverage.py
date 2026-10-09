"""Coverage-threshold tests over the realised document plans.

Building all 100 plans in memory takes well under a second and exercises every
content module plus the whole answer-bank pipeline.  A spec or content
regression therefore fails here immediately, instead of after a multi-minute PDF
render.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.corpus_gen import coverage  # noqa: E402
from tests.fixtures.corpus_gen import spec as S  # noqa: E402


@pytest.fixture(scope="module")
def checked() -> coverage.PlanCoverage:
    return coverage.plan_coverage()


def test_every_threshold_passes(checked: coverage.PlanCoverage) -> None:
    failures = [m.line() for m in checked.failed]
    assert not failures, "coverage thresholds not met:\n" + "\n".join(failures)


def test_threshold_table_matches_measured_metrics(checked: coverage.PlanCoverage) -> None:
    """Every declared threshold must actually be measured, and vice versa."""
    assert set(S.THRESHOLDS) == set(checked.metrics)


def test_all_documents_are_planned(checked: coverage.PlanCoverage) -> None:
    assert len(checked.plans) == 100
    assert set(checked.plans) == set(S.ALL_DOC_IDS)


def test_category_metrics_are_exact(checked: coverage.PlanCoverage) -> None:
    for category, expected in S.CATEGORY_COUNTS.items():
        assert checked.metrics[f"category.{category}"].value == expected


def test_extractability_metrics_are_exact(checked: coverage.PlanCoverage) -> None:
    assert checked.metrics["extract.native"].value == 77
    assert checked.metrics["extract.scanned"].value == 15
    assert checked.metrics["extract.ocr"].value == 8


def test_answer_bank_size(checked: coverage.PlanCoverage) -> None:
    stats = checked.stats
    assert stats["document_count"] == 100
    assert stats["per_doc_min"] >= 10
    assert stats["per_doc_max"] <= 20
    assert stats["question_count"] >= 1000
    for kind in ("factual", "table", "cross_page", "cross_doc", "figure", "refusal"):
        assert stats["per_kind"].get(kind, 0) > 0, f"no {kind} questions at all"


def test_every_document_has_a_full_question_mix(checked: coverage.PlanCoverage) -> None:
    for doc_id, plan in checked.plans.items():
        counts = plan.kind_counts()
        assert 10 <= plan.question_count <= 20, f"{doc_id}: {plan.question_count} questions"
        assert counts["refusal"] >= 1, f"{doc_id}: no refusal question"
        assert counts["factual"] >= 1, f"{doc_id}: no factual question"
        spec = checked.specs[doc_id]
        if spec.table_styles:
            assert counts["table"] >= 1, f"{doc_id}: has tables but no table question"
        if plan.figure_count() > 0:
            assert counts["figure"] >= 1, f"{doc_id}: has figures but no figure question"


def test_realised_table_styles_cover_the_spec(checked: coverage.PlanCoverage) -> None:
    """Content modules must actually emit every table class the spec demands."""
    missing: Dict[str, list] = {}
    for doc_id, spec in checked.specs.items():
        realised = set(checked.plans[doc_id].table_styles())
        for style in spec.table_styles:
            if style not in realised:
                missing.setdefault(doc_id, []).append(style)
    assert not missing, f"table styles not realised: {missing}"


def test_no_figure_documents_really_have_no_figure(checked: coverage.PlanCoverage) -> None:
    for doc_id in S.NO_FIGURE_DOCS:
        assert checked.plans[doc_id].figure_count() == 0, doc_id


def test_figure_image_documents_really_have_a_formula_image(checked: coverage.PlanCoverage) -> None:
    for doc_id in S.FORMULA_IMAGE_DOCS:
        arts = {b.art for b in checked.plans[doc_id].body if hasattr(b, "art")}
        assert "formula" in arts, doc_id


def test_advert_documents_really_have_adverts(checked: coverage.PlanCoverage) -> None:
    for doc_id in S.AD_DOCS:
        assert checked.plans[doc_id].ad_count() > 0, doc_id


def test_coverage_is_stable_for_a_different_seed() -> None:
    """Thresholds must hold for any seed: the allocation is not random."""
    other = coverage.plan_coverage(seed=20260101)
    assert other.all_passed, [m.line() for m in other.failed]
