"""Spec invariants for the 100-document corpus.

These tests are fast: they only inspect :mod:`spec`, so they give immediate
feedback when the allocation tables are edited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.corpus_gen import facts as F  # noqa: E402
from tests.fixtures.corpus_gen import spec as S  # noqa: E402


def test_total_and_category_counts() -> None:
    assert len(S.SPECS) == 100
    for category, expected in S.CATEGORY_COUNTS.items():
        actual = sum(1 for spec in S.SPECS if spec.category == category)
        assert actual == expected, f"{category}: {actual} != {expected}"
    assert sum(S.CATEGORY_COUNTS.values()) == 100


def test_document_ids_are_unique_and_grouped_by_category() -> None:
    ids = [spec.doc_id for spec in S.SPECS]
    assert len(ids) == len(set(ids))
    # Documents are emitted grouped by category, in CATEGORIES order, with
    # contiguous 1-based numbering inside each category.
    offset = 0
    for category in S.CATEGORIES:
        count = S.CATEGORY_COUNTS[category]
        block = [s for s in S.SPECS[offset:offset + count]]
        assert all(s.category == category for s in block)
        assert [s.seq for s in block] == list(range(1, count + 1))
        offset += count
    assert offset == 100


def test_titles_are_unique() -> None:
    titles = [spec.title for spec in S.SPECS]
    assert len(titles) == len(set(titles)), "duplicate titles would make retrieval ambiguous"


def test_extractability_partition() -> None:
    native = [s for s in S.SPECS if s.extractability == S.NATIVE]
    scanned = [s for s in S.SPECS if s.extractability == S.SCANNED]
    ocr = [s for s in S.SPECS if s.extractability == S.OCR]

    assert (len(native), len(scanned), len(ocr)) == (77, 15, 8)
    assert not (S.SCANNED_DOCS & S.OCR_DOCS), "scanned and OCR sets must be disjoint"
    assert {s.doc_id for s in scanned} == S.SCANNED_DOCS
    assert {s.doc_id for s in ocr} == S.OCR_DOCS


def test_language_sets_partition_the_corpus() -> None:
    all_ids = set(S.ALL_DOC_IDS)
    assert not (S.LANG_EN_DOCS & S.LANG_MIXED_DOCS)
    assert S.LANG_EN_DOCS | S.LANG_MIXED_DOCS <= all_ids
    zh = all_ids - S.LANG_EN_DOCS - S.LANG_MIXED_DOCS
    assert len(zh) + len(S.LANG_EN_DOCS) + len(S.LANG_MIXED_DOCS) == 100
    assert not (zh & S.LANG_EN_DOCS) and not (zh & S.LANG_MIXED_DOCS)


def test_language_thresholds_hold_on_extractable_subset() -> None:
    extractable = {s.doc_id for s in S.SPECS if s.extractable}
    all_ids = set(S.ALL_DOC_IDS)
    zh = (all_ids - S.LANG_EN_DOCS - S.LANG_MIXED_DOCS) & extractable
    assert len(zh) >= 30, f"only {len(zh)} Chinese extractable documents"
    assert len(S.LANG_EN_DOCS & extractable) >= 30
    assert len(S.LANG_MIXED_DOCS & extractable) >= 15
    assert S.LANG_MIXED_DOCS <= extractable, "mixed-language documents must be extractable"


def test_version_pairs_are_consistent() -> None:
    assert len(S.VERSION_PAIRS) == 6
    for v1, v2 in S.VERSION_PAIRS:
        a, b = S.SPEC_BY_ID[v1], S.SPEC_BY_ID[v2]
        assert a.category == b.category
        assert a.pages == b.pages, f"{v1}/{v2} must have equal length"
        assert a.layout == b.layout, f"{v1}/{v2} must share a layout"
        assert a.language == b.language, f"{v1}/{v2} must share a language"
        assert a.extractable and b.extractable, f"{v1}/{v2} must both be extractable"
        assert a.version_label == "v1" and b.version_label == "v2"
        assert a.version_pair == v2 and b.version_pair == v1
        assert a.filename() != b.filename()


def test_scanned_and_vertical_documents_are_not_anchor_verifiable() -> None:
    """Anchors cannot be asserted when text is absent or reading order is lost."""
    for spec in S.SPECS:
        expected = spec.extractability != S.SCANNED and spec.layout != S.VERTICAL
        assert spec.anchors_verifiable is expected, spec.doc_id


def test_complex_table_documents_have_room_for_their_tables() -> None:
    for doc_id in S.COMPLEX_TABLE_DOCS:
        spec = S.SPEC_BY_ID[doc_id]
        assert spec.pages >= S.COMPLEX_TABLE_MIN_PAGES or doc_id in S.PAGE_FIFTY
        assert len(spec.table_styles) == 4


def test_table_styles_are_known_and_two_column_papers_stay_narrow() -> None:
    for spec in S.SPECS:
        for style in spec.table_styles:
            assert style in S.TABLE_DIFFICULTY_STYLES, (spec.doc_id, style)
        if spec.layout == S.TWO_COL:
            # A five-column table inside a half-width frame would be unreadable.
            assert S.TABLE_MULTIPAGE not in spec.table_styles
            assert S.TABLE_MULTIHEADER not in spec.table_styles


def test_difficulty_table_styles_stay_in_sync_with_facts() -> None:
    """``facts`` keeps a literal copy to avoid importing ``spec``; assert parity."""
    assert set(F._DIFFICULTY_TABLE_STYLES) == set(S.TABLE_DIFFICULTY_STYLES)


def test_page_count_buckets() -> None:
    pages = [s.pages for s in S.SPECS]
    assert sum(1 for p in pages if p == 1) == 3
    assert sum(1 for p in pages if 2 <= p <= 3) == 10
    assert sum(1 for p in pages if 8 <= p <= 14) >= 25
    assert sum(1 for p in pages if 40 <= p <= 60) == 6
    assert sum(1 for p in pages if p >= 200) == 1


def test_page_count_is_deterministic_across_calls() -> None:
    first = [(s.doc_id, s.pages) for s in S.build_specs()]
    second = [(s.doc_id, s.pages) for s in S.build_specs()]
    assert first == second


def test_scanned_profiles_are_assigned_and_valid() -> None:
    profiles = {name for name, *_ in S.SCAN_PROFILES}
    for spec in S.SPECS:
        if spec.extractability in (S.SCANNED, S.OCR):
            assert spec.scan_profile in profiles, spec.doc_id
            assert spec.scan_dpi >= 72
        else:
            assert spec.scan_profile == "clean"


def test_threshold_table_is_well_formed() -> None:
    for name, (low, high, description) in S.THRESHOLDS.items():
        assert isinstance(low, int) and low >= 0, name
        assert high is None or high >= low, name
        assert description.strip(), name
