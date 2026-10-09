"""Build the standard answer bank from :class:`~.facts.DocPlan` objects.

The bank is *derived*, never hand-authored: every question comes from a
:class:`~.facts.Fact` whose ``anchor`` was written into the document by the same
expression that registered it.  This module only serialises, orders and counts.

Output shape (one file per document, plus an index and a stats roll-up)::

    {
      "doc_id": "FIN-07",
      "source_pdf": "pdf/FIN/FIN-07_v1.pdf",
      "expected_ingestion": {"expect": "success", ...},
      "question_count": 15,
      "kind_counts": {"factual": 8, "table": 3, ...},
      "questions": [ {...}, ... ]
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.facts import (
    CROSS_DOC,
    CROSS_PAGE,
    FACTUAL,
    FIGURE,
    QUESTION_KINDS,
    REFUSAL,
    TABLE,
    DocPlan,
)

#: Review-friendly ordering: simple kinds first, hardest last, refusals at the end.
KIND_ORDER: Tuple[str, ...] = (FACTUAL, TABLE, FIGURE, CROSS_PAGE, CROSS_DOC, REFUSAL)

#: Ingestion chunker settings the expectation is derived from.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@dataclass
class QaDocument:
    """The answer bank for a single document."""

    doc_id: str
    payload: Dict[str, Any]

    @property
    def questions(self) -> List[Dict[str, Any]]:
        return self.payload["questions"]

    @property
    def question_count(self) -> int:
        return len(self.questions)


def expected_ingestion(spec: S.DocSpec, plan: DocPlan) -> Dict[str, Any]:
    """What the ingestion pipeline should do with this document.

    Scanned documents have no text layer, so ``DocumentChunker.split_document``
    raises ``ValueError("... has no text content to split")`` and the pipeline
    reports failure.  That is the intended, reportable outcome.
    """
    figures = plan.figure_count()
    if spec.is_scanned:
        return {
            "expect": "failure",
            "error_contains": "no text content",
            "reason": "扫描件无文本层，DocumentChunker 会因空文本抛错；需先 OCR",
            "chunk_count_range": [0, 0],
            "image_count_range": [spec.pages, spec.pages],
        }

    chars = int(plan.notes.get("chars", 0) or 0)
    effective = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    low = max(1, chars // (effective * 2))
    high = max(low + 1, chars // max(1, effective // 2))
    if spec.is_ocr:
        image_range = [spec.pages, spec.pages]
    else:
        image_range = [figures, figures]
    return {
        "expect": "success",
        "reason": "存在文本层，可正常切分与向量化" if spec.is_ocr else "原生文本层",
        "chunk_count_range": [low, high],
        "image_count_range": image_range,
    }


def build_doc_qa(spec: S.DocSpec, plan: DocPlan, source_relpath: str) -> QaDocument:
    """Serialise one document's answer bank."""
    expectation = expected_ingestion(spec, plan)

    records: List[Dict[str, Any]] = []
    for fact in plan.facts:
        records.append(fact.to_qa_record(spec.doc_id, 0, [source_relpath]))
    for refusal in plan.refusals:
        records.append(refusal.to_qa_record(spec.doc_id, 0, [source_relpath]))

    records.sort(key=lambda r: (KIND_ORDER.index(r["category"]), r["fact_id"]))
    for index, record in enumerate(records, start=1):
        record["qid"] = f"{spec.doc_id}-Q{index:02d}"

    payload: Dict[str, Any] = {
        "doc_id": spec.doc_id,
        "category": spec.category,
        "category_name": S.CATEGORY_NAMES[spec.category],
        "title": spec.title,
        "title_alt": spec.title_alt,
        "source_style": spec.source_style,
        "source_pdf": source_relpath,
        "language": spec.language,
        "layout": spec.layout,
        "extractability": spec.extractability,
        "pages": spec.pages,
        "version_label": spec.version_label,
        "version_pair": spec.version_pair,
        "scan_profile": spec.scan_profile,
        "watermark": spec.watermark,
        "expected_ingestion": expectation,
        "question_count": len(records),
        "kind_counts": _kind_counts(records),
        "questions": records,
    }
    return QaDocument(doc_id=spec.doc_id, payload=payload)


def _kind_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {kind: 0 for kind in QUESTION_KINDS}
    for record in records:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    return counts


def build_index(documents: Iterable[QaDocument], seed: int, version: str) -> Dict[str, Any]:
    """Flat index of every question in the corpus, with provenance."""
    ordered = sorted(documents, key=lambda d: d.doc_id)
    questions: List[Dict[str, Any]] = []
    for doc in ordered:
        for record in doc.questions:
            item = dict(record)
            item["doc_id"] = doc.doc_id
            questions.append(item)
    return {
        "description": (
            "Standard answer bank for the 100-document RAG test corpus. "
            "Each question's evidence.anchor_text is guaranteed to occur in the "
            "text that markitdown extracts from the listed source PDF; refusal "
            "questions instead assert their reject_terms occur nowhere in the corpus."
        ),
        "version": version,
        "seed": seed,
        "document_count": len(ordered),
        "question_count": len(questions),
        "questions": questions,
    }


def build_stats(documents: Sequence[QaDocument]) -> Dict[str, Any]:
    """Aggregate roll-up used by the coverage report and the verifier."""
    per_kind: Dict[str, int] = {kind: 0 for kind in QUESTION_KINDS}
    per_doc: Dict[str, int] = {}
    per_category: Dict[str, int] = {}
    refusal_counts: Dict[str, int] = {}

    for doc in documents:
        per_doc[doc.doc_id] = doc.question_count
        category = doc.payload["category"]
        per_category[category] = per_category.get(category, 0) + doc.question_count
        refusals = sum(1 for q in doc.questions if q["category"] == REFUSAL)
        refusal_counts[doc.doc_id] = refusals
        for record in doc.questions:
            per_kind[record["category"]] = per_kind.get(record["category"], 0) + 1

    counts = list(per_doc.values()) or [0]
    return {
        "document_count": len(documents),
        "question_count": sum(counts),
        "per_kind": per_kind,
        "per_category": per_category,
        "per_doc": per_doc,
        "per_doc_min": min(counts),
        "per_doc_max": max(counts),
        "refusal_per_doc_min": min(refusal_counts.values()) if refusal_counts else 0,
        "refusal_counts": refusal_counts,
    }
