#!/usr/bin/env python
"""Verification gate for a generated corpus.

This is the check that makes the corpus trustworthy: it re-derives everything
from the **files on disk** rather than trusting the build, and it validates each
question's claimed evidence against the text the *production* extractor
(``markitdown``, the same call :class:`~.pdf_loader.PdfLoader` makes) actually
recovers.

Usage::

    python tests/fixtures/corpus_gen/verify_corpus.py --out data/test_corpus
    python tests/fixtures/corpus_gen/verify_corpus.py --out data/test_corpus --json
    python tests/fixtures/corpus_gen/verify_corpus.py --out data/test_corpus \\
        --mode post-ingest --collection corpus100

Exit codes: ``0`` all checks passed, ``1`` at least one check failed.

Checks
------
1. every declared PDF exists, opens, and has the declared page count and SHA256;
2. coverage thresholds recomputed from the manifest;
3. extraction outcomes per document class (native / scanned / OCR);
4. **every question's anchor actually occurs in the extracted text**
   (exact after whitespace normalisation for native, fuzzy for OCR);
5. **every refusal question's reject terms occur nowhere in the corpus**;
6. watermark, running header/footer, bookmarks and embedded images are present;
7. answer-bank shape: 10-20 questions per document, >= 1 refusal each;
8. version pairs agree on page count, layout and language;
9. determinism: extracted-text hashes match the previous run at this seed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.corpus_gen import spec as S  # noqa: E402
from tests.fixtures.corpus_gen.fsutil import ensure_dir, sha256_file  # noqa: E402

#: A scanned page must extract no meaningful text.
SCANNED_TEXT_LIMIT = 20

#: Minimum trigram overlap for an anchor in a document whose text is imperfect.
#: Applied to OCR'd text (recognition errors) and to watermarked text (the
#: diagonal watermark is real text and interleaves with body text in the
#: extractor's reading order, so a run of body characters can be interrupted).
LOOSE_ANCHOR_MIN_RATIO = 0.55

#: Mean overlap across all loosely-matched anchors.  The per-anchor floor alone
#: could hide a systematic extraction failure; this cannot.
LOOSE_ANCHOR_MEAN_RATIO = 0.85

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Strip all whitespace so line wrapping cannot break an anchor match."""
    return _WS.sub("", text or "")


def trigrams(text: str) -> set:
    return {text[i:i + 3] for i in range(max(0, len(text) - 2))}


def fuzzy_ratio(anchor_norm: str, doc_trigrams: set) -> float:
    """Fraction of the anchor's 3-grams present in the document."""
    grams = trigrams(anchor_norm)
    if not grams:
        return 1.0
    return len(grams & doc_trigrams) / float(len(grams))


def watermark_in_text_objects(pdf_path: Path, marker: str) -> bool:
    """True when *marker* survives as a single text span in the PDF.

    PyMuPDF keeps one ``drawString`` call as one span, so a rotated watermark is
    recovered intact even when the flowing text extractor interleaves its glyphs
    with the body or with table cells.  That makes this the reliable way to prove
    the watermark really is in the text layer.
    """
    import pymupdf

    target = normalize(marker)
    doc = pymupdf.open(str(pdf_path))
    try:
        for page in doc:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if target in normalize(str(span.get("text", ""))):
                            return True
    finally:
        doc.close()
    return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "failure_count": len(self.failures),
            "failures": self.failures[:40],
        }


class Report:
    """Collects check results and prints a summary."""

    def __init__(self) -> None:
        self.checks: List[CheckResult] = []
        self.extract_hashes: Dict[str, str] = {}

    def add(self, name: str, failures: Sequence[str], detail: str = "") -> CheckResult:
        result = CheckResult(name=name, passed=not failures, detail=detail,
                             failures=list(failures))
        self.checks.append(result)
        status = "OK  " if result.passed else "FAIL"
        extra = f"  {detail}" if detail else ""
        print(f"[{status}] {name}{extra}")
        for failure in result.failures[:8]:
            print(f"        - {failure}")
        if len(result.failures) > 8:
            print(f"        ... and {len(result.failures) - 8} more")
        return result

    @property
    def failed(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": len(self.checks) - len(self.failed),
            "total": len(self.checks),
            "all_passed": not self.failed,
            "checks": [c.to_dict() for c in self.checks],
            "extract_hashes": self.extract_hashes,
        }


# ---------------------------------------------------------------------------
# Extraction (cached by content hash)
# ---------------------------------------------------------------------------


class Extractor:
    """Extract PDF text with markitdown, caching by file content hash.

    markitdown is used deliberately: it is exactly what ``PdfLoader`` calls, so
    the verification measures the text the RAG system will really see.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self._markitdown = None

    def text(self, pdf_path: Path, digest: str) -> str:
        cached = self.cache_dir / f"{digest}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        if self._markitdown is None:
            from markitdown import MarkItDown

            self._markitdown = MarkItDown()
        result = self._markitdown.convert(str(pdf_path))
        text = getattr(result, "text_content", None) or str(result)
        cached.write_text(text, encoding="utf-8")
        return text


# ---------------------------------------------------------------------------
# Cheks
# ---------------------------------------------------------------------------


def check_files(
    report: Report,
    out_root: Path,
    manifest: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Every declared file exists, opens, and matches its recorded facts."""
    import pypdfium2 as pdfium

    failures: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}

    for entry in manifest["documents"]:
        doc_id = entry["doc_id"]
        path = out_root / entry["relpath"]
        info: Dict[str, Any] = {"path": path, "entry": entry}

        if not path.exists():
            failures.append(f"{doc_id}: missing {entry['relpath']}")
            continue

        digest = sha256_file(path)
        if digest != entry["sha256"]:
            failures.append(f"{doc_id}: sha256 mismatch on disk")
        info["sha256"] = digest

        try:
            doc = pdfium.PdfDocument(str(path))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{doc_id}: cannot open as PDF ({exc})")
            continue
        try:
            pages = len(doc)
        finally:
            doc.close()
        info["pages"] = pages
        if pages != entry["pages"]:
            failures.append(f"{doc_id}: {pages} pages on disk, manifest says {entry['pages']}")
        if entry.get("pages_actual") != entry["pages"]:
            failures.append(
                f"{doc_id}: build recorded pages_actual={entry.get('pages_actual')} "
                f"but pages={entry['pages']}"
            )
        meta[doc_id] = info

    report.add("files: exist / open / page count / sha256", failures,
               f"{len(manifest['documents'])} documents")
    return meta


def check_coverage(report: Report, manifest: Dict[str, Any], qa_stats: Dict[str, Any]) -> None:
    """Recompute the coverage matrix from the manifest and assert thresholds."""
    docs = manifest["documents"]
    ids = [d["doc_id"] for d in docs]
    extractable = [d for d in docs if d["extractability"] in (S.NATIVE, S.OCR)]

    measured: Dict[str, int] = {}
    measured["total"] = len(docs)
    for category in S.CATEGORIES:
        measured[f"category.{category}"] = sum(1 for d in docs if d["category"] == category)
    for key, mode in (("extract.native", S.NATIVE), ("extract.scanned", S.SCANNED),
                      ("extract.ocr", S.OCR)):
        measured[key] = sum(1 for d in docs if d["extractability"] == mode)

    measured["dirty.watermark"] = sum(1 for d in docs if d.get("watermark"))
    measured["dirty.complex_tables"] = sum(
        1 for d in docs if len([s for s in d.get("table_styles", {}) if s in S.TABLE_DIFFICULTY_STYLES]) >= 3
    )
    measured["dirty.messy_headings"] = sum(1 for d in docs if d.get("messy_headings"))
    measured["dirty.ads"] = sum(1 for d in docs if d.get("ads", 0) > 0)
    measured["dirty.footnotes"] = sum(1 for d in docs if d.get("footnotes", 0) > 0)

    measured["layout.two_column"] = sum(1 for d in docs if d["layout"] == S.TWO_COL)
    measured["layout.three_column"] = sum(1 for d in docs if d["layout"] == S.THREE_COL)
    measured["layout.vertical"] = sum(1 for d in docs if d["layout"] == S.VERTICAL)

    for key, lang in (("lang.zh", S.LANG_ZH), ("lang.en", S.LANG_EN), ("lang.mixed", S.LANG_MIXED)):
        measured[key] = sum(1 for d in extractable if d["language"] == lang)

    for key, style in (("table.simple", S.TABLE_SIMPLE), ("table.merged", S.TABLE_MERGED),
                       ("table.multipage", S.TABLE_MULTIPAGE),
                       ("table.borderless", S.TABLE_BORDERLESS),
                       ("table.multiheader", S.TABLE_MULTIHEADER)):
        measured[key] = sum(1 for d in docs if d.get("table_styles", {}).get(style, 0) > 0)

    # Figures: which authored kinds each document carries (from the plan), so
    # the metric is independent of how many images the renderer emitted.
    def has_kind(doc: Dict[str, Any], predicate) -> bool:
        return any(predicate(kind) for kind in doc.get("figure_kinds", []))

    measured["figure.flow"] = sum(1 for d in docs if has_kind(d, lambda k: k == "flow"))
    measured["figure.chart"] = sum(
        1 for d in docs if has_kind(d, lambda k: k in ("bar", "pie", "line")))
    measured["figure.photo"] = sum(1 for d in docs if has_kind(d, lambda k: k == "photo"))
    measured["figure.formula_image"] = sum(
        1 for d in docs if has_kind(d, lambda k: k == "formula"))
    measured["figure.none"] = sum(1 for d in extractable if d.get("figures", 0) == 0)

    def page_bucket(pred) -> int:
        return sum(1 for d in docs if pred(d["pages"]))

    measured["length.one_page"] = page_bucket(lambda p: p == 1)
    measured["length.short"] = page_bucket(lambda p: 2 <= p <= 3)
    measured["length.mid"] = page_bucket(lambda p: 8 <= p <= 14)
    measured["length.fifty"] = page_bucket(lambda p: 40 <= p <= 60)
    measured["length.huge"] = page_bucket(lambda p: p >= 200)

    # has_toc is defined as "at least 8 pages" in the spec.
    measured["structure.toc"] = sum(1 for d in docs if d["pages"] >= 8)
    measured["structure.bookmarks"] = sum(1 for d in docs if d.get("bookmarks_requested"))
    measured["structure.version_pairs"] = sum(1 for d in docs if d.get("version_label") == "v1")
    measured["structure.references"] = sum(1 for d in docs if d.get("references", 0) > 0)

    measured["qa.per_doc_min"] = qa_stats.get("per_doc_min", 0)
    measured["qa.per_doc_max"] = qa_stats.get("per_doc_max", 0)
    measured["qa.refusal_per_doc"] = qa_stats.get("refusal_per_doc_min", 0)
    measured["qa.total"] = qa_stats.get("question_count", 0)

    failures: List[str] = []
    for name, (low, high, description) in S.THRESHOLDS.items():
        value = measured.get(name)
        if value is None:
            failures.append(f"{name}: not measurable from the manifest")
            continue
        if value < low:
            failures.append(f"{name} = {value} < {low} ({description})")
        elif high is not None and value > high:
            failures.append(f"{name} = {value} > {high} ({description})")

    report.add("coverage: thresholds", failures,
               f"{len(S.THRESHOLDS) - len(failures)}/{len(S.THRESHOLDS)} metrics within range")


def check_extraction(
    report: Report,
    out_root: Path,
    manifest: Dict[str, Any],
    meta: Dict[str, Dict[str, Any]],
    extractor: Extractor,
) -> Dict[str, str]:
    """Per-class extraction expectations, watermark presence and footers."""
    failures: List[str] = []
    texts: Dict[str, str] = {}
    watermarks_total = 0
    watermarks_contiguous = 0

    for entry in manifest["documents"]:
        doc_id = entry["doc_id"]
        info = meta.get(doc_id)
        if not info:
            continue
        text = extractor.text(info["path"], info["sha256"])
        texts[doc_id] = text
        norm = normalize(text)
        length = len(norm)

        if entry["extractability"] == S.NATIVE:
            if length == 0:
                failures.append(f"{doc_id}: native PDF extracted no text")
        elif entry["extractability"] == S.SCANNED:
            if length > SCANNED_TEXT_LIMIT:
                failures.append(
                    f"{doc_id}: scanned PDF should have no text layer but extracted "
                    f"{length} chars"
                )
        else:  # OCR
            if length == 0:
                failures.append(f"{doc_id}: OCR PDF extracted no text")

        if entry.get("watermark") and entry["extractability"] == S.NATIVE:
            watermark = entry.get("watermark_text", "")
            watermarks_total += 1
            if not watermark_in_text_objects(info["path"], watermark):
                failures.append(f"{doc_id}: watermark {watermark!r} not in the text layer")
            elif normalize(watermark) in norm:
                watermarks_contiguous += 1

        if entry["pages"] >= 3 and entry["extractability"] == S.NATIVE:
            has_footer = bool(re.search(r"第\d+页", norm)) or bool(re.search(r"page\d+", norm))
            if not has_footer:
                failures.append(f"{doc_id}: running footer not found in extraction")

    detail = f"{len(texts)} documents extracted"
    if watermarks_total:
        detail += (
            f"; {watermarks_total - watermarks_contiguous}/{watermarks_total} watermarks "
            f"interleaved with body text in flow extraction"
        )
    report.add("extraction: per-class text expectations", failures, detail)
    return texts


def check_anchors(
    report: Report,
    out_root: Path,
    texts: Dict[str, str],
    manifest: Dict[str, Any],
) -> None:
    """Every question's anchor must occur in the extracted text (or be OCR-fuzzy)."""
    failures: List[str] = []
    exact_checked = 0
    fuzzy_checked = 0
    skipped = 0
    ratios: List[float] = []

    entry_by_id = {e["doc_id"]: e for e in manifest["documents"]}

    for doc_id, entry in entry_by_id.items():
        qa_path = out_root / "qa" / f"{doc_id}.json"
        if not qa_path.exists():
            failures.append(f"{doc_id}: missing qa/{doc_id}.json")
            continue
        payload = json.loads(qa_path.read_text(encoding="utf-8"))

        if not entry.get("anchors_verifiable", True):
            # Scanned: no text layer.  Vertical: extraction order does not match
            # the drawing order.  Anchors are recorded but cannot be asserted.
            skipped += len(payload["questions"])
            continue

        text = texts.get(doc_id, "")
        norm = normalize(text)
        loose = entry["extractability"] == S.OCR or bool(entry.get("watermark"))
        grams = trigrams(norm) if loose else set()

        for question in payload["questions"]:
            if question["expect_no_answer"]:
                continue
            anchors = [question["evidence"].get("anchor_text")]
            if question["evidence"].get("secondary_anchor_text"):
                anchors.append(question["evidence"]["secondary_anchor_text"])
            for anchor in anchors:
                if not anchor:
                    continue
                anchor_norm = normalize(anchor)
                if loose:
                    ratio = fuzzy_ratio(anchor_norm, grams)
                    ratios.append(ratio)
                    fuzzy_checked += 1
                    if ratio < LOOSE_ANCHOR_MIN_RATIO:
                        failures.append(
                            f"{doc_id}/{question['qid']}: fuzzy anchor match "
                            f"{ratio:.2f} < {LOOSE_ANCHOR_MIN_RATIO} ({anchor[:40]!r})"
                        )
                else:
                    exact_checked += 1
                    if anchor_norm not in norm:
                        failures.append(
                            f"{doc_id}/{question['qid']}: anchor not found in extraction "
                            f"({anchor[:48]!r})"
                        )

    mean_ratio = sum(ratios) / len(ratios) if ratios else 1.0
    detail = (
        f"{exact_checked} exact, {fuzzy_checked} fuzzy (mean {mean_ratio:.3f}), "
        f"{skipped} skipped (scanned / vertical)"
    )
    if ratios and mean_ratio < LOOSE_ANCHOR_MEAN_RATIO:
        failures.append(
            f"aggregate fuzzy anchor overlap {mean_ratio:.3f} < {LOOSE_ANCHOR_MEAN_RATIO} "
            f"across {len(ratios)} anchors"
        )
    report.add("anchors: evidence present in extracted text", failures, detail)


def check_refusals(report: Report, out_root: Path, texts: Dict[str, str]) -> None:
    """Refusal questions must be unanswerable: their terms appear nowhere."""
    failures: List[str] = []
    checked = 0
    corpus_norm = "\n".join(normalize(t) for t in texts.values())

    for qa_path in sorted((out_root / "qa").glob("*.json")):
        payload = json.loads(qa_path.read_text(encoding="utf-8"))
        for question in payload["questions"]:
            if not question["expect_no_answer"]:
                continue
            for term in question["reject_terms"]:
                checked += 1
                if normalize(term) in corpus_norm:
                    failures.append(
                        f"{payload['doc_id']}/{question['qid']}: reject term "
                        f"{term!r} DOES occur in the corpus"
                    )

    report.add("refusals: reject terms absent corpus-wide", failures,
               f"{checked} terms checked")


def check_structure(
    report: Report,
    manifest: Dict[str, Any],
    meta: Dict[str, Dict[str, Any]],
) -> None:
    """Bookmarks, embedded images and version-pair consistency."""
    import pymupdf

    failures: List[str] = []
    bookmark_checked = 0
    image_checked = 0

    for entry in manifest["documents"]:
        doc_id = entry["doc_id"]
        info = meta.get(doc_id)
        if not info:
            continue
        try:
            doc = pymupdf.open(str(info["path"]))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{doc_id}: pymupdf cannot open ({exc})")
            continue
        try:
            if entry.get("bookmarks_requested") and entry["extractability"] == S.NATIVE:
                toc = doc.get_toc()
                bookmark_checked += 1
                if not toc:
                    failures.append(f"{doc_id}: expected PDF bookmarks, found none")

            images = 0
            for page in doc:
                images += len(page.get_images(full=True))
            image_checked += 1
            expected = entry.get("figures", 0)
            if entry["extractability"] == S.NATIVE:
                if expected == 0 and images != 0:
                    failures.append(
                        f"{doc_id}: no authored figures but {images} embedded images"
                    )
                elif expected > 0 and images < 1:
                    failures.append(
                        f"{doc_id}: {expected} authored figures but no embedded image found"
                    )
            elif entry["extractability"] == S.SCANNED:
                if images < entry["pages"]:
                    failures.append(
                        f"{doc_id}: scanned pages {entry['pages']} but only {images} images"
                    )
        finally:
            doc.close()

    # version pairs must agree on shape and only differ in content.  When the
    # corpus is only a subset (smoke builds), an absent partner is informational
    # rather than a failure.
    by_id = {e["doc_id"]: e for e in manifest["documents"]}
    full_corpus = manifest.get("document_count", 0) >= len(S.SPECS)
    incomplete: List[str] = []
    for v1, v2 in S.VERSION_PAIRS:
        a, b = by_id.get(v1), by_id.get(v2)
        if not a or not b:
            if full_corpus:
                failures.append(f"version pair {v1}/{v2}: missing member")
            else:
                incomplete.append(f"{v1}/{v2}")
            continue
        if a["pages"] != b["pages"]:
            failures.append(f"version pair {v1}/{v2}: pages {a['pages']} != {b['pages']}")
        if a["layout"] != b["layout"]:
            failures.append(f"version pair {v1}/{v2}: layout differs")
        if a["language"] != b["language"]:
            failures.append(f"version pair {v1}/{v2}: language differs")
        if a["sha256"] == b["sha256"]:
            failures.append(f"version pair {v1}/{v2}: identical files, no version change")

    report.add("structure: bookmarks / images / version pairs", failures,
               f"{bookmark_checked} bookmark docs, {image_checked} inspected"
               + (f", {len(incomplete)} pair(s) not built" if incomplete else ""))


def check_answer_bank(report: Report, out_root: Path, manifest: Dict[str, Any]) -> None:
    """Answer-bank shape: per-document count, refusal presence, category coverage.

    Only categories the document can actually support are required: a document
    with no table cannot pose a table question, and the spec reallocates that
    quota to factual questions instead.
    """
    failures: List[str] = []

    for entry in manifest["documents"]:
        doc_id = entry["doc_id"]
        qa_path = out_root / "qa" / f"{doc_id}.json"
        if not qa_path.exists():
            failures.append(f"{doc_id}: missing answer bank")
            continue
        payload = json.loads(qa_path.read_text(encoding="utf-8"))
        questions = payload["questions"]
        count = len(questions)

        if not (10 <= count <= 20):
            failures.append(f"{doc_id}: {count} questions, expected 10-20")
        if count != entry.get("questions"):
            failures.append(
                f"{doc_id}: manifest says {entry.get('questions')} questions, file has {count}"
            )
        refusals = sum(1 for q in questions if q["category"] == "refusal")
        if refusals < 1:
            failures.append(f"{doc_id}: no refusal question")

        kinds = {q["category"] for q in questions}
        required = {"factual", "refusal"}
        if entry.get("table_styles"):
            required.add("table")
        if entry.get("figures", 0) > 0:
            required.add("figure")
        missing = required - kinds
        if missing:
            failures.append(f"{doc_id}: missing question categories {sorted(missing)}")

        for question in questions:
            if question["expect_no_answer"]:
                continue
            if not question["evidence"].get("anchor_text"):
                failures.append(f"{doc_id}/{question['qid']}: no anchor_text")
            if not question["expected_sources"]:
                failures.append(f"{doc_id}/{question['qid']}: no expected_sources")

    report.add("answer bank: shape and coverage", failures,
               f"{len(manifest['documents'])} documents")


def check_determinism(
    report: Report,
    previous: Optional[Dict[str, Any]],
    extract_hashes: Dict[str, str],
    seed: int,
) -> None:
    """Extracted-text hashes must match the previous run at the same seed."""
    if previous is None:
        report.add("determinism: vs previous run", [], "no previous report; recorded baseline")
        return
    if previous.get("seed") != seed:
        report.add("determinism: vs previous run", [],
                   f"previous run used seed {previous.get('seed')}; recorded new baseline")
        return

    import hashlib

    old = previous.get("extract_hashes", {})
    failures: List[str] = []
    for doc_id, digest in extract_hashes.items():
        if doc_id in old and old[doc_id] != digest:
            failures.append(f"{doc_id}: extracted text changed between identical seeds")
    report.add("determinism: vs previous run", failures,
               f"{len(extract_hashes)} documents compared")


def check_post_ingest(
    report: Report,
    out_root: Path,
    collection: str,
) -> None:
    """Compare what ingestion actually did against ``expected_ingestion.json``."""
    import sqlite3

    expected_path = out_root / "expected_ingestion.json"
    if not expected_path.exists():
        report.add("post-ingest: expectations file", ["expected_ingestion.json missing"])
        return
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["documents"]

    db_path = _REPO_ROOT / "data" / "db" / "ingestion_history.db"
    if not db_path.exists():
        report.add("post-ingest: ingestion history",
                   [f"no ingestion database at {db_path}; run scripts/ingest.py first"])
        return

    failures: List[str] = []
    succeeded: set = set()
    failed: Dict[str, str] = {}
    with sqlite3.connect(str(db_path)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(ingestion_history)")}
        path_col = "file_path" if "file_path" in columns else None
        if path_col is None:
            report.add("post-ingest: ingestion history",
                       ["ingestion_history has no file_path column"])
            return
        for file_path, status, error in conn.execute(
            f"SELECT {path_col}, status, error_msg FROM ingestion_history"
        ):
            name = Path(str(file_path)).stem
            doc_id = name.split("_v")[0]
            if status == "success":
                succeeded.add(doc_id)
            else:
                failed[doc_id] = str(error or "")

    for doc_id, expectation in expected.items():
        if doc_id not in succeeded and doc_id not in failed:
            failures.append(f"{doc_id}: not present in ingestion history")
            continue
        if expectation["expect"] == "failure":
            if doc_id in succeeded:
                failures.append(f"{doc_id}: expected ingestion failure but it succeeded")
            elif expectation.get("error_contains") and \
                    expectation["error_contains"] not in failed.get(doc_id, ""):
                failures.append(
                    f"{doc_id}: failure message {failed.get(doc_id, '')[:60]!r} does not "
                    f"contain {expectation['error_contains']!r}"
                )
        else:
            if doc_id in failed:
                failures.append(f"{doc_id}: expected success but failed: {failed[doc_id][:60]}")

    report.add("post-ingest: expectations vs history", failures,
               f"collection={collection}, {len(succeeded)} succeeded, {len(failed)} failed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a generated corpus.")
    parser.add_argument("--out", default="test_corpus", help="corpus root")
    parser.add_argument("--seed", type=int, default=1337, help="expected seed")
    parser.add_argument("--mode", choices=("build", "post-ingest"), default="build")
    parser.add_argument("--collection", default="corpus100",
                        help="collection name for --mode post-ingest")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = _REPO_ROOT / out_root

    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        print(f"[FAIL] no manifest at {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = Report()

    print("Corpus verification")
    print(f"  root : {out_root}")
    print(f"  seed : {manifest.get('seed')}  docs: {manifest.get('document_count')}")
    print("=" * 74)

    qa_stats_path = out_root / "qa_stats.json"
    qa_stats = json.loads(qa_stats_path.read_text(encoding="utf-8")) if qa_stats_path.exists() else {}

    # 1-2: files and coverage
    meta = check_files(report, out_root, manifest)
    check_coverage(report, manifest, qa_stats)

    if args.mode == "post-ingest":
        check_post_ingest(report, out_root, args.collection)
    else:
        # 3-7: content-level checks
        extractor = Extractor(out_root / "_extract")
        texts = check_extraction(report, out_root, manifest, meta, extractor)
        check_anchors(report, out_root, texts, manifest)
        check_refusals(report, out_root, texts)
        check_structure(report, manifest, meta)
        check_answer_bank(report, out_root, manifest)

        import hashlib

        for doc_id, text in texts.items():
            report.extract_hashes[doc_id] = hashlib.sha256(
                normalize(text).encode("utf-8")
            ).hexdigest()[:16]

        # 8: determinism against the previous report
        report_path = out_root / "verify_report.json"
        previous = None
        if report_path.exists():
            try:
                previous = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                previous = None
        check_determinism(report, previous, report.extract_hashes, args.seed)

    payload = report.to_dict()
    payload["seed"] = manifest.get("seed")
    payload["mode"] = args.mode
    (out_root / "verify_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 74)
    print(f"checks passed: {payload['passed']}/{payload['total']}")
    if report.failed:
        for check in report.failed:
            print(f"  FAILED {check.name}: {len(check.failures)} problem(s)")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    print("ALL CHECKS PASSED")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
