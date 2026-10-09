#!/usr/bin/env python
"""Build the 100-document RAG test corpus and its standard answer bank.

Usage::

    python tests/fixtures/corpus_gen/build_corpus.py                 # full 100 documents
    python tests/fixtures/corpus_gen/build_corpus.py --profile smoke # 8 documents, fast
    python tests/fixtures/corpus_gen/build_corpus.py --category ACAD --only ACAD-01
    python tests/fixtures/corpus_gen/build_corpus.py --out test_corpus --seed 1337
    python tests/fixtures/corpus_gen/build_corpus.py --list          # show the plan only
    python tests/fixtures/corpus_gen/build_corpus.py --check-coverage  # plan-only check

Outputs (default ``test_corpus``)::

    manifest.json            per-document facts + the flags that were realised
    build_log.json           per-document timings, scan parameters and warnings
    expected_ingestion.json  what ingestion should do with each document
    coverage_report.md/.json coverage matrix against the required thresholds
    qa/<DOC>.json            standard answer bank, one file per document
    qa_index.json            every question, flattened
    qa_stats.json            answer-bank roll-up
    pdf/<CAT>/<DOC>.pdf      the corpus itself
    _sources/<DOC>.pdf       pre-degradation native PDFs (only with --keep-sources)

The build is deterministic: the same ``--seed`` produces the same extracted text,
the same page counts and the same manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.corpus_gen import coverage, spec as S  # noqa: E402
from tests.fixtures.corpus_gen import content_academic, content_finance, content_legal  # noqa: E402
from tests.fixtures.corpus_gen import content_medical, content_news, content_tech  # noqa: E402
from tests.fixtures.corpus_gen import degrade, depcheck, qa  # noqa: E402
from tests.fixtures.corpus_gen.facts import DocPlan  # noqa: E402
from tests.fixtures.corpus_gen.content_common import FILL_ATTEMPTS  # noqa: E402
from tests.fixtures.corpus_gen.fsutil import (  # noqa: E402
    atomic_write_text,
    ensure_dir,
    sha256_file,
)
from tests.fixtures.corpus_gen.pdf_writer import RenderStats, render_pdf  # noqa: E402

VERSION = "1.0.0"
DEFAULT_SEED = 1337

#: Default output root.
#:
#: Deliberately at the repository root rather than ``data/test_corpus``: under the
#: DSH file sandbox only paths directly at the workspace root are writable, while
#: ``data/**`` is refused with WinError 5.  ``data/`` is still the right home for
#: the RAG indexes themselves; this is only about where generated PDFs are
#: allowed to be written.
DEFAULT_OUT = "test_corpus"

#: Small, fast subset that still touches every hard case (scanned, OCR,
#: watermark, complex tables, vertical, three-column, huge-page-count clamp).
SMOKE_DOCS: Tuple[str, ...] = (
    "ACAD-01", "ACAD-21", "ACAD-24",
    "TECH-05", "FIN-03", "LEGAL-10", "MED-02", "NEWS-01",
)

#: Longest page count allowed in the smoke profile, so iteration stays fast.
SMOKE_MAX_PAGES = 12

CONTENT_MODULES = {
    S.ACADEMIC: content_academic,
    S.TECH: content_tech,
    S.FINANCE: content_finance,
    S.LEGAL: content_legal,
    S.MEDICAL: content_medical,
    S.NEWS: content_news,
}


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 100-document RAG test corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"output root (default: {DEFAULT_OUT})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"deterministic seed (default: {DEFAULT_SEED})")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full",
                        help="full = 100 documents, smoke = 8 documents")
    parser.add_argument("--only", default="", help="comma-separated doc ids to build")
    parser.add_argument("--category", default="", help="limit to one category code")
    parser.add_argument("--keep-sources", action="store_true",
                        help="keep the pre-degradation native PDFs under _sources/")
    parser.add_argument("--resume", action="store_true",
                        help="skip documents already present with the right page count")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the verification gate after building")
    parser.add_argument("--list", action="store_true",
                        help="print the document plan and exit without rendering")
    parser.add_argument("--check-coverage", action="store_true",
                        help="build every plan in memory and check coverage; render nothing")
    return parser.parse_args(argv)


def select_specs(args: argparse.Namespace) -> List[S.DocSpec]:
    """Resolve the requested subset of documents."""
    specs = list(S.SPECS)

    if args.profile == "smoke":
        specs = [s for s in specs if s.doc_id in SMOKE_DOCS]
        specs = [
            dataclasses.replace(s, pages=min(s.pages, SMOKE_MAX_PAGES))
            if s.pages > SMOKE_MAX_PAGES else s
            for s in specs
        ]

    if args.category:
        wanted = {args.category.strip().upper()}
        specs = [s for s in specs if s.category in wanted]

    if args.only:
        wanted = {token.strip() for token in args.only.split(",") if token.strip()}
        known = {s.doc_id for s in S.SPECS}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"unknown document id(s): {', '.join(sorted(unknown))}")
        specs = [s for s in specs if s.doc_id in wanted]

    return specs


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_plan(spec: S.DocSpec, seed: int, fill: float | None = None) -> DocPlan:
    """Dispatch to the content module for this document's category."""
    module = CONTENT_MODULES[spec.category]
    return module.build(spec, seed, fill=fill)


def build_one(
    spec: S.DocSpec,
    seed: int,
    out_root: Path,
    sources_dir: Path,
    resume: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], qa.QaDocument, DocPlan]:
    """Render one document, degrade it if required, and collect its metadata.

    If a unit overflows its page the rendered page count exceeds the
    specification.  Because every unit ends with a page break, shrinking the
    per-unit content budget can only reduce the page count, so the render is
    retried with progressively smaller budgets until the count matches.
    """
    started = time.monotonic()
    warnings: List[str] = []

    native_path = sources_dir / f"{spec.doc_id}.pdf"
    final_path = out_root / spec.relpath()
    ensure_dir(final_path.parent)

    plan: DocPlan | None = None
    native_stats: RenderStats | None = None
    for attempt, fill in enumerate(FILL_ATTEMPTS, start=1):
        plan = build_plan(spec, seed, fill=None if attempt == 1 else fill)
        native_stats = render_pdf(plan, spec, native_path, seed)
        if native_stats.pages <= spec.pages:
            if attempt > 1:
                warnings.append(
                    f"page count fitted on attempt {attempt} with fill={fill}"
                )
            break
        warnings.append(
            f"attempt {attempt} (fill={fill}): rendered {native_stats.pages} pages, "
            f"expected {spec.pages}"
        )

    assert plan is not None and native_stats is not None

    scan_info: Dict[str, Any] = {}
    if spec.is_scanned:
        result = degrade.make_scanned(native_path, final_path, spec)
        scan_info = result.to_dict()
        actual_pages = result.pages
    elif spec.is_ocr:
        result = degrade.make_ocr(native_path, final_path, spec)
        scan_info = result.to_dict()
        actual_pages = result.pages
    else:
        shutil.copyfile(native_path, final_path)
        actual_pages = native_stats.pages

    if actual_pages != spec.pages:
        warnings.append(
            f"final page count mismatch: expected {spec.pages}, got {actual_pages}"
        )

    doc_qa = qa.build_doc_qa(spec, plan, spec.relpath())
    realised_tables = plan.table_styles()

    entry: Dict[str, Any] = {
        "doc_id": spec.doc_id,
        "category": spec.category,
        "category_name": S.CATEGORY_NAMES[spec.category],
        "title": spec.title,
        "title_alt": spec.title_alt,
        "source_style": spec.source_style,
        "language": spec.language,
        "layout": spec.layout,
        "extractability": spec.extractability,
        "version_label": spec.version_label,
        "version_pair": spec.version_pair,
        "file": spec.filename(),
        "relpath": spec.relpath(),
        "pages": spec.pages,
        "pages_actual": actual_pages,
        "watermark": spec.watermark,
        "messy_headings": spec.messy_headings,
        "bookmarks_requested": spec.bookmarks,
        "comments": "",
        # realised content facts
        "tables": native_stats.tables,
        "table_styles": realised_tables,
        "figures": native_stats.figures,
        "figure_kinds": sorted({b.art for b in plan.body if hasattr(b, "art")}),
        "bookmarks": native_stats.bookmarks,
        "footnotes": plan.footnote_count(),
        "references": plan.reference_count(),
        "formulas_text": plan.formula_counts()[0],
        "formulas_image": plan.formula_counts()[1],
        "ads": plan.ad_count(),
        "headings": plan.heading_count(),
        "anchors": len(plan.facts),
        "watermark_text": native_stats.watermark_text,
        "header_text": native_stats.header_text,
        "footer_pattern": native_stats.footer_pattern,
        "vertical": native_stats.vertical,
        "anchors_verifiable": spec.anchors_verifiable,
        "questions": doc_qa.question_count,
        "kind_counts": doc_qa.payload["kind_counts"],
        "expected_ingestion": doc_qa.payload["expected_ingestion"],
        "sha256": sha256_file(final_path),
        "bytes": final_path.stat().st_size,
        "scan": scan_info,
        "tags": sorted(spec.tags_set()),
    }

    log: Dict[str, Any] = {
        "doc_id": spec.doc_id,
        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
        "native_pages": native_stats.pages,
        "native_bytes": native_stats.bytes_written,
        "warnings": warnings,
    }
    return entry, log, doc_qa, plan


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    code = depcheck.check_requirements()
    if code != 0:
        return code

    specs = select_specs(args)

    if args.check_coverage:
        started = time.monotonic()
        checked = coverage.plan_coverage(args.seed)
        elapsed = time.monotonic() - started
        print(f"plan-only coverage check ({len(checked.plans)} documents, {elapsed:.1f}s)")
        print("=" * 74)
        for metric in checked.metrics.values():
            print(metric.line())
        print("=" * 74)
        print(f"questions: {checked.stats['question_count']}  "
              f"per doc: {checked.stats['per_doc_min']}..{checked.stats['per_doc_max']}  "
              f"by kind: {checked.stats['per_kind']}")
        for metric in checked.failed:
            print(f"  FAIL {metric.name} = {metric.value} (need {metric.low}"
                  f"{'' if metric.high is None else '..' + str(metric.high)})")
        return 0 if checked.all_passed else 1

    if args.list:
        print(f"{len(specs)} document(s) selected")
        print(f"{'doc_id':<10} {'cat':<6} {'lang':<6} {'layout':<13} "
              f"{'extract':<9} {'pages':>5}  title")
        for spec in specs:
            print(f"{spec.doc_id:<10} {spec.category:<6} {spec.language:<6} "
                  f"{spec.layout:<13} {spec.extractability:<9} {spec.pages:>5}  {spec.title[:44]}")
        return 0

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = _REPO_ROOT / out_root
    sources_dir = ensure_dir(out_root / "_sources")
    ensure_dir(out_root / "pdf")
    ensure_dir(out_root / "qa")

    print(f"Modular RAG test corpus — building {len(specs)} document(s)")
    print(f"  output : {out_root}")
    print(f"  seed   : {args.seed}")
    print(f"  profile: {args.profile}")
    print("=" * 74)

    entries: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    plans: Dict[str, DocPlan] = {}
    documents: List[qa.QaDocument] = []
    spec_by_id: Dict[str, S.DocSpec] = {}
    failures = 0
    started = time.monotonic()

    for index, spec in enumerate(specs, start=1):
        final_path = out_root / spec.relpath()
        prefix = f"[{index:>3}/{len(specs)}] {spec.doc_id:<10}"

        if args.resume and final_path.exists():
            print(f"{prefix} skipped (exists)")
            spec_by_id[spec.doc_id] = spec
            continue

        try:
            entry, log, doc_qa, plan = build_one(
                spec, args.seed, out_root, sources_dir, args.resume
            )
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop the build
            failures += 1
            print(f"{prefix} FAILED: {type(exc).__name__}: {exc}")
            logs.append({"doc_id": spec.doc_id, "error": f"{type(exc).__name__}: {exc}"})
            continue

        entries.append(entry)
        logs.append(log)
        documents.append(doc_qa)
        plans[spec.doc_id] = plan
        spec_by_id[spec.doc_id] = spec
        write_json(out_root / "qa" / f"{spec.doc_id}.json", doc_qa.payload)

        warn = " !" if log["warnings"] else ""
        print(
            f"{prefix} {entry['pages_actual']:>3}p {entry['tables']:>2}t "
            f"{entry['figures']:>2}f {entry['questions']:>2}q "
            f"{entry['scan'].get('mode', 'native'):<8}{warn}"
        )

    if not entries:
        print("nothing was built")
        return 1

    # ---- aggregate artefacts -------------------------------------------
    stats = qa.build_stats(documents)
    index_payload = qa.build_index(documents, args.seed, VERSION)
    metrics = coverage.compute_metrics(plans, spec_by_id)
    coverage.add_qa_metrics(metrics, stats)

    manifest = {
        "version": VERSION,
        "seed": args.seed,
        "profile": args.profile,
        "document_count": len(entries),
        "page_total": sum(e["pages_actual"] for e in entries),
        "question_count": stats["question_count"],
        "documents": entries,
    }
    write_json(out_root / "manifest.json", manifest)
    write_json(out_root / "build_log.json", {
        "seed": args.seed,
        "profile": args.profile,
        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
        "documents": logs,
    })
    write_json(out_root / "qa_index.json", index_payload)
    write_json(out_root / "qa_stats.json", stats)
    write_json(out_root / "expected_ingestion.json", {
        "description": "Per-document expectation for scripts/ingest.py.",
        "collection_suggestion": "corpus100",
        "documents": {
            e["doc_id"]: e["expected_ingestion"] for e in entries
        },
    })
    write_json(out_root / "coverage_report.json", coverage.to_json(metrics))
    atomic_write_text(out_root / "coverage_report.md",
                      coverage.render_report(metrics, manifest) + "\n")

    # ---- summary --------------------------------------------------------
    print("=" * 74)
    print(f"documents : {len(entries)}   pages: {manifest['page_total']}   "
          f"questions: {stats['question_count']}")
    print(f"per-doc questions: {stats['per_doc_min']}..{stats['per_doc_max']}   "
          f"refusals/doc min: {stats['refusal_per_doc_min']}")
    print(f"by kind   : {stats['per_kind']}")
    bad = coverage.failing(metrics)
    print(f"coverage  : {len(metrics) - len(bad)}/{len(metrics)} thresholds pass")
    for metric in bad:
        print(f"  FAIL {metric.name} = {metric.value} (need {metric.low}"
              f"{'' if metric.high is None else '..' + str(metric.high)})")
    if failures:
        print(f"build failures: {failures}")

    if not args.keep_sources:
        shutil.rmtree(sources_dir, ignore_errors=True)

    if not args.no_verify:
        print("-" * 74)
        from tests.fixtures.corpus_gen import verify_corpus

        verify_code = verify_corpus.main(["--out", str(out_root), "--seed", str(args.seed)])
        if verify_code != 0:
            return verify_code

    if failures or bad:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
