#!/usr/bin/env python
"""Resolve the answer bank against the live index and emit a golden test set.

The corpus answer bank anchors are plain text.  ``scripts/evaluate.py`` consumes
``expected_chunk_ids`` instead, so this script maps each anchor to the chunk IDs
that actually contain it.

Why read the index rather than predict IDs
------------------------------------------
Chunk IDs are generated in two places with *different* prefixes::

    DocumentChunker._generate_chunk_id  -> {doc_id}_{index:04d}_{content_hash}
    VectorUpserter._generate_chunk_id   -> {sha256(source_path)[:8]}_{index:04d}_{content_hash}

and the Transform stage rewrites chunk text before storage.  Predicting IDs
offline is therefore fragile; reading them back from Chroma is exact.

Usage::

    python tests/fixtures/corpus_gen/build_golden_test_set.py --collection corpus100
    python tests/fixtures/corpus_gen/build_golden_test_set.py --out data/test_corpus \\
        --collection corpus100 --top-k 10

Outputs ``golden_test_set_corpus.json`` (schema-compatible with
``EvalRunner``/``scripts/evaluate.py``) and ``qa_resolved.json`` (per-question
resolution status, including anchors that could not be located).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.fixtures.corpus_gen import spec as S  # noqa: E402
from tests.fixtures.corpus_gen.qa import KIND_ORDER  # noqa: E402

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub("", text or "")


def load_chunks(collection_name: str) -> Tuple[Dict[str, List[Tuple[str, str]]], str]:
    """Return ``{filename -> [(chunk_id, normalized_text), ...]}`` and the path used."""
    import chromadb

    settings_path = _REPO_ROOT / "config" / "settings.yaml"
    persist = _REPO_ROOT / "data" / "db" / "chroma"
    if settings_path.exists():
        import yaml

        data = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        configured = (data.get("vector_store") or {}).get("persist_directory")
        if configured:
            candidate = Path(configured)
            persist = candidate if candidate.is_absolute() else _REPO_ROOT / candidate

    client = chromadb.PersistentClient(path=str(persist))
    collection = client.get_collection(collection_name)
    payload = collection.get(include=["documents", "metadatas"])

    by_filename: Dict[str, List[Tuple[str, str]]] = {}
    for chunk_id, document, metadata in zip(
        payload.get("ids") or [], payload.get("documents") or [], payload.get("metadatas") or []
    ):
        source = str((metadata or {}).get("source_path") or (metadata or {}).get("source") or "")
        if not source:
            continue
        name = Path(source.replace("\\", "/")).name
        by_filename.setdefault(name, []).append((str(chunk_id), normalize(document or "")))
    return by_filename, str(persist)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve the answer bank to chunk IDs.")
    parser.add_argument("--out", default="test_corpus", help="corpus root")
    parser.add_argument("--collection", default="corpus100", help="Chroma collection to read")
    parser.add_argument("--top-k", type=int, default=10, help="documented top-K for the run")
    parser.add_argument("--max-chunks-per-question", type=int, default=3,
                        help="cap chunk IDs recorded per question")
    parser.add_argument("--resolved-only", action="store_true",
                        help="also write golden_test_set_resolved.json containing only the "
                             "questions whose anchor resolved to a real chunk. Those are the "
                             "only ones the IR metrics (hit_rate / mrr) can score; unresolved "
                             "and refusal questions would otherwise be counted as misses.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = _REPO_ROOT / out_root

    index_path = out_root / "qa_index.json"
    if not index_path.exists():
        print(f"[FAIL] no answer bank at {index_path}", file=sys.stderr)
        return 2

    try:
        by_filename, persist = load_chunks(args.collection)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] cannot read collection {args.collection!r}: {exc}", file=sys.stderr)
        print("       Run ingestion first, e.g.\n"
              "       python scripts/ingest.py --path data/test_corpus/pdf --collection corpus100",
              file=sys.stderr)
        return 2

    bank = json.loads(index_path.read_text(encoding="utf-8"))
    test_cases: List[Dict[str, Any]] = []
    resolved: List[Dict[str, Any]] = []
    unresolved = 0
    refusal_count = 0

    for question in bank["questions"]:
        sources = question.get("expected_sources") or []
        reference = question["answer"]
        entry: Dict[str, Any] = {
            "qid": question["qid"],
            "category": question["category"],
            "expected_sources": sources,
            "expected_chunk_ids": [],
            "located": 0,
        }

        if question.get("expect_no_answer"):
            refusal_count += 1
            entry["resolution"] = "refusal"
            resolved.append(entry)
            test_cases.append({
                "query": question["question"],
                "expected_chunk_ids": [],
                "expected_sources": [],
                "reference_answer": reference,
                "category": question["category"],
                "expect_no_answer": True,
            })
            continue

        anchor = normalize(question["evidence"].get("anchor_text") or "")
        found: List[str] = []
        for source in sources:
            name = Path(source.replace("\\", "/")).name
            for chunk_id, text in by_filename.get(name, []):
                if anchor and anchor in text:
                    found.append(chunk_id)
                    if len(found) >= args.max_chunks_per_question:
                        break
            if len(found) >= args.max_chunks_per_question:
                break

        entry["expected_chunk_ids"] = found
        entry["located"] = len(found)
        if found:
            entry["resolution"] = "resolved"
        else:
            entry["resolution"] = "unresolved"
            unresolved += 1
        resolved.append(entry)

        test_cases.append({
            "query": question["question"],
            "expected_chunk_ids": found,
            "expected_sources": sources,
            "reference_answer": reference,
            "category": question["category"],
            "expect_no_answer": False,
        })

    # Order by the answer-bank kind order so reports read consistently.
    test_cases.sort(key=lambda c: KIND_ORDER.index(c["category"])
                    if c["category"] in KIND_ORDER else len(KIND_ORDER))

    golden = {
        "description": (
            "Golden test set derived from the 100-document corpus answer bank. "
            "expected_chunk_ids were resolved against the live Chroma index after "
            "ingestion; refusal cases intentionally carry no expected chunks."
        ),
        "version": "1.0",
        "collection": args.collection,
        "top_k": args.top_k,
        "resolved_from": persist,
        "test_cases": test_cases,
    }
    (out_root / "golden_test_set_corpus.json").write_text(
        json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_root / "qa_resolved.json").write_text(
        json.dumps({
            "collection": args.collection,
            "documents_in_index": len(by_filename),
            "questions": len(resolved),
            "unresolved": unresolved,
            "refusals": refusal_count,
            "detail": resolved,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"index            : {persist} :: {args.collection}")
    print(f"documents in index: {len(by_filename)}")
    print(f"questions         : {len(resolved)} ({refusal_count} refusal)")
    print(f"unresolved anchors: {unresolved}")

    # Diagnostic: which document classes the unresolved questions come from.  A
    # high count for a class is a property of the corpus (no text layer, unstable
    # reading order), not a resolver bug.
    manifest_path = out_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        klass = {d["doc_id"]: d["extractability"] for d in manifest.get("documents", [])}
        by_class: Dict[str, int] = {}
        by_doc: Dict[str, int] = {}
        for item in resolved:
            if item["resolution"] != "unresolved":
                continue
            doc = str(item["qid"]).rsplit("-Q", 1)[0]
            key = klass.get(doc, "unknown")
            by_class[key] = by_class.get(key, 0) + 1
            by_doc[doc] = by_doc.get(doc, 0) + 1
        print("unresolved by class:", dict(sorted(by_class.items())))
        top = sorted(by_doc.items(), key=lambda kv: -kv[1])[:8]
        print("unresolved by doc  :", ", ".join(f"{d}={n}" for d, n in top))

    scorable = [c for c in test_cases if c["expected_chunk_ids"]]
    print(f"scorable by IR     : {len(scorable)} / {len(test_cases)}")
    if args.resolved_only:
        subset = dict(golden)
        subset["description"] = (
            "Scorable subset of the corpus golden test set: only questions whose anchor "
            "resolved to at least one real chunk. Unresolved anchors (documents with no "
            "text layer, unstable reading order or OCR corruption) and refusal questions "
            "are excluded, because hit_rate/mrr would score them as automatic misses."
        )
        subset["test_cases"] = scorable
        subset["full_test_set_size"] = len(test_cases)
        (out_root / "golden_test_set_resolved.json").write_text(
            json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote             : {out_root / 'golden_test_set_resolved.json'}")

    print(f"wrote             : {out_root / 'golden_test_set_corpus.json'}")
    if unresolved:
        print("  note: unresolved anchors are expected for documents whose text the "
              "index cannot contain or whose reading order is not recoverable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
