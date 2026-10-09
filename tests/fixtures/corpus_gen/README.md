# Synthetic RAG test corpus

Generates **100 PDF documents** plus a **machine-checkable answer bank** (10–20
questions each, ~1 400 in total) for exercising the Modular RAG MCP Server.

The corpus is designed to break a naive RAG pipeline in specific, measurable
ways: scanned pages with no text layer, OCR'd text with recognition errors,
two-column papers, vertical writing, merged-cell and page-spanning tables,
watermarks, running headers/footers, adverts, bookmarks, and six version pairs.

## Quick start

```bash
pip install "reportlab>=4.0" "pymupdf>=1.24"     # or: pip install -e ".[corpus]"

# 1. Fast feedback: check the coverage matrix without rendering anything (~0.1s)
python tests/fixtures/corpus_gen/build_corpus.py --check-coverage

# 2. Generate the corpus (a few minutes) and verify it
python tests/fixtures/corpus_gen/build_corpus.py

# 3. Inspect what was produced
python tests/fixtures/corpus_gen/verify_corpus.py --out test_corpus
```

Output goes to **`test_corpus/`** at the repository root (see *Why not `data/`*).

```
test_corpus/
  pdf/<CAT>/<DOC>.pdf        100 PDFs
  qa/<DOC>.json              answer bank, one file per document
  qa_index.json              every question, flattened
  qa_stats.json              answer-bank roll-up
  manifest.json              per-document facts (the on-disk source of truth)
  expected_ingestion.json    what ingestion should do with each document
  coverage_report.md/.json   the coverage matrix
  verify_report.json         verification results
  _extract/                  cached markitdown extraction (verification only)
```

## Why the documents are synthetic

Real arXiv / PMC / CNKI / SEC EDGAR / WHO PDFs are **not** downloaded. They
cannot satisfy the requirements: they are not reproducible offline, their
licensing varies, they cannot be made to cover a specified flag matrix, and —
decisively — **no answer key can be authored for them automatically**. Instead
each document strictly imitates the layout conventions and register of its source
family (arXiv two-column paper, 10-K statement, statute, drug label, magazine
report), using entirely fictional companies, authors, document numbers and
figures. No real personal or corporate data is used.

## The key design property: content and answers share one source

A content module returns a `DocPlan` containing:

* `body` — the blocks that will be rendered;
* `facts` — `Fact` objects whose `anchor` is the exact substring written into
  that body, together with the question and answer it supports.

Because the question is authored in the same expression that writes its evidence
into the PDF, **an answer cannot drift away from the corpus**. `verify_corpus.py`
then re-checks every anchor against the text that `markitdown` — the same
extractor `PdfLoader` uses — actually recovers. Anchors are whitespace-normalised
before comparison, so line wrapping cannot break them.

## Coverage matrix

All thresholds are asserted by `verify_corpus.py` and by
`tests/unit/test_corpus_gen_coverage.py`.

| Dimension | Coverage (measured) |
|---|---|
| Categories | 25 academic / 20 technical / 15 financial / 15 legal / 10 medical / 15 news |
| Text extractability | 77 native · 15 scanned (image-only) · 8 OCR'd (invisible text layer) |
| Layout | 24 two-column · 3 three-column · 2 vertical · 71 single-column |
| Language (extractable subset) | 30 Chinese · 40 English · 15 mixed |
| Tables | simple 69 · merged 20 · page-spanning 39 · borderless 10 · grouped-header 30 |
| Figures | flow 54 · chart 70 · photo 23 · formula-as-image 9 · documents with no figure 21 |
| Length | 3 × 1 page · 10 × 2–3 pages · 75 × 8–14 pages · 6 × 40–60 pages · 1 × 220 pages |
| Structure | 87 with a table of contents · 25 with PDF bookmarks · 6 with broken heading hierarchy |
| Dirty data | 6 watermarked · 29 with ≥3 hard table classes · 10 with adverts · 100 with footnotes |
| Versions | 6 v1/v2 pairs (12 documents) differing only in content |

Totals for the shipped build: **100 documents, 1 446 pages, 1 425 questions
(10–17 per document), 429 MB**. All 44 coverage thresholds and all 8 verification
checks pass.

Question mix across the corpus: factual 793 · table 159 · figure 148 ·
cross-page 97 · cross-document 12 · refusal 216.

## Document classes and what they test

| Class | Documents | Why it matters |
|---|---|---|
| **native** | 77 | the happy path |
| **scanned** | 15 | no text layer at all. `DocumentChunker` raises `ValueError("... has no text content to split")`, so ingestion **fails** — this is the expected outcome and is recorded in `expected_ingestion.json`. Documents that genuinely need OCR. |
| **ocr** | 8 | a degraded page image plus an **invisible** (alpha 0) text layer carrying OCR-style recognition errors (~1–8.5 % per character). Text is extractable but imperfect, so anchors are matched by trigram overlap ≥ 0.70 rather than exactly. |
| **vertical** | 2 | characters top-to-bottom, columns right-to-left. PDF extraction reads left-to-right, so the recovered reading order does **not** match the drawing order (measured trigram overlap 0.25–0.55). Anchors are recorded but not asserted. |
| **version pairs** | 12 | same layout and length, different facts. Exercises deduplication and recency handling; the pair also supplies the cross-document questions. |

## Question categories

Every document carries 10–20 questions across these kinds. A document only
receives a kind it can support (a document with no table gets no table question;
that quota is reallocated to factual questions).

| Kind | What it exercises |
|---|---|
| `factual` | single-fact lookup, including numeric values |
| `table` | reading a value out of a table cell |
| `cross_page` | two anchors on different pages, answer needs both |
| `cross_doc` | comparison against the version-pair partner |
| `figure` | the figure caption and the diagram it describes |
| `refusal` | an entity that exists **nowhere** in the corpus; the system must decline |

Refusal questions are the reason `reject_terms` are checked corpus-wide: the
verifier proves the asked-about entity appears in no document, which makes "the
system should decline" a checkable claim rather than a hope.

## Command-line reference

### `build_corpus.py`

| Flag | Meaning |
|---|---|
| `--out PATH` | output root (default `test_corpus`) |
| `--seed N` | deterministic seed (default `1337`) |
| `--profile full\|smoke` | `full` = 100 documents; `smoke` = 8 documents covering every hard case, with long documents clamped to 12 pages |
| `--only ACAD-01,TECH-05` | build specific documents |
| `--category ACAD` | build one category |
| `--keep-sources` | keep the pre-degradation native PDFs under `_sources/` |
| `--resume` | skip documents already present |
| `--no-verify` | skip the verification gate |
| `--list` | print the plan without rendering |
| `--check-coverage` | build every plan in memory and check coverage; renders nothing |

### `verify_corpus.py`

| Flag | Meaning |
|---|---|
| `--out PATH` | corpus root (default `test_corpus`) |
| `--seed N` | expected seed, for the determinism check |
| `--mode build\|post-ingest` | audit the files, or compare ingestion results against `expected_ingestion.json` |
| `--collection NAME` | collection name for `--mode post-ingest` |
| `--json` | print the report as JSON |

Exit code `0` means every check passed.

Checks performed in `build` mode:

1. every declared PDF exists, opens, and matches its recorded page count and SHA256;
2. every coverage threshold, recomputed from the manifest;
3. extraction outcomes per document class (native > 0 chars, scanned ≤ 20 chars, OCR > 0);
4. **every question's anchor occurs in the extracted text**;
5. **every refusal question's reject terms occur nowhere in the corpus**;
6. watermarks, running footers, bookmarks and embedded images are really present;
7. answer-bank shape: 10–20 questions per document, ≥ 1 refusal each;
8. version pairs agree on page count, layout and language, and are not byte-identical;
9. determinism: extracted-text hashes match the previous run at the same seed.

## Ingesting the corpus

```bash
python scripts/ingest.py --path test_corpus/pdf --collection corpus100
python tests/fixtures/corpus_gen/verify_corpus.py --mode post-ingest --collection corpus100
python tests/fixtures/corpus_gen/build_golden_test_set.py --collection corpus100
python scripts/evaluate.py --test-set test_corpus/golden_test_set_corpus.json --collection corpus100
```

`build_golden_test_set.py` resolves each anchor to real chunk IDs **read back from
Chroma** rather than predicted. Chunk IDs are generated in two places with
different prefixes (`DocumentChunker` uses `doc_<hash>`, `VectorUpserter` uses
`sha256(source_path)[:8]`) and the Transform stage rewrites chunk text, so offline
prediction is unreliable. Unresolved anchors are reported — they are expected for
the scanned documents, whose text never reaches the index.

## Findings this corpus exposes

These are real behaviours of the existing pipeline, measured by the verification
run rather than assumed. They are left visible rather than engineered around,
because they are exactly what the corpus exists to surface.

| Finding | Evidence |
|---|---|
| **Scanned pages fail ingestion.** `DocumentChunker` raises `ValueError("... has no text content to split")` when a PDF has no text layer, so all 15 scanned documents report failure. The system has no OCR. | `expected_ingestion.json` marks them `expect: failure`; `verify_corpus.py --mode post-ingest` asserts it. |
| **A diagonal text watermark damages retrieval.** The watermark is real text; over a dense table the flow extractor interleaves its glyphs with the cell text, so the token never appears contiguously. | 2 of 6 watermarked documents (`FIN-01`, `FIN-02`) lose the watermark string in flow extraction; all 6 retain it as a text span. Reported in the verification detail line. |
| **Vertical writing is not order-recoverable.** Characters are drawn top-to-bottom in right-to-left columns while extraction reads left-to-right. | Measured trigram overlap 0.25–0.55 on `LEGAL-10`; these documents are recorded as `anchors_verifiable: false`. |
| **Long tables inside a half-width frame are unreadable.** Two-column papers therefore only carry narrow tables — the generator enforces this. | `spec.table_styles_for` refuses `multipage`/`multiheader` for `layout == two_column`. |
| **PyMuPDF activates a dead code path.** With PyMuPDF installed, `PdfLoader` extracts images and appends every `[IMAGE: …]` placeholder to the *end* of the document rather than beside the figure. | `image_refs` lands in the final chunk; the loader's own comment calls this "simplified". |
| **OCR errors are survivable but measurable.** At 1–8.5 % per-character error, mean anchor overlap is 0.98, with the weakest short anchor at 0.60. | `verify_corpus.py` reports the exact/fuzzy split and the mean. |
| **FIXED — embedding failures were swallowed silently.** `BatchProcessor` caught every batch exception and only counted it (never logging, never raising), so an exhausted API quota surfaced as `Chunk count (18) must match vector count (0)` with nothing in the log explaining why. Now each failed batch is logged at `ERROR` with the provider's own message, and a total failure raises an actionable `RuntimeError`. | Before: the opaque count mismatch alone. After: `Batch 1/2 failed (10 chunks, ids …): … Error code: 400 … Free quota exhausted` followed by `Encoding failed for all 2 batch(es) … First error: …`. |
| **OPEN — installing PyMuPDF masks the scanned-document failure.** `PdfLoader` appends one `[IMAGE: …]` placeholder per page, so an image-only PDF is no longer "empty text" and `DocumentChunker` does not raise. Scanned documents therefore ingest as a **single chunk of pure placeholders** rather than failing loudly. | Real ingestion traces: `MED-09.pdf: text_length=327 images=14 chunks=1`, preview `'\n[IMAGE: 3d1a88a0_1_1]\n\n[IMAGE: …]'` — the same shape for all 15 scanned documents. `expected_ingestion.json` still predicts `failure` for them, so `verify_corpus.py --mode post-ingest` will correctly flag the mismatch once ingestion completes. |

## Where the corpus is written, and the `data/` permission situation

The corpus is written to `test_corpus/` at the repository root, not
`data/test_corpus/`, because the generator normally runs under the DSH file
sandbox and only paths created at the workspace root accept confined writes.
Pass `--out` to override.

The underlying cause is now understood, and it is **not** a normal ACL problem:
the confined process runs at **low integrity**, and DSH labels the workspace root
accordingly, so newly created top-level paths inherit that label and stay
writable. The pre-existing `data/` and `logs/` trees have **no matching
integrity label**, so every write into them is refused with `WinError 5` — even
though the on-disk ACLs and ownership are perfectly correct and the same writes
succeed for an unconfined process.

Consequences:

* the RAG index (`data/db/chroma`, `data/db/bm25`, `data/images`,
  `data/db/ingestion_history.db`) **can** be written, but only by a run with
  wider file access — ingestion therefore needs one escalated run;
* `icacls /setintegritylevel` fixes the label only partially when invoked
  unescalated, because setting an integrity label requires `SeSecurityPrivilege`;
* reading those files (`sqlite3` read-only opens included) is likewise refused
  under confinement, so status queries need the same wider access.

## Environment notes

* **`reportlab` and `pymupdf` are required.** They were not installed in this
  checkout, which is why the pre-existing `tests/fixtures/generate_*.py` scripts
  could not run. `pip install -e ".[corpus]"` adds them.
* Installing **PyMuPDF activates `PdfLoader`'s image extraction**, which was
  previously dead code (`PYMUPDF_AVAILABLE` was False). Documents ingested after
  this change will have images stored under `data/images/` and `[IMAGE: …]`
  placeholders appended — the loader appends every placeholder to the *end* of
  the document, so `image_refs` lands in the final chunk rather than beside the
  figure caption. That is a real defect of the existing loader, deliberately left
  visible rather than worked around.
* **Never use `tempfile` in this package.** On this platform `os.mkdir(path,
  0o700)` produces a directory the creating user cannot write into, and
  `tempfile.mkdtemp` uses exactly that mode — this is also why `pip install`
  fails under the confined sandbox. Use `fsutil.ensure_dir` / `fsutil.scratch_dir`.

## Module map

| File | Responsibility |
|---|---|
| `spec.py` | the authoritative 100-document table and every coverage threshold |
| `facts.py` | `DocPlan` / `Fact` / `Block` model and the plan flattening helper |
| `content_common.py` | content engine: page units, content budget, prose pools, table builders, plan assembly |
| `content_{academic,tech,finance,legal,medical,news}.py` | per-category rendering profile |
| `pdf_writer.py` | reportlab rendering: column layouts, vertical writing, TOC, bookmarks, header/footer, watermark, five table classes |
| `art.py` | Pillow-drawn figures, charts, photos, screenshots, formula images |
| `fonts.py` | CJK font registration with a glyph-coverage self-check |
| `degrade.py` | rasterise + blur/skew/noise/JPEG/aged degradation, invisible OCR text layer |
| `qa.py` | answer-bank assembly and `expected_ingestion` derivation |
| `coverage.py` | coverage matrix, thresholds, plan-only pre-flight |
| `build_corpus.py` | CLI orchestrator |
| `verify_corpus.py` | acceptance gate |
| `build_golden_test_set.py` | anchor → chunk-ID resolution |
| `depcheck.py` | dependency guard and API probe (`--probe`) |
| `fsutil.py` | directory/atomic-write helpers that avoid the `0o700` defect |

## Determinism

The same `--seed` produces the same extracted text, the same page counts and the
same manifest. The seed influences prose, numeric values and degradation
parameters only — **never which document carries which property**, so the
coverage matrix holds for any seed (asserted by a unit test).

Two consecutive runs are compared by `verify_corpus.py`, which stores a hash of
each document's normalised extracted text and fails if any of them changes.

## Tests

```bash
pytest tests/unit/test_corpus_gen_spec.py \
       tests/unit/test_corpus_gen_coverage.py \
       tests/unit/test_corpus_gen_qa_shape.py -q
```

38 tests, ~2 s. They build all 100 plans in memory — no rendering, no API calls —
and assert the coverage matrix, answer-bank shape, anchor uniqueness, that every
anchor is a real substring of the text the plan will render, and that refusal
terms occur nowhere in the corpus.
