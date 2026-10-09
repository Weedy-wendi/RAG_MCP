"""Compute the corpus coverage matrix and check it against the required thresholds.

Metrics are derived from the **realised** :class:`~.facts.DocPlan` objects, not
from the spec's intent, so a content module that silently fails to emit a table
shows up as a failing threshold rather than passing on paper.

``figure.*`` metrics count *authored* figures.  For scanned documents the page
raster is itself an image, but that is not an authored figure, so scanned and
OCR'd documents are excluded from ``figure.none`` -- otherwise rasterisation
would fabricate coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.facts import DocPlan

CHART_ARTS = ("bar", "pie", "line")


@dataclass
class Metric:
    """One measured coverage number and whether it satisfies its threshold."""

    name: str
    value: int
    low: int
    high: Optional[int]
    description: str
    docs: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.value < self.low:
            return False
        if self.high is not None and self.value > self.high:
            return False
        return True

    def line(self) -> str:
        status = "OK  " if self.ok else "FAIL"
        bound = f"{self.low}" if self.high is None else f"{self.low}..{self.high}"
        return f"[{status}] {self.name:<26} {self.value:>5}  (need {bound})  {self.description}"


def _count(
    name: str,
    docs: Sequence[str],
    low: int,
    high: Optional[int],
    description: str,
) -> Metric:
    return Metric(name=name, value=len(docs), low=low, high=high,
                  description=description, docs=list(docs))


def compute_metrics(plans: Dict[str, DocPlan], specs: Dict[str, S.DocSpec]) -> Dict[str, Metric]:
    """Measure every threshold metric over the realised plans."""
    thresholds = S.THRESHOLDS
    metrics: Dict[str, Metric] = {}

    def thr(name: str) -> Tuple[int, Optional[int], str]:
        low, high, desc = thresholds[name]
        return low, high, desc

    all_ids = list(plans.keys())
    extractable = [d for d in all_ids if specs[d].extractable]

    # -- categories and totals -------------------------------------------
    for category in S.CATEGORIES:
        name = f"category.{category}"
        low, high, desc = thr(name)
        metrics[name] = _count(name, [d for d in all_ids if specs[d].category == category],
                               low, high, desc)
    low, high, desc = thr("total")
    metrics["total"] = _count("total", all_ids, low, high, desc)

    # -- extractability --------------------------------------------------
    for key, mode in (("extract.native", S.NATIVE), ("extract.scanned", S.SCANNED),
                      ("extract.ocr", S.OCR)):
        low, high, desc = thr(key)
        metrics[key] = _count(key, [d for d in all_ids if specs[d].extractability == mode],
                              low, high, desc)

    # -- dirty data ------------------------------------------------------
    low, high, desc = thr("dirty.watermark")
    metrics["dirty.watermark"] = _count("dirty.watermark", [d for d in all_ids if specs[d].watermark],
                                        low, high, desc)

    low, high, desc = thr("dirty.complex_tables")
    complex_docs = [d for d in all_ids if len(plans[d].table_styles()) >= 3]
    metrics["dirty.complex_tables"] = _count("dirty.complex_tables", complex_docs, low, high, desc)

    low, high, desc = thr("dirty.messy_headings")
    metrics["dirty.messy_headings"] = _count(
        "dirty.messy_headings", [d for d in all_ids if specs[d].messy_headings], low, high, desc)

    low, high, desc = thr("dirty.ads")
    metrics["dirty.ads"] = _count("dirty.ads", [d for d in all_ids if plans[d].ad_count() > 0],
                                  low, high, desc)

    low, high, desc = thr("dirty.footnotes")
    metrics["dirty.footnotes"] = _count(
        "dirty.footnotes", [d for d in all_ids if plans[d].footnote_count() > 0], low, high, desc)

    # -- layout ----------------------------------------------------------
    for key, layout in (("layout.two_column", S.TWO_COL), ("layout.three_column", S.THREE_COL),
                        ("layout.vertical", S.VERTICAL)):
        low, high, desc = thr(key)
        metrics[key] = _count(key, [d for d in all_ids if specs[d].layout == layout], low, high, desc)

    # -- language (extractable documents only) ---------------------------
    for key, lang in (("lang.zh", S.LANG_ZH), ("lang.en", S.LANG_EN), ("lang.mixed", S.LANG_MIXED)):
        low, high, desc = thr(key)
        metrics[key] = _count(key, [d for d in extractable if specs[d].language == lang],
                              low, high, desc)

    # -- tables ----------------------------------------------------------
    for key, style in (("table.simple", S.TABLE_SIMPLE), ("table.merged", S.TABLE_MERGED),
                       ("table.multipage", S.TABLE_MULTIPAGE),
                       ("table.borderless", S.TABLE_BORDERLESS),
                       ("table.multiheader", S.TABLE_MULTIHEADER)):
        low, high, desc = thr(key)
        docs = [d for d in all_ids if plans[d].table_styles().get(style, 0) > 0]
        metrics[key] = _count(key, docs, low, high, desc)

    # -- figures ---------------------------------------------------------
    def docs_with_art(predicate) -> List[str]:
        out: List[str] = []
        for doc_id in all_ids:
            arts = {b.art for b in plans[doc_id].body if getattr(b, "art", None)}
            if any(predicate(a) for a in arts):
                out.append(doc_id)
        return out

    low, high, desc = thr("figure.flow")
    metrics["figure.flow"] = _count("figure.flow", docs_with_art(lambda a: a == "flow"), low, high, desc)

    low, high, desc = thr("figure.chart")
    metrics["figure.chart"] = _count("figure.chart", docs_with_art(lambda a: a in CHART_ARTS),
                                     low, high, desc)

    low, high, desc = thr("figure.photo")
    metrics["figure.photo"] = _count("figure.photo", docs_with_art(lambda a: a == "photo"),
                                     low, high, desc)

    low, high, desc = thr("figure.formula_image")
    metrics["figure.formula_image"] = _count(
        "figure.formula_image", docs_with_art(lambda a: a == "formula"), low, high, desc)

    low, high, desc = thr("figure.none")
    metrics["figure.none"] = _count(
        "figure.none",
        [d for d in extractable if plans[d].figure_count() == 0],
        low, high, desc,
    )

    # -- length ----------------------------------------------------------
    def page_docs(predicate) -> List[str]:
        return [d for d in all_ids if predicate(specs[d].pages)]

    for key, predicate in (
        ("length.one_page", lambda p: p == 1),
        ("length.short", lambda p: 2 <= p <= 3),
        ("length.mid", lambda p: 8 <= p <= 14),
        ("length.fifty", lambda p: 40 <= p <= 60),
        ("length.huge", lambda p: p >= 200),
    ):
        low, high, desc = thr(key)
        metrics[key] = _count(key, page_docs(predicate), low, high, desc)

    # -- structure -------------------------------------------------------
    low, high, desc = thr("structure.toc")
    metrics["structure.toc"] = _count(
        "structure.toc", [d for d in all_ids if specs[d].has_toc], low, high, desc)

    low, high, desc = thr("structure.bookmarks")
    metrics["structure.bookmarks"] = _count(
        "structure.bookmarks", [d for d in all_ids if specs[d].bookmarks], low, high, desc)

    low, high, desc = thr("structure.version_pairs")
    pairs = [d for d in all_ids if specs[d].version_label == "v1"]
    metrics["structure.version_pairs"] = _count(
        "structure.version_pairs", pairs, low, high, desc)

    low, high, desc = thr("structure.references")
    metrics["structure.references"] = _count(
        "structure.references", [d for d in all_ids if plans[d].reference_count() > 0],
        low, high, desc)

    return metrics


def add_qa_metrics(metrics: Dict[str, Metric], stats: Dict[str, Any]) -> None:
    """Fold answer-bank metrics into the coverage matrix."""
    thresholds = S.THRESHOLDS
    for key, value in (
        ("qa.per_doc_min", stats.get("per_doc_min", 0)),
        ("qa.per_doc_max", stats.get("per_doc_max", 0)),
        ("qa.refusal_per_doc", stats.get("refusal_per_doc_min", 0)),
        ("qa.total", stats.get("question_count", 0)),
    ):
        low, high, desc = thresholds[key]
        metrics[key] = Metric(name=key, value=int(value), low=low, high=high, description=desc)


def failing(metrics: Dict[str, Metric]) -> List[Metric]:
    return [m for m in metrics.values() if not m.ok]


def render_report(
    metrics: Dict[str, Metric],
    manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the human-readable coverage report."""
    lines: List[str] = []
    lines.append("# 测试语料覆盖矩阵（Coverage Matrix）")
    lines.append("")
    if manifest:
        lines.append(f"- 语料版本：{manifest.get('version', '')}")
        lines.append(f"- 随机种子：{manifest.get('seed', '')}")
        lines.append(f"- 文档总数：{manifest.get('document_count', '')}")
        lines.append(f"- 生成时间（不含时区）：{manifest.get('generated_at', '')}")
        lines.append("")

    ok = [m for m in metrics.values() if m.ok]
    bad = failing(metrics)
    lines.append(f"**阈值检查：{len(ok)}/{len(metrics)} 项通过**")
    lines.append("")
    lines.append("```")
    for metric in metrics.values():
        lines.append(metric.line())
    lines.append("```")
    lines.append("")

    if bad:
        lines.append("## 未达标项")
        lines.append("")
        for metric in bad:
            lines.append(f"- `{metric.name}` = {metric.value}，要求 {metric.low}"
                         f"{'' if metric.high is None else '..' + str(metric.high)}（{metric.description}）")
        lines.append("")
    else:
        lines.append("全部覆盖阈值通过。")
        lines.append("")

    return "\n".join(lines)


def to_json(metrics: Dict[str, Metric]) -> Dict[str, Any]:
    """Machine-readable coverage matrix."""
    return {
        "passed": len([m for m in metrics.values() if m.ok]),
        "total": len(metrics),
        "all_passed": all(m.ok for m in metrics.values()),
        "metrics": {
            name: {
                "value": m.value,
                "min": m.low,
                "max": m.high,
                "ok": m.ok,
                "description": m.description,
                "doc_count": len(m.docs),
            }
            for name, m in metrics.items()
        },
    }


# ---------------------------------------------------------------------------
# Fast in-memory pre-flight
# ---------------------------------------------------------------------------


@dataclass
class PlanCoverage:
    """Coverage computed from plans alone -- no rendering, no API calls."""

    metrics: Dict[str, Metric]
    plans: Dict[str, DocPlan]
    specs: Dict[str, S.DocSpec]
    stats: Dict[str, Any]

    @property
    def failed(self) -> List[Metric]:
        return failing(self.metrics)

    @property
    def all_passed(self) -> bool:
        return not self.failed


def plan_coverage(seed: int = 1337) -> PlanCoverage:
    """Build every document plan in memory and evaluate the coverage matrix.

    This is the fast feedback loop: it exercises all six content modules and the
    whole answer-bank pipeline in a couple of seconds, so a spec or content
    regression is caught long before 100 PDFs are rendered.
    """
    from tests.fixtures.corpus_gen import qa as qa_module
    from tests.fixtures.corpus_gen.build_corpus import build_plan

    plans: Dict[str, DocPlan] = {}
    specs: Dict[str, S.DocSpec] = {}
    documents: List[Any] = []

    for spec in S.SPECS:
        plan = build_plan(spec, seed)
        plans[spec.doc_id] = plan
        specs[spec.doc_id] = spec
        documents.append(qa_module.build_doc_qa(spec, plan, spec.relpath()))

    stats = qa_module.build_stats(documents)
    metrics = compute_metrics(plans, specs)
    add_qa_metrics(metrics, stats)
    return PlanCoverage(metrics=metrics, plans=plans, specs=specs, stats=stats)
