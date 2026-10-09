"""Shared content engine for all six document categories.

Design
------
A document is produced as a sequence of **page units**.  Each unit is a chunk of
content sized to fit inside one page, and the engine emits exactly ``spec.pages``
units separated by page breaks.  Because every unit ends with a page break and no
unit may overflow, the rendered page count equals the target *by construction* --
no render-measure-adjust loop is needed.

Each unit returns two things:

* :class:`Block` objects, which ``pdf_writer`` renders;
* :class:`Anchor` objects -- exact substrings that were just written into the
  document, together with the question and answer they support.

``assemble_plan`` then converts anchors into :class:`~.facts.Fact` objects,
choosing how many of each question kind to emit so that every document satisfies
the required mix (factual / table / cross-page / cross-document / figure / refusal).

Because a question is authored in the same expression that writes its evidence
into the PDF, the answer bank cannot drift from the corpus.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tests.fixtures.corpus_gen import spec as S
from tests.fixtures.corpus_gen.art import ArtSpec
from tests.fixtures.corpus_gen.facts import (
    CROSS_DOC,
    CROSS_PAGE,
    FACTUAL,
    FIGURE,
    TABLE,
    Block,
    Bullets,
    CodeBlock,
    DocPlan,
    Fact,
    FigureBlock,
    FootnoteBlock,
    FormulaBlock,
    Heading,
    PageBreak,
    Para,
    ReferenceBlock,
    RefusalQuestion,
    Rule,
    TableBlock,
)
from tests.fixtures.corpus_gen.facts import AdBlock

# ---------------------------------------------------------------------------
# Capacity model: characters that comfortably fit on one page per layout.
# Units are deliberately filled to ~72% of capacity so that a unit can never
# spill onto a second page (which would break the exact page count).
# ---------------------------------------------------------------------------

CHARS_PER_PAGE: Dict[str, int] = {
    S.SINGLE: 2600,
    S.TWO_COL: 3800,
    S.THREE_COL: 4100,
    S.VERTICAL: 560,
}
FILL = 0.72

#: Fill factors tried in order when a unit overflows its page.  Shrinking the
#: per-unit budget can only reduce the rendered page count (each unit always ends
#: with a page break), so this loop converges.
FILL_ATTEMPTS: Tuple[float, ...] = (FILL, 0.62, 0.52, 0.42, 0.34)


def unit_capacity(spec: S.DocSpec, fill: Optional[float] = None) -> int:
    """Target character budget for one unit of this document."""
    factor = FILL if fill is None else fill
    return max(400, int(CHARS_PER_PAGE[spec.layout] * factor))


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------


@dataclass
class Anchor:
    """An exact substring written into the document, plus the Q/A it supports."""

    text: str
    question: str
    answer: str
    section: str
    page: int
    kind: str = "fact"  # fact | table | figure
    answer_type: str = "text"
    figure_id: str = ""
    difficulty: str = "medium"

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("anchor text must be non-empty")
        if not self.question.strip():
            raise ValueError("anchor question must be non-empty")


# ---------------------------------------------------------------------------
# DocBuilder
# ---------------------------------------------------------------------------


class DocBuilder:
    """Accumulates blocks and anchors for one document."""

    def __init__(self, spec: S.DocSpec, rng: random.Random) -> None:
        self.spec = spec
        self.rng = rng
        self.blocks: List[Block] = []
        self.anchors: List[Anchor] = []
        self.headings: List[Tuple[int, str]] = []
        self.chars = 0
        self.page = 1

    # -- structure ---------------------------------------------------------

    def set_page(self, page: int) -> None:
        self.page = page

    def h(self, text: str, level: int = 1) -> "DocBuilder":
        self.blocks.append(Heading(text=text, level=level))
        self.headings.append((level, text))
        self.chars += len(text)
        return self

    def p(self, text: str, style: str = "body") -> "DocBuilder":
        self.blocks.append(Para(text=text, style=style))
        self.chars += len(text)
        return self

    def bullets(self, items: Sequence[str], ordered: bool = False) -> "DocBuilder":
        self.blocks.append(Bullets(items=list(items), ordered=ordered))
        self.chars += sum(len(i) for i in items)
        return self

    def code(self, lines: Sequence[str], caption: Optional[str] = None) -> "DocBuilder":
        self.blocks.append(CodeBlock(lines=list(lines), caption=caption))
        self.chars += sum(len(l) for l in lines)
        return self

    def rule(self) -> "DocBuilder":
        self.blocks.append(Rule())
        return self

    def formula(self, text: str) -> "DocBuilder":
        self.blocks.append(FormulaBlock(text=text))
        self.chars += len(text)
        return self

    def footnotes(self, items: Sequence[str]) -> "DocBuilder":
        self.blocks.append(FootnoteBlock(items=list(items)))
        self.chars += sum(len(i) for i in items)
        return self

    def references(self, items: Sequence[str]) -> "DocBuilder":
        self.blocks.append(ReferenceBlock(items=list(items)))
        self.chars += sum(len(i) for i in items)
        return self

    def ad(self, headline: str, lines: Sequence[str]) -> "DocBuilder":
        self.blocks.append(AdBlock(headline=headline, lines=list(lines)))
        self.chars += len(headline) + sum(len(l) for l in lines)
        return self

    def table(
        self,
        rows: Sequence[Sequence[str]],
        caption: str,
        style: str = S.TABLE_SIMPLE,
        col_widths: Optional[Sequence[float]] = None,
    ) -> "DocBuilder":
        self.blocks.append(
            TableBlock(
                rows=[list(r) for r in rows],
                caption=caption,
                style=style,
                col_widths=list(col_widths) if col_widths else None,
            )
        )
        self.chars += sum(len(str(c)) for r in rows for c in r) + len(caption)
        return self

    def figure(
        self,
        fig_id: str,
        caption: str,
        art: str = "flow",
        width_ratio: float = 0.86,
        **art_kwargs,
    ) -> "DocBuilder":
        self.blocks.append(
            FigureBlock(
                fig_id=fig_id,
                caption=caption,
                art=art,
                width_ratio=width_ratio,
                art_kwargs=dict(art_kwargs),
            )
        )
        self.chars += len(caption)
        return self

    # -- anchors -----------------------------------------------------------

    def anchor(
        self,
        text: str,
        question: str,
        answer: str,
        section: str,
        kind: str = "fact",
        answer_type: str = "text",
        figure_id: str = "",
        difficulty: str = "medium",
    ) -> Anchor:
        """Register an anchor and record the text in the character budget.

        The caller is responsible for having already written *text* into the
        document (or for passing it to :meth:`p_with_anchor`).
        """
        item = Anchor(
            text=text,
            question=question,
            answer=answer,
            section=section,
            page=self.page,
            kind=kind,
            answer_type=answer_type,
            figure_id=figure_id,
            difficulty=difficulty,
        )
        self.anchors.append(item)
        return item

    def p_anchor(
        self,
        prefix: str,
        evidence: str,
        suffix: str,
        question: str,
        answer: str,
        section: str,
        kind: str = "fact",
        answer_type: str = "text",
        figure_id: str = "",
        difficulty: str = "medium",
    ) -> Anchor:
        """Write ``prefix + evidence + suffix`` as a paragraph, anchored on *evidence*."""
        self.p(f"{prefix}{evidence}{suffix}")
        return self.anchor(
            evidence, question, answer, section, kind, answer_type, figure_id, difficulty
        )

    def pagebreak(self) -> "DocBuilder":
        self.blocks.append(PageBreak())
        return self

    def budget_left(self, capacity: int) -> int:
        return capacity - self.chars


# ---------------------------------------------------------------------------
# Prose pools
# ---------------------------------------------------------------------------

CONNECTORS_ZH = (
    "在此基础上", "与此相关", "进一步看", "从工程实践出发", "综合上述分析",
    "值得注意的是", "需要强调的是", "结合实际部署经验", "就实现细节而言", "更具体地说",
)

CONNECTORS_EN = (
    "Building on this", "Relatedly", "Looking further", "From an engineering standpoint",
    "Taken together", "It is worth noting that", "To emphasise the practical side",
    "Drawing on deployment experience", "On implementation detail", "More concretely",
)

VERBS_ZH = (
    "显著提升了", "稳定改善了", "在一定程度上优化了", "明显降低了", "有效缓解了",
    "系统性重构了", "以较低成本实现了", "受限于算力而牺牲了", "针对长尾场景优化了", "逐步收敛于",
)

VERBS_EN = (
    "materially improves", "stabilises", "partially optimises", "clearly reduces",
    "effectively mitigates", "systematically restructures", "achieves at low cost",
    "trades accuracy for throughput in", "tunes for tail cases in", "converges on",
)

NOUNS_ZH = (
    "召回质量", "端到端延迟", "索引吞吐", "上下文一致性", "标注成本", "推理显存占用",
    "长尾查询的命中率", "分块边界的语义完整性", "元数据覆盖率", "跨语言检索表现",
    "缓存命中率", "重排阶段的稳定性", "离线评估的可复现性", "数据摄取的成功率",
)

NOUNS_EN = (
    "recall quality", "end-to-end latency", "index throughput", "context consistency",
    "annotation cost", "inference memory footprint", "tail-query hit rate",
    "semantic integrity of chunk boundaries", "metadata coverage", "cross-lingual retrieval",
    "cache hit rate", "reranking stability", "offline evaluation reproducibility",
    "ingestion success rate",
)

METRIC_UNITS = ("ms", "%", "QPS", "GB", "MB", "tokens", "req/s", "pts", "x", "µs")


def _num(rng: random.Random, low: float, high: float, digits: int = 2) -> float:
    return round(rng.uniform(low, high), digits)


def sentence_zh(rng: random.Random, subject: str, index: int) -> str:
    connector = rng.choice(CONNECTORS_ZH)
    verb = rng.choice(VERBS_ZH)
    noun = rng.choice(NOUNS_ZH)
    value = _num(rng, 3.5, 92.0, 1)
    unit = rng.choice(("个百分点", "毫秒", "%"))
    return (
        f"{connector}，第 {index} 组对照实验表明，{subject}{verb}{noun}，"
        f"实测变化约 {value}{unit}。"
    )


def sentence_en(rng: random.Random, subject: str, index: int) -> str:
    connector = rng.choice(CONNECTORS_EN)
    verb = rng.choice(VERBS_EN)
    noun = rng.choice(NOUNS_EN)
    value = _num(rng, 3.5, 92.0, 1)
    unit = rng.choice(("percentage points", "ms", "%"))
    return (
        f"{connector}, the {index}th control run shows that {subject} {verb} {noun}, "
        f"with a measured change of about {value} {unit}."
    )


def sentence_mixed(rng: random.Random, subject: str, index: int) -> str:
    connector = rng.choice(CONNECTORS_ZH + CONNECTORS_EN)
    verb = rng.choice(VERBS_ZH)
    noun = rng.choice(NOUNS_EN)
    value = _num(rng, 3.5, 92.0, 1)
    return (
        f"{connector}，在第 {index} 组对照实验中 {subject} {verb} {noun} "
        f"(measured change ≈ {value}%)."
    )


def prose(rng: random.Random, spec: S.DocSpec, subject: str, index: int) -> str:
    """One locale-appropriate paragraph sentence."""
    if spec.language == S.LANG_EN:
        return sentence_en(rng, subject, index)
    if spec.language == S.LANG_MIXED:
        return sentence_mixed(rng, subject, index)
    return sentence_zh(rng, subject, index)


def paragraph(rng: random.Random, spec: S.DocSpec, subject: str, index: int, n: int = 3) -> str:
    """A paragraph of *n* distinct sentences."""
    return " ".join(prose(rng, spec, subject, index * 10 + i) for i in range(n))


# ---------------------------------------------------------------------------
# Section headings
# ---------------------------------------------------------------------------


def chapter_title(rng: random.Random, spec: S.DocSpec, index: int) -> str:
    """A plausible section title for unit *index*."""
    subjects = SECTION_SUBJECTS.get(spec.category, SECTION_SUBJECTS[ACADEMIC_FALLBACK])
    subject = subjects[(index - 1) % len(subjects)]
    if spec.language == S.LANG_EN:
        return f"{index}. {subject[1]}"
    return f"第 {index} 章  {subject[0]}"


ACADEMIC_FALLBACK = "ACAD"

SECTION_SUBJECTS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ACAD": (
        ("引言与问题定义", "Introduction and Problem Definition"),
        ("相关工作", "Related Work"),
        ("方法概述", "Method Overview"),
        ("模型架构设计", "Model Architecture"),
        ("训练与优化细节", "Training and Optimisation"),
        ("实验设置", "Experimental Setup"),
        ("主要结果", "Main Results"),
        ("消融实验", "Ablation Study"),
        ("错误分析", "Error Analysis"),
        ("效率与成本分析", "Efficiency and Cost Analysis"),
        ("定性案例分析", "Qualitative Case Study"),
        ("泛化性与鲁棒性", "Generalisation and Robustness"),
        ("讨论与局限性", "Discussion and Limitations"),
        ("结论与未来工作", "Conclusion and Future Work"),
    ),
    "TECH": (
        ("概述与适用版本", "Overview and Supported Versions"),
        ("快速开始", "Quick Start"),
        ("认证与鉴权", "Authentication"),
        ("核心接口说明", "Core API"),
        ("参数与返回结构", "Parameters and Responses"),
        ("错误码与异常处理", "Error Codes and Handling"),
        ("配置项参考", "Configuration Reference"),
        ("性能与配额限制", "Performance and Quotas"),
        ("版本差异与迁移", "Version Differences and Migration"),
        ("故障排查", "Troubleshooting"),
        ("安全与合规", "Security and Compliance"),
        ("附录：变更记录", "Appendix: Changelog"),
    ),
    "FIN": (
        ("公司概况与业务回顾", "Company Overview and Business Review"),
        ("行业与竞争格局", "Industry and Competition"),
        ("经营成果分析", "Results of Operations"),
        ("分季度财务数据", "Quarterly Financial Data"),
        ("资产负债与现金流", "Balance Sheet and Cash Flow"),
        ("分部业绩", "Segment Performance"),
        ("主要风险因素", "Principal Risk Factors"),
        ("管理层讨论与分析", "Management Discussion and Analysis"),
        ("公司治理", "Corporate Governance"),
        ("财务报表附注", "Notes to Financial Statements"),
    ),
    "LEGAL": (
        ("总则", "General Provisions"),
        ("适用范围与定义", "Scope and Definitions"),
        ("权利与义务", "Rights and Obligations"),
        ("数据与信息安全", "Data and Information Security"),
        ("合规要求", "Compliance Requirements"),
        ("监督管理", "Supervision"),
        ("法律责任", "Legal Liability"),
        ("附则", "Supplementary Provisions"),
        ("争议解决", "Dispute Resolution"),
        ("生效与修订", "Effectiveness and Amendment"),
    ),
    "MED": (
        ("流行病学与疾病负担", "Epidemiology and Burden of Disease"),
        ("诊断与评估", "Diagnosis and Assessment"),
        ("治疗方案", "Treatment Regimen"),
        ("用药剂量与调整", "Dosage and Adjustment"),
        ("特殊人群用药", "Special Populations"),
        ("不良反应与监测", "Adverse Reactions and Monitoring"),
        ("随访与预后", "Follow-up and Prognosis"),
        ("患者教育要点", "Patient Education"),
    ),
    "NEWS": (
        ("现场纪实", "Dispatches from the Field"),
        ("数据与背景", "Data and Background"),
        ("关键人物访谈", "Interview: Key Figures"),
        ("产业链透视", "Inside the Supply Chain"),
        ("争议与反驳", "Contention and Rebuttal"),
        ("区域对比", "Regional Comparison"),
        ("时间线梳理", "Timeline"),
        ("前景展望", "Outlook"),
    ),
}


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def simple_table_rows(rng: random.Random, spec: S.DocSpec, index: int, rows: int = 5) -> List[List[str]]:
    """A plain parameter/comparison table with unique values."""
    if spec.language == S.LANG_EN:
        header = ["Parameter", "Baseline", "Proposed", "Delta"]
    else:
        header = ["参数", "基线", "本方案", "变化"]
    body = [header]
    for r in range(rows):
        base = _num(rng, 10.0, 400.0, 1)
        new = round(base * rng.uniform(0.55, 1.25), 1)
        delta = round((new - base) / base * 100, 1)
        body.append([f"metric_{index:02d}_{r + 1}", f"{base}", f"{new}", f"{delta:+}%"])
    return body


def merged_table_rows(rng: random.Random, spec: S.DocSpec, index: int) -> List[List[str]]:
    """Header cells that span columns -- layout carries meaning, not just text."""
    if spec.language == S.LANG_EN:
        return [
            ["Segment", "FY2024", "", "FY2025", ""],
            ["", "Revenue", "Margin", "Revenue", "Margin"],
            ["Cloud", f"{_num(rng, 800, 2400, 1)}", f"{_num(rng, 12, 38, 1)}%",
             f"{_num(rng, 900, 2600, 1)}", f"{_num(rng, 13, 40, 1)}%"],
            ["Devices", f"{_num(rng, 300, 900, 1)}", f"{_num(rng, 4, 19, 1)}%",
             f"{_num(rng, 280, 880, 1)}", f"{_num(rng, 3, 18, 1)}%"],
            ["Services", f"{_num(rng, 150, 600, 1)}", f"{_num(rng, 20, 44, 1)}%",
             f"{_num(rng, 170, 640, 1)}", f"{_num(rng, 21, 46, 1)}%"],
        ]
    return [
        ["业务分部", "2024 年", "", "2025 年", ""],
        ["", "营业收入", "毛利率", "营业收入", "毛利率"],
        ["云计算", f"{_num(rng, 800, 2400, 1)}", f"{_num(rng, 12, 38, 1)}%",
         f"{_num(rng, 900, 2600, 1)}", f"{_num(rng, 13, 40, 1)}%"],
        ["智能终端", f"{_num(rng, 300, 900, 1)}", f"{_num(rng, 4, 19, 1)}%",
         f"{_num(rng, 280, 880, 1)}", f"{_num(rng, 3, 18, 1)}%"],
        ["服务业务", f"{_num(rng, 150, 600, 1)}", f"{_num(rng, 20, 44, 1)}%",
         f"{_num(rng, 170, 640, 1)}", f"{_num(rng, 21, 46, 1)}%"],
    ]


def multipage_table_rows(rng: random.Random, spec: S.DocSpec, index: int, rows: int = 34) -> List[List[str]]:
    """A long table that must break across a page boundary."""
    if spec.language == S.LANG_EN:
        header = ["#", "Endpoint", "Method", "Quota/min", "p95 (ms)"]
    else:
        header = ["序号", "接口", "方法", "每分钟配额", "P95 延迟(ms)"]
    body = [header]
    for r in range(rows):
        body.append([
            str(r + 1),
            f"/v{index % 3 + 1}/resource_{index:02d}_{r + 1}",
            ("GET", "POST", "PUT", "DELETE")[r % 4],
            str(int(_num(rng, 60, 6000, 0))),
            str(int(_num(rng, 12, 940, 0))),
        ])
    return body


def borderless_table_rows(rng: random.Random, spec: S.DocSpec, index: int) -> List[List[str]]:
    """A table with no rules at all: structure is purely positional."""
    if spec.language == S.LANG_EN:
        return [
            ["Stage", "Owner", "Duration", "Exit criteria"],
            ["Intake", "Data team", f"{int(_num(rng, 2, 9, 0))} days", "hash recorded"],
            ["Parse", "Platform", f"{int(_num(rng, 1, 5, 0))} days", "text length > 0"],
            ["Index", "Search team", f"{int(_num(rng, 3, 14, 0))} days", "recall@10 stable"],
            ["Review", "QA", f"{int(_num(rng, 1, 6, 0))} days", "sign-off recorded"],
        ]
    return [
        ["阶段", "负责方", "周期", "退出条件"],
        ["接入", "数据组", f"{int(_num(rng, 2, 9, 0))} 天", "哈希已登记"],
        ["解析", "平台组", f"{int(_num(rng, 1, 5, 0))} 天", "文本长度大于 0"],
        ["索引", "检索组", f"{int(_num(rng, 3, 14, 0))} 天", "Recall@10 稳定"],
        ["复核", "质量组", f"{int(_num(rng, 1, 6, 0))} 天", "签署完成"],
    ]


def multiheader_table_rows(rng: random.Random, spec: S.DocSpec, index: int) -> List[List[str]]:
    """Two-level grouped header expressed with spans."""
    if spec.language == S.LANG_EN:
        return [
            ["Region", "Q1", "", "Q2", ""],
            ["", "Units", "Revenue", "Units", "Revenue"],
            ["North", str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}",
             str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}"],
            ["South", str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}",
             str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}"],
            ["East", str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}",
             str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}"],
        ]
    return [
        ["区域", "第一季度", "", "第二季度", ""],
        ["", "销量", "收入", "销量", "收入"],
        ["华北", str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}",
         str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}"],
        ["华东", str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}",
         str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}"],
        ["华南", str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}",
         str(int(_num(rng, 100, 999, 0))), f"{_num(rng, 10, 90, 1)}"],
    ]


TABLE_ROW_BUILDERS: Dict[str, Tuple[Callable[..., List[List[str]]], int]] = {
    S.TABLE_SIMPLE: (simple_table_rows, 5),
    S.TABLE_MERGED: (merged_table_rows, 5),
    S.TABLE_MULTIPAGE: (multipage_table_rows, 34),
    S.TABLE_BORDERLESS: (borderless_table_rows, 5),
    S.TABLE_MULTIHEADER: (multiheader_table_rows, 5),
}

#: Rough character cost of rendering each table class, used by the per-unit
#: content budget so that a heavy table cannot push a unit onto a second page.
TABLE_CHAR_COST: Dict[str, int] = {
    S.TABLE_SIMPLE: 320,
    S.TABLE_MERGED: 340,
    S.TABLE_MULTIPAGE: 1450,
    S.TABLE_BORDERLESS: 300,
    S.TABLE_MULTIHEADER: 340,
}


#: ``style -> (anchor cell, answer cell)`` as ``(row, col)`` indices into the
#: generated rows.  The anchor is always a **single cell** so that extractors
#: which render tables as Markdown pipes, or reflow cells across lines, cannot
#: break it apart; the answer is the neighbouring value cell.
TABLE_ANCHOR_CELLS: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    S.TABLE_SIMPLE: ((2, 0), (2, 1)),
    S.TABLE_MERGED: ((2, 0), (2, 1)),
    S.TABLE_MULTIHEADER: ((2, 0), (2, 1)),
    S.TABLE_BORDERLESS: ((2, 0), (2, 2)),
    S.TABLE_MULTIPAGE: ((2, 1), (2, 3)),
}


def build_table(
    builder: DocBuilder,
    style: str,
    index: int,
    caption: str,
    anchor_question: Optional[str] = None,
    anchor_answer: Optional[str] = None,
) -> Optional[Anchor]:
    """Render one table of the requested style and optionally anchor a fact on it."""
    spec = builder.spec
    rows_fn, rows = TABLE_ROW_BUILDERS[style]
    data = rows_fn(builder.rng, spec, index, rows) if style == S.TABLE_MULTIPAGE else rows_fn(builder.rng, spec, index)
    builder.table(data, caption=caption, style=style)

    if anchor_question is None:
        return None

    (arow, acol), (vrow, vcol) = TABLE_ANCHOR_CELLS.get(style, ((1, 0), (1, 1)))
    anchor_cell = _cell(data, arow, acol)
    answer_cell = anchor_answer or _cell(data, vrow, vcol)
    if not anchor_cell:
        return None
    return builder.anchor(
        anchor_cell, anchor_question, answer_cell, caption, kind="table",
        answer_type="numeric",
    )


def _cell(rows: Sequence[Sequence[str]], row: int, col: int) -> str:
    """Fetch a table cell as a string, tolerating short rows."""
    if row >= len(rows):
        return ""
    cells = rows[row]
    if col >= len(cells):
        return ""
    return str(cells[col]).strip()


# ---------------------------------------------------------------------------
# Slot distribution and page furniture
# ---------------------------------------------------------------------------


def spread_slots(pages: int, count: int, start: int = 0) -> List[int]:
    """Distribute *count* zero-based unit slots evenly across *pages* units.

    Used to decide which page carries a table, a figure, or a formula so that
    those elements are spread through the document instead of clustered.
    """
    available = pages - start
    if available <= 0 or count <= 0:
        return []
    count = min(count, available)
    if count >= available:
        return list(range(start, pages))
    step = available / count
    return sorted({start + min(available - 1, int(round(i * step))) for i in range(count)})


def distribute_elements(
    units: Sequence[int],
    requests: Sequence[Tuple[str, Any]],
) -> Dict[int, List[Tuple[str, Any]]]:
    """Assign heavy elements to page units, spreading them as evenly as possible.

    Each request is placed on the unit that currently carries the fewest
    elements (leftmost wins ties), which keeps at most one heavy element per page
    whenever the document has enough pages for them.  When a document genuinely
    has more elements than pages, the excess is stacked rather than dropped, and
    the caller's per-unit content budget shrinks the prose to compensate.
    """
    assignment: Dict[int, List[Tuple[str, Any]]] = {u: [] for u in units}
    if not units:
        return assignment
    for request in requests:
        target = min(units, key=lambda u: (len(assignment[u]), u))
        assignment[target].append(request)
    return assignment


def figure_caption(spec: S.DocSpec, number: int, text: str) -> str:
    """Locale-appropriate figure caption, e.g. ``图 3：RAG 数据摄取流程图``."""
    if spec.language == S.LANG_EN:
        return f"Figure {number}: {text}"
    return f"图 {number}：{text}"


def table_caption(spec: S.DocSpec, number: int, text: str) -> str:
    """Locale-appropriate table caption, e.g. ``表 2：分层参数对照``."""
    if spec.language == S.LANG_EN:
        return f"Table {number}: {text}"
    return f"表 {number}：{text}"


def cover_blocks(builder: DocBuilder) -> Optional[Anchor]:
    """Render a title page.  Returns a factual anchor on the version string."""
    spec = builder.spec
    rng = builder.rng
    version = f"v{1 + (spec.seq % 4)}.{spec.seq % 10}"
    doc_code = f"{spec.category}-{spec.seq:02d}/{rng.randint(2000, 2999)}"

    builder.h(spec.title, level=0)
    if spec.title_alt:
        builder.p(spec.title_alt, style="subtitle")

    meta = [
        ["来源", spec.source_style] if spec.language != S.LANG_EN else ["Source", spec.source_style],
        ["版本", version] if spec.language != S.LANG_EN else ["Version", version],
        ["文档编号", doc_code] if spec.language != S.LANG_EN else ["Document ID", doc_code],
        ["页数", str(spec.pages)] if spec.language != S.LANG_EN else ["Length", f"{spec.pages} pages"],
    ]
    if spec.language == S.LANG_EN:
        meta[0][0] = "Source"
    builder.table(
        meta,
        caption=table_caption(spec, 1, "文档信息" if spec.language != S.LANG_EN else "Document information"),
        style=S.TABLE_INFO,
    )

    question = (
        f"该文档的文档编号是什么？" if spec.language != S.LANG_EN
        else f"What is the document ID of this report?"
    )
    return builder.anchor(
        doc_code, question, doc_code,
        section="封面" if spec.language != S.LANG_EN else "Cover",
        kind="fact",
    )


def toc_blocks(builder: DocBuilder, first_chapter: int) -> None:
    """Render a table of contents listing the remaining chapters.

    Capped at 40 entries: a 220-page document would otherwise need a table of
    contents longer than the page that holds it, which would push that unit onto
    a second page and break the exact page count.
    """
    spec = builder.spec
    builder.h("目录" if spec.language != S.LANG_EN else "Contents", level=1)
    entries = [
        chapter_title(builder.rng, spec, i)
        for i in range(first_chapter, spec.pages + 1)
    ]
    if len(entries) > 40:
        step = len(entries) / 40.0
        entries = [entries[min(len(entries) - 1, int(i * step))] for i in range(40)]
    builder.bullets(entries, ordered=False)


# ---------------------------------------------------------------------------
# Footnotes / references / formulas
# ---------------------------------------------------------------------------


def footnote_items(rng: random.Random, spec: S.DocSpec, index: int, count: int = 2) -> List[str]:
    out = []
    for i in range(count):
        value = _num(rng, 1.0, 99.0, 1)
        if spec.language == S.LANG_EN:
            out.append(f"[{index}.{i + 1}] See Appendix {index}.{i + 1}, table {value} for the full breakdown.")
        else:
            out.append(f"[{index}.{i + 1}] 详见附录 {index}.{i + 1}，完整明细见第 {value} 页表格。")
    return out


def reference_items(rng: random.Random, spec: S.DocSpec, index: int, count: int = 6) -> List[str]:
    out = []
    for i in range(count):
        year = 2016 + ((index + i) % 10)
        if spec.language == S.LANG_EN:
            out.append(
                f"[{index * count + i + 1}] Author, A. et al. ({year}). "
                f"A study of method {index}.{i + 1}. Journal of Applied Systems, "
                f"{10 + (index % 40)}({i + 1}), {100 + index * 3 + i}."
            )
        else:
            out.append(
                f"[{index * count + i + 1}] 作者等（{year}）。方法 {index}.{i + 1} 的实证研究。"
                f"《应用系统学报》，{10 + (index % 40)}（{i + 1}），{100 + index * 3 + i}。"
            )
    return out


FORMULAS = (
    "Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V",
    "RRF(d) = Σ_i 1 / (k + rank_i(d))",
    "BM25(q, d) = Σ_t IDF(t) · (f(t,d)(k₁+1)) / (f(t,d) + k₁(1 - b + b·|d|/avgdl))",
    "P(y | x) = Π_{t=1}^{T} P(y_t | y_{<t}, x; θ)",
    "L = -Σ_i log P(y_i | x_i; θ) + λ‖θ‖²",
    "cos(u, v) = u·v / (‖u‖₂ · ‖v‖₂)",
)


# ---------------------------------------------------------------------------
# Refusal pool -- entities that appear in NO document of the corpus
# ---------------------------------------------------------------------------

REFUSAL_POOL: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("贵公司 2019 年在开曼群岛的递延税项负债具体是多少？",
     ("开曼群岛", "递延税项负债")),
    ("该产品在火星表面大气条件下的额定功率是多少瓦？",
     ("火星表面大气条件", "额定功率")),
    ("文档中提到的 ZX-9942 型磁悬浮压缩机的保修期是多久？",
     ("ZX-9942", "磁悬浮压缩机")),
    ("作者在 1998 年的那篇论文中给出的对照实验样本量是多少？",
     ("1998 年", "对照实验样本量")),
    ("该法规对深海锰结核开采的许可证编号有何规定？",
     ("深海锰结核", "许可证编号")),
    ("该药品在妊娠 D 类人群中的半衰期是多少小时？",
     ("妊娠 D 类", "半衰期")),
    ("请给出本文未披露的第六作者的全名与邮箱。",
     ("第六作者", "全名与邮箱")),
    ("该公司的审计机构在 2011 年出具了哪种非标意见？",
     ("2011 年", "非标意见")),
    ("手册中 Aurora-7 固件的校验和是多少？",
     ("Aurora-7 固件", "校验和")),
    ("该研究使用的冷冻电镜型号与加速电压是多少？",
     ("冷冻电镜型号", "加速电压")),
)


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------


UnitFactory = Callable[[DocBuilder, int, bool], None]


def fact_unit_indices(pages: int, count: int) -> List[int]:
    """Evenly spread *count* fact-bearing units across *pages* units.

    Spread rather than front-loaded so that cross-page questions are genuinely
    cross-page even in the 220-page document.
    """
    if pages <= 0:
        return []
    count = min(count, pages)
    if count >= pages:
        return list(range(pages))
    step = pages / count
    return sorted({min(pages - 1, int(round(i * step))) for i in range(count)})


def assemble_plan(
    spec: S.DocSpec,
    seed: int,
    unit_factory: UnitFactory,
    fact_units: int = 12,
    refusals: int = 2,
    min_questions: int = 10,
) -> DocPlan:
    """Build a complete :class:`DocPlan` for *spec*.

    Args:
        spec: The document specification.
        seed: Global corpus seed; combined with the doc id so that adding or
            removing documents never perturbs another document.
        unit_factory: Emits one page unit's blocks (and anchors) per call.
        fact_units: How many units should register answerable anchors.
        refusals: How many unanswerable questions to attach.
        min_questions: Floor on the answer-bank size for this document.  Very
            short documents cannot host enough distinct anchors, so the remainder
            is topped up from anchors the kind-balanced selection did not use.
    """
    rng = random.Random(f"{seed}:{spec.doc_id}")
    builder = DocBuilder(spec, rng)

    emit_at = set(fact_unit_indices(spec.pages, fact_units))
    for index in range(spec.pages):
        builder.set_page(index + 1)
        unit_factory(builder, index + 1, index in emit_at)
        builder.pagebreak()

    plan = DocPlan(
        doc_id=spec.doc_id,
        category=spec.category,
        title=spec.title,
        subtitle=spec.title_alt,
        body=builder.blocks,
        notes={
            "anchors": len(builder.anchors),
            "headings": len(builder.headings),
            "chars": builder.chars,
        },
    )

    used = _emit_facts(spec, plan, builder.anchors, rng)

    chosen = pick_refusals(rng, refusals)
    for i, (question, terms) in enumerate(chosen):
        plan.add_refusals(
            RefusalQuestion(
                fact_id=f"{spec.fact_prefix()}-R{i + 1:02d}",
                question=question,
                reject_terms=terms,
                reason="该实体在整个语料中不存在，系统应如实拒答",
                difficulty="hard" if i else "medium",
            )
        )

    _enforce_floor(spec, plan, builder.anchors, used, min_questions, rng, chosen)
    return plan


def _enforce_floor(
    spec: S.DocSpec,
    plan: DocPlan,
    anchors: List[Anchor],
    used: set,
    min_questions: int,
    rng: random.Random,
    already_asked: Sequence[Tuple[str, Tuple[str, ...]]],
) -> None:
    """Guarantee the answer bank reaches its floor for every document.

    First from unused anchors, then -- for a one-page document with no tables and
    no figures, which simply has too little content to host ten distinct
    assertions -- from additional refusal questions.
    """
    counter = len(plan.facts) + 1
    for anchor in anchors:
        if plan.question_count >= min_questions:
            break
        if anchor.text in used:
            continue
        used.add(anchor.text)
        plan.add_facts(
            Fact(
                fact_id=f"{spec.fact_prefix()}-F{counter:02d}",
                kind=FACTUAL,
                question=anchor.question,
                answer=anchor.answer,
                anchor=anchor.text,
                answer_type=anchor.answer_type,
                section=anchor.section,
                page_hint=anchor.page,
                figure_id=anchor.figure_id,
                difficulty=anchor.difficulty,
                requires_ocr=spec.is_ocr,
            )
        )
        counter += 1

    if plan.question_count >= min_questions:
        return

    asked = {question for question, _ in already_asked}
    spare = [item for item in REFUSAL_POOL if item[0] not in asked]
    rng.shuffle(spare)
    index = len(plan.refusals) + 1
    for question, terms in spare:
        if plan.question_count >= min_questions:
            break
        plan.add_refusals(
            RefusalQuestion(
                fact_id=f"{spec.fact_prefix()}-R{index:02d}",
                question=question,
                reject_terms=terms,
                reason="该实体在整个语料中不存在，系统应如实拒答",
                difficulty="hard",
            )
        )
        index += 1


def pick_refusals(rng: random.Random, count: int) -> List[Tuple[str, Tuple[str, ...]]]:
    """Pick *count* distinct refusal questions."""
    pool = list(REFUSAL_POOL)
    rng.shuffle(pool)
    return pool[:count]


def _emit_facts(
    spec: S.DocSpec,
    plan: DocPlan,
    anchors: List[Anchor],
    rng: random.Random,
) -> set:
    """Turn collected anchors into a balanced set of :class:`Fact` objects.

    Returns the set of anchor texts that were consumed, so the caller can top up
    the answer bank from whatever is left over.
    """
    if not anchors:
        return set()

    factual = [a for a in anchors if a.kind == "fact"]
    tables = [a for a in anchors if a.kind == "table"]
    figures = [a for a in anchors if a.kind == "figure"]

    # Target mix: ~8 factual, up to 3 table, 2 figure, 1 cross-page, 1 cross-doc.
    picked: List[Tuple[str, Anchor]] = []
    picked.extend((FACTUAL, a) for a in factual[:8])
    picked.extend((TABLE, a) for a in tables[:3])
    picked.extend((FIGURE, a) for a in figures[:2] if spec.has_figures)

    # Cross-page: two anchors far apart in the document.
    if len(anchors) >= 4:
        ordered = sorted(anchors, key=lambda a: a.page)
        first = ordered[0]
        last = ordered[-1]
        if first.page != last.page:
            cross = Fact(
                fact_id=f"{spec.fact_prefix()}-FXP",
                kind=CROSS_PAGE,
                question=(
                    f"第 {first.page} 页所披露的「{first.answer}」与第 {last.page} 的"
                    f"「{last.answer}」之间是什么关系？"
                    if spec.language != S.LANG_EN
                    else f"How does the figure reported on page {first.page} relate to "
                         f"the one reported on page {last.page}?"
                ),
                answer=(
                    f"第 {first.page} 页给出 {first.answer}；第 {last.page} 页给出 {last.answer}，"
                    f"两者需跨页对照才能得出完整结论。"
                    if spec.language != S.LANG_EN
                    else f"Page {first.page} reports {first.answer} and page {last.page} "
                         f"reports {last.answer}; the conclusion requires both."
                ),
                anchor=first.text,
                secondary_anchor=last.text,
                answer_type="text",
                section=f"{first.section} / {last.section}",
                page_hint=first.page,
                difficulty="hard",
                requires_ocr=spec.is_ocr,
            )
            plan.add_facts(cross)

    # Cross-document: reference the version-pair partner when there is one.
    if spec.version_pair:
        partner = spec.version_pair
        seed_anchor = picked[0][1] if picked else anchors[0]
        plan.add_facts(
            Fact(
                fact_id=f"{spec.fact_prefix()}-FXD",
                kind=CROSS_DOC,
                question=(
                    f"{spec.doc_id} 与 {partner} 在「{seed_anchor.section}」上的表述有何差异？"
                    if spec.language != S.LANG_EN
                    else f"What differs between {spec.doc_id} and {partner} regarding "
                         f"\"{seed_anchor.section}\"?"
                ),
                answer=(
                    f"两版结构一致，但 {spec.doc_id} 在该处的取值为 {seed_anchor.answer}，"
                    f"另一版为修订后的数值。"
                    if spec.language != S.LANG_EN
                    else f"Both versions share the same structure; {spec.doc_id} reports "
                         f"{seed_anchor.answer} where the other version reports the revised value."
                ),
                anchor=seed_anchor.text,
                secondary_anchor=seed_anchor.text,
                secondary_doc=partner,
                answer_type="text",
                section=seed_anchor.section,
                page_hint=seed_anchor.page,
                difficulty="hard",
                requires_ocr=spec.is_ocr,
            )
        )

    seen: set[str] = set()
    counter = 0
    for kind, anchor in picked:
        if anchor.text in seen:
            continue
        seen.add(anchor.text)
        counter += 1
        plan.add_facts(
            Fact(
                fact_id=f"{spec.fact_prefix()}-F{counter:02d}",
                kind=kind,  # type: ignore[arg-type]
                question=anchor.question,
                answer=anchor.answer,
                anchor=anchor.text,
                answer_type=anchor.answer_type,
                section=anchor.section,
                page_hint=anchor.page,
                figure_id=anchor.figure_id,
                difficulty=anchor.difficulty,
                requires_ocr=spec.is_ocr,
            )
        )
    return seen


# ---------------------------------------------------------------------------
# Generic document builder
# ---------------------------------------------------------------------------

#: Per-category label for the unique "spec item" that factual anchors are built
#: on.  Each label embeds the unit index, which makes every anchor unique inside
#: its document.
SPEC_ITEM_LABELS: Dict[str, Tuple[str, str]] = {
    "ACAD": ("超参数", "hyper-parameter"),
    "TECH": ("配置项", "configuration key"),
    "FIN": ("统计口径", "reporting line"),
    "LEGAL": ("条款阈值", "provision threshold"),
    "MED": ("剂量档位", "dose tier"),
    "NEWS": ("观测指标", "observed indicator"),
}

#: Figure artwork used by each category, in the order they are distributed.
CATEGORY_FIGURE_ARTS: Dict[str, Tuple[str, ...]] = {
    "ACAD": ("flow", "line", "line", "flow"),
    "TECH": ("screenshot", "flow", "bar", "screenshot"),
    "FIN": ("pie", "bar", "line"),
    "LEGAL": ("flow",),
    "MED": ("flow", "photo", "bar"),
    "NEWS": ("photo", "bar", "photo", "pie"),
}


@dataclass
class Profile:
    """Per-category rendering preferences."""

    figure_arts: Tuple[str, ...] = ()
    formula_text: int = 0
    formula_images: int = 0
    code_blocks: int = 0
    footnote_every: int = 3
    references: bool = False
    extra_tables_per_12_pages: int = 0
    body_sentences: int = 3
    difficulty: str = "medium"


def category_profile(category: str) -> Profile:
    """Default profile per category."""
    if category == "ACAD":
        return Profile(
            figure_arts=CATEGORY_FIGURE_ARTS["ACAD"], formula_text=5,
            footnote_every=3, references=True, body_sentences=4,
        )
    if category == "TECH":
        return Profile(
            figure_arts=CATEGORY_FIGURE_ARTS["TECH"], code_blocks=6,
            footnote_every=4, extra_tables_per_12_pages=1, body_sentences=3,
        )
    if category == "FIN":
        return Profile(
            figure_arts=CATEGORY_FIGURE_ARTS["FIN"], formula_text=1,
            footnote_every=2, extra_tables_per_12_pages=2, body_sentences=3,
        )
    if category == "LEGAL":
        return Profile(
            figure_arts=CATEGORY_FIGURE_ARTS["LEGAL"], footnote_every=6,
            references=True, body_sentences=4, difficulty="hard",
        )
    if category == "MED":
        return Profile(
            figure_arts=CATEGORY_FIGURE_ARTS["MED"], formula_text=2,
            footnote_every=3, extra_tables_per_12_pages=1, body_sentences=3,
        )
    if category == "NEWS":
        return Profile(
            figure_arts=CATEGORY_FIGURE_ARTS["NEWS"], footnote_every=5,
            body_sentences=5, difficulty="easy",
        )
    return Profile()


def article_subject(spec: S.DocSpec) -> str:
    """The noun phrase prose is written about."""
    return spec.title_alt if spec.language != S.LANG_EN else spec.title


def messy_level(index: int) -> int:
    """Heading level for a document with a deliberately broken hierarchy.

    Produces level skips (h1 -> h3 -> h2) and an occasional depth that should not
    exist, which is exactly the structure noise the corpus needs to cover.
    """
    return (1, 3, 2, 1, 4, 2, 5, 3)[index % 8]


def spec_item(builder: DocBuilder, index: int, j: int) -> Tuple[str, str, str]:
    """A unique ``(label, value, evidence)`` triple for a factual anchor."""
    spec = builder.spec
    label_word, _ = SPEC_ITEM_LABELS.get(spec.category, ("参数", "parameter"))
    value = f"{_num(builder.rng, 1.0, 99.0, 2)}"
    if spec.language == S.LANG_EN:
        label = f"{spec.doc_id}-P{index:02d}-{j + 1}"
        evidence = f"{label} = {value}"
    else:
        label = f"{spec.doc_id}-P{index:02d}-{j + 1}"
        evidence = f"{label_word} {label} 取值为 {value}"
    return label, value, evidence


def build_document(
    spec: S.DocSpec,
    seed: int,
    profile: Optional[Profile] = None,
    fill: Optional[float] = None,
) -> DocPlan:
    """Build the full :class:`DocPlan` for *spec* using *profile*.

    Args:
        spec: The document to build.
        seed: Corpus seed (combined with the doc id).
        profile: Category rendering profile; defaults to the category's own.
        fill: Per-unit character budget as a fraction of a full page.  Lower
            values shrink each unit and are used to recover from page-count
            overflow (see :data:`FILL_ATTEMPTS`).

    Layout of the page units:

    * unit 1 is the cover and unit 2 the table of contents (documents >= 8 pages);
    * every remaining unit is one chapter, ending with a page break;
    * tables, figures, formulas, code blocks, footnotes, references and adverts
      are distributed across those units by :func:`spread_slots`.
    """
    profile = profile or category_profile(spec.category)
    pages = spec.pages
    has_front = pages >= 8
    first_content = 3 if has_front else 1

    # -- slot maps ---------------------------------------------------------
    content_units = list(range(first_content, pages + 1))
    n_units = len(content_units)

    extra_tables = profile.extra_tables_per_12_pages * max(0, pages // 12)
    style_seq = list(spec.table_styles) + [S.TABLE_SIMPLE] * extra_tables
    figure_arts = list(profile.figure_arts) if spec.has_figures else []
    formula_image_count = profile.formula_images or spec.formula_images

    # A document cannot be asked to hold more heavy elements than it has pages:
    # a page carrying a 34-row table *and* a figure will overflow regardless of
    # how little prose it also carries.
    # At most one heavy element per page: a page carrying a 34-row table *and* a
    # figure overflows no matter how little prose it also carries, and that would
    # push the document past its specified page count.  Spec-required tables win
    # over formula images, which win over profile figures.
    style_seq = style_seq[:n_units]
    formula_image_count = min(formula_image_count, max(0, n_units - len(style_seq)))
    figure_cap = max(0, n_units - len(style_seq) - formula_image_count)
    figure_arts = figure_arts[:figure_cap]

    heavy: List[Tuple[str, Any]] = (
        [("table", style) for style in style_seq]
        + [("figure", art) for art in figure_arts]
        + [("formula_image", None)] * formula_image_count
    )
    element_at = distribute_elements(content_units, heavy)

    # Light elements are spread independently of the heavy ones.
    formula_slots = spread_slots(pages, profile.formula_text, start=first_content - 1)
    formula_at = {slot + 1 for slot in formula_slots}
    formula_image_at = {u for u, items in element_at.items()
                        if any(kind == "formula_image" for kind, _ in items)}

    # Code blocks are the one heavy element that is optional: give them only to
    # pages that carry nothing else.
    busy = {u for u, items in element_at.items() if items}
    free_units = [u for u in content_units if u not in busy]
    code_at = set(free_units[: max(0, min(profile.code_blocks, len(free_units)))])

    footnote_at = {
        i for i in content_units if (i - first_content) % profile.footnote_every == 0
    }
    reference_at = {pages} if profile.references else set()
    ad_slots = spread_slots(pages, spec.ads, start=first_content - 1)
    ad_at = {slot + 1 for slot in ad_slots}
    # Adverts are light but should not share a page with a heavy element either.
    ad_at = {u for u in ad_at if u not in busy} or ({content_units[-1]} if spec.ads else set())

    counters = {"table": 1, "figure": 0}

    # Short documents have few units to hang anchors on, so each fact-bearing
    # unit carries more of them; they also take an extra refusal question.
    short_doc = pages <= 3
    anchors_per_unit = 8 if short_doc else 2
    refusal_count = 3 if short_doc else 2

    # -- unit factory ------------------------------------------------------
    def unit(builder: DocBuilder, index: int, emit_facts: bool) -> None:
        builder.set_page(index)
        rng = builder.rng

        if has_front and index == 1:
            if emit_facts:
                cover_blocks(builder)
            else:
                cover_blocks(builder)
            return

        if has_front and index == 2:
            toc_blocks(builder, 3)
            return

        # chapter heading (hierarchy may be deliberately broken)
        title = chapter_title(rng, spec, index)
        level = messy_level(index) if spec.messy_headings else 1
        builder.h(title, level=level)
        section = title

        # ---- content budget ------------------------------------------------
        # A unit must fit inside one page, otherwise it spills onto a second page
        # and the document's page count exceeds the specification.  Reserve space
        # for the heavy elements first, then spend what is left on prose.
        budget = unit_capacity(spec, fill)
        reserved = len(title) + 40
        this_unit = element_at.get(index, [])
        for kind, payload in this_unit:
            if kind == "table":
                reserved += TABLE_CHAR_COST.get(payload, 320)
            else:
                reserved += 520
        if index in code_at:
            reserved += 380
        if index in formula_at:
            reserved += 140
        if index in footnote_at:
            reserved += 200
        if index in reference_at:
            reserved += 900
        if index in ad_at:
            reserved += 260

        sentence_cost = 175 if spec.language != S.LANG_EN else 205
        anchor_cost = 78

        # On a fact-bearing page the answer anchors take priority over filler
        # prose: they are what makes the document usable in the answer bank, and
        # a page that spends its whole budget on prose would push the anchors
        # onto a page that already has a heavy element and overflow it.
        room = max(0, budget - reserved)
        n_anchors = 0
        if emit_facts and room > 0:
            n_anchors = min(anchors_per_unit, int(room * 0.5) // anchor_cost)
        anchor_chars = n_anchors * anchor_cost

        # Prose is only ever *reduced* from the category profile, never increased:
        # the profile is already calibrated to fill roughly one page.
        sentences = profile.body_sentences
        prose_room = max(0, room - anchor_chars)
        if sentences * sentence_cost > prose_room:
            sentences = prose_room // sentence_cost

        # body prose
        subject = article_subject(spec)
        for k in range(sentences):
            builder.p(paragraph(rng, spec, subject, index + k, n=3))

        # a numbered clause list, which also makes links look like headings,
        # but only when there is room left after the prose.
        clause_cost = 3 * sentence_cost
        if spec.messy_headings and index % 3 == 0:
            builder.h(f"{index}.{index % 4 + 1} 补充说明", level=2)
        elif reserved + sentences * sentence_cost + anchor_chars + clause_cost <= budget * 1.12:
            items = [
                f"{index}.{k + 1} {prose(rng, spec, subject, index * 100 + k)}"
                for k in range(3)
            ]
            builder.bullets(items, ordered=True)

        # heavy elements allocated to this page, in assignment order
        for kind, payload in this_unit:
            if kind == "table":
                style = payload
                counters["table"] += 1
                caption = table_caption(spec, counters["table"], table_style_label(spec, style))
                # Table and figure anchors are always registered: there are only a
                # handful per document, and gating them on the fact-bearing units
                # would leave a document with a table but no table question.
                anchor_q = (
                    f"{caption} 中第一行数据项的取值是多少？"
                    if spec.language != S.LANG_EN
                    else f"What value does the first data row report in \"{caption}\"?"
                )
                build_table(builder, style, index, caption, anchor_q, None)
                continue

            if kind == "figure":
                counters["figure"] += 1
                art = payload
                fig_id = f"fig-{spec.doc_id}-{counters['figure']:02d}"
                caption = figure_caption(spec, counters["figure"], figure_label(spec, art, index))
                builder.figure(fig_id, caption, art=art)
                builder.anchor(
                    caption,
                    question=(
                        f"{caption} 展示的主要内容是什么？"
                        if spec.language != S.LANG_EN
                        else f"What does \"{caption}\" show?"
                    ),
                    answer=figure_label(spec, art, index),
                    section=section,
                    kind="figure",
                    figure_id=fig_id,
                )
                continue

            if kind == "formula_image":
                counters["figure"] += 1
                fig_id = f"fig-{spec.doc_id}-f{counters['figure']:02d}"
                caption = figure_caption(
                    spec, counters["figure"], "注意力计算公式（图片形式，不可抽取）"
                )
                builder.figure(
                    fig_id, caption, art="formula", width_ratio=0.62,
                    body=[FORMULAS[index % len(FORMULAS)]],
                )
                continue

        # extractable formula
        if index in formula_at:
            builder.formula(FORMULAS[index % len(FORMULAS)])

        # code block
        if index in code_at:
            builder.code(
                code_lines(rng, spec, index),
                caption=f"示例 {index}：{spec.doc_id} 调用示例",
            )

        # factual anchors on units selected to carry them
        if emit_facts:
            for j in range(n_anchors):
                label, value, evidence = spec_item(builder, index, j)
                builder.p(
                    f"{evidence}。该取值由第 {index} 组对照实验确定，"
                    f"并在后续版本的回归测试中保持稳定。"
                    if spec.language != S.LANG_EN
                    else f"{evidence}. This value was fixed by control run {index} and "
                         f"is held stable by the regression suite."
                )
                builder.anchor(
                    evidence,
                    question=(
                        f"{label} 的取值为多少？"
                        if spec.language != S.LANG_EN
                        else f"What is the value of {label}?"
                    ),
                    answer=value,
                    section=section,
                    answer_type="numeric",
                    difficulty=profile.difficulty,
                )

        # footnotes
        if index in footnote_at:
            builder.footnotes(footnote_items(rng, spec, index))

        # references
        if index in reference_at:
            builder.h("参考文献" if spec.language != S.LANG_EN else "References", level=1)
            builder.references(reference_items(rng, spec, index, count=8 if pages > 20 else 6))

        # mock advertisement
        if index in ad_at:
            builder.ad(ad_headline(spec, index), ad_lines(spec, index))

    return assemble_plan(
        spec, seed, unit, fact_units=14, refusals=refusal_count, min_questions=10,
    )


def table_style_label(spec: S.DocSpec, style: str) -> str:
    """Human caption for a table difficulty class."""
    labels = {
        S.TABLE_SIMPLE: ("分层参数对照", "Layered parameter comparison"),
        S.TABLE_MERGED: ("分部经营数据（含合并单元格）", "Segment results (merged cells)"),
        S.TABLE_MULTIPAGE: ("接口配额与延迟明细（跨页）", "Quota and latency detail (spans pages)"),
        S.TABLE_BORDERLESS: ("流程责任矩阵（无边框）", "Process responsibility matrix (borderless)"),
        S.TABLE_MULTIHEADER: ("区域季度数据（多级表头）", "Regional quarterly data (grouped header)"),
    }
    zh, en = labels[style]
    return en if spec.language == S.LANG_EN else zh


def figure_label(spec: S.DocSpec, art: str, index: int) -> str:
    """Human caption body for a figure kind."""
    labels = {
        "flow": ("数据摄取与检索流程图", "Ingestion and retrieval flow"),
        "bar": ("各方案延迟对比（毫秒）", "Latency comparison by configuration (ms)"),
        "pie": ("成本构成占比", "Cost composition"),
        "line": ("召回率随参数变化曲线", "Recall against parameter value"),
        "photo": ("现场观测照片", "Field observation photograph"),
        "screenshot": ("控制台调用截图", "Console invocation screenshot"),
        "formula": ("公式渲染图", "Rendered formula"),
    }
    zh, en = labels.get(art, ("示意图", "Illustration"))
    base = en if spec.language == S.LANG_EN else zh
    return f"{base}（第 {index} 组）" if spec.language != S.LANG_EN else f"{base} (set {index})"


def code_lines(rng: random.Random, spec: S.DocSpec, index: int) -> List[str]:
    """A short, plausible code sample."""
    token = f"{spec.doc_id.lower()}-{index:02d}"
    return [
        "from acme_client import Gateway",
        "",
        f"client = Gateway(api_key=os.environ['ACME_KEY'], region='cn-north-{index % 4 + 1}')",
        f"resp = client.query(index='{token}', top_k={5 + index % 12})",
        "for hit in resp.hits:",
        "    print(hit.id, round(hit.score, 4), hit.source)",
    ]


def ad_headline(spec: S.DocSpec, index: int) -> str:
    heads = (
        "广告 · 示例工业解决方案", "广告 · 示例云计算服务", "广告 · 示例储能设备",
    )
    return f"{heads[index % len(heads)]}"

def ad_lines(spec: S.DocSpec, index: int) -> List[str]:
    return [
        f"限时优惠：示例产品 {index} 系列，首年立减 {10 + index % 40}%。",
        "本广告与正文内容无关，用于测试检索系统对广告噪声的抑制能力。",
        f"咨询热线 400-000-{1000 + index}，网址 example.invalid/promo{index}。",
    ]

