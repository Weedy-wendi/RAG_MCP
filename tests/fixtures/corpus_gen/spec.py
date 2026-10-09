"""The authoritative 100-document specification.

This module is the single place where the corpus is defined.  Every other module
derives from it:

* ``content_*`` build a ``DocPlan`` for a :class:`DocSpec`;
* ``build_corpus`` iterates :func:`build_specs`;
* ``coverage`` asserts the realised corpus against :data:`THRESHOLDS`;
* ``verify_corpus`` checks it against the PDFs on disk.

Flag allocation is fixed here (not randomised) so that the corpus satisfies the
required coverage matrix for *any* ``--seed``; the seed only influences prose,
numbers and degradation parameters, never which document carries which property.

Layout of the allocation
------------------------
Counts               25 academic / 20 tech / 15 finance / 15 legal / 10 medical / 15 news
Non-native           15 scanned (image-only) + 8 OCR'd (image + invisible text layer)
Watermark            6            Complex tables  6
Version pairs        6 pairs (12 docs)
Two-column           24           Three-column    3        Vertical  2
One page             3            8-14 pages      79       40-60 pages  6      220 pages  1
Bookmarks            25           Messy headings  6        TOC          87
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

ACADEMIC = "ACAD"
TECH = "TECH"
FINANCE = "FIN"
LEGAL = "LEGAL"
MEDICAL = "MED"
NEWS = "NEWS"

CATEGORIES: Tuple[str, ...] = (ACADEMIC, TECH, FINANCE, LEGAL, MEDICAL, NEWS)

CATEGORY_NAMES: Dict[str, str] = {
    ACADEMIC: "学术论文",
    TECH: "技术手册",
    FINANCE: "财务商业报告",
    LEGAL: "法律政策文件",
    MEDICAL: "医疗生命科学",
    NEWS: "新闻杂志综合",
}

CATEGORY_COUNTS: Dict[str, int] = {
    ACADEMIC: 25,
    TECH: 20,
    FINANCE: 15,
    LEGAL: 15,
    MEDICAL: 10,
    NEWS: 15,
}

# Source families the documents imitate (never downloaded -- see README).
SOURCE_STYLES: Dict[str, Tuple[str, ...]] = {
    ACADEMIC: ("arXiv preprint", "PubMed Central", "CNKI 期刊论文"),
    TECH: ("GitHub Docs", "厂商用户手册", "API Reference", "Release Notes"),
    FINANCE: ("SEC Form 10-K", "巨潮资讯 年度报告", "券商行业研报", "招股说明书"),
    LEGAL: ("政府公报", "法律法规汇编", "合同范本", "政策白皮书"),
    MEDICAL: ("WHO 临床指南", "FDA 药品说明书", "NMPA 说明书", "科研综述"),
    NEWS: ("The Economist", "National Geographic", "维基百科条目", "深度调查报道"),
}

# ---------------------------------------------------------------------------
# Layouts / languages / extractability
# ---------------------------------------------------------------------------

SINGLE = "single"
TWO_COL = "two_column"
THREE_COL = "three_column"
VERTICAL = "vertical"

NATIVE = "native"
SCANNED = "scanned"
OCR = "ocr"

LANG_ZH = "zh"
LANG_EN = "en"
LANG_MIXED = "mixed"


def _ids(category: str, numbers: Sequence[int]) -> Set[str]:
    return {f"{category}-{n:02d}" for n in numbers}


def _range(category: str, first: int, last: int) -> Set[str]:
    return _ids(category, range(first, last + 1))


# ---------------------------------------------------------------------------
# Flag allocation  (fixed -- independent of --seed)
# ---------------------------------------------------------------------------

#: Image-only PDFs: no text layer at all.  Ingestion is *expected to fail*.
SCANNED_DOCS: Set[str] = (
    _ids(ACADEMIC, (21, 22, 23))
    | _ids(TECH, (18, 19))
    | _ids(FINANCE, (13, 14))
    | _ids(LEGAL, (11, 12, 13))
    | _ids(MEDICAL, (9, 10))
    | _ids(NEWS, (13, 14, 15))
)

#: Scanned page image + invisible (alpha 0) text layer carrying OCR-style errors.
OCR_DOCS: Set[str] = (
    _ids(ACADEMIC, (24,))
    | _ids(TECH, (20,))
    | _ids(FINANCE, (15,))
    | _ids(LEGAL, (14, 15))
    | _ids(MEDICAL, (8,))
    | _ids(NEWS, (11, 12))
)

NON_NATIVE: Set[str] = SCANNED_DOCS | OCR_DOCS

#: Diagonal translucent text watermark on every page.  Native/OCR only: on a
#: rasterised page a text watermark becomes pixels, which tests something else.
WATERMARKED: Set[str] = (
    {"ACAD-01"}
    | _ids(FINANCE, (1, 2))
    | {"LEGAL-01"}
    | {"MED-01"}
    | {"NEWS-01"}
)

#: "Tables especially complex": all four hard table styles in one document.
COMPLEX_TABLE_DOCS: Set[str] = (
    _ids(FINANCE, (3, 4, 5, 6)) | {"MED-02"} | {"TECH-05"}
)

#: (v1, v2) pairs.  Both native, identical layout and page count, differing only
#: in a handful of facts -- exercises dedup and recency handling.
VERSION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("ACAD-19", "ACAD-20"),
    ("TECH-03", "TECH-04"),
    ("FIN-07", "FIN-08"),
    ("LEGAL-08", "LEGAL-09"),
    ("MED-04", "MED-06"),
    ("NEWS-09", "NEWS-10"),
)
VERSION_V1: Set[str] = {a for a, _ in VERSION_PAIRS}
VERSION_V2: Set[str] = {b for _, b in VERSION_PAIRS}
VERSION_ALL: Set[str] = VERSION_V1 | VERSION_V2

#: Heading hierarchy deliberately broken (level skips, duplicate numbering,
#: numbered list items masquerading as headings).
MESSY_HEADINGS: Set[str] = (
    {"TECH-17", "LEGAL-07", "NEWS-08", "FIN-12", "ACAD-25", "MED-07"}
)

#: PDF outline entries (real bookmarks).
BOOKMARKED: Set[str] = _range(ACADEMIC, 1, 20) | _ids(TECH, (1, 2, 3, 4, 5))

#: True two-column frame flow.
TWO_COLUMN_DOCS: Set[str] = _range(ACADEMIC, 1, 24)

#: Three-column magazine flow.
THREE_COLUMN_DOCS: Set[str] = _ids(NEWS, (1, 2, 3))

#: Vertical writing (characters top-to-bottom, columns right-to-left).
VERTICAL_DOCS: Set[str] = {"LEGAL-10", "NEWS-15"}

#: Mock advertisement blocks inside news/magazine documents.
AD_DOCS: Set[str] = _range(NEWS, 1, 10)

# ---------------------------------------------------------------------------
# Lengths
# ---------------------------------------------------------------------------

PAGE_ONE: Set[str] = {"NEWS-01", "NEWS-02", "LEGAL-01"}
PAGE_SHORT: Set[str] = (
    _ids(NEWS, (4, 5, 6, 7)) | _ids(LEGAL, (2, 3)) | {"TECH-11"} | _ids(FINANCE, (9, 10)) | {"MED-03"}
)

#: 40-60 page documents, explicit so the manifest is seed-independent.
PAGE_FIFTY: Dict[str, int] = {
    "ACAD-19": 48,
    "ACAD-20": 48,
    "TECH-10": 52,
    "FIN-03": 56,
    "LEGAL-05": 44,
    "MED-05": 50,
}

#: One genuinely huge document (220 pages of procedurally-unique API reference).
PAGE_HUGE: Dict[str, int] = {"TECH-16": 220}


#: Version-pair members must be identical in layout *and length* so that a
#: difference question can only be answered from content.  Both members therefore
#: derive their page count from the v1 member's position.
_PAIR_CANONICAL_SEQ: Dict[str, int] = {}
for _v1, _v2 in VERSION_PAIRS:
    _n = int(_v1.split("-")[1])
    _PAIR_CANONICAL_SEQ[_v1] = _n
    _PAIR_CANONICAL_SEQ[_v2] = _n


#: Documents required to carry all four hard table classes plus figures need
#: enough pages to place them one per page; otherwise a page would have to hold a
#: 34-row table *and* a figure and would overflow regardless of prose length.
COMPLEX_TABLE_MIN_PAGES = 16


def page_count(doc_id: str, category: str, seq: int) -> int:
    """Deterministic page count for a document (seed-independent)."""
    if doc_id in PAGE_ONE:
        return 1
    if doc_id in PAGE_HUGE:
        return PAGE_HUGE[doc_id]
    if doc_id in PAGE_FIFTY:
        return PAGE_FIFTY[doc_id]
    seq = _PAIR_CANONICAL_SEQ.get(doc_id, seq)
    if doc_id in PAGE_SHORT:
        return 2 + (seq % 2)  # 2 or 3
    pages = 8 + ((seq * 5 + len(category)) % 7)  # 8..14
    if doc_id in COMPLEX_TABLE_DOCS:
        pages = max(pages, COMPLEX_TABLE_MIN_PAGES)
    return pages


# ---------------------------------------------------------------------------
# Language allocation
# ---------------------------------------------------------------------------

#: Documents whose primary language is English.
LANG_EN_DOCS: Set[str] = (
    _range(ACADEMIC, 1, 18)
    | _ids(ACADEMIC, (21, 22, 23))
    | _range(TECH, 1, 10)
    | {"TECH-16"}
    | _ids(FINANCE, (1, 3, 4, 13, 14))
    | _range(NEWS, 1, 8)
    | _ids(NEWS, (13, 14))
)

#: Documents that deliberately interleave Chinese and English.  Version-pair
#: members always share a language, so both members of a pair appear here or
#: neither does.
LANG_MIXED_DOCS: Set[str] = {
    "ACAD-19", "ACAD-20",
    "TECH-12", "TECH-13",
    "FIN-02", "FIN-07", "FIN-08",
    "LEGAL-14", "LEGAL-15",
    "MED-03", "MED-04", "MED-06", "MED-08",
    "NEWS-09", "NEWS-10",
}

#: Everything else is Chinese-primary.
def language_of(doc_id: str) -> str:
    if doc_id in LANG_EN_DOCS:
        return LANG_EN
    if doc_id in LANG_MIXED_DOCS:
        return LANG_MIXED
    return LANG_ZH


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

#: Documents that intentionally contain NO authored figure (blank-figure case).
NO_FIGURE_DOCS: Set[str] = (
    _range(LEGAL, 1, 10)
    | _ids(FINANCE, (5, 6, 11, 12))
    | _ids(TECH, (11, 14, 15, 17))
    | {"ACAD-25", "MED-04"}
)

#: Documents carrying a formula rendered as an image (NOT text-extractable).
FORMULA_IMAGE_DOCS: Set[str] = _ids(ACADEMIC, (1, 2, 3, 4, 5, 6, 19, 20, 24))

# ---------------------------------------------------------------------------
# Table styles
# ---------------------------------------------------------------------------

TABLE_SIMPLE = "simple"
TABLE_MERGED = "merged"
TABLE_MULTIPAGE = "multipage"
TABLE_BORDERLESS = "borderless"
TABLE_MULTIHEADER = "multiheader"

#: Cover-page metadata only.  Deliberately outside the difficulty classes so it
#: never contributes to the table coverage metrics.
TABLE_INFO = "info"

TABLE_STYLES: Tuple[str, ...] = (
    TABLE_SIMPLE, TABLE_MERGED, TABLE_MULTIPAGE, TABLE_BORDERLESS, TABLE_MULTIHEADER,
)

#: The styles that count as a genuine table-difficulty witness.
TABLE_DIFFICULTY_STYLES: Tuple[str, ...] = TABLE_STYLES

_COMPLEX_SET = (TABLE_MERGED, TABLE_MULTIPAGE, TABLE_MULTIHEADER, TABLE_BORDERLESS)
_FIN_ROTATION = (
    (TABLE_MERGED, TABLE_MULTIPAGE),
    (TABLE_MULTIPAGE, TABLE_BORDERLESS),
    (TABLE_MERGED, TABLE_MULTIHEADER, TABLE_MULTIPAGE),
    (TABLE_BORDERLESS, TABLE_SIMPLE),
    (TABLE_MULTIHEADER, TABLE_MULTIPAGE),
)


def table_styles_for(doc_id: str, category: str, seq: int) -> Tuple[str, ...]:
    """Which table difficulty classes this document must contain.

    Two-column papers only get narrow tables: a five-column table inside a
    half-width frame would be unreadable, and the corpus is meant to be realistic
    as well as hard.
    """
    if doc_id in COMPLEX_TABLE_DOCS:
        return _COMPLEX_SET
    if category == FINANCE:
        return _FIN_ROTATION[seq % len(_FIN_ROTATION)]
    if category == TECH:
        # Manuals and API references always carry a long, page-spanning table.
        return (TABLE_SIMPLE, TABLE_MULTIPAGE, TABLE_MULTIHEADER)
    if category == MEDICAL:
        return (TABLE_MERGED, TABLE_MULTIPAGE) if seq % 2 == 0 else (TABLE_SIMPLE, TABLE_MERGED)
    if category == ACADEMIC:
        # ACAD-25 is the single-column survey and can host a wide table.
        if doc_id in TWO_COLUMN_DOCS:
            return (TABLE_SIMPLE,)
        return (TABLE_SIMPLE, TABLE_MULTIPAGE)
    if category == LEGAL:
        return (TABLE_SIMPLE,) if seq % 4 == 0 else ()
    if category == NEWS:
        return (TABLE_SIMPLE,) if seq % 3 == 0 else ()
    return (TABLE_SIMPLE,)


# ---------------------------------------------------------------------------
# Scan degradation profiles
# ---------------------------------------------------------------------------

#: (profile name, dpi, blur radius, skew degrees, noise sigma, jpeg quality)
SCAN_PROFILES: Tuple[Tuple[str, int, float, float, float, int], ...] = (
    ("clean", 150, 0.0, 0.0, 0.0, 88),
    ("blur", 150, 1.4, 0.0, 4.0, 80),
    ("skew", 150, 0.4, 1.4, 3.0, 78),
    ("noisy", 120, 0.6, 0.6, 11.0, 62),
    ("lowq_jpeg", 100, 0.9, 1.1, 7.0, 34),
    ("aged", 150, 0.7, -0.9, 6.0, 70),
)


def scan_profile_for(doc_id: str, seq: int) -> Tuple[str, int, float, float, float, int]:
    """Pick a deterministic degradation profile for a scanned/OCR document."""
    return SCAN_PROFILES[seq % len(SCAN_PROFILES)]


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

TOPICS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    ACADEMIC: (
        ("Retrieval-Augmented Generation for Long-Context Question Answering", "长上下文问答中的检索增强生成"),
        ("Sparse-Dense Hybrid Retrieval with Reciprocal Rank Fusion", "稀疏-稠密混合检索与倒数排名融合"),
        ("Cross-Encoder Reranking under Latency Constraints", "延迟约束下的交叉编码器重排序"),
        ("Vision-Language Models for Scientific Figure Understanding", "面向科学图表的视觉语言模型"),
        ("Chunking Strategies and Their Effect on Retrieval Recall", "分块策略对检索召回率的影响"),
        ("Metadata Enrichment for Domain-Specific Retrieval", "面向垂直领域的元数据增强检索"),
        ("Evaluating Faithfulness in Grounded Generation", "基于证据生成的忠实度评估"),
        ("Semi-Supervised Learning for Low-Resource Biomedical NER", "低资源生物医学命名的半监督学习"),
        ("Graph Neural Networks for Drug-Target Interaction Prediction", "药物-靶点相互作用的图神经网络预测"),
        ("Single-Cell Transcriptomics Clustering via Variational Autoencoders", "变分自编码器的单细胞转录组聚类"),
        ("Differential Privacy in Clinical Text Mining", "临床文本挖掘中的差分隐私"),
        ("CRISPR Off-Target Prediction with Attention Models", "注意力模型的 CRISPR 脱靶预测"),
        ("Knowledge Distillation for Edge-Deployed Transformers", "面向边缘部署的 Transformer 知识蒸馏"),
        ("Long-Context Attention Approximation with Linear Kernels", "线性核长上下文注意力近似"),
        ("Instruction Tuning Data Quality over Quantity", "指令微调数据的质量优于数量"),
        ("Hallucination Detection via Self-Consistency Sampling", "自一致性采样的幻觉检测"),
        ("Multimodal Retrieval over Technical Diagrams", "技术图表的跨模态检索"),
        ("Benchmark Contamination in Retrieval Evaluation", "检索评测中的基准污染"),
        ("A Survey of Retrieval-Augmented Generation Architectures", "检索增强生成架构综述"),
        ("Retrieval-Augmented Generation: A Comprehensive Survey", "检索增强生成：综合评述"),
        ("Efficient Fine-Tuning of Medical Language Models", "医学语言模型的高效微调"),
        ("Automated Segmentation of Histopathology Slides", "组织病理切片的自动分割"),
        ("Proteomic Biomarker Discovery via Mass Spectrometry", "质谱蛋白质组学生物标志物发现"),
        ("Biomedical Ontology Alignment with Contrastive Learning", "对比学习的生物医学本体对齐"),
        ("Retrieval Augmentation in Clinical Decision Support", "临床决策支持中的检索增强"),
    ),
    TECH: (
        ("Acme Gateway API Reference", "Acme 网关 API 参考"),
        ("Acme Gateway Authentication Guide", "Acme 网关认证指南"),
        ("Acme Gateway API Reference (v2)", "Acme 网关 API 参考（v2）"),
        ("Acme Gateway API Reference (v2.1)", "Acme 网关 API 参考（v2.1）"),
        ("Acme Storage Adapter Configuration", "Acme 存储适配器配置手册"),
        ("Acme Vector Store Integration Guide", "Acme 向量存储集成指南"),
        ("Acme Observability Pipeline Setup", "Acme 可观测性管道搭建"),
        ("Acme CLI Command Reference", "Acme 命令行参考手册"),
        ("Acme Deployment and Upgrade Manual", "Acme 部署与升级手册"),
        ("Acme Platform Administrator Guide", "Acme 平台管理员手册"),
        ("Acme 数据摄取运维手册", "Acme 数据摄取运维手册"),
        ("Acme 检索服务调优指南", "Acme 检索服务调优指南"),
        ("Acme 缓存层双语配置说明", "Acme 缓存层双语配置说明"),
        ("Acme SDK 双语文档", "Acme SDK 双语文档"),
        ("Acme 日志采集接入说明", "Acme 日志采集接入说明"),
        ("Acme Platform Complete Reference", "Acme 平台完整参考"),
        ("Acme 插件开发手册（结构混乱版）", "Acme 插件开发手册（结构混乱版）"),
        ("Acme 迁移工具说明（结构混乱版）", "Acme 迁移工具说明（结构混乱版）"),
        ("Acme 边缘节点扫描手册", "Acme 边缘节点扫描手册"),
        ("Acme 遗留系统 OCR 手册", "Acme 遗留系统 OCR 手册"),
    ),
    FINANCE: (
        ("Meridian Industrial Holdings 2025 Annual Report", "Meridian 工业控股 2025 年度报告"),
        ("Meridian Industrial Holdings 2025 Annual Report (Bilingual)", "Meridian 工业控股 2025 年度报告（中英对照）"),
        ("Northwind Logistics 2025 Form 10-K", "Northwind 物流 2025 年 10-K 报告"),
        ("Northwind Logistics 2025 Form 10-K (Revised)", "Northwind 物流 2025 年 10-K 报告（修订版）"),
        ("Cobalt Semiconductor Industry Review", "Cobalt 半导体行业研究"),
        ("Cobalt Semiconductor Prospectus Extract", "Cobalt 半导体招股书节选"),
        ("Vantage Retail Group FY2025 Results", "Vantage 零售集团 2025 财年业绩"),
        ("Vantage Retail Group FY2025 Results (Corrected)", "Vantage 零售集团 2025 财年业绩（更正版）"),
        ("示例新能源股份有限公司 2025 年年度报告", "示例新能源 2025 年年度报告"),
        ("示例医药集团 2025 年度报告摘要", "示例医药集团 2025 年度报告摘要"),
        ("示例消费行业季度研报", "示例消费行业季度研报"),
        ("示例建材行业研究报告（结构混乱版）", "示例建材行业研究报告（结构混乱版）"),
        ("Meridian 2024 Annual Report (Scanned)", "Meridian 2024 年度报告（扫描件）"),
        ("Northwind 2023 Form 10-K (Scanned)", "Northwind 2023 年 10-K（扫描件）"),
        ("示例物流集团 2024 年报（OCR 版）", "示例物流集团 2024 年报（OCR 版）"),
    ),
    LEGAL: (
        ("示例数据安全管理办法", "示例数据安全管理办法"),
        ("示例个人信息处理告知书", "示例个人信息处理告知书"),
        ("示例技术服务合同范本", "示例技术服务合同范本"),
        ("示例保密协议范本", "示例保密协议范本"),
        ("示例数据交易管理条例汇编", "示例数据交易管理条例汇编"),
        ("示例算法推荐管理规定", "示例算法推荐管理规定"),
        ("示例平台责任政策白皮书（结构混乱版）", "示例平台责任政策白皮书（结构混乱版）"),
        ("示例人工智能治理条例（征求意见稿）", "示例人工智能治理条例（征求意见稿）"),
        ("示例人工智能治理条例（正式稿）", "示例人工智能治理条例（正式稿）"),
        ("古籍影印·示例条例（竖排）", "古籍影印·示例条例（竖排）"),
        ("示例跨境数据流动规定（扫描件）", "示例跨境数据流动规定（扫描件）"),
        ("示例网络安全处罚办法（扫描件）", "示例网络安全处罚办法（扫描件）"),
        ("示例商业秘密保护指引（扫描件）", "示例商业秘密保护指引（扫描件）"),
        ("示例数据出境安全评估办法（中英双语·OCR）", "示例数据出境安全评估办法（中英双语·OCR）"),
        ("示例人工智能伦理准则（中英双语·OCR）", "示例人工智能伦理准则（中英双语·OCR）"),
    ),
    MEDICAL: (
        ("示例高血压诊疗指南（带水印）", "示例高血压诊疗指南（带水印）"),
        ("示例抗生素用药剂量手册", "示例抗生素用药剂量手册"),
        ("示例心血管药物说明书（中英对照）", "示例心血管药物说明书（中英对照）"),
        ("示例糖尿病管理指南 v1", "示例糖尿病管理指南 v1"),
        ("示例慢性肾病临床指南", "示例慢性肾病临床指南"),
        ("示例糖尿病管理指南 v2", "示例糖尿病管理指南 v2"),
        ("示例肿瘤标志物解读（结构混乱版）", "示例肿瘤标志物解读（结构混乱版）"),
        ("示例疫苗说明书（中英对照·OCR）", "示例疫苗说明书（中英对照·OCR）"),
        ("示例呼吸系统疾病指南（扫描件）", "示例呼吸系统疾病指南（扫描件）"),
        ("示例儿科用药手册（扫描件）", "示例儿科用药手册（扫描件）"),
    ),
    NEWS: (
        ("The Quiet Revolution in Industrial Robotics", "工业机器人领域的静默革命"),
        ("Deep Sea Mining: The Next Frontier", "深海采矿：下一个前沿"),
        ("Who Owns the Weather?", "谁拥有天气？"),
        ("The Economics of Urban Farming", "都市农业的经济学"),
        ("Inside the Rare Earth Supply Chain", "稀土供应链内幕"),
        ("The Return of the Night Train", "夜行列车的回归"),
        ("Water Rights in the American West", "美国西部的用水权"),
        ("The Rise of Portable Nuclear Power", "便携式核电的兴起"),
        ("Grid Storage: A Special Report", "电网储能专题报道"),
        ("Grid Storage: A Special Report (Corrected)", "电网储能专题报道（勘误版）"),
        ("示例城市轨道交通发展综述", "示例城市轨道交通发展综述"),
        ("示例半导体产业链观察", "示例半导体产业链观察"),
        ("The Last Ice Road (Scanned)", "最后的冰路（扫描件）"),
        ("Ports of the Future (Scanned)", "未来的港口（扫描件）"),
        ("古籍影印·示例日报（竖排·扫描件）", "古籍影印·示例日报（竖排·扫描件）"),
    ),
}

# ---------------------------------------------------------------------------
# DocSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocSpec:
    """Everything the generator needs to know about one document."""

    doc_id: str
    category: str
    seq: int
    title: str
    title_alt: str
    source_style: str
    language: str
    layout: str
    extractability: str
    pages: int
    table_styles: Tuple[str, ...]
    has_figures: bool
    formula_images: int
    watermark: bool
    ads: int
    messy_headings: bool
    bookmarks: bool
    has_toc: bool
    version_label: str = ""
    version_pair: str = ""
    scan_profile: str = "clean"
    scan_dpi: int = 150
    scan_blur: float = 0.0
    scan_skew: float = 0.0
    scan_noise: float = 0.0
    scan_jpeg: int = 90
    tags: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_scanned(self) -> bool:
        return self.extractability == SCANNED

    @property
    def is_ocr(self) -> bool:
        return self.extractability == OCR

    @property
    def is_native(self) -> bool:
        return self.extractability == NATIVE

    @property
    def extractable(self) -> bool:
        """True when a text layer exists (native or OCR)."""
        return self.extractability in (NATIVE, OCR)

    @property
    def columns(self) -> int:
        return {SINGLE: 1, TWO_COL: 2, THREE_COL: 3, VERTICAL: 1}[self.layout]

    @property
    def anchors_verifiable(self) -> bool:
        """Whether a question's anchor can be asserted against extracted text.

        False for two document classes, for different reasons:

        * **scanned** -- there is no text layer at all, so nothing is extractable;
        * **vertical** -- characters are drawn top-to-bottom in right-to-left
          columns, and PDF text extraction reads left-to-right, so the recovered
          reading order does not match the drawing order (measured trigram
          overlap for these documents is only 0.25-0.55).

        Anchors are still recorded for these documents; they are simply reported
        rather than asserted.
        """
        return self.extractability != SCANNED and self.layout != VERTICAL

    def filename(self, extension: str = ".pdf") -> str:
        suffix = f"_{self.version_label}" if self.version_label else ""
        return f"{self.doc_id}{suffix}{extension}"

    def relpath(self) -> str:
        """POSIX-style path relative to the corpus ``pdf/`` root."""
        return f"pdf/{self.category}/{self.filename()}"

    def fact_prefix(self) -> str:
        return self.doc_id

    def tags_set(self) -> Set[str]:
        out = set(self.tags)
        out.add(f"lang:{self.language}")
        out.add(f"layout:{self.layout}")
        out.add(f"extract:{self.extractability}")
        if self.watermark:
            out.add("watermark")
        if self.messy_headings:
            out.add("messy_headings")
        if self.bookmarks:
            out.add("bookmarks")
        if self.doc_id in VERSION_ALL:
            out.add("version_pair")
        if self.doc_id in COMPLEX_TABLE_DOCS:
            out.add("complex_tables")
        return out


# ---------------------------------------------------------------------------
# Building the 100 specs
# ---------------------------------------------------------------------------


def _assign_extractability(doc_id: str) -> str:
    if doc_id in SCANNED_DOCS:
        return SCANNED
    if doc_id in OCR_DOCS:
        return OCR
    return NATIVE


def _assign_layout(doc_id: str) -> str:
    if doc_id in VERTICAL_DOCS:
        return VERTICAL
    if doc_id in TWO_COLUMN_DOCS:
        return TWO_COL
    if doc_id in THREE_COLUMN_DOCS:
        return THREE_COL
    return SINGLE


def _version_label(doc_id: str) -> Tuple[str, str]:
    for v1, v2 in VERSION_PAIRS:
        if doc_id == v1:
            return "v1", v2
        if doc_id == v2:
            return "v2", v1
    return "", ""


def build_specs() -> List[DocSpec]:
    """Build all 100 :class:`DocSpec` objects in deterministic order."""
    specs: List[DocSpec] = []
    for category in CATEGORIES:
        topics = TOPICS[category]
        count = CATEGORY_COUNTS[category]
        if len(topics) < count:
            raise AssertionError(f"{category}: need {count} topics, have {len(topics)}")
        styles = SOURCE_STYLES[category]

        for seq in range(1, count + 1):
            doc_id = f"{category}-{seq:02d}"
            title_en, title_zh = topics[seq - 1]
            language = language_of(doc_id)
            if language == LANG_EN:
                title, title_alt = title_en, title_zh
            elif language == LANG_ZH:
                title, title_alt = title_zh, title_en
            else:
                title, title_alt = title_zh, title_en

            extractability = _assign_extractability(doc_id)
            pages = page_count(doc_id, category, seq)

            profile_name, dpi, blur, skew, noise, jpeg = ("clean", 150, 0.0, 0.0, 0.0, 90)
            if extractability in (SCANNED, OCR):
                profile_name, dpi, blur, skew, noise, jpeg = scan_profile_for(doc_id, seq)

            version_label, version_pair = _version_label(doc_id)

            specs.append(
                DocSpec(
                    doc_id=doc_id,
                    category=category,
                    seq=seq,
                    title=title,
                    title_alt=title_alt,
                    source_style=styles[seq % len(styles)],
                    language=language,
                    layout=_assign_layout(doc_id),
                    extractability=extractability,
                    pages=pages,
                    table_styles=table_styles_for(doc_id, category, seq),
                    has_figures=doc_id not in NO_FIGURE_DOCS,
                    formula_images=2 if doc_id in FORMULA_IMAGE_DOCS else 0,
                    watermark=doc_id in WATERMARKED,
                    ads=(seq % 3) + 1 if doc_id in AD_DOCS else 0,
                    messy_headings=doc_id in MESSY_HEADINGS,
                    bookmarks=doc_id in BOOKMARKED,
                    has_toc=pages >= 8,
                    version_label=version_label,
                    version_pair=version_pair,
                    scan_profile=profile_name if extractability in (SCANNED, OCR) else "clean",
                    scan_dpi=dpi,
                    scan_blur=blur,
                    scan_skew=skew,
                    scan_noise=noise,
                    scan_jpeg=jpeg,
                )
            )
    return specs


SPECS: List[DocSpec] = build_specs()
SPEC_BY_ID: Dict[str, DocSpec] = {spec.doc_id: spec for spec in SPECS}
ALL_DOC_IDS: Tuple[str, ...] = tuple(spec.doc_id for spec in SPECS)


# ---------------------------------------------------------------------------
# Coverage thresholds asserted by coverage.py / verify_corpus.py / unit tests
# ---------------------------------------------------------------------------

#: ``metric name -> (minimum, maximum or None, human description)``.
#: Counts are over the subset named in the third element where relevant.
THRESHOLDS: Dict[str, Tuple[int, int | None, str]] = {
    # categories
    "category.ACAD": (25, 25, "学术论文篇数"),
    "category.TECH": (20, 20, "技术手册篇数"),
    "category.FIN": (15, 15, "财务报告篇数"),
    "category.LEGAL": (15, 15, "法律文件篇数"),
    "category.MED": (10, 10, "医疗文档篇数"),
    "category.NEWS": (15, 15, "新闻杂志篇数"),
    "total": (100, 100, "文档总数"),
    # extractability
    "extract.native": (77, 77, "原生可抽取 PDF"),
    "extract.scanned": (15, 15, "扫描件（无文本层）"),
    "extract.ocr": (8, 8, "OCR 后（隐形文本层）"),
    # quality / dirty data
    "dirty.watermark": (5, None, "带水印文档"),
    "dirty.complex_tables": (5, None, "表格特别复杂文档"),
    "dirty.messy_headings": (6, None, "层级混乱文档"),
    "dirty.ads": (6, None, "含广告干扰文档"),
    "dirty.footnotes": (10, None, "含脚注文档"),
    # layout
    "layout.two_column": (20, None, "双栏文档"),
    "layout.three_column": (3, None, "三栏文档"),
    "layout.vertical": (2, None, "竖排文档"),
    # language (over extractable docs only)
    "lang.zh": (30, None, "中文为主（可抽取）"),
    "lang.en": (30, None, "英文为主（可抽取）"),
    "lang.mixed": (15, None, "中英混排（可抽取）"),
    # tables
    "table.simple": (40, None, "含简单表文档"),
    "table.merged": (10, None, "含合并单元格表文档"),
    "table.multipage": (10, None, "含跨页表文档"),
    "table.borderless": (5, None, "含无边框表文档"),
    "table.multiheader": (5, None, "含多级表头文档"),
    # figures
    "figure.flow": (25, None, "含流程图文档"),
    "figure.chart": (20, None, "含图表（柱/饼/折线）文档"),
    "figure.photo": (10, None, "含照片文档"),
    "figure.formula_image": (8, None, "含公式截图文档"),
    "figure.none": (20, None, "无插图文档（可抽取）"),
    # length
    "length.one_page": (3, None, "1 页文档"),
    "length.short": (10, None, "2-3 页文档"),
    "length.mid": (25, None, "8-14 页文档"),
    "length.fifty": (6, None, "40-60 页文档"),
    "length.huge": (1, None, "200 页以上文档"),
    # structure
    "structure.toc": (30, None, "含目录文档"),
    "structure.bookmarks": (20, None, "含 PDF 书签文档"),
    "structure.version_pairs": (6, 6, "版本对数量"),
    "structure.references": (25, None, "含参考文献文档"),
    # answer bank
    "qa.per_doc_min": (10, None, "每篇最少题目数"),
    "qa.per_doc_max": (10, 20, "每篇题目数上限"),
    "qa.refusal_per_doc": (1, None, "每篇最少拒答题数"),
    "qa.total": (1000, None, "标准答案总题数"),
}


def threshold(name: str) -> Tuple[int, int | None]:
    """Return ``(minimum, maximum)`` for a metric."""
    low, high, _ = THRESHOLDS[name]
    return low, high


def describe_thresholds() -> List[str]:
    """Render the threshold table for the coverage report."""
    return [f"{name}: {low}..{high if high is not None else '∞'}  ({desc})"
            for name, (low, high, desc) in THRESHOLDS.items()]
