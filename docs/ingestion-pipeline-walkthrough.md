# 摄取链路（Ingestion Pipeline）逐行拆解

> 目标：把 `PDF 文件 → 可检索的知识库` 这条链路从入口到落库，一行一行讲清楚。
> 面向有前端基础、刚接触 RAG 的同学，每个关键决策都会说明「为什么这么写」。

---

## 0. 阅读地图：这一条链路涉及哪些文件

按数据流动的顺序读，不要按目录顺序读。

| 顺序 | 文件 | 职责（一句话） |
|---|---|---|
| 1 | `src/core/types.py` | 定义 Document / Chunk / ChunkRecord，全链路的数据契约 |
| 2 | `scripts/ingest.py` | 命令行入口，收文件路径参数 |
| 3 | `src/ingestion/pipeline.py` | **总指挥**，串联六个阶段 |
| 4 | `src/libs/loader/file_integrity.py` | 阶段1：SHA256 判重，已处理过就跳过 |
| 5 | `src/libs/loader/pdf_loader.py` | 阶段2：PDF 解析 + 抽图 |
| 6 | `src/ingestion/chunking/document_chunker.py` | 阶段3：切块 + 生成 ID + 元数据继承 |
| 7 | `src/ingestion/transform/*.py` | 阶段4：精炼 / 元数据增强 / 图片描述 |
| 8 | `src/ingestion/embedding/*.py` | 阶段5：向量化 + BM25 词频统计 |
| 9 | `src/ingestion/storage/*.py` | 阶段6：写入 ChromaDB / BM25 索引 / 图片索引 |

一句话概括这条链路的本质：

> **把「一个 PDF 文件」拆成「一堆带向量和元数据的文本片段」，并写进三种存储，让后面的检索能查到。**

---

## 1. 数据契约：先看三个数据结构（`src/core/types.py`）

RAG 系统最容易混乱的地方就是数据结构。这个项目先把契约定死，全链路复用同一套类型——**类似前端里先定好 TS interface，再写业务**。

### 1.1 `Document`（L19-L78）：一个原始文档

```python
@dataclass
class Document:
    id: str                              # 文档唯一 ID（这里是 doc_{hash前16位}）
    text: str                            # 标准化后的 Markdown 文本
    metadata: Dict[str, Any] = field(default_factory=dict)
```

逐个看：

- **`id`**：文档的唯一标识。注意它的生成规则在 `PdfLoader` 里，是 **文件内容的 SHA256**，所以「同一个文件重复摄取」会得到同一个 ID——这是幂等的基础。
- **`text`**：**注意不是原始 PDF 文本，而是 Markdown**。因为中间经过 markitdown 转换，表格、标题都会变成 Markdown 语法。图片不会真插进文本，而是留一个占位符 `[IMAGE: {image_id}]`（L28 注释写得很清楚）。
- **`metadata`**：必须包含 `source_path`，否则构造时就抛错：

```python
def __post_init__(self):                    # L66
    if "source_path" not in self.metadata:
        raise ValueError("Document metadata must contain 'source_path'")
```

  `__post_init__` 是 dataclass 的钩子，相当于构造函数末尾的校验。**这类"契约校验"是后面所有阶段能放心用 metadata 的原因**。

- **图片元数据的结构**（L37-L48 注释）：每个图片是一个 dict，含 `id / path / page / text_offset / text_length / position`。`text_offset` 记录占位符在正文中的字符位置——这样即使同一张图在文中出现多次，也能定位。

### 1.2 `Chunk`（L81-L140）：切块后的片段

比 Document 多了三个字段：

```python
start_offset: Optional[int] = None      # 在原文中的起始字符位置
end_offset: Optional[int] = None        # 结束位置
source_ref: Optional[str] = None        # 指向父文档 ID
```

`source_ref` 是**溯源链**：检索命中一个片段后，能顺着它找到是哪个文档来的。前端类比：类似 React 组件树里的 parent ref，或者一条记录的 `foreign key`。

`metadata` 里还约定要有 `chunk_index`（在文档里的顺序号）。

### 1.3 `ChunkRecord`（L143-L221）：准备入库的完整记录

```python
dense_vector: Optional[List[float]] = None       # 稠密向量（1536维）
sparse_vector: Optional[Dict[str, float]] = None # 稀疏向量（词→权重）
```

它是 `Chunk` + 向量。还有个工厂方法：

```python
@classmethod
def from_chunk(cls, chunk, dense_vector=None, sparse_vector=None):   # L203
    return cls(
        id=chunk.id,
        text=chunk.text,
        metadata=chunk.metadata.copy(),   # 注意是 copy，不共享引用
        dense_vector=dense_vector,
        sparse_vector=sparse_vector
    )
```

**`metadata.copy()` 这一行值得注意**：多个 Chunk 如果共享同一个 metadata 字典对象，任何一个 transform 改了它，会"隔空"影响其他 chunk。这种浅拷贝是廉价的防御。

> ⚠️ 但要注意：`copy()` 是**浅拷贝**。metadata 里的 `images` 是 list，仍然是共享引用。全链路没有深拷贝，所以 transform 里写 `chunk.metadata["image_captions"].extend(...)` 是有共享风险的（见阶段4）。

---

## 2. 总指挥：`IngestionPipeline`（`src/ingestion/pipeline.py`）

### 2.1 构造函数：把六个阶段的零件全部装配好（L120-L195）

```python
def __init__(self, settings, collection: str = "default", force: bool = False):
    self.settings = settings
    self.collection = collection      # 集合名 = 逻辑分组，类似"表名"
    self.force = force                # 强制重跑，忽略判重
```

然后是一串"装配零件"的代码，注意它的组织方式：

```python
# Stage 1: File Integrity
self.integrity_checker = SQLiteIntegrityChecker(
    db_path=str(resolve_path("data/db/ingestion_history.db"))
)
logger.info("  ✓ FileIntegrityChecker initialized")
```

这段代码有两个值得学的点：

1. **`resolve_path()`**：把 `data/db/...` 这种相对路径解析成"以项目根为锚点"的绝对路径。因为脚本可能在任意目录被调用（`python scripts/ingest.py` / `python -m ...`），cwd 不一定是项目根。**路径锚定是工程化项目的最低要求**。
2. **装配完就 log 一行**：这不是废话日志。流水线有 6 阶段 14 个零件，某一环装配失败时，日志能立刻告诉你是哪一环。

装配顺序严格按数据流：判重 → 加载 → 切块 → 三个 transform → 两个编码器 → 批处理器 → 三个存储。

注意几个细节：

```python
embedding = EmbeddingFactory.create(settings)          # L167
batch_size = settings.ingestion.batch_size if settings.ingestion else 100
self.dense_encoder = DenseEncoder(embedding, batch_size=batch_size)
```

- **通过工厂拿后端**：`EmbeddingFactory.create(settings)` 内部读 `settings.yaml` 的 `embedding.provider` 决定用 OpenAI / Azure / Ollama。Pipeline 完全不关心是谁。
- **`DenseEncoder` 是依赖注入**：它接收一个 `BaseEmbedding` 实例，自己不调工厂。这样测试时可以塞一个假的 embedding 进去。

```python
self.image_captioner = ImageCaptioner(settings)
has_vision = self.image_captioner.llm is not None
logger.info(f"  ✓ ImageCaptioner initialized (vision_enabled={has_vision})")
```

- **Vision LLM 初始化失败不抛异常**，只是 `llm` 为 None。后面 `transform()` 看到 `llm` 为空就直接原样返回。**这是"降级不中断"原则的第一次出现**，整条链路会反复用到。

```python
self.bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
```

- **BM25 索引按 collection 分目录**。因为 BM25 的 IDF 依赖整个语料库的统计量，不同 collection 的语料必须分开。

### 2.2 `run()`：六阶段的骨架（L197-L227）

```python
def run(self, file_path, trace=None, on_progress=None) -> PipelineResult:
    file_path = Path(file_path)
    stages: Dict[str, Any] = {}
    _total_stages = 6
```

三个参数各有用途：

- `file_path`：要处理的文件
- `trace`：可观测性上下文，每阶段往里记录细节（后面会看到大量 `trace.record_stage(...)`）
- `on_progress`：进度回调。**Dashboard 的实时进度条就是靠它**——注意 callback 签名是 `(stage_name, current, total)`。

```python
    def _notify(stage_name: str, step: int) -> None:
        if on_progress is not None:
            on_progress(stage_name, step, _total_stages)
```

一个 3 行的闭包，把"当前是第几阶段/共几个阶段"封起来，避免每个阶段都写一遍 `if on_progress`。**这就是为什么 `run()` 里每个阶段开头都有一行 `_notify("load", 2)`。**

```python
    try:
        ...
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        self.integrity_checker.mark_failed(file_hash, str(file_path), str(e))
```

**这里有一个真实的缺陷，值得你记下来**：`file_hash` 是在 try 的第 3 行（L236）才赋值的。如果它之前的代码出错（比如 `compute_sha256` 因为文件被占用而抛 IOError），进入 except 后 `file_hash` 这个变量根本不存在，`mark_failed(file_hash, ...)` 会再抛一个 `NameError`，把原始错误掩盖掉。

> 对比 L549 的写法 `doc_id=file_hash if 'file_hash' in locals() else None` —— 作者在构造返回值时想到了这个边界，但忘了同一件事在 `mark_failed` 上也要处理。**面试时讲"我阅读代码时发现的一个健壮性缺陷"，比讲"我用了某某框架"有说服力得多。**

### 2.3 返回值：`PipelineResult`（L49-L94）

```python
self.success = success
self.chunk_count = chunk_count
self.image_count = image_count
self.vector_ids = vector_ids or []
self.stages = stages or {}          # 每个阶段的局部统计
```

`stages` 是个嵌套字典，形如：

```python
{
  "integrity": {"file_hash": "...", "skipped": False},
  "loading":   {"doc_id": "...", "text_length": 12000, "image_count": 3},
  "chunking":  {"chunk_count": 42, "avg_chunk_size": 980},
  ...
}
```

**同一个 `stages` 字典同时服务两个下游**：① `trace.record_stage()` 存进 `logs/traces.jsonl` 给 Dashboard 做时间线；② `PipelineResult.to_dict()` 返回给调用方（脚本 / Dashboard 的进度页）。一份数据两个用途，避免重复计算。

---

## 3. 阶段一：完整性校验（跨进程判重）

### 3.1 `pipeline.py` L229-L249

```python
logger.info("\n📋 Stage 1: File Integrity Check")
_notify("integrity", 1)                                    # 通知进度：第1阶段

file_hash = self.integrity_checker.compute_sha256(str(file_path))
logger.info(f"  File hash: {file_hash[:16]}...")

if not self.force and self.integrity_checker.should_skip(file_hash):
    logger.info(f"  ⏭️  File already processed, skipping (use force=True to reprocess)")
    return PipelineResult(
        success=True,                      # 注意：跳过也算成功
        file_path=str(file_path),
        doc_id=file_hash,
        stages={"integrity": {"skipped": True, "reason": "already_processed"}}
    )
```

三个关键决策：

1. **`not self.force and should_skip(...)`**：`force` 优先。批量摄取时加 `--force` 就能全量重跑。
2. **跳过时 `success=True`**：因为对调用方来说"这个文件已经在库里了"是符合预期的结果，不是错误。**如果把跳过当失败，批量脚本会误报。**
3. **跳过时直接 return**：不进入后续 5 个阶段，也不重复消耗 Embedding API 的钱。

### 3.2 `compute_sha256`：为什么分块读（`file_integrity.py` L206-L241）

```python
sha256_hash = hashlib.sha256()
with open(file_path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):    # 64KB 一块
        sha256_hash.update(chunk)
return sha256_hash.hexdigest()
```

`iter(lambda: f.read(65536), b"")` 是个 Python 惯用法，等价于"反复读 64KB，直到读到空字节串就停"。

**为什么不直接 `f.read()` 一次读完？** 因为要处理大 PDF（几百 MB）。一次性读进内存会爆，而且哈希算法本身就是流式的，没必要全量加载。**这是体现工程素养的细节，值得记。**

### 3.3 `should_skip`：状态机（L243-L268）

```python
cursor = conn.execute(
    "SELECT status FROM ingestion_history WHERE file_hash = ?",
    (file_hash,)
)
result = cursor.fetchone()
if result is None:
    return False            # 从没处理过 → 不跳过
return result[0] == "success"   # 只有成功的才跳过
```

**注意最后一行：只有 `status == 'success'` 才跳过。** 这意味着：

- 处理失败的文件 → 下次会重试（不会被永久卡住）
- 修改了文件内容 → SHA256 变了 → 视为新文件，重新处理

这就是 `mark_success` / `mark_failed` 两个方法存在的意义（L270 / L324），分别把状态写成 `'success'` 和 `'failed'`。

数据库表结构（L184-L193）：

```sql
CREATE TABLE IF NOT EXISTS ingestion_history (
    file_hash TEXT PRIMARY KEY,   -- 主键 = 文件哈希，天然去重
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,         -- 'success' | 'failed'
    collection TEXT,
    error_msg TEXT,
    processed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

`PRAGMA journal_mode=WAL`（L181）：开启 WAL 模式，**允许"一个进程写、多个进程读"并发**。为什么需要？因为 Dashboard 可能在摄取（写），同时 MCP Server 在查询（读）。默认的 SQLite 锁会让它们互相阻塞。

> 注意 `mark_success` 里有一段"先 SELECT 再决定 INSERT 还是 UPDATE"的代码（L293-L316）。这是为了**保留 `processed_at`（首次处理时间）**，只更新 `updated_at`。SQLite 的 `INSERT OR REPLACE` 会把整行删掉重建，首次时间就丢了。这是很典型的"ORM 之外必须懂的 SQL 细节"。

---

## 4. 阶段二：PDF 加载

### 4.1 `pipeline.py` L251-L281

```python
_t0 = time.monotonic()
document = self.loader.load(str(file_path))
_elapsed = (time.monotonic() - _t0) * 1000.0
```

`time.monotonic()` 而不是 `time.time()`：monotonic 是单调递增时钟，**不受系统时间调整（NTP 校时、改时区）影响**，测耗时只能用这个。乘 1000 转成毫秒。

```python
image_count = len(document.metadata.get("images", []))
```

`document.metadata.get("images", [])` 用 `.get` 带默认值，因为**不是每个 PDF 都有图**。这种防御式读取在全链路到处都是。

```python
if trace is not None:
    trace.record_stage("load", {
        "method": "markitdown",
        "doc_id": document.id,
        "text_length": len(document.text),
        "image_count": image_count,
        "text_preview": document.text,      # ⚠️ 注意这里是全文
    }, elapsed_ms=_elapsed)
```

**`"text_preview": document.text` 传的是完整正文**（不是前 200 字）。目的是让 Dashboard 能展示"这个阶段产出的完整文本"。

> ⚠️ 代价是 `logs/traces.jsonl` 会变得很大——一个 10 万字的 PDF 就会往日志里写 10 万字。这是一处典型的"可观测性换来存储成本"的权衡。**面试里被问到"你的 trace 怎么设计的、有什么代价"，这就是一个真实答案。**

### 4.2 `PdfLoader.load`：解析主流程（`pdf_loader.py` L79-L139）

```python
path = self._validate_file(file_path)
if path.suffix.lower() != '.pdf':
    raise ValueError(f"File is not a PDF: {path}")
```

**先校验再干活**。`.suffix.lower()` 是为了兼容 `.PDF` 大写后缀（Windows 上很常见）。

```python
doc_hash = self._compute_file_hash(path)
doc_id = f"doc_{doc_hash[:16]}"
```

同一份文件算出同一个 hash → 同一个 doc_id。**`[:16]` 是截断**：完整 SHA256 是 64 位十六进制，截 16 位足够避免碰撞，同时让 ID 短一些、方便阅读。

注意这里又算了一次文件哈希（`_compute_file_hash`，L141-L154，8KB 分块）。**和阶段一的 `compute_sha256` 重复了**——同一个文件被读了两次算两次哈希。这是可优化的点（把阶段一的 hash 传进来即可）。你如果要做性能优化，这是一个"投入产出比很高"的改造点。

```python
result = self._markitdown.convert(str(path))
text_content = result.text_content if hasattr(result, 'text_content') else str(result)
```

markitdown 是微软出的"万物转 Markdown"库。这里用 `hasattr` 做兼容——**不同版本的返回对象结构不同**（有的是 `.text_content` 属性，有的直接是字符串）。写库代码时这种"版本抖动兼容"很常见。

```python
title = self._extract_title(text_content)
if title:
    metadata["title"] = title
```

`_extract_title`（L156-L179）的策略：先在前 20 行找 `# ` 开头的 Markdown 标题；找不到就退化为"前 10 行里第一个非空行"。**先精确后降级，是规则提取的通用套路。**

```python
if self.extract_images:
    try:
        text_content, images_metadata = self._extract_and_process_images(...)
        if images_metadata:
            metadata["images"] = images_metadata
    except Exception as e:
        logger.warning(f"Image extraction failed for {path}, continuing with text-only: {e}")
```

**图片提取失败只 warning 不 raise**。如果这里抛异常，整个文件就废了。而实际上"没有图片的纯文本"仍然是可检索的——**降级后依然交付价值**，这是 RAG 摄取链路的核心设计哲学。

### 4.3 `_extract_and_process_images`：用 PyMuPDF 抽图（L181-L299）

```python
image_dir = self.image_storage_dir / doc_hash        # data/images/{collection}/{doc_hash}/
image_dir.mkdir(parents=True, exist_ok=True)
doc = fitz.open(pdf_path)                            # PyMuPDF 打开 PDF
```

图片按文档 hash 分目录存放，天然避免不同文档的图片重名。

```python
for page_num in range(len(doc)):
    page = doc[page_num]
    image_list = page.get_images(full=True)
    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]                # 图片在 PDF 内部的对象引用号
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]     # png / jpeg / ...
```

PyMuPDF 的取图方式是"两步走"：`page.get_images()` 拿到 xref 列表，再用 `doc.extract_image(xref)` 取实际字节。**xref 是 PDF 文件格式的内部概念**（交叉引用表条目），面试被问到时能说出这层就很有料。

```python
image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)
# → f"{doc_hash[:8]}_{page}_{sequence}"   例如 3f8a91c2_2_1
```

**ID 是确定性的**：文档 hash + 页码 + 页内序号。所以同一份 PDF 重跑，图片 ID 一模一样——这是幂等链条的一环。

```python
insert_position = len(modified_text)
modified_text += f"\n{placeholder}\n"
```

**注意这两行——这里是本文件最"诚实"的缺陷**：

- `modified_text` 始终是"当前文本的末尾"，所以**占位符是被追加到全文最后的**，而不是插在图片真正出现的段落里（L253-L254 的注释自己写明了 "simplified - in production, you'd parse page boundaries"）。
- 后果：`[IMAGE: xxx]` 全都堆在文末，切块后可能**图片和它原本的说明文字落进不同的 chunk**，图文关联会弱化。

> **改进方向**：markitdown 输出的是 Markdown 文本，可以在文本中按页/按位置匹配后再插入。这是"项目深度"类面试问题的好素材——**能指出问题 + 说清改进方案，就是深度**。

```python
image_metadata = {
    "id": image_id,
    "path": str(relative_path),
    "page": page_num + 1,
    "text_offset": insert_position + 1,
    "text_length": len(placeholder),
    "position": {"width": width, "height": height, "page": page_num + 1, "index": img_index}
}
```

这就是 `types.py` 里约定的图片结构。`text_offset` / `text_length` 是**给下游做精确定位用的**（虽然当前实现下 offset 总是文末）。

```python
try:
    relative_path = image_path.relative_to(Path.cwd())
except ValueError:
    relative_path = image_path.absolute()
```

**优先存相对路径，存不了才用绝对路径**。好处：项目换目录（本地 → Docker 容器）后相对路径依然有效。这是"可移植性"意识。

```python
except Exception as e:
    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
    continue                                        # 单张图失败，继续下一张
```

**单张图片失败不影响整页，单页失败不影响整文档**——三层防御。

---

## 5. 阶段三：切块

### 5.1 `pipeline.py` L283-L316

```python
chunks = self.chunker.split_document(document)
logger.info(f"  Chunks generated: {len(chunks)}")
if chunks:
    logger.info(f"  First chunk ID: {chunks[0].id}")
    logger.info(f"  First chunk preview: {chunks[0].text[:100]}...")
```

`stages["chunking"]` 里存了 `chunk_count` 和 `avg_chunk_size`：

```python
"avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0
```

**为什么关心平均长度？** 因为 chunk 太短 → 语义不完整；太长 → 检索精度下降（一个 chunk 里混了多个主题）。**平均长度是调参时第一个要看的指标**，Dashboard 上会展示。

trace 里还记录了**每个 chunk 的完整文本**（L307-L314）：

```python
"chunks": [
    {"chunk_id": c.id, "text": c.text, "char_len": len(c.text),
     "chunk_index": c.metadata.get("chunk_index", i)}
    for i, c in enumerate(chunks)
]
```

又见"全量数据进 trace"。目的是让 Dashboard 的摄取追踪页能点开每个 chunk 看内容——**可观测性做到底就是"每个中间产物都能看"**。

### 5.2 `DocumentChunker`：适配器模式（`document_chunker.py`）

构造函数只有一行核心：

```python
def __init__(self, settings: Settings):
    self._settings = settings
    self._splitter = SplitterFactory.create(settings)      # L73
```

**它不自己实现切分算法**，而是从工厂拿一个 splitter（默认是 `RecursiveSplitter`，内部用 langchain 的 `RecursiveCharacterTextSplitter`）。

这个类的定位写在文件头注释里，非常清楚：

> Core Value-Add (vs libs.splitter)：1. Chunk ID Generation 2. Metadata Inheritance 3. chunk_index 4. source_ref 5. Type Conversion

**这就是分层架构的分工**：

- `libs/splitter`：纯粹的"字符串 → 字符串列表"，只管切
- `ingestion/chunking`：业务层，负责"给每块发身份证、登记户口、建父子关系"

前端类比：`libs` 相当于通用工具库（比如 dayjs），`ingestion` 相当于你的业务 service 层。

### 5.3 `split_document` 主流程（L75-L138）

```python
if not document.text or not document.text.strip():
    raise ValueError(f"Document {document.id} has no text content to split")
```

**空文本直接抛错**。这里的选择很关键：空文档是"上游出错了"的信号（比如扫描版 PDF 没 OCR 出文字），**必须让调用方知道**，而不是悄悄产出一个空索引。

```python
text_fragments = self._splitter.split_text(document.text)

if not text_fragments:
    raise ValueError(f"Splitter returned no chunks for document {document.id}. ...")
```

第二道校验，防止 splitter 返回空列表。

```python
for index, text in enumerate(text_fragments):
    chunk_id = self._generate_chunk_id(document.id, index, text)
    chunk_metadata = self._inherit_metadata(document, index, text)
    chunk = Chunk(id=chunk_id, text=text, metadata=chunk_metadata)
    chunks.append(chunk)
```

**每个 chunk 独立计算 ID 和 metadata**，没有任何跨 chunk 的状态。这种"纯函数式"的写法让后续并行化（阶段 4/5 都用了线程池）成为可能。

### 5.4 `_generate_chunk_id`：三段式 ID（L140-L169）

```python
content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
return f"{doc_id}_{index:04d}_{content_hash}"
# 例如 doc_3f8a91c2d4e5f6a7_0000_c0535e4b
```

三个组成部分各有分工：

| 部分 | 作用 |
|---|---|
| `doc_id` | 归属哪个文档（可读、可追溯） |
| `index:04d` | 在文档中的顺序（`04d` = 补零到 4 位，排序时不会 `10` 排在 `2` 前面） |
| `content_hash[:8]` | 内容指纹（内容变了 ID 就变） |

**"内容指纹进 ID"是幂等的关键**：修改文档中某一段，只有那一段的 ID 变了，其余 chunk 的 ID 保持不变，可以精确地只更新变化的部分。

> ⚠️ **一个需要你自己注意的细节**：这个 ID **不是最终入库的 ID**。阶段六的 `VectorUpserter._generate_chunk_id`（L140-L168）会**重新生成一遍**：
>
> ```python
> source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
> return f"{source_hash}_{chunk_index:04d}_{content_hash}"
> ```
>
> 区别在前缀：这里是 `doc_{hash}`（文件内容 hash），那里是 `hash(source_path)`（**路径** hash）。
>
> 后果：**同一个文件移动到另一个路径，会得到新的向量 ID**，但阶段一的判重（按内容 hash）会认为"已处理过"从而跳过，导致"库里是新 ID、判重记录是旧 hash"的不一致。
>
> 这属于"两次 ID 生成规则没统一"的设计瑕疵。**面试时如果有人问"你的 chunk id 怎么设计的"，一定要能分清这两处。**

### 5.5 `_inherit_metadata`：元数据继承（L171-L249）

```python
chunk_metadata = document.metadata.copy()          # 继承文档级元数据
doc_images = document.metadata.get("images", [])
chunk_metadata.pop("images", None)                 # 但把 images 拿掉
```

**先全量继承，再删掉会冗余的字段**。因为 doc 级的 images 是"文档里所有图"，直接塞进每个 chunk 会让每个 chunk 都背上全文的图片清单（几百 KB 的重复数据）。

```python
chunk_metadata["chunk_index"] = chunk_index
chunk_metadata["source_ref"] = document.id
```

补上两个 chunk 级字段。`source_ref` 建立父子链。

```python
pattern = r'\[IMAGE:\s*([^\]]+)\]'
matches = re.findall(pattern, chunk_text)
image_refs = [m.strip() for m in matches]
chunk_metadata["image_refs"] = image_refs
```

**正则提取本块引用了哪些图**。`\s*` 容忍 `[IMAGE:xxx]` 和 `[IMAGE: xxx]` 两种写法。

```python
chunk_images = []
if image_refs and doc_images:
    image_lookup = {img.get("id"): img for img in doc_images}
    for img_id in image_refs:
        if img_id in image_lookup:
            chunk_images.append(image_lookup[img_id])
if chunk_images:
    chunk_metadata["images"] = chunk_images
```

**把"ID"还原成"完整图片信息"**：先用字典把 doc 级图片列表建成 `id → 图片信息` 的索引（O(1) 查找，避免对每个 ref 都遍历一遍 doc_images），再只保留本块涉及的。

为什么要把完整信息（含 path）挂到 chunk 上？因为阶段 4 的 `ImageCaptioner` 需要图片的 `path` 才能调 Vision API，而它只拿到 chunks，拿不到 document。

```python
if chunk_images:
    chunk_metadata["page_num"] = chunk_images[0].get("page")
```

从第一张图的页码推断本块的页码。**这是"顺带补充元数据"的小技巧**——有页码信息，检索时就能按页过滤。

---

## 6. 阶段四：三路变换（Pipeline 里最贵的一段）

### 6.1 `pipeline.py` L318-L375

三个 transform **串行执行**，前一个的输出是后一个的输入：

```python
chunks = self.chunk_refiner.transform(chunks, trace)
chunks = self.metadata_enricher.transform(chunks, trace)
chunks = self.image_captioner.transform(chunks, trace)
```

它们共享同一个接口（`base_transform.py` L10-L41）：

```python
class BaseTransform(ABC):
    @abstractmethod
    def transform(self, chunks: List[Chunk], trace=None) -> List[Chunk]:
        pass
```

**输入输出都是 `List[Chunk]`，长度不变。** 这个统一契约带来两个好处：

1. 可以在 `pipeline` 里任意增删、换序（比如不想要图片描述，把第三行注释掉即可）
2. 单独测试一个 transform 极其简单——造几个 Chunk 传进去，断言输出

`base_transform.py` 的注释明确了四条设计原则，其中 **"Atomic Operations: Failure in one chunk doesn't affect others"（单块失败不影响其他块）** 和 **"Graceful Degradation: Returns original chunk on unrecoverable errors"（无法恢复时返回原块）** 是这段代码的灵魂。

### 6.2 trace 的打点方式（L326-L375）

```python
_t0_transform = time.monotonic()
_pre_refine_texts = {c.id: c.text for c in chunks}      # 精炼前快照
chunks = self.chunk_refiner.transform(chunks, trace)
refined_by_llm = sum(1 for c in chunks if c.metadata.get("refined_by") == "llm")
refined_by_rule = sum(1 for c in chunks if c.metadata.get("refined_by") == "rule")
```

**先拍快照再变换**，然后把 trace 里存 `text_before` 和 `text_after` 两份文本（L364-L365）：

```python
"text_before": _pre_refine_texts.get(c.id, ""),
"text_after": c.text,
```

**为什么要存前后两份？** 因为 Dashboard 的摄取追踪页要能展示"LLM 到底把这段文本改成什么样了"。**没有前后对比，你无法判断 LLM 精炼是变好了还是变坏了**——这就把"黑盒变换"变成了"可审计变换"。

`refined_by` 这个字段是整个设计的点睛之笔：**每个 chunk 自己记录"我是被 LLM 处理的还是被规则处理的"**，于是可以一行统计：

```python
logger.info(f"      LLM refined: {refined_by_llm}, Rule refined: {refined_by_rule}")
```

> ⚡ **注意：三个 transform 是串行的，这可能是这条链路最大的性能瓶颈。** 因为每个 chunk 都要调一次 LLM，一个有 50 个 chunk 的 PDF 就要 50 次 API 调用。虽然每个 transform 内部用了线程池（5 个 worker），但三个 transform 之间是串行的。
>
> **可优化方向**：三个 transform 的 LLM 调用互相独立，理论上可以合并成一次调用（让 LLM 同时返回精炼文本 + title/summary/tags），把 3N 次调用降到 N 次。这是很好的面试加分点。

### 6.3 `ChunkRefiner`：规则前置 + LLM 增强

单个 chunk 的处理逻辑（L100-L146）：

```python
try:
    rule_refined_text = self._rule_based_refine(chunk.text)      # 第1步：规则清洗（必做）

    if self.use_llm and self.llm:
        llm_refined_text = self._llm_refine(rule_refined_text, trace)   # 第2步：LLM 增强
        if llm_refined_text:
            refined_text = llm_refined_text
            refined_by = "llm"
        else:
            refined_text = rule_refined_text      # LLM 失败 → 用规则结果
            refined_by = "rule"
    else:
        refined_text = rule_refined_text          # LLM 未启用 → 用规则结果
        refined_by = "rule"
```

**注意调用顺序：规则在前，LLM 在后，且 LLM 的输入是"规则清洗后的文本"而不是原文。**

这样做有三个理由：

1. **省 token**：先把页眉页脚、多余空行这些垃圾去掉，再送给 LLM，token 更少
2. **LLM 更容易做好**：给它干净的 Markdown 比给脏文本效果好
3. **兜底天然存在**：规则结果随时可用，LLM 一失败就退回它

```python
except Exception as e:
    logger.error(f"Failed to refine chunk {chunk.id}: {e}")
    return (chunk, "error", str(e))               # 单块异常 → 返回原块
```

**最外层再兜一层**：连规则清洗都炸了，就返回原 chunk，绝不让单个 chunk 毁掉整批。

### 6.4 `_rule_based_refine`：代码块保护（L275-L344）

这段是纯字符串处理，但思路很讲究：

```python
code_blocks = []
code_block_pattern = r'```[\s\S]*?```'

def extract_code_block(match):
    code_blocks.append(match.group(0))
    return f"__CODE_BLOCK_{len(code_blocks)-1}__"

text = re.sub(code_block_pattern, extract_code_block, text)
```

**第一步先把代码块"挖出来"换成占位符**。因为后面的清洗规则会压缩空格、删标签，而代码块里的缩进和 `<xxx>` 是有意义的，不能被破坏。

```python
text = re.sub(r'─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}', '', text, flags=re.IGNORECASE | re.DOTALL)
text = re.sub(r'─{10,}', '', text)
```

**第二步删页眉页脚**：匹配"连续 10 个以上的横线 + 页码/页脚关键词 + 横线"这种典型排版。第二条兜底删掉残留的分隔线。`re.DOTALL` 让 `.` 能匹配换行（否则跨行的页脚匹配不到）。

```python
text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)     # 删 HTML 注释
text = re.sub(r'<[^>]+>', '', text)                          # 删 HTML 标签，保留内容
text = re.sub(r' {2,}', ' ', text)                           # 多个空格 → 1 个
text = re.sub(r'\n{3,}', '\n\n', text)                       # 3+ 换行 → 2 个
lines = [line.rstrip() for line in text.split('\n')]         # 去掉每行行尾空白
```

- `\n{3,}` → `\n\n`：**保留段落分隔（空行），但不留大段空白**。这个尺度很重要——全删了段落结构就丢了。
- `line.rstrip()` 而不是 `strip()`：**只去行尾，保留行首缩进**（Markdown 的列表、代码缩进都靠行首空格）。

```python
for i, code_block in enumerate(code_blocks):
    text = text.replace(f"__CODE_BLOCK_{i}__", code_block)
return text.strip()
```

**最后把代码块原样放回去**。注意占位符用的是 `__CODE_BLOCK_0__` 这种形式——不会和正文内容撞车（普通文本里不会出现这种串）。

> 这个"提取 → 清洗 → 还原"的三段式，是所有"要处理但又要保护某类内容"的文本处理的通用模式。前端里类似"代码高亮时先抽离 `<pre>` 再处理其余 HTML"。

### 6.5 `ChunkRefiner` 的并行实现（L147-L200）

```python
max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))       # DEFAULT_MAX_WORKERS = 5
refined_chunks = [None] * len(chunks)                     # 预分配，按下标写回

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_idx = {
        executor.submit(self._refine_single_chunk, chunk, trace): idx
        for idx, chunk in enumerate(chunks)
    }
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        refined_chunk, refined_by, error = future.result()
        refined_chunks[idx] = refined_chunk               # 按原下标写回，保证顺序
```

三个细节：

1. **`min(5, len(chunks))`**：chunk 只有 2 个时不要开 5 个线程。
2. **`[None] * len(chunks)` + 按 idx 写回**：`as_completed` 是"谁先完成谁先返回"，**顺序是乱的**。只有按下标写回才能保证输出顺序和输入一致。这是**并发编程里最容易踩的坑**（如果你直接 `append`，最终顺序就乱了）。
3. **为什么用线程而不是进程？** 因为瓶颈在**网络 IO（等 API 返回）**，不是在算 CPU。Python 的 GIL 虽然限制 CPU 并行，但**IO 等待时会释放 GIL**，所以线程池对 API 调用有效。

```python
except Exception as e:
    logger.error(f"Unexpected error in parallel refinement: {e}")
    refined_chunks[idx] = chunks[idx]            # 兜底：用原块
```

### 6.6 `MetadataEnricher`：给每个 chunk 打标签

处理框架和 `ChunkRefiner` 完全一致（规则 → LLM → 兜底），重点看规则版和 LLM 版各自怎么做。

**规则版**（L326-L354）产出 `title / summary / tags` 三个字段：

```python
def _extract_title(self, text) -> str:                # L356
    heading_match = re.match(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()          # 1. Markdown 标题
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) <= 100 and not first_line.endswith(('.', ',', ';')):
        return first_line                              # 2. 短首行（且不像句子）
    sentences = re.split(r'[.!?]\s+', text)
    if sentences and sentences[0]:                     # 3. 首句
        title = re.sub(r'[.!?]+$', '', sentences[0].strip())
        return title if len(title) <= 150 else title[:147] + "..."
    return text[:100].strip()                          # 4. 前 100 字兜底
```

**四级降级策略**，每一级都有明确的判断条件。特别看第 2 级的 `not first_line.endswith(('.', ',', ';'))`——**用一个简单的标点判断来区分"标题行"和"正文行"**，很朴素但有效。

```python
def _extract_tags(self, text, max_tags=10) -> List[str]:      # L417
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)   # 专有名词
    identifiers = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b', text)  # camelCase/snake_case
    markdown_keywords = re.findall(r'\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_', text)  # 加粗/斜体
```

**三条启发式规则**：

- 大写开头的词 → 可能是专有名词（Azure、Docker）
- camelCase / snake_case → 代码标识符（`getUserId`、`user_name`）→ **技术文档的关键词**
- 加粗/斜体的词 → 作者本人在强调 → 高价值关键词

**这是一种"用排版信号反推语义重要性"的思路，在没有 LLM 或想省钱时非常实用。**

```python
tags.update(capitalized[:5])       # 每类最多取 5 个
...
tag_list = sorted(list(tags))[:max_tags]
```

用 `set()` 天然去重，再 `sorted()` 保证**输出确定性**（同一段文本每次跑出来的 tags 顺序一致，便于比对测试）。

**LLM 版**（L452-L578）的关键在于**让 LLM 输出结构化文本再解析**：

```python
prompt = self._load_prompt()                            # config/prompts/metadata_enrichment.txt
formatted_prompt = prompt.replace("{chunk_text}", text[:2000])   # 截断到 2000 字
```

`text[:2000]`：**控制输入长度**，既省 token 又避免超上下文。（代价是长 chunk 的后半段信息不参与生成——是个需要权衡的点。）

```python
title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
summary_match = re.search(r'Summary:\s*(.+?)(?:\n(?:Tags:|$))', response, re.IGNORECASE | re.DOTALL)
tags_match = re.search(r'Tags:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
```

**用正则解析 LLM 的回复**，因为 prompt 里要求它按 `Title: / Summary: / Tags:` 的格式输出。

> 这是 **2024 年之前的典型做法**。现在的标准做法是 **Function Calling / JSON Schema 结构化输出**（让模型直接返回 JSON，由 SDK 保证格式）。项目里用正则解析的原因大概是"多 Provider 兼容"（不是所有 provider 都支持结构化输出）。
>
> **面试时这是一个绝佳的"我知道更好的方案"谈资**：说明你知道 `response_format={"type": "json_schema"}` 或 tool calling，并且能解释为什么这个项目没用（跨 Provider 兼容性）。

```python
if not metadata['title']:
    metadata['title'] = 'Untitled'
if not metadata['summary']:
    metadata['summary'] = response[:500]     # 格式解析失败 → 把原始回复当前 500 字当摘要
```

**解析失败也要有兜底值**，不能让 metadata 缺字段导致下游 KeyError。

### 6.7 `ImageCaptioner`：只处理"真正被引用"的图（`image_captioner.py`）

这是阶段四里最值得细看的一个，因为它涉及**成本控制**。

```python
self._caption_cache: Dict[str, str] = {}
self._cache_lock = threading.Lock()
```

**一个缓存 + 一把锁**。共享可变状态在多线程下必须加锁，否则可能缓存读写错乱。

```python
image_lookup: Dict[str, dict] = {}
for chunk in chunks:
    if chunk.metadata and "images" in chunk.metadata:
        for img_meta in chunk.metadata.get("images", []):
            img_id = img_meta.get("id")
            if img_id and img_id not in image_lookup:
                image_lookup[img_id] = img_meta
```

**先汇总全文所有图片，建 `id → 图片信息` 索引**（同样的字典索引技巧）。

```python
images_to_caption: Dict[str, str] = {}      # img_id -> img_path
for chunk in chunks:
    referenced_ids = self._find_referenced_image_ids(chunk.text)     # 找 [IMAGE: xxx]
    for img_id in referenced_ids:
        if img_id not in images_to_caption:
            images_to_caption[img_id] = img_meta.get("path")
```

**第一遍：收集"被正文引用过的图"**。

> 🎯 **这就是本文件的核心优化**。PDF 里经常有装饰性图片（logo、边框、分隔线），它们不会在正文中以 `[IMAGE: xxx]` 出现。如果无脑对所有图调 Vision API，一个 30 页的 PPT 可能白烧几十次调用。

```python
if images_to_caption:
    self._generate_captions_parallel(images_to_caption, trace)
```

**第二遍之前先并行把 caption 全部生成好**，写入缓存。

```python
max_workers = min(DEFAULT_MAX_WORKERS, len(images_to_caption))    # DEFAULT_MAX_WORKERS = 3
```

**这里是 3 不是 5**，文件注释写明了原因：`Lower than text LLM due to higher cost/latency`——**Vision 调用更贵更慢，并发调低**。

```python
def _get_caption(self, img_id, img_path, trace=None):     # L91
    with self._cache_lock:
        if img_id in self._caption_cache:
            return self._caption_cache[img_id]            # 命中缓存直接返回
    if not img_path or not Path(img_path).exists():
        logger.warning(f"Image path not found: {img_path}")
        return None                                        # 文件不存在 → 跳过
    try:
        image_input = ImageInput(path=img_path)
        response = self.llm.chat_with_image(text=self.prompt, image=image_input, trace=trace)
        caption = response.content
        with self._cache_lock:
            self._caption_cache[img_id] = caption           # 写缓存
        return caption
    except Exception as e:
        logger.error(f"Failed to caption image {img_path}: {e}")
        return None                                         # 单图失败 → None，不影响其他
```

**同一张图在多个 chunk 里出现时只会调一次 API**（后续命中缓存）。注意缓存是**每次 `transform()` 开头清空的**（L163-L165），因为跨文档复用缓存会串数据。

**第二遍：把 caption 缝进正文**

```python
replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
new_text = new_text.replace(placeholder, replacement)
```

**关键设计：caption 是拼进 `chunk.text` 的，不是只存在 metadata 里。**

为什么？因为**只有进了 `text`，它才会被 Embedding 向量化、被 BM25 分词**。这样用户搜"系统架构图"时，就能命中那张架构图所在的 chunk——**这就是"搜文字出图"的实现原理**。

```python
if captions:
    if "image_captions" not in chunk.metadata:
        chunk.metadata["image_captions"] = []
    chunk.metadata["image_captions"].extend(captions)
```

同时在 metadata 里留一份结构化记录（`[{id, caption}, ...]`），供检索阶段做多模态结果组装（返回图片给用户）。

> ⚠️ **这里有两个可以改进的细节**（能讲出来说明你读细了）：
>
> 1. **`(Description: ...)` 这个英文前缀进入了向量空间**。中文库检索时，"Description" 这个词是无意义噪声。更好的做法是只拼 caption 内容，或者用中文括号。
> 2. **`extend` 有共享风险**：`chunk.metadata` 是浅拷贝自 document 的，如果多个 chunk 引用同一个 list 对象，`extend` 会互相污染。当前实现里每个 chunk 的 `image_captions` 是新建的 list，所以没出问题，但这是"靠实现恰好正确"而非"靠设计保证正确"。

---

## 7. 阶段五：双路编码

### 7.1 `pipeline.py` L377-L427

```python
batch_result = self.batch_processor.process(chunks, trace)
dense_vectors = batch_result.dense_vectors      # List[List[float]]，每 chunk 一个 1536 维向量
sparse_stats  = batch_result.sparse_stats       # List[Dict]，每 chunk 一份词频统计
```

**一次调用拿到两种表示**——这就是"双路"的意思：同一个 chunk，同时产出稠密向量（给语义检索）和词频统计（给关键词检索）。

```python
logger.info(f"  Dense vectors: {len(dense_vectors)} (dim={len(dense_vectors[0]) if dense_vectors else 0})")
```

**顺手校验维度**：`len(dense_vectors[0])` 应该是 1536。如果换了个 embedding 模型（比如 BGE-large 是 1024 维），这里会立刻暴露出来。

trace 里记录了非常细的编码信息（L400-L427）：

```python
for idx, c in enumerate(chunks):
    detail = {"chunk_id": c.id, "char_len": len(c.text)}
    if idx < len(dense_vectors):
        detail["dense_dim"] = len(dense_vectors[idx])
    if idx < len(sparse_stats):
        ss = sparse_stats[idx]
        detail["doc_length"] = ss.get("doc_length", 0)
        detail["unique_terms"] = ss.get("unique_terms", 0)
        tf = ss.get("term_frequencies", {})
        top_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]
        detail["top_terms"] = [{"term": t, "freq": f} for t, f in top_terms]
```

注意 `top_terms` 这一行：`sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]` —— **按词频降序取前 10 个词**。

**为什么要在 trace 里看词频？** 因为 BM25 检索不准时，第一个要排查的就是"分词是否正确"。如果 dashboard 上看到一个中文 chunk 的 top terms 是 `的 / 是 / 在`，你就知道是停用词没过滤；如果切出来一堆单字，就知道 jieba 没生效。**这是排障视角的可观测性，不是装饰。**

### 7.2 `BatchProcessor`：一个"薄编排层"（`batch_processor.py` L103-L209）

它自己几乎不干活，只做三件事：

```python
batches = self._create_batches(chunks)              # 1. 分批
for batch_idx, batch in enumerate(batches):
    try:
        batch_dense = self.dense_encoder.encode(batch, trace=trace)
        dense_vectors.extend(batch_dense)
        batch_sparse = self.sparse_encoder.encode(batch, trace=trace)
        sparse_stats.extend(batch_sparse)
        successful_chunks += len(batch)
    except Exception as e:
        failed_chunks += len(batch)                 # 2. 单批失败不中断
        if trace:
            trace.record_stage(f"batch_{batch_idx}_error", {"error": str(e), "batch_size": len(batch)})
```

```python
def _create_batches(self, chunks):                  # L211
    batches = []
    for i in range(0, len(chunks), self.batch_size):     # 3. 切片分批
        batches.append(chunks[i:i + self.batch_size])
    return batches
```

**`range(0, n, size)` + 切片**是最典型的分批写法。`batch_size` 默认 100（来自 `settings.ingestion.batch_size`）。

```python
return BatchResult(
    dense_vectors=dense_vectors,
    sparse_stats=sparse_stats,
    batch_count=batch_count,
    total_time=total_time,
    successful_chunks=successful_chunks,
    failed_chunks=failed_chunks
)
```

**返回值是一个 dataclass 而不是裸 tuple**——因为字段多（6 个），返回 tuple 后调用方要靠位置解包，极易搞错顺序。

> ⚠️ **注意这里的降级是有代价的**：某一批失败后，`dense_vectors` 和 `sparse_stats` 的长度会**小于** `chunks` 的长度，但 `BatchResult` 里没有任何"哪些 chunk 失败了"的信息。Pipeline 后面执行 `zip(sparse_stats, vector_ids)`（L443）时就会错位。
>
> 更稳健的做法是返回带占位符的等长列表（失败的位置填 `None`），或者抛出带批次信息的异常让上层决定。**这是一个可以指出来的隐患。**

### 7.3 `DenseEncoder`：只负责"对齐"（`dense_encoder.py` L66-L158）

```python
texts = [chunk.text for chunk in chunks]                    # 1. 抽出所有文本
for i, text in enumerate(texts):
    if not text or not text.strip():
        raise ValueError(f"Chunk at index {i} (id={chunks[i].id}) has empty or whitespace-only text")
```

**编码前先检查空文本**。因为空字符串送进 embedding API 通常返回垃圾向量或直接报错，**报错信息里带上 `id` 和 `index`** 是刻意为之——让你能立刻定位是哪一块。

```python
for batch_start in range(0, len(texts), self.batch_size):
    batch_texts = texts[batch_start:batch_end]
    batch_vectors = self.embedding.embed(texts=batch_texts, trace=trace)

    if len(batch_vectors) != len(batch_texts):
        raise RuntimeError(f"Embedding provider returned {len(batch_vectors)} vectors for {len(batch_texts)} texts ...")
    all_vectors.extend(batch_vectors)
```

**"数量必须对得上"的运行时校验**。向量和文本一旦错位，整个知识库就废了（A 的向量挂在 B 的文本上），而且**这种错误后期极难发现**——检索照样返回结果，只是结果莫名其妙。**在数据流边界上做长度断言，是廉价的保险。**

```python
if all_vectors:
    expected_dim = len(all_vectors[0])
    for i, vec in enumerate(all_vectors):
        if len(vec) != expected_dim:
            raise RuntimeError(f"Inconsistent vector dimensions: vector {i} has ...")
```

**维度一致性校验**。如果某个 provider 偶发返回不同维度的向量（或者中途换了模型），这里会兜住。

```python
except Exception as e:
    raise RuntimeError(f"Failed to encode batch {batch_start}-{batch_end}: {str(e)}") from e
```

**`raise ... from e`**：这是 Python 3 的异常链语法，保留原始异常（`__cause__`），打印时能看到 `The above exception was the direct cause of...`。**包装异常而不丢原始信息，是排查问题的基本功。**

> 📌 **一个重要的认知**：这个 encoder **只 embed `chunk.text`**，没有把 `title` / `summary` 拼进去。
>
> 那阶段四辛苦生成的 title/summary 有什么用？——它们存在 metadata 里，用于：① 检索结果展示（给用户看标题）；② 元数据过滤；③ 未来可以改成"标题 + 正文"一起 embed 来提升召回。
>
> **"元数据增强到底有没有提升检索效果"是一个必须用评估来回答的问题**，不能凭感觉。这也是项目里 `observability/evaluation/` 存在的意义。

### 7.4 `SparseEncoder`：中文分词是关键（`sparse_encoder.py` L72-L169）

```python
terms = self._tokenize(chunk.text)
term_frequencies = Counter(terms)
stat_dict = {
    "chunk_id": chunk.id,
    "term_frequencies": dict(term_frequencies),   # term -> 出现次数
    "doc_length": len(terms),                     # 该块总词数
    "unique_terms": len(term_frequencies),        # 去重后的词数
}
```

**输出的是"词频统计"，不是"向量"**。BM25 不算向量，它算的是"词在该文档里出现几次（tf）、在全库里出现过几个文档里（df）"——这是稀疏检索的本质，也是它和向量检索最大的区别。

```python
def _tokenize(self, text: str) -> List[str]:          # L134
    raw_tokens = jieba.lcut(text)                     # 中文分词

    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r'[\s\W]+', token, re.UNICODE):     # 纯标点/空白 → 丢掉
            continue
        tokens.append(token)

    if self.lowercase:
        tokens = [t.lower() for t in tokens]
    terms = [t for t in tokens if len(t) >= self.min_term_length]    # 默认 >= 2
    return terms
```

**`jieba.lcut()` 是这里最重要的一个调用。** 为什么？因为：

- **英文**天然有空格分隔，`text.split()` 就够
- **中文**没有空格，"机器学习"必须切成 `["机器", "学习"]`，否则整句就是一个 token，BM25 完全匹配不上

```python
if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
```

`fullmatch` 要求**整个 token 都是**标点/空白才丢弃。用 `fullmatch` 而不是 `match`（match 只要求开头匹配）——否则 `"AI,"` 这种带标点的会被误丢。

```python
if self.lowercase:
    tokens = [t.lower() for t in tokens]
```

**统一小写**。这里有个必须注意的联动：`BM25Indexer.query` 里也做了 `query_terms = [t.lower() for t in query_terms]`（`bm25_indexer.py` L257）。

**索引侧和查询侧的分词规则必须完全一致**，否则查不到。文件注释（L138-L139）特意强调了这一点：`This ensures consistent tokenization with the query-side (QueryProcessor), which is required for BM25 matching.`

> ⚠️ **`min_term_length = 2` 会丢掉所有单字词**。中文里"图"、"表"、"码"这类单字有时是关键（比如查"架构图"时，"图"有帮助）。这是个可调参数，**调参时可以用评估集验证"改成 1 会不会更好"**。

```python
def get_corpus_stats(self, encoded_chunks):          # L171
    num_docs = len(encoded_chunks)
    total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)
    avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0

    doc_freq: Dict[str, int] = {}
    for chunk_stats in encoded_chunks:
        for term in chunk_stats["term_frequencies"].keys():
            doc_freq[term] = doc_freq.get(term, 0) + 1
    return {"num_docs": num_docs, "avg_doc_length": avg_doc_length, "document_frequency": doc_freq}
```

**这是"语料库级统计"**，注意两个概念的区别：

- `doc_length`：**本块**有多少词
- `avg_doc_length`：**全库平均**每块有多少词 ← 必须遍历全部 chunk 才能算

`df`（document frequency）算的是"**有多少个块**包含这个词"（不是出现了几次）。遍历时用 `.keys()` 而不是 `.values()`——**每块对同一个词只贡献 1**，否则就变成 tf 了。

---

## 8. 阶段六：落库（三写）

### 8.1 `pipeline.py` L429-L518

```python
vector_ids = self.vector_upserter.upsert(chunks, dense_vectors, trace)

# ⭐ 这一行是整条链路最关键的"接头"
for stat, vid in zip(sparse_stats, vector_ids):
    stat["chunk_id"] = vid

self.bm25_indexer.add_documents(sparse_stats, collection=self.collection, doc_id=document.id, trace=trace)
```

**为什么要把 BM25 的 `chunk_id` 替换成 Chroma 的 `vector_id`？**

推演一遍检索流程就明白了：

1. 用户查询 → BM25 索引算出命中 → 拿到 `chunk_id`（**BM25 只知道 ID 和分数，它不存正文**）
2. 拿着这个 ID 去 ChromaDB 取正文和元数据 → 返回给用户

**如果两边的 ID 不一致，第 2 步就查不到东西。** 所以在写入阶段就把它们的 ID 对齐。

这也是 D3（`SparseRetriever`）任务里"BaseVectorStore.get_by_ids"存在的原因。**这是一个典型的"跨存储一致性"问题**，面试里问"你系统里有两个索引，怎么保证它们一致"就是这个点。

```python
images = document.metadata.get("images", [])
for img in images:
    img_path = Path(img["path"])
    if img_path.exists():                                    # 先确认文件真的在
        self.image_storage.register_image(
            image_id=img["id"], file_path=img_path,
            collection=self.collection, doc_hash=file_hash, page_num=img.get("page", 0)
        )
logger.info(f"      Indexed {len(images)} images")
```

**注意这里是"登记"而不是"存储"**——图片文件在阶段二就已经由 `PdfLoader` 写到磁盘了，这里只是把它的位置登记到 SQLite 索引表，让检索时能按 `image_id` 找回文件。

`if img_path.exists()` 这个判断很实在：**索引里不能登记不存在的文件**，否则检索时返回 404。

```python
self.integrity_checker.mark_success(file_hash, str(file_path), self.collection)
```

**最后才写"成功"标记**。如果前面任何一步失败，标记不会写，下次跑还会重试。**"最后确认"是幂等设计的关键顺序**：先干活，全干完了再记账。

### 8.2 `VectorUpserter`：幂等写入（`vector_upserter.py` L73-L168）

```python
if len(chunks) != len(vectors):
    raise ValueError(f"Chunk count ({len(chunks)}) must match vector count ({len(vectors)})")
```

第一道防线：数量和维度都要对上。

```python
for chunk, vector in zip(chunks, vectors):
    chunk_id = self._generate_chunk_id(chunk)          # 重新生成稳定 ID
    chunk_ids.append(chunk_id)
    record = {
        "id": chunk_id,
        "vector": vector,
        "metadata": {
            **chunk.metadata,      # 展开所有元数据（title/tags/source_path/images/...）
            "text": chunk.text,    # ⭐ 正文塞进 metadata
            "chunk_id": chunk_id,
        },
    }
```

**`"text": chunk.text` 这一行很关键。** 为什么正文要塞进 metadata？因为 ChromaDB 的 `metadata` 字段是**跟着向量一起存、并且查询时一起返回**的。检索命中后直接就能拿到正文，**不用再回源读文件**（PDF 已经不在原路径或已被删除的场景也能工作）。

```python
def _generate_chunk_id(self, chunk: Chunk) -> str:      # L140
    if "source_path" not in chunk.metadata:
        raise ValueError("Chunk metadata must contain 'source_path'")
    if "chunk_index" not in chunk.metadata:
        raise ValueError("Chunk metadata must contain 'chunk_index'")
    source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
    content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
    return f"{source_hash}_{chunk_index:04d}_{content_hash}"
```

**这里显式校验了两个必填字段**，等于把 `Chunk` 契约（`document_chunker` 负责填的字段）在消费端再确认了一次。**在系统边界做契约校验**是好习惯。

ID 三段式的语义：

| 段 | 变化时机 | 作用 |
|---|---|---|
| `source_hash` | 换路径就变 | 区分文件 |
| `chunk_index` | 内容插入/删除时变 | 定位位置 |
| `content_hash` | 内容改了才变 | 内容版本 |

```python
self.vector_store.upsert(records, trace=trace)
```

**`upsert` = update + insert**：ID 已存在就覆盖，不存在就新增。所以**同一个文件重复摄取不会产生重复数据**——这就是幂等。整个系统能放心地"重跑"，靠的就是这个。

### 8.3 `BM25Indexer`：全量重建 + 原子写（`bm25_indexer.py`）

这个类有两个写入方法，Pipeline 用的是 `add_documents`（增量）：

```python
def add_documents(self, term_stats, collection="default", doc_id=None, trace=None):    # L311
    if not term_stats:
        return

    if not self._index:
        self.load(collection)                        # 1. 先把磁盘上的旧索引读进内存

    if doc_id and self._index:
        self.remove_document(doc_id, collection)     # 2. 删掉这个文档的旧倒排项（支持重新摄取）

    existing_stats = {}
    for term, term_data in self._index.items():
        for posting in term_data["postings"]:
            cid = posting["chunk_id"]
            if cid not in existing_stats:
                existing_stats[cid] = {"chunk_id": cid, "term_frequencies": {}, "doc_length": posting["doc_length"]}
            existing_stats[cid]["term_frequencies"][term] = posting["tf"]
```

**这段代码在做一件很聪明的事：从倒排索引"反推"出正向的词频统计。**

因为 `build()` 需要的输入是"每块的词频"（正排视角），而当前内存里只有"词 → 块列表"（倒排视角）。所以它把倒排遍历一遍，重新组装成正排格式，才能和新的 term_stats 合并。

**代价是 O(词数 × 平均倒排长度)**——每次增量摄取都要把整个索引重build一遍。对这个项目的数据规模（几千个 chunk）完全够用，但**在百万级语料下会成为瓶颈**。

> **改进方向**（面试可讲）：为每个 chunk 单独持久化一份正排词频（比如 `data/db/bm25/{collection}_forward.json`），增量时就不用反推了。或者改用真正的检索库（Elasticsearch / Tantivy / rank-bm25 库）来管倒排索引。

```python
    combined = list(existing_stats.values()) + list(term_stats)
    self.build(combined, collection, trace)          # 3. 全量重建（IDF 必须全量重算）
```

**为什么必须全量 rebuild？** 因为 **IDF 依赖 N（文档总数）和 df（文档频率）**，加一批新文档会改变所有词的 IDF。**这是 BM25 绕不开的成本。**

**`build()` 的计算（L100-L185）**

```python
num_docs = len(term_stats)
total_length = sum(stat["doc_length"] for stat in term_stats)
avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0

doc_freq: Dict[str, int] = {}
for stat in term_stats:
    for term in stat["term_frequencies"].keys():
        doc_freq[term] = doc_freq.get(term, 0) + 1
```

先算语料库级统计：文档数、平均文档长度、每个词的 df。

```python
for term, df in doc_freq.items():
    idf = self._calculate_idf(num_docs, df)
    postings = []
    for stat in term_stats:
        tf = stat["term_frequencies"].get(term, 0)
        if tf > 0:                                    # 只收录出现过的块
            postings.append({"chunk_id": stat["chunk_id"], "tf": tf, "doc_length": stat["doc_length"]})
    index[term] = {"idf": idf, "df": df, "postings": postings}
```

**这一步是双循环（词 × 文档），构建时间随语料增长是平方级的。** 优化空间在于：先构建"块 → 词"正排，再一次性倒排；或者用 `defaultdict` 累加。对这个项目够用，但值得知道。

IDF 公式（L436-L448）：

```python
return math.log((num_docs - df + 0.5) / (df + 0.5))
```

**`+0.5` 是平滑项（smoothing）**，防止 df 极小时分母为 0、或 df ≈ N 时分子为 0。

注意注释写着 **"can be negative for very common terms"**——当某个词出现在超过一半的文档里时，IDF 会变成负数。负数 IDF 在理论上会让"常见词命中的文档"反而被降权，这是 BM25 的已知特性（也是为什么需要停用词过滤）。**能解释这一点，说明你真的看懂了公式而不是抄下来的。**

BM25 打分公式（L450-L478）：

```python
numerator = tf * (self.k1 + 1)
denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
return idf * (numerator / denominator)
```

两个超参数的物理意义：

- **`k1 = 1.5`**：控制 **tf 的饱和速度**。词出现 10 次不等于得分是 1 次的 10 倍——`k1` 让 tf 增长到一定程度后收益递减（避免"堆关键词"刷分）。
- **`b = 0.75`**：控制 **长度归一化强度**。`doc_length / avg_doc_length > 1`（比平均长）时会被降权，防止长文档因为词多而天然占优。`b = 0` 表示完全不归一化，`b = 1` 表示完全归一化。

查询时的打分（L225-L291）：

```python
query_terms = [t.lower() for t in query_terms]        # 必须和索引侧一致

scores: Dict[str, float] = {}
for term in query_terms:
    if term not in self._index:
        continue                                       # 库里没这个词 → 跳过
    term_data = self._index[term]
    for posting in term_data["postings"]:              # 只遍历"含该词"的块（倒排的核心优势）
        term_score = self._calculate_bm25_score(tf=..., doc_length=..., avg_doc_length=..., idf=term_data["idf"])
        scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score     # 多个词累加

sorted_results = sorted([...], key=lambda x: x["score"], reverse=True)
return sorted_results[:top_k]
```

**这一段就是"倒排索引为什么快"的答案**：查询时**只遍历包含该词的 posting list**，而不是扫描全部文档。一个词只出现在 3 个块里，就只算 3 次。

**持久化的原子写（L518-L548）**

```python
temp_path = index_path.with_suffix('.tmp')
with open(temp_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
temp_path.replace(index_path)              # 原子 rename
```

**"先写临时文件，再 rename 覆盖"是文件写入的标准安全模式。** `Path.replace()` 在大多数系统上是原子操作，保证读方要么看到旧版本、要么看到新版本，**绝不会读到写了一半的残缺 JSON**。

**为什么需要？** 因为 Dashboard 可能正在读索引做查询，同时 MCP Server 在写入。如果直接覆盖写，读到一半的文件就崩了。

`ensure_ascii=False`：中文不转义成 `\uXXXX`，文件可读性更好、体积更小。

```python
except Exception as e:
    if temp_path.exists():
        temp_path.unlink()                 # 写失败就清掉临时文件
    raise
```

**失败时清理残留**，不让 `.tmp` 垃圾堆在磁盘上。`raise` 是裸 raise，表示"清理完继续往上抛原异常"，不吞掉错误。

---

## 9. 串起来看：这条链路的全貌与衔接点

```
                         ┌── force? ─────────────────────────────┐
PDF ──► ① SHA256 判重 ──►│ 命中 success → 直接返回（零成本）      │
                         └── 未命中 → 继续 ──────────────────────┘
                                     │
② markitdown 转 Markdown + PyMuPDF 抽图 ──► Document(text, metadata.images)
                                     │
③ RecursiveSplitter 切块 ──► List[Chunk]（带 id / chunk_index / source_ref / image_refs）
                                     │
④ ChunkRefiner（规则→LLM）──► MetadataEnricher（规则→LLM）──► ImageCaptioner（Vision）
   └─ 每个 chunk 新增：refined_by / enriched_by / title·summary·tags / image_captions
   └─ caption 被拼进 chunk.text（这才让"搜文字出图"成为可能）
                                     │
⑤ BatchProcessor：DenseEncoder（embedding API）＋ SparseEncoder（jieba 词频）
                                     │
                    ┌────────────────┴────────────────┐
⑥ ChromaDB 向量+正文      BM25 倒排索引（全量重建 IDF）   图片 SQLite 索引
   （upsert 幂等）          （chunk_id 与 Chroma 对齐！）   （只登记，文件已在②落盘）
                                     │
                          mark_success（最后才记账）
```

### 三个必须记住的衔接点

1. **`chunk_id` 对齐**（`pipeline.py` L443）：BM25 索引里的 ID 必须替换成 Chroma 的 vector_id，否则稀疏检索拿不到正文。
2. **caption 进 `chunk.text`**：只有进 text 才会被向量化/分词，否则"搜文字出图"失效。
3. **`mark_success` 放最后**：保证"没跑完的文件下次还会重跑"，这是幂等的前提。

### 四个可以改进的点（面试时都是加分项）

| # | 位置 | 问题 | 改进方向 |
|---|---|---|---|
| 1 | `pipeline.py` L544 | 异常时 `file_hash` 未定义 → 掩盖原始错误 | 先初始化 `file_hash = None` 再做用 |
| 2 | `pdf_loader.py` L253-L256 | 图片占位符统一追加到文末，图文位置错乱 | 在 Markdown 文本中按页/位置匹配后插入 |
| 3 | `document_chunker` vs `vector_upserter` | 两处 chunk_id 生成规则不一致（doc hash vs path hash） | 统一为同一套 ID 生成函数 |
| 4 | 阶段四三连串行 | 3N 次 LLM 调用，是最大延迟来源 | 合并成一次调用（精炼+摘要+标签一起返回），3N → N |

### 建议的动手实验顺序

1. **先跑通**：`python scripts/ingest.py <某个.pdf>`，看 `logs/traces.jsonl` 里六个阶段的 `elapsed_ms`，找出最慢的阶段。
2. **改一个参数**：把 `settings.yaml` 的 `ingestion.chunk_size` 从 1000 改成 500，重新摄取（加 `--force`），对比 `chunk_count` 和检索结果的变化。
3. **关掉 LLM**：把 `chunk_refiner.use_llm` 设为 `false`，观察 trace 里 `refined_by` 从 `llm` 全变成 `rule`，对比耗时差异。
4. **修一个缺陷**：把上表第 1 条修掉（3 行代码），写一个单测覆盖"文件不存在时 pipeline 返回失败而不是崩在 except 里"。
5. **做一次调优**：按第 4 条把三个 transform 合并，用评估面板对比 `hit_rate` 有没有变化。

> 完成第 4、5 步，你就有了一个"我读过这个项目并且改过它"的真实故事——**这比简历上写"熟悉 RAG"有说服力得多。**
