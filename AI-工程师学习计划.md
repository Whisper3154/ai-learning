# AI 工程师学习计划 — 大模型应用开发方向

**目标读者：** 有编程基础和 Git 使用经验，接触过 AI 工具，但对 AI 工程化（框架使用、应用落地、生产部署）缺乏系统经验
**周期：** 12 周（约 3 个月）
**核心目标：** 能独立搭建端到端的 LLM 应用（RAG、Agent、API 服务）并部署上线

---

## 阶段一：夯实基础（第 1-3 周）

> **阶段目标：** 建立 LLM 系统认知，掌握 AI 项目工程化搭建，能独立调用各类大模型 API 并构建可用应用。
> **阶段心法：** 这三周是"从会用 AI 工具"到"能搭建 AI 应用"的跨越。重点不在语言基础，而在 AI 生态的工程化实践。

---

### 第 1 周 — AI 项目工程化与环境搭建

**目标：** 学会搭建规范的 AI 项目脚手架，掌握 AI 开发的工程化习惯

**学习清单：**
- [ ] AI 项目标准结构：Prompt 管理、模型配置、工具注册、评估数据集
- [ ] AI 专属配置管理：多模型切换、环境隔离、Key 管理
- [ ] AI 开发工具链：Jupyter 探索 → 脚本固化 → 服务化 的演进路径
- [ ] AI 项目依赖管理：poetry、版本锁定、避免依赖地狱
- [ ] Prompt 版本管理：把 Prompt 当代码管，支持回滚和对比

**详细学习内容：**

1. **AI 项目结构（不是普通 Python 项目）**
   ```
   ai-app/
   ├── src/
   │   └── myapp/
   │       ├── prompts/          # Prompt 模板（独立管理，方便迭代）
   │       │   ├── system.txt
   │       │   └── rag_answer.txt
   │       ├── tools/            # Agent 工具注册
   │       │   ├── __init__.py
   │       │   ├── search.py
   │       │   └── calculator.py
   │       ├── chains/           # LangChain / 自定义链
   │       ├── config.py         # pydantic-settings 统一配置
   │       ├── models.py         # LLM 选型与切换
   │       └── evaluation/       # 评估数据集和脚本
   │           ├── datasets/
   │           └── runner.py
   ├── notebooks/                # Jupyter 探索阶段
   ├── tests/
   ├── .env.example
   └── pyproject.toml
   ```
   - 和普通项目的区别：多了 `prompts/`、`tools/`、`evaluation/` 目录
   - 核心思想：Prompt、工具、模型配置都要独立管理，方便 A/B 测试和迭代

2. **AI 专属配置管理**
   - `pydantic-settings` 管理所有配置：模型名称、API Key、温度参数、Top-K 等
   - 多环境配置：开发用便宜模型 + 小 Top-K，生产用强模型
   - API Key 安全管理：`.env` + 密钥管理服务，绝不硬编码
   - 模型切换抽象层：一行配置切换 OpenAI / Claude / 本地模型

3. **AI 开发的工作流**
   - **Jupyter 阶段：** 快速实验 Prompt、测试模型效果、验证想法
   - **脚本阶段：** 把验证过的想法固化成可复现的脚本
   - **服务阶段：** 封装成 API 服务，对外提供能力
   - 关键习惯：Jupyter 中验证好的 Prompt 要提取到 `prompts/` 目录管理

4. **Prompt 版本管理**
   - 把 Prompt 当代码：存文件、做版本、写注释说明改动原因
   - Prompt 模板引擎：Jinja2 模板化，支持变量注入
   - 对比测试：同一个问题用不同版本的 Prompt，量化效果差异
   - 进阶工具：promptfoo 管理 Prompt 版本和评估

5. **依赖管理要点**
   - AI 项目的依赖特别容易冲突（LangChain 版本更新快）
   - 用 poetry 锁定版本，避免"我本地能跑你那边不行"
   - 分层依赖：核心依赖 vs 可选依赖（如本地模型依赖 torch 很大）

**建议：**
- 第一天就建好项目模板，后续所有开发都在这个结构上进行
- 从 Jupyter 开始探索，验证后再搬到项目代码中
- Prompt 管理是 AI 工程化最容易被忽视的部分，从第一天就养成习惯

**常见坑：**
- 所有代码塞在一个 Jupyter 里，无法复现和部署
- API Key 写死在代码里，提交到 Git 后泄露
- Prompt 改了不知道改了啥，没有版本记录

**实践项目：** 搭建一个 AI 项目脚手架模板
- 包含 prompts 目录（2-3 个 Prompt 模板）
- 包含 tools 目录（1-2 个示例工具）
- pydantic-settings 统一配置（支持多模型切换）
- poetry 管理依赖
- Jupyter notebook 做探索演示

---

### 第 2 周 — LLM 核心概念

**目标：** 搞清楚"大模型到底是什么"，建立正确的心智模型，理解推理模型时代的新范式

**学习清单：**
- [ ] Transformer 架构核心思想（注意力机制、编码器-解码器）
- [ ] Token、Embedding、上下文窗口、温度参数的含义
- [ ] Prompt Engineering 基础：系统提示、Few-shot、CoT
- [ ] 推理模型（Reasoning Models）：o1/o3、DeepSeek-R1 的新范式
- [ ] 主流模型对比：GPT-4o、Claude、Llama、Qwen、DeepSeek-R1 的定位
- [ ] API 调用 vs 本地部署的权衡

**详细学习内容：**

1. **Transformer 架构（理解原理，不需要写代码实现）**
   - 核心思想：Self-Attention，让模型在处理每个词时能"看到"所有其他词
   - Q/K/V 三兄弟：Query（我在找什么）、Key（我有什么）、Value（我的内容是什么）
   - 编码器 vs 解码器 vs 编码器-解码器的区别
   - 推荐理解方式：看 3Blue1Brown 的可视化视频，比看论文直观得多

2. **关键概念扫盲**
   - **Token：** 模型处理文本的最小单位，不等于字或词。中文大约 1 个字 = 1-2 个 token
   - **Embedding：** 将文本映射到高维向量空间，语义相近的文本向量也相近
   - **上下文窗口：** 模型一次能"看到"的文本长度，如 128K token ≈ 10 万字
   - **温度参数：** 控制输出随机性。temperature=0 几乎确定性输出，=1 更有创造力
   - **System / User / Assistant 消息：** 对话的三种角色和它们的作用

3. **Prompt Engineering 基础**
   - **系统提示（System Prompt）：** 给模型设定角色和行为规则
   - **Few-shot：** 在提示中给出几个示例，引导模型模仿
   - **Chain-of-Thought（CoT）：** 让模型"一步步想"，显著提升推理准确率
   - **角色扮演：** "你是一个资深的 Python 开发者"这类技巧
   - 关键认知：Prompt 是和模型沟通的接口，写好 Prompt 是 AI 工程师的核心技能之一

4. **推理模型（2025-2026 新范式）**
   - 什么是推理模型：内置思维链（Thinking Process），能在回答前进行长链条推理
   - 代表模型：OpenAI o1/o3-mini、DeepSeek-R1、Claude extended thinking
   - 和传统模型的区别：推理模型自主思考，传统 CoT 提示词在推理模型上可能"反向优化"
   - Prompt 差异：推理模型不需要手动写 CoT，只需给清晰目标和约束条件
   - 注意：输出 Token 更多、耗时更长、成本更高，适合复杂推理但不适合简单任务

5. **主流模型对比**
   - **GPT-4o / GPT-4.5：** OpenAI 旗舰，通用能力强，生态最成熟
   - **OpenAI o1/o3：** 推理模型，数学/编程/逻辑推理极强，耗时和成本高
   - **Claude 4 (Opus/Sonnet)：** Anthropic 出品，长上下文、代码能力强，注重安全性
   - **DeepSeek-R1：** 开源推理模型，推理能力强，中文友好，成本低
   - **Llama 3：** Meta 开源，可本地部署，社区生态活跃
   - **Qwen 2.5：** 国产开源，中文能力强，性价比高
   - 选择原则：通用任务用通用模型，复杂推理用推理模型，按需混合

6. **API 调用 vs 本地部署**
   - API 调用：门槛低、效果好、按量付费、依赖网络
   - 本地部署：隐私好、无网络依赖、可控性强、需要 GPU 资源
   - 初学者建议：先用 API，理解了再探索本地部署

**建议：**
- 这周以"理解概念"为主，不要纠结太多细节
- 动手调几个 API，感受 temperature、top_p、max_tokens 等参数的实际效果
- 写一份自己的"LLM 概念速查表"，后续遇到不懂的就往里加

**常见坑：**
- 不要试图一次看懂 Transformer 论文，先看可视化讲解建立直觉
- 不要迷信"万能 Prompt 模板"，理解原理才能举一反三

**实践项目：** 用 OpenAI / Anthropic API 写一个多轮对话机器人
- 支持 system/user/assistant 多轮对话
- 让用户可以调节 temperature、model 等参数
- 用日志记录每次调用的 token 消耗

**推荐资源：**
- 3Blue1Brown: "Attention in Transformers, visually explained"（视频）
- Andrej Karpathy: "Let's build GPT from scratch"（YouTube，有编程基础可看）
- Anthropic Prompt Engineering Guide（docs.anthropic.com）

---

### 第 3 周 — API 集成与 SDK

**目标：** 熟练调用和管理 API，能处理真实场景中的各种边界情况

**学习清单：**
- [ ] OpenAI Python SDK 精讲（chat.completions、流式输出、函数调用）
- [ ] Anthropic Claude SDK（Messages API、Tool Use）
- [ ] 统一接口方案：LiteLLM / OpenAI 兼容格式
- [ ] Token 计数、成本估算、速率限制处理
- [ ] 异步调用：asyncio + httpx

**详细学习内容：**

1. **OpenAI Python SDK**
   - 基础调用：`client.chat.completions.create()`
   - 流式输出：`stream=True`，逐 token 返回，提升用户体验
   - 结构化输出：`response_format={"type": "json_object"}` 强制 JSON 输出
   - 错误处理：捕获 `RateLimitError`、`APIError`、`APITimeoutError`

2. **Anthropic Claude SDK**
   - Messages API 格式：和 OpenAI 的区别（无 system 参数，system 作为顶层字段）
   - Tool Use：定义工具 schema，处理模型返回的 tool_use 块
   - 流式输出：`stream=True` + `text_stream` 遍历
   - 长上下文优势：Claude 支持 200K token 上下文

3. **统一接口方案**
   - 为什么需要：项目可能需要切换模型，不想到处改代码
   - LiteLLM：支持 100+ 模型，统一 OpenAI 格式调用
   - OpenAI 兼容格式：很多国产模型（DeepSeek、Qwen）都兼容 OpenAI API 格式
   - 抽象一层 Wrapper，封装模型切换、重试、fallback 逻辑

4. **Token 计数与成本管理**
   - `tiktoken`（OpenAI 官方）计算 token 数
   - 估算成本：输入/输出 token 单价不同，要分开算
   - 速率限制处理：指数退避重试（exponential backoff）
   - 设定 token 预算上限，防止失控

5. **异步调用**
   - `asyncio` + `httpx.AsyncClient` 实现并发请求
   - 适用场景：批量处理文档、同时查询多个知识库
   - 注意 API 并发限制，用 `asyncio.Semaphore` 控制并发数

**建议：**
- 这周要大量写代码，不要只看文档
- 尝试同时用 OpenAI 和 Claude 的 SDK，体会设计差异
- 做一个简单的成本追踪器，记录每次 API 调用的花费

**常见坑：**
- 不要把 API Key 硬编码在代码里，一定用环境变量
- 流式输出时注意处理中断（用户取消请求），否则浪费 token
- 异步调用容易出 bug，先保证同步版本跑通再优化

**实践项目：** 构建一个支持切换模型的 API 代理服务
- 通过配置切换 OpenAI / Claude / 其他模型
- 支持流式和非流式两种模式
- 记录每次调用的 token 数和耗时
- 实现基础的重试和降级逻辑

**推荐资源：**
- OpenAI Python SDK 文档（platform.openai.com/docs）
- Anthropic SDK 文档（docs.anthropic.com）
- LiteLLM 文档（docs.litellm.ai）

---

## 阶段二：核心技能（第 4-7 周）

> **阶段目标：** 掌握大模型应用开发的四大核心能力：向量检索、RAG、工具调用、Agent。
> **阶段心法：** 这四周是整个计划的硬核部分。每个技能都是后面项目的基石，务必动手写代码，不要只看不练。

---

### 第 4 周 — 向量数据库与 Embedding

**目标：** 掌握 RAG 的存储层，理解"语义搜索"的工作原理

**学习清单：**
- [ ] Embedding 模型选型（OpenAI text-embedding、BGE、Jina）
- [ ] 向量相似度计算（余弦、欧氏、内积）
- [ ] 向量数据库入门：ChromaDB / FAISS（本地）
- [ ] 生产级方案：PGVector（PostgreSQL 向量扩展）——企业首选
- [ ] 分块策略：固定长度、语义分割、重叠窗口
- [ ] 元数据过滤与混合检索

**详细学习内容：**

1. **Embedding 原理与选型**
   - Embedding 就是把文本变成一串数字（向量），语义相似的文本在向量空间中距离近
   - 选型指南：
     - **OpenAI text-embedding-3-small：** 简单好用，按量付费，维度 1536
     - **BGE-M3（开源）：** 支持多语言、多粒度，可本地部署，免费
     - **Jina Embeddings v3：** 长文本能力强，有免费额度
   - 维度越高不一定越好，要平衡效果和存储成本
   - 向量需要存入数据库才能高效检索（不能每次调 API 重新算）

2. **相似度计算**
   - **余弦相似度（Cosine）：** 最常用，只看方向不看大小，范围 [-1, 1]
   - **欧氏距离（Euclidean）：** 看绝对距离，对向量长度敏感
   - **内积（Dot Product）：** 计算最快，但要注意向量是否已归一化
   - 实际中余弦相似度用得最多，大部分向量库都支持

3. **向量数据库**
   - **ChromaDB：** 最简单的入门选择，纯 Python，嵌入式，适合开发和小规模项目
     - `chromadb.Client()` 创建客户端
     - `collection.add()` / `collection.query()` 增删查
     - 支持 metadata 过滤
   - **FAISS：** Meta 开源，性能极强，适合大规模数据，但只是索引不是数据库
   - **PGVector（⭐ 企业首选）：** PostgreSQL 的向量扩展
     - 优势：零额外运维、事务一致性、真正的业务+向量混合查询
     - 在已有 PostgreSQL 上 `CREATE EXTENSION vector` 即可使用
     - 一条 SQL 同时做向量搜索 + 业务条件过滤 + 排序
     - 索引选型：HNSW（推荐，查询快）、IVFFlat（构建快）
   - 其他生产级选项：Pinecone（托管）、Weaviate、Qdrant、Milvus（自建）
   - 选择建议：学习用 ChromaDB，生产优先 PGVector，大规模独立向量库选 Qdrant

4. **分块策略（Chunking）**
   - **固定长度切分：** 按 token 数或字数切，简单但可能切断语义
   - **语义切分：** 按段落、标题、句子边界切，保留语义完整性
   - **重叠窗口：** 相邻块之间重叠 10-20%，避免边界信息丢失
   - 分块大小建议：500-1500 token，太大检索不精确，太小上下文不够
   - 进阶：父文档检索（先检索小块，返回大块上下文）

5. **元数据过滤与混合检索**
   - 给每个 chunk 附加元数据：来源文件、创建时间、文档类型、章节
   - 检索时先过滤元数据再向量搜索，缩小范围提升精度
   - **混合检索：** 向量搜索 + 关键词搜索（BM25）结合，取两者优点
   - 实际场景中混合检索效果通常优于纯向量搜索

**建议：**
- 先用 ChromaDB 跑通整个流程，再考虑其他方案
- 不同分块策略对效果影响很大，多尝试几种并对比
- 用可视化工具（如 TensorBoard）看看 embedding 的分布，建立直觉

**常见坑：**
- 不要一上来就选最复杂的向量库，ChromaDB 足够学习和开发
- 分块太小会导致检索到碎片信息，太大会导致包含太多无关内容
- Embedding 模型和向量库要匹配，不同模型的向量不能混用

**实践项目：** 把一批文档切块后存入 ChromaDB，做语义搜索
- 准备 20-50 篇技术文档（Markdown/HTML）
- 实现至少两种分块策略并对比效果
- 支持按元数据过滤（如只搜某个来源的文档）
- 输出检索结果的相似度分数

**推荐资源：**
- ChromaDB 官方文档（docs.trychroma.com）
- Pinecone 学习中心（pinecone.io/learn）
- "Chunking Strategies for LLM Applications"（LlamaIndex 博客）

---

### 第 5 周 — RAG 应用开发

**目标：** 搭建完整的 RAG（检索增强生成）系统

**学习清单：**
- [ ] RAG 完整流程：文档加载 → 分块 → Embedding → 检索 → 生成
- [ ] LangChain / LlamaIndex 框架入门
- [ ] 文档加载器：PDF、Markdown、网页、数据库
- [ ] 检索优化：HyDE、多查询、重排序（Reranker）
- [ ] Dense-Sparse 混合检索（BM25 + 向量检索）
- [ ] GraphRAG：基于知识图谱的全局检索（解决"总结全书"类问题）
- [ ] 评估 RAG 质量：忠实度、相关性、召回率

**详细学习内容：**

1. **RAG 全流程梳理**
   ```
   用户提问
     ↓
   Query 改写（可选）
     ↓
   向量检索 → Top-K 相关文档块
     ↓
   组装 Prompt：系统提示 + 检索到的上下文 + 用户问题
     ↓
   LLM 生成回答（基于检索到的上下文）
     ↓
   返回答案 + 引用来源
   ```
   - 核心思想：让模型基于你的数据回答，而不是靠训练时的知识
   - RAG 解决的问题：知识过时、领域专有知识、减少幻觉

2. **框架选择**
   - **LangChain：** 生态最大，组件最全，但抽象层多，学习曲线较陡
     - 适合：快速原型、组件丰富、社区支持好
   - **LlamaIndex：** 专注 RAG 和数据索引，文档清晰
     - 适合：复杂 RAG 场景、数据连接器多
   - **纯手写：** 不用框架，直接调 API + 向量库
     - 适合：学习原理、完全掌控、避免框架锁定
   - 建议：先用框架快速跑通，理解原理后再考虑纯手写

3. **文档加载器**
   - PDF：`PyPDFLoader`（简单）、`UnstructuredPDFLoader`（复杂布局）
   - Markdown：`UnstructuredMarkdownLoader`，保留标题结构
   - 网页：`WebBaseLoader`（BeautifulSoup）、`PlaywrightLoader`（JS 渲染）
   - 数据库：SQL Loader、MongoDB Loader
   - 核心问题：不同格式的文档结构不同，加载质量直接影响后续效果

4. **检索优化技巧**
   - **HyDE（Hypothetical Document Embeddings）：** 先让 LLM 生成假设性答案，用假设答案去检索
   - **多查询改写：** 将用户问题改写成多个不同角度的查询，合并检索结果
   - **Reranker（重排序）：** 第一轮检索取 Top-20，用 Reranker 精排取 Top-5
     - 推荐：Cohere Rerank、BGE-reranker（开源）
   - **查询路由：** 根据问题类型决定走向量检索还是关键词搜索还是直接回答
   - **上下文压缩：** 对检索到的长文本做摘要，只保留和问题相关的部分

5. **Dense-Sparse 混合检索**
   - **Dense（稠密检索）：** 向量搜索，语义匹配，理解同义词和上下文
   - **Sparse（稀疏检索）：** BM25 关键词搜索，精确匹配专有名词和术语
   - 为什么混合：向量搜索理解语义但可能忽略精确关键词；BM25 精确但不理解同义词
   - 实现方式：两种检索各取 Top-K，用 RRF（Reciprocal Rank Fusion）合并排序
   - 混合检索是目前工业界最成熟、效果最稳的方案

6. **GraphRAG（知识图谱增强的 RAG）**
   - 解决什么问题：向量检索擅长"找相关段落"，但遇到"请总结全书核心观点"这类全局性问题会抓瞎
   - 原理：先用 LLM 从文档中提取实体和关系，构建知识图谱；查询时结合图谱的全局结构做推理
   - 微软 GraphRAG：Build Community Summaries → 逐层聚合回答全局问题
   - LightRAG：轻量级替代，更快更便宜
   - 适用场景：需要跨文档推理、全局总结、发现隐藏关联
   - 不适合：简单的事实检索（常规 RAG 足够）、实时性要求高（图谱构建慢）

5. **RAG 质量评估**
   - **忠实度（Faithfulness）：** 回答是否基于检索到的文档，没有编造
   - **相关性（Relevancy）：** 检索到的文档是否和问题相关
   - **召回率（Recall）：** 是否找到了所有相关的信息
   - **答案正确性：** 和标准答案对比
   - 评估工具：RAGAS 框架（自动化评估）、人工标注（金标准）

**建议：**
- 框架不是目的，理解 RAG 流程才是。即使不用框架，也能手写出来
- 多做 A/B 测试：同一个问题，不同配置下效果差异很大
- 关注"检索失败"的 case，通常比"成功"的 case 更有学习价值

**常见坑：**
- 不要把所有文档一股脑塞进去，质量比数量重要
- 分块粒度和检索 Top-K 要配合调优，不是越大越好
- RAG 不是万能的，有些问题用 Agent + 工具更合适

**实践项目：** 搭建一个"企业知识库问答"系统，支持上传文档和对话查询
- 支持上传 PDF / Markdown 文档，自动入库
- 对话式查询，支持多轮追问
- 回答中引用来源文档和具体段落
- 记录检索失败的情况，给出"我不知道"的诚实回答

**推荐资源：**
- LangChain 官方教程（python.langchain.com）
- LlamaIndex 官方文档（docs.llamaindex.ai）
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"（原始论文，可选读）

---

### 第 6 周 — Function Calling 与 Tool Use

**目标：** 让 LLM 能调用外部工具，突破"只会聊天"的限制

**学习清单：**
- [ ] Function Calling 原理与协议（OpenAI / Anthropic 格式）
- [ ] 工具设计模式：参数校验、错误处理、结果摘要
- [ ] 常用工具集：搜索、数据库查询、代码执行、文件操作
- [ ] MCP (Model Context Protocol) 基础
- [ ] 安全边界：权限控制、沙箱执行

**详细学习内容：**

1. **Function Calling 原理**
   - 流程：用户提问 → 模型判断需要调用工具 → 返回工具调用请求 → 你执行工具 → 将结果返回模型 → 模型生成最终回答
   - OpenAI 格式：在 `tools` 参数中定义函数的 name、description、parameters（JSON Schema）
   - Anthropic 格式：`tools` 参数格式类似，但响应中用 `tool_use` 块标识
   - 关键认知：模型不直接执行函数，它只决定"调什么"和"传什么参数"，执行在你这边

2. **工具设计模式**
   - **命名清晰：** 工具名和描述要让模型理解"这个工具能干什么"
   - **参数简单：** 尽量用扁平参数，避免深层嵌套的 JSON Schema
   - **返回结构化：** 工具返回 JSON 格式的结果，方便模型理解和总结
   - **错误友好：** 工具执行失败时返回有意义的错误信息，让模型能做出合理响应
   - **结果截断：** 大量结果时分页或摘要，避免撑爆上下文窗口

3. **常用工具集**
   - **Web 搜索：** SerpAPI、Tavily、Brave Search API
   - **数据库查询：** SQL 查询执行器（带只读权限）
   - **代码执行：** 沙箱中的 Python 解释器（E2B、Docker 沙箱）
   - **文件操作：** 读写文件、目录列表（限权限）
   - **API 调用：** HTTP 请求、第三方服务集成
   - **数学计算：** 安全的表达式求值器

4. **MCP（Model Context Protocol）**
   - Anthropic 推出的开放协议，标准化模型与外部工具/数据源的连接方式
   - 类比：USB 接口之于设备，MCP 之于 AI 工具
   - 架构：MCP Server（提供工具）←→ MCP Client（模型端调用）
   - 已有生态：文件系统、GitHub、数据库、浏览器等 MCP Server
   - 学会配置和使用已有的 MCP Server，了解如何自建

5. **安全边界**
   - 权限最小化：工具只给必要的权限，数据库只读，文件限目录
   - 沙箱执行：代码执行必须在隔离环境中，防止系统被破坏
   - 输入验证：对模型生成的参数做校验，不要直接信任
   - 审计日志：记录所有工具调用，便于追溯和调试

**建议：**
- 从简单的工具开始（计算器、搜索），再逐步增加复杂度
- 重点打磨"工具描述"，模型是否选对工具完全取决于你的描述写得好不好
- 调试技巧：打印模型返回的完整响应，观察它是怎么"思考"的

**常见坑：**
- 工具描述写得模糊，模型会选错工具或传错参数
- 不做参数校验，模型偶尔会传入不合法的值
- 忘记处理"模型没有调用工具"的情况，要加兜底逻辑

**实践项目：** 构建一个能查询天气、搜索网页、操作文件的助手
- 注册 3-5 个工具（天气 API、搜索 API、文件读写、计算器）
- 支持工具链式调用（先搜索，再对结果做分析）
- 处理工具执行失败的情况，让模型给出友好的错误提示
- 记录完整的工具调用链，方便调试

**推荐资源：**
- OpenAI Function Calling 文档（platform.openai.com/docs/guides/function-calling）
- Anthropic Tool Use 文档（docs.anthropic.com/en/docs/build-with-claude/tool-use）
- MCP 官方文档（modelcontextprotocol.io）

---

### 第 7 周 — AI Agent 开发

**目标：** 构建能自主规划、执行、反思的 Agent 系统

**学习清单：**
- [ ] Agent 架构：ReAct、Plan-and-Execute、Reflection
- [ ] 短期记忆 vs 长期记忆管理
- [ ] 多工具协同与任务分解
- [ ] Agent 框架：LangGraph / CrewAI / AutoGen
- [ ] 调试 Agent：可观测性、Trace 追踪（LangSmith）

**详细学习内容：**

1. **Agent 核心架构**
   - **ReAct（Reasoning + Acting）：** 最经典的 Agent 模式
     - 循环：思考（Thought）→ 行动（Action）→ 观察（Observation）→ 再思考...
     - 直到模型判断任务完成，输出最终答案
     - 简单有效，适合大多数场景
   - **Plan-and-Execute：** 先制定计划，再逐步执行
     - 适合复杂任务：先拆解成子任务，再逐个完成
     - 中间可调整计划（Re-planning）
   - **Reflection：** 执行后自我评估，发现问题则重新执行
     - 类似"写完后自己检查一遍"
     - 适合代码生成、内容创作等需要质量的场景

2. **记忆管理**
   - **短期记忆（对话历史）：** 维护在上下文窗口中，随对话滚动
     - 问题：上下文窗口有限，长对话会溢出
     - 解决：摘要压缩、滑动窗口、关键信息提取
   - **长期记忆（持久化存储）：** 存到向量数据库或文件中
     - 按需检索：需要时从长期记忆中检索相关内容注入上下文
     - 记忆分类：用户偏好、历史决策、知识积累
   - 实际工程中，短期和长期记忆需要配合使用

3. **多工具协同**
   - 任务分解：把复杂任务拆成多个子任务，每个子任务可能需要不同工具
   - 依赖管理：某些任务需要等前置任务完成（如：先搜索，再分析，再写报告）
   - 并行执行：无依赖的任务可以同时执行，提升效率
   - 结果汇总：将多个工具的输出整合成最终回答

4. **Agent 框架**
   - **LangGraph：** LangChain 生态，用有向图定义 Agent 流程
     - 节点（Node）= 步骤，边（Edge）= 流转条件
     - 优势：可视化流程、支持人机协作、状态管理清晰
     - 推荐作为首选框架，学习曲线合理
   - **CrewAI：** 多 Agent 协作框架
     - 定义多个 Agent（各有角色和工具），协同完成任务
     - 适合：需要多个专业角色配合的场景
   - **AutoGen：** 微软出品，Agent 之间可以对话协商
   - 建议：先学 LangGraph，它最接近底层原理

5. **调试与可观测性**
   - Agent 比普通程序更难调试，因为行为有随机性
   - **LangSmith：** LangChain 官方的 Trace 工具
     - 记录每次 LLM 调用、工具调用的完整链路
     - 可以回放 Agent 的"思考过程"
   - 打印中间步骤：在开发阶段，把每一步的 Thought/Action/Observation 打出来
   - 设定最大循环次数：防止 Agent 陷入死循环

**建议：**
- Agent 开发是"组合技"，前几周的基础全在这里用上
- 从简单 Agent 开始（单工具 + ReAct），再逐步增加复杂度
- Agent 的行为不可预测，一定要加安全边界和循环上限

**常见坑：**
- 不设最大循环次数，Agent 可能陷入无限循环消耗 token
- 工具太多反而让模型"选择困难"，控制在 5-10 个以内
- 不观察中间步骤就调试，出了问题完全不知道错在哪

**实践项目：** 构建一个能拆解复杂任务并自动执行的 Agent
- 任务示例："帮我调研 XX 技术栈，整理成一份对比报告"
- Agent 需要：搜索信息 → 整理结构 → 写报告 → 自我检查
- 支持中断和人机协作（在关键步骤让人类确认）
- 展示完整的执行链路（每一步的思考和行动）

**推荐资源：**
- LangGraph 官方教程（langchain-ai.github.io/langgraph）
- "Building Effective Agents"（Anthropic 工程博客，强烈推荐）
- CrewAI 文档（docs.crewai.com）

---

## 阶段三：AI 工程化落地（第 8-10 周）

> **阶段目标：** 把 AI 能力包装成可交付、可部署、可评估的工程产品。掌握 AI 应用从 demo 到生产的全链路。
> **阶段心法：** 写出能跑的 demo 容易，写成能上线的 AI 服务难。这三周重点补 AI 工程化能力——模型调用治理、效果评估、生产部署。

---

### 第 8 周 — AI 应用工程化：服务封装与模型治理

**目标：** 把 AI 能力包装成可交付的 HTTP 服务，掌握 AI 应用的工程化模式

**学习清单：**
- [ ] FastAPI 封装 AI 服务：路由、流式响应（SSE）、异步任务
- [ ] 模型治理：多模型路由、Fallback 降级、重试策略
- [ ] AI 可观测性：LLM 调用链路追踪、Token 用量监控、成本告警
- [ ] 缓存策略：语义缓存、Embedding 缓存、响应缓存
- [ ] 容器化部署：Docker + docker-compose 一键启动

**详细学习内容：**

1. **FastAPI 封装 AI 服务**
   - 为什么选 FastAPI：异步支持好、自动文档（Swagger）、类型校验、性能高
   - 基础三件套：路由（`@app.post`）、请求体（Pydantic Model）、响应模型
   - **流式响应（SSE）：** AI 应用的标配，用户不想等 10 秒才看到第一个字
     - `StreamingResponse` + `text/event-stream` 实现
     - 前端对接：`EventSource` API 或 `fetch` + `ReadableStream`
   - **异步任务处理：** 文档解析、批量 Embedding 不能阻塞请求
     - `BackgroundTasks`：简单场景
     - Celery + Redis：生产级方案，支持任务队列、重试、状态追踪

2. **模型治理（AI 工程化核心）**
   - **多模型路由：** 根据任务类型自动选择模型
     - 简单问题 → 便宜模型（Haiku/GPT-4o-mini）
     - 复杂推理 → 强模型（Claude Opus/GPT-4o）
     - 代码任务 → 代码专用模型
   - **Fallback 降级：** 主模型不可用时自动切换备选模型
     - OpenAI 超时 → 降级到 Claude → 降级到本地模型
     - 关键：降级后的效果要能接受，不是随便降
   - **重试策略：** 指数退避重试，处理速率限制和临时故障
   - **统一抽象层：** 用 LiteLLM 或自建 Wrapper 统一不同模型的调用接口

3. **AI 可观测性**
   - **为什么重要：** AI 应用是"黑盒"，不追踪就不知道内部发生了什么
   - **LLM 调用链路追踪：**
     - LangSmith：自动追踪 LangChain 应用的每一步调用
     - OpenLLMetry / Langfuse：开源的 LLM 可观测性平台
     - 记录内容：输入 Prompt、输出结果、耗时、Token 数、模型名
   - **Token 用量监控：** 按用户、按功能、按时间段统计 Token 消耗
   - **成本告警：** 单日 Token 消耗超过阈值时自动通知
   - **质量监控：** 用户踩/举报率突增时告警

4. **缓存策略（降本增效关键）**
   - **精确缓存：** 完全相同的输入直接返回缓存结果
     - 适合：FAQ 类问题、固定模板的生成任务
   - **语义缓存：** 语义相似的问题返回缓存结果
     - 实现：对用户问题做 Embedding → 查向量缓存 → 相似度 > 阈值则命中
     - 工具：GPTCache、Redis Vector Cache
   - **Embedding 缓存：** 缓存文档的 Embedding 向量，避免重复计算
   - **Prompt 缓存：** 相同的 system prompt + 历史上下文复用之前的结果
   - 缓存能大幅降低成本（常见问题可能节省 30-50% 的 API 调用）

5. **Docker 容器化**
   - Dockerfile：基础镜像选择、依赖安装、启动命令
   - docker-compose：多服务编排（应用 + Redis + 向量数据库 + 缓存）
   - 镜像优化：多阶段构建、`.dockerignore`、slim 基础镜像
   - 环境变量注入：通过 `.env` 或 docker-compose `environment` 配置

**建议：**
- 模型治理和可观测性是 AI 工程化的核心区别于普通后端的地方，重点学
- 从项目第一天就加 Token 追踪，不要等账单爆了才加
- 缓存策略是"低投入高回报"的优化，尽早引入

**常见坑：**
- 同步的 LLM 调用阻塞了异步 FastAPI，需要用 `run_in_executor` 包装
- 不做 Fallback，主模型一挂整个服务就不可用
- 语义缓存的相似度阈值设置不当，导致返回不相关的缓存结果
- 只缓存最终结果不缓存中间结果（如 Embedding），优化效果有限

**实践项目：** 将 RAG 系统封装为完整的 AI 服务
- `POST /chat` 接口：支持普通和流式两种模式
- `POST /documents/upload` 接口：上传文档并自动入库
- 模型路由：简单问题用便宜模型，复杂问题用强模型
- Token 追踪：记录每次调用的 Token 消耗和成本
- 基础语义缓存：相似问题命中缓存
- Docker 化，`docker-compose up` 一键启动

**推荐资源：**
- FastAPI 官方教程（fastapi.tiangolo.com）
- Langfuse 文档（langfuse.com）— 开源 LLM 可观测性
- GPTCache（github.com/zilliztech/GPTCache）— 语义缓存
- LiteLLM 文档（docs.litellm.ai）— 统一模型调用

---

### 第 9 周 — AI 效果评估与 Prompt 工程化

**目标：** 建立数据驱动的 AI 开发习惯，用量化指标指导 Prompt 迭代和模型调优

**学习清单：**
- [ ] 数据集构建与标注方法
- [ ] LLM 评估框架：promptfoo、RAGAS、DeepEval
- [ ] Prompt 工程化：版本管理、A/B 测试、自动化回归测试
- [ ] AI 应用特有的测试策略（非确定性输出的测试方法）
- [ ] 成本优化实战：模型降级、Token 精简、批量处理

**详细学习内容：**

1. **数据集构建**
   - 从真实用户场景中收集数据，不要凭空造数据
   - 标注方法：人工标注、模型辅助标注（先让模型标，人工校对）
   - 数据格式：Question-Context-Answer 三元组
   - 数据质量 > 数量：100 条高质量数据 > 1000 条噪声数据
   - 数据集分层：训练集 / 验证集 / 测试集（或开发集 / 评估集）

2. **LLM 评估框架**
   - **promptfoo：** 命令行评估工具，配置简单，支持多模型对比
     - 定义 prompt + 测试用例，自动跑评估并输出对比报告
     - 特别适合：Prompt 迭代时快速对比新旧版本效果
   - **RAGAS：** 专门评估 RAG 系统
     - 四大指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall
   - **DeepEval：** 通用 LLM 评估框架
     - 内置多种指标：G-Eval、Hallucination、Answer Relevancy、Summarization
   - 评估方式：自动化指标 + 人工评审，两者结合最可靠

3. **Prompt 工程化（AI 工程化核心）**
   - **Prompt 版本管理：** 每次改 Prompt 都要有记录，和代码一样做 Git 管理
     - Prompt 文件独立管理（`prompts/` 目录），用 Jinja2 模板化
     - 改动记录：改了什么、为什么改、效果变化（前后指标对比）
   - **Prompt A/B 测试：** 同一批测试用例，跑两个版本的 Prompt，量化对比
     - 工具：promptfoo、自建评估脚本
     - 指标：准确率、响应格式合规率、Token 消耗、延迟
   - **Prompt 回归测试：** 改了 Prompt 后自动跑测试集，确保没有"改好一个搞坏三个"
     - 集成到 CI/CD：每次提交 Prompt 变更自动跑评估
   - **Prompt 优化策略：**
     - 从失败 case 中找规律，针对性改进
     - 控制变量：一次只改一个因素（温度、措辞、示例数量）
     - 用量化的指标替代主观判断

4. **AI 应用特有的测试策略**
   - AI 应用的输出是非确定性的，传统 `assertEqual` 不适用
   - **结构化输出测试：** 如果要求返回 JSON，验证 JSON 格式和必需字段
   - **语义相似度测试：** 用 Embedding 比较输出和期望答案的语义距离
   - **LLM-as-Judge：** 用另一个 LLM 评判输出质量（评分、分类）
   - **模糊匹配：** 关键词包含、正则匹配、长度范围检查
   - **护栏测试：** 测试模型是否会被 Prompt Injection 攻击、是否泄露 system prompt
   - 最佳实践：分层测试 — 底层用确定性测试（格式、字段），上层用 LLM 评估（质量、相关性）

5. **成本优化实战**
   - Token 用量追踪：记录每次调用的 input/output token
   - 缓存策略：相同问题缓存回答，避免重复调用 API（语义缓存更优）
   - 模型降级：简单问题用便宜模型（Haiku/GPT-4o-mini），复杂问题用强模型
   - Prompt 精简：去掉不必要的上下文，减少 token 消耗
   - 批量处理：非实时任务合并请求，减少 API 调用次数

**建议：**
- 从项目第一天起就积累评估数据集，不要等项目做完了才想起来
- **每次改 Prompt 后必须跑评估**，这是 AI 工程化和"随便调调"的根本区别
- 成本意识很重要，一个小优化可能每月省几百美元
- 重点关注"差 case"，它们比平均指标更有指导意义

**常见坑：**
- 用训练数据评估，效果虚高，必须用独立的测试集
- 只看平均指标，忽略了"长尾"的差 case
- 不追踪成本，上线后发现 API 账单爆表
- Prompt 改了不记录，过两周忘了为什么这么改

**实践项目：** 给前面的项目搭建完整的评估和 Prompt 管理体系
- 构建 50+ 条测试用例（覆盖常见和边界情况）
- 用 promptfoo / RAGAS 跑自动评估
- 对比 2-3 个 Prompt 版本的效果（A/B 测试）
- 生成评估报告（指标 + 具体 case 分析）
- 集成到 CI 中，每次改 Prompt 自动跑评估

**推荐资源：**
- promptfoo 官方文档（promptfoo.dev）
- RAGAS 文档（docs.ragas.io）
- "Evaluating RAG Pipelines"（LlamaIndex 博客）

---

### 第 10 周 — 部署与运维

**目标：** 让应用跑在生产环境，能被真实用户使用

**学习清单：**
- [ ] 部署方案选型：Vercel、Railway、AWS、自建
- [ ] CI/CD 基础：GitHub Actions 自动化
- [ ] 监控告警：日志、指标、异常追踪
- [ ] 扩展策略：负载均衡、队列、缓存层（Redis）
- [ ] 安全实践：输入过滤、Prompt Injection 防御、数据脱敏

**详细学习内容：**

1. **部署方案选型**
   - **Vercel / Railway：** 最简单，适合小项目和原型，push 即部署
   - **AWS（ECS/Lambda）：** 弹性强，适合中大规模，学习曲线陡
   - **自建（VPS + Docker）：** 完全掌控，适合对成本敏感的场景
   - 选型建议：学习阶段用 Railway/Fly.io，生产环境按需选 AWS/GCP

2. **CI/CD（持续集成 / 持续部署）**
   - GitHub Actions 入门：触发条件、Job、Step、Runner
   - 基础 Pipeline：代码检查 → 单元测试 → 构建镜像 → 部署
   - 环境管理：开发 / 预发布 / 生产 环境隔离
   - 自动化的好处：减少人为错误、加速迭代、代码即基础设施

3. **监控告警**
   - **日志：** 结构化日志（JSON），集中收集（ELK / Loki）
   - **指标：** 请求量、延迟、错误率、Token 用量（Prometheus + Grafana）
   - **异常追踪：** Sentry 自动捕获和报告异常
   - **AI 特有指标：** 平均响应时间、Token 消耗趋势、检索命中率
   - 告警规则：错误率突增、延迟飙升、API 额度接近上限

4. **扩展策略**
   - 水平扩展：多实例 + 负载均衡（Nginx / 云 LB）
   - 异步队列：Redis + Celery 处理高峰期请求
   - 缓存层：Redis 缓存频繁查询的结果，减少 API 调用
   - 数据库优化：向量数据库索引调优、分片策略

5. **安全实践**
   - **输入过滤：** 清理用户输入，防止注入攻击
   - **Prompt Injection 防御：**
     - 输入检测：识别和过滤恶意提示
     - 隔离策略：用户输入和系统提示严格分离
     - 输出过滤：检查模型输出是否泄露了系统提示
   - **数据脱敏：** 日志中不记录敏感信息（PII、API Key）
   - **速率限制：** 防止滥用，per-user / per-IP 限流
   - **安全 Headers：** CORS、CSP、HSTS

**建议：**
- 部署不是"最后一步"，尽早部署，尽早发现环境问题
- 监控和日志要提前配置好，不要等出问题才加
- 安全意识从第一天培养，AI 应用的攻击面比传统应用更大

**常见坑：**
- 本地能跑但部署失败，通常是环境变量或路径问题
- 不做健康检查，服务挂了都不知道
- API Key 泄露到前端代码或日志中

**实践项目：** 将应用部署到云平台，配置 CI/CD 和监控
- 选择一个云平台部署你的 AI 应用
- 配置 GitHub Actions：push 到 main 自动测试 + 部署
- 配置基础监控：请求日志、错误率、Token 用量
- 配置至少一个告警规则（如：错误率 > 5% 时通知）

**推荐资源：**
- GitHub Actions 官方文档（docs.github.com/en/actions）
- "Deploying LLM Apps to Production"（Chip Huyen 博客）
- OWASP LLM Top 10（安全最佳实践）

---

## 阶段四：综合实战（第 11-12 周）

> **阶段目标：** 完成一个完整的作品集项目，串联前 10 周所学。
> **阶段心法：** 这是"毕业设计"，不是练习。要按真实产品的标准来做。

---

### 第 11-12 周 — 毕业项目

**目标：** 完成一个完整作品集项目，展示你的 AI 工程能力

**选择一个方向深入：**

| 项目 | 核心技术栈 | 亮点 |
|------|-----------|------|
| 智能客服系统 | RAG + 多轮对话 + 工单流转 | 知识库管理、意图识别、人机协作 |
| AI 代码助手 | 代码理解 + 生成 + Review | 代码 Embedding、AST 解析、Diff 分析 |
| 文档智能平台 | 多格式解析 + 结构化提取 + 知识图谱 | OCR、表格提取、跨文档推理 |
| 自动化工作流 Agent | 多 Agent 协作 + 外部工具集成 + 定时任务 | Agent 编排、状态机、调度系统 |

**项目要求：**
- [ ] 完整的项目文档和 README（别人能看懂并复现）
- [ ] 前后端分离或全栈实现
- [ ] Docker 化部署，`docker-compose up` 一键启动
- [ ] 评估指标和测试覆盖（至少有自动化测试）
- [ ] 公开分享：GitHub 开源 + 至少一篇技术博客

**开发流程建议：**

1. **第 11 周前半：设计与核心实现**
   - 明确功能范围（MVP，不要贪多）
   - 画系统架构图（数据流、组件关系）
   - 实现核心链路（先把主流程跑通）
   - 每天 commit，保持增量开发

2. **第 11 周后半：完善与测试**
   - 补充边界处理和错误处理
   - 写自动化测试（核心路径至少覆盖）
   - 优化 Prompt 和检索效果
   - 做一轮评估，对比优化前后的效果

3. **第 12 周前半：部署与打磨**
   - 部署到云平台
   - 配置监控和日志
   - 邀请 2-3 个人试用，收集反馈
   - 根据反馈迭代

4. **第 12 周后半：文档与分享**
   - 写 README：项目介绍、架构图、快速开始、功能演示
   - 写技术博客：选一个技术点深入讲（如 RAG 优化、Agent 调试）
   - 录一个 Demo 视频（2-3 分钟）
   - 准备技术分享（可以发到掘金、知乎、Twitter）

**建议：**
- MVP 思维：先做一个能用的最小版本，再逐步增加功能
- 记录开发过程中的坑和解决方案，这本身就是最好的学习笔记
- 不要追求完美，"完成"比"完美"重要

---

## 推荐学习资源汇总

### 书籍
- 《Build a Large Language Model (From Scratch)》— Sebastian Raschka（理解原理）
- 《Hands-On Large Language Models》— Jay Alammar（实操向）
- 《Designing Machine Learning Systems》— Chip Huyen（工程化思维）

### 在线课程
- DeepLearning.AI: ChatGPT Prompt Engineering for Developers
- DeepLearning.AI: Building Systems with the ChatGPT API
- DeepLearning.AI: AI Agentic Design Patterns with AutoGen
- Full Stack LLM Bootcamp（免费）
- LLM Engineering Bootcamp（GitHub 上有免费版）

### 实践平台
- OpenAI Cookbook（cookbook.openai.com）
- Anthropic Prompt Engineering Guide（docs.anthropic.com）
- Hugging Face 教程（huggingface.co/learn）
- LangChain / LlamaIndex 官方教程

### 社区与信息源
- GitHub Trending（关注 AI 项目）
- r/LocalLLaMA（Reddit，开源模型社区）
- AI Twitter/X 圈子（关注 @kaboroevich、@simonw、@haboroevich）
- Hacker News AI 板块
- 国内：掘金 AI 专栏、知乎 AI 话题

### 每周必做习惯
- 浏览 2-3 个 AI 开源项目，看别人的代码和架构
- 关注 AI 领域新闻（模型发布、框架更新、行业动态）
- 写学习笔记（不用发布，给自己看就行）

---

## 每周时间安排建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 工作日晚上 | 概念学习 + 编码练习 | 每天 1-1.5h |
| 周末上午 | 实践项目开发 | 3-4h |
| 周末下午 | 复盘总结 + 写笔记 | 1-2h |

**每周最低投入：8-12 小时**

### 进度检查点
- **第 1 周结束：** 拥有一个规范的 AI 项目脚手架，会管理 Prompt 版本和模型配置
- **第 3 周结束：** 能独立调用 OpenAI/Claude API，写出完整的多轮对话应用，会成本追踪
- **第 7 周结束：** 能搭建 RAG 系统 + Agent，掌握核心 AI 应用架构模式
- **第 10 周结束：** 能将 AI 应用部署到生产环境，有模型治理、效果评估、可观测性
- **第 12 周结束：** 有 1 个完整作品集项目 + GitHub + 技术博客
