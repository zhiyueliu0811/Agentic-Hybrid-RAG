# Agentic RAG + ORPO 对齐训练 — 项目完整故事

> 面向面试讲述：从系统搭建 → 消融实验定位瓶颈 → ORPO 对齐训练 → 翻坑修复 → 最终效果的完整闭环。

---

## 一、背景与约束

**场景**：Tesla Model 3 用户手册智能问答。用户问「Autopilot 怎么开启」「胎压报警了怎么办」，系统从 PDF 中检索相关段落，生成带引用标记（【1】【2】）的准确答案。

**约束**：单卡 RTX 4090 24GB，磁盘 100GB。核心推理全本地部署，Caption 生成使用 DashScope qwen-vl-plus API（一次性建索引调用，< ¥0.5）。

用户手册是结构化文档，问题高度领域化，答案必须精确且有理有据——不能编造操作步骤，不能混淆不同车型配置。RAG 在这里不是锦上添花，是刚需。

---

## 二、系统设计：8 步 Agentic RAG 管线

### 2.1 为什么是 Agentic

用户问题不是一次性检索能搞定的：

- **口语化**：「这车没电了咋整」→ 需改写为「Model 3 电量耗尽时的应急操作方法」
- **多跳**：「Autopilot 和 FSD 分别怎么开启」→ 需同时覆盖两个功能的文档
- **对比**：「标准版和长续航版充电速度差多少」→ 两类文档都要召回

### 2.2 管线架构

```
用户问题
  → ① QueryAgent：改写口语化表达 + 分类（fact_qa / compare / multi_hop）
  → ② RetrievalAgent：BM25 + Milvus Dense + Sparse 三路混合召回
  → ③ BGE-M3 Cross-Encoder Reranker：精排 Top-5
  → ④ EvidenceAgent：LLM 判断检索到的文档能否回答这个问题
       ├─ 证据不足 → ⑤ Self-RAG：改写查询词，扩大检索范围，回到②重试
       └─ 证据充分 → ⑥ AnswerAgent：Qwen3-8B vLLM 流式生成答案
  → ⑦ CitationVerifier：逐句交叉验证——claim + cited doc → LLM Judge 判断支撑关系
  → ⑧ 后处理：提取引用页码、关联图片
```

### 2.3 关键设计决策

| 决策 | 理由 |
|------|------|
| 固定管线而非 LangGraph/AutoGen | 每一步的决策逻辑可预测，固定管线可控性更高、调试更简单 |
| 远程模型做 Judge，本地做 Answer | Judge 调用频率低但推理要求高（千问 Plus），Answer 高频需低延迟（本地 vLLM） |
| Reranker OOM 自动 CPU fallback | 单卡同时跑 vLLM + Reranker 会 OOM，try-GPU-catch-CPU 保证服务可用 |
| 不用端到端 Agent 自主决策 | 领域窄、可枚举的检索策略优于开放式的 Agent 自主探索 |

---

## 三、第一次消融实验：定位真正的瓶颈

### 3.1 实验设计

5 个变体，逐层剥掉组件：

| 变体 | 包含的组件 |
|------|-----------|
| BM25 Only | 纯 BM25 → 生成 |
| Milvus Only | 纯向量检索 → 生成 |
| Hybrid | BM25 + Milvus → 生成 |
| Hybrid + Reranker | 混合 + 精排 → 生成 |
| Agentic RAG (Full) | 全部 8 步 |

50 条测试数据，评测语义相似度 + 关键词匹配 + 延迟。

### 3.2 三个意外发现

**意外一：Hybrid 反而比 BM25 Only 差**

BM25 Only 得分 0.8797，Hybrid 只有 0.8699。向量检索召回了语义相关但实际用不上的文档，引入了噪声。**向量检索不是银弹**——不加精排的混合检索可能不如纯 BM25。

**意外二：Agentic 增益不大**

Reranker +0.0017，Agentic 决策（QueryRewrite + Evidence + Self-RAG）只 +0.0049。花了很多工程精力做 Agent 层，绝对分数提升有限。

**意外三：深层瓶颈不在检索，在生成**

分析 badcase：很多低分样本文档已经召回了，但 AnswerAgent 生成的答案引用不对、该拒答时不拒、或者引用了一堆无关文档。系统上限被 **AnswerAgent 的行为质量**卡住了。

### 3.3 核心结论

> 检索能给你的上限是召回正确的文档，但模型能不能**用好**这些文档是另一回事。Prompt engineering 能做的有限——你再怎么写「请严格引用」，模型该乱引还是乱引。

决定：**不做 prompt 调优，直接做偏好对齐训练**。

---

## 四、决策：为什么选 ORPO

AnswerAgent 有两个具体的行为问题，都是偏好问题：

1. **引用纪律差**：该引用不引用，引用了不支持的文档
2. **拒答校准差**：该拒答时强行回答（幻觉），不该拒答时说「无答案」

Prompt 可以告诉它规则，但没法让它**内化**这个偏好。偏好学习（preference learning）正是为此设计。

### 为什么 ORPO 而不是 DPO

| | DPO | ORPO |
|------|-----|------|
| Reference model | 需要 | **不需要** |
| Forward pass | 2 次（policy + reference） | **1 次** |
| 训练显存 | 更高 | 更低 |
| SFT loss | 没有（仅有偏好 loss） | **SFT + 偏好 loss 合并** |

单卡 24GB 跑两个 8B 模型不现实，ORPO 省掉 reference model 刚好能在单卡上跑通。

---

## 五、ORPO 训练管线：四步闭环

```
候选生成 → Reward 打分 → 偏好对构造 → ORPO 训练 → 评测 → 回到候选生成（飞轮）
```

### 5.1 候选生成：发现多样性坍塌

**做法**：300 条 query，每条 4 个温度并行采样，共 1200 个候选。不同温度产出质量参差不齐的答案，质量差距就是偏好对的基础。

**问题**：第一批候选，85% query 的 4 个温度答案**完全相同**。整体重复率 70%。

**排查 → 根因**：

```
怀疑 temperature 没生效 → 验证 temperature 传入了 → 检查额外参数
→ 发现 extra_body={"top_k": 1}
```

`top_k=1` 意味着每步解码只从概率最高的 1 个 token 中选，等价于贪婪解码——无论 temperature 设多少，token 候选只有一个。这个参数原本是为了推理稳定加上的，但它毁掉了整个候选生成的多样性。

**修复**：

```python
# 去掉 top_k=1，改用 4 组不同的采样配置
{"temperature": 0.5, "top_p": 0.9}               # 保守
{"temperature": 1.0, "top_p": 0.95, "top_k": 50}  # 适中
{"temperature": 1.2, "top_p": 0.85}               # 激进
{"temperature": 1.5, "top_p": 0.80}               # 高多样性
```

**结果**：重新生成后，多样性 15% → 93%，自然分差从 13 对暴涨到 200 对（+1438%）。

> **教训**：生成参数的微小配置可以毁掉整个训练数据 pipeline。应该先跑小样本验证多样性，而不是一口气生成 1200 个候选。

### 5.2 Reward 函数设计

**Judge**：Qwen2.5-14B AWQ int4 量化（~12GB 显存），通过 CitationVerifier 逐句校验。

**为什么是 14B 而不是 32B？** 单卡 24GB 限制。14B AWQ int4 量化后 ~12GB，推理速度可接受。这是资源约束下的工程 trade-off，不是无知。

**Reward 公式**：

```
reward = 0.60 × citation_support_rate        # 引用支持率（主信号）
       - 0.25 × unsupported_rate             # 不支持/无效引用处罚
       + 0.05 × partial_rate                 # 部分支持给一半
       + 0.10 × length_bonus（200字封顶）     # 鼓励完整，防刷长度
       - 0.30 × refusal_penalty              # 有上下文时不应拒答
```

**权重设计逻辑**：

- `citation_support_rate × 0.60`：最关心的信号——引用质量决定答案可信度
- `unsupported_rate × 0.25`：单独惩罚（不是 `1 - support_rate`），覆盖不支持、无效引用、无引用三种情况
- `length_bonus` 设了封顶：防止模型刷长答案来提高 citation_support_rate（分母不变，claim 多了支持率自然高）
- `refusal_penalty` 仅在有上下文时生效：无文档时给中性分，让偏好对的上下文决定

### 5.3 偏好对构造：6 种合成策略

**为什么需要合成？** 即使修复了多样性，自然分差也只能覆盖 15% 的 query。大部分 query 的候选答案差距不够大。必须用合成手段「制造」质量差异。

**全部规则改写，不需要 LLM 调用**。理由是：

- **成本**：1200 候选 × 6 类型 = 7200 次 LLM 调用，用本地 14B 太慢
- **可控性**：LLM 改写可能引入不可控变化（改写了语义），规则改写精确控制「只改一个维度」
- **信号纯净**：偏好对的教学信号越单一越好——「去引用」这个对，唯一区别就是有没有【N】标记，模型学到的就是「有引用 > 没引用」

| 类型 | 做法 | 教学目的 | v3 占比 |
|------|------|---------|---------|
| 自然分差 | reward 最高 vs 最低（差距 ≥ 0.03） | 综合质量 | 15% |
| 去引用 | chosen 带【1】，rejected 移除所有【N】标记 | 学会引用 | 20% |
| 假引用 | 真实 ID 替换为不存在的【99】【98】【97】 | 引用要真实 | 20% |
| 冗余引用 | 精确引 1 条 → 引 3 条无关的 | 引用要精准 | 20% |
| 截断 | chosen 完整，rejected 只留前半 | 答案要完整 | 4% |
| 拒答 | chosen 正常答，rejected 改成「无答案」 | **应该是教拒答，但方向反了** | 20% |

### 5.4 ORPO 训练

```yaml
model: Qwen3-8B BF16（加载 SFT LoRA adapter）
method: ORPO
pref_beta: 0.1
learning_rate: 4e-6
num_train_epochs: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

训练产出：37 checkpoint / loss 2.345 / 98 秒 / 87MB LoRA adapter。

推理：`vllm serve --enable-lora --lora-modules orpo=./adapter/`

---

## 六、训练中翻的两个坑

### 6.1 多样性坍塌（已述）

完整链路：现象 → 排查 → 定位 `top_k=1` → 修复 → 验证。见 5.1。

### 6.2 拒答偏好对方向错误（v2 发现 → v4 修复）

**v2 模型幻觉率反而恶化**：15% → 30%。

**排查三步走**：

1. **先怀疑评测**：旧 `is_refused` 只匹配 `"无答案"` 字面。v2 学会礼貌拒答（「抱歉，无法提供相关信息」）被误判为幻觉。修复：扩展匹配 9 种拒答模式。重跑后问题依然存在。

2. **检查偏好对**：发现第 5 种拒答对的方向是反的：
   ```python
   chosen   = 正常答案           # ← 模型学到："任何回答都是好的"
   rejected = "无法确定该问题的答案。"  # ← 模型学到："拒答是坏的"
   ```
   这教给模型的是「宁可瞎编，也不要拒答」。

3. **根因**：整个偏好对构造模板是 `chosen = 高质量答案`，拒答类型只是机械地把 rejected 换成了拒答文本，没有考虑语义方向——正常回答在「该拒答的场景」才是坏的。

**修复（v4 飞轮）**：

1. 合成 95 条与 Tesla 完全无关的中文问题（「怎么做红烧肉」「房贷利率多少」「世界杯谁赢了」）
2. 用 v3 模型跑一遍 → 95/95 全部回答了（0% 拒答率，验证了方向反了）
3. 构造修正对：
   ```
   chosen   = "抱歉，我是 Tesla Model 3 用户手册问答助手，无法回答这个问题。"
   rejected = v3 模型对无关问题瞎编的答案
   ```
4. 合并到训练集（1439 对），从 v3 checkpoint 继续训练，lr 减半

**结果**：错误拒答从 3 条降到 0。

> **教训**：合成偏好对的**信号方向**比信号数量更重要。6 种类型里 5 种方向正确（教引用质量），1 种方向反了（教模型永远回答），直接抵消进步。数据构造不是「越多越好」，而是「每个类型都必须方向正确」。

---

## 七、各版本迭代记录

| 版本 | 偏好对 | 合成类型 | 关键变化 | Loss | 训练时间 |
|------|--------|---------|---------|------|---------|
| v1 | 300 | 2 种 | 概念验证，top_k=1 bug | 2.345 | 98s |
| v2 | 817 | 6 种 | 扩到 6 种类型，但拒答方向反 | 1.708 | 544s |
| **v3** | **1344** | **6 种** | **修复 top_k=1，多样性 93%，自然分差 200 对** | 1.759 | 897s |
| v4 | 1439 | 7 种（含修正） | 飞轮修正拒答，消除错误拒答 | 1.622 | 490s |

---

## 八、最终效果

| 指标 | SFT 基线 | ORPO v3/v4 | 变化 |
|------|---------|-----------|------|
| Agentic RAG 得分 | 0.8301 | 0.8765 | **+5.6%** |
| 引用支持率 | 51.2% | 61.1% | **+19%** |
| 引用不支持率 | 30.1% | 11.8% | **-61%** |
| 答案平均 claim 数 | 1.70 | 1.23 | 更简洁精确 |
| 有引用的答案占比 | 86.7% | 93.3% | +6.6% |
| 拒答精确率 | 100% | 100% | 持平 |
| 幻觉率 | 15% | ~17% | 基本持平 |
| 错误拒答率 | 0% | 0% | 持平 |

**最有意义的提升**：引用支持率 +19%，不支持率 -61%。模型确实学会了「引用要精准、引用要真实」。拒答指标持平说明没有因偏好训练引入行为退化。

**直言局限**：幻觉率 15-17% 依然偏高——300 条微调数据、领域窄（一本用户手册），这是数据量的天花板，不是方法问题。更大的 query 池和更多飞轮迭代可以继续改善，这是后续方向。

---

## 九、这个项目展示了什么

### 核心能力

| 能力 | 体现 |
|------|------|
| **能从实验中定位问题** | 消融实验 → 发现瓶颈在 AnswerAgent 而非检索 |
| **能设计解决方案** | 选择 ORPO 而非 DPO，选择合成数据而非 LLM 标注，每个选择都有明确理由 |
| **能 debug 训练问题** | top_k=1 多样性坍塌、拒答方向错误——不是玄学调参，是逐层排查 |
| **能量化效果** | 每个改动都有 before/after，每个数字都知道怎么来的 |
| **能理解工具边界** | 合成数据的局限、小数据集的天花板、单卡 24GB 的 trade-off |

### 面试一句话版本

> 我搭了一个 RAG 系统，消融实验告诉我瓶颈在模型行为而非检索，于是从零建了一条 ORPO 对齐训练管线来修复这些行为。过程中翻了两坑——候选多样性坍塌和拒答方向错误——通过飞轮迭代逐个解决。最终引用质量提升 19%，但我最大的收获是理解了「什么时候该做对齐训练、数据该怎么设计、bug 该怎么排查」。

### 面试常见追问速查

| 问题 | 回答要点 |
|------|---------|
| 为什么 Hybrid 不如 BM25 Only？ | 向量检索引入语义相关但无用的文档作为噪声，不加精排不如纯关键词 |
| 为什么选 ORPO 而不是 DPO？ | 不需要 reference model，单卡 24GB 限制下的最优解 |
| 为什么用 Qwen2.5-14B 做 Judge？ | AWQ int4 量化后 ~12GB，单卡能跑，再大需要双卡 |
| 合成数据为什么用规则而不用 LLM？ | 成本低、信号纯净、可控性强——每个对比维度只改一个变量 |
| Reward 函数权重怎么定的？ | 引用支持率是核心信号（60%），其他是辅助，length bonus 设封顶防刷分 |
| top_k=1 怎么发现的？ | 候选重复率太高 → 排查 temperature → 排查额外参数 → 定位 `extra_body` |
| 拒答方向错误怎么发现的？ | v2 幻觉率恶化 → 先怀疑评测（确实有 bug）→ 修完评测再查偏好对 → 发现方向反 |
| 为什么不做 GRPO？ | LLaMA-Factory 已不支持，且在线训练对单卡不友好 |
| 为什么不扩 query 池？ | 300 → 1000+ 候选生成和打分时间线性增长，先验证方向再扩规模 |
| 幻觉率为什么没降？ | 300 条数据覆盖不了所有边界场景，需要更多样的拒答训练数据和更大的 query 池 |

---

## 十、多模态检索：从纯文本到图文混合

### 10.1 为什么需要多模态

用户手册 PDF 中包含 ~170 张图片：仪表盘 UI、按钮位置图、警告图标、操作流程图。纯文本 RAG 的一个盲区：

- 用户问「充电口在哪」→ 文本段落描述了位置，但图片比文字直观得多
- 用户问「这个图标什么意思」→ 纯文本根本答不了
- 用户问「方向盘上有哪些按钮」→ 一张图片胜过三行文字

但系统存在两个待解决问题：

| 问题 | 现状 |
|------|------|
| 生成模型 | ORPO v4 LoRA adapter 从未 merge 上线，线上仍是 SFT 模型 |
| 图片处理 | PDF 图片只存文件路径，内容未被检索管线感知 |

**目标**：生成模型升级到 ORPO v4 + 图片可被检索和前端展示。

**约束不变**：单卡 RTX 4090 24GB，不能往显存里塞第二个模型。

### 10.2 ORPO v4 上线：四层 LoRA 合并

**背景**：ORPO 训练产出了四个 83MB 的 LoRA adapter（SFT / v2 / v3 / v4），每个都是相对于原始 Qwen3-8B 基座的累计 delta。但线上 vLLM 一直跑的是只 merge 了 SFT 的 INT4 模型。

**过程**：

```
# 错误的做法（第一次）
Base → merge SFT → merge v2 → merge v3 → merge v4
# 问题：每个 adapter 保存的是对原始基座的累计 delta，逐层 merge 会重复叠加
# SFT × 4, v2 × 3, v3 × 2 → 权重彻底错乱

# 正确的做法
Base + ORPO v4 → merge_and_unload() 一次
# ORPO v4 = SFT + v2 + v3 + v4 累计增量（全都相对于原始基座）
```

**量化尝试**：BF16 模型 16.4GB，在 24GB 卡上刚好踩线。尝试 INT8/INT4 量化：
- AutoAWQ：Qwen3 架构不兼容，报 `torch.empty()` 空张量
- GPTQModel：拉高 transformers 到 5.x → vLLM 崩溃，torch 升级 → 全环境版本错乱
- LLaMA-Factory export：缺 `gptqmodel>=2.0.0`，安装后 PEFT 版本冲突

**最终方案**：BF16 直接上。16.4GB + KV cache ~3GB + Reranker CPU fallback，24GB 刚好够。量化后续可做，但当前不是 blocker。

**部署问题**：vLLM 0.9.0.1 默认开启 Qwen3 thinking 模式，每次回答先输出 `<think>` 内部推理，吃光 token 配额。排查发现：`enable_thinking: False` 在 `gptqmodel` 拉坏 torch/transformers 环境后失效。降回 `torch 2.7.0 + transformers 4.51.3` 后恢复正常。

**结果**：ORPO v4 上线，thinking 已禁，引用格式正确，答案准确。

### 10.3 图片检索：三层索引互补

**设计理念**：不把图片当二进制 blob，而是同时建立三套索引，各自覆盖不同的检索模式。

```
每张图片入库时：
  ├── 中文 Caption（qwen-vl-plus API）→ BM25 + 文本 Milvus
  │   作用：让关键词检索和文本语义检索都能命中图片
  │   生成：DashScope API，169 张图一次调用，< ¥0.5
  │
  ├── 视觉向量（Chinese-CLIP）→ 图片 Milvus Collection
  │   作用：用自然语言问题直接搜到语义匹配的图片
  │   模型：OFA-Sys/chinese-clip-vit-base-patch16，512 维，CPU 运行
  │
  └── 元信息（标题/页码/周围文字）→ MongoDB
      作用：图片来源追溯、结构化定位
```

**模型选型过程**：

| 尝试 | 模型 | 结果 |
|------|------|------|
| 1 | `jina-clip-v2` | 仅提供 ONNX 格式，加载需要 `trust_remote_code` 联网下载实现代码，网络不通 |
| 2 | `clip-ViT-B-32-multilingual-v1` | sentence-transformers 版本要求 torch 2.11+，环境不兼容 |
| **3** | **`OFA-Sys/chinese-clip-vit-base-patch16`** | **transformers 原生支持，直接加载，中文原生，512 维** |

**检索管线**：三路并行

```python
ThreadPoolExecutor(max_workers=3)
  ├── BM25 (jieba 关键词)
  ├── 文本 Milvus (BGE-M3 语义)
  └── 图片 Milvus (Chinese-CLIP 跨模态)  # 新增
```

检索结果合并后：
- 文本结果 → Reranker → AnswerAgent 生成答案
- 图片结果 → 附带 base64 编码 → API 返回相关图片

**验证**：

```
$ curl http://localhost:9000/chat -d '{"query":"充电口在哪"}'
{
  "answer": "Model 3 的充电口位于车辆后方左侧...",
  "related_images": [
    {"title": "充电口位置", "base64": "data:image/jpeg;base64,...", "caption": "..."},
    ...
  ]
}
```

### 10.4 服务部署与配置调整

为适配 BF16 模型（16.4GB），调整了 vLLM 启动参数：

```
原：max-model-len 4096, gpu-memory-utilization 0.75
新：max-model-len 3072, gpu-memory-utilization 0.92
```

同时将 answer_agent 中文档截断从 400 字符缩到 200 字符，max_tokens 从 2048 降到 1024，确保上下文不超限。

**服务冲突**：Milvus Lite 只支持单进程。Gradio 和 FastAPI 不能同时启动（后者拿不到 Milvus 锁）。线上只保留 FastAPI（9000 端口），Gradio 按需手动启动。

---

## 十一、技术栈总结

| 层级 | 技术 |
|------|------|
| LLM 推理 | vLLM 0.9, Qwen3-8B ORPO v4 BF16, Qwen2.5-14B AWQ |
| 训练框架 | LLaMA-Factory 0.9.3 (ORPO), LoRA, BF16 |
| 文本检索 | BM25 (jieba), Milvus Lite 2.5, BGE-M3 Embedding |
| 图片检索 | Chinese-CLIP (512d 向量) + Caption 文本（qwen-vl-plus 生成） |
| 精排 | BGE-M3 Cross-Encoder Reranker (GPU + CPU fallback) |
| Agent 框架 | 自建固定管线（非 LangGraph/AutoGen） |
| 评测 | 自定义消融框架, ragas, text2vec |
| Web 服务 | FastAPI + SSE 流式, Gradio |
| 数据库 | MongoDB 7.0, Milvus Lite |
| 图片处理 | PyMuPDF 提取 + qwen-vl-plus Caption + PIL base64 编码 |
| 硬件 | 单卡 RTX 4090 24GB, 磁盘 100GB |
