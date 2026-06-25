# RL 训练计划书：全本地闭环的答案生成器强化学习

## Context

### 为什么做这个
当前 RAG 系统的**答案生成器（Qwen3-8B LoRA SFT）是性能瓶颈**。消融实验和 badcase 分析的数据一致指向：

| 证据 | 数据 |
|------|------|
| Agentic RAG 总分低于 BM25 Only | 0.8301 vs 0.8486（Agentic 组件反而拖后腿） |
| badcase 中大量"检索错误"实为生成错误 | 上下文已有证据，模型仍输出「无答案」 |
| 幻觉率 | 15%（9/60 条应拒答的 query 被错误回答） |
| Reranker 增益仅 +0.0124 | 检索和重排序已接近天花板，生成器才是瓶颈 |

### 目标
在 SFT checkpoint 基础上，通过 RL 让生成器学会三件事：
1. **证据利用**：上下文有答案时必须提取使用，不能偷懒输出「无答案」
2. **引用纪律**：每个事实声明必须带 `【ref_id】` 引用标记
3. **拒答校准**：区分「真的无答案」和「有答案但我不确定」

### 核心架构决策：全本地闭环

```
┌─────────────────────────────────────────────────────────┐
│                    全本地 RL Pipeline                      │
│                                                         │
│   生成器模型                  评判者模型                    │
│   Qwen3-8B AWQ INT4          Qwen2.5-14B AWQ INT4       │
│   vLLM :8000                 vLLM :8001                  │
│   (已 SFT, 待 RL)             (全新部署)                   │
│       │                          │                       │
│       │  生成 4 个候选答案         │  实时打分                │
│       └──────────────────────────┤                       │
│                                  │                       │
│          ┌───────────────────────┘                       │
│          ▼                                               │
│   偏好对数据集 → LLaMA-Factory ORPO/DPO 训练               │
│   未来可直接升级为 GRPO 在线训练                              │
│                                                         │
│   全部本地 · 零 API 费用 · 无限迭代 · 自主可控                 │
└─────────────────────────────────────────────────────────┘
```

### 为什么选 ORPO 而非 GRPO/PPO

| 方法 | LLaMA-Factory 支持 | 复杂度 | 当前可用？ |
|------|-------------------|--------|-----------|
| GRPO | ❌ 已拆分到 EasyR1 | — | ❌ |
| PPO | ✅ `stage: ppo` | 高（需先训练 RM） | ✅ 可作为升级方向 |
| **ORPO** | ✅ `pref_loss: orpo` | 低（离线偏好对→直接训练，无需 ref model） | ✅ **首选** |
| DPO | ✅ `pref_loss: sigmoid` | 中（需 ref model） | ✅ 备选 |

**选择 ORPO**：无需 reference model，单卡可跑，内置 SFT loss 防遗忘。

---

## 基础设施

### GPU 需求分析

| 场景 | 组件 | VRAM 占用 |
|------|------|----------|
| 候选答案生成 | Qwen3-8B AWQ INT4 (vLLM :8000) | ~8GB |
| Citation 打分 | Qwen2.5-14B AWQ INT4 (vLLM :8001) | ~10GB |
| **两模型同时跑** | 两个 vLLM 实例 | ~18GB + KV cache ~4GB ≈ **22GB** |
| ORPO 训练 | Qwen3-8B LoRA (rank=8) | ~12GB |

- **24GB GPU (A10/3090/4090)**：两个模型可同时跑，刚好够
- **48GB GPU (A6000/L40)**：绰绰有余，可用 BF16 全精度 judge
- **如果不能同时跑**：分批运行（先生成候选 → 关闭生成器 → 启动 judge 打分），总时长多加约 30 分钟

### 模型部署计划

| 模型 | 用途 | 源 | 大小 | 端口 |
|------|------|-----|------|------|
| Qwen3-8B AWQ INT4 | 答案生成器（已部署） | 已有 | ~8GB | :8000 |
| **Qwen2.5-14B-Instruct-AWQ** | Citation Judge（新部署） | ModelScope 下载 | ~8GB | :8001 |

### 为什么选 Qwen2.5-14B 做 Citation Judge

| 对比维度 | Qwen3-8B INT4（用现有） | Qwen2.5-14B INT4（推荐） |
|---------|----------------------|------------------------|
| 与生成器关系 | 同模型，有自我偏好偏差 | 不同模型，天然避免偏差 |
| 中文阅读理解 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 细粒度 classify | 中-高 | 高 |
| 与 Qwen-Plus 一致率（预估） | 82-88% | **90-95%** |
| 偏好对排序一致性（预估） | 85-90% | **93-96%** |

---

## 实施路线图

### Phase 0: 部署本地 Citation Judge ← 当前步骤

**目标**：下载 Qwen2.5-14B，启动 vLLM，验证可与现有 CitationVerifier 对接。

**步骤**：
```bash
# 1. 下载模型
pip install modelscope
modelscope download Qwen/Qwen2.5-14B-Instruct-AWQ \
    --local_dir /root/autodl-tmp/RAG/models/Qwen2.5-14B-Instruct-AWQ/

# 2. 启动 vLLM judge 服务（端口 :8001，与生成器 :8000 错开）
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/RAG/models/Qwen2.5-14B-Instruct-AWQ/ \
    --served-model-name Qwen2.5-14B-Instruct-AWQ \
    --port 8001 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype auto

# 3. 创建本地 judge 客户端（新增文件）
# src/client/llm_judge_client.py — 指向 :8001 的 OpenAI 兼容客户端
```

**新增文件**：`src/client/llm_judge_client.py`（约 15 行，复制 `llm_local_client.py` 改端口）

**验证**：取 5 条已有答案，用本地 judge 跑 CitationVerifier，确认 JSON 输出格式正确。

---

### Phase 0.5: Citation Judge 校准（可选但推荐）

**目标**：验证 Qwen2.5-14B 的 citation 判断与远程 Qwen-Plus 的一致性。

**步骤**：
1. 从 `test_qa_pair_pred.json` 随机抽 50 条答案
2. 分别用 Qwen-Plus（远程）和 Qwen2.5-14B（本地）跑 CitationVerifier
3. 计算一致率：`support/partial/not_support` 三元一致率
4. 重点检查：偏好对排序是否一致（同 query 的 4 个答案排序对比）

**预期**：三元一致率 ≥ 88%，排序一致率 ≥ 92%

**如果一致率不达标（< 85%）**：
- 检查 prompt 是否需要针对 Qwen2.5 调整（Qwen2.5 vs Qwen-Plus 的 system prompt 格式差异）
- 如果调整后仍不达标，降级为 text2vec semantic similarity 作为主 reward（全本地、零 LLM）

---

### Phase 1: 生成候选答案

**输入**：`data/qa_pairs/train_qa_pair.json`（~2300 条 has-answer QA 对）

**步骤**：
1. 对每条 query 用现有 RAGPipeline 检索上下文（BM25 + Milvus + Reranker）
2. 用 SFT checkpoint 的 AnswerAgent（vLLM :8000）生成 4 个候选答案
3. temperature 分别取 [0.7, 0.8, 0.9, 1.0]，max_tokens=1024
4. 保存为 `data/rl/candidates.json`

**新增文件**：`src/rl/generate_candidates.py`（约 150 行）

**预估耗时**：~3 小时（2300 queries × 4 candidates × ~1.2s/vLLM call）

---

### Phase 2: 本地 Citation Judge 打分

**目标**：用 Qwen2.5-14B 给每个候选答案的 citation 质量打分。

**步骤**：
1. 对每个候选答案调用 `CitationVerifier`（已复用，只换 client import）
2. 计算主分数：`citation_support_rate = support_count / total_claims`
3. 计算辅助分数：
   - `unsupported_penalty = unsupported_count / max(total_claims, 1)`
   - `no_citation_penalty = 1.0 if 答案有实质内容但无引用 else 0.0`
   - `answer_length = len(answer_text)`
4. 组合成最终 reward（用于偏好对排序）：
   ```python
   reward = (
       0.5 * citation_support_rate          # 引用质量
     - 0.3 * unsupported_penalty            # 虚假引用惩罚
     - 0.1 * no_citation_penalty            # 无引用惩罚
     + 0.1 * min(answer_length / 200, 1.0)  # 鼓励完整回答（封顶）
   )
   ```

**输出**：`data/rl/scored_candidates.json`

**新增文件**：`src/rl/score_candidates.py`（约 80 行）+ `src/rl/reward.py`（约 40 行）

**预估耗时**：~6-8 小时（9200 个答案 × 4 workers × ~1.5s/vLLM judge call）

---

### Phase 3: 构造偏好对

**偏好对构造规则**：
```python
def build_pair(candidates_with_scores):
    sorted_cands = sorted(candidates, key=lambda c: c["reward"], reverse=True)
    chosen = sorted_cands[0]    # 最高 reward
    rejected = sorted_cands[-1] # 最低 reward

    # 过滤：reward 差距太小说明区分度不够
    if chosen["reward"] - rejected["reward"] < 0.2:
        return None

    return {"chosen": chosen["text"], "rejected": rejected["text"]}
```

**LLaMA-Factory pairwise 格式**：
```json
{
  "messages": [
    {"role": "system", "content": "你是特斯拉Model 3的智能助手..."},
    {"role": "user", "content": "<query>"}
  ],
  "response": ["<chosen_answer>", "<rejected_answer>"]
}
```

**预期产出**：~1500-2000 个高质量偏好对（过滤后）

**新增文件**：`src/rl/build_preference_pairs.py`（约 100 行）

---

### Phase 4: ORPO 训练

**训练配置** `LLaMA-Factory-main/examples/train_lora/qwen3_lora_orpo.yaml`：

```yaml
### model
model_name_or_path: /root/autodl-tmp/RAG/models/Qwen3-8B/
adapter_name_or_path: saves/qwen3-8b/lora/sft/  # 从 SFT checkpoint 开始

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_loss: orpo           # 不需要 reference model
pref_beta: 0.1

### dataset
dataset: rag_preference
template: qwen3
cutoff_len: 4096
max_samples: 2000

### training
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6      # SFT 的 1/4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### output
output_dir: saves/qwen3-8b/lora/orpo
```

**数据集注册**（在 `data/dataset_info.json` 中添加）：
```json
"rag_preference": {
    "file_name": "rag_preference.json",
    "ranking": true
}
```

**训练时长**：~20-30 分钟（2000 条偏好对，1 epoch）

---

### Phase 5: 评估

复用现有评估框架，对比 SFT vs ORPO：

```bash
# 1. Ablation 评估（50 题 × 5 variants）
python eval/ablation_eval.py
# 关注：Agentic RAG 总分变化

# 2. 拒答评估
python eval/no_answer_eval.py --mode agentic
# 关注：幻觉率、拒答召回率

# 3. Badcase 分析
python badcase_analyzer.py
# 关注：retrieval_error（假阳性）和 citation_error 占比

# 4. 人工抽查（关键！）
# 随机抽 30 条 ORPO 答案，人工对比 SFT 答案
# 判断：是否更好？哪里好了？有没有变差？
```

---

## 迭代实验设计（本地部署的核心价值）

因为全本地、零 API 费用，可以多轮迭代：

```
第 1 轮：ORPO with citation_support_rate 作为主 reward
  → 预期：citation 覆盖率显著提升，但答案可能偏短

第 2 轮：根据第 1 轮结果，调整 reward 权重
  → 如果发现答案偏短 → 加大 answer_length 权重
  → 如果发现虚假引用 → 加大 unsupported_penalty

第 3 轮：改进候选答案生成策略
  → 加入 top_p sampling, 不同 system prompt
  → 候选更丰富 → 偏好对质量更高

第 4 轮：如果 VRAM 允许，升级到 GRPO 在线训练
```

---

## 升级路径：ORPO → GRPO

当全本地 pipeline 跑通后，如果 VRAM 允许两个模型同时跑（24GB+），可以直接升级：

| | ORPO（离线） | GRPO（在线） |
|---|-------------|-------------|
| 训练方式 | 预先生成偏好对 → 一次性训练 | 训练中不断采样 → 实时打分 → 更新 |
| 需要框架 | LLaMA-Factory | EasyR1 或 TRL |
| 采样效率 | 固定 4 个候选/temperature | 在线采样，可动态调整 |
| 收敛速度 | 较慢 | 较快（策略持续接触新的负样本） |
| 实现难度 | 低 | 中-高 |

**升级条件**：
1. ORPO 跑通且有效果
2. GPU 能同时跑两个 vLLM（24GB+ VRAM）
3. 安装 EasyR1 或升级 TRL 到支持 GRPO 的版本

---

## 风险评估（全本地版本更新）

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 本地 judge 打分不够准 | 低-中 | 偏好对质量差 | Phase 0.5 校准，不达标降级为 text2vec |
| 奖励黑客（刷引用格式） | 中 | 模型学会形式化引用 | reward 组合 + ORPO SFT loss 正则化 |
| 训练数据质量问题 | 中 | gold answer 有误影响偏好 | Phase 3 后人工抽查 50 对 |
| GPU OOM（两模型同时跑） | 低 | 不能同时跑 | 分批运行，只多花 30 分钟 |
| 灾难性遗忘 | 低 | 丢 SFT 能力 | ORPO 内置 SFT loss，LoRA rank=8 学习量小 |

---

## 成功标准

| 指标 | SFT Baseline | ORPO 目标 | 测量方式 |
|------|-------------|-----------|---------|
| Agentic RAG 总分 | 0.8301 | ≥ 0.86 | `ablation_eval.py` |
| 幻觉率 | 15% | ≤ 8% | `no_answer_eval.py` |
| 拒答召回率 | 85% | ≥ 90% | `no_answer_eval.py` |
| citation_error 占比 | ~12% | ≤ 5% | `badcase_analyzer.py` |
| retrieval_error（假阳性） | ~40% | ≤ 20% | `badcase_analyzer.py` |
| 人工主观评估 | — | ≥ 70% 答案优于 SFT | 人工抽查 30 条 |

---

## 把握度评估

### 全本地闭环后：**70-80%**（比远程方案提升 10%+）

**提升信心的因素**：
- 不同模型做 judge（Qwen2.5-14B vs Qwen3-8B），天然避免自我偏好
- 零成本迭代 → 可以多轮调优 reward 函数，不需要第一次就完美
- 14B 模型的分类判断力足够靠谱

**仍然不确定的部分**：
- reward 函数需要多轮调整才能找到最优权重
- 训练集 2300 条可能不够多样，泛化能力待验证
- 总分提升幅度能否达到 0.03+

### 如果 ORPO 效果不好，Plan B

1. **纯 SFT 改进**：用本地 judge 过滤 SFT 数据，只保留高质量答案重新 SFT
2. **训 Reward Model → PPO**：人工标注 200-500 条，用 LLaMA-Factory RM stage 训 reward model
3. **转向重排序器 RL**：用全本地的 text2vec 打分，对 BGE-M3 Reranker 做 listwise RL

---

## 文件变更清单

### 新增文件

| 文件 | 用途 | 预估行数 |
|------|------|---------|
| `src/client/llm_judge_client.py` | Phase 0: 指向 :8001 的本地 judge 客户端 | ~15 |
| `src/rl/__init__.py` | RL 模块初始化 | ~5 |
| `src/rl/generate_candidates.py` | Phase 1: 批量生成候选答案 | ~150 |
| `src/rl/score_candidates.py` | Phase 2: Citation Judge 批量打分 | ~80 |
| `src/rl/reward.py` | 奖励函数（多信号组合） | ~40 |
| `src/rl/build_preference_pairs.py` | Phase 3: 构造 LLaMA-Factory 偏好对 | ~100 |
| `LLaMA-Factory-main/data/rag_preference.json` | Phase 3 产出：偏好对数据集 | ~8MB |
| `LLaMA-Factory-main/examples/train_lora/qwen3_lora_orpo.yaml` | Phase 4: ORPO 训练配置 | ~35 |

### 修改文件

| 文件 | 改动 | 行数 |
|------|------|------|
| `LLaMA-Factory-main/data/dataset_info.json` | 添加 `rag_preference` 数据集定义 | +4 |

### 不修改的文件

- `src/pipeline/rag_pipeline.py` — pipeline 逻辑不变
- `src/agents/` — 所有 agent 逻辑不变（只换 CitationVerifier 的 client import）
- `src/retriever/` — 检索器不变
- `src/reranker/` — 重排序器不变
- `eval/` — 评估框架直接复用
- `src/agents/citation_verifier.py` — 不改，构造函数接受任意 OpenAI client

---

## 验证步骤（完整流程）

```bash
# ===== Phase 0: 部署本地 Judge =====
modelscope download Qwen/Qwen2.5-14B-Instruct-AWQ \
    --local_dir /root/autodl-tmp/RAG/models/Qwen2.5-14B-Instruct-AWQ/

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/RAG/models/Qwen2.5-14B-Instruct-AWQ/ \
    --port 8001 --max-model-len 8192 &

# 验证 judge 可用
curl http://localhost:8001/v1/models

# ===== Phase 0.5: 校准（可选） =====
python src/rl/calibrate_judge.py --samples 50

# ===== Phase 1: 生成候选答案 =====
python src/rl/generate_candidates.py \
    --qa_pairs data/qa_pairs/train_qa_pair.json \
    --output data/rl/candidates.json

# ===== Phase 2: 打分 =====
python src/rl/score_candidates.py \
    --candidates data/rl/candidates.json \
    --output data/rl/scored_candidates.json

# ===== Phase 3: 构造偏好对 =====
python src/rl/build_preference_pairs.py \
    --scored data/rl/scored_candidates.json \
    --output LLaMA-Factory-main/data/rag_preference.json

# ===== 人工抽查 50 对偏好对 =====
# 确认 chosen 确实比 rejected 好

# ===== Phase 4: ORPO 训练 =====
cd LLaMA-Factory-main
llamafactory-cli train examples/train_lora/qwen3_lora_orpo.yaml

# ===== Phase 5: 评估 =====
cd ..
python eval/ablation_eval.py
python eval/no_answer_eval.py --mode agentic
python badcase_analyzer.py
```
