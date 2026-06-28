# RL Training TODO

> 目标：对答案生成器（Qwen3-8B）做 ORPO 强化学习，全本地闭环，零 API 依赖。
> 详细方案见 [RL_PLAN.md](./RL_PLAN.md)

---

## 阶段

### ✅ Phase 0: 基础设施准备

- [x] 创建 git 分支 `rl/orpo-training`
- [x] 保存计划书 `RL_PLAN.md`
- [x] 创建 `TODO.md`
- [x] 下载 Qwen2.5-14B-Instruct-AWQ 模型（9.4GB）
- [x] 新增 `src/client/llm_judge_client.py`
- [x] 启动 vLLM judge 服务，验证可用

### ⏭️ Phase 0.5: Citation Judge 校准（跳过）

- [ ] 跳过了形式化校准（因采用合成偏好对方案，不依赖 judge 精度）

### ✅ Phase 1: 生成候选答案

- [x] 新增 `src/rl/__init__.py`
- [x] 新增 `src/rl/generate_candidates.py`（ThreadPoolExecutor 并行，断点续跑）
- [x] 生成候选答案（300 queries × 4 temperatures = 1200 个候选）
- [x] 发现：SFT 模型输出高度确定性，85% query 的 4 个温度答案完全相同，整体重复率 70%

### ✅ Phase 2: Citation Judge 打分

- [x] 新增 `src/rl/reward.py`
- [x] 新增 `src/rl/score_candidates.py`
- [x] 用本地 Qwen2.5-14B judge 给所有候选打分
- [x] 发现：同一 query 的 4 个 temperature 答案 reward 差距极小，难以构造自然偏好对

### ✅ Phase 3: 构造偏好对

- [x] 新增 `src/rl/build_preference_pairs.py`
- [x] **策略调整**：用合成方式构造偏好对（去引用/截断），300 对
- [x] 实际分布：去引用 166 对（55%）、截断 47 对（16%）、自然分差 ~87 对（29%）
- [x] 数据格式：LLaMA-Factory sharegpt pairwise（`conversations` + `chosen`/`rejected` dict）
- [x] 注册 `rag_preference` 数据集到 `dataset_info.json`

### ✅ Phase 4: ORPO 训练

- [x] 新增 `LLaMA-Factory-main/examples/train_lora/qwen3_lora_orpo.yaml`
- [x] LLaMA-Factory 0.9.3 安装完成
- [x] **解决 AWQ + Triton 不兼容**：下载 Qwen3-8B BF16 基础模型到 `/root/autodl-tmp/RAG/models/Qwen3-8B/`（16GB），加载 BF16 基础模型 + SFT LoRA adapter 做 ORPO
- [x] 训练结果：37 checkpoint，loss 2.345，98 秒，LoRA adapter 87MB

### ✅ Phase 5: 评估

- [x] 运行 `eval/ablation_eval.py` → score **0.8450**（+0.0149 vs SFT 0.8301）
- [x] 运行 `eval/no_answer_eval.py` → 幻觉率 15%（持平，预期内）
- [ ] 人工抽查 30 条 ORPO 答案 vs SFT 答案（待做）

---

## 🔜 Phase 6: ORPO 改进（优先做，成本低收益高）

> 当前问题：300 对偏好数据 70% 只教了"带引用 > 不带引用"，多样性不足。不是样本少，是对比维度太单一。

### 6.1 rejected 类型扩充（优先级最高）

当前仅 2 种，建议扩到 4-5 种：

```
✅ 去引用    — chosen 带【1】，rejected 去掉引用标记
✅ 截断      — chosen 完整答案，rejected 只留前半
❌ 假引用    — chosen 引【真实引用】，rejected 引不存在的【99】  → 教模型引用要真实
❌ 拒答      — chosen 正常答，rejected 改成"无答案"            → 教模型该答时必须答
❌ 冗余引用  — chosen 精确引 1 条，rejected 引 5 条无关的       → 教模型引用要精准（可选）
```

全部纯规则改写，不动 LLM，改动量 < 20 行代码。

### 6.2 候选生成优化（增强多样性）

```
问题：85% query 四个温度产出一模一样的答案

方案 A：加大 temperature 范围
  [0.7, 0.8, 0.9, 1.0] → [0.5, 1.0, 1.2, 1.5]

方案 B：加采样参数组合（不只是 temperature）
  configs = [
    {"temperature": 0.5, "top_p": 0.9},
    {"temperature": 0.8, "top_p": 0.95},
    {"temperature": 1.0, "top_p": 0.9, "top_k": 50},
    {"temperature": 1.2, "top_p": 0.85},
  ]

方案 C：不同 prompt 扰动
  同一 query 用 "请详细回答" / "请简洁回答" / "列出关键步骤" 三种指令

预期：> 1.2 的温度下模型产出多样性明显改善，即使部分候选质量差，也天然适合做 rejected
```

### 6.3 训练超参调整

```yaml
# 当前
pref_beta: 0.1
num_train_epochs: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# 建议
pref_beta: 0.05              # 降低偏好 loss 权重，防止 300 对小数据集过拟合
num_train_epochs: 2           # 小数据集多跑一轮
per_device_train_batch_size: 1
gradient_accumulation_steps: 4 # 更频繁更新
```

### 6.4 评测维度补齐

当前只看了 avg_score，加测：

| 指标 | 说明 |
|------|------|
| citation_support_rate（RL reward 同口径） | 直接验证引用质量有没有提升 |
| 按 badcase 类型分组看分差 | 定位 ORPO 具体改善了哪类错误 |
| 人工抽查 30 条 ORPO vs SFT | 定性判断输出风格变化 |

---

## 📋 后续扩展（做完 6.1-6.3 后考虑）

### 迭代 RL 飞轮

```
ORPO v1 → 重新生成候选 → 打分 → 构造新偏好对 → ORPO v2
```

每轮约 2 小时+，等改进方向验证成立后再跑。

### 扩大 query 池

当前 300 条 → 扩到 1000+ 条，从 `train_qa_pair.json` 全量采样。代价：候选生成和打分时间按比例增长。

### Judge 升级

当前 Qwen2.5-14B 打分一致性预估 90-95%，若硬件允许可换 Qwen2.5-32B。**但需要双卡，不紧急。**

---

## 不可行 / 现阶段不做的

| 项 | 原因 |
|----|------|
| 编造内容 rejected（LLM 幻觉） | 需要专用坏模型，规则改写够用 |
| Judge 换 32B | 单卡 24GB 放不下 |
| GRPO 在线训练 | LLaMA-Factory 已不支持 |
| 全量微调替代 LoRA | 至少 48GB 显存 |

---

## GPU 用量记录

| 步骤 | 模型 | VRAM | 状态 |
|------|------|------|------|
| Phase 1 候选生成 | Qwen3-8B INT4 vLLM :8000 | ~10GB | ✅ |
| Phase 2 打分 | Qwen2.5-14B INT4 vLLM :8001 | ~12GB | ✅ |
| Phase 4 ORPO 训练 | Qwen3-8B BF16 LoRA | ~18GB | ✅ |

---

## 新增/修改文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/client/llm_judge_client.py` | ✅ | 本地 judge 客户端（端口 :8001） |
| `src/rl/__init__.py` | ✅ | RL 模块 |
| `src/rl/generate_candidates.py` | ✅ | 并行候选生成 + 断点续跑 |
| `src/rl/reward.py` | ✅ | 组合奖励函数 |
| `src/rl/score_candidates.py` | ✅ | 本地 judge 批量打分 |
| `src/rl/build_preference_pairs.py` | ✅ | 偏好对构造（当前仅去引用+截断，待扩充） |
| `LLaMA-Factory-main/data/rag_preference.json` | ✅ | 300 对 sharegpt pairwise 格式 |
| `LLaMA-Factory-main/examples/train_lora/qwen3_lora_orpo.yaml` | ✅ | ORPO 训练配置 |
| `LLaMA-Factory-main/data/dataset_info.json` | ✏️ | 注册 rag_preference |
| `LLaMA-Factory-main/saves/qwen3-8b/lora/orpo/` | ✅ | 训练产出（87MB LoRA adapter） |
| `data/rl/candidates.jsonl` | ✅ | 300 条 query × 4 temps（85% 重复） |
| `data/rl/scored_candidates.jsonl` | ✅ | 298 条评分结果 |
