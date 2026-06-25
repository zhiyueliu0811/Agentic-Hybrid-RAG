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
- [x] 生成 160 条候选答案（300 queries × 4 temperatures）
- [x] 发现：SFT 模型输出高度确定性（72% 答案完全相同）

### ✅ Phase 2: Citation Judge 打分

- [x] 新增 `src/rl/reward.py`
- [x] 新增 `src/rl/score_candidates.py`
- [x] 用本地 Qwen2.5-14B judge 给所有候选打分
- [x] 发现：同一 query 的 4 个 temperature 答案 reward 差距极小，难以构造自然偏好对

### ✅ Phase 3: 构造偏好对

- [x] 新增 `src/rl/build_preference_pairs.py`
- [x] **策略调整**：用合成方式构造偏好对（去引用/截断/假引用），300 对
- [x] 数据格式：LLaMA-Factory sharegpt pairwise（`conversations` + `chosen`/`rejected` dict）
- [x] 注册 `rag_preference` 数据集到 `dataset_info.json`

### ⚠️ Phase 4: ORPO 训练（受阻）

- [x] 新增 `LLaMA-Factory-main/examples/train_lora/qwen3_lora_orpo.yaml`
- [x] LLaMA-Factory 0.9.3 安装完成
- [ ] **受阻**：AWQ 量化模型 + Triton 3.3.0 内核编译失败
- [ ] **修复方案**：下载 Qwen3-8B BF16 基础模型（~16GB），以 BF16 精度做 ORPO 训练

### ⏭️ Phase 5: 评估（待 Phase 4 完成后）

- [ ] 运行 `eval/ablation_eval.py`
- [ ] 运行 `eval/no_answer_eval.py --mode agentic`
- [ ] 运行 `badcase_analyzer.py`
- [ ] 人工抽查 30 条 ORPO 答案 vs SFT 答案

---

## 当前阻塞与解决方案

### 问题：AWQ 量化模型 + Triton 兼容性

```
triton.compiler.errors.CompilationError: at 108:22:
    accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)
```

**原因**：合并后的 SFT 模型是 AWQ INT4 量化格式。Triton 3.3.0（torch 2.7.0 要求）与 AWQ 库的内核不兼容。

**解决方案（按推荐度排序）**：

1. **下载 Qwen3-8B BF16 基础模型**（推荐，16GB）：
   ```bash
   modelscope download Qwen/Qwen3-8B --local_dir /root/autodl-tmp/RAG/models/Qwen3-8B/
   ```
   然后在 YAML 中加载基础模型 + SFT LoRA adapter：
   ```yaml
   model_name_or_path: /root/autodl-tmp/RAG/models/Qwen3-8B/
   adapter_name_or_path: saves/qwen3-8b/lora/sft/
   ```

2. **用 vLLM 做推理时优化替代训练**：跳过 ORPO 训练，直接在推理时用更严格的 prompt + citation 校验来提升质量

3. **转向重排序器 RL**：用 text2vec 做奖励信号，对 BGE-M3 Reranker 做 listwise RL

---

## GPU 用量记录

| 步骤 | 模型 | VRAM | 状态 |
|------|------|------|------|
| Phase 1 候选生成 | Qwen3-8B INT4 vLLM :8000 | ~10GB | ✅ |
| Phase 2 打分 | Qwen2.5-14B INT4 vLLM :8001 | ~12GB | ✅ |
| Phase 4 ORPO 训练 | Qwen3-8B BF16 LoRA | ~18GB | ⚠️ 受阻 |

---

## 新增/修改文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/client/llm_judge_client.py` | ✅ | 本地 judge 客户端（端口 :8001） |
| `src/rl/__init__.py` | ✅ | RL 模块 |
| `src/rl/generate_candidates.py` | ✅ | 并行候选生成 + 断点续跑 |
| `src/rl/reward.py` | ✅ | 组合奖励函数 |
| `src/rl/score_candidates.py` | ✅ | 本地 judge 批量打分 |
| `src/rl/build_preference_pairs.py` | ✅ | 偏好对构造 |
| `LLaMA-Factory-main/data/rag_preference.json` | ✅ | 300 对 sharegpt 格式 |
| `LLaMA-Factory-main/examples/train_lora/qwen3_lora_orpo.yaml` | ✅ | ORPO 训练配置 |
| `LLaMA-Factory-main/data/dataset_info.json` | ✏️ | 注册 rag_preference |
| `data/rl/candidates.jsonl` | ✅ | 160 条候选答案 |
| `data/rl/scored_candidates.jsonl` | ✅ | 298 条评分结果 |
