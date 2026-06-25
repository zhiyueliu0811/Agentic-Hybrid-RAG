# RL Training TODO

> 目标：对答案生成器（Qwen3-8B）做 ORPO 强化学习，全本地闭环，零 API 依赖。
> 详细方案见 [RL_PLAN.md](./RL_PLAN.md)

---

## 阶段

### ✅ Phase 0: 基础设施准备

- [x] 创建 git 分支 `rl/orpo-training`
- [x] 保存计划书 `RL_PLAN.md`
- [x] 创建 `TODO.md`
- [x] 下载 Qwen2.5-14B-Instruct-AWQ 模型（9.4GB, done）
- [ ] 新增 `src/client/llm_judge_client.py`（本地 judge 客户端，端口 :8001）
- [ ] 启动 vLLM judge 服务（需要 GPU）

### Phase 0.5: Citation Judge 校准

- [ ] 新增 `src/rl/calibrate_judge.py`
- [ ] 抽 50 条答案，对比本地 judge vs 远程 Qwen-Plus 一致率
- [ ] 一致率 ≥ 88% → 继续；< 85% → 调整 prompt 或降级为 text2vec

### Phase 1: 生成候选答案

- [ ] 新增 `src/rl/__init__.py`
- [ ] 新增 `src/rl/generate_candidates.py`
- [ ] 运行：2300 queries × 4 temperatures → `data/rl/candidates.json`
- [ ] 需要 GPU（vLLM :8000）

### Phase 2: Citation Judge 打分

- [ ] 新增 `src/rl/reward.py`
- [ ] 新增 `src/rl/score_candidates.py`
- [ ] 运行：9200 个答案 × judge → `data/rl/scored_candidates.json`
- [ ] 需要 GPU（vLLM :8001）

### Phase 3: 构造偏好对

- [ ] 新增 `src/rl/build_preference_pairs.py`
- [ ] 运行：过滤 + 构造 pairwise 格式 → `LLaMA-Factory-main/data/rag_preference.json`
- [ ] 人工抽查 50 对，确认 chosen > rejected
- [ ] 不需要 GPU

### Phase 4: ORPO 训练

- [ ] 新增 `LLaMA-Factory-main/examples/train_lora/qwen3_lora_orpo.yaml`
- [ ] 修改 `LLaMA-Factory-main/data/dataset_info.json`，注册 `rag_preference` 数据集
- [ ] 运行 `llamafactory-cli train`
- [ ] 需要 GPU

### Phase 5: 评估

- [ ] 运行 `eval/ablation_eval.py`
- [ ] 运行 `eval/no_answer_eval.py --mode agentic`
- [ ] 运行 `badcase_analyzer.py`
- [ ] 人工抽查 30 条 ORPO 答案 vs SFT 答案
- [ ] 需要 GPU

### Plan B（如果 ORPO 效果不好）

- [ ] 方案 1：用 judge 过滤 SFT 数据，重新 SFT
- [ ] 方案 2：人工标注 200-500 条，训 Reward Model → PPO
- [ ] 方案 3：转向重排序器 listwise RL

---

## GPU 需求总览

| 步骤 | GPU | 预估时长 |
|------|-----|---------|
| 下载模型 | 否 | ~10 分钟 |
| 代码开发 | 否 | 不限 |
| 启动 judge vLLM | **是** | ~2 分钟 |
| Phase 1: 生成候选 | **是** | ~3 小时 |
| Phase 2: 打分 | **是** | ~6-8 小时 |
| Phase 3: 构造偏好对 | 否 | ~10 分钟 |
| Phase 4: ORPO 训练 | **是** | ~30 分钟 |
| Phase 5: 评估 | **是** | ~2 小时 |

---

## 新增文件清单

| 文件 | 状态 |
|------|------|
| `RL_PLAN.md` | ✅ |
| `TODO.md` | ✅ |
| `src/client/llm_judge_client.py` | ⬜ |
| `src/rl/__init__.py` | ⬜ |
| `src/rl/calibrate_judge.py` | ⬜ |
| `src/rl/generate_candidates.py` | ⬜ |
| `src/rl/reward.py` | ⬜ |
| `src/rl/score_candidates.py` | ⬜ |
| `src/rl/build_preference_pairs.py` | ⬜ |
| `LLaMA-Factory-main/data/rag_preference.json` | ⬜ |
| `LLaMA-Factory-main/examples/train_lora/qwen3_lora_orpo.yaml` | ⬜ |
