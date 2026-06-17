# TODO

## 已完成（2026-05-28 上午）

### 工程安全与配置化
- [x] S1 清理硬编码 API Key：新增 `.env.example` + `src/config.py`
- [x] S1 删除 `config.ini`（已被 `.env` 替代），start.sh 改为 source `.env`
- [x] S1 统一配置源：llm_client / llm_local_client / semantic_chunk_client 均改用 config.py
- [x] S2 修复 `post_processing()` 健壮性：metadata 访问改用 `.get()`，增加类型检查
- [x] B4 QueryAgent 增加 Pydantic schema 校验 + 重试 + fallback
- [x] 新增 `src/config.py` / `src/agents/query_schema.py` / `README.md`
- [x] `.env` 已在 `.gitignore` 中，真实密钥不会提交

### Citation Verifier（可信 RAG）
- [x] S3 新增 `src/agents/citation_verifier.py` — 句子级引用校验
- [x] S3 新增 `CITATION_VERIFY_PROMPT` 到 `src/agents/prompts.py`
- [x] S3 集成到 `web_demo.py` — Gradio 展示逐句校验结果
- [x] S3 集成到 `infer_agentic.py` — 终端输出校验详情
- [x] 端到端验证通过：LLM Judge 正确判断 support/partial/not_support

### 消融评测
- [x] S4 新增 `eval/eval_config.py` — 5 个管道变体配置
- [x] S4 新增 `eval/ablation_eval.py` — 消融评测主脚本（支持 --limit / --variant / 缓存）
- [x] S4 新增 `eval/report_generator.py` — JSON/CSV/Markdown 报告生成
- [x] 按需初始化组件：bm25_only 等变体不加载 Milvus，避免文件锁冲突

### 无答案专项评测
- [x] S5 新增 `data/qa_pairs/no_answer_qa.json` — 35 条无答案测试数据
- [x] S5 新增 `eval/no_answer_eval.py` — 4 项指标评测
- [x] 报告输出：JSON 原始数据 + Markdown 汇总报告，含分类统计和典型案例

### RAGPipeline 抽象 + FastAPI 服务
- [x] A4 新增 `src/pipeline/rag_pipeline.py` — 统一管线（CLI / Web / API / 评测共用）
- [x] A3 新增 `src/api/main.py` + `src/api/schemas.py` — FastAPI 服务（/health /chat /chat/stream）
- [x] 改造 `web_demo.py`、`infer_agentic.py` — 删除重复代码，导入共享管线
- [x] 3 个入口（CLI / Web / API）共享同一套 RAGPipeline，行为一致

### 启动修复 + 千问替换（2026-05-27）
- [x] vLLM torch.compile 崩溃修复、模型名 404 修复、Milvus Lite 文件锁修复
- [x] Web Demo (Gradio) 上线运行：http://localhost:7860
- [x] 全链路验证通过：千问 API + vLLM + BM25/Milvus 混合检索 + Reranker
- [x] 千问 API 替换豆包：环境变量 DOUBAO_* → LLM_*
- [x] STARTUP_NOTES.md 启动文档、项目源码打包

---

## 已完成（2026-05-28 下午）— 工程质量修复

### P0：必须优先修复
- [x] **FastAPI `/chat/stream` 事件循环 bug** — `asyncio.get_event_loop()` → `get_running_loop()` 提前捕获，修复后台线程 RuntimeError
- [x] **RAGPipeline 空召回兜底** — rerank 前判断 `merged_docs` 为空时直接返回「无答案」，避免崩溃
- [x] **RAGPipeline 返回字段统一** — 新增 `_default_result()`，三个分支（无需检索/空召回/正常）字段完全一致，消灭 KeyError 风险

### P1：强烈建议修复
- [x] **Citation Verifier 无引用判断** — 有事实性答案但无引用标记 → `verified=False` + `no_citation`
- [x] **Citation Verifier partial 纳入风险** — `partial` 计入 `unsupported_count`，新增 `partial_count` 统计
- [x] **校验失败触发重答闭环** — AnswerAgent 新增 `rewrite_with_supported_evidence()`，Pipeline 中校验不通过 → 保守重答 → 二次校验，记录 `citation_rewrite_triggered`
- [x] **Web Demo 字段防御性读取** — 全部 `r["key"]` → `r.get("key", default)`

### P2：评测链路修复
- [x] **`no_answer_eval.py` 增加 `--mode agentic`** — agentic 模式调用完整 RAGPipeline（含 QueryAgent / EvidenceAgent / Self-RAG / CitationVerifier），baseline 保持原逻辑
- [x] **`ablation_eval.py` 评分逻辑修复** — 无关键词时 `score = semantic_sim`（与 `final_score.py` 一致），修复系统性低分 bug
- [x] **`ablation_eval.py` `--output` 参数生效** — 修复参数已解析但未传递到 `generate_report()` 的 bug
- [x] **新增 `examples/qa/*.example.json`** — 8 条无答案 + 7 条 QA 示例数据，不被 `.gitignore` 忽略，clone 后可立即跑评测

### P3：工程规范
- [x] **`final_score.py` 迁移到 `src.config`** — `os.environ[...]` → `from src.config import ...`
- [x] **`TODO.md` Markdown 格式修复** — 删除底部多余的代码块结束符
- [x] **`start.sh` `.env` 加载优化** — `set -a; source .env; set +a` 替代 `grep | xargs`；API Key 兼容 `LLM_API_KEY` 和 `DASHSCOPE_API_KEY`

---

## 有卡模式下待验证（当前 GPU 可用，立即执行）

> 当前环境：RTX 4090 (24GB)，GPU 空闲，无服务在运行。

### 第一步：启动基础服务栈
```bash
bash start.sh
# 依次启动：MongoDB → 语义分块服务 → vLLM (Qwen3-8B) → Web Demo (:7860)
# 等待约 3 分钟，vLLM 加载 AWQ int4 模型
```

### 第二步：Web Demo + CLI + FastAPI 回归测试（已验证 2026-05-28）
- [x] 浏览器打开 http://localhost:7860 → HTTP 200，Web Demo 运行中
- [x] `python infer_agentic.py` CLI 流式输出正常 — 8 步管线完整跑通，引用校验通过，12.64s
- [x] FastAPI 启动：`uvicorn src.api.main:app --host 0.0.0.0 --port 9000`
- [x] `curl http://localhost:9000/health` → `{"status":"ok"}`
- [x] `/chat` 非流式问答 → 返回完整 JSON，答案正确，引用校验通过，耗时 11.77s
- [x] `/chat/stream` SSE 流式问答 → token 逐字返回，结尾 `{"done": true}` 正常，**事件循环修复验证通过**

### 第三步：S4 消融评测（已验证 2026-05-28）
- [x] 停止 Web Demo：`pkill -f web_demo.py`
- [x] 快速验证：`python eval/ablation_eval.py --limit 5 --variant hybrid_rerank` → 5条平均分 0.6311
- [x] 完整评测：`python eval/ablation_eval.py --limit 50` → **50条平均分 0.8442，延迟 1.73s，高分率 72%**
- [x] 报告已生成：`ablation_result.json` / `.csv` / `ablation_report.md`

### 第四步：S5 无答案评测（已验证 2026-05-28）
- [x] baseline 模式 → 精确率 100% / 召回率 83.33% / 幻觉率 16.67%
- [x] agentic 模式 → 精确率 100% / 召回率 85.00% / 幻觉率 15.00%
- [x] **对比结论：Agentic 比 Baseline 幻觉率降低 1.67%，召回率提升 1.67%**
- [x] 报告已生成：`no_answer_baseline_*` + `no_answer_agentic_*`

### 第五步：验证后重启
```bash
bash start.sh
```

---

## 待实施

- [ ] A1 统一 Trace 可观测性（`src/tracing.py`）
- [ ] A2 EvidenceAgent 规则增强
- [ ] Badcase Analyzer 增强
- [ ] 跑出真实实验数据后，在简历中补充具体百分比（Context Recall / No-Answer Precision / Hallucination Rate）

---

## 已完成（2026-06-04）— 意图识别

- [x] **意图识别融入 QueryAgent** — 一次 LLM 调用同时输出 `intent` + `query_type`，零额外成本
- [x] 意图分类：`knowledge_qa` / `chitchat` / `action_request` / `ambiguous`
- [x] 路由逻辑：非知识问答意图跳过检索管线，直接返回引导性拒答（节省远程 API 调用）
- [x] 改动的文件：`prompts.py` / `query_schema.py` / `query_agent.py` / `rag_pipeline.py` / `schemas.py` / `main.py` / `web_demo.py` / `infer_agentic.py`
- [x] 向后兼容：`intent` 默认 `knowledge_qa`，旧 prompt 返回不带 intent 时自动 fallback
