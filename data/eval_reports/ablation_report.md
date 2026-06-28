# RAG Ablation Report

## Overall Metrics

| Variant | Count | Avg Score | Semantic Sim | Keyword Score | Avg Latency |
|---------|-------|-----------|-------------|---------------|-------------|
| BM25 Only | 50 | 0.8797 | 0.9096 | 0.6763 | 3.3s |
| Milvus Only | 50 | 0.862 | 0.8921 | 0.663 | 3.77s |
| Hybrid (BM25 + Milvus) | 50 | 0.8699 | 0.9034 | 0.6563 | 4.01s |
| Hybrid + Reranker | 50 | 0.8716 | 0.9067 | 0.6497 | 4.9s |
| Agentic RAG (Full) | 50 | 0.8765 | 0.9098 | 0.6597 | 8.14s |

## Per-Component Gain

- **Milvus vs BM25**: -0.0177 (0.8797 → 0.862)
- **Hybrid vs BM25**: -0.0098 (0.8797 → 0.8699)
- **Reranker gain**: +0.0017 (0.8699 → 0.8716)
- **Agentic gain (QueryRewrite + Evidence + Self-RAG)**: +0.0049 (0.8716 → 0.8765)

## Score Distribution

| Variant | Min | Max | Std |
|---------|-----|-----|-----|
| BM25 Only | 0.2962 | 1.0000 | 0.1283 |
| Milvus Only | 0.0000 | 0.9918 | 0.1611 |
| Hybrid (BM25 + Milvus) | 0.2962 | 1.0000 | 0.1327 |
| Hybrid + Reranker | 0.4836 | 1.0000 | 0.1095 |
| Agentic RAG (Full) | 0.4836 | 1.0000 | 0.1147 |
