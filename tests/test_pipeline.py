"""RAGPipeline._default_result() 字段完整性测试

注意：此测试需要 torch/vllm 等 GPU 推理依赖。在 CI 或无 GPU 环境中自动跳过。
"""

import pytest


# 在收集阶段尝试导入，失败则跳过全部测试
try:
    import torch  # noqa
    import vllm   # noqa
    from src.pipeline.rag_pipeline import RAGPipeline
except ImportError as e:
    RAGPipeline = None
    _skip_reason = str(e)


pytestmark = pytest.mark.skipif(
    RAGPipeline is None,
    reason=f"缺少 GPU 推理依赖: {_skip_reason if RAGPipeline is None else ''}"
)


class TestDefaultResult:
    def test_all_fields_present(self):
        assert RAGPipeline is not None
        result = RAGPipeline._default_result("测试问题")
        expected_fields = [
            "query", "answer", "cite_pages", "related_images",
            "rewritten_query", "query_type", "intent", "rewrite_reason",
            "need_retrieval", "bm25_count", "milvus_count", "merged_count",
            "retrieval_time", "ranked_doc_count", "evidence_enough",
            "evidence_reason", "self_rag", "suggested_query",
            "second_merged_count", "raw_answer", "citation_verified",
            "citation_unsupported", "citation_partial",
            "citation_rewrite_triggered", "citation_details",
            "total_time", "_bm25_docs", "_milvus_docs",
            "_ranked_docs", "_ranked_scores",
        ]
        for field in expected_fields:
            assert field in result, f"缺少字段: {field}"

    def test_query_preserved(self):
        result = RAGPipeline._default_result("怎么开空调")
        assert result["query"] == "怎么开空调"

    def test_default_values(self):
        result = RAGPipeline._default_result("")
        assert result["answer"] == ""
        assert result["cite_pages"] == []
        assert result["related_images"] == []
        assert result["need_retrieval"] is True
        assert result["citation_verified"] is True
        assert result["self_rag"] is False
        assert result["citation_rewrite_triggered"] is False
        assert result["bm25_count"] == 0
        assert isinstance(result["total_time"], float)

    def test_independent_instances(self):
        r1 = RAGPipeline._default_result("q1")
        r2 = RAGPipeline._default_result("q2")
        r1["answer"] = "modified"
        assert r2["answer"] == ""
