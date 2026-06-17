"""_extract_claims() 边界测试"""

import pytest

# 模块级导入（纯函数，无副作用）
from src.agents.citation_verifier import _extract_claims


class TestExtractClaims:
    def test_normal_citation(self):
        """标准引用格式：事实【1，2】"""
        claims = _extract_claims("Model 3支持自动上锁功能【1，2】")
        assert len(claims) == 1
        assert claims[0]["claim"] == "Model 3支持自动上锁功能"
        assert claims[0]["cited_doc_ids"] == [1, 2]

    def test_no_citation(self):
        """无引用标记"""
        claims = _extract_claims("Model 3是一辆电动汽车。")
        assert len(claims) == 1
        assert claims[0]["claim"] == "Model 3是一辆电动汽车。"
        assert claims[0]["cited_doc_ids"] == []

    def test_multiple_sentences(self):
        """多句话，各有引用"""
        claims = _extract_claims("离车后自动上锁【1】。大灯延时关闭【2，3】。")
        assert len(claims) == 2
        assert claims[0]["cited_doc_ids"] == [1]
        assert claims[1]["cited_doc_ids"] == [2, 3]

    def test_empty_string(self):
        """空字符串"""
        claims = _extract_claims("")
        assert claims == []

    def test_whitespace_only(self):
        """空白字符串"""
        claims = _extract_claims("   \n  ")
        assert claims == []

    def test_mixed_format_comma(self):
        """逗号分隔的引用编号"""
        claims = _extract_claims("功能说明【1, 2, 3】")
        assert len(claims) == 1
        assert claims[0]["cited_doc_ids"] == [1, 2, 3]

    def test_fullwidth_comma(self):
        """全角逗号分隔"""
        claims = _extract_claims("功能说明【1，2】")
        assert len(claims) == 1
        assert 1 in claims[0]["cited_doc_ids"]
        assert 2 in claims[0]["cited_doc_ids"]

    def test_dedup_cite_ids(self):
        """同一编号只出现一次"""
        claims = _extract_claims("说明【1，1，2】")
        assert claims[0]["cited_doc_ids"] == [1, 2]

    def test_claim_without_period(self):
        """无句末标点的文本"""
        claims = _extract_claims("这是答案【1】")
        assert len(claims) == 1
        assert claims[0]["claim"] == "这是答案"
