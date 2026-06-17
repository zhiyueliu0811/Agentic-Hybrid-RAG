"""QueryAgent._parse_response() 边界测试"""

import json
import pytest

from src.agents.query_agent import QueryAgent


class TestParseResponse:
    def setup_method(self):
        self.agent = QueryAgent(llm_client=None, model_name="test")

    def test_valid_json(self):
        """正常 JSON"""
        raw = json.dumps({
            "rewritten_query": "如何充电",
            "need_retrieval": True,
            "reasoning": "简单改写",
            "query_type": "fact_qa",
            "intent": "knowledge_qa",
        })
        result = self.agent._parse_response(raw, "怎么充电")
        assert result is not None
        assert result["rewritten_query"] == "如何充电"

    def test_json_with_markdown_fence(self):
        """markdown 代码块包裹的 JSON"""
        raw = '```json\n{"rewritten_query": "test", "need_retrieval": true, "reasoning": "", "query_type": "fact_qa", "intent": "knowledge_qa"}\n```'
        result = self.agent._parse_response(raw, "test")
        assert result is not None
        assert result["rewritten_query"] == "test"

    def test_malformed_json_returns_none(self):
        """畸形 JSON"""
        result = self.agent._parse_response("not json at all", "test")
        assert result is None

    def test_empty_string_returns_none(self):
        """空字符串"""
        result = self.agent._parse_response("", "test")
        assert result is None

    def test_missing_field_still_valid(self):
        """缺少部分字段（setdefault 兜底）"""
        raw = json.dumps({
            "rewritten_query": "test",
            "need_retrieval": True,
            "reasoning": "",
            "query_type": "fact_qa",
            "intent": "knowledge_qa",
        })
        result = self.agent._parse_response(raw, "test")
        assert result is not None
        assert result["need_retrieval"] is True
