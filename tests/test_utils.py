"""merge_docs() 边界测试"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def _make_doc(unique_id, page_content="test content", parent_id=None):
    meta = {"unique_id": unique_id}
    if parent_id:
        meta["parent_id"] = parent_id
    return Document(page_content=page_content, metadata=meta)


class TestMergeDocs:
    @pytest.fixture(autouse=True)
    def _mock_mongo(self):
        """隔离 MongoDB 连接，避免导入时连数据库"""
        with patch("pymongo.MongoClient", MagicMock()):
            yield

    def test_empty_both(self, monkeypatch):
        from src.utils import merge_docs
        monkeypatch.setattr("src.utils.manual_collection", None)
        result = merge_docs([], [])
        assert result == []

    def test_dedup_same_unique_id(self, monkeypatch):
        from src.utils import merge_docs
        monkeypatch.setattr("src.utils.manual_collection", None)
        d1 = _make_doc("id1")
        d2 = _make_doc("id1")
        result = merge_docs([d1], [d2])
        assert len(result) == 1
        assert result[0].metadata["unique_id"] == "id1"

    def test_keep_distinct_ids(self, monkeypatch):
        from src.utils import merge_docs
        monkeypatch.setattr("src.utils.manual_collection", None)
        d1 = _make_doc("id1")
        d2 = _make_doc("id2")
        result = merge_docs([d1], [d2])
        assert len(result) == 2

    def test_doc_without_unique_id_skipped(self, monkeypatch):
        from src.utils import merge_docs
        monkeypatch.setattr("src.utils.manual_collection", None)
        d1 = Document(page_content="no id")
        result = merge_docs([d1], [])
        assert len(result) == 0

    def test_mixed_dedup_across_lists(self, monkeypatch):
        from src.utils import merge_docs
        monkeypatch.setattr("src.utils.manual_collection", None)
        d1 = _make_doc("a")
        d2 = _make_doc("b")
        d3 = _make_doc("a")
        result = merge_docs([d1, d2], [d3])
        assert len(result) == 2
        ids = {d.metadata["unique_id"] for d in result}
        assert ids == {"a", "b"}
