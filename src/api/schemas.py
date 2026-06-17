# -*- coding: utf-8 -*-
# API Pydantic schemas

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    stream: bool = Field(default=False, description="是否流式返回")


class CitationItem(BaseModel):
    claim: str
    cited_doc_ids: list[int]
    support_status: str
    reason: str


class ChatResponse(BaseModel):
    answer: str
    cite_pages: list = Field(default_factory=list)
    related_images: list = Field(default_factory=list)
    citations: list = Field(default_factory=list)
    citation_verified: bool = True
    citation_unsupported: int = 0
    rewritten_query: str = ""
    query_type: str = ""
    intent: str = "knowledge_qa"
    evidence_enough: bool = True
    evidence_reason: str = ""
    self_rag_triggered: bool = False
    elapsed: float = 0.0
