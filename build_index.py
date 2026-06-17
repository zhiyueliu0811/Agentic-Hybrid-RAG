# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------


import os
import pickle
from src.parser.pdf_parse import load_pdf, texts_split
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.constant import raw_docs_path, clean_docs_path, split_docs_path
from src.client.llm_clean_client import request_llm_clean

# 解析pdf
raw_docs = load_pdf()
print("文档page数:", len(raw_docs))
with open(raw_docs_path, "wb") as f:
    pickle.dump(raw_docs, f)

# 文本清洗和整理
clean_docs = request_llm_clean(raw_docs)
print("清洗后文档page数:", len(clean_docs))
with open(clean_docs_path, "wb") as f:
    pickle.dump(clean_docs, f)

# 文档切分
split_docs = texts_split(clean_docs)
print("解析后文档总数:", len(split_docs))
with open(split_docs_path, "wb") as f:
    pickle.dump(split_docs, f)

# 索引入库
bm25_retriever = BM25(split_docs) 
candidate_docs = bm25_retriever.retrieve_topk("介绍一下离车后自动上锁功能", topk=3)
print("BM25召回样例:")
print(candidate_docs)
print("="*100)

milvus_retriever = MilvusRetriever(split_docs) 
candidate_docs = milvus_retriever.retrieve_topk("介绍一下离车后自动上锁功能", topk=3)
print("BGE-M3召回样例:")
print(candidate_docs)
