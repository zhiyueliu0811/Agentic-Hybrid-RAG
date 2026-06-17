# -*- coding: utf-8 -*-
# RetrievalAgent：BM25 + Milvus 混合召回封装

import time
from concurrent.futures import ThreadPoolExecutor
from src.utils import merge_docs


class RetrievalAgent:
    def __init__(self, bm25_retriever, milvus_retriever):
        self.bm25 = bm25_retriever
        self.milvus = milvus_retriever

    def retrieve(self, query: str, bm25_topk: int = 5, milvus_topk: int = 8) -> dict:
        t1 = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_bm25 = executor.submit(self.bm25.retrieve_topk, query, bm25_topk)
            future_milvus = executor.submit(self.milvus.retrieve_topk, query, milvus_topk)
            bm25_docs = future_bm25.result()
            milvus_docs = future_milvus.result()

        t2 = time.time()
        merged_docs = merge_docs(bm25_docs, milvus_docs)

        return {
            "bm25_docs": bm25_docs,
            "milvus_docs": milvus_docs,
            "merged_docs": merged_docs,
            "elapsed": t2 - t1,
        }
