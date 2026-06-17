# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------


import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from src.retriever.bm25_retriever import BM25
from src.retriever.milvus_retriever import MilvusRetriever
from src.client.llm_local_client import request_chat
from src.reranker.bge_m3_reranker import BGEM3ReRanker
from src.constant import bge_reranker_tuned_model_path
from src.utils import merge_docs, post_processing

# warmstart
bm25_retriever = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)


while True:
    query = input("输入—>")

    # BM25 + Milvus 并行检索
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bm25 = executor.submit(bm25_retriever.retrieve_topk, query, 10)
        future_milvus = executor.submit(milvus_retriever.retrieve_topk, query, 10)
        bm25_docs = future_bm25.result()
        milvus_docs = future_milvus.result()
    t2 = time.time()
    print(f"检索耗时: {t2 - t1:.2f}s")

    print("BM25召回样例:")
    print(bm25_docs)
    print("="*100)

    print("BGE-M3召回样例:")
    print(milvus_docs)
    print("="*100)


    # 去重
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    print(merged_docs)
    print("="*100)


    # 精排 
    ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=5)
    print(ranked_docs)
    print("="*100)


    # 答案
    context = "\n".join(["【" + str(idx+1) + "】" + doc.page_content for idx, doc in enumerate(ranked_docs)])
    res_handler = request_chat(query, context, stream=True)
    response = ""
    for r in res_handler:
        uttr = r.choices[0].delta.content
        if uttr is None:
            continue
        response += uttr 
        print(uttr, end='')
    print("\n" + "="*100)

    # 后处理
    answer = post_processing(response, ranked_docs)
    print("\n答案—>", answer)

