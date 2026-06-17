# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  ©丁师兄大模型
# --------------------------------------------

import re
from langchain_core.documents import Document
from src.client.mongodb_config import MongoConfig

manual_collection = MongoConfig.get_collection("manual_text")


def merge_docs(docs1, docs2):
    merged_docs = []
    merged_ids = set()
    candidate_docs = docs1 + docs2

    # 收集所有需要查询的parent_id，批量查询MongoDB
    parent_ids = []
    for doc in candidate_docs:
        pid = doc.metadata.get("parent_id")
        if pid:
            parent_ids.append(pid)

    parent_docs_map = {}
    if parent_ids:
        mongo_results = list(manual_collection.find({"unique_id": {"$in": parent_ids}}))
        parent_docs_map = {r["unique_id"]: r for r in mongo_results}

    for doc in candidate_docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            parent_mg = parent_docs_map.get(parent_id)
            if parent_mg is None:
                continue
            unique_id = parent_mg["unique_id"]
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                parent_doc = Document(page_content=parent_mg["page_content"], metadata=parent_mg["metadata"])
                merged_docs.append(parent_doc)
        else:
            unique_id = doc.metadata.get("unique_id")
            if unique_id and unique_id not in merged_ids:
                merged_ids.add(unique_id)
                merged_docs.append(doc)
    return merged_docs




def post_processing(response, docs):
    all_cites = re.findall("[【](.*?)[】]", response) 
    cites = []
    for cite in all_cites:
        cite = re.sub("[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cite = [int(k) for k in cite.split("，") if k.isdigit()]
        cites.extend(cite)
    cites = list(set(cites))
    answer = re.sub("[【](.*?)[】]", "", response)
    answer = re.sub("[{}【】]", "", answer)

    related_images = []
    pages = []
    for index in cites:
        if index < 1 or index > len(docs):
            continue
        images = docs[index-1].metadata.get("images_info", [])
        page = docs[index-1].metadata.get("page")
        if page is not None:
            pages.append(page)
        for image in images:
            if isinstance(image, dict) and image.get("title"):
                related_images.append(image)
    pages = sorted(list(set(pages)))
    return {
        "answer": answer,
        "cite_pages": pages,
        "related_images": related_images
    }
