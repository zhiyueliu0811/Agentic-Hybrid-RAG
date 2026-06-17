# -*- coding: utf-8 -*-
# Badcase Analyzer：读取 RAG 预测结果，自动归类错误类型

import json
import csv
import os
import numpy as np
from text2vec import SentenceModel
from tqdm import tqdm

from src.constant import text2vec_model_path

INPUT_PATH = "data/qa_pairs/test_qa_pair_pred.json"
OUTPUT_JSON = "badcases/badcase_report.json"
OUTPUT_CSV = "badcases/badcase_report.csv"

# 错误类型
RETRIEVAL_ERROR = "retrieval_error"        # 召回不到正确证据
RERANK_ERROR = "rerank_error"             # 召回到了但排序靠后
GENERATION_ERROR = "generation_error"      # 证据正确但生成错误
CITATION_ERROR = "citation_error"          # 答案正确但引用错误
NO_ANSWER_ERROR = "no_answer_error"        # 应该无答案但回答了
UNKNOWN_ERROR = "unknown_error"            # 无法判断

ERROR_LABELS = {
    RETRIEVAL_ERROR: "检索未召回",
    RERANK_ERROR: "精排靠后",
    GENERATION_ERROR: "生成错误",
    CITATION_ERROR: "引用错误",
    NO_ANSWER_ERROR: "应无答案却回答",
    UNKNOWN_ERROR: "无法判断",
}


def calc_sim(sim_model, text_a, text_b):
    """用 text2vec SentenceModel 计算余弦相似度"""
    emb_a = sim_model.encode([text_a])
    emb_b = sim_model.encode([text_b])
    return float(np.dot(emb_a, emb_b.T)[0][0])


def classify(pred_item, sim_model):
    """用规则 + 语义相似度分类错误"""
    gold_answer = pred_item.get("answer", "")
    pred = pred_item.get("pred", {})
    pred_answer = pred.get("answer", "")
    context = pred_item.get("context", "")
    score = pred_item.get("score", 0.0)

    # 1. 应该无答案却回答了
    if gold_answer == "无答案" and pred_answer != "无答案":
        return NO_ANSWER_ERROR, "gold 为无答案但模型给出了回答"

    # 2. 应该有答案却输出无答案
    if gold_answer != "无答案" and pred_answer == "无答案":
        return RETRIEVAL_ERROR, "模型输出无答案，可能未检索到相关证据"

    # 3. 双方都无答案 → 正确
    if gold_answer == "无答案" and pred_answer == "无答案":
        return None, None

    # 4. 答案语义相似度高 → 检查引用
    gold_sim = calc_sim(sim_model, gold_answer, pred_answer)
    if gold_sim > 0.7:
        cite_pages = pred.get("cite_pages", [])
        if not cite_pages:
            return CITATION_ERROR, f"答案正确但无引用页码 (语义相似度: {gold_sim:.3f})"
        return None, None

    # 5. 检查证据是否包含答案
    if context:
        ctx_sim_gold = calc_sim(sim_model, gold_answer, context[:1000])
        ctx_sim_pred = calc_sim(sim_model, pred_answer, context[:1000])

        if ctx_sim_gold > 0.5 and ctx_sim_pred < 0.4:
            return GENERATION_ERROR, f"证据存在答案(gold_sim={ctx_sim_gold:.3f})但生成偏差(pred_sim={ctx_sim_pred:.3f})"
        if ctx_sim_gold < 0.3:
            return RETRIEVAL_ERROR, f"证据与参考答案相似度过低({ctx_sim_gold:.3f})"

    # 6. 低分 + 低相似度 → 检索问题
    if score < 0.3 and gold_sim < 0.5:
        return RETRIEVAL_ERROR, f"得分{score:.2f}且答案相似度{gold_sim:.3f}，可能未召回正确证据"

    return None, None


def main():
    os.makedirs("badcases", exist_ok=True)

    print("加载语义相似度模型...")
    sim_model = SentenceModel(model_name_or_path=text2vec_model_path, device="cpu")

    print(f"读取预测结果: {INPUT_PATH}")
    with open(INPUT_PATH) as f:
        data = json.load(f)
    print(f"共 {len(data)} 条用例")

    # 分类
    badcases = []
    stats = {k: 0 for k in ERROR_LABELS}
    correct = 0

    for item in tqdm(data):
        error_type, reason = classify(item, sim_model)
        if error_type is None:
            correct += 1
            continue

        stats[error_type] += 1
        badcases.append({
            "question": item["question"],
            "gold_answer": item.get("answer", ""),
            "pred_answer": item.get("pred", {}).get("answer", ""),
            "score": item.get("score", 0.0),
            "error_type": error_type,
            "error_label": ERROR_LABELS[error_type],
            "reason": reason,
            "context": item.get("context", "")[:500],
            "cite_pages": item.get("pred", {}).get("cite_pages", []),
        })

    # 输出报告
    with open(OUTPUT_JSON, "w") as f:
        json.dump(badcases, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "gold_answer", "pred_answer", "score",
            "error_type", "error_label", "reason", "cite_pages",
        ])
        writer.writeheader()
        for bc in badcases:
            writer.writerow({k: bc[k] for k in writer.fieldnames})

    # 打印汇总
    print(f"\n正确: {correct} ({correct/len(data)*100:.1f}%)")
    print(f"错误: {len(badcases)}")
    print(f"\n错误分布:")
    for etype, label in ERROR_LABELS.items():
        print(f"  {label}: {stats[etype]}")
    print(f"\n报告已保存:")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  CSV : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
