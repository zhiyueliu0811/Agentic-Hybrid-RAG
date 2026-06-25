# -*- coding: utf-8 -*-
# Phase 3: 偏好对构造器
# 读取打分后的候选答案，构造 chosen/rejected 偏好对（LLaMA-Factory pairwise 格式）

import json
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

# --- 配置 ---
SCORED_PATH = PROJECT_DIR / "data/rl/scored_candidates.jsonl"
OUTPUT_PATH = PROJECT_DIR / "LLaMA-Factory-main/data/rag_preference.json"
MIN_REWARD_GAP = 0.03  # chosen-rejected 最小分差（降低，因 SFT 模型输出 diversity 有限）
LLAMA_FACTORY_DIR = PROJECT_DIR / "LLaMA-Factory-main"

SYSTEM_PROMPT = "你是特斯拉电动汽车Model 3车型的用户手册问答系统，请严格基于提供的信息回答用户问题。"


def main():
    print(f"Reading {SCORED_PATH}...")
    records = []
    with open(SCORED_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} scored records.")

    # 构造偏好对
    pairs = []
    skipped_low_gap = 0
    skipped_no_variation = 0
    for record in tqdm(records, desc="Building pairs"):
        query = record["query"]
        candidates = record["candidates"]

        if len(candidates) < 2:
            skipped_no_variation += 1
            continue

        # 按 reward 降序排列
        sorted_cands = sorted(candidates, key=lambda c: c["score"]["reward"], reverse=True)
        chosen = sorted_cands[0]
        rejected = sorted_cands[-1]

        gap = chosen["score"]["reward"] - rejected["score"]["reward"]
        if gap < MIN_REWARD_GAP:
            skipped_low_gap += 1
            continue

        # 只要分差够大就保留（不设最低 reward 门槛）

        pair = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "response": [chosen["text"], rejected["text"]],
        }
        pairs.append(pair)

    print(f"\nBuilt {len(pairs)} preference pairs.")
    print(f"  Skipped (low reward gap): {skipped_low_gap}")
    print(f"  Skipped (no variation): {skipped_no_variation}")

    # 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Output: {OUTPUT_PATH}")
    print(f"Total pairs: {len(pairs)}")

    # 打印几个例子供人工检查
    print("\n=== Sample pairs for manual review ===")
    for pair in pairs[:3]:
        print(f"\nQuery: {pair['messages'][1]['content'][:80]}...")
        print(f"  Chosen (reward ≈high): {pair['response'][0][:120]}...")
        print(f"  Rejected (reward ≈low): {pair['response'][1][:120]}...")
        print("---")


if __name__ == "__main__":
    main()
