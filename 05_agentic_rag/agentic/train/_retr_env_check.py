#!/usr/bin/env python3
"""检索环境只读验证(用于确认某个 venv 与 Milvus 索引口径一致)。

验证内容:FirstStageRetriever 能否正常初始化(BGE-M3 文本 + Qwen3-VL 图像 + Milvus),
并对一条真实 query 做一次文本检索、一次图像检索,打印命中分数与文本,
用来判断"查询向量"是否与"索引里存的文档向量"口径一致(分数正常、命中相关即一致)。
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

from retrieval_adapter import FirstStageRetriever, load_config  # noqa: E402
from evidence_selection import select_top_evidence  # noqa: E402

INPUT = str(HERE.parent / "outputs/stage2_calibration/agent_context_v11_train3200.jsonl")


def main() -> None:
    # 从数据集里取第一条带图像的真实 query
    q_img, q_txt = "", ""
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t.get("query_image_path"):
                q_img = t["query_image_path"]
                q_txt = t.get("original_query") or t.get("question") or ""
                break
    print(f"[验证] 查询图像 = {q_img}", flush=True)
    print(f"[验证] 查询文本 = {q_txt[:100]}", flush=True)
    print(f"[验证] 图像文件存在 = {Path(q_img).exists()}", flush=True)

    cfg = load_config()
    cfg["retrieval"]["text_device"] = "cuda:2"
    cfg["retrieval"]["image_device"] = "cuda:1"
    print("[验证] 开始初始化检索器(加载 BGE-M3 + Qwen3-VL 图像编码器 + 连接 Milvus)……", flush=True)
    t0 = time.time()
    r = FirstStageRetriever(cfg)
    print(f"[验证] 检索器初始化完成,用时 {time.time()-t0:.1f}s", flush=True)

    # 文本检索
    t1 = time.time()
    txt = r.search_text(q_txt, k=20)
    print(f"[验证] 文本检索:返回 {len(txt)} 条,用时 {time.time()-t1:.2f}s", flush=True)
    for i, h in enumerate(txt[:3]):
        print(f"    文本命中[{i}] 分数={h.get('score'):.4f} | {h.get('text','')[:70]}", flush=True)

    # 图像检索
    t2 = time.time()
    img = r.search_image(q_img, k=20)
    print(f"[验证] 图像检索:返回 {len(img)} 条,用时 {time.time()-t2:.2f}s", flush=True)
    for i, h in enumerate(img[:3]):
        print(f"    图像命中[{i}] 分数={h.get('score'):.4f} | doc={h.get('doc_id','')} page={h.get('page_idx')} | {h.get('text','')[:50]}", flush=True)

    top = select_top_evidence(txt + img, select_k=12)
    print(f"[验证] 合并后 select_top_evidence 取到 {len(top)} 条候选", flush=True)
    print("[验证] 结论:检索环境可用,口径请人工看上面分数/文本是否合理", flush=True)


if __name__ == "__main__":
    main()
