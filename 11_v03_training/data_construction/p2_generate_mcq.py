#!/usr/bin/env python3
"""P2 题库重建（放量版）：两阶段——并发出题（GPT-5.6-luna）→ 分片三层验证（4B logprob）。

阶段：
  generate（无 GPU，--concurrency 并发）:
    读 p2_source_pool.jsonl（5982 源），每源 2 个不同题型任务，按配额出 ~11000 题；
    输出行含完整 source（排除键 + pdf_context_text），verify 阶段独立于源池文件；
    行号=任务号（确定性生成），断点续跑=跳过已写行数。
  verify（GPU，--shard-index/--shard-count 分片多进程）:
    L1 题干盲答弱（logP < -0.25，argmax 碰对不淘汰）
    L2 看图盲答弱（logP < -0.5）
    L3 oracle 留一检索（排除自身+同书）top3 文段合并：correct 或 logP 提升 > 0.5
    断点续跑=按 gen_id 跳过已验证行。

200 题小样实测（28.5% 通过率）后的 prompt 强化：题干禁用选项关键词与操作/器械/
分型命名（治 procedure L1 只有 32% 的短板）。

用法：
  # 出题（约 2 小时，无 GPU）
  python p2_generate_mcq.py generate --pool ./p2_source_pool.jsonl --n 11000 \
      --outdir ./p2_full --concurrency 8
  # 验证 worker A（卡0 + 检索卡1/2）
  python p2_generate_mcq.py verify --config ../../10_harness/harness_config.json \
      --indir ./p2_full --outdir ./p2_full --shard-index 0 --shard-count 2 \
      --generator-device cuda:0 --text-device cuda:1 --image-device cuda:2
  # 验证 worker B（卡3 + BGE卡1 + 8B共卡3）
  python p2_generate_mcq.py verify --config ../../10_harness/harness_config.json \
      --indir ./p2_full --outdir ./p2_full --shard-index 1 --shard-count 2 \
      --generator-device cuda:3 --text-device cuda:1 --image-device cuda:3
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from leave_one_out_retriever import LeaveOneOutRetriever  # noqa: E402
from p0_check_retrieval_env import gold_keys, load_questions  # noqa: E402
from p1_support_sampling import OptionLogprobScorer, build_oracle_queries, dedup_candidates  # noqa: E402

L1_THRESHOLD = -0.25
L2_THRESHOLD = -0.5
L3_GAIN = 0.5
TOPK_L3 = 3
API_RETRY = 5
API_BACKOFF = [5, 10, 20, 40, 60]  # 秒；502/503 多为瞬时过载，指数退避等它恢复

# 出题配额（先 8000 批，按 200 题小样实测通过率反推，不够再补第二批）
GEN_QUOTA = {"lesion": 2510, "procedure": 2690, "spatial": 1985, "anatomy": 815}
TYPE_NAMES = {
    "lesion": "lesion_finding_identification",
    "procedure": "procedure_operation_recognition",
    "spatial": "spatial_region_understanding",
    "anatomy": "anatomical_site_recognition",
}

TYPE_PROMPTS = {
    "lesion": "出1道【病灶识别/病变严重程度分级/病灶定量】类单选题。焦点：图中病灶的类型、分级（如息肉病理分型、早癌浸润深度分级、炎症程度评分）或大小数量定量",
    "procedure": "出1道【手术流程/操作识别】类单选题。焦点：图中正在进行的操作步骤、检查阶段、器械用途或操作技术要点",
    "spatial": "出1道【空间定位/区域识别】类单选题。焦点：图像视野中的目标位于哪个解剖区域/部位，或视野覆盖的区域范围",
    "anatomy": "出1道【解剖部位/解剖标志识别】类单选题。焦点：图中显示的器官、解剖标志或正常结构辨认",
}

PROMPT_TMPL = """你是内镜医学多选题出题专家。请根据给定的内窥镜图像和原文片段，{task_desc}。

出题要求（防泄露硬约束）：
1. 题干只描述图像中可见的现象、动作与场景（形态、颜色、位置关系、器械外观），【严禁】出现任何选项文本中的关键词及其同义词——特别是操作名称、器械名称、病理分型名称、分级术语、解剖部位命名；这些名称只能出现在选项中
2. 不看图像、只读题干与选项时，无法确定正确答案；题干对四个选项的描述必须保持等可能的中性
3. 4个选项，干扰项与正确答案高度相似（同器官相邻部位/同病灶其他级别/相邻操作步骤/同类器械），不得让答案因语义距离过远而被排除法猜中
4. 正确答案必须能从原文片段或图像中找到明确依据；若原文与图像不符，以图像为准并在 evidence_span 说明
5. 同时写1-2条疑惑点(info_need)：一位看到此图但不熟悉该领域的医生最可能提出的具体问题
6. 严格只输出JSON（无其他文字）：
{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A", "answer_text": "...", "evidence_span": "原文或图中依据", "info_need": ["...", "..."]}}

原文片段：
{ctx}"""


class McqApiClient:
    def __init__(self, cfg_path: str):
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.hdr = {
            "Authorization": "Bearer " + self.cfg["api_key"],
            "User-Agent": "curl/8.5.0",  # WAF 拦 python-urllib 默认 UA，必须带
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 1600, timeout: int = 240) -> str:
        body = json.dumps({"model": self.cfg["model"], "messages": messages, "max_tokens": max_tokens}).encode()
        last = None
        for i in range(API_RETRY):
            try:
                req = urllib.request.Request(self.cfg["endpoint"] + "/chat/completions", data=body, headers=self.hdr)
                r = json.load(urllib.request.urlopen(req, timeout=timeout))
                return r["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(API_BACKOFF[min(i, len(API_BACKOFF) - 1)])
        raise RuntimeError(f"API {API_RETRY}次重试失败: {type(last).__name__}: {last}")

    def generate_mcq(self, image_path: str, ctx: str, qtype: str) -> Dict[str, Any]:
        img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        prompt = PROMPT_TMPL.format(task_desc=TYPE_PROMPTS[qtype], ctx=ctx)
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}]
        s = self.chat(msgs).strip()
        if s.startswith("```"):
            s = s.split("```")[1].lstrip("json").strip()
        d = json.loads(s)
        opts = {str(k).strip().upper()[:1]: str(v).strip() for k, v in (d.get("options") or {}).items()}
        if len(opts) != 4 or set(opts) != {"A", "B", "C", "D"}:
            raise ValueError(f"选项格式不合格: {list(opts)}")
        ans = str(d.get("answer", "")).strip().upper()[:1]
        if ans not in opts:
            raise ValueError(f"答案不在选项中: {ans}")
        q = str(d.get("question", "")).strip()
        if len(q) < 8:
            raise ValueError("题干过短")
        return {
            "question": q, "options": opts, "answer": ans,
            "answer_text": str(opts.get(ans, "")).strip(),
            "evidence_span": str(d.get("evidence_span", "")).strip(),
            "info_need": [str(x) for x in (d.get("info_need") or []) if str(x).strip()][:2],
        }


def build_tasks(rows: List[Dict[str, Any]], n_target: int, seed: int) -> List[Dict[str, Any]]:
    """确定性任务列表：每源 2 个不同题型，按配额取最缺的题型；行号=任务号。"""
    rng = random.Random(seed)
    remain = dict(GEN_QUOTA)
    total_quota = sum(remain.values())
    tasks: List[Dict[str, Any]] = []
    pool = rows[:]
    rng.shuffle(pool)
    for src in pool:
        if len(tasks) >= min(n_target, total_quota):
            break
        picks = []
        for _ in range(2):
            avail = [t for t in remain if remain[t] > 0 and t not in picks]
            if not avail:
                break
            avail.sort(key=lambda t: -remain[t])
            t = avail[0] if len(avail) == 1 or rng.random() < 0.7 else rng.choice(avail)
            picks.append(t)
            remain[t] -= 1
        for t in picks:
            if len(tasks) >= min(n_target, total_quota):
                break
            tasks.append({"qtype": t, "src": src})
    return tasks


def run_generate(args: argparse.Namespace) -> int:
    rows = load_questions(args.pool)
    tasks = build_tasks(rows, args.n, args.seed)
    out_path = Path(args.outdir) / "p2_mcq_raw.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：只记成功的 gen_id；error 行重新出题（追加写，汇总阶段按 gen_id 去重取最后行）
    done_ids: set = set()
    n_err_prev = 0
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                if "error" in r:
                    n_err_prev += 1
                else:
                    done_ids.add(r["gen_id"])
            except (ValueError, KeyError):
                continue
    todo = [(i, t) for i, t in enumerate(tasks) if i not in done_ids]
    print(f"[gen] 任务 {len(tasks)}，已成功 {len(done_ids)}，待跑 {len(todo)}（含重跑 error {n_err_prev}）", flush=True)
    if not todo:
        return 0

    api = McqApiClient(args.api_config)
    t0 = time.time()
    n_ok = n_err = 0

    def work(item: tuple) -> Dict[str, Any]:
        idx, task = item
        src = task["src"]
        rec: Dict[str, Any] = {"gen_id": idx, "qtype": task["qtype"], "query_type": TYPE_NAMES[task["qtype"]],
                               "src_qid": src.get("qid", ""),
                               "src_doc": (src.get("source") or {}).get("doc_name", ""),
                               "query_image_path": str(src.get("query_image_path") or ""),
                               "source": src.get("source") or {}}
        try:
            mcq = api.generate_mcq(rec["query_image_path"], str(rec["source"].get("pdf_context_text", ""))[:1500], task["qtype"])
            rec.update(mcq)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:180]}"
        return rec

    with open(out_path, "a", encoding="utf-8") as fp:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for i, fut in enumerate(as_completed(futs), start=1):
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001  work 内部已捕获，此处防worker级异常中断整批
                    rec = {"gen_id": -1, "error": f"worker: {type(e).__name__}: {str(e)[:180]}"}
                if "error" in rec:
                    n_err += 1
                else:
                    n_ok += 1
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fp.flush()  # 每行落盘：杀进程零丢失，进度实时可见
                if i % 50 == 0 or i == len(todo):
                    print(f"[gen] {i}/{len(todo)} ok={n_ok} err={n_err} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[gen] 完成: ok={n_ok} err={n_err} -> {out_path}")
    return 0


def build_l3_evidence(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ev = []
    for h in pool:
        t = str(h.get("text", "")).strip()
        if t:
            ev.append({"text": t})
        if len(ev) >= TOPK_L3:
            break
    return ev


def verify_one(scorer, retriever, rec: Dict[str, Any]) -> Dict[str, Any]:
    q, opts, gold = rec["question"], rec["options"], rec["answer"]
    img = str(rec.get("query_image_path") or "")

    l1 = scorer.judge(q, opts, gold, image_path="", evidence=[])
    l2 = scorer.judge(q, opts, gold, image_path=img, evidence=[])

    per_query: Dict[str, List[Dict[str, Any]]] = {}
    for name, oq in build_oracle_queries({"question": q, "options": opts, "answer": gold,
                                          "answer_text": rec.get("answer_text", ""),
                                          "source": rec.get("source") or {}}).items():
        if not oq:
            per_query[name] = []
            continue
        # oracle_answer 走双路；oracle_context 是文本上下文查询，跳过图像路省 8B 编码
        q_img = img if name == "oracle_answer" else ""
        per_query[name] = retriever.retrieve(oq, q_img, **gold_keys(rec))["combined"]
    pool = dedup_candidates(per_query)
    ev = build_l3_evidence(pool)
    l3 = scorer.judge(q, opts, gold, image_path=img, evidence=ev) if ev else l2

    return {
        "l1_logp": round(l1["logp_gold"], 4), "l1_correct": l1["correct"],
        "pass_l1": l1["logp_gold"] < L1_THRESHOLD,
        "l2_logp": round(l2["logp_gold"], 4), "l2_pred": l2["pred"],
        "pass_l2": l2["logp_gold"] < L2_THRESHOLD,
        "l3_logp": round(l3["logp_gold"], 4), "l3_correct": l3["correct"],
        "pass_l3": l3["correct"] or (l3["logp_gold"] - l2["logp_gold"]) > L3_GAIN,
        "pool_n": len(pool), "ev_n": len(ev),
    }


def run_verify(args: argparse.Namespace) -> int:
    raw_path = Path(args.indir) / "p2_mcq_raw.jsonl"
    # 按 gen_id 去重（后行覆盖前行：重跑成功的新行覆盖旧 error 行）
    by_id: Dict[int, Dict[str, Any]] = {}
    for line in open(raw_path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            by_id[r["gen_id"]] = r
        except (ValueError, KeyError):
            continue
    rows = [by_id[k] for k in sorted(by_id)]
    todo = [r for r in rows if "error" not in r]
    shard = [r for i, r in enumerate(todo) if i % args.shard_count == args.shard_index]

    out_path = Path(args.outdir) / f"p2_mcq_verified_shard{args.shard_index}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["gen_id"])
            except (ValueError, KeyError):
                continue
    remain = [r for r in shard if r.get("gen_id") not in done_ids]
    print(f"[ver#{args.shard_index}] 分片任务 {len(shard)}，已验证 {len(shard)-len(remain)}，待验证 {len(remain)}", flush=True)
    if not remain:
        return 0

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    gen_cfg = dict(cfg["generator"])
    gen_cfg["device"] = args.generator_device
    ret_cfg = dict(cfg["retrieval"])
    ret_cfg["text_device"] = args.text_device
    ret_cfg["image_device"] = args.image_device

    print(f"[ver#{args.shard_index}] 加载检索器(text@{args.text_device}, image@{args.image_device}) + 4B打分器(@{args.generator_device})", flush=True)
    retriever = LeaveOneOutRetriever(ret_cfg)
    scorer = OptionLogprobScorer(gen_cfg)

    t0 = time.time()
    n_done = 0
    with open(out_path, "a", encoding="utf-8") as fp:
        for r in remain:
            try:
                out = dict(r)
                out.update(verify_one(scorer, retriever, r))
                out["verified"] = out["pass_l1"] and out["pass_l2"] and out["pass_l3"]
            except Exception as e:  # noqa: BLE001
                out = dict(r)
                out["verify_error"] = f"{type(e).__name__}: {str(e)[:180]}"
            fp.write(json.dumps(out, ensure_ascii=False) + "\n")
            fp.flush()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(remain):
                ok = sum(1 for l in open(out_path, encoding="utf-8")
                         if '"verified": true' in l)
                print(f"[ver#{args.shard_index}] {n_done}/{len(remain)} verified累计={ok} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[ver#{args.shard_index}] 完成 -> {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--pool", required=True)
    g.add_argument("--api-config", default=str(_HERE / "api_config.local.json"))
    g.add_argument("--n", type=int, default=8000)
    g.add_argument("--seed", type=int, default=11)
    g.add_argument("--concurrency", type=int, default=3)
    g.add_argument("--outdir", default="./p2_full")

    v = sub.add_parser("verify")
    v.add_argument("--config", required=True)
    v.add_argument("--indir", required=True)
    v.add_argument("--outdir", default=None, help="默认同 indir")
    v.add_argument("--shard-index", type=int, default=0)
    v.add_argument("--shard-count", type=int, default=1)
    v.add_argument("--generator-device", default="cuda:0")
    v.add_argument("--text-device", default="cuda:1")
    v.add_argument("--image-device", default="cuda:2")

    args = ap.parse_args()
    if args.stage == "generate":
        return run_generate(args)
    if args.outdir is None:
        args.outdir = args.indir
    return run_verify(args)


if __name__ == "__main__":
    sys.exit(main())
