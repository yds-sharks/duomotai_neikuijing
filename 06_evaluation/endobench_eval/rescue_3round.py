#!/usr/bin/env python3
"""3-round rescue evaluation on wrong answers from single-round eval.

Key design decisions:
- M+ (retained) candidates keep their ORIGINAL origin label (image/text),
  NOT "retained", because the model has never seen "(retained)" in training.
- M- (search_history) keeps the "search_history" label (model saw 11 in SFT).
- For REWRITE+wrong: force inject top-3 original candidates as M+ (model dropped all).
- For ACCEPT+wrong: carry forward the model's kept evidence as M+.
- Run up to 3 rounds; use the last round's kept evidence for generation.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parent
AGENTIC_ROOT = Path("/mnt/data_1/yds/多模态/rerank_image_and_text/agentic")
sys.path.insert(0, str(AGENTIC_ROOT / "code"))
sys.path.insert(0, str(AGENTIC_ROOT / "train"))

from context_agent import SYSTEM_PROMPT_V11, USER_TEMPLATE, format_candidates  # noqa: E402
from ctrl_data_common import build_messages, render_chat  # noqa: E402
from grpo_reward import parse_action  # noqa: E402
from trajectory_runtime import (candidate_id, normalize_candidate,  # noqa: E402
                                search_history_item_multi)

DEFAULT_GEN_MODEL = "/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]
NULL_STRINGS = {"", "none", "null", "n/a"}

SCENE_TO_ORGAN = {
    "Gastroscopy": "胃",
    "Colonoscopy": "结直肠",
    "Capsule Endoscopy": "小肠",
}


def is_null_option(v: Any) -> bool:
    return str(v or "").strip().lower() in NULL_STRINGS


def build_options(ex: Dict[str, Any]) -> Dict[str, str]:
    return {L: str(ex[L]).strip() for L in OPTION_LETTERS
            if ex.get(L) is not None and not is_null_option(ex.get(L))}


def build_text_query(ex: Dict[str, Any], options: Dict[str, str]) -> str:
    q = str(ex.get("question") or "").strip()
    opt = "\n".join(f"{L}: {t}" for L, t in options.items())
    return f"{q}\n{opt}" if opt else q


def dump_query_image(ex: Dict[str, Any], img_dir: Path) -> str:
    idx = ex.get("index")
    out = img_dir / f"eb_{idx}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    raw = ex.get("image")
    if not raw:
        return ""
    try:
        out.write_bytes(base64.b64decode(raw))
        return str(out)
    except Exception:
        return ""


def hit_to_ev(h: Dict[str, Any]) -> Dict[str, Any]:
    return {"text": h.get("text", ""), "image_path": h.get("image_path", ""),
            "doc_name": h.get("doc_name", ""), "page_idx": h.get("page_idx", ""),
            "score": h.get("score", 0.0)}


def build_round_candidates_safe(
    retained: List[Dict[str, Any]],
    failed_queries: List[str],
    new_cands: List[Dict[str, Any]],
    *,
    no_mplus: bool = False,
    no_mminus: bool = False,
) -> tuple:
    """[M+ with ORIGINAL labels] + [M- breadcrumb] + [new retrieval].

    Ablation flags:
      no_mplus  – skip cross-round evidence retention (M+)
      no_mminus – skip failed-query breadcrumb (M-)
    """
    cand: List[Dict[str, Any]] = []
    n_retained = 0
    if retained and not no_mplus:
        cand += [dict(it) for it in retained]
        n_retained = len(cand)
    n_bc = 0
    if failed_queries and not no_mminus:
        cand += [search_history_item_multi(failed_queries)]
        n_bc = 1
    cand += new_cands
    return cand, n_retained, n_bc


class RescueEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        from retrieval_adapter import FirstStageRetriever, load_config as load_retr_config
        cfg = load_retr_config()
        cfg["retrieval"]["text_device"] = args.retr_text_device
        cfg["retrieval"]["image_device"] = args.retr_image_device
        if getattr(args, 'milvus_db_path', '') and args.milvus_db_path:
            cfg["retrieval"]["milvus_db_path"] = args.milvus_db_path
        self.retriever = FirstStageRetriever(cfg)
        self.retriever.search_text("warmup", k=1)
        print(f"[retr] online: text={args.retr_text_device} image={args.retr_image_device}", flush=True)

        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.ctrl_proc = AutoProcessor.from_pretrained(args.ctrl_model, trust_remote_code=True)
        self.ctrl_model = AutoModelForImageTextToText.from_pretrained(
            args.ctrl_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).to(args.ctrl_device)
        self.ctrl_model.eval()
        self.ctrl_model.config.use_cache = True
        print(f"[ctrl] loaded {Path(args.ctrl_model).name} on {args.ctrl_device}", flush=True)

        from gen_scorer import AnswerScorer
        self.scorer = AnswerScorer(args.gen_model, device=args.gen_device,
                                   query_edge=768, ev_edge=384, max_images=8)
        print(f"[gen] scorer loaded on {args.gen_device}", flush=True)

    def initial_candidates(self, qimg: str, text_query: str, *,
                           level1: str = "", text_query_zh: str = "") -> List[Dict[str, Any]]:
        from evidence_selection import select_top_evidence
        a = self.args
        search_query = text_query_zh if text_query_zh else text_query
        img_hits = self.retriever.search_image(qimg, k=a.image_k, level1=level1) if qimg else []
        txt_hits = self.retriever.search_text(search_query, k=a.text_k, level1=level1) if search_query else []
        cands = (select_top_evidence(img_hits, select_k=a.image_select_k)
                 + select_top_evidence(txt_hits, select_k=a.text_select_k))
        return [normalize_candidate(h) for h in cands]

    @torch.no_grad()
    def decide(self, ex, qid, options, text_query, qimg, cands, current_query="") -> Dict[str, Any]:
        user_text = USER_TEMPLATE.format(
            qid=qid, query_type=f"{ex.get('scene', '')}/{ex.get('task', '')}",
            original_query=text_query, current_query=current_query or text_query,
            question=str(ex.get("question") or ""),
            options=json.dumps(options, ensure_ascii=False),
            candidate_block=format_candidates(cands),
        )
        sample = {"system": SYSTEM_PROMPT_V11, "user_text": user_text,
                  "query_image_path": qimg,
                  "evidence_image_paths": [str(c.get("image_path") or "") for c in cands]}
        msgs, images = build_messages(sample, max_images=8, query_edge=768, ev_edge=384,
                                      include_target=False)
        text = render_chat(self.ctrl_proc, msgs, add_generation_prompt=True)
        enc = self.ctrl_proc(text=[text], images=images if images else None,
                             return_tensors="pt").to(self.args.ctrl_device)
        eos = self.ctrl_proc.tokenizer.eos_token_id
        gen = self.ctrl_model.generate(**enc, do_sample=False,
                                       max_new_tokens=self.args.max_new_tokens,
                                       pad_token_id=eos)
        new = gen[0][enc["input_ids"].shape[1]:]
        out_text = self.ctrl_proc.tokenizer.decode(new, skip_special_tokens=True)
        act = parse_action(out_text, len(cands))
        act["raw_text"] = out_text
        return act

    def run_sample(self, ex: Dict[str, Any], qimg: str, orig_result: Dict[str, Any],
                   *, level1: str = "", text_query_zh: str = "") -> Dict[str, Any]:
        """Run multi-round rescue on a single sample (wrong or correct).

        ParseFail fallback: when any round's controller output is unparseable,
        skip that round's state update and continue.  If no valid evidence
        survives after all rounds, fall back to the original single-round
        prediction (deterministic, no re-generation).
        """
        a = self.args
        idx = ex.get("index")
        qid = f"eb_{idx}"
        options = build_options(ex)
        gold = str(ex.get("answer") or "").strip()
        text_query = build_text_query(ex, options)
        no_mplus = getattr(a, "no_mplus", False)
        no_mminus = getattr(a, "no_mminus", False)
        mplus_topk = getattr(a, "mplus_topk", 3)

        # ---- Round 0: initial retrieval + controller ----
        t0 = time.time()
        cands_r0 = self.initial_candidates(qimg, text_query, level1=level1,
                                           text_query_zh=text_query_zh)
        action_r0 = self.decide(ex, qid, options, text_query, qimg, cands_r0)
        r0_latency = time.time() - t0

        # Build M+/M- from FRESH round-0 action (respecting ablation flags)
        retained: List[Dict[str, Any]] = []
        failed_queries: List[str] = []
        suppressed: set = set()
        for c in cands_r0:
            cid = candidate_id(c)
            if cid:
                suppressed.add(cid)

        fresh_action = action_r0["action"]
        fresh_keep = action_r0.get("keep", [])
        r0_parse_ok = action_r0.get("parsed", True)

        if r0_parse_ok:
            if fresh_action == "ACCEPT" and fresh_keep and not no_mplus:
                retained = [cands_r0[i] for i in fresh_keep if 0 <= i < len(cands_r0)]
            elif fresh_action == "REWRITE":
                if not no_mplus:
                    retained = list(cands_r0[:mplus_topk])
                if not no_mminus:
                    failed_queries.append(text_query)

        # Determine round-1 retrieval query
        rw_query_r0 = action_r0.get("rewrite_query", "") or orig_result.get("rewrite_query", "")
        current_query = rw_query_r0 if rw_query_r0 else text_query

        best_evidence: List[Dict[str, Any]] = []

        rounds_log = [{
            "round_idx": 0,
            "action": action_r0["action"],
            "keep": action_r0.get("keep", []),
            "rewrite_query": action_r0.get("rewrite_query", ""),
            "query": text_query,
            "n_cands": len(cands_r0),
            "n_retained": 0,
            "n_breadcrumb": 0,
            "parse_ok": r0_parse_ok,
            "latency_s": round(r0_latency, 2),
            "keep_docs": [str(cands_r0[i].get("doc_name", "")) for i in action_r0.get("keep", [])
                          if 0 <= i < len(cands_r0)],
        }]

        # Extract round-0 evidence (only if parse ok)
        if r0_parse_ok and action_r0["action"] == "ACCEPT":
            ev = [hit_to_ev(cands_r0[i]) for i in action_r0.get("keep", [])
                  if 0 <= i < len(cands_r0)]
            if ev:
                best_evidence = ev

        # ---- Rounds 1..max_rounds-1: rescue ----
        cand_list = cands_r0
        for round_idx in range(1, a.max_rounds):
            t_round = time.time()
            hits = self.retriever.search_text(current_query, k=a.rewrite_topk, level1=level1)
            new_cands = [normalize_candidate(h) for h in hits[:a.rewrite_topk]
                         if candidate_id(normalize_candidate(h)) not in suppressed]

            cand_list, n_ret, n_bc = build_round_candidates_safe(
                retained, failed_queries, new_cands,
                no_mplus=no_mplus, no_mminus=no_mminus)

            action_info = self.decide(ex, qid, options, text_query, qimg, cand_list,
                                      current_query=current_query)
            round_latency = time.time() - t_round
            r_parse_ok = action_info.get("parsed", True)

            # Suppress seen
            for c in new_cands:
                cid = candidate_id(c)
                if cid:
                    suppressed.add(cid)

            # Extract evidence (exclude breadcrumb)
            bc_start, bc_end = n_ret, n_ret + n_bc
            kept_evidence = [hit_to_ev(cand_list[i]) for i in action_info.get("keep", [])
                             if 0 <= i < len(cand_list) and not (bc_start <= i < bc_end)]

            rounds_log.append({
                "round_idx": round_idx,
                "action": action_info["action"],
                "keep": action_info.get("keep", []),
                "rewrite_query": action_info.get("rewrite_query", ""),
                "query": current_query,
                "n_cands": len(cand_list),
                "n_retained": n_ret,
                "n_breadcrumb": n_bc,
                "parse_ok": r_parse_ok,
                "latency_s": round(round_latency, 2),
                "keep_docs": [str(cand_list[i].get("doc_name", "")) for i in action_info.get("keep", [])
                              if 0 <= i < len(cand_list) and not (bc_start <= i < bc_end)],
            })

            # ParseFail: skip state update, don't break — continue to next round
            if not r_parse_ok:
                continue

            if action_info["action"] == "ACCEPT":
                if kept_evidence:
                    best_evidence = kept_evidence
                break
            else:
                # REWRITE: update M+/M- for next round
                if kept_evidence:
                    best_evidence = kept_evidence
                    if not no_mplus:
                        retained = [cand_list[i] for i in action_info.get("keep", [])
                                    if 0 <= i < len(cand_list) and not (bc_start <= i < bc_end)]
                if not no_mminus:
                    failed_queries.append(current_query)
                current_query = action_info.get("rewrite_query", "") or current_query

        # Deterministic fallback: if no valid evidence, use original single-round prediction
        if not best_evidence:
            return {
                "index": idx, "qid": qid, "scene": ex.get("scene", ""), "task": ex.get("task", ""),
                "category": ex.get("category", ""), "dataset": ex.get("dataset", ""),
                "gold": gold, "pred": orig_result.get("pred", ""),
                "correct": orig_result.get("correct", False),
                "logp_gold": orig_result.get("logp_gold", 0.0),
                "p_gold": orig_result.get("p_gold", 0.0),
                "orig_action": fresh_action, "orig_correct": orig_result.get("correct", False),
                "rescued": False,
                "n_rounds": len(rounds_log),
                "n_evidence": 0,
                "evidence_docs": [],
                "fallback_to_orig": True,
                "rounds": rounds_log,
            }

        # Final generation
        verdict = self.scorer.judge(
            str(ex.get("question") or ""), options, gold, qimg, best_evidence)

        return {
            "index": idx, "qid": qid, "scene": ex.get("scene", ""), "task": ex.get("task", ""),
            "category": ex.get("category", ""), "dataset": ex.get("dataset", ""),
            "gold": gold, "pred": verdict["pred"], "correct": verdict["correct"],
            "logp_gold": verdict["logp_gold"], "p_gold": verdict["p_gold"],
            "orig_action": fresh_action, "orig_correct": orig_result.get("correct", False),
            "rescued": bool(verdict["correct"] and not orig_result.get("correct", False)),
            "n_rounds": len(rounds_log),
            "n_evidence": len(best_evidence),
            "evidence_docs": [str(e.get("doc_name", "")) for e in best_evidence],
            "fallback_to_orig": False,
            "rounds": rounds_log,
        }

    def close(self):
        if self.retriever is not None:
            try:
                self.retriever.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(HERE / "wrong50_subset.jsonl"))
    ap.add_argument("--benchmark", default="Saint-lsy/EndoBench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--img-dir", default=str(HERE / "endobench_images"))
    ap.add_argument("--translate-cache", default=str(HERE / "endobench_translated_queries.jsonl"))
    ap.add_argument("--ctrl-model", default="/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/train/ckpt_grpo_v4_u50")
    ap.add_argument("--ctrl-device", default="cuda:0")
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--gen-device", default="cuda:1")
    ap.add_argument("--retr-text-device", default="cuda:1")
    ap.add_argument("--retr-image-device", default="cuda:0")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--image-k", type=int, default=20)
    ap.add_argument("--text-k", type=int, default=20)
    ap.add_argument("--image-select-k", type=int, default=6)
    ap.add_argument("--text-select-k", type=int, default=6)
    ap.add_argument("--rewrite-topk", type=int, default=5)
    ap.add_argument("--use-organ-filter", action="store_true", default=False)
    ap.add_argument("--milvus-db-path", default="",
                    help="override Milvus DB path (for parallel evals)")
    ap.add_argument("--no-mplus", action="store_true", default=False,
                    help="Ablation: disable M+ (cross-round evidence retention)")
    ap.add_argument("--no-mminus", action="store_true", default=False,
                    help="Ablation: disable M- (failed query breadcrumb)")
    ap.add_argument("--mplus-topk", type=int, default=3,
                    help="Number of top candidates to inject as M+ for REWRITE (default 3)")
    ap.add_argument("--out", default=str(HERE / "rescue50_results.jsonl"))
    args = ap.parse_args()

    # Load wrong subset
    subset_results = {}
    for line in open(args.subset):
        d = json.loads(line)
        subset_results[d["qid"]] = d

    # Load translated queries
    translate_map: Dict[int, str] = {}
    tc_path = Path(args.translate_cache)
    if tc_path.exists():
        for line in tc_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                try:
                    row = json.loads(line)
                    translate_map[row["index"]] = row.get("query_zh", "")
                except Exception:
                    pass
        print(f"[trans] {len(translate_map)} translated queries loaded", flush=True)

    # Load EndoBench from HF, filter to wrong subset
    from datasets import load_dataset
    ds = load_dataset(args.benchmark)[args.split]
    img_dir = Path(args.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Build index -> ds position mapping
    subset_indices = set()
    for qid in subset_results:
        # qid = "eb_{index}"
        idx = int(qid.replace("eb_", ""))
        subset_indices.add(idx)

    print(f"[rescue] {len(subset_indices)} wrong samples to rescue", flush=True)

    evaluator = RescueEvaluator(args)

    out_path = Path(args.out)
    done_qids = set()
    if out_path.exists():
        for line in open(out_path):
            if line.strip():
                try:
                    done_qids.add(json.loads(line)["qid"])
                except Exception:
                    pass
        print(f"[rescue] resuming: {len(done_qids)} already done", flush=True)

    t0 = time.time()
    n_done = 0
    n_rescued = 0
    with open(out_path, "a", encoding="utf-8") as fo:
        for i in range(len(ds)):
            ex = ds[i]
            idx = ex.get("index")
            if idx not in subset_indices:
                continue
            qid = f"eb_{idx}"
            if qid in done_qids:
                continue

            orig = subset_results[qid]
            qimg = dump_query_image(ex, img_dir)
            level1 = SCENE_TO_ORGAN.get(str(ex.get("scene", "")), "") if args.use_organ_filter else ""
            text_query_zh = translate_map.get(idx, "")

            t1 = time.time()
            try:
                result = evaluator.run_sample(ex, qimg, orig,
                                              level1=level1, text_query_zh=text_query_zh)
            except Exception as e:
                print(f"[rescue] {qid} ERROR: {e}", flush=True)
                result = {"qid": qid, "index": idx, "error": str(e),
                          "correct": False, "rescued": False,
                          "orig_action": orig.get("action", ""), "orig_correct": False}

            fo.write(json.dumps(result, ensure_ascii=False) + "\n")
            fo.flush()
            n_done += 1
            if result.get("rescued"):
                n_rescued += 1
            dt = time.time() - t1
            print(f"[rescue] {n_done}/{len(subset_indices)} {qid} "
                  f"rescued={result.get('rescued', False)} "
                  f"({dt:.1f}s, total={time.time()-t0:.0f}s)", flush=True)

    evaluator.close()

    # Summary
    results = [json.loads(l) for l in open(out_path) if l.strip()]
    n = len(results)
    n_rescued = sum(1 for r in results if r.get("rescued"))
    n_still_wrong = sum(1 for r in results if not r.get("correct"))
    n_error = sum(1 for r in results if r.get("error"))

    by_orig_action: Dict[str, Dict[str, int]] = {}
    for r in results:
        act = r.get("orig_action", "unknown")
        if act not in by_orig_action:
            by_orig_action[act] = {"n": 0, "rescued": 0}
        by_orig_action[act]["n"] += 1
        if r.get("rescued"):
            by_orig_action[act]["rescued"] += 1

    print(f"\n===== RESCUE SUMMARY =====")
    print(f"Total: {n}")
    print(f"Rescued (wrong->correct): {n_rescued} ({100*n_rescued/max(n,1):.1f}%)")
    print(f"Still wrong: {n_still_wrong}")
    print(f"Errors: {n_error}")
    for act, stats in sorted(by_orig_action.items()):
        print(f"  {act}: {stats['rescued']}/{stats['n']} rescued "
              f"({100*stats['rescued']/max(stats['n'],1):.1f}%)")


if __name__ == "__main__":
    main()
