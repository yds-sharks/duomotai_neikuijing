#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: align_seeds_to_corpus.py
Purpose:
  Pick ONE anchor docid(pk) per query (qid) by aligning rewritten P1 seeds to corpus via dense+sparse retrieval
  + consistency checks, using a 3-stage decision rule:

  A) Threshold accept (strict/soft) on any seed's top1
  B) If A fails: consensus between two seeds (top1 same, or topK intersection)
     with a minimal safety floor
  C) If B fails: fallback to the best top1 across seeds (low confidence)

Inputs:
  --config: config.yaml
  --seeds:  seeds.jsonl (qid/query/seeds[])

Outputs:
  --qrels_out: qrels.tsv (qid 0 docid rel)  (ONE line per eligible qid)
  --report_out: alignment_report.json (full audit)
  --eligible_qids_out: list of eligible qids that have an anchor
  --audit_tsv_out: anchor-level audit table
  --review_out: TSV for manual review: question + two P1 + anchor text (docid)
"""
import argparse
import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional
import yaml

# --------------------------
# Text similarity helpers
# --------------------------
def norm_for_chargrams(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    # keep CJK + alnum
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)
    return s

def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = norm_for_chargrams(s)
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def containment(seed_toks: List[str], cand_toks: List[str]) -> float:
    if not seed_toks:
        return 0.0
    A, B = set(seed_toks), set(cand_toks)
    return len(A & B) / max(1, len(A))

def clean_one_line(s: str) -> str:
    return (s or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# --------------------------
# Core
# --------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", required=True)

    ap.add_argument("--qrels_out", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--eligible_qids_out", required=True)
    ap.add_argument("--audit_tsv_out", required=True)

    # output for manual review (qid-level, includes question + 2 seeds + anchor)
    ap.add_argument("--review_out", default="", help="TSV for manual review: question + seeds + anchor")
    ap.add_argument("--max_text_chars", type=int, default=400, help="truncate long texts in review output")

    # retrieval depth
    ap.add_argument("--topk_dense", type=int, default=50)
    ap.add_argument("--topk_sparse", type=int, default=50)

    # acceptance thresholds (Stage A)
    ap.add_argument("--min_cos_strict", type=float, default=0.92)
    ap.add_argument("--min_char3_strict", type=float, default=0.20)

    ap.add_argument("--min_cos_soft", type=float, default=0.88)
    ap.add_argument("--min_char3_soft", type=float, default=0.30)
    ap.add_argument("--min_contain_soft", type=float, default=0.60)

    # Stage B consensus
    ap.add_argument("--consensus_topk", type=int, default=5, help="intersection over topK per seed")
    ap.add_argument("--min_cos_floor", type=float, default=0.78, help="minimal cosine floor for consensus")
    ap.add_argument("--min_contain_floor", type=float, default=0.35, help="minimal containment floor for consensus (OR with cos floor)")

    # qrels rel
    ap.add_argument("--rel_p1", type=int, default=3)

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Import your retriever
    from bm25_bge_vectorstore_v2.retrieval_service import HybridRetriever
    retr = HybridRetriever(cfg)

    report: Dict[str, Any] = {
        "mode": "P1_only_anchor_selection_A_threshold_B_consensus_C_fallback",
        "config": {
            "milvus_uri": cfg["milvus"]["uri"],
            "collection": cfg["milvus"]["collection"],
            "fusion_weights": cfg["retrieval"]["fusion"].get("weights", None),
        },
        "thresholds": {
            "strict": {"min_cos": args.min_cos_strict, "min_char3": args.min_char3_strict},
            "soft": {"min_cos": args.min_cos_soft, "min_char3": args.min_char3_soft, "min_contain": args.min_contain_soft},
            "consensus_floor": {"min_cos_floor": args.min_cos_floor, "min_contain_floor": args.min_contain_floor, "topk": args.consensus_topk},
        },
        "items": []
    }

    # Anchor-level audit
    audit_header = [
        "qid", "decision_stage", "confidence", "accept_reason",
        "anchor_pk", "cosine", "char3_jaccard", "token_containment", "file_name", "text_snippet",
        "seed0_top1_pk", "seed0_cos", "seed0_char3", "seed0_contain",
        "seed1_top1_pk", "seed1_cos", "seed1_char3", "seed1_contain",
    ]
    audit_rows = ["\t".join(audit_header)]

    # Review TSV (qid-level)
    review_rows: List[str] = []
    if args.review_out:
        review_header = [
            "qid", "question",
            "seed0_text", "seed1_text",
            "decision_stage", "confidence", "accept_reason",
            "anchor_pk", "cosine", "char3_jaccard", "token_containment", "file_name",
            "anchor_text"
        ]
        review_rows.append("\t".join(review_header))

    eligible_qids: List[str] = []
    qrels_lines: List[str] = []

    # ---- helper: retrieve & score for one seed text ----
    def retrieve_and_score(seed_text: str) -> List[Dict[str, Any]]:
        seed_text = (seed_text or "").strip()
        if not seed_text:
            return []

        dense_out = retr.search_batch([seed_text], topk=args.topk_dense, mode="dense")[0]["results"]
        sparse_out = retr.search_batch([seed_text], topk=args.topk_sparse, mode="sparse")[0]["results"]

        pool: Dict[int, Dict[str, Any]] = {}

        def push(res_list, src: str):
            for r in res_list:
                pk = int(r["pk"])
                item = pool.setdefault(pk, {
                    "pk": pk,
                    "text": r.get("text", "") or "",
                    "file_name": r.get("file_name", ""),
                    "dense": None,
                    "sparse": None,
                })
                if src == "dense":
                    item["dense"] = float(r.get("dense", r.get("score", 0.0)))
                else:
                    item["sparse"] = float(r.get("sparse", r.get("score", 0.0)))

        push(dense_out, "dense")
        push(sparse_out, "sparse")

        seed_char3 = char_ngrams(seed_text, 3)
        seed_toks = retr.tok.tokenize_mixed(seed_text)

        scored: List[Dict[str, Any]] = []
        for pk, item in pool.items():
            cand_text = item["text"] or ""
            cand_char3 = char_ngrams(cand_text, 3)
            char3_jac = jaccard(seed_char3, cand_char3)

            cand_toks = retr.tok.tokenize_mixed(cand_text)
            cont = containment(seed_toks, cand_toks)

            cos = item["dense"] if item["dense"] is not None else -1e9

            c_clean = clean_one_line(cand_text)
            scored.append({
                "pk": pk,
                "cosine": float(cos),
                "char3_jaccard": float(char3_jac),
                "token_containment": float(cont),
                "file_name": item.get("file_name", ""),
                "text_snippet": (c_clean[:160] + ("..." if len(c_clean) > 160 else "")),
                "gold_text": c_clean,  # full clean text
                "has_dense": item["dense"] is not None,
                "has_sparse": item["sparse"] is not None,
            })

        # Sort by cosine then char3 then containment
        scored.sort(key=lambda x: (x["cosine"], x["char3_jaccard"], x["token_containment"]), reverse=True)
        return scored

    # ---- helper: stage A accept for a seed's top1 ----
    def accept_by_threshold(top: Dict[str, Any]) -> Tuple[bool, str]:
        if not top:
            return False, "no_candidate"
        # strict
        if (top["cosine"] >= args.min_cos_strict) and (top["char3_jaccard"] >= args.min_char3_strict):
            return True, "strict_cos+char3"
        # soft
        if (top["cosine"] >= args.min_cos_soft) and (top["char3_jaccard"] >= args.min_char3_soft) and (top["token_containment"] >= args.min_contain_soft):
            return True, "soft_contain+cos+char3"
        return False, "rejected_by_threshold"

    # ---- helper: aggregate score for consensus candidate ----
    def agg_score(x: Dict[str, Any]) -> float:
        # cosine dominates; char3/contain provide tie-breaker
        return float(x.get("cosine", -1e9)) + 0.2 * float(x.get("char3_jaccard", 0.0)) + 0.1 * float(x.get("token_containment", 0.0))

    # ---- process each qid ----
    for row in iter_jsonl(args.seeds):
        qid = str(row["qid"])
        question = row.get("query", "")

        seeds = row.get("seeds", []) or []
        # Only keep non-empty seed texts
        seed_texts = []
        seed_meta = []
        for s in seeds:
            st = (s.get("text") or "").strip()
            if st:
                seed_texts.append(st)
                seed_meta.append({
                    "level": s.get("level", "P1"),
                    "rank_score": s.get("rank_score", None),
                })

        # if no seeds -> cannot anchor
        if not seed_texts:
            report["items"].append({
                "qid": qid,
                "query": question,
                "decision": {"eligible": False, "reason": "no_seed_text"},
                "seeds": []
            })
            continue

        # score each seed
        seed_infos: List[Dict[str, Any]] = []
        for i, st in enumerate(seed_texts):
            scored = retrieve_and_score(st)
            top1 = scored[0] if scored else None
            accepted, reason = accept_by_threshold(top1) if top1 else (False, "no_candidate")
            seed_infos.append({
                "seed_i": i,
                "seed_text": st,
                "seed_level": seed_meta[i].get("level", "P1") if i < len(seed_meta) else "P1",
                "seed_rank_score": seed_meta[i].get("rank_score", None) if i < len(seed_meta) else None,
                "top1": top1,
                "topk": scored[: max(1, args.consensus_topk)],
                "accepted_A": accepted,
                "accept_reason_A": reason,
            })

        # --------------------------
        # Decision A -> B -> C
        # --------------------------
        anchor: Optional[Dict[str, Any]] = None
        decision_stage = ""
        accept_reason = ""
        confidence = ""

        # A) threshold accept: choose best among accepted seeds
        accepted_seeds = [si for si in seed_infos if si["accepted_A"] and si["top1"]]
        if accepted_seeds:
            # pick the best top1 by agg_score
            accepted_seeds.sort(key=lambda si: agg_score(si["top1"]), reverse=True)
            anchor = accepted_seeds[0]["top1"]
            decision_stage = "A_threshold"
            accept_reason = accepted_seeds[0]["accept_reason_A"]
            confidence = "high"
        else:
            # B) consensus between seeds
            # support N seeds, but your case is 2; implement generic intersection voting
            # Build pk -> list of appearances with scores
            pk_votes: Dict[int, List[Dict[str, Any]]] = {}
            for si in seed_infos:
                for cand in si["topk"]:
                    pk_votes.setdefault(int(cand["pk"]), []).append(cand)

            # candidates with >=2 votes (i.e., appear in at least two seeds' topK)
            consensus_pks = [pk for pk, lst in pk_votes.items() if len(lst) >= 2]

            # prefer top1 consensus if exists
            top1_pks = []
            for si in seed_infos:
                if si["top1"]:
                    top1_pks.append(int(si["top1"]["pk"]))
            top1_consensus_pk = None
            if len(top1_pks) >= 2 and len(set(top1_pks)) == 1:
                top1_consensus_pk = top1_pks[0]

            def pass_floor(c: Dict[str, Any]) -> bool:
                # OR condition: cosine >= floor OR containment >= floor
                return (float(c.get("cosine", -1e9)) >= args.min_cos_floor) or (float(c.get("token_containment", 0.0)) >= args.min_contain_floor)

            if top1_consensus_pk is not None:
                # take the best instance (they are same pk; choose the one with higher score record)
                cand_list = pk_votes.get(top1_consensus_pk, [])
                if cand_list:
                    cand_list.sort(key=lambda c: agg_score(c), reverse=True)
                    if pass_floor(cand_list[0]):
                        anchor = cand_list[0]
                        decision_stage = "B_consensus"
                        accept_reason = "pk_consensus_top1(votes=2)"
                        confidence = "mid"

            if anchor is None and consensus_pks:
                # select best pk in topK intersection by mean agg_score
                scored_consensus: List[Tuple[int, float, Dict[str, Any]]] = []
                for pk in consensus_pks:
                    lst = pk_votes[pk]
                    # compute mean aggregate score
                    mean_s = sum(agg_score(c) for c in lst) / max(1, len(lst))
                    # pick representative cand (best one)
                    rep = sorted(lst, key=lambda c: agg_score(c), reverse=True)[0]
                    scored_consensus.append((pk, mean_s, rep))
                scored_consensus.sort(key=lambda x: x[1], reverse=True)
                best_pk, _, rep = scored_consensus[0]
                if pass_floor(rep):
                    anchor = rep
                    decision_stage = "B_consensus"
                    accept_reason = f"pk_consensus_topK(votes=2,k={args.consensus_topk})"
                    confidence = "mid"

            # C) fallback: best top1 across seeds
            if anchor is None:
                top1s = [si["top1"] for si in seed_infos if si["top1"]]
                if top1s:
                    top1s.sort(key=lambda c: agg_score(c), reverse=True)
                    anchor = top1s[0]
                    decision_stage = "C_fallback"
                    accept_reason = "fallback_top1_best_score"
                    confidence = "low"

        eligible = anchor is not None
        if eligible:
            eligible_qids.append(qid)
            qrels_lines.append(f"{qid}\t0\t{int(anchor['pk'])}\t{args.rel_p1}")

        # Prepare seed0/seed1 info for audit/review (your common case is 2 seeds)
        def seed_top_fields(idx: int) -> Tuple[str, str, str, str]:
            if idx < len(seed_infos) and seed_infos[idx]["top1"]:
                t = seed_infos[idx]["top1"]
                return (str(t["pk"]), f"{t['cosine']:.4f}", f"{t['char3_jaccard']:.4f}", f"{t['token_containment']:.4f}")
            return ("", "", "", "")

        s0_pk, s0_cos, s0_jac, s0_con = seed_top_fields(0)
        s1_pk, s1_cos, s1_jac, s1_con = seed_top_fields(1)

        # audit row
        if eligible:
            audit_rows.append("\t".join([
                qid, decision_stage, confidence, accept_reason,
                str(int(anchor["pk"])),
                f"{anchor['cosine']:.4f}", f"{anchor['char3_jaccard']:.4f}", f"{anchor['token_containment']:.4f}",
                clean_one_line(anchor.get("file_name", "")),
                clean_one_line(anchor.get("text_snippet", "")),
                s0_pk, s0_cos, s0_jac, s0_con,
                s1_pk, s1_cos, s1_jac, s1_con,
            ]))
        else:
            audit_rows.append("\t".join([qid, "NONE", "NONE", "no_anchor", "", "", "", "", "", "", s0_pk, s0_cos, s0_jac, s0_con, s1_pk, s1_cos, s1_jac, s1_con]))

        # review row (qid-level)
        if args.review_out:
            seed0_text = clean_one_line(seed_infos[0]["seed_text"]) if len(seed_infos) > 0 else ""
            seed1_text = clean_one_line(seed_infos[1]["seed_text"]) if len(seed_infos) > 1 else ""
            if len(seed0_text) > args.max_text_chars:
                seed0_text = seed0_text[:args.max_text_chars] + "..."
            if len(seed1_text) > args.max_text_chars:
                seed1_text = seed1_text[:args.max_text_chars] + "..."

            anchor_text = ""
            anchor_pk = ""
            cos_v = ""
            jac_v = ""
            con_v = ""
            fn_v = ""
            if eligible:
                anchor_pk = str(int(anchor["pk"]))
                cos_v = f"{anchor['cosine']:.4f}"
                jac_v = f"{anchor['char3_jaccard']:.4f}"
                con_v = f"{anchor['token_containment']:.4f}"
                fn_v = clean_one_line(anchor.get("file_name", ""))
                anchor_text = clean_one_line(anchor.get("gold_text", ""))
                if len(anchor_text) > args.max_text_chars:
                    anchor_text = anchor_text[:args.max_text_chars] + "..."

            review_rows.append("\t".join([
                qid, clean_one_line(question),
                seed0_text, seed1_text,
                decision_stage if eligible else "NONE",
                confidence if eligible else "NONE",
                accept_reason if eligible else "no_anchor",
                anchor_pk, cos_v, jac_v, con_v, fn_v,
                anchor_text
            ]))

        # report item
        report["items"].append({
            "qid": qid,
            "query": question,
            "decision": {
                "eligible": eligible,
                "decision_stage": decision_stage if eligible else "NONE",
                "accept_reason": accept_reason if eligible else "no_anchor",
                "confidence": confidence if eligible else "NONE",
                "anchor": anchor if eligible else None,
            },
            "seeds": seed_infos
        })

    # --------------------------
    # Write outputs
    # --------------------------
    os.makedirs(os.path.dirname(args.qrels_out), exist_ok=True)
    with open(args.qrels_out, "w", encoding="utf-8") as f:
        f.write("\n".join(qrels_lines) + ("\n" if qrels_lines else ""))

    os.makedirs(os.path.dirname(args.eligible_qids_out), exist_ok=True)
    with open(args.eligible_qids_out, "w", encoding="utf-8") as f:
        f.write("\n".join(eligible_qids) + ("\n" if eligible_qids else ""))

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(args.audit_tsv_out), exist_ok=True)
    with open(args.audit_tsv_out, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_rows) + "\n")

    if args.review_out:
        os.makedirs(os.path.dirname(args.review_out), exist_ok=True)
        with open(args.review_out, "w", encoding="utf-8") as f:
            f.write("\n".join(review_rows) + "\n")

    # --------------------------
    # Print summary
    # --------------------------
    total_q = len({it["qid"] for it in report["items"]})
    eligible_q = len(eligible_qids)

    # Stage counts
    stage_cnt = {"A_threshold": 0, "B_consensus": 0, "C_fallback": 0, "NONE": 0}
    for it in report["items"]:
        st = it["decision"]["decision_stage"]
        stage_cnt[st] = stage_cnt.get(st, 0) + 1

    print(json.dumps({
        "num_queries_total": total_q,
        "num_queries_eligible": eligible_q,
        "eligible_ratio": eligible_q / max(1, total_q),
        "qrels_lines": len(qrels_lines),
        "stage_counts": stage_cnt,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
