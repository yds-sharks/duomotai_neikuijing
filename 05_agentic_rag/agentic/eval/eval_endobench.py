#!/usr/bin/env python3
"""EndoBench evaluation for the AgenticRL controller stack (paper main table).

One script -> all rows of the main comparison:

  --mode baseline     : frozen generator, no retrieval, option-argmax
  --mode vanilla_rag  : frozen RAG pipeline (first-stage retrieval img6+txt6, no controller)
  --mode agentic      : + trained controller (greedy) keep/rewrite; REWRITE re-runs
                        real text retrieval (top-5) exactly as in training
  --mode gpt4o        : + proprietary controller (GPTContextAgent) with the IDENTICAL
                        v11 system prompt and action space as the trained controller

Scoring = the frozen generator's option-argmax (AnswerScorer.judge), i.e. the exact
same deterministic primitive as the training reward, so eval and reward are consistent.

Outputs (per --out-dir):
  <mode>_samples.jsonl : one line per question (resumable: existing qids are skipped)
  <mode>_summary.json  : overall / per-scene / per-task accuracy + controller behavior

Initial-candidate recipe mirrors rebuild_obs_candidates.py:
  search_image(qimg, k=20) -> select 6 ; search_text(question+options, k=20) -> select 6
Candidates are cached (--cand-cache) so repeated modes/runs skip retrieval.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
AGENTIC_ROOT = HERE.parent
sys.path.insert(0, str(AGENTIC_ROOT / "code"))
sys.path.insert(0, str(AGENTIC_ROOT / "train"))

from context_agent import SYSTEM_PROMPT_V11, USER_TEMPLATE, format_candidates  # noqa: E402
from ctrl_data_common import build_messages, render_chat  # noqa: E402
from grpo_reward import parse_action  # noqa: E402

DEFAULT_GEN_MODEL = "/mnt/data_10/mwx/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]
NULL_STRINGS = {"", "none", "null", "n/a"}

# EndoBench scene -> Milvus body_site_main filter (organ module)
SCENE_TO_ORGAN = {
    "Gastroscopy": "胃",
    "Colonoscopy": "结直肠",
    "Capsule Endoscopy": "小肠",
}


def weight_adjusted_k(dep_level: str, base_text_k: int, base_image_k: int) -> tuple:
    """Adjust text/image select_k based on weight module's image dependency level."""
    if dep_level == "R3":  # high image dependency -> more images
        return max(base_text_k - 2, 2), base_image_k + 2
    elif dep_level == "R1":  # low image dependency -> more text
        return base_text_k + 2, max(base_image_k - 2, 2)
    return base_text_k, base_image_k


def is_null_option(v: Any) -> bool:
    return str(v or "").strip().lower() in NULL_STRINGS


def build_options(ex: Dict[str, Any]) -> Dict[str, str]:
    return {L: str(ex[L]).strip() for L in OPTION_LETTERS
            if ex.get(L) is not None and not is_null_option(ex.get(L))}


def build_text_query(ex: Dict[str, Any], options: Dict[str, str]) -> str:
    """question+options retrieval query (matches the 'question_options' convention)."""
    q = str(ex.get("question") or "").strip()
    opt = "\n".join(f"{L}: {t}" for L, t in options.items())
    return f"{q}\n{opt}" if opt else q


def dump_query_image(ex: Dict[str, Any], img_dir: Path) -> str:
    """EndoBench stores the image as base64 in the `image` column; materialize to disk."""
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


class EndoBenchEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.need_retrieval = args.mode != "baseline"
        self.retriever = None
        self.ctrl_model = None
        self.ctrl_proc = None
        self.gpt_agent = None
        self.scorer = None

        if self.need_retrieval:
            from retrieval_adapter import FirstStageRetriever, load_config as load_retr_config
            cfg = load_retr_config()
            cfg["retrieval"]["text_device"] = args.retr_text_device
            cfg["retrieval"]["image_device"] = args.retr_image_device
            self.retriever = FirstStageRetriever(cfg)
            self.retriever.search_text("warmup", k=1)
            print(f"[retr] online: text={args.retr_text_device} image={args.retr_image_device}", flush=True)

        self.weight_service = None
        if args.use_weight_module:
            try:
                _wpath = "/mnt/data_1/yds/多模态/权重模块"
                if _wpath not in sys.path:
                    sys.path.insert(0, _wpath)
                from multimodal_weight_service import MultimodalWeightService
                self.weight_service = MultimodalWeightService()
                _ = self.weight_service.analyze_text("测试")  # warmup
                print(f"[weight] MultimodalWeightService loaded", flush=True)
            except Exception as e:
                print(f"[weight] FAILED to load weight module: {e}", flush=True)
                self.weight_service = None

        if args.mode == "agentic":
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.ctrl_proc = AutoProcessor.from_pretrained(args.ctrl_model, trust_remote_code=True)
            self.ctrl_model = AutoModelForImageTextToText.from_pretrained(
                args.ctrl_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                trust_remote_code=True).to(args.ctrl_device)
            self.ctrl_model.eval()
            self.ctrl_model.config.use_cache = True
            print(f"[ctrl] loaded {Path(args.ctrl_model).name} on {args.ctrl_device}", flush=True)
        elif args.mode == "gpt4o":
            from context_agent import GPTContextAgent
            kw = {}
            if args.api_config:
                kw["api_config_path"] = args.api_config
            self.gpt_agent = GPTContextAgent(**kw)
            print(f"[ctrl] GPTContextAgent model={self.gpt_agent.model}", flush=True)

        from gen_scorer import AnswerScorer
        self.scorer = AnswerScorer(args.gen_model, device=args.gen_device,
                                   query_edge=768, ev_edge=384, max_images=8)
        print(f"[gen] scorer loaded on {args.gen_device}", flush=True)

    # ---------------- retrieval ----------------
    def initial_candidates(self, qimg: str, text_query: str, *,
                           level1: str = "", text_query_zh: str = "") -> List[Dict[str, Any]]:
        from evidence_selection import select_top_evidence
        from trajectory_runtime import normalize_candidate
        a = self.args
        # Use translated Chinese query for text retrieval if available
        search_query = text_query_zh if text_query_zh else text_query
        # Weight module: adaptively adjust text/image select_k
        text_sel_k = a.text_select_k
        image_sel_k = a.image_select_k
        if self.weight_service is not None and search_query:
            try:
                w = self.weight_service.analyze_text(search_query[:512])
                dep = w.get("image_dependency_level", "R2")
                text_sel_k, image_sel_k = weight_adjusted_k(dep, a.text_select_k, a.image_select_k)
            except Exception:
                pass
        img_hits = self.retriever.search_image(qimg, k=a.image_k, level1=level1) if qimg else []
        txt_hits = self.retriever.search_text(search_query, k=a.text_k, level1=level1) if search_query else []
        cands = (select_top_evidence(img_hits, select_k=image_sel_k)
                 + select_top_evidence(txt_hits, select_k=text_sel_k))
        return [normalize_candidate(h) for h in cands]

    def rewrite_retrieve(self, rewrite_query: str, *, level1: str = "") -> List[Dict[str, Any]]:
        hits = self.retriever.search_text(rewrite_query, k=self.args.rewrite_topk, level1=level1)
        return [hit_to_ev(h) for h in hits[: self.args.rewrite_topk]]

    # ---------------- controllers ----------------
    @torch.no_grad()
    def decide_local(self, ex, qid, options, text_query, qimg, cands) -> Dict[str, Any]:
        user_text = USER_TEMPLATE.format(
            qid=qid, query_type=f"{ex.get('scene','')}/{ex.get('task','')}",
            original_query=text_query, current_query=text_query,
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

    def decide_gpt(self, ex, qid, options, text_query, qimg, cands) -> Dict[str, Any]:
        d = self.gpt_agent.decide(
            qid=qid, query_type=f"{ex.get('scene','')}/{ex.get('task','')}",
            original_query=text_query, current_query=text_query,
            question=str(ex.get("question") or ""), options=options,
            candidates=cands, image_path=qimg)
        keep = [i for i in d.get("keep", []) if 0 <= int(i) < len(cands)]
        action = str(d.get("action") or "ACCEPT").upper()
        if action not in ("ACCEPT", "REWRITE"):
            action = "ACCEPT"
        return {"keep": sorted(keep), "action": action,
                "rewrite_query": str(d.get("rewrite_query") or "").strip(),
                "reason": str(d.get("reason") or ""), "parsed": True}

    # ---------------- per-sample flow ----------------
    def run_sample(self, ex, qimg, cands, *, level1: str = "") -> Dict[str, Any]:
        a = self.args
        idx = ex.get("index")
        qid = f"eb_{idx}"
        options = build_options(ex)
        gold = str(ex.get("answer") or "").strip()
        text_query = build_text_query(ex, options)

        action_info: Dict[str, Any] = {"action": "NONE", "keep": [], "rewrite_query": "",
                                       "parsed": True}
        if a.mode == "baseline":
            evidence: List[Dict[str, Any]] = []
        elif a.mode == "vanilla_rag":
            evidence = [hit_to_ev(c) for c in cands]
        else:
            if a.mode == "agentic":
                action_info = self.decide_local(ex, qid, options, text_query, qimg, cands)
            else:
                action_info = self.decide_gpt(ex, qid, options, text_query, qimg, cands)
            if action_info["action"] == "REWRITE" and action_info.get("rewrite_query"):
                evidence = self.rewrite_retrieve(action_info["rewrite_query"], level1=level1)
            elif action_info["action"] == "ACCEPT" and not action_info.get("parsed", True):
                # unparsable controller output -> conservative fallback: all candidates
                evidence = [hit_to_ev(c) for c in cands]
            else:
                evidence = [hit_to_ev(cands[i]) for i in action_info.get("keep", [])
                            if 0 <= i < len(cands)]

        verdict = self.scorer.judge(
            str(ex.get("question") or ""), options, gold, qimg, evidence)
        return {
            "index": idx, "qid": qid, "scene": ex.get("scene", ""), "task": ex.get("task", ""),
            "category": ex.get("category", ""), "dataset": ex.get("dataset", ""),
            "gold": gold, "pred": verdict["pred"], "correct": verdict["correct"],
            "logp_gold": verdict["logp_gold"], "p_gold": verdict["p_gold"],
            "probs": verdict["probs"], "mode": a.mode,
            "action": action_info["action"], "keep": action_info.get("keep", []),
            "rewrite_query": action_info.get("rewrite_query", ""),
            "parse_ok": action_info.get("parsed", True),
            "n_evidence": len(evidence),
            "evidence_docs": [str(e.get("doc_name", "")) for e in evidence],
        }

    def close(self):
        if self.retriever is not None:
            try:
                self.retriever.close()
            except Exception:
                pass


def load_done_qids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line).get("qid"))
                    except Exception:
                        pass
    return done


def summarize(records: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    def acc(rows):
        n = len(rows)
        return {"n": n, "acc": sum(r["correct"] for r in rows) / n if n else 0.0,
                "mean_logp_gold": sum(r["logp_gold"] for r in rows) / n if n else 0.0}

    out: Dict[str, Any] = {"mode": mode, "overall": acc(records)}
    for key in ("scene", "task", "category"):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            groups.setdefault(str(r.get(key) or "?"), []).append(r)
        out[f"by_{key}"] = {k: acc(v) for k, v in sorted(groups.items())}
    if mode in ("agentic", "gpt4o"):
        rw = [r for r in records if r["action"] == "REWRITE"]
        ac = [r for r in records if r["action"] == "ACCEPT"]
        out["behavior"] = {
            "rewrite_rate": len(rw) / len(records) if records else 0.0,
            "accept_acc": acc(ac)["acc"], "rewrite_acc": acc(rw)["acc"],
            "mean_keep_accept": (sum(len(r["keep"]) for r in ac) / len(ac)) if ac else 0.0,
            "parse_fail_rate": sum(1 for r in records if not r["parse_ok"]) / len(records) if records else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["baseline", "vanilla_rag", "agentic", "gpt4o"])
    ap.add_argument("--benchmark", default="Saint-lsy/EndoBench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--img-dir", default=str(HERE / "endobench_images"))
    ap.add_argument("--ctrl-model", default="", help="controller ckpt dir (agentic mode)")
    ap.add_argument("--ctrl-device", default="cuda:0")
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--gen-device", default="cuda:1")
    ap.add_argument("--retr-text-device", default="cuda:1")
    ap.add_argument("--retr-image-device", default="cuda:0")
    ap.add_argument("--text-k", type=int, default=20)
    ap.add_argument("--image-k", type=int, default=20)
    ap.add_argument("--text-select-k", type=int, default=6)
    ap.add_argument("--image-select-k", type=int, default=6)
    ap.add_argument("--rewrite-topk", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--cand-cache", default=str(HERE / "results" / "cand_cache.jsonl"))
    ap.add_argument("--api-config", default="")
    ap.add_argument("--translate-cache", default=str(HERE / "endobench_translated_queries.jsonl"),
                    help="Pre-translated EndoBench queries (Chinese) for retrieval")
    ap.add_argument("--use-organ-filter", action="store_true", default=False,
                    help="Enable organ-based Milvus filtering by scene")
    ap.add_argument("--use-weight-module", action="store_true", default=False,
                    help="Enable adaptive channel fusion via weight module")
    ap.add_argument("--weight-device", default="cuda:0",
                    help="Device for weight module model")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "agentic" and not args.ctrl_model:
        ap.error("--ctrl-model is required for agentic mode")

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir); img_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"{args.mode}_samples.jsonl"
    summary_path = out_dir / f"{args.mode}_summary.json"

    from datasets import load_dataset
    ds = load_dataset(args.benchmark)[args.split]
    n_total = len(ds)
    start = max(args.offset, 0)
    stop = n_total if args.limit <= 0 else min(n_total, start + args.limit)
    indices = list(range(start, stop))
    print(f"[data] {args.benchmark}[{args.split}] n={n_total} -> eval [{start},{stop}) = {len(indices)}", flush=True)

    # candidate cache (shared across modes): qid -> candidates
    cand_cache: Dict[str, List[Dict[str, Any]]] = {}
    cache_path = Path(args.cand_cache)
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        cand_cache[d["qid"]] = d["candidates"]
                    except Exception:
                        pass
    cache_f = open(cache_path, "a", encoding="utf-8") if args.mode != "baseline" else None

    done = load_done_qids(samples_path)
    if done:
        print(f"[resume] {len(done)} samples already done, skipping", flush=True)
    out_f = open(samples_path, "a", encoding="utf-8")

    # Load translated queries (English -> Chinese for retrieval)
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
    elif args.mode != "baseline":
        print(f"[trans] WARNING: no translation cache at {tc_path}, using English queries", flush=True)

    ev = EndoBenchEvaluator(args)
    t0 = time.time()
    n_new = 0
    records: List[Dict[str, Any]] = []
    # reload previous records for a complete summary
    if samples_path.exists():
        pass  # records rebuilt at the end from the file
    try:
        for i in indices:
            ex = ds[i]
            qid = f"eb_{ex.get('index')}"
            if qid in done:
                continue
            qimg = dump_query_image(ex, img_dir)
            level1 = SCENE_TO_ORGAN.get(str(ex.get("scene", "")), "") if args.use_organ_filter else ""
            if args.mode == "baseline":
                cands: List[Dict[str, Any]] = []
            elif qid in cand_cache:
                cands = cand_cache[qid]
            else:
                options = build_options(ex)
                text_query_zh = translate_map.get(ex.get("index"), "")
                cands = ev.initial_candidates(qimg, build_text_query(ex, options),
                                              level1=level1, text_query_zh=text_query_zh)
                if cache_f is not None:
                    cache_f.write(json.dumps({"qid": qid, "candidates": cands},
                                             ensure_ascii=False) + "\n")
                    cache_f.flush()
            try:
                rec = ev.run_sample(ex, qimg, cands, level1=level1)
            except Exception as exc:
                rec = {"index": ex.get("index"), "qid": qid, "scene": ex.get("scene", ""),
                       "task": ex.get("task", ""), "category": ex.get("category", ""),
                       "dataset": ex.get("dataset", ""), "gold": str(ex.get("answer") or "").strip(),
                       "pred": "", "correct": False, "logp_gold": 0.0, "p_gold": 0.0,
                       "probs": {}, "mode": args.mode, "action": "ERROR", "keep": [],
                       "rewrite_query": "", "parse_ok": False, "n_evidence": 0,
                       "evidence_docs": [], "error": f"{type(exc).__name__}: {exc}"}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n_new += 1
            if n_new % args.log_every == 0:
                dt = time.time() - t0
                print(f"[{n_new}/{len(indices) - len(done)}] {dt / n_new:.2f}s/sample "
                      f"elapsed={dt / 60:.1f}min", flush=True)
    finally:
        out_f.close()
        if cache_f is not None:
            cache_f.close()
        ev.close()

    # rebuild full record list for the summary (includes previously-done samples)
    all_records: List[Dict[str, Any]] = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass
    summary = summarize(all_records, args.mode)
    summary["n_samples_file"] = len(all_records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    ov = summary["overall"]
    print(f"[done] mode={args.mode} acc={ov['acc']:.4f} ({ov['n']} samples) "
          f"mean_logp_gold={ov['mean_logp_gold']:.3f}", flush=True)
    print(f"[save] {samples_path} / {summary_path}", flush=True)


if __name__ == "__main__":
    main()
