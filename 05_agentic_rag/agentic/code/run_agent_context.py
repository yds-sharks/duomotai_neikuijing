#!/usr/bin/env python3
"""Stage-2 trajectory generation driver.

For each benchmark question, run the multi-round keep/drop + ACCEPT/REWRITE
control loop with the GPT teacher and record the full per-round trajectory.
Output is one JSON trajectory per line (resumable by qid).

Retrieval (Milvus Lite single-connection + GPU encoders) is serialized behind a
lock; GPT API calls run concurrently across worker threads.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from context_agent import GPTContextAgent
from retrieval_adapter import FirstStageRetriever, load_config
from trajectory_runtime import run_trajectory

DEFAULT_INPUT = "/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/outputs/mcq_image_v2_4000/train.jsonl"
DEFAULT_OUTPUT = "/mnt/data_1/yds/多模态/rerank_image_and_text/agentic/outputs/stage2_calibration/agent_context_v11_train3200.jsonl"
DEFAULT_API_CONFIG = "/mnt/data_1/yds/多模态/agentic/data_construction/api_config.local.json"

LOW_QUERY_BY_TYPE = {
    "image_organ_identification": "这张图是什么部位？",
    "anatomical_site_recognition": "这张图是什么部位？",
    "image_content_type_identification": "这张图主要展示什么？",
    "lesion_or_finding_identification": "图中是什么异常？",
    "procedure_or_operation_recognition": "图中在做什么操作？",
    "spatial_region_understanding": "图中异常在哪里？",
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def get_qid(row: Dict[str, Any], idx: int) -> str:
    return str(row.get("qid") or row.get("candidate_id") or row.get("sample_id") or f"row_{idx:06d}")


def get_options(row: Dict[str, Any]) -> Dict[str, Any]:
    opts = row.get("options")
    return opts if isinstance(opts, dict) else {k: row[k] for k in "ABCDEF" if k in row}


def get_query_image(row: Dict[str, Any]) -> str:
    src = row.get("source") if isinstance(row.get("source"), dict) else {}
    return str(row.get("query_image_path") or row.get("image_path") or src.get("image_path") or "")


def get_original_query(row: Dict[str, Any]) -> str:
    for key in ("low_information_query_seed", "original_query", "retrieval_query"):
        if row.get(key):
            return str(row[key]).strip()
    return LOW_QUERY_BY_TYPE.get(str(row.get("query_type") or ""), str(row.get("question") or "").strip())


def get_gold_source(row: Dict[str, Any]) -> Dict[str, Any]:
    src = row.get("source") if isinstance(row.get("source"), dict) else {}
    return {
        "doc_id": str(src.get("doc_id") or row.get("doc_id") or ""),
        "page_idx": src.get("page_idx", row.get("page_idx")),
        "sample_id": str(src.get("sample_id") or row.get("sample_id") or ""),
    }


def select_rows(rows: Iterable[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("question") or not get_options(row) or not get_query_image(row):
            continue
        selected.append(row)
        if limit and len(selected) >= limit:
            break
    return selected


def load_done_qids(path: Path) -> set:
    done: set = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("qid")
            if qid:
                done.add(str(qid))
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "agentic_runtime_config.json"))
    p.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    p.add_argument("--output-jsonl", default=DEFAULT_OUTPUT)
    p.add_argument("--api-config", default=DEFAULT_API_CONFIG)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--select-k", type=int, default=12)
    p.add_argument("--text-k", type=int, default=20)
    p.add_argument("--image-k", type=int, default=20)
    p.add_argument("--max-evidence-images", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = select_rows(iter_jsonl(Path(args.input_jsonl)), limit=args.limit)
    done = load_done_qids(out_path)
    rows = [(i, r) for i, r in enumerate(all_rows) if get_qid(r, i) not in done]
    if not rows:
        print(f"[done] nothing to do (all {len(all_rows)} already in {out_path})")
        return
    print(f"[start] total={len(all_rows)} done={len(done)} todo={len(rows)} workers={args.workers} max_rounds={args.max_rounds}")

    retriever_lock = threading.Lock()
    write_lock = threading.Lock()
    local = threading.local()

    def get_agent() -> GPTContextAgent:
        agent = getattr(local, "agent", None)
        if agent is None:
            agent = GPTContextAgent(api_config_path=args.api_config, max_evidence_images=args.max_evidence_images)
            local.agent = agent
        return agent

    counters = {"n": 0, "ok": 0, "err": 0, "rewrite": 0, "t0": time.time()}
    out_handle = out_path.open("a", encoding="utf-8")

    def work(idx_row):
        idx, row = idx_row
        qid = get_qid(row, idx)
        traj = run_trajectory(
            qid=qid,
            query_type=str(row.get("query_type") or ""),
            original_query=get_original_query(row),
            question=str(row.get("question") or ""),
            options=get_options(row),
            answer=str(row.get("answer") or ""),
            answer_text=str(row.get("answer_text") or ""),
            image_path=get_query_image(row),
            gold_source=get_gold_source(row),
            retriever=retriever,
            agent=get_agent(),
            max_rounds=args.max_rounds,
            select_k=args.select_k,
            text_k=args.text_k,
            image_k=args.image_k,
            retriever_lock=retriever_lock,
        )
        return traj

    with FirstStageRetriever(config) as retriever:
        with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
            futures = [pool.submit(work, ir) for ir in rows]
            for fut in as_completed(futures):
                try:
                    traj = fut.result()
                except Exception as exc:
                    counters["err"] += 1
                    print(f"[error] {type(exc).__name__}: {exc}")
                    continue
                with write_lock:
                    out_handle.write(json.dumps(traj, ensure_ascii=False) + "\n")
                    out_handle.flush()
                counters["n"] += 1
                if str(traj.get("status")) == "ok":
                    counters["ok"] += 1
                else:
                    counters["err"] += 1
                if str(traj.get("final_action")) == "REWRITE":
                    counters["rewrite"] += 1
                if counters["n"] % max(int(args.log_every), 1) == 0:
                    dt = time.time() - counters["t0"]
                    rate = counters["n"] / dt if dt > 0 else 0.0
                    eta = (len(rows) - counters["n"]) / rate if rate > 0 else 0.0
                    print(f"[{counters['n']}/{len(rows)}] ok={counters['ok']} err={counters['err']} "
                          f"rewrite={counters['rewrite']} rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)

    out_handle.close()
    print(f"[done] wrote {counters['n']} trajectories -> {out_path}")


if __name__ == "__main__":
    main()
