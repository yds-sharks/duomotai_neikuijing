import argparse
import base64
import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import io
import json
import os
import random
import re
import tempfile
import time
from typing import Any, Dict, Optional

from PIL import Image
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "baseline")
DEFAULT_RAG_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "rag")
DEFAULT_BENCHMARK = "Saint-lsy/EndoBench"
DEFAULT_PROMPT_TEMPLATE = "{question}\n{options_text}\n\n{prompt_suffix}"
DEFAULT_PROMPT_SUFFIX = (
    "请先给出简短分析过程，然后严格按照下面要求作答：\n"
    "1. 最后一行必须单独输出：最终答案：X\n"
    "2. X 必须为 {valid_choices} 之一。\n"
    "3. 如果某选项内容为 null 或 None，则该选项无效，不要选择。\n"
    "4. 除最后一行外，不要在别处重复写最终答案。\n"
    "5. 最后一行在“最终答案：X”后立刻停止，不要再输出任何文字、解释、标点或换行内容。\n"
    "6. 最后一行不要写成“正确答案是B”“B: 小肠”“最终答案：B 小肠”“最终答案：B：小肠”这类格式，只能写“最终答案：X”，不要跟具体的选项内容。\n"
    "示例：\n"
    "最终答案：B"
)
DEFAULT_ANSWER_REGEX = r"最终答案[:：]\s*([A-F])"
DEFAULT_RAG_INSTRUCTION = (
    "以下是从知识库中检索到的参考资料，仅供辅助判断。"
    "请结合图像、题目和选项综合作答，不要机械照搬参考资料。"
)
CHOICE_COLUMNS = tuple("ABCDEF")
FILTER_FIELDS = ("dataset", "task", "scene", "category", "subtask")
CSV_FIELDS = [
    "index",
    "category",
    "task",
    "subtask",
    "scene",
    "dataset",
    "answer",
    "pred",
    "status",
    "error_type",
    "error_message",
]


def write_metrics_file(
    *,
    output_dir: str,
    timestamp: str,
    run_index: int,
    backend_name: str,
    model_name: str,
    benchmark: str,
    split: str,
    correct: int,
    total: int,
) -> str:
    accuracy = correct / total if total else 0.0
    metrics_path = os.path.join(output_dir, f"{timestamp}_metrics_run{run_index}.json")
    payload = {
        "backend": backend_name,
        "model": model_name,
        "benchmark": benchmark,
        "split": split,
        "run": run_index,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }
    with open(metrics_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    return metrics_path


def write_log_file(
    *,
    output_dir: str,
    timestamp: str,
    run_index: int,
    backend_name: str,
    model_name: str,
    benchmark: str,
    split: str,
    mode: str,
    total: int,
    sample_logs,
    run_start_time: float,
    run_end_time: float,
) -> str:
    log_path = os.path.join(output_dir, f"{timestamp}_log_run{run_index}.log")
    elapsed_seconds = run_end_time - run_start_time
    lines = [
        f"backend={backend_name}",
        f"model={model_name}",
        f"benchmark={benchmark}",
        f"split={split}",
        f"mode={mode}",
        f"run={run_index}",
        f"started_at={datetime.fromtimestamp(run_start_time).isoformat()}",
        f"ended_at={datetime.fromtimestamp(run_end_time).isoformat()}",
        f"elapsed_seconds={elapsed_seconds:.6f}",
        f"completed_samples={len(sample_logs)}",
        f"total={total}",
        f"avg_sample_seconds={((elapsed_seconds / len(sample_logs)) if sample_logs else 0.0):.6f}",
        "",
        "[sample_logs]",
    ]
    for sample in sample_logs:
        lines.append(
            "index={index} status={status} elapsed_seconds={elapsed:.6f}".format(
                index=sample.get("index"),
                status=sample.get("status"),
                elapsed=float(sample.get("elapsed_seconds", 0.0)),
            )
        )
    with open(log_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")
    return log_path


@dataclass
class EvalContext:
    benchmark: str
    split: str
    total_samples: int
    image_column: str
    output_dir: str
    output_prefix: str
    timestamp: str


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: Optional[str] = None,
    default_model_path: Optional[str] = None,
    default_output_subdir: Optional[str] = None,
    default_dtype: str = "auto",
    default_attn_impl: Optional[str] = None,
    default_trust_remote_code: bool = False,
) -> None:
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark dataset id.")
    parser.add_argument("--split", default="test", help="Benchmark split.")
    parser.add_argument("--dataset", default="all", help="Dataset filter: 'all' or comma-separated values.")
    parser.add_argument("--task", default="all", help="Task filter: 'all' or comma-separated values.")
    parser.add_argument("--scene", default="all", help="Scene filter: 'all' or comma-separated values.")
    parser.add_argument("--category", default="all", help="Category filter: 'all' or comma-separated values.")
    parser.add_argument("--subtask", default="all", help="Subtask filter: 'all' or comma-separated values.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N filtered samples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the filtered dataset before slicing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model", default=default_model, help="Model name or served model id.")
    parser.add_argument("--model-path", default=default_model_path, help="Optional local model path.")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of full evaluation runs.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top_p.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generated tokens.")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value for local transformers backends.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map argument.")
    parser.add_argument(
        "--dtype",
        default=default_dtype,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Weight dtype for local transformers backends.",
    )
    parser.add_argument("--attn-impl", default=default_attn_impl, help="Optional attention implementation.")
    parser.add_argument("--api-key", default=None, help="API key for API or vLLM backends.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var used when --api-key is omitted.")
    parser.add_argument("--base-url", default=None, help="Base URL for API or vLLM backends.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Root directory for outputs.")
    parser.add_argument("--output-subdir", default=default_output_subdir, help="Optional subdirectory under --output-dir.")
    parser.add_argument("--output-csv", default="predictions.csv", help="CSV filename prefix.")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Prompt template. Supported placeholders: {question}, {options_text}, {prompt_suffix}, {valid_choices}.",
    )
    parser.add_argument("--prompt-template-file", default=None, help="Path to a prompt template file.")
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX, help="Prompt suffix appended to the question.")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt.")
    parser.add_argument("--system-prompt-file", default=None, help="Path to a file containing the system prompt.")
    parser.add_argument("--answer-regex", default=DEFAULT_ANSWER_REGEX, help="Regex used to parse the final option.")
    parser.add_argument("--extra-model-kwargs", default=None, help="JSON object merged into local model loading kwargs.")
    parser.add_argument(
        "--extra-generation-kwargs",
        default=None,
        help="JSON object merged into generation or request kwargs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=default_trust_remote_code,
        help="Enable trust_remote_code for local transformers backends.",
    )
    parser.add_argument("--save-response", action="store_true", help="Save raw responses to a separate JSONL file.")
    parser.add_argument("--response-jsonl", default="responses.jsonl", help="Response JSONL filename prefix.")
    parser.add_argument("--fsync-every-row", action="store_true", help="Call fsync after every row write.")


def build_parser(
    description: str,
    *,
    default_model: Optional[str] = None,
    default_model_path: Optional[str] = None,
    default_output_subdir: Optional[str] = None,
    default_dtype: str = "auto",
    default_attn_impl: Optional[str] = None,
    default_trust_remote_code: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_common_args(
        parser,
        default_model=default_model,
        default_model_path=default_model_path,
        default_output_subdir=default_output_subdir,
        default_dtype=default_dtype,
        default_attn_impl=default_attn_impl,
        default_trust_remote_code=default_trust_remote_code,
    )
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.prompt_template = resolve_text_arg(args.prompt_template, args.prompt_template_file, DEFAULT_PROMPT_TEMPLATE)
    args.system_prompt = resolve_text_arg(args.system_prompt, args.system_prompt_file, None)
    args.extra_model_kwargs = parse_json_dict(args.extra_model_kwargs, "--extra-model-kwargs")
    args.extra_generation_kwargs = parse_json_dict(args.extra_generation_kwargs, "--extra-generation-kwargs")
    return args


def configure_runtime(args: argparse.Namespace) -> None:
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass


def load_filtered_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    dataset = load_dataset(args.benchmark)[args.split]
    for field in FILTER_FIELDS:
        values = parse_csv_values(getattr(args, field))
        if values is None:
            continue
        dataset = dataset.filter(lambda row, field=field, values=values: str(row.get(field, "")) in values)

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)

    start = max(args.offset, 0)
    stop = len(dataset) if args.limit is None else min(len(dataset), start + max(args.limit, 0))
    return dataset.select(range(start, stop))


def detect_image_column(dataset) -> str:
    for column in dataset.column_names:
        if "image" in column.lower():
            return column
    raise ValueError("No image column found in the benchmark split.")


def create_eval_context(args: argparse.Namespace, backend_name: str, dataset) -> EvalContext:
    model_tag = sanitize_name(args.model or backend_name)
    output_subdir = args.output_subdir or os.path.join(backend_name, model_tag)
    output_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    return EvalContext(
        benchmark=args.benchmark,
        split=args.split,
        total_samples=len(dataset),
        image_column=detect_image_column(dataset),
        output_dir=output_dir,
        output_prefix=os.path.splitext(os.path.basename(args.output_csv))[0],
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )


def build_prompt(example: Dict[str, Any], args: argparse.Namespace):
    option_columns = [column for column in CHOICE_COLUMNS if column in example]
    valid_columns = [column for column in option_columns if not is_null_option(example.get(column))]
    display_columns = valid_columns or option_columns
    options_text = "\n".join(f"{column}: {example.get(column)}" for column in display_columns)
    valid_choices = "/".join(valid_columns or option_columns or CHOICE_COLUMNS)
    prompt_suffix = (args.prompt_suffix or DEFAULT_PROMPT_SUFFIX).format(valid_choices=valid_choices)
    prompt = args.prompt_template.format(
        question=example.get("question", ""),
        options_text=options_text,
        prompt_suffix=prompt_suffix,
        valid_choices=valid_choices,
    )
    return prompt.strip(), (valid_columns or option_columns)


def build_retrieval_query(example: Dict[str, Any], query_mode: str) -> str:
    question = str(example.get("question", "")).strip()
    valid_columns = [
        column for column in CHOICE_COLUMNS
        if column in example and not is_null_option(example.get(column))
    ]
    options_text = "\n".join(f"{column}: {example.get(column)}" for column in valid_columns)

    if query_mode == "question_options" and options_text:
        return f"{question}\n{options_text}"
    return question


def rewrite_retrieval_query_via_api(
    client,
    model: str,
    query: str,
    target_lang: str = "en",
) -> str:
    """
    Rewrite a single retrieval query using OpenAI API.

    Args:
        client: OpenAI client
        model: Model name to use for rewriting
        query: Original retrieval query
        target_lang: Target language for translation

    Returns:
        Rewritten query
    """
    prompt = f"""请帮我改写以下用于医学问答检索的 query。

要求：
1. 去掉选项前的字母标识（如 A. B. C. D. 等），但是要保留选项内容
2. 将 query 翻译成 {target_lang}
3. 可以适当使用更易于检索的表述改写

原始 query：
{query}

请直接输出改写后的 query，不要包含任何解释或其他内容。"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    result = response.choices[0].message.content
    if result:
        return result.strip()
    raise RuntimeError("Query rewrite API returned empty result")


def extract_doc_text(doc: Dict[str, Any]) -> str:
    for key in ("text", "content", "chunk", "page_content", "passage", "snippet", "summary"):
        value = doc.get(key)
        if value:
            return str(value)
    return ""


def format_retrieved_docs(docs, *, max_chars: int) -> str:
    if not docs:
        return "未检索到可用参考资料。"

    blocks = []
    for doc in docs:
        text = extract_doc_text(doc).strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        score = doc.get("score", 0)
        try:
            score_text = f"{float(score):.4f}"
        except (TypeError, ValueError):
            score_text = str(score)
        blocks.append(
            "\n".join(
                [
                    f"[{doc.get('rank', '?')}] {doc.get('doc_name', 'unknown')} "
                    f"(p.{doc.get('page_idx', '?')}, score={score_text})",
                    text or "[empty chunk]",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_rag_prompt(example: Dict[str, Any], args: argparse.Namespace, docs):
    base_prompt, valid_choices = build_prompt(example, args)
    retrieval_block = format_retrieved_docs(docs, max_chars=args.doc_max_chars)
    rag_prompt = (
        f"{args.rag_instruction}\n\n"
        f"参考资料：\n{retrieval_block}\n\n"
        f"{base_prompt}"
    )
    return rag_prompt, valid_choices


def parse_choice(text: Any, valid_choices, answer_regex: str) -> Optional[str]:
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    if text is None:
        return None

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return None

    match = re.search(answer_regex, lines[-1])
    if not match:
        return None

    prediction = match.group(1)
    return prediction if prediction in set(valid_choices) else None


def run_evaluation(args: argparse.Namespace, *, backend_name: str, adapter_factory) -> None:
    args = normalize_args(args)
    configure_runtime(args)

    dataset = load_filtered_dataset(args)
    context = create_eval_context(args, backend_name, dataset)
    adapter = adapter_factory(args)

    accuracies = []
    try:
        for run_idx in range(args.num_runs):
            run_number = run_idx + 1
            print(f"[{backend_name}] model={args.model or backend_name} run={run_number}/{args.num_runs}")
            run_start_time = time.time()
            log_path = os.path.join(context.output_dir, f"{context.timestamp}_log_run{run_number}.json")
            output_csv = os.path.join(
                context.output_dir,
                f"{context.timestamp}_{context.output_prefix}_run{run_number}.csv",
            )
            response_path = None
            if args.save_response:
                response_prefix = os.path.splitext(os.path.basename(args.response_jsonl))[0]
                response_path = os.path.join(
                    context.output_dir,
                    f"{context.timestamp}_{response_prefix}_run{run_number}.jsonl",
                )

            correct = 0
            sample_logs = []
            with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                response_file = open(response_path, "w", encoding="utf-8") if response_path else None

                try:
                    for idx, example in tqdm(enumerate(dataset), total=context.total_samples):
                        sample_start_time = time.time()
                        prompt, valid_choices = build_prompt(example, args)
                        answer = str(example.get("answer") or "").strip()
                        response_text = None
                        prediction = None
                        status = "ok"
                        error_type = ""
                        error_message = ""

                        try:
                            response_text = adapter.generate(example, prompt, context)
                            prediction = parse_choice(response_text, valid_choices, args.answer_regex)
                        except Exception as exc:
                            error_type = type(exc).__name__
                            error_message = str(exc).strip()
                            if "content_policy_violation" in error_message:
                                status = "skipped_content_policy"
                            else:
                                status = "skipped_error"
                            print(
                                f"[{backend_name}] skipped sample index={example.get('index', idx)} "
                                f"status={status} error={error_type}: {error_message}"
                            )

                        if status == "ok" and prediction == answer:
                            correct += 1

                        row = {
                            "index": example.get("index", idx),
                            "category": example.get("category", ""),
                            "task": example.get("task", ""),
                            "subtask": example.get("subtask", ""),
                            "scene": example.get("scene", ""),
                            "dataset": example.get("dataset", ""),
                            "answer": answer,
                            "pred": prediction,
                            "status": status,
                            "error_type": error_type,
                            "error_message": error_message,
                        }
                        writer.writerow(row)
                        csv_file.flush()
                        if args.fsync_every_row:
                            os.fsync(csv_file.fileno())

                        if response_file is not None:
                            response_record = dict(row)
                            response_record["response"] = response_text
                            response_file.write(json.dumps(response_record, ensure_ascii=False) + "\n")
                            response_file.flush()
                            if args.fsync_every_row:
                                os.fsync(response_file.fileno())

                        sample_logs.append(
                            {
                                "index": example.get("index", idx),
                                "status": status,
                                "elapsed_seconds": time.time() - sample_start_time,
                            }
                        )
                        write_log_file(
                            output_dir=context.output_dir,
                            timestamp=context.timestamp,
                            run_index=run_number,
                            backend_name=backend_name,
                            model_name=args.model or backend_name,
                            benchmark=args.benchmark,
                            split=args.split,
                            mode="baseline",
                            total=context.total_samples,
                            sample_logs=sample_logs,
                            run_start_time=run_start_time,
                            run_end_time=time.time(),
                        )
                finally:
                    if response_file is not None:
                        response_file.close()

            run_end_time = time.time()
            accuracy = correct / context.total_samples if context.total_samples else 0.0
            accuracies.append(accuracy)
            write_metrics_file(
                output_dir=context.output_dir,
                timestamp=context.timestamp,
                run_index=run_number,
                backend_name=backend_name,
                model_name=args.model or backend_name,
                benchmark=args.benchmark,
                split=args.split,
                correct=correct,
                total=context.total_samples,
            )
            write_log_file(
                output_dir=context.output_dir,
                timestamp=context.timestamp,
                run_index=run_number,
                backend_name=backend_name,
                model_name=args.model or backend_name,
                benchmark=args.benchmark,
                split=args.split,
                mode="baseline",
                total=context.total_samples,
                sample_logs=sample_logs,
                run_start_time=run_start_time,
                run_end_time=run_end_time,
            )
            print(
                f"[{backend_name}] run {run_number}/{args.num_runs} accuracy: "
                f"{accuracy:.4f} ({correct}/{context.total_samples})"
            )

        if accuracies:
            mean_accuracy = sum(accuracies) / len(accuracies)
            print(f"[{backend_name}] mean accuracy over {len(accuracies)} run(s): {mean_accuracy:.4f}")
    finally:
        close_fn = getattr(adapter, "close", None)
        if callable(close_fn):
            close_fn()


def run_rag_evaluation(args: argparse.Namespace, *, backend_name: str, adapter_factory) -> None:
    args = normalize_args(args)
    configure_runtime(args)

    dataset = load_filtered_dataset(args)
    context = create_eval_context(args, backend_name, dataset)
    adapter = adapter_factory(args)

    from search_vector_store import batch_search

    accuracies = []
    try:
        for run_idx in range(args.num_runs):
            run_number = run_idx + 1
            print(f"[{backend_name}] model={args.model or backend_name} run={run_number}/{args.num_runs}")
            run_start_time = time.time()
            log_path = os.path.join(context.output_dir, f"{context.timestamp}_log_run{run_number}.json")
            output_csv = os.path.join(
                context.output_dir,
                f"{context.timestamp}_{context.output_prefix}_run{run_number}.csv",
            )
            response_path = None
            if args.save_response:
                response_prefix = os.path.splitext(os.path.basename(args.response_jsonl))[0]
                response_path = os.path.join(
                    context.output_dir,
                    f"{context.timestamp}_{response_prefix}_run{run_number}.jsonl",
                )

            correct = 0
            sample_logs = []
            with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                response_file = open(response_path, "w", encoding="utf-8") if response_path else None

                try:
                    retrieval_batch_size = max(int(getattr(args, "retrieval_batch_size", 64) or 64), 1)
                    # Get query rewrite settings
                    query_rewrite = getattr(args, "retrieval_query_rewrite", False)
                    query_rewrite_model = getattr(args, "retrieval_query_rewrite_model", "gpt-4o-mini")
                    query_translate_lang = getattr(args, "retrieval_query_translate_lang", "en")

                    # Create API client for query rewriting if enabled
                    # Always use external API (from env) for query rewriting, not the backend's vLLM URL
                    rewrite_client = None
                    if query_rewrite:
                        external_base_url = os.environ.get("BASE_URL")
                        external_api_key = os.environ.get("DEER_API_KEY") or os.environ.get("OPENAI_API_KEY")
                        rewrite_client = create_openai_client(
                            base_url=external_base_url,
                            api_key=external_api_key,
                        )
                        print(f"[{backend_name}] Using API model '{query_rewrite_model}' for batch query rewriting (URL: {external_base_url})")

                    progress = tqdm(total=context.total_samples)
                    try:
                        for batch_start in range(0, context.total_samples, retrieval_batch_size):
                            batch_end = min(context.total_samples, batch_start + retrieval_batch_size)
                            batch_examples = [dataset[idx] for idx in range(batch_start, batch_end)]
                            batch_queries = [
                                build_retrieval_query(example, args.retrieval_query_mode)
                                for example in batch_examples
                            ]
                            # Rewrite queries using API (one call per query)
                            if query_rewrite and rewrite_client:
                                batch_rewritten_queries = [
                                    rewrite_retrieval_query_via_api(
                                        rewrite_client,
                                        query_rewrite_model,
                                        query,
                                        target_lang=query_translate_lang,
                                    )
                                    for query in batch_queries
                                ]
                            else:
                                # No rewrite, use original query as-is
                                batch_rewritten_queries = batch_queries
                            unique_rewritten_queries = list(dict.fromkeys(batch_rewritten_queries))
                            retrieval_map = (
                                batch_search(unique_rewritten_queries, topk=args.topk) if args.topk > 0 else {}
                            )

                            for idx, (example, query, rewritten_query) in enumerate(
                                zip(batch_examples, batch_queries, batch_rewritten_queries), start=batch_start
                            ):
                                sample_start_time = time.time()
                                # Use rewritten query for retrieval, but keep original for response log
                                docs = retrieval_map.get(rewritten_query, [])
                                prompt, valid_choices = build_rag_prompt(example, args, docs)
                                answer = str(example.get("answer") or "").strip()
                                response_text = None
                                prediction = None
                                status = "ok"
                                error_type = ""
                                error_message = ""

                                try:
                                    response_text = adapter.generate(example, prompt, context)
                                    prediction = parse_choice(response_text, valid_choices, args.answer_regex)
                                except Exception as exc:
                                    error_type = type(exc).__name__
                                    error_message = str(exc).strip()
                                    if "content_policy_violation" in error_message:
                                        status = "skipped_content_policy"
                                    else:
                                        status = "skipped_error"
                                    print(
                                        f"[{backend_name}] skipped sample index={example.get('index', idx)} "
                                        f"status={status} error={error_type}: {error_message}"
                                    )

                                if status == "ok" and prediction == answer:
                                    correct += 1

                                row = {
                                    "index": example.get("index", idx),
                                    "category": example.get("category", ""),
                                    "task": example.get("task", ""),
                                    "subtask": example.get("subtask", ""),
                                    "scene": example.get("scene", ""),
                                    "dataset": example.get("dataset", ""),
                                    "answer": answer,
                                    "pred": prediction,
                                    "status": status,
                                    "error_type": error_type,
                                    "error_message": error_message,
                                }
                                writer.writerow(row)
                                csv_file.flush()
                                if args.fsync_every_row:
                                    os.fsync(csv_file.fileno())

                                if response_file is not None:
                                    response_record = dict(row)
                                    response_record["response"] = response_text
                                    response_record["retrieval_query"] = query
                                    response_record["retrieval_query_rewritten"] = rewritten_query
                                    response_record["retrieval_docs"] = docs
                                    response_file.write(json.dumps(response_record, ensure_ascii=False) + "\n")
                                    response_file.flush()
                                    if args.fsync_every_row:
                                        os.fsync(response_file.fileno())

                                sample_logs.append(
                                    {
                                        "index": example.get("index", idx),
                                        "status": status,
                                        "elapsed_seconds": time.time() - sample_start_time,
                                    }
                                )
                                write_log_file(
                                    output_dir=context.output_dir,
                                    timestamp=context.timestamp,
                                    run_index=run_number,
                                    backend_name=backend_name,
                                    model_name=args.model or backend_name,
                                    benchmark=args.benchmark,
                                    split=args.split,
                                    mode="rag",
                                    total=context.total_samples,
                                    sample_logs=sample_logs,
                                    run_start_time=run_start_time,
                                    run_end_time=time.time(),
                                )
                                progress.update(1)

                            try:
                                import search_vector_store
                                from pymilvus import connections

                                connections.disconnect("default")
                                search_vector_store._global_engine = None
                            except Exception:
                                pass
                    finally:
                        progress.close()
                finally:
                    if response_file is not None:
                        response_file.close()

            run_end_time = time.time()
            accuracy = correct / context.total_samples if context.total_samples else 0.0
            accuracies.append(accuracy)
            write_metrics_file(
                output_dir=context.output_dir,
                timestamp=context.timestamp,
                run_index=run_number,
                backend_name=backend_name,
                model_name=args.model or backend_name,
                benchmark=args.benchmark,
                split=args.split,
                correct=correct,
                total=context.total_samples,
            )
            write_log_file(
                output_dir=context.output_dir,
                timestamp=context.timestamp,
                run_index=run_number,
                backend_name=backend_name,
                model_name=args.model or backend_name,
                benchmark=args.benchmark,
                split=args.split,
                mode="rag",
                total=context.total_samples,
                sample_logs=sample_logs,
                run_start_time=run_start_time,
                run_end_time=run_end_time,
            )
            print(
                f"[{backend_name}] run {run_number}/{args.num_runs} accuracy: "
                f"{accuracy:.4f} ({correct}/{context.total_samples})"
            )

        if accuracies:
            mean_accuracy = sum(accuracies) / len(accuracies)
            print(f"[{backend_name}] mean accuracy over {len(accuracies)} run(s): {mean_accuracy:.4f}")
    finally:
        close_fn = getattr(adapter, "close", None)
        if callable(close_fn):
            close_fn()


def resolve_model_source(args: argparse.Namespace) -> str:
    return args.model_path or args.model


def resolve_torch_dtype(dtype_name: str, *, fallback=None):
    if dtype_name == "auto":
        return fallback if fallback is not None else "auto"

    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def infer_main_device(model, torch_module):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for device_name in model.hf_device_map.values():
            if isinstance(device_name, str) and device_name.startswith("cuda"):
                return torch_module.device(device_name)
        if "cpu" in model.hf_device_map.values():
            return torch_module.device("cpu")

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device("cpu")


def merge_kwargs(base_kwargs: Dict[str, Any], extra_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base_kwargs)
    if extra_kwargs:
        merged.update(extra_kwargs)
    return merged


def build_generation_kwargs(args: argparse.Namespace, *, include_top_p: bool = True) -> Dict[str, Any]:
    generation_kwargs = {"max_new_tokens": args.max_tokens}
    if args.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
        if include_top_p:
            generation_kwargs["top_p"] = args.top_p
    return merge_kwargs(generation_kwargs, args.extra_generation_kwargs)


def build_openai_request_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    request_kwargs = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    return merge_kwargs(request_kwargs, args.extra_generation_kwargs)


def build_openai_messages(prompt: str, image_data_url: str, system_prompt: Optional[str] = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    )
    return messages


def create_openai_client(base_url: Optional[str], api_key: Optional[str]):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def get_api_key(args: argparse.Namespace, *, default: Optional[str] = None, required: bool = False) -> str:
    api_key = args.api_key or os.environ.get(args.api_key_env) or default
    if required and not api_key:
        raise ValueError(f"Missing API key. Provide --api-key or set {args.api_key_env}.")
    return api_key


def maybe_prepend_system_prompt(system_prompt: Optional[str], prompt: str) -> str:
    if not system_prompt:
        return prompt
    return f"{system_prompt.strip()}\n\n{prompt}"


def get_image_pil(example: Dict[str, Any], context: EvalContext) -> Image.Image:
    raw_b64 = get_raw_base64(example, context.image_column)
    return Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")


def get_image_data_url(example: Dict[str, Any], context: EvalContext) -> str:
    image = get_image_pil(example, context)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"


@contextmanager
def temporary_image_path(example: Dict[str, Any], context: EvalContext):
    raw_b64 = get_raw_base64(example, context.image_column)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(base64.b64decode(raw_b64))
        temp_path = temp_file.name

    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def parse_csv_values(raw_value: Optional[str]):
    if raw_value is None or str(raw_value).lower() == "all":
        return None
    values = {item.strip() for item in str(raw_value).split(",") if item.strip()}
    return values or None


def parse_json_dict(raw_value: Optional[str], flag_name: str) -> Dict[str, Any]:
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object.")
    return parsed


def resolve_text_arg(inline_value: Optional[str], file_path: Optional[str], default: Optional[str]) -> Optional[str]:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as file_obj:
            return file_obj.read()
    if inline_value is not None:
        return inline_value
    return default


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "model"


def is_null_option(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return True
    return False


def get_raw_base64(example: Dict[str, Any], image_column: str) -> str:
    raw_value = example[image_column]
    if isinstance(raw_value, str) and "base64," in raw_value:
        return raw_value.split("base64,", 1)[1]
    return raw_value
