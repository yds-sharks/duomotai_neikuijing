"""
Unified evaluation entry for EndoBench.

This file supports two usage patterns:

1. Direct backend mode
   - Use `--backend` to run a single backend implementation directly.
   - Example backends: `qwen3-vl`, `deepseek-vl`, `hulumed`,
     `huatuogpt-vision`, `gpt`.
   - In this mode, the script imports the target backend module and runs
     evaluation in the current process.

2. Profile mode
   - Use `--profiles` to run one or more predefined model profiles.
   - A profile bundles together:
       - backend name
       - model repo id
       - deployment style (`vllm`, `transformers`, `api`)
       - Python environment
       - per-model overrides such as gpu ids, dtype, base_url, etc.
   - In this mode, the script may:
       - start a vLLM service automatically
       - wait until the service becomes ready
       - launch a child evaluation process
       - stop the vLLM service after evaluation finishes

Two evaluation modes are supported:

- `baseline`
  Run the benchmark without retrieval augmentation.

- `rag`
  Retrieve reference passages with `search_vector_store.batch_search(...)`
  and prepend the retrieved content to the prompt before inference.

Typical usage
-------------

Run all configured profiles in baseline mode:

    cd /mnt/data_10/mwx/workspace/endo_benchmark/project_ver4
    python evaluate_unified.py \
      --mode baseline \
      --profiles Qwen3-VL-2B-Instruct,Qwen3-VL-8B-Instruct,deepseek-vl2-tiny,gpt-4o-mini,Hulu-Med-7B,HuatuoGPT-Vision-7B \
      --benchmark Saint-lsy/EndoBench \
      --split test \
      --dataset all \
      --task all \
      --scene all \
      --category all \
      --subtask all \
      --limit 500 \
      --num-runs 1 \
      --temperature 0.2 \
      --top-p 0.9 \
      --max-tokens 512 \
      --save-response \
      --gpu 0,1,2,3 \
      --gpu-memory-utilization 0.9

Run all configured profiles in RAG mode:

    python evaluate_unified.py \
      --mode rag \
      --profiles Qwen3-VL-2B-Instruct,Qwen3-VL-8B-Instruct,deepseek-vl2-tiny,gpt-4o-mini,Hulu-Med-7B,HuatuoGPT-Vision-7B \
      --benchmark Saint-lsy/EndoBench \
      --split test \
      --dataset all \
      --task all \
      --scene all \
      --category all \
      --subtask all \
      --limit 500 \
      --num-runs 1 \
      --temperature 0.2 \
      --top-p 0.9 \
      --max-tokens 512 \
      --retrieval-query-mode question_options \
      --topk 5 \
      --retrieval-batch-size 64 \
      --save-response \
      --gpu 0,1,2,3 \
      --gpu-memory-utilization 0.9

Run a single backend directly:

    python evaluate_unified.py \
      --backend gpt \
      --mode baseline \
      --model gpt-4o-mini \
      --limit 20 \
      --num-runs 1 \
      --temperature 0.2 \
      --max-tokens 512
"""



import argparse
import importlib.util
import os
import subprocess
import time
from urllib import error, request

from dotenv import load_dotenv

from utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAG_INSTRUCTION,
    DEFAULT_RAG_OUTPUT_DIR,
    PROJECT_DIR,
    add_common_args,
    run_rag_evaluation,
)


load_dotenv()


BACKEND_CONFIGS = {
    "qwen3-vl": {
        "file": "evaluate_qwen3_vl.py",
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "deployment": "vllm",
        "base_url": "http://127.0.0.1:8888/v1",
        "api_key": "EMPTY",
    },
    "deepseek-vl": {
        "file": "evaluate_deepseek_vl.py",
        "model": "deepseek-ai/deepseek-vl2-tiny",
        "deployment": "vllm",
        "base_url": "http://127.0.0.1:8888/v1",
        "api_key": "EMPTY",
    },
    "hulumed": {
        "file": "evaluate_hulumed.py",
        "model": "ZJU-AI4H/Hulu-Med-7B",
        "dtype": "bfloat16",
        "deployment": "transformers",
        "trust_remote_code": True,
    },
    "huatuogpt-vision": {
        "file": "evaluate_huatuogpt_vision.py",
        "model": "FreedomIntelligence/HuatuoGPT-Vision-7B",
        "deployment": "transformers",
    },
    "gpt": {
        "file": "evaluate_gpt.py",
        "model": "gpt-4o-mini",
        "deployment": "api",
        "api_key_env": "DEER_API_KEY",
    },
}

CONDA_ENVS_DIR = "/mnt/data_1/mwx/anaconda3/envs"
DEFAULT_VLLM_PORT = 8888

# =========== 若添加模型, 则修改此处 =========== 
PROFILE_CONFIGS = {
    "Qwen3-VL-2B-Instruct": {
        "python_env": "endo",
        "deployment": "vllm",
        "backend": "qwen3-vl",
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "eval_overrides": {
            "gpu": "0,1,2,3",
            "base_url": f"http://127.0.0.1:{DEFAULT_VLLM_PORT}/v1",
            "api_key": "EMPTY",
        },
        "serve": {
            "gpu": "0,1,2,3",
            "port": DEFAULT_VLLM_PORT,
            "max_model_len": 4096,
            "tensor_parallel_size": 4,
            "allowed_tensor_parallel_sizes": [1, 2, 4],
            "extra_args": ["--trust-remote-code"],
        },
    },
    "Qwen3-VL-8B-Instruct": {
        "python_env": "endo",
        "deployment": "vllm",
        "backend": "qwen3-vl",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "eval_overrides": {
            "gpu": "0,1,2,3",
            "base_url": f"http://127.0.0.1:{DEFAULT_VLLM_PORT}/v1",
            "api_key": "EMPTY",
        },
        "serve": {
            "gpu": "0,1,2,3",
            "port": DEFAULT_VLLM_PORT,
            "max_model_len": 8192,
            "tensor_parallel_size": 4,
            "allowed_tensor_parallel_sizes": [1, 2, 4],
            "extra_args": ["--trust-remote-code"],
        },
    },
    "deepseek-vl2-tiny": {
        "python_env": "endo",
        "deployment": "vllm",
        "backend": "deepseek-vl",
        "model": "deepseek-ai/deepseek-vl2-tiny",
        "eval_overrides": {
            "gpu": "0,1",
            "base_url": f"http://127.0.0.1:{DEFAULT_VLLM_PORT}/v1",
            "api_key": "EMPTY",
        },
        "serve": {
            "gpu": "0,1",
            "port": DEFAULT_VLLM_PORT,
            "max_model_len": 4096,
            "tensor_parallel_size": 2,
            "allowed_tensor_parallel_sizes": [1, 2],
            "extra_args": ['--hf-overrides', '{"architectures":["DeepseekVLV2ForCausalLM"]}'],
        },
    },
    "gpt-4o-mini": {
        "python_env": "endo",
        "deployment": "api",
        "backend": "gpt",
        "model": "gpt-4o-mini",
        "eval_overrides": {
            "api_key_env": "DEER_API_KEY",
        },
    },
        "gpt-4o": {
        "python_env": "endo",
        "deployment": "api",
        "backend": "gpt",
        "model": "gpt-4o",
        "eval_overrides": {
            "api_key_env": "DEER_API_KEY",
        },
    },
    "Hulu-Med-7B": {
        "python_env": "endo",
        "deployment": "transformers",
        "backend": "hulumed",
        "model": "ZJU-AI4H/Hulu-Med-7B",
        "eval_overrides": {
            "gpu": "0,1,2,3",
            "device_map": "auto",
            "dtype": "bfloat16",
        },
    },
    "HuatuoGPT-Vision-7B": {
        "python_env": "huatuo",
        "deployment": "transformers",
        "backend": "huatuogpt-vision",
        "model": "FreedomIntelligence/HuatuoGPT-Vision-7B",
        "eval_overrides": {
            "gpu": "0",  # cli 的设置只能单卡
        },
    },
}


def load_backend_module(backend_name: str):
    config = BACKEND_CONFIGS[backend_name]
    file_path = os.path.join(PROJECT_DIR, config["file"])
    module_name = f"project_ver3_{backend_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backend module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_backend_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = BACKEND_CONFIGS[args.backend]
    if not args.model and config.get("model"):
        args.model = config["model"]
    if not args.model_path and config.get("model_path"):
        args.model_path = config["model_path"]
    if args.dtype == "auto" and config.get("dtype"):
        args.dtype = config["dtype"]
    if not args.attn_impl and config.get("attn_impl"):
        args.attn_impl = config["attn_impl"]
    if not args.base_url and config.get("base_url"):
        args.base_url = config["base_url"]
    if not args.api_key and config.get("api_key"):
        args.api_key = config["api_key"]
    if getattr(args, "api_key_env", None) == "OPENAI_API_KEY" and config.get("api_key_env"):
        args.api_key_env = config["api_key_env"]
    if args.backend == "gpt" and not args.base_url:
        args.base_url = os.environ.get("BASE_URL")
    if config.get("trust_remote_code"):
        args.trust_remote_code = True
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified EndoBench evaluation entry for vLLM, transformers and API backends."
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--backend",
        choices=sorted(BACKEND_CONFIGS.keys()),
        help="Backend to evaluate directly.",
    )
    target_group.add_argument(
        "--profiles",
        default=None,
        help="Comma-separated predefined profiles to run sequentially.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print available profile names and exit.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu-memory-utilization used when profiles auto-start a vLLM server.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rag"],
        default="baseline",
        help="Evaluation mode: baseline or rag.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of retrieved chunks injected into the prompt when --mode rag.",
    )
    parser.add_argument(
        "--retrieval-batch-size",
        type=int,
        default=64,
        help="Number of questions retrieved together in each RAG retrieval batch.",
    )
    parser.add_argument(
        "--retrieval-query-mode",
        choices=["question", "question_options"],
        default="question",
        help="How to build the retrieval query in RAG mode.",
    )
    parser.add_argument(
        "--rag-instruction",
        default=DEFAULT_RAG_INSTRUCTION,
        help="Optional instruction prepended before retrieved documents in RAG mode.",
    )
    parser.add_argument(
        "--doc-max-chars",
        type=int,
        default=1200,
        help="Maximum characters kept from each retrieved chunk in RAG mode.",
    )
    parser.add_argument(
        "--retrieval-query-rewrite",
        action="store_true",
        help="Whether to rewrite the retrieval query using API before searching (removes option letters and optionally translates).",
    )
    parser.add_argument(
        "--retrieval-query-rewrite-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for retrieval query rewriting.",
    )
    parser.add_argument(
        "--retrieval-query-translate-lang",
        type=str,
        default="en",
        help="Target language for retrieval query translation (e.g., en, zh, ja, ko). Used when --retrieval-query-rewrite is enabled.",
    )
    add_common_args(parser)
    args = parser.parse_args()
    if args.list_profiles:
        print("\n".join(sorted(PROFILE_CONFIGS.keys())))
        raise SystemExit(0)
    if not args.backend and not args.profiles:
        parser.error("one of --backend or --profiles is required")
    if args.mode == "rag" and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_RAG_OUTPUT_DIR
    if args.backend:
        return apply_backend_defaults(args)
    return args


def get_python_executable(env_name: str) -> str:
    candidate = os.path.join(CONDA_ENVS_DIR, env_name, "bin", "python")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Python executable not found for env '{env_name}': {candidate}")


def split_profiles(raw_profiles: str):
    profiles = [item.strip() for item in str(raw_profiles).split(",") if item.strip()]
    if not profiles:
        raise ValueError("No valid profiles were provided.")
    invalid = [profile for profile in profiles if profile not in PROFILE_CONFIGS]
    if invalid:
        raise ValueError(f"Unknown profile(s): {', '.join(invalid)}")
    return profiles


def parse_gpu_ids(raw_gpu_value):
    if not raw_gpu_value:
        return []
    return [item.strip() for item in str(raw_gpu_value).split(",") if item.strip()]


def choose_tensor_parallel_size(args, profile_config: dict) -> int:
    serve_config = profile_config["serve"]
    requested_gpu_ids = parse_gpu_ids(args.gpu)
    if not requested_gpu_ids:
        return serve_config["tensor_parallel_size"]

    allowed_sizes = serve_config.get("allowed_tensor_parallel_sizes")
    if not allowed_sizes:
        allowed_sizes = [serve_config["tensor_parallel_size"]]

    valid_sizes = [size for size in sorted(allowed_sizes) if size <= len(requested_gpu_ids)]
    if not valid_sizes:
        raise ValueError(
            f"Profile '{profile_config['model']}' requires one of TP sizes {allowed_sizes}, "
            f"but only got {len(requested_gpu_ids)} GPU ids from --gpu={args.gpu}."
        )
    return valid_sizes[-1]


def select_profile_gpu_ids(args, profile_config: dict, *, for_vllm: bool, tensor_parallel_size: int | None = None) -> str:
    requested_gpu_ids = parse_gpu_ids(args.gpu)
    if not requested_gpu_ids:
        key = "serve" if for_vllm else "eval_overrides"
        return profile_config.get(key, {}).get("gpu")

    if for_vllm:
        required_count = tensor_parallel_size or profile_config["serve"]["tensor_parallel_size"]
        if len(requested_gpu_ids) < required_count:
            raise ValueError(
                f"Profile '{profile_config['model']}' requires at least {required_count} GPU ids, "
                f"but only got {len(requested_gpu_ids)} from --gpu={args.gpu}."
            )
        return ",".join(requested_gpu_ids[:required_count])

    return ",".join(requested_gpu_ids)


def add_arg(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_common_eval_cli_args(args):
    cmd = []
    add_arg(cmd, "--mode", args.mode)
    add_arg(cmd, "--topk", args.topk)
    add_arg(cmd, "--retrieval-batch-size", args.retrieval_batch_size)
    add_arg(cmd, "--retrieval-query-mode", args.retrieval_query_mode)
    add_arg(cmd, "--rag-instruction", args.rag_instruction)
    add_arg(cmd, "--doc-max-chars", args.doc_max_chars)
    if args.retrieval_query_rewrite:
        cmd.append("--retrieval-query-rewrite")
    add_arg(cmd, "--retrieval-query-rewrite-model", args.retrieval_query_rewrite_model)
    add_arg(cmd, "--retrieval-query-translate-lang", args.retrieval_query_translate_lang)
    add_arg(cmd, "--benchmark", args.benchmark)
    add_arg(cmd, "--split", args.split)
    add_arg(cmd, "--dataset", args.dataset)
    add_arg(cmd, "--task", args.task)
    add_arg(cmd, "--scene", args.scene)
    add_arg(cmd, "--category", args.category)
    add_arg(cmd, "--subtask", args.subtask)
    add_arg(cmd, "--limit", args.limit)
    add_arg(cmd, "--offset", args.offset)
    if args.shuffle:
        cmd.append("--shuffle")
    add_arg(cmd, "--seed", args.seed)
    add_arg(cmd, "--num-runs", args.num_runs)
    add_arg(cmd, "--temperature", args.temperature)
    add_arg(cmd, "--top-p", args.top_p)
    add_arg(cmd, "--max-tokens", args.max_tokens)
    add_arg(cmd, "--output-dir", args.output_dir)
    add_arg(cmd, "--output-subdir", args.output_subdir)
    add_arg(cmd, "--output-csv", args.output_csv)
    add_arg(cmd, "--prompt-template", args.prompt_template)
    add_arg(cmd, "--prompt-template-file", args.prompt_template_file)
    add_arg(cmd, "--prompt-suffix", args.prompt_suffix)
    add_arg(cmd, "--system-prompt", args.system_prompt)
    add_arg(cmd, "--system-prompt-file", args.system_prompt_file)
    add_arg(cmd, "--answer-regex", args.answer_regex)
    add_arg(cmd, "--extra-model-kwargs", args.extra_model_kwargs)
    add_arg(cmd, "--extra-generation-kwargs", args.extra_generation_kwargs)
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.save_response:
        cmd.append("--save-response")
    add_arg(cmd, "--response-jsonl", args.response_jsonl)
    if args.fsync_every_row:
        cmd.append("--fsync-every-row")
    return cmd


def wait_for_http_ready(base_url: str, timeout_seconds: int = 360) -> None:
    models_url = f"{base_url.rstrip('/')}/models"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with request.urlopen(models_url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (error.URLError, TimeoutError, OSError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for service readiness: {models_url}")


def start_vllm_service(profile_name: str, profile_config: dict):
    return _start_vllm_service(profile_name, profile_config, gpu_memory_utilization=0.9)


def _start_vllm_service(
    profile_name: str,
    profile_config: dict,
    *,
    gpu_memory_utilization: float,
    gpu_ids: str,
    tensor_parallel_size: int,
    model_override=None,
):
    serve_config = profile_config["serve"]
    python_exec = get_python_executable(profile_config["python_env"])
    port = serve_config["port"]
    cmd = [
        python_exec,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_override or profile_config["model"],
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max-model-len",
        str(serve_config["max_model_len"]),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    cmd.extend(serve_config.get("extra_args", []))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"[profile:{profile_name}] starting vLLM service")
    process = subprocess.Popen(cmd, env=env, cwd=PROJECT_DIR)
    return process, f"http://127.0.0.1:{port}/v1"


def run_profile_eval(args, profile_name: str, profile_config: dict, *, model_override=None, base_url_override=None):
    python_exec = get_python_executable(profile_config["python_env"])
    script_path = os.path.abspath(__file__)
    eval_overrides = dict(profile_config.get("eval_overrides", {}))

    cmd = [
        python_exec,
        script_path,
        "--backend",
        profile_config["backend"],
        "--model",
        model_override or profile_config["model"],
    ]
    cmd.extend(build_common_eval_cli_args(args))

    add_arg(cmd, "--gpu", select_profile_gpu_ids(args, profile_config, for_vllm=False))
    add_arg(cmd, "--device-map", eval_overrides.get("device_map"))
    add_arg(cmd, "--dtype", eval_overrides.get("dtype"))
    add_arg(cmd, "--attn-impl", eval_overrides.get("attn_impl"))
    add_arg(cmd, "--base-url", base_url_override or args.base_url or eval_overrides.get("base_url"))
    add_arg(cmd, "--api-key", args.api_key or eval_overrides.get("api_key"))
    add_arg(cmd, "--api-key-env", eval_overrides.get("api_key_env"))
    add_arg(cmd, "--model-path", args.model_path)

    print(f"[profile:{profile_name}] running evaluation")
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def run_profiles(args):
    profiles = split_profiles(args.profiles)
    single_profile_model_override = args.model if len(profiles) == 1 and args.model else None

    for profile_name in profiles:
        profile_config = PROFILE_CONFIGS[profile_name]
        deployment = profile_config["deployment"]
        print(f"[profile:{profile_name}] deployment={deployment}")

        if deployment == "vllm":
            tensor_parallel_size = choose_tensor_parallel_size(args, profile_config)
            vllm_gpu_ids = select_profile_gpu_ids(
                args,
                profile_config,
                for_vllm=True,
                tensor_parallel_size=tensor_parallel_size,
            )
            process, local_base_url = _start_vllm_service(
                profile_name,
                profile_config,
                gpu_memory_utilization=args.gpu_memory_utilization,
                gpu_ids=vllm_gpu_ids,
                tensor_parallel_size=tensor_parallel_size,
                model_override=single_profile_model_override,
            )
            try:
                wait_for_http_ready(local_base_url)
                run_profile_eval(
                    args,
                    profile_name,
                    profile_config,
                    model_override=single_profile_model_override,
                    base_url_override=local_base_url,
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        else:
            run_profile_eval(
                args,
                profile_name,
                profile_config,
                model_override=single_profile_model_override,
            )


def main():
    args = parse_args()
    if args.profiles:
        run_profiles(args)
        return
    module = load_backend_module(args.backend)
    if args.mode == "rag":
        run_rag_evaluation(args, backend_name=args.backend, adapter_factory=module.ADAPTER_CLASS)
    else:
        module.run(args)


if __name__ == "__main__":
    main()
