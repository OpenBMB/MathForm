import argparse
import os
import sys

import config
from config import DEFAULT_DATASET_PATHS, DEFAULT_MAX_WORKERS
from inference import run_inference
from judge import run_judge
from prompts import DEFAULT_INFER_PROMPT_TEMPLATE, INFERENCE_PROMPT_TEMPLATES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoformalizer inference/evaluation via an OpenAI-compatible API backend")
    parser.add_argument("--mode", choices=["infer", "judge"], default="judge", help="run mode")
    parser.add_argument("--output_path", type=str, required=True, help="output path (jsonl)")
    parser.add_argument("--dataset_path", type=str, nargs="+", default=DEFAULT_DATASET_PATHS, help="benchmark dataset paths (one or more)")
    parser.add_argument("--predictions_path", type=str, default=None, help="predictions jsonl (optional)")
    parser.add_argument("--num_samples", type=int, default=8, help="number of candidates per problem (k)")
    parser.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    parser.add_argument("--eval_max_tokens", type=int, default=16384, help="max generated tokens for inference")
    parser.add_argument("--judge_max_tokens", type=int, default=51200, help="max generated tokens for judge")
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_INFER_PROMPT_TEMPLATE,
        choices=sorted(INFERENCE_PROMPT_TEMPLATES.keys()),
        help="prompt template for inference",
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="skip think/code-block parsing and use the full answer as the code",
    )

    parser.add_argument(
        "--api_base_url",
        type=str,
        required=True,
        help="remote API base URL (OpenAI-compatible), e.g. https://api.example.com/v1",
    )
    parser.add_argument(
        "--api_model",
        type=str,
        required=True,
        help="model name on the remote API",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="API_KEY",
        help="environment variable holding the API key (never passed via command line)",
    )
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="number of concurrent requests")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(f"[error] API key not found in environment variable {args.api_key_env}")
        sys.exit(1)
    config.API_BASE_URL = args.api_base_url
    config.API_KEY = api_key
    config.API_MODEL = args.api_model
    print(f"API backend: {config.API_BASE_URL} (model={config.API_MODEL}), max_workers: {args.max_workers}")
    print(f"Datasets: {args.dataset_path}")

    if args.mode == "infer":
        run_inference(
            output_path=args.output_path,
            dataset_paths=args.dataset_path,
            max_workers=args.max_workers,
            num_samples=args.num_samples,
            timeout=args.timeout,
            eval_max_tokens=args.eval_max_tokens,
            prompt_template=args.prompt_template,
            limit=args.limit,
            no_think=args.no_think,
        )
    else:
        run_judge(
            output_path=args.output_path,
            dataset_paths=args.dataset_path,
            max_workers=args.max_workers,
            num_samples=args.num_samples,
            timeout=args.timeout,
            predictions_path=args.predictions_path,
            limit=args.limit,
            judge_max_tokens=args.judge_max_tokens,
        )
