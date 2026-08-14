#!/usr/bin/env python3
"""
Data construction pipeline entry point.

Two run modes:
1. Sample mode (default): each sample runs the full pipeline independently
2. Batch mode (--batch-mode): generate all -> compile all -> semantic-check all
   -> retry failed samples; recommended for large datasets

Examples:
    python main.py --input data.jsonl --output-dir ./output
    python main.py --input data.jsonl --output-dir ./output --batch-mode
"""

import argparse

from config import Config
from pipeline import SamplePipeline
from batch_pipeline import BatchPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data construction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    
    parser.add_argument(
        "--base-model-url",
        type=str,
        default="http://localhost:8001/v1/chat/completions",
        help="Base model API URL",
    )
    parser.add_argument(
        "--retrieval-planner-url",
        type=str,
        default=None,
        help="Retrieval planner API URL (defaults to base model)",
    )
    parser.add_argument(
        "--judge-model-url",
        type=str,
        default="http://localhost:8002/v1/chat/completions",
        help="Judge model API URL",
    )
    parser.add_argument(
        "--lean-server-url",
        type=str,
        default="http://localhost:8000",
        help="Lean server URL",
    )
    parser.add_argument(
        "--lean-explore-url",
        type=str,
        default=None,
        help="Lean Explore API URL (optional)",
    )
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--request-timeout", type=int, default=300)
    
    parser.add_argument("--base-model-name", type=str, default=None,
                        help="Base model name for code generation")
    parser.add_argument("--retrieval-planner-name", type=str, default=None,
                        help="Retrieval planner name for search query generation")
    parser.add_argument("--judge-model-name", type=str, default=None,
                        help="Judge model name for semantic consistency check")
    
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum iterations per sample")
    parser.add_argument("--retrieval-limit", type=int, default=15,
                        help="Maximum retrieval history length")
    parser.add_argument("--query-top-k", type=int, default=3,
                        help="Top-k results per query")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Maximum concurrent workers (default for all models)")
    parser.add_argument("--retrieval-workers", type=int, default=32,
                        help="Maximum concurrent retrieval workers")
    parser.add_argument("--base-model-workers", type=int, default=None,
                        help="Maximum concurrent workers for base model (defaults to max-workers)")
    parser.add_argument("--retrieval-planner-workers", type=int, default=None,
                        help="Maximum concurrent workers for retrieval planner (defaults to max-workers)")
    parser.add_argument("--judge-model-workers", type=int, default=None,
                        help="Maximum concurrent workers for judge model (defaults to max-workers)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--pipeline-batch-size", type=int, default=200,
                        help="Batch size for pipeline processing in batch mode")
    
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        dest="use_batch_mode",
        help="Use batch processing mode (recommended for large datasets like 120k samples). "
             "Process flow: generate all → compile all → semantic all → retry failed"
    )
    parser.add_argument(
        "--sample-mode",
        action="store_false",
        dest="use_batch_mode",
        help="Use sample-level pipeline mode (default). "
             "Each sample completes the full pipeline independently"
    )
    parser.set_defaults(use_batch_mode=False)

    parser.add_argument("--generation-batch-size", type=int, default=1000,
                        help="Batch size for generation phase (batch mode only)")
    parser.add_argument("--compile-batch-size", type=int, default=500,
                        help="Batch size for compilation phase (batch mode only)")
    parser.add_argument("--semantic-batch-size", type=int, default=500,
                        help="Batch size for semantic check phase (batch mode only)")
    
    parser.add_argument("--use-async", action="store_true",
                        help="[Deprecated] Use --batch-mode instead")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.use_async and not args.use_batch_mode:
        print("Warning: --use-async is deprecated, using --batch-mode instead")
        args.use_batch_mode = True
    
    config = Config.from_args(args)
    
    if config.pipeline.use_batch_mode:
        print(f"Running in BATCH mode (recommended for large datasets)")
        print(f"  - max_workers: {config.pipeline.max_workers}")
        print(f"  - max_iterations: {config.pipeline.max_iterations}")
        pipeline = BatchPipeline(config)
    else:
        print(f"Running in SAMPLE mode (default)")
        print(f"  - max_workers: {config.pipeline.max_workers}")
        print(f"  - max_iterations: {config.pipeline.max_iterations}")
        pipeline = SamplePipeline(config)
    
    pipeline.run()


if __name__ == "__main__":
    main()
