"""Concurrent benchmark for lean_explore local search."""

import argparse
import itertools
import logging
import pathlib
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

# Ensure the src directory is in the Python path
TRUE_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRUE_PROJECT_ROOT / "src"))

try:
    from lean_explore.local.search import (
        fetch_embeddings_batch,
        perform_search,
        perform_search_batch,
    )
    from lean_explore.local.service import Service
except ImportError as e:
    print(
        f"Error: Could not import project modules: {e}\n"
        "Ensure 'lean_explore' is installed (e.g., 'pip install -e .') "
        "and that 'src' is discoverable.",
        file=sys.stderr,
    )
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


DEFAULT_QUERIES_FILENAME = "queries.txt"
DEFAULT_TOTAL_REQUESTS = 100
DEFAULT_CONCURRENCY = 4


THREAD_LOCAL = threading.local()
THREAD_SERVICE: Optional[Service] = None
PROCESS_SERVICE: Optional[Service] = None
THREAD_SERVICE_LOCK = threading.Lock()


def _init_process_worker() -> None:
    global PROCESS_SERVICE
    PROCESS_SERVICE = Service()


def _get_thread_service() -> Service:
    global THREAD_SERVICE
    if THREAD_SERVICE is None:
        with THREAD_SERVICE_LOCK:
            if THREAD_SERVICE is None:
                THREAD_SERVICE = Service()
    return THREAD_SERVICE


def _run_query_thread(
    query: str, package_filters: Optional[List[str]], limit: Optional[int]
) -> Tuple[float, bool, int, Optional[str], Optional[str]]:
    start = time.perf_counter()
    try:
        service = _get_thread_service()
        resp = service.search(query, package_filters=package_filters, limit=limit)
        latency = time.perf_counter() - start
        top_item = resp.results[0] if resp.results else None
        top_preview = None
        if top_item:
            name = (
                top_item.primary_declaration.lean_name
                if top_item.primary_declaration
                else "N/A"
            )
            top_preview = f"id={top_item.id} name={name}"
        return latency, True, resp.count, None, top_preview
    except Exception as exc:
        latency = time.perf_counter() - start
        return latency, False, 0, type(exc).__name__, None


def _run_query_process(
    query: str, package_filters: Optional[List[str]], limit: Optional[int]
) -> Tuple[float, bool, int, Optional[str], Optional[str]]:
    global PROCESS_SERVICE
    if PROCESS_SERVICE is None:
        PROCESS_SERVICE = Service()
    start = time.perf_counter()
    try:
        resp = PROCESS_SERVICE.search(query, package_filters=package_filters, limit=limit)
        latency = time.perf_counter() - start
        top_item = resp.results[0] if resp.results else None
        top_preview = None
        if top_item:
            name = (
                top_item.primary_declaration.lean_name
                if top_item.primary_declaration
                else "N/A"
            )
            top_preview = f"id={top_item.id} name={name}"
        return latency, True, resp.count, None, top_preview
    except Exception as exc:
        latency = time.perf_counter() - start
        return latency, False, 0, type(exc).__name__, None


def _run_query_batch_thread(
    queries: List[str],
    package_filters: Optional[List[str]],
    limit: Optional[int],
) -> List[Tuple[float, bool, int, Optional[str], Optional[str]]]:
    start = time.perf_counter()
    service = _get_thread_service()
    try:
        embeddings = fetch_embeddings_batch(queries, service.embedding_model)
    except Exception as exc:
        latency = time.perf_counter() - start
        return [(latency, False, 0, type(exc).__name__, None) for _ in queries]

    batch_latency = time.perf_counter() - start
    per_query_latency = batch_latency / max(1, len(queries))
    results: List[Tuple[float, bool, int, Optional[str], Optional[str]]] = []

    with service.SessionLocal() as session:
        try:
            ranked_batches = perform_search_batch(
                session=session,
                queries=queries,
                embeddings=embeddings,
                faiss_index=service.faiss_index,
                text_chunk_id_map=service.text_chunk_id_map,
                faiss_k=service.default_faiss_k,
                pagerank_weight=service.default_pagerank_weight,
                text_relevance_weight=service.default_text_relevance_weight,
                name_match_weight=service.default_name_match_weight,
                selected_packages=package_filters,
                semantic_similarity_threshold=service.default_semantic_similarity_threshold,
                faiss_nprobe=service.default_faiss_nprobe,
                faiss_oversampling_factor=service.default_faiss_oversampling_factor,
            )
        except Exception as exc:
            return [(per_query_latency, False, 0, type(exc).__name__, None) for _ in queries]

        for ranked_results in ranked_batches:
            actual_limit = limit if limit is not None else service.default_results_limit
            final_results = ranked_results[:actual_limit]
            top_item = final_results[0][0] if final_results else None
            top_preview = None
            if top_item:
                name = (
                    top_item.primary_declaration.lean_name
                    if top_item.primary_declaration
                    else "N/A"
                )
                top_preview = f"id={top_item.id} name={name}"
            results.append((per_query_latency, True, len(final_results), None, top_preview))

    return results


def _load_queries(file_path: pathlib.Path) -> List[str]:
    if not file_path.exists():
        logger.error("Queries file not found: %s", file_path)
        return []
    try:
        with open(file_path, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d queries from %s", len(queries), file_path)
        return queries
    except Exception as exc:
        logger.error("Failed to load queries from %s: %s", file_path, exc, exc_info=True)
        return []


def _percentile(sorted_values: List[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent benchmark for lean_explore local search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=DEFAULT_QUERIES_FILENAME,
        help="Query list file (relative to benchmarking/ if not absolute).",
    )
    parser.add_argument(
        "--total_requests",
        type=int,
        default=DEFAULT_TOTAL_REQUESTS,
        help="Total number of queries to execute.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of concurrent workers.",
    )
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Concurrency model for the benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit per query (None uses default).",
    )
    parser.add_argument(
        "--packages",
        metavar="PKG",
        type=str,
        nargs="*",
        default=None,
        help="Optional package filters (e.g., Mathlib Std).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for embedding requests.",
    )
    parser.add_argument(
        "--print_each",
        action="store_true",
        help="Print a short result summary for each query.",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable performance logging to reduce I/O overhead.",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=None,
        help="Set PyTorch inter/intra-op threads (also sets OMP/MKL threads).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup queries before benchmark (not counted in results).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Set PyTorch/OMP/MKL thread counts early (before any torch import in workers)
    if args.torch_threads is not None and args.torch_threads > 0:
        import os as _os
        import torch as _torch
        _os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
        _os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
        _torch.set_num_threads(args.torch_threads)
        _torch.set_num_interop_threads(args.torch_threads)
        logger.info("Set torch/OMP/MKL threads to %d", args.torch_threads)

    # Disable performance logging if requested
    if args.no_log:
        from lean_explore.local import search as _search_module
        # Monkey-patch the log function to no-op
        _search_module.log_search_event_to_json = lambda *a, **kw: None
        logger.info("Performance logging disabled.")

    queries_path = (
        pathlib.Path(__file__).resolve().parent / args.queries_file
        if not pathlib.Path(args.queries_file).is_absolute()
        else pathlib.Path(args.queries_file)
    )
    queries = _load_queries(queries_path)
    if not queries:
        sys.exit(1)

    if args.total_requests <= 0 or args.concurrency <= 0:
        logger.error("total_requests and concurrency must be > 0.")
        sys.exit(1)

    # Warmup phase
    if args.warmup > 0 and args.executor == "thread":
        logger.info("Running %d warmup queries...", args.warmup)
        service = _get_thread_service()
        warmup_queries = list(itertools.islice(itertools.cycle(queries), args.warmup))
        for wq in warmup_queries:
            try:
                service.search(wq, package_filters=args.packages, limit=args.limit)
            except Exception:
                pass
        logger.info("Warmup complete.")

    logger.info(
        "Benchmark starting: executor=%s, concurrency=%d, total_requests=%d",
        args.executor,
        args.concurrency,
        args.total_requests,
    )

    run_fn = _run_query_process if args.executor == "process" else _run_query_thread
    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    init = _init_process_worker if args.executor == "process" else None

    latencies: List[float] = []
    success = 0
    errors = Counter()

    start_all = time.perf_counter()
    with executor_cls(max_workers=args.concurrency, initializer=init) as executor:
        futures = []
        if args.batch_size > 1 and args.executor == "thread":
            query_list = list(
                itertools.islice(itertools.cycle(queries), args.total_requests)
            )
            batches = [
                query_list[i : i + args.batch_size]
                for i in range(0, len(query_list), args.batch_size)
            ]
            for batch in batches:
                futures.append(
                    executor.submit(
                        _run_query_batch_thread, batch, args.packages, args.limit
                    )
                )
            for future in as_completed(futures):
                for latency, ok, count, err, preview in future.result():
                    latencies.append(latency)
                    if ok:
                        success += 1
                        if args.print_each:
                            preview_text = preview if preview else "no results"
                            print(
                                f"[OK] latency={latency:.4f}s count={count} top={preview_text}"
                            )
                    else:
                        errors[err or "UnknownError"] += 1
                        if args.print_each:
                            err_text = err or "UnknownError"
                            print(f"[ERR] latency={latency:.4f}s error={err_text}")
        else:
            for i, query in zip(range(args.total_requests), itertools.cycle(queries)):
                futures.append(executor.submit(run_fn, query, args.packages, args.limit))
            for future in as_completed(futures):
                latency, ok, count, err, preview = future.result()
                latencies.append(latency)
                if ok:
                    success += 1
                    if args.print_each:
                        preview_text = preview if preview else "no results"
                        print(
                            f"[OK] latency={latency:.4f}s count={count} top={preview_text}"
                        )
                else:
                    errors[err or "UnknownError"] += 1
                    if args.print_each:
                        err_text = err or "UnknownError"
                        print(f"[ERR] latency={latency:.4f}s error={err_text}")
    end_all = time.perf_counter()

    total_time = end_all - start_all
    latencies.sort()

    qps = success / total_time if total_time > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p50 = _percentile(latencies, 50)
    p90 = _percentile(latencies, 90)
    p99 = _percentile(latencies, 99)

    print("\n--- Concurrent Benchmark Results ---")
    print(f"Executor:                     {args.executor}")
    print(f"Concurrency:                  {args.concurrency}")
    print(f"Total Requests:               {args.total_requests}")
    print(f"Total Time:                   {total_time:.3f} s")
    print(f"Successful Requests:          {success}")
    print(f"Failed Requests:              {sum(errors.values())}")
    print(f"QPS (successful):             {qps:.2f}")
    print(f"Average Latency:              {avg_latency:.4f} s")
    if p50 is not None:
        print(f"P50 Latency:                  {p50:.4f} s")
    if p90 is not None:
        print(f"P90 Latency:                  {p90:.4f} s")
    if p99 is not None:
        print(f"P99 Latency:                  {p99:.4f} s")
    if latencies:
        print(f"Min Latency:                  {latencies[0]:.4f} s")
        print(f"Max Latency:                  {latencies[-1]:.4f} s")
    if errors:
        print("Errors by type:")
        for err_type, count in errors.most_common():
            print(f"  {err_type}: {count}")
    print("------------------------------------\n")


if __name__ == "__main__":
    main()
