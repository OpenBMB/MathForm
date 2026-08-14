"""HTTP concurrent search load-testing script."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests


THREAD_LOCAL = threading.local()


def _get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        THREAD_LOCAL.session = session
    return session


def _load_queries(file_path: pathlib.Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"query file not found: {file_path}")
    with file_path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _normalize_packages(packages: Optional[List[str]]) -> Optional[List[str]]:
    if not packages:
        return None
    expanded: List[str] = []
    for item in packages:
        if not item:
            continue
        for part in item.split(","):
            part = part.strip()
            if part:
                expanded.append(part)
    return expanded or None


def _percentile(sorted_values: List[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _post_once(
    url: str,
    payload: Dict[str, object],
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> Tuple[float, bool, Optional[int], Optional[str]]:
    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            session = _get_session()
            resp = session.post(url, json=payload, timeout=timeout)
            latency = time.perf_counter() - start
            if resp.status_code == 200:
                return latency, True, resp.status_code, None
            last_error = resp.text[:200]
            if resp.status_code in {429, 502, 503, 504} and attempt < retries:
                time.sleep(retry_backoff * (attempt + 1))
                continue
            return latency, False, resp.status_code, last_error
        except requests.RequestException as exc:
            latency = time.perf_counter() - start
            last_error = type(exc).__name__
            if attempt < retries:
                time.sleep(retry_backoff * (attempt + 1))
                continue
            return latency, False, None, last_error
    return 0.0, False, None, last_error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTTP concurrent search load test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/search")
    parser.add_argument(
        "--queries-file",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "queries.txt",
        help="query file (one per line)",
    )
    parser.add_argument("--total", type=int, default=1000, help="total number of requests")
    parser.add_argument("--concurrency", type=int, default=50, help="concurrency level")
    parser.add_argument("--limit", type=int, default=3, help="max results per request")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="queries per request; >1 sends the batch queries field",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=None,
        help="package filter (repeatable or comma-separated)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="retries on failure")
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=0.2,
        help="retry backoff base in seconds",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable server-side cache for requests",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    packages = _normalize_packages(args.package)
    queries = _load_queries(args.queries_file)
    if not queries:
        print("query file is empty.")
        return 2

    latencies: List[float] = []
    errors = collections.Counter()
    ok_count = 0
    total_queries_sent = 0

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for _ in range(args.total):
            if args.batch_size > 1:
                q = random.sample(queries, k=min(args.batch_size, len(queries)))
                total_queries_sent += len(q)
                payload: Dict[str, object] = {"queries": q, "limit": args.limit}
            else:
                q = random.choice(queries)
                total_queries_sent += 1
                payload = {"query": q, "limit": args.limit}
            if packages:
                payload["packages"] = packages
            if args.no_cache:
                payload["no_cache"] = True
            futures.append(
                executor.submit(
                    _post_once,
                    args.url,
                    payload,
                    args.timeout,
                    args.retries,
                    args.retry_backoff,
                )
            )
        for fut in as_completed(futures):
            latency, ok, status, err = fut.result()
            latencies.append(latency)
            if ok:
                ok_count += 1
            else:
                key = str(status) if status is not None else "exception"
                if err:
                    key = f"{key}:{err}"
                errors[key] += 1

    total_time = time.perf_counter() - start_all
    latencies.sort()

    p50 = _percentile(latencies, 50)
    p90 = _percentile(latencies, 90)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)

    print(f"url: {args.url}")
    print(f"total: {args.total}  concurrency: {args.concurrency}")
    if args.batch_size > 1:
        print(f"batch_size: {args.batch_size}")
    print(f"success: {ok_count}  failed: {args.total - ok_count}")
    request_qps = args.total / total_time
    print(f"qps: {request_qps:.2f}")
    if args.batch_size > 1:
        query_qps = total_queries_sent / total_time
        print(f"query_qps: {query_qps:.2f}")
    print(
        "latency(s): "
        f"p50={p50:.4f} p90={p90:.4f} p95={p95:.4f} p99={p99:.4f}"
    )
    if errors:
        print("errors:")
        for k, v in errors.most_common():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
