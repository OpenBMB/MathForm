"""Local search HTTP server."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import socket
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

TRUE_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRUE_PROJECT_ROOT / "src"))

try:
    from lean_explore.local.service import Service
except ImportError as exc:
    print(
        "Cannot import project modules; make sure src is installed/discoverable."
        f"\nOriginal error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


SERVICE_LOCK = threading.Lock()
SERVICE_INSTANCE: Optional[Service] = None
REQUEST_SEMAPHORE: Optional[threading.Semaphore] = None
REQUEST_SEMAPHORE_TIMEOUT = 0.0
BATCH_MANAGER: Optional["BatchManager"] = None
BATCH_REQUEST_TIMEOUT = 30.0
CACHE_LOCK = threading.Lock()
CACHE_INSTANCE: Optional["LRUCache"] = None
STATS_LOCK = threading.Lock()
STATS = {
    "requests_total": 0,
    "cache_hit": 0,
    "cache_miss": 0,
}


def _get_service() -> Service:
    """Per-process singleton Service to avoid repeated loading under load."""
    global SERVICE_INSTANCE
    if SERVICE_INSTANCE is None:
        with SERVICE_LOCK:
            if SERVICE_INSTANCE is None:
                SERVICE_INSTANCE = Service()
    return SERVICE_INSTANCE


class LRUCache:
    """Simple LRU cache with TTL."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.max_size = max(0, max_size)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._data: OrderedDict[object, tuple[float, object]] = OrderedDict()

    def _is_expired(self, timestamp: float) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - timestamp) > self.ttl_seconds

    def get(self, key: object) -> Optional[object]:
        if self.max_size <= 0:
            return None
        item = self._data.get(key)
        if item is None:
            return None
        ts, value = item
        if self._is_expired(ts):
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: object, value: object) -> None:
        if self.max_size <= 0:
            return
        self._data[key] = (time.time(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)


class BatchRequest:
    def __init__(
        self, query: str, package_filters: Optional[List[str]], limit: Optional[int]
    ) -> None:
        self.query = query
        self.package_filters = package_filters
        self.limit = limit
        self.event = threading.Event()
        self.response: Optional[Any] = None
        self.error: Optional[Exception] = None


class BatchManager:
    def __init__(self, max_size: int, wait_ms: float) -> None:
        self.max_size = max(1, max_size)
        self.wait_s = max(0.0, wait_ms / 1000.0)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queues: Dict[Tuple[Tuple[str, ...], Optional[int]], List[BatchRequest]] = {}
        self._running = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, req: BatchRequest) -> None:
        key = (tuple(req.package_filters or []), req.limit)
        with self._cond:
            self._queues.setdefault(key, []).append(req)
            self._cond.notify()

    def _drain(self) -> Dict[Tuple[Tuple[str, ...], Optional[int]], List[BatchRequest]]:
        batches: Dict[Tuple[Tuple[str, ...], Optional[int]], List[BatchRequest]] = {}
        for key, queue in list(self._queues.items()):
            if not queue:
                continue
            batches[key] = queue[:]
            self._queues[key] = []
        return batches

    def _run(self) -> None:
        while self._running:
            with self._cond:
                while self._running and not any(self._queues.values()):
                    self._cond.wait(timeout=0.1)
                if not self._running:
                    break
                if self.wait_s > 0:
                    self._cond.wait(timeout=self.wait_s)
                batches = self._drain()
            if not batches:
                continue

            service = _get_service()
            for (pkg_filters, limit), reqs in batches.items():
                if not reqs:
                    continue
                for i in range(0, len(reqs), self.max_size):
                    chunk = reqs[i : i + self.max_size]
                    queries = [req.query for req in chunk]
                    try:
                        responses = service.search(
                            queries,
                            package_filters=list(pkg_filters) if pkg_filters else None,
                            limit=limit,
                        )
                        for req, resp in zip(chunk, responses):
                            req.response = resp
                    except Exception as exc:
                        for req in chunk:
                            req.error = exc
                    finally:
                        for req in chunk:
                            req.event.set()


def _get_cache(max_size: int, ttl_seconds: float) -> Optional[LRUCache]:
    global CACHE_INSTANCE
    if max_size <= 0:
        return None
    if CACHE_INSTANCE is None:
        with CACHE_LOCK:
            if CACHE_INSTANCE is None:
                CACHE_INSTANCE = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return CACHE_INSTANCE


def _stats_inc(key: str, delta: int = 1) -> None:
    with STATS_LOCK:
        STATS[key] = STATS.get(key, 0) + delta


def _normalize_packages(packages: Optional[List[str]]) -> Optional[List[str]]:
    if not packages:
        return None
    expanded: List[str] = []
    for item in packages:
        if not item:
            continue
        if isinstance(item, str):
            parts = item.split(",")
        else:
            parts = [str(item)]
        for part in parts:
            part = part.strip()
            if part:
                expanded.append(part)
    return expanded or None


class SearchHandler(BaseHTTPRequestHandler):

    server_version = "LeanExploreSearchServer/1.0"
    protocol_version = "HTTP/1.1"

    def _send_json(
        self,
        status_code: int,
        payload: Dict[str, Any],
        close: bool = False,
        cache: Optional[str] = None,
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            if cache is not None:
                self.send_header("X-Cache", cache)
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except BrokenPipeError:
            # Client disconnected early; ignore write-back failure
            return
        except ConnectionResetError:
            return

    def _drain_body(self) -> None:
        """Read and discard the request body to keep the connection clean."""
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            content_length = 0
        if content_length > 0:
            _ = self.rfile.read(content_length)

    def _read_json(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            raise ValueError("empty request body")
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok"})
            return
        if path == "/stats":
            with STATS_LOCK:
                payload = dict(STATS)
            self._send_json(200, payload)
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in ("/search", "/search_batch"):
            self._send_json(404, {"error": "not_found"})
            return
        acquired = False
        if REQUEST_SEMAPHORE is not None:
            acquired = REQUEST_SEMAPHORE.acquire(timeout=REQUEST_SEMAPHORE_TIMEOUT)
            if not acquired:
                self._drain_body()
                self._send_json(503, {"error": "server_busy"}, close=True)
                return
        try:
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": "invalid_json", "message": str(exc)})
                return

            query = body.get("query")
            queries = body.get("queries")
            packages = body.get("packages") or body.get("package_filters")
            limit = body.get("limit")
            no_cache = bool(body.get("no_cache", False))

            if query is None and queries is None:
                self._send_json(
                    400, {"error": "missing_query", "message": "provide query or queries"}
                )
                return

            package_filters = _normalize_packages(packages)
            try:
                service = _get_service()
                _stats_inc("requests_total", 1)
                cache = None
                if not no_cache:
                    cache = _get_cache(self.server.cache_size, self.server.cache_ttl)
                cache_status = None

                if queries is not None or path == "/search_batch":
                    query_list = list(queries or [])
                    if not query_list:
                        self._send_json(
                            400,
                            {"error": "missing_queries", "message": "queries must not be empty"},
                        )
                        return
                    results_payload: List[Dict[str, Any]] = []
                    missing_queries: List[str] = []
                    cached_map: Dict[str, Dict[str, Any]] = {}
                    if cache is not None:
                        with CACHE_LOCK:
                            for q in query_list:
                                key = ("q", q, tuple(package_filters or []), limit)
                                cached = cache.get(key)
                                if cached is None:
                                    missing_queries.append(q)
                                    _stats_inc("cache_miss", 1)
                                else:
                                    cached_map[q] = cached
                                    _stats_inc("cache_hit", 1)
                    else:
                        missing_queries = query_list

                    if missing_queries:
                        responses = service.search(
                            missing_queries,
                            package_filters=package_filters,
                            limit=limit,
                        )
                        for q, resp in zip(missing_queries, responses):
                            payload_item = resp.model_dump()
                            results_payload.append(payload_item)
                            if cache is not None:
                                with CACHE_LOCK:
                                    cache.set(
                                        ("q", q, tuple(package_filters or []), limit),
                                        payload_item,
                                    )
                    if cached_map:
                        for q in query_list:
                            if q in cached_map:
                                results_payload.append(cached_map[q])

                    payload = {"results": results_payload}
                    self._send_json(200, payload, cache=cache_status)
                else:
                    key = ("q", str(query), tuple(package_filters or []), limit)
                    cached_resp = None
                    if cache is not None:
                        with CACHE_LOCK:
                            cached_resp = cache.get(key)
                            if cached_resp is not None:
                                _stats_inc("cache_hit", 1)
                                cache_status = "HIT"
                            else:
                                _stats_inc("cache_miss", 1)
                                cache_status = "MISS"
                    if cached_resp is not None:
                        payload = {"result": cached_resp}
                        self._send_json(200, payload, cache=cache_status)
                    else:
                        if BATCH_MANAGER is not None:
                            req = BatchRequest(
                                str(query),
                                package_filters=package_filters,
                                limit=limit,
                            )
                            BATCH_MANAGER.submit(req)
                            if not req.event.wait(timeout=BATCH_REQUEST_TIMEOUT):
                                self._send_json(
                                    503,
                                    {"error": "batch_timeout", "message": "batch processing timed out"},
                                    close=True,
                                )
                                return
                            if req.error is not None or req.response is None:
                                self._send_json(
                                    500,
                                    {
                                        "error": "search_failed",
                                        "message": type(req.error).__name__
                                        if req.error
                                        else "UnknownError",
                                    },
                                )
                                return
                            payload_item = req.response.model_dump()
                        else:
                            resp = service.search(
                                str(query), package_filters=package_filters, limit=limit
                            )
                            payload_item = resp.model_dump()
                        if cache is not None:
                            with CACHE_LOCK:
                                cache.set(key, payload_item)
                        payload = {"result": payload_item}
                        self._send_json(200, payload, cache=cache_status)
            except Exception as exc:  # pragma: no cover - runtime errors
                self._send_json(
                    500,
                    {"error": "search_failed", "message": type(exc).__name__},
                )
        finally:
            if REQUEST_SEMAPHORE is not None and acquired:
                REQUEST_SEMAPHORE.release()


class HighConcurrencyHTTPServer(ThreadingHTTPServer):
    """HTTP server with SO_REUSEPORT support for multi-process serving."""

    allow_reuse_address = True
    allow_reuse_port = True
    daemon_threads = True
    request_queue_size = 1024

    def server_bind(self) -> None:
        if getattr(self, "allow_reuse_port", False):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                # SO_REUSEPORT unsupported on some systems/kernels
                pass
        super().server_bind()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Lean Explore local search HTTP server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="listen host")
    parser.add_argument("--port", type=int, default=8000, help="listen port")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes (>1 enables port reuse for higher concurrency)",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=32,
        help="max concurrent requests (thread cap) to bound memory usage",
    )
    parser.add_argument(
        "--queue-timeout",
        type=float,
        default=0.0,
        help="queue wait time in seconds; 0 rejects immediately",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=2000,
        help="cache size limit; 0 disables caching",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=120.0,
        help="cache TTL in seconds; 0 means no expiry",
    )
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help="prewarm the Service at startup (for high-concurrency serving)",
    )
    parser.add_argument(
        "--batch-max-size",
        type=int,
        default=0,
        help="max auto-merged batch size per request (<=1 disables)",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=5.0,
        help="batching wait window in milliseconds",
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=30.0,
        help="max seconds a request waits for its batched result",
    )
    return parser.parse_args()


def _serve(
    host: str,
    port: int,
    max_threads: int,
    queue_timeout: float,
    cache_size: int,
    cache_ttl: float,
    prewarm: bool,
    batch_max_size: int,
    batch_wait_ms: float,
    batch_timeout: float,
) -> None:
    global REQUEST_SEMAPHORE
    global REQUEST_SEMAPHORE_TIMEOUT
    if max_threads > 0:
        REQUEST_SEMAPHORE = threading.Semaphore(max_threads)
        REQUEST_SEMAPHORE_TIMEOUT = max(0.0, queue_timeout)
    if batch_max_size > 1:
        global BATCH_MANAGER
        global BATCH_REQUEST_TIMEOUT
        BATCH_MANAGER = BatchManager(batch_max_size, batch_wait_ms)
        BATCH_REQUEST_TIMEOUT = max(0.1, batch_timeout)
    print(f"Search server listening on http://{host}:{port}", flush=True)
    server = HighConcurrencyHTTPServer((host, port), SearchHandler)
    server.cache_size = cache_size
    server.cache_ttl = cache_ttl
    if prewarm:
        _get_service()
    server.serve_forever()


def main() -> int:
    args = _parse_args()
    if args.workers <= 1:
        global REQUEST_SEMAPHORE
        global REQUEST_SEMAPHORE_TIMEOUT
        if args.max_threads > 0:
            REQUEST_SEMAPHORE = threading.Semaphore(args.max_threads)
            REQUEST_SEMAPHORE_TIMEOUT = max(0.0, args.queue_timeout)
        if args.batch_max_size > 1:
            global BATCH_MANAGER
            global BATCH_REQUEST_TIMEOUT
            BATCH_MANAGER = BatchManager(args.batch_max_size, args.batch_wait_ms)
            BATCH_REQUEST_TIMEOUT = max(0.1, args.batch_timeout)
        server = HighConcurrencyHTTPServer((args.host, args.port), SearchHandler)
        try:
            server.cache_size = args.cache_size
            server.cache_ttl = args.cache_ttl
            if args.prewarm:
                _get_service()
            print(
                f"Search server listening on http://{args.host}:{args.port}",
                flush=True,
            )
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
        finally:
            server.server_close()
        return 0

    workers = []
    for _ in range(args.workers):
        proc = multiprocessing.Process(
            target=_serve,
            args=(
                args.host,
                args.port,
                args.max_threads,
                args.queue_timeout,
                args.cache_size,
                args.cache_ttl,
                args.prewarm,
                args.batch_max_size,
                args.batch_wait_ms,
                args.batch_timeout,
            ),
        )
        proc.start()
        workers.append(proc)

    try:
        for proc in workers:
            proc.join()
    except KeyboardInterrupt:
        for proc in workers:
            proc.terminate()
        for proc in workers:
            proc.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
