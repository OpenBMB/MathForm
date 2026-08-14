"""Retrieval: search-query generation and Lean Explore lookup."""

import asyncio
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

import aiohttp
from tqdm import tqdm

from api_client import APIClient
from config import APIConfig, PipelineConfig
from utils import Logger, TextProcessor


class RetrievalService:
    
    def __init__(
        self,
        api_client: APIClient,
        api_config: APIConfig,
        pipeline_config: PipelineConfig,
        logger: Optional[Logger] = None,
    ):
        self.api_client = api_client
        self.api_config = api_config
        self.pipeline_config = pipeline_config
        self.logger = logger or Logger()
        self._lean_explore_ready = False
        self._lean_explore_ready_lock = threading.Lock()
        self._lean_explore_batch_max = int(os.getenv("LEAN_EXPLORE_BATCH_MAX", "256"))
        self._lean_explore_batch_min = int(os.getenv("LEAN_EXPLORE_BATCH_MIN", "32"))
        self._lean_explore_busy_cooldown = float(
            os.getenv("LEAN_EXPLORE_BUSY_COOLDOWN", "0.5")
        )
        self._lean_explore_busy_until = 0.0
        self._lean_explore_busy_lock = threading.Lock()
        
        self._retrieval_semaphore = (
            threading.Semaphore(pipeline_config.retrieval_workers)
            if pipeline_config.retrieval_workers and pipeline_config.retrieval_workers > 0
            else None
        )
    
    def _get_lean_explore_batch_url(self) -> Optional[str]:
        if not self.api_config.lean_explore_url:
            return None
        if self.api_config.lean_explore_url.endswith("/search_batch"):
            return self.api_config.lean_explore_url
        return self.api_config.lean_explore_url.rstrip("/") + "/search_batch"

    def _get_lean_explore_health_url(self) -> Optional[str]:
        if not self.api_config.lean_explore_url:
            return None
        base = self.api_config.lean_explore_url.rstrip("/")
        if base.endswith("/search_batch"):
            base = base[: -len("/search_batch")]
        return base + "/health"

    def _ensure_lean_explore_ready(self, timeout: int = 180, check_interval: float = 2.0) -> bool:
        if self._lean_explore_ready:
            return True
        url = self._get_lean_explore_health_url()
        if not url:
            return False
        with self._lean_explore_ready_lock:
            if self._lean_explore_ready:
                return True
            session = self.api_client._get_session()
            start = time.time()
            while time.time() - start < timeout:
                try:
                    resp = session.get(url, timeout=5)
                    if resp.status_code == 200:
                        self._lean_explore_ready = True
                        return True
                except Exception:
                    pass
                time.sleep(check_interval)
        return False

    def _mark_lean_explore_busy(self, extra_delay: float = 0.0) -> None:
        if self._lean_explore_busy_cooldown <= 0:
            return
        delay = self._lean_explore_busy_cooldown + max(0.0, extra_delay)
        now = time.time()
        with self._lean_explore_busy_lock:
            self._lean_explore_busy_until = max(self._lean_explore_busy_until, now + delay)

    def _sleep_if_lean_explore_busy(self) -> None:
        if self._lean_explore_busy_cooldown <= 0:
            return
        with self._lean_explore_busy_lock:
            wait = self._lean_explore_busy_until - time.time()
        if wait > 0:
            time.sleep(wait)

    async def _sleep_if_lean_explore_busy_async(self) -> None:
        if self._lean_explore_busy_cooldown <= 0:
            return
        with self._lean_explore_busy_lock:
            wait = self._lean_explore_busy_until - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
    
    def _build_query_prompt(
        self,
        statement: str,
        retrieval_context: str = "",
        previous_queries: Optional[List[str]] = None,
        compile_error: str = "",
        semantic_feedback: str = "",
    ) -> str:
        previous_queries = previous_queries or []
        previous_queries_block = "\n".join(f"- {q}" for q in previous_queries) or "(none)"
        prompt = f"""You are a Lean 4/Mathlib expert. I need to formalize the mathematical statement below into Lean 4 code.

To ensure accurate formalization, I will search the Lean mathematical library (Mathlib) for relevant definitions and theorems.

First review the existing queries and retrieved results to decide whether any new search queries are truly necessary. Only generate new queries if they are essential to resolve ambiguity or missing definitions. If more info is needed, generate 1-3 new queries; otherwise output nothing.

Each query should be:
- Concise and specific
- Focus on mathematical concepts, definitions, or theorems

Do not repeat any previous queries.
Each query must be highly necessary and directly helpful for formalization; avoid broad or speculative queries.
If no additional queries are needed, output an EMPTY code block only.
Output format: output ONLY a single triple-backtick block. One query per line, no numbering or bullets. No extra text, no explanations.

Examples:
```
group definition
ring homomorphism
topological space compactness
natural number induction
vector space dimension
```

Empty output example (no additional queries needed):
```
```

Previous queries:
{previous_queries_block}"""

        if retrieval_context:
            prompt += f"""

Retrieved results so far:
{retrieval_context}"""

        if compile_error or semantic_feedback:
            prompt += "\n\nPrevious verification feedback:"
            if compile_error:
                prompt += f"\n- Compilation error: {compile_error}"
            if semantic_feedback:
                prompt += f"\n- Semantic feedback: {semantic_feedback}"

        prompt += f"""

Statement to formalize:
{statement}"""
        return prompt
    
    def generate_queries(
        self,
        statement: str,
        retrieval_history: Optional[List[Dict[str, Any]]] = None,
        compile_error: str = "",
        semantic_feedback: str = "",
    ) -> List[str]:
        retrieval_history = retrieval_history or []
        previous_queries = []
        seen = set()
        for entry in retrieval_history:
            q = entry.get("query", "").strip()
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                previous_queries.append(q)
        
        retrieval_context = self.format_retrieval_context(retrieval_history)
        prompt = self._build_query_prompt(
            statement=statement,
            retrieval_context=retrieval_context,
            previous_queries=previous_queries,
            compile_error=compile_error,
            semantic_feedback=semantic_feedback,
        )
        
        response = self.api_client.call_retrieval_planner(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        
        queries = TextProcessor.parse_query_list(response)
        if previous_queries:
            prev_set = {q.lower().strip() for q in previous_queries if q.strip()}
            queries = [q for q in queries if q.lower().strip() not in prev_set]
        return queries[:5]
    
    def _format_lean_explore_item(self, item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, str):
            cleaned = TextProcessor.collapse_whitespace(item)
            return {"raw": cleaned} if cleaned else None
        if not isinstance(item, dict):
            return None
        primary_decl = item.get("primary_declaration") or {}
        lean_name = ""
        if isinstance(primary_decl, dict):
            lean_name = primary_decl.get("lean_name") or ""
        formatted = {
            "id": item.get("id"),
            "primary_declaration": primary_decl if isinstance(primary_decl, dict) else None,
            "lean_name": lean_name,
            "source_file": item.get("source_file"),
            "range_start_line": item.get("range_start_line"),
            "display_statement_text": item.get("display_statement_text"),
            "statement_text": item.get("statement_text"),
            "docstring": item.get("docstring"),
            "informal_description": item.get("informal_description"),
        }
        return formatted
    
    def _search_lean_explore_batch(
        self,
        queries: List[str],
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        if not queries:
            return {}
        if not self._ensure_lean_explore_ready():
            self.logger.log("Lean Explore not ready (health check failed)")
            return None
        url = self._get_lean_explore_batch_url()
        if not url:
            return None
        
        session = self.api_client._get_session()
        semaphore = self._retrieval_semaphore
        if semaphore:
            semaphore.acquire()
        try:
            for attempt in range(5):
                try:
                    self._sleep_if_lean_explore_busy()
                    resp = session.post(
                        url,
                        json={"queries": queries, "limit": self.pipeline_config.query_top_k},
                        timeout=self.api_config.request_timeout,
                    )
                    if resp.status_code == 200:
                        data = resp.json() if resp.content else {}
                        results = data.get("results") or []
                        mapping: Dict[str, List[Dict[str, Any]]] = {}
                        if isinstance(results, list):
                            for entry in results:
                                if not isinstance(entry, dict):
                                    continue
                                query = entry.get("query") or ""
                                raw_results = entry.get("results") or []
                                formatted: List[Dict[str, Any]] = []
                                if isinstance(raw_results, list):
                                    for item in raw_results:
                                        rendered = self._format_lean_explore_item(item)
                                        if rendered:
                                            formatted.append(rendered)
                                if query:
                                    mapping[query] = formatted
                        return mapping
                    if resp.status_code in {429, 500, 502, 503, 504}:
                        self.logger.log(f"Lean Explore error {resp.status_code}: {resp.text[:200]}")
                        self._mark_lean_explore_busy(min(2.0, 0.25 * (2 ** attempt)))
                        time.sleep(min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.2))
                        continue
                    self.logger.log(f"Lean Explore error {resp.status_code}: {resp.text[:200]}")
                    return None
                except Exception as exc:
                    self.logger.log(f"Lean Explore batch call failed: {exc}")
                    self._mark_lean_explore_busy(min(2.0, 0.25 * (2 ** attempt)))
                    time.sleep(min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.2))
            return None
        finally:
            if semaphore:
                semaphore.release()

    async def _search_lean_explore_batch_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: Optional[asyncio.Semaphore],
        queries: List[str],
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        if not queries:
            return {}
        if not self._ensure_lean_explore_ready():
            self.logger.log("Lean Explore not ready (health check failed)")
            return None
        url = self._get_lean_explore_batch_url()
        if not url:
            return None

        async def _do_request() -> Optional[Dict[str, List[Dict[str, Any]]]]:
            for attempt in range(5):
                try:
                    await self._sleep_if_lean_explore_busy_async()
                    async with session.post(
                        url,
                        json={"queries": queries, "limit": self.pipeline_config.query_top_k},
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json() if resp.content_length != 0 else {}
                            results = data.get("results") or []
                            mapping: Dict[str, List[Dict[str, Any]]] = {}
                            if isinstance(results, list):
                                for entry in results:
                                    if not isinstance(entry, dict):
                                        continue
                                    query = entry.get("query") or ""
                                    raw_results = entry.get("results") or []
                                    formatted: List[Dict[str, Any]] = []
                                    if isinstance(raw_results, list):
                                        for item in raw_results:
                                            rendered = self._format_lean_explore_item(item)
                                            if rendered:
                                                formatted.append(rendered)
                                    if query:
                                        mapping[query] = formatted
                            return mapping
                        if resp.status in {429, 500, 502, 503, 504}:
                            text = await resp.text()
                            self.logger.log(f"Lean Explore error {resp.status}: {text[:200]}")
                            self._mark_lean_explore_busy(min(2.0, 0.25 * (2 ** attempt)))
                            await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.2))
                            continue
                        text = await resp.text()
                        self.logger.log(f"Lean Explore error {resp.status}: {text[:200]}")
                        return None
                except Exception as exc:
                    self.logger.log(f"Lean Explore async batch call failed: {exc}")
                    self._mark_lean_explore_busy(min(2.0, 0.25 * (2 ** attempt)))
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.2))
            return None

        if semaphore:
            async with semaphore:
                return await _do_request()
        return await _do_request()
    
    def run_retrieval(self, queries: List[str]) -> List[Dict[str, Any]]:
        if not queries:
            return []
        if not self._ensure_lean_explore_ready():
            self.logger.log("Lean Explore not ready (health check failed)")
            return []
        mapping = self._search_lean_explore_batch(queries)
        if mapping is None:
            self.logger.log("Lean Explore returned no results (mapping is None)")
            return []
        entries: List[Dict[str, Any]] = []
        with tqdm(total=len(queries), desc="Retrieval", unit="query", ncols=120, leave=True) as pbar:
            for q in queries:
                results = mapping.get(q, [])
                if results:
                    for res in results:
                        entries.append({"query": q, "result": res})
                else:
                    entries.append({"query": q, "result": {"raw": "(no results)"}})
                time.sleep(random.uniform(0.05, 0.2))
                pbar.update(1)
        return entries

    def run_retrieval_batch(
        self,
        queries: List[str],
        batch_size: int = 200,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run retrieval for all queries and return a merged query->results map."""
        if not queries:
            return {}
        if not self._ensure_lean_explore_ready():
            self.logger.log("Lean Explore not ready (health check failed)")
            return {}
        if self._lean_explore_batch_max > 0 and batch_size > self._lean_explore_batch_max:
            self.logger.log(
                f"Lean Explore batch_size {batch_size} exceeds max {self._lean_explore_batch_max}, clamping"
            )
            batch_size = self._lean_explore_batch_max
        batch_min = max(self._lean_explore_batch_min, 1)

        merged: Dict[str, List[Dict[str, Any]]] = {}
        total = len(queries)
        chunks = [queries[i:i + batch_size] for i in range(0, total, batch_size)]

        # Fall back to the sync path if this thread already runs an event loop
        try:
            asyncio.get_running_loop()
            use_async = False
        except RuntimeError:
            use_async = True

        if not use_async:
            def _fetch_chunk_sync(chunk: List[str]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
                mapping = self._search_lean_explore_batch(chunk)
                if mapping is not None:
                    return mapping
                if len(chunk) <= batch_min:
                    return None
                self.logger.log(
                    f"Lean Explore batch failed for {len(chunk)} queries; retrying with smaller chunks"
                )
                mid = len(chunk) // 2
                left = _fetch_chunk_sync(chunk[:mid])
                right = _fetch_chunk_sync(chunk[mid:])
                merged_map: Dict[str, List[Dict[str, Any]]] = {}
                if left:
                    merged_map.update(left)
                if right:
                    merged_map.update(right)
                return merged_map or None

            with tqdm(total=total, desc="Retrieval(Batch)", unit="query", ncols=120, leave=True) as pbar:
                for i in range(0, total, batch_size):
                    chunk = queries[i:i + batch_size]
                    mapping = _fetch_chunk_sync(chunk)
                    if mapping is None:
                        self.logger.log(f"Lean Explore batch returned None for queries[{i}:{i + len(chunk)}]")
                        for q in chunk:
                            merged[q] = []
                        pbar.update(len(chunk))
                        continue
                    for q, results in mapping.items():
                        merged[q] = results
                    for q in chunk:
                        merged.setdefault(q, [])
                    pbar.update(len(chunk))
            return merged

        async def _run_async() -> Dict[str, List[Dict[str, Any]]]:
            timeout = aiohttp.ClientTimeout(total=self.api_config.request_timeout)
            max_workers = max(self.pipeline_config.retrieval_workers or 0, 1)
            semaphore = asyncio.Semaphore(max_workers) if max_workers > 0 else None

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async def _fetch_chunk_async(
                    chunk: List[str],
                ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
                    mapping = await self._search_lean_explore_batch_async(session, semaphore, chunk)
                    if mapping is not None:
                        return mapping
                    if len(chunk) <= batch_min:
                        return None
                    self.logger.log(
                        f"Lean Explore batch failed for {len(chunk)} queries; retrying with smaller chunks"
                    )
                    mid = len(chunk) // 2
                    left = await _fetch_chunk_async(chunk[:mid])
                    right = await _fetch_chunk_async(chunk[mid:])
                    merged_map: Dict[str, List[Dict[str, Any]]] = {}
                    if left:
                        merged_map.update(left)
                    if right:
                        merged_map.update(right)
                    return merged_map or None

                async def _run_chunk(
                    chunk: List[str],
                ) -> tuple[List[str], Optional[Dict[str, List[Dict[str, Any]]]]]:
                    result = await _fetch_chunk_async(chunk)
                    return chunk, result

                tasks = [
                    asyncio.create_task(_run_chunk(chunk))
                    for chunk in chunks
                ]

                with tqdm(total=total, desc="Retrieval(Batch)", unit="query", ncols=120, leave=True) as pbar:
                    for task in asyncio.as_completed(tasks):
                        try:
                            chunk, mapping = await task
                        except Exception as exc:
                            self.logger.log(f"Lean Explore async task failed: {exc}")
                            chunk, mapping = [], None

                        if mapping is None:
                            self.logger.log(f"Lean Explore batch returned None for queries[{len(chunk)}]")
                            for q in chunk:
                                merged[q] = []
                            pbar.update(len(chunk))
                            continue
                        for q, results in mapping.items():
                            merged[q] = results
                        for q in chunk:
                            merged.setdefault(q, [])
                        pbar.update(len(chunk))
            return merged

        return asyncio.run(_run_async())

    def build_entries_from_mapping(
        self,
        queries: List[str],
        mapping: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for q in queries:
            results = mapping.get(q, [])
            if results:
                for res in results:
                    entries.append({"query": q, "result": res})
            else:
                entries.append({"query": q, "result": {"raw": "(no results)"}})
        return entries
    
    def append_retrieval_history(
        self,
        history: List[Dict[str, Any]],
        entries: List[Dict[str, Any]],
    ) -> None:
        history.extend(entries)
        if len(history) > self.pipeline_config.retrieval_limit:
            del history[:-self.pipeline_config.retrieval_limit]
    
    def format_retrieval_context(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return ""
        lines: List[str] = []
        last_query = None
        query_idx = 0
        for entry in history:
            query = entry.get("query", "").strip()
            result = entry.get("result")
            if isinstance(result, dict):
                result_lines = []
                for key in (
                    "lean_name",
                    "display_statement_text",
                    "statement_text",
                    "docstring",
                    "informal_description",
                    "source_file",
                    "range_start_line",
                    "id",
                ):
                    value = result.get(key)
                    if value is None or value == "":
                        continue
                    if key in ("display_statement_text", "statement_text", "docstring", "informal_description"):
                        value = TextProcessor.truncate_text(
                            TextProcessor.collapse_whitespace(str(value)), 800
                        )
                    result_lines.append(f"{key}: {value}")
                if not result_lines and "raw" in result:
                    result_lines.append(f"raw: {result.get('raw')}")
                result_text = "; ".join(result_lines)
            else:
                result_text = str(result).strip() if result else ""
            if query and query != last_query:
                query_idx += 1
                lines.append(f"[{query_idx}] Query: {query}")
                last_query = query
            if result_text:
                lines.append(f"- {result_text}")
        return "\n".join(lines)
