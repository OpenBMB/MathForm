"""Compilation-check client for the Kimina Lean Server."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from kimina_client import AsyncKiminaClient, Snippet
except ImportError:
    # If kimina_client is not installed, point KIMINA_CLIENT_PATH to its source directory
    kimina_client_path = Path(os.environ.get("KIMINA_CLIENT_PATH", ""))
    if kimina_client_path.is_dir() and str(kimina_client_path) not in sys.path:
        sys.path.insert(0, str(kimina_client_path))
    from kimina_client import AsyncKiminaClient, Snippet

from config import (
    LEAN_SERVER_HOST,
    LEAN_SERVER_MAX_WORKERS,
    LEAN_SERVER_PORT,
    LEAN_SERVER_TIMEOUT,
)


class KiminaLeanClient:
    BACKOFF = 3
    MAX_RETRIES = 10

    def __init__(self, host: str, port: int, timeout: int, max_workers: int):
        self._url = f"http://{host}:{port}"
        self._timeout = timeout
        self._max_workers = max_workers
        self._client = AsyncKiminaClient(api_url=self._url)

    @property
    def url(self) -> str:
        return self._url

    def collect_results(self, code: str, repl_response) -> dict[str, Any]:
        if repl_response.error is not None:
            return {
                "pass": False,
                "complete": False,
                "system_errors": repl_response.error,
                "time": repl_response.time,
                "verified_code": code,
                "errors": [],
                "warnings": [],
                "infos": [],
                "sorries": [],
                "tactics": [],
            }

        response = repl_response.response
        if response is None or "message" in response:
            error_msg = response.get("message", "Unknown error") if response else "No response"
            return {
                "pass": False,
                "complete": False,
                "system_errors": error_msg,
                "time": repl_response.time,
                "verified_code": code,
                "errors": [],
                "warnings": [],
                "infos": [],
                "sorries": [],
                "tactics": [],
            }

        result = {
            "sorries": response.get("sorries", []),
            "tactics": response.get("tactics", []),
            "errors": [m for m in response.get("messages", []) if m["severity"] == "error"],
            "warnings": [m for m in response.get("messages", []) if m["severity"] == "warning"],
            "infos": [m for m in response.get("messages", []) if m["severity"] == "info"],
            "system_errors": None,
            "verified_code": code,
            "time": repl_response.time,
        }
        result["pass"] = not result["errors"]
        result["complete"] = result["pass"] and not result["sorries"] and not any(
            "declaration uses 'sorry'" in warning["data"] or "failed" in warning["data"]
            for warning in result["warnings"]
        )
        return result

    async def _query_impl(
        self,
        code_list: list[str],
        is_clear_cache: bool = False,
        disable_process_bar: bool = False,
    ) -> list[dict[str, Any]]:
        snippets = [Snippet(id=str(i), code=code) for i, code in enumerate(code_list)]
        reuse_repl = not is_clear_cache
        check_response = await self._client.check(
            snips=snippets,
            timeout=self._timeout,
            reuse=reuse_repl,
            batch_size=1,
            max_workers=self._max_workers,
            show_progress=not disable_process_bar,
        )
        results = sorted(check_response.results, key=lambda x: int(x.id))
        return [self.collect_results(code, result) for code, result in zip(code_list, results)]

    async def query(self, **kwargs):
        backoff = self.BACKOFF
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await self._query_impl(**kwargs)
            except Exception as err:
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"Lean server query failed after {self.MAX_RETRIES} attempts: {err}"
                    ) from err
                print(
                    f"Error in {self.__class__.__name__}: {err}, "
                    f"retrying in {backoff}s ({attempt}/{self.MAX_RETRIES})..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)


def compile_lean_codes(lean_codes: List[str]) -> List[Dict[str, Any]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(lean_codes)
    code_list: List[str] = []
    index_map: List[int] = []
    for idx, code in enumerate(lean_codes):
        if isinstance(code, str) and code.strip():
            code_list.append(code)
            index_map.append(idx)
        else:
            results[idx] = {
                "pass": False,
                "complete": False,
                "system_errors": "empty_code",
                "time": 0,
                "verified_code": code or "",
                "errors": [],
                "warnings": [],
                "infos": [],
                "sorries": [],
                "tactics": [],
            }

    if not code_list:
        return [r or {} for r in results]

    async def _run() -> List[Dict[str, Any]]:
        client = KiminaLeanClient(
            host=LEAN_SERVER_HOST,
            port=LEAN_SERVER_PORT,
            timeout=LEAN_SERVER_TIMEOUT,
            max_workers=LEAN_SERVER_MAX_WORKERS,
        )
        return await client.query(code_list=code_list, is_clear_cache=False, disable_process_bar=False)

    compiled = asyncio.run(_run())
    for idx, res in zip(index_map, compiled):
        results[idx] = res
    return [r or {} for r in results]
