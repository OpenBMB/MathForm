"""Lean compilation check via the Kimina Lean Server."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

from config import APIConfig
from utils import Logger


LEAN_SERVER_TIMEOUT = 120
LEAN_SERVER_MAX_WORKERS = 24


try:
    from kimina_client import AsyncKiminaClient, Snippet
except ImportError:
    # If kimina_client is not installed, point KIMINA_CLIENT_PATH to its source directory
    kimina_client_path = Path(os.environ.get("KIMINA_CLIENT_PATH", ""))
    if kimina_client_path.is_dir() and str(kimina_client_path) not in sys.path:
        sys.path.insert(0, str(kimina_client_path))
    try:
        from kimina_client import AsyncKiminaClient, Snippet
    except ImportError:
        AsyncKiminaClient = None
        Snippet = None


def _parse_lean_server_url(url: str) -> Tuple[str, int]:
    if "://" not in url:
        url = f"http://{url}"
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return host, port


class KiminaLeanClient:
    
    BACKOFF = 3
    
    def __init__(
        self, 
        host: str, 
        port: int, 
        timeout: int = LEAN_SERVER_TIMEOUT, 
        max_workers: int = LEAN_SERVER_MAX_WORKERS
    ):
        if AsyncKiminaClient is None:
            raise ImportError(
                "kimina_client not found. Please install it or add kimina-lean-server/client to your path."
            )
        
        self._url = f"http://{host}:{port}"
        self._timeout = timeout
        self._max_workers = max_workers
        self._client = AsyncKiminaClient(api_url=self._url)
    
    @property
    def url(self) -> str:
        return self._url
    
    def collect_results(self, code: str, repl_response) -> dict[str, Any]:
        """Convert a ReplResponse into a result dict (pass/complete/errors/...)."""
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
                "tactics": []
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
                "tactics": []
            }
        
        result = {
            "sorries": response.get('sorries', []),
            "tactics": response.get('tactics', []),
            "errors": [m for m in response.get('messages', []) if m['severity'] == 'error'],
            "warnings": [m for m in response.get('messages', []) if m['severity'] == 'warning'],
            "infos": [m for m in response.get('messages', []) if m['severity'] == 'info'],
            "system_errors": None,
            "verified_code": code,
            "time": repl_response.time
        }
        result['pass'] = not result['errors']
        result['complete'] = (
            result['pass'] 
            and not result['sorries'] 
            and not any(
                "declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] 
                for warning in result['warnings']
            )
        )
        return result
    
    async def _query_impl(
        self, 
        code_list: list[str], 
        is_clear_cache: bool = False, 
        disable_progress_bar: bool = False
    ) -> list[dict[str, Any]]:
        snippets = [Snippet(id=str(i), code=code) for i, code in enumerate(code_list)]
        
        # is_clear_cache=True means do not reuse the REPL (reuse=False)
        reuse_repl = not is_clear_cache
        
        check_response = await self._client.check(
            snips=snippets,
            timeout=self._timeout,
            reuse=reuse_repl,
            batch_size=1,
            max_workers=self._max_workers,
            show_progress=not disable_progress_bar
        )
        
        results = sorted(check_response.results, key=lambda x: int(x.id))
        return [self.collect_results(code, result) for code, result in zip(code_list, results)]
    
    async def query(self, **kwargs) -> list[dict[str, Any]]:
        curr_backoff = self.BACKOFF
        while True:
            try:
                return await self._query_impl(**kwargs)
            except Exception as err:
                print(f"Error in KiminaLeanClient: {err}, waiting {curr_backoff} seconds and then retrying...")
                await asyncio.sleep(curr_backoff)
                continue


_lean_client: KiminaLeanClient | None = None
_lean_client_config: Tuple[str, int] | None = None


def _get_lean_client(host: str, port: int) -> KiminaLeanClient:
    global _lean_client, _lean_client_config
    
    new_config = (host, port)
    if _lean_client is None or _lean_client_config != new_config:
        _lean_client = KiminaLeanClient(host=host, port=port)
        _lean_client_config = new_config
    return _lean_client


async def _async_batch_verify(
    code_list: list[str], 
    host: str,
    port: int,
    disable_progress_bar: bool = False
) -> list[dict[str, Any]]:
    client = _get_lean_client(host, port)
    return await client.query(code_list=code_list, is_clear_cache=False, disable_progress_bar=disable_progress_bar)


class LeanCompiler:
    
    def __init__(
        self,
        api_config: APIConfig,
        logger: Optional[Logger] = None,
    ):
        self.api_config = api_config
        self.logger = logger or Logger()
        self._host, self._port = _parse_lean_server_url(api_config.lean_server_url)
        self.logger.log(f"Initialized Lean compiler for {api_config.lean_server_url}")
    
    def verify_batch(self, codes: List[str], batch_size: int = 10) -> List[Tuple[bool, str]]:
        if not codes:
            return []
        
        if AsyncKiminaClient is None:
            return [(False, "Lean client not available")] * len(codes)
        
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(
                _async_batch_verify(
                    codes, 
                    host=self._host, 
                    port=self._port,
                    disable_progress_bar=False
                )
            )
        finally:
            loop.close()
        
        output: List[Tuple[bool, str]] = []
        for code, result in zip(codes, results):
            if not code:
                output.append((False, "Empty code"))
                continue
            if result.get("pass"):
                output.append((True, ""))
                continue
            if result.get("system_errors"):
                output.append((False, result["system_errors"]))
                continue
            errors = result.get("errors", [])
            if errors:
                output.append((False, "; ".join([str(e.get("data", "")) for e in errors[:3]])))
            else:
                output.append((False, "Compilation failed"))
        return output
    
    def verify_compilation(self, code: str) -> Tuple[bool, str]:
        if not code:
            return False, "Empty code"
        results = self.verify_batch([code])
        return results[0] if results else (False, "Verification failed")
