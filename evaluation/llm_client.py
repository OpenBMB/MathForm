"""LLM client: calls to a remote OpenAI-compatible API with streaming and retries."""

import json
import time
from typing import Any, Dict, List, Optional

import requests

import config


def _resolve_endpoint() -> tuple[str, Optional[Dict[str, str]], str]:
    """Return (api_url, headers, request_model)."""
    base_url = config.API_BASE_URL.rstrip("/")
    api_url = (
        f"{base_url}/chat/completions"
        if base_url.endswith("/v1")
        else f"{base_url}/v1/chat/completions"
    )
    headers = {"Authorization": f"Bearer {config.API_KEY}"} if config.API_KEY else None
    return api_url, headers, config.API_MODEL


class EmptyContentError(Exception):
    """Judge returned empty content (typically the reasoning looped until
    max_tokens, finish_reason=length). Retried with a separate budget so the
    sample is not silently counted as incorrect."""

    def __init__(self, finish_reason: Optional[str] = None, reasoning_content: str = ""):
        self.finish_reason = finish_reason
        self.reasoning_content = reasoning_content
        super().__init__(f"empty content (finish_reason={finish_reason})")


def _stream_response(
    api_url: str,
    payload: Dict[str, Any],
    timeout: int,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Read a streaming (SSE) /v1/chat/completions response into full text.

    The (connect, read) timeout tuple is the key: the read timeout only
    bounds the gap between consecutive chunks, not the total response time,
    so long generations are not falsely treated as timeouts as long as
    tokens keep arriving. The return structure matches the non-streaming path.
    """
    connect_timeout = min(30, timeout) if timeout and timeout > 0 else 30
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    finish_reason: Optional[str] = None
    with requests.post(
        api_url,
        json=payload,
        stream=True,
        timeout=(connect_timeout, timeout),
        headers=headers,
    ) as response:
        response.raise_for_status()
        # Decode per line: SSE is newline-delimited, and line boundaries never
        # split multi-byte UTF-8 characters
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if not line:
                continue
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
            reasoning_piece = delta.get("reasoning_content")
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
    return {
        "output": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "finish_reason": finish_reason,
    }


def call_chat_api(
    prompt: str,
    request_id: int,
    temperature: float,
    max_tokens: int,
    timeout: int,
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    stream: bool = True,
    retry_on_empty: bool = False,
    max_empty_retries: int = 50,
) -> Dict[str, Any]:
    api_url, headers, request_model = _resolve_endpoint()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": request_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "n": 1,
        "stream": stream,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    max_retries = 50
    # Empty content (judge looping until max_tokens) uses a separate retry
    # budget: it is model degeneration rather than a service error, and
    # resampling escapes the loop without consuming connection-error retries.
    attempt = 0
    empty_attempts = 0
    while True:
        try:
            if stream:
                streamed = _stream_response(api_url, payload, timeout, headers=headers)
                output = streamed["output"]
                reasoning_content = streamed["reasoning_content"]
                finish_reason = streamed.get("finish_reason")
            else:
                response = requests.post(
                    api_url,
                    json=payload,
                    timeout=timeout,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()
                choice = result["choices"][0]
                message = choice["message"]
                output = message.get("content") or ""
                reasoning_content = message.get("reasoning_content") or ""
                finish_reason = choice.get("finish_reason")
            if retry_on_empty and not (output or "").strip():
                raise EmptyContentError(finish_reason, reasoning_content)
            return {
                "request_id": request_id,
                "status": "success",
                "output": output,
                "reasoning_content": reasoning_content,
                "finish_reason": finish_reason,
            }
        except EmptyContentError as e:
            empty_attempts += 1
            if empty_attempts <= max_empty_retries:
                print(
                    f"[warn] Request {request_id} judge returned empty content (finish_reason={e.finish_reason}), "
                    f"retrying {empty_attempts}/{max_empty_retries} ..."
                )
                time.sleep(1)
                continue
            print(
                f"[fail] Request {request_id} returned empty content {max_empty_retries} times in a row, "
                f"marking as failed (finish_reason={e.finish_reason})"
            )
            return {
                "request_id": request_id,
                "status": "failed",
                "error": f"empty_content_after_{max_empty_retries}_retries (finish_reason={e.finish_reason})",
                "output": "",
                "reasoning_content": e.reasoning_content,
                "finish_reason": e.finish_reason,
            }
        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                wait_time = min(1 * (2 ** (attempt - 1)), 90)
                print(
                    f"[warn] Request {request_id} failed (attempt {attempt}/{max_retries}): {str(e)[:200]}, retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                print(f"[fail] Request {request_id} failed after {max_retries} attempts: {str(e)[:200]}")
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "error": str(e),
                    "output": "",
                    "reasoning_content": "",
                    "finish_reason": None,
                }


def call_eval_api(
    prompt: str,
    request_id: int,
    timeout: int,
    max_tokens: int,
) -> Dict[str, Any]:
    return call_chat_api(
        prompt,
        request_id,
        0.6,
        max_tokens,
        timeout,
    )


def call_judge_api(
    prompt: str,
    request_id: int,
    timeout: int,
    max_tokens: int = 16384,
) -> Dict[str, Any]:
    return call_chat_api(
        prompt,
        request_id,
        0.1,
        max_tokens,
        timeout,
        system_prompt=None,
        reasoning_effort="high",
        stream=True,
        retry_on_empty=True,
    )
