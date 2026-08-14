"""Client for OpenAI-compatible chat/completions APIs."""

import threading
import time
from typing import Any, Dict, List, Optional

import requests

from config import APIConfig
from utils import Logger


class APIClient:
    """API client.

    call_api returns {"content", "reasoning_content"}; the base/retrieval planner/judge
    helpers bind their respective URL, model name and sampling parameters.
    """

    def __init__(self, config: APIConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or Logger()
        self._session_local = threading.local()
        self._url_lock = threading.Lock()
        self._url_counters = {
            "base": 0,
            "retrieval_planner": 0,
            "judge": 0,
        }

    def _get_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            session.headers.update(headers)
            self._session_local.session = session
        return session

    def _extract_message(self, result: Dict[str, Any]) -> Optional[Dict[str, str]]:
        message = (
            result.get("choices", [{}])[0].get("message", {})
            if isinstance(result, dict)
            else {}
        )
        if not isinstance(message, dict):
            return None
        return {
            "content": message.get("content", "") or "",
            "reasoning_content": message.get("reasoning_content", "") or "",
        }

    def _pick_url(self, urls: List[str], key: str) -> str:
        if not urls:
            raise ValueError(f"Empty URL list for {key} model")
        if len(urls) == 1:
            return urls[0]
        with self._url_lock:
            idx = self._url_counters.get(key, 0)
            url = urls[idx % len(urls)]
            self._url_counters[key] = idx + 1
            return url

    def call_api(
        self,
        url: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 16384,
        max_retries: int = 3,
        model_name: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Return {"content", "reasoning_content"}, or None on failure."""
        session = self._get_session()
        for attempt in range(max_retries):
            try:
                payload: Dict[str, Any] = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if model_name:
                    payload["model"] = model_name
                if reasoning_effort:
                    payload["reasoning_effort"] = reasoning_effort
                resp = session.post(
                    url,
                    json=payload,
                    timeout=self.config.request_timeout,
                )
                if resp.status_code == 200:
                    return self._extract_message(resp.json())
                if resp.status_code == 429:
                    time.sleep(1 + attempt)
            except Exception as exc:
                if attempt == max_retries - 1:
                    self.logger.log(f"API call failed: {exc}")
                time.sleep(0.5 * (attempt + 1))
        return None

    def call_base_model(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 16384,
    ) -> Optional[Dict[str, str]]:
        return self.call_api(
            url=self._pick_url(self.config.base_model_urls, "base"),
            messages=messages,
            temperature=temperature or 0.7,
            max_tokens=max_tokens,
            model_name=self.config.base_model_name,
            reasoning_effort="high",
        )

    def call_retrieval_planner(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Optional[str]:
        message = self.call_api(
            url=self._pick_url(self.config.retrieval_planner_urls, "retrieval_planner"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=self.config.retrieval_planner_name,
        )
        return message["content"] if message else None

    def call_judge_model(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 16384,
    ) -> Optional[str]:
        message = self.call_api(
            url=self._pick_url(self.config.judge_model_urls, "judge"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=self.config.judge_model_name,
        )
        return message["content"] if message else None
