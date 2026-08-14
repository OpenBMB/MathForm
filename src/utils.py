"""Shared utilities: logging, file IO, text processing."""

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Logger:
    
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self._lock = threading.Lock()
    
    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        with self._lock:
            print(line)
            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
    
    def set_log_file(self, log_file: Path) -> None:
        with self._lock:
            self.log_file = log_file


class FileIO:
    
    def __init__(self):
        self._lock = threading.Lock()
    
    def load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def append_jsonl(self, record: Dict[str, Any], path: Path) -> None:
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def write_jsonl(self, records: List[Dict[str, Any]], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def save_checkpoint(self, states: List[Dict[str, Any]], path: Path) -> None:
        self.write_jsonl(states, path)
    
    def load_checkpoint(self, path: Path) -> Optional[List[Dict[str, Any]]]:
        if path.exists():
            return self.load_jsonl(path)
        return None


class TextProcessor:
    
    @staticmethod
    def collapse_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
    
    @staticmethod
    def truncate_text(text: str, max_len: int = 600) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len - 3].rstrip() + "..."
    
    @staticmethod
    def strip_after_last_think(text: str) -> str:
        """Drop everything before the last </think> tag."""
        tag = "</think>"
        idx = text.rfind(tag)
        if idx != -1:
            return text[idx + len(tag):].strip()
        return text.strip()
    
    @staticmethod
    def extract_lean_code(text: str) -> str:
        if not text:
            return ""
        cleaned = TextProcessor.strip_after_last_think(text)
        patterns = [
            r"```lean4\s*(.*?)```",
            r"```lean\s*(.*?)```",
            r"```\s*(.*?)```",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, cleaned, re.DOTALL | re.IGNORECASE))
            if matches:
                return matches[-1].group(1).strip()
        return cleaned.strip()
    
    @staticmethod
    def contains_sorry(lean_code: str) -> bool:
        return bool(re.search(r"\bsorry\b", lean_code))
    
    @staticmethod
    def has_proof_steps(lean_code: str) -> bool:
        if not lean_code:
            return False
        if re.search(r"\b(sorry|admit)\b", lean_code):
            return False
        proof_tokens = [
            "exact", "simp", "simp?", "aesop", "linarith", "nlinarith",
            "ring", "omega", "tauto", "by_contra", "intro", "intros",
            "cases", "case", "induction", "funext", "rw", "rewrite",
            "calc", "have", "refine", "apply", "constructor", "obtain",
            "simp_all", "simp_rw", "rfl", "decide", "trivial",
        ]
        token_pattern = r"\b(" + "|".join(re.escape(tok) for tok in proof_tokens) + r")\b"
        return re.search(token_pattern, lean_code) is not None
    
    @staticmethod
    def parse_query_list(text: Optional[str]) -> List[str]:
        if not text:
            return []
        raw = text.strip()
        queries_match = re.search(r"<queries>(.*?)</queries>", raw, re.DOTALL | re.IGNORECASE)
        if queries_match:
            raw = queries_match.group(1).strip()
        fenced = list(re.finditer(r"```(?:\w+)?\s*(.*?)```", raw, re.DOTALL))
        if fenced:
            raw = fenced[-1].group(1).strip()
        else:
            return []
        if "\n" not in raw and ("," in raw or ";" in raw or "；" in raw or "，" in raw):
            raw = re.sub(r"[；，;]", "\n", raw)
        queries: List[str] = []
        for line in raw.splitlines():
            cleaned = re.sub(r"^[\s\-\*\d\.\)\]]+", "", line).strip()
            cleaned = re.sub(r"^query\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip().strip("\"'`")
            if cleaned:
                queries.append(cleaned)
        return TextProcessor.dedup_queries(queries)
    
    @staticmethod
    def dedup_queries(queries: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for q in queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q.strip())
        return deduped


def get_statement(item: Dict[str, Any]) -> str:
    return (
        item.get("informal_theorem_qa")
        or item.get("statement")
        or item.get("nl_statement")
        or item.get("informal_statement")
        or item.get("natural_language_statement")
        or item.get("query")
        or ""
    )
