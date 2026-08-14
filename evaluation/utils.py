"""Shared utilities: jsonl IO, field extraction, model output parsing,
Lean server readiness check, and resume-completeness checks."""

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(paths: Iterable[str], benchmark_from_filename: bool = False) -> list[dict[str, Any]]:
    """Read jsonl files; benchmark_from_filename uses the file stem as benchmark_name."""
    rows = []
    for path in paths:
        dataset_name = Path(path).stem if benchmark_from_filename else Path(path).name
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if benchmark_from_filename:
                    item["benchmark_name"] = dataset_name
                elif not item.get("benchmark_name") or item.get("benchmark_name") == "unknown":
                    item["benchmark_name"] = dataset_name
                rows.append(item)
    return rows


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_nl(data: Dict[str, Any]) -> str:
    for key in ("nl", "problem", "statement", "text", "informal_statement"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("missing natural-language field: nl/problem/statement/text/informal_statement")


def extract_candidates_from_item(data: Dict[str, Any]) -> List[str]:
    for key in ("lean_candidates", "candidates", "autoformalizations"):
        value = data.get(key)
        if isinstance(value, list):
            return [str(v) for v in value if isinstance(v, str) and v.strip()]
    for key in ("lean_code", "formal_statement", "autoformalization", "prediction", "output"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def attach_predictions(
    examples: List[Dict[str, Any]],
    predictions: Optional[List[Dict[str, Any]]],
) -> List[List[str]]:
    if predictions is None:
        candidates = []
        for ex in examples:
            cand = extract_candidates_from_item(ex)
            candidates.append(cand)
        return candidates

    pred_map: Dict[Any, List[str]] = defaultdict(list)
    for idx, pred in enumerate(predictions):
        key = pred.get("problem_id", pred.get("id", pred.get("name", idx)))
        cand = extract_candidates_from_item(pred)
        for item in cand:
            pred_map[key].append(item)

    candidates = []
    for idx, ex in enumerate(examples):
        key = ex.get("problem_id", ex.get("id", ex.get("name", idx)))
        cand = pred_map.get(key, [])
        if not cand:
            cand = extract_candidates_from_item(ex)
        candidates.append(cand)
    return candidates


def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def extract_lean_code_from_inference(text: str, no_think: bool = False) -> Optional[str]:
    if not text:
        return None
    if no_think:
        return text.strip()
    # Truncate at the last </think>; the opening tag may live in the chat
    # template prefix, so a paired regex would not match.
    close_think = list(re.finditer(r"</think>", text, flags=re.I))
    if close_think:
        text = text[close_think[-1].end():].strip()
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    fenced = list(re.finditer(r"```([^\n`]*)\n(.*?)```", text, flags=re.S))
    if fenced:
        return fenced[-1].group(2).strip()

    open_fences = list(re.finditer(r"```([^\n`]*)\s*", text, flags=re.I))
    if open_fences:
        last_open = open_fences[-1]
        start = last_open.end()
        if start < len(text) and text[start] == "\n":
            start += 1
        end = text.find("```", start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    # No code fence: accept plain text if it looks like a Lean statement
    if re.search(r"\b(theorem|lemma|example|def|abbrev|instance|import)\b", text):
        return text.strip()
    return None


def ensure_mathlib_import(code: str) -> str:
    if not code or not code.strip():
        return code
    if re.search(r"^\s*import\s+", code, flags=re.M):
        return code
    return f"import Mathlib\n\n{code.lstrip()}"


def parse_judge_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    comments_match = re.search(r"<comments>(.*?)</comments>", text, flags=re.S | re.I)
    result_match = re.search(r"<result>(.*?)</result>", text, flags=re.S | re.I)
    if comments_match or result_match:
        comments = comments_match.group(1).strip() if comments_match else ""
        result = result_match.group(1).strip().lower() if result_match else ""
        return {"comments": comments, "result": result}
    block = _extract_json_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except Exception:
        return None


def judge_is_correct(parsed: Optional[Dict[str, Any]]) -> bool:
    if not parsed:
        return False
    if "result" in parsed:
        return str(parsed.get("result", "")).strip().lower() == "correct"
    value = str(parsed.get("is_assistant_correct", "")).strip().lower()
    return value == "correct"


def wait_for_lean_server(host: str, port: int, timeout: int = 120, check_interval: int = 2) -> bool:
    import socket

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        time.sleep(check_interval)
    return False


def check_infer_complete(output_path: str, num_samples: int, dataset_paths: List[str]) -> bool:
    if not Path(output_path).exists():
        return False

    dataset = load_jsonl(dataset_paths)
    if not dataset:
        return False

    rows = load_jsonl([output_path])
    results = {r.get("problem_id"): r for r in rows if isinstance(r, dict)}

    for i in range(len(dataset)):
        item = results.get(i)
        if not item:
            return False
        leans = item.get("lean_candidates") or []
        raws = item.get("raw_outputs") or []
        if len(leans) < num_samples or len(raws) < num_samples:
            return False
        for j in range(num_samples):
            if not str(leans[j]).strip() or not str(raws[j]).strip():
                return False
    return True


def check_judge_complete(output_path: str, num_samples: int, dataset_paths: List[str]) -> bool:
    if not Path(output_path).exists():
        return False

    dataset = load_jsonl(dataset_paths)
    if not dataset:
        return False

    rows = load_jsonl([output_path])
    results = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        pid = r.get("problem_id")
        sid = r.get("sample_id")
        if isinstance(pid, int) and isinstance(sid, int):
            results[(pid, sid)] = r

    expected = len(dataset) * num_samples
    if len(results) < expected:
        return False

    for pid in range(len(dataset)):
        for sid in range(num_samples):
            item = results.get((pid, sid))
            if not item:
                return False
            status = str(item.get("request_status", "")).strip().lower()
            if status not in {"success", "skipped"}:
                return False
    return True


if __name__ == "__main__":
    # Completeness-check CLI used by run.sh; exit 0 = complete, 1 = incomplete.
    import sys

    if len(sys.argv) < 5 or sys.argv[1] not in {"infer", "judge"}:
        print(
            "usage: python utils.py {infer|judge} <output_path> <num_samples> <dataset_path> [...]",
            file=sys.stderr,
        )
        sys.exit(2)
    _mode = sys.argv[1]
    _output_path = sys.argv[2]
    _num_samples = int(sys.argv[3])
    _dataset_paths = sys.argv[4:]
    _checker = check_infer_complete if _mode == "infer" else check_judge_complete
    sys.exit(0 if _checker(_output_path, _num_samples, _dataset_paths) else 1)
