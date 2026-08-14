"""Inference stage: generate Lean candidates with the model under evaluation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import tqdm

import config
from config import SAVE_INTERVAL
from llm_client import call_eval_api
from prompts import build_inference_prompt
from utils import (
    ensure_mathlib_import,
    extract_lean_code_from_inference,
    extract_nl,
    load_jsonl,
    write_jsonl,
)


def run_inference(
    output_path: str,
    dataset_paths: List[str],
    max_workers: int,
    num_samples: int,
    timeout: int,
    eval_max_tokens: int,
    prompt_template: str,
    limit: Optional[int],
    no_think: bool,
) -> None:
    print(f"Inference via remote API backend: {config.API_BASE_URL} (model={config.API_MODEL})")

    dataset = load_jsonl(dataset_paths, benchmark_from_filename=True)
    if limit is not None:
        dataset = dataset[:limit]

    existing_map: Dict[int, Dict[str, Any]] = {}
    output_file = Path(output_path)
    if output_file.exists():
        try:
            existing_rows = load_jsonl([output_path])
            for row in existing_rows:
                if isinstance(row, dict) and "problem_id" in row:
                    existing_map[int(row["problem_id"])] = row
            if existing_map:
                print(f"Found existing inference output, resuming: {output_path}")
        except Exception as e:
            print(f"[warn] Failed to read existing inference output, running from scratch: {e}")

    examples = []
    for idx, data in enumerate(dataset):
        example = dict(data)
        example["problem_id"] = idx
        example["nl"] = extract_nl(data)
        examples.append(example)

    results_map: Dict[int, Dict[str, Any]] = {}
    for ex in examples:
        existing = existing_map.get(ex["problem_id"])
        existing_candidates = []
        existing_raw = []
        existing_reasonings = []
        if existing:
            existing_candidates = existing.get("lean_candidates", []) or []
            existing_raw = existing.get("raw_outputs", []) or []
            existing_reasonings = existing.get("raw_reasonings", []) or []
        results_map[ex["problem_id"]] = {
            "problem_id": ex["problem_id"],
            "name": ex.get("name"),
            "benchmark_name": ex.get("benchmark_name", "unknown"),
            "nl": ex["nl"],
            "lean_candidates": (existing_candidates + [""] * num_samples)[:num_samples],
            "raw_outputs": (existing_raw + [""] * num_samples)[:num_samples],
            "raw_reasonings": (existing_reasonings + [""] * num_samples)[:num_samples],
        }

    entries = []
    request_id = 0
    for ex in examples:
        prompt = build_inference_prompt(ex["nl"], prompt_template)
        for sample_id in range(num_samples):
            existing_item = results_map[ex["problem_id"]]
            if (
                existing_item["lean_candidates"][sample_id]
                and existing_item["raw_outputs"][sample_id]
            ):
                continue
            entries.append(
                {
                    "problem_id": ex["problem_id"],
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "request_id": request_id,
                }
            )
            request_id += 1

    total_requests = len(entries)
    print(f"Starting inference (total requests={total_requests}, max_workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_eval_api,
                entry["prompt"],
                entry["request_id"],
                timeout,
                eval_max_tokens,
            ): entry
            for entry in entries
        }
        with tqdm.tqdm(total=total_requests, desc="Inference") as pbar:
            completed = 0
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    response = future.result(timeout=timeout + 30)
                except Exception as e:
                    response = {"status": "failed", "output": "", "error": str(e)}
                raw = response.get("output", "")
                reasoning = response.get("reasoning_content", "") or ""
                lean_code = ensure_mathlib_import(extract_lean_code_from_inference(raw, no_think=no_think) or "")
                result = results_map[entry["problem_id"]]
                result["lean_candidates"][entry["sample_id"]] = lean_code
                result["raw_outputs"][entry["sample_id"]] = raw
                result["raw_reasonings"][entry["sample_id"]] = reasoning
                pbar.update(1)
                completed += 1
                if completed % SAVE_INTERVAL == 0:
                    write_jsonl(output_path, [results_map[ex["problem_id"]] for ex in examples])

    write_jsonl(output_path, [results_map[ex["problem_id"]] for ex in examples])
