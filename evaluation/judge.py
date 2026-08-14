"""Judge stage: compilation check, LLM semantic-consistency evaluation,
and Pass@k aggregation with table output."""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tqdm

import config
from config import LEAN_SERVER_HOST, LEAN_SERVER_PORT, SAVE_INTERVAL
from lean_compiler import compile_lean_codes
from llm_client import call_judge_api
from prompts import build_prompt
from utils import (
    attach_predictions,
    ensure_mathlib_import,
    extract_nl,
    judge_is_correct,
    load_jsonl,
    parse_judge_response,
    wait_for_lean_server,
    write_jsonl,
)


def compute_pass_at_k(is_correct: List[bool], k: int) -> float:
    n = len(is_correct)
    c = sum(1 for v in is_correct if v)
    if k <= 0 or n == 0:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def run_judge(
    output_path: str,
    dataset_paths: List[str],
    max_workers: int,
    num_samples: int,
    timeout: int,
    predictions_path: Optional[str],
    limit: Optional[int],
    judge_max_tokens: int = 16384,
) -> None:
    dataset = load_jsonl(dataset_paths, benchmark_from_filename=True)
    if limit is not None:
        dataset = dataset[:limit]

    existing_map: Dict[tuple[int, int], Dict[str, Any]] = {}
    output_file = Path(output_path)
    if output_file.exists():
        try:
            existing_rows = load_jsonl([output_path])
            for row in existing_rows:
                if isinstance(row, dict) and "problem_id" in row and "sample_id" in row:
                    key = (int(row["problem_id"]), int(row["sample_id"]))
                    existing_map[key] = row
            if existing_map:
                print(f"Found existing judge output, resuming: {output_path}")
        except Exception as e:
            print(f"[warn] Failed to read existing judge output, running from scratch: {e}")

    examples = []
    for idx, data in enumerate(dataset):
        example = dict(data)
        example["problem_id"] = idx
        example["nl"] = extract_nl(data)
        examples.append(example)

    predictions = None
    if predictions_path:
        predictions = load_jsonl([predictions_path])

    candidates_list = attach_predictions(examples, predictions)

    entries = []
    missing_candidates = 0
    request_id = 0
    for ex, candidates in zip(examples, candidates_list):
        if not candidates:
            missing_candidates += 1
        if len(candidates) < num_samples:
            candidates = candidates + [""] * (num_samples - len(candidates))
        candidates = candidates[:num_samples]
        for sample_id, lean_code in enumerate(candidates):
            lean_code = ensure_mathlib_import(lean_code)
            entries.append(
                {
                    "problem_id": ex["problem_id"],
                    "sample_id": sample_id,
                    "name": ex.get("name"),
                    "benchmark_name": ex.get("benchmark_name", "unknown"),
                    "lean_code": lean_code,
                    "nl": ex["nl"],
                    "request_id": request_id,
                }
            )
            request_id += 1

    if missing_candidates > 0:
        print(f"[warn] {missing_candidates} samples have no Lean candidates")

    compile_cache_path = str(Path(output_path).with_suffix("")) + ".compile.jsonl"
    compile_results: Optional[List[Dict[str, Any]]] = None
    if Path(compile_cache_path).exists():
        try:
            cached_rows = load_jsonl([compile_cache_path])
            cached_map = {
                (int(row["problem_id"]), int(row["sample_id"])): row.get("compile_result")
                for row in cached_rows
                if isinstance(row, dict) and "problem_id" in row and "sample_id" in row
            }
            resolved_cache = []
            for entry in entries:
                cached_result = cached_map.get((entry["problem_id"], entry["sample_id"]))
                if (
                    not isinstance(cached_result, dict)
                    or cached_result.get("verified_code") != entry["lean_code"]
                ):
                    break
                resolved_cache.append(cached_result)
            if len(resolved_cache) == len(entries):
                compile_results = resolved_cache
                print(f"Complete compilation cache found, skipping compilation: {compile_cache_path}")
            else:
                print("[warn] Compilation cache incomplete or mismatched, recompiling")
        except Exception as e:
            print(f"[warn] Failed to read compilation cache, recompiling: {e}")

    if compile_results is None:
        print("Starting compilation check (Kimina Lean Server)...")
        if not wait_for_lean_server(LEAN_SERVER_HOST, LEAN_SERVER_PORT, timeout=120, check_interval=2):
            print(
                f"[error] Cannot connect to Lean server ({LEAN_SERVER_HOST}:{LEAN_SERVER_PORT}), aborting. "
                "Make sure the server is running and set LEAN_SERVER_HOST / LEAN_SERVER_PORT if needed"
            )
            return
        compile_results = compile_lean_codes([e["lean_code"] for e in entries])
        write_jsonl(
            compile_cache_path,
            [
                {
                    "problem_id": entry["problem_id"],
                    "sample_id": entry["sample_id"],
                    "compile_result": compile_result,
                }
                for entry, compile_result in zip(entries, compile_results)
            ],
        )
        print(f"Compilation results saved: {compile_cache_path}")

    results = []
    judge_entries = []
    compile_pass_by_bench = defaultdict(list)
    for entry, compile_result in zip(entries, compile_results):
        compile_pass = bool(compile_result.get("pass"))
        entry["compile_pass"] = compile_pass
        entry["compile_result"] = compile_result
        compile_pass_by_bench[entry["benchmark_name"]].append(compile_pass)
        key = (entry["problem_id"], entry["sample_id"])
        existing = existing_map.get(key)
        if compile_pass:
            # Reuse a historical record only if it succeeded with non-empty judge
            # output; "success but empty output" is treated as not judged and re-sent.
            if (
                existing
                and existing.get("request_status") == "success"
                and (existing.get("judge_raw") or "").strip()
            ):
                existing["compile_pass"] = True
                existing["compile_result"] = compile_result
                results.append(existing)
            else:
                entry["prompt"] = build_prompt(entry["nl"], entry["lean_code"])
                judge_entries.append(entry)
        else:
            results.append(
                {
                    "problem_id": entry["problem_id"],
                    "sample_id": entry["sample_id"],
                    "name": entry["name"],
                    "benchmark_name": entry["benchmark_name"],
                    "nl": entry["nl"],
                    "lean_code": entry["lean_code"],
                    "compile_pass": False,
                    "compile_result": compile_result,
                    "judge_raw": "",
                    "judge_reasoning": "",
                    "judge_json": None,
                    "judge_correct": False,
                    "judge_skipped": True,
                    "request_status": "skipped",
                    "error": None,
                }
            )

    total_requests = len(judge_entries)
    if total_requests > 0:
        print(f"Judge via remote API backend: {config.API_BASE_URL} (model={config.API_MODEL})")
        print(f"Starting judge evaluation (total requests={total_requests}, max_workers={max_workers})...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    call_judge_api,
                    pd["prompt"],
                    pd["request_id"],
                    timeout,
                    judge_max_tokens,
                ): pd
                for pd in judge_entries
            }
            with tqdm.tqdm(total=total_requests, desc="Judge") as pbar:
                completed = 0
                for future in as_completed(futures):
                    prompt_data = futures[future]
                    try:
                        response = future.result(timeout=timeout + 30)
                    except Exception as e:
                        response = {"status": "failed", "output": "", "error": str(e)}
                    raw = response.get("output", "")
                    reasoning = response.get("reasoning_content", "") or ""
                    parsed = parse_judge_response(raw)
                    correct = judge_is_correct(parsed)
                    results.append(
                        {
                            "problem_id": prompt_data["problem_id"],
                            "sample_id": prompt_data["sample_id"],
                            "name": prompt_data["name"],
                            "benchmark_name": prompt_data["benchmark_name"],
                            "nl": prompt_data["nl"],
                            "lean_code": prompt_data["lean_code"],
                            "compile_pass": prompt_data["compile_pass"],
                            "compile_result": prompt_data["compile_result"],
                            "judge_raw": raw,
                            "judge_reasoning": reasoning,
                            "judge_json": parsed,
                            "judge_correct": correct,
                            "judge_skipped": False,
                            "request_status": response.get("status", "failed"),
                            "error": response.get("error"),
                        }
                    )
                    pbar.update(1)
                    completed += 1
                    if completed % SAVE_INTERVAL == 0:
                        write_jsonl(output_path, results)

    write_jsonl(output_path, results)

    grouped = defaultdict(list)
    compile_grouped = defaultdict(list)
    for item in results:
        grouped[item["problem_id"]].append(item["judge_correct"])
        compile_grouped[item["problem_id"]].append(item["compile_pass"])

    pass_ks = sorted({1, num_samples})
    benchmark_pass = {k: defaultdict(list) for k in pass_ks}
    compile_pass_by_bench_k = {k: defaultdict(list) for k in pass_ks}
    for ex in examples:
        correct_list = grouped.get(ex["problem_id"], [])
        compile_list = compile_grouped.get(ex["problem_id"], [])
        for k in pass_ks:
            pass_k = compute_pass_at_k(correct_list, k)
            benchmark_pass[k][ex.get("benchmark_name", "unknown")].append(pass_k)
            compile_pass = compute_pass_at_k(compile_list, k)
            compile_pass_by_bench_k[k][ex.get("benchmark_name", "unknown")].append(compile_pass)

    def macro_avg(by_bench: Dict[str, List[float]]) -> float:
        """Macro average: mean within each benchmark, then mean across benchmarks."""
        bench_means = [float(np.mean(vals)) for vals in by_bench.values() if vals]
        return float(np.mean(bench_means)) if bench_means else 0.0

    # Benchmark column order follows the dataset file order
    bench_order = []
    for p in dataset_paths:
        stem = Path(p).stem
        if stem not in bench_order:
            bench_order.append(stem)

    def _fmt_pct(v: float) -> str:
        return f"{v * 100:.2f}"

    def _print_table(compile_by_bench, judge_by_bench,
                     compile_overall, judge_overall) -> None:
        """Print a Pass@k table: AVG (macro) + one column per benchmark,
        each with SC / CC sub-columns."""
        all_benches_in_data = set(compile_by_bench.keys()) | set(judge_by_bench.keys())
        ordered = [b for b in bench_order if b in all_benches_in_data]
        for b in sorted(all_benches_in_data):
            if b not in ordered:
                ordered.append(b)

        header_names = ["AVG"] + ordered
        col_count = len(header_names)

        sc_vals = []
        cc_vals = []
        sc_vals.append(_fmt_pct(compile_overall))
        cc_vals.append(_fmt_pct(judge_overall))
        for bench in ordered:
            sv = compile_by_bench.get(bench)
            jv = judge_by_bench.get(bench)
            sc_vals.append(_fmt_pct(float(np.mean(sv))) if sv else "-")
            cc_vals.append(_fmt_pct(float(np.mean(jv))) if jv else "-")

        col_widths = []
        for i in range(col_count):
            w = max(
                len(header_names[i]),
                len("SC") + 2 + len("CC"),  # "SC  CC"
                len(sc_vals[i]) + 2 + len(cc_vals[i]),
            )
            w = max(w, 14)  # minimum column width
            col_widths.append(w)

        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        print(sep)
        row = "|"
        for i, name in enumerate(header_names):
            row += f" {name:^{col_widths[i]}} |"
        print(row)

        print(sep)

        row = "|"
        for i in range(col_count):
            half = col_widths[i] // 2
            cell = f"{'SC':^{half}}{'CC':^{col_widths[i] - half}}"
            row += f" {cell} |"
        print(row)

        print(sep)

        row = "|"
        for i in range(col_count):
            half = col_widths[i] // 2
            cell = f"{sc_vals[i]:^{half}}{cc_vals[i]:^{col_widths[i] - half}}"
            row += f" {cell} |"
        print(row)

        print(sep)

    # Print only the largest k (Pass@num_samples); all k values are still saved
    print_k = max(pass_ks)
    # Macro AVG avoids larger benchmarks dominating the overall metric
    print(f"\n{'=' * 20} Pass@{print_k} results {'=' * 20}")
    _print_table(
        compile_by_bench={bench: vals for bench, vals in compile_pass_by_bench_k[print_k].items()},
        judge_by_bench={bench: vals for bench, vals in benchmark_pass[print_k].items()},
        compile_overall=macro_avg(compile_pass_by_bench_k[print_k]),
        judge_overall=macro_avg(benchmark_pass[print_k]),
    )

    summary = {
        "compile_pass": {
            "overall_macro": macro_avg(compile_pass_by_bench),
            "by_benchmark": {bench: float(np.mean(vals)) for bench, vals in compile_pass_by_bench.items()},
        },
        "compile_pass_at_k": {
            "k": pass_ks,
            "overall_macro": {k: macro_avg(compile_pass_by_bench_k[k]) for k in pass_ks},
            "by_benchmark": {
                k: {bench: float(np.mean(vals)) for bench, vals in compile_pass_by_bench_k[k].items()}
                for k in pass_ks
            },
        },
        "pass_at_k": {
            "k": pass_ks,
            "overall_macro": {k: macro_avg(benchmark_pass[k]) for k in pass_ks},
            "by_benchmark": {
                k: {bench: float(np.mean(vals)) for bench, vals in benchmark_pass[k].items()}
                for k in pass_ks
            },
        },
    }
    summary_path = str(Path(output_path).with_suffix("")) + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
