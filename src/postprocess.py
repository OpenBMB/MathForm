#!/usr/bin/env python3
"""Post-processing for pipeline outputs: normalize Lean code and filter samples.

Usage:
    python postprocess.py normalize --input success.jsonl --output normalized.jsonl
    python postprocess.py filter --input normalized.jsonl --output filtered.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# normalize: strip comments, unify sorry, strip proofs, rename theorems
# ---------------------------------------------------------------------------


def remove_comments(code: str) -> str:
    # Remove nested block comments /- ... -/ by repeatedly deleting innermost pairs
    while True:
        new_code = re.sub(r'/-[\s\S]*?-/', '', code)
        if len(new_code) == len(code):
            break
        code = new_code
    code = re.sub(r'--.*$', '', code, flags=re.MULTILINE)
    return code


def find_main_assignment(text_after_name: str) -> int:
    """Index of the top-level ':=' (skipping brackets), or -1 if not found."""
    depth = 0
    i = 0
    n = len(text_after_name)
    while i < n:
        char = text_after_name[i]
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ':' and i + 1 < n and text_after_name[i + 1] == '=':
            if depth == 0:
                return i
            i += 1
        i += 1
    return -1


def strip_theorem_proofs(code: str) -> str:
    """Replace theorem/lemma/example proofs with `:= by sorry`."""
    decl_start_re = re.compile(r'^\s*(theorem|lemma|example)\b', re.MULTILINE)
    next_top_re = re.compile(
        r'^\s*(?:theorem|lemma|example|def|abbrev|structure|inductive|class|instance|axiom|macro|'
        r'section|namespace|end|open|variable|variables|set_option|attribute|notation)\b',
        re.MULTILINE,
    )

    pieces = []
    last_idx = 0
    matches = list(decl_start_re.finditer(code))
    for match in matches:
        start = match.start()
        if start < last_idx:
            continue

        next_top = next_top_re.search(code, match.end())
        end = next_top.start() if next_top else len(code)

        decl = code[start:end]
        assign_idx = find_main_assignment(decl)
        if assign_idx != -1:
            decl = decl[:assign_idx].rstrip() + " := by sorry\n"

        pieces.append(code[last_idx:start])
        pieces.append(decl)
        last_idx = end

    pieces.append(code[last_idx:])
    return "".join(pieces)


def normalize_lean_code(code: str) -> str:
    code = remove_comments(code)

    # Normalize ":= sorry" variants to ":= by sorry"
    code = re.sub(r':=\s*(?!by\s+sorry)sorry\b', ':= by sorry', code, flags=re.DOTALL)

    code = strip_theorem_proofs(code)

    # Rename the last theorem to my_favorite_theorem
    matches = list(re.finditer(r'^\s*theorem\s+(\S+)', code, re.MULTILINE))
    if matches:
        last_match = matches[-1]
        code = code[:last_match.start(1)] + "my_favorite_theorem" + code[last_match.end(1):]

    code = re.sub(r':=\s+by\s+sorry\b', ':= by sorry', code, flags=re.DOTALL)

    # Normalize whitespace: strip trailing spaces, collapse 3+ newlines
    lines = [line.rstrip() for line in code.split('\n')]
    code = '\n'.join(lines)
    code = re.sub(r'\n{3,}', '\n\n', code)

    return code.strip()


def run_normalize(input_file: str, output_file: str) -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {input_path} -> {output_path} ...")
    count = 0

    with input_path.open('r', encoding='utf-8') as fin, \
         output_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print("Warning: Skipped invalid JSON line")
                continue
            if data.get('lean_code'):
                original_code = data['lean_code']
                data['lean_code_raw'] = original_code
                data['lean_code'] = normalize_lean_code(original_code)
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1

    print(f"Done. Processed {count} lines.")


# ---------------------------------------------------------------------------
# filter: keep samples with exactly one theorem and no axiom
# ---------------------------------------------------------------------------

THEOREM_RE = re.compile(r"^\s*theorem\b", re.MULTILINE)
AXIOM_RE = re.compile(r"\baxiom\b")


def check_sample(obj: Dict) -> Tuple[bool, str]:
    if "lean_code" not in obj:
        return False, "missing_lean_code"
    lean_code = obj.get("lean_code")
    if lean_code in (None, ""):
        return False, "empty_lean_code"
    lean_code = str(lean_code)
    theorem_count = len(THEOREM_RE.findall(lean_code))
    if theorem_count == 0:
        return False, "no_theorem_in_code"
    if theorem_count > 1:
        return False, "multiple_theorems"
    if AXIOM_RE.search(lean_code):
        return False, "has_axiom"
    return True, ""


def run_filter(input_file: str, output_file: str, log_every: int = 5000) -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    total = 0
    kept = 0
    skipped: Dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped["invalid_json"] = skipped.get("invalid_json", 0) + 1
                continue

            ok, reason = check_sample(obj)
            if not ok:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

            if log_every > 0 and total % log_every == 0:
                print(f"processed={total} kept={kept} skipped={skipped}")

    print(f"done processed={total} kept={kept} skipped={skipped} output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_norm = sub.add_parser("normalize", help="normalize Lean code (strip comments, unify sorry, rename theorems)")
    p_norm.add_argument("--input", required=True, help="input jsonl path")
    p_norm.add_argument("--output", required=True, help="output jsonl path")

    p_filter = sub.add_parser("filter", help="keep samples with exactly one theorem and no axiom")
    p_filter.add_argument("--input", required=True, help="input jsonl path")
    p_filter.add_argument("--output", required=True, help="output jsonl path")
    p_filter.add_argument("--log-every", type=int, default=5000)

    args = parser.parse_args()
    if args.command == "normalize":
        run_normalize(args.input, args.output)
    else:
        run_filter(args.input, args.output, args.log_every)


if __name__ == "__main__":
    main()
