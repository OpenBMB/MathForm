"""Multi-query script for quick local search validation."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import textwrap
from typing import Iterable, List, Optional

TRUE_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRUE_PROJECT_ROOT / "src"))

try:
    from lean_explore.local.service import Service
except ImportError as exc:
    print(
        "Cannot import project modules; make sure src is installed/discoverable."
        f"\nOriginal error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple queries and print result summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("queries", nargs="*", help="query strings (one or more)")
    parser.add_argument(
        "--queries-file",
        type=pathlib.Path,
        default=None,
        help="read queries from a file (one per line)",
    )
    parser.add_argument("--limit", type=int, default=3, help="max results per query")
    parser.add_argument(
        "--package",
        action="append",
        default=None,
        help="package filter (repeatable or comma-separated)",
    )
    parser.add_argument("--show", type=int, default=5, help="show the first N results")
    parser.add_argument("--json", action="store_true", help="print full JSON output")
    return parser.parse_args()


def _normalize_packages(packages: Optional[List[str]]) -> Optional[List[str]]:
    if not packages:
        return None
    expanded: List[str] = []
    for item in packages:
        if not item:
            continue
        for part in item.split(","):
            part = part.strip()
            if part:
                expanded.append(part)
    return expanded or None


def _load_queries(file_path: pathlib.Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"query file not found: {file_path}")
    with file_path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _flatten_queries(cli_queries: Iterable[str], file_queries: Iterable[str]) -> List[str]:
    merged = [q.strip() for q in list(cli_queries) + list(file_queries) if q.strip()]
    return merged


def _truncate(text: Optional[str], max_len: int = 200) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def _wrap(text: Optional[str], width: int = 88) -> Optional[str]:
    if text is None:
        return None
    return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False))


def _print_human(resp, show_count: int, index: Optional[int] = None) -> None:
    header = f"query: {resp.query}" if index is None else f"[{index}] query: {resp.query}"
    print(header)
    print(f"count: {resp.count}")
    print(f"total_candidates_considered: {resp.total_candidates_considered}")
    print(f"processing_time_ms: {resp.processing_time_ms}")
    print("")
    print(f"top {min(show_count, resp.count)} results:")
    print("-" * 80)
    for idx, item in enumerate(resp.results[:show_count], start=1):
        name = item.primary_declaration.lean_name if item.primary_declaration else None
        statement = item.display_statement_text or item.statement_text
        statement = _wrap(_truncate(statement, max_len=400))
        docstring = _wrap(_truncate(item.docstring, max_len=400))
        informal = _wrap(_truncate(item.informal_description, max_len=400))

        print(f"[{idx}] id={item.id} name={name}")
        print(f"file: {item.source_file}:{item.range_start_line}")
        if statement:
            print("statement:")
            print(textwrap.indent(statement, "  "))
        if docstring:
            print("docstring:")
            print(textwrap.indent(docstring, "  "))
        if informal:
            print("informal_description:")
            print(textwrap.indent(informal, "  "))
        print("-" * 80)


def main() -> int:
    args = _parse_args()
    packages = _normalize_packages(args.package)
    file_queries = []
    if args.queries_file is not None:
        file_queries = _load_queries(args.queries_file)

    queries = _flatten_queries(args.queries, file_queries)
    if not queries:
        print("No queries provided; pass queries or use --queries-file.", file=sys.stderr)
        return 2

    service = Service()
    responses = service.search(queries, package_filters=packages, limit=args.limit)

    if args.json:
        payload = [resp.model_dump() for resp in responses]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for i, resp in enumerate(responses, start=1):
            _print_human(resp, args.show, index=i if len(responses) > 1 else None)
            if i != len(responses):
                print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
