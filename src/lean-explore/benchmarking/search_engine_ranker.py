# benchmarking/search_engine_ranker.py

"""Module for evaluating search engine results using an LLM."""

import asyncio
import json
import os
import sys
from itertools import permutations
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tqdm.asyncio import tqdm

current_script_path = os.path.abspath(__file__)
benchmarking_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(benchmarking_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from dev_tools.llm_caller import GeminiClient, GeminiCostTracker
except ImportError:
    print(
        "Error: Could not import GeminiClient or GeminiCostTracker from "
        "dev_tools.llm_caller."
    )
    print(
        "Please ensure that the 'dev_tools' directory is accessible, "
        "the project root is in your PYTHONPATH, or you are running the script "
        "from the project root directory."
    )
    GeminiClient = None  # type: ignore
    GeminiCostTracker = None  # type: ignore
    import sys

    sys.exit(1)


# --- Configuration Constants ---
SEARCH_RESULTS_FILE = "search_results.json"
EVALUATION_OUTPUT_FILE = "evaluation_output.json"
LLM_GENERATION_MODEL = "gemini-2.5-flash-preview-05-20"
MAX_RESULTS_TO_DISPLAY_PER_ENGINE = 5
MAX_CONCURRENT_LLM_CALLS = 50
MAX_LLM_PARSE_RETRIES = 2  # Number of retries after the initial attempt fails parsing

SEARCH_ENGINE_KEYS = [
    "leansearch_results",
    "moogle_results",
    "leanexplore_results",
]
SEARCH_ENGINE_NAMES = {
    "leansearch_results": "LeanSearch",
    "moogle_results": "Moogle",
    "leanexplore_results": "LeanExplore",
}


# -----------------------------


def convert_sets_to_lists_in_evaluations(
    eval_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Converts sets in llm_parsed_ranking to lists for JSON serialization.

    Args:
        eval_data: A list of evaluation dictionaries.

    Returns:
        A list of evaluation dictionaries with sets converted to lists.
    """
    serializable_data = []
    for entry in eval_data:
        new_entry = entry.copy()
        if "llm_parsed_ranking" in new_entry and isinstance(
            new_entry["llm_parsed_ranking"], list
        ):
            new_entry["llm_parsed_ranking"] = [
                list(rank_set) if isinstance(rank_set, set) else rank_set
                for rank_set in new_entry["llm_parsed_ranking"]
            ]
        serializable_data.append(new_entry)
    return serializable_data


def strip_moogle_docstring(text: Optional[str]) -> str:
    """Strips common Lean comment markers from Moogle docstrings.

    Args:
        text: The raw docstring text from Moogle.

    Returns:
        The cleaned docstring text, or "N/A" if input is empty.
    """
    if not text:
        return "N/A"
    text = text.strip()
    if (text.startswith("/--") or text.startswith("/-!")) and text.endswith("-/"):
        text = text[3:-2].strip()

    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("--"):
            cleaned_lines.append(stripped_line[2:].strip())
        elif stripped_line.startswith("- "):
            cleaned_lines.append(stripped_line[2:].strip())
        else:
            cleaned_lines.append(line)

    processed_text = "\n".join(cleaned_lines).strip()
    return processed_text if processed_text else "N/A"


def format_single_hit(hit: Dict[str, Any], engine_key: str) -> Tuple[str, str, str]:
    """Formats a single search hit into Code, Docstring, and Informal Desc.

    Args:
        hit: A dictionary representing a single search result.
        engine_key: The key of the search engine.

    Returns:
        A tuple containing the code text, docstring text, and informal
        description text. Each will be "N/A" if not available.
    """
    code_text = "N/A"
    docstring_text = "N/A"
    informal_description_text = "N/A"

    if engine_key == "leansearch_results":
        actual_result = hit.get("result", {})

        kind = actual_result.get("kind", "")
        name_list = actual_result.get("name", [])
        full_name = (
            ".".join(name_list) if name_list and isinstance(name_list, list) else ""
        )
        signature = actual_result.get("signature", "")

        statement_parts = []
        if kind:
            statement_parts.append(kind)
        if full_name:
            statement_parts.append(full_name)
        code_text = " ".join(statement_parts) + signature
        code_text = code_text.strip() if code_text.strip() else "N/A"

        docstring_text = "N/A"

        informal_name = actual_result.get("informal_name")
        informal_desc_from_json = actual_result.get("informal_description")

        if informal_name and informal_desc_from_json:
            informal_description_text = f"{informal_name}\n{informal_desc_from_json}"
        elif informal_name:
            informal_description_text = informal_name
        elif informal_desc_from_json:
            informal_description_text = informal_desc_from_json
        else:
            informal_description_text = "N/A"
        if informal_description_text is None:
            informal_description_text = "N/A"

    elif engine_key == "moogle_results":
        code_text = hit.get("declarationCode", "N/A")
        docstring_content = hit.get("declarationDocstring")
        docstring_text = strip_moogle_docstring(docstring_content)
        informal_description_text = "N/A"

    elif engine_key == "leanexplore_results":
        code_text = hit.get("statement_text", hit.get("display_statement_text", "N/A"))
        docstring_content = hit.get("docstring")
        docstring_text = (
            docstring_content.strip()
            if docstring_content and docstring_content.strip()
            else "N/A"
        )

        informal_desc_from_json = hit.get("informal_description")
        informal_description_text = (
            informal_desc_from_json.strip()
            if informal_desc_from_json and informal_desc_from_json.strip()
            else "N/A"
        )

    else:
        return f"Unknown engine key: {engine_key}", "N/A", "N/A"

    if not code_text.strip():
        code_text = "N/A"
    if not docstring_text.strip():
        docstring_text = "N/A"
    if not informal_description_text.strip():
        informal_description_text = "N/A"

    return code_text, docstring_text, informal_description_text


def format_search_results_for_prompt(
    result_data: Optional[Union[Dict[str, Any], List[Any]]],
    engine_key: str,
    max_hits: int,
) -> str:
    """Formats search results for a single engine for inclusion in an LLM prompt.

    Args:
        result_data: The data for a specific engine from search_results.json.
        engine_key: The key identifying the search engine.
        max_hits: The maximum number of search hits to include in the formatted string.

    Returns:
        A string summarizing the search results for the engine.
    """
    if result_data is None:
        return "Results not available for this engine."

    if isinstance(result_data, dict) and "error" in result_data:
        return f"Error fetching results: {result_data['error']}"

    hits: List[Any] = []
    if engine_key == "leansearch_results":
        if isinstance(result_data, list):
            if result_data and isinstance(result_data[0], list):
                hits = result_data[0]
            else:
                hits = result_data
        elif isinstance(result_data, dict):
            hits = result_data.get("hits", [])
        else:
            return f"Unexpected data format for {engine_key}."
    elif engine_key == "moogle_results":
        if isinstance(result_data, dict):
            hits = result_data.get(
                "data",
                result_data.get(
                    "items", result_data.get("results", result_data.get("hits", []))
                ),
            )
        elif isinstance(result_data, list):
            hits = result_data
        else:
            return f"Unexpected data format for {engine_key}."
    elif engine_key == "leanexplore_results":
        if isinstance(result_data, dict):
            hits = result_data.get("results", [])
        else:
            return f"Unexpected data format for {engine_key} (expected dict)."
    else:
        return "Unknown engine key for formatting."

    if not hits:
        return "No results found by this engine."

    valid_hits = [hit for hit in hits if isinstance(hit, dict)]
    if not valid_hits:
        if hits:
            return "No valid result items found to display."
        return "No results found by this engine."

    formatted_item_blocks_list = []

    for i, hit_dict in enumerate(valid_hits[:max_hits]):
        code, docstring, informal_desc = format_single_hit(hit_dict, engine_key)

        item_parts = [
            f"RESULT {i + 1}\n",
            "Code:",
            f"{code}\n",
            "Docstring:",
            f"{docstring}\n",
            "Informal Description:",
            f"{informal_desc}",
        ]
        formatted_item_blocks_list.append("\n".join(item_parts))

    return "\n\n".join(formatted_item_blocks_list)


def construct_llm_prompt(
    query: str, engine_placeholders: Dict[str, str], formatted_results: Dict[str, str]
) -> Tuple[str, str]:
    """Constructs the system and user prompts for the LLM.

    Args:
        query: The original search query.
        engine_placeholders: Maps placeholder names (Engine A, B, C) to actual
            engine names.
        formatted_results: Maps placeholder names to their formatted search
            result strings.

    Returns:
        A tuple containing the system prompt and the user prompt.
    """
    system_prompt_str = (
        "You are an expert search result evaluator. Your task is to analyze search "
        "results from three different search engines for a given query. "
        "These engines are presented as Engine A, Engine B, and Engine C. "
        "You need to rank these three engines from best to worst based on "
        "how accurate the search results are for the provided query. "
        "If two or more engines are of comparable quality "
        "for a given rank, you can declare them as tied.\n\n"
        "First, provide your detailed reasoning for the ranking. "
        "After your reasoning, on a new and final line, provide your ranking. "
        "Use the placeholders 'Engine A', 'Engine B', 'Engine C'. "
        "Separate distinct ranks with a comma and a space (e.g., "
        "'Engine A, Engine B'). "
        "Separate tied engines at the same rank with an equals sign and spaces "
        "around it (e.g., 'Engine A = Engine B'). "
        "All three engine placeholders must be present in the ranking line.\n"
        "Examples of valid ranking lines:\n"
        "- Engine B, Engine A, Engine C\n"
        "- Engine A = Engine B, Engine C\n"
        "- Engine A, Engine B = Engine C\n"
        "- Engine A = Engine B = Engine C"
    )

    user_prompt_parts = [f'Original Query: "{query}"\n']
    user_prompt_parts.append("Search Results:\n")

    for placeholder in sorted(engine_placeholders.keys()):
        results_str = formatted_results[placeholder]
        user_prompt_parts.append(f"\n----- {placeholder} -----\n\n{results_str}\n")

    user_prompt_parts.append(
        "\nBased on the original query, please provide your detailed reasoning "
        "for ranking Engine A, Engine B, and Engine C. After your reasoning, "
        "on a new and final line, state your ranking of Engine A, Engine B, "
        "and Engine C from best to worst. Use only the placeholders 'Engine A', "
        "'Engine B', 'Engine C'.\n"
        "If engines are tied, list them separated by ' = ' "
        "(e.g., 'Engine A = Engine B').\n"
        "If ranks are distinct, separate them by ', ' "
        "(e.g., 'Engine A, Engine B').\n"
        "This final line must contain only the ranking and all three engine "
        "placeholders.\n"
        "For example:\n"
        "Engine B, Engine A, Engine C\n"
        "OR\n"
        "Engine A = Engine B, Engine C\n"
        "OR\n"
        "Engine A, Engine B = Engine C\n"
        "OR\n"
        "Engine A = Engine B = Engine C"
    )
    user_prompt_str = "\n".join(user_prompt_parts)
    return system_prompt_str, user_prompt_str


def parse_llm_ranking(
    ranking_line: str, placeholder_to_engine_map: Dict[str, str]
) -> List[Set[str]]:
    """Parses the LLM's single-line ranking, handling ties.

    The ranking line is expected to use ',' to separate distinct ranks and
    '=' to denote ties within a rank. All three engine placeholders
    (e.g., "Engine A", "Engine B", "Engine C") must be present exactly once
    in the ranking.

    Args:
        ranking_line: The single line from the LLM response.
                      Examples: "Engine C, Engine A, Engine B"
                                "Engine A = Engine B, Engine C"
                                "Engine A, Engine B = Engine C"
                                "Engine A = Engine B = Engine C"
        placeholder_to_engine_map: Maps placeholders like "Engine A"
                                   to actual engine names (e.g., "Moogle").

    Returns:
        A list of sets, where each set contains the actual engine names
        for a given rank, ordered from best rank (index 0) to worst.
        Example: [ {"Moogle", "LeanSearch"}, {"LeanExplore"} ] for the
                 ranking "Engine A = Engine B, Engine C" (assuming A is Moogle,
                 B is LeanSearch, C is LeanExplore).
        Returns an empty list if critical parsing errors occur or if the
        ranking is fundamentally malformed (e.g., missing engines, engines
        ranked multiple times in different groups, or unrecognized placeholders).
    """
    parsed_ranks: List[Set[str]] = []
    engines_assigned_to_a_rank: Set[str] = set()

    rank_group_segments = ranking_line.split(",")

    for segment_str in rank_group_segments:
        stripped_segment = segment_str.strip()
        if not stripped_segment:
            print(
                f"Warning: Empty segment found after splitting by comma in "
                f"ranking line: '{ranking_line}'. Skipping segment."
            )
            continue

        placeholder_candidates = stripped_segment.split("=")

        current_rank_engine_names: Set[str] = set()

        for ph_candidate_raw in placeholder_candidates:
            placeholder_str = ph_candidate_raw.strip()
            if not placeholder_str:
                print(
                    f"Warning: Empty placeholder string found after splitting by "
                    f"'=' in segment '{stripped_segment}'. Skipping placeholder."
                )
                continue

            if placeholder_str in placeholder_to_engine_map:
                engine_name = placeholder_to_engine_map[placeholder_str]
                current_rank_engine_names.add(engine_name)
            else:
                print(
                    f"Warning: Unrecognized placeholder '{placeholder_str}' found in "
                    f"segment '{stripped_segment}' from LLM ranking line: "
                    f"'{ranking_line}'."
                )

        if current_rank_engine_names:
            for engine_in_current_group in current_rank_engine_names:
                if engine_in_current_group in engines_assigned_to_a_rank:
                    print(
                        f"Critical Error: Engine '{engine_in_current_group}' in "
                        f"current group {current_rank_engine_names} (from segment "
                        f"'{stripped_segment}') was already assigned to a previous "
                        f"rank. Malformed LLM response: '{ranking_line}'. "
                        "Aborting parse."
                    )
                    return []

            parsed_ranks.append(current_rank_engine_names)
            engines_assigned_to_a_rank.update(current_rank_engine_names)
        elif stripped_segment:
            all_placeholders_in_segment_were_problematic = True
            for ph_candidate_raw_check in placeholder_candidates:
                ph_stripped_check = ph_candidate_raw_check.strip()
                if ph_stripped_check and ph_stripped_check in placeholder_to_engine_map:
                    all_placeholders_in_segment_were_problematic = False
                    break
            if all_placeholders_in_segment_were_problematic:
                print(
                    f"Warning: Segment '{stripped_segment}' from ranking line "
                    f"'{ranking_line}' contained only unrecognized or empty "
                    "placeholders, resulting in no engines for this rank group."
                )

    expected_engine_names_set = set(placeholder_to_engine_map.values())

    if engines_assigned_to_a_rank != expected_engine_names_set:
        error_messages = []
        missing_engines = expected_engine_names_set - engines_assigned_to_a_rank
        extra_engines = engines_assigned_to_a_rank - expected_engine_names_set

        if missing_engines:
            error_messages.append(f"Missing expected engines: {missing_engines}")
        if extra_engines:
            error_messages.append(
                "Found unexpected/unmapped engines in final ranked set: "
                f"{extra_engines}"
            )

        if (
            not error_messages
            and engines_assigned_to_a_rank != expected_engine_names_set
        ):
            error_messages.append(
                f"The set of finally ranked engines ({engines_assigned_to_a_rank}) "
                f"does not exactly match the expected set of engines "
                f"({expected_engine_names_set})."
            )

        if not error_messages:
            error_messages.append(
                "General mismatch between ranked engines and expected engines."
            )

        print(
            f"Critical Error in ranking consistency: {'; '.join(error_messages)}. "
            f"LLM ranking line: '{ranking_line}'. Parsed structure: {parsed_ranks}. "
            f"Engines assigned to rank: {engines_assigned_to_a_rank}. Aborting parse."
        )
        return []

    if not parsed_ranks and ranking_line.strip():
        print(
            f"Critical Error: Failed to parse any valid ranking structure from the "
            f"non-empty LLM ranking line: '{ranking_line}'. Ensure the LLM response "
            "adheres to the expected format (e.g., 'Engine A, Engine B = Engine C'). "
            "Aborting parse."
        )
        return []

    return parsed_ranks


async def evaluate_single_query(
    query_data: Dict[str, Any],
    engine_permutation: Tuple[str, ...],
    gemini_client: GeminiClient,
) -> Dict[str, Any]:
    """Evaluates search results for a single query using the LLM.

    Implements a retry mechanism if the LLM response is received but
    the ranking cannot be parsed.

    Args:
        query_data: Contains the query and its search results for all engines.
        engine_permutation: A tuple defining the order (Engine A, B, C)
                            in which engine results are presented to the LLM.
        gemini_client: An initialized GeminiClient instance.

    Returns:
        A dictionary containing the evaluation details for this query,
        including parsed ranking (as List[Set[str]]).
    """
    query_string = query_data["query"]
    evaluation_entry: Dict[str, Any] = {
        "original_query": query_string,
        "engines_presented_order": [
            SEARCH_ENGINE_NAMES.get(ek, ek) for ek in engine_permutation
        ],
        "placeholder_to_engine_map": {},
        "llm_system_prompt": "",
        "llm_user_prompt": "",
        "llm_raw_response": None,
        "llm_parsed_ranking": [],
        "llm_error": None,
    }

    placeholders = ["Engine A", "Engine B", "Engine C"]
    current_placeholder_to_engine_map: Dict[str, str] = {}
    formatted_results_for_prompt: Dict[str, str] = {}

    for i, engine_key in enumerate(engine_permutation):
        placeholder = placeholders[i]
        actual_engine_name = SEARCH_ENGINE_NAMES.get(engine_key, engine_key)
        current_placeholder_to_engine_map[placeholder] = actual_engine_name

        engine_specific_results = query_data.get(engine_key)
        formatted_results_for_prompt[placeholder] = format_search_results_for_prompt(
            engine_specific_results, engine_key, MAX_RESULTS_TO_DISPLAY_PER_ENGINE
        )

    evaluation_entry["placeholder_to_engine_map"] = current_placeholder_to_engine_map

    system_prompt, user_prompt = construct_llm_prompt(
        query_string, current_placeholder_to_engine_map, formatted_results_for_prompt
    )
    evaluation_entry["llm_system_prompt"] = system_prompt
    evaluation_entry["llm_user_prompt"] = user_prompt

    for attempt in range(MAX_LLM_PARSE_RETRIES + 1):
        evaluation_entry["llm_error"] = None
        evaluation_entry["llm_raw_response"] = None
        # evaluation_entry["llm_parsed_ranking"] is reset implicitly if parsing fails

        ranking_line_this_attempt = ""

        try:
            llm_response_text = await gemini_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=LLM_GENERATION_MODEL,
            )
            evaluation_entry["llm_raw_response"] = llm_response_text

            if llm_response_text:
                lines = llm_response_text.strip().split("\n")
                if lines:
                    for line_idx in range(len(lines) - 1, -1, -1):
                        current_line_stripped = lines[line_idx].strip()
                        if current_line_stripped:
                            ranking_line_this_attempt = current_line_stripped
                            break

                if ranking_line_this_attempt:
                    parsed_ranking_result = parse_llm_ranking(
                        ranking_line_this_attempt, current_placeholder_to_engine_map
                    )
                    if parsed_ranking_result:
                        evaluation_entry["llm_parsed_ranking"] = parsed_ranking_result
                        evaluation_entry["llm_error"] = None
                        break  # Successful parse, exit retry loop
                    else:
                        evaluation_entry["llm_parsed_ranking"] = []
                        evaluation_entry["llm_error"] = (
                            "LLM response's ranking line "
                            f"('{ranking_line_this_attempt}') could not be parsed."
                        )
                        print(
                            "Info: Parsing of ranking line "
                            f"'{ranking_line_this_attempt}' for query "
                            f"'{query_string[:50]}...' (Attempt {attempt + 1}) "
                            "resulted in an empty ranking. Check warnings from parser."
                        )
                else:  # No ranking line extracted from non-empty response
                    evaluation_entry["llm_parsed_ranking"] = []
                    if any(line.strip() for line in lines):
                        evaluation_entry["llm_error"] = (
                            "LLM response had content but no final ranking line "
                            "was identified."
                        )
                        print(
                            f"Warning: LLM response for query '{query_string[:50]}...' "
                            f"(Attempt {attempt + 1}) contained text but no "
                            "clear ranking line was extracted."
                        )
                    else:  # Response became empty after strip
                        evaluation_entry["llm_error"] = (
                            "LLM returned an effectively empty response "
                            "(e.g., only newlines)."
                        )
            else:  # LLM returned None or empty string
                evaluation_entry["llm_parsed_ranking"] = []
                evaluation_entry["llm_error"] = (
                    "LLM returned no response text (None or empty string)."
                )

        except Exception as e:
            print(
                f"Error during LLM call for query '{query_string[:50]}...' "
                f"(Attempt {attempt + 1}): {e}"
            )
            evaluation_entry["llm_error"] = str(e)
            evaluation_entry["llm_parsed_ranking"] = []
            # llm_raw_response may be None or from a partial failure

        if not evaluation_entry["llm_parsed_ranking"]:
            if attempt < MAX_LLM_PARSE_RETRIES:
                print(
                    "Warning: Attempt "
                    f"{attempt + 1}/{MAX_LLM_PARSE_RETRIES + 1} for query "
                    f"'{query_string[:50]}...' failed. Error: "
                    f"{evaluation_entry['llm_error']}. Retrying..."
                )
                await asyncio.sleep(1 + attempt)
            else:
                print(
                    "Error: Failed to get a valid, parsable ranking for query "
                    f"'{query_string[:50]}...' after {MAX_LLM_PARSE_RETRIES + 1} "
                    "attempts. "
                    f"Last error: {evaluation_entry['llm_error']}"
                )
        else:
            pass

    return evaluation_entry


async def wrapped_evaluate_single_query_with_semaphore(
    query_data: Dict[str, Any],
    engine_permutation: Tuple[str, ...],
    gemini_client: GeminiClient,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Acquires semaphore then calls evaluate_single_query.

    Args:
        query_data: Contains the query and its search results for all engines.
        engine_permutation: A tuple defining the order (Engine A, B, C)
                            in which engine results are presented to the LLM.
        gemini_client: An initialized GeminiClient instance.
        semaphore: An asyncio.Semaphore to limit concurrent calls.

    Returns:
        A dictionary containing the evaluation details for this query.
    """
    async with semaphore:
        return await evaluate_single_query(
            query_data, engine_permutation, gemini_client
        )


async def main_evaluation_loop():
    """Main asynchronous function to orchestrate the evaluation process."""
    if GeminiClient is None:
        return

    search_results_path = os.path.join(os.path.dirname(__file__), SEARCH_RESULTS_FILE)
    if not os.path.exists(search_results_path):
        print(f"Error: Search results file not found at '{search_results_path}'.")
        print(f"Please run gather_data.py first to generate '{SEARCH_RESULTS_FILE}'.")
        return

    try:
        with open(search_results_path, encoding="utf-8") as f:
            all_search_data = json.load(f)
        if not isinstance(all_search_data, list):
            print(f"Error: '{search_results_path}' does not contain a JSON list.")
            return
        if not all_search_data:
            print(f"No search data found in '{search_results_path}'.")
            return
        print(
            f"Loaded {len(all_search_data)} query records from '{search_results_path}'."
        )
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{search_results_path}'.")
        return
    except Exception as e:
        print(f"Error reading '{search_results_path}': {e}")
        return

    try:
        gemini_client = GeminiClient()
        print(
            f"GeminiClient initialized. Using model: {LLM_GENERATION_MODEL} "
            "for evaluations."
        )
    except Exception as e:
        print(f"Failed to initialize GeminiClient: {e}")
        print("Ensure GEMINI_API_KEY is set and config is accessible if used.")
        return

    engine_key_permutations = list(permutations(SEARCH_ENGINE_KEYS))
    num_permutations = len(engine_key_permutations)

    all_evaluations: List[Dict[str, Any]] = []
    evaluation_output_path = os.path.join(
        os.path.dirname(__file__), EVALUATION_OUTPUT_FILE
    )
    processed_queries_set: Set[str] = set()

    if os.path.exists(evaluation_output_path):
        try:
            with open(evaluation_output_path, encoding="utf-8") as f_in:
                existing_evaluations = json.load(f_in)
                if isinstance(existing_evaluations, list):
                    for eval_item in existing_evaluations:
                        if (
                            isinstance(eval_item, dict)
                            and "original_query" in eval_item
                        ):
                            processed_queries_set.add(eval_item["original_query"])
                            all_evaluations.append(eval_item)  # type: ignore
                    print(
                        f"Loaded {len(processed_queries_set)} existing evaluations "
                        f"from '{evaluation_output_path}'."
                    )
        except Exception as e:
            print(
                f"Warning: Could not load or parse existing evaluations from "
                f"'{evaluation_output_path}': {e}"
            )
            all_evaluations = []
            processed_queries_set = set()

    valid_queries_for_run: List[Dict[str, Any]] = []
    non_dict_items_count = 0
    already_processed_count = 0

    for item in all_search_data:
        if isinstance(item, dict) and "query" in item:
            if item["query"] not in processed_queries_set:
                valid_queries_for_run.append(item)
            else:
                already_processed_count += 1
        else:
            non_dict_items_count += 1
            print(
                f"Warning: Found a non-dictionary or malformed item in "
                f"'{SEARCH_RESULTS_FILE}': {str(item)[:100]}... This will be skipped."
            )

    if non_dict_items_count > 0:
        print(f"Total non-dictionary/malformed items skipped: {non_dict_items_count}.")
    if already_processed_count > 0:
        print(
            f"Found {already_processed_count} queries in '{SEARCH_RESULTS_FILE}' "
            f"that were already processed and present in '{EVALUATION_OUTPUT_FILE}'."
        )

    if not valid_queries_for_run:
        print(
            "No new queries to evaluate. All queries from input file are already "
            "in the output file or input was empty/invalid."
        )
    else:
        print(
            f"Starting LLM evaluation for {len(valid_queries_for_run)} new queries, "
            f"using up to {MAX_CONCURRENT_LLM_CALLS} concurrent tasks."
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
        tasks_for_as_completed = []

        initial_completed_eval_count = len(all_evaluations)

        for i, query_data_item in enumerate(valid_queries_for_run):
            current_permutation_index = (
                initial_completed_eval_count + i
            ) % num_permutations
            current_permutation = engine_key_permutations[current_permutation_index]

            task = wrapped_evaluate_single_query_with_semaphore(
                query_data_item, current_permutation, gemini_client, semaphore
            )
            tasks_for_as_completed.append(task)

        for future in tqdm(
            asyncio.as_completed(tasks_for_as_completed),
            total=len(tasks_for_as_completed),
            desc="Evaluating Queries (LLM Calls)",
        ):
            try:
                evaluation_result = await future
                all_evaluations.append(evaluation_result)

                try:
                    serializable_evaluations = convert_sets_to_lists_in_evaluations(
                        all_evaluations
                    )
                    with open(evaluation_output_path, "w", encoding="utf-8") as f_out:
                        json.dump(
                            serializable_evaluations,
                            f_out,
                            indent=4,
                            ensure_ascii=False,
                        )
                except Exception as e:
                    tqdm.write(
                        "Error writing intermediate results to "
                        f"'{evaluation_output_path}': {e}"
                    )
            except Exception as e:
                tqdm.write(f"Critical error processing a query evaluation task: {e}")

    try:
        serializable_evaluations = convert_sets_to_lists_in_evaluations(all_evaluations)
        with open(evaluation_output_path, "w", encoding="utf-8") as f_out:
            json.dump(serializable_evaluations, f_out, indent=4, ensure_ascii=False)
        print(
            f"Successfully saved {len(all_evaluations)} total evaluations to "
            f"'{evaluation_output_path}'."
        )
    except Exception as e:
        print(f"Error writing final results to '{evaluation_output_path}': {e}")

    if hasattr(gemini_client, "cost_tracker") and gemini_client.cost_tracker:
        cost_summary = gemini_client.cost_tracker.get_summary()
        print("\n--- LLM Usage Cost Summary ---")
        print(json.dumps(cost_summary, indent=2))
        total_cost = cost_summary.get("total_estimated_cost", 0.0)
        if not isinstance(total_cost, (int, float)):
            total_cost = 0.0
        print(f"Total Estimated Cost: ${total_cost:.4f}")
    else:
        print("\nCost tracker not available or not initialized in GeminiClient.")


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print(
            "Warning: GEMINI_API_KEY environment variable is not set. "
            "The script may fail if the API key is not configured elsewhere."
        )
    asyncio.run(main_evaluation_loop())
