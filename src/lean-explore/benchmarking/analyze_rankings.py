# benchmarking/analyze_rankings.py

"""Module for analyzing LLM ranking results from evaluation_output.json."""

import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

EVALUATION_OUTPUT_FILE = "evaluation_output.json"
PLACE_MAP = {1: "1st", 2: "2nd", 3: "3rd"}
PLACES_TO_REPORT = ["1st", "2nd", "3rd"]
TIE_DETAIL_KEYS = ["solo", "2-way tie", "3-way tie"]


def calculate_rank_statistics(
    evaluation_data: List[Dict[str, Any]],
) -> DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]]:
    """Calculates 1st, 2nd, 3rd place finishes, detailing ties.

    Args:
        evaluation_data: A list of evaluation dictionaries, each containing
                         an 'llm_parsed_ranking' field.

    Returns:
        A defaultdict where keys are engine names, and values are
        defaultdicts mapping place strings ('1st', '2nd', '3rd') to
        another defaultdict. This inner defaultdict maps tie type
        ('total', 'solo', '2-way tie', '3-way tie') to counts.

    Example:
        {'EngineA': {
            '1st': {'total': 10, 'solo': 5, '2-way tie': 5},
            '2nd': {'total': 3, 'solo': 3}
            }
        }
    """
    engine_scores: DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )

    for entry_idx, entry in enumerate(evaluation_data):
        parsed_ranking = entry.get("llm_parsed_ranking")
        original_query = entry.get(
            "original_query", f"Unknown query (entry {entry_idx})"
        )

        if not parsed_ranking:
            if entry.get("llm_error"):
                print(
                    f"Info (Rank Stats): Skipping query '{original_query}' due to "
                    f"LLM error: {entry['llm_error']}"
                )
            elif isinstance(parsed_ranking, list):
                print(
                    f"Info (Rank Stats): Skipping query '{original_query}' as it has "
                    "an empty LLM parsed ranking."
                )
            else:
                print(
                    f"Warning (Rank Stats): Skipping query '{original_query}' due to "
                    "missing or invalid 'llm_parsed_ranking' field: "
                    f"{parsed_ranking}"
                )
            continue

        current_place = 1
        for rank_group_idx, rank_group in enumerate(parsed_ranking):
            if not rank_group:
                continue
            if not isinstance(rank_group, list):
                print(
                    f"Warning (Rank Stats): Malformed rank group in query "
                    f"'{original_query}'. Expected list, got: {rank_group}. "
                    "Skipping this group."
                )
                continue

            if current_place > 3:
                break

            place_key = PLACE_MAP.get(current_place)
            if place_key:
                valid_engines_in_group = []
                for engine_name_obj in rank_group:
                    if isinstance(engine_name_obj, str) and engine_name_obj.strip():
                        valid_engines_in_group.append(engine_name_obj.strip())
                    else:
                        print(
                            f"Warning (Rank Stats): Invalid engine name found in query "
                            f"'{original_query}'. Skipping engine: {engine_name_obj}"
                        )

                if not valid_engines_in_group:
                    current_place += 0
                    continue

                num_tied_in_group = len(valid_engines_in_group)
                tie_type_detail_key = ""

                if num_tied_in_group == 1:
                    tie_type_detail_key = "solo"
                elif num_tied_in_group == 2:
                    tie_type_detail_key = "2-way tie"
                elif num_tied_in_group == 3:
                    tie_type_detail_key = "3-way tie"
                elif num_tied_in_group > 3 and current_place <= 3:
                    pass

                if tie_type_detail_key:
                    for engine_name in valid_engines_in_group:
                        engine_scores[engine_name][place_key][tie_type_detail_key] += 1
                        engine_scores[engine_name][place_key]["total"] += 1
                elif num_tied_in_group > 0:
                    for engine_name in valid_engines_in_group:
                        engine_scores[engine_name][place_key]["total"] += 1

            current_place += len(valid_engines_in_group)

    return engine_scores


def display_rank_statistics(
    engine_scores: DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]],
) -> None:
    """Prints the rank statistics for each engine, including tie details.

    Args:
        engine_scores: A dictionary containing the rank counts and tie details
                       for each engine, as returned by `calculate_rank_statistics`.
    """
    if not engine_scores:
        print("\n--- Search Engine Ranking Statistics ---")
        print("No engine scores to display for rank statistics.")
        return

    print("\n--- Search Engine Ranking Statistics ---")
    sorted_engine_names = sorted(engine_scores.keys())

    for engine_name in sorted_engine_names:
        print(f"\nEngine: {engine_name}")

        has_any_score = any(
            engine_scores[engine_name][place].get("total", 0) > 0
            for place in PLACES_TO_REPORT
        )
        if not has_any_score:
            print("  No 1st, 2nd, or 3rd place finishes recorded in this analysis.")
            continue

        for place in PLACES_TO_REPORT:
            total_for_place = engine_scores[engine_name][place].get("total", 0)
            if total_for_place > 0:
                print(f"  {place} Place: {total_for_place} times")

                for tie_key in TIE_DETAIL_KEYS:
                    count = engine_scores[engine_name][place].get(tie_key, 0)
                    if count > 0:
                        display_tie_key = tie_key[0].upper() + tie_key[1:]
                        print(f"    - {display_tie_key}: {count} times")


def calculate_head_to_head_statistics(
    evaluation_data: List[Dict[str, Any]], all_engine_names_list: List[str]
) -> DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]]:
    """Calculates head-to-head comparison statistics between pairs of engines.

    For each pair of engines (engineA, engineB), where engineA is
    lexicographically smaller than engineB, it counts how many times
    engineA ranked higher than engineB ('wins'), engineB ranked higher
    than engineA ('losses' for engineA), or they had the same rank
    and were both present ('ties').

    Args:
        evaluation_data: A list of evaluation dictionaries, each containing
                         an 'llm_parsed_ranking' field.
        all_engine_names_list: A sorted list of unique engine names found
                               across all valid rankings in evaluation_data.

    Returns:
        A defaultdict structure: h2h_stats[lex_smaller_engine][lex_larger_engine]
        containing counts for 'wins', 'losses', and 'ties'.
        'wins': lex_smaller_engine ranked higher than lex_larger_engine.
        'losses': lex_smaller_engine ranked lower than lex_larger_engine.
        'ties': Both engines had the same rank and were present.
    """
    h2h_stats: DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for entry_idx, entry in enumerate(evaluation_data):
        parsed_ranking = entry.get("llm_parsed_ranking")
        original_query = entry.get(
            "original_query", f"Unknown query (entry {entry_idx})"
        )

        if not parsed_ranking:
            if entry.get("llm_error"):
                print(
                    f"Info (H2H): Skipping query '{original_query}' due to "
                    f"LLM error: {entry['llm_error']}"
                )
            elif isinstance(parsed_ranking, list):
                print(
                    f"Info (H2H): Skipping query '{original_query}' as it has an "
                    "empty LLM parsed ranking."
                )
            else:
                print(
                    f"Warning (H2H): Skipping query '{original_query}' due to missing "
                    f"or invalid 'llm_parsed_ranking' field: {parsed_ranking}"
                )
            continue

        engine_actual_ranks_in_query: Dict[str, int] = {}
        current_rank_marker = 1
        has_any_valid_engine_in_this_ranking = False

        if not isinstance(parsed_ranking, list):
            print(
                f"Warning (H2H): Skipping query '{original_query}' due to "
                f"'llm_parsed_ranking' not being a list: {parsed_ranking}"
            )
            continue

        for rank_group in parsed_ranking:
            if not rank_group:
                continue
            if not isinstance(rank_group, list):
                print(
                    f"Warning (H2H): Malformed rank group in query '{original_query}'. "
                    f"Expected list, got: {rank_group}. Skipping this group."
                )
                continue

            group_engines = []
            for engine_name_obj in rank_group:
                if isinstance(engine_name_obj, str) and engine_name_obj.strip():
                    group_engines.append(engine_name_obj.strip())
                else:
                    print(
                        f"Warning (H2H): Invalid engine name found in rank group for "
                        f"query '{original_query}'. Skipping engine: {engine_name_obj}"
                    )

            if not group_engines:
                continue

            for engine_name in group_engines:
                engine_actual_ranks_in_query[engine_name] = current_rank_marker
                has_any_valid_engine_in_this_ranking = True
            current_rank_marker += len(group_engines)

        if not has_any_valid_engine_in_this_ranking:
            if not any(parsed_ranking):
                print(
                    f"Info (H2H): Skipping query '{original_query}' as its parsed "
                    "ranking contained no valid engine names after processing."
                )
            continue

        for i in range(len(all_engine_names_list)):
            for j in range(i + 1, len(all_engine_names_list)):
                engine1 = all_engine_names_list[i]
                engine2 = all_engine_names_list[j]

                rank1 = engine_actual_ranks_in_query.get(engine1, float("inf"))
                rank2 = engine_actual_ranks_in_query.get(engine2, float("inf"))

                if rank1 == float("inf") and rank2 == float("inf"):
                    continue

                if rank1 < rank2:
                    h2h_stats[engine1][engine2]["wins"] += 1
                elif rank2 < rank1:
                    h2h_stats[engine1][engine2]["losses"] += 1
                else:
                    h2h_stats[engine1][engine2]["ties"] += 1
    return h2h_stats


def display_head_to_head_statistics(
    h2h_stats: DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]],
    all_engine_names_list: List[str],
) -> None:
    """Prints the head-to-head comparison statistics for each pair of engines.

    Output includes win rates for each engine in the pair, the tie rate,
    and the raw counts for wins and ties, based on their direct comparisons.

    Args:
        h2h_stats: The head-to-head statistics data as returned by
                   `calculate_head_to_head_statistics`.
        all_engine_names_list: A sorted list of all unique engine names
                               involved in the comparison.
    """
    print("\n--- Head-to-Head Engine Comparison (Win/Tie Rates) ---")
    if not h2h_stats and len(all_engine_names_list) >= 2:
        print("No head-to-head data was generated from the analyzed queries.")
        return
    if len(all_engine_names_list) < 2:
        print(
            "Not enough unique engines (need at least 2) for head-to-head comparison."
        )
        return

    compared_at_least_one_pair_with_interactions = False
    for i in range(len(all_engine_names_list)):
        for j in range(i + 1, len(all_engine_names_list)):
            engine1 = all_engine_names_list[i]
            engine2 = all_engine_names_list[j]

            stats_for_pair = h2h_stats.get(engine1, {}).get(engine2, {})

            engine1_wins = stats_for_pair.get("wins", 0)
            engine2_wins = stats_for_pair.get("losses", 0)
            ties = stats_for_pair.get("ties", 0)

            total_comparisons_for_pair = engine1_wins + engine2_wins + ties

            if total_comparisons_for_pair == 0:
                continue

            compared_at_least_one_pair_with_interactions = True
            print(f"\n{engine1} vs {engine2}:")

            engine1_win_rate = (engine1_wins / total_comparisons_for_pair) * 100
            engine2_win_rate = (engine2_wins / total_comparisons_for_pair) * 100
            tie_rate = (ties / total_comparisons_for_pair) * 100

            print(
                f"  {engine1} Win Rate: {engine1_win_rate:.2f}% ({engine1_wins} wins)"
            )
            print(
                f"  {engine2} Win Rate: {engine2_win_rate:.2f}% ({engine2_wins} wins)"
            )
            print(f"  Tie Rate: {tie_rate:.2f}% ({ties} ties)")
            print(
                f"  (Based on {total_comparisons_for_pair} direct comparisons where "
                "at least one was ranked)"
            )

    if (
        not compared_at_least_one_pair_with_interactions
        and len(all_engine_names_list) >= 2
    ):
        print(
            "Although engines were found, no specific head-to-head matchups "
            "(wins, losses, or ties) were recorded to calculate rates."
        )
    elif not compared_at_least_one_pair_with_interactions:
        if len(all_engine_names_list) >= 2:
            print("No head-to-head statistics with interactions to display.")


def main() -> None:
    """Loads, calculates, and displays rank and head-to-head statistics."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_file_path = os.path.join(script_dir, EVALUATION_OUTPUT_FILE)

    if not os.path.exists(eval_file_path):
        print(f"Error: Evaluation file not found at '{eval_file_path}'.")
        print(
            f"Please ensure '{EVALUATION_OUTPUT_FILE}' exists in the same "
            "directory as this script."
        )
        return

    try:
        with open(eval_file_path, encoding="utf-8") as f:
            evaluation_data = json.load(f)
        if not isinstance(evaluation_data, list):
            print(f"Error: Content of '{eval_file_path}' is not a JSON list.")
            return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{eval_file_path}'.")
        return
    except Exception as e:
        print(f"Error reading or parsing '{eval_file_path}': {e}")
        return

    if not evaluation_data:
        print(f"No evaluation entries found in '{eval_file_path}'.")
        return

    print(f"Loaded {len(evaluation_data)} evaluation entries for analysis.")

    engine_scores = calculate_rank_statistics(evaluation_data)
    display_rank_statistics(engine_scores)

    all_engine_names_set = set()
    valid_queries_for_h2h_count = 0

    for entry in evaluation_data:
        parsed_ranking = entry.get("llm_parsed_ranking")
        if not parsed_ranking or not isinstance(parsed_ranking, list):
            continue

        has_engines_in_this_ranking = False
        for rank_group in parsed_ranking:
            if rank_group and isinstance(rank_group, list):
                for engine_name_obj in rank_group:
                    if isinstance(engine_name_obj, str) and engine_name_obj.strip():
                        all_engine_names_set.add(engine_name_obj.strip())
                        has_engines_in_this_ranking = True

        if has_engines_in_this_ranking:
            valid_queries_for_h2h_count += 1

    sorted_all_engine_names = sorted(list(all_engine_names_set))

    if valid_queries_for_h2h_count == 0:
        print("\n--- Head-to-Head Engine Comparison ---")
        print("No queries with valid rankings found to perform head-to-head analysis.")
    elif len(sorted_all_engine_names) < 2:
        print("\n--- Head-to-Head Engine Comparison ---")
        print(
            "Fewer than two unique engines found in valid rankings. "
            "Head-to-head comparison is not applicable."
        )
    else:
        print(
            f"\nAnalyzing {valid_queries_for_h2h_count} queries for head-to-head "
            f"statistics involving {len(sorted_all_engine_names)} unique engines."
        )
        h2h_engine_scores = calculate_head_to_head_statistics(
            evaluation_data, sorted_all_engine_names
        )
        display_head_to_head_statistics(h2h_engine_scores, sorted_all_engine_names)


if __name__ == "__main__":
    main()
