# benchmarking/gather_search_data.py

"""Module for gathering search data from various sources."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import requests
from tqdm.asyncio import tqdm  # For progress bar

try:
    from lean_explore.api.client import Client as LeanExploreApiClient
    from lean_explore.cli import config_utils as lean_explore_api_config_utils
    from lean_explore.local.service import Service as LeanExploreLocalService

    LEANEXPLORE_PACKAGE_AVAILABLE = True
except ImportError:
    LEANEXPLORE_PACKAGE_AVAILABLE = False
    LeanExploreApiClient = None  # type: ignore
    LeanExploreLocalService = None  # type: ignore
    lean_explore_api_config_utils = None  # type: ignore
    print(
        "Warning: lean_explore package or its core components (API client, "
        "Local service) not found. LeanExplore search functionalities will "
        "be limited or unavailable."
    )


# --- Configuration Constants ---
BENCHMARK_QUERIES_FILE = "benchmarking/queries.txt"  # Input file for search queries.
OUTPUT_JSON_FILE = "benchmarking/search_results.json"  # Output file for search results.
NUM_LEANSEARCH_RESULTS = 10  # Default number of results from LeanSearch.
MOOGLE_LIMIT = 10  # Default number of results from Moogle.
NUM_LEANEXPLORE_RESULTS = 10  # Default number of results from LeanExplore.
MAX_CONCURRENT_REQUESTS = 9  # Max concurrent search operations.
# -----------------------------


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Gather search data from various engines for benchmarking."
    )
    parser.add_argument(
        "--leanexplore-mode",
        type=str,
        choices=["api", "local"],
        default="api",
        help=(
            "Mode for LeanExplore: 'api' to use the remote API, "
            "'local' to use the local search service. Default is 'api'."
        ),
    )
    parser.add_argument(
        "--update-only-leanexplore",
        action="store_true",
        help=(
            "If set, only fetch/update data for LeanExplore. "
            "Data for other search engines will be loaded from the existing "
            "output file if available and not re-fetched."
        ),
    )
    return parser.parse_args()


def search_leansearch(query: str, num_results: int = 50) -> dict:
    """Sends a POST request to LeanSearch to perform a search.

    Args:
        query: The natural language query string.
        num_results: The desired number of results (defaults to 50, max 100).

    Returns:
        A dictionary containing the JSON response from LeanSearch,
        or an error dictionary if the request fails.
    """
    if not (1 <= num_results <= 100):
        print(
            "Warning: num_results for LeanSearch should be between 1 and 100."
            " Clamping to range."
        )
        num_results = max(1, min(100, num_results))

    url = "https://leansearch.net/search"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
    }
    payload = {
        "query": [query],
        "num_results": num_results,
    }

    response_obj = None
    try:
        response_obj = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=10
        )
        response_obj.raise_for_status()
        return response_obj.json()
    except requests.exceptions.RequestException as e:
        print(f"Error LeanSearch '{query[:30]}...': {e}")
        status_code = e.response.status_code if e.response is not None else None
        return {"error": str(e), "status_code": status_code}
    except json.JSONDecodeError as e:
        print(f"JSON Error LeanSearch '{query[:30]}...': {e}")
        raw_text = response_obj.text if response_obj is not None else None
        return {
            "error": "Invalid JSON response from LeanSearch",
            "details": str(e),
            "raw_response": raw_text,
        }


def search_moogle(query: str, limit: int = 10) -> dict:
    """Sends a POST request to Moogle to perform a search.

    Args:
        query: The natural language query string.
        limit: The desired number of results.

    Returns:
        A dictionary containing the JSON response from Moogle,
        or an error dictionary if the request fails.
    """
    url = "https://www.moogle.ai/api/search"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) "
            "Gecko/20100101 Firefox/138.0"
        ),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.5",
        "Origin": "https://www.moogle.ai",
        "Referer": "https://www.moogle.ai/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Sites": "same-origin",
        "DNT": "1",
        "Sec-GPC": "1",
        "TE": "trailers",
        "Connection": "keep-alive",
    }
    payload = [{"isFind": False, "contents": query}]

    response_obj = None
    try:
        response_obj = requests.post(url, headers=headers, json=payload, timeout=10)
        response_obj.raise_for_status()
        return response_obj.json()
    except requests.exceptions.RequestException as e:
        print(f"Error Moogle '{query[:30]}...': {e}")
        status_code = e.response.status_code if e.response is not None else None
        return {"error": str(e), "status_code": status_code}
    except json.JSONDecodeError as e:
        print(f"JSON Error Moogle '{query[:30]}...': {e}")
        raw_text = response_obj.text if response_obj is not None else None
        return {
            "error": "Invalid JSON response from Moogle",
            "details": str(e),
            "raw_response": raw_text,
        }


async def initialize_leanexplore_api_client() -> Optional[LeanExploreApiClient]:
    """Initializes and returns the LeanExplore API client.

    Attempts to load the API key using lean_explore's config utilities,
    then falls back to the LEANEXPLORE_API_KEY environment variable.

    Returns:
        An initialized LeanExploreApiClient instance if an API key is successfully
        found and the client can be instantiated, otherwise None.
    """
    if not LEANEXPLORE_PACKAGE_AVAILABLE or LeanExploreApiClient is None:
        tqdm.write("LeanExplore API client components not available.")
        return None
    if lean_explore_api_config_utils is None:
        tqdm.write(
            "LeanExplore API config utilities not available for API key loading."
        )
        # Fallback to environment variable only
        api_key = os.getenv("LEANEXPLORE_API_KEY")
    else:
        api_key = None
        try:
            api_key = lean_explore_api_config_utils.load_api_key()
        except Exception as e:
            tqdm.write(
                "Note: Could not load API key using lean_explore.cli.config_utils:"
                f" {e}. Trying environment variable."
            )
        if not api_key:
            api_key = os.getenv("LEANEXPLORE_API_KEY")

    if not api_key:
        tqdm.write(
            "LeanExplore API key not found via config_utils or "
            "LEANEXPLORE_API_KEY environment variable."
        )
        tqdm.write("Please configure an API key to use LeanExplore remote search.")
        tqdm.write(
            "You can configure it using 'leanexplore configure api-key YOUR_KEY_HERE' "
            "or by setting the LEANEXPLORE_API_KEY environment variable."
        )
        return None

    try:
        client = LeanExploreApiClient(api_key=api_key)
        tqdm.write("LeanExplore API Client initialized successfully.")
        return client
    except Exception as e:
        tqdm.write(f"Failed to initialize LeanExplore API Client: {e}")
        return None


def initialize_leanexplore_local_service() -> Optional[LeanExploreLocalService]:
    """Initializes and returns the LeanExplore Local Service.

    Returns:
        An initialized LeanExploreLocalService instance if successful,
        otherwise None.
    """
    if not LEANEXPLORE_PACKAGE_AVAILABLE or LeanExploreLocalService is None:
        tqdm.write("LeanExplore Local Service components not available.")
        return None
    try:
        service = LeanExploreLocalService()
        tqdm.write("LeanExplore Local Service initialized successfully.")
        return service
    except FileNotFoundError as e:
        tqdm.write(
            "Failed to initialize LeanExplore Local Service: Critical data file "
            f"not found. {e}"
        )
        tqdm.write(
            "Please ensure local data (DB, FAISS index, models) is correctly set up."
        )
        tqdm.write("You may need to run 'leanexplore data fetch'.")
        return None
    except RuntimeError as e:
        tqdm.write(
            f"Failed to initialize LeanExplore Local Service: Runtime error. {e}"
        )
        return None
    except Exception as e:
        tqdm.write(
            f"An unexpected error occurred initializing LeanExplore Local Service: {e}"
        )
        return None


async def search_leanexplore_api(
    query_str: str, client: LeanExploreApiClient, limit: int = 10
) -> dict:
    """Searches LeanExplore using the remote API client.

    Args:
        query_str: The natural language query string.
        client: An initialized LeanExplore API Client instance.
        limit: The desired number of results to return (client-side slicing).

    Returns:
        A dictionary containing the search results from LeanExplore,
        or an error dictionary if the request fails.
    """
    try:
        api_search_response = await client.search(query=query_str)

        results_list = []
        if (
            api_search_response
            and hasattr(api_search_response, "results")
            and api_search_response.results
        ):
            for item in api_search_response.results[:limit]:
                results_list.append(item.model_dump())

        return {
            "count": (
                api_search_response.count
                if hasattr(api_search_response, "count")
                else len(results_list)  # Fallback if count attribute missing
            ),
            "processing_time_ms": (
                api_search_response.processing_time_ms
                if hasattr(api_search_response, "processing_time_ms")
                else -1  # Indicate not available
            ),
            "results": results_list,
        }
    except httpx.HTTPStatusError as e_http:
        error_msg = (
            f"LeanExplore API Error for '{query_str[:30]}...': "
            f"{e_http.response.status_code}"
        )
        if hasattr(e_http.response, "text") and e_http.response.text:
            error_msg += f" - {e_http.response.text[:200]}"
        tqdm.write(error_msg)

        error_details: Dict[str, Any] = {"details": str(e_http)}
        if hasattr(e_http.response, "text"):
            error_details["response_text"] = e_http.response.text
        if e_http.response.status_code == 401:
            error_details["hint"] = "LeanExplore API key might be invalid or expired."
        return {
            "error": f"API HTTP Status Error: {e_http.response.status_code}",
            **error_details,
        }
    except httpx.RequestError as e_req:
        tqdm.write(
            f"LeanExplore Network request failed for '{query_str[:30]}...': {e_req}"
        )
        return {"error": f"Network request failed: {str(e_req)}", "query": query_str}
    except Exception as e:
        tqdm.write(
            "An unexpected error occurred searching LeanExplore (API) for "
            f"'{query_str[:30]}...': {e}"
        )
        return {"error": str(e), "query": query_str}


def search_leanexplore_local(
    query_str: str, service: LeanExploreLocalService, limit: int = 10
) -> dict:
    """Searches LeanExplore using the local service.

    Args:
        query_str: The natural language query string.
        service: An initialized LeanExplore Local Service instance.
        limit: The desired number of results.

    Returns:
        A dictionary containing the search results from LeanExplore local service,
        or an error dictionary if the operation fails.
    """
    try:
        api_search_response = service.search(
            query=query_str,
            limit=limit,
            package_filters=["Batteries", "Init", "Lean", "Mathlib", "Std"],
        )

        results_list = []
        if (
            api_search_response
            and hasattr(api_search_response, "results")
            and api_search_response.results
        ):
            for item in api_search_response.results:
                results_list.append(item.model_dump())

        return {
            "count": (
                api_search_response.count
                if hasattr(api_search_response, "count")
                else len(results_list)
            ),
            "total_candidates_considered": (
                api_search_response.total_candidates_considered
                if hasattr(api_search_response, "total_candidates_considered")
                else -1
            ),
            "processing_time_ms": (
                api_search_response.processing_time_ms
                if hasattr(api_search_response, "processing_time_ms")
                else -1
            ),
            "results": results_list,
        }
    except RuntimeError as e_rt:
        tqdm.write(
            f"LeanExplore Local Service runtime error for '{query_str[:30]}...': {e_rt}"
        )
        return {
            "error": f"Local Service runtime error: {str(e_rt)}",
            "query": query_str,
        }
    except Exception as e:
        tqdm.write(
            "An unexpected error occurred searching LeanExplore (Local) for "
            f"'{query_str[:30]}...': {e}"
        )
        return {"error": str(e), "query": query_str}


async def _fetch_leansearch_concurrently(
    query_str: str, num_results: int, semaphore: asyncio.Semaphore
) -> Tuple[str, str, Dict[str, Any]]:
    """Wraps LeanSearch call for concurrent execution with a semaphore.

    Args:
        query_str: The natural language query string.
        num_results: The desired number of results.
        semaphore: An asyncio.Semaphore to limit concurrent requests.

    Returns:
        A tuple containing the original query string, the search engine
        name ("leansearch"), and the JSON response from LeanSearch or an
        error dictionary.
    """
    async with semaphore:
        result = await asyncio.to_thread(search_leansearch, query_str, num_results)
        return query_str, "leansearch", result


async def _fetch_moogle_concurrently(
    query_str: str, limit: int, semaphore: asyncio.Semaphore
) -> Tuple[str, str, Dict[str, Any]]:
    """Wraps Moogle search call for concurrent execution with a semaphore.

    Args:
        query_str: The natural language query string.
        limit: The desired number of results.
        semaphore: An asyncio.Semaphore to limit concurrent requests.

    Returns:
        A tuple containing the original query string, the search engine
        name ("moogle"), and the JSON response from Moogle or an
        error dictionary.
    """
    async with semaphore:
        result = await asyncio.to_thread(search_moogle, query_str, limit)
        return query_str, "moogle", result


async def _fetch_leanexplore_concurrently(
    query_str: str,
    leanexplore_source: Union[LeanExploreApiClient, LeanExploreLocalService],
    leanexplore_mode: str,
    limit: int,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, str, Dict[str, Any]]:
    """Wraps LeanExplore search call for concurrent execution with a semaphore.

    Dispatches to either API or local search based on leanexplore_mode.

    Args:
        query_str: The natural language query string.
        leanexplore_source: An initialized LeanExploreApiClient or
                            LeanExploreLocalService instance.
        leanexplore_mode: Specifies 'api' or 'local' mode.
        limit: The desired number of results.
        semaphore: An asyncio.Semaphore to limit concurrent operations.

    Returns:
        A tuple containing the original query string, the search engine
        name ("leanexplore"), and the JSON response from LeanExplore or an
        error dictionary.
    """
    async with semaphore:
        result: Dict[str, Any]
        if leanexplore_mode == "api":
            if not isinstance(leanexplore_source, LeanExploreApiClient):
                msg = "Internal error: Mismatched LeanExplore source type for API mode."
                tqdm.write(msg)
                return query_str, "leanexplore", {"error": msg}
            result = await search_leanexplore_api(query_str, leanexplore_source, limit)
        elif leanexplore_mode == "local":
            if not isinstance(leanexplore_source, LeanExploreLocalService):
                msg = (
                    "Internal error: Mismatched LeanExplore source type for local mode."
                )
                tqdm.write(msg)
                return query_str, "leanexplore", {"error": msg}
            result = await asyncio.to_thread(
                search_leanexplore_local, query_str, leanexplore_source, limit
            )
        else:
            msg = f"Invalid LeanExplore mode specified: {leanexplore_mode}"
            tqdm.write(msg)
            return query_str, "leanexplore", {"error": msg}
        return query_str, "leanexplore", result


async def gather_all_search_data(
    queries_filepath: str,
    output_filepath: str,
    lean_num_results: int,
    moogle_num_results: int,
    leanexplore_num_results: int,
    max_concurrent_requests: int,
    leanexplore_mode: str,
    update_only_leanexplore: bool,
) -> None:
    """Reads queries, performs searches concurrently, and saves aggregated results.

    Checks for existing results and skips searches based on arguments.
    Limits concurrent search operations using a semaphore and displays progress.

    Args:
        queries_filepath: Path to the text file containing search queries.
        output_filepath: Path to the JSON file for saving results.
        lean_num_results: Number of results for LeanSearch.
        moogle_num_results: Number of results for Moogle.
        leanexplore_num_results: Number of results for LeanExplore.
        max_concurrent_requests: Maximum number of concurrent search operations.
        leanexplore_mode: Mode for LeanExplore ('api' or 'local').
        update_only_leanexplore: If True, only fetch/update LeanExplore data.
    """
    all_query_data_map: Dict[str, Dict[str, Any]] = {}
    tqdm.write("Starting data gathering process...")

    existing_results_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(output_filepath):
        tqdm.write(
            f"Found existing output file: '{output_filepath}'. "
            "Attempting to load previous results."
        )
        try:
            with open(output_filepath, encoding="utf-8") as f_in:
                loaded_data = json.load(f_in)
                if isinstance(loaded_data, list):
                    for item in loaded_data:
                        if isinstance(item, dict) and "query" in item:
                            existing_results_map[item["query"]] = item
                        else:
                            tqdm.write(
                                "Warning: Skipping malformed item in existing data: "
                                f"{str(item)[:100]}..."
                            )
                    tqdm.write(
                        f"Successfully loaded {len(existing_results_map)} existing"
                        " query results."
                    )
                else:
                    tqdm.write(
                        f"Warning: Existing output file '{output_filepath}' does not"
                        " contain a JSON list. It will be overwritten if new data is"
                        " saved."
                    )
        except json.JSONDecodeError:
            tqdm.write(
                f"Warning: Could not decode JSON from '{output_filepath}'. File might"
                " be corrupted. It will be overwritten if new data is saved."
            )
        except Exception as e:
            tqdm.write(
                f"Warning: An error occurred reading existing output file "
                f"'{output_filepath}': {e}. It may be overwritten if new data is"
                " saved."
            )
    else:
        tqdm.write(f"Output file '{output_filepath}' not found. Will create a new one.")

    tqdm.write(f"Attempting to read queries from: '{queries_filepath}'")
    try:
        with open(queries_filepath, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        tqdm.write(
            f"Error: Query file '{queries_filepath}' not found. Please ensure the"
            " file exists."
        )
        return
    except Exception as e:
        tqdm.write(
            f"An unexpected error occurred while reading '{queries_filepath}': {e}"
        )
        return

    if not queries:
        tqdm.write(f"No queries found in '{queries_filepath}'. Exiting.")
        return

    tqdm.write(
        f"Successfully read {len(queries)} unique, non-empty queries to process."
    )

    for query_str_item in queries:
        if query_str_item in existing_results_map:
            all_query_data_map[query_str_item] = existing_results_map[query_str_item]
        else:
            all_query_data_map[query_str_item] = {
                "query": query_str_item,
                "leansearch_results": None,
                "moogle_results": None,
                "leanexplore_results": None,
            }

    semaphore = asyncio.Semaphore(max_concurrent_requests)
    tasks_for_this_run: List[asyncio.Task[Tuple[str, str, Dict[str, Any]]]] = []

    leanexplore_source_instance: Optional[
        Union[LeanExploreApiClient, LeanExploreLocalService]
    ] = None
    can_run_leanexplore = False

    if LEANEXPLORE_PACKAGE_AVAILABLE:
        if leanexplore_mode == "api":
            leanexplore_source_instance = await initialize_leanexplore_api_client()
        elif leanexplore_mode == "local":
            leanexplore_source_instance = initialize_leanexplore_local_service()

        if leanexplore_source_instance:
            can_run_leanexplore = True
        else:
            tqdm.write(
                f"LeanExplore {leanexplore_mode} source initialization failed. "
                "LeanExplore searches will be skipped for this run."
            )
    else:
        tqdm.write(
            "LeanExplore package not available. LeanExplore searches will be skipped."
        )

    tqdm.write("\n--- Identifying and preparing search tasks ---")
    for query_str_item in queries:
        query_needs_processing = False

        # LeanSearch tasks
        if not update_only_leanexplore:
            current_ls_results = all_query_data_map[query_str_item].get(
                "leansearch_results"
            )
            if current_ls_results is None or (
                isinstance(current_ls_results, dict) and "error" in current_ls_results
            ):
                tasks_for_this_run.append(
                    asyncio.create_task(
                        _fetch_leansearch_concurrently(
                            query_str_item, lean_num_results, semaphore
                        )
                    )
                )
                query_needs_processing = True
            else:
                tqdm.write(
                    f"   LeanSearch: Skipping '{query_str_item[:30]}...', "
                    "results exist."
                )

        # Moogle tasks
        if not update_only_leanexplore:
            current_moogle_results = all_query_data_map[query_str_item].get(
                "moogle_results"
            )
            if current_moogle_results is None or (
                isinstance(current_moogle_results, dict)
                and "error" in current_moogle_results
            ):
                tasks_for_this_run.append(
                    asyncio.create_task(
                        _fetch_moogle_concurrently(
                            query_str_item, moogle_num_results, semaphore
                        )
                    )
                )
                query_needs_processing = True
            else:
                tqdm.write(
                    f"   Moogle: Skipping '{query_str_item[:30]}...', results exist."
                )

        # LeanExplore tasks
        if can_run_leanexplore and leanexplore_source_instance:
            current_le_results = all_query_data_map[query_str_item].get(
                "leanexplore_results"
            )
            if current_le_results is None or (
                isinstance(current_le_results, dict) and "error" in current_le_results
            ):
                tasks_for_this_run.append(
                    asyncio.create_task(
                        _fetch_leanexplore_concurrently(
                            query_str_item,
                            leanexplore_source_instance,
                            leanexplore_mode,
                            leanexplore_num_results,
                            semaphore,
                        )
                    )
                )
                query_needs_processing = True
            else:
                tqdm.write(
                    f"   LeanExplore ({leanexplore_mode}): Skipping "
                    f"'{query_str_item[:30]}...', results exist."
                )
        elif not query_needs_processing and update_only_leanexplore:
            pass

    if not tasks_for_this_run:
        tqdm.write(
            "\nNo new search tasks to perform. All results may already be cached or "
            "targets are not selected for update."
        )
    else:
        tqdm.write(
            f"\nProcessing {len(tasks_for_this_run)} search tasks (max "
            f"{max_concurrent_requests} concurrently):"
        )
        all_results_tuples = await tqdm.gather(
            *tasks_for_this_run, desc="Searching", unit="task"
        )
        tqdm.write("All search tasks processed. Aggregating results.")

        for query_str, engine_name, result_data in all_results_tuples:
            if engine_name == "leansearch":
                all_query_data_map[query_str]["leansearch_results"] = result_data
            elif engine_name == "moogle":
                all_query_data_map[query_str]["moogle_results"] = result_data
            elif engine_name == "leanexplore":
                all_query_data_map[query_str]["leanexplore_results"] = result_data

    if (
        leanexplore_mode == "api"
        and isinstance(leanexplore_source_instance, LeanExploreApiClient)
        and hasattr(leanexplore_source_instance, "_client")
        and hasattr(leanexplore_source_instance._client, "is_closed")
        and not leanexplore_source_instance._client.is_closed
    ):
        tqdm.write("Closing LeanExplore API client's underlying HTTP client.")
        await leanexplore_source_instance._client.aclose()

    generic_le_error_message: Optional[str] = None
    if not LEANEXPLORE_PACKAGE_AVAILABLE:
        generic_le_error_message = "LeanExplore package not installed or importable."
    elif not can_run_leanexplore:
        if leanexplore_mode == "api":
            generic_le_error_message = (
                "LeanExplore API client could not be initialized "
                "(e.g., API key issue or client error)."
            )
        else:
            generic_le_error_message = (
                "LeanExplore local service could not be initialized "
                "(e.g., data files missing or service error)."
            )

    if generic_le_error_message:
        tqdm.write(
            f"\nUpdating LeanExplore status for all queries: {generic_le_error_message}"
        )
        for query_str_item in queries:
            current_le_res = all_query_data_map[query_str_item].get(
                "leanexplore_results"
            )
            if current_le_res is None or (
                isinstance(current_le_res, dict) and "error" in current_le_res
            ):
                all_query_data_map[query_str_item]["leanexplore_results"] = {
                    "error": generic_le_error_message
                }

    final_results_list = list(all_query_data_map.values())

    tqdm.write(
        f"\nAll queries processed. Attempting to save collected data to: "
        f"'{output_filepath}'"
    )
    try:
        with open(output_filepath, "w", encoding="utf-8") as f_out:
            json.dump(final_results_list, f_out, indent=4, ensure_ascii=False)
        tqdm.write(
            f"Successfully saved data for {len(final_results_list)} queries to"
            f" '{output_filepath}'."
        )
    except Exception as e:
        tqdm.write(
            f"An error occurred while writing results to '{output_filepath}': {e}"
        )


async def main() -> None:
    """Main asynchronous function to orchestrate the data gathering process."""
    args = parse_arguments()
    await gather_all_search_data(
        queries_filepath=BENCHMARK_QUERIES_FILE,
        output_filepath=OUTPUT_JSON_FILE,
        lean_num_results=NUM_LEANSEARCH_RESULTS,
        moogle_num_results=MOOGLE_LIMIT,
        leanexplore_num_results=NUM_LEANEXPLORE_RESULTS,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
        leanexplore_mode=args.leanexplore_mode,
        update_only_leanexplore=args.update_only_leanexplore,
    )


if __name__ == "__main__":
    asyncio.run(main())
