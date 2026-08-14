# File: scripts/get_summaries.py

"""Processes StatementGroups to generate concise, search-oriented summaries.

This script iterates through StatementGroups in the database, using their
Lean code, docstring (from the StatementGroup model), existing informal
description, and primary declaration details (name, type) as context for an LLM.
The LLM is tasked with generating a short summary phrase for each group,
optimized for searchability (i.e., what a user might type to find that
statement).

The generated summaries are stored in the 'informal_summary' field of the
StatementGroup. The script supports concurrent LLM calls with rate limiting,
cost tracking, and progress display. Database updates for summaries are performed
individually upon successful generation. The progress bar is updated in real-time
as each asynchronous summary generation attempt completes. This script is intended
to be run after 'lean_to_english.py' has populated the 'informal_description' field.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# --- Project Path Setup for Imports ---
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- Dependency Imports ---
try:
    from sqlalchemy import create_engine, inspect, select
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session, selectinload, sessionmaker
except ImportError:
    print(
        "Error: Missing 'sqlalchemy'. Install with 'pip install sqlalchemy'",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: Missing 'tqdm'. Install with 'pip install tqdm'", file=sys.stderr)
    sys.exit(1)


# --- Project Model & Config/LLM Imports ---
try:
    from dev_tools.config import APP_CONFIG, get_gemini_api_key
    from dev_tools.llm_caller import GeminiClient, GeminiCostTracker
    from lean_explore.shared.models.db import Declaration, StatementGroup
except ImportError as e:
    print(
        f"Error: Could not import project modules: {e}\n\n"
        "POSSIBLE CAUSES & SOLUTIONS:\n"
        "1. RUNNING LOCATION: Ensure you are running this script from the "
        "project's ROOT directory,\n"
        "   e.g., 'python scripts/get_summaries.py'.\n\n"
        "2. FOR 'lean_explore' (e.g., models, config):\n"
        "   - RECOMMENDED: The 'lean_explore' package might not be installed "
        "in your current Python environment.\n"
        "     From your project root, run: 'pip install -e .'\n"
        "     This installs it in editable mode, making it available.\n"
        "   - Alternatively (if not using editable install), ensure the 'src/' "
        "directory is correctly added to sys.path.\n\n"
        "3. FOR 'dev_tools.llm_caller':\n"
        "   - The 'dev_tools/' directory (containing 'llm_caller.py' and an "
        "'__init__.py' file)\n"
        "     must be at the project root.\n"
        "   - The script attempts to add the project root to sys.path;\n"
        "     verify this is working for your environment if issues persist.\n"
        "     You might need to set your PYTHONPATH environment variable to "
        "include the project root.\n\n"
        f"Current sys.path (at time of error): {sys.path}",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as e:
    print(
        f"An unexpected error occurred during project module import: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MAX_CONCURRENT_LLM_CALLS = 20
DEFAULT_SUMMARY_PROMPT_TEMPLATE_PATH = "scripts/summary_prompt_template.txt"
DEFAULT_DATABASE_URL = "sqlite:///data/lean_explore_data.db"

DEFAULT_SUMMARY_PROMPT_TEXT = """You are an expert in the Lean theorem prover and \
formal mathematics.
Your task is to generate a summary phrase for the following Lean code.
This summary should capture the essence of the Lean code in a way that would help \
someone search for and find it. \
Think: "What keywords or short phrase would someone type into a search engine if \
they were looking for this specific concept, theorem, definition, or utility?"
The summary MUST be a short phrase, not a full sentence or explanation. \
Your summary should aim to be almost entirely informal mathematics; \
it should not have Lean code in it unless this is necessary. \
It should be like a title or a very specific keyword tag.
The summary should also be distinctive enough to differentiate similar blocks of \
Lean code:
While there is only one definition of a scheme in Mathlib, there are multiple \
versions of the fundamental theorem of calculus.

Here is the information about the statement block:

Primary Declaration Name: {primary_lean_name}
Primary Declaration Type: {primary_decl_type}

Lean Code:
```lean
{statement_text}
```

Docstring (if available):
{docstring_context}

Informal English Description (previously generated):
{informal_description_context}

Based on all the above, provide ONLY the short summary phrase optimized for \
searchability and discoverability:
"""


# --- Helper Functions ---


def load_prompt_template(template_path_str: str, default_template_text: str) -> str:
    """Loads a prompt template from a file or returns a default text.

    Args:
        template_path_str: The path to the prompt template file.
        default_template_text: The default template text to use if the file
            is not found or an error occurs.

    Returns:
        The loaded or default prompt template string.
    """
    try:
        template_path = Path(template_path_str).resolve()
        if template_path.exists():
            logger.info("Loading summary prompt template from %s", template_path)
            return template_path.read_text(encoding="utf-8")
        logger.warning(
            "Prompt template file '%s' not found. Using default prompt text.",
            template_path,
        )
    except Exception as e:
        logger.error(
            "Error loading prompt template from %s: %s. Using default prompt text.",
            template_path_str,
            e,
            exc_info=True,
        )
    return default_template_text


def load_groups_for_summary_processing(
    session: Session, startover: bool = False
) -> List[StatementGroup]:
    """Loads StatementGroups that require summary generation.

    Fetches StatementGroups, prioritizing those without an existing
    'informal_summary' (unless 'startover' is True). It also ensures that
    essential fields like 'statement_text' and 'informal_description'
    are present. It eagerly loads primary declaration details needed for the prompt.

    Args:
        session: The SQLAlchemy session object.
        startover: If True, all applicable groups are considered for processing,
            regardless of existing summaries.

    Returns:
        A list of StatementGroup ORM objects to be processed.
    """
    logger.info("Loading StatementGroups for summary generation...")
    stmt = (
        select(StatementGroup)
        .options(
            selectinload(StatementGroup.primary_declaration).load_only(
                Declaration.lean_name, Declaration.decl_type
            )
        )
        .where(StatementGroup.statement_text.is_not(None))
        .where(StatementGroup.informal_description.is_not(None))
    )

    if not startover:
        stmt = stmt.where(StatementGroup.informal_summary.is_(None))

    db_groups = session.execute(stmt).scalars().unique().all()

    logger.info(
        "Found %d StatementGroups meeting criteria for summary generation.",
        len(db_groups),
    )
    return list(db_groups)


async def _generate_summary_for_single_group_async(
    sg_orm_item: StatementGroup,
    client: GeminiClient,
    semaphore: asyncio.Semaphore,
    prompt_template: str,
    is_test_mode: bool,
    pbar: Optional[tqdm],
) -> Tuple[int, Optional[str], Optional[Exception]]:
    """Generates a summary for a single StatementGroup using the LLM.

    This internal helper formats the prompt, calls the LLM (rate-limited by
    the semaphore), handles the result, and updates the progress bar upon completion.

    Args:
        sg_orm_item: The StatementGroup ORM object to process.
        client: The GeminiClient instance.
        semaphore: The asyncio Semaphore for concurrent call limiting.
        prompt_template: The loaded prompt template string.
        is_test_mode: Flag indicating if running in test mode.
        pbar: The tqdm progress bar instance to update.

    Returns:
        A tuple containing: (group_id, generated_summary_text or None, error or None).
    """
    group_id = sg_orm_item.id
    primary_decl_name = "UnknownPrimaryDeclaration"
    primary_decl_type = "UnknownType"

    if sg_orm_item.primary_declaration:
        primary_decl_name = (
            sg_orm_item.primary_declaration.lean_name or primary_decl_name
        )
        primary_decl_type = (
            sg_orm_item.primary_declaration.decl_type or primary_decl_type
        )

    docstring_context = (
        sg_orm_item.docstring
        if sg_orm_item.docstring and sg_orm_item.docstring.strip()
        else "No docstring available for this statement group."
    )
    informal_desc_context = (
        sg_orm_item.informal_description
        if sg_orm_item.informal_description and sg_orm_item.informal_description.strip()
        else "No informal description available for this statement group."
    )

    try:
        prompt = prompt_template.format(
            primary_lean_name=primary_decl_name,
            primary_decl_type=primary_decl_type,
            statement_text=sg_orm_item.statement_text,
            docstring_context=docstring_context,
            informal_description_context=informal_desc_context,
        )

        if is_test_mode:
            print(
                f"\n--- LLM PROMPT for Summary (Group ID: {group_id}) ---\n"
                f"Primary Decl: {primary_decl_name} ({primary_decl_type})\n"
                f"Source File: {sg_orm_item.source_file}\n"
                f"{prompt}"
                f"\n--- END LLM PROMPT ---"
            )

        async with semaphore:
            generated_text = await client.generate(prompt=prompt)

        if generated_text and generated_text.strip():
            summary_text = generated_text.strip()
            if is_test_mode:
                print(
                    f"\n--- Test Output for Summary (Group ID: {group_id}) ---\n"
                    f"Statement Text:\n{sg_orm_item.statement_text[:200]}...\n"
                    f"Generated Summary:\n{summary_text}\n"
                    f"---------------------------------------------"
                )
            return group_id, summary_text, None

        logger.warning(
            "LLM returned empty summary for Group %d (%s).",
            group_id,
            primary_decl_name,
        )
        return group_id, None, None

    except Exception as llm_err:
        logger.error(
            "LLM call failed for summary generation (Group ID %d, Name: %s): %s",
            group_id,
            primary_decl_name,
            llm_err,
            exc_info=is_test_mode,
        )
        return group_id, None, llm_err
    finally:
        if pbar:
            pbar.update(1)


async def generate_summaries_for_statement_groups(
    db_url: str,
    test_limit: Optional[int] = None,
    startover: bool = False,
) -> bool:
    """Orchestrates StatementGroup summary generation using an LLM.

    Loads group data, then processes groups by generating summaries via
    concurrent LLM calls. Updates the database with generated summaries
    individually or prints to console if in test mode. The progress bar
    is updated as each asynchronous summary generation attempt completes.

    Args:
        db_url: SQLAlchemy database URL.
        test_limit: If set, process at most N items in test mode (no DB writes).
        startover: If True, re-generate summaries for all applicable groups.

    Returns:
        True if all targeted groups were attempted for summary generation,
        False otherwise (e.g., due to critical errors).
    """
    is_test_mode = test_limit is not None
    mode_desc_str = (
        f"Test Mode (limit {test_limit}, NO DB writes)" if is_test_mode else "Live Mode"
    )
    if startover:
        mode_desc_str += " [STARTOVER]"

    logger.info(
        "--- Starting LLM Group Summary Generation Workflow (%s) ---", mode_desc_str
    )
    logger.info("Using database: %s", db_url)

    summaries_config = APP_CONFIG.get("get_summaries", {})
    prompt_template_path_str = summaries_config.get(
        "prompt_template_path", DEFAULT_SUMMARY_PROMPT_TEMPLATE_PATH
    )
    prompt_template = load_prompt_template(
        prompt_template_path_str, DEFAULT_SUMMARY_PROMPT_TEXT
    )

    max_concurrent_calls = summaries_config.get(
        "max_concurrent_llm_calls", DEFAULT_MAX_CONCURRENT_LLM_CALLS
    )
    if not (isinstance(max_concurrent_calls, int) and max_concurrent_calls > 0):
        logger.warning(
            "Invalid max_concurrent_llm_calls '%s' in config, using default %d.",
            max_concurrent_calls,
            DEFAULT_MAX_CONCURRENT_LLM_CALLS,
        )
        max_concurrent_calls = DEFAULT_MAX_CONCURRENT_LLM_CALLS
    logger.info("Max concurrent LLM calls for summaries: %d", max_concurrent_calls)
    llm_semaphore = asyncio.Semaphore(max_concurrent_calls)

    max_gather_batch = summaries_config.get("max_gather_batch_size")
    if max_gather_batch is not None and not (
        isinstance(max_gather_batch, int) and max_gather_batch > 0
    ):
        logger.warning(
            "Invalid max_gather_batch_size '%s' in config, applying no limit.",
            max_gather_batch,
        )
        max_gather_batch = None
    logger.info(
        "Max items per gather batch for summaries: %s", max_gather_batch or "No limit"
    )

    api_key = get_gemini_api_key()
    if not api_key:
        logger.error("GEMINI_API_KEY not found. Cannot initialize LLM client.")
        return False

    client: GeminiClient
    try:
        llm_config = APP_CONFIG.get("llm", {})
        generation_model_name = llm_config.get("generation_model")
        if not generation_model_name:
            logger.error(
                "Configuration error: 'llm.generation_model' is not set. "
                "Cannot initialize GeminiClient."
            )
            return False

        cost_tracker = GeminiCostTracker(model_costs_override=APP_CONFIG.get("costs"))
        client = GeminiClient(
            api_key=api_key,
            cost_tracker=cost_tracker,
            default_generation_model=generation_model_name,
        )
        logger.info(
            "GeminiClient initialized for summaries, using Generation Model: %s",
            generation_model_name,
        )
    except Exception as e:
        logger.error(
            "Failed to initialize GeminiClient for summaries: %s", e, exc_info=True
        )
        return False

    engine = None
    SessionLocal = None
    try:
        engine = create_engine(db_url, echo=False)
        with engine.connect():  # Test connection
            inspector = inspect(engine)
            if not inspector.has_table(StatementGroup.__tablename__):
                logger.error(
                    "Database missing required table: %s",
                    StatementGroup.__tablename__,
                )
                return False
            sg_columns = [
                col["name"]
                for col in inspector.get_columns(StatementGroup.__tablename__)
            ]
            if "informal_summary" not in sg_columns and not is_test_mode:
                logger.error(
                    "DB table '%s' missing 'informal_summary' column. "
                    "Please update your schema (e.g., add the column).",
                    StatementGroup.__tablename__,
                )
                return False
            if "informal_description" not in sg_columns:
                logger.error(
                    "DB table '%s' missing 'informal_description' column. "
                    "Ensure 'lean_to_english.py' has run successfully.",
                    StatementGroup.__tablename__,
                )
                return False
            if "docstring" not in sg_columns:
                logger.error(
                    "DB table '%s' missing 'docstring' column. "
                    "This field is required for summary generation.",
                    StatementGroup.__tablename__,
                )
                return False

        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
            expire_on_commit=False,
        )
        logger.info(
            "Database connection for summaries successful. Schema checks passed "
            "(target column existence check skipped in test mode if applicable)."
        )
    except Exception as e:
        logger.error(
            "Failed to setup database connection for summaries: %s", e, exc_info=True
        )
        return False

    processed_items_count = 0
    summary_success_count = 0
    summary_failure_count = 0
    pbar: Optional[tqdm] = None
    session_obj_for_processing: Optional[Session] = None

    try:
        with SessionLocal() as session_obj_for_processing:
            groups_to_process_list = load_groups_for_summary_processing(
                session_obj_for_processing, startover
            )

            total_to_process_initially = len(groups_to_process_list)
            if total_to_process_initially == 0:
                logger.info(
                    "No StatementGroups found needing summary generation. Exiting."
                )
                if pbar:
                    pbar.close()
                return True

            pbar_total = (
                min(test_limit, total_to_process_initially)
                if is_test_mode
                else total_to_process_initially
            )
            pbar = tqdm(total=pbar_total, desc="Generating Summaries", unit="group")
            initial_cost_str = f"${client.cost_tracker.get_total_cost():.4f}"
            pbar.set_postfix_str(
                f"Cost: {initial_cost_str} | OK: 0 | Fail: 0 | "
                f"{mode_desc_str.split(' [')[0]}"
            )

            item_idx = 0
            while item_idx < total_to_process_initially:
                if is_test_mode and pbar.n >= test_limit:
                    logger.info(
                        "Reached test limit of %d items for summaries.", test_limit
                    )
                    break

                num_remaining_total = total_to_process_initially - item_idx
                current_gather_size = num_remaining_total
                if max_gather_batch is not None:
                    current_gather_size = min(current_gather_size, max_gather_batch)

                if is_test_mode:
                    remaining_in_test_limit = test_limit - pbar.n
                    current_gather_size = min(
                        current_gather_size, remaining_in_test_limit
                    )

                if current_gather_size <= 0:
                    break

                current_batch_sgs = groups_to_process_list[
                    item_idx : item_idx + current_gather_size
                ]
                item_idx += len(current_batch_sgs)

                tasks = [
                    _generate_summary_for_single_group_async(
                        sg_orm,
                        client,
                        llm_semaphore,
                        prompt_template,
                        is_test_mode,
                        pbar,
                    )
                    for sg_orm in current_batch_sgs
                ]

                results_from_gather = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                to_write = []  # type: List[Tuple[int, str]]
                for res_item in results_from_gather:
                    processed_items_count += 1

                    if isinstance(res_item, Exception):
                        summary_failure_count += 1
                    else:
                        res_group_id, summary_text, error_obj = res_item
                        if error_obj:
                            summary_failure_count += 1
                        elif summary_text:
                            summary_success_count += 1
                            to_write.append((res_group_id, summary_text))
                        else:
                            summary_failure_count += 1

                if to_write and not is_test_mode:
                    batch_mappings = [
                        {"id": gid, "informal_summary": summary}
                        for (gid, summary) in to_write
                    ]
                    try:
                        session_obj_for_processing.bulk_update_mappings(
                            StatementGroup, batch_mappings
                        )
                        session_obj_for_processing.commit()
                    except SQLAlchemyError as db_err_bulk:
                        session_obj_for_processing.rollback()
                        logger.error(
                            "Bulk update failed for %d rows in this batch: %s",
                            len(batch_mappings),
                            db_err_bulk,
                        )
                        summary_failure_count += len(batch_mappings)

                if pbar:
                    cost_str = f"${client.cost_tracker.get_total_cost():.4f}"
                    pbar.set_postfix_str(
                        f"Cost: {cost_str} | OK: {summary_success_count} | "
                        f"Fail: {summary_failure_count} | "
                        f"{mode_desc_str.split(' [')[0]}"
                    )

    except SQLAlchemyError as db_err:
        logger.error(
            "Database error during summary processing: %s", db_err, exc_info=True
        )
        if session_obj_for_processing and session_obj_for_processing.is_active:
            session_obj_for_processing.rollback()
        return False
    except Exception as e:
        logger.error("Unexpected error during summary processing: %s", e, exc_info=True)
        if session_obj_for_processing and session_obj_for_processing.is_active:
            session_obj_for_processing.rollback()
        return False
    finally:
        if pbar:
            pbar.close()
            print()
        if client and client.cost_tracker:
            summary_stats = client.cost_tracker.get_summary()
            logger.info(
                "--- Final Estimated LLM Costs for Summaries (%s) ---", mode_desc_str
            )
            logger.info(
                "Total Estimated Cost: $%.4f",
                summary_stats.get("total_estimated_cost", 0.0),
            )
            final_processed_count = (
                pbar.n if pbar and pbar.n > 0 else processed_items_count
            )
            logger.info(
                "LLM Calls Attempted (Items Processed): %d", final_processed_count
            )
            logger.info("Summaries Generated Successfully: %d", summary_success_count)
            logger.info(
                "LLM Calls Failed or Yielded No Summary: %d", summary_failure_count
            )
            for model_name, usage in summary_stats.get("usage_by_model", {}).items():
                cost_value = usage.get("cost")
                calls_value = usage.get("calls", 0)

                logger.info(
                    "    Model: %s -> Calls: %d, Cost: %s",
                    model_name,
                    calls_value,
                    f"${cost_value:.4f}"
                    if isinstance(cost_value, (int, float))
                    else cost_value,
                )
            logger.info("---------------------------------------------------")

    if is_test_mode:
        final_processed_count = pbar.n if pbar and pbar.n > 0 else processed_items_count
        logger.info(
            "Summary Test mode finished. Attempted %d items (limit was %d). "
            "Successes: %d, Failures: %d.",
            final_processed_count,
            test_limit or 0,
            summary_success_count,
            summary_failure_count,
        )
        return True

    final_attempted_count = pbar.n if pbar else processed_items_count
    all_targeted_items_attempted = final_attempted_count >= total_to_process_initially

    if all_targeted_items_attempted:
        if (
            summary_failure_count == 0
            and summary_success_count == total_to_process_initially
        ):
            logger.info(
                "Live mode: Successfully generated and saved summaries for all %d "
                "targeted groups.",
                total_to_process_initially,
            )
        else:
            logger.warning(
                "Live mode: Attempted all %d targeted groups. %d summaries succeeded "
                "and were saved, %d attempts failed or yielded no summary.",
                total_to_process_initially,
                summary_success_count,
                summary_failure_count,
            )
        return True
    else:
        logger.error(
            "Live mode: Summary generation INCOMPLETE. Attempted (%d) != Initial "
            "Total (%d). Successes: %d, Failures: %d. Check logs for errors.",
            final_attempted_count,
            total_to_process_initially,
            summary_success_count,
            summary_failure_count,
        )
        return False


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the summary generation script.

    Returns:
        An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate informal English summaries for Lean StatementGroups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=DEFAULT_DATABASE_URL,
        help=(f"SQLAlchemy database URL. (default: {DEFAULT_DATABASE_URL})"),
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="Enable test mode: process at most N items. LLM calls are active, "
        "but no database writes are performed.",
    )
    parser.add_argument(
        "--startover",
        action="store_true",
        help="Ignore existing summaries in the database and re-generate for all "
        "applicable StatementGroups.",
    )
    return parser.parse_args()


async def main_async():
    """Asynchronous main function to run the summary generation logic."""
    args = parse_arguments()

    startover_msg_part = " [STARTOVER ENABLED]" if args.startover else ""
    current_mode_log_header = "--- "
    if args.test is not None:
        current_mode_log_header += (
            f"TEST MODE (Max {args.test} items, LLM ON, DB OFF){startover_msg_part}"
        )
    else:
        current_mode_log_header += f"LIVE MODE{startover_msg_part}"
    current_mode_log_header += " ---"
    logger.info(current_mode_log_header)

    if args.db_url.startswith("sqlite:///"):
        db_path_str = args.db_url.split("sqlite:///", 1)[1]
        if db_path_str != ":memory:":
            db_file = Path(db_path_str).resolve()
            logger.info("Using SQLite database for summaries: %s", db_file)
            if not db_file.exists() and not args.test:
                logger.warning(
                    "SQLite database file for summaries not found: %s", db_file
                )
        else:
            logger.info("Using in-memory SQLite database for summaries.")

    success = await generate_summaries_for_statement_groups(
        db_url=args.db_url,
        test_limit=args.test,
        startover=args.startover,
    )

    final_mode_desc = "Test Mode" if args.test is not None else "Live Mode"
    if success:
        logger.info(
            "--- Summary Generation Completed (%s%s) ---",
            final_mode_desc,
            startover_msg_part,
        )
        sys.exit(0)
    else:
        logger.error(
            "--- Summary Generation FAILED or Incomplete (%s%s) ---",
            final_mode_desc,
            startover_msg_part,
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("\nSummary generation interrupted by user (KeyboardInterrupt).")
        sys.exit(130)
    except Exception as e:
        logger.critical(
            "An unexpected critical error occurred in main_async for summaries: %s",
            e,
            exc_info=True,
        )
        sys.exit(1)
