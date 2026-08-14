# File: scripts/update_primary_declarations.py

"""Updates primary declarations for statement groups in the database.

This script iterates through all StatementGroup entries and, for each group,
evaluates if its primary_decl_id should be changed. The new primary
declaration is chosen based on a multi-phase logic involving prefix
relationships and hierarchical presence of declaration names in the
statement group's text. The StatementGroup's docstring is also
updated to match the new primary declaration's docstring.
"""

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import joinedload, scoped_session, sessionmaker

# Determine project root and add src to sys.path to import models
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lean_explore.shared.models.db import (  # noqa: E402
    Base,
    Declaration,
    StatementGroup,
)

# --- Configuration ---
DATABASE_FILE = PROJECT_ROOT / "data" / "lean_explore_data.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE.resolve()}"
# Adjustable batch size for committing database updates
BATCH_SIZE = 10000

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _is_word_in_text(word: str, text: str) -> bool:
    """Checks if a word is in text as a whole word.

    Args:
        word (str): The word to search for.
        text (str): The text to search within.

    Returns:
        bool: True if the word is found as a whole word, False otherwise.
    """
    if not word or not text:
        return False
    escaped_word = re.escape(word)
    pattern = r"\b" + escaped_word + r"\b"
    return bool(re.search(pattern, text))


def _get_declarations_found_in_code_hierarchically(
    declarations: List[Declaration], statement_text: str
) -> List[Declaration]:
    """Performs a hierarchical search for declaration names in statement text.

    It iterates through suffix levels of declaration names (from full FQN
    down to the last component). If matches are found at a certain level,
    only those declarations are returned, and deeper levels are not checked.

    Args:
        declarations (List[Declaration]): The list of declarations to check.
        statement_text (str): The text of the statement group.

    Returns:
        List[Declaration]: A list of declarations found at the earliest
                           (longest name part) suffix level. Empty if none found.
    """
    s_found_in_code: List[Declaration] = []
    if not declarations or not statement_text:
        return s_found_in_code

    max_levels = 0
    for decl in declarations:
        if decl.lean_name:
            max_levels = max(max_levels, len(decl.lean_name.split(".")))

    if max_levels == 0:
        return s_found_in_code

    for level in range(max_levels):  # level 0 to max_levels-1
        current_level_matches: List[Declaration] = []
        for decl in declarations:
            if not decl.lean_name:
                continue

            name_parts = decl.lean_name.split(".")
            if level >= len(name_parts):  # Cannot strip this many components
                name_part_to_check = None
            else:
                # Suffix at level 'l' means stripping 'l' leading components
                name_part_to_check = ".".join(name_parts[level:])

            if name_part_to_check and _is_word_in_text(
                name_part_to_check, statement_text
            ):
                current_level_matches.append(decl)

        if current_level_matches:
            s_found_in_code = current_level_matches
            break  # Terminate: matches found at this level

    return s_found_in_code


def update_primary_declarations_in_db(db_url: str):
    """Updates the primary_decl_id for StatementGroups.

    Iterates through StatementGroups, applying a multi-phase logic to select
    a new primary declaration. This involves an initial selection based on
    prefix relationships to the current primary, followed by a refinement
    based on hierarchical presence of declaration names in the statement
    group's source text. Updates are committed in batches.

    Args:
        db_url (str): The database connection URL.
    """
    engine = create_engine(db_url)
    Base.metadata.bind = engine
    SessionLocal = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    session = SessionLocal()

    updated_groups_count = 0
    processed_groups_count = 0
    pending_updates_in_batch = 0

    try:
        logger.info("Fetching all statement groups with their declarations...")
        result = session.execute(
            select(StatementGroup).options(
                joinedload(StatementGroup.primary_declaration),
                joinedload(StatementGroup.declarations),
            )
        )
        stmt_groups = result.unique().scalars().all()

        total_groups = len(stmt_groups)
        logger.info(f"Found {total_groups} statement groups to process.")

        for i, sg in enumerate(stmt_groups):
            processed_groups_count += 1
            if (i + 1) % 1000 == 0 or (i + 1) == total_groups:
                logger.info(f"Processing group {i + 1}/{total_groups} (ID: {sg.id})...")

            current_primary_decl = sg.primary_declaration
            if not current_primary_decl:
                logger.warning(
                    f"StatementGroup ID {sg.id} has no current primary declaration. "
                    "Skipping."
                )
                continue

            member_declarations = sg.declarations
            if not member_declarations:
                logger.warning(
                    f"StatementGroup ID {sg.id} has no member declarations. Skipping."
                )
                continue

            non_internal_members = [d for d in member_declarations if not d.is_internal]
            candidate_pool = (
                non_internal_members if non_internal_members else member_declarations
            )
            if not candidate_pool:  # Ensure pool is not empty for logic below
                logger.warning(
                    f"StatementGroup ID {sg.id} has an empty candidate pool. Skipping."
                )
                continue

            # Phase 1: Initial Candidate Selection (Prefix Logic)
            # Find the shortest declaration that is a prefix of current_primary_decl.
            # If none, prefix_contender_decl remains current_primary_decl.
            initial_shorter_prefix_candidate = None
            for decl_ph1 in candidate_pool:
                if decl_ph1.id == current_primary_decl.id:
                    continue

                if (
                    decl_ph1.lean_name
                    and current_primary_decl.lean_name
                    and len(decl_ph1.lean_name) < len(current_primary_decl.lean_name)
                    and current_primary_decl.lean_name.startswith(decl_ph1.lean_name)
                ):
                    if initial_shorter_prefix_candidate is None or len(
                        decl_ph1.lean_name
                    ) < len(initial_shorter_prefix_candidate.lean_name):
                        initial_shorter_prefix_candidate = decl_ph1

            prefix_contender_decl: Declaration
            if initial_shorter_prefix_candidate:
                prefix_contender_decl = initial_shorter_prefix_candidate
            else:
                prefix_contender_decl = current_primary_decl

            # Phase 2: Code Presence Evaluation & Final Selection
            s_found_in_code = _get_declarations_found_in_code_hierarchically(
                candidate_pool, sg.statement_text
            )

            final_primary_decl: Optional[Declaration] = None

            if not s_found_in_code:
                # Case B: No declarations found in code
                final_primary_decl = prefix_contender_decl
            else:
                # Case A: Some declarations found in code
                if len(s_found_in_code) == 1:
                    final_primary_decl = s_found_in_code[0]
                else:
                    # Apply selection criteria to S_found_in_code
                    # 1. Prefer by prefix relationship
                    eligible_candidates = s_found_in_code

                    prefix_based_candidates: List[Declaration] = []
                    is_any_candidate_a_prefix = False
                    for d1 in s_found_in_code:
                        is_d1_prefix_of_other = False
                        for d2 in s_found_in_code:
                            if d1.id == d2.id or not d1.lean_name or not d2.lean_name:
                                continue
                            if len(d1.lean_name) < len(
                                d2.lean_name
                            ) and d2.lean_name.startswith(d1.lean_name):
                                is_d1_prefix_of_other = True
                                is_any_candidate_a_prefix = True
                                break
                        if is_d1_prefix_of_other:
                            prefix_based_candidates.append(d1)

                    if is_any_candidate_a_prefix:
                        eligible_candidates = prefix_based_candidates
                    # If no candidate is a prefix of another,
                    # eligible_candidates remains S_found_in_code.

                    # 2. Sort by lean_name length, then by ID for stability
                    eligible_candidates.sort(
                        key=lambda d: (
                            len(d.lean_name) if d.lean_name else float("inf"),
                            d.id,
                        )
                    )

                    best_after_sort = eligible_candidates[0]

                    # 3. If tied (same shortest length), prefer prefix_contender_decl
                    # if among them
                    shortest_len = (
                        len(best_after_sort.lean_name)
                        if best_after_sort.lean_name
                        else float("inf")
                    )

                    tied_shortest_candidates = [
                        decl
                        for decl in eligible_candidates
                        if (len(decl.lean_name) if decl.lean_name else float("inf"))
                        == shortest_len
                    ]

                    is_prefix_contender_chosen = False
                    if len(tied_shortest_candidates) > 1:
                        for tied_decl in tied_shortest_candidates:
                            if tied_decl.id == prefix_contender_decl.id:
                                final_primary_decl = tied_decl
                                is_prefix_contender_chosen = True
                                break

                    if not is_prefix_contender_chosen:
                        final_primary_decl = best_after_sort  # First from sorted list

            # Update StatementGroup if the chosen primary is different
            if final_primary_decl and sg.primary_decl_id != final_primary_decl.id:
                old_primary_name = current_primary_decl.lean_name
                new_primary_name = final_primary_decl.lean_name
                logger.info(
                    f"Updating StatementGroup ID {sg.id}: "
                    f"Old primary: '{old_primary_name}' "
                    f"(ID: {sg.primary_decl_id}), "
                    f"New primary: '{new_primary_name}' "
                    f"(ID: {final_primary_decl.id})"
                )
                sg.primary_decl_id = final_primary_decl.id
                sg.docstring = final_primary_decl.docstring
                session.add(sg)
                updated_groups_count += 1
                pending_updates_in_batch += 1

            if pending_updates_in_batch >= BATCH_SIZE:
                logger.info(
                    f"Committing batch of {pending_updates_in_batch} updates. "
                    f"Total updates so far: {updated_groups_count}."
                )
                session.commit()
                pending_updates_in_batch = 0

        if session.dirty:
            logger.info(
                f"Committing final {len(list(session.dirty))} outstanding changes..."
            )
            session.commit()

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        session.rollback()
    finally:
        logger.info(
            f"Finished processing. Total groups processed: {processed_groups_count}. "
            f"Total groups updated: {updated_groups_count}."
        )
        session.close()


if __name__ == "__main__":
    logger.info("Starting script to update primary declarations with enhanced logic.")
    update_primary_declarations_in_db(DATABASE_URL)
    logger.info("Script finished.")
