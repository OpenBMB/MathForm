# src/lean_explore/local/service.py

"""Provides a service class for local Lean data exploration.

This module defines the Service class, which offers methods to search,
retrieve by ID, and get dependencies for statement groups using local
data assets (SQLite database, FAISS index, and embedding models).
"""

import logging
import time
from typing import List, Optional, Union, overload

import faiss  # For type hinting if needed
from sentence_transformers import SentenceTransformer  # For type hinting if needed
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session as SQLAlchemySessionType
from sqlalchemy.orm import joinedload, sessionmaker

from lean_explore import defaults
from lean_explore.shared.models.api import (
    APICitationsResponse,
    APIPrimaryDeclarationInfo,
    APISearchResponse,
    APISearchResultItem,
)
from lean_explore.shared.models.db import (
    StatementGroup,
    StatementGroupDependency,
)

from .search import (
    load_embedding_model,
    load_faiss_assets,
    perform_search,
)

logger = logging.getLogger(__name__)


class Service:
    """A service for interacting with local Lean explore data.

    This service loads necessary data assets (embedding model, FAISS index,
    database connection) upon initialization using default paths and parameters
    derived from the active toolchain. It provides methods for searching
    statement groups, retrieving them by ID, and fetching dependencies (citations).

    Attributes:
        embedding_model: The loaded sentence embedding model.
        faiss_index: The loaded FAISS index.
        text_chunk_id_map: A list mapping FAISS indices to text chunk IDs.
        engine: The SQLAlchemy engine for database connections.
        SessionLocal: The SQLAlchemy sessionmaker for creating sessions.
        default_faiss_k (int): Default number of FAISS neighbors to retrieve.
        default_pagerank_weight (float): Default weight for PageRank.
        default_text_relevance_weight (float): Default weight for text relevance.
        default_name_match_weight (float): Default weight for name matching (BM25).
        default_semantic_similarity_threshold (float): Default similarity threshold.
        default_results_limit (int): Default limit for search results.
        default_faiss_nprobe (int): Default nprobe for FAISS IVF indexes.
        default_faiss_oversampling_factor (int): Default oversampling factor for
            FAISS when package filters are active.
    """

    def __init__(self):
        """Initializes the Service by loading data assets and configurations.

        Checks for essential local data files first, then loads the
        embedding model, FAISS index, and sets up the database engine.
        Paths for data assets are sourced from `lean_explore.defaults`.

        Raises:
            FileNotFoundError: If essential data files (DB, FAISS index, map)
                                are not found at their expected locations.
            RuntimeError: If the embedding model fails to load or if other
                        critical initialization steps (like database connection
                        after file checks) fail.
        """
        logger.info("Initializing local Service...")
        try:
            defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(
                "User toolchains base directory ensured: "
                f"{defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR}"
            )
        except OSError as e:
            logger.error(
                f"Could not create user toolchains base directory "
                f"{defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR}: {e}"
            )

        db_path = defaults.DEFAULT_DB_PATH
        db_url = defaults.DEFAULT_DB_URL
        is_file_db = db_url.startswith("sqlite:///")

        if is_file_db and not db_path.exists():
            error_message = (
                f"Database file not found at the expected location: {db_path}\n"
                "Please run 'leanexplore data fetch' to download the data toolchain."
            )
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        logger.info(f"Loading embedding model: {defaults.DEFAULT_EMBEDDING_MODEL_NAME}")
        self.embedding_model = load_embedding_model(defaults.DEFAULT_EMBEDDING_MODEL_NAME)
        if self.embedding_model is None:
            logger.info("Using remote embedding service; local model not loaded.")

        faiss_index_path = defaults.DEFAULT_FAISS_INDEX_PATH
        faiss_map_path = defaults.DEFAULT_FAISS_MAP_PATH
        logger.info(
            f"Attempting to load FAISS assets: Index='{faiss_index_path}', "
            f"Map='{faiss_map_path}'"
        )

        faiss_assets = load_faiss_assets(str(faiss_index_path), str(faiss_map_path))
        if faiss_assets[0] is None or faiss_assets[1] is None:
            error_message = (
                "Failed to load critical FAISS assets (index or ID map).\n"
                "Expected at:\n"
                f"  Index path: {faiss_index_path}\n"
                f"  ID map path: {faiss_map_path}\n"
                "Please run 'leanexplore data fetch' to download or update the data "
                "toolchain."
            )
            logger.error(error_message)
            raise FileNotFoundError(error_message)
        self.faiss_index: faiss.Index = faiss_assets[0]
        self.text_chunk_id_map: List[str] = faiss_assets[1]
        logger.info("FAISS assets loaded successfully.")

        logger.info(f"Initializing database engine. Expected DB path: {db_path}")
        try:
            self.engine = create_engine(db_url)
            # Test connection
            with self.engine.connect():  # type: ignore[attr-defined] # sqlalchemy stubs might be incomplete
                logger.info("Database connection successful.")
            self.SessionLocal: sessionmaker[SQLAlchemySessionType] = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
        except OperationalError as oe:
            guidance = (
                "Please check your database configuration or connection parameters."
            )
            if is_file_db:
                guidance = (
                    f"The database file at '{db_path}' might be corrupted, "
                    "inaccessible, or not a valid SQLite file. "
                    "Consider running 'leanexplore data fetch' to get a fresh copy."
                )
            logger.error(
                f"Failed to initialize database engine or connection to {db_url}: "
                f"{oe}\n{guidance}"
            )
            raise RuntimeError(
                f"Database initialization failed: {oe}. {guidance}"
            ) from oe
        except Exception as e:
            logger.error(
                f"Unexpected error during database engine initialization: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Database initialization failed unexpectedly: {e}"
            ) from e

        self.default_faiss_k: int = defaults.DEFAULT_FAISS_K
        self.default_pagerank_weight: float = defaults.DEFAULT_PAGERANK_WEIGHT
        self.default_text_relevance_weight: float = (
            defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT
        )
        self.default_name_match_weight: float = defaults.DEFAULT_NAME_MATCH_WEIGHT
        self.default_semantic_similarity_threshold: float = (
            defaults.DEFAULT_SEM_SIM_THRESHOLD
        )
        self.default_results_limit: int = defaults.DEFAULT_RESULTS_LIMIT
        self.default_faiss_nprobe: int = defaults.DEFAULT_FAISS_NPROBE
        self.default_faiss_oversampling_factor: int = (
            defaults.DEFAULT_FAISS_OVERSAMPLING_FACTOR
        )

        logger.info("Local Service initialized successfully.")

    def _serialize_sg_to_api_item(self, sg_orm: StatementGroup) -> APISearchResultItem:
        """Converts a StatementGroup ORM obj to APISearchResultItem Pydantic model.

        Args:
            sg_orm: The SQLAlchemy StatementGroup object.

        Returns:
            An APISearchResultItem Pydantic model instance.
        """
        primary_decl_info = APIPrimaryDeclarationInfo(
            lean_name=sg_orm.primary_declaration.lean_name
            if sg_orm.primary_declaration
            else None
        )
        return APISearchResultItem(
            id=sg_orm.id,
            primary_declaration=primary_decl_info,
            source_file=sg_orm.source_file,
            range_start_line=sg_orm.range_start_line,
            display_statement_text=sg_orm.display_statement_text,
            statement_text=sg_orm.statement_text,
            docstring=sg_orm.docstring,
            informal_description=sg_orm.informal_description,
        )

    def _perform_one_search(
        self,
        query: str,
        package_filters: Optional[List[str]],
        limit: Optional[int],
    ) -> APISearchResponse:
        """Helper to perform and package a single local search.

        Args:
            query: The search query string.
            package_filters: An optional list of package names to filter results by.
            limit: An optional limit on the number of results to return.

        Returns:
            An APISearchResponse for the given query.
        """
        start_time = time.time()
        actual_limit = limit if limit is not None else self.default_results_limit

        with self.SessionLocal() as session:
            try:
                ranked_results_orm = perform_search(
                    session=session,
                    query_string=query,
                    model=self.embedding_model,
                    faiss_index=self.faiss_index,
                    text_chunk_id_map=self.text_chunk_id_map,
                    faiss_k=self.default_faiss_k,
                    pagerank_weight=self.default_pagerank_weight,
                    text_relevance_weight=self.default_text_relevance_weight,
                    name_match_weight=self.default_name_match_weight,
                    log_searches=True,
                    selected_packages=package_filters,
                    semantic_similarity_threshold=(
                        self.default_semantic_similarity_threshold
                    ),
                    faiss_nprobe=self.default_faiss_nprobe,
                    faiss_oversampling_factor=self.default_faiss_oversampling_factor,
                )
            except Exception as e:
                logger.error(
                    f"Error during perform_search execution: {e}", exc_info=True
                )
                raise

        api_results = [
            self._serialize_sg_to_api_item(sg_obj)
            for sg_obj, _scores in ranked_results_orm
        ]

        final_results = api_results[:actual_limit]
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        return APISearchResponse(
            query=query,
            packages_applied=package_filters,
            results=final_results,
            count=len(final_results),
            total_candidates_considered=len(api_results),
            processing_time_ms=processing_time_ms,
        )

    @overload
    def search(
        self,
        query: str,
        package_filters: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> APISearchResponse: ...

    @overload
    def search(
        self,
        query: List[str],
        package_filters: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[APISearchResponse]: ...

    def search(
        self,
        query: Union[str, List[str]],
        package_filters: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Union[APISearchResponse, List[APISearchResponse]]:
        """Performs a local search for statement groups.

        This method can handle a single query string or a list of query strings.
        When a list is provided, searches are performed serially.

        Args:
            query: The search query string or a list of query strings.
            package_filters: An optional list of package names to filter results by.
            limit: An optional limit on the number of results to return.
                    If None, defaults.DEFAULT_RESULTS_LIMIT is used.

        Returns:
            An APISearchResponse object if a single query was provided, or a
            list of APISearchResponse objects if a list of queries was provided.

        Raises:
            RuntimeError: If service not properly initialized (e.g., assets missing).
            Exception: Propagates exceptions from `perform_search`.
        """
        if self.faiss_index is None or self.text_chunk_id_map is None:
            logger.error(
                "Search service assets not loaded. Service may not have initialized "
                "correctly."
            )
            raise RuntimeError(
                "Search service assets not loaded. Please ensure data has been fetched."
            )

        was_single_query = isinstance(query, str)
        queries = [query] if was_single_query else query
        results = []

        for q in queries:
            response = self._perform_one_search(q, package_filters, limit)
            results.append(response)

        if was_single_query:
            return results[0]
        return results

    @overload
    def get_by_id(self, group_id: int) -> Optional[APISearchResultItem]: ...

    @overload
    def get_by_id(self, group_id: List[int]) -> List[Optional[APISearchResultItem]]: ...

    def get_by_id(
        self, group_id: Union[int, List[int]]
    ) -> Union[Optional[APISearchResultItem], List[Optional[APISearchResultItem]]]:
        """Retrieves a specific statement group by its ID from local data.

        Args:
            group_id: The unique identifier of the statement group, or a list of IDs.

        Returns:
            An APISearchResultItem if a single ID was found, None if not found.
            A list of Optional[APISearchResultItem] if a list of IDs was provided.
        """
        was_single_id = isinstance(group_id, int)
        group_ids = [group_id] if was_single_id else group_id
        results = []

        with self.SessionLocal() as session:
            for g_id in group_ids:
                try:
                    stmt_group_orm = (
                        session.query(StatementGroup)
                        .options(joinedload(StatementGroup.primary_declaration))
                        .filter(StatementGroup.id == g_id)
                        .first()
                    )
                    if stmt_group_orm:
                        results.append(self._serialize_sg_to_api_item(stmt_group_orm))
                    else:
                        results.append(None)
                except SQLAlchemyError as e:
                    logger.error(
                        f"Database error in get_by_id for group_id {g_id}: {e}",
                        exc_info=True,
                    )
                    results.append(None)
                except Exception as e:
                    logger.error(
                        f"Unexpected error in get_by_id for group_id {g_id}: {e}",
                        exc_info=True,
                    )
                    results.append(None)

        if was_single_id:
            return results[0]
        return results

    @overload
    def get_dependencies(self, group_id: int) -> Optional[APICitationsResponse]: ...

    @overload
    def get_dependencies(
        self, group_id: List[int]
    ) -> List[Optional[APICitationsResponse]]: ...

    def get_dependencies(
        self, group_id: Union[int, List[int]]
    ) -> Union[Optional[APICitationsResponse], List[Optional[APICitationsResponse]]]:
        """Retrieves citations for a specific statement group from local data.

        Citations are the statement groups that the specified group_id depends on.

        Args:
            group_id: The unique ID of the source group, or a list of IDs.

        Returns:
            An APICitationsResponse if a single ID was provided, or a list of
            Optional[APICitationsResponse] if a list of IDs was given. Returns
            None for IDs that are not found or cause an error.
        """
        was_single_id = isinstance(group_id, int)
        group_ids = [group_id] if was_single_id else group_id
        results = []

        with self.SessionLocal() as session:
            for g_id in group_ids:
                try:
                    source_group_exists = (
                        session.query(StatementGroup.id)
                        .filter(StatementGroup.id == g_id)
                        .first()
                    )
                    if not source_group_exists:
                        logger.warning(
                            f"Source statement group ID {g_id} not found for "
                            "dependency lookup."
                        )
                        results.append(None)
                        continue

                    cited_target_groups_orm = (
                        session.query(StatementGroup)
                        .join(
                            StatementGroupDependency,
                            StatementGroup.id
                            == StatementGroupDependency.target_statement_group_id,
                        )
                        .filter(
                            StatementGroupDependency.source_statement_group_id == g_id
                        )
                        .options(joinedload(StatementGroup.primary_declaration))
                        .all()
                    )

                    citations_api_items = [
                        self._serialize_sg_to_api_item(sg_orm)
                        for sg_orm in cited_target_groups_orm
                    ]

                    results.append(
                        APICitationsResponse(
                            source_group_id=g_id,
                            citations=citations_api_items,
                            count=len(citations_api_items),
                        )
                    )
                except SQLAlchemyError as e:
                    logger.error(
                        f"Database error in get_dependencies for group_id {g_id}: {e}",
                        exc_info=True,
                    )
                    results.append(None)
                except Exception as e:
                    logger.error(
                        f"Unexpected error in get_dependencies for "
                        f"group_id {g_id}: {e}",
                        exc_info=True,
                    )
                    results.append(None)

        if was_single_id:
            return results[0]
        return results
