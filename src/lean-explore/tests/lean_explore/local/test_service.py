# tests/local/test_service.py

"""Tests for the lean_explore.local.service.Service class.

This module focuses on testing the initialization and core functionalities
of the Service class, which orchestrates local data asset loading and
provides methods for interacting with the local Lean explore data.
"""

import logging
import pathlib
from typing import TYPE_CHECKING, List
from unittest.mock import MagicMock

import faiss
import pytest
from sentence_transformers import SentenceTransformer
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session as SQLAlchemySession

from lean_explore import defaults
from lean_explore.local.service import Service
from lean_explore.shared.models.api import (
    APICitationsResponse,
    APIPrimaryDeclarationInfo,
    APISearchResponse,
    APISearchResultItem,
)

# Import your ORM and API models
from lean_explore.shared.models.db import (
    Declaration,
    StatementGroup,
    StatementGroupDependency,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestServiceInitialization:
    """Tests focused on the __init__ method of the Service class."""

    def test_successful_initialization(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
        mocker: "MockerFixture",
    ):
        """Verifies successful instantiation of Service when all assets load.

        This test ensures that when all underlying asset loading functions
        (mocked) succeed and prerequisite files (like the database file)
        are present in their isolated locations, the Service initializes
        correctly and sets its internal attributes as expected.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Fixture mocking embedding model loading.
            mock_load_faiss_assets: Fixture mocking FAISS asset loading.
            mock_sqlalchemy_engine_setup: Fixture mocking SQLAlchemy engine setup.
            mocker: Pytest-mock's mocker fixture.
        """
        defaults.DEFAULT_DB_PATH.touch()

        mock_model_instance = MagicMock(spec=SentenceTransformer)
        mock_faiss_index_instance = MagicMock(spec=faiss.Index)
        mock_id_map_instance: List[str] = ["id1", "id2"]

        mock_load_embedding_model.return_value = mock_model_instance
        mock_load_faiss_assets.return_value = (
            mock_faiss_index_instance,
            mock_id_map_instance,
        )

        service = Service()

        assert service.embedding_model is mock_model_instance
        assert service.faiss_index is mock_faiss_index_instance
        assert service.text_chunk_id_map is mock_id_map_instance
        assert service.engine is mock_sqlalchemy_engine_setup.return_value
        assert service.SessionLocal is not None

        assert service.default_faiss_k == defaults.DEFAULT_FAISS_K
        assert service.default_pagerank_weight == defaults.DEFAULT_PAGERANK_WEIGHT
        assert (
            service.default_text_relevance_weight
            == defaults.DEFAULT_TEXT_RELEVANCE_WEIGHT
        )
        assert service.default_name_match_weight == defaults.DEFAULT_NAME_MATCH_WEIGHT
        assert (
            service.default_semantic_similarity_threshold
            == defaults.DEFAULT_SEM_SIM_THRESHOLD
        )
        assert service.default_results_limit == defaults.DEFAULT_RESULTS_LIMIT
        assert service.default_faiss_nprobe == defaults.DEFAULT_FAISS_NPROBE
        assert (
            service.default_faiss_oversampling_factor
            == defaults.DEFAULT_FAISS_OVERSAMPLING_FACTOR
        )

        mock_load_embedding_model.assert_called_once_with(
            defaults.DEFAULT_EMBEDDING_MODEL_NAME
        )
        mock_load_faiss_assets.assert_called_once_with(
            str(defaults.DEFAULT_FAISS_INDEX_PATH), str(defaults.DEFAULT_FAISS_MAP_PATH)
        )
        mock_sqlalchemy_engine_setup.assert_called_once_with(defaults.DEFAULT_DB_URL)
        service.engine.connect.assert_called_once()

    def test_init_fails_if_embedding_model_load_fails(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
    ):
        """Tests Service init raises RuntimeError if embedding model loading fails.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
        """
        defaults.DEFAULT_DB_PATH.touch()
        mock_load_embedding_model.return_value = None

        expected_msg = (
            f"Failed to load embedding model: {defaults.DEFAULT_EMBEDDING_MODEL_NAME}"
        )
        with pytest.raises(RuntimeError, match=expected_msg):
            Service()

    def test_init_fails_if_faiss_assets_load_fails(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
    ):
        """Tests Service init raises FileNotFoundError if FAISS assets are None.

        Covers cases where either the FAISS index or the ID map fails to load,
        signified by `mock_load_faiss_assets` returning `None` for one of
        its tuple elements.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
        """
        defaults.DEFAULT_DB_PATH.touch()
        base_error_message_regex = (
            r"Failed to load critical FAISS assets \(index or ID map\)"
        )

        mock_load_faiss_assets.return_value = (None, ["id1"])
        with pytest.raises(FileNotFoundError, match=base_error_message_regex):
            Service()

        mock_load_faiss_assets.return_value = (MagicMock(spec=faiss.Index), None)
        with pytest.raises(FileNotFoundError, match=base_error_message_regex):
            Service()

    def test_init_fails_if_db_file_does_not_exist(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
    ):
        """Tests Service init raises FileNotFoundError if the DB file is missing.

        This test ensures that the `Service` checks for the database file's
        existence at the path specified by `defaults.DEFAULT_DB_PATH` (which
        is an isolated path in this test context) and fails appropriately.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
        """
        if defaults.DEFAULT_DB_PATH.exists():
            defaults.DEFAULT_DB_PATH.unlink()

        expected_msg_regex = (
            "Database file not found at the expected location: "
            f"{str(defaults.DEFAULT_DB_PATH).replace('.', r'.')}"
        )
        with pytest.raises(FileNotFoundError, match=expected_msg_regex):
            Service()

    def test_init_fails_if_db_connection_fails(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
    ):
        """Tests Service init raises RuntimeError if DB connection fails.

        Even if the database file exists, this test simulates a failure
        during the engine connection attempt.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
        """
        defaults.DEFAULT_DB_PATH.touch()

        mock_engine_instance = mock_sqlalchemy_engine_setup.return_value
        mock_dbapi_exception = Exception("DBAPI connection error detail")
        mock_engine_instance.connect.side_effect = OperationalError(
            "Test DB connection failed", {"param": 1}, mock_dbapi_exception
        )

        expected_match_str = (
            r"Database initialization failed: \(builtins\.Exception\) DBAPI connection "
            r"error detail\n\[SQL: Test DB connection failed\]\n"
            r"\[parameters: \{'param': 1\}\]\n"
            r"\(Background on this error at: https://sqlalche\.me/e/20/e3q8\)\. "
            r"The database file at .* might be corrupted"
        )
        with pytest.raises(RuntimeError, match=expected_match_str):
            Service()

    def test_init_proceeds_despite_toolchain_dir_OSError_if_exist_ok(
        self,
        isolated_data_paths: pathlib.Path,
        mock_load_embedding_model: MagicMock,
        mock_load_faiss_assets: MagicMock,
        mock_sqlalchemy_engine_setup: MagicMock,
        caplog: pytest.LogCaptureFixture,
        mocker: "MockerFixture",
    ):
        """Verifies Service init proceeds if toolchain base dir mkdir is handled.

        The `isolated_data_paths` fixture already creates the toolchain base
        directory. The `mkdir(parents=True, exist_ok=True)` call in
        `Service.__init__` should not raise an error. This test confirms
        that no error log regarding this directory creation appears and the
        expected info log is present.

        Args:
            isolated_data_paths: Fixture redirecting default data paths.
            mock_load_embedding_model: Mock for embedding model loading.
            mock_load_faiss_assets: Mock for FAISS asset loading.
            mock_sqlalchemy_engine_setup: Mock for SQLAlchemy engine setup.
            caplog: Pytest fixture to capture log output.
            mocker: Pytest-mock's mocker fixture.
        """
        defaults.DEFAULT_DB_PATH.touch()
        mock_load_embedding_model.return_value = MagicMock(spec=SentenceTransformer)
        mock_load_faiss_assets.return_value = (MagicMock(spec=faiss.Index), ["id_map"])

        with caplog.at_level(logging.INFO):
            service = Service()
            assert service is not None

        assert (
            "User toolchains base directory ensured: "
            f"{str(defaults.LEAN_EXPLORE_TOOLCHAINS_BASE_DIR)}" in caplog.text
        )
        assert "Could not create user toolchains base directory" not in caplog.text


@pytest.fixture
def initialized_service(
    isolated_data_paths: pathlib.Path,
    mock_load_embedding_model: MagicMock,
    mock_load_faiss_assets: MagicMock,
    mock_sqlalchemy_engine_setup: MagicMock,
    mocker: "MockerFixture",
) -> Service:
    """Provides a successfully initialized Service instance for method testing.

    This fixture ensures that all dependencies for Service.__init__ are
    mocked for success, and dummy files (like the database file) are
    touched to satisfy existence checks.

    Args:
        isolated_data_paths: Fixture that redirects default data paths.
        mock_load_embedding_model: Fixture that mocks embedding model loading.
        mock_load_faiss_assets: Fixture that mocks FAISS asset loading.
        mock_sqlalchemy_engine_setup: Fixture that mocks SQLAlchemy engine setup.
        mocker: Pytest-mock's mocker fixture.

    Returns:
        Service: A successfully initialized Service instance.
    """
    defaults.DEFAULT_DB_PATH.touch()

    mock_model_instance = MagicMock(spec=SentenceTransformer)
    mock_faiss_index_instance = MagicMock(spec=faiss.Index)
    mock_id_map_instance: List[str] = ["id1", "id2"]

    mock_load_embedding_model.return_value = mock_model_instance
    mock_load_faiss_assets.return_value = (
        mock_faiss_index_instance,
        mock_id_map_instance,
    )
    return Service()


class TestServiceMethods:
    """Tests for the data access and utility methods of the Service class."""

    def test_serialize_sg_to_api_item(self, initialized_service: Service):
        """Tests StatementGroup ORM object to APISearchResultItem conversion.

        Args:
            initialized_service: A successfully initialized Service instance.
        """
        mock_sg_orm = MagicMock(spec=StatementGroup)
        mock_sg_orm.id = 1
        mock_sg_orm.source_file = "Test.lean"
        mock_sg_orm.range_start_line = 10
        mock_sg_orm.display_statement_text = "display text"
        mock_sg_orm.statement_text = "full statement text"
        mock_sg_orm.docstring = "This is a docstring."
        mock_sg_orm.informal_description = "An informal description."

        mock_primary_decl = MagicMock(spec=Declaration)
        mock_primary_decl.lean_name = "Nat.add"
        mock_sg_orm.primary_declaration = mock_primary_decl

        api_item = initialized_service._serialize_sg_to_api_item(mock_sg_orm)

        assert isinstance(api_item, APISearchResultItem)
        assert api_item.id == mock_sg_orm.id
        assert api_item.primary_declaration.lean_name == "Nat.add"
        assert api_item.source_file == "Test.lean"
        assert api_item.range_start_line == 10
        assert api_item.display_statement_text == "display text"
        assert api_item.statement_text == "full statement text"
        assert api_item.docstring == "This is a docstring."
        assert api_item.informal_description == "An informal description."

    def test_serialize_sg_to_api_item_no_primary_decl(
        self, initialized_service: Service
    ):
        """Tests serialization when the StatementGroup has no primary_declaration.

        Args:
            initialized_service: A successfully initialized Service instance.
        """
        mock_sg_orm = MagicMock(spec=StatementGroup)
        mock_sg_orm.id = 2
        mock_sg_orm.primary_declaration = None
        mock_sg_orm.source_file = "Another.lean"
        mock_sg_orm.range_start_line = 1
        mock_sg_orm.statement_text = "some text"
        mock_sg_orm.display_statement_text = None
        mock_sg_orm.docstring = None
        mock_sg_orm.informal_description = None

        api_item = initialized_service._serialize_sg_to_api_item(mock_sg_orm)
        assert api_item.primary_declaration.lean_name is None

    def test_get_by_id_single_id(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
    ):
        """Tests retrieving a StatementGroup by a single ID.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session with schema created.
            mocker: Pytest-mock's mocker fixture.
        """
        decl = Declaration(id=1, lean_name="Nat.zero", decl_type="axiom")
        sg = StatementGroup(
            id=101,
            text_hash="hash1",
            statement_text="def zero := 0",
            source_file="Init.Nat.Zero.lean",
            range_start_line=5,
            range_start_col=0,
            range_end_line=5,
            range_end_col=15,
            primary_decl_id=decl.id,
            primary_declaration=decl,
        )
        db_session.add_all([decl, sg])
        db_session.commit()

        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=db_session
        )

        result = initialized_service.get_by_id(101)

        assert isinstance(result, APISearchResultItem)
        assert result.id == 101
        assert result.primary_declaration.lean_name == "Nat.zero"

    def test_get_by_id_batch_ids(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
    ):
        """Tests retrieving multiple StatementGroups by a list of IDs.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session.
            mocker: Pytest-mock's mocker fixture.
        """
        decl1 = Declaration(id=1, lean_name="Test1", decl_type="def")
        sg1 = StatementGroup(
            id=101,
            text_hash="h1",
            statement_text="t1",
            source_file="f1",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            primary_decl_id=decl1.id,
        )
        decl2 = Declaration(id=2, lean_name="Test2", decl_type="def")
        sg2 = StatementGroup(
            id=102,
            text_hash="h2",
            statement_text="t2",
            source_file="f2",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            primary_decl_id=decl2.id,
        )
        db_session.add_all([decl1, decl2, sg1, sg2])
        db_session.commit()

        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=db_session
        )

        results = initialized_service.get_by_id([101, 102, 999])  # 999 is not found

        assert isinstance(results, list)
        assert len(results) == 3
        assert isinstance(results[0], APISearchResultItem)
        assert results[0].id == 101
        assert isinstance(results[1], APISearchResultItem)
        assert results[1].id == 102
        assert results[2] is None

    def test_get_by_id_sqlalchemy_error(
        self,
        initialized_service: Service,
        mocker: "MockerFixture",
        caplog: pytest.LogCaptureFixture,
    ):
        """Tests get_by_id when an SQLAlchemyError occurs.

        Args:
            initialized_service: An initialized Service instance.
            mocker: Pytest-mock's mocker fixture.
            caplog: Pytest fixture to capture log output.
        """
        mock_session_query = mocker.MagicMock()
        mock_session_query.side_effect = SQLAlchemyError("DB query failed")

        mock_session = mocker.MagicMock(spec=SQLAlchemySession)
        query_chain = (
            mock_session.query.return_value.options.return_value.filter.return_value
        )
        query_chain.first = mock_session_query

        mock_session_local_cm = mocker.MagicMock()
        mock_session_local_cm.__enter__.return_value = mock_session
        mock_session_local_cm.__exit__.return_value = None

        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=mock_session_local_cm
        )

        with caplog.at_level(logging.ERROR):
            result = initialized_service.get_by_id([101])

        assert isinstance(result, list)
        assert result[0] is None
        assert "Database error in get_by_id for group_id 101" in caplog.text
        assert "DB query failed" in caplog.text

    def test_get_dependencies_batch_ids(
        self,
        initialized_service: Service,
        db_session: SQLAlchemySession,
        mocker: "MockerFixture",
    ):
        """Tests retrieving dependencies for a list of StatementGroup IDs.

        Args:
            initialized_service: An initialized Service instance.
            db_session: An in-memory SQLite session.
            mocker: Pytest-mock's mocker fixture.
        """
        decl1 = Declaration(id=1, lean_name="Src1", decl_type="def")
        decl2 = Declaration(id=2, lean_name="Src2", decl_type="def")
        decl3 = Declaration(id=3, lean_name="Tgt1", decl_type="def")
        sg_source1 = StatementGroup(
            id=201,
            text_hash="h1",
            statement_text="t1",
            source_file="f1",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            primary_decl_id=1,
            primary_declaration=decl1,
        )
        sg_source2 = StatementGroup(
            id=202,
            text_hash="h2",
            statement_text="t2",
            source_file="f2",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            primary_decl_id=2,
            primary_declaration=decl2,
        )
        sg_target = StatementGroup(
            id=301,
            text_hash="h3",
            statement_text="t3",
            source_file="f3",
            range_start_line=1,
            range_start_col=1,
            range_end_line=1,
            range_end_col=1,
            primary_decl_id=3,
            primary_declaration=decl3,
        )
        dep1 = StatementGroupDependency(
            source_statement_group_id=201, target_statement_group_id=301
        )
        db_session.add_all(
            [decl1, decl2, decl3, sg_source1, sg_source2, sg_target, dep1]
        )
        db_session.commit()

        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=db_session
        )

        results = initialized_service.get_dependencies([201, 202, 999])

        assert isinstance(results, list)
        assert len(results) == 3
        # Result 1: Found with dependencies
        assert isinstance(results[0], APICitationsResponse)
        assert results[0].source_group_id == 201
        assert results[0].count == 1
        assert results[0].citations[0].id == 301
        # Result 2: Found with no dependencies
        assert isinstance(results[1], APICitationsResponse)
        assert results[1].source_group_id == 202
        assert results[1].count == 0
        # Result 3: Not found
        assert results[2] is None

    def test_search_single_query(
        self, initialized_service: Service, mocker: "MockerFixture"
    ):
        """Tests the search method with a single query.

        Args:
            initialized_service: An initialized Service instance.
            mocker: Pytest-mock's mocker fixture.
        """
        mock_perform_search = mocker.patch(
            "lean_explore.local.service.perform_search",
            return_value=[(MagicMock(id=10), {"final_score": 0.9})],
        )
        mocker.patch.object(
            initialized_service,
            "_serialize_sg_to_api_item",
            return_value=APISearchResultItem(
                id=10,
                primary_declaration=APIPrimaryDeclarationInfo(),
                source_file="f",
                range_start_line=1,
                statement_text="t",
            ),
        )
        mock_session_cm = MagicMock()
        mock_session_cm.__enter__.return_value = MagicMock(spec=SQLAlchemySession)
        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=mock_session_cm
        )

        query = "test query"
        response = initialized_service.search(query=query)

        assert isinstance(response, APISearchResponse)
        assert response.query == query
        assert len(response.results) == 1
        assert response.results[0].id == 10
        mock_perform_search.assert_called_once()

    def test_search_batch_queries(
        self, initialized_service: Service, mocker: "MockerFixture"
    ):
        """Tests the search method with a list of queries.

        Args:
            initialized_service: An initialized Service instance.
            mocker: Pytest-mock's mocker fixture.
        """
        mock_perform_search = mocker.patch("lean_explore.local.service.perform_search")
        mock_serialize = mocker.patch.object(
            initialized_service, "_serialize_sg_to_api_item"
        )

        # Simulate different results for each query
        mock_perform_search.side_effect = [
            [(MagicMock(id=10), {"final_score": 0.9})],
            [(MagicMock(id=20), {"final_score": 0.8})],
        ]
        mock_serialize.side_effect = [
            APISearchResultItem(
                id=10,
                primary_declaration=APIPrimaryDeclarationInfo(),
                source_file="f",
                range_start_line=1,
                statement_text="t",
            ),
            APISearchResultItem(
                id=20,
                primary_declaration=APIPrimaryDeclarationInfo(),
                source_file="f",
                range_start_line=1,
                statement_text="t",
            ),
        ]

        mock_session_cm = MagicMock()
        mock_session_cm.__enter__.return_value = MagicMock(spec=SQLAlchemySession)
        mocker.patch.object(
            initialized_service, "SessionLocal", return_value=mock_session_cm
        )

        queries = ["query one", "query two"]
        responses = initialized_service.search(query=queries)

        assert isinstance(responses, list)
        assert len(responses) == 2
        assert mock_perform_search.call_count == 2

        assert responses[0].query == queries[0]
        assert responses[0].results[0].id == 10
        assert responses[1].query == queries[1]
        assert responses[1].results[0].id == 20

    def test_search_fails_if_assets_not_loaded(self, initialized_service: Service):
        """Tests that Service.search raises RuntimeError if core assets are missing.

        This tests the internal safeguard in the search method.

        Args:
            initialized_service: An initialized Service instance.
        """
        initialized_service.embedding_model = None
        with pytest.raises(RuntimeError, match="Search service assets not loaded"):
            initialized_service.search(query="any")
