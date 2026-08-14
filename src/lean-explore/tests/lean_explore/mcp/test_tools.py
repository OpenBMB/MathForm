# tests/lean_explore/mcp/test_tools.py

"""Tests for the MCP tools defined in `lean_explore.mcp.tools`.

This module verifies the functionality of the MCP tools (search, get_by_id,
get_dependencies), ensuring they correctly interact with the backend service
(either APIClient or LocalService) via the application context and return
responses in the expected format for MCP consumers. External service calls
are mocked to isolate test execution.
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from mcp.server.fastmcp import Context as MCPContext

from lean_explore.api.client import Client as APIClient
from lean_explore.local.service import Service as LocalService
from lean_explore.mcp.app import AppContext
from lean_explore.mcp.tools import (
    _get_backend_from_context,
    _prepare_mcp_result_item,
    get_by_id,
    get_dependencies,
    search,
)
from lean_explore.shared.models.api import (
    APICitationsResponse,
    APIPrimaryDeclarationInfo,
    APISearchResponse,
    APISearchResultItem,
)


@pytest.fixture
def backend_service_mock(request):
    """Resolve mock_api_backend and mock_local_backend."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def mock_mcp_context(mocker: "MockerFixture") -> MCPContext:
    """Provides a mock MCPContext for testing tools."""
    mock_app_context = AppContext(backend_service=None)  # Will be set by other fixtures
    mock_request_context = mocker.MagicMock()
    mock_request_context.lifespan_context = mock_app_context

    mock_ctx = mocker.MagicMock(spec=MCPContext)
    mock_ctx.request_context = mock_request_context
    return mock_ctx


@pytest.fixture
def mock_api_backend(mocker: "MockerFixture") -> APIClient:
    """Provides a mock APIClient instance as a backend service."""
    mock_client = mocker.MagicMock(spec=APIClient)
    # Set default mock return values for methods that are awaited
    mock_client.search = AsyncMock()
    mock_client.get_by_id = AsyncMock()
    mock_client.get_dependencies = AsyncMock()
    return mock_client


@pytest.fixture
def mock_local_backend(mocker: "MockerFixture") -> LocalService:
    """Provides a mock LocalService instance as a backend service."""
    mock_service = mocker.MagicMock(spec=LocalService)
    # Set default mock return values for methods that are not awaited (synchronous)
    mock_service.search = MagicMock()
    mock_service.get_by_id = MagicMock()
    mock_service.get_dependencies = MagicMock()
    return mock_service


@pytest.fixture
def set_backend_on_context(
    mock_mcp_context: MCPContext, backend_service_mock: MagicMock
):
    """Sets the provided backend service mock onto the MCP context."""
    mock_mcp_context.request_context.lifespan_context.backend_service = (
        backend_service_mock
    )
    return mock_mcp_context


@pytest.mark.asyncio
class TestBackendRetrievalHelper:
    """Test suite for the internal `_get_backend_from_context` helper function."""

    async def test_get_backend_from_context_successful(
        self, mock_mcp_context: MCPContext, mock_api_backend: APIClient
    ):
        """Verifies that the helper correctly retrieves the backend service."""
        mock_mcp_context.request_context.lifespan_context.backend_service = (
            mock_api_backend
        )
        backend = await _get_backend_from_context(mock_mcp_context)
        assert backend is mock_api_backend

    async def test_get_backend_from_context_raises_error_if_none(
        self, mock_mcp_context: MCPContext
    ):
        """Verifies that the helper raises an error if the backend service is None."""
        mock_mcp_context.request_context.lifespan_context.backend_service = None
        with pytest.raises(
            RuntimeError,
            match="Backend service not configured or available for MCP tool.",
        ):
            await _get_backend_from_context(mock_mcp_context)


class TestPrepareMcpResultItemHelper:
    """Test suite for the internal `_prepare_mcp_result_item` helper function."""

    def test_prepare_mcp_result_item_omits_display_statement_text(self):
        """Verifies that `display_statement_text` is set to None."""
        original_item = APISearchResultItem(
            id=1,
            primary_declaration=APIPrimaryDeclarationInfo(lean_name="Test.Declaration"),
            source_file="Test.lean",
            range_start_line=10,
            statement_text="def Test.Declaration := ...",
            display_statement_text="def Test.Declaration := ... (formatted)",
            docstring="A test declaration.",
            informal_description=None,
        )

        prepared_item = _prepare_mcp_result_item(original_item)

        assert prepared_item.id == original_item.id
        assert prepared_item.statement_text == original_item.statement_text
        assert prepared_item.display_statement_text is None
        assert prepared_item.docstring == original_item.docstring
        assert (
            prepared_item.primary_declaration.lean_name
            == original_item.primary_declaration.lean_name
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_service_mock", ["mock_api_backend", "mock_local_backend"], indirect=True
)
class TestSearchTool:
    """Test suite for the `search` MCP tool."""

    async def test_search_single_query(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies successful search with a single query string."""
        ctx = set_backend_on_context
        query = "function.continuous"
        mock_backend_response = APISearchResponse(
            query=query,
            results=[],
            count=0,
            total_candidates_considered=0,
            processing_time_ms=50,
        )
        backend_service_mock.search.return_value = mock_backend_response

        result = await search(ctx, query=query)

        backend_service_mock.search.assert_called_once_with(
            query=query, package_filters=None
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["query"] == query
        assert "display_statement_text" not in result[0]

    async def test_search_batch_query(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies successful search with a list of query strings."""
        ctx = set_backend_on_context
        queries = ["function.continuous", "group.is_monoid"]
        package_filters = ["Mathlib"]
        limit = 5
        mock_backend_responses = [
            APISearchResponse(
                query=q,
                results=[],
                count=0,
                total_candidates_considered=0,
                processing_time_ms=1,
            )
            for q in queries
        ]
        backend_service_mock.search.return_value = mock_backend_responses

        result = await search(
            ctx, query=queries, package_filters=package_filters, limit=limit
        )

        backend_service_mock.search.assert_called_once_with(
            query=queries, package_filters=package_filters
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["query"] == queries[0]
        assert result[1]["query"] == queries[1]

    async def test_search_returns_empty_list_if_backend_returns_none(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies search tool returns an empty list if the backend returns None."""
        ctx = set_backend_on_context
        backend_service_mock.search.return_value = None

        result = await search(ctx, query="nonexistent")

        assert result == []

    async def test_search_propagates_backend_exceptions(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies search propagates exceptions raised by the backend service."""
        ctx = set_backend_on_context
        backend_service_mock.search.side_effect = ValueError("Backend search failed")

        with pytest.raises(ValueError, match="Backend search failed"):
            await search(ctx, query="error_query")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_service_mock", ["mock_api_backend", "mock_local_backend"], indirect=True
)
class TestGetByIdTool:
    """Test suite for the `get_by_id` MCP tool."""

    async def test_get_by_id_single_id_found(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies successful retrieval for a single, existing ID."""
        ctx = set_backend_on_context
        group_id = 123
        mock_backend_item = APISearchResultItem(
            id=group_id,
            primary_declaration=APIPrimaryDeclarationInfo(lean_name="L.D"),
            statement_text="def L.D",
            source_file="L.lean",
            range_start_line=1,
        )
        backend_service_mock.get_by_id.return_value = mock_backend_item

        result = await get_by_id(ctx, group_id=group_id)

        backend_service_mock.get_by_id.assert_called_once_with(group_id=group_id)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["id"] == group_id

    async def test_get_by_id_batch_ids(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies successful retrieval for a list of IDs."""
        ctx = set_backend_on_context
        group_ids = [123, 456, 999]  # One will be "not found"
        mock_item1 = APISearchResultItem(
            id=123,
            primary_declaration=APIPrimaryDeclarationInfo(),
            statement_text="t",
            source_file="f",
            range_start_line=1,
        )
        mock_item2 = APISearchResultItem(
            id=456,
            primary_declaration=APIPrimaryDeclarationInfo(),
            statement_text="t",
            source_file="f",
            range_start_line=1,
        )
        backend_service_mock.get_by_id.return_value = [mock_item1, mock_item2, None]

        result = await get_by_id(ctx, group_id=group_ids)

        backend_service_mock.get_by_id.assert_called_once_with(group_id=group_ids)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["id"] == 123
        assert result[1]["id"] == 456
        assert result[2] is None

    async def test_get_by_id_single_id_not_found(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies behavior for a single, non-existent ID."""
        ctx = set_backend_on_context
        group_id = 999
        backend_service_mock.get_by_id.return_value = None

        result = await get_by_id(ctx, group_id=group_id)

        backend_service_mock.get_by_id.assert_called_once_with(group_id=group_id)
        assert result == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_service_mock", ["mock_api_backend", "mock_local_backend"], indirect=True
)
class TestGetDependenciesTool:
    """Test suite for the `get_dependencies` MCP tool."""

    async def test_get_dependencies_single_id(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies successful retrieval of dependencies for a single ID."""
        ctx = set_backend_on_context
        group_id = 456
        mock_backend_response = APICitationsResponse(
            source_group_id=group_id, citations=[], count=0
        )
        backend_service_mock.get_dependencies.return_value = mock_backend_response

        result = await get_dependencies(ctx, group_id=group_id)

        backend_service_mock.get_dependencies.assert_called_once_with(group_id=group_id)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["source_group_id"] == group_id

    async def test_get_dependencies_batch_ids(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies successful retrieval for a list of dependency IDs."""
        ctx = set_backend_on_context
        group_ids = [456, 789]
        mock_response1 = APICitationsResponse(
            source_group_id=456, citations=[], count=0
        )
        mock_response2 = APICitationsResponse(
            source_group_id=789, citations=[], count=0
        )
        backend_service_mock.get_dependencies.return_value = [
            mock_response1,
            mock_response2,
        ]

        result = await get_dependencies(ctx, group_id=group_ids)

        backend_service_mock.get_dependencies.assert_called_once_with(
            group_id=group_ids
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["source_group_id"] == 456
        assert result[1]["source_group_id"] == 789

    async def test_get_dependencies_single_id_not_found(
        self, set_backend_on_context: MCPContext, backend_service_mock: MagicMock
    ):
        """Verifies behavior for a single, non-existent source ID."""
        ctx = set_backend_on_context
        group_id = 998
        backend_service_mock.get_dependencies.return_value = None

        result = await get_dependencies(ctx, group_id=group_id)

        backend_service_mock.get_dependencies.assert_called_once_with(group_id=group_id)
        assert result == [None]
