# tests/lean_explore/api/test_client.py

"""Tests for the API client in `lean_explore.api.client`.

This module verifies the functionality of the `Client` class,
ensuring it correctly initializes, constructs API requests,
handles responses (both successful and erroneous), and parses
data into the appropriate Pydantic models. External HTTP calls
are mocked to isolate test execution.
"""

from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import ANY, AsyncMock, MagicMock

import httpx
import pytest

from lean_explore.api.client import _DEFAULT_API_BASE_URL, Client
from lean_explore.shared.models.api import (
    APICitationsResponse,
    APIPrimaryDeclarationInfo,
    APISearchResponse,
    APISearchResultItem,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_async_client_constructor(mocker: "MockerFixture") -> MagicMock:
    """Provides a mock for the httpx.AsyncClient constructor.

    This fixture sets up a mock for `httpx.AsyncClient` that allows for inspection
    of how the client is instantiated and used within the tested methods. The mock
    is configured to return a context-manageable async client instance.

    Args:
        mocker: Pytest-mock's mocker fixture.

    Returns:
        unittest.mock.MagicMock: The mock for the `httpx.AsyncClient` constructor.
    """
    mock_async_client_instance = MagicMock()
    # Individual methods like `get` will be configured in specific tests.
    mock_async_client_instance.get = AsyncMock()

    mock_constructor = mocker.patch("httpx.AsyncClient")
    mock_constructor.return_value.__aenter__.return_value = mock_async_client_instance
    return mock_constructor


@pytest.mark.asyncio
class TestAPIClient:
    """Test suite for the lean_explore.api.client.Client class."""

    def test_client_initialization(self):
        """Verifies correct initialization of Client attributes.

        Checks that api_key, timeout (default and custom), base_url,
        and HTTP headers are set as expected upon client instantiation.
        """
        api_key = "test_api_key_123"
        custom_timeout = 20.0

        client_default_timeout = Client(api_key=api_key)
        assert client_default_timeout.api_key == api_key
        assert client_default_timeout.timeout == 10.0  # Default timeout
        assert client_default_timeout.base_url == _DEFAULT_API_BASE_URL
        assert client_default_timeout._headers == {"Authorization": f"Bearer {api_key}"}

        client_custom_timeout = Client(api_key=api_key, timeout=custom_timeout)
        assert client_custom_timeout.timeout == custom_timeout

    async def test_search_single_query(self, mocker: "MockerFixture"):
        """Tests search with a single query string.

        Verifies that a single query correctly calls the internal fetcher
        and returns a single APISearchResponse object.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        query = "Nat.add"
        mock_response = APISearchResponse(
            query=query,
            results=[],
            count=0,
            total_candidates_considered=0,
            processing_time_ms=1,
        )

        mock_fetch_one = mocker.patch.object(
            client,
            "_fetch_one_search",
            new_callable=AsyncMock,
            return_value=mock_response,
        )

        result = await client.search(query=query)

        mock_fetch_one.assert_awaited_once_with(ANY, query, None)
        assert result is mock_response

    async def test_search_batch_query(self, mocker: "MockerFixture"):
        """Tests search with a list of query strings.

        Verifies that a list of queries calls the internal fetcher multiple
        times via asyncio.gather and returns a list of APISearchResponse objects.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        queries = ["Nat.add", "Nat.mul"]
        mock_response1 = APISearchResponse(
            query=queries[0],
            results=[],
            count=0,
            total_candidates_considered=0,
            processing_time_ms=1,
        )
        mock_response2 = APISearchResponse(
            query=queries[1],
            results=[],
            count=0,
            total_candidates_considered=0,
            processing_time_ms=1,
        )

        mock_fetch_one = mocker.patch.object(
            client,
            "_fetch_one_search",
            new_callable=AsyncMock,
            side_effect=[mock_response1, mock_response2],
        )

        results = await client.search(query=queries)

        assert mock_fetch_one.await_count == 2
        mock_fetch_one.assert_any_await(ANY, queries[0], None)
        mock_fetch_one.assert_any_await(ANY, queries[1], None)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0] is mock_response1
        assert results[1] is mock_response2

    async def test_fetch_one_search_successful(
        self, mock_async_client_constructor: MagicMock
    ):
        """Tests the internal `_fetch_one_search` method for success.

        Args:
            mock_async_client_constructor: Mock for httpx.AsyncClient constructor.
        """
        client = Client(api_key="test_key")
        query = "is_prime"
        package_filters = ["Mathlib"]
        mock_response_json: Dict[str, Any] = {
            "query": query,
            "results": [],
            "count": 0,
            "total_candidates_considered": 0,
            "processing_time_ms": 1,
        }

        mock_http_response = MagicMock(spec=httpx.Response, status_code=200)
        mock_http_response.json.return_value = mock_response_json
        mock_http_response.raise_for_status = MagicMock()

        mock_async_client = (
            mock_async_client_constructor.return_value.__aenter__.return_value
        )
        mock_async_client.get.return_value = mock_http_response

        response_model = await client._fetch_one_search(
            mock_async_client, query=query, package_filters=package_filters
        )

        mock_async_client.get.assert_awaited_once_with(
            f"{client.base_url}/search",
            params={"q": query, "pkg": package_filters},
            headers=client._headers,
        )
        mock_http_response.raise_for_status.assert_called_once()
        assert isinstance(response_model, APISearchResponse)

    async def test_get_by_id_single_id(self, mocker: "MockerFixture"):
        """Tests get_by_id with a single integer ID.

        Verifies it calls the internal fetcher and returns a single item.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        group_id = 123
        mock_response = APISearchResultItem(
            id=group_id,
            primary_declaration=APIPrimaryDeclarationInfo(lean_name="Test.Item"),
            source_file="f",
            range_start_line=1,
            statement_text="t",
        )

        mock_fetch_one = mocker.patch.object(
            client,
            "_fetch_one_by_id",
            new_callable=AsyncMock,
            return_value=mock_response,
        )

        result = await client.get_by_id(group_id=group_id)

        mock_fetch_one.assert_awaited_once_with(ANY, group_id)
        assert result is mock_response

    async def test_get_by_id_batch_ids(self, mocker: "MockerFixture"):
        """Tests get_by_id with a list of integer IDs.

        Verifies it calls the internal fetcher multiple times and returns a list.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        group_ids = [123, 456, 789]
        mock_response1 = APISearchResultItem(
            id=group_ids[0],
            primary_declaration=APIPrimaryDeclarationInfo(lean_name="Test.Item1"),
            source_file="f",
            range_start_line=1,
            statement_text="t",
        )
        mock_response2 = None  # Simulate a not-found case
        mock_response3 = APISearchResultItem(
            id=group_ids[2],
            primary_declaration=APIPrimaryDeclarationInfo(lean_name="Test.Item3"),
            source_file="f",
            range_start_line=1,
            statement_text="t",
        )

        mock_fetch_one = mocker.patch.object(
            client,
            "_fetch_one_by_id",
            new_callable=AsyncMock,
            side_effect=[mock_response1, mock_response2, mock_response3],
        )

        results = await client.get_by_id(group_id=group_ids)

        assert mock_fetch_one.await_count == 3
        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0] is mock_response1
        assert results[1] is mock_response2
        assert results[2] is mock_response3

    async def test_fetch_one_by_id_not_found_404(
        self, mock_async_client_constructor: MagicMock
    ):
        """Tests the internal `_fetch_one_by_id` returning None for a 404.

        Args:
            mock_async_client_constructor: Mock for httpx.AsyncClient constructor.
        """
        client = Client(api_key="test_key")
        mock_http_response = MagicMock(spec=httpx.Response, status_code=404)
        mock_http_response.raise_for_status = MagicMock()

        mock_async_client = (
            mock_async_client_constructor.return_value.__aenter__.return_value
        )
        mock_async_client.get.return_value = mock_http_response

        result = await client._fetch_one_by_id(mock_async_client, group_id=404)

        assert result is None
        mock_http_response.raise_for_status.assert_not_called()

    async def test_fetch_one_by_id_http_error_other_than_404(
        self, mock_async_client_constructor: MagicMock
    ):
        """Tests `_fetch_one_by_id` raising HTTPStatusError for non-404s.

        Args:
            mock_async_client_constructor: Mock for httpx.AsyncClient constructor.
        """
        client = Client(api_key="test_key")
        mock_http_response = MagicMock(spec=httpx.Response, status_code=401)
        mock_http_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_http_response
        )
        mock_async_client = (
            mock_async_client_constructor.return_value.__aenter__.return_value
        )
        mock_async_client.get.return_value = mock_http_response

        with pytest.raises(httpx.HTTPStatusError):
            await client._fetch_one_by_id(mock_async_client, group_id=123)
        mock_http_response.raise_for_status.assert_called_once()

    async def test_get_dependencies_single_id(self, mocker: "MockerFixture"):
        """Tests get_dependencies with a single integer ID.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        group_id = 456
        mock_response = APICitationsResponse(
            source_group_id=group_id, citations=[], count=0
        )
        mock_fetch_one = mocker.patch.object(
            client,
            "_fetch_one_dependencies",
            new_callable=AsyncMock,
            return_value=mock_response,
        )

        result = await client.get_dependencies(group_id=group_id)

        mock_fetch_one.assert_awaited_once_with(ANY, group_id)
        assert result is mock_response

    async def test_get_dependencies_batch_ids(self, mocker: "MockerFixture"):
        """Tests get_dependencies with a list of integer IDs.

        Args:
            mocker: Pytest-mock's mocker fixture.
        """
        client = Client(api_key="test_key")
        group_ids = [456, 789]
        mock_response1 = APICitationsResponse(
            source_group_id=group_ids[0], citations=[], count=0
        )
        mock_response2 = APICitationsResponse(
            source_group_id=group_ids[1], citations=[], count=0
        )
        mock_fetch_one = mocker.patch.object(
            client,
            "_fetch_one_dependencies",
            new_callable=AsyncMock,
            side_effect=[mock_response1, mock_response2],
        )

        results = await client.get_dependencies(group_id=group_ids)

        assert mock_fetch_one.await_count == 2
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0] is mock_response1
        assert results[1] is mock_response2

    async def test_fetch_one_dependencies_http_error(
        self, mock_async_client_constructor: MagicMock
    ):
        """Tests `_fetch_one_dependencies` for non-200/404 HTTP errors.

        Args:
            mock_async_client_constructor: Mock for httpx.AsyncClient constructor.
        """
        client = Client(api_key="test_key")
        mock_http_response = MagicMock(spec=httpx.Response, status_code=503)
        mock_http_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", request=MagicMock(), response=mock_http_response
        )
        mock_async_client = (
            mock_async_client_constructor.return_value.__aenter__.return_value
        )
        mock_async_client.get.return_value = mock_http_response

        with pytest.raises(httpx.HTTPStatusError):
            await client._fetch_one_dependencies(mock_async_client, group_id=123)
        mock_http_response.raise_for_status.assert_called_once()
