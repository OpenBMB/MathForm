# src/lean_explore/api/client.py

"""Provides a client for interacting with the remote Lean Explore API.

This module contains the Client class, which facilitates
communication with the backend Lean Explore search engine API for
performing searches and retrieving detailed information.
"""

import asyncio
from typing import List, Optional, Union, overload

import httpx

from lean_explore.shared.models.api import (
    APICitationsResponse,
    APISearchResponse,
    APISearchResultItem,
)

_DEFAULT_API_BASE_URL = "https://www.leanexplore.com/api/v1"


class Client:
    """An asynchronous client for the Lean Explore backend API.

    This client handles making HTTP requests to the production API base URL,
    authenticating with an API key, and parsing responses into Pydantic models.

    Attributes:
        api_key: The API key used for authenticating requests.
        timeout: The timeout for HTTP requests in seconds.
        base_url: The hardcoded base URL for the API.
    """

    def __init__(self, api_key: str, timeout: float = 10.0):
        """Initializes the API Client.

        Args:
            api_key: The API key for authentication.
            timeout: Default timeout for HTTP requests in seconds.
        """
        self.base_url: str = _DEFAULT_API_BASE_URL
        self.api_key: str = api_key
        self.timeout: float = timeout
        self._headers: dict = {"Authorization": f"Bearer {self.api_key}"}

    async def _fetch_one_search(
        self,
        client: httpx.AsyncClient,
        query: str,
        package_filters: Optional[List[str]],
    ) -> APISearchResponse:
        """Coroutine to fetch a single search result.

        Args:
            client: An active httpx.AsyncClient instance.
            query: The search query string.
            package_filters: An optional list of package names.

        Returns:
            An APISearchResponse object.
        """
        endpoint = f"{self.base_url}/search"
        params = {"q": query}
        if package_filters:
            params["pkg"] = package_filters

        response = await client.get(endpoint, params=params, headers=self._headers)
        response.raise_for_status()
        return APISearchResponse(**response.json())

    @overload
    async def search(
        self, query: str, package_filters: Optional[List[str]] = None
    ) -> APISearchResponse: ...

    @overload
    async def search(
        self, query: List[str], package_filters: Optional[List[str]] = None
    ) -> List[APISearchResponse]: ...

    async def search(
        self,
        query: Union[str, List[str]],
        package_filters: Optional[List[str]] = None,
    ) -> Union[APISearchResponse, List[APISearchResponse]]:
        """Performs a search for statement groups via the API.

        This method can handle a single query string or a list of query strings.
        When a list is provided, requests are sent concurrently.

        Args:
            query: The search query string or a list of query strings.
            package_filters: An optional list of package names to filter the
                search by. This filter is applied to all queries.

        Returns:
            An APISearchResponse object if a single query was provided, or a
            list of APISearchResponse objects if a list of queries was provided.

        Raises:
            httpx.HTTPStatusError: If the API returns an HTTP error status (4xx or 5xx).
            httpx.RequestError: For network-related issues or other request errors.
        """
        was_single_query = isinstance(query, str)
        queries = [query] if was_single_query else query

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [
                self._fetch_one_search(client, q, package_filters) for q in queries
            ]
            results = await asyncio.gather(*tasks)

        if was_single_query:
            return results[0]
        return results

    async def _fetch_one_by_id(
        self, client: httpx.AsyncClient, group_id: int
    ) -> Optional[APISearchResultItem]:
        endpoint = f"{self.base_url}/statement_groups/{group_id}"
        response = await client.get(endpoint, headers=self._headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return APISearchResultItem(**response.json())

    @overload
    async def get_by_id(self, group_id: int) -> Optional[APISearchResultItem]: ...

    @overload
    async def get_by_id(
        self, group_id: List[int]
    ) -> List[Optional[APISearchResultItem]]: ...

    async def get_by_id(
        self, group_id: Union[int, List[int]]
    ) -> Union[Optional[APISearchResultItem], List[Optional[APISearchResultItem]]]:
        """Retrieves a specific statement group by its unique ID via the API.

        Args:
            group_id: The unique identifier of the statement group, or a list of IDs.

        Returns:
            An APISearchResultItem object if a single ID was found, None if it was
            not found. A list of Optional[APISearchResultItem] if a list of
            IDs was provided.

        Raises:
            httpx.HTTPStatusError: If the API returns an HTTP error status
                                other than 404 (e.g., 401, 403, 5xx).
            httpx.RequestError: For network-related issues or other request errors.
        """
        was_single_id = isinstance(group_id, int)
        group_ids = [group_id] if was_single_id else group_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._fetch_one_by_id(client, g_id) for g_id in group_ids]
            results = await asyncio.gather(*tasks)

        if was_single_id:
            return results[0]
        return results

    async def _fetch_one_dependencies(
        self, client: httpx.AsyncClient, group_id: int
    ) -> Optional[APICitationsResponse]:
        endpoint = f"{self.base_url}/statement_groups/{group_id}/dependencies"
        response = await client.get(endpoint, headers=self._headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return APICitationsResponse(**response.json())

    @overload
    async def get_dependencies(
        self, group_id: int
    ) -> Optional[APICitationsResponse]: ...

    @overload
    async def get_dependencies(
        self, group_id: List[int]
    ) -> List[Optional[APICitationsResponse]]: ...

    async def get_dependencies(
        self, group_id: Union[int, List[int]]
    ) -> Union[Optional[APICitationsResponse], List[Optional[APICitationsResponse]]]:
        """Retrieves the dependencies (citations) for a specific statement group.

        This method fetches the statement groups that the specified 'group_id'(s)
        depend on (i.e., cite).

        Args:
            group_id: The unique identifier of the statement group, or a list of IDs.

        Returns:
            An APICitationsResponse object if a single ID was provided. A list
            of Optional[APICitationsResponse] if a list of IDs was provided.
            None is returned for IDs that are not found.

        Raises:
            httpx.HTTPStatusError: If the API returns an HTTP error status
                                other than 404 (e.g., 401, 403, 5xx).
            httpx.RequestError: For network-related issues or other request errors.
        """
        was_single_id = isinstance(group_id, int)
        group_ids = [group_id] if was_single_id else group_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._fetch_one_dependencies(client, g_id) for g_id in group_ids]
            results = await asyncio.gather(*tasks)

        if was_single_id:
            return results[0]
        return results
