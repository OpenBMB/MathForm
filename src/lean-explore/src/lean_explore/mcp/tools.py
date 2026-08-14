# src/lean_explore/mcp/tools.py

"""Defines MCP tools for interacting with the Lean Explore search engine.

These tools provide functionalities such as searching for statement groups,
retrieving specific groups by ID, and getting their dependencies. They
utilize a backend service (either an API client or a local service)
made available through the MCP application context.
"""

import asyncio  # Needed for asyncio.iscoroutinefunction
import logging
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import Context as MCPContext

from lean_explore.mcp.app import AppContext, BackendServiceType, mcp_app

# Import Pydantic models for type hinting and for creating response dicts
from lean_explore.shared.models.api import (
    APICitationsResponse,
    APISearchResponse,
    APISearchResultItem,
)

logger = logging.getLogger(__name__)


async def _get_backend_from_context(ctx: MCPContext) -> BackendServiceType:
    """Retrieves the backend service from the MCP context.

    Args:
        ctx: The MCP context provided to the tool.

    Returns:
        The configured backend service (APIClient or LocalService).
        Guaranteed to be non-None if this function returns, otherwise
        it raises an exception.

    Raises:
        RuntimeError: If the backend service is not available in the context,
                      indicating a server configuration issue.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    backend = app_ctx.backend_service
    if not backend:
        logger.error(
            "MCP Tool Error: Backend service is not available in lifespan_context."
        )
        raise RuntimeError("Backend service not configured or available for MCP tool.")
    return backend


def _prepare_mcp_result_item(backend_item: APISearchResultItem) -> APISearchResultItem:
    """Prepares an APISearchResultItem for MCP response.

    This helper ensures that the item sent over MCP does not include
    the display_statement_text, as the full statement_text is preferred
    for model consumption.

    Args:
        backend_item: The item as received from the backend service.

    Returns:
        A new APISearchResultItem instance suitable for MCP responses.
    """
    # Create a new instance or use .model_copy(update=...) for Pydantic v2
    return APISearchResultItem(
        id=backend_item.id,
        primary_declaration=backend_item.primary_declaration.model_copy()
        if backend_item.primary_declaration
        else None,
        source_file=backend_item.source_file,
        range_start_line=backend_item.range_start_line,
        statement_text=backend_item.statement_text,
        docstring=backend_item.docstring,
        informal_description=backend_item.informal_description,
        display_statement_text=None,  # Ensure this is not sent over MCP
    )


@mcp_app.tool()
async def search(
    ctx: MCPContext,
    query: Union[str, List[str]],
    package_filters: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Searches Lean statement groups by a query string or list of strings.

    This tool allows for filtering by package names and limits the number
    of results returned per query.

    Args:
        ctx: The MCP context, providing access to shared resources like the
             backend service.
        query: A single search query string or a list of query strings. For
               example, "continuous function" or ["prime number theorem",
               "fundamental theorem of arithmetic"].
        package_filters: An optional list of package names to filter the search
                         results by. For example, `["Mathlib.Analysis",
                         "Mathlib.Order"]`. If None or empty, no package filter
                         is applied.
        limit: The maximum number of search results to return per query.
               Defaults to 10. Must be a positive integer.

    Returns:
        A list of dictionaries, where each dictionary corresponds to the
        APISearchResponse model. Each response contains the search results
        for a single query. The `display_statement_text` field within each
        result item is omitted.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(
        f"MCP Tool 'search' called with query/queries: '{query}', "
        f"packages: {package_filters}, tool_limit: {limit}"
    )

    if not hasattr(backend, "search"):
        logger.error("Backend service does not have a 'search' method.")
        raise RuntimeError("Search functionality not available on configured backend.")

    tool_limit = max(1, limit)
    backend_responses: Union[APISearchResponse, List[APISearchResponse]]

    # Conditionally await based on the backend's search method type
    if asyncio.iscoroutinefunction(backend.search):
        backend_responses = await backend.search(
            query=query, package_filters=package_filters
        )
    else:
        backend_responses = backend.search(query=query, package_filters=package_filters)

    # Normalize to a list for consistent processing, handling None from backend.
    if backend_responses is None:
        responses_list = []
    else:
        responses_list = (
            [backend_responses]
            if isinstance(backend_responses, APISearchResponse)
            else backend_responses
        )

    final_mcp_responses = []

    for response_pydantic in responses_list:
        if not response_pydantic:
            logger.warning("A backend search returned None; skipping this response.")
            continue

        actual_backend_results = response_pydantic.results
        mcp_results_list = []
        for backend_item in actual_backend_results[:tool_limit]:
            mcp_results_list.append(_prepare_mcp_result_item(backend_item))

        final_mcp_response = APISearchResponse(
            query=response_pydantic.query,
            packages_applied=response_pydantic.packages_applied,
            results=mcp_results_list,
            count=len(mcp_results_list),
            total_candidates_considered=response_pydantic.total_candidates_considered,
            processing_time_ms=response_pydantic.processing_time_ms,
        )
        final_mcp_responses.append(final_mcp_response.model_dump(exclude_none=True))

    return final_mcp_responses


@mcp_app.tool()
async def get_by_id(
    ctx: MCPContext, group_id: Union[int, List[int]]
) -> List[Optional[Dict[str, Any]]]:
    """Retrieves specific statement groups by their unique identifier(s).

    The `display_statement_text` field is omitted from the response. This tool
    always returns a list of results.

    Args:
        ctx: The MCP context, providing access to the backend service.
        group_id: A single unique integer identifier or a list of identifiers
                  of the statement group(s) to retrieve. For example, `12345` or
                  `[12345, 67890]`.

    Returns:
        A list of dictionaries, where each dictionary corresponds to the
        APISearchResultItem model. If an ID is not found, its corresponding
        entry in the list will be None (serialized as JSON null by MCP).
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(f"MCP Tool 'get_by_id' called for group_id(s): {group_id}")

    backend_items: Union[
        Optional[APISearchResultItem], List[Optional[APISearchResultItem]]
    ]
    if asyncio.iscoroutinefunction(backend.get_by_id):
        backend_items = await backend.get_by_id(group_id=group_id)
    else:
        backend_items = backend.get_by_id(group_id=group_id)

    # Normalize to a list for consistent return type
    items_list = (
        [backend_items] if not isinstance(backend_items, list) else backend_items
    )

    mcp_items = []
    for item in items_list:
        if item:
            mcp_item = _prepare_mcp_result_item(item)
            mcp_items.append(mcp_item.model_dump(exclude_none=True))
        else:
            mcp_items.append(None)

    return mcp_items


@mcp_app.tool()
async def get_dependencies(
    ctx: MCPContext, group_id: Union[int, List[int]]
) -> List[Optional[Dict[str, Any]]]:
    """Retrieves direct dependencies (citations) for specific statement group(s).

    The `display_statement_text` field within each cited item is omitted
    from the response. This tool always returns a list of results.

    Args:
        ctx: The MCP context, providing access to the backend service.
        group_id: A single unique integer identifier or a list of identifiers for
                  the statement group(s) for which to fetch direct dependencies.
                  For example, `12345` or `[12345, 67890]`.

    Returns:
        A list of dictionaries, where each dictionary corresponds to the
        APICitationsResponse model. If a source group ID is not found or has
        no dependencies, its corresponding entry will be None.
    """
    backend = await _get_backend_from_context(ctx)
    logger.info(f"MCP Tool 'get_dependencies' called for group_id(s): {group_id}")

    backend_responses: Union[
        Optional[APICitationsResponse], List[Optional[APICitationsResponse]]
    ]
    if asyncio.iscoroutinefunction(backend.get_dependencies):
        backend_responses = await backend.get_dependencies(group_id=group_id)
    else:
        backend_responses = backend.get_dependencies(group_id=group_id)

    # Normalize to a list for consistent return type
    responses_list = (
        [backend_responses]
        if not isinstance(backend_responses, list)
        else backend_responses
    )
    final_mcp_responses = []

    for response in responses_list:
        if response:
            mcp_citations_list = []
            for backend_item in response.citations:
                mcp_citations_list.append(_prepare_mcp_result_item(backend_item))

            final_response = APICitationsResponse(
                source_group_id=response.source_group_id,
                citations=mcp_citations_list,
                count=len(mcp_citations_list),
            )
            final_mcp_responses.append(final_response.model_dump(exclude_none=True))
        else:
            final_mcp_responses.append(None)

    return final_mcp_responses
