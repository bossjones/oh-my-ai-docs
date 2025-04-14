#!/usr/bin/env python3
# pyright: reportMissingImports=false
# pyright: reportUnusedVariable=warning
# pyright: reportUntypedBaseClass=error
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import pytest
from pytest_mock import MockerFixture
from pathlib import Path
from typing import Any, Dict, TypeVar, cast, Optional
from collections.abc import Awaitable, Mapping
from collections.abc import Callable, Generator
from mcp.types import TextContent, Tool, ResourceTemplate, Resource
from mcp.server.fastmcp.resources.base import Resource as FunctionResource
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.exceptions import ResourceError, ToolError
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from oh_my_ai_docs.avectorstore_mcp import (
    mcp_server,
    QueryConfig,
    DocumentResponse,
    BASE_PATH,
    DOCS_PATH
)

# Create a mock context class that extends the actual Context class
class MockContext(Context[Any, Any]):
    """Mock context for testing that extends the actual Context class."""

    info_fn: Callable[[str], Awaitable[None]] = Field(default=None)
    error_fn: Callable[[str], Awaitable[None]] = Field(default=None)
    report_progress_fn: Callable[[int, int], Awaitable[None]] = Field(default=None)
    app_context: Any = Field(default=None)
    mock_request_id: str = Field(default="test-request-id")

    @property
    def request_id(self) -> str:
        """Override request_id to avoid request context dependency."""
        return self.mock_request_id

    async def info(self, message: str) -> None:
        """Send info message."""
        if self.info_fn:
            await self.info_fn(message)

    async def error(self, message: str) -> None:
        """Send error message."""
        if self.error_fn:
            await self.error_fn(message)

    async def report_progress(self, current: int, total: int) -> None:
        """Report progress."""
        if self.report_progress_fn:
            await self.report_progress_fn(current, total)

    def model_dump(self) -> dict[str, Any]:
        """Override model_dump to include our custom attributes."""
        return {
            "info": self.info_fn,
            "error": self.error_fn,
            "report_progress": self.report_progress_fn,
            "app_context": self.app_context,
            "request_id": self.request_id
        }

    @classmethod
    def create(cls, info: Callable[[str], Awaitable[None]],
             error: Callable[[str], Awaitable[None]],
             report_progress: Callable[[int, int], Awaitable[None]],
             app_context: Any = None) -> MockContext:
        """Factory method to create MockContext instance."""
        return cls(
            info_fn=info,
            error_fn=error,
            report_progress_fn=report_progress,
            app_context=app_context
        )

# Test class for avectorstore MCP server
class TestAVectorStoreMCPServer:
    """
    Test suite for AVectorStore MCP Server functionality.
    Tests the actual server instance rather than creating new ones.
    """

    @pytest.fixture
    def mock_context(self, mocker: MockerFixture, mock_vectorstore: Any) -> Generator[Context[Any, Any], None, None]:
        """
        Fixture providing a mocked MCP Context for testing.

        Scope: function - ensures test isolation
        Args:
            mocker: pytest-mock fixture for creating mocks
            mock_vectorstore: Mocked vectorstore fixture

        Returns:
            Generator yielding mocked Context with Any types
        """
        # Create AsyncMock objects for the async methods
        info_mock = mocker.AsyncMock(name="info")
        error_mock = mocker.AsyncMock(name="error")
        progress_mock = mocker.AsyncMock(name="report_progress")

        # Create app context with store
        app_context = mocker.MagicMock()
        app_context.store = mock_vectorstore.return_value

        # Create a context using the factory method
        mock_ctx = MockContext.create(
            info=info_mock,
            error=error_mock,
            report_progress=progress_mock,
            app_context=app_context
        )

        yield mock_ctx

    @pytest.fixture
    def mock_vectorstore(self, mocker: MockerFixture) -> Generator[Any, None, None]:
        """
        Fixture providing a mocked SKLearnVectorStore.

        Scope: function - ensures fresh mock for each test
        Args:
            mocker: pytest-mock fixture

        Returns:
            Generator yielding mocked vectorstore
        Cleanup: Automatically handled by pytest-mock
        """
        mock_store = mocker.patch(
            "oh_my_ai_docs.avectorstore_mcp.SKLearnVectorStore",
            autospec=True
        )

        # Setup mock retriever behavior
        mock_retriever = mocker.MagicMock()
        mock_store.return_value.as_retriever.return_value = mock_retriever

        # Setup default document return
        mock_doc = Document(
            page_content="Test content",
            metadata={"score": 0.95}
        )
        mock_retriever.invoke.return_value = [mock_doc]

        yield mock_store

    @pytest.fixture
    def test_docs_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
        """
        Fixture creating a temporary docs directory structure and updating paths.

        Scope: function - ensures clean test environment
        Args:
            tmp_path: pytest fixture providing temporary directory
            monkeypatch: pytest fixture for patching

        Returns:
            Generator yielding path to test docs directory
        """
        docs_dir = tmp_path / "ai_docs" / "dpytest"
        docs_dir.mkdir(parents=True)

        # Create test docs file
        docs_file = docs_dir / "dpytest_docs.txt"
        docs_file.write_text("Test documentation content")

        # Create vectorstore directory
        vectorstore_dir = docs_dir / "vectorstore"
        vectorstore_dir.mkdir()

        # Patch the paths
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.DOCS_PATH", tmp_path / "ai_docs")
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.BASE_PATH", tmp_path)

        yield docs_dir

    @pytest.mark.anyio
    async def test_server_initialization(self) -> None:
        """Test that the MCP server is initialized with correct default name"""
        assert mcp_server.name == "dpytest-docs-mcp-server"

    @pytest.mark.anyio
    async def test_query_tool_registration(self) -> None:
        """Test that query_docs tool is properly registered with correct parameters"""
        tools = mcp_server._tool_manager.list_tools()

        assert len(tools) == 1
        tool = tools[0]

        assert tool.name == "query_docs"
        assert "Search through module documentation" in tool.description
        assert tool.parameters is not None

        # Check parameters structure
        params = tool.parameters.get("properties", {})
        assert "query" in params
        assert "config" in params
        assert params["query"]["type"] == "string"

    @pytest.mark.anyio
    async def test_successful_query(
        self,
        mocker: MockerFixture,
        mock_context: Context[Any, Any],
        mock_vectorstore: Any,
        test_docs_path: Path
    ) -> None:
        """
        Test successful query execution through the server.

        Args:
            mock_context: Mocked MCP context
            mock_vectorstore: Mocked vectorstore
            test_docs_path: Test documentation directory
        """
        result = await mcp_server._tool_manager.call_tool(
            "query_docs",
            {
                "query": "test query",
                "config": QueryConfig(k=3, min_relevance_score=0.0).model_dump(),
                "ctx": mock_context
            },
            context=mock_context
        )

        assert isinstance(result, DocumentResponse)
        assert len(result.documents) == 1
        assert result.documents[0] == "Test content"
        assert result.scores[0] == 0.95
        assert result.total_found == 1

        # Verify context method calls
        mock_info = cast(mocker.AsyncMock, mock_context.info_fn)
        mock_progress = cast(mocker.AsyncMock, mock_context.report_progress_fn)

        assert mock_info.await_count >= 2
        assert mock_info.await_args_list[0][0][0] == "Querying vectorstore with k=3"
        assert mock_info.await_args_list[1][0][0] == "Retrieved 1 relevant documents"
        assert mock_progress.await_count == 1
        assert mock_progress.await_args_list[0][0] == (1, 1)

    @pytest.mark.anyio
    async def test_query_with_low_relevance_threshold(
        self,
        mock_context: Context[Any, Any],
        mock_vectorstore: Any
    ) -> None:
        """Test query filtering based on relevance score threshold"""
        result = await mcp_server._tool_manager.call_tool(
            "query_docs",
            {
                "query": "test query",
                "config": QueryConfig(k=3, min_relevance_score=0.98).model_dump(),
                "ctx": mock_context
            },
            context=mock_context
        )

        assert isinstance(result, DocumentResponse)
        assert len(result.documents) == 0  # Should filter out doc with score 0.95
        assert len(result.scores) == 0
        assert result.total_found == 1

    @pytest.mark.anyio
    async def test_query_timeout(
        self,
        mock_context: Context[Any, Any],
        mock_vectorstore: Any,
        mocker: MockerFixture
    ) -> None:
        """Test query timeout handling"""
        # Make retriever.invoke take too long
        mock_vectorstore.return_value.as_retriever.return_value.invoke.side_effect = TimeoutError()

        with pytest.raises(ToolError, match="Query operation timed out"):
            await mcp_server._tool_manager.call_tool(
                "query_docs",
                {
                    "query": "test query",
                    "config": QueryConfig().model_dump(),
                    "ctx": mock_context
                },
                context=mock_context
            )

        # Verify error was logged
        mock_error = cast(mocker.AsyncMock, mock_context.error_fn)
        assert mock_error.await_count == 1
        assert mock_error.await_args_list[0][0][0] == "Query timed out"

    @pytest.mark.anyio
    async def test_get_all_docs_success(
        self,
        test_docs_path: Path
    ) -> None:
        """Test successful documentation retrieval"""
        resource = await mcp_server._resource_manager.get_resource("docs://dpytest/full")

        # Verify we got a Resource
        assert isinstance(resource, FunctionResource)

        # Get the actual content by calling the resource function
        content = resource.fn()
        assert isinstance(content, str)
        assert content == "Test documentation content"

    @pytest.mark.anyio
    async def test_get_all_docs_module_mismatch(self) -> None:
        """Test documentation retrieval with mismatched module"""
        with pytest.raises(ValueError, match="Requested module 'discord' does not match server module 'dpytest'"):
            await mcp_server._resource_manager.get_resource("docs://discord/full")
