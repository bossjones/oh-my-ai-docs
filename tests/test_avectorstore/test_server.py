#!/usr/bin/env python3
from __future__ import annotations

import pytest
from pytest_mock import MockerFixture
from pathlib import Path
from typing import Any, Dict
from collections.abc import Generator
from mcp.types import TextContent, Tool, ResourceTemplate
from mcp.server.fastmcp import Context
from langchain_core.documents import Document

from oh_my_ai_docs.avectorstore_mcp import (
    mcp,
    QueryConfig,
    DocumentResponse,
    BASE_PATH,
    DOCS_PATH
)

# Test class for avectorstore MCP server
class TestAVectorStoreMCPServer:
    """
    Test suite for AVectorStore MCP Server functionality.
    Tests the actual server instance rather than creating new ones.
    """

    @pytest.fixture
    def mock_context(self, mocker: MockerFixture) -> Generator[Context[Any, Any], None, None]:
        """
        Fixture providing a mocked MCP Context for testing.

        Scope: function - ensures test isolation
        Args:
            mocker: pytest-mock fixture for creating mocks

        Returns:
            Generator yielding mocked Context with Any types
        """
        mock_ctx = mocker.MagicMock(spec=Context)
        mock_ctx.info = mocker.AsyncMock()
        mock_ctx.error = mocker.AsyncMock()
        mock_ctx.report_progress = mocker.AsyncMock()
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
        assert mcp.name == "dpytest-docs-mcp-server"

    @pytest.mark.anyio
    async def test_query_tool_registration(self) -> None:
        """Test that query_docs tool is properly registered with correct parameters"""
        tools = mcp._tool_manager.list_tools()

        assert len(tools) == 1
        tool = tools[0]

        assert tool.name == "query_docs"
        assert "Search through module documentation" in tool.description
        assert tool.parameters is not None

        # Check parameters structure
        params = tool.parameters
        assert "query" in params
        assert "config" in params
        assert params["query"]["type"] == "string"

    @pytest.mark.anyio
    async def test_successful_query(
        self,
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
        config = QueryConfig(k=3, min_relevance_score=0.0)

        result = await mcp._tool_manager.call_tool(
            "query_docs",
            {
                "query": "test query",
                "config": config.model_dump()
            },
            context=mock_context
        )

        assert isinstance(result, DocumentResponse)
        assert len(result.documents) == 1
        assert result.documents[0] == "Test content"
        assert result.scores[0] == 0.95
        assert result.total_found == 1

        # Verify context method calls
        mock_context.info.assert_any_call("Querying vectorstore with k=3")
        mock_context.info.assert_any_call("Retrieved 1 relevant documents")
        mock_context.report_progress.assert_called_once_with(1, 1)

    @pytest.mark.anyio
    async def test_query_with_low_relevance_threshold(
        self,
        mock_context: Context[Any, Any],
        mock_vectorstore: Any
    ) -> None:
        """Test query filtering based on relevance score threshold"""
        config = QueryConfig(k=3, min_relevance_score=0.98)  # Set high threshold

        result = await mcp._tool_manager.call_tool(
            "query_docs",
            {
                "query": "test query",
                "config": config.model_dump()
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

        with pytest.raises(Exception, match="Query operation timed out"):
            await mcp._tool_manager.call_tool(
                "query_docs",
                {
                    "query": "test query",
                    "config": QueryConfig().model_dump()
                },
                context=mock_context
            )

        mock_context.error.assert_called_once_with("Query timed out")

    @pytest.mark.anyio
    async def test_get_all_docs_success(
        self,
        test_docs_path: Path
    ) -> None:
        """Test successful documentation retrieval"""
        content = await mcp._resource_manager.get_resource("docs://dpytest/full")

        assert isinstance(content, str)
        assert content == "Test documentation content"

    @pytest.mark.anyio
    async def test_get_all_docs_module_mismatch(self) -> None:
        """Test documentation retrieval with mismatched module"""
        with pytest.raises(ResourceError, match="Requested module 'discord' does not match"):
            await mcp._resource_manager.get_resource("docs://discord/full")
