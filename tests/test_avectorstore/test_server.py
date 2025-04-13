#!/usr/bin/env python3
from __future__ import annotations

import pytest
from pytest_mock import MockerFixture
from pathlib import Path
from typing import Any
from collections.abc import Generator
from mcp.types import TextContent
from mcp.server.fastmcp import Context

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
    def mock_context(self, mocker: MockerFixture) -> Generator[Context, None, None]:
        """
        Fixture providing a mocked MCP Context for testing.

        Args:
            mocker: pytest-mock fixture

        Returns:
            Generator yielding mocked Context
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

        Args:
            mocker: pytest-mock fixture

        Returns:
            Generator yielding mocked vectorstore
        """
        mock_store = mocker.patch(
            "oh_my_ai_docs.avectorstore_mcp.SKLearnVectorStore",
            autospec=True
        )
        yield mock_store

    @pytest.fixture
    def mock_embeddings(self, mocker: MockerFixture) -> Generator[Any, None, None]:
        """
        Fixture providing mocked OpenAI embeddings.

        Args:
            mocker: pytest-mock fixture

        Returns:
            Generator yielding mocked embeddings
        """
        mock_embed = mocker.patch(
            "oh_my_ai_docs.avectorstore_mcp.OpenAIEmbeddings",
            autospec=True
        )
        yield mock_embed

    @pytest.fixture
    def test_docs_dir(self, tmp_path: Path) -> Generator[Path, None, None]:
        """
        Fixture creating a temporary docs directory structure.

        Args:
            tmp_path: pytest fixture providing temporary directory

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
        assert "query" in tool.parameters.properties

    @pytest.mark.anyio
    async def test_docs_resource_registration(self) -> None:
        """Test that docs resource is properly registered with correct URI template"""
        resources = mcp._resource_manager._templates

        assert len(resources) == 1
        resource = resources[0]

        assert resource.uri_template == "docs://{module}/full"
        assert resource.mime_type == "text/plain"
        assert "documentation content" in resource.description.lower()

    @pytest.mark.anyio
    async def test_empty_query_raises_error(
        self,
        mock_context: Context
    ) -> None:
        """Test that empty query raises ValueError"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await mcp._tool_manager.call_tool(
                "query_docs",
                {"query": "   "},  # Empty query with whitespace
                context=mock_context
            )
