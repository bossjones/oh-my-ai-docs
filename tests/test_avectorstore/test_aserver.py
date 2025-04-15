#!/usr/bin/env python3

"""Okay, let's construct a test_server.py file from scratch, drawing heavily on the structure and techniques demonstrated in the provided context (avectorstore_mcp.py and the example test snippets).

This version aims for clarity, good fixture usage, and comprehensive testing of the server's core MCP functionality (tools and resources)."""

# --- Core Imports ---
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast
from collections.abc import AsyncGenerator, Generator

# --- Testing Imports ---
from langchain_core.utils.iter import Tee
import pytest
from pytest_mock import MockerFixture
from pytest import MonkeyPatch

# --- Langchain/AI Imports ---
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

# --- MCP Imports ---
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.session import ServerSession
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session as client_session
from mcp.types import TextContent, Tool, Resource

# --- Project Imports ---
# Assume the server code is accessible, adjust path if needed
import oh_my_ai_docs.avectorstore_mcp
from oh_my_ai_docs.avectorstore_mcp import (
    AppContext,
    DocumentResponse,
    QueryConfig,
    ToolError,
    vectorstore_factory, # Use the factory
    get_vectorstore_path,
)
from tests.fake_embeddings import FakeEmbeddings # Use the fake embeddings for testing

# --- Constants ---
TEST_MODULE = "dpytest" # Default module for testing

# --- Fixtures ---

@pytest.fixture(scope="function")
def mock_app_context(mocker: MockerFixture, mock_vectorstore: SKLearnVectorStore) -> AppContext:
    """Provides a mocked AppContext, simulating the lifespan context."""
    app_ctx = AppContext(store=mock_vectorstore)
    # Mock the get_context().request_context.lifespan_context chain
    # This is tricky, we need to mock how the server retrieves the context
    mock_req_ctx = mocker.MagicMock()
    mock_req_ctx.lifespan_context = app_ctx
    mock_server_ctx = mocker.MagicMock(spec=Context)
    mock_server_ctx.request_context = mock_req_ctx
    # Patch the server's get_context method
    mocker.patch("oh_my_ai_docs.avectorstore_mcp.mcp_server.get_context", return_value=mock_server_ctx)
    return app_ctx

@pytest.fixture(scope="function")
def test_file_structure(tmp_path: Path, monkeypatch: MonkeyPatch) -> dict[str, Path]:
    """Creates a temporary file structure for testing resources and patches paths."""
    base_dir = tmp_path / "test_repo"
    docs_root = base_dir / "docs" / "ai_docs"
    module_docs_path = docs_root / TEST_MODULE
    vectorstore_path = module_docs_path / "vectorstore"

    # Create directories
    vectorstore_path.mkdir(parents=True, exist_ok=True)

    # Create dummy docs file
    docs_file = module_docs_path / f"{TEST_MODULE}_docs.txt"
    docs_file.write_text(f"Full documentation content for {TEST_MODULE}.")

    # Create dummy vectorstore file (though it won't be loaded by mock)
    vectorstore_file = vectorstore_path / f"{TEST_MODULE}_vectorstore.parquet"
    vectorstore_file.touch()

    # Patch the paths used in the server code
    monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.BASE_PATH", base_dir)
    monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.DOCS_PATH", docs_root)

    # Mock aiofiles operations
    # We need to mock aiofiles globally for the test function's scope
    # This is better done within the test or using a dedicated fixture if complex

    return {
        "base": base_dir,
        "docs_root": docs_root,
        "module_docs": module_docs_path,
        "vectorstore": vectorstore_path,
        "docs_file": docs_file,
        "vectorstore_file": vectorstore_file,
    }

# --- Test Class ---

class TestAVectorStoreMCPServer:
    """Test suite for the AVectorStore MCP Server."""

    # -- Initialization and Registration Tests --

    @pytest.mark.anyio
    @pytest.mark.fastmcp_basic
    async def test_server_initialization(self, mcp_server_instance: FastMCP):
        """Verify the server name is correctly set based on args."""
        assert mcp_server_instance.name == f"{TEST_MODULE}-docs-mcp-server"

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    async def test_tool_registration(self, mcp_server_instance: FastMCP):
        """Check if the 'query_docs' tool is registered correctly."""
        tools = mcp_server_instance._tool_manager.list_tools()
        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, Tool)
        assert tool.name == "query_docs"
        assert "Search through module documentation" in tool.description
        assert tool.parameters is not None
        props = tool.parameters.get("properties", {})
        assert "query" in props
        assert props["query"]["type"] == "string"
        # Check if QueryConfig parameters are reflected (optional, depends on FastMCP internals)
        # assert "k" in props
        # assert props["k"]["type"] == "integer"
        # assert props["k"]["default"] == 3

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    async def test_resource_registration(self, mcp_server_instance: FastMCP):
        """Check if the 'module_documentation' resource is registered correctly."""
        # Resources defined with URI templates are stored differently
        templates = mcp_server_instance._resource_manager._templates
        assert len(templates) == 1
        template_uri = f"docs://{{module}}/full" # Note the curly braces for template
        assert template_uri in templates
        resource_template = templates[template_uri]
        # We can't easily get a Resource object directly without resolving
        # Check the template details if possible (depends on FastMCP structure)
        assert resource_template.name == "module_documentation"
        if resource_template.description is not None:
            assert "Retrieves the full documentation content" in resource_template.description
        assert resource_template.mime_type == "text/plain"

    # -- Tool ('query_docs') Tests --

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.vectorstore
    async def test_query_success(
        self,
        mock_app_context: AppContext, # Ensures context is mocked
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Test a successful query returning documents."""
        # Configure the mock retriever within the mocked AppContext's store
        mock_retriever = mock_app_context.store.as_retriever()
        test_docs = [
            Document(page_content="Relevant doc 1", metadata={"score": 0.9}),
            Document(page_content="Relevant doc 2", metadata={"score": 0.8}),
        ]
        # Mock the actual call used by the tool (asyncio.to_thread(retriever.invoke, ...))
        mock_invoke = mocker.patch("asyncio.to_thread", return_value=test_docs)

        query_text = "find relevant info"
        async with client_session(mcp_server_instance._mcp_server) as client:
            result = await client.call_tool("query_docs", {"query": query_text})

        assert result is not None
        assert not result.isError
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        # The tool returns a DocumentResponse model, which FastMCP likely serializes to JSON string
        response_data = json.loads(result.content[0].text)
        assert isinstance(response_data, dict)
        assert response_data["documents"] == ["Relevant doc 1", "Relevant doc 2"]
        assert response_data["scores"] == [0.9, 0.8]
        assert response_data["total_found"] == 2

        # Verify retriever interaction - use a different approach to check if as_retriever was called
        # We can't use .assert_any_call() since it's not a mock method
        # Just verify that mock_invoke was called correctly
        mock_invoke.assert_called_once()
        call_args, call_kwargs = mock_invoke.call_args
        assert call_args[0] == mock_retriever.invoke # Check the function being called
        assert call_args[1] == query_text # Check the query argument

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    async def test_query_empty_string(self, mcp_server_instance: FastMCP):
        """Test calling the query tool with an empty string."""
        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.call_tool("query_docs", {"query": "  "}) # Whitespace only

        # FastMCP should catch the ValueError raised by the tool and convert it
        assert "Query cannot be empty" in str(exc_info.value)
        # The error code might vary depending on FastMCP's mapping
        # assert exc_info.value.code == "INVALID_PARAMS" or exc_info.value.code == -32602

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.vectorstore
    async def test_query_retriever_error(
        self,
        mock_app_context: AppContext,
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Test when the underlying retriever raises an exception."""
        mock_retriever = mock_app_context.store.as_retriever()
        error_message = "Vector store connection failed"
        # Mock asyncio.to_thread to raise an exception
        mock_invoke = mocker.patch("asyncio.to_thread", side_effect=RuntimeError(error_message))

        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.call_tool("query_docs", {"query": "trigger error"})

        # The server should catch the underlying error and wrap it
        assert "Failed to query vectorstore" in str(exc_info.value)
        assert error_message in str(exc_info.value) # Include original error
        # assert exc_info.value.code == "INTERNAL_ERROR" or exc_info.value.code == -32603

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.vectorstore
    async def test_query_timeout(
        self,
        mock_app_context: AppContext,
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Test when the query operation times out."""
        mock_retriever = mock_app_context.store.as_retriever()
        # Mock asyncio.to_thread to raise TimeoutError
        mock_invoke = mocker.patch("asyncio.to_thread", side_effect=asyncio.TimeoutError("Query took too long"))

        # The server code currently doesn't have an explicit timeout block,
        # so this test assumes the underlying call might timeout or FastMCP handles it.
        # If the server *did* have `async with timeout(...)`, this mock would trigger it.
        # Let's assume the server catches the TimeoutError from the mocked call.
        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.call_tool("query_docs", {"query": "timeout query"})

        # Check if the server translates TimeoutError appropriately
        assert "Query operation timed out" in str(exc_info.value) # Based on server's catch block
        # assert exc_info.value.code == "INTERNAL_ERROR" # Or a specific timeout code if defined

    # -- Resource ('module_documentation') Tests --

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    async def test_resource_success(
        self,
        test_file_structure: dict[str, Path], # Provides paths and patches BASE/DOCS_PATH
        mocker: MockerFixture,
        mock_app_context: AppContext, # Needed to mock context access
        mcp_server_instance: FastMCP
    ):
        """Test successfully reading the full documentation resource."""
        expected_content = f"Full documentation content for {TEST_MODULE}."
        docs_file_path = test_file_structure["docs_file"]

        # Mock aiofiles.open to return the expected content
        mock_async_file = mocker.AsyncMock()
        mock_async_file.read.return_value = expected_content
        # Use __aenter__ and __aexit__ for async context manager mocking
        mock_open = mocker.patch("aiofiles.open", return_value=mocker.MagicMock(__aenter__=mocker.AsyncMock(return_value=mock_async_file), __aexit__=mocker.AsyncMock()))
        # Mock path check
        mocker.patch("aiofiles.os.path.exists", return_value=True) # Assume file exists

        resource_uri = f"docs://{TEST_MODULE}/full"
        async with client_session(mcp_server_instance._mcp_server) as client:
            result = await client.read_resource(resource_uri)

        assert result is not None
        assert len(result.contents) == 1
        content = result.contents[0]
        assert isinstance(content, TextContent)
        assert content.text == expected_content
        assert content.mimeType == "text/plain" # As defined in the decorator

        # Verify aiofiles.open was called with the correct path
        mock_open.assert_called_once_with(docs_file_path)
        mock_async_file.read.assert_called_once()

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    async def test_resource_module_mismatch(
        self,
        test_file_structure: dict[str, Path],
        mock_app_context: AppContext,
        mcp_server_instance: FastMCP
    ):
        """Test reading resource with a module name different from the server's."""
        wrong_module = "other_module"
        resource_uri = f"docs://{wrong_module}/full"

        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.read_resource(resource_uri)

        # Server should raise ResourceError, FastMCP converts it
        assert f"Requested module '{wrong_module}' does not match server module '{TEST_MODULE}'" in str(exc_info.value)
        # assert exc_info.value.code == "RESOURCE_ERROR" # Or similar code

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    async def test_resource_file_not_found(
        self,
        test_file_structure: dict[str, Path],
        mocker: MockerFixture,
        mock_app_context: AppContext,
        mcp_server_instance: FastMCP
    ):
        """Test reading resource when the underlying documentation file is missing."""
        # Mock aiofiles.os.path.exists to return False
        mock_exists = mocker.patch("aiofiles.os.path.exists", return_value=False)

        resource_uri = f"docs://{TEST_MODULE}/full"
        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.read_resource(resource_uri)

        assert "Documentation file not found" in str(exc_info.value)
        # assert exc_info.value.code == "RESOURCE_UNAVAILABLE" # Or similar

        # Verify the path check was made
        docs_file_path = test_file_structure["docs_file"]
        mock_exists.assert_called_once_with(docs_file_path)

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    async def test_resource_read_error(
        self,
        test_file_structure: dict[str, Path],
        mocker: MockerFixture,
        mock_app_context: AppContext,
        mcp_server_instance: FastMCP
    ):
        """Test reading resource when aiofiles raises an error during read."""
        error_message = "Disk read permission denied"
        # Mock aiofiles.open to raise an OSError
        mock_open = mocker.patch("aiofiles.open", side_effect=OSError(error_message))
        # Mock path check
        mocker.patch("aiofiles.os.path.exists", return_value=True)

        resource_uri = f"docs://{TEST_MODULE}/full"
        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.read_resource(resource_uri)

        assert "Error reading documentation file" in str(exc_info.value)
        assert error_message in str(exc_info.value) # Include original error
        # assert exc_info.value.code == "RESOURCE_ERROR" # Or similar

    # -- Lifespan / Context Tests --

    @pytest.mark.anyio
    @pytest.mark.fastmcp_context
    @pytest.mark.fastmcp_tools
    @pytest.mark.vectorstore
    async def test_lifespan_context_available_in_tool(
        self,
        mock_app_context: AppContext, # Fixture sets up the mock context
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Verify that the AppContext from the lifespan is accessible within the tool."""
        # We already mocked get_context in the mock_app_context fixture.
        # Now, we just need to ensure the tool *uses* it.
        # Spy on the store access within the tool
        store_spy = mocker.spy(mock_app_context.store, "as_retriever")

        # Mock the actual retrieval part to avoid errors
        mock_retriever = mock_app_context.store.as_retriever()
        mocker.patch("asyncio.to_thread", return_value=[Document(page_content="Doc")])

        # Call the tool
        async with client_session(mcp_server_instance._mcp_server) as client:
            await client.call_tool("query_docs", {"query": "check context"})

        # Assert that the store's method (accessed via context) was called
        store_spy.assert_called_once()
