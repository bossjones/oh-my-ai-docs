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
from mcp.server.fastmcp.tools.base import Tool
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.session import ServerSession
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session as client_session
from mcp.types import TextContent

# --- Project Imports ---
import oh_my_ai_docs.avectorstore_mcp
from oh_my_ai_docs.avectorstore_mcp import (
    AppContext,
    DocumentResponse,
    QueryConfig,
    ToolError,
    vectorstore_factory,
    get_vectorstore_path,
)
from tests.fake_embeddings import FakeEmbeddings

# --- Constants ---
TEST_MODULE = "dpytest"

# --- Test Class ---

class TestAVectorStoreMCPServer:
    """Test suite for the AVectorStore MCP Server."""

    # -- Initialization and Registration Tests --

    @pytest.mark.anyio
    @pytest.mark.fastmcp_basic
    @pytest.mark.langchain_tool_integration
    async def test_server_initialization(self, mcp_server_instance: FastMCP):
        """Verify the server name is correctly set based on args."""
        assert mcp_server_instance.name == f"{TEST_MODULE}-docs-mcp-server"

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.langchain_tool_integration
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
    @pytest.mark.langchain_vectorstore_integration
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
    @pytest.mark.langchain_vectorstore_integration
    async def test_query_success(
        self,
        fastmcp_vector_store: SKLearnVectorStore,
        sample_texts: list[str],
        mcp_server_instance: FastMCP,
        mocker: MockerFixture
    ):
        """Test a successful query returning documents from the vectorstore.

        Verifies:
        - Correct integration between FastMCP server and SKLearnVectorStore
        - Proper document retrieval and formatting in response
        - Appropriate logging and progress reporting
        """
        # Only mock logging/progress reporting as per rules
        mock_log = mocker.patch("mcp.server.session.ServerSession.send_log_message")
        mock_progress = mocker.patch("mcp.server.session.ServerSession.send_progress_update")

        # Set up the embeddings provider for the test
        oh_my_ai_docs.avectorstore_mcp.set_embeddings_provider(FakeEmbeddings(size=128))

        query_text = "protocol LLM"  # Should match first sample text
        async with client_session(mcp_server_instance._mcp_server) as client:
            result = await client.call_tool("query_docs", {"query": query_text})

        # Verify the result
        assert result is not None
        assert not result.isError
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        # Parse the response
        response_data = json.loads(result.content[0].text)
        assert isinstance(response_data, dict)
        assert len(response_data["documents"]) > 0
        # First result should be the most relevant text about FastMCP and LLM
        assert "FastMCP" in response_data["documents"][0]
        assert "LLM" in response_data["documents"][0]

        # Verify logging was called appropriately
        mock_log.assert_any_call(
            level="info",
            data="Querying vectorstore with k=3",  # Default k value
            logger=None
        )
        mock_log.assert_any_call(
            level="info",
            data=mocker.ANY,  # Number of docs may vary
            logger=None
        )

        # Verify progress was reported
        mock_progress.assert_called()

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.langchain_tool_integration
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
    @pytest.mark.langchain_retrievers_integration
    async def test_query_retriever_error(
        self,
        mock_app_context: AppContext,
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Test when the underlying retriever raises an exception."""
        error_message = "Vector store connection failed"

        # Get the mock retriever instance
        mock_retriever = mock_app_context.store.as_retriever()
        # Mock the invoke method ON the retriever instance to raise an error
        mock_retriever.invoke = mocker.AsyncMock(side_effect=RuntimeError(error_message))

        # Patch get_context if necessary (as above)
        mocker.patch("oh_my_ai_docs.avectorstore_mcp.mcp_server.get_context", return_value=mock_app_context)

        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.call_tool("query_docs", {"query": "trigger error"})

        # The server should catch the underlying error and wrap it
        assert "Failed to query vectorstore" in str(exc_info.value)
        assert error_message in str(exc_info.value) # Include original error
        # assert exc_info.value.code == "INTERNAL_ERROR" or exc_info.value.code == -32603
        # Verify as_retriever and invoke were called
        mock_app_context.store.as_retriever.assert_called_once()
        mock_retriever.invoke.assert_called_once_with("trigger error")

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.vectorstore
    @pytest.mark.langchain_retrievers_integration
    async def test_query_timeout(
        self,
        mock_app_context: AppContext,
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Test when the query operation times out."""
        # Get the mock retriever instance
        mock_retriever = mock_app_context.store.as_retriever()
        # Mock the invoke method ON the retriever instance to raise TimeoutError
        mock_retriever.invoke = mocker.AsyncMock(side_effect=asyncio.TimeoutError("Query took too long"))

        # Patch get_context if necessary (as above)
        mocker.patch("oh_my_ai_docs.avectorstore_mcp.mcp_server.get_context", return_value=mock_app_context)

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
        # Verify as_retriever and invoke were called
        mock_app_context.store.as_retriever.assert_called_once()
        mock_retriever.invoke.assert_called_once_with("timeout query")

    # -- Resource ('module_documentation') Tests --

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    @pytest.mark.langchain_vectorstore_integration
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
    @pytest.mark.langchain_vectorstore_unit
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
    @pytest.mark.langchain_vectorstore_unit
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
    @pytest.mark.langchain_vectorstore_unit
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
    @pytest.mark.langchain_vectorstore_integration
    async def test_lifespan_context_available_in_tool(
        self,
        mock_app_context: AppContext, # Fixture sets up the mock context
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Verify that the AppContext from the lifespan is accessible within the tool."""
        # Patch get_context to ensure the tool receives our mock context
        # This assumes the tool uses `mcp_server.get_context()` instead of directly
        # accessing the lifespan context from the raw request context.
        mocker.patch("oh_my_ai_docs.avectorstore_mcp.mcp_server.get_context", return_value=mock_app_context)

        # Get the mock retriever instance from the context's store
        mock_retriever = mock_app_context.store.as_retriever()
        # Mock the actual retrieval part to avoid errors and return a dummy doc
        mock_retriever.invoke = mocker.AsyncMock(return_value=[Document(page_content="Doc")])

        # Call the tool
        async with client_session(mcp_server_instance._mcp_server) as client:
            await client.call_tool("query_docs", {"query": "check context"})

        # Assert that the store's as_retriever method was called
        mock_app_context.store.as_retriever.assert_called_once()
        # Assert that the invoke method on the returned retriever was called
        mock_retriever.invoke.assert_called_once_with("check context")
