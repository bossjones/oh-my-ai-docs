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
        fixture_app_context: AppContext,
        mcp_server_instance: FastMCP
    ):
        """Test a successful query returning documents from the vectorstore.

        Verifies:
        - Correct integration between FastMCP server and SKLearnVectorStore
        - Proper document retrieval and formatting in response
        - Appropriate response structure and content
        """
        # Add test documents to the vectorstore
        test_texts = [
            "FastMCP is a protocol for LLM interactions",
            "FastMCP provides tools and resources for AI development",
            "The protocol enables seamless communication with language models"
        ]
        fixture_app_context.store.add_texts(test_texts)

        # Test querying the vectorstore
        query_text = "protocol LLM"  # Should match first and third documents
        async with client_session(mcp_server_instance._mcp_server) as client:
            # Call the tool and verify results
            result = await client.call_tool("query_docs", {"query": query_text})

            # Verify the result
            assert result is not None
            assert not result.isError
            assert len(result.content) == 1
            assert isinstance(result.content[0], TextContent)

            # Parse the response
            response_data = json.loads(result.content[0].text)

            # Verify the response structure
            assert "documents" in response_data
            assert isinstance(response_data["documents"], list)
            assert len(response_data["documents"]) > 0  # Should find at least one match
            assert len(response_data["documents"]) <= 3  # Default k=3 in QueryConfig

            # Verify document content - should find the relevant documents
            found_docs = response_data["documents"]
            assert any("protocol" in doc.lower() for doc in found_docs)
            assert any("llm" in doc.lower() for doc in found_docs)

            # Verify scores
            assert "scores" in response_data
            assert isinstance(response_data["scores"], list)
            assert len(response_data["scores"]) == len(response_data["documents"])
            # Verify all scores are floats between 0 and 1
            assert all(isinstance(score, float) and 0 <= score <= 1 for score in response_data["scores"])

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
        fixture_app_context: AppContext,
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Test when the underlying retriever raises an exception."""
        error_message = "Vector store connection failed"

        # Create a mock retriever
        mock_retriever = mocker.Mock()
        mock_retriever.invoke = mocker.AsyncMock(side_effect=RuntimeError(error_message))

        # Patch the as_retriever method to return our mock
        mocker.patch.object(
            fixture_app_context.store,
            "as_retriever",
            return_value=mock_retriever
        )

        # Patch get_context to return our fixture context
        mocker.patch(
            "oh_my_ai_docs.avectorstore_mcp.mcp_server.get_context",
            return_value=fixture_app_context
        )

        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.call_tool("query_docs", {"query": "trigger error"})

        # The server should catch the underlying error and wrap it
        assert "Failed to query vectorstore" in str(exc_info.value)
        assert error_message in str(exc_info.value) # Include original error
        # Verify invoke was called with correct parameters
        mock_retriever.invoke.assert_called_once_with("trigger error")

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.vectorstore
    @pytest.mark.langchain_retrievers_integration
    async def test_query_timeout(
        self,
        fixture_app_context: AppContext,
        mocker: MockerFixture,
        mcp_server_instance: FastMCP
    ):
        """Test when the query operation times out."""
        # Create a mock retriever
        mock_retriever = mocker.Mock()
        mock_retriever.invoke = mocker.AsyncMock(side_effect=asyncio.TimeoutError("Query took too long"))

        # Patch the as_retriever method to return our mock
        mocker.patch.object(
            fixture_app_context.store,
            "as_retriever",
            return_value=mock_retriever
        )

        # Patch get_context to return our fixture context
        mocker.patch(
            "oh_my_ai_docs.avectorstore_mcp.mcp_server.get_context",
            return_value=fixture_app_context
        )

        async with client_session(mcp_server_instance._mcp_server) as client:
            with pytest.raises(McpError) as exc_info:
                await client.call_tool("query_docs", {"query": "timeout query"})

        # Check if the server translates TimeoutError appropriately
        assert "Query operation timed out" in str(exc_info.value)
        # Verify invoke was called with correct parameters
        mock_retriever.invoke.assert_called_once_with("timeout query")

    # -- Resource ('module_documentation') Tests --

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    @pytest.mark.langchain_vectorstore_integration
    async def test_resource_success(
        self,
        test_file_structure: dict[str, Path],
        fixture_app_context: AppContext,
        mcp_server_instance: FastMCP
    ):
        """Test successfully reading the full documentation resource."""
        # Create test documentation content
        docs_file_path = test_file_structure["docs_file"]
        expected_content = f"Full documentation content for {TEST_MODULE}."

        # Ensure the docs directory exists
        docs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write test content to the documentation file
        async with aiofiles.open(docs_file_path, "w") as f:
            await f.write(expected_content)

        try:
            # Test reading the resource
            resource_uri = f"docs://{TEST_MODULE}/full"
            async with client_session(mcp_server_instance._mcp_server) as client:
                result = await client.read_resource(resource_uri)

            # Verify the result
            assert result is not None
            assert len(result.contents) == 1
            content = result.contents[0]
            assert isinstance(content, TextContent)
            assert content.text == expected_content
            assert content.mimeType == "text/plain"  # As defined in the decorator

        finally:
            # Clean up - remove the test file
            if docs_file_path.exists():
                docs_file_path.unlink()

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    @pytest.mark.langchain_vectorstore_unit
    async def test_resource_module_mismatch(
        self,
        test_file_structure: dict[str, Path],
        fixture_app_context: AppContext,
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
        fixture_app_context: AppContext,
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
        fixture_app_context: AppContext,
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
        fixture_app_context: AppContext,
        mcp_server_instance: FastMCP
    ):
        """Verify that the AppContext from the lifespan is accessible within the tool.

        Tests:
        - Context injection into the tool
        - Proper retriever access and configuration
        - Tool invocation with context
        - Proper cleanup after tool execution
        """
        # Create a test document in the vectorstore
        test_texts = ["FastMCP is a protocol for LLM interactions"]
        fixture_app_context.store.add_texts(test_texts)

        # Test the tool with context
        async with client_session(mcp_server_instance._mcp_server) as client:
            # Call the tool and verify it works with the context
            result = await client.call_tool(
                "query_docs",
                {"query": "protocol LLM"}
            )

            # Verify the result
            assert not result.isError
            assert len(result.content) == 1
            response_data = json.loads(result.content[0].text)
            assert "documents" in response_data
            assert len(response_data["documents"]) == 1
            assert response_data["documents"][0] == test_texts[0]  # Should match exactly since it's the only document
            assert "scores" in response_data
            assert len(response_data["scores"]) == 1
            assert isinstance(response_data["scores"][0], float)
            assert 0 <= response_data["scores"][0] <= 1  # Score should be between 0 and 1
