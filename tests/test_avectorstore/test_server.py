#!/usr/bin/env python3

from __future__ import annotations

import pytest
from pytest_mock import MockerFixture, AsyncMockType, MockType
from pathlib import Path
from typing import Any, Dict, TypeVar, cast, Optional
from collections.abc import Callable, Awaitable
from collections.abc import Generator
from mcp.types import TextContent, Tool, ResourceTemplate, Resource
from mcp.server.fastmcp.resources.base import Resource as FunctionResource
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)
from mcp.server.fastmcp.exceptions import ResourceError, ToolError
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import json
from tests.fake_embeddings import FakeEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

from oh_my_ai_docs.avectorstore_mcp import (
    mcp_server,
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
    def real_vectorstore(self, tmp_path: Path) -> Generator[SKLearnVectorStore, None, None]:
        """
        Fixture providing a real SKLearnVectorStore with FakeEmbeddings.

        Scope: function - ensures fresh vectorstore for each test
        Args:
            tmp_path: pytest fixture providing temporary directory

        Returns:
            Generator yielding vectorstore
        """
        # Create test documents
        texts = ["Test content", "Another test", "Final test"]
        metadatas = [{"score": 0.95}, {"score": 0.85}, {"score": 0.75}]

        # Initialize vectorstore with FakeEmbeddings
        store = SKLearnVectorStore.from_texts(
            texts=texts,
            embedding=FakeEmbeddings(),
            metadatas=metadatas,
        )

        yield store

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
    @pytest.mark.fastmcp_basic
    async def test_server_initialization(self) -> None:
        """Test that the MCP server is initialized with correct default name"""
        assert mcp_server.name == "dpytest-docs-mcp-server"

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
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
        assert params["query"]["type"] == "string"

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.fastmcp_context
    @pytest.mark.vectorstore
    async def test_successful_query(
        self,
        real_vectorstore: SKLearnVectorStore,
        test_docs_path: Path,
        mocker: MockerFixture
    ) -> None:
        """
        Test successful query execution through the server.

        Args:
            real_vectorstore: Real vectorstore with test data
            test_docs_path: Test documentation directory
            mocker: Pytest mocker fixture
        """
        # Mock the log message and progress update methods
        mock_log = mocker.patch("mcp.server.session.ServerSession.send_log_message")
        mock_progress = mocker.patch("mcp.server.session.ServerSession.send_progress_update")

        async with client_session(mcp_server._mcp_server) as client:
            # Patch the vectorstore in the server
            mcp_server._vectorstore = real_vectorstore

            result = await client.call_tool(
                "query_docs",
                {
                    "query": "test query",
                }
            )

            assert isinstance(result.content[0], TextContent)
            content = result.content[0]
            response = DocumentResponse.model_validate_json(content.text)

            assert len(response.documents) == 1
            assert response.documents[0] == "Test content"
            assert response.scores[0] == 0.95
            assert response.total_found == 1

            # Verify log messages
            mock_log.assert_any_call(
                level="info",
                data="Querying vectorstore with k=3",
                logger=None
            )
            mock_log.assert_any_call(
                level="info",
                data="Retrieved 1 relevant documents",
                logger=None
            )

            # Verify progress updates
            mock_progress.assert_called_once_with(1, 1)

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.fastmcp_context
    @pytest.mark.vectorstore
    async def test_query_with_low_relevance_threshold(
        self,
        real_vectorstore: SKLearnVectorStore,
        mocker: MockerFixture
    ) -> None:
        """Test query filtering based on relevance score threshold"""
        # Mock the log message method
        mock_log = mocker.patch("mcp.server.session.ServerSession.send_log_message")

        async with client_session(mcp_server._mcp_server) as client:
            # Patch the vectorstore in the server
            mcp_server._vectorstore = real_vectorstore

            result = await client.call_tool(
                "query_docs",
                {
                    "query": "test query",
                    "config": QueryConfig(k=3, min_relevance_score=0.98).model_dump()
                }
            )

            assert isinstance(result.content[0], TextContent)
            content = result.content[0]
            response = DocumentResponse.model_validate_json(content.text)

            assert len(response.documents) == 0  # Should filter out doc with score 0.95
            assert len(response.scores) == 0
            assert response.total_found == 1

            # Verify appropriate log messages
            mock_log.assert_any_call(
                level="info",
                data="Querying vectorstore with k=3",
                logger=None
            )
            mock_log.assert_any_call(
                level="info",
                data="Retrieved 1 relevant documents",
                logger=None
            )
            mock_log.assert_any_call(
                level="info",
                data="Filtered to 0 documents with min_relevance_score=0.98",
                logger=None
            )

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    @pytest.mark.fastmcp_context
    @pytest.mark.vectorstore
    async def test_query_timeout(
        self,
        real_vectorstore: SKLearnVectorStore,
        mocker: MockerFixture
    ) -> None:
        """Test query timeout handling"""
        # Make retriever.invoke take too long
        mocker.patch.object(
            real_vectorstore.as_retriever(),
            'invoke',
            side_effect=TimeoutError()
        )

        # Mock the log message method
        mock_log = mocker.patch("mcp.server.session.ServerSession.send_log_message")

        async with client_session(mcp_server._mcp_server) as client:
            # Patch the vectorstore in the server
            mcp_server._vectorstore = real_vectorstore

            with pytest.raises(ToolError, match="Query operation timed out"):
                await client.call_tool(
                    "query_docs",
                    {
                        "query": "test query",
                        "config": QueryConfig().model_dump()
                    }
                )

            # Verify error was logged
            mock_log.assert_any_call(
                level="error",
                data="Query timed out",
                logger=None
            )

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    async def test_get_all_docs_success(
        self,
        test_docs_path: Path
    ) -> None:
        """Test successful documentation retrieval"""
        resource: FunctionResource | None = await mcp_server._resource_manager.get_resource("docs://dpytest/full")

        # Verify we got a Resource
        assert isinstance(resource, FunctionResource)

        # Get the actual content by calling the resource function
        content: str = resource.fn()  # type: ignore[no-any-return]
        assert isinstance(content, str)
        assert content == "Test documentation content"

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    async def test_get_all_docs_module_mismatch(self) -> None:
        """Test documentation retrieval with mismatched module"""
        with pytest.raises(ValueError, match="Requested module 'discord' does not match server module 'dpytest'"):
            await mcp_server._resource_manager.get_resource("docs://discord/full")

    @pytest.mark.anyio
    @pytest.mark.fastmcp_basic
    async def test_argument_parsing(self) -> None:
        """Test argument parsing functionality"""
        from oh_my_ai_docs.avectorstore_mcp import parser

        # Test default arguments
        args = parser.parse_args([])
        assert args.module == "dpytest"
        assert args.stdio is True
        assert not args.debug
        assert not args.dry_run

        # Test custom arguments
        args = parser.parse_args(["--module", "discord", "--debug", "--dry-run"])
        assert args.module == "discord"
        assert args.debug is True
        assert args.dry_run is True

    @pytest.mark.fastmcp_basic
    @pytest.mark.vectorstore
    def test_list_vectorstores(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test vectorstore listing functionality"""
        from oh_my_ai_docs.avectorstore_mcp import list_vectorstores, DOCS_PATH, BASE_PATH

        # Setup test vectorstore files
        test_docs = tmp_path / "ai_docs"
        test_docs.mkdir(parents=True)

        # Create test vectorstores
        modules = ["discord", "dpytest", "langgraph"]
        for module in modules:
            module_path = test_docs / module / "vectorstore"
            module_path.mkdir(parents=True)
            (module_path / f"{module}_vectorstore.parquet").touch()

        # Patch both DOCS_PATH and BASE_PATH to use our temporary directory
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.DOCS_PATH", test_docs)
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.BASE_PATH", tmp_path)

        # Capture stdout to verify output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        list_vectorstores()

        # Restore stdout
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # Verify output contains all modules and their vectorstores
        for module in modules:
            assert module in output
            assert f"{module}_vectorstore.parquet" in output
            assert str(Path("ai_docs") / module / "vectorstore" / f"{module}_vectorstore.parquet") in output

    @pytest.mark.fastmcp_basic
    @pytest.mark.vectorstore
    def test_list_vectorstores_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test vectorstore listing with no files present"""
        from oh_my_ai_docs.avectorstore_mcp import list_vectorstores, DOCS_PATH, BASE_PATH

        # Setup empty test directory
        test_docs = tmp_path / "ai_docs"
        test_docs.mkdir(parents=True)

        # Patch paths
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.DOCS_PATH", test_docs)
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.BASE_PATH", tmp_path)

        # Capture stdout
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        list_vectorstores()

        # Restore stdout
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "No vector stores found." in output

    @pytest.mark.fastmcp_basic
    @pytest.mark.vectorstore
    def test_list_vectorstores_invalid_structure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test vectorstore listing with invalid directory structure"""
        from oh_my_ai_docs.avectorstore_mcp import list_vectorstores, DOCS_PATH, BASE_PATH

        # Setup test directory with invalid structure
        test_docs = tmp_path / "ai_docs"
        test_docs.mkdir(parents=True)
        invalid_path = test_docs / "invalid" / "vectorstore.parquet"
        invalid_path.parent.mkdir(parents=True)
        invalid_path.touch()

        # Patch paths
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.DOCS_PATH", test_docs)
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.BASE_PATH", tmp_path)

        # Capture stdout
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        list_vectorstores()

        # Restore stdout
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        assert "Total vector stores found: 1" in output
        assert "invalid" in output

    @pytest.fixture
    def mock_script_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Fixture to set up a mock script path for testing"""
        script_path = tmp_path / "src" / "oh_my_ai_docs" / "avectorstore_mcp.py"
        script_path.parent.mkdir(parents=True)
        script_path.touch()
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.__file__", str(script_path))
        return script_path

    @pytest.mark.fastmcp_basic
    def test_generate_mcp_config_all_modules(self, tmp_path: Path, mock_script_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test MCP config generation for all modules"""
        from oh_my_ai_docs.avectorstore_mcp import generate_mcp_config, BASE_PATH

        # Patch BASE_PATH
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.BASE_PATH", tmp_path)

        # Generate config
        config = generate_mcp_config()

        # Verify all modules are configured
        assert "mcpServers" in config
        for module in ["discord", "dpytest", "langgraph"]:
            server_name = f"{module}-docs-mcp-server"
            assert server_name in config["mcpServers"]
            server_config = config["mcpServers"][server_name]
            assert server_config["command"] == "uv"
            assert isinstance(server_config["args"], list)
            assert "--module" in server_config["args"]
            assert module in server_config["args"]
            assert str(tmp_path) in server_config["args"]

    @pytest.mark.fastmcp_basic
    def test_save_mcp_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test saving MCP configuration to disk"""
        from oh_my_ai_docs.avectorstore_mcp import save_mcp_config, BASE_PATH

        # Patch BASE_PATH
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.BASE_PATH", tmp_path)

        # Create test config
        test_config = {
            "mcpServers": {
                "test-server": {
                    "command": "uv",
                    "args": ["run", "test"]
                }
            }
        }

        # Save config
        save_mcp_config(test_config)

        # Verify file was created and contains correct content
        config_path = tmp_path / "mcp.json"
        assert config_path.exists()
        with open(config_path) as f:
            saved_config = json.load(f)
            assert saved_config == test_config

    @pytest.mark.anyio
    @pytest.mark.fastmcp_context
    @pytest.mark.vectorstore
    async def test_vectorstore_session_cleanup(self, test_docs_path: Path) -> None:
        """Test vectorstore session cleanup"""
        from oh_my_ai_docs.avectorstore_mcp import vectorstore_session, DOCS_PATH

        # Create a test vectorstore file
        vectorstore_path = test_docs_path / "vectorstore" / "test_vectorstore.parquet"
        vectorstore_path.parent.mkdir(exist_ok=True)

        # Create a real vectorstore and save it
        store = SKLearnVectorStore.from_texts(
            texts=["Test content"],
            embedding=FakeEmbeddings(),
            metadatas=[{"score": 0.95}],
        )
        store.save_local(str(vectorstore_path))

        async with vectorstore_session(str(vectorstore_path)) as session:
            assert isinstance(session.store, SKLearnVectorStore)
            # Test basic functionality
            results = await session.store.asimilarity_search("test", k=1)
            assert len(results) == 1
            assert results[0].page_content == "Test content"

    @pytest.mark.anyio
    @pytest.mark.fastmcp_tools
    async def test_query_empty_query(self) -> None:
        """Test query handling with empty query string"""
        async with client_session(mcp_server._mcp_server) as client:
            with pytest.raises(ToolError, match="Query cannot be empty"):
                await client.call_tool(
                    "query_docs",
                    {
                        "query": "   ",  # Empty query with whitespace
                    }
                )

    @pytest.mark.fastmcp_tools
    @pytest.mark.vectorstore
    def test_query_invalid_k(self) -> None:
        """Test query config validation with invalid k value"""
        with pytest.raises(ValueError, match="Input should be less than or equal to 10"):
            QueryConfig(k=11)

    @pytest.mark.anyio
    @pytest.mark.fastmcp_resources
    async def test_get_all_docs_missing_file(self, test_docs_path: Path) -> None:
        """Test documentation retrieval with missing file"""
        # Remove the test docs file
        (test_docs_path / "dpytest_docs.txt").unlink()

        with pytest.raises(ValueError, match="Error creating resource from template: Documentation file not found for module: dpytest"):
            await mcp_server._resource_manager.get_resource("docs://dpytest/full")

    @pytest.mark.anyio
    @pytest.mark.fastmcp_context
    @pytest.mark.fastmcp_tools
    @pytest.mark.vectorstore
    async def test_context_logging_comprehensive(
        self,
        real_vectorstore: SKLearnVectorStore,
        test_docs_path: Path,
        mocker: MockerFixture
    ) -> None:
        """Test comprehensive logging across all server functionality."""

        # Mock the send_log_message method
        mock_log = mocker.patch("mcp.server.session.ServerSession.send_log_message")

        async with client_session(mcp_server._mcp_server) as client:
            # Patch the vectorstore in the server
            mcp_server._vectorstore = real_vectorstore

            # Test 1: Empty query should trigger error log
            with pytest.raises(ValueError, match="Query cannot be empty"):
                await client.call_tool(
                    "query_docs",
                    {
                        "query": "   ",  # Empty query with whitespace
                    }
                )

            # Verify error log for empty query
            mock_log.assert_any_call(
                level="error",
                data="Query cannot be empty",
                logger=None
            )

            # Reset mock for next test
            mock_log.reset_mock()

            # Test 2: Successful query should have debug, info logs
            result = await client.call_tool(
                "query_docs",
                {
                    "query": "test query",
                }
            )

            # Verify logs for successful query
            mock_log.assert_any_call(
                level="debug",
                data=mocker.ANY,  # State will be dynamic
                logger=None
            )
            mock_log.assert_any_call(
                level="info",
                data="Querying vectorstore with k=3",
                logger=None
            )
            mock_log.assert_any_call(
                level="info",
                data=mocker.ANY,  # Number of docs will be dynamic
                logger=None
            )

            mock_log.reset_mock()

            # Test 3: Documentation retrieval with module mismatch
            with pytest.raises(ResourceError, match="Requested module 'discord' does not match server module 'dpytest'"):
                await mcp_server._resource_manager.get_resource("docs://discord/full")

            # Verify error log for module mismatch
            mock_log.assert_any_call(
                level="error",
                data="Module mismatch",
                logger=None,
                extra={"requested_module": "discord", "server_module": "dpytest"}
            )

            mock_log.reset_mock()

            # Test 4: Successful documentation retrieval
            resource = await mcp_server._resource_manager.get_resource("docs://dpytest/full")
            content = resource.fn()  # type: ignore[no-any-return]

            # Verify info logs for successful doc retrieval
            mock_log.assert_any_call(
                level="info",
                data="Retrieving documentation for module: dpytest",
                logger=None
            )
            mock_log.assert_any_call(
                level="info",
                data="Successfully read documentation",
                logger=None,
                extra={"doc_module": "dpytest", "size": mocker.ANY}
            )

            mock_log.reset_mock()

            # Test 5: Query timeout scenario
            mocker.patch.object(
                real_vectorstore.as_retriever(),
                'invoke',
                side_effect=TimeoutError()
            )

            with pytest.raises(ToolError, match="Query operation timed out"):
                await client.call_tool(
                    "query_docs",
                    {
                        "query": "test query",
                    }
                )

            # Verify error log for timeout
            mock_log.assert_any_call(
                level="error",
                data="Query timed out",
                logger=None
            )
