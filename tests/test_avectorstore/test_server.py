#!/usr/bin/env python3
# pyright: reportMissingImports=false
# pyright: reportUnusedVariable=warning
# pyright: reportUntypedBaseClass=error
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import pytest
from pytest_mock import MockerFixture, AsyncMockType, MockType
from pathlib import Path
from typing import Any, Dict, TypeVar, cast, Optional
from collections.abc import Callable, Awaitable
from collections.abc import Generator
from mcp.types import TextContent, Tool, ResourceTemplate, Resource
from mcp.server.fastmcp.resources.base import Resource as FunctionResource
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.exceptions import ResourceError, ToolError
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import json

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

    info_fn: Callable[[str], Awaitable[None]] | None = Field(default=None)
    error_fn: Callable[[str], Awaitable[None]] | None = Field(default=None)
    report_progress_fn: Callable[[int, int], Awaitable[None]] | None = Field(default=None)
    app_context: Any = Field(default=None)
    mock_request_id: str = Field(default="test-request-id")

    @property
    def request_id(self) -> str:
        """Override request_id to avoid request context dependency."""
        return self.mock_request_id

    async def info(self, message: str) -> None:
        """Send info message."""
        if self.info_fn is not None:
            await self.info_fn(message)

    async def error(self, message: str) -> None:
        """Send error message."""
        if self.error_fn is not None:
            await self.error_fn(message)

    async def report_progress(self, current: int, total: int) -> None:
        """Report progress."""
        if self.report_progress_fn is not None:
            await self.report_progress_fn(current, total)

    def model_dump(self) -> dict[str, Any]:
        """Override model_dump to include our custom attributes."""
        # Access model_fields from the class instead of instance
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
        info_mock: AsyncMockType = mocker.AsyncMock(name="info")
        error_mock: AsyncMockType = mocker.AsyncMock(name="error")
        progress_mock: AsyncMockType = mocker.AsyncMock(name="report_progress")

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
    def mock_vectorstore(self, mocker: MockerFixture) -> Generator[MockType, None, None]:
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
        # assert "config" in params
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
            },
            context=mock_context
        )

        assert isinstance(result, DocumentResponse)
        assert len(result.documents) == 1
        assert result.documents[0] == "Test content"
        assert result.scores[0] == 0.95
        assert result.total_found == 1

        # Verify context method calls
        mock_info = cast(AsyncMockType, mock_context.info_fn)
        mock_progress = cast(AsyncMockType, mock_context.report_progress_fn)

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
        mock_error = cast(AsyncMockType, mock_context.error_fn)
        assert mock_error.await_count == 1
        assert mock_error.await_args_list[0][0][0] == "Query timed out"

    @pytest.mark.anyio
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
    async def test_get_all_docs_module_mismatch(self) -> None:
        """Test documentation retrieval with mismatched module"""
        with pytest.raises(ValueError, match="Requested module 'discord' does not match server module 'dpytest'"):
            await mcp_server._resource_manager.get_resource("docs://discord/full")

    @pytest.mark.anyio
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
    async def test_vectorstore_session_cleanup(self, mock_context: Context[Any, Any], mock_vectorstore: Any) -> None:
        """Test vectorstore session cleanup"""
        from oh_my_ai_docs.avectorstore_mcp import vectorstore_session, DOCS_PATH

        vectorstore_path = DOCS_PATH / "test" / "vectorstore" / "test_vectorstore.parquet"

        async with vectorstore_session(str(vectorstore_path)) as session:
            assert isinstance(session.store, mock_vectorstore.return_value.__class__)

        # Verify cleanup (mock_vectorstore cleanup would be called if implemented)
        mock_vectorstore.assert_called_once()

    @pytest.mark.anyio
    async def test_query_empty_query(self, mock_context: Context[Any, Any]) -> None:
        """Test query handling with empty query string"""
        with pytest.raises(ToolError, match="Query cannot be empty"):
            await mcp_server._tool_manager.call_tool(
                "query_docs",
                {
                    "query": "   ",  # Empty query with whitespace
                    # "config": QueryConfig().model_dump(),
                    # "ctx": mock_context
                },
                context=mock_context
            )

    @pytest.mark.anyio
    async def test_query_invalid_k(self, mock_context: Context[Any, Any]) -> None:
        """Test query config validation with invalid k value"""
        with pytest.raises(ValueError, match="Input should be less than or equal to 10"):
            QueryConfig(k=11)

    @pytest.mark.anyio
    async def test_get_all_docs_missing_file(self, test_docs_path: Path) -> None:
        """Test documentation retrieval with missing file"""
        # Remove the test docs file
        (test_docs_path / "dpytest_docs.txt").unlink()

        with pytest.raises(ValueError, match="Error creating resource from template: Documentation file not found for module: dpytest"):
            await mcp_server._resource_manager.get_resource("docs://dpytest/full")
