"""Tests for avectorstore_mcp.py"""
import sys
import pytest
from pathlib import Path
from typing import Any
from collections.abc import Generator
from pytest_mock import MockerFixture
from _pytest.monkeypatch import MonkeyPatch
from mcp.types import TextContent
from mcp.server.fastmcp.exceptions import ResourceError

# Mock sys.argv before importing the module
sys.argv = ["avectorstore_mcp.py", "--module", "discord"]

from oh_my_ai_docs.avectorstore_mcp import (
    mcp_server,
    QueryConfig,
    DocumentResponse,
    MCPError,
    ToolError,
    BASE_PATH,
    DOCS_PATH,
    get_args
)

# Fixtures
@pytest.fixture
def mock_args(monkeypatch: MonkeyPatch) -> None:
    """Mock command line arguments"""
    monkeypatch.setattr(sys, "argv", ["avectorstore_mcp.py", "--module", "discord"])

@pytest.fixture
def mock_vectorstore(mocker: MockerFixture) -> Any:
    """Mock SKLearnVectorStore"""
    mock = mocker.patch("langchain_community.vectorstores.SKLearnVectorStore")
    mock.return_value.as_retriever.return_value.invoke.return_value = [
        mocker.Mock(
            page_content="Test content",
            metadata={"score": 0.95}
        )
    ]
    return mock

@pytest.fixture
def mock_context(mocker: MockerFixture) -> Any:
    """Mock MCP Context"""
    context = mocker.Mock()
    context.info = mocker.AsyncMock()
    context.error = mocker.AsyncMock()
    context.report_progress = mocker.AsyncMock()
    return context

@pytest.fixture
def mock_docs_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create mock documentation file"""
    docs_dir = tmp_path / "docs" / "ai_docs" / "discord"
    docs_dir.mkdir(parents=True)
    docs_file = docs_dir / "discord_docs.txt"
    docs_file.write_text("Test documentation content")
    yield docs_file

class TestAVectorStoreMCP:
    """Test suite for AVectorStore MCP server"""

    def test_mcp_instance_configuration(self, mock_args):
        """Test that mcp instance is configured correctly"""
        assert mcp.name == "discord-docs-mcp-server"  # default module

    def test_command_line_args(self, mock_args):
        """Test command line argument parsing"""
        args = get_args()
        assert args.module == "discord"
        assert args.stdio is True
        assert not args.debug
        assert not args.dry_run

    def test_query_tool_exists(self, mock_args):
        """Test that query_docs tool is registered"""
        tools = mcp._tool_manager.list_tools()
        assert any(tool.name == "query_docs" for tool in tools)

    def test_query_tool_configuration(self, mock_args):
        """Test query_docs tool configuration"""
        tool = next(t for t in mcp._tool_manager.list_tools() if t.name == "query_docs")
        assert tool.description is not None
        assert "Search through module documentation" in tool.description

    def test_docs_resource_exists(self, mock_args):
        """Test that docs resource is registered"""
        templates = mcp._resource_manager._templates
        assert any(t.name == "module_documentation" for t in templates)

    def test_docs_resource_configuration(self, mock_args):
        """Test docs resource configuration"""
        resource = next(t for t in mcp._resource_manager._templates
                     if t.name == "module_documentation")
        assert resource.mime_type == "text/plain"
        assert "docs://{module}/full" in str(resource.uri_template)

    @pytest.mark.anyio
    async def test_query_tool_execution(self, mock_args, mock_vectorstore: Any, mock_context: Any):
        """Test query_docs tool execution"""
        config = QueryConfig(k=3, min_relevance_score=0.0)
        result = await mcp._tool_manager.get_tool("query_docs").invoke(
            "test query",
            mock_context,
            config
        )
        assert isinstance(result, DocumentResponse)
        assert len(result.documents) > 0
        assert all(isinstance(score, float) for score in result.scores)
        assert result.total_found > 0

    @pytest.mark.anyio
    async def test_query_tool_empty_query(self, mock_args, mock_context: Any):
        """Test query_docs with empty query"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await mcp._tool_manager.get_tool("query_docs").invoke(
                "   ",
                mock_context
            )

    @pytest.mark.anyio
    async def test_query_tool_timeout(self, mock_args, mock_vectorstore: Any, mock_context: Any, mocker: MockerFixture):
        """Test query_docs timeout handling"""
        # Mock the vectorstore to simulate a timeout
        mock_vectorstore.return_value.as_retriever.return_value.invoke.side_effect = TimeoutError("Query timed out")

        with pytest.raises(ToolError, match="Query operation timed out"):
            await mcp._tool_manager.get_tool("query_docs").invoke(
                "test query",
                mock_context
            )

    @pytest.mark.anyio
    async def test_docs_resource_access(self, mock_args, mock_docs_file: Path, monkeypatch: MonkeyPatch):
        """Test docs resource access"""
        # Patch the DOCS_PATH to use our mock file
        monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.DOCS_PATH", mock_docs_file.parent.parent.parent)

        result = await mcp._resource_manager.get_resource("docs://discord/full").get()
        assert isinstance(result, str)
        assert "Test documentation content" in result

    @pytest.mark.anyio
    async def test_docs_resource_invalid_module(self, mock_args):
        """Test docs resource with invalid module"""
        with pytest.raises(ResourceError):
            await mcp._resource_manager.get_resource("docs://invalid/full").get()

    @pytest.mark.anyio
    async def test_query_tool_with_config(self, mock_args, mock_vectorstore: Any, mock_context: Any):
        """Test query_docs with custom configuration"""
        config = QueryConfig(k=5, min_relevance_score=0.8)
        result = await mcp._tool_manager.get_tool("query_docs").invoke(
            "test query",
            mock_context,
            config
        )
        assert isinstance(result, DocumentResponse)
        assert all(score >= 0.8 for score in result.scores)

    def test_query_config_validation(self, mock_args):
        """Test QueryConfig validation"""
        # Test valid config
        config = QueryConfig(k=5, min_relevance_score=0.5)
        assert config.k == 5
        assert config.min_relevance_score == 0.5

        # Test invalid k
        with pytest.raises(ValueError):
            QueryConfig(k=11)  # k cannot be greater than 10

        # Test invalid min_relevance_score
        with pytest.raises(ValueError):
            QueryConfig(min_relevance_score=1.5)  # must be between 0 and 1
