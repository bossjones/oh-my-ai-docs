#!/usr/bin/env python3
"""Configure test fixtures for AVectorStore MCP tests."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, cast
from collections.abc import AsyncGenerator

import pytest
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import SKLearnVectorStore

# Create a protocol to represent VectorStoreRetriever's interface
# This avoids direct import issues
class VectorStoreRetrieverProtocol(Protocol):
    """Protocol defining VectorStoreRetriever's interface for testing."""
    def get_relevant_documents(self, query: str) -> list[Document]: ...

    @property
    def vectorstore(self) -> Any: ...

from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager

from tests.fake_embeddings import FakeEmbeddings

# Import vectorstore_session with proper module reference validation
try:
    from oh_my_ai_docs.avectorstore_mcp import vectorstore_session as real_vectorstore_session
except ImportError:
    # Mock for linter - in actual running this would be properly imported
    real_vectorstore_session = None

# --- Constants ---
TEST_MODULE = "dpytest"  # Default module for testing

# --- Environment Setup Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Sets up the test environment once per session.

    Scope: session - ensures module patching happens only once
    Returns: None - this fixture only applies patches

    This fixture patches sys.argv directly to simulate command-line arguments for the server
    and reloads the module to apply the patched args *before* any server instances are created.
    """
    # Store original args to restore later if needed
    original_argv = sys.argv.copy()

    # Set the new argv
    sys.argv = ['avectorstore_mcp.py', '--module', TEST_MODULE]

    # Handle potential import errors for the linter
    try:
        import oh_my_ai_docs.avectorstore_mcp
        importlib.reload(oh_my_ai_docs.avectorstore_mcp)
    except ImportError:
        # This is handled at runtime, just for linter satisfaction
        pass

    # No need to yield/restore since this is session-scoped
    # If we want to restore, we'd need to use a finalizer

# --- Server Fixtures ---

@pytest.fixture(scope="function")
def mcp_server_instance(setup_test_environment: None) -> FastMCP:
    """Provides the configured FastMCP server instance.

    Scope: function - ensures test isolation
    Args:
        setup_test_environment: Environment setup fixture (dependency)
    Returns: FastMCP server instance
    """
    try:
        from oh_my_ai_docs.avectorstore_mcp import mcp_server
        return mcp_server
    except ImportError:
        # This won't happen at runtime, just for linter satisfaction
        raise RuntimeError("oh_my_ai_docs module not found")

# --- Test Classes --- #

class AVectorStoreMCPServer:
    """Wrapper class for MCP server to use in tests"""
    def __init__(self, mcp_server: FastMCP):
        self._mcp_server = mcp_server
        self._vectorstore = None

# --- Embedding Fixtures --- #

@pytest.fixture(scope="function")
def mock_openai_embeddings(mocker: MockerFixture) -> FakeEmbeddings:
    """Fixture to provide fake embeddings, mocking the real import.

    Scope: function - ensures fresh embeddings for each test
    Args:
        mocker: pytest-mock's mocker fixture
    Returns: FakeEmbeddings instance

    This fixture mocks the OpenAIEmbeddings import in the server code to return
    fake embeddings, preventing real API calls during tests.
    """
    fake_embeddings = FakeEmbeddings()
    mocker.patch("oh_my_ai_docs.avectorstore_mcp.OpenAIEmbeddings", return_value=fake_embeddings)
    return fake_embeddings

# --- VectorStore Fixtures --- #

@pytest.fixture(scope="function")
def mock_vectorstore(mocker: MockerFixture, mock_openai_embeddings: FakeEmbeddings) -> SKLearnVectorStore:
    """Provides a mocked SKLearnVectorStore instance.

    Scope: function - ensures test isolation
    Args:
        mocker: pytest-mock's mocker fixture
        mock_openai_embeddings: The fake embeddings fixture
    Returns: Mocked SKLearnVectorStore instance

    This fixture mocks the SKLearnVectorStore class and the vectorstore_factory function
    to ensure tests don't create real vector stores or make API calls.
    """
    mock_store = mocker.MagicMock(spec=SKLearnVectorStore)
    mock_store.embedding = mock_openai_embeddings

    mock_retriever = mocker.MagicMock()  # Use generic mock instead of spec
    mock_store.as_retriever.return_value = mock_retriever

    mocker.patch("oh_my_ai_docs.avectorstore_mcp.vectorstore_factory", return_value=mock_store)
    return mock_store

@pytest.fixture(scope="function")
async def vectorstore_session() -> AsyncGenerator[SKLearnVectorStore, None]:
    """Fixture providing a mock vectorstore session.

    Scope: function - ensures test isolation
    Yields: Configured SKLearnVectorStore instance
    Cleanup: Automatically handled via AsyncGenerator
    """
    # Create the vectorstore with proper initialization
    store = SKLearnVectorStore(
        embedding=FakeEmbeddings(),
        persist_path="mock_path",
        serializer="parquet",
    )

    # Create a properly typed mock for add_documents
    original_add_documents = store.add_documents
    store.add_documents = lambda documents, **kwargs: []  # type: ignore

    # Create a properly typed mock for as_retriever
    original_as_retriever = store.as_retriever

    # Mock retriever behavior without using VectorStoreRetriever directly
    class MockRetriever:
        """Mock retriever implementation."""
        def __init__(self, vectorstore: Any):
            self.vectorstore = vectorstore

        def get_relevant_documents(self, query: str) -> list[Document]:
            """Return mock documents for testing."""
            return [
                Document(page_content="Relevant document 1", metadata={"source": "doc1", "score": 0.9}),
                Document(page_content="Relevant document 2", metadata={"source": "doc2", "score": 0.8}),
            ]

    # Configure the mock retriever
    mock_retriever = MockRetriever(vectorstore=store)
    store.as_retriever = lambda **kwargs: mock_retriever  # type: ignore

    # Yield the store for test use
    yield store

    # Restore original methods (if needed for cleanup)
    store.add_documents = original_add_documents  # type: ignore
    store.as_retriever = original_as_retriever  # type: ignore

# --- Server Fixtures --- #

@pytest.fixture(scope="function")
def mcp_server(monkeypatch: pytest.MonkeyPatch) -> AVectorStoreMCPServer:
    """Fixture to provide a configured MCP server for testing.

    Scope: function - ensures test isolation
    Args:
        monkeypatch: pytest's monkeypatch fixture
    Returns: Configured AVectorStoreMCPServer instance
    """
    try:
        # Import here to avoid import errors during fixture collection
        from oh_my_ai_docs.avectorstore_mcp import mcp_server as real_mcp_server

        # Create test server wrapper
        server = AVectorStoreMCPServer(real_mcp_server)

        # Return the test server
        return server
    except ImportError:
        # This won't happen at runtime, just for linter satisfaction
        raise RuntimeError("oh_my_ai_docs module not found")

# --- Logging Fixtures --- #

@pytest.fixture(scope="function")
def mock_vectorstore_session_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to mock vectorstore_session for logging tests.

    Scope: function - ensures test isolation
    Args:
        monkeypatch: pytest's monkeypatch fixture
    Returns: None - this fixture only applies patches
    """
    # Skip if real_vectorstore_session is None (would only happen in linter)
    if real_vectorstore_session is None:
        return

    # Original function to wrap
    original_session = real_vectorstore_session

    # Create a wrapper that adds logging
    @asynccontextmanager
    async def wrapped_session(*args: Any, **kwargs: dict[str, Any]) -> AsyncGenerator[SKLearnVectorStore, None]:
        module = kwargs.get("module", "unknown")
        logger = logging.getLogger("oh_my_ai_docs.avectorstore_mcp")
        logger.info(f"Entering vectorstore session for {module}")
        try:
            store = SKLearnVectorStore(
                embedding=FakeEmbeddings(),
                persist_path="mock_path",
                serializer="parquet",
            )

            # Create mock methods with proper signatures - avoid direct VectorStoreRetriever reference
            class MockRetriever:
                """Mock retriever for logging tests."""
                def __init__(self, vectorstore: Any):
                    self.vectorstore = vectorstore

                def get_relevant_documents(self, query: str) -> list[Document]:
                    """Return mock documents with module reference."""
                    return [
                        Document(page_content=f"Relevant doc 1 for {module}", metadata={"source": "doc1", "score": 0.9}),
                        Document(page_content=f"Relevant doc 2 for {module}", metadata={"source": "doc2", "score": 0.8}),
                    ]

            # Apply mocks with proper type handling
            store.add_documents = lambda documents, **kwargs: []  # type: ignore
            mock_retriever = MockRetriever(vectorstore=store)
            store.as_retriever = lambda **kwargs: mock_retriever  # type: ignore

            logger.info(f"Loaded vector store for {module}")
            yield store
            logger.info(f"Retrieved all 2 documents for {module}")
        finally:
            logger.info(f"Exiting vectorstore session for {module}")

    # Apply the patch
    monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.vectorstore_session", wrapped_session)
