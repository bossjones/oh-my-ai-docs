#!/usr/bin/env python3
"""Configure test fixtures for AVectorStore MCP tests."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, cast
from collections.abc import AsyncGenerator, Iterator

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

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager

from tests.fake_embeddings import FakeEmbeddings

# Import vectorstore_session with proper module reference validation
try:
    from oh_my_ai_docs.avectorstore_mcp import vectorstore_session as real_vectorstore_session, AppContext
except ImportError:
    # Mock for linter - in actual running this would be properly imported
    real_vectorstore_session = None
    # AppContext = Any # type: ignore

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

@pytest.fixture(scope="session")
def mock_openai_embeddings() -> FakeEmbeddings:
    """Provides a consistent fake embeddings generator for testing."""
    return FakeEmbeddings()

# --- VectorStore Fixtures --- #

# Remove the old mock_vectorstore fixture entirely
# Remove the old vectorstore_session fixture entirely

# Add the new mock_app_context fixture
@pytest.fixture(scope="function")
def mock_app_context(
    mocker: MockerFixture,
    mock_openai_embeddings: FakeEmbeddings,
    tmp_path: Path # Use tmp_path for persistence path in factory
) -> Iterator[AppContext]:
    """
    Provides an AppContext suitable for testing tool/resource functions.

    - Injects FakeEmbeddings globally for the test's scope via set_embeddings_provider.
    - Uses the real vectorstore_factory to create an SKLearnVectorStore instance,
      ensuring it uses the FakeEmbeddings.
    - Mocks the as_retriever() method on the created store instance to return
      a mock retriever object, allowing tests to control retriever behavior
      (e.g., mock the invoke method on the returned retriever).
    - Returns an AppContext containing the store.
    - Resets the global embeddings provider upon teardown for test isolation.
    """
    # Import necessary functions from the module under test
    from oh_my_ai_docs.avectorstore_mcp import (
        _EMBEDDINGS_PROVIDER,
        set_embeddings_provider,
        vectorstore_factory,
        AppContext,
    )
    from langchain_core.vectorstores import VectorStoreRetriever # For spec

    # Inject fake embeddings for this test's scope
    original_embeddings = _EMBEDDINGS_PROVIDER
    set_embeddings_provider(mock_openai_embeddings)

    try:
        # Create a store using the real factory, which will pick up the fake embeddings
        # Provide a temporary path for persistence
        vectorstore_path = tmp_path / "mock_vectorstore.parquet"
        store = vectorstore_factory(
            vector_store_cls=SKLearnVectorStore,
            vector_store_kwargs={
                "persist_path": str(vectorstore_path),
                "serializer": "parquet",
                # embeddings are handled by the factory via get_embeddings_provider
            }
        )

        # Mock the as_retriever METHOD of the store INSTANCE
        # It should return a mock RETRIEVER object.
        mock_retriever_instance = mocker.MagicMock(spec=VectorStoreRetriever)
        # Tests can now configure this instance, e.g.:
        # retriever = app_context.store.as_retriever()
        # retriever.invoke = AsyncMock(...)
        mocker.patch.object(store, 'as_retriever', return_value=mock_retriever_instance)


        # Create the AppContext with the configured store
        app_context = AppContext(store=store)

        yield app_context

    finally:
        # Restore original embeddings provider after the test
        set_embeddings_provider(original_embeddings)

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
