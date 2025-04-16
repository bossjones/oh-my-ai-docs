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

@pytest.fixture(scope="function")
def test_file_structure(tmp_path: Path, monkeypatch: MonkeyPatch) -> dict[str, Path | str]:
    """Creates a temporary file structure for testing resources and patches paths.

    Scope: function - ensures test isolation
    Args:
        tmp_path: pytest's temporary path fixture
        monkeypatch: pytest's monkeypatch fixture
    Returns: Dictionary containing paths to test directories and files

    Creates a standardized file structure for vectorstore testing that mirrors
    the actual dpytest documentation structure from tests/fixtures/dpytest:
    - base_dir/
      - docs/
        - ai_docs/
          - dpytest/
            - vectorstore/
              - dpytest_vectorstore.parquet
            - dpytest_docs.txt (contains real dpytest docs)
            - llms.txt (contains real URL mappings)
            - summaries/
              - dpytest.readthedocs.io_*.txt (real doc summaries)
    """
    # Set up base paths
    base_dir = tmp_path / "test_repo"
    docs_root = base_dir / "docs" / "ai_docs"
    module_docs_path = docs_root / TEST_MODULE
    vectorstore_path = module_docs_path / "vectorstore"
    summaries_path = module_docs_path / "summaries"

    # Create all required directories
    for path in [vectorstore_path, summaries_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Copy real dpytest documentation content
    fixtures_path = Path("tests/fixtures/dpytest")

    # Copy main docs file
    docs_file = module_docs_path / f"{TEST_MODULE}_docs.txt"
    docs_file.write_text(fixtures_path.joinpath("dpytest_docs.txt").read_text())

    # Copy llms.txt
    llms_file = module_docs_path / "llms.txt"
    llms_file.write_text(fixtures_path.joinpath("llms.txt").read_text())

    # Copy all summary files
    fixtures_summaries = fixtures_path / "summaries"
    if fixtures_summaries.exists():
        for summary_file in fixtures_summaries.glob("*.txt"):
            dest_file = summaries_path / summary_file.name
            dest_file.write_text(summary_file.read_text())

    # Create dummy vectorstore file (though it won't be loaded by mock)
    vectorstore_file = vectorstore_path / f"{TEST_MODULE}_vectorstore.parquet"
    # vectorstore_file.touch()

    # Patch the paths used in the server code
    monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.BASE_PATH", base_dir)
    monkeypatch.setattr("oh_my_ai_docs.avectorstore_mcp.DOCS_PATH", docs_root)

    return {
        "base": base_dir,
        "docs_root": docs_root,
        "module_docs": module_docs_path,
        "vectorstore": vectorstore_path,
        "summaries": summaries_path,
        "docs_file": docs_file,
        "llms_file": llms_file,
        "vectorstore_file": vectorstore_file,
        "module_name": TEST_MODULE,
    }

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

# Refactored app_context fixture using test_file_structure
@pytest.fixture(scope="function")
def fixture_app_context(
    test_file_structure: dict[str, Path],
    mock_openai_embeddings: FakeEmbeddings,
    mocker: MockerFixture,
) -> Iterator[AppContext]:
    """
    Provides a realistic AppContext for testing, leveraging test_file_structure.

    - Injects FakeEmbeddings globally for the test's scope.
    - Uses TextLoader to load documents with proper metadata.
    - Uses the real vectorstore_factory to create an SKLearnVectorStore instance.
    - Splits and processes documents using RecursiveCharacterTextSplitter.
    - Properly persists the store using parquet serialization.
    - Returns an AppContext containing the initialized store.
    - Resets the global embeddings provider upon teardown.
    """
    # Import necessary functions from the module under test
    from oh_my_ai_docs.avectorstore_mcp import (
        _EMBEDDINGS_PROVIDER,
        set_embeddings_provider,
        vectorstore_factory,
        AppContext,
    )
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    import tiktoken

    # Inject fake embeddings for this test's scope
    original_embeddings = _EMBEDDINGS_PROVIDER
    set_embeddings_provider(mock_openai_embeddings)

    # --- Store Initialization ---
    vectorstore_path = test_file_structure["vectorstore_file"]
    try:


        # --- Document Processing ---
        docs_file_path = test_file_structure["docs_file"]

        if docs_file_path.exists():
            # Use TextLoader to load the document with proper metadata
            loader = TextLoader(str(docs_file_path))
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=8000,
                chunk_overlap=500
            )

            # >>> documents[0].metadata
            # {'source': '/private/var/folders/q_/d5r_s8wd02zdx6qmc5f_96mw0000gp/T/pytest-of-malcolm/pytest-99/test_query_success0/test_repo/docs/ai_docs/dpytest/dpytest_docs.tx
            # t'}

            # Split documents into chunks
            split_docs: list[Document] = text_splitter.split_documents(documents)

            # Create a store using the real factory
            store: SKLearnVectorStore = vectorstore_factory(
                vector_store_cls=SKLearnVectorStore,
                vector_store_kwargs={
                    "persist_path": str(vectorstore_path),
                    "serializer": "parquet",
                },
                embeddings=mock_openai_embeddings
            ).from_documents(documents=split_docs, embedding=mock_openai_embeddings, persist_path=str(vectorstore_path), serializer="parquet")

            # Add split documents to the store
            if split_docs:
                # store.add_documents(split_docs)
                store.from_documents(split_docs, mock_openai_embeddings)

                # Persist the store
                if hasattr(store, 'persist'):
                    store.persist()



        # --- Create AppContext ---
        app_context_instance = AppContext(store=store)  # type: ignore

        yield app_context_instance

    finally:
        # Cleanup
        set_embeddings_provider(original_embeddings)

        # Clean up persisted files if they exist
        if vectorstore_path.exists():
            try:
                vectorstore_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete vectorstore file: {e}")



# def count_tokens(text: str, model: str = "cl100k_base") -> int:
#     """Count tokens in text using tiktoken."""
#     encoder = tiktoken.get_encoding(model)
#     return len(encoder.encode(text))
