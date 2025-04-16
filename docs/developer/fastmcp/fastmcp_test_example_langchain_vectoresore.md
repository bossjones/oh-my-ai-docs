# FastMCP LangChain VectorStore Testing Guide

This guide provides best practices and examples for testing FastMCP implementations that use LangChain's VectorStore functionality with scikit-learn.

## Table of Contents
- [Overview](#overview)
- [Test Setup Best Practices](#test-setup-best-practices)
- [Basic Testing Examples](#basic-testing-examples)
- [Advanced Testing Scenarios](#advanced-testing-scenarios)
- [Performance Testing](#performance-testing)
- [Common Pitfalls](#common-pitfalls)

## Overview

When testing FastMCP implementations that use LangChain's VectorStore with scikit-learn, we need to ensure:
1. Proper vector embedding generation
2. Accurate similarity search functionality
3. Efficient data storage and retrieval
4. Robust error handling
5. Performance under various load conditions

## Test Setup Best Practices

### Fixture Setup in conftest.py

```python
import pytest
from typing import List, Dict, Any, AsyncIterator
from langchain.vectorstores import SKLearnVectorStore
from langchain.schema import Document
from fastmcp.vectorstore import FastMCPVectorStore
from fastmcp.server import FastMCP, Context
from mcp.shared.memory import create_connected_server_and_client_session as client_session
from tests.fake_embeddings import ConsistentFakeEmbeddings, AngularTwoDimensionalEmbeddings

@pytest.fixture
def mock_embeddings():
    """Provides consistent mock embeddings for testing.

    Uses ConsistentFakeEmbeddings to ensure the same text always gets the same embedding vector.
    This is crucial for deterministic testing.
    """
    return ConsistentFakeEmbeddings(dimensionality=128)

@pytest.fixture
def angular_embeddings():
    """Provides 2D embeddings for geometric testing scenarios.

    Useful for testing similarity search with known geometric relationships.
    """
    return AngularTwoDimensionalEmbeddings()

@pytest.fixture
def sample_texts():
    """Sample texts for testing vector store functionality."""
    return [
        "FastMCP is a protocol for LLM interaction",
        "Vector stores help with semantic search",
        "Testing ensures reliability",
        "Embeddings represent text as vectors"
    ]

@pytest.fixture
def vector_store(mock_embeddings, sample_texts):
    """Creates a test vector store with sample data."""
    return SKLearnVectorStore.from_texts(
        texts=sample_texts,
        embedding=mock_embeddings
    )

@pytest.fixture
def fastmcp_vector_store(vector_store):
    """Creates a FastMCP-wrapped vector store for testing."""
    return FastMCPVectorStore(vector_store=vector_store)

@pytest.fixture
def mcp_server(fastmcp_vector_store):
    """Creates a FastMCP server with vector store integration."""
    mcp = FastMCP(instructions="Vector Store Test Server")

    @mcp.resource("vectorstore://data")
    async def vector_data() -> AsyncIterator[str]:
        for text in sample_texts:
            yield text

    return mcp._mcp_server
```

### Test Environment Configuration

```python
@pytest.fixture(autouse=True)
def setup_test_env():
    """Configure test environment with necessary settings."""
    import os
    os.environ["FASTMCP_TEST_MODE"] = "1"
    os.environ["VECTOR_STORE_DIMENSION"] = "128"
    yield
    os.environ.pop("FASTMCP_TEST_MODE", None)
    os.environ.pop("VECTOR_STORE_DIMENSION", None)
```

## Basic Testing Examples

### Test Vector Store Creation

```python
@pytest.mark.anyio
async def test_vector_store_creation(fastmcp_vector_store, mcp_server):
    """Test basic vector store initialization and server integration."""
    async with client_session(mcp_server) as client:
        assert fastmcp_vector_store is not None
        assert hasattr(fastmcp_vector_store, "similarity_search")
        assert hasattr(fastmcp_vector_store, "add_texts")

        # Verify server connection
        assert client.is_connected()

@pytest.mark.anyio
async def test_add_texts(fastmcp_vector_store, mock_embeddings, mcp_server):
    """Test adding new texts to the vector store."""
    async with client_session(mcp_server) as client:
        new_texts = ["New document for testing"]
        result = await fastmcp_vector_store.add_texts(new_texts)

        assert len(result) == 1
        assert isinstance(result[0], str)  # Should return document IDs
```

### Test Similarity Search with Geometric Verification

```python
@pytest.mark.anyio
async def test_geometric_similarity_search(angular_embeddings):
    """Test similarity search with known geometric relationships.

    Uses AngularTwoDimensionalEmbeddings to create vectors with known angles,
    making it possible to verify similarity search results mathematically.
    """
    # Create vectors at 0, 45, and 90 degrees
    texts = ["0", "0.25", "0.5"]  # angles in units of π
    store = SKLearnVectorStore.from_texts(
        texts=texts,
        embedding=angular_embeddings
    )

    # Query at 22.5 degrees (0.125π)
    results = store.similarity_search(
        query="0.125",
        k=2
    )

    # Should return 0° then 45° as closest matches
    assert results[0].page_content == "0"
    assert results[1].page_content == "0.25"
```

### Test Context Integration

```python
@pytest.mark.anyio
async def test_vector_store_with_context(mcp_server, mocker):
    """Test vector store operations with FastMCP context."""
    async with client_session(mcp_server) as client:
        # Mock the session's log message method
        mock_log = mocker.patch("mcp.server.session.ServerSession.send_log_message")

        @mcp_server.tool()
        async def search_docs(query: str, ctx: Context) -> List[str]:
            await ctx.info(f"Searching for: {query}")
            results = await ctx.vector_store.similarity_search(query, k=2)
            return [doc.page_content for doc in results]

        result = await client.call_tool("search_docs", {"query": "protocol"})

        # Verify logging
        mock_log.assert_any_call(
            level="info",
            data="Searching for: protocol",
            logger=None
        )

        # Verify results
        assert len(result.content) > 0
        assert "FastMCP" in result.content[0].text
```

## Advanced Testing Scenarios

### Test Error Handling

```python
@pytest.mark.anyio
async def test_invalid_dimension_handling(mock_embeddings, mcp_server):
    """Test handling of dimension mismatch errors."""
    async with client_session(mcp_server) as client:
        with pytest.raises(ValueError) as exc_info:
            SKLearnVectorStore.from_texts(
                texts=["Test"],
                embedding=mock_embeddings,
                dimension=256  # Intentionally wrong dimension
            )
        assert "dimension mismatch" in str(exc_info.value).lower()

@pytest.mark.anyio
async def test_empty_query_handling(fastmcp_vector_store, mcp_server):
    """Test handling of empty queries."""
    async with client_session(mcp_server) as client:
        with pytest.raises(ValueError) as exc_info:
            await fastmcp_vector_store.similarity_search("")
        assert "empty query" in str(exc_info.value).lower()
```

## Performance Testing

### Test Search Performance

```python
def test_search_performance(fastmcp_vector_store, benchmark):
    """Test similarity search performance."""
    query = "FastMCP protocol"

    result = benchmark(
        fastmcp_vector_store.similarity_search,
        query=query,
        k=5
    )

    assert len(result) == 5
    assert benchmark.stats.stats.mean < 0.1  # Should complete within 100ms

@pytest.mark.parametrize("n_docs", [100, 1000, 10000])
def test_scaling_performance(mock_embeddings, n_docs, benchmark):
    """Test performance with increasing document counts."""
    texts = [f"Document {i}" for i in range(n_docs)]

    def create_and_search():
        store = SKLearnVectorStore.from_texts(
            texts=texts,
            embedding=mock_embeddings
        )
        return store.similarity_search("test", k=5)

    result = benchmark(create_and_search)
    assert len(result) == 5
```

## Common Pitfalls

1. **Dimension Mismatch**
   - Always ensure embedding dimensions match between model and vector store
   - Use consistent embedding models across tests

2. **Memory Management**
   - Clear large vector stores after tests
   - Use appropriate batch sizes for large datasets

3. **Test Data Quality**
   - Use diverse test data that represents real-world scenarios
   - Include edge cases and special characters

4. **Async Operations**
   - Test both sync and async interfaces if available
   - Handle timeouts appropriately

## Best Practices Summary

1. Always use fixtures for common test data and configurations
2. Test both basic and edge cases
3. Include performance benchmarks for critical operations
4. Test error handling thoroughly
5. Use appropriate batch sizes for different test scenarios
6. Clean up resources after tests
7. Document test assumptions and requirements

## Example Test Suite Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_basic.py           # Basic functionality tests
├── test_advanced.py        # Advanced features tests
├── test_performance.py     # Performance benchmarks
├── test_error_handling.py  # Error cases
└── test_integration.py     # Integration tests
```

## Running Tests

```bash
# Run all vector store tests
uv run pytest -s --verbose --showlocals --tb=short tests/test_vectorstore/

# Run specific test with debugging info
uv run pytest -s --verbose --showlocals --tb=short tests/test_vectorstore/test_basic.py::test_vector_store_creation

# Run with coverage
uv run pytest --cov=fastmcp.vectorstore tests/test_vectorstore/

# Run performance tests
uv run pytest tests/test_vectorstore/test_performance.py --benchmark-only
```

## Common Pitfalls to Avoid

1. **Inconsistent Embeddings**
   - Use `ConsistentFakeEmbeddings` for deterministic testing
   - Avoid `FakeEmbeddings` for similarity tests as it doesn't maintain consistency

2. **Resource Management**
   - Always use `client_session` context manager
   - Clean up vector stores after large-scale tests
   - Use appropriate batch sizes for large datasets

3. **Context Integration**
   - Always test with proper FastMCP context
   - Use `mocker.patch` instead of direct mocking of context methods
   - Verify logging through `ServerSession.send_log_message`

4. **Async Operations**
   - Always use `@pytest.mark.anyio` for async tests
   - Properly await all async operations
   - Handle timeouts appropriately

5. **Test Data Quality**
   - Use `AngularTwoDimensionalEmbeddings` for geometric verification
   - Include edge cases and special characters
   - Test with various embedding dimensions
