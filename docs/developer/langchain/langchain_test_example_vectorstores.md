Here is the Markdown version of the second LangChain guide image, focusing on **Vector store standard tests**:

---

# How to add standard tests to an integration

When creating either a custom class for yourself or to publish in a LangChain integration, it is important to add standard tests to ensure it works as expected. This guide will show you how to add standard tests to each integration type.

---

## Setup

First, let's install 2 dependencies:

- `langchain-core`: will define the interfaces we want to import to define our custom tool.
- `langchain-tests`: will provide the standard tests we want to use, as well as pytest plugins necessary to run them.

> **Recommended**: pin to the latest version
> ![PyPI badge](https://img.shields.io/pypi/v/langchain-tests)

> **Note**
> Because added tests in new versions of `langchain-tests` can break your CI/CD pipelines, we recommend pinning the version of `langchain-tests` to avoid unexpected changes.

### Poetry

```bash
uv add langchain-core
uv add --group test langchain-tests@<latest_version>
uv sync --dev
```

---

## Add and configure standard tests

There are 2 namespaces in the `langchain-tests` package:

- **unit tests** (`langchain_tests.unit_tests`): Test the component in isolation and without access to external services.
- **integration tests** (`langchain_tests.integration_tests`): Test the component with access to external services (like an actual vector store or API).

Both types are implemented as `pytest` class-based test suites.

We recommend placing test files in:

- `tests/unit_tests/` for unit tests
- `tests/integration_tests/` for integration tests

---

## Implementing standard tests

In the following tabs, we show how to implement the standard tests for each component type:

**Chat models | Vector stores | Embeddings | Tools | Retrievers**

---

### Vector store tests

Here's how you would configure the standard tests for a typical vector store using `ParrotVectorStore` as a placeholder:

#### `tests/integration_tests/test_vectorstores.py`

```python
from typing import Generator

import pytest
from langchain_parrot_link.vectorstores import ParrotVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

class TestParrotStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Set up an empty vectorstore for unit tests."""
        store = ParrotVectorStore(embedding=self._get_embeddings())
        # Note: store should be EMPTY at this point
        try:
            yield store
        finally:
            # Cleanup operations, or deleting data
            pass
```

---

### Fixture Configuration

| Fixture       | Description |
|---------------|-------------|
| `vectorstore` | A generator that yields an empty vector store for unit tests. The vector store is cleaned up after the test run ends. |

---

### Example: Chroma integration

```python
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_community.vectorstores import Chroma

class TestChromaStandard(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Set up an empty vectorstore for unit tests."""
        store = Chroma(embedding_function=self._get_embeddings())
        try:
            yield store
        finally:
            store.delete_collection()
```

---

### API Reference

- [`VectorStore`](https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.VectorStore.html)
- [`VectorStoreIntegrationTests`](https://api.python.langchain.com/en/latest/tests/langchain_tests.integration_tests.vectorstores.VectorStoreIntegrationTests.html)

> **Note**
> Before the initial `yield`, the vector store is instantiated with an embeddings object. This is a pre-defined "fake" embedding model. You can use a different embeddings object if desired.
> In the `finally` block, clean-up logic is executed, even if a test fails.

---

## Running the tests

You can run these with the following commands from your project root:

```bash
# run unit tests without network access
uv run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests

# run integration tests
uv run pytest --asyncio-mode=auto tests/integration_tests
```

---

## Test suite information and troubleshooting

For a full list of the standard test suites that are available, and how to troubleshoot common issues, see the [Standard Tests API Reference](https://api.python.langchain.com/en/latest/tests/index.html).

Example:
- [`ChatModelIntegrationTests.test_usage_metadata`](https://api.python.langchain.com/en/latest/tests/langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.html#langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_usage_metadata)

---

Let me know if you'd like this exported to a `.md` file or added to a README!
