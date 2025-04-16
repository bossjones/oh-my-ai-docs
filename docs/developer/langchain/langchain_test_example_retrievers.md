Here is the Markdown version of the LangChain guide titled **"How to add standard tests to an integration"**:

---

# How to add standard tests to an integration

When creating either a custom class for yourself or to publish in a LangChain integration, it is important to add standard tests to ensure it works as expected. This guide will show you how to add standard tests to each integration type.

---

## Setup

First, let's install 2 dependencies:

- `langchain-core` will define the interfaces we want to import to define our custom tool.
- `langchain-tests` will provide the standard tests we want to use, as well as pytest plugins necessary to run them.

> Recommended to pin to the latest version: `v0.1.31`

> **Note:**
> Because added tests in new versions of `langchain-tests` can break your CI/CD pipelines, we recommend pinning the version of `langchain-tests` to avoid unexpected changes.

### UV

```bash
uv add langchain-core
uv add --group test langchain-tests@latest_version
uv sync --dev
```

---

## Add and configure standard tests

There are 2 namespaces in the `langchain-tests` package:

- `unit_tests (langchain_tests.unit_tests)`: designed to be used to test the component in isolation and without access to external services.
- `integration_tests (langchain_tests.integration_tests)`: designed to be used to test the component with access to external services (in particular, the external service that the component is designed to interact with).

Both types of tests are implemented as [pytest class-based test suites](https://docs.pytest.org/en/latest/how-to/writing_plugins.html#creating-plugins).

By subclassing the base classes for each type of standard test (see below), you get all of the standard tests for that type, and you can override the properties that the test suite uses to configure the tests.

We recommend subclassing these classes in test files under two test subdirectories:

- `tests/unit_tests` for unit tests
- `tests/integration_tests` for integration tests

---

## Implementing standard tests

In the following tabs, we show how to implement the standard tests for each component type:

- Chat models
- Vector stores
- Embeddings
- Tools
- **Retrievers** ← *example shown*

To configure standard tests for a retriever, we subclass `RetrieverUnitTests` and `RetrieverIntegrationTests`. On each subclass, we override the following `@property` methods:

| Property                    | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `retriever_constructor`    | The class for the retriever to be tested                     |
| `retriever_constructor_params` | The parameters to pass to the retriever's constructor     |
| `retriever_query_example`  | An example of the query to pass to the retriever's `invoke` method |

```python
# tests/integration_tests/test_retrievers.py

from typing import Type
from langchain_parrot_link.retrievers import ParrotRetriever
from langchain_tests.integration_tests import (
    RetrieverIntegrationTests,
)

class TestParrotRetriever(RetrieverIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[ParrotRetriever]:
        """Return an empty vectorstore for unit tests."""
        return ParrotRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """Returns a str representing the `query` of an example retriever call."""
        return "example query"
```

---

## Running the tests

You can run these with the following commands from your project root:

### UV

```bash
# run unit tests without network access
uv run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests

# run integration tests
uv run pytest --asyncio-mode=auto tests/integration_tests
```

---

## Test suite information and troubleshooting

For a full list of the standard test suites that are available, as well as information on which tests are included and how to troubleshoot common issues, see the [Standard Tests API Reference](https://api.python.langchain.com/en/latest/tests/langchain_tests.unit_tests.html).

You can see troubleshooting guides under the individual test suites listed in that API Reference. For example, here is the guide for `ChatModelIntegrationTests.test_usage_metadata`.

---

Let me know if you want this exported as a `.md` file or included in a documentation system!
