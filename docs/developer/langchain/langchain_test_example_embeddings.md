Here's the content from the screenshot converted into Markdown format:

---

# How to add standard tests to an integration

When creating either a custom class for yourself or to publish in a LangChain integration, it is important to add standard tests to ensure it works as expected. This guide will show you how to add standard tests to each integration type.

---

## Setup

First, let's install 2 dependencies:

- `langchain-core` will define the interfaces we want to import to define our custom tool.
- `langchain-tests` will provide the standard tests we want to use, as well as pytest plugins necessary to run them.

> **Recommended** to pin to the latest version: `==0.1.11`

> **Note**
> Because added tests in new versions of `langchain-tests` can break your CI/CD pipelines, we recommend pinning the version of `langchain-tests` to avoid unexpected changes.

### UV

If you followed the [previous guide](https://python.langchain.com/docs/integrations/), you should already have these dependencies installed!

```bash
uv add langchain-core
uv add --group test langchain-tests@latest_version
uv sync --dev
```

---

## Add and configure standard tests

There are 2 namespaces in the `langchain-tests` package:

- **Unit tests** (`langchain_tests.unit_tests`): designed to be used to test the component in isolation and without access to external services.
- **Integration tests** (`langchain_tests.integration_tests`): designed to be used to test the component with access to external services (i.e. services the component is designed to interact with).

Both types of tests are implemented as [pytest](https://docs.pytest.org/) class-based test suites.

By subclassing the base classes for each type of standard test (see below), you get all of the standard tests for that type, and you can override the properties that the test suite uses to configure the tests.

To run the tests in the same way as in this guide, we recommend subclassing these classes in test files under two test subdirectories:

```bash
tests/unit_tests/      # for unit tests
tests/integration_tests/  # for integration tests
```

---

## Implementing standard tests

In the following tabs, we show how to implement the standard tests for each component type:

- Chat models
- Vector stores
- **Embeddings**
- Tools
- Retrievers

To configure standard tests for an embeddings model, we subclass `EmbeddingUnitTests` and `EmbeddingIntegrationTests`. On each subclass, we override the following `@property` methods to specify the embeddings model to be tested and the embeddings model's configuration:

| Property                | Description                                              |
|------------------------|----------------------------------------------------------|
| `embedding_cls`        | The class for the embeddings model to be tested          |
| `embedding_model_params` | The parameters to pass to the embeddings model's constructor |

> **Note**
> Details on what tests are run, how each test can be skipped, and troubleshooting tips for each test can be found in the API references. See details:
> - [Unit tests API Reference](https://python.langchain.com/docs/testing/unit_tests)
> - [Integration tests API Reference](https://python.langchain.com/docs/testing/integration_tests)

---

### Unit test example (`tests/unit_tests/test_embeddings.py`):

```python
"""Test embedding model integration."""

from typing import Type
from langchain_parrot_link.embeddings import ParrotLinkEmbeddings
from langchain_tests.unit_tests import EmbeddingUnitTests

class TestParrotLinkEmbeddingUnit(EmbeddingUnitTests):
    @property
    def embedding_cls(self) -> Type[ParrotLinkEmbeddings]:
        return ParrotLinkEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "test-embed-001"}
```

---

### Integration test example (`tests/integration_tests/test_embeddings.py`):

```python
from typing import Type
from langchain_parrot_link.embeddings import ParrotLinkEmbeddings
from langchain_tests.integration_tests import EmbeddingIntegrationTests

class TestParrotLinkEmbeddingIntegration(EmbeddingIntegrationTests):
    @property
    def embedding_cls(self) -> Type[ParrotLinkEmbeddings]:
        return ParrotLinkEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "test-embed-001"}
```

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

For a full list of the standard test suites that are available, as well as information on which tests are included and how to troubleshoot common issues, see the [Standard Tests API Reference](https://python.langchain.com/docs/testing/reference).

You can see troubleshooting guides under the individual test suites listed in that API Reference. For example, here is the guide for `ChatModelIntegrationTests.test_usage_metadata`.

---

Was this page helpful? 👍 👎

[Write a comment]

---

**Navigation:**
← [How to implement an integration package](https://python.langchain.com/docs/integrations/)
→ [Publishing your package](https://python.langchain.com/docs/integrations/publish/)

---

Let me know if you want this in a downloadable `.md` file too.
