Here is the Markdown version of the guide titled **"How to add standard tests to an integration"** from LangChain:

---

# How to add standard tests to an integration

When creating either a custom class for yourself or to publish in a LangChain integration, it is important to add standard tests to ensure it works as expected. This guide will show you how to add standard tests to each integration type.

---

## Setup

First, let's install 2 dependencies:

- `langchain-core` will define the interfaces we want to import to define our custom tool.
- `langchain-tests` will provide the standard tests we want to use, as well as pytest plugins necessary to run them.

Recommended to pin to the latest version: ![latest](https://img.shields.io/pypi/v/langchain-tests)

> **Note**
> Because added tests in new versions of `langchain-tests` can break your CI/CD pipelines, we recommend pinning the version of `langchain-tests` to avoid unexpected changes.

### Poetry

If you followed the [previous guide](https://python.langchain.com/v0.1/docs/integrations/how_to_guides/integration_package/), you should already have these dependencies installed!

```bash
uv add langchain-core
uv add --group test langchain-tests@<latest_version>
uv sync --dev
```

---

## Add and configure standard tests

There are 2 namespaces in the `langchain-tests` package:

- **unit tests** (`langchain_tests.unit_tests`): Designed to be used to test the component in isolation and without access to external services.
- **integration tests** (`langchain_tests.integration_tests`): Designed to be used to test the component with access to external services (in particular, the external service that the component is designed to interact with).

Both types of tests are implemented as `pytest` class-based test suites.

By subclassing the base classes for each type of standard test (see below), you get all of the standard tests for that type, and you can override the properties that the test suite uses to configure the tests.

In order to run the tests in the same way as this guide, we recommend subclassing these classes in test files under two test subdirectories:

- `tests/unit_tests/` for unit tests
- `tests/integration_tests/` for integration tests

---

## Implementing standard tests

In the following tabs, we show how to implement the standard tests for each component type:

**Chat models | Vector stores | Embeddings | Tools | Retrievers**

To configure standard tests for a chat model, we subclass `ChatModelUnitTests` and `ChatModelIntegrationTests`. On each subclass, we override the following `@property` methods to specify the chat model to be tested and the chat model's constructor:

| Property            | Description                                           |
|---------------------|-------------------------------------------------------|
| `chat_model_class`  | The class for the chat model to be tested            |
| `chat_model_kwargs` | The parameters to pass to the chat model's constructor |

Additionally, chat model standard tests test a range of behaviors, from the most basic requirements (generating a response to a query) to optional capabilities like multi-modal support and tool-calling. For a test run to be successful:

1. If a feature is intended to be supported by the model, it should pass.
2. If a feature is not intended to be supported by the model, it should be skipped.

Tests for "optional" capabilities are controlled via a set of properties that can be overridden on the test model subclass.

You can see the **entire list of configurable capabilities** in the API references for [unit tests](https://api.python.langchain.com/en/latest/tests/langchain_tests.unit_tests.chat_models.ChatModelUnitTests.html) and [integration tests](https://api.python.langchain.com/en/latest/tests/langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.html).

For example, to enable integration tests for image inputs, we can implement:

```python
@property
def supports_image_inputs(self) -> bool:
    return True
```

on the integration test class.

> **Note**
> Details on what tests are run, how each test can be skipped, and troubleshooting tips for each test can be found in the API references. See details:
> - Unit tests API reference
> - Integration tests API reference

---

### Unit test example:

`tests/unit_tests/test_chat_models.py`

```python
"""Test chat model integration."""

from typing import Type

from langchain_parrot_link.chat_models import ChatParrotLink
from langchain_tests.unit_tests import ChatModelUnitTests

class TestChatParrotLinkUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatParrotLink]:
        return ChatParrotLink

    @property
    def chat_model_kwargs(self) -> dict:
        """These should be parameters used to initialize your integration for testing"""
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "prompt_buffer_length": 50,
        }
```

---

### Integration test example:

`tests/integration_tests/test_chat_models.py`

```python
"""Test ChatParrotLink chat model."""

from typing import Type

from langchain_parrot_link.chat_models import ChatParrotLink
from langchain_tests.integration_tests import ChatModelIntegrationTests

class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatParrotLink]:
        return ChatParrotLink

    @property
    def chat_model_kwargs(self) -> dict:
        """These should be parameters used to initialize your integration for testing"""
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "prompt_buffer_length": 50,
        }
```

---

## Running the tests

You can run these with the following commands from your project root:

### Poetry

```bash
# run unit tests without network access
uv run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests

# run integration tests
uv run pytest --asyncio-mode=auto tests/integration_tests
```

---

## Test suite information and troubleshooting

For a full list of the standard test suites that are available, as well as information on which tests are included and how to troubleshoot common issues, see the [Standard Tests API Reference](https://api.python.langchain.com/en/latest/tests/index.html).

You can see troubleshooting guides under the individual test suites listed in that API Reference. For example, here is the guide for:

[`ChatModelIntegrationTests.test_usage_metadata`](https://api.python.langchain.com/en/latest/tests/langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.html#langchain_tests.integration_tests.chat_models.ChatModelIntegrationTests.test_usage_metadata)

---

Let me know if you'd like this exported to a `.md` file or rendered with specific formatting or section emphasis.
