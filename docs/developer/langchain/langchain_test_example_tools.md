Here's the second screenshot converted into clean Markdown format:

---

# How to add standard tests to an integration

When creating either a custom class for yourself or to publish in a LangChain integration, it is important to add standard tests to ensure it works as expected. This guide will show you how to add standard tests to each integration type.

---

## Setup

First, let's install 2 dependencies:

- `langchain-core` will define the interfaces we want to import to define our custom tool.
- `langchain-tests` will provide the standard tests we want to use, as well as pytest plugins necessary to run them.

> **Recommended to pin to the latest version:** `==0.1.11`

> **Note**
> Because added tests in new versions of `langchain-tests` can break your CI/CD pipelines, we recommend pinning the version of `langchain-tests` to avoid unexpected changes.

### Poetry

If you followed the [previous guide](https://python.langchain.com/docs/integrations/), you should already have these dependencies installed!

```bash
uv add langchain-core
uv add --group test langchain-tests@latest_version
uv sync --dev
```

---

## Add and configure standard tests

There are 2 namespaces in the `langchain-tests` package:

- `unit_tests` (`langchain_tests.unit_tests`) – designed to test the component in isolation and without access to external services.
- `integration_tests` (`langchain_tests.integration_tests`) – designed to test the component with access to external services.

Both test types are implemented as [pytest](https://docs.pytest.org/) class-based test suites.

Create two test directories:

```bash
tests/unit_tests/         # for unit tests
tests/integration_tests/  # for integration tests
```

---

## Implementing standard tests

Tabs:
- Chat models
- Vector stores
- Embeddings
- **Tools**
- Retrievers

To configure standard tests for a tool, we subclass `ToolUnitTests` and `ToolIntegrationTests`. On each subclass, we override the following `@property` methods to specify the tool to be tested and the tool's configuration:

| Property                    | Description                                                                          |
|----------------------------|--------------------------------------------------------------------------------------|
| `tool_constructor`         | The constructor for the tool to be tested, or an instantiated tool.                 |
| `tool_constructor_params`  | The parameters to pass to the tool (optional).                                      |
| `tool_invoke_params_example` | An example of the parameters to pass to the tool's `invoke` method.                 |

If you're testing a tool class and pass a class like `MyTool` to `tool_constructor`, you can pass the parameters to the constructor in `tool_constructor_params`.

If you're testing an instantiated tool, you can pass the instantiated tool to `tool_constructor` and **do not override** `tool_constructor_params`.

> **Note**
> Details on what tests are run, how each test can be skipped, and troubleshooting tips for each test can be found in the API references:
> - [Unit tests API reference](https://python.langchain.com/docs/testing/unit_tests)
> - [Integration tests API reference](https://python.langchain.com/docs/testing/integration_tests)

---

### Unit test example (`tests/unit_tests/test_tools.py`):

```python
from typing import Type
from langchain_parrot_link.tools import ParrotTool
from langchain_tests.unit_tests import ToolUnitTests

class TestParrotMultiplyToolUnit(ToolUnitTests):
    @property
    def tool_constructor(self) -> Type[ParrotTool]:
        return ParrotTool

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor requires init args like:
        # def __init__(self, some_arg: int):
        # you would return those here as a dictionary, e.g. {"some_arg": 42}
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        # Should represent the args of a valid tool call
        # This must NOT be a ToolCall dict like {"name": ..., "id": ..., "args": ...}
        return {"a": 2, "b": 3}
```

---

### Integration test example (`tests/integration_tests/test_tools.py`):

```python
from typing import Type
from langchain_parrot_link.tools import ParrotTool
from langchain_tests.integration_tests import ToolIntegrationTests

class TestParrotMultiplyToolIntegration(ToolIntegrationTests):
    @property
    def tool_constructor(self) -> Type[ParrotTool]:
        return ParrotTool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"a": 2, "b": 3}
```

---

## Running the tests

Use the following commands from your project root:

```bash
# Run unit tests without network access
uv run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests

# Run integration tests
uv run pytest --asyncio-mode=auto tests/integration_tests
```

---

## Test suite information and troubleshooting

For a full list of the standard test suites that are available, as well as information on which tests are included and how to troubleshoot common issues, see the [Standard Tests API Reference](https://python.langchain.com/docs/testing/reference).

You can also check test-specific documentation under each class in the API Reference.
Example: [`ChatModelIntegrationTests.test_usage_metadata`](https://python.langchain.com/docs/testing/reference/chatmodelintegrationtests.test_usage_metadata)

---

Would you like me to bundle both markdown files into a `.zip` download or convert them into `.md` files?
