## v0.3.0 (2025-04-17)

### Feat

- add release preparation and GitHub release scripts
- add GitHub label creation for official releases in release preparation script

### Refactor

- enhance query_tool and get_all_docs function signatures with detailed Field descriptions
- update MCP server configuration and entry points
- enhance documentation and type hints in build_llmstxt_context.py
- comment out unused test cases in test_aserver.py
- enhance vectorstore and query_tool return types, improve error handling
- update test assertions and enhance query validation in test_aserver.py
- update QueryConfig default value and enhance test fixture
- remove unused logging fixture and comment out test case

## v0.2.0 (2025-04-16)

### Feat

- expand LangChain testing markers for integration and unit tests
- introduce LangChain testing standards for embeddings, retrievers, and tools
- add LangChain chat model integration and unit testing standards
- update dependencies and enhance testing documentation
- update pre-commit configuration and enhance ruff linting settings
- add future annotations support across multiple scripts
- update mvp-testing-agent.mdc with pytest fixture guidelines
- enhance vectorstore testing setup and introduce new test file structure
- enhance pytest fixture guidelines and refactor vectorstore testing setup
- enhance FastMCP testing documentation with new lifespan and vectorstore guidelines
- update mvp-testing-agent.mdc with enhanced debugging guidelines and isolated test command format
- add comprehensive pre-implementation and code reference guidelines to mvp-testing-agent.mdc
- add detailed guidelines for debugging test failures in mvp-testing-agent.mdc
- enhance logging and improve error messages in avectorstore_mcp.py
- update mocking guidelines and add new test cases for FastMCP
- enhance FastMCP testing documentation and add new test cases
- implement comprehensive context testing for FastMCP
- expand testing rules and documentation in SCRATCH.md
- implement comprehensive MVP testing rules and documentation updates
- enhance testing and VSCode configurations for improved development workflow
- update VSCode configurations for improved testing and debugging
- enhance testing setup and update VSCode configurations
- add FastMCP stdio server rules and enhance logging in avectorstore_mcp.py
- add discord avectorstore configuration to mcp-sample.json

### Refactor

- update vectorstore factory and improve test fixtures
- remove unused dependencies and enhance timeout settings
- reorganize test fixtures and update client session usage
- clean up VSCode launch configurations by removing unused entries
