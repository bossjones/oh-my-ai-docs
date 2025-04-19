# oh-my-ai-docs

A repository for managing and utilizing documentation through llms.txt files and sklearn vectorstores, integrated with Model Context Protocol (MCP). This project helps create efficient documentation search and retrieval systems for various Python modules.

## Overview

This repository provides tools and workflows for:
1. Generating and maintaining llms.txt files for different Python modules
2. Building sklearn vectorstores from llms.txt files
3. Serving these vectorstores through MCP servers for AI-powered documentation search

## Prerequisites

- Python 3.12+
- UV package manager
- Node.js (for MCP inspector)
- Just command runner
- direnv (recommended for environment management)
- Docker (optional, for containerized deployment)

### Required API Keys

The following API keys are required for full functionality:

```bash
# Required for embeddings
export OPENAI_API_KEY=your_api_key

# Optional but recommended
export ANTHROPIC_API_KEY=your_api_key  # For advanced LLM features
export LANGCHAIN_API_KEY=your_api_key  # For LangChain tracing
export LANGCHAIN_PROJECT=your_project  # For LangChain project organization
```

### Optional Features

The project supports several optional features that can be enabled via environment variables:

```bash
# Development and debugging
export LANGCHAIN_DEBUG_LOGS=1        # Enable LangChain debug logging
export LOCAL_TEST_DEBUG=1            # Enable local test debugging
export LOCAL_TEST_ENABLE_EVALS=1     # Enable evaluation features
export BETTER_EXCEPTIONS=1           # Improved exception formatting

# Tracing and monitoring
export LANGCHAIN_TRACING_V2=true     # Enable LangChain tracing
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

## Supported Modules

Currently supports documentation management for:
- discord.py
- LangGraph
- LangChain
- dpytest

## Building and Updating llms.txt Files

The repository uses [llmstxt-architect](https://github.com/rlancemartin/llmstxt_architect) to generate and maintain llms.txt files. These files contain curated URLs and descriptions for module documentation.

### Generate New llms.txt Files

```bash
# For discord.py documentation
just llmstxt-discord

# For LangGraph documentation
just llmstxt-langgraph

# For LangChain documentation
just llmstxt-langchain

# For dpytest documentation
just llmstxt-dpytest
```

### Update Existing llms.txt Files

```bash
# Update discord.py documentation
just llmstxt-discord-update

# Update LangGraph documentation
just llmstxt-langgraph-update

# Update LangChain documentation
just llmstxt-langchain-update

# Update dpytest documentation
just llmstxt-dpytest-update
```

## Building sklearn Vectorstores

After generating or updating llms.txt files, you can build sklearn vectorstores for efficient documentation search:

```bash
# Build vectorstore for discord.py
just avectorstore-build-context-discord

# Build vectorstore for LangGraph
just avectorstore-build-context-langgraph

# Build vectorstore for LangChain
just avectorstore-build-context-langchain

# Build vectorstore for dpytest
just avectorstore-build-context-dpytest

# Build all vectorstores
just avectorstore-build-context-all
```

## Using Vectorstores with MCP

The repository provides MCP servers to query the vectorstores:

### Environment Setup

Before running the MCP servers, ensure you have the following environment variables set:

```bash
export OPENAI_API_KEY=your_api_key  # Required for text-embedding-3-large model
```

### Running MCP Servers

```bash
# Run discord.py vectorstore server
just avectorstore-discord

# Debug with MCP inspector
just avectorstore-discord-inspector

# Additional options:
# --dry-run: Show configuration without starting the server
# --list-vectorstores: List available vector stores
# --generate-mcp-config: Generate mcp.json configuration
# --save: Save the generated configuration
# --debug: Enable verbose logging
```

### MCP Server Configuration

The vectorstore servers can be configured in your MCP client configuration:

```json
{
  "servers": {
    "discord-docs": {
      "command": ["just", "avectorstore-discord"]
    }
  }
}
```

### Available Endpoints

The MCP server provides two main endpoints:

1. Tool Endpoint:
   - Name: `query_docs`
   - Description: Search through module documentation using semantic search
   - Parameters:
     - `query`: The search query string
     - `k`: Number of documents to retrieve (1-10, default: 2)
     - `min_relevance_score`: Minimum relevance score threshold (0.0-1.0)

2. Resource Endpoint:
   - URI: `docs://{module}/full`
   - Description: Retrieve full documentation content for a module
   - Supported modules: discord, dpytest, langgraph
   - Returns raw text content in plain text format

### Query Examples

```python
# Using the query_docs tool
result = await client.call_tool("query_docs", {
    "query": "How to test discord bots with dpytest?",
    "k": 3,
    "min_relevance_score": 0.5
})

# Accessing full documentation
docs = await client.get_resource("docs://discord/full")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Insert your license here]

## Development Setup

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/bossjones/oh-my-ai-docs.git
   cd oh-my-ai-docs
   ```

2. Set up the development environment:
   ```bash
   # Install UV package manager if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --all-groups --dev

   # Install pre-commit hooks
   pre-commit install
   ```

3. Configure environment variables:
   ```bash
   cp sample.env .env
   # Edit .env with your settings
   ```

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Ruff**: Fast Python linter
  ```bash
  uv run ruff check .
  ```

- **Pyright**: Static type checker
  ```bash
  uv run pyright
  ```

- **Pre-commit**: Git hooks for code quality
  ```bash
  uv run pre-commit run --all-files
  ```

### Testing

The project has comprehensive test coverage using pytest:

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=oh_my_ai_docs

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/
```

## Docker Support

The project includes Docker support with multi-stage builds for optimized image size:

### Building the Docker Image

```bash
docker build -t oh-my-ai-docs .
```

### Running with Docker

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key \
  oh-my-ai-docs
```

### Docker Development

For development with Docker:

```bash
# Build with development dependencies
docker build --target builder -t oh-my-ai-docs-dev .

# Run tests in container
docker run oh-my-ai-docs-dev uv run pytest
```

## Documentation

The project uses MkDocs for documentation:

### Building Documentation

```bash
# Install documentation dependencies
uv sync --dev

# Build documentation
uv run mkdocs build

# Serve documentation locally
uv run mkdocs serve
```

### Documentation Structure

- `docs/`: Main documentation directory
- `docs_templates/`: Documentation templates
- `mkdocs.yml`: MkDocs configuration

## Command Line Tools

The project provides two CLI tools:

1. `avectorstore_mcp`: MCP server for vectorstore operations
   ```bash
   uv run avectorstore_mcp --help
   ```

2. `goobctl`: Project management CLI
   ```bash
   uv run goobctl --help
   ```

## Environment Setup

The project uses direnv for environment management. To set up your environment:

1. Install direnv:
   ```bash
   # macOS
   brew install direnv

   # Linux
   curl -sfL https://direnv.net/install.sh | bash
   ```

2. Add direnv hook to your shell:
   ```bash
   # Add to your ~/.bashrc or ~/.zshrc
   eval "$(direnv hook bash)"  # or zsh
   ```

3. Configure your environment:
   ```bash
   cp sample.env .env
   cp sample.envrc .envrc
   # Edit .env and .envrc with your settings
   direnv allow
   ```
