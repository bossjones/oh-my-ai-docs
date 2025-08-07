# AUGMENTCODE.md

## Project Overview

**oh-my-ai-docs** is a comprehensive documentation management and AI-powered search system that leverages LLMs.txt files and sklearn vectorstores integrated with the Model Context Protocol (MCP). This project enables efficient documentation search and retrieval for various Python modules through semantic search capabilities.

### Purpose

The repository serves as a centralized hub for:
- **Documentation Curation**: Generating and maintaining structured llms.txt files for popular Python libraries
- **Semantic Search**: Building sklearn vectorstores for fast, AI-powered documentation search
- **MCP Integration**: Serving documentation through standardized Model Context Protocol servers
- **Developer Productivity**: Providing instant access to relevant documentation through AI assistants

### Key Features

- 🔍 **Semantic Documentation Search**: Vector-based similarity search across multiple Python library docs
- 📚 **Multi-Module Support**: Currently supports discord.py, LangGraph, LangChain, and dpytest
- 🚀 **MCP Server Integration**: Standards-compliant servers for AI assistant integration
- 🔄 **Automated Updates**: Tools for keeping documentation current and synchronized
- 🛠️ **Developer Tools**: CLI utilities for managing vectorstores and documentation
- 📊 **Rich Monitoring**: Comprehensive logging, metrics, and health checks

## Architecture

### Core Components

1. **llms.txt Generation**: Uses `llmstxt-architect` to create structured documentation files
2. **Vector Store Engine**: sklearn-based vectorstores with OpenAI embeddings for semantic search
3. **MCP Server**: FastMCP-based server providing standardized AI assistant integration
4. **CLI Tools**: Command-line utilities for management and operations

### Supported Documentation Modules

| Module | Description | Documentation Source |
|--------|-------------|---------------------|
| **discord.py** | Discord bot development library | https://discordpy.readthedocs.io/ |
| **LangGraph** | Graph-based LLM application framework | https://langchain-ai.github.io/langgraph/ |
| **LangChain** | LLM application development framework | https://python.langchain.com/ |
| **dpytest** | Discord.py testing framework | https://dpytest.readthedocs.io/ |

## Installation & Setup

### Prerequisites

- **Python 3.12+** (Required)
- **UV Package Manager** (Recommended)
- **Node.js** (For MCP inspector)
- **Just Command Runner** (For task automation)
- **direnv** (Optional, for environment management)
- **Docker** (Optional, for containerized deployment)

### Quick Start

1. **Clone and Setup**:
   ```bash
   git clone https://github.com/bossjones/oh-my-ai-docs.git
   cd oh-my-ai-docs

   # Install UV if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --all-groups --dev
   ```

2. **Environment Configuration**:
   ```bash
   cp sample.env .env
   # Edit .env with your API keys and settings
   ```

3. **Required API Keys**:
   ```bash
   # Essential for embeddings and vectorstore operations
   export OPENAI_API_KEY=your_openai_api_key

   # Optional but recommended for enhanced features
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   export LANGCHAIN_API_KEY=your_langchain_api_key
   export LANGCHAIN_PROJECT=your_project_name
   ```

### Development Setup

```bash
# Install pre-commit hooks for code quality
pre-commit install

# Verify installation
just --list  # Shows all available commands
uv run pytest  # Run test suite
```

## Usage Guide

### 1. Documentation Generation

#### Generate New llms.txt Files

```bash
# Generate documentation for specific modules
just llmstxt-discord      # Discord.py documentation
just llmstxt-langgraph    # LangGraph documentation
just llmstxt-langchain    # LangChain documentation
just llmstxt-dpytest      # dpytest documentation
```

#### Update Existing Documentation

```bash
# Update existing llms.txt files with latest content
just llmstxt-discord-update
just llmstxt-langgraph-update
just llmstxt-langchain-update
just llmstxt-dpytest-update
```

### 2. Vector Store Operations

#### Build Vector Stores

```bash
# Build vectorstores for semantic search
just avectorstore-build-context-discord
just avectorstore-build-context-langgraph
just avectorstore-build-context-langchain
just avectorstore-build-context-dpytest

# Build all vectorstores at once
just avectorstore-build-context-all
```

### 3. MCP Server Operations

#### Running MCP Servers

```bash
# Start MCP server for discord.py documentation
just avectorstore-discord

# Debug with MCP inspector (opens web interface)
just avectorstore-discord-inspector
```

#### MCP Server Features

The MCP server provides two main endpoints:

1. **Tool Endpoint** (`query_docs`):
   - **Purpose**: Semantic search through documentation
   - **Parameters**:
     - `query`: Search query string
     - `k`: Number of results (1-10, default: 2)
     - `min_relevance_score`: Minimum relevance threshold (0.0-1.0)

2. **Resource Endpoint** (`docs://{module}/full`):
   - **Purpose**: Retrieve complete documentation content
   - **Supported modules**: discord, dpytest, langgraph
   - **Format**: Plain text

#### Example MCP Usage

```python
# Using the query_docs tool
result = await client.call_tool("query_docs", {
    "query": "How to create a Discord bot command?",
    "k": 3,
    "min_relevance_score": 0.7
})

# Accessing full documentation
docs = await client.get_resource("docs://discord/full")
```

### 4. CLI Tools

The project provides two main CLI utilities:

#### avectorstore_mcp
```bash
# MCP server for vectorstore operations
uv run avectorstore_mcp --help

# Common options:
# --module: Specify documentation module (discord, langgraph, etc.)
# --stdio: Use stdio transport
# --debug: Enable verbose logging
# --dry-run: Show configuration without starting server
# --list-vectorstores: List available vector stores
```

#### goobctl
```bash
# Project management CLI
uv run goobctl --help
```

## Configuration

### Environment Variables

#### Essential Configuration
```bash
# Core API Keys
OPENAI_API_KEY="your_openai_api_key"          # Required for embeddings
ANTHROPIC_API_KEY="your_anthropic_api_key"    # Optional, for enhanced LLM features

# LangChain Integration
LANGCHAIN_API_KEY="your_langchain_api_key"    # Optional, for tracing
LANGCHAIN_PROJECT="your_project_name"         # Project organization
LANGCHAIN_TRACING_V2="true"                   # Enable tracing
```

#### Development & Debugging
```bash
# Debugging Options
DEBUG="true"                                  # Enable debug mode
LANGCHAIN_DEBUG_LOGS="1"                     # LangChain debug logging
LOCAL_TEST_DEBUG="1"                         # Local test debugging
BETTER_EXCEPTIONS="1"                        # Enhanced exception formatting

# Monitoring
LOG_LEVEL="DEBUG"                            # Logging verbosity
ENABLE_METRICS="true"                        # Prometheus metrics
METRICS_PORT="9090"                          # Metrics endpoint port
```

#### Storage & Performance
```bash
# Storage Configuration
STORAGE_ROOT="/tmp/oh-my-ai-docs"            # Root storage directory
MAX_FILE_SIZE_MB="50"                        # Maximum file size limit
MAX_CONCURRENT_DOWNLOADS="5"                 # Download concurrency
MAX_QUEUE_SIZE="50"                          # Queue size limit

# Performance Tuning
RATE_LIMIT_REQUESTS="100"                    # Requests per window
RATE_LIMIT_WINDOW_SECONDS="60"               # Rate limit window
```

### MCP Client Configuration

Add to your MCP client configuration:

```json
{
  "servers": {
    "discord-docs": {
      "command": ["just", "avectorstore-discord"],
      "description": "Discord.py documentation search"
    },
    "langgraph-docs": {
      "command": ["uv", "run", "./src/oh_my_ai_docs/avectorstore_mcp.py", "--module", "langgraph", "--stdio"],
      "description": "LangGraph documentation search"
    }
  }
}
```

## Development

### Code Quality & Testing

#### Linting and Formatting
```bash
# Run code quality checks
just check                    # Run all checks
uv run ruff check .          # Linting
uv run pyright              # Type checking
uv run pre-commit run --all-files  # Pre-commit hooks
```

#### Testing
```bash
# Run test suite
uv run pytest                           # All tests
uv run pytest --cov=oh_my_ai_docs      # With coverage
uv run pytest tests/unit/              # Unit tests only
uv run pytest tests/integration/       # Integration tests only

# Test with specific markers
uv run pytest -m "vectorstore"         # Vectorstore tests
uv run pytest -m "fastmcp_basic"       # FastMCP basic tests
```

#### Performance Testing
```bash
# Memory profiling
uv run pytest --memray

# Performance benchmarks
uv run pytest -m "slow"
```

### Project Structure

```
oh-my-ai-docs/
├── src/oh_my_ai_docs/           # Main source code
│   ├── avectorstore_mcp.py      # MCP server implementation
│   ├── cli.py                   # CLI utilities
│   └── ...
├── docs/                        # Documentation
│   ├── ai_docs/                 # Generated llms.txt files
│   │   ├── discord/             # Discord.py docs
│   │   ├── langgraph/           # LangGraph docs
│   │   ├── langchain/           # LangChain docs
│   │   └── dpytest/             # dpytest docs
│   └── ...
├── scripts/                     # Utility scripts
├── tests/                       # Test suite
├── justfiles/                   # Just command definitions
├── pyproject.toml              # Project configuration
└── README.md                   # Basic project info
```

### Adding New Documentation Modules

1. **Create llmstxt command** in `justfiles/llmstxt.just`:
   ```bash
   [group('llmstxt')]
   llmstxt-newmodule:
       uv run llmstxt-architect \
       --urls https://newmodule.readthedocs.io/ \
       --max-depth 3 \
       --llm-name claude-3-7-sonnet-latest \
       --llm-provider anthropic \
       --project-dir docs/ai_docs/newmodule
   ```

2. **Add vectorstore build command** in `justfiles/vectorstore.just`:
   ```bash
   [group('vectorstore')]
   avectorstore-build-context-newmodule:
       uv run ./scripts/build_llmstxt_context.py --module newmodule
   ```

3. **Update MCP server** to support the new module in `avectorstore_mcp.py`

### Contributing Guidelines

1. **Code Style**: Follow PEP 8, use Ruff for linting
2. **Type Hints**: All functions must have proper type annotations
3. **Testing**: Maintain >80% test coverage
4. **Documentation**: Update docs for any new features
5. **Commits**: Use conventional commit format

```bash
# Example contribution workflow
git checkout -b feature/new-module
# Make changes
just check                    # Verify code quality
uv run pytest               # Run tests
git commit -m "feat: add support for new documentation module"
git push origin feature/new-module
# Create pull request
```

## Docker Support

### Building and Running

```bash
# Build Docker image
docker build -t oh-my-ai-docs .

# Run with environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key \
  -e LANGCHAIN_API_KEY=your_langchain_key \
  oh-my-ai-docs

# Development with volume mounting
docker run -v $(pwd):/app \
  -e OPENAI_API_KEY=your_api_key \
  oh-my-ai-docs
```

### Multi-stage Build

The Dockerfile uses multi-stage builds for optimization:
- **Builder stage**: Installs dependencies and builds the application
- **Runtime stage**: Minimal image with only necessary components

## Monitoring & Observability

### Logging

The project uses structured logging with multiple levels:

```python
# Configure logging level
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR

# Enable specific debug features
LANGCHAIN_DEBUG_LOGS=1      # LangChain operations
LOCAL_TEST_DEBUG=1          # Test debugging
```

### Metrics

Prometheus metrics are available when enabled:

```bash
# Enable metrics
ENABLE_METRICS=true
METRICS_PORT=9090

# Access metrics
curl http://localhost:9090/metrics
```

### Health Checks

Built-in health check endpoints:

```bash
# Enable health checks
ENABLE_HEALTH_CHECK=true
HEALTH_CHECK_PORT=8080

# Check health
curl http://localhost:8080/health
```

## Troubleshooting

### Common Issues

#### 1. Missing API Keys
```bash
# Error: OpenAI API key not found
export OPENAI_API_KEY=your_actual_api_key
```

#### 2. Vector Store Not Found
```bash
# Build the vectorstore first
just avectorstore-build-context-discord
```

#### 3. MCP Server Connection Issues
```bash
# Check server logs
just avectorstore-discord-inspector
# Verify environment variables are set
```

#### 4. Permission Errors
```bash
# Ensure proper permissions for storage directory
chmod 755 /tmp/oh-my-ai-docs
```

### Debug Mode

Enable comprehensive debugging:

```bash
export DEBUG=true
export LANGCHAIN_DEBUG_LOGS=1
export LOCAL_TEST_DEBUG=1
export BETTER_EXCEPTIONS=1

# Run with debug output
just avectorstore-discord
```

### Performance Issues

1. **Slow Search**: Increase `min_relevance_score` to filter results
2. **Memory Usage**: Reduce `k` parameter in queries
3. **Rate Limits**: Adjust `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SECONDS`

## Advanced Usage

### Custom Embeddings

```python
from oh_my_ai_docs.avectorstore_mcp import set_embeddings_provider
from langchain_openai import OpenAIEmbeddings

# Use custom embeddings configuration
custom_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536
)
set_embeddings_provider(custom_embeddings)
```

### Batch Operations

```bash
# Process multiple modules
for module in discord langgraph langchain dpytest; do
    just llmstxt-${module}-update
    just avectorstore-build-context-${module}
done
```

### Integration with AI Assistants

The MCP servers can be integrated with various AI assistants:

1. **Claude Desktop**: Add server configuration to `claude_desktop_config.json`
2. **Custom Clients**: Use the MCP protocol directly
3. **LangChain**: Integrate as a retriever component

### Custom MCP Server Configuration

```json
{
  "servers": {
    "custom-docs": {
      "command": ["uv", "run", "avectorstore_mcp"],
      "args": ["--module", "custom", "--stdio", "--debug"],
      "env": {
        "OPENAI_API_KEY": "your_key_here",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

## API Reference

### MCP Server Endpoints

#### Tool: query_docs
- **Description**: Semantic search through documentation
- **Parameters**:
  - `query` (string, required): Search query
  - `k` (integer, optional): Number of results (1-10, default: 2)
  - `min_relevance_score` (float, optional): Minimum relevance (0.0-1.0)
- **Returns**: List of relevant documentation snippets with metadata

#### Resource: docs://{module}/full
- **Description**: Retrieve complete documentation content
- **URI Pattern**: `docs://{module}/full`
- **Supported Modules**: discord, dpytest, langgraph, langchain
- **Returns**: Full documentation text content

### CLI Commands

#### avectorstore_mcp
```bash
uv run avectorstore_mcp [OPTIONS]

Options:
  --module TEXT          Documentation module to serve
  --stdio               Use stdio transport
  --debug               Enable debug logging
  --dry-run             Show configuration without starting
  --list-vectorstores   List available vector stores
  --generate-mcp-config Generate MCP configuration
  --save                Save generated configuration
```

#### goobctl
```bash
uv run goobctl [COMMAND] [OPTIONS]

Commands:
  version               Show version information
  config                Manage configuration
```

## Best Practices

### Performance Optimization

1. **Vector Store Management**:
   - Rebuild vectorstores periodically to maintain accuracy
   - Use appropriate `k` values (2-5 for most use cases)
   - Set reasonable `min_relevance_score` thresholds (0.5-0.8)

2. **Memory Management**:
   - Monitor memory usage during large document processing
   - Use batch processing for multiple modules
   - Clear unused vectorstores periodically

3. **API Usage**:
   - Implement proper rate limiting
   - Cache frequently accessed documentation
   - Use environment-specific API keys

### Security Considerations

1. **API Key Management**:
   - Store API keys in environment variables, not code
   - Use different keys for development and production
   - Rotate keys regularly

2. **Access Control**:
   - Implement proper authentication for production deployments
   - Use HTTPS for all external communications
   - Validate all input parameters

3. **Data Privacy**:
   - Be mindful of sensitive information in documentation
   - Implement proper logging practices
   - Consider data retention policies

## License & Contributing

This project is open source under the MIT License. Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

For detailed contributing guidelines, see the project's GitHub repository.

## Support & Resources

- **Documentation**: https://bossjones.github.io/oh-my-ai-docs/
- **Issues**: https://github.com/bossjones/oh-my-ai-docs/issues
- **Discussions**: GitHub Discussions
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

*This documentation is maintained as part of the oh-my-ai-docs project. For the most up-to-date information, please refer to the project repository.*
