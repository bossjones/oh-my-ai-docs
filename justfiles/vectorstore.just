# vectorstore mcp server targets
[group('vectorstore')]
vectorstore-discord:
    uv run ./scripts/vectorstore_mcp.py --module discord --stdio

[group('vectorstore')]
vectorstore-discord-inspector:
    npx -y @modelcontextprotocol/inspector@0.8.2 uv run ./scripts/vectorstore_mcp.py --module discord --stdio

[group('vectorstore')]
vectorstore-discord-dry-run:
    uv run ./scripts/vectorstore_mcp.py --module discord --dry-run

[group('vectorstore')]
vectorstore-discord-dry-run-inspector:
    npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module discord --dry-run

[group('vectorstore')]
vectorstore-langgraph:
    uv run ./scripts/vectorstore_mcp.py --module langgraph --stdio --debug

[group('vectorstore')]
vectorstore-langgraph-inspector:
    npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module langgraph

[group('vectorstore')]
vectorstore-langgraph-dry-run:
    uv run ./scripts/vectorstore_mcp.py --module langgraph --dry-run --stdio --debug

[group('vectorstore')]
vectorstore-langgraph-dry-run-inspector:
    npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module langgraph --dry-run

[group('vectorstore')]
vectorstore-langchain:
    uv run ./scripts/vectorstore_mcp.py --module langchain --stdio --debug

[group('vectorstore')]
vectorstore-langchain-inspector:
    npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module langchain

[group('vectorstore')]
vectorstore-langchain-dry-run:
    uv run ./scripts/vectorstore_mcp.py --module langchain --dry-run --stdio --debug

[group('vectorstore')]
vectorstore-langchain-dry-run-inspector:
    npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module langchain --dry-run

[group('vectorstore')]
vectorstore-dpytest:
    uv run ./scripts/vectorstore_mcp.py --module dpytest --stdio --debug

[group('vectorstore')]
vectorstore-dpytest-inspector:
    npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module dpytest

[group('vectorstore')]
vectorstore-dpytest-dry-run:
    uv run ./scripts/vectorstore_mcp.py --module dpytest --dry-run --stdio --debug

[group('vectorstore')]
vectorstore-dpytest-dry-run-inspector:
    npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module dpytest --dry-run

# Run all vectorstore servers
[group('vectorstore')]
vectorstore-all:
    just vectorstore-discord
    just vectorstore-langgraph
    just vectorstore-langchain
    just vectorstore-dpytest

# Run all vectorstore servers with inspector
[group('vectorstore')]
vectorstore-all-inspector:
    just vectorstore-discord-inspector
    just vectorstore-langgraph-inspector
    just vectorstore-langchain-inspector
    just vectorstore-dpytest-inspector

# Run all vectorstore servers in dry-run mode
[group('vectorstore')]
vectorstore-all-dry-run:
    just vectorstore-discord-dry-run
    just vectorstore-langgraph-dry-run
    just vectorstore-langchain-dry-run
    just vectorstore-dpytest-dry-run

# Run all vectorstore servers in dry-run mode with inspector
[group('vectorstore')]
vectorstore-all-dry-run-inspector:
    just vectorstore-discord-dry-run-inspector
    just vectorstore-langgraph-dry-run-inspector
    just vectorstore-langchain-dry-run-inspector
    just vectorstore-dpytest-dry-run-inspector

# Run avectorstore-discord
[group('vectorstore')]
avectorstore-discord:
    uv run ./src/oh_my_ai_docs/avectorstore_mcp.py --module discord --stdio --debug

# Debug avectorstore-discord
[group('vectorstore')]
avectorstore-discord-inspector:
    npx -y @modelcontextprotocol/inspector@0.8.2 uv run ./src/oh_my_ai_docs/avectorstore_mcp.py --module discord --stdio --debug
