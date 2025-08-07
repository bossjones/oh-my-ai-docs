# # vectorstore mcp server targets
# [group('vectorstore')]
# vectorstore-discord:
#     uv run ./scripts/vectorstore_mcp.py --module discord --stdio

# [group('vectorstore')]
# vectorstore-discord-inspector:
#     npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module discord --stdio

# [group('vectorstore')]
# vectorstore-discord-dry-run:
#     uv run ./scripts/vectorstore_mcp.py --module discord --dry-run

# [group('vectorstore')]
# vectorstore-discord-dry-run-inspector:
#     npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module discord --dry-run

# [group('vectorstore')]
# vectorstore-langgraph:
#     uv run ./scripts/vectorstore_mcp.py --module langgraph --stdio --debug

# [group('vectorstore')]
# vectorstore-langgraph-inspector:
#     npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module langgraph

# [group('vectorstore')]
# vectorstore-langgraph-dry-run:
#     uv run ./scripts/vectorstore_mcp.py --module langgraph --dry-run --stdio --debug

# [group('vectorstore')]
# vectorstore-langgraph-dry-run-inspector:
#     npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module langgraph --dry-run

# [group('vectorstore')]
# vectorstore-langchain:
#     uv run ./scripts/vectorstore_mcp.py --module langchain --stdio --debug

# [group('vectorstore')]
# vectorstore-langchain-inspector:
#     npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module langchain

# [group('vectorstore')]
# vectorstore-langchain-dry-run:
#     uv run ./scripts/vectorstore_mcp.py --module langchain --dry-run --stdio --debug

# [group('vectorstore')]
# vectorstore-langchain-dry-run-inspector:
#     npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module langchain --dry-run

# [group('vectorstore')]
# vectorstore-dpytest:
#     uv run ./scripts/vectorstore_mcp.py --module dpytest --stdio --debug

# [group('vectorstore')]
# vectorstore-dpytest-inspector:
#     npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module dpytest

# [group('vectorstore')]
# vectorstore-dpytest-dry-run:
#     uv run ./scripts/vectorstore_mcp.py --module dpytest --dry-run --stdio --debug

# [group('vectorstore')]
# vectorstore-dpytest-dry-run-inspector:
#     npx -y @modelcontextprotocol/inspector uv run ./scripts/vectorstore_mcp.py --module dpytest --dry-run

# # Run all vectorstore servers
# [group('vectorstore')]
# vectorstore-all:
#     just vectorstore-discord
#     just vectorstore-langgraph
#     just vectorstore-langchain
#     just vectorstore-dpytest

# # Run all vectorstore servers with inspector
# [group('vectorstore')]
# vectorstore-all-inspector:
#     just vectorstore-discord-inspector
#     just vectorstore-langgraph-inspector
#     just vectorstore-langchain-inspector
#     just vectorstore-dpytest-inspector

# # Run all vectorstore servers in dry-run mode
# [group('vectorstore')]
# vectorstore-all-dry-run:
#     just vectorstore-discord-dry-run
#     just vectorstore-langgraph-dry-run
#     just vectorstore-langchain-dry-run
#     just vectorstore-dpytest-dry-run

# # Run all vectorstore servers in dry-run mode with inspector
# [group('vectorstore')]
# vectorstore-all-dry-run-inspector:
#     just vectorstore-discord-dry-run-inspector
#     just vectorstore-langgraph-dry-run-inspector
#     just vectorstore-langchain-dry-run-inspector
#     just vectorstore-dpytest-dry-run-inspector

# Run avectorstore-discord
[group('vectorstore')]
avectorstore-discord:
    uv run ./src/oh_my_ai_docs/avectorstore_mcp.py --module discord --stdio --debug

# Debug avectorstore-discord
[group('vectorstore')]
avectorstore-discord-inspector:
    npx -y @modelcontextprotocol/inspector uv run ./src/oh_my_ai_docs/avectorstore_mcp.py --module discord --stdio --debug

# Build sklearn vectorstore context for discord module
[group('vectorstore')]
avectorstore-build-context-discord:
    uv run ./scripts/build_llmstxt_context.py --module discord

# Build sklearn vectorstore context for langgraph module
[group('vectorstore')]
avectorstore-build-context-langgraph:
    uv run ./scripts/build_llmstxt_context.py --module langgraph

# Build sklearn vectorstore context for langchain module
[group('vectorstore')]
avectorstore-build-context-langchain:
    uv run ./scripts/build_llmstxt_context.py --module langchain

# Build sklearn vectorstore context for dpytest module
[group('vectorstore')]
avectorstore-build-context-dpytest:
    uv run ./scripts/build_llmstxt_context.py --module dpytest

# Build all sklearn vectorstore contexts
[group('vectorstore')]
avectorstore-build-context-all:
    just avectorstore-build-context-discord
    just avectorstore-build-context-langgraph
    just avectorstore-build-llmstxt-langchain
    just avectorstore-build-llmstxt-dpytest
