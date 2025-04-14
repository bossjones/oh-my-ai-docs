# FastMCP MVP Testing Agent Updates
Reference: See docs/fastmcp-examples.md for detailed examples and patterns to implement.

## Test Infrastructure
[x] Add rule for using `create_connected_server_and_client_session` alias as `client_session`
[x] Add rule for consistent server instance handling (`mcp._mcp_server`)
[x] Add rule for proper test isolation using in-memory connections

## Context Testing
[x] Add specific rules for testing async context support
[x] Add rules for testing context resource iteration (`r_iter = await ctx.read_resource()`)
[x] Add rules for testing optional context parameters
[x] Add specific assertions for context attribute verification

## Resource Testing
[x] Add rules for testing resource registration (`@mcp.resource`)
[x] Add rules for testing resource content verification
[x] Add rules for testing resource iteration patterns
[x] Add rules for testing resource cleanup in async contexts

## Tool Testing
[x] Add rules for tool registration testing (`mcp._tool_manager.add_tool`)
[x] Add rules for tool parameter detection testing
[x] Add rules for testing tool return value handling
[x] Add rules for testing tool async/sync compatibility

## Client Testing
[x] Add rules for testing client tool invocation patterns
[x] Add rules for testing client response handling
[x] Add rules for testing client error scenarios
[x] Add rules for testing client connection states

## Integration Testing
[x] Add rules for testing tool-resource interactions
[x] Add rules for testing context-resource interactions
[x] Add rules for testing client-server communication patterns
[x] Add rules for testing end-to-end workflows

## Documentation
[x] Add examples of proper test docstring formats
[x] Add examples of proper test class organization
[x] Add examples of proper test method naming
[x] Add examples of proper test assertion patterns

Progress tracking:
- Total tasks: 28
- Completed: 28
- Remaining: 0
