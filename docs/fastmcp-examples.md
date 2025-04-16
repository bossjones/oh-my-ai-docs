- [FastMCP Context Usage Examples (GitHub Code Search)](https://github.com/search?q=from+mcp.server.fastmcp+import+Context%2C+FastMCP+request_context&type=code)
- [iTerm MCP Tests](https://github.com/research-developer/iterm-mcp/tree/main/tests)
- [Logfire MCP Implementation](https://github.com/pydantic/logfire-mcp/blob/de90f038bf7175ef2e4916903e388d222670bb2d/logfire_mcp/__main__.py#L13)
- [OSO MCP Warehouse Implementation](https://github.com/opensource-observer/oso/blob/778de2bd353e576e41648ea5332b323970d16bea/warehouse/oso_mcp/main.py#L8)
- [AMemCP FastMCP Server Implementation](https://github.com/jpollard-cs/AMemCP/blob/015559c70b492a426b9c98c99b21e3cdc5a50db8/amem/server/mcp_fastmcp_server.py#L16)


Let me analyze the testing patterns for context in this codebase, particularly focusing on the `TestContextInjection` class and related tests. Here are the key patterns I observe:

1. **Basic Context Detection Testing**
```python
@pytest.mark.anyio  # Always used for async tests
async def test_context_detection():
    # Test that context parameters are properly identified
    def tool_with_context(x: int, ctx: Context) -> str:
        return f"Request {ctx.request_id}: {x}"

    tool = mcp._tool_manager.add_tool(tool_with_context)
    assert tool.context_kwarg == "ctx"  # Verify context param is detected
```

2. **Context Injection Testing**
```python
@pytest.mark.anyio
async def test_context_injection():
    # Test context is properly injected with required attributes
    def tool_with_context(x: int, ctx: Context) -> str:
        assert ctx.request_id is not None  # Verify context has required attributes
        return f"Request {ctx.request_id}: {x}"
```

3. **Async Context Support**
```python
@pytest.mark.anyio
async def test_async_context():
    # Test context works in async functions
    async def async_tool(x: int, ctx: Context) -> str:
        assert ctx.request_id is not None
        return f"Async request {ctx.request_id}: {x}"
```

4. **Context Logging Testing**
```python
@pytest.mark.anyio
async def test_context_logging():
    # Test all logging levels
    async def logging_tool(msg: str, ctx: Context) -> str:
        await ctx.debug("Debug message")
        await ctx.info("Info message")
        await ctx.warning("Warning message")
        await ctx.error("Error message")

    # Mock logging to verify calls
    with patch("mcp.server.session.ServerSession.send_log_message") as mock_log:
        # Verify each log level was called
        mock_log.assert_any_call(level="debug", data="Debug message", logger=None)
```

5. **Optional Context Testing**
```python
@pytest.mark.anyio
async def test_optional_context():
    # Test tools work without context
    def no_context(x: int) -> int:
        return x * 2
```

6. **Context Resource Access Testing**
```python
@pytest.mark.anyio
async def test_context_resource_access():
    # Test context can access resources
    @mcp.resource("test://data")
    def test_resource() -> str:
        return "resource data"

    @mcp.tool()
    async def tool_with_resource(ctx: Context) -> str:
        r_iter = await ctx.read_resource("test://data")
        # Verify resource access and content
```

Key Testing Patterns:

1. **Always Use `@pytest.mark.anyio`**: Every context test is marked with `@pytest.mark.anyio` for async support.

2. **Client Session Pattern**: Tests consistently use `client_session` with async context management:
```python
async with client_session(mcp._mcp_server) as client:
    result = await client.call_tool(...)
```

3. **Verification Layers**:
   - Verify context detection/injection
   - Verify context attributes (e.g., `request_id`)
   - Verify context functionality (logging, resource access)
   - Verify async/sync compatibility
   - Verify optional context scenarios

4. **Mock Integration**: Uses mocking for external dependencies (e.g., logging)

5. **Resource Cleanup**: Uses context managers to ensure proper cleanup

6. **Comprehensive Coverage**:
   - Tests sync and async functions
   - Tests all logging levels
   - Tests resource access
   - Tests optional context scenarios
   - Tests error cases

7. **Type Safety**: Tests include proper type hints for context parameters

These patterns ensure robust testing of context functionality across different scenarios and use cases in the FastMCP framework 🧪

------------------------------

Looking at this import and its usage throughout the test file, here are the key patterns I notice:

1. **Consistent Alias Usage**
```python
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)
```
- The long function name is aliased to a shorter, more readable `client_session`
- This alias is used consistently throughout all tests
- Makes the test code more concise and readable

2. **Context Manager Pattern**
```python
async with client_session(mcp._mcp_server) as client:
    result = await client.call_tool(...)
```
- Always used as an async context manager with `async with`
- Ensures proper setup and cleanup of server/client connections
- Handles connection lifecycle automatically

3. **Testing Infrastructure Abstraction**
- The import is from `mcp.shared.memory`, suggesting it's a testing utility
- The name suggests it creates an in-memory server-client connection for testing
- Abstracts away the complexity of setting up test connections

4. **Consistent Usage Pattern**
Throughout the tests, it's always used:
- Inside async test functions (marked with `@pytest.mark.anyio`)
- Takes the server instance (`mcp._mcp_server`) as an argument
- Returns a client that's used to interact with the server
- Used for all server interaction tests (tools, resources, prompts)

5. **Single Connection Per Test**
- Each test creates its own isolated connection
- Connection is scoped to the test's context manager block
- Ensures test isolation and clean state

6. **Testing Helper Function**
- The function name suggests it's specifically designed for testing
- Creates both server and client in one call
- "Connected" in the name implies it handles connection setup

7. **Memory-Based Testing**
- The `memory` module name suggests this creates in-memory connections
- Avoids actual network/IO operations during tests
- Makes tests faster and more reliable

Here's a typical usage pattern seen throughout the tests:
```python
@pytest.mark.anyio
async def test_something():
    mcp = FastMCP()
    # Setup test resources/tools

    async with client_session(mcp._mcp_server) as client:
        # Perform test operations using client
        result = await client.some_operation()
        # Assert expected results
```

This pattern provides:
- Clean setup and teardown
- Isolated test environments
- Consistent server-client interaction model
- Proper resource management
- Clear test structure 🔍

The import and its usage demonstrate good testing practices by providing a reliable, consistent way to test server-client interactions in an isolated environment 🧪


----

Let me analyze the test file to identify patterns around mocking. I'll use the sequential thinking tool to break this down systematically.

Ran tool

Ran tool

Ran tool

Ran tool
Based on my analysis, here are the key patterns around mocking in this test file:

1. **Minimal Use of Mocks**: The file strongly favors real implementations over mocks. Only one explicit mock exists in the entire test suite.

2. **Mocking Reserved for Side Effects**: The only mock is used for logging in `test_context_logging`, where it's necessary to:
   - Verify logging behavior without actually writing logs
   - Assert specific log levels and messages were called
   - Avoid actual I/O operations during testing

3. **Architecture Favors Testability Without Mocks**: The codebase appears designed to be testable without heavy mocking through:
   - Use of `client_session` for isolated test environments
   - In-memory operations where possible
   - Temporary files for file-based tests
   - Real async/await operations instead of mocked ones
   - Direct error triggering instead of mocked errors

4. **Integration-Style Testing**: The tests lean towards integration-style testing where components work together with real implementations, rather than unit tests with mocked dependencies. For example:
   - Real FastMCP instances are created
   - Actual tools and resources are registered
   - Real client-server communication occurs
   - Actual file operations are performed (with temporary files)

This approach suggests a design philosophy that values testing real component interactions over isolated unit testing with mocks 🔍
