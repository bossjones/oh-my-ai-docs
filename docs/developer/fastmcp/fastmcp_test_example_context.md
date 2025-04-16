# FastMCP Context Tests

This document demonstrates context functionality tests for FastMCP.

## Basic Context Detection and Injection

```python
@pytest.mark.anyio
async def test_context_detection():
    """Test that context parameters are properly detected."""
    mcp = FastMCP()

    def tool_with_context(x: int, ctx: Context) -> str:
        return f"Request {ctx.request_id}: {x}"

    tool = mcp._tool_manager.add_tool(tool_with_context)
    assert tool.context_kwarg == "ctx"

@pytest.mark.anyio
async def test_context_injection():
    """Test that context is properly injected into tool calls."""
    mcp = FastMCP()

    def tool_with_context(x: int, ctx: Context) -> str:
        assert ctx.request_id is not None
        return f"Request {ctx.request_id}: {x}"

    mcp.add_tool(tool_with_context)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("tool_with_context", {"x": 42})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "Request" in content.text
        assert "42" in content.text
```

## Async Context Support

```python
@pytest.mark.anyio
async def test_async_context():
    """Test that context works in async functions."""
    mcp = FastMCP()

    async def async_tool(x: int, ctx: Context) -> str:
        assert ctx.request_id is not None
        return f"Async request {ctx.request_id}: {x}"

    mcp.add_tool(async_tool)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("async_tool", {"x": 42})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "Async request" in content.text
        assert "42" in content.text
```

## Context Logging

```python
@pytest.mark.anyio
async def test_context_logging():
    """Test that context logging methods work."""
    mcp = FastMCP()

    async def logging_tool(msg: str, ctx: Context) -> str:
        await ctx.debug("Debug message")
        await ctx.info("Info message")
        await ctx.warning("Warning message")
        await ctx.error("Error message")
        return f"Logged messages for {msg}"

    mcp.add_tool(logging_tool)

    with patch("mcp.server.session.ServerSession.send_log_message") as mock_log:
        async with client_session(mcp._mcp_server) as client:
            result = await client.call_tool("logging_tool", {"msg": "test"})
            assert len(result.content) == 1
            content = result.content[0]
            assert isinstance(content, TextContent)
            assert "Logged messages for test" in content.text

            assert mock_log.call_count == 4
            mock_log.assert_any_call(
                level="debug", data="Debug message", logger=None
            )
            mock_log.assert_any_call(level="info", data="Info message", logger=None)
            mock_log.assert_any_call(
                level="warning", data="Warning message", logger=None
            )
            mock_log.assert_any_call(
                level="error", data="Error message", logger=None
            )
```

## Optional Context and Resource Access

```python
@pytest.mark.anyio
async def test_optional_context():
    """Test that context is optional."""
    mcp = FastMCP()

    def no_context(x: int) -> int:
        return x * 2

    mcp.add_tool(no_context)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("no_context", {"x": 21})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert content.text == "42"

@pytest.mark.anyio
async def test_context_resource_access():
    """Test that context can access resources."""
    mcp = FastMCP()

    @mcp.resource("test://data")
    def test_resource() -> str:
        return "resource data"

    @mcp.tool()
    async def tool_with_resource(ctx: Context) -> str:
        r_iter = await ctx.read_resource("test://data")
        r_list = list(r_iter)
        assert len(r_list) == 1
        r = r_list[0]
        return f"Read resource: {r.content} with mime type {r.mime_type}"

    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("tool_with_resource", {})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "Read resource: resource data" in content.text
```
