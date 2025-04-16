# Basic FastMCP Server Tests

This document demonstrates basic server functionality tests for FastMCP.

## Server Creation and Configuration

```python
@pytest.mark.anyio
async def test_create_server():
    mcp = FastMCP(instructions="Server instructions")
    assert mcp.name == "FastMCP"
    assert mcp.instructions == "Server instructions"
```

## Unicode and Non-ASCII Support

```python
@pytest.mark.anyio
async def test_non_ascii_description():
    """Test that FastMCP handles non-ASCII characters in descriptions correctly"""
    mcp = FastMCP()

    @mcp.tool(
        description=(
            "🌟 This tool uses emojis and UTF-8 characters: á é í ó ú ñ 漢字 🎉"
        )
    )
    def hello_world(name: str = "世界") -> str:
        return f"¡Hola, {name}! 👋"

    async with client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 1
        tool = tools.tools[0]
        assert tool.description is not None
        assert "🌟" in tool.description
        assert "漢字" in tool.description
        assert "🎉" in tool.description

        result = await client.call_tool("hello_world", {})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "¡Hola, 世界! 👋" == content.text
```

## Tool Decorator Tests

```python
@pytest.mark.anyio
async def test_add_tool_decorator():
    mcp = FastMCP()

    @mcp.tool()
    def add(x: int, y: int) -> int:
        return x + y

    assert len(mcp._tool_manager.list_tools()) == 1

@pytest.mark.anyio
async def test_add_tool_decorator_incorrect_usage():
    mcp = FastMCP()

    with pytest.raises(TypeError, match="The @tool decorator was used incorrectly"):
        @mcp.tool  # Missing parentheses
        def add(x: int, y: int) -> int:
            return x + y
```

## Resource Decorator Tests

```python
@pytest.mark.anyio
async def test_add_resource_decorator():
    mcp = FastMCP()

    @mcp.resource("r://{x}")
    def get_data(x: str) -> str:
        return f"Data: {x}"

    assert len(mcp._resource_manager._templates) == 1

@pytest.mark.anyio
async def test_add_resource_decorator_incorrect_usage():
    mcp = FastMCP()

    with pytest.raises(TypeError, match="The @resource decorator was used incorrectly"):
        @mcp.resource  # Missing parentheses
        def get_data(x: str) -> str:
            return f"Data: {x}"
```
