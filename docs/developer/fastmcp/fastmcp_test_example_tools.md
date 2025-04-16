# FastMCP Tool Tests

This document demonstrates tool functionality tests for FastMCP.

## Basic Tool Operations

```python
@pytest.mark.anyio
async def test_add_tool():
    mcp = FastMCP()
    mcp.add_tool(tool_fn)
    mcp.add_tool(tool_fn)  # Adding same tool twice
    assert len(mcp._tool_manager.list_tools()) == 1

@pytest.mark.anyio
async def test_list_tools():
    mcp = FastMCP()
    mcp.add_tool(tool_fn)
    async with client_session(mcp._mcp_server) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 1

@pytest.mark.anyio
async def test_call_tool():
    mcp = FastMCP()
    mcp.add_tool(tool_fn)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("my_tool", {"arg1": "value"})
        assert not hasattr(result, "error")
        assert len(result.content) > 0
```

## Error Handling

```python
@pytest.mark.anyio
async def test_tool_exception_handling():
    mcp = FastMCP()
    mcp.add_tool(error_tool_fn)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("error_tool_fn", {})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "Test error" in content.text
        assert result.isError is True

@pytest.mark.anyio
async def test_tool_error_details():
    """Test that exception details are properly formatted in the response"""
    mcp = FastMCP()
    mcp.add_tool(error_tool_fn)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("error_tool_fn", {})
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert isinstance(content.text, str)
        assert "Test error" in content.text
        assert result.isError is True
```

## Return Value Handling

```python
@pytest.mark.anyio
async def test_tool_return_value_conversion():
    mcp = FastMCP()
    mcp.add_tool(tool_fn)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("tool_fn", {"x": 1, "y": 2})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert content.text == "3"
```

## Image and Mixed Content Handling

```python
@pytest.mark.anyio
async def test_tool_image_helper(tmp_path: Path):
    # Create a test image
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"fake png data")

    mcp = FastMCP()
    mcp.add_tool(image_tool_fn)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("image_tool_fn", {"path": str(image_path)})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, ImageContent)
        assert content.type == "image"
        assert content.mimeType == "image/png"
        # Verify base64 encoding
        decoded = base64.b64decode(content.data)
        assert decoded == b"fake png data"

@pytest.mark.anyio
async def test_tool_mixed_content():
    mcp = FastMCP()
    mcp.add_tool(mixed_content_tool_fn)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("mixed_content_tool_fn", {})
        assert len(result.content) == 2
        content1 = result.content[0]
        content2 = result.content[1]
        assert isinstance(content1, TextContent)
        assert content1.text == "Hello"
        assert isinstance(content2, ImageContent)
        assert content2.mimeType == "image/png"
        assert content2.data == "abc"

@pytest.mark.anyio
async def test_tool_mixed_list_with_image(tmp_path: Path):
    """Test that lists containing Image objects and other types are handled correctly"""
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"test image data")

    def mixed_list_fn() -> list:
        return [
            "text message",
            Image(image_path),
            {"key": "value"},
            TextContent(type="text", text="direct content"),
        ]

    mcp = FastMCP()
    mcp.add_tool(mixed_list_fn)
    async with client_session(mcp._mcp_server) as client:
        result = await client.call_tool("mixed_list_fn", {})
        assert len(result.content) == 4
        # Check text conversion
        content1 = result.content[0]
        assert isinstance(content1, TextContent)
        assert content1.text == "text message"
        # Check image conversion
        content2 = result.content[1]
        assert isinstance(content2, ImageContent)
        assert content2.mimeType == "image/png"
        assert base64.b64decode(content2.data) == b"test image data"
        # Check dict conversion
        content3 = result.content[2]
        assert isinstance(content3, TextContent)
        assert '"key": "value"' in content3.text
        # Check direct TextContent
        content4 = result.content[3]
        assert isinstance(content4, TextContent)
        assert content4.text == "direct content"
```
