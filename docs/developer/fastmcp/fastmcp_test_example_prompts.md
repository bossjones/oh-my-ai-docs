# FastMCP Prompt Tests

This document demonstrates prompt functionality tests for FastMCP.

## Basic Prompt Registration

```python
@pytest.mark.anyio
async def test_prompt_decorator():
    """Test that the prompt decorator registers prompts correctly."""
    mcp = FastMCP()

    @mcp.prompt()
    def fn() -> str:
        return "Hello, world!"

    prompts = mcp._prompt_manager.list_prompts()
    assert len(prompts) == 1
    assert prompts[0].name == "fn"
    # Don't compare functions directly since validate_call wraps them
    content = await prompts[0].render()
    assert isinstance(content[0].content, TextContent)
    assert content[0].content.text == "Hello, world!"
```

## Prompt Configuration

```python
@pytest.mark.anyio
async def test_prompt_decorator_with_name():
    """Test prompt decorator with custom name."""
    mcp = FastMCP()

    @mcp.prompt(name="custom_name")
    def fn() -> str:
        return "Hello, world!"

    prompts = mcp._prompt_manager.list_prompts()
    assert len(prompts) == 1
    assert prompts[0].name == "custom_name"
    content = await prompts[0].render()
    assert isinstance(content[0].content, TextContent)
    assert content[0].content.text == "Hello, world!"

@pytest.mark.anyio
async def test_prompt_decorator_with_description():
    """Test prompt decorator with custom description."""
    mcp = FastMCP()

    @mcp.prompt(description="A custom description")
    def fn() -> str:
        return "Hello, world!"

    prompts = mcp._prompt_manager.list_prompts()
    assert len(prompts) == 1
    assert prompts[0].description == "A custom description"
    content = await prompts[0].render()
    assert isinstance(content[0].content, TextContent)
    assert content[0].content.text == "Hello, world!"
```

## Error Handling

```python
def test_prompt_decorator_error():
    """Test error when decorator is used incorrectly."""
    mcp = FastMCP()
    with pytest.raises(TypeError, match="decorator was used incorrectly"):
        @mcp.prompt  # Missing parentheses
        def fn() -> str:
            return "Hello, world!"
```

## Prompt Management

```python
@pytest.mark.anyio
async def test_list_prompts():
    """Test listing prompts through MCP protocol."""
    mcp = FastMCP()

    @mcp.prompt()
    def fn(name: str, optional: str = "default") -> str:
        return f"Hello, {name}!"

    async with client_session(mcp._mcp_server) as client:
        result = await client.list_prompts()
        assert result.prompts is not None
        assert len(result.prompts) == 1
        prompt = result.prompts[0]
        assert prompt.name == "fn"
        assert prompt.arguments is not None
        assert len(prompt.arguments) == 2
        assert prompt.arguments[0].name == "name"
        assert prompt.arguments[0].required is True
        assert prompt.arguments[1].name == "optional"
        assert prompt.arguments[1].required is False
```

## Prompt Execution

```python
@pytest.mark.anyio
async def test_get_prompt():
    """Test getting a prompt through MCP protocol."""
    mcp = FastMCP()

    @mcp.prompt()
    def fn(name: str) -> str:
        return f"Hello, {name}!"

    async with client_session(mcp._mcp_server) as client:
        result = await client.get_prompt("fn", {"name": "World"})
        assert len(result.messages) == 1
        message = result.messages[0]
        assert message.role == "user"
        content = message.content
        assert isinstance(content, TextContent)
        assert content.text == "Hello, World!"

@pytest.mark.anyio
async def test_get_prompt_with_resource():
    """Test getting a prompt that returns resource content."""
    mcp = FastMCP()

    @mcp.prompt()
    def fn() -> Message:
        return UserMessage(
            content=EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri=AnyUrl("file://file.txt"),
                    text="File contents",
                    mimeType="text/plain",
                ),
            )
        )

    async with client_session(mcp._mcp_server) as client:
        result = await client.get_prompt("fn")
        assert len(result.messages) == 1
        message = result.messages[0]
        assert message.role == "user"
        content = message.content
        assert isinstance(content, EmbeddedResource)
        resource = content.resource
        assert isinstance(resource, TextResourceContents)
        assert resource.text == "File contents"
        assert resource.mimeType == "text/plain"
```

## Error Cases

```python
@pytest.mark.anyio
async def test_get_unknown_prompt():
    """Test error when getting unknown prompt."""
    mcp = FastMCP()
    async with client_session(mcp._mcp_server) as client:
        with pytest.raises(McpError, match="Unknown prompt"):
            await client.get_prompt("unknown")

@pytest.mark.anyio
async def test_get_prompt_missing_args():
    """Test error when required arguments are missing."""
    mcp = FastMCP()

    @mcp.prompt()
    def prompt_fn(name: str) -> str:
        return f"Hello, {name}!"

    async with client_session(mcp._mcp_server) as client:
        with pytest.raises(McpError, match="Missing required arguments"):
            await client.get_prompt("prompt_fn")
```
