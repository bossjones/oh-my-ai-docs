# FastMCP Resource Tests

This document demonstrates resource functionality tests for FastMCP.

## Basic Resource Types

### Text Resources

```python
@pytest.mark.anyio
async def test_text_resource():
    mcp = FastMCP()

    def get_text():
        return "Hello, world!"

    resource = FunctionResource(
        uri=AnyUrl("resource://test"), name="test", fn=get_text
    )
    mcp.add_resource(resource)

    async with client_session(mcp._mcp_server) as client:
        result = await client.read_resource(AnyUrl("resource://test"))
        assert isinstance(result.contents[0], TextResourceContents)
        assert result.contents[0].text == "Hello, world!"
```

### Binary Resources

```python
@pytest.mark.anyio
async def test_binary_resource():
    mcp = FastMCP()

    def get_binary():
        return b"Binary data"

    resource = FunctionResource(
        uri=AnyUrl("resource://binary"),
        name="binary",
        fn=get_binary,
        mime_type="application/octet-stream",
    )
    mcp.add_resource(resource)

    async with client_session(mcp._mcp_server) as client:
        result = await client.read_resource(AnyUrl("resource://binary"))
        assert isinstance(result.contents[0], BlobResourceContents)
        assert result.contents[0].blob == base64.b64encode(b"Binary data").decode()
```

## File Resources

### Text Files

```python
@pytest.mark.anyio
async def test_file_resource_text(tmp_path: Path):
    mcp = FastMCP()

    # Create a text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("Hello from file!")

    resource = FileResource(
        uri=AnyUrl("file://test.txt"), name="test.txt", path=text_file
    )
    mcp.add_resource(resource)

    async with client_session(mcp._mcp_server) as client:
        result = await client.read_resource(AnyUrl("file://test.txt"))
        assert isinstance(result.contents[0], TextResourceContents)
        assert result.contents[0].text == "Hello from file!"
```

### Binary Files

```python
@pytest.mark.anyio
async def test_file_resource_binary(tmp_path: Path):
    mcp = FastMCP()

    # Create a binary file
    binary_file = tmp_path / "test.bin"
    binary_file.write_bytes(b"Binary file data")

    resource = FileResource(
        uri=AnyUrl("file://test.bin"),
        name="test.bin",
        path=binary_file,
        mime_type="application/octet-stream",
    )
    mcp.add_resource(resource)

    async with client_session(mcp._mcp_server) as client:
        result = await client.read_resource(AnyUrl("file://test.bin"))
        assert isinstance(result.contents[0], BlobResourceContents)
        assert (
            result.contents[0].blob
            == base64.b64encode(b"Binary file data").decode()
        )
```

## Resource Templates

### Parameter Validation

```python
@pytest.mark.anyio
async def test_resource_with_params():
    """Test that a resource with function parameters raises an error if the URI
    parameters don't match"""
    mcp = FastMCP()

    with pytest.raises(ValueError, match="Mismatch between URI parameters"):
        @mcp.resource("resource://data")
        def get_data_fn(param: str) -> str:
            return f"Data: {param}"

@pytest.mark.anyio
async def test_resource_with_uri_params():
    """Test that a resource with URI parameters is automatically a template"""
    mcp = FastMCP()

    with pytest.raises(ValueError, match="Mismatch between URI parameters"):
        @mcp.resource("resource://{param}")
        def get_data() -> str:
            return "Data"
```

### Parameter Matching

```python
@pytest.mark.anyio
async def test_resource_matching_params():
    """Test that a resource with matching URI and function parameters works"""
    mcp = FastMCP()

    @mcp.resource("resource://{name}/data")
    def get_data(name: str) -> str:
        return f"Data for {name}"

    async with client_session(mcp._mcp_server) as client:
        result = await client.read_resource(AnyUrl("resource://test/data"))
        assert isinstance(result.contents[0], TextResourceContents)
        assert result.contents[0].text == "Data for test"

@pytest.mark.anyio
async def test_resource_multiple_params():
    """Test that multiple parameters work correctly"""
    mcp = FastMCP()

    @mcp.resource("resource://{org}/{repo}/data")
    def get_data(org: str, repo: str) -> str:
        return f"Data for {org}/{repo}"

    async with client_session(mcp._mcp_server) as client:
        result = await client.read_resource(
            AnyUrl("resource://cursor/fastmcp/data")
        )
        assert isinstance(result.contents[0], TextResourceContents)
        assert result.contents[0].text == "Data for cursor/fastmcp"
```

### Template to Resource Conversion

```python
@pytest.mark.anyio
async def test_template_to_resource_conversion():
    """Test that templates are properly converted to resources when accessed"""
    mcp = FastMCP()

    @mcp.resource("resource://{name}/data")
    def get_data(name: str) -> str:
        return f"Data for {name}"

    # Should be registered as a template
    assert len(mcp._resource_manager._templates) == 1
    assert len(await mcp.list_resources()) == 0

    # When accessed, should create a concrete resource
    resource = await mcp._resource_manager.get_resource("resource://test/data")
    assert isinstance(resource, FunctionResource)
    result = await resource.read()
    assert result == "Data for test"
```
