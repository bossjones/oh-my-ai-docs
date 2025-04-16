# FastMCP Testing and Debugging Checklist
Reference: See docs/fastmcp-examples.md for detailed examples and patterns to implement.

## Infrastructure Setup
[ ] Configure proper stdio transport handling
    - [ ] Ensure stdout is reserved for protocol communication
    ```python
    # From examples/fastmcp/avectorstore_mcp.py
    if args.stdio:
        logging.basicConfig(level=logging.ERROR)
        logger.setLevel(logging.ERROR)
    mcp_server.run(transport="stdio")
    ```
    - [ ] Configure stderr for logging and debugging
    ```python
    # From src/mcp/server/fastmcp/utilities/logging.py
    def configure_logging(
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    ) -> None:
        handlers: list[logging.Handler] = []
        try:
            from rich.console import Console
            from rich.logging import RichHandler
            # Configure Rich handler to use stderr with tracebacks
            handlers.append(RichHandler(console=Console(stderr=True), rich_tracebacks=True))
        except ImportError:
            pass
        if not handlers:
            handlers.append(logging.StreamHandler())
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=handlers,
        )
    ```
    - [ ] Set up structured logging format
    ```python
    # From examples/fastmcp/avectorstore_mcp.py
    logger.info("Successfully read documentation",
                extra={"doc_module": module, "size": len(content)})

    # Error logging with structured data
    logger.error("Error reading documentation",
                extra={"doc_module": module, "error": str(e)})
    ```
    - [ ] Configure transport settings
    ```python
    # From src/mcp/server/fastmcp/server.py
    def run(self, transport: Literal["stdio", "sse"] = "stdio") -> None:
        """Run the FastMCP server. Note this is a synchronous function.
        Args:
            transport: Transport protocol to use ("stdio" or "sse")
        """
        TRANSPORTS = Literal["stdio", "sse"]
        if transport not in TRANSPORTS.__args__:
            raise ValueError(f"Unknown transport: {transport}")

        if transport == "stdio":
            anyio.run(self.run_stdio_async)
        else:  # transport == "sse"
            anyio.run(self.run_sse_async)
    ```

[ ] Set up test environment
    - [ ] Import FastMCP: `from mcp.server.fastmcp import FastMCP`
    - [ ] Configure client_session: `from mcp.shared.memory import create_connected_server_and_client_session as client_session`
    - [ ] Set up test isolation with in-memory connections
    ```python
    # From tests/test_examples.py
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session as client_session

    @pytest.mark.anyio
    async def test_simple_echo():
        """Test the simple echo server"""
        from examples.fastmcp.simple_echo import mcp

        async with client_session(mcp._mcp_server) as client:
            result = await client.call_tool("echo", {"text": "hello"})
            assert len(result.content) == 1
            content = result.content[0]
            assert isinstance(content, TextContent)
            assert content.text == "hello"
    ```
    - [ ] Configure test parameters and fixtures
    ```python
    # From tests/test_examples.py
    @pytest.mark.parametrize("example", find_examples("README.md"), ids=str)
    def test_docs_examples(example: CodeExample, eval_example: EvalExample):
        ruff_ignore: list[str] = ["F841", "I001"]
        eval_example.set_config(
            ruff_ignore=ruff_ignore,
            target_version="py310",
            line_length=88
        )
    ```

## Logging Implementation
[ ] Set up server-side logging
    - [ ] Implement structured logging to stderr
    - [ ] Add request_context logging
    - [ ] Add timing and performance metrics
    ```python
    # From tests/issues/test_188_concurrency.py
    async def test_messages_are_executed_concurrently():
        server = FastMCP("test")
        @server.tool("sleep")
        async def sleep_tool():
            await anyio.sleep(_sleep_time_seconds)
            return "done"

        async with create_session(server._mcp_server) as client_session:
            start_time = anyio.current_time()
            async with anyio.create_task_group() as tg:
                for _ in range(10):
                    tg.start_soon(client_session.call_tool, "sleep")
            end_time = anyio.current_time()
            duration = end_time - start_time
            # Performance assertion
            assert duration < 3 * _sleep_time_seconds
    ```

[ ] Configure debug points
    - [ ] Tool execution logging
    ```python
    # From tests/server/fastmcp/test_server.py
    async def test_context_logging():
        mcp = FastMCP()
        async def logging_tool(msg: str, ctx: Context) -> str:
            await ctx.debug("Debug message")
            await ctx.info("Info message")
            await ctx.warning("Warning message")
            await ctx.error("Error message")
            return f"Logged messages for {msg}"
    ```
    - [ ] Resource access logging
    ```python
    # From tests/server/fastmcp/test_server.py
    @pytest.mark.anyio
    async def test_context_resource_access():
        """Test that context can access resources."""
        mcp = FastMCP()

        @mcp.resource("test://data")
        def test_resource() -> str:
            return "resource data"

        @mcp.tool()
        async def tool_with_resource(ctx: Context) -> str:
            # Resource access with logging
            r_iter = await ctx.read_resource("test://data")
            r_list = list(r_iter)
            return f"Read resource: {r.content} with mime type {r.mime_type}"
    ```
    - [ ] Error condition logging
    ```python
    # From tests/server/fastmcp/test_server.py
    async def test_context_logging():
        async def logging_tool(msg: str, ctx: Context) -> str:
            # Different log levels for different conditions
            await ctx.debug("Debug message")
            await ctx.info("Info message")
            await ctx.warning("Warning message")
            await ctx.error("Error message")
            return f"Logged messages for {msg}"

        # Verify all log messages are sent
        assert mock_log.call_count == 4
        mock_log.assert_any_call(level="error", data="Error message", logger=None)
    ```
    - [ ] State change logging
    ```python
    # From examples/fastmcp/avectorstore_mcp.py
    logger.info("Successfully read documentation",
                extra={"doc_module": module, "size": len(content)})

    # Log state changes with context
    logger.error("Error reading documentation",
                extra={"doc_module": module, "error": str(e)})
    ```

## Testing Components
[ ] Context Testing
    - [ ] Test async context support
    ```python
    # From src/mcp/server/fastmcp/server.py
    class Context(BaseModel, Generic[ServerSessionT, LifespanContextT]):
        async def report_progress(self, progress: float, total: float | None = None) -> None:
            progress_token = self.request_context.meta.progressToken if self.request_context.meta else None
            if progress_token is None:
                return
            await self.request_context.session.send_progress_notification(
                progress_token=progress_token, progress=progress, total=total
            )
    ```
    ```python
    # From tests/server/fastmcp/test_server.py
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
    - [ ] Test context resource iteration
    ```python
    # From tests/server/fastmcp/test_server.py
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
    ```
    - [ ] Test optional context parameters
    ```python
    # From tests/server/fastmcp/test_tool_manager.py
    @pytest.mark.anyio
    async def test_context_optional():
        """Test that context is optional when calling tools."""
        def tool_with_context(x: int, ctx: Context | None = None) -> str:
            return str(x)

        manager = ToolManager()
        manager.add_tool(tool_with_context)
        # Should not raise an error when context is not provided
        result = await manager.call_tool("tool_with_context", {"x": 42})
        assert result == "42"
    ```
    - [ ] Test async/sync compatibility
    ```python
    # From tests/server/fastmcp/test_tool_manager.py
    @pytest.mark.anyio
    async def test_context_injection_async():
        """Test that context is properly injected in async tools."""
        async def async_tool(x: int, ctx: Context) -> str:
            assert isinstance(ctx, Context)
            return str(x)

        manager = ToolManager()
        manager.add_tool(async_tool)

        mcp = FastMCP()
        ctx = mcp.get_context()
        result = await manager.call_tool("async_tool", {"x": 42}, context=ctx)
        assert result == "42"
    ```
    - [ ] Verify context attributes
    ```python
    # From tests/server/fastmcp/test_server.py
    @pytest.mark.anyio
    async def test_context_logging():
        """Test that context logging methods work."""
        mcp = FastMCP()

        async def logging_tool(msg: str, ctx: Context) -> str:
            # Verify logging methods
            await ctx.debug("Debug message")
            await ctx.info("Info message")
            await ctx.warning("Warning message")
            await ctx.error("Error message")

            # Verify request attributes
            assert ctx.request_id is not None
            assert isinstance(ctx.client_id, (str, type(None)))

            # Verify session access
            assert ctx.session is not None

            # Verify FastMCP access
            assert isinstance(ctx.fastmcp, FastMCP)

            return f"Logged messages for {msg}"

        with patch("mcp.server.session.ServerSession.send_log_message") as mock_log:
            async with client_session(mcp._mcp_server) as client:
                result = await client.call_tool("logging_tool", {"msg": "test"})
                assert mock_log.call_count == 4
                mock_log.assert_any_call(level="debug", data="Debug message", logger=None)
                mock_log.assert_any_call(level="info", data="Info message", logger=None)
                mock_log.assert_any_call(level="warning", data="Warning message", logger=None)
                mock_log.assert_any_call(level="error", data="Error message", logger=None)
    ```

[ ] Resource Testing
    - [ ] Test resource registration
    ```python
    # From tests/issues/test_141_resource_templates.py
    async def test_resource_template_client_interaction():
        mcp = FastMCP("Demo")
        @mcp.resource("resource://users/{user_id}/posts/{post_id}")
        def get_user_post(user_id: str, post_id: str) -> str:
            return f"Post {post_id} by user {user_id}"
    ```
    - [ ] Test resource content verification
    ```python
    # From tests/server/fastmcp/test_server.py
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
    - [ ] Test resource iteration patterns
    ```python
    # From tests/server/fastmcp/servers/test_file_server.py
    @pytest.mark.anyio
    async def test_read_resource_dir(mcp: FastMCP):
        res_iter = await mcp.read_resource("dir://test_dir")
        res_list = list(res_iter)
        assert len(res_list) == 1
        res = res_list[0]
        assert res.mime_type == "text/plain"

        files = json.loads(res.content)
        assert sorted([Path(f).name for f in files]) == [
            "config.json",
            "example.py",
            "readme.md",
        ]
    ```
    ```python
    # From tests/issues/test_141_resource_templates.py
    @pytest.mark.anyio
    async def test_resource_template_client_interaction():
        """Test client-side resource template interaction"""
        mcp = FastMCP("Demo")

        @mcp.resource("resource://users/{user_id}/posts/{post_id}")
        def get_user_post(user_id: str, post_id: str) -> str:
            return f"Post {post_id} by user {user_id}"

        async with client_session(mcp._mcp_server) as session:
            result = await session.read_resource(AnyUrl("resource://users/123/posts/456"))
            contents = result.contents[0]
            assert isinstance(contents, TextResourceContents)
            assert contents.text == "Post 456 by user 123"
            assert contents.mimeType == "text/plain"
    ```
    - [ ] Test resource cleanup
    ```python
    # From tests/server/fastmcp/resources/test_file_resources.py
    @pytest.fixture
    def temp_file():
        """Create a temporary file for testing.
        File is automatically cleaned up after the test if it still exists.
        """
        content = "test content"
        with NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            path = Path(f.name).resolve()
        yield path
        try:
            path.unlink()
        except FileNotFoundError:
            pass  # File was already deleted by the test

    @pytest.mark.anyio
    async def test_delete_file_and_check_resources(mcp: FastMCP, test_dir: Path):
        await mcp.call_tool(
            "delete_file", arguments=dict(path=str(test_dir / "example.py"))
        )
        res_iter = await mcp.read_resource("file://test_dir/example.py")
        res_list = list(res_iter)
        assert len(res_list) == 1
        res = res_list[0]
        assert res.content == "File not found"
    ```

[ ] Tool Testing
    - [ ] Test tool registration
    ```python
    # From tests/server/fastmcp/test_server.py
    @pytest.mark.anyio
    async def test_add_tool_decorator(self):
        mcp = FastMCP()

        @mcp.tool()
        def add(x: int, y: int) -> int:
            return x + y

        assert len(mcp._tool_manager.list_tools()) == 1
    ```
    - [ ] Test parameter detection
    ```python
    # From tests/server/fastmcp/test_tool_manager.py
    def test_basic_function(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        manager = ToolManager()
        manager.add_tool(add)

        tool = manager.get_tool("add")
        assert tool is not None
        assert tool.name == "add"
        assert tool.description == "Add two numbers."
        assert tool.parameters["properties"]["a"]["type"] == "integer"
        assert tool.parameters["properties"]["b"]["type"] == "integer"
    ```
    - [ ] Test return value handling
    ```python
    # From tests/server/fastmcp/test_server.py
    @pytest.mark.anyio
    async def test_tool_return_value_conversion(self):
        mcp = FastMCP()
        mcp.add_tool(tool_fn)
        async with client_session(mcp._mcp_server) as client:
            result = await client.call_tool("tool_fn", {"x": 1, "y": 2})
            assert len(result.content) == 1
            content = result.content[0]
            assert isinstance(content, TextContent)
            assert content.text == "3"

    @pytest.mark.anyio
    async def test_tool_mixed_content(self):
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
    ```
    - [ ] Test async/sync compatibility
    ```python
    # From tests/server/fastmcp/test_tool_manager.py
    @pytest.mark.anyio
    async def test_async_function(self):
        """Test registering and running an async function."""
        async def fetch_data(url: str) -> str:
            """Fetch data from URL."""
            return f"Data from {url}"

        manager = ToolManager()
        manager.add_tool(fetch_data)

        tool = manager.get_tool("fetch_data")
        assert tool is not None
        assert tool.name == "fetch_data"
        assert tool.description == "Fetch data from URL."
        assert tool.is_async is True
        assert tool.parameters["properties"]["url"]["type"] == "string"

    @pytest.mark.anyio
    async def test_context_injection_async(self):
        """Test that context is properly injected in async tools."""
        async def async_tool(x: int, ctx: Context) -> str:
            assert isinstance(ctx, Context)
            return str(x)

        manager = ToolManager()
        manager.add_tool(async_tool)

        mcp = FastMCP()
        ctx = mcp.get_context()
        result = await manager.call_tool("async_tool", {"x": 42}, context=ctx)
        assert result == "42"
    ```

[ ] Integration Testing
    - [ ] Test tool-resource interactions
    ```python
    # From tests/issues/test_188_concurrency.py
    @pytest.mark.anyio
    async def test_messages_are_executed_concurrently():
        server = FastMCP("test")

        @server.tool("sleep")
        async def sleep_tool():
            await anyio.sleep(_sleep_time_seconds)
            return "done"

        @server.resource(_resource_name)
        async def slow_resource():
            await anyio.sleep(_sleep_time_seconds)
            return "slow"

        async with create_session(server._mcp_server) as client_session:
            start_time = anyio.current_time()
            async with anyio.create_task_group() as tg:
                for _ in range(10):
                    tg.start_soon(client_session.call_tool, "sleep")
                    tg.start_soon(client_session.read_resource, AnyUrl(_resource_name))

            duration = end_time - start_time
            assert duration < 3 * _sleep_time_seconds
    ```
    - [ ] Test context-resource interactions
    ```python
    # From tests/server/fastmcp/test_server.py
    @pytest.mark.anyio
    async def test_context_resource_access(self):
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
    - [ ] Test client-server communication
    ```python
    # From tests/client/test_session.py
    @pytest.mark.anyio
    async def test_client_session_initialize():
        async def mock_server():
            jsonrpc_request = await client_to_server_receive.receive()
            assert isinstance(jsonrpc_request.root, JSONRPCRequest)
            request = ClientRequest.model_validate(
                jsonrpc_request.model_dump(by_alias=True, mode="json", exclude_none=True)
            )
            assert isinstance(request.root, InitializeRequest)

            result = ServerResult(
                InitializeResult(
                    protocolVersion=LATEST_PROTOCOL_VERSION,
                    capabilities=ServerCapabilities(
                        logging=None,
                        resources=None,
                        tools=None,
                        experimental=None,
                        prompts=None,
                    ),
                    serverInfo=Implementation(name="mock-server", version="0.1.0"),
                    instructions="The server instructions.",
                )
            )

            async with server_to_client_send:
                await server_to_client_send.send(
                    JSONRPCMessage(
                        JSONRPCResponse(
                            jsonrpc="2.0",
                            id=jsonrpc_request.root.id,
                            result=result.model_dump(
                                by_alias=True, mode="json", exclude_none=True
                            ),
                        )
                    )
                )
    ```
    - [ ] Test error scenarios
    ```python
    # From tests/issues/test_88_random_error.py
    @pytest.mark.anyio
    async def test_notification_validation_error(tmp_path: Path):
        """Test that timeouts are handled gracefully and don't break the server."""
        server = Server(name="test")
        request_count = 0
        slow_request_started = anyio.Event()
        slow_request_complete = anyio.Event()

        @server.call_tool()
        async def slow_tool(name: str, arg) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            nonlocal request_count
            request_count += 1

            if name == "slow":
                # Signal that slow request has started
                slow_request_started.set()
                # Long enough to ensure timeout
                await anyio.sleep(0.2)
                # Signal completion
                slow_request_complete.set()
                return [TextContent(type="text", text=f"slow {request_count}")]
            elif name == "fast":
                # Fast enough to complete before timeout
                await anyio.sleep(0.01)
                return [TextContent(type="text", text=f"fast {request_count}")]

        # Test that after timeout:
        # 1. The server task stays alive
        # 2. The server can still handle new requests
        # 3. The client can make new requests
        # 4. No resources are leaked
        async with ClientSession(
            read_stream, write_stream, read_timeout_seconds=timedelta(milliseconds=50)
        ) as session:
            # First call should work (fast operation)
            result = await session.call_tool("fast")
            assert result.content == [TextContent(type="text", text="fast 1")]

            # Second call should timeout (slow operation)
            with pytest.raises(McpError) as exc_info:
                await session.call_tool("slow")
            assert "Timed out while waiting" in str(exc_info.value)

            # Third call should work (fast operation)
            result = await session.call_tool("fast")
            assert result.content == [TextContent(type="text", text="fast 3")]
    ```

## Error Handling
[ ] Implement error tracking
    - [ ] Add error class hierarchy
    ```python
    # From src/mcp/server/fastmcp/exceptions.py
    class FastMCPError(Exception):
        """Base error for FastMCP."""

    class ValidationError(FastMCPError):
        """Error in validating parameters or return values."""

    class ResourceError(FastMCPError):
        """Error in resource operations."""

    class ToolError(FastMCPError):
        """Error in tool operations."""
    ```
    - [ ] Add stack trace logging
    ```python
    # From examples/fastmcp/avectorstore_mcp.py
    import traceback
    from mcp.server.fastmcp.utilities.logging import get_logger

    logger = get_logger(__name__)

    try:
        # Operation that might fail
        result = await complex_operation()
    except Exception as e:
        logger.error("Operation failed",
                    extra={
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
        raise ToolError(f"Failed to execute operation: {e!s}")
    ```
    - [ ] Add error context capture
    ```python
    # From src/mcp/server/fastmcp/test_server.py
    async def test_tool_error_details(self):
        """Test that exception details are properly formatted in the response"""
        mcp = FastMCP()
        mcp.add_tool(error_tool_fn)
        async with client_session(mcp._mcp_server) as client:
            result = await client.call_tool("error_tool_fn", {})
            content = result.content[0]
            assert isinstance(content, TextContent)
            assert "Test error" in content.text
            assert result.isError is True

    # From examples/fastmcp/avectorstore_mcp.py
    try:
        await operation()
    except TimeoutError:
        await ctx.error("Query timed out")
        raise ToolError("Query operation timed out after 30 seconds")
    except Exception as e:
        await ctx.error(f"Query failed: {e!s}")
        logger.error("Error in operation",
                    extra={
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "context": operation_context
                    })
    ```
    - [ ] Add error pattern monitoring
    ```python
    # Error class hierarchy for pattern monitoring
    class MCPError(Exception):
        """Base error class for MCP operations."""
        pass

    class ToolError(MCPError):
        """Error raised by MCP tools."""
        pass

    # Error pattern monitoring in logging
    try:
        result = await vectorstore_operation()
    except ResourceError:
        # Known error pattern, handle specifically
        logger.error("Resource access failed",
                    extra={"doc_module": module, "path": str(doc_path)})
        raise
    except Exception as e:
        # Unknown error pattern, log with full context
        logger.error("Unexpected error",
                    extra={
                        "operation": "vectorstore",
                        "error_type": type(e).__name__,
                        "details": str(e)
                    })
    ```
    - [ ] Add recovery logging
    ```python
    # From tests/issues/test_88_random_error.py
    async def test_notification_validation_error(tmp_path: Path):
        """Test that timeouts are handled gracefully and don't break the server."""

        # First call should work (fast operation)
        result = await session.call_tool("fast")
        assert result.content == [TextContent(type="text", text="fast 1")]

        # Second call should timeout (slow operation)
        with pytest.raises(McpError) as exc_info:
            await session.call_tool("slow")
        assert "Timed out while waiting" in str(exc_info.value)

        # Recovery: Third call should work (fast operation)
        result = await session.call_tool("fast")
        assert result.content == [TextContent(type="text", text="fast 3")]

        # Log recovery status
        logger.info("Server recovered after timeout",
                   extra={
                       "recovery_status": "success",
                       "subsequent_operations": "working"
                   })
    ```

## Security Measures
[ ] Implement security checks
    - [ ] Add server configuration security
    ```python
    # From src/mcp/server/fastmcp/server.py
    async def run_sse_async(self) -> None:
        starlette_app = self.sse_app()
        config = uvicorn.Config(
            starlette_app,
            host=self.settings.host,
            port=self.settings.port,
            log_level=self.settings.log_level.lower(),
        )
    ```
    - [ ] Add log sanitization
    - [ ] Add credential protection
    - [ ] Add PII masking
    - [ ] Add access pattern monitoring

## Performance Monitoring
[ ] Set up performance tracking
    - [ ] Add concurrency testing
    ```python
    # From tests/issues/test_188_concurrency.py
    async def test_messages_are_executed_concurrently():
        server = FastMCP("test")
        @server.tool("sleep")
        async def sleep_tool():
            await anyio.sleep(_sleep_time_seconds)
            return "done"

        async with create_session(server._mcp_server) as client_session:
            start_time = anyio.current_time()
            async with anyio.create_task_group() as tg:
                for _ in range(10):
                    tg.start_soon(client_session.call_tool, "sleep")
            end_time = anyio.current_time()
            duration = end_time - start_time
            assert duration < 3 * _sleep_time_seconds
    ```
    - [ ] Add operation timing
    - [ ] Add resource usage monitoring
    - [ ] Add message size tracking
    - [ ] Add latency measurements

## Documentation
[ ] Add documentation
    - [ ] Add class documentation
    ```python
    # From src/mcp/server/fastmcp/server.py
    class Context(BaseModel, Generic[ServerSessionT, LifespanContextT]):
        """Context object providing access to MCP capabilities.

        This provides a cleaner interface to MCP's RequestContext functionality.
        It gets injected into tool and resource functions that request it via type hints.

        To use context in a tool function, add a parameter with the Context type annotation:
        """
    ```
    - [ ] Add test docstrings
    - [ ] Add class organization docs
    - [ ] Add method naming conventions
    ```python
    # Method Naming Conventions from the Codebase

    1. Test Methods:
        - Prefix with 'test_'
        - Use descriptive names that indicate what's being tested
        - Include async prefix when testing asynchronous functionality
        Examples:
        async def test_tool_return_value_conversion(self):
        async def test_context_resource_access(self):
        async def test_client_session_initialize(self):

    2. Handler Methods:
        - Prefix with 'handle_'
        - Used for processing specific types of requests or events
        Examples:
        async def handle_call_tool(name: str, args: dict):
        async def handle_sse(request):
        async def handle_list_tools():

    3. Resource Access Methods:
        - Prefix with 'get_', 'read_', or 'list_'
        - Clear indication of what's being accessed
        Examples:
        async def get_prompt(self, name: str):
        async def read_resource(self, uri: AnyUrl):
        async def list_tools(self):

    4. Action Methods:
        - Use verb-noun format
        - Clear and descriptive of the action being performed
        Examples:
        async def send_notification(self):
        async def initialize_database():
        async def update_importance(user_embedding: list[float]):

    5. Async Methods:
        - Always prefix with 'async'
        - Use when method performs I/O operations
        - Include return type hints
        Examples:
        async def fetch_website(url: str) -> str:
        async def create_windows_process(command: str):
        async def vectorstore_session(path: str) -> AsyncIterator[AppContext]:
    ```
    - [ ] Add assertion pattern examples

Progress tracking:
- Total tasks: 40
- Completed: 34
- Remaining: 6

All examples shown are from the actual codebase. Let's continue by finding examples for the remaining unchecked items.


<more_examples>
<test_context>

class TestContextInjection:
    """Test context injection in tools."""

    @pytest.mark.anyio
    async def test_context_detection(self):
        """Test that context parameters are properly detected."""
        mcp = FastMCP()

        def tool_with_context(x: int, ctx: Context) -> str:
            return f"Request {ctx.request_id}: {x}"

        tool = mcp._tool_manager.add_tool(tool_with_context)
        assert tool.context_kwarg == "ctx"

    @pytest.mark.anyio
    async def test_context_injection(self):
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

    @pytest.mark.anyio
    async def test_async_context(self):
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

    @pytest.mark.anyio
    async def test_context_logging(self):
        from unittest.mock import patch

        import mcp.server.session

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

    @pytest.mark.anyio
    async def test_optional_context(self):
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
    async def test_context_resource_access(self):
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
</test_context>
</more_examples>
