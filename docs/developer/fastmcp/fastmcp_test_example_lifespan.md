# FastMCP Server Lifespan Tests

This document demonstrates lifespan functionality tests for FastMCP servers, showing how to implement and test server lifespan contexts.

## Basic Lifespan Context Test

```python
@pytest.mark.anyio
async def test_fastmcp_server_lifespan():
    """Test that lifespan works in FastMCP server."""

    @asynccontextmanager
    async def test_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        """Test lifespan context that tracks startup/shutdown."""
        context = {"started": False, "shutdown": False}
        try:
            context["started"] = True
            yield context
        finally:
            context["shutdown"] = True

    server = FastMCP("test", lifespan=test_lifespan)
```

## Lifespan Context in Tools

```python
@pytest.mark.anyio
async def test_lifespan_context_in_tools():
    server = FastMCP("test", lifespan=test_lifespan)

    @server.tool()
    def check_lifespan(ctx: Context) -> bool:
        """Tool that checks lifespan context."""
        assert isinstance(ctx.request_context.lifespan_context, dict)
        assert ctx.request_context.lifespan_context["started"]
        assert not ctx.request_context.lifespan_context["shutdown"]
        return True
```

## Complete Lifespan Test Example

Here's a complete example showing how to test lifespan functionality with server initialization, tool calls, and proper cleanup:

```python
@pytest.mark.anyio
async def test_fastmcp_server_lifespan():
    """Test that lifespan works in FastMCP server."""

    @asynccontextmanager
    async def test_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        context = {"started": False, "shutdown": False}
        try:
            context["started"] = True
            yield context
        finally:
            context["shutdown"] = True

    server = FastMCP("test", lifespan=test_lifespan)

    # Add a tool that checks lifespan context
    @server.tool()
    def check_lifespan(ctx: Context) -> bool:
        assert isinstance(ctx.request_context.lifespan_context, dict)
        assert ctx.request_context.lifespan_context["started"]
        assert not ctx.request_context.lifespan_context["shutdown"]
        return True

    # Test the lifespan context through a tool call
    async with client_session(server._mcp_server) as client:
        result = await client.call_tool("check_lifespan", {})
        assert result.content[0].text == "true"
```

## Error Handling Examples

### Handling Initialization Errors

```python
@pytest.mark.anyio
async def test_lifespan_initialization_error():
    """Test handling of errors during lifespan initialization."""

    @asynccontextmanager
    async def failing_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        try:
            # Simulate a database connection error
            raise ConnectionError("Failed to connect to database")
            yield {}  # This line won't be reached
        finally:
            # Cleanup should still run
            pass

    server = FastMCP("test", lifespan=failing_lifespan)

    with pytest.raises(ConnectionError, match="Failed to connect to database"):
        async with client_session(server._mcp_server) as client:
            # This should fail before we can make any calls
            await client.call_tool("some_tool", {})

### Handling Cleanup Errors

```python
@pytest.mark.anyio
async def test_lifespan_cleanup_error():
    """Test handling of errors during lifespan cleanup."""
    cleanup_attempted = False

    @asynccontextmanager
    async def cleanup_error_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        try:
            yield {"resource": "some_resource"}
        finally:
            nonlocal cleanup_attempted
            cleanup_attempted = True
            # Simulate error during cleanup
            raise RuntimeError("Failed to cleanup resources")

    server = FastMCP("test", lifespan=cleanup_error_lifespan)

    with pytest.raises(RuntimeError, match="Failed to cleanup resources"):
        async with client_session(server._mcp_server) as client:
            # The tool call should succeed
            result = await client.call_tool("some_tool", {})
            assert result is not None
            # Error should be raised during context exit

    # Verify cleanup was attempted
    assert cleanup_attempted

### Graceful Degradation

```python
@pytest.mark.anyio
async def test_lifespan_graceful_degradation():
    """Test graceful degradation when optional resources fail."""

    @asynccontextmanager
    async def partial_failure_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        context = {
            "critical_resource": None,
            "optional_resource": None,
            "errors": []
        }

        try:
            # Critical resource initialization
            context["critical_resource"] = "database_connection"

            try:
                # Optional resource that might fail
                raise ConnectionError("Cache server unavailable")
                context["optional_resource"] = "cache_connection"
            except Exception as e:
                # Log error but continue
                context["errors"].append(str(e))

            yield context
        finally:
            # Cleanup resources in reverse order
            if context["optional_resource"]:
                context["optional_resource"] = None
            if context["critical_resource"]:
                context["critical_resource"] = None

    server = FastMCP("test", lifespan=partial_failure_lifespan)

    @server.tool()
    def check_resources(ctx: Context) -> dict:
        """Tool that checks available resources."""
        lifespan_ctx = ctx.request_context.lifespan_context
        return {
            "has_critical": lifespan_ctx["critical_resource"] is not None,
            "has_optional": lifespan_ctx["optional_resource"] is not None,
            "errors": lifespan_ctx["errors"]
        }

    async with client_session(server._mcp_server) as client:
        result = await client.call_tool("check_resources", {})
        response = eval(result.content[0].text)  # Convert string response to dict
        assert response["has_critical"] is True
        assert response["has_optional"] is False
        assert len(response["errors"]) == 1
        assert "Cache server unavailable" in response["errors"][0]

### Recovery and Retry

```python
@pytest.mark.anyio
async def test_lifespan_recovery():
    """Test recovery attempts for failed resource initialization."""
    retry_count = 0
    max_retries = 3

    @asynccontextmanager
    async def retry_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        nonlocal retry_count
        context = {"connection": None, "retry_count": 0}

        while retry_count < max_retries:
            try:
                # Simulate connection attempt
                if retry_count < 2:  # Succeed on third try
                    retry_count += 1
                    raise ConnectionError(f"Connection failed, attempt {retry_count}")

                context["connection"] = "established"
                context["retry_count"] = retry_count
                break
            except ConnectionError:
                if retry_count >= max_retries - 1:
                    raise
                await anyio.sleep(0.1)  # Add delay between retries

        try:
            yield context
        finally:
            context["connection"] = None

    server = FastMCP("test", lifespan=retry_lifespan)

    @server.tool()
    def check_connection(ctx: Context) -> dict:
        """Tool that checks connection status."""
        lifespan_ctx = ctx.request_context.lifespan_context
        return {
            "connection": lifespan_ctx["connection"],
            "retry_count": lifespan_ctx["retry_count"]
        }

    async with client_session(server._mcp_server) as client:
        result = await client.call_tool("check_connection", {})
        response = eval(result.content[0].text)  # Convert string response to dict
        assert response["connection"] == "established"
        assert response["retry_count"] == 2  # Succeeded on third try (index 2)
```

## Resource Management Examples

### Database Connection Lifecycle

```python
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncIterator

@pytest.mark.anyio
async def test_database_lifecycle():
    """Test database connection lifecycle in lifespan context."""

    @asynccontextmanager
    async def database_lifespan(server: FastMCP) -> AsyncIterator[dict[str, sqlite3.Connection]]:
        # Initialize database connection
        connection = sqlite3.connect(":memory:")
        try:
            # Set up schema
            with connection:
                connection.execute("""
                    CREATE TABLE test_data (
                        id INTEGER PRIMARY KEY,
                        value TEXT
                    )
                """)

            yield {"db": connection}
        finally:
            connection.close()

    server = FastMCP("test", lifespan=database_lifespan)

    @server.tool()
    def db_operation(value: str, ctx: Context) -> str:
        """Tool that uses the database connection."""
        db = ctx.request_context.lifespan_context["db"]
        with db:
            db.execute("INSERT INTO test_data (value) VALUES (?)", (value,))
            result = db.execute("SELECT value FROM test_data").fetchone()
        return f"Stored value: {result[0]}"

    async with client_session(server._mcp_server) as client:
        result = await client.call_tool("db_operation", {"value": "test_value"})
        assert "Stored value: test_value" in result.content[0].text

### File Handle Management

```python
from pathlib import Path
import tempfile
import shutil

@pytest.mark.anyio
async def test_file_handle_lifecycle():
    """Test file handle management in lifespan context."""

    @asynccontextmanager
    async def file_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        log_file = temp_dir / "app.log"

        try:
            # Initialize log file
            log_handle = open(log_file, "w+")
            yield {
                "log_file": log_handle,
                "temp_dir": temp_dir
            }
        finally:
            # Clean up resources in reverse order
            if "log_file" in locals():
                log_handle.close()
            shutil.rmtree(temp_dir)

    server = FastMCP("test", lifespan=file_lifespan)

    @server.tool()
    def log_message(message: str, ctx: Context) -> str:
        """Tool that writes to the log file."""
        log_file = ctx.request_context.lifespan_context["log_file"]
        log_file.write(f"{message}\n")
        log_file.flush()
        return "Message logged successfully"

    async with client_session(server._mcp_server) as client:
        result = await client.call_tool("log_message", {"message": "test log"})
        assert "Message logged successfully" in result.content[0].text

### Cache Management

```python
from typing import Any
import time

class SimpleCache:
    def __init__(self, ttl: int = 300):
        self.cache: dict[str, tuple[Any, float]] = {}
        self.ttl = ttl

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (value, time.time())

    def get(self, key: str) -> Any | None:
        if key not in self.cache:
            return None
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        return value

@pytest.mark.anyio
async def test_cache_lifecycle():
    """Test cache initialization and cleanup in lifespan context."""

    @asynccontextmanager
    async def cache_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        # Initialize cache with 5-second TTL
        cache = SimpleCache(ttl=5)
        try:
            yield {"cache": cache}
        finally:
            # Clear cache on shutdown
            cache.cache.clear()

    server = FastMCP("test", lifespan=cache_lifespan)

    @server.tool()
    def cache_operation(key: str, value: str | None, ctx: Context) -> str:
        """Tool that interacts with the cache."""
        cache = ctx.request_context.lifespan_context["cache"]

        if value is not None:
            cache.set(key, value)
            return f"Stored in cache: {key}={value}"

        result = cache.get(key)
        return f"Cache value for {key}: {result}"

    async with client_session(server._mcp_server) as client:
        # Test cache storage
        result = await client.call_tool("cache_operation", {"key": "test", "value": "data"})
        assert "Stored in cache" in result.content[0].text

        # Test cache retrieval
        result = await client.call_tool("cache_operation", {"key": "test", "value": None})
        assert "Cache value for test: data" in result.content[0].text

## Advanced Scenarios

### Shared Resources Between Tools

```python
@pytest.mark.anyio
async def test_shared_resources():
    """Test multiple tools sharing the same lifespan resources."""

    @asynccontextmanager
    async def shared_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        context = {
            "counter": 0,
            "data": {}
        }
        try:
            yield context
        finally:
            context.clear()

    server = FastMCP("test", lifespan=shared_lifespan)

    @server.tool()
    def increment_counter(ctx: Context) -> int:
        """First tool that modifies shared state."""
        ctx.request_context.lifespan_context["counter"] += 1
        return ctx.request_context.lifespan_context["counter"]

    @server.tool()
    def store_data(key: str, value: str, ctx: Context) -> str:
        """Second tool that modifies shared state."""
        ctx.request_context.lifespan_context["data"][key] = value
        return f"Stored: {key}={value}"

    @server.tool()
    def get_state(ctx: Context) -> dict:
        """Third tool that reads shared state."""
        return {
            "counter": ctx.request_context.lifespan_context["counter"],
            "data": ctx.request_context.lifespan_context["data"]
        }

    async with client_session(server._mcp_server) as client:
        # Test counter increment
        result = await client.call_tool("increment_counter", {})
        assert result.content[0].text == "1"

        # Test data storage
        result = await client.call_tool("store_data", {"key": "test", "value": "data"})
        assert "Stored: test=data" in result.content[0].text

        # Verify shared state
        result = await client.call_tool("get_state", {})
        state = eval(result.content[0].text)
        assert state["counter"] == 1
        assert state["data"]["test"] == "data"

### Custom State Types

```python
from dataclasses import dataclass
from typing import TypedDict

class ResourceState(TypedDict):
    is_ready: bool
    error: str | None

@dataclass
class AppConfig:
    debug: bool
    max_connections: int
    timeout: float

@pytest.mark.anyio
async def test_custom_state_types():
    """Test using custom types for lifespan state."""

    @asynccontextmanager
    async def typed_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        config = AppConfig(
            debug=True,
            max_connections=10,
            timeout=5.0
        )

        state: ResourceState = {
            "is_ready": True,
            "error": None
        }

        try:
            yield {
                "config": config,
                "state": state
            }
        finally:
            pass

    server = FastMCP("test", lifespan=typed_lifespan)

    @server.tool()
    def get_config(ctx: Context) -> dict:
        """Tool that accesses typed config."""
        config: AppConfig = ctx.request_context.lifespan_context["config"]
        return {
            "debug": config.debug,
            "max_connections": config.max_connections,
            "timeout": config.timeout
        }

    @server.tool()
    def get_state(ctx: Context) -> ResourceState:
        """Tool that accesses typed state."""
        return ctx.request_context.lifespan_context["state"]

    async with client_session(server._mcp_server) as client:
        # Test config access
        result = await client.call_tool("get_config", {})
        config = eval(result.content[0].text)
        assert config["debug"] is True
        assert config["max_connections"] == 10
        assert config["timeout"] == 5.0

        # Test state access
        result = await client.call_tool("get_state", {})
        state = eval(result.content[0].text)
        assert state["is_ready"] is True
        assert state["error"] is None

### Nested Lifespan Contexts

```python
@pytest.mark.anyio
async def test_nested_lifespan():
    """Test nested lifespan contexts with dependency chain."""

    @asynccontextmanager
    async def database_lifespan() -> AsyncIterator[sqlite3.Connection]:
        connection = sqlite3.connect(":memory:")
        try:
            yield connection
        finally:
            connection.close()

    @asynccontextmanager
    async def cache_lifespan() -> AsyncIterator[SimpleCache]:
        cache = SimpleCache(ttl=60)
        try:
            yield cache
        finally:
            cache.cache.clear()

    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        async with database_lifespan() as db:
            async with cache_lifespan() as cache:
                yield {
                    "db": db,
                    "cache": cache
                }

    server = FastMCP("test", lifespan=app_lifespan)

    @server.tool()
    def check_resources(ctx: Context) -> dict:
        """Tool that verifies all nested resources are available."""
        context = ctx.request_context.lifespan_context
        return {
            "has_db": isinstance(context["db"], sqlite3.Connection),
            "has_cache": isinstance(context["cache"], SimpleCache)
        }

    async with client_session(server._mcp_server) as client:
        result = await client.call_tool("check_resources", {})
        resources = eval(result.content[0].text)
        assert resources["has_db"] is True
        assert resources["has_cache"] is True
```

## Integration Patterns

### External Service Integration

```python
import aiohttp
from dataclasses import dataclass
from typing import AsyncIterator
import json

@dataclass
class ExternalServiceConfig:
    base_url: str
    api_key: str
    timeout: float = 30.0

class ExternalServiceClient:
    def __init__(self, config: ExternalServiceConfig):
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_data(self, endpoint: str) -> dict:
        if not self._session:
            raise RuntimeError("Client not initialized")
        async with self._session.get(f"{self.config.base_url}{endpoint}") as response:
            response.raise_for_status()
            return await response.json()

@pytest.mark.anyio
async def test_external_service_integration():
    """Test integration with an external service."""

    @asynccontextmanager
    async def service_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        # Initialize external service client
        config = ExternalServiceConfig(
            base_url="https://api.example.com",
            api_key="test_key"
        )
        client = ExternalServiceClient(config)

        try:
            await client.initialize()
            yield {"external_service": client}
        finally:
            await client.close()

    server = FastMCP("test", lifespan=service_lifespan)

    @server.tool()
    async def fetch_external_data(endpoint: str, ctx: Context) -> dict:
        """Tool that fetches data from external service."""
        client = ctx.request_context.lifespan_context["external_service"]
        return await client.get_data(endpoint)

    # Mock external service for testing
    async with aiohttp.ClientSession() as session:
        async with client_session(server._mcp_server) as client:
            # Test would typically interact with the external service
            pass

### State Persistence

```python
import pickle
from pathlib import Path
import asyncio
from typing import Any

class PersistentState:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        if self.file_path.exists():
            async with self._lock:
                self.data = pickle.loads(self.file_path.read_bytes())

    async def save(self) -> None:
        async with self._lock:
            self.file_path.write_bytes(pickle.dumps(self.data))

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self.data[key] = value
            await self.save()

    async def get(self, key: str) -> Any:
        async with self._lock:
            return self.data.get(key)

@pytest.mark.anyio
async def test_state_persistence():
    """Test persistent state management across server lifecycles."""

    state_file = Path("test_state.pkl")

    @asynccontextmanager
    async def persistent_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        state = PersistentState(state_file)
        try:
            await state.load()
            yield {"state": state}
        finally:
            await state.save()
            if state_file.exists():
                state_file.unlink()

    # First server instance
    server1 = FastMCP("test", lifespan=persistent_lifespan)

    @server1.tool()
    async def set_state(key: str, value: Any, ctx: Context) -> str:
        """Tool that sets persistent state."""
        state = ctx.request_context.lifespan_context["state"]
        await state.set(key, value)
        return f"State set: {key}={value}"

    # Test first server instance
    async with client_session(server1._mcp_server) as client:
        result = await client.call_tool("set_state", {
            "key": "test_key",
            "value": "test_value"
        })
        assert "State set" in result.content[0].text

    # Second server instance to verify persistence
    server2 = FastMCP("test", lifespan=persistent_lifespan)

    @server2.tool()
    async def get_state(key: str, ctx: Context) -> Any:
        """Tool that gets persistent state."""
        state = ctx.request_context.lifespan_context["state"]
        return await state.get(key)

    async with client_session(server2._mcp_server) as client:
        result = await client.call_tool("get_state", {"key": "test_key"})
        assert "test_value" in result.content[0].text

### Multi-Server State Sharing

```python
import redis.asyncio as redis
from typing import TypedDict

class SharedState(TypedDict):
    counter: int
    last_updated: str

@pytest.mark.anyio
async def test_multi_server_state():
    """Test state sharing between multiple server instances."""

    @asynccontextmanager
    async def redis_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        # Initialize Redis connection
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        try:
            await redis_client.ping()  # Verify connection
            yield {"redis": redis_client}
        finally:
            await redis_client.close()

    # Create multiple server instances
    server1 = FastMCP("server1", lifespan=redis_lifespan)
    server2 = FastMCP("server2", lifespan=redis_lifespan)

    @server1.tool()
    async def increment_counter(ctx: Context) -> int:
        """Tool that increments shared counter on server 1."""
        redis_client = ctx.request_context.lifespan_context["redis"]
        return await redis_client.incr("shared_counter")

    @server2.tool()
    async def get_counter(ctx: Context) -> int:
        """Tool that reads shared counter on server 2."""
        redis_client = ctx.request_context.lifespan_context["redis"]
        value = await redis_client.get("shared_counter")
        return int(value) if value else 0

    # Test state sharing between servers
    async with (
        client_session(server1._mcp_server) as client1,
        client_session(server2._mcp_server) as client2
    ):
        # Increment counter on server 1
        result = await client1.call_tool("increment_counter", {})
        assert result.content[0].text == "1"

        # Read counter from server 2
        result = await client2.call_tool("get_counter", {})
        assert result.content[0].text == "1"

### Configuration Management

```python
import yaml
from typing import TypedDict
from pathlib import Path

class DatabaseConfig(TypedDict):
    host: str
    port: int
    database: str

class AppConfig(TypedDict):
    debug: bool
    database: DatabaseConfig
    api_keys: dict[str, str]

@pytest.mark.anyio
async def test_configuration_management():
    """Test configuration loading and management in lifespan."""

    # Create test configuration
    config_data = {
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "test_db"
        },
        "api_keys": {
            "service1": "key1",
            "service2": "key2"
        }
    }

    config_file = Path("test_config.yml")
    config_file.write_text(yaml.dump(config_data))

    @asynccontextmanager
    async def config_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        try:
            # Load configuration
            config: AppConfig = yaml.safe_load(config_file.read_text())

            # Validate configuration
            assert isinstance(config["debug"], bool)
            assert all(k in config["database"] for k in ("host", "port", "database"))
            assert isinstance(config["api_keys"], dict)

            yield {"config": config}
        finally:
            if config_file.exists():
                config_file.unlink()

    server = FastMCP("test", lifespan=config_lifespan)

    @server.tool()
    def get_database_config(ctx: Context) -> DatabaseConfig:
        """Tool that retrieves database configuration."""
        return ctx.request_context.lifespan_context["config"]["database"]

    @server.tool()
    def get_api_key(service: str, ctx: Context) -> str:
        """Tool that retrieves API key for a service."""
        api_keys = ctx.request_context.lifespan_context["config"]["api_keys"]
        if service not in api_keys:
            raise KeyError(f"No API key found for service: {service}")
        return api_keys[service]

    async with client_session(server._mcp_server) as client:
        # Test database config access
        result = await client.call_tool("get_database_config", {})
        db_config = eval(result.content[0].text)
        assert db_config["host"] == "localhost"
        assert db_config["port"] == 5432

        # Test API key access
        result = await client.call_tool("get_api_key", {"service": "service1"})
        assert result.content[0].text == "key1"
```

## Testing Patterns

### Basic Lifespan Testing

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import anyio
import pytest
from mcp.server.fastmcp import Context, FastMCP

@pytest.mark.anyio
async def test_basic_lifespan():
    """Test basic lifespan functionality with memory streams."""

    @asynccontextmanager
    async def test_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        context = {"initialized": False, "cleaned_up": False}
        try:
            context["initialized"] = True
            yield context
        finally:
            context["cleaned_up"] = True

    server = FastMCP("test", lifespan=test_lifespan)

    # Create memory streams for testing
    send_stream1, receive_stream1 = anyio.create_memory_object_stream(100)
    send_stream2, receive_stream2 = anyio.create_memory_object_stream(100)

    @server.tool()
    def verify_state(ctx: Context) -> dict:
        """Tool that verifies lifespan state."""
        state = ctx.request_context.lifespan_context
        return {
            "initialized": state["initialized"],
            "cleaned_up": state["cleaned_up"]
        }

    async with (
        anyio.create_task_group() as tg,
        send_stream1,
        receive_stream1,
        send_stream2,
        receive_stream2,
    ):
        # Start server in background
        tg.start_soon(lambda: server._mcp_server.run(
            receive_stream1,
            send_stream2,
            server._mcp_server.create_initialization_options(),
            raise_exceptions=True
        ))

        async with client_session(server._mcp_server) as client:
            result = await client.call_tool("verify_state", {})
            state = eval(result.content[0].text)
            assert state["initialized"] is True
            assert state["cleaned_up"] is False

        # Cancel server to test cleanup
        tg.cancel_scope.cancel()

### Controlled Shutdown Testing

```python
@pytest.mark.anyio
async def test_controlled_shutdown():
    """Test lifespan behavior during controlled shutdown."""

    shutdown_steps = []

    @asynccontextmanager
    async def shutdown_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        shutdown_steps.append("initialize")
        try:
            yield {"status": "running"}
        finally:
            shutdown_steps.append("cleanup")

    server = FastMCP("test", lifespan=shutdown_lifespan)

    # Create memory streams for testing
    send_stream1, receive_stream1 = anyio.create_memory_object_stream(100)
    send_stream2, receive_stream2 = anyio.create_memory_object_stream(100)

    async with (
        anyio.create_task_group() as tg,
        send_stream1,
        receive_stream1,
        send_stream2,
        receive_stream2,
    ):
        # Start server
        tg.start_soon(lambda: server._mcp_server.run(
            receive_stream1,
            send_stream2,
            server._mcp_server.create_initialization_options(),
            raise_exceptions=True
        ))

        async with client_session(server._mcp_server) as client:
            # Verify server is running
            assert shutdown_steps == ["initialize"]

        # Trigger controlled shutdown
        tg.cancel_scope.cancel()

        # Verify shutdown sequence
        assert shutdown_steps == ["initialize", "cleanup"]

### State Verification During Lifecycle

```python
@pytest.mark.anyio
async def test_lifecycle_state():
    """Test state verification throughout server lifecycle."""

    class LifecycleTracker:
        def __init__(self):
            self.states = []
            self.current_state = "initial"

        def transition_to(self, state: str) -> None:
            self.states.append(state)
            self.current_state = state

        def get_history(self) -> list[str]:
            return self.states.copy()

    tracker = LifecycleTracker()

    @asynccontextmanager
    async def lifecycle_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        tracker.transition_to("initializing")
        try:
            tracker.transition_to("running")
            yield {"tracker": tracker}
        finally:
            tracker.transition_to("shutting_down")
            tracker.transition_to("terminated")

    server = FastMCP("test", lifespan=lifecycle_lifespan)

    @server.tool()
    def get_lifecycle_state(ctx: Context) -> dict:
        """Tool that returns current lifecycle state."""
        tracker: LifecycleTracker = ctx.request_context.lifespan_context["tracker"]
        return {
            "current_state": tracker.current_state,
            "history": tracker.get_history()
        }

    # Create memory streams for testing
    send_stream1, receive_stream1 = anyio.create_memory_object_stream(100)
    send_stream2, receive_stream2 = anyio.create_memory_object_stream(100)

    async with (
        anyio.create_task_group() as tg,
        send_stream1,
        receive_stream1,
        send_stream2,
        receive_stream2,
    ):
        # Start server
        tg.start_soon(lambda: server._mcp_server.run(
            receive_stream1,
            send_stream2,
            server._mcp_server.create_initialization_options(),
            raise_exceptions=True
        ))

        async with client_session(server._mcp_server) as client:
            result = await client.call_tool("get_lifecycle_state", {})
            state = eval(result.content[0].text)
            assert state["current_state"] == "running"
            assert state["history"] == ["initializing", "running"]

        # Trigger shutdown
        tg.cancel_scope.cancel()

        # Verify final state
        assert tracker.get_history() == [
            "initializing",
            "running",
            "shutting_down",
            "terminated"
        ]

### Concurrent Access Testing

```python
@pytest.mark.anyio
async def test_concurrent_access():
    """Test concurrent access to lifespan state."""

    @asynccontextmanager
    async def concurrent_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        context = {
            "counter": 0,
            "lock": asyncio.Lock()
        }
        try:
            yield context
        finally:
            pass

    server = FastMCP("test", lifespan=concurrent_lifespan)

    @server.tool()
    async def increment_counter(ctx: Context) -> int:
        """Tool that safely increments counter."""
        async with ctx.request_context.lifespan_context["lock"]:
            current = ctx.request_context.lifespan_context["counter"]
            # Simulate some work
            await anyio.sleep(0.1)
            ctx.request_context.lifespan_context["counter"] = current + 1
            return ctx.request_context.lifespan_context["counter"]

    # Create memory streams for testing
    send_stream1, receive_stream1 = anyio.create_memory_object_stream(100)
    send_stream2, receive_stream2 = anyio.create_memory_object_stream(100)

    async with (
        anyio.create_task_group() as tg,
        send_stream1,
        receive_stream1,
        send_stream2,
        receive_stream2,
    ):
        # Start server
        tg.start_soon(lambda: server._mcp_server.run(
            receive_stream1,
            send_stream2,
            server._mcp_server.create_initialization_options(),
            raise_exceptions=True
        ))

        async with client_session(server._mcp_server) as client:
            # Run concurrent increments
            tasks = [
                client.call_tool("increment_counter", {})
                for _ in range(5)
            ]
            results = await asyncio.gather(*tasks)

            # Verify all increments were processed
            final_values = [int(r.content[0].text) for r in results]
            assert sorted(final_values) == [1, 2, 3, 4, 5]

        # Clean shutdown
        tg.cancel_scope.cancel()
```

## Key Points

1. The lifespan context manager is used to manage server startup and shutdown states
2. Lifespan context is accessible in tools through the `ctx.request_context.lifespan_context`
3. The lifespan context is initialized before any tool calls and cleaned up after server shutdown
4. Tools can verify the lifespan state during their execution
5. The `asynccontextmanager` decorator is required for implementing lifespan context managers

## Common Patterns

- Use a dictionary to track lifespan states
- Initialize resources in the setup phase (before `yield`)
- Clean up resources in the teardown phase (after `yield`)
- Access lifespan context in tools through the context object
- Verify lifespan states in tools for proper server lifecycle management
