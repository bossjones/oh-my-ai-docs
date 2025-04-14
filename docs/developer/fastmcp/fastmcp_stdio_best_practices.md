Okay, let's break down the best practices for creating a **STDIO-only FastMCP server**. This assumes your server communicates solely via standard input (`stdin`) and standard output (`stdout`), likely launched as a subprocess by a host application (like Claude Desktop, VS Code Copilot, etc.), and aims for efficiency.

Here's a step-by-step thinking process leading to the Do's and Don'ts:

1.  **Understand the Core Constraint: STDIO Only.** This is paramount. All communication *must* happen over `stdin` (for receiving messages) and `stdout` (for sending messages). `stderr` can be used for logging that the *host* might capture, but it's not part of the MCP protocol itself. No HTTP, no SSE, just process streams.
2.  **Understand "FastMCP":** While not a formal term in the spec, it implies using helper libraries or patterns (like Python's `FastMCP` class mentioned in the context) that simplify server creation, often using decorators or type hints for defining tools/resources, and aiming for performance.
3.  **Identify Key MCP Requirements:** Regardless of transport, the server must adhere to the MCP specification: JSON-RPC 2.0 message format, the initialization lifecycle, capability negotiation, and correct implementation of features like Tools, Resources, or Prompts.
4.  **Consider the Environment:** STDIO servers are typically local processes launched by a host. This has implications for the working directory, environment variables, and security (access to local files/resources).
5.  **Focus on Performance ("Fast"):** This means non-blocking operations for I/O, efficient resource usage, and potentially caching.
6.  **Structure as Do's and Don'ts:** This provides clear, actionable guidance.

---

Here is the list of best practices (Do's and Don'ts) for a STDIO-only FastMCP server:

**Do's:**

1.  **DO Strictly Adhere to JSON-RPC 2.0 over STDIO:**
    *   Read JSON-RPC messages line-by-line from `stdin`.
    *   Write JSON-RPC messages line-by-line to `stdout`.
    *   Ensure each message (request, response, notification) is a single, valid JSON object on one line, terminated by a newline (`\n`).
    *   Use UTF-8 encoding for all messages.

2.  **DO Implement the MCP Lifecycle Correctly:**
    *   Handle the `initialize` request first. Respond with your server's capabilities and info.
    *   Wait for the `initialized` notification from the client before sending any server-initiated requests (if applicable, though less common for simple STDIO servers).
    *   Accurately declare your server's capabilities (e.g., `tools`, `resources`, `prompts`, `logging`) in the `initialize` response. Only implement features you declare.

3.  **DO Use `stderr` for Logging:**
    *   Direct all diagnostic logging, debugging information, and non-protocol output to `stderr`. Host applications (like Claude Desktop) often capture `stderr` from MCP servers for debugging purposes.
    *   *Never* write anything to `stdout` that isn't a valid, newline-terminated JSON-RPC message.

4.  **DO Leverage Frameworks/Helpers (like FastMCP):**
    *   If using a library like Python's `mcp.server.fastmcp`, use its decorators (`@mcp.tool()`, `@mcp.resource()`, etc.) and type hints to define tools, resources, and prompts. This simplifies schema generation and request handling.
    *   Ensure your function signatures match what the framework expects for automatic argument parsing and validation based on the generated schema.

5.  **DO Implement Asynchronous Handlers for I/O:**
    *   Use `async`/`await` (or equivalent non-blocking patterns in your language) for any operation that involves waiting for external resources (network requests, file I/O, database queries, subprocess calls).
    *   This prevents the server from becoming unresponsive while waiting for slow operations.

6.  **DO Handle Errors Gracefully:**
    *   Implement robust error handling within your tool/resource/prompt handlers.
    *   Report tool execution errors correctly within the `CallToolResult` by setting `isError: true` and providing error details in the `content`. Do *not* use JSON-RPC protocol errors (like code `-32603`) for errors *within* a successful tool call flow.
    *   Use standard JSON-RPC error responses for protocol-level issues (e.g., `MethodNotFound`, `InvalidParams`, `InternalError` if something unexpected breaks *outside* a specific tool execution).
    *   Log detailed errors to `stderr` for debugging.

7.  **DO Be Mindful of the Execution Environment:**
    *   Be aware that the working directory might be unpredictable (e.g., `/` on macOS when launched by Claude Desktop). Use absolute paths or resolve paths relative to the script's location or user's home directory.
    *   Don't assume environment variables are inherited from the user's shell. If your server needs specific environment variables (like API keys), ensure they are provided when the server is configured in the host application (e.g., via the `env` key in `claude_desktop_config.json`).

8.  **DO Validate Inputs:**
    *   Rigorously validate arguments passed to tools, especially if they involve file paths, shell commands, or external API calls. Use JSON Schema validation (often handled by frameworks like FastMCP if using type hints) and perform runtime checks.

9.  **DO Handle Shutdown Signals:**
    *   Your server process might be terminated by the host (e.g., by closing `stdin` or sending `SIGTERM`). Implement cleanup logic if necessary (e.g., closing file handles, shutting down background tasks).

10. **DO Test with the MCP Inspector and Target Clients:**
    *   Use the `npx @modelcontextprotocol/inspector` tool to test your server's responses independently.
    *   Test integration with the actual client applications (Claude Desktop, VS Code, etc.) that you intend to support.

**Don'ts:**

1.  **DON'T Write Anything to `stdout` Except Valid JSON-RPC Messages:**
    *   No logs, no prompts, no debugging prints, no partial messages. This is the most common way to break STDIO communication.

2.  **DON'T Use Blocking I/O Operations in Handlers:**
    *   Avoid synchronous file reads/writes, network calls, or long computations that block the server's event loop. This will make the server unresponsive.

3.  **DON'T Rely on the Default Working Directory:**
    *   It's often not what you expect when launched by a host application.

4.  **DON'T Assume Environment Variables are Inherited:**
    *   Explicitly require and check for necessary environment variables.

5.  **DON'T Hardcode Sensitive Information:**
    *   Avoid embedding API keys, passwords, or sensitive paths directly in the code. Use environment variables or secure configuration methods.

6.  **DON'T Ignore the MCP Lifecycle:**
    *   Skipping or incorrectly implementing the `initialize` handshake will prevent the client from connecting.

7.  **DON'T Report Tool Execution Errors as Protocol Errors:**
    *   A tool failing (e.g., API returns 404) is usually *not* a protocol error. Use `CallToolResult` with `isError: true`. A protocol error (`InternalError`, `-32603`) should be reserved for unexpected crashes *within the server's MCP handling logic itself*.

8.  **DON'T Embed Newlines Within a Single JSON Message:**
    *   Ensure the entire JSON object for a single message is on one line before the terminating newline character.

9.  **DON'T Forget Capability Negotiation:**
    *   Only attempt to use features (like sending `notifications/message` for logging) if the client declared support for the corresponding capability during initialization.

10. **DON'T Make Assumptions About Client Support:**
    *   While many clients support Tools, fewer might support Resources, Prompts, or Sampling over STDIO. Check the client documentation or test directly if relying on less common features.

By following these guidelines, you can build robust, efficient, and compatible STDIO-only MCP servers.


-----------------


Okay, let's update the best practices list with relevant Python code examples, focusing on the `mcp.server.fastmcp` library and the STDIO transport constraint.

---

**Best Practices for STDIO-Only FastMCP Servers (with Code Examples)**

This list assumes you are using Python and the `mcp.server.fastmcp` library.

**Do's:**

1.  **DO Strictly Adhere to JSON-RPC 2.0 over STDIO:**
    *   The `FastMCP` framework and the underlying `mcp` library handle this when using `transport='stdio'`. You focus on writing handlers.
    *   Ensure your server script's *only* output to `stdout` is generated by the `mcp.run()` call.

2.  **DO Implement the MCP Lifecycle Correctly:**
    *   `FastMCP` handles the `initialize` handshake automatically based on the tools/resources/prompts you define.
    *   Declare capabilities implicitly by defining features (e.g., defining tools enables the `tools` capability).

3.  **DO Use `stderr` for Logging:**
    *   Direct all diagnostic logging, debugging information, and non-protocol output to `stderr`. Host applications often capture this.

    ```python
    import sys
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("my_stdio_server")

    @mcp.tool()
    async def my_tool(param: str) -> str:
        # Correct: Log to stderr
        print(f"Received request for my_tool with param: {param}", file=sys.stderr)

        # Incorrect: DO NOT print to stdout directly
        # print("Processing request...") # <--- This would break the protocol!

        # ... tool logic ...
        result = f"Processed: {param}"
        print(f"Returning result: {result}", file=sys.stderr) # Log result to stderr
        return result # Return value is handled by FastMCP for stdout protocol message

    # ... rest of server ...
    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

4.  **DO Leverage `FastMCP` Decorators and Type Hints:**
    *   Use `@mcp.tool()`, `@mcp.resource()`, etc., with Python type hints and docstrings. `FastMCP` uses these to automatically generate the necessary MCP schemas and handle argument parsing/validation.

    ```python
    from mcp.server.fastmcp import FastMCP
    import httpx # For async example

    mcp = FastMCP("api_wrapper")

    @mcp.tool(
        annotations={ # Optional annotations for clients
            "title": "Fetch Web Content",
            "readOnlyHint": True,
            "openWorldHint": True
        }
    )
    async def fetch_url(url: str) -> str:
        """Fetches the content of a given URL.

        Args:
            url: The URL to fetch (must start with http:// or https://).
        """
        print(f"Tool 'fetch_url' called with URL: {url}", file=sys.stderr)
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL scheme. Must be http or https.") # FastMCP handles this exception

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status() # Raises HTTPError for bad responses
            return response.text[:500] # Return first 500 chars

    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

5.  **DO Implement Asynchronous Handlers for I/O:**
    *   Use `async def` for your tool/resource handlers if they perform any network, file, or subprocess operations. Use `await` for these operations.

    ```python
    # See the `fetch_url` example above which uses `async def` and `await client.get(url)`
    ```

6.  **DO Handle Errors Gracefully:**
    *   For expected errors within a tool's logic (e.g., invalid input, API error), raise specific exceptions. `FastMCP` will typically catch these and format them as a `CallToolResult` with `isError: true`.
    *   Avoid letting unexpected errors crash the entire server; wrap critical sections if necessary, though `FastMCP` provides some top-level catching.

    ```python
    import sys
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("error_handler_example")

    class ToolSpecificError(Exception):
        """Custom exception for clear error reporting."""
        pass

    @mcp.tool()
    async def process_data(data_id: int) -> str:
        """Processes data based on ID."""
        print(f"Processing data_id: {data_id}", file=sys.stderr)
        try:
            if data_id < 0:
                # Raise an exception for invalid input - FastMCP catches this
                raise ValueError("Data ID cannot be negative.")
            elif data_id == 42:
                 # Raise a custom exception for specific business logic errors
                 raise ToolSpecificError("Processing failed for special ID 42.")

            # Simulate processing
            await asyncio.sleep(0.1) # Example async operation
            if data_id > 1000:
                 # Simulate an unexpected internal issue
                 # This might crash the handler if not caught,
                 # but FastMCP should still report an InternalError (-32603)
                 # It's better to catch where possible.
                 result = 1 / 0

            return f"Successfully processed data ID: {data_id}"

        except ToolSpecificError as e:
             print(f"Caught ToolSpecificError: {e}", file=sys.stderr)
             # Re-raise for FastMCP to handle as isError=True
             raise e
        except ValueError as e:
             print(f"Caught ValueError: {e}", file=sys.stderr)
             # Re-raise for FastMCP to handle as isError=True
             raise e
        except Exception as e:
             # Catch unexpected errors within the handler
             print(f"Caught unexpected error: {e}", file=sys.stderr)
             # Raise a generic exception that FastMCP will report as InternalError
             # or potentially isError=True depending on its configuration.
             raise RuntimeError(f"An internal error occurred processing ID {data_id}")

    # import asyncio
    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```
    *(Note: The exact way FastMCP converts exceptions to `isError=True` vs. protocol errors might depend on the framework version, but raising exceptions is the standard way to signal failure.)*

7.  **DO Be Mindful of the Execution Environment:**
    *   Use absolute paths or resolve paths carefully.

    ```python
    import os
    import sys
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("path_example")

    @mcp.tool()
    async def read_user_config() -> str:
        """Reads a config file from the user's home directory."""
        home_dir = os.path.expanduser("~")
        config_path = os.path.join(home_dir, ".my_mcp_app", "config.txt")
        print(f"Attempting to read config from: {config_path}", file=sys.stderr)
        try:
            # Use async file I/O if available (e.g., with aiofiles)
            # For simplicity, using sync here, but convert to async for real servers
            if not os.path.exists(config_path):
                 raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading config: {e}", file=sys.stderr)
            raise # Let FastMCP handle reporting the error

    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

8.  **DO Validate Inputs:**
    *   Use type hints for basic validation via `FastMCP`. Add explicit checks for complex constraints.

    ```python
    # See the `fetch_url` example above which checks the URL scheme.
    # See the `process_data` example which checks if data_id is negative.
    ```

9.  **DO Handle Shutdown Signals:**
    *   Python's `asyncio` loop often handles standard signals like `SIGTERM` gracefully, but if you start background tasks or hold resources, use `try...finally` or `asyncio.shield` / signal handlers if needed. (Simple `FastMCP` servers often don't require complex handling here).

10. **DO Test with the MCP Inspector and Target Clients:**
    *   Run `npx @modelcontextprotocol/inspector python your_server_script.py` to test.

**Don'ts:**

1.  **DON'T Write Anything to `stdout` Except Valid JSON-RPC Messages:**
    *   Avoid `print("debug message")`. Use `print("debug message", file=sys.stderr)`.

2.  **DON'T Use Blocking I/O Operations in `async` Handlers:**
    *   Don't use `requests.get()` in an `async def` handler; use `httpx.AsyncClient().get()` or `aiohttp`.
    *   Don't use standard `open()` for large files if it blocks the event loop; use `aiofiles` or run it in a thread pool executor.

3.  **DON'T Rely on the Default Working Directory:**
    *   Use `os.path.expanduser("~")`, `os.path.dirname(__file__)`, or absolute paths passed via configuration/arguments.

4.  **DON'T Assume Environment Variables are Inherited:**
    *   Check for needed variables using `os.environ.get()` and raise an error or log to `stderr` if missing.

    ```python
    import os
    import sys
    # ...
    API_KEY = os.environ.get("MY_SERVICE_API_KEY")
    if not API_KEY:
        print("ERROR: MY_SERVICE_API_KEY environment variable not set.", file=sys.stderr)
        # Option 1: Raise an exception to prevent the tool from working
        # raise ValueError("API Key is missing")
        # Option 2: The tool could return an error message directly
        # return "Error: API Key is missing" # If the tool returns str
    ```

5.  **DON'T Hardcode Sensitive Information:**
    *   Load keys from environment variables (see above) or secure configuration files outside the codebase.

6.  **DON'T Ignore the MCP Lifecycle:**
    *   (Handled by `FastMCP`, just ensure `mcp.run()` is called correctly).

7.  **DON'T Report Tool Execution Errors as Protocol Errors:**
    *   If `fetch_url` gets a 404 from the target website, raise an `httpx.HTTPStatusError` or a custom exception. `FastMCP` should report this as `isError: true` in the `CallToolResult`. Don't manually construct a JSON-RPC error object with code `-32603`.

8.  **DON'T Embed Newlines Within a Single JSON Message:**
    *   (Handled by `FastMCP` and the `json` library).

9.  **DON'T Forget Capability Negotiation:**
    *   (Handled implicitly by `FastMCP` based on defined features).

10. **DON'T Make Assumptions About Client Support:**
    *   Stick to widely supported features like `tools` for STDIO servers unless you know the specific client supports more (like `resources` or `prompts`).

----------------------------

Okay, let's update the best practices list with relevant Python code examples, focusing on the `mcp.server.fastmcp` library and the STDIO transport constraint.

---

**Best Practices for STDIO-Only FastMCP Servers (with Code Examples)**

This list assumes you are using Python and the `mcp.server.fastmcp` library.

**Do's:**

1.  **DO Strictly Adhere to JSON-RPC 2.0 over STDIO:**
    *   The `FastMCP` framework and the underlying `mcp` library handle this when using `transport='stdio'`. You focus on writing handlers.
    *   Ensure your server script's *only* output to `stdout` is generated by the `mcp.run()` call.

2.  **DO Implement the MCP Lifecycle Correctly:**
    *   `FastMCP` handles the `initialize` handshake automatically based on the tools/resources/prompts you define.
    *   Declare capabilities implicitly by defining features (e.g., defining tools enables the `tools` capability).

3.  **DO Use `stderr` for Logging:**
    *   Direct all diagnostic logging, debugging information, and non-protocol output to `stderr`. Host applications often capture this.

    ```python
    import sys
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("my_stdio_server")

    @mcp.tool()
    async def my_tool(param: str) -> str:
        # Correct: Log to stderr
        print(f"Received request for my_tool with param: {param}", file=sys.stderr)

        # Incorrect: DO NOT print to stdout directly
        # print("Processing request...") # <--- This would break the protocol!

        # ... tool logic ...
        result = f"Processed: {param}"
        print(f"Returning result: {result}", file=sys.stderr) # Log result to stderr
        return result # Return value is handled by FastMCP for stdout protocol message

    # ... rest of server ...
    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

4.  **DO Leverage `FastMCP` Decorators and Type Hints:**
    *   Use `@mcp.tool()`, `@mcp.resource()`, etc., with Python type hints and docstrings. `FastMCP` uses these to automatically generate the necessary MCP schemas and handle argument parsing/validation.

    ```python
    from mcp.server.fastmcp import FastMCP
    import httpx # For async example

    mcp = FastMCP("api_wrapper")

    @mcp.tool(
        annotations={ # Optional annotations for clients
            "title": "Fetch Web Content",
            "readOnlyHint": True,
            "openWorldHint": True
        }
    )
    async def fetch_url(url: str) -> str:
        """Fetches the content of a given URL.

        Args:
            url: The URL to fetch (must start with http:// or https://).
        """
        print(f"Tool 'fetch_url' called with URL: {url}", file=sys.stderr)
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL scheme. Must be http or https.") # FastMCP handles this exception

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status() # Raises HTTPError for bad responses
            return response.text[:500] # Return first 500 chars

    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

5.  **DO Implement Asynchronous Handlers for I/O:**
    *   Use `async def` for your tool/resource handlers if they perform any network, file, or subprocess operations. Use `await` for these operations.

    ```python
    # See the `fetch_url` example above which uses `async def` and `await client.get(url)`
    ```

6.  **DO Handle Errors Gracefully:**
    *   For expected errors within a tool's logic (e.g., invalid input, API error), raise specific exceptions. `FastMCP` will typically catch these and format them as a `CallToolResult` with `isError: true`.
    *   Avoid letting unexpected errors crash the entire server; wrap critical sections if necessary, though `FastMCP` provides some top-level catching.

    ```python
    import sys
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("error_handler_example")

    class ToolSpecificError(Exception):
        """Custom exception for clear error reporting."""
        pass

    @mcp.tool()
    async def process_data(data_id: int) -> str:
        """Processes data based on ID."""
        print(f"Processing data_id: {data_id}", file=sys.stderr)
        try:
            if data_id < 0:
                # Raise an exception for invalid input - FastMCP catches this
                raise ValueError("Data ID cannot be negative.")
            elif data_id == 42:
                 # Raise a custom exception for specific business logic errors
                 raise ToolSpecificError("Processing failed for special ID 42.")

            # Simulate processing
            await asyncio.sleep(0.1) # Example async operation
            if data_id > 1000:
                 # Simulate an unexpected internal issue
                 # This might crash the handler if not caught,
                 # but FastMCP should still report an InternalError (-32603)
                 # It's better to catch where possible.
                 result = 1 / 0

            return f"Successfully processed data ID: {data_id}"

        except ToolSpecificError as e:
             print(f"Caught ToolSpecificError: {e}", file=sys.stderr)
             # Re-raise for FastMCP to handle as isError=True
             raise e
        except ValueError as e:
             print(f"Caught ValueError: {e}", file=sys.stderr)
             # Re-raise for FastMCP to handle as isError=True
             raise e
        except Exception as e:
             # Catch unexpected errors within the handler
             print(f"Caught unexpected error: {e}", file=sys.stderr)
             # Raise a generic exception that FastMCP will report as InternalError
             # or potentially isError=True depending on its configuration.
             raise RuntimeError(f"An internal error occurred processing ID {data_id}")

    # import asyncio
    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```
    *(Note: The exact way FastMCP converts exceptions to `isError=True` vs. protocol errors might depend on the framework version, but raising exceptions is the standard way to signal failure.)*

7.  **DO Be Mindful of the Execution Environment:**
    *   Use absolute paths or resolve paths carefully.

    ```python
    import os
    import sys
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("path_example")

    @mcp.tool()
    async def read_user_config() -> str:
        """Reads a config file from the user's home directory."""
        home_dir = os.path.expanduser("~")
        config_path = os.path.join(home_dir, ".my_mcp_app", "config.txt")
        print(f"Attempting to read config from: {config_path}", file=sys.stderr)
        try:
            # Use async file I/O if available (e.g., with aiofiles)
            # For simplicity, using sync here, but convert to async for real servers
            if not os.path.exists(config_path):
                 raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading config: {e}", file=sys.stderr)
            raise # Let FastMCP handle reporting the error

    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

8.  **DO Validate Inputs:**
    *   Use type hints for basic validation via `FastMCP`. Add explicit checks for complex constraints.

    ```python
    # See the `fetch_url` example above which checks the URL scheme.
    # See the `process_data` example which checks if data_id is negative.
    ```

9.  **DO Handle Shutdown Signals:**
    *   Python's `asyncio` loop often handles standard signals like `SIGTERM` gracefully, but if you start background tasks or hold resources, use `try...finally` or `asyncio.shield` / signal handlers if needed. (Simple `FastMCP` servers often don't require complex handling here).

10. **DO Test with the MCP Inspector and Target Clients:**
    *   Run `npx @modelcontextprotocol/inspector python your_server_script.py` to test.

**Don'ts:**

1.  **DON'T Write Anything to `stdout` Except Valid JSON-RPC Messages:**
    *   Avoid `print("debug message")`. Use `print("debug message", file=sys.stderr)`.

2.  **DON'T Use Blocking I/O Operations in `async` Handlers:**
    *   Don't use `requests.get()` in an `async def` handler; use `httpx.AsyncClient().get()` or `aiohttp`.
    *   Don't use standard `open()` for large files if it blocks the event loop; use `aiofiles` or run it in a thread pool executor.

3.  **DON'T Rely on the Default Working Directory:**
    *   Use `os.path.expanduser("~")`, `os.path.dirname(__file__)`, or absolute paths passed via configuration/arguments.

4.  **DON'T Assume Environment Variables are Inherited:**
    *   Check for needed variables using `os.environ.get()` and raise an error or log to `stderr` if missing.

    ```python
    import os
    import sys
    # ...
    API_KEY = os.environ.get("MY_SERVICE_API_KEY")
    if not API_KEY:
        print("ERROR: MY_SERVICE_API_KEY environment variable not set.", file=sys.stderr)
        # Option 1: Raise an exception to prevent the tool from working
        # raise ValueError("API Key is missing")
        # Option 2: The tool could return an error message directly
        # return "Error: API Key is missing" # If the tool returns str
    ```

5.  **DON'T Hardcode Sensitive Information:**
    *   Load keys from environment variables (see above) or secure configuration files outside the codebase.

6.  **DON'T Ignore the MCP Lifecycle:**
    *   (Handled by `FastMCP`, just ensure `mcp.run()` is called correctly).

7.  **DON'T Report Tool Execution Errors as Protocol Errors:**
    *   If `fetch_url` gets a 404 from the target website, raise an `httpx.HTTPStatusError` or a custom exception. `FastMCP` should report this as `isError: true` in the `CallToolResult`. Don't manually construct a JSON-RPC error object with code `-32603`.

8.  **DON'T Embed Newlines Within a Single JSON Message:**
    *   (Handled by `FastMCP` and the `json` library).

9.  **DON'T Forget Capability Negotiation:**
    *   (Handled implicitly by `FastMCP` based on defined features).

10. **DON'T Make Assumptions About Client Support:**
    *   Stick to widely supported features like `tools` for STDIO servers unless you know the specific client supports more (like `resources` or `prompts`).

--------------------

You are absolutely correct! That's a crucial point I glossed over in the examples for simplicity, but it's vital for a truly *asynchronous* and *responsive* FastMCP server.

**Yes, you absolutely SHOULD replace synchronous file I/O calls like `open()`, `read()`, `write()` with asynchronous alternatives like `aiofiles` when inside an `async def` handler.**

Here's why and how:

1.  **Blocking the Event Loop:** Standard Python file operations are *blocking*. When you call `open()` or `f.read()` inside an `async def` function running in an `asyncio` event loop, that call will block the *entire loop* until the disk operation completes. This means your server cannot handle any other incoming requests or perform any other concurrent tasks during that time, defeating the purpose of using `asyncio`.
2.  **Responsiveness:** For a "FastMCP" server, responsiveness is key. If reading a large file or writing to a slow disk blocks the server, it won't be fast or efficient.
3.  **`aiofiles` Solution:** The `aiofiles` library provides asynchronous versions of file operations that integrate correctly with `asyncio`. They typically use a thread pool behind the scenes to perform the blocking disk I/O without blocking the main event loop.
4.  **Alternative (`run_in_executor`):** You *could* also use `asyncio.get_running_loop().run_in_executor(None, blocking_function, args...)` to run synchronous code in a thread pool, but `aiofiles` is generally cleaner and more idiomatic for file I/O.

**Let's update the relevant examples:**

First, you'd need to install `aiofiles`:

```bash
uv add aiofiles
```

**Updated Example: `read_user_config` (from Do #7)**

```python
import os
import sys
from mcp.server.fastmcp import FastMCP
import aiofiles # Import aiofiles
import asyncio # Often needed with async file ops

mcp = FastMCP("path_example_async")

@mcp.tool()
async def read_user_config() -> str:
    """Reads a config file from the user's home directory asynchronously."""
    home_dir = os.path.expanduser("~")
    config_path = os.path.join(home_dir, ".my_mcp_app", "config.txt")
    print(f"Attempting to read config from: {config_path}", file=sys.stderr)
    try:
        # Use async file I/O with aiofiles
        if not await aiofiles.os.path.exists(config_path):
             raise FileNotFoundError(f"Config file not found at {config_path}")
        if not await aiofiles.os.path.isfile(config_path):
             raise IsADirectoryError(f"Path is not a file: {config_path}") # Or similar error

        async with aiofiles.open(config_path, mode='r', encoding='utf-8') as f:
            content = await f.read() # Await the read operation
            return content
    except Exception as e:
        print(f"Error reading config: {e}", file=sys.stderr)
        raise # Let FastMCP handle reporting the error

# if __name__ == "__main__":
#     mcp.run(transport='stdio')
```

**Updated Example: `get_file_as_resource` (from Scenario 3)**

```python
import os
import sys
from mcp.server.fastmcp import FastMCP
from mcp.types import EmbeddedResource, Resource # Import types
import aiofiles # Import aiofiles
import asyncio

mcp = FastMCP("file_server_async")

@mcp.tool()
async def get_file_as_resource(path: str) -> list[EmbeddedResource]:
    """Reads a file asynchronously and returns its content as an embedded resource."""
    print(f"Attempting to read file: {path}", file=sys.stderr)
    # WARNING: Real server needs strict path validation!
    if not await aiofiles.os.path.exists(path):
         raise FileNotFoundError(f"File not found: {path}")
    if not await aiofiles.os.path.isfile(path):
         raise IsADirectoryError(f"Path is not a file: {path}")

    try:
        # Use async file I/O
        async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
            content = await f.read() # Await the read

        # Create the Resource object first
        resource_obj = Resource(
            uri=f"file://{os.path.abspath(path)}", # Construct a URI
            name=os.path.basename(path),
            mimeType="text/plain", # Or detect MIME type
            text=content # Embed the content directly
        )
        # Return a list containing the EmbeddedResource
        return [EmbeddedResource(type="resource", resource=resource_obj)]
    except Exception as e:
        print(f"Error reading file {path}: {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to read file {path}")


# if __name__ == "__main__":
#     mcp.run(transport='stdio')
```

Thanks for pointing out that important detail! Using asynchronous I/O correctly is fundamental to building performant `asyncio`-based applications like an MCP server.

-------------------------------------

Okay, I will integrate the provided information, focusing on the corrected code examples using `aiofiles` and incorporating the refinements mentioned in the review.

Here is the updated `fastmcp_stdio_best_practices.md`:

```markdown
# Best Practices for STDIO-Only FastMCP Servers (Python)

This document outlines best practices for creating **STDIO-only FastMCP servers** using Python and the `mcp.server.fastmcp` library. This assumes your server communicates solely via standard input (`stdin`) and standard output (`stdout`), is typically launched as a subprocess by a host application (like Claude Desktop, VS Code Copilot, etc.), and aims for efficiency and robustness.

## Understanding the Constraints

1.  **STDIO Only:** All protocol communication *must* happen over `stdin` (for receiving messages) and `stdout` (for sending messages). `stderr` should be used for logging and diagnostics that the *host* might capture, but it's not part of the MCP protocol itself. No HTTP, no SSE, just process streams.
2.  **"FastMCP":** This implies using helper libraries like Python's `mcp.server.fastmcp` that simplify server creation, often using decorators and type hints for defining tools/resources, and aiming for performance through asynchronous operations.
3.  **MCP Requirements:** Adherence to the MCP specification is mandatory: JSON-RPC 2.0 message format, the initialization lifecycle (`initialize`/`initialized`), capability negotiation, and correct implementation of features like Tools, Resources, or Prompts.
4.  **Execution Environment:** STDIO servers are local processes. Be mindful of the working directory, environment variables, and security implications (access to local files/resources).
5.  **Performance ("Fast"):** Requires non-blocking operations (especially for I/O), efficient resource usage, and potentially caching.

---

## Best Practices: Do's and Don'ts

Here is the list of best practices with Python code examples using `mcp.server.fastmcp`.

**(Note:** You'll likely need `aiofiles` and `httpx` for asynchronous operations: `pip install aiofiles httpx` or `uv add aiofiles httpx`)

**Do's:**

1.  **DO Strictly Adhere to JSON-RPC 2.0 over STDIO:**
    *   The `FastMCP` framework handles JSON-RPC formatting and STDIO transport when using `mcp.run(transport='stdio')`.
    *   Your primary responsibility is to ensure your script writes *nothing* to `stdout` outside of what `FastMCP` generates. Use `stderr` for all other output.

2.  **DO Implement the MCP Lifecycle Correctly:**
    *   `FastMCP` handles the `initialize` handshake automatically based on the tools/resources/prompts you define using decorators.
    *   Capabilities are declared implicitly by defining features (e.g., defining tools with `@mcp.tool()` enables the `tools` capability).

3.  **DO Use `stderr` for All Logging and Diagnostics:**
    *   Direct all logging, debugging information, progress updates, and non-protocol output to `stderr`. Host applications often capture this stream.

    ```python
    import sys
    from mcp.server.fastmcp import FastMCP
    import asyncio # For async example

    mcp = FastMCP("my_stdio_server")

    @mcp.tool()
    async def my_tool(param: str) -> str:
        # Correct: Log to stderr
        print(f"INFO: Received request for my_tool with param: {param}", file=sys.stderr)

        # Incorrect: DO NOT print to stdout directly
        # print("Processing request...") # <--- This would break the protocol!

        try:
            # Simulate work
            await asyncio.sleep(0.1)
            result = f"Processed: {param}"
            print(f"DEBUG: Returning result: {result}", file=sys.stderr) # Log result to stderr
            return result # Return value is handled by FastMCP for stdout protocol message
        except Exception as e:
            print(f"ERROR: Exception in my_tool: {e}", file=sys.stderr)
            raise # Re-raise for FastMCP to handle

    # Example of how to run the server (typically at the end of the script)
    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

4.  **DO Leverage `FastMCP` Decorators and Type Hints:**
    *   Use `@mcp.tool()`, `@mcp.resource()`, etc., with Python type hints and docstrings. `FastMCP` uses these to automatically generate MCP schemas, validate inputs, and parse arguments.

    ```python
    import sys
    from mcp.server.fastmcp import FastMCP
    import httpx # For async HTTP requests

    mcp = FastMCP("api_wrapper")

    @mcp.tool(
        annotations={ # Optional annotations for clients
            "title": "Fetch Web Content",
            "readOnlyHint": True,
            "openWorldHint": True
        }
    )
    async def fetch_url(url: str) -> str:
        """Fetches the first 500 characters of a given URL's content.

        Args:
            url: The URL to fetch (must start with http:// or https://).
        """
        print(f"INFO: Tool 'fetch_url' called with URL: {url}", file=sys.stderr)
        if not url.startswith(("http://", "https://")):
            # Raising ValueError is caught by FastMCP and reported as tool error
            raise ValueError("Invalid URL scheme. Must be http or https.")

        try:
            # Use async with for proper resource management
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=15.0, follow_redirects=True)
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
                content = response.text
                print(f"DEBUG: Fetched {len(content)} chars from {url}", file=sys.stderr)
                return content[:500] # Return first 500 chars
        except httpx.HTTPStatusError as e:
            print(f"ERROR: HTTP error fetching {url}: {e}", file=sys.stderr)
            # Re-raise the specific error for clear reporting
            raise ToolSpecificError(f"Failed to fetch URL: {e.response.status_code} {e.response.reason_phrase}") from e
        except httpx.RequestError as e:
            print(f"ERROR: Network error fetching {url}: {e}", file=sys.stderr)
            raise ToolSpecificError(f"Network error accessing URL: {e}") from e
        except Exception as e:
            print(f"ERROR: Unexpected error in fetch_url: {e}", file=sys.stderr)
            raise # Let FastMCP handle unexpected errors

    class ToolSpecificError(Exception):
        """Custom exception for clearer tool-level errors."""
        pass

    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

5.  **DO Implement Asynchronous Handlers for I/O and CPU-Bound Tasks:**
    *   Use `async def` for handlers performing network, file, or subprocess operations. Use `await` for these calls.
    *   **Crucially, use asynchronous libraries for I/O:**
        *   Network: `httpx.AsyncClient`, `aiohttp`
        *   File I/O: `aiofiles` (see example in Do #7)
        *   Subprocesses: `asyncio.create_subprocess_exec`
    *   For **CPU-bound tasks** that would block the event loop, run them in a separate thread or process pool using `asyncio.run_in_executor`:

    ```python
    import asyncio
    import time

    def cpu_intensive_task(data):
        print("Starting CPU-bound task...", file=sys.stderr)
        # Simulate heavy computation
        time.sleep(2) # Blocking sleep - BAD in async def directly!
        print("Finished CPU-bound task.", file=sys.stderr)
        return f"Computed result for {data}"

    @mcp.tool()
    async def run_computation(data: str) -> str:
        print(f"INFO: Received request for computation on: {data}", file=sys.stderr)
        loop = asyncio.get_running_loop()
        # Run the blocking function in the default thread pool executor
        result = await loop.run_in_executor(None, cpu_intensive_task, data)
        print(f"DEBUG: Computation result: {result}", file=sys.stderr)
        return result
    ```

6.  **DO Handle Errors Gracefully within Tool Logic:**
    *   Raise specific Python exceptions for expected errors (invalid input, file not found, API errors). `FastMCP` catches these and typically translates them into a `CallToolResult` with `isError: true` and the exception message as content.
    *   Use custom exception classes (`ToolSpecificError` in the `fetch_url` example) for better clarity.
    *   Do *not* manually create JSON-RPC protocol error responses (like `-32603 Internal error`) for failures *within* your tool's logic. Reserve protocol errors for issues with the MCP communication itself (handled by the framework).
    *   Log detailed error information to `stderr` before raising the exception.

    ```python
    import sys
    from mcp.server.fastmcp import FastMCP
    import asyncio

    mcp = FastMCP("error_handler_example")

    class ToolProcessingError(Exception):
        """Custom exception for processing errors."""
        pass

    @mcp.tool()
    async def process_data(data_id: int) -> str:
        """Processes data based on ID, demonstrating error handling."""
        print(f"INFO: Processing data_id: {data_id}", file=sys.stderr)
        try:
            if data_id < 0:
                print(f"WARN: Invalid input: Data ID {data_id} is negative.", file=sys.stderr)
                raise ValueError("Data ID cannot be negative.") # Caught by FastMCP -> isError: true

            elif data_id == 42:
                 print(f"ERROR: Specific processing failure for ID 42.", file=sys.stderr)
                 raise ToolProcessingError("Processing failed for special ID 42.") # Caught -> isError: true

            # Simulate async work
            await asyncio.sleep(0.1)

            if data_id > 1000:
                 # Simulate an unexpected internal issue
                 print(f"CRITICAL: Unexpected division by zero for ID {data_id}!", file=sys.stderr)
                 # This will likely crash the handler. FastMCP should catch this
                 # and report a generic InternalError (-32603) or similar.
                 # It's better to have broad try/excepts if possible.
                 result = 1 / 0

            return f"Successfully processed data ID: {data_id}"

        except (ValueError, ToolProcessingError) as e:
             # Log expected errors and re-raise for FastMCP to format
             print(f"DEBUG: Raising expected error: {e}", file=sys.stderr)
             raise e
        except Exception as e:
             # Catch unexpected errors within the handler
             print(f"ERROR: Caught unexpected error during processing: {e}", file=sys.stderr)
             # Raise a generic exception; FastMCP might report this as isError: true
             # or potentially a protocol-level InternalError depending on configuration.
             raise RuntimeError(f"An internal error occurred processing ID {data_id}") from e

    # if __name__ == "__main__":
    #     mcp.run(transport='stdio')
    ```

7.  **DO Be Mindful of the Execution Environment (Paths, Permissions):**
    *   The working directory is often unpredictable (e.g., `/` on macOS when launched by Claude Desktop).
    *   Use absolute paths or resolve paths relative to the script's location (`os.path.dirname(__file__)`) or the user's home directory (`os.path.expanduser("~")`).
    *   Assume minimal permissions. Ensure the server process has read/write access if needed.
    *   **Use `aiofiles` for asynchronous file operations.**

    ```python
    import os
    import sys
    from mcp.server.fastmcp import FastMCP
    import aiofiles # Use async file I/O
    import aiofiles.os # For async path checks
    import asyncio

    mcp = FastMCP("path_example_async")

    @mcp.tool()
    async def read_user_config(filename: str = "config.txt") -> str:
        """Reads a config file from ~/.my_mcp_app asynchronously."""
        # Basic validation to prevent path traversal - NEEDS MORE ROBUST VALIDATION IN REAL APPS
        if ".." in filename or "/" in filename or "\\" in filename:
             raise ValueError("Invalid filename.")

        home_dir = os.path.expanduser("~")
        config_dir = os.path.join(home_dir, ".my_mcp_app")
        config_path = os.path.join(config_dir, filename)

        print(f"INFO: Attempting to read config from: {config_path}", file=sys.stderr)
        try:
            # Use async file/path operations
            if not await aiofiles.os.path.exists(config_path):
                 print(f"WARN: Config file not found at {config_path}", file=sys.stderr)
                 raise FileNotFoundError(f"Config file not found: {config_path}")
            if not await aiofiles.os.path.isfile(config_path):
                 print(f"ERROR: Path is not a file: {config_path}", file=sys.stderr)
                 raise IsADirectoryError(f"Path is not a file: {config_path}")

            # Use async with for file handling
            async with aiofiles.open(config_path, mode='r', encoding='utf-8') as f:
                content = await f.read() # Await the read operation
                print(f"DEBUG: Read {len(content)} bytes from {config_path}", file=sys.stderr)
                return content
        except FileNotFoundError:
            # Re-raise specific expected errors
            raise
        except Exception as e:
            print(f"ERROR: Error reading config {config_path}: {e}", file=sys.stderr)
            raise RuntimeError(f"Failed to read configuration file.") from e # Generic error

    # if __name__ == "__main__":
    #     # Ensure the config directory exists for testing
    #     # home_dir = os.path.expanduser("~")
    #     # config_dir = os.path.join(home_dir, ".my_mcp_app")
    #     # os.makedirs(config_dir, exist_ok=True)
    #     # with open(os.path.join(config_dir, "config.txt"), "w") as f:
    #     #     f.write("Default config content")
    #     mcp.run(transport='stdio')
    ```

8.  **DO Validate Inputs Rigorously:**
    *   Use type hints for basic validation via `FastMCP`.
    *   Add explicit runtime checks for constraints not covered by types (e.g., string patterns, number ranges, valid enum values).
    *   **CRITICAL:** Be extremely careful validating inputs used in file paths, shell commands, database queries, or external API calls to prevent security vulnerabilities (e.g., path traversal, injection attacks). Sanitize or reject invalid inputs early. (See basic check in `read_user_config` example).

9.  **DO Handle Shutdown Signals Gracefully (If Necessary):**
    *   Python's `asyncio` loop handles `SIGINT` and `SIGTERM` reasonably well for simple servers.
    *   If your server manages external resources (e.g., background tasks, persistent connections, temporary files), implement cleanup logic using `try...finally` blocks or `asyncio` signal handlers (`loop.add_signal_handler`) to ensure resources are released properly when the host terminates the server. `async with` helps greatly with resource management within handlers.

10. **DO Test with the MCP Inspector and Target Clients:**
    *   Use the official MCP Inspector tool to verify your server's protocol adherence independently:
        ```bash
        npx @modelcontextprotocol/inspector python your_server_script.py
        ```
    *   Test integration with the actual client applications (Claude Desktop, VS Code, etc.) you intend to support, as client capabilities and behaviors can vary.

11. **DO Manage Resources Properly:**
    *   Use `async with` for resources like file handles (`aiofiles.open`) and network clients (`httpx.AsyncClient`) to ensure they are automatically closed/released, even if errors occur.

**Don'ts:**

1.  **DON'T Write Anything to `stdout` Except Valid JSON-RPC Messages:**
    *   This is the most common mistake. No logs, no prompts, no debug prints, no partial messages, no non-JSON text. Use `print(..., file=sys.stderr)` exclusively for non-protocol output.

2.  **DON'T Use Blocking I/O Operations in `async` Handlers:**
    *   Avoid standard `open()`, `read()`, `write()`, `time.sleep()`, `requests.get()`, or synchronous database calls directly within an `async def` handler. Use asynchronous alternatives (`aiofiles`, `asyncio.sleep`, `httpx`, `asyncpg`/`aiomysql`, etc.) or `run_in_executor` for blocking code.

3.  **DON'T Rely on the Default Working Directory:**
    *   It's unreliable when launched by host applications. Construct absolute paths or paths relative to known locations (home directory, script directory).

4.  **DON'T Assume Environment Variables are Inherited:**
    *   Host applications might not pass the user's full shell environment. Explicitly check for required environment variables using `os.environ.get()` and handle missing variables gracefully (log an error to `stderr`, raise an exception).

    ```python
    import os
    import sys

    # At the start of your script or tool function:
    API_KEY = os.environ.get("MY_SERVICE_API_KEY")
    if not API_KEY:
        print("ERROR: Required environment variable MY_SERVICE_API_KEY is not set.", file=sys.stderr)
        # Depending on the server's design, you might:
        # 1. Exit the server immediately:
        # sys.exit("Required environment variable MY_SERVICE_API_KEY is missing.")
        # 2. Raise an exception within tools that need it:
        # raise EnvironmentError("API Key is missing, cannot perform operation.")
        # 3. Have tools return an error state if the key is missing.
    ```

5.  **DON'T Hardcode Sensitive Information:**
    *   Avoid embedding API keys, passwords, or sensitive paths directly in the source code. Use environment variables, secure configuration files (read asynchronously!), or secrets management systems.

6.  **DON'T Ignore the MCP Lifecycle:**
    *   (Mostly handled by `FastMCP`, but ensure `mcp.run()` is called correctly and your tool definitions are sound).

7.  **DON'T Report Tool Execution Errors as Protocol Errors:**
    *   A tool failing due to invalid input, a network timeout, or an external API returning an error is *not* typically an MCP protocol error. Raise Python exceptions within the tool handler. `FastMCP` will report these using `CallToolResult` with `isError: true`. Reserve JSON-RPC errors (`-32603`, `-32602`, etc.) for fundamental issues in request processing or server state, usually handled by the framework itself.

8.  **DON'T Embed Unescaped Newlines Within a Single JSON Message:**
    *   (Handled automatically by `FastMCP` and the `json` library when serializing strings).

9.  **DON'T Forget Capability Negotiation (If Applicable):**
    *   While `FastMCP` handles declaring capabilities based on defined features, if your server *initiates* communication (less common for simple STDIO tools), ensure the client declared support for the relevant capability (e.g., `workspace/didChangeWatchedFiles` notification) during its `initialize` request.

10. **DON'T Make Assumptions About Client Feature Support:**
    *   While `tools` are widely supported, features like `resources`, `prompts`, or advanced capabilities might not be implemented by all clients, especially over STDIO. Verify client support if using features beyond basic tools.

By following these guidelines, you can build robust, efficient, and compatible STDIO-only MCP servers using Python and `FastMCP`.
```
