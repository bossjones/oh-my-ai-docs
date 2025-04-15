#!/usr/bin/env python3
"""Configure global test fixtures for the project."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import pytest
import anyio
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP


@pytest.fixture
def anyio_backend():
    """Specify the async backend for pytest-anyio.

    This fixture is required by pytest-anyio and specifies
    that we want to use asyncio as the backend.
    """
    return "asyncio"


@pytest.fixture
async def client_session(mcp_server_instance: FastMCP) -> AsyncGenerator[ClientSession, None]:
    """Provides a connected client session for interacting with any MCP server.

    Scope: function (default) - ensures isolated client sessions for each test
    Args:
        mcp_server_instance: The FastMCP server instance to connect to
    Yields: Connected ClientSession
    Cleanup: Automatically disconnects the session when the test completes

    This is a project-wide fixture for connecting to any MCP server instance.
    It's placed in the root conftest because it's a general-purpose fixture
    that could be used by multiple test modules.
    """
    # We need to directly connect to the server's underlying MCP server object
    # This follows the pattern shown in the FastMCP documentation
    server = mcp_server_instance._mcp_server

    # No need to check if server is running - connect() should handle that
    client = await server.connect()

    try:
        # Initialize with empty options
        await client.initialize({})
        yield client
    finally:
        # Clean up
        await client.disconnect()
