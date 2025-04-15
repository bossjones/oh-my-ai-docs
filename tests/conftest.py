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
