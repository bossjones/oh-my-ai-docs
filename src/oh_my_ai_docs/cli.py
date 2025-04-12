from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.traceback import install as install_rich_traceback

from oh_my_ai_docs.__version__ import __version__

# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Create Typer app
app = typer.Typer(
    name="mcp-client",
    help="Command-line client for MCP servers",
    add_completion=False,
)

# Create console for rich output
console = Console()

# Config command group
config_app = typer.Typer(help="Manage MCP client configuration")
app.add_typer(config_app, name="config")


# Utility commands
@app.command()
def version() -> None:
    """Show the version of the MCP CLI client."""
    console.print(f"MCP CLI Client v{__version__}")


@app.callback()
def callback(
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
) -> None:
    """MCP CLI Client - Interact with MCP servers from the command line."""
    # Set debug mode
    if debug:
        # Show full traceback for errors
        def exception_handler(exc_type, exc_value, exc_traceback):
            console.print_exception(show_locals=True)

        sys.excepthook = exception_handler


def run_async(func: Any) -> Any:
    """Run an async function in the event loop.

    Args:
        func: Async function to run.

    Returns:
        The result of the async function.

    """
    return asyncio.run(func)


def main() -> None:
    """Run the MCP CLI client."""
    app()


if __name__ == "__main__":
    main()
