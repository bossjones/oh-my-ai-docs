#!/usr/bin/env python3
# async version, will come back to this
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingImports=false
# pyright: reportUnusedVariable=warning
# pyright: reportUntypedBaseClass=error
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from asyncio import timeout
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, cast

import aiofiles
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from mcp import ClientSession, StdioServerParameters
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.fastmcp.utilities.logging import get_logger
from mcp.server.session import ServerSession
from mcp.types import (
    TextContent,
)
from pydantic import BaseModel, Field, field_validator

# Configure logging - UNCOMMENTED this line
# logger = get_logger(__name__)

T = TypeVar("T")

# Define common path to the repo locally
BASE_PATH = Path("/Users/malcolm/dev/bossjones/oh-my-ai-docs")
DOCS_PATH = BASE_PATH / "docs/ai_docs"


class MCPError(Exception):
    """Base error class for MCP operations."""

    pass


class ToolError(MCPError):
    """Error raised by MCP tools."""

    pass


class DocumentResponse(BaseModel):
    documents: list[str]
    scores: list[float]
    total_found: int


@dataclass
class AppContext:
    store: SKLearnVectorStore


# Create argument parser
parser = argparse.ArgumentParser(description="MCP Server for vectorstore queries")
parser.add_argument(
    "--module",
    type=str,
    choices=["discord", "dpytest", "langgraph"],
    default="dpytest",
    help="Module to query (default: dpytest)",
)
parser.add_argument("--dry-run", action="store_true", help="Show configuration without starting the server")
parser.add_argument(
    "--list-vectorstores", action="store_true", help="List available vector stores (searches for .parquet files)"
)
parser.add_argument(
    "--generate-mcp-config", action="store_true", help="Generate mcp.json configuration for all modules"
)
parser.add_argument("--save", action="store_true", help="Save the generated mcp.json configuration to disk")
parser.add_argument(
    "--stdio", action="store_true", default=True, help="Run in stdio mode with all logging/printing disabled"
)
parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")

# Only parse args when run directly
if __name__ == "__main__":
    args = parser.parse_args()
else:
    # For testing and importing, use default args
    args = parser.parse_args([])


async def get_config_info():
    """Get configuration information for display"""
    module_path = DOCS_PATH / args.module
    config = {
        "server_name": f"{args.module}-docs-mcp-server".lower(),
        "module": args.module,
        "paths": {
            "base_path": str(BASE_PATH),
            "docs_path": str(DOCS_PATH),
            "module_path": str(module_path),
            "vectorstore": str(module_path / "vectorstore" / f"{args.module}_vectorstore.parquet"),
            "docs_file": str(module_path / f"{args.module}_docs.txt"),
        },
        "available_endpoints": {
            "tool": f"query_tool - Query {args.module} documentation",
            "resource": f"docs://{args.module}/full - Get full {args.module} documentation",
        },
    }
    return config


async def list_vectorstores():
    """Search for and list all .parquet files in the docs directory"""
    # logger.info("Listing available vector stores")

    # Find all .parquet files recursively
    parquet_files: list[Path] = list(DOCS_PATH.glob("**/*.parquet"))

    if not parquet_files:
        # logger.info("No vector stores found.")
        return

    # Group by module
    stores_by_module: dict[str, list[Path]] = {}
    for file in parquet_files:
        module_name = file.parent.parent.name
        if module_name not in stores_by_module:
            stores_by_module[module_name] = []
        stores_by_module[module_name].append(file)

    # Log the results
    for module, files in stores_by_module.items():
        # logger.info(f"Module: {module}")
        for file in files:
            try:
                relative_path = file.relative_to(BASE_PATH)
            except ValueError:
                # If file is not under BASE_PATH, show path relative to DOCS_PATH parent
                relative_path = file.relative_to(DOCS_PATH.parent)
            # logger.info(f"  - {relative_path}")

    # logger.info(f"Total vector stores found: {len(parquet_files)}")


async def generate_mcp_config() -> dict[str, dict[str, Any]]:
    """Generate mcp.json configuration for all modules"""
    # logger.info("Generating MCP configuration for all modules")
    modules = ["discord", "dpytest", "langgraph"]

    # Get the script path relative to BASE_PATH
    script_path = os.path.abspath(__file__)
    relative_script_path = os.path.relpath(script_path, BASE_PATH)

    mcp_config: dict[str, dict[str, Any]] = {"mcpServers": {}}

    for module in modules:
        server_name = f"{module}-docs-mcp-server".lower()
        mcp_config["mcpServers"][server_name] = {
            "command": "uv",
            "args": ["run", "--directory", str(BASE_PATH), f"./{relative_script_path}", "--module", module],
        }

    # logger.info("MCP configuration generated successfully")
    return mcp_config


async def save_mcp_config(config: dict[str, dict[str, Any]]) -> None:
    """Save the MCP configuration to disk"""
    save_path = BASE_PATH / "mcp.json"
    async with aiofiles.open(save_path, "w") as f:
        await f.write(json.dumps(config, indent=2))
    # logger.info(f"Configuration saved to {save_path}")


# Define validation models
class QueryConfig(BaseModel):
    k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")
    min_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score threshold")

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v > 10:
            raise ValueError("k cannot be greater than 10 to prevent excessive token usage")
        return v


def get_vectorstore_path() -> Path:
    return DOCS_PATH / args.module / "vectorstore" / f"{args.module}_vectorstore.parquet"


@asynccontextmanager
async def vectorstore_session(server: FastMCP) -> AsyncIterator[AppContext]:
    """Context manager for vectorstore operations."""
    vectorstore_path = get_vectorstore_path()
    try:
        # logger.debug(f"Opening vectorstore session: {vectorstore_path}")
        store = SKLearnVectorStore(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            persist_path=str(vectorstore_path),
            serializer="parquet",
        )
        yield AppContext(store=store)
        # logger.debug("Vectorstore session completed")
    except Exception as e:
        # logger.error(f"Error in vectorstore session: {e}", exc_info=True)
        raise
    finally:
        # Cleanup if needed
        # logger.debug("Vectorstore session cleanup complete")
        pass


# Create an MCP server with module name
mcp_server = FastMCP(f"{args.module}-docs-mcp-server".lower(), lifespan=vectorstore_session)


# Add a tool to query the documentation
@mcp_server.tool(
    name="query_docs",
    description="Search through module documentation using semantic search to find relevant information based on your query",
)
async def query_tool(query: str) -> DocumentResponse:
    """
    Query the documentation using a retriever.

    Args:
        query (str): The query string to search for in the documentation
        ctx (Context[Any, Any]): Tool context for progress reporting and status updates
        config (Optional[QueryConfig]): Query configuration parameters including:
            - k (int): Number of documents to retrieve (1-10, default=3)
            - min_relevance_score (float): Minimum relevance score threshold (0.0-1.0, default=0.0)

    Returns:
        DocumentResponse: A structured response containing:
            - documents (List[str]): List of retrieved document contents
            - scores (List[float]): Corresponding relevance scores for each document
            - total_found (int): Total number of documents found before filtering

    Raises:
        ToolError: If the query operation fails or returns invalid results
        ValueError: If the query string is empty or invalid
        TimeoutError: If the query operation takes longer than 30 seconds
    """
    ctx: Context[ServerSession, object] = mcp_server.get_context()
    if not query.strip():
        await ctx.error("Query cannot be empty")
        raise ValueError("Query cannot be empty")

    config = QueryConfig()

    vectorstore_path = get_vectorstore_path()

    state: AppContext = cast(AppContext, ctx.request_context.lifespan_context)
    await ctx.debug(f"state: {state}")

    try:
        # async with timeout(30):  # Prevent hanging on API calls
        #     async with vectorstore_session(str(vectorstore_path)) as app_ctx:
        await ctx.info(f"Querying vectorstore with k={config.k}")

        # import bpdb; bpdb.set_trace()
        retriever: VectorStoreRetriever = state.as_retriever(search_kwargs={"k": config.k})

        relevant_docs: list[Document] = await asyncio.to_thread(retriever.invoke, query)

        await ctx.info(f"Retrieved {len(relevant_docs)} relevant documents")

        documents: list[str] = []
        scores: list[float] = []

        for i, doc in enumerate(relevant_docs):
            if hasattr(doc, "metadata") and doc.metadata.get("score", 1.0) < config.min_relevance_score:
                continue

            documents.append(doc.page_content)
            scores.append(doc.metadata.get("score", 1.0) if hasattr(doc, "metadata") else 1.0)
            await ctx.report_progress(i + 1, len(relevant_docs))

        return DocumentResponse(documents=documents, scores=scores, total_found=len(relevant_docs))

    except TimeoutError:
        await ctx.error("Query timed out")
        # logger.error("Query operation timed out after 30 seconds")
        raise ToolError("Query operation timed out after 30 seconds")
    except Exception as e:
        await ctx.error(f"Query failed: {e!s}")
        # logger.error(f"Failed to query vectorstore: {e!s}", exc_info=True)
        raise ToolError(f"Failed to query vectorstore: {e!s}")


@mcp_server.resource(
    uri="docs://{module}/full",
    name="module_documentation",
    description="Retrieves the full documentation content for a specified module (discord, dpytest, or langgraph). Returns the raw text content from the module's documentation file.",
    mime_type="text/plain",
)
async def get_all_docs(module: str) -> str:
    """
    Get all the documentation for the specified module. Returns the contents of the {module}_docs.txt file,
    which contains a curated set of documentation. This is useful for a comprehensive response to questions.

    Args:
        module (str): The module name (discord, dpytest, or langgraph)

    Returns:
        str: The contents of the module's documentation

    Raises:
        ResourceError: If the module doesn't match or if there's an error reading the documentation
    """
    ctx: Context[ServerSession, object] = mcp_server.get_context()
    # import bpdb; bpdb.set_trace()
    try:
        await ctx.info(f"Retrieving documentation for module: {module}")

        if module != args.module:
            await ctx.error("Module mismatch", extra={"requested_module": module, "server_module": args.module})
            raise ResourceError(f"Requested module '{module}' does not match server module '{args.module}'")

        # Local path to the documentation
        doc_path = DOCS_PATH / module / f"{module}_docs.txt"

        if not doc_path.exists():
            await ctx.error("Documentation file not found", extra={"doc_module": module, "path": str(doc_path)})
            raise ResourceError(f"Documentation file not found for module: {module}")

        async with aiofiles.open(doc_path) as file:
            content = await file.read()
            await ctx.info("Successfully read documentation", extra={"doc_module": module, "size": len(content)})
            return content

    except ResourceError:
        raise
    except Exception as e:
        # logger.error("Error reading documentation", extra={"doc_module": module, "error": str(e)}, exc_info=True)
        raise ResourceError(f"Error reading documentation file: {e}")


if __name__ == "__main__":
    import asyncio

    # async def main():
    try:
        if args.list_vectorstores:
            pass
            # await list_vectorstores()
            # logger.info("Vectorstore listing completed")
        # elif args.generate_mcp_config:
        #     config = await generate_mcp_config()
        #     if args.save:
        #         await save_mcp_config(config)
        # elif args.dry_run:
        #     config = await get_config_info()
        #     logger.info("MCP Server Configuration:")
        #     logger.info(json.dumps(config, indent=2))
        #     logger.info("Dry run completed. Use without --dry-run to start the server.")
        else:
            # Initialize and run the server
            # logger.info(f"Starting MCP server for {args.module} documentation...")
            mcp_server.run(transport="stdio")
    except Exception as e:
        # logger.error(f"Error in main execution: {e}", exc_info=True)
        # sys.exit(1)
        # except Exception as e:
        # logger.error(f"Error starting server: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Run the main async function
    # asyncio.run(main())
