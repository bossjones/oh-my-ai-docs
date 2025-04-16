#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langchain-community",
#     "langchain-core",
#     "langchain-openai",
#     "mcp[cli]",
#     "aiofiles",
#     "pydantic",
# ]
# ///

# async version, will come back to this
# pyright: reportUnknownArgumentType=false
# pyright: reportMissingImports=false
# pyright: reportUnusedVariable=warning
# pyright: reportUntypedBaseClass=error
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportUnusedVariable=false
# pyright: reportConstantRedefinition=false

"""
avectorstore_mcp.py - Documentation Vector Store MCP Server

This module implements a FastMCP server that provides semantic search capabilities over documentation using vector stores.
It supports querying documentation for different modules (discord, dpytest, langgraph) through a vector store interface,
allowing for semantic similarity search and document retrieval.

Key features:
- Vector store-based semantic search over documentation
- Support for multiple documentation modules
- Configurable search parameters (k, relevance scores)
- Full documentation retrieval endpoints
- Async operation with proper resource management

Based on the vectorstore_session implementation from langchain-community/fastmcp project.
"""

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

# Add these imports at the top of the file
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, cast

import aiofiles
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from mcp import ClientSession, StdioServerParameters
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.fastmcp.utilities.logging import get_logger
from mcp.server.session import ServerSession
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)
from pydantic import BaseModel, Field, field_validator

logger = get_logger(__name__)

# Add this after the BASE_PATH and DOCS_PATH declarations
# Global embeddings configuration
_EMBEDDINGS_PROVIDER: Embeddings | None = None


def set_embeddings_provider(embeddings: Embeddings) -> None:
    """
    Set the embeddings provider to use for vectorstore operations.
    This allows for dependency injection, particularly useful in testing.

    Args:
        embeddings: The embeddings provider to use
    """
    global _EMBEDDINGS_PROVIDER
    _EMBEDDINGS_PROVIDER = embeddings


def get_embeddings_provider() -> Embeddings:
    """
    Get the configured embeddings provider, or create a default one if none is set.

    Returns:
        The embeddings provider to use for vectorstore operations
    """
    global _EMBEDDINGS_PROVIDER
    if _EMBEDDINGS_PROVIDER is None:
        from langchain_openai import OpenAIEmbeddings

        _EMBEDDINGS_PROVIDER = OpenAIEmbeddings(model="text-embedding-3-large")
    return _EMBEDDINGS_PROVIDER


# Configure logging
logger = get_logger(__name__)

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
        "server_name": f"{args.module}-avectorstore-mcp".lower(),
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
    # Find all .parquet files recursively
    parquet_files: list[Path] = list(DOCS_PATH.glob("**/*.parquet"))

    if not parquet_files:
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
        for file in files:
            try:
                relative_path = file.relative_to(BASE_PATH)
            except ValueError:
                # If file is not under BASE_PATH, show path relative to DOCS_PATH parent
                relative_path = file.relative_to(DOCS_PATH.parent)


async def generate_mcp_config() -> dict[str, dict[str, Any]]:
    """Generate mcp.json configuration for all modules"""
    # logger.info("Generating MCP configuration for all modules")
    modules = ["discord", "dpytest", "langgraph"]

    # Get the script path relative to BASE_PATH
    script_path = os.path.abspath(__file__)
    relative_script_path = os.path.relpath(script_path, BASE_PATH)

    mcp_config: dict[str, dict[str, Any]] = {"mcpServers": {}}

    for module in modules:
        server_name = f"{module}-avectorstore-mcp".lower()
        mcp_config["mcpServers"][server_name] = {
            "command": "uv",
            "args": ["run", "--directory", str(BASE_PATH), f"./{relative_script_path}", "--module", module],
        }

    return mcp_config


async def save_mcp_config(config: dict[str, dict[str, Any]]) -> None:
    """Save the MCP configuration to disk"""
    save_path = BASE_PATH / "mcp.json"
    async with aiofiles.open(save_path, "w") as f:
        await f.write(json.dumps(config, indent=2))


# Define validation models
class QueryConfig(BaseModel):
    k: int = Field(default=2, ge=1, le=10, description="Number of documents to retrieve")
    min_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score threshold")

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v > 10:
            raise ValueError("k cannot be greater than 10 to prevent excessive token usage")
        return v


def get_vectorstore_path() -> Path:
    return DOCS_PATH / args.module / "vectorstore" / f"{args.module}_vectorstore.parquet"


def vectorstore_factory(
    store: VectorStore | None = None,
    embeddings: Embeddings | None = None,
    vector_store_cls: type[VectorStore] = SKLearnVectorStore,
    vector_store_kwargs: dict[str, Any] | None = None,
) -> SKLearnVectorStore | VectorStore:
    """
    Factory function to create or return a vector store.

    Args:
        store: An existing vector store to return if provided
        embeddings: Embedding model to use for vectorization
        vector_store_cls: Vector store class to instantiate if store is None
        vector_store_kwargs: Additional arguments to pass to the vector store constructor

    Returns:
        A vector store instance
    """
    if store:
        return store

    # Use the provided embeddings or get the configured one
    embeddings = embeddings or get_embeddings_provider()

    if vector_store_kwargs is None:
        vector_store_kwargs = {}

    # Add default kwargs if not provided
    if "persist_path" not in vector_store_kwargs:
        vectorstore_path = get_vectorstore_path()
        vector_store_kwargs["persist_path"] = str(vectorstore_path)

    if "serializer" not in vector_store_kwargs:
        vector_store_kwargs["serializer"] = "parquet"

    # Ensure embedding is set
    vector_store_kwargs["embedding"] = embeddings

    return vector_store_cls(**vector_store_kwargs)


@asynccontextmanager
async def vectorstore_session(server: FastMCP) -> AsyncIterator[AppContext]:
    """Context manager for vectorstore operations."""
    try:
        # Use the factory with the configured embeddings provider
        store = vectorstore_factory(
            embeddings=get_embeddings_provider(),
            vector_store_cls=SKLearnVectorStore,
            vector_store_kwargs={"persist_path": str(get_vectorstore_path()), "serializer": "parquet"},
        )
        yield AppContext(store=store)
    except Exception as e:
        raise
    finally:
        # Cleanup if needed
        pass


# Create an MCP server with module name
mcp_server = FastMCP(f"{args.module}-avectorstore-mcp".lower(), lifespan=vectorstore_session)


# Add a tool to query the documentation
@mcp_server.tool(
    name="query_docs",
    description="Search through module documentation using semantic search to find relevant information based on your query",
)
async def query_tool(
    query: str = Field(
        description="The query string to search for in the documentation",
        examples=["What is the best way to test discord bots with dpytest?"],
        min_length=1,
    ),
) -> DocumentResponse | list[TextContent]:
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
    # import bpdb
    # bpdb.set_trace()
    if not query.strip():
        raise ValueError("Query cannot be empty")

    config = QueryConfig()

    vectorstore_path = get_vectorstore_path()

    state: AppContext = cast(AppContext, ctx.request_context.lifespan_context)

    try:
        # Get the retriever from the store, not from state directly
        retriever: VectorStoreRetriever = state.store.as_retriever(search_kwargs={"k": config.k})

        relevant_docs: list[Document] = await asyncio.to_thread(retriever.invoke, query)

        documents: list[str] = []
        scores: list[float] = []

        for i, doc in enumerate(relevant_docs):
            if hasattr(doc, "metadata") and doc.metadata.get("score", 1.0) < config.min_relevance_score:
                continue

            documents.append(doc.page_content)
            scores.append(doc.metadata.get("score", 1.0) if hasattr(doc, "metadata") else 1.0)

        return DocumentResponse(documents=documents, scores=scores, total_found=len(relevant_docs))
    except Exception as ex:
        return [TextContent(type="text", text=f"Error: {ex!s}")]


@mcp_server.resource(
    uri="docs://{module}/full",
    name="module_documentation",
    description="Retrieves the full documentation content for a specified module (discord, dpytest, or langgraph). Returns the raw text content from the module's documentation file.",
    mime_type="text/plain",
)
async def get_all_docs(
    module: str = Field(
        description="The module name (discord, dpytest, or langgraph)", examples=["discord", "dpytest", "langgraph"]
    ),
) -> str:
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
    # Get the current server context which contains session and request information
    ctx: Context[ServerSession, object] = mcp_server.get_context()

    try:
        # # Validate that the requested module matches the server's configured module
        # if module != args.module:
        #     raise ResourceError(f"Requested module '{module}' does not match server module '{args.module}'")

        # Construct the full path to the module's documentation file
        doc_path = DOCS_PATH / module / f"{module}_docs.txt"

        # Check if the documentation file exists at the specified path
        if not doc_path.exists():
            # Raise a ValueError if the documentation file is not found
            raise ValueError(f"Documentation file not found for module: {module}")

        # Open the documentation file asynchronously using aiofiles
        async with aiofiles.open(doc_path) as file:
            # Read the entire contents of the file asynchronously
            content = await file.read()

            # Return the documentation content as a string
            return content

    # Catch any other unexpected errors and wrap them in a ResourceError with a descriptive message
    except Exception as e:
        raise ValueError(f"Error reading documentation file: {e}")


def main() -> None:
    """Entry point for the avectorstore MCP server."""
    try:
        if args.list_vectorstores:
            pass
        else:
            # Initialize and run the server
            mcp_server.run(transport="stdio")
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        logger.error(f"Error: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
