#!/usr/bin/env python3
# async version, will come back to this
# pyright: reportUnknownArgumentType=false

import argparse
import glob
import json
import logging
import os
from asyncio import timeout
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import aiofiles
import anyio
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.fastmcp.utilities.logging import get_logger
from pydantic import BaseModel, Field, field_validator

# Configure logging
logger = get_logger(__name__)

T = TypeVar("T")

# Define common path to the repo locally
BASE_PATH = Path("/Users/malcolm/dev/bossjones/oh-my-ai-docs")
DOCS_PATH = BASE_PATH / "docs/ai_docs"

# Create argument parser
parser = argparse.ArgumentParser(description="MCP Server for vectorstore queries")
parser.add_argument(
    "--module",
    type=str,
    choices=["discord", "dpytest", "langgraph"],
    default="langgraph",
    help="Module to query (default: langgraph)",
)
parser.add_argument("--dry-run", action="store_true", help="Show configuration without starting the server")
parser.add_argument(
    "--list-vectorstores", action="store_true", help="List available vector stores (searches for .parquet files)"
)
parser.add_argument(
    "--generate-mcp-config", action="store_true", help="Generate mcp_server.json configuration for all modules"
)
parser.add_argument("--save", action="store_true", help="Save the generated mcp_server.json configuration to disk")
parser.add_argument(
    "--stdio", action="store_true", default=True, help="Run in stdio mode with all logging/printing disabled"
)
parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")


def get_config_info(args: argparse.Namespace):
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


def list_vectorstores():
    """Search for and list all .parquet files in the docs directory"""
    print("\n=== Available Vector Stores ===\n")

    # Find all .parquet files recursively
    parquet_files: list[Path] = list(DOCS_PATH.glob("**/*.parquet"))

    if not parquet_files:
        print("No vector stores found.")
        return

    # Group by module
    stores_by_module: dict[str, list[Path]] = {}
    for file in parquet_files:
        module_name = file.parent.parent.name
        if module_name not in stores_by_module:
            stores_by_module[module_name] = []
        stores_by_module[module_name].append(file)

    # Print the results
    for module, files in stores_by_module.items():
        print(f"Module: {module}")
        for file in files:
            print(f"  - {file.relative_to(BASE_PATH)}")
        print()

    print(f"Total vector stores found: {len(parquet_files)}")


def generate_mcp_config() -> dict[str, dict[str, Any]]:
    """Generate mcp_server.json configuration for all modules"""
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

    # Print the generated config
    print("\n=== Generated MCP Configuration ===\n")
    print(json.dumps(mcp_config, indent=2))

    return mcp_config


def save_mcp_config(config: dict[str, dict[str, Any]]) -> None:
    """Save the MCP configuration to disk"""
    save_path = BASE_PATH / "mcp_server.json"
    with open(save_path, "w") as f:
        json.dump(config, indent=2, fp=f)
    print(f"\nConfiguration saved to {save_path}")


def get_args():
    """Get command line arguments"""
    return parser.parse_args()


def create_mcp_server(module_name: str = "discord"):
    """Create MCP server instance"""
    return FastMCP(f"{module_name}-docs-mcp-server".lower())


# Create an MCP server with module name
mcp_server: FastMCP = create_mcp_server()

# # Option 2: Run with STDIO transport
# async def start_server_stdio():
#     await mcp_server.run_stdio_async()


class MCPError(Exception):
    """Base error class for MCP operations."""

    pass


class ToolError(MCPError):
    """Error raised by MCP tools."""

    pass


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


class DocumentResponse(BaseModel):
    documents: list[str]
    scores: list[float]
    total_found: int


@asynccontextmanager
async def vectorstore_session(vectorstore_path: str):
    """Context manager for vectorstore operations."""
    try:
        store = SKLearnVectorStore(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            persist_path=str(vectorstore_path),
            serializer="parquet",
        )
        yield store
    finally:
        # Cleanup if needed
        pass


# Add a tool to query the documentation
@mcp_server.tool(
    name="query_docs",
    description="Search through module documentation using semantic search to find relevant information based on your query",
)
async def query_tool(query: str, ctx: Context[Any, Any], config: QueryConfig | None = None) -> DocumentResponse:
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
    if not query.strip():
        raise ValueError("Query cannot be empty")

    config = config or QueryConfig()
    vectorstore_path = DOCS_PATH / args.module / "vectorstore" / f"{args.module}_vectorstore.parquet"

    try:
        async with timeout(30):  # Prevent hanging on API calls
            async with vectorstore_session(str(vectorstore_path)) as store:
                await ctx.info(f"Querying vectorstore with k={config.k}")

                retriever = store.as_retriever(search_kwargs={"k": config.k})

                relevant_docs: list[Document] = retriever.invoke(query)

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
        raise ToolError("Query operation timed out after 30 seconds")
    except Exception as e:
        await ctx.error(f"Query failed: {e!s}")
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
    try:
        if module != args.module:
            logger.error("Module mismatch", extra={"requested_module": module, "server_module": args.module})
            raise ResourceError(f"Requested module '{module}' does not match server module '{args.module}'")

        # Local path to the documentation
        doc_path = DOCS_PATH / module / f"{module}_docs.txt"

        if not doc_path.exists():
            logger.error("Documentation file not found", extra={"module": module, "path": str(doc_path)})
            raise ResourceError(f"Documentation file not found for module: {module}")

        async with aiofiles.open(doc_path) as file:
            content = await file.read()
            logger.info("Successfully read documentation", extra={"module": module, "size": len(content)})
            return content

    except ResourceError:
        raise
    except Exception as e:
        logger.error("Error reading documentation", extra={"module": module, "error": str(e)})
        raise ResourceError(f"Error reading documentation file: {e}")


if __name__ == "__main__":
    import asyncio

    # Get command line arguments
    args = get_args()

    # Update MCP server with correct module
    global mcp_server
    mcp_server = create_mcp_server(args.module)

    # Configure logging based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        # In stdio mode, disable all output except through the MCP protocol
        if args.stdio:
            logging.basicConfig(level=logging.ERROR)
            logger.setLevel(logging.ERROR)
        else:
            logging.basicConfig(level=logging.INFO)
            logger.setLevel(logging.INFO)

    if args.list_vectorstores:
        list_vectorstores()
    elif args.generate_mcp_config:
        config = generate_mcp_config()
        if args.save:
            save_mcp_config(config)
    elif args.dry_run:
        config = get_config_info(args)
        print("\n=== MCP Server Configuration ===\n")
        print(json.dumps(config, indent=2))
        print("\nDry run completed. Use without --dry-run to start the server.")
    else:
        # Initialize and run the server
        if not args.stdio:
            print(f"Starting MCP server for {args.module} documentation...")
        mcp_server.run(transport="stdio")
