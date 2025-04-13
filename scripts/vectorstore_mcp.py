#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP

# Define common path to the repo locally
BASE_PATH = Path("/Users/malcolm/dev/bossjones/oh-my-ai-docs")
DOCS_PATH = BASE_PATH / "docs/ai_docs"

# Create argument parser
parser = argparse.ArgumentParser(description='MCP Server for vectorstore queries')
parser.add_argument('--module', type=str, choices=['discord', 'dpytest', 'langgraph'],
                  default='langgraph', help='Module to query (default: langgraph)')
parser.add_argument('--dry-run', action='store_true',
                  help='Show configuration without starting the server')
parser.add_argument('--list-vectorstores', action='store_true',
                  help='List available vector stores (searches for .parquet files)')

args = parser.parse_args()

def get_config_info():
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
            "docs_file": str(module_path / f"{args.module}_docs.txt")
        },
        "available_endpoints": {
            "tool": f"query_tool - Query {args.module} documentation",
            "resource": f"docs://{args.module}/full - Get full {args.module} documentation"
        }
    }
    return config

def list_vectorstores():
    """Search for and list all .parquet files in the docs directory"""
    print("\n=== Available Vector Stores ===\n")

    # Find all .parquet files recursively
    parquet_files = list(DOCS_PATH.glob("**/*.parquet"))

    if not parquet_files:
        print("No vector stores found.")
        return

    # Group by module
    stores_by_module = {}
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

# Create an MCP server with module name
mcp = FastMCP(f"{args.module}-docs-mcp-server".lower())

# Add a tool to query the documentation
@mcp.tool()
def query_tool(query: str):
    """
    Query the documentation using a retriever.

    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    vectorstore_path = DOCS_PATH / args.module / "vectorstore" / f"{args.module}_vectorstore.parquet"

    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_path=str(vectorstore_path),
        serializer="parquet").as_retriever(search_kwargs={"k": 3}
        )

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context

# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://{module}/full")
def get_all_docs(module: str) -> str:
    """
    Get all the documentation for the specified module. Returns the contents of the {module}_docs.txt file,
    which contains a curated set of documentation. This is useful for a comprehensive response to questions.

    Args:
        module (str): The module name (discord, dpytest, or langgraph)

    Returns:
        str: The contents of the module's documentation
    """
    if module != args.module:
        return f"Error: Requested module '{module}' does not match server module '{args.module}'"

    # Local path to the documentation
    doc_path = DOCS_PATH / module / f"{module}_docs.txt"
    try:
        with open(doc_path) as file:
            return file.read()
    except Exception as e:
        return f"Error reading documentation file: {e!s}"

if __name__ == "__main__":
    if args.list_vectorstores:
        list_vectorstores()
    elif args.dry_run:
        config = get_config_info()
        print("\n=== MCP Server Configuration ===\n")
        print(json.dumps(config, indent=2))
        print("\nDry run completed. Use without --dry-run to start the server.")
    else:
        # Initialize and run the server
        print(f"Starting MCP server for {args.module} documentation...")
        mcp.run(transport='stdio')
