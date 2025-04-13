#!/Users/malcolm/dev/bossjones/oh-my-ai-docs/.venv/bin/python

import argparse
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

args = parser.parse_args()

# Create an MCP server with module name
mcp = FastMCP(f"{args.module.capitalize()}-Docs-MCP-Server")

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
    # Initialize and run the server
    mcp.run(transport='stdio')
