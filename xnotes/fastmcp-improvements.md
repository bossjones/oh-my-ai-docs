Here are key improvements to your FastMCP server based on MCP SDK best practices and your code:

### 1. Add Lifespan Management
```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
from mcp.server.fastmcp import Context, FastMCP

@asynccontextmanager
async def vectorstore_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Preload vectorstore once at startup"""
    vectorstore_path = DOCS_PATH / args.module / "vectorstore" / f"{args.module}_vectorstore.parquet"

    server.context.info(f"Loading vectorstore from {vectorstore_path}")
    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_path=str(vectorstore_path),
        serializer="parquet"
    ).as_retriever(search_kwargs={"k": 3})

    yield {"retriever": retriever}

mcp = FastMCP(
    name=f"{args.module}-docs-mcp-server".lower(),
    lifespan=vectorstore_lifespan
)
```

### 2. Enhanced Error Handling
```python
from mcp.server.fastmcp import ToolError

@mcp.tool()
async def query_tool(query: str, ctx: Context) -> str:
    """Query documentation with error handling"""
    try:
        retriever = ctx.request_context.lifespan_context["retriever"]
        relevant_docs = await ctx.run_in_executor(retriever.invoke, query)

        if not relevant_docs:
            ctx.warning("No documents found for query")
            return "No relevant documentation found"

        return format_docs(relevant_docs)

    except Exception as e:
        ctx.error(f"Vectorstore query failed: {str(e)}")
        raise ToolError("Document query service unavailable") from e

def format_docs(docs: list) -> str:
    """Structured response with metadata"""
    return "\n\n".join(
        f"### Document {i+1}\n"
        f"**Source**: {doc.metadata.get('source', 'unknown')}\n"
        f"**Content**: {doc.page_content[:500]}..."
        for i, doc in enumerate(docs)
    )
```

### 3. Resource Metadata
```python
@mcp.resource(
    "docs://{module}/full",
    description="Full module documentation",
    metadata={
        "format": "text/plain",
        "updated": (DOCS_PATH / args.module / f"{args.module}_docs.txt").stat().st_mtime
    }
)
def get_all_docs(module: str) -> str:
    """Documentation resource with versioning"""
    # ... existing code ...
```

### 4. Security Additions
```python
from fastapi import HTTPException
from mcp.server.fastmcp import RateLimiter

# Add rate limiting
mcp.add_middleware(
    RateLimiter(requests_per_minute=30, error_message="Too many requests")
)

@mcp.tool()
async def query_tool(query: str, ctx: Context, api_key: str = None):
    """Add API key validation"""
    if not validate_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of tool code ...
```

### 5. Configuration Validation
```python
from pydantic import BaseModel, Field
from typing import Literal

class ServerConfig(BaseModel):
    module: Literal['discord', 'dpytest', 'langgraph'] = Field(...)
    dry_run: bool = False
    api_keys: list[str] = Field(default_factory=list)

def parse_args():
    """Type-safe config handling"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', choices=['discord', 'dpytest', 'langgraph'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--api-keys', nargs='+', default=[])
    return ServerConfig(**vars(parser.parse_args()))
```

### 6. Client Usage Example
```python
async def example_client_usage():
    """Show how clients would use the service"""
    async with mcp_client() as client:
        # Get full docs
        full_docs = await client.read_resource(f"docs://{args.module}/full")

        # Search docs
        results = await client.call_tool(
            "query_tool",
            arguments={"query": "How to handle messages?"}
        )
```

Key improvements made:
1. **Efficiency**: Vectorstore loaded once at startup[1]
2. **Reliability**: Added error boundaries and structured logging[1]
3. **Discoverability**: Enhanced resource metadata
4. **Security**: Rate limiting and API key support
5. **Maintainability**: Pydantic config validation

To implement these changes, you'll need to update dependencies:
```bash
uv add pydantic fastapi python-multipart
```

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/39473573/769ffba5-9f0a-4aaf-aab3-66c13a64cf2a/paste.txt

---
Answer from Perplexity: pplx.ai/share
