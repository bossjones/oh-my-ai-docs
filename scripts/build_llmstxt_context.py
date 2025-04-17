#!/usr/bin/env python3
"""
Build a local vector store from module documentation for semantic search and retrieval.

This script creates a searchable vector store from documentation of specified modules
(like LangGraph, Discord.py, etc.) by:
1. Fetching documentation from predefined URLs for each module
2. Processing and cleaning the HTML content
3. Splitting documents into semantically meaningful chunks
4. Converting text chunks into vector embeddings using OpenAI's text-embedding-3-large model
5. Storing the embeddings in a local SKLearnVectorStore for efficient retrieval

The resulting vector store enables semantic search capabilities, allowing for
context-aware documentation lookups and enhanced AI assistance.

Key Features:
- Supports multiple documentation sources (LangGraph, Discord.py, dpytest, etc.)
- Uses BeautifulSoup for clean HTML content extraction
- Implements efficient document chunking with tiktoken-based text splitting
- Creates persistent vector stores using SKLearn for local storage
- Includes dry-run capability for testing configuration

Usage:
    python build_llmstxt_context.py --module [module_name]
    python build_llmstxt_context.py --module langgraph --dry-run

Available modules:
    - langgraph
    - langchain
    - dpytest
    - discord

Dependencies:
    - tiktoken: For token counting and text splitting
    - beautifulsoup4: For HTML parsing
    - langchain: For document processing and vector store creation
    - openai: For text embeddings
    - scikit-learn: For vector store backend
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Iterator
from typing import Any, Dict, List, Tuple

import tiktoken
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

# Define URL mappings for different modules
MODULE_URLS: dict[str, list[str]] = {
    "langgraph": [
        "https://langchain-ai.github.io/langgraph/concepts/",
        "https://langchain-ai.github.io/langgraph/how-tos/",
        "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
        "https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/",
        "https://python.langchain.com/docs/get_started/introduction/",
        "https://python.langchain.com/docs/modules/",
        "https://python.langchain.com/docs/use_cases/",
    ],
    "langchain": [
        "https://python.langchain.com/docs/get_started/introduction/",
        "https://python.langchain.com/docs/modules/",
        "https://python.langchain.com/docs/use_cases/",
    ],
    "dpytest": [
        "https://dpytest.readthedocs.io/en/latest/index.html",
        "https://dpytest.readthedocs.io/en/latest/tutorials/index.html",
        "https://dpytest.readthedocs.io/en/latest/modules/index.html",
        "https://dpytest.readthedocs.io/en/latest/modules/backend.html",
        "https://dpytest.readthedocs.io/en/latest/modules/callbacks.html",
        "https://dpytest.readthedocs.io/en/latest/modules/factories.html",
        "https://dpytest.readthedocs.io/en/latest/modules/runner.html",
        "https://dpytest.readthedocs.io/en/latest/modules/state.html",
        "https://dpytest.readthedocs.io/en/latest/modules/utils.html",
        "https://dpytest.readthedocs.io/en/latest/modules/verify.html",
        "https://dpytest.readthedocs.io/en/latest/modules/websocket.html",
        "https://dpytest.readthedocs.io/en/latest/tutorials/using_pytest.html",
    ],
    "discord": [
        "https://discordpy.readthedocs.io/en/stable/index.html",
        "https://discordpy.readthedocs.io/en/stable/intro.html",
        "https://discordpy.readthedocs.io/en/stable/quickstart.html",
        "https://discordpy.readthedocs.io/en/stable/logging.html",
        "https://discordpy.readthedocs.io/en/stable/discord.html",
        "https://discordpy.readthedocs.io/en/stable/intents.html",
        "https://github.com/Rapptz/discord.py/tree/v2.5.2/examples",
        "https://discordpy.readthedocs.io/en/stable/faq.html",
        "https://discordpy.readthedocs.io/en/stable/genindex.html",
        "https://discordpy.readthedocs.io/en/stable/ext/commands/index.html",
        "https://discordpy.readthedocs.io/en/stable/ext/tasks/index.html",
        "https://discordpy.readthedocs.io/en/stable/interactions/api.html",
        "https://discordpy.readthedocs.io/en/stable/ext/commands/api.html",
        "https://discordpy.readthedocs.io/en/stable/ext/tasks/index.html",
        "https://discordpy.readthedocs.io/en/stable/migrating.html"
    ],
    # Add more modules as needed
}

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count the number of tokens in the text using tiktoken.

    Args:
        text (str): The text to count tokens for
        model (str): The tokenizer model to use (default: cl100k_base for GPT-4)

    Returns:
        int: Number of tokens in the text
    """
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

def bs4_extractor(html: str) -> str:
    """
    Extract clean text content from HTML using BeautifulSoup.

    This function:
    1. Parses HTML content using BeautifulSoup with lxml parser
    2. Attempts to find the main article content (typically in documentation sites)
    3. Falls back to whole document text if no article is found
    4. Cleans up excessive whitespace in the extracted text

    Args:
        html (str): Raw HTML content to extract text from

    Returns:
        str: Cleaned text content with normalized whitespace
    """
    soup = BeautifulSoup(html, "lxml")

    # Target the main article content for documentation
    main_content = soup.find("article", class_="md-content__inner")

    # If found, use that, otherwise fall back to the whole document
    content = main_content.get_text() if main_content else soup.text

    # Clean up whitespace
    content = re.sub(r"\n\n+", "\n\n", content).strip()

    return content

def load_docs(module: str, dry_run: bool = False) -> tuple[list[Document | Any], list[int]]:
    """
    Load documentation from specified module URLs.

    This function:
    1. Uses RecursiveUrlLoader to fetch pages from the specified URLs
    2. Counts the total documents and tokens loaded

    Args:
        module (str): The module name to load URLs for
        dry_run (bool): If True, only echo what would be done without loading

    Returns:
        tuple[list[Document | Any], list[int]]: A tuple containing:
            - A list of Document objects (or Any for dry runs)
            - A list of token counts per document
    """
    if module not in MODULE_URLS:
        raise ValueError(f"Module '{module}' not found. Available modules: {list(MODULE_URLS.keys())}")

    urls: list[str] = MODULE_URLS[module]
    print(f"{'[DRY RUN] Would load' if dry_run else 'Loading'} {module} documentation...")

    docs: list[Document | Any] = []
    for url in urls:
        if dry_run:
            print(f"[DRY RUN] Would load documents from URL: {url}")
            print("[DRY RUN] Would perform recursive URL loading with max_depth=5")
        else:
            loader = RecursiveUrlLoader(
                url,
                max_depth=5,
                extractor=bs4_extractor,
            )

            # Load documents using lazy loading (memory efficient)
            docs_lazy: Iterator[Document] = loader.lazy_load()

            # Load documents and track URLs
            for d in docs_lazy:
                docs.append(d)

    if dry_run:
        print(f"[DRY RUN] Would have loaded documents from {len(urls)} URLs for {module} documentation.")
        return [], []
    else:
        print(f"Loaded {len(docs)} documents from {module} documentation.")
        print("\nLoaded URLs:")
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.metadata.get('source', 'Unknown URL')}")

        # Count total tokens in documents
        total_tokens = 0
        tokens_per_doc: list[int | Any] = []
        for doc in docs:
            doc_tokens: int = count_tokens(doc.page_content)
            total_tokens += doc_tokens
            tokens_per_doc.append(doc_tokens)
        print(f"Total tokens in loaded documents: {total_tokens}")

        return docs, tokens_per_doc

def split_documents(documents: list[Document], dry_run: bool = False) -> list[Document]:
    """
    Split documents into smaller chunks for improved retrieval.

    This function:
    1. Uses RecursiveCharacterTextSplitter with tiktoken to create semantically meaningful chunks
    2. Ensures chunks are appropriately sized for embedding and retrieval
    3. Counts the resulting chunks and their total tokens

    Args:
        documents (list): List of Document objects to split
        dry_run (bool): If True, only echo what would be done without splitting

    Returns:
        list: A list of split Document objects
    """
    print(f"{'[DRY RUN] Would split' if dry_run else 'Splitting'} documents...")

    if dry_run:
        print("[DRY RUN] Would use RecursiveCharacterTextSplitter with tiktoken encoder")
        print("[DRY RUN] Would set chunk_size=8000 and chunk_overlap=500")
        print(f"[DRY RUN] Would split {len(documents)} documents into chunks")
        return []
    else:
        # Initialize text splitter using tiktoken for accurate token counting
        # chunk_size=8,000 creates relatively large chunks for comprehensive context
        # chunk_overlap=500 ensures continuity between chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=8000,
            chunk_overlap=500
        )

        # Split documents into chunks
        split_docs = text_splitter.split_documents(documents)

        print(f"Created {len(split_docs)} chunks from documents.")

        # Count total tokens in split documents
        total_tokens = 0
        for doc in split_docs:
            total_tokens += count_tokens(doc.page_content)

        print(f"Total tokens in split documents: {total_tokens}")

        return split_docs

def create_vectorstore(splits: list[Document], module: str, dry_run: bool = False) -> SKLearnVectorStore:
    """
    Create a vector store from document chunks using SKLearnVectorStore.

    This function:
    1. Initializes an embedding model to convert text into vector representations
    2. Creates a vector store from the document chunks

    Args:
        splits (list): List of split Document objects to embed
        module (str): The module name used for naming the vector store file
        dry_run (bool): If True, only echo what would be done without creating

    Returns:
        SKLearnVectorStore: A vector store containing the embedded documents
    """
    print(f"{'[DRY RUN] Would create' if dry_run else 'Creating'} SKLearnVectorStore...")

    # Create directory path for vector store
    vector_dir = os.path.join(os.getcwd(), "docs", "ai_docs", module, "vectorstore")
    persist_path = os.path.join(vector_dir, f"{module}_vectorstore.parquet")

    if dry_run:
        print("[DRY RUN] Would initialize OpenAI embeddings with model='text-embedding-3-large'")
        print(f"[DRY RUN] Would create directory if needed: {vector_dir}")
        print(f"[DRY RUN] Would create SKLearnVectorStore from {len(splits) if splits else 'N/A'} document chunks")
        print(f"[DRY RUN] Would persist vector store to: {persist_path}")
        return None
    else:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Create directory if it doesn't exist
        if not os.path.exists(vector_dir):
            os.makedirs(vector_dir)
            print(f"Created directory: {vector_dir}")

        # Create vector store from documents using SKLearn
        vectorstore = SKLearnVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_path=persist_path,
            serializer="parquet",
        )
        print("SKLearnVectorStore created successfully.")

        vectorstore.persist()
        print(f"SKLearnVectorStore was persisted to {persist_path}")

        return vectorstore

def save_docs_to_text(documents: list[Document], module: str, dry_run: bool = False) -> None:
    """
    Save the documents to a text file for reference.

    Args:
        documents (list): List of Document objects to save
        module (str): The module name used for naming the output file
        dry_run (bool): If True, only echo what would be done without saving
    """
    # Create directory path for document text files
    doc_dir = os.path.join(os.getcwd(), "docs", "ai_docs", module)
    output_path = os.path.join(doc_dir, f"{module}_docs.txt")

    if dry_run:
        print(f"[DRY RUN] Would create directory if needed: {doc_dir}")
        print(f"[DRY RUN] Would save {len(documents) if documents else 'N/A'} documents to: {output_path}")
    else:
        # Create directory if it doesn't exist
        if not os.path.exists(doc_dir):
            os.makedirs(doc_dir)
            print(f"Created directory: {doc_dir}")

        with open(output_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(documents):
                f.write(f"Document {i+1}: {doc.metadata.get('source', 'Unknown URL')}\n")
                f.write("-" * 80 + "\n")
                f.write(doc.page_content)
                f.write("\n\n" + "=" * 80 + "\n\n")

        print(f"Saved document contents to {output_path}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate a vector store for documentation")
    parser.add_argument("--module", type=str, default="langgraph",
                      choices=list(MODULE_URLS.keys()),
                      help="Module to generate vector store for")
    parser.add_argument("--dry-run", action="store_true",
                      help="Echo what would be done without actually performing operations")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Starting dry run - no operations will be performed")

    # Load the documents for the specified module
    documents, tokens_per_doc = load_docs(args.module, args.dry_run)

    # Save the documents to a file
    save_docs_to_text(documents, args.module, args.dry_run)

    # Split the documents
    split_docs = split_documents(documents, args.dry_run)

    # Create the vector store
    vectorstore = create_vectorstore(split_docs, args.module, args.dry_run)

    print(f"{'[DRY RUN] Would complete' if args.dry_run else 'Process completed for'} {args.module} documentation!")

if __name__ == "__main__":
    main()
