"""
Macro definitions for https://mkdocs-macros-plugin.readthedocs.io/

This module provides macro functions for use with the mkdocs-macros-plugin.
It enables dynamic content generation and template rendering in MkDocs pages.
"""
# pyright: reportUnusedFunction=false

import os
from collections.abc import Callable
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment

SIGNATURE: str = "mkdocs_macro_plugin"


def define_env(env: Environment) -> None:
    """Define macros for the mkdocs-macros-plugin environment.

    This function serves as the entry point for registering custom macros
    with the mkdocs-macros-plugin. It receives the plugin environment and
    decorates functions to make them available as macros in MkDocs pages.

    Args:
        env: The mkdocs-macros-plugin environment object that provides access
            to the configuration and page context.
    """
    # activate trace
    chatter: Callable[..., None] = env.start_chatting(SIGNATURE)

    @env.macro
    def include_ai_markdown():
        ai_dir = "./.ai"
        content = ""
        for filename in os.listdir(ai_dir):
            if filename.endswith(".md"):
                with open(os.path.join(ai_dir, filename)) as file:
                    content += file.read() + "\n\n"
        return content

    @env.macro
    def include_ai_files_with_headers() -> str:
        """Include all markdown files from .ai directory with source headers.

        This macro recursively scans the .ai directory and its subdirectories for
        markdown files, reads their contents, and formats them with headers indicating
        the source file. Files are sorted alphabetically for consistent output.

        The output includes:
        1. A table of contents at the top for easy navigation
        2. Clear section headers for each file
        3. Horizontal rules between sections
        4. Proper spacing and formatting

        Returns:
            str: Combined content of all markdown files with TOC and headers

        Example:
            ```markdown
            {{ include_ai_files_with_headers() }}
            ```
        """
        ai_dir = os.path.join(env.project_dir, ".ai")
        content_parts: list[str] = []
        toc_entries: list[tuple[str, str]] = []  # [(file_path, anchor), ...]

        def get_markdown_files(directory: str) -> list[str]:
            """Recursively get all markdown files from directory."""
            markdown_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith((".md", ".mmd")):
                        # Get relative path from .ai directory
                        rel_path = os.path.relpath(os.path.join(root, file), ai_dir)
                        markdown_files.append(rel_path)
            return sorted(markdown_files)

        def create_anchor(filepath: str) -> str:
            """Create a valid HTML anchor from a filepath."""
            # Replace non-alphanumeric chars with hyphens and make lowercase
            return f"file-{filepath.replace('/', '-').replace('.', '-').lower()}"

        # Get all markdown files recursively and sort them
        md_files = get_markdown_files(ai_dir)

        # First, generate TOC
        content_parts.append("# Table of Contents\n")

        for rel_filepath in md_files:
            anchor = create_anchor(rel_filepath)
            toc_entries.append((rel_filepath, anchor))
            content_parts.append(f"- [.ai/{rel_filepath}](#{anchor})")

        # Add spacing after TOC
        content_parts.append("\n---\n")

        # Now add the actual content
        for rel_filepath, anchor in toc_entries:
            # Add a header with anchor
            content_parts.append(f"<h1 id='{anchor}'>.ai/{rel_filepath}</h1>\n")

            # Read and add the file content
            abs_filepath = os.path.join(ai_dir, rel_filepath)
            with open(abs_filepath) as file:
                file_content = file.read().strip()
                content_parts.append(file_content)

            # Add clear separation between files
            content_parts.append("\n\n---\n\n")

        # Combine all parts with proper spacing
        return "\n".join(content_parts)

    @env.macro
    def include_file(filename: str, start_line: int = 0, end_line: int | None = None) -> str:
        """Include a file's contents with optional line range selection.

        Include a file, optionally indicating start_line and end_line
        (start counting from 0). The path is relative to the top directory
        of the documentation project.

        Args:
            filename: The path to the file to include, relative to the project directory
            start_line: The line number to start including from (0-based index)
            end_line: The line number to end including at (0-based index, exclusive)

        Returns:
            str: The content of the file between start_line and end_line

        Example:
            ```markdown
            {{ include_file("docs/example.md", start_line=5, end_line=10) }}
            ```
        """
        chatter("Including:", filename)
        full_filename = os.path.join(env.project_dir, filename)
        with open(full_filename) as f:
            lines = f.readlines()
        line_range = lines[start_line:end_line]
        return "\n".join(line_range)

    @env.macro
    def doc_env() -> dict[str, Any]:
        """Document the environment by returning visible attributes.

        Returns a dictionary containing all visible attributes of the environment
        object (those not starting with '_' or 'register').

        Returns:
            Dict[str, Any]: A dictionary mapping attribute names to their values
                           for all visible environment attributes.

        Example:
            ```markdown
            {{ doc_env() }}
            ```
        """
        return {
            name: getattr(env, name) for name in dir(env) if not (name.startswith("_") or name.startswith("register"))
        }

    @env.macro
    def render_with_page_template(page_template: str) -> str:
        """Render a page using a template from the templates directory.

        This macro loads a template file from the templates directory and
        processes it through the MkDocs markdown pipeline, allowing for
        dynamic content generation based on templates.

        Args:
            page_template: The name of the template file (without extension)
                         to load from the templates directory.

        Returns:
            str: The rendered markdown content after template processing.

        Example:
            ```markdown
            {{ render_with_page_template("project") }}
            ```

        For more detailed example usages, see:
          - streamlit-outline-display.md
          - streamlit-dcai-image-describe.md
          - streamlit-dcai-vortex-insights

        Raises:
            FileNotFoundError: If the template file does not exist
            IOError: If there are issues reading the template file
        """
        with open(f"templates/{page_template}.jinja") as template_file:
            return env.conf.plugins.on_page_markdown(
                template_file.read(),
                page=env.page,
                config=env.conf,
                files=[],
            )


def on_post_build(env: Environment) -> None:
    """Post-build hook for mkdocs-macros-plugin.

    This function is called after the MkDocs build process completes.
    It can be used to perform cleanup tasks or post-processing operations.

    Args:
        env: The mkdocs-macros-plugin environment object that provides access
            to the configuration and page context.
    """
    # activate trace
    chatter: Callable[..., None] = env.start_chatting(SIGNATURE)
    chatter("This means `on_post_build(env)` works")
