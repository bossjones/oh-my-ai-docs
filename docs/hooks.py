#!/usr/bin/env python3
"""

Serving Raw Text Files with MkDocs
Based on your directory structure, you want to make all llms.txt files available as raw text files with the plain/text MIME type. There are several approaches to accomplish this with MkDocs. Here's a comprehensive guide to implementing this functionality.

Understanding How MkDocs Handles Non-Markdown Files
By default, MkDocs copies any non-Markdown files from your documentation directory to the built site directory without processing them. This means your llms.txt files are already being copied to the output directory, but you need to ensure they're served correctly.

plugins:
  - search
  - mkdocs-simple-hooks:
      hooks:
        on_post_build: "docs.hooks:copy_txt_files"
"""
from __future__ import annotations

import glob
import os
import shutil
from typing import Any, Dict, List, Tuple


def copy_txt_files(config: dict[str, str], **kwargs: Any) -> None:
    """Copy all llms.txt files to ensure they're available as raw text."""
    site_dir: str = config['site_dir']
    docs_dir: str = config['docs_dir']

    # Find all llms.txt files in the docs directory
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file == "llms.txt":
                # Get relative path from docs directory
                rel_path: str = os.path.relpath(os.path.join(root, file), docs_dir)
                # Create the destination path
                dest_path: str = os.path.join(site_dir, rel_path)
                # Ensure the destination directory exists
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                # Copy the file
                shutil.copy2(os.path.join(root, file), dest_path)
