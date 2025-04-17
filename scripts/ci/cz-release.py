#!/usr/bin/env python3
"""
# NOTE: This is a WIP and not ready for use yet.
cz-release.py - Creates a GitHub release for the current version using Commitizen

DESCRIPTION:
  This script automates the GitHub release creation process by:
  - Verifying GitHub CLI installation and authentication
  - Determining the current version from Commitizen
  - Creating a GitHub release with auto-generated release notes

REQUIREMENTS:
  - gh (GitHub CLI)
  - uv (Python package manager)
  - commitizen (cz)

USAGE:
  python scripts/ci/cz-release.py

ENVIRONMENT VARIABLES:
  DRY_RUN - Optional. If set to "1", "true", or "yes", commands will be displayed but not executed

NOTE:
  This script is typically run after cz-prepare-release.py and the release PR
  has been merged to create the official GitHub release.

EXIT CODES:
  0 - Success
  1 - Various error conditions (see error messages)
    - GitHub CLI not found
    - GitHub CLI not authenticated
    - Version determination failed
    - Release creation failed
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional, Union

import bpdb


def is_dry_run() -> bool:
    """Check if we're in dry run mode."""
    return os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

def dry_run_echo(cmd: str | list[str], shell: bool = False) -> str:
    """Echo what would be run in dry run mode."""
    cmd_str = ' '.join(cmd) if isinstance(cmd, list) else cmd
    print(f"🔍 [DRY RUN] Would execute: {cmd_str}")
    return ""  # Always return empty string in dry run mode

def run_command(
    cmd: str | list[str],
    capture_output: bool = True,
    check: bool = True,
    shell: bool = False
) -> str:
    """Run a command and return the output."""
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()

        if is_dry_run():
            return dry_run_echo(cmd, shell)

        print(f"[running] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
            shell=shell
        )
        return result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"Error output: {e.stderr}")

        # Launch post-mortem debugger
        print("Launching post-mortem debugger to help troubleshoot...")
        bpdb.pm()

        raise

def check_github_cli() -> bool:
    """Check for GitHub CLI and authentication."""
    try:
        # Check if gh is installed
        try:
            run_command("which gh")
            print("✅ GitHub CLI is installed")
        except subprocess.CalledProcessError:
            print("❌ GitHub CLI (gh) not found")
            return False

        # Verify authentication
        try:
            run_command("gh auth status", check=False)
            print("✅ GitHub CLI is authenticated")
            return True
        except subprocess.CalledProcessError:
            print("❌ GitHub CLI not authenticated")
            return False

    except Exception as e:
        print(f"❌ GitHub CLI check failed: {e!s}")
        bpdb.pm()
        return False

def get_current_version() -> str | None:
    """Get the current version from Commitizen."""
    try:
        version = run_command("uv run cz version -p")
        if not version and not is_dry_run():
            print("❌ Failed to determine current version")
            return None
        if is_dry_run():
            version = "0.0.0"  # Placeholder version for dry run
        print(f"✅ Current version: {version}")
        return version
    except Exception as e:
        print(f"❌ Failed to get current version: {e!s}")
        bpdb.pm()
        return None

def create_github_release(version: str) -> bool:
    """Create a GitHub release for the specified version."""
    try:
        run_command(["gh", "release", "create", f"v{version}", "--generate-notes"])
        print(f"🎉 Successfully created release v{version}")
        return True
    except Exception as e:
        print(f"❌ Release creation failed: {e!s}")
        bpdb.pm()
        return False

def main() -> int:
    """Main function to run the release creation process."""
    try:
        if is_dry_run():
            print("\n🔍 === DRY RUN MODE ENABLED ===")
            print("Commands will be displayed but not executed")
            print("Set DRY_RUN=0 to execute commands")
            print("================================\n")

        print("===== GITHUB RELEASE CREATION =====")

        if not check_github_cli():
            return 1

        print("-- Determining current version --")
        version = get_current_version()
        if version is None:
            return 1

        print(f"-- Creating release v{version} --")
        if not create_github_release(version):
            return 1

        if is_dry_run():
            print("\n🔍 === DRY RUN COMPLETED ===")
            print("Above are all commands that would be executed")
            print("Set DRY_RUN=0 to execute commands")
            print("=============================")

        return 0

    except Exception as e:
        if not is_dry_run():
            print(f"❌ Unexpected error: {e!s}")
            print("Launching post-mortem debugger...")
            bpdb.pm()
        return 1

if __name__ == "__main__":
    sys.exit(main())
