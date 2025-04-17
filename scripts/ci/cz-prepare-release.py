#!/usr/bin/env python3
"""
# NOTE: This is a WIP and not ready for use yet.

cz-prepare-release.py - Automates the release preparation process using Commitizen

DESCRIPTION:
  This script automates the process of preparing a new release by:
  - Checking for uncommitted changes
  - Determining the next version based on conventional commits
  - Creating a release branch
  - Bumping version numbers
  - Running pre-commit hooks
  - Creating a pull request

REQUIREMENTS:
  - uv (Python package manager)
  - commitizen (cz)
  - pre-commit
  - gh (GitHub CLI, optional but recommended for PR creation)

USAGE:
  python scripts/ci/cz-prepare-release.py

ENVIRONMENT VARIABLES:
  PRERELEASE_PHASE - Optional. Set to 'alpha', 'beta', or 'rc' for prerelease versions
  CI              - Optional. If set, indicates running in CI environment

EXAMPLES:
  # Standard release
  python scripts/ci/cz-prepare-release.py

  # Beta release
  # Set PRERELEASE_PHASE=beta before running
  python scripts/ci/cz-prepare-release.py

EXIT CODES:
  0 - Success
  1 - Various error conditions (see error messages)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any, List, Optional, Union

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
        if is_dry_run():
            return dry_run_echo(cmd, shell)

        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()

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

def check_environment() -> bool:
    """Verify that the required tools are installed."""
    try:
        # Check for required commands
        for cmd in ["git", "uv", "gh"]:
            try:
                run_command(f"which {cmd}")
                print(f"✅ {cmd} is installed")
            except subprocess.CalledProcessError:
                if cmd == "gh":
                    print(f"⚠️ {cmd} is not installed (optional but recommended)")
                else:
                    print(f"❌ {cmd} is required but not installed")
                    return False
        return True
    except Exception as e:
        print(f"❌ Environment check failed: {e!s}")
        bpdb.pm()
        return False

def check_and_stash_uncommitted_changes() -> bool:
    """Check for uncommitted changes and stash them if necessary."""
    try:
        uncommitted = run_command("git status --porcelain")
        if uncommitted:
            print("⚠️ Stashing uncommitted changes...")
            run_command("git stash push --include-untracked --message \"Stash for release preparation\"")
        return True
    except Exception as e:
        print(f"❌ Failed to check/stash uncommitted changes: {e!s}")
        bpdb.pm()
        return False

def get_current_version() -> str | None:
    """Get the current version from Commitizen."""
    try:
        version = run_command("uv run cz version -p")
        if not version:
            print("❌ Failed to determine current version")
            return None
        print(f"✅ Current version: {version}")
        return version
    except Exception as e:
        print(f"❌ Failed to get current version: {e!s}")
        bpdb.pm()
        return None

def determine_new_version() -> str | None:
    """Determine the new version using Commitizen dry-run."""
    try:
        output = run_command("uv run cz bump --dry-run", check=False)
        match = re.search(r'bump: version .* → (.*)', output)
        if not match:
            print("❌ Failed to determine new version from Commitizen")
            print("Possible causes:")
            print("1. No conventional commits since last release")
            print("2. Version file inconsistencies")
            print("3. Invalid commit message format")
            return None
        version = match.group(1)
        return version
    except Exception as e:
        print(f"❌ Failed to determine new version: {e!s}")
        bpdb.pm()
        return None

def create_release_branch(version: str) -> str | None:
    """Create a release branch for the new version."""
    try:
        branch_name = f"task/prepare-release-{version}"

        # Check if branch already exists
        branches = run_command("git branch")
        if branch_name in branches:
            print(f"❌ Release branch {branch_name} already exists")
            print("Resolve conflicts or delete existing branch before proceeding")
            return None

        # Create branch
        run_command(f"git checkout -b {branch_name}")
        print(f"✅ Created branch {branch_name}")
        return branch_name
    except Exception as e:
        print(f"❌ Failed to create release branch: {e!s}")
        bpdb.pm()
        return None

def bump_version(prerelease_phase: str | None = None) -> bool:
    """Bump the version using Commitizen."""
    try:
        cmd: list[str] = ["uv", "run", "cz", "bump"]
        if prerelease_phase:
            if prerelease_phase not in ['alpha', 'beta', 'rc']:
                print(f"❌ Invalid prerelease phase: {prerelease_phase}")
                print("Valid options: alpha, beta, rc")
                return False
            cmd.extend(["--prerelease", prerelease_phase])
            print(f"🚧 Prerelease phase: {prerelease_phase}")

        run_command(cmd)
        return True
    except subprocess.CalledProcessError:
        print("❌ Version bump failed")
        print("Possible reasons:")
        print("- No version-changing commits since last release")
        print("- Conflicts in version files")
        bpdb.pm()
        return False
    except Exception as e:
        print(f"❌ Failed to bump version: {e!s}")
        bpdb.pm()
        return False

def run_pre_commit_hooks() -> bool:
    """Run pre-commit hooks on all files."""
    try:
        run_command("uv run pre-commit run -a --show-diff-on-failure")
        return True
    except subprocess.CalledProcessError:
        print("❌ Pre-commit checks failed - resolve formatting issues and retry")
        print("Some fixes may have been applied automatically - check git diff")
        bpdb.pm()
        return False
    except Exception as e:
        print(f"❌ Failed to run pre-commit hooks: {e!s}")
        bpdb.pm()
        return False

def verify_changes() -> bool:
    """Verify that files were changed after version bump and pre-commit."""
    try:
        changes = run_command("git diff --name-only")
        if not changes:
            print("❌ No files changed after version bump and pre-commit")
            print("Check Commitizen configuration and commit history")
            return False
        return True
    except Exception as e:
        print(f"❌ Failed to verify changes: {e!s}")
        bpdb.pm()
        return False

def commit_changes(current_version: str, new_version: str) -> bool:
    """Commit changes from version bump and pre-commit."""
    try:
        run_command("git add .")

        # Check if there are staged changes
        try:
            run_command("git diff --cached --quiet", check=True)
            print("❌ No changes to commit after version bump and pre-commit")
            return False
        except subprocess.CalledProcessError:
            # This is actually good - it means there are changes to commit
            pass

        commit_message = f"chore: bump version from {current_version} to {new_version}"
        run_command(["git", "commit", "-m", commit_message])
        print("✅ Committed version changes")
        return True
    except Exception as e:
        print(f"❌ Failed to commit changes: {e!s}")
        bpdb.pm()
        return False

def push_branch(branch_name: str) -> bool:
    """Push the branch to the remote repository if in CI environment."""
    try:
        if os.environ.get("CI"):
            print("-- Verifying branch existence on remote --")
            try:
                run_command(f"git ls-remote --exit-code origin {branch_name}")
                print("✅ Remote branch already exists")
            except subprocess.CalledProcessError:
                print("⬆️ Pushing new branch to remote")
                run_command(f"git push origin {branch_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to push branch: {e!s}")
        bpdb.pm()
        return False

def create_pull_request(branch_name: str, current_version: str, new_version: str) -> bool:
    """Create a pull request for the release branch."""
    try:
        # Check if gh is available and authenticated
        try:
            run_command("which gh")
            run_command("gh auth status", check=False)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ GitHub CLI not found or not authenticated")
            print(f"Create PR manually: https://github.com/bossjones/codegen-lab/compare/{branch_name}?expand=1")
            return True

        # Create release label if it doesn't exist
        try:
            run_command("gh label create release --description \"Label for marking official releases\" --color 28a745", check=False)
        except subprocess.CalledProcessError:
            # Label might already exist, that's fine
            pass

        # Create PR
        pr_title = f"Prepare for release of {current_version} to {new_version}"
        pr_body = f"Release preparation triggered by @{run_command('git config user.name')}.\n\nOnce merged, create a GitHub release for `{new_version}` to publish."

        cmd: list[str] = [
            "gh", "pr", "create",
            "--title", pr_title,
            "--body", pr_body,
            "--assignee", "@me",
            "--label", "release",
            "--fill",
            "--base", "main",
            "--head", branch_name
        ]

        run_command(cmd)
        print("✅ Created pull request")
        return True
    except Exception as e:
        print(f"❌ Failed to create PR: {e!s}")
        bpdb.pm()
        return False

def main() -> int:
    """Main function to run the release preparation process."""
    if is_dry_run():
        print("\n🔍 === DRY RUN MODE ENABLED ===")
        print("Commands will be displayed but not executed")
        print("Set DRY_RUN=0 to execute commands")
        print("================================\n")

    try:
        print("===== ENVIRONMENT PRE-CHECKS =====")
        if not check_environment():
            return 1

        if not check_and_stash_uncommitted_changes():
            return 1

        print("===== CURRENT VERSION CHECK =====")
        current_version = get_current_version()
        if current_version is None and not is_dry_run():
            return 1

        print("===== VERSION DETERMINATION =====")
        new_version = determine_new_version()
        if new_version is None and not is_dry_run():
            return 1

        if is_dry_run():
            # Use placeholder values in dry run mode
            current_version = "0.0.0"
            new_version = "0.0.0"
            branch_name = "task/prepare-release-0.0.0"
        else:
            print(f"✅ New version determined: {new_version} (from {current_version})")
            print("===== BRANCH MANAGEMENT =====")
            branch_name = create_release_branch(new_version)
            if not branch_name:
                return 1

        print("===== VERSION BUMP EXECUTION =====")
        prerelease_phase = os.environ.get("PRERELEASE_PHASE")
        if not bump_version(prerelease_phase):
            return 1

        print("===== RUNNING PRE-COMMIT HOOKS =====")
        if not run_pre_commit_hooks():
            return 1

        print("===== CHANGE VERIFICATION =====")
        if not verify_changes():
            return 1

        print("===== COMMIT SAFEGUARDS =====")
        if not commit_changes(current_version, new_version):
            return 1

        print("===== REMOTE SYNC CHECK =====")
        if not push_branch(branch_name):
            return 1

        print("===== PR CREATION SAFEGUARDS =====")
        if not create_pull_request(branch_name, current_version, new_version):
            return 1

        if is_dry_run():
            print("\n🔍 === DRY RUN COMPLETED ===")
            print("Above are all commands that would be executed")
            print("Set DRY_RUN=0 to execute commands")
            print("=============================")
        else:
            print("🎉 Release preparation complete with enhanced safeguards!")
        return 0

    except Exception as e:
        if not is_dry_run():
            print(f"❌ Unexpected error: {e!s}")
            print("Launching post-mortem debugger...")
            bpdb.pm()
        return 1

if __name__ == "__main__":
    sys.exit(main())
