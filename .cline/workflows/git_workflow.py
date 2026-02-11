#!/usr/bin/env python3
"""
Git Workflow for Cline

Comprehensive Git workflow automation including:
- Pre-commit hooks setup
- Branch management (feature, release, hotfix)
- Pull request preparation
- Release management and tagging
- Status checking and cleanup
- Changelog generation

Usage:
  /git-workflow                  - Show available commands
  /git-workflow setup-hooks      - Setup pre-commit hooks
  /git-workflow feature <name>   - Create feature branch
  /git-workflow release <version> - Create release branch
  /git-workflow pr               - Prepare pull request
  /git-workflow release-tag <version> - Create release tag
  /git-workflow status           - Check repository status
  /git-workflow cleanup          - Clean up merged branches
  /git-workflow changelog [tag]  - Generate changelog
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class GitWorkflow:
    """Comprehensive Git workflow management."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from .cline-git-config.json."""
        config_path = self.project_root / ".cline-git-config.json"

        default_config = {
            "base_branch": "main",
            "release_branch_prefix": "release/",
            "feature_branch_prefix": "feature/",
            "hotfix_branch_prefix": "hotfix/",
            "pre_commit_hooks": [
                "ruff",
                "black",
                "mypy",
                "safety"
            ],
            "protected_branches": [
                "main",
                "master",
                "develop"
            ],
            "auto_cleanup": True,
            "changelog_format": "keepachangelog"
        }

        if config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                # Merge with defaults
                return {**default_config, **user_config}
            except:
                pass

        return default_config

    def _run_command(self, cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
        """Run a command and return success status."""
        print(f"🔧 {description}")
        print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                check=False,
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(f"   ✅ {result.stdout.strip()}")
            if result.stderr:
                print(f"   ⚠️  {result.stderr.strip()}")

            return result.returncode == 0

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    def setup_hooks(self) -> bool:
        """Setup pre-commit hooks."""
        print("🔧 Setting up pre-commit hooks...")

        # Create .pre-commit-config.yaml
        pre_commit_config = {
            "repos": [
                {
                    "repo": "https://github.com/pre-commit/pre-commit-hooks",
                    "rev": "v4.4.0",
                    "hooks": [
                        {"id": "trailing-whitespace"},
                        {"id": "end-of-file-fixer"},
                        {"id": "check-yaml"},
                        {"id": "check-added-large-files"}
                    ]
                },
                {
                    "repo": "https://github.com/psf/black",
                    "rev": "22.3.0",
                    "hooks": [
                        {"id": "black"}
                    ]
                },
                {
                    "repo": "https://github.com/astral-sh/ruff-pre-commit",
                    "rev": "v0.0.253",
                    "hooks": [
                        {"id": "ruff"}
                    ]
                },
                {
                    "repo": "https://github.com/pre-commit/mirrors-mypy",
                    "rev": "v0.991",
                    "hooks": [
                        {"id": "mypy"}
                    ]
                }
            ]
        }

        config_path = self.project_root / ".pre-commit-config.yaml"
        with open(config_path, 'w') as f:
            json.dump(pre_commit_config, f, indent=2)

        print(f"   ✅ Created {config_path}")

        # Install pre-commit hooks
        success = self._run_command(
            ["pre-commit", "install"],
            "Installing pre-commit hooks"
        )

        return success

    def feature(self, name: str) -> bool:
        """Create and switch to feature branch."""
        print(f"🔧 Creating feature branch: {name}")

        base_branch = self.config.get("base_branch", "main")

        # Fetch latest from remote
        success = self._run_command(
            ["git", "fetch", "origin"],
            "Fetching latest changes"
        )
        if not success:
            return False

        # Create and switch to feature branch
        feature_branch = f"{self.config.get('feature_branch_prefix', 'feature/')}{name}"

        success = self._run_command(
            ["git", "checkout", "-b", feature_branch, f"origin/{base_branch}"],
            f"Creating feature branch {feature_branch}"
        )

        if success:
            # Push to remote with -u flag
            self._run_command(
                ["git", "push", "-u", "origin", feature_branch],
                f"Pushing {feature_branch} to remote"
            )

        return success

    def release(self, version: str) -> bool:
        """Create release branch."""
        print(f"🔧 Creating release branch: {version}")

        base_branch = self.config.get("base_branch", "main")
        release_branch = f"{self.config.get('release_branch_prefix', 'release/')}{version}"

        # Fetch latest from remote
        success = self._run_command(
            ["git", "fetch", "origin"],
            "Fetching latest changes"
        )
        if not success:
            return False

        # Create and switch to release branch
        success = self._run_command(
            ["git", "checkout", "-b", release_branch, f"origin/{base_branch}"],
            f"Creating release branch {release_branch}"
        )

        if success:
            # Push to remote with -u flag
            self._run_command(
                ["git", "push", "-u", "origin", release_branch],
                f"Pushing {release_branch} to remote"
            )

        return success

    def pr(self) -> bool:
        """Prepare pull request."""
        print("🔧 Preparing pull request...")

        # Check if we're on a feature branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        current_branch = result.stdout.strip()

        if not current_branch:
            print("   ❌ Not on a branch")
            return False

        # Update branch with base branch
        base_branch = self.config.get("base_branch", "main")
        success = self._run_command(
            ["git", "fetch", "origin"],
            "Fetching latest changes"
        )
        if not success:
            return False

        success = self._run_command(
            ["git", "rebase", f"origin/{base_branch}"],
            f"Rebasing {current_branch} on {base_branch}"
        )
        if not success:
            print("   ⚠️  Rebase failed, trying merge")
            success = self._run_command(
                ["git", "merge", f"origin/{base_branch}"],
                f"Merging {base_branch} into {current_branch}"
            )

        if not success:
            return False

        # Run quality checks
        print("   🔍 Running quality checks...")
        quality_checks = [
            (["ruff", "check", "."], "Ruff linting"),
            (["black", "--check", "."], "Black formatting check"),
            (["mypy", "."], "MyPy type checking")
        ]

        all_passed = True
        for cmd, description in quality_checks:
            success = self._run_command(cmd, description)
            if not success:
                all_passed = False

        if not all_passed:
            print("   ❌ Quality checks failed")
            return False

        # Push to remote
        success = self._run_command(
            ["git", "push", "origin", current_branch],
            f"Pushing {current_branch} to remote"
        )

        if success:
            print(f"   ✅ Ready for PR: {current_branch} -> {base_branch}")
            print("   💡 Create PR at: https://github.com/owner/repo/compare/main...feature-name")

        return success

    def release_tag(self, version: str) -> bool:
        """Create release tag."""
        print(f"🔧 Creating release tag: {version}")

        # Fetch latest from remote
        success = self._run_command(
            ["git", "fetch", "origin"],
            "Fetching latest changes"
        )
        if not success:
            return False

        # Create tag
        success = self._run_command(
            ["git", "tag", "-a", version, "-m", f"Release {version}"],
            f"Creating tag {version}"
        )
        if not success:
            return False

        # Push tag to remote
        success = self._run_command(
            ["git", "push", "origin", version],
            f"Pushing tag {version} to remote"
        )

        return success

    def status(self) -> bool:
        """Check repository status."""
        print("🔧 Checking repository status...")

        # Check working directory status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        status_output = result.stdout.strip()

        if status_output:
            print("   📁 Working directory status:")
            for line in status_output.split('\n'):
                print(f"      {line}")
        else:
            print("   ✅ Working directory clean")

        # Check current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        current_branch = result.stdout.strip()
        print(f"   🌿 Current branch: {current_branch}")

        # Check if up to date with remote
        result = subprocess.run(
            ["git", "status", "-uno"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        status_info = result.stdout.strip()

        if "Your branch is up to date" in status_info:
            print("   ✅ Branch up to date with remote")
        elif "Your branch is ahead" in status_info:
            print("   ⚠️  Branch ahead of remote")
        elif "Your branch is behind" in status_info:
            print("   ⚠️  Branch behind remote")
        else:
            print("   ❓ Unknown branch status")

        # Check for merge conflicts
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        conflicts = result.stdout.strip()

        if conflicts:
            print("   ❌ Merge conflicts detected:")
            for conflict in conflicts.split('\n'):
                print(f"      {conflict}")
        else:
            print("   ✅ No merge conflicts")

        return True

    def cleanup(self) -> bool:
        """Clean up merged branches."""
        print("🔧 Cleaning up merged branches...")

        # Fetch latest from remote
        success = self._run_command(
            ["git", "fetch", "--prune"],
            "Fetching and pruning remote branches"
        )
        if not success:
            return False

        # List merged branches
        result = subprocess.run(
            ["git", "branch", "--merged"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        merged_branches = result.stdout.strip().split('\n')

        protected_branches = self.config.get("protected_branches", ["main", "master", "develop"])
        branches_to_delete = []

        for branch in merged_branches:
            branch = branch.strip().lstrip('* ')
            if branch and branch not in protected_branches:
                branches_to_delete.append(branch)

        if not branches_to_delete:
            print("   ✅ No merged branches to clean up")
            return True

        print(f"   🗑️  Found {len(branches_to_delete)} merged branches to delete:")
        for branch in branches_to_delete:
            print(f"      {branch}")

        # Delete local branches
        for branch in branches_to_delete:
            self._run_command(
                ["git", "branch", "-d", branch],
                f"Deleting local branch {branch}"
            )

        # Delete remote branches
        for branch in branches_to_delete:
            self._run_command(
                ["git", "push", "origin", "--delete", branch],
                f"Deleting remote branch {branch}"
            )

        return True

    def changelog(self, tag: Optional[str] = None) -> bool:
        """Generate changelog."""
        print("🔧 Generating changelog...")

        if tag:
            # Generate changelog for specific tag
            since_commit = f"{tag}~1"
            print(f"   📝 Generating changelog since {tag}")
        else:
            # Generate full changelog
            since_commit = "HEAD~50"
            print("   📝 Generating recent changelog")

        # Get commit messages
        result = subprocess.run(
            ["git", "log", "--oneline", f"{since_commit}..HEAD"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        commits = result.stdout.strip().split('\n')

        if not commits or (len(commits) == 1 and not commits[0]):
            print("   ⚠️  No commits found")
            return True

        print("   📋 Recent commits:")
        for commit in commits[:20]:  # Show last 20 commits
            print(f"      {commit}")

        if len(commits) > 20:
            print(f"      ... and {len(commits) - 20} more commits")

        # Generate changelog file
        changelog_content = f"""# Changelog

Generated on {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}

## Recent Changes

"""

        for commit in commits:
            if commit:
                changelog_content += f"- {commit}\n"

        changelog_path = self.project_root / "CHANGELOG.md"
        with open(changelog_path, 'w') as f:
            f.write(changelog_content)

        print(f"   ✅ Generated {changelog_path}")

        return True

    def show_help(self) -> bool:
        """Show available commands."""
        print("🔧 Git Workflow Commands:")
        print()
        print("  setup-hooks              - Setup pre-commit hooks")
        print("  feature <name>           - Create feature branch")
        print("  release <version>        - Create release branch")
        print("  pr                       - Prepare pull request")
        print("  release-tag <version>    - Create release tag")
        print("  status                   - Check repository status")
        print("  cleanup                  - Clean up merged branches")
        print("  changelog [tag]          - Generate changelog")
        print()
        print("Configuration file: .cline-git-config.json")
        print("Pre-commit config: .pre-commit-config.yaml")

        return True


def main():
    """Main entry point for the workflow."""
    if len(sys.argv) < 2:
        workflow = GitWorkflow()
        success = workflow.show_help()
        sys.exit(0 if success else 1)

    command = sys.argv[1].lower()

    if command == "setup-hooks":
        workflow = GitWorkflow()
        success = workflow.setup_hooks()
        sys.exit(0 if success else 1)
    elif command == "feature" and len(sys.argv) > 2:
        workflow = GitWorkflow()
        success = workflow.feature(sys.argv[2])
        sys.exit(0 if success else 1)
    elif command == "release" and len(sys.argv) > 2:
        workflow = GitWorkflow()
        success = workflow.release(sys.argv[2])
        sys.exit(0 if success else 1)
    elif command == "pr":
        workflow = GitWorkflow()
        success = workflow.pr()
        sys.exit(0 if success else 1)
    elif command == "release-tag" and len(sys.argv) > 2:
        workflow = GitWorkflow()
        success = workflow.release_tag(sys.argv[2])
        sys.exit(0 if success else 1)
    elif command == "status":
        workflow = GitWorkflow()
        success = workflow.status()
        sys.exit(0 if success else 1)
    elif command == "cleanup":
        workflow = GitWorkflow()
        success = workflow.cleanup()
        sys.exit(0 if success else 1)
    elif command == "changelog":
        workflow = GitWorkflow()
        tag = sys.argv[2] if len(sys.argv) > 2 else None
        success = workflow.changelog(tag)
        sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  /git-workflow                  - Show available commands")
        print("  /git-workflow setup-hooks      - Setup pre-commit hooks")
        print("  /git-workflow feature <name>   - Create feature branch")
        print("  /git-workflow release <version> - Create release branch")
        print("  /git-workflow pr               - Prepare pull request")
        print("  /git-workflow release-tag <version> - Create release tag")
        print("  /git-workflow status           - Check repository status")
        print("  /git-workflow cleanup          - Clean up merged branches")
        print("  /git-workflow changelog [tag]  - Generate changelog")
        sys.exit(1)


if __name__ == "__main__":
    main()
