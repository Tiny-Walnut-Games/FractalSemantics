"""
Global Git Workflow for Cline

Comprehensive Git workflow automation including:
- Pre-commit hooks setup
- Branch management
- Pull request preparation
- Code review automation
- Release management
"""

import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class GitWorkflow:
    """Comprehensive Git workflow automation."""

    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path or os.getcwd())
        self.is_git_repo = self._check_git_repo()

    def _check_git_repo(self) -> bool:
        """Check if the current directory is a Git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False

    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"🔧 {description}")
        print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
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

    def setup_pre_commit_hooks(self) -> bool:
        """Set up pre-commit hooks for code quality."""
        print("🪝 Setting up pre-commit hooks...")

        # Create .pre-commit-config.yaml if it doesn't exist
        pre_commit_config = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
"""

        config_path = self.repo_path / ".pre-commit-config.yaml"
        if not config_path.exists():
            with open(config_path, 'w') as f:
                f.write(pre_commit_config.strip())
            print("   ✅ Created .pre-commit-config.yaml")

        # Install pre-commit hooks
        success = self._run_command(
            ["pre-commit", "install"],
            "Installing pre-commit hooks"
        )

        if success:
            # Run pre-commit on all files
            self._run_command(
                ["pre-commit", "run", "--all-files"],
                "Running pre-commit on all files"
            )

        return success

    def create_feature_branch(self, feature_name: str) -> bool:
        """Create and switch to a new feature branch."""
        print(f"🌿 Creating feature branch: {feature_name}")

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        result.stdout.strip()

        # Create new branch
        branch_name = f"feature/{feature_name}"
        success = self._run_command(
            ["git", "checkout", "-b", branch_name],
            f"Creating branch {branch_name}"
        )

        if success:
            print(f"   ✅ Switched to branch: {branch_name}")

        return success

    def create_release_branch(self, version: str) -> bool:
        """Create a release branch."""
        print(f"📦 Creating release branch: {version}")

        branch_name = f"release/{version}"
        success = self._run_command(
            ["git", "checkout", "-b", branch_name, "main"],
            f"Creating release branch {branch_name}"
        )

        if success:
            print(f"   ✅ Created release branch: {branch_name}")

        return success

    def prepare_pull_request(self, base_branch: str = "main") -> bool:
        """Prepare a pull request by updating branch and running checks."""
        print("📋 Preparing pull request...")

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        current_branch = result.stdout.strip()

        if current_branch == base_branch:
            print(f"   ⚠️  Already on base branch {base_branch}")
            return False

        # Fetch latest changes
        success = self._run_command(
            ["git", "fetch", "origin"],
            "Fetching latest changes"
        )

        if not success:
            return False

        # Rebase onto base branch
        success = self._run_command(
            ["git", "rebase", f"origin/{base_branch}"],
            f"Rebasing onto {base_branch}"
        )

        if not success:
            print("   ⚠️  Rebase failed, trying merge instead")
            success = self._run_command(
                ["git", "merge", f"origin/{base_branch}"],
                f"Merging {base_branch}"
            )

        if success:
            # Run pre-commit checks
            self._run_command(
                ["pre-commit", "run", "--all-files"],
                "Running pre-commit checks"
            )

            # Push to remote
            self._run_command(
                ["git", "push", "origin", current_branch, "-u"],
                "Pushing to remote"
            )

        return success

    def create_release(self, version: str, message: str = "") -> bool:
        """Create a new release."""
        print(f"🚀 Creating release: {version}")

        # Create tag
        tag_name = f"v{version}"
        commit_message = message or f"Release {version}"

        success = self._run_command(
            ["git", "tag", "-a", tag_name, "-m", commit_message],
            f"Creating tag {tag_name}"
        )

        if success:
            # Push tag
            self._run_command(
                ["git", "push", "origin", tag_name],
                "Pushing tag to remote"
            )

            print(f"   ✅ Created and pushed release tag: {tag_name}")

        return success

    def check_status(self) -> bool:
        """Check Git status and provide recommendations."""
        print("📊 Checking Git status...")

        # Check working tree status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            print("   📝 Untracked changes:")
            for line in result.stdout.strip().split('\n'):
                print(f"      {line}")
        else:
            print("   ✅ Working tree is clean")

        # Check current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        current_branch = result.stdout.strip()
        print(f"   🌿 Current branch: {current_branch}")

        # Check for unpushed commits
        result = subprocess.run(
            ["git", "status", "-uno"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if "Your branch is ahead" in result.stdout:
            print("   ⚠️  You have unpushed commits")
        elif "Your branch is up to date" in result.stdout:
            print("   ✅ Branch is up to date with remote")

        return True

    def cleanup_branches(self) -> bool:
        """Clean up local branches that have been merged."""
        print("🧹 Cleaning up merged branches...")

        # Fetch all remotes and prune
        self._run_command(
            ["git", "fetch", "--all", "--prune"],
            "Fetching and pruning remotes"
        )

        # List merged branches
        result = subprocess.run(
            ["git", "branch", "--merged"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        merged_branches = []
        for branch in result.stdout.strip().split('\n'):
            branch = branch.strip()
            if branch and branch not in ['main', 'master', 'develop']:
                merged_branches.append(branch)

        if merged_branches:
            print(f"   🗑️  Found {len(merged_branches)} merged branches to delete:")
            for branch in merged_branches:
                print(f"      - {branch}")

            # Delete merged branches
            for branch in merged_branches:
                self._run_command(
                    ["git", "branch", "-d", branch],
                    f"Deleting branch {branch}"
                )
        else:
            print("   ✅ No merged branches to clean up")

        return True

    def generate_changelog(self, since_tag: Optional[str] = None) -> bool:
        """Generate changelog from Git commits."""
        print("📝 Generating changelog...")

        # Get latest tag if not provided
        if not since_tag:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            since_tag = result.stdout.strip()

        if since_tag:
            # Generate changelog
            result = subprocess.run(
                ["git", "log", f"{since_tag}..HEAD", "--pretty=format:- %s"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            changelog_content = f"""# Changelog

## {datetime.datetime.now().strftime('%Y-%m-%d')}

### Changes since {since_tag}:

{result.stdout}

"""

            changelog_path = self.repo_path / "CHANGELOG.md"
            if changelog_path.exists():
                with open(changelog_path) as f:
                    existing_content = f.read()
                changelog_content = changelog_content + "\n" + existing_content

            with open(changelog_path, 'w') as f:
                f.write(changelog_content)

            print(f"   ✅ Generated changelog since {since_tag}")
        else:
            print("   ⚠️  No tags found, cannot generate changelog")

        return True


def main():
    """Main entry point for the Git workflow."""
    if len(sys.argv) < 2:
        print("Usage: python git_workflow.py <command> [args]")
        print("Commands:")
        print("  setup-hooks           Setup pre-commit hooks")
        print("  feature <name>        Create feature branch")
        print("  release <version>     Create release branch")
        print("  pr                    Prepare pull request")
        print("  release-tag <version> Create release tag")
        print("  status                Check Git status")
        print("  cleanup               Clean up merged branches")
        print("  changelog [tag]       Generate changelog")
        sys.exit(1)

    workflow = GitWorkflow()

    command = sys.argv[1]

    if command == "setup-hooks":
        success = workflow.setup_pre_commit_hooks()
    elif command == "feature" and len(sys.argv) > 2:
        success = workflow.create_feature_branch(sys.argv[2])
    elif command == "release" and len(sys.argv) > 2:
        success = workflow.create_release_branch(sys.argv[2])
    elif command == "pr":
        success = workflow.prepare_pull_request()
    elif command == "release-tag" and len(sys.argv) > 2:
        success = workflow.create_release(sys.argv[2])
    elif command == "status":
        success = workflow.check_status()
    elif command == "cleanup":
        success = workflow.cleanup_branches()
    elif command == "changelog":
        tag = sys.argv[2] if len(sys.argv) > 2 else None
        success = workflow.generate_changelog(tag)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
