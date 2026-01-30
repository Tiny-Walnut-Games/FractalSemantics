#!/usr/bin/env python3
"""
FractalStat Easy Installation Script

This script provides an automated way to install FractalStat with platform-specific optimizations.
It handles dependency installation, platform detection, and common troubleshooting scenarios.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


class FractalStatInstaller:
    """Automated installer for FractalStat with platform-specific optimizations."""

    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / 'venv'
        self.venv_python = self.venv_path / 'bin' / 'python' if self.system != 'windows' else self.venv_path / 'Scripts' / 'python.exe'
        self.venv_pip = self.venv_path / 'bin' / 'pip' if self.system != 'windows' else self.venv_path / 'Scripts' / 'pip.exe'

    def run_command(self, cmd, description="", check=True, shell=False):
        """Run a command with proper error handling."""
        print(f"🔧 {description}")
        print(f"   Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=shell,
                check=check,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.stdout:
                print(f"   ✅ {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {e}")
            if e.stderr:
                print(f"   Error: {e.stderr.strip()}")
            if check:
                sys.exit(1)
            return e

    def check_python_version(self):
        """Check if Python version is compatible."""
        print(f"🐍 Checking Python version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")

        if self.python_version < (3, 9):
            print("❌ Python 3.9+ is required")
            sys.exit(1)

        print("✅ Python version is compatible")

    def detect_platform(self):
        """Detect platform and architecture."""
        print("🖥️  Detecting platform...")

        is_arm = self.machine in ['arm64', 'aarch64']
        is_raspberry_pi = False

        if self.system == 'linux' and is_arm:
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'Raspberry Pi' in cpuinfo:
                        is_raspberry_pi = True
            except (FileNotFoundError, PermissionError, OSError) as e:
                print(f"Warning: Could not detect Raspberry Pi: {e}")
                pass

        platform_info = {
            'system': self.system,
            'machine': self.machine,
            'is_arm': is_arm,
            'is_raspberry_pi': is_raspberry_pi,
            'is_apple_silicon': self.system == 'darwin' and self.machine == 'arm64'
        }

        print(f"   System: {platform_info['system']}")
        print(f"   Architecture: {platform_info['machine']}")
        print(f"   ARM: {platform_info['is_arm']}")
        print(f"   Raspberry Pi: {platform_info['is_raspberry_pi']}")
        print(f"   Apple Silicon: {platform_info['is_apple_silicon']}")

        return platform_info

    def create_virtual_environment(self):
        """Create a virtual environment for the project."""
        print("🔧 Creating virtual environment...")

        if self.venv_path.exists():
            print(f"   Virtual environment already exists at {self.venv_path}")
            return

        self.run_command([
            sys.executable, '-m', 'venv', str(self.venv_path)
        ], "Creating virtual environment")

        print(f"   Created virtual environment at {self.venv_path}")

    def install_system_dependencies(self, platform_info):
        """Install system-level dependencies."""
        if platform_info['system'] == 'linux':
            if platform_info['is_raspberry_pi']:
                print("🍓 Installing Raspberry Pi system dependencies...")
                self.run_command([
                    'sudo', 'apt', 'update'
                ], "Updating package lists", check=False)

                self.run_command([
                    'sudo', 'apt', 'install', '-y',
                    'libopenblas-dev', 'libblas-dev', 'liblapack-dev',
                    'libatlas-base-dev', 'gfortran', 'build-essential'
                ], "Installing BLAS/LAPACK for PyTorch", check=False)

            elif platform_info['system'] == 'linux':
                print("🐧 Installing Linux system dependencies...")
                self.run_command([
                    'sudo', 'apt', 'install', '-y', 'build-essential'
                ], "Installing build tools", check=False)

        elif platform_info['system'] == 'darwin':
            print("🍎 macOS detected - no additional system dependencies needed")

        elif platform_info['system'] == 'windows':
            print("🪟 Windows detected - no additional system dependencies needed")

    def install_pytorch(self, platform_info):
        """Install PyTorch with platform-specific optimizations."""
        print("🔥 Installing PyTorch...")

        # Determine PyTorch installation command
        if platform_info['is_raspberry_pi']:
            # Raspberry Pi - CPU only
            pytorch_cmd = [
                str(self.venv_pip), 'install',
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ]
        elif platform_info['is_apple_silicon']:
            # Apple Silicon Macs
            pytorch_cmd = [
                str(self.venv_pip), 'install',
                'torch', 'torchvision', 'torchaudio'
            ]
        else:
            # Standard installation
            pytorch_cmd = [
                str(self.venv_pip), 'install',
                'torch', 'torchvision', 'torchaudio'
            ]

        self.run_command(pytorch_cmd, "Installing PyTorch", check=False)

    def install_fractalstat(self, dev=False, minimal=False):
        """Install FractalStat dependencies."""
        print("🎯 Installing FractalStat dependencies...")

        if minimal:
            # Install only core dependencies
            core_deps = [
                'pydantic>=2.0.0',
                'numpy>=1.20.0',
                'click>=8.1.0'
            ]
            if self.python_version < (3, 11):
                core_deps.append('tomli>=2.0.0')

            for dep in core_deps:
                self.run_command([
                    str(self.venv_pip), 'install', dep
                ], f"Installing {dep}")

        else:
            # Install from requirements.txt
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                install_cmd = [str(self.venv_pip), 'install', '-r', str(requirements_file)]
                self.run_command(install_cmd, "Installing dependencies from requirements.txt")
            else:
                print(f"   Warning: requirements.txt not found at {requirements_file}")
                # Fallback to core dependencies
                core_deps = [
                    'pydantic>=2.0.0',
                    'numpy>=1.20.0',
                    'click>=8.1.0'
                ]
                if not minimal:
                    core_deps.extend([
                        'torch>=2.0.0',
                        'transformers>=4.30.0',
                        'sentence-transformers>=2.2.0'
                    ])

                for dep in core_deps:
                    self.run_command([
                        str(self.venv_pip), 'install', dep
                    ], f"Installing {dep}")

            # Install development dependencies if requested
            if dev:
                dev_deps = [
                    'pytest>=7.0.0',
                    'black>=22.0.0',
                    'ruff>=0.1.0',
                    'mypy>=1.0.0'
                ]
                for dep in dev_deps:
                    self.run_command([
                        str(self.venv_pip), 'install', dep
                    ], f"Installing dev dependency {dep}")

    def test_installation(self):
        """Test that FractalStat dependencies were installed correctly."""
        print("🧪 Testing installation...")

        try:
            # Test that core dependencies can be imported
            self.run_command([
                str(self.venv_python), '-c', 'import pydantic, numpy, click; print("Core dependencies imported successfully!")'
            ], "Testing core dependencies")

            # Test that the fractalstat module can be imported (by adding current dir to path)
            self.run_command([
                str(self.venv_python), '-c', '''
import sys
sys.path.insert(0, ".")
import fractalstat
print("FractalStat modules imported successfully!")
'''
            ], "Testing FractalStat modules")

            print("✅ Installation test passed!")
            return True

        except Exception as e:
            print(f"❌ Installation test failed: {e}")
            return False

    def create_launcher_script(self, platform_info):
        """Create a convenient launcher script."""
        print("📜 Creating launcher script...")

        # Get relative path to venv
        venv_rel_path = self.venv_path.relative_to(self.project_root)

        launcher_content = f'''#!/bin/bash
# FractalStat Launcher Script

echo "🚀 Starting FractalStat..."

# Activate virtual environment
source {venv_rel_path}/bin/activate

# Set environment based on first argument
if [ "$1" = "dev" ]; then
    export FRACTALSTAT_ENV=dev
    echo "Using development configuration (faster, smaller samples)"
elif [ "$1" = "ci" ]; then
    export FRACTALSTAT_ENV=ci
    echo "Using CI configuration"
else
    export FRACTALSTAT_ENV=production
    echo "Using production configuration"
fi

# Run experiments
python -m fractalstat.fractalstat_experiments
'''

        launcher_path = self.project_root / 'run_experiments.sh'
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)

        # Make executable on Unix-like systems
        if platform_info['system'] != 'windows':
            os.chmod(launcher_path, 0o755)

        print(f"   Created: {launcher_path}")

    def show_post_installation_info(self, platform_info):
        """Show post-installation information."""
        print("\n" + "="*60)
        print("🎉 FractalStat Installation Complete!")
        print("="*60)

        print("\n📚 Quick Start:")
        print("  ./run_experiments.sh dev    # Fast development mode")
        print("  ./run_experiments.sh        # Full production mode")

        print("\n🧪 Test Individual Experiments:")
        print("  source venv/bin/activate && python -m fractalstat.exp01_geometric_collision")
        print("  source venv/bin/activate && python -m fractalstat.exp02_retrieval_efficiency")

        print("\n🐍 Virtual Environment:")
        print("  All dependencies are installed in a virtual environment (venv/)")
        print("  The launcher script automatically activates the venv")
        print("  To manually activate: source venv/bin/activate")

        if platform_info['is_raspberry_pi']:
            print("\n🍓 Raspberry Pi Notes:")
            print("  - Experiments may run slower on ARM")
            print("  - Use 'dev' mode for faster testing")
            print("  - Monitor memory usage with 'htop'")
            print("  - Consider external cooling for long runs")

        print("\n📖 Documentation:")
        print("  - Installation Guide: INSTALL.md")
        print("  - Experiment Details: docs/")
        print("  - Configuration: fractalstat/config/")

        print("\n🔧 Troubleshooting:")
        print("  - Check logs for error messages")
        print("  - Use 'dev' mode for memory-constrained systems")
        print("  - See INSTALL.md for detailed troubleshooting")

        print("\n✨ Happy experimenting with FractalStat!")

    def run(self, args):
        """Main installation workflow."""
        print("🎯 FractalStat Automated Installer")
        print("="*50)

        # Preliminary checks
        self.check_python_version()
        platform_info = self.detect_platform()

        # Create virtual environment
        self.create_virtual_environment()

        # Installation steps
        if not args.skip_system_deps:
            self.install_system_dependencies(platform_info)

        if not args.minimal:
            self.install_pytorch(platform_info)

        self.install_fractalstat(dev=args.dev, minimal=args.minimal)

        # Post-installation
        self.create_launcher_script(platform_info)

        if not args.skip_test:
            success = self.test_installation()
            if not success and not args.force:
                print("❌ Installation test failed. Use --force to continue anyway.")
                sys.exit(1)

        self.show_post_installation_info(platform_info)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Automated installer for FractalStat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                    # Full installation
  python install.py --dev             # Include development tools
  python install.py --minimal         # Core dependencies only
  python install.py --skip-system-deps # Skip system package installation

For Raspberry Pi:
  python install.py --minimal         # Lightweight installation
        """
    )

    parser.add_argument(
        '--dev', action='store_true',
        help='Install development dependencies (pytest, mypy, etc.)'
    )

    parser.add_argument(
        '--minimal', action='store_true',
        help='Install only core dependencies (no ML libraries)'
    )

    parser.add_argument(
        '--skip-system-deps', action='store_true',
        help='Skip installation of system-level dependencies'
    )

    parser.add_argument(
        '--skip-test', action='store_true',
        help='Skip post-installation testing'
    )

    parser.add_argument(
        '--force', action='store_true',
        help='Continue even if tests fail'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.minimal and args.dev:
        print("❌ Cannot use --minimal and --dev together")
        sys.exit(1)

    # Run installer
    installer = FractalStatInstaller()
    installer.run(args)


if __name__ == '__main__':
    main()
