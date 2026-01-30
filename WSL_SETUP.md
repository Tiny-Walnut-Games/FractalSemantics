# WSL Terminal Setup for VS Code

This document explains how to use WSL (Windows Subsystem for Linux) as your terminal in VS Code for this project.

## What's Been Set Up

✅ **WSL Ubuntu Distribution**: Installed and configured
✅ **Python Environment**: Python 3.12.3 with all project dependencies
✅ **Virtual Environment**: `.venv_wsl` in the project root
✅ **VS Code Integration**: Configured to use WSL as the default terminal

## How to Use WSL Terminal in VS Code

### Method 1: Open VS Code in WSL Mode
1. Open VS Code
2. Press `Ctrl+Shift+P` to open the Command Palette
3. Type "WSL: Connect to WSL"
4. Select "WSL: Connect to WSL" to open a new window connected to WSL

### Method 2: Use WSL Terminal in Current Window
1. Open the terminal in VS Code (`Ctrl+``)
2. Click the dropdown arrow next to the `+` button
3. Select "WSL: Ubuntu" from the list
4. A new WSL terminal will open

## Virtual Environment Activation

The virtual environment is automatically activated when you open a WSL terminal in this project. You should see:

```
(.venv_wsl) jerry@JerrysLaptop:/mnt/c/Users/jerio/RiderProjects/fractalstat$
```

If it doesn't activate automatically, you can manually activate it:

```bash
source .venv_wsl/bin/activate
```

## Available Dependencies

All project dependencies are installed in the WSL environment:
- pydantic, numpy, click
- torch, transformers, sentence-transformers
- fastapi, uvicorn
- pytest, black, ruff, mypy

## Troubleshooting

### Virtual Environment Not Activating
If the virtual environment doesn't activate automatically:
1. Make sure you're using the WSL terminal (not PowerShell/CMD)
2. Manually activate: `source .venv_wsl/bin/activate`
3. Check that the `.venv_wsl` directory exists

### WSL Terminal Not Appearing
1. Make sure the WSL extension is installed in VS Code
2. Restart VS Code after installing the WSL extension
3. Check that Ubuntu is properly installed: `wsl --list --verbose`

### Python Path Issues
The Python interpreter is configured to use the WSL virtual environment. If you see path issues:
1. Make sure you're in the WSL terminal
2. Check the Python path: `which python`
3. It should point to: `/mnt/c/Users/jerio/RiderProjects/fractalstat/.venv_wsl/bin/python`

## Benefits of Using WSL

- Full Linux terminal environment within Windows
- Access to Linux-specific tools and commands
- Better development experience for cross-platform projects
- Seamless integration with VS Code
- Proper Python virtual environment isolation

## Project-Specific Commands

Once in the WSL terminal with the virtual environment activated:

```bash
# Run the main application
python app.py

# Run tests
pytest

# Run experiments
python experiment_runner.py

# Format code
black .

# Check type hints
mypy .