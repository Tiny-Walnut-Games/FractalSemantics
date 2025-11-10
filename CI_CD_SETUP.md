# CI/CD Pipeline Setup Guide

## Overview
FractalStat has a basic three-stage GitLab CI/CD pipeline:
1. **test** - Runs pytest on all commits to main and MRs
2. **build** - Builds the Python package on tags and main
3. **deploy** - Publishes to HuggingFace (manual trigger on tags)

## Required GitLab CI Variables

Set these in **GitLab → Settings → CI/CD → Variables**:

### HuggingFace Credentials
- **HF_TOKEN**: Your HuggingFace API token
  - Get it from: https://huggingface.co/settings/tokens
  - Ensure token has write access to your models
  - Mark as **Protected** and **Masked**

- **HF_REPO_ID**: Your HuggingFace model repository ID
  - Format: `username/model-name`
  - Example: `tinywalnutgames/fractalstat-v1`

## Pipeline Behavior

### Test Stage
- **Trigger**: On every push to `main` or merge request
- **Action**: Installs dev dependencies and runs `pytest`
- **Artifacts**: Coverage report (if pytest-cov configured)

### Build Stage
- **Trigger**: On tags and main branch
- **Action**: Builds wheel and sdist packages
- **Artifacts**: Saved to `dist/` directory

### Deploy Stage
- **Trigger**: Manually on tagged releases
- **Action**: Uploads `dist/` folder to HuggingFace Model Hub
- **Requirements**: Build stage must pass

## Usage

### Create a Release
```bash
git tag v0.1.0
git push origin v0.1.0
```

This triggers build automatically. Then:
1. Go to **GitLab → CI/CD → Pipelines**
2. Find your tag pipeline
3. Click **Play** on the `deploy_huggingface` job

### Monitor Pipeline
- GitLab → Project → CI/CD → Pipelines
- View logs and artifacts for each stage

## Next Steps

### Optional Enhancements
- Add more Python versions (3.9, 3.10) to test stage
- Add linting (black, flake8)
- Add type checking (mypy)
- Add SAST/security scanning
- Create a model card for HuggingFace
- Add automatic version bumping

### Local Testing
Before pushing, test locally:
```bash
pip install -e ".[dev]"
pytest -v
python -m build
```
