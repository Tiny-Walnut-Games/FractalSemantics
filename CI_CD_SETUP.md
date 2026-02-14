# CI/CD Pipeline Setup Guide

## Overview

FractalSemantics has a comprehensive four-stage GitLab CI/CD pipeline:

1. **Quality** - Code formatting, linting, type checking
2. **Validate** - Run experiments to prove system correctness
3. **Build** - Create Python package
4. **Deploy** - Publish to PyPI, HuggingFace, or API endpoints

## Pipeline Stages

### Quality Stage (Parallel Jobs)

#### `code_format`

- **Trigger**: On merge requests and main branch
- **Tool**: Black (code formatter)
- **Action**: Checks if code adheres to Black formatting standards
- **Fail Policy**: Hard fail - MRs cannot merge if formatting is incorrect

**Local equivalent:**

```bash
black --check fractalsemantics/
black fractalsemantics/  # to auto-fix
```

#### `lint`

- **Trigger**: On merge requests and main branch
- **Tool**: Ruff (fast Python linter)
- **Action**: Checks for style violations, unused imports, etc.
- **Fail Policy**: Hard fail

**Local equivalent:**

```bash
ruff check fractalsemantics/
ruff check --fix fractalsemantics/  # auto-fix
```

#### `type_check`

- **Trigger**: On merge requests and main branch
- **Tool**: MyPy (static type checker)
- **Action**: Validates type hints and catches type errors
- **Fail Policy**: Soft fail (warnings only) - helps catch issues early

**Local equivalent:**

```bash
mypy fractalsemantics/ --ignore-missing-imports
```

### Validate Stage (Experiment-Based)

#### `validate_experiments`

- **Trigger**: On merge requests and main branch
- **Duration**: Up to 2 hours
- **Action**: Runs 9 nested and standalone experiments sequentially:

  **Phase 1 Doctrine (nested in `fractalsemantics_experiments.py`)**:
  - **EXP-01**: Address Uniqueness Test
  - **EXP-02**: Retrieval Efficiency Test
  - **EXP-03**: Dimension Necessity Test
  
  **Phase 2+ Experiments (separate files)**:
  - **EXP-04**: Fractal Scaling (`exp04_fractal_scaling`)
  - **EXP-05**: Compression/Expansion (`exp05_compression_expansion`)
  - **EXP-06**: Entanglement Detection (`exp06_entanglement_detection`)
  - **EXP-07**: LUCA Bootstrap (`exp07_luca_bootstrap`)
  - **EXP-08**: RAG Integration (`exp08_rag_integration`)
  - **EXP-09**: Concurrency & Thread Safety (`exp09_concurrency`)

- **Artifacts**: Experiment reports (.json files) stored for 30 days
- **Retry**: Automatically retries once on runner failure
- **Fail Policy**: Hard fail - must pass to merge

#### `stress_tests` (EXP-10: Bob Stress Test)

- **Trigger**: On merge requests and main branch
- **Duration**: Up to 1 hour
- **Action**: Runs Bob Stress Test (`bob_stress_test`) - comprehensive high-volume query testing
- **Fail Policy**: Soft fail (optional) - reports issues but doesn't block merges
- **Artifacts**: Stress test results stored for 30 days
- **Purpose**: Validates system stability under sustained load with concurrent queries

### Build Stage

#### `build_package`

- **Trigger**: On tags and main branch
- **Action**:
  - Runs build transformation script
  - Creates wheel and source distributions
  - Uploads to `dist/` directory
- **Artifacts**: Python packages (wheel, sdist)

### Deploy Stage (Manual Triggers)

#### `deploy_pypi`

- **Trigger**: Manual on tagged releases
- **Requires**: Build stage success
- **Action**: Uploads package to PyPI (Python Package Index)
- **Environment**: <https://pypi.org/project/fractalsemantics/>

#### `deploy_huggingface`

- **Trigger**: Manual on tagged releases
- **Requires**: Build stage success
- **Action**: Uploads package to HuggingFace Model Hub
- **Environment**: <https://huggingface.co/${HF_REPO_ID}>

#### `deploy_api_dev` & `deploy_api_prod`

- **Trigger**: Manual
- **Action**: Deploys as API service (configurable)
- **Environments**: Development and Production

## Required GitLab CI Variables

Set these in **GitLab → Project → Settings → CI/CD → Variables**:

### For HuggingFace Deployment

- **HF_TOKEN**: Your HuggingFace API token
  - Get from: <https://huggingface.co/settings/tokens>
  - Mark as **Protected** and **Masked**
  - Requires write permissions

- **HF_REPO_ID**: Your HuggingFace model repository
  - Format: `username/model-name`
  - Example: `tinywalnutgames/fractalsemantics-v1`

### For PyPI Deployment (Optional)

- **PYPI_API_TOKEN**: Your PyPI API token
  - Get from: <https://pypi.org/manage/account/>
  - Mark as **Protected** and **Masked**

### For API Deployment (Optional)

- **API_DEV_URL**: Development API endpoint
- **API_PROD_URL**: Production API endpoint
- **API_AUTH_TOKEN**: Authentication token for deployments

## Workflow Examples

### Submit a Merge Request

1. Push to a feature branch: `git push origin feature/new-experiment`
2. Open merge request on GitLab
3. Pipeline automatically runs:
   - [Success] Quality checks (black, ruff, mypy)
   - [Success] Validation experiments (all exp*_*.py modules)
   - [Success] Stress tests
4. If all pass → MR can be merged; if fail → fix issues and re-push

### Create a Release

1. **Tag on main branch:**

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

2. **Pipeline automatically:**
   - Runs all quality & validation checks
   - Builds package if checks pass
   - Waits for manual deployment trigger

3. **Deploy to PyPI (manual):**
   - Go to **GitLab → CI/CD → Pipelines**
   - Find your tag pipeline
   - Click **Play** (▶) on `deploy_pypi` job

4. **Deploy to HuggingFace (manual):**
   - Same as PyPI, trigger `deploy_huggingface` job

### Run Experiments Locally Before Pushing

```bash
# Install in dev mode
pip install -e ".[dev]"

# Check code quality
black --check fractalsemantics/
ruff check fractalsemantics/
mypy fractalsemantics/ --ignore-missing-imports

# Run all 9 Phase 1-2+ experiments (EXP-01 through EXP-09)
# Phase 1 Doctrine (nested: EXP-01, EXP-02, EXP-03)
python -m fractalsemantics.fractalsemantics_experiments

# Phase 2+ experiments (separate files)
python -m fractalsemantics.exp04_fractal_scaling
python -m fractalsemantics.exp05_compression_expansion
python -m fractalsemantics.exp06_entanglement_detection
python -m fractalsemantics.exp07_luca_bootstrap
python -m fractalsemantics.exp08_rag_integration
python -m fractalsemantics.exp09_concurrency

# Run EXP-10: Bob Stress Test (optional, high-volume load testing)
python -m fractalsemantics.bob_stress_test

# Build package
python copy_and_transform.py
python -m build
```

- **Total: 10 Experiments**

- 3 Phase 1 Doctrine experiments nested in `fractalsemantics_experiments.py` (EXP-01, EXP-02, EXP-03)
- 6 Phase 2+ experiments in separate files (EXP-04 through EXP-09)
- 1 Stress testing framework (EXP-10: Bob Stress Test)

## Cache Strategy

The pipeline uses caching for:

- **`.cache/pip`**: pip package cache (speeds up dependency installation)
- **`venv/`**: Virtual environment (if needed)

This reduces pipeline runtime significantly by avoiding re-downloading dependencies.

## Monitoring Pipeline

- **GitLab Dashboard**: Project → CI/CD → Pipelines
- **View Logs**: Click any job to see detailed output
- **Download Artifacts**: Click job → Artifacts → Download
- **Pipeline Status Badge**: Add to README:

  ```markdown
  [![pipeline status](https://gitlab.com/tiny-walnut-games/the-seed/badges/main/pipeline.svg)](https://gitlab.com/tiny-walnut-games/the-seed/-/commits/main)
  ```

## Next Steps for Enhancement

1. **Multi-version Testing**: Add Python 3.9, 3.10 testing alongside 3.11
2. **Performance Benchmarking**: Compare experiment results across releases
3. **Artifact Reports**: Generate HTML reports for experiments
4. **Slack Notifications**: Alert team on build failures
5. **Scheduled Nightly Tests**: Run full suite nightly with alerts
6. **Deployment Status Tracking**: Monitor API deployment health
7. **Rollback Automation**: Add automatic rollback on deployment failure

## Troubleshooting

### Black/Ruff Failures

```bash
# Auto-fix locally
black fractalsemantics/
ruff check --fix fractalsemantics/
git add .
git commit -m "Fix code formatting"
git push
```

### Experiments Timeout

- Increase `timeout:` value in `.gitlab-ci.yml`
- Split long experiments into separate jobs
- Run only critical experiments in pre-merge checks

### Deploy Failures

- Verify token is set and not expired
- Check token permissions in service (PyPI/HuggingFace)
- Verify repository/model IDs are correct
- Check GitLab variable scope (protected vs. unprotected branches)

### Build Failures

- Run `python copy_and_transform.py` locally to verify
- Check that all dependencies in `pyproject.toml` are available
- Verify Python version compatibility in GitHub Actions logs
