# FractalStat Academic Publication Automation

Automated workflow for preparing FractalStat for academic publication.

## Quick Setup

### 1. Create Milestones

Go to Project → Plan → Milestones and create:

- **Academic Publication** (Due: target submission date)
- **Peer Review Round 1**
- **Peer Review Round 2**  
- **Camera Ready**

### 2. Create Labels

Go to Project → Manage → Labels and create:

- `experiment` (color: #1F77B4)
- `validation` (color: #FF7F0E)
- `documentation` (color: #2CA02C)
- `peer-review` (color: #D62728)
- `publication-ready` (color: #9467BD)
- `statistical-analysis` (color: #8C564B)
- `reproducibility` (color: #E377C2)

### 3. Automated Tagging Strategy

Use semantic commit messages for automatic versioning:

```bash
# New experiment → minor version bump
git commit -m "feat(experiment): add EXP-11 validation"

# Bug fix → patch version bump  
git commit -m "fix(exp04): correct scaling calculation"

# Documentation → no version bump
git commit -m "docs: update methodology section"
```

**Version Format:** `v1.0.0`, `v1.1.0`, `v1.0.1`

### 4. Automated Release Process

When you're ready to create a publication version:

```bash
# Tag the release
git tag -a v1.0.0 -m "Publication version 1.0.0"
git push origin v1.0.0
```

This triggers:
1. Full experiment validation
2. Artifact archival
3. DOI metadata generation
4. Changelog creation
5. GitLab release with notes

## Workflow Phases

### Phase 1: Experiment Validation

1. Create issue for each experiment
2. Implement and validate
3. Create merge request
4. Get peer review
5. Merge to main

### Phase 2: Documentation

1. Write paper sections
2. Generate figures and tables
3. Compile bibliography
4. Peer review

### Phase 3: Publication Prep

1. Close all milestone issues
2. Run full validation suite
3. Archive all artifacts
4. Create release tag

### Phase 4: Submission

1. Get DOI from Zenodo
2. Submit to arXiv/journal
3. Update citations

## Reproducibility Checklist

For each experiment:

- [ ] Random seeds documented
- [ ] Dependencies locked
- [ ] Execution time recorded
- [ ] Hardware specs documented
- [ ] Results archived
- [ ] Code frozen at tag

## Publication Artifacts

Each release generates:

```
publication/
├── VERSION
├── CITATION.cff
├── doi_metadata.json
├── validation_report.pdf
├── figures/
├── tables/
└── archives/
```

## Next Steps

1. Create milestones and labels (see above)
2. Review existing experiments
3. Create issues for any missing validation
4. Begin documentation
5. Set up Zenodo for DOI
