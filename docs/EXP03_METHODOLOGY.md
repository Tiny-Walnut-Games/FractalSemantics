# EXP-03 Methodology: Coordinate Space Entropy Test

**Experiment**: EXP-03  
**Name**: Coordinate Space Entropy Test  
**Version**: 2.0 (Entropy-Based)  
**Date**: 2025-11-18

## Abstract

This document describes the methodology for EXP-03, which quantifies the entropy contribution of each STAT7 dimension to the coordinate space. Unlike the previous collision-based approach, this entropy-based method measures information content directly in the coordinate space before hashing, providing deeper insights into semantic disambiguation power.

## Hypothesis

**Primary Hypothesis**: Each STAT7 dimension contributes measurable entropy to the coordinate space. Removing or omitting a dimension reduces entropy and semantic clarity, even if hash collisions remain at 0%.

**Secondary Hypothesis**: Dimensions with higher entropy contribution are more critical for semantic disambiguation.

**Null Hypothesis**: Some dimensions contribute negligible entropy (<5% reduction when removed) and could be considered optional.

## STAT7 Dimensions

The seven dimensions under test:

1. **realm** - Domain classification (data, narrative, system, faculty, event, pattern, void)
2. **lineage** - Generation from LUCA (Last Universal Common Ancestor)
3. **adjacency** - Relational neighbor mapping (graph connections)
4. **horizon** - Lifecycle stage (genesis, emergence, peak, decay, crystallization)
5. **resonance** - Charge/alignment (-1.0 to 1.0)
6. **velocity** - Rate of change (-1.0 to 1.0)
7. **density** - Compression distance (0.0 to 1.0)

## Methodology

### Phase 1: Baseline Measurement

**Objective**: Establish baseline entropy with all 7 dimensions.

**Procedure**:
1. Generate N random bit-chains (default: N=1000) with deterministic seed
2. Extract coordinate representations using all 7 dimensions
3. Compute Shannon entropy of coordinate distribution
4. Normalize entropy to [0, 1] scale
5. Calculate semantic disambiguation score
6. Record unique coordinate count

**Metrics**:
- Shannon Entropy (H): Information content in bits
- Normalized Entropy (H_norm): H / log₂(N)
- Unique Coordinates: Count of distinct coordinate combinations
- Disambiguation Score: Weighted combination of uniqueness and uniformity

### Phase 2: Ablation Testing

**Objective**: Measure entropy contribution of each dimension individually.

**Procedure** (repeated for each dimension):
1. Remove one dimension from coordinate representation
2. Extract coordinates using remaining 6 dimensions
3. Compute Shannon entropy of reduced coordinate space
4. Calculate entropy reduction percentage from baseline
5. Determine if reduction exceeds threshold (5%)
6. Record all metrics

**Ablation Order**: realm, lineage, adjacency, horizon, resonance, velocity, density

### Phase 3: Analysis

**Objective**: Rank dimensions by entropy contribution and identify critical dimensions.

**Procedure**:
1. Sort dimensions by entropy reduction percentage (descending)
2. Identify critical dimensions (>5% reduction)
3. Identify optional dimensions (<5% reduction)
4. Generate visualization of entropy contributions
5. Compile summary statistics

## Mathematical Framework

### Shannon Entropy

Shannon entropy measures the average information content of a random variable:

```
H(X) = -Σ p(xᵢ) · log₂(p(xᵢ))
```

Where:
- X is the random variable (coordinate values)
- p(xᵢ) is the probability of observing coordinate value xᵢ
- The sum is over all unique coordinate values

**Properties**:
- H(X) ≥ 0 (non-negative)
- H(X) = 0 when all coordinates are identical (no information)
- H(X) = log₂(N) when all N coordinates are unique (maximum information)

### Entropy Normalization

To enable comparison across different sample sizes:

```
H_norm = H(X) / H_max
H_max = log₂(N)
```

Where:
- H_norm ∈ [0, 1]
- H_norm = 1 indicates maximum entropy (all unique)
- H_norm = 0 indicates minimum entropy (all identical)

### Entropy Reduction

Quantifies the information loss when a dimension is removed:

```
ΔH% = ((H_baseline - H_ablation) / H_baseline) × 100%
```

Where:
- ΔH% is the percentage entropy reduction
- H_baseline is entropy with all 7 dimensions
- H_ablation is entropy with 6 dimensions (one removed)

**Interpretation**:
- ΔH% > 5%: Dimension is CRITICAL (significant information loss)
- ΔH% ≤ 5%: Dimension is OPTIONAL (minimal information loss)

### Semantic Disambiguation Score

Measures how well coordinates separate semantically different entities:

```
D = 0.7 × U + 0.3 × E_uniform
```

Where:
- D is the disambiguation score ∈ [0, 1]
- U is uniqueness ratio = (unique coordinates) / (total samples)
- E_uniform is distribution uniformity based on entropy

**Rationale**:
- Uniqueness (70% weight): Primary indicator of separation
- Uniformity (30% weight): Ensures even distribution, not clustering

## Experimental Parameters

### Default Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample Size | 1,000 | Balance between statistical power and runtime |
| Random Seed | 42 | Reproducibility for peer review |
| Entropy Threshold | 5% | Distinguishes critical from optional dimensions |
| Uniqueness Weight | 0.7 | Prioritizes separation over distribution |
| Uniformity Weight | 0.3 | Ensures balanced coordinate space |

### Alternative Configurations

**Quick Mode** (--quick):
- Sample Size: 100
- Use Case: Rapid iteration during development

**Full Mode** (--full):
- Sample Size: 5,000
- Use Case: Publication-quality validation

## Success Criteria

### Primary Criteria

1. **Baseline Entropy**: H_norm ≥ 0.95 (near-maximum information content)
2. **All Dimensions Critical**: All 7 dimensions show ΔH% > 5%
3. **Semantic Disambiguation**: D ≥ 0.90 for baseline
4. **Reproducibility**: Identical results with same random seed

### Secondary Criteria

1. **Dimension Ranking**: Clear ordering by entropy contribution
2. **Statistical Significance**: Results stable across multiple runs
3. **Visualization Quality**: Publication-ready figures generated

## Data Collection

### Coordinate Extraction

For each bit-chain, extract canonical coordinate representation:

```python
{
  "adjacency": ["id1", "id2"],  # Sorted list
  "density": 0.12345679,         # Normalized float
  "horizon": "peak",
  "lineage": 42,
  "realm": "data",
  "resonance": 0.87654321,       # Normalized float
  "velocity": -0.23456789        # Normalized float
}
```

**Canonicalization**:
- Keys sorted alphabetically
- Floats normalized to 8 decimal places
- Adjacency list sorted
- JSON serialization with no whitespace

### Metrics Recorded

**Per Test** (baseline + 7 ablations):
- Dimensions used (list)
- Sample size (int)
- Shannon entropy (float, bits)
- Normalized entropy (float, [0,1])
- Entropy reduction % (float)
- Unique coordinates (int)
- Semantic disambiguation score (float, [0,1])
- Meets threshold (bool)

**Summary**:
- Sample size
- Random seed
- Baseline entropy
- Total tests (8)
- All critical (bool)
- Results array

## Statistical Analysis

### Entropy Distribution

Expected baseline entropy for N=1000 samples:
- **Theoretical Maximum**: log₂(1000) ≈ 9.97 bits
- **Expected Range**: 9.5 - 9.97 bits (95-100% of maximum)
- **Minimum Acceptable**: 9.47 bits (95% of maximum)

### Dimension Contribution

Expected entropy reduction ranges:
- **High Contribution**: ΔH% > 15% (e.g., realm, lineage)
- **Medium Contribution**: 5% < ΔH% ≤ 15% (e.g., adjacency, horizon)
- **Low Contribution**: ΔH% ≤ 5% (potentially optional)

### Confidence Intervals

With N=1000 samples:
- **95% CI for entropy**: ±0.1 bits
- **99% CI for entropy**: ±0.15 bits

## Reproducibility

### Random Seed Control

All randomness is controlled via a single seed parameter:
- Bit-chain generation: `seed + i` for i-th bit-chain
- Coordinate values: Deterministic from seed
- Adjacency lists: Deterministic UUID generation

### Platform Independence

Entropy calculations use:
- **numpy**: Cross-platform numerical library
- **Python's Counter**: Deterministic frequency counting
- **JSON serialization**: Canonical string representation

### Verification

To verify reproducibility:
1. Run experiment with seed=42
2. Record baseline entropy
3. Re-run with same seed
4. Verify baseline entropy matches to 4 decimal places

## Visualization

### Entropy Contribution Chart

**Type**: Horizontal bar chart  
**X-axis**: Entropy Reduction (%)  
**Y-axis**: STAT7 Dimensions  
**Colors**: Green (>5%), Orange (≤5%)  
**Threshold Line**: Red dashed line at 5%

**Elements**:
- Bar labels with exact percentages
- Threshold indicator
- Title and axis labels
- Legend

**Output**: PNG file at 300 DPI (publication quality)

## Limitations

### Sample Size

- **Current**: N=1000 (default)
- **Limitation**: May not capture rare coordinate combinations
- **Mitigation**: Full mode with N=5000 for publication

### Dimension Independence

- **Assumption**: Dimensions contribute independently
- **Reality**: Some dimensions may be correlated
- **Future Work**: Measure joint entropy and mutual information

### Threshold Selection

- **Current**: 5% reduction threshold
- **Limitation**: Somewhat arbitrary cutoff
- **Justification**: Based on information theory conventions

## Future Enhancements

### Planned Improvements

1. **Joint Entropy Analysis**: Measure entropy of dimension pairs
2. **Mutual Information**: Quantify dimension dependencies
3. **Conditional Entropy**: Entropy of dimension given others
4. **Dimension Subsets**: Test all 2^7 = 128 combinations
5. **STAT8 Testing**: Evaluate 8th dimension candidates

### Research Questions

1. What is the optimal number of dimensions?
2. Are there redundant dimensions?
3. How does entropy scale with sample size?
4. What is the minimum dimension set for 95% entropy?

## References

### Information Theory

1. Shannon, C. E. (1948). "A Mathematical Theory of Communication". *Bell System Technical Journal*, 27(3), 379-423.

2. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

3. MacKay, D. J. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.

### STAT7 System

1. FractalStat Issue #3: EXP-03 Coordinate Space Entropy Test
2. FractalStat Issue #37: Collision detection at lower dimensions
3. FractalStat EXP-11: Dimension Cardinality Analysis

## Appendix A: Coordinate Examples

### Example 1: High Entropy (All Unique)
```
Coordinates: ["coord_0", "coord_1", "coord_2", ..., "coord_999"]
Shannon Entropy: 9.97 bits
Normalized Entropy: 1.00
Interpretation: Maximum information content
```

### Example 2: Low Entropy (Many Duplicates)
```
Coordinates: ["same", "same", "same", ..., "same"]
Shannon Entropy: 0.00 bits
Normalized Entropy: 0.00
Interpretation: No information content
```

### Example 3: Medium Entropy (50% Unique)
```
Coordinates: ["A", "A", "B", "B", "C", "C", ...]
Shannon Entropy: ~8.97 bits
Normalized Entropy: ~0.90
Interpretation: Good but not maximum information
```

## Appendix B: Calculation Examples

### Shannon Entropy Calculation

Given coordinates: ["A", "A", "B", "C"]

1. Count frequencies: {A: 2, B: 1, C: 1}
2. Calculate probabilities: {A: 0.5, B: 0.25, C: 0.25}
3. Apply formula:
   ```
   H = -(0.5 × log₂(0.5) + 0.25 × log₂(0.25) + 0.25 × log₂(0.25))
   H = -(0.5 × -1 + 0.25 × -2 + 0.25 × -2)
   H = -(-0.5 - 0.5 - 0.5)
   H = 1.5 bits
   ```

### Normalization

Given H = 1.5 bits, N = 4 samples:
```
H_max = log₂(4) = 2 bits
H_norm = 1.5 / 2 = 0.75
```

### Entropy Reduction

Given H_baseline = 9.8 bits, H_ablation = 8.5 bits:
```
ΔH% = ((9.8 - 8.5) / 9.8) × 100%
ΔH% = (1.3 / 9.8) × 100%
ΔH% = 13.27%
```

Interpretation: Dimension is CRITICAL (>5% reduction)

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-18  
**Status**: Final
