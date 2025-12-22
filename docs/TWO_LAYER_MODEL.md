# FractalStat Two-Layer Addressing Model

## Overview

FractalStat uses a two-layer addressing system. Understanding this distinction is critical for interpreting experiment results.

## Layer 1: Coordinate Space (Semantic Layer)

**What it is:**

- 7-dimensional semantic coordinates
- Pre-hash positioning in conceptual space
- Provides semantic meaning and queryability

**The 7 Dimensions:**

1. **realm**: Domain classification (data, narrative, system, etc.)
2. **lineage**: Generation from LUCA (temporal context)
3. **adjacency**: Relational neighbors (graph structure)
4. **horizon**: Lifecycle stage (genesis, peak, decay, etc.)
5. **resonance**: Charge/alignment (-1.0 to 1.0)
6. **velocity**: Rate of change (-1.0 to 1.0)
7. **density**: Compression distance (0.0 to 1.0)

**Properties:**

- Coordinate collisions ARE possible (same coordinates, different entities)
- Entropy depends on dimension design
- This is what we're testing in experiments

**Why it matters:**

- Enables coordinate-based queries ("find all in realm X")
- Provides semantic disambiguation
- Validates dimension design choices

## Layer 2: Address Space (Cryptographic Layer)

**What it is:**

- SHA-256 hash of canonical serialization
- 256-bit cryptographic address
- Post-hash uniqueness guarantee

**Properties:**

- Address collisions are cryptographically impossible
- 2^256 address space (~10^77 unique addresses)
- Birthday paradox: need ~2^128 attempts for 50% collision chance
- This layer is already proven by cryptography

**Why it matters:**

- Guarantees unique addresses regardless of coordinate entropy
- Enables content-addressable storage
- Provides cryptographic integrity verification

## The Key Distinction

### What We CAN Test

**Coordinate Space (Layer 1):**

- Do 7 dimensions provide sufficient entropy?
- How much does each dimension contribute?
- Are coordinates semantically meaningful?
- Can we query by coordinate properties?

**Implementation Correctness:**

- Does canonical serialization work?
- Is addressing deterministic?
- Are results reproducible?

### What We CANNOT Test

**SHA-256 Collision Resistance:**

- Already proven by cryptography
- Would require 2^128 attempts to test
- Not feasible at any realistic scale
- Not what our experiments measure

## Experiment Mapping

### EXP-01: Address Determinism

**Tests:** Layer 2 implementation (canonical serialization)
**Does NOT test:** SHA-256 collision resistance
**Value:** Validates our implementation is correct

### EXP-03: Coordinate Space Entropy

**Tests:** Layer 1 design (coordinate entropy)
**Does NOT test:** Address uniqueness (guaranteed by SHA-256)
**Value:** Validates dimension design provides semantic value

### EXP-11: Dimension Cardinality

**Tests:** Layer 1 optimization (how many dimensions needed)
**Does NOT test:** Hash collisions
**Value:** Finds optimal dimension count for coordinate space

## Historical Context

### What We Learned

**From previous experiments:**

- At 3 dimensions or fewer: coordinate collisions happen
- At 4+ dimensions: coordinate space is large enough
- At 7 dimensions: coordinate space is so large that SHA-256 becomes the only boundary

**The realization:**

- Collision testing is no longer a real talking point
- SHA-256 already guarantees address uniqueness
- What matters is coordinate space design

### Why We Reframed

**Old framing (misleading):**

- "Testing address uniqueness with zero hash collisions"
- Implied we're testing SHA-256
- Overstated value of collision measurements

**New framing (honest):**

- "Testing canonical serialization and coordinate entropy"
- Focuses on what we can actually learn
- Clarifies the value of each experiment

## Implications for Research

### What This Means

1. **Collision measurements are not the goal**
   - SHA-256 already guarantees uniqueness
   - Finding zero collisions tells us nothing new

2. **Coordinate entropy is the real metric**
   - How well do coordinates disambiguate?
   - Do dimensions provide semantic value?
   - Is 7 the minimal necessary set?

3. **Implementation correctness matters**
   - Canonical serialization must be deterministic
   - Cross-platform consistency is critical
   - Any "collision" = our bug, not SHA-256 failure

### Future Experiments

When designing new experiments, ask:

- **Layer 1 or Layer 2?** Which layer are we testing?
- **What can we learn?** What's not already proven?
- **Why does it matter?** What decision does this inform?

Don't test SHA-256 (it's proven). Test our design and implementation.

## Summary

**Two layers, two purposes:**

1. **Coordinate Space** (Layer 1)
   - Semantic positioning
   - Queryable properties
   - Design validation
   - **This is what we test**

2. **Address Space** (Layer 2)
   - Cryptographic uniqueness
   - Content-addressable storage
   - Integrity verification
   - **This is already proven**

Understanding this distinction makes experiments scientifically honest and practically valuable.
