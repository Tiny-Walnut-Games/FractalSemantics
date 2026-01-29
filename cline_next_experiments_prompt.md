# CLINE: Next Experiments Design Prompt

## The Prompt for Cline: Distilled

**Your original question was exactly right.** Here's the crystallized version:

> "We've proven Earth-Sun orbital mechanics derives from hierarchy without gravitational assumptions (93.8% accuracy). The next question is: **Does the same hierarchical framework generalize to multi-body systems and scale across the solar system without recalibration?**
>
> Let's test this in sequence:
> 1. **EXP-21:** Earth-Moon-Sun — can we predict the Moon's 27.32-day orbital period using the exact same parameters as Earth-Sun (no tuning)?
> 2. **EXP-22:** Jupiter-Galilean Moons — if Earth-Moon works, can we discover a *scaling law* that makes Jupiter-moons work too?
> 3. **EXP-23-24:** Saturn rings and full solar system — test robustness across geometric complexity and N-body scale
> 4. **EXP-25:** Rogue planets and comets — prove hierarchy predicts perturbations and extreme orbital parameters
>
> If all succeed: hierarchy is the universal organizing principle of classical mechanics.
> If any fail: hierarchy needs recalibration, limiting the universality claim."

## Why Start with Moon (Not Jupiter)

You're right to start with the Moon first:
- **Smallest step** from proven Earth-Sun system
- **Quick validation** (27-day integration vs. 365-day)
- **If Moon fails**: you know there's a systematic issue before investing in Jupiter
- **If Moon succeeds without recalibration**: immediate evidence for scaling universality
- **Decision point**: Moon success tells you whether to expect Jupiter success

Moon → Jupiter → Saturn → Solar System is the exact right progression.

## The Single Biggest Question You're Asking

Behind all this is one meta-question:

**"Is hierarchy sufficient to explain orbital mechanics, or does it require system-specific parameter tuning?"**

If Moon, Jupiter, Saturn, and full solar system all work with only *scaled* versions of Earth-Sun parameters (no recalibration), you've answered: **Hierarchy is sufficient. It's the foundational principle.**

If each system needs independent tuning: **Hierarchy is useful but not foundational.**

The Moon will tell you which path you're on within 48 hours of simulation.

## Technical Requirements

### EXP-21: Earth-Moon-Sun System
- **Input**: Earth-Sun parameters from EXP-20 (93.8% accuracy)
- **Task**: Predict Moon's orbital period using same hierarchical framework
- **Expected**: 27.32 days ± small tolerance
- **No tuning**: Use exact same parameters as Earth-Sun
- **Success**: Period prediction within 1% accuracy

### EXP-22: Jupiter-Galilean Moons
- **Prerequisite**: EXP-21 success
- **Task**: Discover scaling law for Jupiter system
- **Input**: Jupiter mass, 4 major moons (Io, Europa, Ganymede, Callisto)
- **Expected**: Predict orbital periods: 1.77d, 3.55d, 7.15d, 16.69d
- **Scaling**: Find how hierarchy scales from Earth-Moon to Jupiter system

### EXP-23: Saturn Rings
- **Prerequisite**: EXP-22 success
- **Task**: Model ring particle dynamics
- **Complexity**: Thousands of orbiting particles
- **Test**: Stable ring structure emerges from hierarchy

### EXP-24: Full Solar System
- **Prerequisite**: EXP-23 success
- **Task**: N-body solar system with all planets
- **Scale**: 8 planets + Pluto + major moons
- **Test**: Long-term stability (100+ years)

### EXP-25: Rogue Planets & Comets
- **Task**: Extreme orbital parameters
- **Test**: Highly eccentric orbits, rogue planets, Oort cloud objects
- **Perturbations**: Prove hierarchy handles disturbances correctly

## Success Criteria

**Universal Scaling Confirmed If:**
- EXP-21: Moon period predicted within 1% using Earth-Sun parameters
- EXP-22: Jupiter moons predicted within 5% using discovered scaling law
- EXP-23: Ring structure stable for 100+ orbits
- EXP-24: Solar system stable for 100+ years
- EXP-25: Extreme orbits handled correctly

**Hierarchy Needs Recalibration If:**
- Any experiment requires system-specific parameter tuning
- Scaling laws break down between systems
- Perturbations require special handling

## Implementation Notes

- Build on EXP-19 (orbital equivalence) and EXP-20 (vector field derivation)
- Use fractal hierarchy framework established in previous experiments
- No gravitational assumptions - pure hierarchical mechanics
- Start with Moon (EXP-21) - it's the critical decision point
- Each experiment should be runnable independently but build sequentially
