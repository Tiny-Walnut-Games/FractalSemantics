# Constants Audit (Initial Pass)

Date: 2026-02-21
Scope: `fractalsemantics/*.py` (focused on runner, progress transport, and high-signal experiment modules)

## 1) Confirmed safe categories

### A. Display / formatting constants (safe)

- Precision specifiers and display rounding (e.g., duration/percent formatting)
- Progress bar width/limits and output truncation limits
- JSON pretty-print indentation and table width values

### B. Known domain / physics constants (likely intentional)

- Orbital and astrophysics scales in EXP-19/20/21
- Scientific thresholds documented in experiment narratives (e.g., entropy threshold in EXP-03)
- Unit conversions (e.g., seconds to milliseconds via `* 1000`)

## 2) Refactors applied in this pass

### `fractalsemantics/experiment_runner.py`

- Centralized UI/progress heuristics:
  - `PROGRESS_PERCENT_MIN`, `PROGRESS_PERCENT_MAX`
  - `PROGRESS_BAR_TOTAL`, `PROGRESS_BAR_COLUMNS`
  - `PROGRESS_STAGE_LABEL_MAX_CHARS`
  - `QUEUE_POLL_TIMEOUT_SECONDS`
  - `OUTPUT_DISPLAY_MAX_LINES`
  - `PROGRESS_MESSAGE_LIMIT`
  - `SEQUENTIAL_PROGRESS_MIN_INTERVAL_SECONDS`
  - `PROGRESS_SEPARATOR_EVERY`
- Replaced hard-coded values in queue polling, output compression, progress bar setup, and progress clamping.
- Normalized summary bullet indentation in the batch report section.

### `fractalsemantics/progress_comm.py`

- Centralized progress protocol constants:
  - `PROGRESS_PERCENT_MIN`, `PROGRESS_PERCENT_MAX`
  - `NON_PROGRESS_SENTINEL`
  - `PROGRESS_COMPLETION_PERCENT`
  - `MIN_MESSAGE_INTERVAL_SECONDS`
- Replaced raw sentinel/range/time literals in message construction and clamping.

## 3) Review candidates (not auto-refactored)

These values may be intentional, but should be reviewed and named if they represent policy/heuristics rather than science:

- `fractalsemantics/exp08_self_organizing_memory.py`
  - Similarity weights and thresholds (`0.35`, `0.2`, `0.15`, `0.1`, `0.8`, `0.2`)
- `fractalsemantics/exp10_multidimensional_query.py`
  - Query heuristics and penalties (`0.2`, `0.1`, `0.15`, `0.05`), sampling caps (`1000`)
- `fractalsemantics/exp11_dimension_cardinality.py`
  - Expressiveness weighting vector and improvement thresholds
- `fractalsemantics/exp12_benchmark_comparison.py`
  - Composite scoring weights (`0.25`, `0.20`, etc.) and fallback expressiveness scores
- `fractalsemantics/exp03_coordinate_entropy.py`
  - Composite contribution weights and thresholds (`5.0`, `1e-9`) are likely valid but should remain documented as scientific policy constants

## 4) Suggested next hardening step

Introduce per-module constant blocks in the above experiment files to separate:

- scientific-model constants,
- evaluation-policy thresholds,
- UI/reporting formatting constants.

That keeps model semantics explicit and prevents confusion between "measured value" vs "display/policy knob".

## 5) Phase-2 completion (this session)

Refactored policy/heuristic literals into named constants in:

- `fractalsemantics/exp08_self_organizing_memory.py`
  - clustering thresholds, similarity penalties, weights, neighbor limits, retrieval cutoff
- `fractalsemantics/exp10_multidimensional_query.py`
  - similarity defaults/penalties, filter thresholds, fallback sizes, accuracy baseline map, ratio penalties/bonuses
- `fractalsemantics/exp11_dimension_cardinality.py`
  - expressiveness weight map, diversity bonus factors, latency sample cap, diminishing-return threshold, 7D validation thresholds
- `fractalsemantics/exp12_benchmark_comparison.py`
  - baseline semantic/query flexibility scores, normalization floors, weighted overall score constants

Validation:

- No diagnostics errors in any of the four edited files.

## 6) Phase-3 completion (continuation)

Refactored additional non-physics policy/heuristic literals into named constants in:

- `fractalsemantics/exp02_retrieval_efficiency.py`
  - scale defaults and latency target map, warmup/query-mix policy ratios, payload/memory-pressure simulation knobs, CLI mode query-count defaults
- `fractalsemantics/exp06_entanglement_detection.py`
  - identity tolerance and non-identity score multiplier, identity base/bonus weights, synthetic-data generation knobs, iteration-level validation gates, default mode parameters
- `fractalsemantics/exp09_memory_pressure.py`
  - pressure-allocation caps/intervals, optimization estimate baselines, monitoring and fragmentation windows, success thresholds, progress-stage percentages, quick/full memory-mode defaults

Validation:

- No diagnostics errors in edited files (`exp02`, `exp06`, `exp09`).
- Quick smoke runs passed:
  - `fractalsemantics/exp02_retrieval_efficiency.py --quick`
  - `fractalsemantics/exp06_entanglement_detection.py --quick`
  - `fractalsemantics/exp09_memory_pressure.py --quick`

## 7) Final sweep (EXP-13 .. EXP-21)

Refactored additional policy/heuristic constants in:

- `fractalsemantics/exp13_fractal_gravity.py`
  - random seed, default element density fallback, flatness/falloff consistency thresholds, mode/config defaults, element-stage progress cap
- `fractalsemantics/exp14_atomic_fractal_mapping.py`
  - structure-success thresholds and threshold echo fields in results payload
- `fractalsemantics/exp15_topological_conservation.py`
  - topology full-conservation tolerance threshold (`0.999`) centralized
- `fractalsemantics/exp17_thermodynamic_validation.py`
  - synthetic evolution noise/growth factors, thermodynamic pass-rate threshold, density defaults, sampling window sizes, temperature scaling factors
- `fractalsemantics/exp18_falloff_thermodynamics.py`
  - synthetic evolution factors (with/without falloff), temperature scaling, CLI/default falloff exponent
- `fractalsemantics/exp20_vector_field_derivation.py`
  - inverse-square confirmation ratio threshold (`80%`) centralized
- `fractalsemantics/exp21_earth_moon_sun.py`
  - runtime defaults (simulation days/steps/coefficient), progress-stage mapping, tolerance percentages (Moon/Earth), generic fallback ratios, unit-conversion constants

Validation:

- Diagnostics:
  - No errors in `exp14` and `exp15`.
  - Existing unresolved import warnings remain in `exp17`, `exp18`, `exp20` for `fractalsemantics.progress_reporter` (pre-existing fallback path still present).
- Quick smoke runs:
  - `exp13 --quick`: PASSED
  - `exp14 --quick`: PASSED
  - `exp15 --quick`: FAILED (experiment criteria; execution healthy)
  - `exp17 --quick`: PASSED
  - `exp18 --quick`: FAILED (experiment criteria; execution healthy)
  - `exp20 --quick`: FAILED (experiment criteria; execution healthy)
  - `exp21 --quick`: FAILED (experiment criteria; execution healthy)
