# gsax — Global Sensitivity Analysis in JAX

## Context

Build a publishable Python package (`gsax`) that implements Sobol/Saltelli variance-based global sensitivity analysis using JAX. The package clones SALib's Sobol method but leverages JAX's `vmap` and `jit` for efficient computation over multiple time points and outputs simultaneously. The core value proposition: vectorized, JIT-compiled sensitivity index computation for models with high-dimensional outputs.

## Package Structure

```
gsax/
├── pyproject.toml          # uv, Python >=3.12, deps: jax, jaxlib, scipy
├── src/gsax/
│   ├── __init__.py         # exports: sample, analyze, Problem, SamplingResult, SAResult
│   ├── problem.py          # Problem dataclass
│   ├── sampling.py         # Saltelli sampling via scipy Sobol + matrix construction
│   ├── analyze.py          # JAX-based Sobol index computation with vmap/jit
│   ├── _indices.py         # Core JIT-able index functions (first_order, total_order, second_order)
│   ├── _bootstrap.py       # JAX-vectorized bootstrap CI computation
│   ├── results.py          # SAResult dataclass
│   └── benchmarks/
│       ├── __init__.py
│       └── ishigami.py     # Ishigami function + analytical solutions
├── tests/
│   ├── test_problem.py
│   ├── test_sampling.py
│   ├── test_indices.py     # Unit tests for core index functions
│   ├── test_analyze.py     # Integration tests with Ishigami benchmark
│   └── test_shapes.py      # Edge cases: 1 param, 1 timestep, 2D vs 3D input
└── ruff.toml               # Ruff config
```

## TODO — Implementation Checklist

### PR 1: Project scaffold
- [ ] Update `pyproject.toml` with dependencies (jax, jaxlib, scipy) and dev deps (pytest, ruff)
- [ ] Add `ruff.toml` config
- [ ] Create `src/gsax/__init__.py` with stub exports
- [ ] Create empty module files for all planned modules
- [ ] Create `tests/` directory with empty test files
- [ ] Verify `uv sync` installs everything correctly

### PR 2: Problem dataclass + Saltelli sampling
- [ ] Implement `Problem` frozen dataclass in `problem.py` (names, bounds, from_dict, num_vars)
- [ ] Implement `SamplingResult` frozen dataclass in `sampling.py` (samples, base_n, n_params, calc_second_order)
- [ ] Implement `sample()` function:
  - [ ] Back-calculate base N from desired n_samples: `ceil(n_samples / (2D+2))`
  - [ ] Round base N up to next power of 2
  - [ ] Generate Sobol sequence via `scipy.stats.qmc.Sobol(d=2*D, scramble=True)`
  - [ ] Construct Saltelli matrix (A, AB, BA, B interleaved) — stride `2D+2`
  - [ ] Scale samples from [0,1] to parameter bounds
  - [ ] Return `SamplingResult`
- [ ] Write `test_problem.py` — construction, from_dict, num_vars
- [ ] Write `test_sampling.py` — output shape, bounds, power-of-2 enforcement, back-calculation

### PR 3: Core JAX index computation
- [ ] Implement `first_order(A, AB_j, B)` in `_indices.py` — pure JAX, JIT-able
  - Formula: `mean(B * (AB_j - A)) / var([A, B])`
- [ ] Implement `total_order(A, AB_j, B)` — pure JAX, JIT-able
  - Formula: `0.5 * mean((A - AB_j)^2) / var([A, B])`
- [ ] Implement `second_order(A, AB_j, AB_k, BA_j, B)` — pure JAX, JIT-able
  - Formula: `mean(BA_j * AB_k - A * B) / var([A,B]) - S1_j - S1_k`
- [ ] Write `test_indices.py`:
  - [ ] Unit tests with known small arrays
  - [ ] JIT compatibility: `jax.jit(first_order)(...)` succeeds
  - [ ] Numerical correctness vs manual calculation

### PR 4: Bootstrap CIs in JAX
- [ ] Implement `bootstrap_indices()` in `_bootstrap.py`:
  - [ ] Generate resample indices: `(num_resamples, N)` using JAX PRNG
  - [ ] vmap `compute_one_resample` over all resamples
  - [ ] Each resample: reindex A, B, AB, BA → compute S1, ST for all params
  - [ ] S2 bootstrap: compute for all (j,k) pairs
  - [ ] CI = Z * std(resampled_indices, ddof=1), Z from normal ppf
  - [ ] Parameters: `key: PRNGKey`, `num_resamples=100`, `conf_level=0.95`
- [ ] Unit test bootstrap produces reasonable CI widths

### PR 5: analyze() with vmap over time/outputs
- [ ] Implement `SAResult` dataclass in `results.py`
  - Fields: S1, S1_conf, ST, ST_conf, S2, S2_conf (JAX arrays), problem
- [ ] Implement `analyze()` in `analyze.py`:
  - [ ] Accept Y as 2D `(n_total, K)` or 3D `(n_total, T, K)`
  - [ ] Auto-detect shape; if 2D expand to `(n_total, 1, K)`, flag squeeze_time
  - [ ] Normalize Y: `(Y - mean) / std` along sample axis per time/output
  - [ ] Separate output values into A, B, AB, BA using stride indexing
    - A: `(base_N, T, K)`, B: `(base_N, T, K)`
    - AB: `(base_N, D, T, K)`, BA: `(base_N, D, T, K)`
  - [ ] Define `_compute_for_single_output()` — computes S1, ST, S2 + CIs for one (t,k)
  - [ ] Nested vmap: inner over outputs (K), outer over time (T)
  - [ ] If squeeze_time, remove time dimension from results
  - [ ] Return `SAResult`
- [ ] Update `__init__.py` exports

### PR 6: Ishigami benchmark + full test suite
- [ ] Implement `benchmarks/ishigami.py`:
  - [ ] `evaluate(X, A=7.0, B=0.1)` — JAX implementation
  - [ ] `ANALYTICAL_S1 = [0.3139, 0.4424, 0.0]`
  - [ ] `ANALYTICAL_ST = [0.5576, 0.4424, 0.2437]`
  - [ ] `ANALYTICAL_S2 = {(0,2): 0.2437}` (others ~0)
  - [ ] `PROBLEM` — Problem instance for Ishigami (x1,x2,x3 in [-pi, pi])
- [ ] Write `test_analyze.py` — Ishigami validation:
  - [ ] Sample with base N=2^14
  - [ ] Run Ishigami, reshape Y to (n_total, 1)
  - [ ] Check S1, ST within 5% relative tolerance of analytical values
  - [ ] Check S2[0,2] ≈ 0.2437, S2[0,1] ≈ 0, S2[1,2] ≈ 0
- [ ] Write `test_shapes.py` — edge cases:
  - [ ] 2D input: time dim squeezed in output
  - [ ] 3D input: full (T, K, D) shape preserved
  - [ ] Single parameter (D=1): S2 has shape (1,1)
  - [ ] Single output (K=1): shapes correct
- [ ] Final `ruff check` + `ruff format` pass
- [ ] All `pytest` tests green

## Design Decisions (Locked In)

| Decision | Choice |
|---|---|
| Package name | `gsax` |
| Build system | `pyproject.toml` + `uv`, Python >=3.12, ruff for lint/format |
| Problem definition | Frozen dataclass, uniform bounds only, `{name: (low, high)}` dict constructor |
| Sampling | Scipy `qmc.Sobol` for sequence, Saltelli matrix construction in gsax |
| Sample count | User specifies desired total N, back-calculate base N, round up to power-of-2 |
| Workflow | Two-step: `sample()` → `SamplingResult` → user runs model → `analyze(result, Y)` |
| Output Y shape | 2D `(n_eval, K)` or 3D `(n_eval, T, K)`. Auto-detect, squeeze time if 2D. |
| Indices | S1 (first-order), ST (total-order), S2 (second-order) |
| CI method | JAX bootstrap, vmapped. `num_resamples=100`, `conf_level=0.95`, configurable |
| RNG | User passes `jax.random.PRNGKey` for bootstrap |
| vmap strategy | Nested: outer over time, inner over outputs |
| JIT | Fully JIT-compatible analysis step |
| API surface | Two public functions: `gsax.sample()`, `gsax.analyze()` |
| Results | `SAResult` dataclass with JAX arrays. Time dim squeezed if input was 2D |
| Testing | Ishigami benchmark (5% tolerance) + edge case tests |
| Sobol N | Enforce power-of-2 for base N (best low-discrepancy properties) |

## Key Formulas (from SALib / Saltelli 2010)

**First-order**: `S1_j = E[B * (AB_j - A)] / Var([A, B])`

**Total-order**: `ST_j = 0.5 * E[(A - AB_j)^2] / Var([A, B])`

**Second-order**: `S2_jk = E[BA_j * AB_k - A * B] / Var([A, B]) - S1_j - S1_k`

**Bootstrap CI**: `CI = Z * std(resampled_indices)`, where `Z = ppf(0.5 + conf/2)`

## Verification

1. `uv run ruff check src/ tests/` — no lint errors
2. `uv run ruff format src/ tests/` — formatted
3. `uv run pytest tests/ -v` — all tests pass
4. Ishigami S1, ST within 5% of analytical values
5. `jax.jit(gsax.analyze)` compiles without error
