# Methods

gsax implements two complementary approaches to variance-based global sensitivity analysis.

## Sobol Indices (Saltelli Sampling)

Sobol indices decompose a model's output variance into contributions from each input parameter and their interactions. gsax uses the Saltelli sampling scheme, which constructs a structured set of quasi-random sample matrices so that first-order (S1), total-order (ST), and second-order (S2) indices can be estimated from a single batch of model evaluations.

**How it works:**

1. `gsax.sample()` generates a Sobol quasi-random sequence (via `scipy.stats.qmc.Sobol`) and builds the Saltelli cross-matrices. Duplicate rows are removed so your model only evaluates unique points.
2. You evaluate your model on `sampling_result.samples`.
3. `gsax.analyze()` reconstructs the Saltelli layout internally and computes all indices in a single `jit(vmap(...))` pass.

**What the indices mean:**

| Index | Meaning |
|-------|---------|
| S1(i) | Fraction of output variance due to parameter i alone (main effect). |
| ST(i) | Fraction of output variance due to parameter i including all its interactions. ST >= S1 always. |
| S2(i,j) | Fraction of output variance due to the pairwise interaction between i and j, beyond their individual effects. |

**When to use Sobol:** You can afford to run the structured Saltelli sampling design and want exact, model-free variance decomposition.

## RS-HDMR (Random Sampling High-Dimensional Model Representation)

RS-HDMR fits a B-spline surrogate to decompose a model's input-output relationship into additive component functions (main effects, pairwise interactions, and optionally triple interactions). Sensitivity indices are derived from an ANCOVA (analysis of covariance) decomposition of the surrogate's variance.

**How it works:**

1. You provide any set of `(X, Y)` pairs -- no structured sampling design required.
2. `gsax.analyze_hdmr()` normalizes inputs to [0, 1], builds B-spline basis matrices, and fits component functions via backfitting with Tikhonov regularization.
3. The ANCOVA decomposition splits each component's variance into structural (Sa) and correlative (Sb) parts. Total-order indices (ST) sum contributions from all terms involving a given parameter.

**HDMR-specific indices:**

| Index | Meaning |
|-------|---------|
| Sa(t) | Structural (uncorrelated) variance contribution of term t. For first-order terms, equivalent to Sobol S1. |
| Sb(t) | Correlative variance contribution of term t (due to input correlations). |
| S(t) | Total contribution per term: Sa + Sb. |
| ST(i) | Total-order per parameter: sum of S for all terms involving parameter i. |

**When to use HDMR:**
- Model evaluations are expensive and you want to reuse existing runs
- Inputs may be correlated (Sobol assumes independent inputs)
- You need a surrogate/emulator for fast prediction at new inputs

## Choosing Between Them

| Consideration | Sobol | HDMR |
|---------------|-------|------|
| Sampling requirement | Structured Saltelli design | Any (X, Y) pairs |
| Input independence | Assumed | Handled via ANCOVA decomposition |
| Surrogate/emulator | No | Yes (`emulate_hdmr`) |
| Accuracy | Exact (given enough samples) | Depends on B-spline fit quality |
| Second-order indices | Direct estimation | From interaction component functions |

## Output Shapes

Both methods support scalar, multi-output, and time-series outputs. The shape of `Y` determines the shape of all returned index arrays:

| Y shape | S1 / ST shape | S2 shape |
|---------|---------------|----------|
| `(N,)` | `(D,)` | `(D, D)` |
| `(N, K)` | `(K, D)` | `(K, D, D)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K, D, D)` |

D is always the last axis. Confidence interval arrays (when using bootstrap) prepend a leading dimension of 2 for `[lower, upper]`.

## Data Cleaning

`gsax.analyze()` automatically drops sample groups that contain non-finite values (NaN, Inf). The Saltelli layout requires groups of rows to stay together, so if any row in a group is non-finite, the entire group is removed. A message is printed when this happens. The `nan_counts` field on the result reports how many NaN values remain in the computed indices.
