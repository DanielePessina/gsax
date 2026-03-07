# HDMR

## `gsax.analyze_hdmr()`

Compute sensitivity indices via RS-HDMR with B-spline surrogate modelling. Works with any (X, Y) pairs.

```python
def analyze_hdmr(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    lambdax: float = 0.01,
    chunk_size: int = 2048,
) -> HDMRResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | `Problem` | required | Parameter names and bounds (used to normalize X to [0, 1]). |
| `X` | `Array` | required | Input samples, shape `(N, D)`. |
| `Y` | `Array` | required | Model outputs. Shape `(N,)`, `(N, K)`, or `(N, T, K)`. A 2D array is always interpreted as `(N, K)`; for time-series with a single output, reshape to `(N, T, 1)`. |
| `maxorder` | `int` | `2` | Maximum expansion order. 1 = main effects, 2 = + pairwise, 3 = + triple. |
| `maxiter` | `int` | `100` | Maximum backfitting iterations for first-order terms. |
| `m` | `int` | `2` | B-spline intervals. Each dimension gets `m + 3` basis functions. |
| `lambdax` | `float` | `0.01` | Tikhonov regularization parameter. |
| `chunk_size` | `int` | `2048` | `(T, K)` outputs per `jit(vmap(...))` call. |

**Returns:** [`HDMRResult`](#hdmrresult)

---

## `gsax.emulate_hdmr()`

Predict at new input points using the fitted HDMR surrogate.

```python
def emulate_hdmr(result: HDMRResult, X_new: Array) -> Array
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `HDMRResult` | Result from `analyze_hdmr()`. |
| `X_new` | `Array` | New input points, shape `(N_new, D)`. |

**Returns:** `Array` of shape `(N_new,)`, `(N_new, K)`, or `(N_new, T, K)` matching the output layout used during `analyze_hdmr()`.

---

## `HDMRResult`

Dataclass holding HDMR sensitivity indices and emulator data.

```python
class HDMREmulator(TypedDict):
    C1: Array
    C2: Array | None
    C3: Array | None
    f0: Array
    m: int
    maxorder: int
    c2: list[tuple[int, int]]
    c3: list[tuple[int, int, int]]

@dataclass
class HDMRResult:
    Sa: Array
    Sb: Array
    S: Array
    ST: Array
    problem: Problem
    terms: tuple[str, ...]
    emulator: HDMREmulator | None = None
    select: Array | None = None
    rmse: Array | None = None
```

### Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `Sa` | `(n_terms,)` / `(K, n_terms)` / `(T, K, n_terms)` | Structural (uncorrelated) contribution per term. For first-order terms, equivalent to Sobol S1. |
| `Sb` | same as Sa | Correlative contribution per term (input correlations). |
| `S` | same as Sa | Total contribution per term: `Sa + Sb`. |
| `ST` | `(D,)` / `(K, D)` / `(T, K, D)` | Total-order per parameter (sum of S for all terms involving that parameter). |
| `terms` | `tuple[str, ...]` | Human-readable labels, e.g. `("x1", "x2", "x1/x2")`. |
| `emulator` | `HDMREmulator \| None` | Fitted B-spline coefficients for prediction via `emulate_hdmr()`. Multi-output HDMR stores leading `(K,)` or `(T, K)` axes on `C1`, `C2`, `C3`, and `f0`. |
| `select` | `(n_terms,)` or `None` | F-test selection counts summed across analyzed output combinations. |
| `rmse` | `Array \| None` | Emulator RMSE with shape `()`, `(K,)`, or `(T, K)` matching the analyzed output layout without the sample axis. |

### Emulator Output Shapes

| Y shape passed to `analyze_hdmr()` | `emulate_hdmr(..., X_new)` shape |
|------------------------------------|----------------------------------|
| `(N,)` | `(N_new,)` |
| `(N, K)` | `(N_new, K)` |
| `(N, T, K)` | `(N_new, T, K)` |

### Properties

| Property | Shape | Description |
|----------|-------|-------------|
| `S1` | same as ST | First-order Sobol indices -- extracts `Sa[:D]`. |

### Number of Terms

| maxorder | n_terms | Composition |
|----------|---------|-------------|
| 1 | D | D first-order terms |
| 2 | D + C(D,2) | + pairwise interactions |
| 3 | D + C(D,2) + C(D,3) | + triple interactions |
