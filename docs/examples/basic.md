# Basic Example (Ishigami)

The Ishigami function is a standard benchmark for sensitivity analysis with known analytical solutions.

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# Generate Saltelli samples (unique rows only)
sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)

# Evaluate the model
Y = evaluate(sampling_result.samples)

# Compute Sobol indices
result = gsax.analyze(sampling_result, Y)

print("First-order (S1):", result.S1)
print("Total-order (ST):", result.ST)
print("Second-order (S2):")
print(result.S2)
```

Expected output (A=7, B=0.1):

```
First-order (S1): [~0.31, ~0.44, ~0.00]
Total-order (ST): [~0.56, ~0.44, ~0.24]
```

**Interpreting the results:**
- `x2` has the largest first-order effect (S1 ~ 0.44) and no interactions (ST ~ S1).
- `x1` has a moderate main effect (S1 ~ 0.31) but significant interactions (ST ~ 0.56).
- `x3` has zero main effect (S1 ~ 0.00) but contributes through interactions with `x1` (ST ~ 0.24).

## Using a DataFrame

`sampling_result.samples_df` gives a pandas DataFrame with `SampleID` and parameter columns:

```python
df = sampling_result.samples_df
print(df.head())
#    SampleID        x1        x2        x3
# 0         0 -1.234567  2.345678 -0.123456
# 1         1  0.987654 -1.876543  3.012345
# ...
```
