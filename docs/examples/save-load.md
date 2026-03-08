# Save and Reload Samples

Use `SamplingResult.save()` when you want to generate Sobol/Saltelli samples
once, store them on disk, and reuse them later without re-running the sampling
step.

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
sampling_result.save("runs/ishigami", format="csv")

restored = gsax.load("runs/ishigami", format="csv")
Y = evaluate(restored.samples)
result = gsax.analyze(restored, Y)

print(result.S1)
print(result.ST)
```

## Files written to disk

For `path="runs/ishigami"` and `format="csv"`, `save()` writes:

- `runs/ishigami.csv` with the unique sample matrix
- `runs/ishigami.json` with the `Problem` definition and Saltelli metadata
- `runs/ishigami.npz` only when the expanded-to-unique mapping is not the identity

The `.npz` file is skipped for storage efficiency when the expanded Saltelli
rows already map one-to-one to the unique rows.

## Supported formats

The sample matrix can be stored as:

- `csv`
- `txt`
- `xlsx`
- `parquet`
- `pkl`

Use the same `format` when calling `gsax.load()`. `xlsx` requires `openpyxl`,
and `parquet` requires `pyarrow`.
