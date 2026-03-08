# Getting Started

## Installation

```bash
pip install git+https://github.com/danielepessina/gsax.git
```

For development:

```bash
git clone https://github.com/danielepessina/gsax.git
cd gsax
pip install -e ".[dev]"
```

## Quick Start

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# 1. Generate unique Sobol/Saltelli samples
sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)

# 2. Evaluate your model on the unique samples
Y = evaluate(sampling_result.samples)

# 3. Compute Sobol indices
result = gsax.analyze(sampling_result, Y)

print("S1:", result.S1)   # first-order indices
print("ST:", result.ST)   # total-order indices
print("S2:", result.S2)   # second-order interaction matrix
```

Expected output (Ishigami function, A=7, B=0.1):

```
S1: [~0.31, ~0.44, ~0.00]
ST: [~0.56, ~0.44, ~0.24]
```

## Define a Problem

A `Problem` specifies parameter names and bounds:

```python
from gsax import Problem

problem = Problem.from_dict({
    "x1": (-3.14, 3.14),
    "x2": (-3.14, 3.14),
    "x3": (-3.14, 3.14),
})
```

## What's Next?

- [Methods](/guide/methods) -- understand Sobol vs HDMR and when to use each
- [xarray Output](/examples/xarray) -- labeled results with named dimensions
- [Examples](/examples/basic) -- copy-pasteable code for common workflows
- [API Reference](/api/problem) -- full parameter and return-type documentation
