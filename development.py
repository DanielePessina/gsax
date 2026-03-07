import gsax
from gsax.benchmarks.ishigami import evaluate
from gsax import Problem
import numpy as np

gsa_problem = Problem.from_dict(
    {
        "x1": (-np.pi, np.pi),
        "x2": (-np.pi, np.pi),
        "x3": (-np.pi, np.pi),
    }
)
# 1. Generate Saltelli samples
sampling_result = gsax.sample(gsa_problem, n_samples=4096, seed=42)

# sampling_result.samples.shape == (n_total, D)  where D = 3
