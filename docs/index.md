---
layout: home

hero:
  name: gsax
  text: Global Sensitivity Analysis in JAX
  tagline: GPU-accelerated Sobol indices and RS-HDMR with JIT compilation, vectorized bootstrap, and multi-output support.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /api/problem

features:
  - title: Sobol Indices
    details: First-order, total-order, and second-order indices via Saltelli sampling with Sobol quasi-random sequences.
  - title: RS-HDMR
    details: Surrogate-based sensitivity analysis that works with any (X, Y) pairs. Includes a built-in emulator for prediction.
  - title: Multi-Output & Time-Series
    details: Pass scalar, (N, K), or (N, T, K) outputs. All indices are computed in a single vectorized pass.
  - title: Up to 929× Faster than SALib
    details: Fused JIT kernels and vectorized execution replace Python loops. 4.7× faster even on scalar outputs, up to 929× on multi-output workloads.
---
