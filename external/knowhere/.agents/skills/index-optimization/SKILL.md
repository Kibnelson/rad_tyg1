
# Index Optimization


Respect the user's requested scope and resource budget. Choose representative combinations first, then
expand where a regression, dispatch boundary, or interaction needs investigation. A full Cartesian product
is not required for every change. State which dimensions were measured and which remain unvalidated.

## Construction and memory


Measure baseline and candidate memory in separate fresh processes with equivalent input loading.
Use a construction-phase RSS sampler or a build-only process high-water measurement; state the method,
units, and whether input vectors and shared/mapped data are included. A process-lifetime high-water mark
collected after search or serialization cannot be attributed to construction alone.

`Index::Size()` and serialized bytes do not measure peak RSS. Label buffer-lifetime calculations as
estimates rather than measurements, and include old/new buffer overlap during conversion or reallocation.
A smaller final index can still require more memory to build.

## Query workload coverage

Choose supported values that span the production workload and the implementation's branch thresholds.
The following are starting points, not fixed requirements:

| Axis | Representative coverage |
|------|-------------------------|
| Top-k | Small, typical, and large requests, for example 1, 10, 100, and 1000 where meaningful. |
| Filter ratio | No filtering, light/moderate filtering, and highly selective filtering, for example 0%, 10%, 50%, 90%, and 99% excluded. Define the ratio as excluded documents divided by total documents and report eligible counts. |
| Filter distribution | For changes to filter/window/block handling, compare random masks with clustered or contiguous masks; equal global ratios can produce different local workloads. Include all-filtered and fewer-than-k eligible matches as correctness cases. |
| Sparse query NNZ | Bucket real queries into low, medium, and high NNZ groups using observed ranges or quantiles. Record bucket boundaries and query counts. If pruning or refinement changes the query, report original and effective NNZ. |
| Execution settings | Fix batch size, concurrency, thread pools, CPU affinity, and SIMD dispatch for comparisons. Vary them when the optimization specifically targets those dimensions. |

Check meaningful interactions, such as large top-k with high filtering, or high-NNZ sparse queries with
high filtering. For threshold-based dispatch, include inputs on both sides of the threshold.
Report results by workload group so an overall mean cannot hide a regression in one group.

Bucket existing sparse queries without changing their contents. If synthetic generation, truncation, or
other transformations alter a query, regenerate its ground truth. NNZ alone does not describe posting
work: consider term frequency or posting-list length when it explains differences between query groups.





## Reproducible comparisons


## Existing benchmarks

Look for reusable benchmarks in [benchmark/](../../../benchmark/). Inspect current code and options
before reuse, and extend coverage for the chosen workload matrix where needed. Do not assume every
parameter or platform is supported.
