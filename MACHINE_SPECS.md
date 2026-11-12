# Which machine should I use?

Use **Ubuntu 24.04 LTS, Python 3.12, an x86-64 CPU and SSD storage**.
No GPU is needed. The sizes below are starting recommendations, not measured
minimums. All run commands use **30 worker threads** and need at least **30 available
logical CPUs**. Provision a **32-vCPU VM**. Start with the 320-document sample
before downloading a large dataset.

## Small sample: build and test

| System | vCPUs | RAM | Free SSD space | Workers in the runbook |
|---|---:|---:|---:|---:|
| RAD, scalar kernel | 32 | 32 GB | 60 GB | 30 |
| SIMD-DPK | 32 | 32 GB | 20 GB | 30 |
| Faiss(Jaccard) | 32 | 32 GB | 60 GB | 30 |
| Milvus, including native build | 32 | 64 GB | 100 GB | 30 |

Builds need internet access. Milvus builds many dependencies and can take hours;
its full custom-server/sample test is still pending. See the
[Milvus validation notes](https://github.com/Kibnelson/rad_tyg1/blob/milvus/VALIDATION.md).
`JOBS` controls compilation only; it is separate from the 30 runtime workers.
Use `JOBS=2` instead of four for RAD/Faiss builds if you need to reduce build memory.
The SIMD-DPK setup compiles its small modules separately and does not use `JOBS`.

## Larger runs, up to about 30M documents

Use **one system and one dataset per VM** for timing comparisons. A practical
starting size is **32 vCPUs, 256 GB RAM and a 1 TB SSD**, with `--threads 30`
when at least 30 CPUs are available to the process. This leaves room for the OS
and supporting services. Use the same machine size and thread count for each
system you compare. A vCPU is a logical CPU, not necessarily a physical core.
`--threads` sets a worker limit; it does not mean every worker stays busy.

This is a capacity-planning suggestion, not a guarantee that every 30M input
fits. Full document lengths and saved candidate files change disk use. RAD
keeps all partitions in RAM, Faiss keeps one growing graph, SIMD-DPK rebuilds
its full kept-document reference, and Milvus also needs local service storage.
Increase RAM/disk if your input or saved results require it.

## CPU features matter

A VM's CPU count alone is not enough. Check its instruction support:

| System | Required CPU features |
|---|---|
| RAD `cache_scalar` | AVX2, POPCNT |
| RAD `cache_simd` | AVX2, POPCNT, AVX-512F and VPOPCNTDQ |
| SIMD-DPK | AVX2, POPCNT |
| Milvus | AVX2, POPCNT, AVX-512F/DQ/VL/BW |
| Faiss(Jaccard) | AVX2, F16C, FMA, POPCNT, BMI2, AVX-512F/CD/DQ/BW/VL |

After cloning, run the check for your branch before building:

```bash
python3 scripts/check_machine.py
```

It prints available CPUs, RAM, free disk and CPU compatibility for each system.
An unsupported optional RAD SIMD kernel does not prevent the scalar example
from working. Use `--system rad --distance-mode cache_scalar`, `--system milvus`,
`--system simd_dpk`, or `--system faiss_jaccard` to check just one configuration;
a missing CPU feature or fewer than 30 available CPUs then gives a nonzero exit
status. It does not run an index.

On Google Cloud, `n2-highmem-32` has 32 vCPUs and 256 GB RAM; disk size
is chosen separately. See the
[official N2 machine table](https://docs.cloud.google.com/compute/docs/general-purpose-machines#n2_high-mem).
For an AVX-512 machine, select a suitable Intel CPU platform such as Ice Lake
where available, then verify the actual flags with the command above.
Google documents [CPU features](https://docs.cloud.google.com/compute/docs/cpu-platforms#cpu_features)
and [minimum CPU platform selection](https://docs.cloud.google.com/compute/docs/instances/specify-min-cpu-platform).
Do not assume every machine with the same vCPU count exposes the same instructions.

## Installation checks

See the [installation checks](https://github.com/Kibnelson/rad_tyg1/blob/main/VALIDATION.md)
for the checks completed on each branch.
Run the CPU check on your own VM before choosing a distance kernel or building.
