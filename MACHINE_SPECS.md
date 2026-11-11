# Which machine should I use?

Use **Ubuntu 24.04 LTS, Python 3.12, an x86-64 CPU and SSD storage**.
No GPU is needed. The sizes below are starting recommendations, not measured
minimums. Start with the 320-document sample before downloading a large dataset.

## Small sample: build and test

| System | vCPUs | RAM | Free SSD space | Workers in the runbook |
|---|---:|---:|---:|---:|
| RAD, scalar kernel | 4 | 32 GB | 60 GB | 4 |
| SIMD-DPK | 4 | 8 GB | 20 GB | 4 |
| Faiss(Jaccard) | 4 | 32 GB | 60 GB | 4 |
| Milvus, including native build | 8 | 64 GB | 100 GB | 4 |

Builds need internet access. Milvus builds many dependencies and can take hours;
its full custom-server/sample test is still pending. See the
[Milvus validation notes](https://github.com/Kibnelson/rad_tyg1/blob/milvus/VALIDATION.md).
Use `JOBS=2` instead of four for RAD/Faiss builds if you need to reduce build memory.
The SIMD-DPK setup compiles its small modules separately and does not use `JOBS`.

## Larger runs, up to about 30M documents

Use **one system and one dataset per VM** for timing comparisons. A practical
starting size is **32 vCPUs, 256 GB RAM and a 1 TB SSD**, with `--threads 28`
when at least 28 CPUs are available to the process. This leaves room for the OS
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
a missing CPU feature then gives a nonzero exit status. It does not run an index.

On Google Cloud, `n2-highmem-4` has 4 vCPUs/32 GB, `n2-highmem-8` has 8/64 GB,
and `n2-highmem-32` has 32/256 GB; disk size is chosen separately. See the
[official N2 machine table](https://docs.cloud.google.com/compute/docs/general-purpose-machines#n2_high-mem).
For an AVX-512 machine, select a suitable Intel CPU platform such as Ice Lake
where available, then verify the actual flags with the command above.
Google documents [CPU features](https://docs.cloud.google.com/compute/docs/cpu-platforms#cpu_features)
and [minimum CPU platform selection](https://docs.cloud.google.com/compute/docs/instances/specify-min-cpu-platform).
Do not assume every machine with the same vCPU count exposes the same instructions.

## What has actually been tested?

The public RAD scalar, SIMD-DPK and Faiss native tests passed on a Linux x86-64
host with 16 available logical CPUs and about 62 GiB total RAM, using Python 3.10
and four worker threads for the sample. The host lacks VPOPCNTDQ, so RAD's SIMD
kernel was not tested there. Milvus's SDK, services and setup checks passed;
its full native server test remains pending. The recommended Ubuntu 24.04/
Python 3.12 setup path is separate from those recorded host tests.
