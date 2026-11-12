
# Benchmarking scripts

This directory contains benchmarking scripts that can reproduce the
numbers reported in the two papers

```
@inproceedings{DJP16,
  Author = {Douze, Matthijs and J{\'e}gou, Herv{\'e} and Perronnin, Florent},
  Booktitle = "ECCV",
  Organization = {Springer},
  Title = {Polysemous codes},
  Year = {2016}
}
```
and

```
@inproceedings{JDJ17,
   Author = {Jeff Johnson and Matthijs Douze and Herv{\'e} J{\'e}gou},
   journal = {arXiv:1702.08734},
   Title = {Billion-scale similarity search with GPUs},
   Year = {2017},
}
```

Note that the numbers (especially timings) change slightly due to changes in the implementation, different machines, etc.

The scripts are self-contained. They depend only on Faiss and external training data that should be stored in sub-directories.

## SIFT1M experiments

The script [`bench_polysemous_sift1m.py`](bench_polysemous_sift1m.py) reproduces the numbers in
Figure 3 from the "Polysemous" paper.

### Getting SIFT1M

To run it, please download the ANN_SIFT1M dataset from

http://corpus-texmex.irisa.fr/

and unzip it to the subdirectory sift1M.

### Result

The output looks like:



## Experiments on 1B elements dataset

The script [`bench_polysemous_1bn.py`](bench_polysemous_1bn.py) reproduces a few experiments on
two datasets of size 1B from the Polysemous codes" paper.


### Getting BIGANN

Download the four files of ANN_SIFT1B from
http://corpus-texmex.irisa.fr/ to subdirectory bigann/

### Getting Deep1B

The ground-truth and queries are available here

https://yadi.sk/d/11eDCm7Dsn9GA

For the learning and database vectors, use the script

https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py

to download the data to subdirectory deep1b/, then concatenate the
database files to base.fvecs and the training files to learn.fvecs

### Running the experiments

These experiments are quite long. To support resuming, the script
stores the result of training to a temporary directory, `/tmp/bench_polysemous`.

The script `bench_polysemous_1bn.py` takes at least two arguments:

- the dataset name: SIFT1000M (aka SIFT1B, aka BIGANN) or Deep1B. SIFT1M, SIFT2M,... are also supported to make subsets of for small experiments (note that SIFT1M as a subset of SIFT1B is not the same as the SIFT1M above)

- the type of index to build, which should be a valid [index_factory key](https://github.com/facebookresearch/faiss/wiki/High-level-interface-and-auto-tuning#index-factory) (see below for examples)

- the remaining arguments are parsed as search-time parameters.

### Experiments of Table 2

The `IMI*+PolyD+ADC` results in Table 2 can be reproduced with (for 16 bytes):

```
python bench_polysemous_1bn.par SIFT1000M IMI2x12,PQ16 nprobe=16,max_codes={10000,30000},ht={44..54}
```

Training takes about 2 minutes and adding vectors to the dataset
takes 3.1 h. These operations are multithreaded. Note that in the command
above, we use bash's [brace expansion](https://www.gnu.org/software/bash/manual/html_node/Brace-Expansion.html) to set a grid of parameters.

The search is *not* multithreaded, and the output looks like:


The result reported in table 2 is the one for which the %pass (percentage of code comparisons that pass the Hamming check) is around 20%, which occurs for Hamming threshold `ht=48`.

The 8-byte results can be reproduced with the factory key `IMI2x12,PQ8`

### Experiments of the appendix

The experiments in the appendix are only in the ArXiv version of the paper (table 3).

Here again the runs are not exactly the same but the original result was obtained from nprobe=32,ht=28.

For Deep1B, we used a simple version of [auto-tuning](https://github.com/facebookresearch/faiss/wiki/High-level-interface-and-auto-tuning/_edit#auto-tuning-the-runtime-parameters) to sweep through the set of operating points:

The original results were obtained with `nprobe=1024,ht=66,max_codes=262144`.


## GPU experiments

The benchmarks below run 1 or 4 Titan X GPUs and reproduce the results of the "GPU paper". They are also a good starting point on how to use GPU Faiss.

### Search on SIFT1M

See above on how to get SIFT1M into subdirectory sift1M/. The script [`bench_gpu_sift1m.py`](bench_gpu_sift1m.py) reproduces the "exact k-NN time" plot in the ArXiv paper, and the SIFT1M numbers.

The output is:
The run produces two warnings:

- the clustering complains that it does not have enough training data, there is not much we can do about this.

- the add() function complains that there is an inefficient memory allocation, but this is a concern only when it happens often, and we are not benchmarking the add time anyways.

To index small datasets, it is more efficient to use a `GpuIVFFlat`, which just stores the full vectors in the inverted lists. We did not mention this in the paper because it is not as scalable. To experiment with this setting, change the `index_factory` string from "IVF4096,PQ64" to "IVF16384,Flat". This gives:


### Clustering on MNIST8m

To get the "infinite MNIST dataset", follow the instructions on [Léon Bottou's website](http://leon.bottou.org/projects/infimnist). The script assumes the file `mnist8m-patterns-idx3-ubyte` is in subdirectory `mnist8m`

The script [`kmeans_mnist.py`](kmeans_mnist.py) produces the following output:

```
python kmeans_mnist.py 1 256
...
Clustering 8100000 points in 784D to 256 clusters, redo 1 times, 20 iterations
  Preprocessing in 7.94526 s
  Iteration 19 (131.697 s, search 114.78 s): objective=1.44881e+13 imbalance=1.05963 nsplit=0
final objective: 1.449e+13
total runtime: 140.615 s
```

### search on SIFT1B

The script [`bench_gpu_1bn.py`](bench_gpu_1bn.py) runs multi-gpu searches on the two 1-billion vector datasets we considered. It is more complex than the previous scripts, because it supports many search options and decomposes the dataset build process in Python to exploit the best possible CPU/GPU parallelism and GPU distribution.

Even on multiple GPUs, building the 1B datasets can last several hours. It is often a good idea to validate that everything is working fine on smaller datasets like SIFT1M, SIFT2M, etc.

The search results on SIFT1B in the "GPU paper" can be obtained with

<!-- see P57124181 -->


We use the `-tempmem` option to reduce the temporary memory allocation to 1.5G, otherwise the dataset does not fit in GPU memory

### search on Deep1B

The same script generates the GPU search results on Deep1B.


Here we are a bit tight on memory so we disable precomputed tables (`-noptables`) and restrict the amount of temporary memory. The `-altadd` option avoids GPU memory overflows during add.


### knn-graph on Deep1B

The same script generates the KNN-graph on Deep1B. Note that the inverted file from above will not be re-used because the training sets are different. For the knngraph, the script will first do a pass over the whole dataset to compute the ground-truth knn for a subset of 10k nodes, for evaluation.

```
python bench_gpu_1bn.py Deep1B OPQ20_80,IVF262144,PQ20 -nnn 10 -altadd -knngraph  -R 2 -noptables -tempmem $[1<<30] -ngpu 4
...
CPU index contains 1000000000 vectors, move to GPU
Copy CPU index to 2 sharded GPU indexes
   dispatch to GPUs 0:2
IndexShards shard 0 indices 0:500000000
  IndexIVFPQ size 500000000 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=0 reserveVecs=0
IndexShards shard 1 indices 500000000:1000000000
  IndexIVFPQ size 500000000 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=0 reserveVecs=0
   dispatch to GPUs 2:4
IndexShards shard 0 indices 0:500000000
  IndexIVFPQ size 500000000 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=0 reserveVecs=0
IndexShards shard 1 indices 500000000:1000000000
  IndexIVFPQ size 500000000 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=0 reserveVecs=0
move to GPU done in 151.535 s
search...
999997440/1000000000 (8389.961 s, 0.3379)      probe=1  : 8389.990 s rank-10 intersection results: 0.3379
999997440/1000000000 (9205.934 s, 0.4079)      probe=2  : 9205.966 s rank-10 intersection results: 0.4079
999997440/1000000000 (9741.095 s, 0.4722)      probe=4  : 9741.128 s rank-10 intersection results: 0.4722
999997440/1000000000 (10830.420 s, 0.5256)      probe=8  : 10830.455 s rank-10 intersection results: 0.5256
999997440/1000000000 (12531.716 s, 0.5603)      probe=16 : 12531.758 s rank-10 intersection results: 0.5603
999997440/1000000000 (15922.519 s, 0.5825)      probe=32 : 15922.571 s rank-10 intersection results: 0.5825
999997440/1000000000 (22774.153 s, 0.5950)      probe=64 : 22774.220 s rank-10 intersection results: 0.5950
999997440/1000000000 (36717.207 s, 0.6015)      probe=128: 36717.309 s rank-10 intersection results: 0.6015
999997440/1000000000 (70616.392 s, 0.6047)      probe=256: 70616.581 s rank-10 intersection results: 0.6047
```

# Additional benchmarks

This directory also contains certain additional benchmarks (and serve as an additional source of examples of how to use the Faiss code).
Certain tests / benchmarks might be outdated.

* bench_6bit_codec.cpp - tests vector codecs for SQ6 quantization on a synthetic dataset
* bench_cppcontrib_sa_decode.cpp - benchmarks specialized kernels for vector codecs for PQ, IVFPQ and Resudial+PQ on a synthetic dataset
* bench_for_interrupt.py - evaluates the impact of the interrupt callback handler (which can be triggered from Python code)
* bench_hamming_computer.cpp - specialized implementations for Hamming distance computations
* bench_heap_replace.cpp - benchmarks different implementations of certain calls for a Heap data structure
* bench_hnsw.py - benchmarks HNSW in combination with other ones for SIFT1M dataset
* bench_index_flat.py - benchmarks IndexFlatL2 on a synthetic dataset
* bench_index_pq.py - benchmarks PQ on SIFT1M dataset
* bench_ivf_fastscan_single_query.py - benchmarks a single query for different nprobe levels for IVF{nlist},PQ{M}x4fs on BIGANN dataset
* bench_ivf_fastscan.py - compares IVF{nlist},PQ{M}x4fs against other indices on SIFT1M dataset
* bench_ivf_selector.cpp - checks the possible overhead when using faiss::IDSelectorAll interface
* bench_pairwise_distances.py - benchmarks pairwise distance computation between two synthetic datasets
* bench_partition.py - benchmarks partitioning functions
* bench_pq_tables.py - benchmarks ProductQuantizer.compute_inner_prod_tables() and ProductQuantizer.compute_distance_tables() calls
* bench_quantizer.py - benchmarks various quantizers for SIFT1M, Deep1B, BigANN datasets
* bench_scalar_quantizer.py - benchmarks IVF+SQ on a Sift1M dataset
* bench_vector_ops.py - benchmarks dot product and distances computations on a synthetic dataset
