# Hnswlib - fast approximate nearest neighbor search
Header-only C++ HNSW implementation with python bindings.

**NEWS:**


**version 0.6.2** 

* Fixed a bug in saving of large pickles. The pickles with > 4GB could have been corrupted. Thanks Kai Wohlfahrt for reporting.
* Thanks to ([@GuyAv46](https://github.com/GuyAv46)) hnswlib inner product now is more consitent accross architectures (SSE, AVX, etc). 
* 

**version 0.6.1** 

* Thanks to ([@tony-kuo](https://github.com/tony-kuo)) hnswlib AVX512 and AVX builds are not backwards-compatible with older SSE and non-AVX512 architectures. 
* Thanks to ([@psobot](https://github.com/psobot)) there is now a sencible message instead of segfault when passing a scalar to get_items.
* Thanks to ([@urigoren](https://github.com/urigoren)) hnswlib has a lazy index creation python wrapper.


  


### Highlights:
1) Lightweight, header-only, no dependencies other than C++ 11
2) Interfaces for C++, Java, Python and R (https://github.com/jlmelville/rcpphnsw).
3) Has full support for incremental index construction. Has support for element deletions 
(by marking them in index). Index is picklable.
4) Can work with custom user defined distances (C++).
5) Significantly less memory footprint and faster build time compared to current nmslib's implementation.

Description of the algorithm parameters can be found in [ALGO_PARAMS.md](ALGO_PARAMS.md).


### Python bindings

#### Supported distances:

| Distance         | parameter       | Equation                |
| -------------    |:---------------:| -----------------------:|
|Squared L2        |'l2'             | d = sum((Ai-Bi)^2)      |
|Inner product     |'ip'             | d = 1.0 - sum(Ai\*Bi)   |
|Cosine similarity |'cosine'         | d = 1.0 - sum(Ai\*Bi) / sqrt(sum(Ai\*Ai) * sum(Bi\*Bi))|

Note that inner product is not an actual metric. An element can be closer to some other element than to itself. That allows some speedup if you remove all elements that are not the closest to themselves from the index.

For other spaces use the nmslib library https://github.com/nmslib/nmslib. 

#### Short API description
* `hnswlib.Index(space, dim)` creates a non-initialized index an HNSW in space `space` with integer dimension `dim`.

`hnswlib.Index` methods:
* `init_index(max_elements, M = 16, ef_construction = 200, random_seed = 100)` initializes the index from with no elements. 
    * `max_elements` defines the maximum number of elements that can be stored in the structure(can be increased/shrunk).
    * `ef_construction` defines a construction time/accuracy trade-off (see [ALGO_PARAMS.md](ALGO_PARAMS.md)).
    * `M` defines tha maximum number of outgoing connections in the graph ([ALGO_PARAMS.md](ALGO_PARAMS.md)).
    
* `add_items(data, ids, num_threads = -1)` - inserts the `data`(numpy array of vectors, shape:`N*dim`) into the structure. 
    * `num_threads` sets the number of cpu threads to use (-1 means use default).
    * `ids` are optional N-size numpy array of integer labels for all elements in `data`. 
      - If index already has the elements with the same labels, their features will be updated. Note that update procedure is slower than insertion of a new element, but more memory- and query-efficient.
    * Thread-safe with other `add_items` calls, but not with `knn_query`.
    
* `mark_deleted(label)`  - marks the element as deleted, so it will be omitted from search results. Throws an exception if it is already deleted.
* 
* `unmark_deleted(label)`  - unmarks the element as deleted, so it will be not be omitted from search results.

* `resize_index(new_size)` - changes the maximum capacity of the index. Not thread safe with `add_items` and `knn_query`.

* `set_ef(ef)` - sets the query time accuracy/speed trade-off, defined by the `ef` parameter (
[ALGO_PARAMS.md](ALGO_PARAMS.md)). Note that the parameter is currently not saved along with the index, so you need to set it manually after loading.

* `knn_query(data, k = 1, num_threads = -1)` make a batch query for `k` closest elements for each element of the 
    * `data` (shape:`N*dim`). Returns a numpy array of (shape:`N*k`).
    * `num_threads` sets the number of cpu threads to use (-1 means use default).
    * Thread-safe with other `knn_query` calls, but not with `add_items`.
    
* `load_index(path_to_index, max_elements = 0)` loads the index from persistence to the uninitialized index.
    * `max_elements`(optional) resets the maximum number of elements in the structure.
      
* `save_index(path_to_index)` saves the index from persistence.

* `set_num_threads(num_threads)` set the default number of cpu threads used during data insertion/querying.
  
* `get_items(ids)` - returns a numpy array (shape:`N*dim`) of vectors that have integer identifiers specified in `ids` numpy vector (shape:`N`). Note that for cosine similarity it currently returns **normalized** vectors.
  
* `get_ids_list()`  - returns a list of all elements' ids.

* `get_max_elements()` - returns the current capacity of the index

* `get_current_count()` - returns the current number of element stored in the index

Read-only properties of `hnswlib.Index` class:

* `space` - name of the space (can be one of "l2", "ip", or "cosine"). 

* `dim`   - dimensionality of the space. 

* `M` - parameter that defines the maximum number of outgoing connections in the graph. 

* `ef_construction` - parameter that controls speed/accuracy trade-off during the index construction. 

* `max_elements` - current capacity of the index. Equivalent to `p.get_max_elements()`. 

* `element_count` - number of items in the index. Equivalent to `p.get_current_count()`. 

Properties of `hnswlib.Index` that support reading and writing:

* `ef` - parameter controlling query time/accuracy trade-off.

* `num_threads` - default number of threads to use in `add_items` or `knn_query`. Note that calling `p.set_num_threads(3)` is equivalent to `p.num_threads=3`.

  
        
  
#### Python bindings examples

An example with updates after serialization/deserialization:

### Bindings installation

You can install from sources:
```bash
apt-get install -y python-setuptools python-pip
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
pip install .
```

or you can install via pip:
`pip install hnswlib`


### For developers 
Contributions are highly welcome!

Please make pull requests against the `develop` branch.

When making changes please run tests (and please add a test to `python_bindings/tests` in case there is new functionality):
```bash
python -m unittest discover --start-directory python_bindings/tests --pattern "*_test*.py
```


### Other implementations
* Non-metric space library (nmslib) - main library(python, C++), supports exotic distances: https://github.com/nmslib/nmslib
* Faiss library by facebook, uses own HNSW  implementation for coarse quantization (python, C++):
https://github.com/facebookresearch/faiss
* Code for the paper 
["Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors"](https://arxiv.org/abs/1802.02422) 
(current state-of-the-art in compressed indexes, C++):
https://github.com/dbaranchuk/ivf-hnsw
* TOROS N2 (python, C++): https://github.com/kakao/n2 
* Online HNSW (C++): https://github.com/andrusha97/online-hnsw) 
* Go implementation: https://github.com/Bithack/go-hnsw
* Python implementation (as a part of the clustering code by by Matteo Dell'Amico): https://github.com/matteodellamico/flexible-clustering
* Java implementation: https://github.com/jelmerk/hnswlib
* Java bindings using Java Native Access: https://github.com/stepstone-tech/hnswlib-jna
* .Net implementation: https://github.com/microsoft/HNSW.Net
* CUDA implementation: https://github.com/js1010/cuhnsw


### 200M SIFT test reproduction 
To download and extract the bigann dataset (from root directory):
```bash
python3 download_bigann.py
```
To compile:
```bash
mkdir build
cd build
cmake ..
make all
```

To run the test on 200M SIFT subset:
```bash
./main
```

The size of the BigANN subset (in millions) is controlled by the variable **subset_size_millions** hardcoded in **sift_1b.cpp**.

### Updates test
To generate testing data (from root directory):
```bash
cd examples
python update_gen_data.py
```
To compile (from root directory):
```bash
mkdir build
cd build
cmake ..
make 
```
To run test **without** updates (from `build` directory)
```bash
./test_updates
```

To run test **with** updates (from `build` directory)
```bash
./test_updates update
```

### HNSW example demos

- Visual search engine for 1M amazon products (MXNet + HNSW): [website](https://thomasdelteil.github.io/VisualSearch_MXNet/), [code](https://github.com/ThomasDelteil/VisualSearch_MXNet), demo by [@ThomasDelteil](https://github.com/ThomasDelteil)

### References

@article{malkov2018efficient,
  title={Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs},
  author={Malkov, Yu A and Yashunin, Dmitry A},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={42},
  number={4},
  pages={824--836},
  year={2018},
  publisher={IEEE}
}
