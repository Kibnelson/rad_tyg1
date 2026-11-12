

### Brute force index API
`hnswlib.BFIndex(space, dim)` creates a non-initialized index in space `space` with integer dimension `dim`.

`hnswlib.BFIndex` methods:

`init_index(max_elements)` initializes the index with no elements.

max_elements defines the maximum number of elements that can be stored in the structure.

`add_items(data, ids)` inserts the data (numpy array of vectors, shape:`N*dim`) into the structure.
`ids` are optional N-size numpy array of integer labels for all elements in data.

`delete_vector(label)` delete the element associated with the given `label` so it will be omitted from search results.

`knn_query(data, k = 1)` make a batch query for `k `closest elements for each element of the
`data` (shape:`N*dim`). Returns a numpy array of (shape:`N*k`).

`load_index(path_to_index, max_elements = 0)` loads the index from persistence to the uninitialized index.

`save_index(path_to_index)` saves the index from persistence.
