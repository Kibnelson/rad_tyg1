"""Thin binding to the original in-batch SIMD extension; no Python fallback."""
import importlib
import numpy as np


def keep_indices(sketches, threads):
    matrix = np.ascontiguousarray(sketches, dtype=np.uint64)
    if not len(matrix):
        return np.empty(0, dtype=np.int64)
    name = 'simd_fuzzy_deduplicationv7_unsorted_multi7A_V1'
    nested = getattr(importlib.import_module(name), name)(matrix, threshold=.7,
                                                         num_bands=14, num_threads=threads)
    removed = set()

    def flatten(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            for child in value:
                flatten(child)
        elif value is not None:
            index = int(value)
            if not 0 <= index < len(matrix):
                raise ValueError('Native SIMD returned an invalid row')
            removed.add(index)

    flatten(nested)
    return np.array([i for i in range(len(matrix)) if i not in removed], dtype=np.int64)
