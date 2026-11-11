"""Experimental length and tagged-band Bloom routing; no admission decisions."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import faiss
import numpy as np

from .partitions import MISSING_DISTANCE, atomic_json

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def native_search(number, part, queries, k, ef_search, threads):
    faiss.omp_set_num_threads(threads)
    part.index.hnsw.efSearch = ef_search
    cpu_started, started = time.process_time(), time.perf_counter()
    distances, positions = part.index.search(queries, min(k, int(part.index.ntotal)))
    wall, cpu = time.perf_counter()-started, time.process_time()-cpu_started
    return distances, positions, dict(native_seconds=wall, native_cpu_seconds=cpu,
        native_average_cpu_cores=cpu/wall, configured_faiss_threads=faiss.omp_get_max_threads())

LAYOUTS = {'28x4': (28, 4), '37x3': (37, 3)}
CASES = ('all', 'length', 'bloom_28x4', 'length_bloom_28x4', 'bloom_37x3', 'length_bloom_37x3')
SEED = 0x5241445632525431
_LIBRARY = None


def native_library():
    global _LIBRARY
    if _LIBRARY is None:
        source = ROOT / 'native/routing.cpp'
        directory = ROOT / 'build/routing'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f'routing-{digest(source)[:20]}.so'
        if not target.exists():
            temporary = target.with_suffix(f'.{os.getpid()}.tmp.so')
            command = [os.environ.get('CXX', 'c++'), '-O3', '-std=c++17', '-fopenmp',
                       '-shared', '-fPIC', str(source), '-o', str(temporary)]
            subprocess.run(command, check=True)
            temporary.replace(target)
        library = ctypes.CDLL(str(target))
        u32 = np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS')
        u64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
        library.routing_keys.argtypes = [u32, u64, ctypes.c_int64, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_int]
        library.routing_add.argtypes = [u64, ctypes.c_int64, u64, ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        library.routing_any.argtypes = [u64, ctypes.c_int64, ctypes.c_int, u64,
            ctypes.c_uint64, ctypes.c_int, u8, ctypes.c_int]
        for name in ('routing_keys', 'routing_add', 'routing_any'):
            getattr(library, name).restype = None
        _LIBRARY = library
    return _LIBRARY


def band_keys(signatures, layout, threads=1):
    signatures = np.ascontiguousarray(signatures, dtype=np.uint32)
    bands, rows = LAYOUTS[layout]
    if signatures.ndim != 2 or signatures.shape[1] < bands*rows or threads < 1:
        raise ValueError('Signature shape or routing thread count is invalid')
    output = np.empty((len(signatures), bands), dtype=np.uint64)
    # Include layout dimensions as well as band position in the hash domain.
    seed = SEED ^ (bands << 32) ^ rows
    native_library().routing_keys(signatures, output, len(signatures), signatures.shape[1],
                                 bands, rows, seed, threads)
    return output


class Bloom:
    def __init__(self, key_capacity, false_positive=1e-4, bits=None):
        if key_capacity <= 0 or not 0 < false_positive < 1:
            raise ValueError('Bloom capacity/FPR is invalid')
        self.key_capacity, self.false_positive = int(key_capacity), float(false_positive)
        self.nbits = 64*math.ceil(-key_capacity*math.log(false_positive)/(math.log(2)**2)/64)
        self.probes = max(1, round(-math.log(false_positive)/math.log(2)))
        if self.probes > 64:
            raise ValueError('Requested Bloom FPR requires more than 64 probes')
        self.bits = np.zeros(self.nbits//64, dtype=np.uint64) if bits is None else bits
        if self.bits.dtype != np.uint64 or self.bits.shape != (self.nbits//64,) or not self.bits.flags.c_contiguous:
            raise ValueError('Bloom array configuration differs')

    def add(self, keys, threads=1):
        keys = np.ascontiguousarray(keys, dtype=np.uint64)
        if threads < 1:
            raise ValueError('Routing threads must be positive')
        if not self.bits.flags.writeable:
            raise ValueError('Cannot mutate a read-only routing summary')
        native_library().routing_add(keys, keys.size, self.bits, self.nbits, self.probes, threads)

    def contains_any(self, keys, threads=1):
        keys = np.ascontiguousarray(keys, dtype=np.uint64)
        if keys.ndim != 2 or threads < 1:
            raise ValueError('Expected a query-by-band matrix and positive threads')
        output = np.empty(len(keys), dtype=np.uint8)
        native_library().routing_any(keys, len(keys), keys.shape[1], self.bits,
                                    self.nbits, self.probes, output, threads)
        return output.astype(bool)


def length_mask(sizes, ranges, threshold):
    if not 0 < threshold <= 1:
        raise ValueError('Length threshold must be in (0,1]')
    sizes, ranges = np.asarray(sizes), np.asarray(ranges)
    if sizes.ndim != 1 or ranges.ndim != 2 or ranges.shape[1] != 2 or np.any(sizes < 0):
        raise ValueError('Invalid cardinality arrays')
    if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError('Invalid graph cardinality ranges')
    lower = np.nextafter(sizes*threshold, -np.inf)
    upper = np.nextafter(sizes/threshold, np.inf)
    return (ranges[None, :, 1] >= lower[:, None]) & (ranges[None, :, 0] <= upper[:, None])


def route(case, signatures, sizes, ranges, blooms, threshold, threads):
    if case not in CASES:
        raise ValueError('Unknown routing case')
    mask = np.ones((len(signatures), len(ranges)), dtype=bool)
    if case == 'length' or case.startswith('length_'):
        mask &= length_mask(sizes, ranges, threshold)
    if 'bloom_' in case:
        layout = case.split('bloom_', 1)[1]
        keys = band_keys(signatures, layout, threads)
        for graph, bloom in enumerate(blooms[layout]):
            selected = np.flatnonzero(mask[:, graph])
            if len(selected):
                mask[selected, graph] = bloom.contains_any(keys[selected], threads)
    return mask


def search_selected(store, queries, k, ef_search, threads, mask):
    """Search all selected query/graph pairs, sequentially; never stop on a hit."""
    if mask.shape != (len(queries), len(store.partitions)) or mask.dtype != bool:
        raise ValueError('Query/graph mask differs from the collection')
    cpu_started, started = time.process_time(), time.perf_counter()
    distances = np.full((len(queries), k), MISSING_DISTANCE, dtype=np.int32)
    found = np.full((len(queries), k), -1, dtype=np.int64)
    graphs = []
    for number, part in enumerate(store.partitions):
        pack_started = time.perf_counter()
        selected = np.flatnonzero(mask[:, number])
        batch = queries if len(selected) == len(queries) else np.ascontiguousarray(queries[selected])
        pack_seconds = time.perf_counter()-pack_started
        if not len(selected):
            graphs.append(dict(graph=number, vectors=int(part.index.ntotal), queries=0,
                native_seconds=0., native_cpu_seconds=0., native_average_cpu_cores=0.,
                configured_faiss_threads=None, mapping_merge_seconds=0., packing_seconds=pack_seconds))
            continue
        local_d, positions, profile = native_search(number, part, batch, k, ef_search, threads)
        merge_started = time.perf_counter()
        local_ids = np.full(positions.shape, -1, dtype=np.int64)
        valid = (positions >= 0) & (positions < len(part.labels))
        local_ids[valid] = part.labels[positions[valid]]
        local_d[~valid] = MISSING_DISTANCE
        candidate_d = np.concatenate((distances[selected], local_d), axis=1)
        candidate_ids = np.concatenate((found[selected], local_ids), axis=1)
        order = np.lexsort((np.where(candidate_ids >= 0, candidate_ids, np.iinfo(np.int64).max), candidate_d), axis=1)[:, :k]
        distances[selected] = np.take_along_axis(candidate_d, order, axis=1)
        found[selected] = np.take_along_axis(candidate_ids, order, axis=1)
        graphs.append(dict(graph=number, vectors=int(part.index.ntotal), queries=len(selected),
            native_seconds=profile['native_seconds'], packing_seconds=pack_seconds,
            native_cpu_seconds=profile['native_cpu_seconds'],
            native_average_cpu_cores=profile['native_average_cpu_cores'],
            configured_faiss_threads=profile['configured_faiss_threads'],
            mapping_merge_seconds=time.perf_counter()-merge_started))
    wall, cpu = time.perf_counter()-started, time.process_time()-cpu_started
    return distances, found, dict(search_seconds=wall, search_cpu_seconds=cpu,
        native_seconds=sum(g['native_seconds'] for g in graphs),
        packing_seconds=sum(g['packing_seconds'] for g in graphs),
        mapping_merge_seconds=sum(g['mapping_merge_seconds'] for g in graphs), graphs=graphs)


def save_summaries(directory, blooms, ranges, graph_sha256, spec_fingerprint, fpr, capacity):
    directory = Path(directory)
    directory.mkdir()
    files = {}
    for layout, filters in blooms.items():
        for graph, bloom in enumerate(filters):
            name = f'{layout}-graph{graph}.npy'
            np.save(directory / name, bloom.bits, allow_pickle=False)
            files[name] = digest(directory / name)
    atomic_json(directory / 'manifest.json', dict(schema_version=1, key_scheme='splitmix64-tagged-v1',
        band_seed=hex(SEED), python_source_sha256=digest(Path(__file__)),
        native_source_sha256=digest(ROOT / 'native/routing.cpp'),
        graph_sha256=graph_sha256, spec_fingerprint=spec_fingerprint, fpr=fpr, capacity=capacity,
        ranges=np.asarray(ranges).tolist(), files=files, layouts={k:list(v) for k,v in LAYOUTS.items()}))


def load_summaries(directory, graph_sha256, spec_fingerprint):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['schema_version'] != 1 or manifest['key_scheme'] != 'splitmix64-tagged-v1' or
            manifest.get('band_seed') != hex(SEED) or manifest.get('python_source_sha256') != digest(Path(__file__)) or
            manifest.get('native_source_sha256') != digest(ROOT / 'native/routing.cpp') or
            manifest['graph_sha256'] != graph_sha256 or manifest['spec_fingerprint'] != spec_fingerprint or
            manifest['layouts'] != {k:list(v) for k,v in LAYOUTS.items()}):
        raise ValueError('Routing summary belongs to a different graph, encoding or layout')
    ranges = np.asarray(manifest['ranges'], dtype=np.int64)
    expected = {f'{layout}-graph{g}.npy' for layout in LAYOUTS for g in range(len(ranges))}
    if set(manifest['files']) != expected:
        raise ValueError('Missing routing summary files')
    blooms = {}
    for layout, (bands, _) in LAYOUTS.items():
        blooms[layout] = []
        for graph in range(len(ranges)):
            path = directory / f'{layout}-graph{graph}.npy'
            if digest(path) != manifest['files'][path.name]:
                raise ValueError('Routing summary checksum differs')
            blooms[layout].append(Bloom(manifest['capacity']*bands, manifest['fpr'],
                                       np.load(path, mmap_mode='r', allow_pickle=False)))
    return blooms, ranges, manifest
