"""One active writer, all-partition search, and atomic manifest checkpoints."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time

import faiss
import numpy as np

from .modes import metric_for_mode, validate_mode
from . import defaults

MISSING_DISTANCE = np.iinfo(np.int32).max


def parse_capacity(value):
    match = re.fullmatch(r'([1-9][0-9]*)([KM]?)', str(value).strip().upper())
    if not match:
        raise ValueError('Capacity must be a positive integer, optionally suffixed K or M')
    return int(match[1]) * {'': 1, 'K': 1000, 'M': 1000000}[match[2]]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _sync_file(path):
    with Path(path).open('rb') as handle:
        os.fsync(handle.fileno())


@dataclass
class Partition:
    index: object
    labels: np.ndarray
    files: dict | None = None


class PartitionedIndex:
    """Single-process writer. Sealed partitions retain their vectors/graph.

    All partitions are resident. Capacity bounds a graph, not total RAM.
    Checkpoint files are immutable until the new manifest is committed.
    """
    def __init__(self, directory, capacity, spec, M=defaults.HNSW_M, distance_mode=defaults.DISTANCE_MODE,
                 max_partitions=None):
        if max_partitions is not None and (type(max_partitions) is not int or max_partitions < 1):
            raise ValueError('max_partitions must be a positive integer or None')
        validate_mode(distance_mode, spec.M_bits)
        self.distance_mode = distance_mode
        self.metric = metric_for_mode(distance_mode)
        self.directory = Path(directory)
        self.capacity = parse_capacity(capacity)
        self.max_partitions = max_partitions
        self.spec = spec
        self.M = int(M)
        if self.M < 2:
            raise ValueError('HNSW M must be at least 2')
        self.partitions = []
        self.known_ids = set()
        self.generation = 0
        self.progress = {}
        self._dirty = set()
        self._ef_search = defaults.EF_SEARCH
        self.routing = None
        self._saved_routing = None

    def configure_routing(self, mode='off', layout='37x3', fpr=1e-4, zero_policy='admit', threads=1):
        if threads < 1:
            raise ValueError('Routing threads must be positive')
        if mode == 'off':
            if self.routing is not None:
                raise ValueError('Use a fresh collection to disable persisted routing')
            return
        from .live_routing import LiveBloomRouter
        expected = dict(mode=mode, layout=layout, fpr=float(fpr), zero_policy=zero_policy)
        if self.routing is not None:
            if self.routing.config != expected:
                raise ValueError('Routing configuration differs from the collection')
            self.routing.threads = int(threads)
        else:
            self.routing = LiveBloomRouter(self, threads=threads, saved=self._saved_routing, **expected)

    @property
    def ntotal(self):
        return sum(int(p.index.ntotal) for p in self.partitions)

    @property
    def population_cache_bytes(self):
        return sum(int(p.index.minhash_bitmap_popcount_cache_size()) * 2 for p in self.partitions)

    @property
    def active_partition(self):
        if self.partitions and self.partitions[-1].index.ntotal < self.capacity:
            return len(self.partitions) - 1
        return None  # Next add allocates a new target lazily, without an empty file.

    def _vectors(self, vectors):
        array = np.asarray(vectors)
        if array.dtype != np.uint8 or array.ndim != 2 or array.shape[1] != self.spec.M_bits // 8:
            raise ValueError('Expected uint8[N, M_bits/8] bitmap vectors')
        return np.ascontiguousarray(array)

    def contains(self, ids):
        return np.fromiter((int(i) in self.known_ids for i in ids), dtype=bool, count=len(ids))

    def _check_partition_limit(self, incoming=0, counts=None):
        if self.max_partitions is None:
            return
        counts = [int(part.index.ntotal) for part in self.partitions] if counts is None else counts
        if len(counts) > self.max_partitions:
            raise ValueError(f'Partition limit exceeded: maximum {self.max_partitions}')
        if (any(type(count) is not int or not 0 <= count <= self.capacity for count in counts)
                or any(count != self.capacity for count in counts[:-1])):
            raise ValueError('Invalid existing partition capacity or ordering')
        if sum(counts) + incoming > self.capacity * self.max_partitions:
            raise ValueError(f'Partition limit would be exceeded: maximum {self.max_partitions} '
                             f'with capacity {self.capacity}')

    def add(self, vectors, ids, ef_construction, profile=None, sketches=None):
        started = time.perf_counter() if profile is not None else None
        vectors = self._vectors(vectors)
        ids = np.asarray(ids)
        if ids.ndim != 1 or len(ids) != len(vectors) or ids.dtype.kind not in 'iu':
            raise ValueError('IDs must be an integer vector aligned with bitmap rows')
        if ids.dtype.kind == 'u' and ids.size and ids.max() > np.iinfo(np.int64).max:
            raise ValueError('IDs must fit int64')
        ids = ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or len(np.unique(ids)) != len(ids) or self.contains(ids).any():
            raise ValueError('Document IDs must be nonnegative and unique across the collection')
        if ef_construction <= 0:
            raise ValueError('efConstruction must be positive')
        # Reject the whole call before native insertion, routing, labels or
        # profile changes; an oversized call must not partially fill a graph.
        self._check_partition_limit(len(vectors))
        if profile is not None:
            profile.update(graphs=[], input_vectors=len(vectors))
        routing_update_ms = 0.
        if self.routing is not None:
            routing_started = time.perf_counter()
            keys = self.routing.prepare(sketches, len(vectors))
            routing_update_ms += (time.perf_counter()-routing_started)*1000
        offset = 0
        while offset < len(vectors):
            segment_start = time.perf_counter() if profile is not None else None
            created = self.active_partition is None
            if self.active_partition is None:
                index = faiss.IndexBinaryHNSW(self.spec.M_bits, self.M)
                index.metric_type = self.metric
                index.hnsw.efSearch = self._ef_search
                self.partitions.append(Partition(index, np.empty(0, dtype=np.int64)))
            number = len(self.partitions) - 1
            part = self.partitions[number]
            before = int(part.index.ntotal)
            count = min(self.capacity - part.index.ntotal, len(vectors) - offset)
            part.index.hnsw.efConstruction = int(ef_construction)
            native_start = time.perf_counter() if profile is not None else None
            native_cpu_start = time.process_time() if profile is not None else None
            part.index.add(vectors[offset:offset + count])
            native_ms = (time.perf_counter() - native_start) * 1000 if profile is not None else None
            native_cpu = time.process_time() - native_cpu_start if profile is not None else None
            tail = ids[offset:offset + count]
            part.labels = np.concatenate((part.labels, tail))
            self.known_ids.update(map(int, tail))
            self._dirty.add(number)
            if self.routing is not None:
                routing_started = time.perf_counter()
                self.routing.append(number, before, keys[offset:offset+count])
                routing_update_ms += (time.perf_counter()-routing_started)*1000
            offset += count
            if profile is not None:
                profile['graphs'].append({'graph': number, 'created': created, 'before': before,
                    'inserted': count, 'after': int(part.index.ntotal), 'native_ms': native_ms,
                    'native_cpu_seconds': native_cpu,
                    'native_average_cpu_cores': native_cpu/(native_ms/1000) if native_ms else None,
                    'configured_faiss_threads': faiss.omp_get_max_threads(),
                    'total_ms': (time.perf_counter() - segment_start) * 1000})
        if profile is not None:
            if self.routing is not None:
                profile['routing_update_ms'] = routing_update_ms
            profile['total_ms'] = (time.perf_counter() - started) * 1000

    def search(self, vectors, k, ef_search, profile=None, sketches=None):
        started = time.perf_counter() if profile is not None else None
        if profile is not None:
            profile.update(graphs=[], queries=len(vectors), k=k, ef_search=ef_search)
        vectors = self._vectors(vectors)
        if k <= 0 or ef_search <= 0:
            raise ValueError('k and efSearch must be positive')
        self._ef_search = int(ef_search)
        if self.routing is not None and sketches is not None:
            return self.routing.search(vectors, k, int(ef_search), sketches, profile)
        distances = np.full((len(vectors), k), MISSING_DISTANCE, dtype=np.int32)
        ids = np.full((len(vectors), k), -1, dtype=np.int64)
        if len(vectors) == 0:
            if profile is not None:
                profile['total_ms'] = (time.perf_counter() - started) * 1000
            return distances, ids
        for number, part in enumerate(self.partitions):
            count = min(k, int(part.index.ntotal))
            if not count:
                continue
            part.index.hnsw.efSearch = int(ef_search)
            native_start = time.perf_counter() if profile is not None else None
            local_d, positions = part.index.search(vectors, count)
            native_ms = (time.perf_counter() - native_start) * 1000 if profile is not None else None
            merge_start = time.perf_counter() if profile is not None else None
            local_ids = np.full(positions.shape, -1, dtype=np.int64)
            valid = (positions >= 0) & (positions < len(part.labels))
            local_ids[valid] = part.labels[positions[valid]]
            local_d[~valid] = MISSING_DISTANCE
            candidate_d = np.concatenate((distances, local_d), axis=1)
            candidate_ids = np.concatenate((ids, local_ids), axis=1)
            tie_ids = np.where(candidate_ids >= 0, candidate_ids, np.iinfo(np.int64).max)
            order = np.lexsort((tie_ids, candidate_d), axis=1)[:, :k]
            distances = np.take_along_axis(candidate_d, order, axis=1)
            ids = np.take_along_axis(candidate_ids, order, axis=1)
            if profile is not None:
                graph_stop = time.perf_counter()
                profile['graphs'].append({'graph': number, 'vectors': int(part.index.ntotal),
                    'native_ms': native_ms, 'mapping_merge_ms': (graph_stop - merge_start) * 1000,
                    'total_ms': (graph_stop - native_start) * 1000})
        if profile is not None:
            profile['total_ms'] = (time.perf_counter() - started) * 1000
        return distances, ids

    def save(self, progress=None):
        self._check_partition_limit()
        self.directory.mkdir(parents=True, exist_ok=True)
        files_dir = self.directory / 'partitions'
        files_dir.mkdir(exist_ok=True)
        generation = self.generation + 1
        references = []
        old_files = []
        for number, part in enumerate(self.partitions):
            if number in self._dirty:
                prefix = f'partition_{number:06d}.g{generation:08d}'
                refs = {'index': f'partitions/{prefix}.faiss', 'labels': f'partitions/{prefix}.npy'}
                faiss.write_index_binary(part.index, str(self.directory / refs['index']))
                np.save(self.directory / refs['labels'], part.labels, allow_pickle=False)
                for value in refs.values():
                    _sync_file(self.directory / value)
                if part.files:
                    old_files.extend(part.files.values())
            else:
                refs = part.files
            references.append({'id': number, 'count': len(part.labels), **refs})
        saved_progress = self.progress if progress is None else progress
        manifest = {
            'schema_version': 2, 'metric': 'minhash_bitmap', 'distance_scale': 100,
            'distance_mode': self.distance_mode,
            'capacity': self.capacity, 'M': self.M, 'd_bits': self.spec.M_bits,
            'sketch_fingerprint': self.spec.fingerprint(), 'generation': generation,
            'active_partition': self.active_partition, 'ntotal': self.ntotal,
            'partitions': references, 'progress': saved_progress,
        }
        if self.max_partitions is not None:
            manifest['max_partitions'] = self.max_partitions
        routing_metadata = self.routing.checkpoint(generation, references) if self.routing is not None else None
        if routing_metadata is not None:
            manifest['routing'] = routing_metadata
        atomic_json(self.directory / 'collection.json', manifest)
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.generation = generation
        self.progress = json.loads(json.dumps(saved_progress))
        if self.routing is not None:
            old_files.extend(self.routing.committed(routing_metadata))
            self._saved_routing = routing_metadata
        for part, refs in zip(self.partitions, references):
            part.files = {key: refs[key] for key in ('index', 'labels')}
        self._dirty.clear()
        # Only superseded files generated by this checkpoint writer are removed.
        # A crash before manifest replacement leaves the previous checkpoint valid.
        for filename in old_files:
            (self.directory / filename).unlink(missing_ok=True)

    @classmethod
    def load(cls, directory, capacity, spec, M=defaults.HNSW_M, distance_mode=defaults.DISTANCE_MODE,
             max_partitions=None):
        collection = cls(directory, capacity, spec, M, distance_mode, max_partitions=max_partitions)
        manifest = json.loads((collection.directory / 'collection.json').read_text())
        if manifest.get('schema_version') == 1:
            manifest['distance_mode'] = 'v2_auto'
        elif manifest.get('schema_version') != 2:
            raise ValueError('Collection configuration mismatch for schema_version')
        expected = {'metric': 'minhash_bitmap', 'distance_scale': 100,
                    'distance_mode': collection.distance_mode,
                    'capacity': collection.capacity, 'M': collection.M, 'd_bits': spec.M_bits,
                    'sketch_fingerprint': spec.fingerprint()}
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f'Collection configuration mismatch for {key}')
        saved_limit = manifest.get('max_partitions')
        if saved_limit is not None and (type(saved_limit) is not int or saved_limit < 1):
            raise ValueError('Invalid saved max_partitions')
        if saved_limit != collection.max_partitions:
            raise ValueError('Collection configuration mismatch for max_partitions')
        if collection.max_partitions is not None:
            counts = [entry['count'] for entry in manifest['partitions']]
            collection._check_partition_limit(counts=counts)
            if type(manifest['ntotal']) is not int or manifest['ntotal'] != sum(counts):
                raise ValueError('Collection counts exceed or differ from the partition limit')
        for number, entry in enumerate(manifest['partitions']):
            if entry['id'] != number or not 0 < entry['count'] <= collection.capacity:
                raise ValueError('Invalid partition ordering or capacity')
            if number < len(manifest['partitions']) - 1 and entry['count'] != collection.capacity:
                raise ValueError('Only the last partition may be partially filled')
            for key in ('index', 'labels'):
                target = (collection.directory / entry[key]).resolve()
                if not target.is_relative_to(collection.directory.resolve()):
                    raise ValueError('Partition path escapes collection directory')
            index = faiss.read_index_binary(str(collection.directory / entry['index']))
            labels = np.load(collection.directory / entry['labels'], allow_pickle=False)
            if labels.dtype != np.int64 or labels.ndim != 1 or len(labels) != entry['count']:
                raise ValueError('Invalid partition labels')
            if (index.d != spec.M_bits or index.ntotal != len(labels)
                    or index.metric_type != collection.metric
                    or not hasattr(index, 'hnsw') or index.hnsw.nb_neighbors(1) != M):
                raise ValueError('Partition index metadata mismatch')
            if (np.any(labels < 0) or len(np.unique(labels)) != len(labels)
                    or collection.contains(labels).any()):
                raise ValueError('Duplicate or invalid document IDs in collection')
            collection.known_ids.update(map(int, labels))
            collection.partitions.append(Partition(index, labels, {k: entry[k] for k in ('index', 'labels')}))
        if collection.ntotal != manifest['ntotal'] or collection.active_partition != manifest['active_partition']:
            raise ValueError('Collection counts/active partition mismatch')
        collection.generation = manifest['generation']
        collection.progress = manifest.get('progress', {})
        if collection.partitions:
            collection._ef_search = int(collection.partitions[-1].index.hnsw.efSearch)
        collection._saved_routing = manifest.get('routing')
        if collection._saved_routing is not None:
            collection.configure_routing(**collection._saved_routing['config'])
        return collection
