"""The baseline's encoding, shared by corpus and query processing."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing
import os
import time

import mmh3
import numpy as np


@dataclass(frozen=True)
class SketchSpec:
    K: int = 112
    M_bits: int = 4096
    perms_seed: int = 49037
    mmh3_seed: int = 9173
    window: int = 5
    global_seed: int = 12345

    def __post_init__(self):
        if self.K <= 0 or self.window <= 0 or self.M_bits <= 0 or self.M_bits % 64:
            raise ValueError('K/window must be positive; M_bits must be a positive multiple of 64')

    def permutations(self):
        rng = np.random.RandomState(self.perms_seed)
        a = np.array([rng.randint(0, np.uint64(2**64 - 1), dtype=np.uint64)
                      for _ in range(self.K)], dtype=np.uint64)
        a[a % 2 == 0] += 1
        return a

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()
                              + self.permutations().tobytes()).hexdigest()


def minhash(text, permutations, window=5, block_size=8192):
    words = text.split()
    if not words:
        return np.zeros(len(permutations), dtype=np.uint32)
    shingles = [' '.join(words[i:i + window]) for i in range(max(1, len(words) - window + 1))]
    hashes = np.array([mmh3.hash(s, signed=False) for s in shingles], dtype=np.uint64)
    best = np.full(len(permutations), np.uint64(2**64 - 1), dtype=np.uint64)
    for start in range(0, len(hashes), block_size):
        values = (hashes[start:start + block_size, None] * permutations[None, :]) >> np.uint64(32)
        np.minimum(best, values.min(axis=0), out=best)
    return best.astype(np.uint32)


def _minhash_chunk(args):
    texts, spec = args
    permutations = spec.permutations()
    return np.stack([minhash(text, permutations, spec.window) for text in texts])


def _profiled_minhash_chunk(args):
    started, cpu_started = time.perf_counter(), time.process_time()
    sketches = _minhash_chunk(args)
    return sketches, os.getpid(), started, time.perf_counter(), time.process_time()-cpu_started


def summarize_worker_metrics(results, requested_workers, start_method):
    """Task CPU time measures computation; overlapping wall time is separate."""
    by_pid, events = {}, []
    for _,pid,start,stop,cpu_seconds in results:
        row = by_pid.setdefault(pid, {'pid':pid,'tasks':0,'cpu_seconds':0.0,'compute_wall_seconds':0.0,
                                     'first_task_start':start,'last_task_stop':stop})
        row['tasks'] += 1
        row['cpu_seconds'] += cpu_seconds
        row['compute_wall_seconds'] += stop-start
        row['first_task_start'] = min(row['first_task_start'],start)
        row['last_task_stop'] = max(row['last_task_stop'],stop)
        events.extend(((start,1),(stop,-1)))
    active, peak = 0, 0
    for _,change in sorted(events):
        active += change
        peak = max(peak,active)
    starts = [row['first_task_start'] for row in by_pid.values()]
    stops = [row['last_task_stop'] for row in by_pid.values()]
    return {'tasks':len(results),'requested_workers':requested_workers,'worker_pids':sorted(by_pid),
        'active_worker_count':len(by_pid),'worker_cpu_seconds_sum':sum(row['cpu_seconds'] for row in by_pid.values()),
        'worker_compute_seconds_sum':sum(row['compute_wall_seconds'] for row in by_pid.values()),
        'peak_overlapping_compute_tasks':peak,'worker_cpu_by_pid':[by_pid[pid] for pid in sorted(by_pid)],
        'compute_span_seconds':max(stops)-min(starts) if starts else 0.0,
        'worker_start_spread_seconds':max(starts)-min(starts) if starts else 0.0,
        'tail_after_median_worker_seconds':max(stops)-float(np.median(stops)) if stops else 0.0,
        'max_task_wall_seconds':max((stop-start for _,_,start,stop,_ in results),default=0.0),
        'start_method':start_method}


def compute_minhashes(texts, spec, workers=1, metrics=None):
    texts = list(texts)
    if not texts:
        if metrics is not None:
            metrics.update(summarize_worker_metrics([],int(workers),multiprocessing.get_start_method()))
        return np.empty((0, spec.K), dtype=np.uint32)
    workers = min(max(1, int(workers)), len(texts))
    size = (len(texts) + workers - 1) // workers
    chunks = [(texts[i:i + size], spec) for i in range(0, len(texts), size)]
    if metrics is not None:
        if workers == 1:
            results = [_profiled_minhash_chunk(chunks[0])]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_profiled_minhash_chunk,chunks))
        metrics.update(summarize_worker_metrics(results,workers,multiprocessing.get_start_method()))
        return np.vstack([result[0] for result in results])
    if workers == 1:
        return _minhash_chunk(chunks[0])
    # map preserves input order, so IDs and sketches stay aligned.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.vstack(list(pool.map(_minhash_chunk, chunks)))


def _init_minhash_worker(spec):
    global _worker_spec, _worker_permutations
    _worker_spec, _worker_permutations = spec, spec.permutations()


def _persistent_minhash_chunk(texts):
    started, cpu_started = time.perf_counter(), time.process_time()
    sketches = np.stack([minhash(text, _worker_permutations, _worker_spec.window) for text in texts])
    return sketches, os.getpid(), started, time.perf_counter(), time.process_time()-cpu_started


class PersistentMinHasher:
    """Pilot encoder: reuse spawned workers without inheriting a resident graph.

    Tasks are smaller than one worker's share to balance variable text lengths.
    The first compute includes worker startup; later calls reuse the same pool.
    map preserves order and each worker uses the unchanged baseline arithmetic.
    """
    def __init__(self, spec, workers=1, chunk_size=64):
        if workers <= 0 or chunk_size <= 0:
            raise ValueError('workers and chunk_size must be positive')
        self.spec, self.workers, self.chunk_size = spec, workers, chunk_size
        self.pool = None
        self.closed = False
        self.last_metrics = {}

    def compute(self, texts):
        if self.closed:
            raise RuntimeError('MinHash pool is closed')
        texts = list(texts)
        if not texts:
            self.last_metrics = summarize_worker_metrics([],self.workers,'spawn')
            return np.empty((0, self.spec.K), dtype=np.uint32)
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_minhash_worker, initargs=(self.spec,))
        chunks = [texts[i:i + self.chunk_size] for i in range(0, len(texts), self.chunk_size)]
        results = list(self.pool.map(_persistent_minhash_chunk, chunks))
        self.last_metrics = summarize_worker_metrics(results,self.workers,'spawn')
        return np.vstack([r[0] for r in results])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


def to_bitmaps(sketches, spec, mode='python', threads=1, metrics=None):
    if mode == 'native':
        from .bitmap_native import encode
        return encode(sketches,spec,threads,metrics)
    if mode != 'python':
        raise ValueError('Bitmap mode must be python or native')
    if metrics is not None:
        metrics.update(mode='python',requested_threads=1,actual_team_threads=1)
    return to_bitmaps_python(sketches,spec)


def to_bitmaps_python(sketches, spec):
    """Original baseline implementation, retained as the byte-parity oracle."""
    sketches = np.asarray(sketches)
    if sketches.ndim != 2 or sketches.shape[1] != spec.K:
        raise ValueError('Expected an N × K sketch matrix')
    out = np.zeros((len(sketches), spec.M_bits // 64), dtype=np.uint64)
    pow2 = spec.M_bits & (spec.M_bits - 1) == 0
    for row, values in enumerate(sketches):
        for value in values:
            data = (int(value) & 0xFFFFFFFF).to_bytes(4, 'little')
            h = mmh3.hash64(data, seed=spec.mmh3_seed, signed=False)[0]
            bucket = h & (spec.M_bits - 1) if pow2 else h % spec.M_bits
            out[row, bucket >> 6] |= np.uint64(1) << np.uint64(bucket & 63)
    return out.view(np.uint8).reshape(len(sketches), spec.M_bits // 8)
