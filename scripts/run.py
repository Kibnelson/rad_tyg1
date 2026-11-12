#!/usr/bin/env python3
"""Screen every input document with native RAD; preserve per-batch evidence and checkpoints."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'external/simd')]
from radv2 import defaults


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + '\n'


def fingerprint(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def batches(path, size, maximum):
    rows = []; count = 0
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('text'), str):
                raise ValueError('Input must contain JSONL objects with a text string and optional id.')
            rows.append(dict(id=row.get('id', count), text=row['text']))
            count += 1
            if len(rows) == size:
                yield rows; rows = []
            if maximum and count >= maximum:
                break
    if rows:
        yield rows


def reports(out, state, store, identity, status):
    rows = state['batches']; seconds = math.fsum(r['processing_seconds'] for r in rows)
    counts = {k: sum(r[k] for r in rows) for k in ('screened', 'inserted', 'index_search_drops', 'in_batch_dedup_drops')}
    summary = dict(system='RAD', status=status, **counts, stored_rows=int(store.ntotal),
        processing_seconds=seconds, throughput=counts['screened'] / seconds if seconds else None,
        partition_counts=[int(p.index.ntotal) for p in store.partitions], profile=identity['profile'],
        input_sha256=identity['input_sha256'], exact_jaccard='not_computed',
        timing_scope='Summed input read, MinHash, in-batch SIMD, bitmap, Bloom/search/merge, decisions and accepted insertion. Evidence/checkpoint/report writes are outside processing throughput.',
        index_id_policy='Zero-based input ordinal; source_id preserves the supplied id.',
        evidence='evidence/batch_*.jsonl preserves every ordered returned candidate and insertion decision.')
    atomic(out / 'summary.json', json_text(summary))
    if rows:
        buffer = io.StringIO(newline=''); writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows); atomic(out / 'batches.csv', buffer.getvalue())
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--partition-capacity', type=int, default=1000000)
    ap.add_argument('--max-documents', type=int)
    ap.add_argument('--M', type=int, default=96)
    ap.add_argument('--ef-construction', type=int, choices=(128,), default=128)
    ap.add_argument('--ef-search', type=int, choices=(150,), default=150)
    ap.add_argument('--top-k', type=int, choices=(4,), default=4)
    ap.add_argument('--cutoff', type=int, choices=(defaults.BITMAP_DISTANCE_CUTOFF,),
                    default=defaults.BITMAP_DISTANCE_CUTOFF)
    ap.add_argument('--distance-mode', choices=('cache_simd', 'cache_scalar'), default='cache_simd')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.threads > len(os.sched_getaffinity(0)):
        ap.error('--threads exceeds available logical CPUs')
    if min(args.threads, args.batch_size, args.partition_capacity, args.ef_construction, args.ef_search, args.top_k) < 1 or args.M < 2:
        ap.error('Counts and graph budgets must be positive, with M >= 2')
    if args.max_documents is not None and args.max_documents < 1:
        ap.error('--max-documents must be positive')
    if not 0 <= args.cutoff <= 100:
        ap.error('--cutoff must be between 0 and 100')
    args.input = args.input.expanduser().resolve(); out = args.output.expanduser().resolve()
    if not args.input.is_file():
        ap.error('Input JSONL file does not exist')
    if out.exists() and any(p.name != '.writer.lock' for p in out.iterdir()) and not args.resume:
        ap.error('Output exists; use a fresh path or --resume')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.writer.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('Another process owns this output directory')
    import faiss
    import numpy as np
    from radv2.encoding import SketchSpec, PersistentMinHasher, to_bitmaps
    from radv2.partitions import PartitionedIndex
    from radv2.public_simd import keep_indices
    from radv2.modes import validate_mode
    spec = SketchSpec()
    try:
        validate_mode(args.distance_mode, spec.M_bits)
    except RuntimeError as error:
        ap.error(str(error))
    faiss.omp_set_num_threads(args.threads)
    profile = dict(M=args.M, ef_construction=args.ef_construction, query_ef_construction=args.ef_construction,
        ef_search=args.ef_search, top_k=args.top_k, cutoff=args.cutoff, distance_mode=args.distance_mode,
        bitmap_similarity_target=defaults.BITMAP_SIMILARITY_TARGET,
        distance_serialization='floor(100 * (1 - bitmap_Jaccard) + 1e-6)',
        threads=args.threads, batch_size=args.batch_size, partition_capacity=args.partition_capacity,
        routing='bloom_37x3', zero_route='admit', text_mode='full_text', minhash_values=112)
    identity = dict(kind='public_rad_jsonl_stream_v1', input_sha256=fingerprint(args.input), profile=profile)
    identity_path = out / 'identity.json'; collection = out / 'collection'
    if args.resume:
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            ap.error('Resume input or configuration differs from the saved run')
        store = PartitionedIndex.load(collection, args.partition_capacity, spec, M=args.M, distance_mode=args.distance_mode)
        state = store.progress
        if args.max_documents is not None and args.max_documents < state['screened']:
            ap.error('--max-documents is smaller than the already committed prefix')
    else:
        atomic(identity_path, json_text(identity))
        store = PartitionedIndex(collection, args.partition_capacity, spec, M=args.M, distance_mode=args.distance_mode)
        state = dict(screened=0, batches=[])
    store.configure_routing('bloom', '37x3', 1e-4, 'admit', args.threads)
    if not args.resume:
        store.save(progress=state)
    status = 'completed'; offset = 0
    hasher = PersistentMinHasher(spec, args.threads)
    try:
        iterator = iter(batches(args.input, args.batch_size, args.max_documents))
        while True:
            started = time.perf_counter()
            try:
                rows = next(iterator)
            except StopIteration:
                break
            end = offset + len(rows)
            if end <= state['screened']:
                offset = end; continue
            if offset != state['screened']:
                raise ValueError('Resume cursor does not align with the input batch boundary')
            timing = dict(input_seconds=time.perf_counter() - started)
            t = time.perf_counter(); sketches = hasher.compute([r['text'] for r in rows]); timing['minhash_seconds'] = time.perf_counter() - t
            t = time.perf_counter(); keep = keep_indices(sketches, args.threads); timing['in_batch_dedup_seconds'] = time.perf_counter() - t
            t = time.perf_counter(); vectors = to_bitmaps(sketches[keep], spec, 'native', args.threads); timing['bitmap_seconds'] = time.perf_counter() - t
            ids = np.arange(offset, end, dtype=np.int64); selected_ids = ids[keep]
            search = {}; t = time.perf_counter()
            distances, found = store.search(vectors, args.top_k, args.ef_search, profile=search, sketches=sketches[keep])
            timing['search_seconds'] = time.perf_counter() - t
            t = time.perf_counter(); duplicate = ((found >= 0) & (distances <= args.cutoff)).any(axis=1)
            accepted = ~duplicate; timing['decision_seconds'] = time.perf_counter() - t
            t = time.perf_counter(); store.add(vectors[accepted], selected_ids[accepted], args.ef_construction, sketches=sketches[keep][accepted])
            timing['insert_seconds'] = time.perf_counter() - t
            timing['processing_seconds'] = time.perf_counter() - started
            timing['unattributed_seconds'] = max(0., timing['processing_seconds'] - math.fsum(
                value for key, value in timing.items() if key != 'processing_seconds'))
            positions = {int(pos): j for j, pos in enumerate(keep)}
            evidence = []
            for pos, row in enumerate(rows):
                j = positions.get(pos)
                candidates = ([] if j is None else [dict(index_id=int(i), bitmap_distance=int(d), rank=rank)
                    for rank, (i, d) in enumerate(zip(found[j], distances[j]), 1) if i >= 0])
                evidence.append(dict(index_id=offset + pos, source_id=row['id'], in_batch_drop=j is None,
                    inserted=bool(j is not None and accepted[j]), candidates=candidates))
            number = len(state['batches']) + 1
            atomic(out / 'evidence' / f'batch_{number:06d}.jsonl', ''.join(json.dumps(r, allow_nan=False) + '\n' for r in evidence))
            fanout = search.get('routing', {})
            result = dict(batch=number, screened=len(rows), inserted=int(accepted.sum()),
                index_search_drops=int(duplicate.sum()), in_batch_dedup_drops=len(rows) - len(keep),
                query_partition_visits=fanout.get('actual_query_graph_pairs', 0),
                available_partitions=fanout.get('available_graphs', len(store.partitions)),
                search_eligible_queries=len(keep), zero_route_queries=fanout.get('zero_candidate_queries', len(keep)),
                native_searched_queries=len(keep) - fanout.get('zero_candidate_queries', len(keep)), **timing)
            atomic(out / 'evidence' / f'batch_{number:06d}_metadata.json',
                   json_text(dict(counts_and_timing=result, search_profile=search)))
            state = dict(screened=end, batches=state['batches'] + [result])
            store.save(progress=state)
            summary = reports(out, state, store, identity, 'running')
            print(f"batch={number} screened={end} retained={store.ntotal} QPS={len(rows)/timing['processing_seconds']:.2f}", flush=True)
            offset = end
    except KeyboardInterrupt:
        status = 'interrupted'
        store = PartitionedIndex.load(collection, args.partition_capacity, spec, M=args.M, distance_mode=args.distance_mode)
        state = store.progress
        print('Interrupted. Resume from the last completed checkpoint with the same command plus --resume.', flush=True)
    finally:
        hasher.__exit__(None, None, None)
    summary = reports(out, state, store, identity, status)
    if not summary['screened']:
        raise SystemExit('No input documents were processed')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
