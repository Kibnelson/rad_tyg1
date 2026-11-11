#!/usr/bin/env python3
"""Screen every input batch with the original full-reference SIMD baseline."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/simd')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=28)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--max-documents', type=int)
    parser.add_argument('--partition-capacity', type=int, default=1000000,
                        help='Accepted for the shared CLI; this baseline has one complete retained reference.')
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.partition_capacity) < 1 or (args.max_documents is not None and args.max_documents < 1):
        parser.error('Counts must be positive')
    if args.output.exists():
        parser.error('Use a new output directory; existing results are preserved')
    import numpy as np
    import simd_full_reference_v1 as native
    import simd_fuzzy_deduplicationv7_unsorted_multi7A_V1 as batch_native
    from engine.encoding import PersistentMinHasher, SketchSpec
    from common.stream import iter_batches
    args.output.mkdir(parents=True)
    reference_path = args.output / 'reference.u32'
    reference_path.touch()
    reference_rows = 0
    ordinal = 0
    reports = []
    stage_names = ('minhash_seconds', 'in_batch_dedup_seconds', 'reference_staging_seconds',
                   'index_search_seconds', 'decision_seconds', 'insert_seconds')
    with PersistentMinHasher(SketchSpec(), args.threads, 128) as hasher, \
            (args.output / 'decisions.jsonl').open('x') as decisions, \
            (args.output / 'accepted_ids.jsonl').open('x') as accepted_ids:
        iterator = iter(iter_batches(args.input, args.batch_size, args.max_documents))
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            timings = {'input_seconds': time.perf_counter() - started}
            t = time.perf_counter()
            sketches = hasher.compute([row['text'] for row in batch])
            timings['minhash_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            groups = getattr(batch_native, 'simd_fuzzy_deduplicationv7_unsorted_multi7A_V1')(
                np.ascontiguousarray(sketches, dtype=np.uint64), threshold=.7, num_bands=14, num_threads=args.threads)
            in_batch = {int(i) for group in groups for i in group}
            assert all(0 <= i < len(batch) for i in in_batch)
            keep = np.asarray([i for i in range(len(batch)) if i not in in_batch], dtype=np.int64)
            timings['in_batch_dedup_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            scratch = args.output / 'union.npy'
            matrix = None
            if len(keep):
                matrix = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.uint64,
                                                  shape=(reference_rows + len(keep), 112))
                if reference_rows:
                    old = np.memmap(reference_path, dtype=np.uint32, mode='r', shape=(reference_rows, 112))
                    for begin in range(0, reference_rows, 100000):
                        stop = min(reference_rows, begin + 100000)
                        matrix[begin:stop] = old[begin:stop]
                    del old
                matrix[reference_rows:] = sketches[keep]
                matrix.flush()
            timings['reference_staging_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            found = native.full_reference(matrix, reference_rows, .7, 14, args.threads) if len(keep) else {'removed': [], 'edges': []}
            timings['index_search_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            removed = {int(i) for group in found['removed'] for i in group if int(i) >= reference_rows}
            assert all(i < reference_rows + len(keep) for i in removed)
            accepted = np.asarray([int(pos) for j, pos in enumerate(keep) if reference_rows + j not in removed], dtype=np.int64)
            accepted_set = set(map(int, accepted))
            observations = [dict(ordinal=ordinal + i, source_id=row['id'], inserted=i in accepted_set,
                                 reason='in_batch' if i in in_batch else 'reference_search' if i not in accepted_set else 'accepted')
                            for i, row in enumerate(batch)]
            timings['decision_seconds'] = time.perf_counter() - t
            if matrix is not None:
                del matrix
                scratch.unlink()
            t = time.perf_counter()
            with reference_path.open('ab') as f:
                f.write(np.ascontiguousarray(sketches[accepted], dtype=np.uint32).tobytes())
            reference_rows += len(accepted)
            timings['insert_seconds'] = time.perf_counter() - t
            seconds = time.perf_counter() - started
            report = dict(batch=len(reports) + 1, screened=len(batch), inserted=len(accepted),
                          index_search_drops=len(removed), in_batch_dedup_drops=len(in_batch),
                          retained=reference_rows, processing_seconds=seconds, throughput=len(batch) / seconds, **timings)
            assert len(batch) == report['inserted'] + report['index_search_drops'] + report['in_batch_dedup_drops']
            reports.append(report)
            for row in observations:
                decisions.write(json.dumps(row) + '\n')
            for pos in accepted:
                accepted_ids.write(json.dumps(dict(ordinal=ordinal + int(pos), source_id=batch[int(pos)]['id'])) + '\n')
            decisions.flush()
            accepted_ids.flush()
            ordinal += len(batch)
            print(f"Batch {report['batch']}: screened={len(batch)} inserted={len(accepted)} QPS={report['throughput']:.2f}", flush=True)
    if not reports:
        raise ValueError('No input documents')
    with (args.output / 'batches.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(reports[0])); writer.writeheader(); writer.writerows(reports)
    seconds = math.fsum(r['processing_seconds'] for r in reports)
    summary = dict(system='SIMD-DPK', screened=ordinal, inserted=reference_rows,
                   index_search_drops=sum(r['index_search_drops'] for r in reports),
                   in_batch_dedup_drops=sum(r['in_batch_dedup_drops'] for r in reports),
                   processing_seconds=seconds, throughput=ordinal / seconds,
                   recall='not_measured', exact_jaccard='not_computed', full_text=True,
                   profile=dict(minhash_values=112, bands=14, rows_per_band=8, threshold=.7,
                                threads=args.threads, batch_size=args.batch_size, reference='all_retained_rows'),
                   scope='Native SIMD full-reference processing; input/MinHash/staging/search/decisions/append included. Evidence writes and final summaries excluded. No ranked top-k or index partitions.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
