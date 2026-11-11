#!/usr/bin/env python3
"""Stream full-text inputs through the original raw-MinHash Faiss HNSW metric."""
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
    parser.add_argument('--partition-capacity', type=int, default=20000000,
                        help='Retained-vector ceiling for this single-graph baseline; increase for larger runs.')
    parser.add_argument('--M', type=int, default=96)
    parser.add_argument('--ef-construction', type=int, default=384)
    parser.add_argument('--ef-search', type=int, default=400)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--cutoff', type=int, default=30)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.partition_capacity, args.M, args.ef_construction, args.ef_search, args.top_k) < 1 or not 0 <= args.cutoff <= 100:
        parser.error('Invalid positive count or distance cutoff')
    if args.max_documents is not None and args.max_documents < 1:
        parser.error('--max-documents must be positive')
    if args.output.exists():
        parser.error('Use a new output directory; existing results are preserved')
    import numpy as np
    import faiss
    import simd_fuzzy_deduplicationv7_unsorted_multi7A_V1 as batch_native
    from engine.encoding import PersistentMinHasher, SketchSpec
    from engine.mh_jaccard import pack_signatures
    from common.stream import iter_batches
    if not hasattr(faiss, 'METRIC_MinhashJaccard'):
        raise RuntimeError('Build the vendored modified Faiss with bash setup.sh; the stock pip wheel cannot run this metric.')
    faiss.omp_set_num_threads(args.threads)
    index = faiss.IndexBinaryHNSW(3584, args.M)
    index.metric_type = faiss.METRIC_MinhashJaccard
    if not index.minhash_jaccard_uses_avx512():
        raise RuntimeError('This metric requires AVX-512F, FMA and POPCNT; no fallback metric is substituted.')
    index.hnsw.efConstruction = args.ef_construction
    index.hnsw.efSearch = args.ef_search
    args.output.mkdir(parents=True)
    labels = []
    reports = []
    ordinal = 0
    with PersistentMinHasher(SketchSpec(), args.threads, 128) as hasher, \
            (args.output / 'topk.jsonl').open('x') as evidence, \
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
            filtered = {int(i) for group in groups for i in group}
            assert all(0 <= i < len(batch) for i in filtered)
            keep = np.asarray([i for i in range(len(batch)) if i not in filtered], dtype=np.int64)
            vectors = pack_signatures(sketches[keep])
            timings['in_batch_dedup_and_packing_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            k = min(args.top_k, int(index.ntotal))
            if len(vectors) and k:
                distances, positions = index.search(vectors, k)
            else:
                distances = np.empty((len(vectors), 0), dtype=np.int32)
                positions = np.empty((len(vectors), 0), dtype=np.int64)
            timings['index_search_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            add = np.ones(len(keep), dtype=bool)
            observations = {i: dict(ordinal=ordinal + i, source_id=row['id'],
                                   in_batch_drop=True, candidates=None, inserted=False)
                            for i, row in enumerate(batch) if i in filtered}
            drops = 0
            for j, pos in enumerate(keep):
                candidates = [dict(rank=rank + 1, ordinal=int(labels[int(label)]), distance=int(distance))
                              for rank, (label, distance) in enumerate(zip(positions[j], distances[j]))
                              if 0 <= int(label) < len(labels)]
                rejected = any(c['distance'] <= args.cutoff for c in candidates)
                add[j] = not rejected
                drops += int(rejected)
                observations[int(pos)] = dict(ordinal=ordinal + int(pos), source_id=batch[int(pos)]['id'],
                    in_batch_drop=False, candidates=candidates, inserted=not rejected)
            accepted = keep[add]
            if int(index.ntotal) + len(accepted) > args.partition_capacity:
                raise RuntimeError('Single-graph retained capacity exceeded. Start a new run with a larger --partition-capacity.')
            timings['decision_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            if len(accepted):
                index.add(pack_signatures(sketches[accepted]))
                labels.extend(ordinal + int(pos) for pos in accepted)
            timings['insert_seconds'] = time.perf_counter() - t
            seconds = time.perf_counter() - started
            report = dict(batch=len(reports) + 1, screened=len(batch), inserted=len(accepted),
                          index_search_drops=drops, in_batch_dedup_drops=len(filtered),
                          retained=int(index.ntotal), processing_seconds=seconds,
                          throughput=len(batch) / seconds, **timings)
            assert report['screened'] == report['inserted'] + drops + len(filtered)
            assert len(labels) == int(index.ntotal)
            reports.append(report)
            for i in range(len(batch)):
                evidence.write(json.dumps(observations[i]) + '\n')
            for pos in accepted:
                accepted_ids.write(json.dumps(dict(ordinal=ordinal + int(pos), source_id=batch[int(pos)]['id'])) + '\n')
            evidence.flush(); accepted_ids.flush()
            ordinal += len(batch)
            print(f"Batch {report['batch']}: screened={len(batch)} inserted={len(accepted)} QPS={report['throughput']:.2f}", flush=True)
    if not reports:
        raise ValueError('No input documents')
    with (args.output / 'batches.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(reports[0])); writer.writeheader(); writer.writerows(reports)
    faiss.write_index_binary(index, str(args.output / 'index.faiss'))
    np.save(args.output / 'index_labels.npy', np.asarray(labels, dtype=np.int64), allow_pickle=False)
    seconds = math.fsum(r['processing_seconds'] for r in reports)
    summary = dict(system='Faiss(Jaccard)', screened=ordinal, inserted=int(index.ntotal),
        index_search_drops=sum(r['index_search_drops'] for r in reports),
        in_batch_dedup_drops=sum(r['in_batch_dedup_drops'] for r in reports),
        processing_seconds=seconds, throughput=ordinal / seconds, recall='not_measured', exact_jaccard='not_computed',
        full_text=True, profile=dict(M=args.M, corpus_ef_construction=args.ef_construction,
            query_ef_construction=args.ef_construction, ef_search=args.ef_search, top_k=args.top_k,
            cutoff=args.cutoff, minhash_values=112, graph_count=1, retained_capacity=args.partition_capacity,
            threads=args.threads, batch_size=args.batch_size),
        scope='One raw-MinHash HNSW graph, no Bloom routing or bitmap metric. Input/signature/batchfilter/search/admission/insertion included; evidence/final index writes excluded. Returned top-k evidence precedes admission.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
