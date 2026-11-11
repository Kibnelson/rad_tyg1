#!/usr/bin/env python3
"""Screen every input row with the included native Milvus MinHash-LSH engine."""
import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'external/simd')]


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=28)
    p.add_argument('--batch-size',type=int,default=10_000)
    p.add_argument('--partition-capacity',type=int,default=1_000_000)
    p.add_argument('--max-documents',type=int)
    p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args()
    if min(a.threads,a.batch_size,a.partition_capacity,a.timeout)<=0:
        p.error('Threads, batch size, partition capacity and timeout must be positive')
    if a.max_documents is not None and a.max_documents<=0:
        p.error('--max-documents must be positive')
    if a.threads>len(os.sched_getaffinity(0)):
        p.error('--threads exceeds available CPU affinity')
    return a


def filtered_positions(module, signatures, threads):
    import numpy as np
    name='simd_fuzzy_deduplicationv7_unsorted_multi7A_V1'
    removed=set()
    def walk(v):
        if isinstance(v,(list,tuple,np.ndarray)):
            for x in v: walk(x)
        elif v is not None:
            i=int(v)
            if not 0<=i<len(signatures): raise ValueError('Invalid SIMD removal index')
            removed.add(i)
    walk(getattr(module,name)(np.ascontiguousarray(signatures,dtype=np.uint64),
        threshold=.7,num_bands=14,num_threads=threads))
    return [i for i in range(len(signatures)) if i not in removed]


def main():
    a=arguments()
    from common.stream import iter_batches
    import numpy as np
    import polars as pl
    from radv2.encoding import PersistentMinHasher, SketchSpec
    import radv2.milvus_growing_store as native
    from organic_milvus_store import create_store_class
    simd=importlib.import_module('simd_fuzzy_deduplicationv7_unsorted_multi7A_V1')
    output=a.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'raw').mkdir()
    cpus=sorted(os.sched_getaffinity(0))[:a.threads]
    spec=SketchSpec()
    config=dict(input=str(a.input.resolve()),threads=a.threads,cpus=cpus,
        batch_size=a.batch_size,partition_capacity=a.partition_capacity,topk=4,
        signature_length=112,bands=14,rows_per_band=8,admission='any native LSH candidate',
        exact_jaccard=False,recall_measured=False,text_mode='full_text',
        sketch_fingerprint=spec.fingerprint())
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    columns=['batch','screened','inserted','index_search_drops','in_batch_dedup_drops',
        'minhash_seconds','in_batch_seconds','search_seconds','decision_seconds',
        'insert_seconds','processing_seconds','throughput','stored']
    rows=[];ordinal=0;wall_start=time.perf_counter()
    store_class=create_store_class(native,capacity=a.partition_capacity)
    with (output/'batches.csv').open('w',newline='') as stats:
        writer=csv.DictWriter(stats,fieldnames=columns);writer.writeheader()
        with PersistentMinHasher(spec,a.threads,128) as hasher:
            with store_class(output/'collection','separate_sequential',cpus,timeout=a.timeout) as store:
                for number,batch in enumerate(iter_batches(a.input,a.batch_size,a.max_documents),1):
                    started=time.perf_counter()
                    texts=[r['text'] for r in batch]
                    marks=[started]
                    signatures=hasher.compute(texts);marks.append(time.perf_counter())
                    survivors=filtered_positions(simd,signatures,a.threads);marks.append(time.perf_counter())
                    data=[np.asarray(signatures[i],dtype='<u4').tobytes() for i in survivors]
                    candidates,_,_=store.search(data,topk=4);marks.append(time.perf_counter())
                    assert len(candidates)==len(survivors)
                    accepted=[i for i,found in zip(survivors,candidates) if not found]
                    accepted_set=set(accepted)
                    decisions={i:found for i,found in zip(survivors,candidates)}
                    marks.append(time.perf_counter())
                    if accepted:
                        frame=pl.DataFrame({'int_id_column':[ordinal+i for i in accepted],
                            'contents':[texts[i] for i in accepted]})
                        store.append(frame,signatures[accepted])
                    marks.append(time.perf_counter())
                    processing=marks[-1]-started
                    row=dict(batch=number,screened=len(batch),inserted=len(accepted),
                        index_search_drops=len(survivors)-len(accepted),
                        in_batch_dedup_drops=len(batch)-len(survivors),
                        minhash_seconds=marks[1]-marks[0],in_batch_seconds=marks[2]-marks[1],
                        search_seconds=marks[3]-marks[2],decision_seconds=marks[4]-marks[3],
                        insert_seconds=marks[5]-marks[4],processing_seconds=processing,
                        throughput=len(batch)/processing,stored=store.ntotal)
                    # Ordered top4 evidence and report writes are outside processing_seconds.
                    with (output/'raw'/f'batch_{number:06d}.jsonl').open('w') as raw:
                        for i,source in enumerate(batch):
                            raw.write(json.dumps(dict(id=source['id'],stream_id=ordinal+i,
                                inserted=i in accepted_set,in_batch_dropped=i not in decisions,
                                candidates=[dict(stream_id=int(doc),score=float(score))
                                    for doc,score in decisions.get(i,[])]))+'\n')
                    writer.writerow(row);stats.flush();rows.append(row);ordinal+=len(batch)
                    print(f"Batch {number}: screened={ordinal:,}, retained={store.ntotal:,}, QPS={row['throughput']:.2f}",flush=True)
                seal_started=time.perf_counter()
                final=store.finalize_retained()
                seal_seconds=time.perf_counter()-seal_started
    seconds=math.fsum(r['processing_seconds'] for r in rows)
    summary=dict(status='completed',screened=ordinal,inserted=sum(r['inserted'] for r in rows),
        index_search_drops=sum(r['index_search_drops'] for r in rows),
        in_batch_dedup_drops=sum(r['in_batch_dedup_drops'] for r in rows),
        batches=len(rows),processing_seconds=seconds,throughput=ordinal/seconds if seconds else None,
        final_seal_seconds=seal_seconds,full_wall_seconds=time.perf_counter()-wall_start,
        retained_completion=final,
        timing_scope='Processing includes MinHash, in-batch dedup, native search, admission and accepted append/readiness, including owner creation during append. JSONL input iteration, evidence writes, store-context setup, final partial sealing and shutdown are outside processing. Full wall includes these local run costs. Downloads/build are outside both.')
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='retained_completion'},indent=2))


if __name__=='__main__':
    main()
