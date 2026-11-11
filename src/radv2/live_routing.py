"""Mutable per-partition Bloom routing with generation-bound checkpoint sidecars."""
import hashlib
import inspect
import os
from pathlib import Path
import time

import numpy as np

from .routing import Bloom, LAYOUTS, SEED, ROOT, band_keys, digest, native_library, search_selected


class LiveBloomRouter:
    def __init__(self, store, mode='bloom', layout='37x3', fpr=1e-4, zero_policy='admit', threads=1, saved=None):
        if mode not in ('bloom','shadow') or layout not in LAYOUTS or zero_policy not in ('admit','search_all'):
            raise ValueError('Invalid live routing configuration')
        if not 0<fpr<1 or threads<1 or store.spec.K<np.prod(LAYOUTS[layout]):
            raise ValueError('Invalid Bloom FPR, threads or signature shape')
        self.store,self.mode,self.layout,self.fpr,self.zero_policy,self.threads=store,mode,layout,float(fpr),zero_policy,int(threads)
        self.config=dict(mode=mode,layout=layout,fpr=float(fpr),zero_policy=zero_policy)
        self.identity=dict(schema_version=1,layout=layout,dimensions=list(LAYOUTS[layout]),seed=hex(SEED),capacity=store.capacity,
            spec_fingerprint=store.spec.fingerprint(),fpr=float(fpr),
            native_source_sha256=digest(ROOT/'native/routing.cpp'),
            bloom_function_sha256=hashlib.sha256(inspect.getsource(Bloom).encode()).hexdigest(),
            band_function_sha256=hashlib.sha256(inspect.getsource(band_keys).encode()).hexdigest())
        self.filters=[None]*len(store.partitions)
        self.covered=[0]*len(store.partitions)
        self.files=[None]*len(store.partitions)
        self.reasons={i:'summary unavailable' for i in range(len(store.partitions))}
        self.dirty=set()
        self.last_mask=self.last_actual_mask=None
        native_library()  # Outside query timing during create/reload.
        if saved is not None:
            self._restore(saved)

    def _restore(self,saved):
        if saved.get('config')!=self.config:
            raise ValueError('Checkpoint routing configuration differs')
        if saved.get('identity')!=self.identity:
            self.reasons={i:'routing hash/spec identity mismatch' for i in range(len(self.filters))}
            return
        entries=saved.get('partitions',[])
        for graph,part in enumerate(self.store.partitions):
            try:
                entry=entries[graph]
                if (not entry.get('valid') or entry['graph']!=graph or entry['count']!=int(part.index.ntotal) or
                        entry['index']!=part.files['index'] or entry['labels']!=part.files['labels']):
                    raise ValueError('summary coverage or graph binding differs')
                path=(self.store.directory/entry['file']).resolve()
                if not path.is_relative_to(self.store.directory.resolve()) or digest(path)!=entry['sha256']:
                    raise ValueError('summary file missing, corrupt or outside collection')
                bits=np.load(path,mmap_mode='r',allow_pickle=False)
                # Only the active partition needs a mutable array after restart.
                if int(part.index.ntotal)<self.store.capacity:
                    bits=np.array(bits,copy=True)
                bloom=Bloom(self.store.capacity*LAYOUTS[self.layout][0],self.fpr,bits)
                self.filters[graph],self.covered[graph],self.files[graph]=bloom,int(part.index.ntotal),entry
                self.reasons.pop(graph,None)
            except (OSError,ValueError,KeyError,IndexError,TypeError) as error:
                self.reasons[graph]=str(error)

    def prepare(self,sketches,count):
        values=np.asarray(sketches)
        if values.dtype!=np.uint32 or values.shape!=(count,self.store.spec.K):
            raise ValueError('Bloom-enabled insertion requires aligned uint32 MinHash signatures')
        return band_keys(values,self.layout,self.threads)

    def append(self,graph,before,keys):
        while len(self.filters)<=graph:
            self.filters.append(None)
            self.covered.append(0)
            self.files.append(None)
        if before==0 and self.filters[graph] is None:
            self.filters[graph]=Bloom(self.store.capacity*LAYOUTS[self.layout][0],self.fpr)
            self.covered[graph]=0
        if self.filters[graph] is None or self.covered[graph]!=before:
            self.filters[graph]=None
            self.reasons[graph]='incomplete earlier signature coverage; searching graph normally'
            self.dirty.add(graph)
            return
        try:
            self.filters[graph].add(keys,self.threads)
        except Exception:
            self.filters[graph]=None
            self.reasons[graph]='summary update failed'
            self.dirty.add(graph)
            raise
        self.covered[graph]=before+len(keys)
        self.reasons.pop(graph,None)
        self.dirty.add(graph)

    def search(self,vectors,k,ef_search,sketches,profile=None):
        started,cpu_started=time.perf_counter(),time.process_time()
        keys=self.prepare(sketches,len(vectors))
        mask=np.ones((len(vectors),len(self.store.partitions)),dtype=bool)
        fallback=[]
        for graph,part in enumerate(self.store.partitions):
            if graph>=len(self.filters) or self.filters[graph] is None or self.covered[graph]!=int(part.index.ntotal):
                fallback.append(graph)
            else:
                mask[:,graph]=self.filters[graph].contains_any(keys,self.threads)
        routing_ms=(time.perf_counter()-started)*1000
        fanout=mask.sum(axis=1)
        actual=mask.copy()
        if self.mode=='shadow':
            actual[:]=True
        elif self.zero_policy=='search_all':
            actual[fanout==0]=True
        self.last_mask,self.last_actual_mask=mask,actual
        distances,found,measured=search_selected(self.store,vectors,k,ef_search,self.threads,actual)
        if profile is not None:
            graphs=[]
            for graph in measured['graphs']:
                graphs.append(dict(graph=graph['graph'],vectors=graph['vectors'],queries=graph['queries'],
                    native_ms=graph['native_seconds']*1000,mapping_merge_ms=graph['mapping_merge_seconds']*1000,
                    packing_ms=graph['packing_seconds']*1000,native_cpu_seconds=graph['native_cpu_seconds'],
                    native_average_cpu_cores=graph['native_average_cpu_cores'],
                    configured_faiss_threads=graph['configured_faiss_threads'],
                    total_ms=(graph['native_seconds']+graph['mapping_merge_seconds']+graph['packing_seconds'])*1000))
            profile.update(graphs=graphs,queries=len(vectors),k=k,ef_search=ef_search,routing_ms=routing_ms,
                packing_ms=measured['packing_seconds']*1000,
                total_ms=(time.perf_counter()-started)*1000,
                cpu_seconds=time.process_time()-cpu_started,
                routing=dict(mode=self.mode,layout=self.layout,zero_policy=self.zero_policy,
                    queries=len(vectors),available_graphs=len(self.store.partitions),
                    candidate_query_graph_pairs=int(mask.sum()),actual_query_graph_pairs=int(actual.sum()),
                    possible_query_graph_pairs=int(mask.size),zero_candidate_queries=int((fanout==0).sum()),
                    mean_candidate_graphs=float(fanout.mean()) if len(fanout) else 0.,
                    p95_candidate_graphs=float(np.percentile(fanout,95)) if len(fanout) else 0.,
                    selected_graphs=int(actual.any(axis=0).sum()),fallback_graphs=fallback,
                    fallback_reasons={str(g):self.reasons.get(g,'coverage mismatch') for g in fallback},
                    zero_candidate_fallback_queries=int((fanout==0).sum())
                        if self.mode=='bloom' and self.zero_policy=='search_all' else 0))
        return distances,found

    def checkpoint(self,generation,partitions):
        entries=[]
        for graph,part in enumerate(partitions):
            entry=dict(graph=graph,count=part['count'],index=part['index'],labels=part['labels'],valid=False)
            if graph<len(self.filters) and self.filters[graph] is not None and self.covered[graph]==part['count']:
                if graph in self.dirty or self.files[graph] is None:
                    filename=f'partitions/bloom_{self.layout}_{graph:06d}.g{generation:08d}.npy'
                    path=self.store.directory/filename
                    np.save(path,self.filters[graph].bits,allow_pickle=False)
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
                    checksum=digest(path)
                else:
                    filename,checksum=self.files[graph]['file'],self.files[graph]['sha256']
                entry.update(valid=True,file=filename,sha256=checksum)
            else:
                entry['reason']=self.reasons.get(graph,'incomplete summary')
            entries.append(entry)
        return dict(config=self.config,identity=self.identity,partitions=entries)

    def committed(self,metadata):
        current={e['file'] for e in metadata['partitions'] if e['valid']}
        obsolete=[]
        for previous in self.files:
            if previous and previous.get('file') not in current:
                filename=previous.get('file','')
                if filename.startswith('partitions/bloom_'):
                    obsolete.append(filename)
        self.files=[e if e['valid'] else None for e in metadata['partitions']]
        self.dirty.clear()
        return obsolete
