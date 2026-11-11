"""Fetch full candidate token sets from Milvus; no SQLite reads or cross-batch cache."""
from concurrent.futures import ThreadPoolExecutor, wait
import time

import numpy as np

from .milvus_features import (hit_document_id, token_set_from_fields,
    TOKEN_SET_FIELD, TOKEN_OVERFLOW_FIELD)


class MilvusTokenSets:
    def __init__(self,clients,collection,stored_ids=None,parallel=False,capacity=1_000_000,batch_size=64,
                 owner_lookup=None,size_lookup=None):
        if not clients or capacity<1 or batch_size<1:
            raise ValueError('Use nonempty clients and positive token retrieval limits')
        self.clients,self.collection=clients,collection
        self.owner_lookup=owner_lookup
        self.size_lookup=size_lookup
        if owner_lookup is None:
            ids=np.asarray(stored_ids,dtype=np.int64)
            if ids.ndim!=1 or not len(ids):
                raise ValueError('Use a nonempty stored-ID vector for candidate routing')
            self.order=np.argsort(ids)
            self.ids=ids[self.order]
            if np.any(self.ids[1:]==self.ids[:-1]):
                raise ValueError('Stored candidate IDs must be unique')
            if len(clients)>1 and len(ids)!=len(clients)*capacity:
                raise ValueError('Independent instances must each hold one full physical capacity')
        self.capacity,self.batch_size=capacity,batch_size
        self.pool=ThreadPoolExecutor(max_workers=len(clients)) if parallel else None
        self.last_stats={}

    def _one(self,owner,ids):
        result={}
        rpcs=[]
        payload_bytes=0
        begin=0
        while begin<len(ids):
            count=0
            size=0
            for doc in ids[begin:begin+self.batch_size]:
                size+=self.size_lookup[doc]+1024 if self.size_lookup is not None else 0
                if count and size>8*1024*1024: break
                count+=1
            requested=ids[begin:begin+count]
            begin+=count
            before=time.perf_counter()
            rows=self.clients[owner].get(self.collection,ids=requested,
                output_fields=['doc_id',TOKEN_SET_FIELD,TOKEN_OVERFLOW_FIELD],
                consistency_level='Strong',timeout=600)
            rpcs.append(time.perf_counter()-before)
            expected=set(requested)
            seen=set()
            for row in rows:
                doc=hit_document_id(row)
                if doc not in expected or doc in seen:
                    raise ValueError('Milvus token fetch returned an unexpected or duplicate ID')
                seen.add(doc)
                result[doc]=token_set_from_fields(row)
                payload_bytes+=sum(len(piece.encode('utf-8')) for piece in
                    [row[TOKEN_SET_FIELD],*row[TOKEN_OVERFLOW_FIELD]])
            if seen!=expected:
                raise ValueError('Milvus token fetch omitted a requested candidate; no local fallback is allowed')
        return result,dict(owner=owner,rows=len(ids),rpc_seconds=rpcs,payload_utf8_bytes=payload_bytes)

    def load(self,requested_ids):
        before=time.perf_counter()
        ids=sorted(set(map(int,requested_ids)))
        groups={}
        if ids:
            if self.owner_lookup is not None:
                try: owners=[self.owner_lookup[doc] for doc in ids]
                except KeyError as error:
                    raise ValueError('Search returned an ID outside the stored population') from error
            else:
                positions=np.searchsorted(self.ids,np.asarray(ids,dtype=np.int64))
                if np.any(positions==len(self.ids)) or not np.array_equal(self.ids[positions],ids):
                    raise ValueError('Search returned an ID outside the frozen stored population')
                owners=(self.order[positions]//self.capacity if len(self.clients)>1 else np.zeros(len(ids),dtype=int))
            for doc,owner in zip(ids,owners): groups.setdefault(int(owner),[]).append(doc)
        if self.pool:
            futures=[self.pool.submit(self._one,owner,group) for owner,group in sorted(groups.items())]
            wait(futures)
            replies=[future.result() for future in futures]
        else:
            replies=[self._one(owner,group) for owner,group in sorted(groups.items())]
        tokens={doc:value for result,_ in replies for doc,value in result.items()}
        self.last_stats=dict(seconds=time.perf_counter()-before,unique_candidates=len(ids),
            instances=[stats for _,stats in replies],
            payload_utf8_bytes=sum(stats['payload_utf8_bytes'] for _,stats in replies),
            source='milvus_varchar_fields',cross_batch_cache=False)
        return tokens

    def close(self):
        if self.pool: self.pool.shutdown(wait=True,cancel_futures=True)
