"""Durable Milvus append/search boundary for one collection or sequential 1M servers."""
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import polars as pl

from .milvus_features import (raw_shingles,token_set_fields,token_set_from_fields,hit_document_id,
    TOKEN_SET_FIELD,TOKEN_OVERFLOW_FIELD,VARCHAR_BYTES,TOKEN_OVERFLOW_CAPACITY)
from .milvus_layout_pilot import Servers,Searcher,wait_layout,warmup_search
from .retention import write as write_json
from .milvus_tokens import MilvusTokenSets
from .milvus_live_segments import querynode_segments
from .local_sources import file_hash

CAPACITY=1_000_000
COLLECTION='radv2_growing_reference'
BAND_CONFIGURATION_KEYS=('mh_signature_length','mh_element_bit_width','mh_lsh_band','mh_lsh_rows_per_band')


def band_configuration(mh_lsh_band=None):
    """Keep legacy identities unchanged; explicitly pin the opt-in 112=14x8 profile."""
    bands=28 if mh_lsh_band is None else mh_lsh_band
    if type(bands) is not int or bands not in (14,28) or 112%bands:
        raise ValueError('Use mh_lsh_band 14 or 28, dividing the 112-position signature')
    return (dict(mh_signature_length=112,mh_element_bit_width=32,mh_lsh_band=14,
                 mh_lsh_rows_per_band=8,topk=32) if bands==14 else {})


class GrowingMilvus:
    def __init__(self,root,layout,cpus,timeout=1800,resume=False,capacity=CAPACITY,mode='refined',mh_lsh_band=None):
        self.band_configuration=band_configuration(mh_lsh_band)
        self.mh_lsh_band=14 if self.band_configuration else 28
        if layout not in ('single','separate_sequential') or capacity<1:
            raise ValueError('Use single or separate_sequential and a positive capacity')
        self.root,self.layout,self.cpus,self.timeout=Path(root),layout,list(cpus),timeout
        self.capacity=capacity
        if mode not in ('refined','approximate'): raise ValueError('Unknown Milvus search mode')
        self.mode=mode
        self.clients,self.handles,self.processes=[],[],[]
        self.collection_ids=[]
        self.owner={}
        self.token_bytes={}
        path=self.root/'store.json'
        if path.exists():
            if not resume: raise ValueError('Milvus store exists; resume explicitly')
            self.state=json.loads(path.read_text())
            if (self.state['layout']!=layout or self.state['capacity']!=capacity
                    or self.state.get('mode')!=mode):
                raise ValueError('Saved Milvus mode, layout or capacity differs')
            saved_band={key:self.state[key] for key in BAND_CONFIGURATION_KEYS if key in self.state}
            expected_band={key:self.band_configuration[key] for key in BAND_CONFIGURATION_KEYS
                           if key in self.band_configuration}
            if (saved_band!=expected_band or any(type(value) is not int for value in saved_band.values())
                    or (self.band_configuration and (type(self.state.get('topk')) is not int
                        or self.state['topk']!=self.band_configuration['topk']))):
                raise ValueError('Saved Milvus band configuration differs or is incomplete')
            for part in self.state['id_parts']:
                saved=self.root/part['file']
                if file_hash(saved)!=part['sha256']: raise ValueError('Stored-ID journal changed')
                values=np.load(saved,allow_pickle=False)
                if values.shape!=(part['rows'],2): raise ValueError('Stored-ID journal shape differs')
                for doc,size in values:
                    if int(doc) in self.owner: raise ValueError('Repeated ID in Milvus append journal')
                    self.owner[int(doc)]=part['owner']
                    self.token_bytes[int(doc)]=int(size)
            if len(self.owner)!=self.ntotal: raise ValueError('Stored-ID journal count differs')
        else:
            if resume: raise ValueError('No Milvus store to resume')
            self.state=dict(schema_version=2,backend='milvus_growing_reference',layout=layout,mode=mode,
                capacity=capacity,ntotal=0,instances=[],id_parts=[],pending=None,recoveries=[])
            self.state.update(self.band_configuration)
            self.root.mkdir(parents=True,exist_ok=True)
            self.save()

    @property
    def ntotal(self): return self.state['ntotal']

    def save(self): write_json(self.root/'store.json',self.state)

    def __enter__(self):
        try:
            for owner in range(len(self.state['instances'])): self._connect(owner)
            if not self.clients: self._new_instance()
            self.tokens=(MilvusTokenSets(self.clients,COLLECTION,owner_lookup=self.owner,size_lookup=self.token_bytes)
                if self.mode=='refined' else None)
            if self.state['pending'] is not None:
                self._finish_pending(recover=True)
            for owner,item in enumerate(self.state['instances']):
                if item['rows']:
                    self.synchronize(owner,bytes.fromhex(item['last_signature']))
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def _connect(self,owner):
        from pymilvus import DataType
        slot=self.state['instances'][owner]['slot']
        handle=Servers(self.root/'instances'/f'owner_{owner:03d}','single',self.cpus,self.timeout,instance_slot=slot)
        handle.__enter__()
        self.handles.append(handle)
        client=handle.clients[0]
        self.clients.append(client)
        self.processes.extend(handle.processes)
        if not client.has_collection(COLLECTION):
            if self.state['instances'][owner]['rows']:
                raise ValueError('A persisted Milvus collection is missing; do not rebuild it from a partial cursor')
            schema=client.create_schema(auto_id=False,enable_dynamic_field=False)
            schema.add_field('doc_id',DataType.INT64,is_primary=True)
            schema.add_field('minhash_signature',DataType.BINARY_VECTOR,dim=112*32)
            if self.mode=='refined':
                schema.add_field(TOKEN_SET_FIELD,DataType.VARCHAR,max_length=VARCHAR_BYTES)
                schema.add_field(TOKEN_OVERFLOW_FIELD,DataType.ARRAY,element_type=DataType.VARCHAR,
                    max_length=VARCHAR_BYTES,max_capacity=TOKEN_OVERFLOW_CAPACITY)
            index=client.prepare_index_params()
            index.add_index(field_name='minhash_signature',index_name='minhash_lsh',index_type='MINHASH_LSH',
                metric_type='MHJACCARD',params=dict(mh_element_bit_width=32,mh_lsh_band=self.mh_lsh_band,
                    with_raw_data=self.mode=='refined'))
            if self.band_configuration:
                # The SDK automatically loads when index_params is supplied to
                # create_collection. Verify the actual index before that load.
                client.create_collection(COLLECTION,schema=schema,num_shards=1,consistency_level='Strong')
                client.create_index(COLLECTION,index,timeout=self.timeout)
                self._validate_index(client,client.describe_collection(COLLECTION,timeout=self.timeout))
                client.load_collection(COLLECTION,timeout=self.timeout)
            else:
                client.create_collection(COLLECTION,schema=schema,index_params=index,num_shards=1,
                    consistency_level='Strong')
        else:
            fields={row['name'] for row in client.describe_collection(COLLECTION)['fields']}
            required={'doc_id','minhash_signature'}
            if self.mode=='refined': required.update((TOKEN_SET_FIELD,TOKEN_OVERFLOW_FIELD))
            if not required<=fields or (self.mode=='approximate' and fields&{TOKEN_SET_FIELD,TOKEN_OVERFLOW_FIELD}):
                raise ValueError('Persisted Milvus collection fields differ from the selected search mode')
            if self.band_configuration:
                self._validate_index(client,client.describe_collection(COLLECTION,timeout=self.timeout))
            client.load_collection(COLLECTION,timeout=self.timeout)
        identity=client.describe_collection(COLLECTION,timeout=self.timeout)
        self.collection_ids.append(int(identity['collection_id']))
        # Check route registration before inserting even the first anchors.
        # An empty array/null is valid before this instance has consumed rows.
        self.live_segments(owner)
        print(f'Milvus live metadata: READY; owner={owner}; collection_id={self.collection_ids[owner]}',flush=True)

    def _validate_index(self,client,identity):
        """Reject a swapped/mislabeled native index before loading or using it."""
        from pymilvus import DataType
        fields=[row for row in identity.get('fields',[]) if row.get('name')=='minhash_signature']
        index=client.describe_index(COLLECTION,'minhash_lsh',timeout=self.timeout)
        if not isinstance(index,dict): raise ValueError('Milvus band profile index is missing')
        params=index.get('params',{})
        if not isinstance(params,dict): raise ValueError('Milvus band profile index parameters are malformed')
        field_params=fields[0].get('params') if len(fields)==1 else None
        def exact_integer(value,expected):
            return type(value) is int and value==expected or isinstance(value,str) and value==str(expected)
        def exact_boolean(value,expected):
            return value is expected or isinstance(value,str) and value.lower()==str(expected).lower()
        if (len(fields)!=1 or fields[0].get('type')!=DataType.BINARY_VECTOR
                or not isinstance(field_params,dict) or not exact_integer(field_params.get('dim'),112*32)
                or index.get('field_name')!='minhash_signature' or index.get('index_name')!='minhash_lsh'
                or index.get('index_type')!='MINHASH_LSH' or index.get('metric_type')!='MHJACCARD'
                or not exact_integer(params.get('mh_element_bit_width',index.get('mh_element_bit_width')),32)
                or not exact_integer(params.get('mh_lsh_band',index.get('mh_lsh_band')),self.mh_lsh_band)
                or not exact_boolean(params.get('with_raw_data',index.get('with_raw_data')),self.mode=='refined')
                or index.get('state')=='Failed'):
            raise ValueError('Milvus native index differs from the selected 112-position band profile')

    def _new_instance(self):
        owner=len(self.state['instances'])
        if self.layout=='single' and owner: raise ValueError('Single layout must keep one collection/client')
        slot=1 if self.layout=='single' else 10+owner
        if slot>100: raise ValueError('Milvus instance slot budget exceeded')
        self.state['instances'].append(dict(slot=slot,rows=0))
        self.save()
        self._connect(owner)
        return owner

    def _finish_pending(self,recover=False):
        pending=self.state['pending']
        path=self.root/pending['file']
        if file_hash(path)!=pending['sha256']: raise ValueError('Pending Milvus insertion journal changed')
        rows=pl.read_parquet(path).to_dicts()
        owner=pending['owner']
        requested={int(row['doc_id']):row for row in rows}
        client=self.clients[owner]
        if recover:
            found={}
            fields=['doc_id','minhash_signature']
            if self.mode=='refined': fields.extend((TOKEN_SET_FIELD,TOKEN_OVERFLOW_FIELD))
            for doc_id in requested:
                present=client.get(COLLECTION,ids=[doc_id],
                    output_fields=fields,
                    consistency_level='Strong',timeout=self.timeout)
                for row in present:
                    doc=hit_document_id(row)
                    if doc not in requested or doc in found: raise ValueError('Invalid recovery ID response')
                    if (bytes(row['minhash_signature'])!=requested[doc]['minhash_signature'] or
                            (self.mode=='refined' and token_set_from_fields(row)!=token_set_from_fields(requested[doc]))):
                        raise ValueError('Persisted features differ from the pending insertion journal')
                    found[doc]=row
            rows=[row for row in rows if int(row['doc_id']) not in found]
            self.state['recoveries'].append(dict(owner=owner,already_present=len(found),missing=len(rows),
                note='Interrupted append recovered by IDs; the interrupted wall interval is not reconstructed.'))
        before=time.perf_counter()
        if rows: client.insert(COLLECTION,rows,timeout=self.timeout)
        insert_seconds=time.perf_counter()-before
        final_rows=pending['before_rows']+pending['count']
        ids=np.array(list(requested),dtype=np.int64)
        sizes=np.array([sum(len(piece.encode('utf-8')) for piece in
            [row.get(TOKEN_SET_FIELD,''),*row.get(TOKEN_OVERFLOW_FIELD,[])]) for row in requested.values()],dtype=np.int64)
        relative=Path('ids')/(path.stem+'.npy')
        destination=self.root/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('wb') as stream:
            np.save(stream,np.column_stack((ids,sizes)),allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor=os.open(destination.parent,os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        self.state['id_parts'].append(dict(file=str(relative),sha256=file_hash(destination),
            owner=owner,rows=len(ids)))
        self.state['instances'][owner]['rows']=final_rows
        self.state['instances'][owner]['last_signature']=rows[-1]['minhash_signature'].hex() if rows else next(
            reversed(requested.values()))['minhash_signature'].hex()
        self.state['ntotal']+=len(ids)
        self.state['pending']=None
        self.save()
        for doc,size in zip(ids,sizes):
            self.owner[int(doc)]=owner
            self.token_bytes[int(doc)]=int(size)
        path.unlink()
        return dict(owner=owner,inserted=len(ids),insert_rpc_seconds=insert_seconds)

    def live_segments(self,owner):
        port=self.handles[owner].instances[0]['ports']['metrics']
        return querynode_segments(f'http://127.0.0.1:{port}',self.collection_ids[owner])

    def synchronize(self,owner,signature):
        began=time.perf_counter()
        client=self.clients[owner]
        name=f'{owner:03d}_{self.state["instances"][owner]["rows"]:09d}.json'
        layout_timing={}
        layout=wait_layout(client,COLLECTION,self.state['instances'][owner]['rows'],self.timeout,
            live_observer=lambda:self.live_segments(owner),timing=layout_timing,
            report=self.root/'layout_waits'/name)
        layout_seconds=time.perf_counter()-began
        readiness_started=time.perf_counter()
        searcher=Searcher([client],COLLECTION,False,32,160,self.mode=='refined')
        report=self.root/'visibility'/name
        try:
            warmup_search(searcher,[signature],self.timeout,self.handles[owner].processes,report)
        finally:
            searcher.close()
            if report.exists():
                evidence=json.loads(report.read_text())
                evidence.update(excluded_from_throughput=False,
                    timing_scope='Included in growing full elapsed and append/batch processing')
                write_json(report,evidence)
        readiness_seconds=time.perf_counter()-readiness_started
        return dict(owner=owner,visibility_index_seconds=time.perf_counter()-began,layout=layout,
            layout_wait_seconds=layout_seconds,readiness_seconds=readiness_seconds,
            layout_poll_count=len(layout_timing['polls']),
            observation_source='live_querynode_and_coordinators',
            layout_wait_report=str(Path('layout_waits')/name),readiness_report=str(Path('visibility')/name))

    def append(self,frame,sketches):
        ids=list(map(int,frame['int_id_column'].to_list()))
        if len(set(ids))!=len(ids) or len(sketches)!=len(ids): raise ValueError('Invalid insertion IDs/sketch count')
        profiles=[]
        offset=0
        while offset<len(ids):
            if ids[offset] in self.owner:
                offset+=1  # Idempotent replay after a durable insertion and before the caller cursor.
                continue
            owner=0 if self.layout=='single' else len(self.clients)-1
            if self.layout!='single' and self.state['instances'][owner]['rows']==self.capacity:
                owner=self._new_instance()
            current=self.state['instances'][owner]['rows']
            count=min(len(ids)-offset,self.capacity-current%self.capacity,1000)
            group=[]
            wire_bytes=0
            for i in range(offset,offset+count):
                if ids[i] in self.owner: break
                fields=token_set_fields(raw_shingles(frame['contents'][i])) if self.mode=='refined' else {}
                signature=np.asarray(sketches[i],dtype='<u4').tobytes()
                size=512+len(signature)+sum(len(piece.encode('utf-8')) for piece in
                    [fields.get(TOKEN_SET_FIELD,''),*fields.get(TOKEN_OVERFLOW_FIELD,[])])
                if size>32*1024*1024:
                    raise ValueError('One complete token set exceeds the 32 MiB single-row RPC envelope; no truncation')
                if group and wire_bytes+size>8*1024*1024: break
                group.append(dict(doc_id=ids[i],minhash_signature=signature,**fields))
                wire_bytes+=size
            count=len(group)
            relative=Path('pending')/(uuid.uuid4().hex+'.parquet')
            path=self.root/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            pl.DataFrame(group).write_parquet(path)
            with path.open('rb') as stream: os.fsync(stream.fileno())
            descriptor=os.open(path.parent,os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
            self.state['pending']=dict(file=str(relative),sha256=file_hash(path),owner=owner,
                before_rows=current,count=count)
            self.save()
            profiles.append(self._finish_pending())
            offset+=count
            # Wait at full physical boundaries and at the end of this logical
            # append, rather than forcing a search after every network packet.
            if (current+count)%self.capacity==0 or offset==len(ids):
                profiles.append(self.synchronize(owner,group[-1]['minhash_signature']))
        return profiles

    def search(self,data,topk=32,refine_k=160):
        if self.band_configuration and topk!=self.band_configuration['topk']:
            raise ValueError('Milvus band profile requires topk=32')
        if not data: return [],[],0.
        if not self.ntotal: return [[] for _ in data],[],0.
        searcher=Searcher(self.clients,COLLECTION,False,topk,refine_k,self.mode=='refined')
        try: return searcher.search(data)
        finally: searcher.close()

    def layouts(self):
        return [wait_layout(client,COLLECTION,item['rows'],self.timeout,
            live_observer=lambda owner=owner:self.live_segments(owner))
            for owner,(client,item) in enumerate(zip(self.clients,self.state['instances']))]

    def __exit__(self,*exc):
        if getattr(self,'tokens',None) is not None: self.tokens.close()
        for handle in reversed(self.handles): handle.__exit__(*exc)
