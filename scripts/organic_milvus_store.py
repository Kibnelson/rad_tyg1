"""Verified retained-capacity Milvus adapter, independent of private checkout history."""
from functools import lru_cache
import importlib.util
from pathlib import Path
POLICY="milvus_organic_retained_1m_top4_v1"
CAPACITY=1_000_000
PROFILE=dict(mode="approximate",mh_signature_length=112,mh_element_bit_width=32,mh_lsh_band=14,mh_lsh_rows_per_band=8,topk=4)

def need(condition, message):
    if not condition:
        raise ValueError(message)


def _prepared_tools():
    spec=importlib.util.spec_from_file_location('organic_frozen_prepared_store_tools',
        Path(__file__).with_name('prepared_milvus_store.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _group_tools():
    return _prepared_tools()._group_tools()


def record(path):
    return _prepared_tools().record(path)


def sha(path):
    return _prepared_tools().sha(path)


def _header(capacity):
    return dict(schema_version=1,policy=POLICY,capacity=capacity,profile=dict(PROFILE),
                rollover='retained_row_capacity',source_million_seals=False,
                final_partial_seal='explicit_full_run_completion_only',completion='open')


def create_store_class(native_module, capacity=CAPACITY):
    """Internal capacity parameter is for the isolated mechanics gate/tests."""
    need(type(capacity) is int and capacity > 0, 'Use a positive retained capacity')
    group=_group_tools()
    parent=group.create_store_class(native_module,capacity,policy=POLICY,topk=4)
    native_base=native_module.GrowingMilvus

    class RetainedMilvus(parent):
        def __init__(self, root, layout, cpus, timeout=1800, resume=False,
                     capacity=capacity, mode='approximate', mh_lsh_band=14):
            path=group.physical(root)/'store.json'
            if path.exists():
                saved=group.read(path)
                need(resume and isinstance(saved.get('organic_retained'),dict),
                     'Saved store has no retained-capacity policy; no migration')
                selected=dict(saved['organic_retained']);selected['completion']='open'
                need(selected==_header(capacity), 'Saved retained-row policy/profile differs')
                need(saved['organic_retained']['completion'] in ('open','closing','closed'),
                     'Invalid retained completion state')
            self._rollover_events=[]
            super().__init__(root,layout,cpus,timeout,resume,capacity,mode,mh_lsh_band)

        def save(self):
            self.state.setdefault('organic_retained',_header(self.capacity))
            return super().save()

        def _validate_groups(self):
            super()._validate_groups()
            header=self.state.get('organic_retained',{})
            expected=dict(header);expected['completion']='open'
            need(expected==_header(self.capacity) and header.get('completion') in ('open','closing','closed'),
                 'Saved retained policy changed')
            groups=self.group_state['groups']
            for number,(partition,item) in enumerate(zip(groups,self.state['instances'])):
                if number < len(groups)-1:
                    need(item['rows']==self.capacity and partition['status']=='closed',
                         'Only a full retained partition may precede another owner')
                elif item['rows'] < self.capacity and partition['status'] != 'open':
                    need(header['completion'] in ('closing','closed'),
                         'A partial retained partition was sealed before final completion')
            if header['completion']=='closed':
                need(all(g['status']=='closed' for g in groups), 'Completed retained store has an open partition')

        def _new_instance(self):
            if self._allow_group_instance:
                return super()._new_instance()
            previous=self.current_group
            need(previous is not None and self.state['instances'][previous['owner']]['rows']==self.capacity
                 and self.state['organic_retained']['completion']=='open',
                 'Retained rollover requires a full prior partition')
            if previous['status'] != 'closed':
                closed=super().finalize_group(previous['number'],
                    self.root/'retained_closures'/f'partition_{previous["owner"]:03d}.json')
                self._rollover_events.append(dict(retained_partition_closed=closed))
            next_group=super().begin_group(len(self.group_state['groups'])+1)
            return next_group['owner']

        def append(self, frame, signatures):
            ids=list(map(int,frame['int_id_column'].to_list()))
            need(len(ids)==len(set(ids)) and len(signatures)==len(ids), 'Invalid retained append IDs/signatures')
            if not ids:
                return []
            new=any(identifier not in self.owner for identifier in ids)
            need(not new or self.state['organic_retained']['completion']=='open',
                 'Cannot extend a finally sealed retained store')
            self._rollover_events=[]
            if not self.group_state['groups'] and new:
                super().begin_group(1)
            # Bypass only GroupStore's source-group capacity check. This is the
            # byte-preserved native append/journal/rollover implementation.
            profiles=native_base.append(self,frame,signatures)
            self._validate_groups()
            return profiles+self._rollover_events

        def search(self, data, topk=4, refine_k=160):
            need(type(topk) is int and topk==4, 'Organic retained Milvus requires topk=4')
            if not data:
                return [],[],0.
            if not self.ntotal:
                return [[] for _ in data],[],0.
            return super().search(data,topk,refine_k)

        def begin_group(self, number):
            raise ValueError('Retained partitions are opened only by accepted-row rollover')

        def finalize_group(self, number, receipt_path):
            raise ValueError('Use finalize_retained only at full-run completion')

        def finalize_retained(self):
            """Seal the final partial partition; never call for a resumable prefix."""
            need(self.state.get('pending') is None, 'Finish the native append journal before final sealing')
            header=self.state['organic_retained']
            path=self.root/'RETAINED_COMPLETE.json'
            if header['completion']=='closed':
                need(path.is_file(), 'Closed retained store omitted its durable completion receipt')
                value=group.read(path)
                need(value['policy']==POLICY and value['ntotal']==self.ntotal and
                     value['capacity']==self.capacity and value['profile']==PROFILE,
                     'Retained completion receipt differs')
                for partition in self.group_state['groups']:
                    self._closed_receipt(partition,group.physical(partition['receipt_path']),verify_live=True)
                return value
            header['completion']='closing';self.save()
            last=self.current_group
            if last is not None:
                need(self.state['instances'][last['owner']]['rows'] > 0,
                     'Preserve an empty owner created before an interrupted first append')
                super().finalize_group(last['number'],
                    self.root/'retained_closures'/f'partition_{last["owner"]:03d}.json')
            parts=[dict(owner=p['owner'],rows=p['actual_rows'],path=p['receipt_path'],sha256=p['receipt_sha256'])
                   for p in self.group_state['groups']]
            value=dict(kind='organic_milvus_retained_completion_v1',status='closed',policy=POLICY,
                profile=dict(PROFILE),capacity=self.capacity,ntotal=self.ntotal,partitions=parts,
                source_million_seals=False,
                timing_scope='Native partition close receipts contain seal timers. Caller records finalization in full runtime.')
            prepared=_prepared_tools()
            prepared._immutable(path,value)
            header['completion']='closed';self.save()
            self._validate_groups()
            return value

    RetainedMilvus.__name__='OrganicRetainedTop4Milvus'
    RetainedMilvus.organic_profile=dict(PROFILE)
    RetainedMilvus.organic_policy=POLICY
    return RetainedMilvus


