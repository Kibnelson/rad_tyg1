"""Server/search portions of the verified Milvus layout adapter."""
from concurrent.futures import ThreadPoolExecutor, wait
import importlib.util
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import numpy as np
from .milvus_capacity import inspect_segments, validate_layout
from .milvus_features import hit_document_id
from .retention import write as write_json
ROOT=Path(__file__).resolve().parents[2]
CAPACITY=1_000_000

def cpu_teams(cpus, parallel):
    cpus=list(cpus)
    if len(cpus)<3 or len(set(cpus))!=len(cpus):
        raise ValueError('Use at least three distinct CPUs')
    if not parallel:
        return [cpus.copy() for _ in range(3)]
    # 28 CPUs -> 9/9/10 disjoint teams; all server/client affinity stays in the common budget.
    return [cpus[i*len(cpus)//3:(i+1)*len(cpus)//3] for i in range(3)]


def merge_hits(responses, nq, topk):
    if any(len(response)!=nq for response in responses):
        raise ValueError('An instance returned an incomplete query response')
    merged=[]
    for q in range(nq):
        candidates={}
        for response in responses:
            for hit in response[q]:
                doc=hit_document_id(hit)
                if doc>=0:
                    score=float(hit['distance'])
                    candidates[doc]=max(score,candidates.get(doc,float('-inf')))
        merged.append(sorted(candidates.items(),key=lambda row:(-row[1],row[0]))[:topk])
    return merged


class Servers:
    def __init__(self, directory, case, cpus, timeout, instance_slot=None):
        self.directory,self.case,self.cpus,self.timeout=directory,case,cpus,timeout
        self.instance_slot=instance_slot
        self.processes=[]
        self.clients=[]
        self.logs=[]
        spec=importlib.util.spec_from_file_location('radv2_owned_source_check',ROOT/'scripts/check_milvus_sources.py')
        self.source=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.source)
        self.source.check()
    def __enter__(self):
        from pymilvus import MilvusClient
        single=self.case=='single'
        slots=[self.instance_slot if self.instance_slot is not None else 1] if single else [10,11,12]
        teams=[self.cpus] if single else cpu_teams(self.cpus,self.case=='separate_parallel')
        self.started=time.perf_counter()
        self.instances=[]
        try:
            for i,(slot,team) in enumerate(zip(slots,teams)):
                folder=self.directory/('single' if single else f'separate_{i}')
                config=self.source.config(folder/'config',slot)
                ports=config['ports']
                for port in ports.values():
                    with socket.socket() as probe:
                        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                        probe.bind(('0.0.0.0',port))
                log=(folder/f'{self.case}.log').open('a')
                self.logs.append(log)
                process=subprocess.Popen(['bash',str(ROOT/'scripts/run_milvus_local.sh'),
                    str(folder/'config'),str(len(team)),str(slot),','.join(map(str,team))],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                self.processes.append(process)
                self.instances.append(dict(slot=slot,cpus=team,threads=len(team),pid=process.pid,
                    uri=f'http://127.0.0.1:{ports["proxy"]}',ports=ports,
                    log=str(folder/f'{self.case}.log')))
            for process,info in zip(self.processes,self.instances):
                deadline=time.monotonic()+self.timeout
                last_error=None
                while time.monotonic()<deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f'Milvus exited during startup: {info["log"]}')
                    client=None
                    try:
                        client=MilvusClient(uri=info['uri'],timeout=10)
                        client.list_collections(timeout=10)
                        self.clients.append(client)
                        break
                    except Exception as error:
                        last_error=error
                        if client is not None: client.close()
                        time.sleep(2)
                else: raise RuntimeError(f'Milvus startup timeout: {info["log"]}; {last_error}')
            self.startup_seconds=time.perf_counter()-self.started
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise
    def __exit__(self,*exc):
        for client in self.clients:
            try: client.close()
            except Exception: pass
        for process in self.processes:
            if process.poll() is None:
                try: os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        deadline=time.monotonic()+120
        for process in self.processes:
            try: process.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
        for log in self.logs: log.close()


def wait_layout(client,collection,expected,timeout,*,live_observer=None,timing=None,report=None):
    started=time.monotonic()
    deadline=started+timeout
    observed=[]
    polls=[]
    def save(status,error=None):
        value=dict(status=status,expected_rows=expected,seconds=time.monotonic()-started,
            observation_source='live_querynode_and_coordinators' if live_observer else 'coordinators',
            polls=polls,excluded_from_throughput=False)
        if error is not None: value['error']=str(error)
        if timing is not None:
            timing.clear()
            timing.update(value)
        if report is not None: write_json(report,value)
    while time.monotonic()<deadline:
        before=time.monotonic()
        try:
            if live_observer is None:
                observed=inspect_segments(client._get_connection(),collection)
            else:
                observed=inspect_segments(client._get_connection(),collection,live_segments=live_observer())
        except Exception as error:
            save('failed',error)
            raise
        try:
            result=validate_layout(observed,expected,require_live=live_observer is not None)
        except ValueError as error:
            polls.append(dict(elapsed_seconds=time.monotonic()-started,
                metadata_seconds=time.monotonic()-before,error=str(error),segments=observed))
            bad=[s for s in observed if s['rows']>CAPACITY or s.get('compaction_from') or
                 (s['state']=='Flushed' and s['indexed'] and 0<s['rows']<CAPACITY)]
            if bad:
                save('failed',error)
                raise RuntimeError(f'Invalid physical layout: {error}; {observed}') from error
            if len(polls)==1 or len(polls)%30==0: save('waiting',error)
            time.sleep(1)
        else:
            polls.append(dict(elapsed_seconds=time.monotonic()-started,
                metadata_seconds=time.monotonic()-before,status='ready',segments=observed))
            save('ready')
            return result
    error=RuntimeError(f'Physical layout timeout: expected {expected}; {observed}')
    save('failed',error)
    raise error


class Searcher:
    def __init__(self,clients,collection,parallel,topk=4,refine_k=160,refined=True):
        if type(topk) is not int or topk != 4:
            raise ValueError('Public Milvus search requires topk=4')
        self.clients,self.collection,self.parallel=clients,collection,parallel
        self.topk,self.refine_k=topk,refine_k
        self.refined=refined
        self.pool=ThreadPoolExecutor(max_workers=len(clients)) if parallel else None
    def one(self,client,data,timeout=600):
        start=time.perf_counter()
        params=dict(mh_search_with_jaccard=self.refined)
        if self.refined: params['refine_k']=self.refine_k
        response=client.search(self.collection,data=data,anns_field='minhash_signature',limit=self.topk,
            search_params=dict(metric_type='MHJACCARD',params=params),
            consistency_level='Strong',timeout=timeout)
        return response,time.perf_counter()-start
    def search(self,data,timeout=600):
        if self.pool:
            futures=[self.pool.submit(self.one,client,data,timeout) for client in self.clients]
            # Finish every sibling RPC before returning an error. A warmup retry
            # must not overlap requests from the failed previous attempt.
            wait(futures)
            for future in futures:
                error=future.exception()
                if error is not None and not channel_not_serviceable(error):
                    raise error
            responses=[future.result() for future in futures]
        else:
            responses=[self.one(client,data,timeout) for client in self.clients]
        merge_start=time.perf_counter()
        merged=merge_hits([row[0] for row in responses],len(data),self.topk)
        merge_seconds=time.perf_counter()-merge_start
        return merged,[row[1] for row in responses],merge_seconds
    def close(self,wait=True):
        if self.pool: self.pool.shutdown(wait=wait,cancel_futures=True)


def channel_not_serviceable(error):
    """Only the observed query-view startup condition is eligible for retry."""
    from pymilvus.exceptions import MilvusException
    message=str(error)
    return (isinstance(error,MilvusException) and error.code==503
        and 'channel distribution is not serviceable' in message
        and 'channel not available' in message)


def warmup_search(searcher,data,timeout,processes,report):
    """Wait for a successful complete search before entering measured batches.

    Loaded physical index metadata does not establish a serviceable query view.
    Retain readiness attempts in a separate evidence file.
    """
    if not data or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('Use nonempty warmup queries and a finite positive readiness timeout')
    started=time.monotonic()
    deadline=started+timeout
    attempts=[]
    last_error=None

    def record(status,error=None):
        value=dict(status=status,queries=len(data),seconds=time.monotonic()-started,
            excluded_from_throughput=True,attempts=attempts,
            note='Strong search at the measured top-k/refine-k; no partial-result policy or index changes.')
        if error is not None:
            value['failure']=dict(error_type=type(error).__name__,error=str(error))
        write_json(report,value)

    try:
        while True:
            for process in processes:
                code=process.poll()
                if code is not None:
                    raise RuntimeError(f'Milvus exited before search readiness: PID {process.pid}; exit={code}')
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise RuntimeError(f'Milvus query channel did not become serviceable within {timeout}s; '
                    f'inspect {report} and instances/*/*.log') from last_error
            before=time.monotonic()
            try:
                result=searcher.search(data,timeout=min(30.,remaining))
            except Exception as error:
                attempts.append(dict(status='failed',seconds=time.monotonic()-before,
                    error_type=type(error).__name__,error=str(error)))
                if not channel_not_serviceable(error):
                    raise
                last_error=error
                record('waiting')
                print(f'Milvus search readiness: channel pending; attempt={len(attempts)}; '
                    f'elapsed={time.monotonic()-started:.1f}s; report={report}',flush=True)
                time.sleep(min(1.,max(0.,deadline-time.monotonic())))
            else:
                if len(result[0])!=len(data):
                    raise ValueError('Incomplete warmup response; measured batches must not start')
                if time.monotonic()>deadline:
                    raise RuntimeError('Milvus warmup search exceeded the readiness deadline')
                attempts.append(dict(status='ready',seconds=time.monotonic()-before,
                    rpc_seconds=result[1],client_merge_seconds=result[2]))
                record('ready')
                print(f'Milvus search readiness: READY; queries={len(data):,}; '
                    f'attempts={len(attempts)}; report={report}',flush=True)
                return result
    except BaseException as error:
        # Preserve the underlying failure even if the evidence write also fails.
        try: record('failed',error)
        except Exception as report_error:
            print(f'Could not save readiness evidence: {report_error}',file=sys.stderr,flush=True)
        raise


