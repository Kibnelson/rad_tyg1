#!/usr/bin/env python3
"""Tools-only one-owner-per-raw-group Milvus store and explicit completion seal.

No runtime or native sources are modified. Server configuration and its derived
launcher live only below the new store's instance directories.
"""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from types import SimpleNamespace

POLICY = 'milvus_end_group_sealing_v1'
PROFILE = dict(mode='approximate', mh_signature_length=112, mh_element_bit_width=32,
               mh_lsh_band=14, mh_lsh_rows_per_band=8, topk=32)
_HOOK_LOCK = threading.RLock()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physical(path):
    path = Path(path).absolute()
    require(path.resolve() == path, 'Use a physical group-store path: ' + str(path))
    if path.exists() and path.is_file():
        require(path.stat().st_nlink == 1, 'Use a private group-store metadata file')
    return path


def read(path):
    return json.loads(physical(path).read_text())


def write(path, value):
    path = physical(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = physical(path.with_name(path.name + '.partial'))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bytes_once(path, body, verify=False):
    path = physical(path)
    if path.exists():
        require(path.is_file() and path.read_bytes() == body, 'Existing group server file differs: ' + str(path))
    else:
        require(not verify, 'Missing group server file: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def _integer(value, label, minimum=0):
    require(not isinstance(value, bool) and isinstance(value, (int, str)), 'Invalid ' + label)
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError('Invalid ' + label) from error
    require(number >= minimum and str(number) == str(value), 'Invalid ' + label)
    return number


def _loaded(value):
    require(type(value) is bool or value in ('true', 'false'), 'Invalid physical index loaded flag')
    return value is True or value == 'true'


def _launcher(runtime, folder, capacity, policy, verify=False):
    original = physical(runtime / 'scripts/run_milvus_local.sh')
    body = original.read_text()
    root_line = 'RADV2_SERVER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"'
    config_line = 'python3 "$RADV2_SERVER_ROOT/scripts/check_milvus_sources.py" "${RADV2_CONFIG_ARGS[@]}"'
    banner = 'physical row cap=1000000;'
    require(all(body.count(item) == 1 for item in (root_line, config_line, banner)),
            'Pinned native launcher does not match the reviewed substitutions')
    adapter = physical(__file__)
    derived = body.replace(root_line, 'RADV2_SERVER_ROOT=' + shlex.quote(str(runtime)))
    command = ('python3 ' + shlex.quote(str(adapter)) + ' verify-config --runtime "$RADV2_SERVER_ROOT" '
        '--directory "$RADV2_SERVER_CONFIG" --instance-slot "$RADV2_INSTANCE_SLOT" '
        '--capacity ' + str(capacity) + ' --policy ' + shlex.quote(policy))
    derived = derived.replace(config_line, command).replace(banner, f'physical row cap={capacity};')
    target = folder / 'run_group_milvus_local.sh'
    payload = derived.encode()
    _bytes_once(target, payload, verify)
    return dict(path=str(target), sha256=hashlib.sha256(payload).hexdigest(),
        original_path=str(original), original_sha256=sha(original),
        adapter_path=str(adapter), adapter_sha256=sha(adapter),
        transforms=['fixed_qualified_runtime_root', 'reviewed_group_config_verifier', 'capacity_banner'])


def configure_server(source, directory, instance_slot, capacity, policy=POLICY, *, verify=False):
    """Create the original namespace/ports/profile with only maxRowNum changed."""
    require(type(capacity) is int and capacity > 0, 'Use a positive group capacity')
    require(isinstance(policy, str) and re.fullmatch(r'[a-z][a-z0-9_]+', policy), 'Invalid group policy')
    directory, runtime = physical(directory), physical(source.ROOT)
    checked = source.check()  # The immutable source/native inventory is still checked.
    inputs = {'milvus.yaml':runtime / 'external/milvus/configs/milvus.yaml',
              'glog.conf':runtime / 'external/milvus/configs/glog.conf',
              'user.yaml':runtime / 'reference/milvus/physical_1m_user.yaml'}
    profile = physical(inputs['user.yaml']).read_text()
    pattern = r'(?m)^(\s+maxRowNum:\s*)1000000\s*$'
    require(len(re.findall(pattern, profile)) == 1 and 'enableCompaction: false' in profile
            and 'enableAutoCompaction: false' in profile, 'Qualified physical profile differs')
    overlay = re.sub(pattern, lambda match:match[1] + str(capacity), profile)
    data = directory.parent / 'server_data'
    namespace = 'radv2-' + hashlib.sha256(str(data).encode()).hexdigest()[:16]
    overlay += ('\n# Dedicated writable paths and metadata namespaces for this server.\n'
        f'localStorage:\n  path: {json.dumps(str(data / "local"))}\n'
        f'rocksmq:\n  path: {json.dumps(str(data / "rocksmq"))}\n'
        f'log:\n  file:\n    rootPath: {json.dumps(str(data / "logs"))}\n'
        f'etcd:\n  rootPath: {namespace}\nminio:\n  rootPath: {namespace}\n')
    ports = source.instance_ports(instance_slot) if instance_slot is not None else None
    if ports is not None:
        overlay += (f'proxy:\n  port: {ports["proxy"]}\n  internalPort: {ports["proxy_internal"]}\n'
            f'  http:\n    enabled: true\n    enableV1: true\n    port: {ports["proxy_http"]}\n'
            f'common:\n  MetricsPort: {ports["metrics"]}\n')
        require(overlay.count('dataCoord:\n') == 1, 'Expected one dataCoord mapping')
        overlay = overlay.replace('dataCoord:\n', f'dataCoord:\n  port: {ports["dataCoord"]}\n', 1)
        for name in ('rootCoord', 'queryCoord', 'queryNode', 'dataNode', 'streamingNode'):
            overlay += f'{name}:\n  port: {ports[name]}\n'
    if directory.exists():
        require(directory.is_dir() and not ({p.name for p in directory.iterdir()} - set(inputs)),
                'Use a dedicated group config directory')
    else:
        require(not verify, 'Missing group config directory')
        directory.mkdir(parents=True)
    for name, source_path in inputs.items():
        _bytes_once(directory / name, overlay.encode() if name == 'user.yaml' else physical(source_path).read_bytes(), verify)
    launcher = _launcher(runtime, directory.parent, capacity, policy, verify)
    build = physical(runtime / 'build/milvus_build.json')
    require(build.is_file(), 'Qualified native build receipt is missing')
    receipt = dict(schema_version=1, kind='milvus_group_server_configuration_v1', policy=policy,
        runtime=str(runtime), physical_row_capacity=capacity, compaction=False, namespace=namespace,
        directory=str(directory), ports=ports, files={name:sha(directory / name) for name in inputs},
        source_check=checked, native_build_receipt=dict(path=str(build), sha256=sha(build)),
        original_profile=dict(path=str(inputs['user.yaml']), sha256=sha(inputs['user.yaml'])), launcher=launcher)
    path = directory.parent / 'group_server_configuration.json'
    payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _bytes_once(path, payload, verify)
    return receipt


def merge_group_segments(observed, live, expected, capacity, closed=False):
    """Use actual QueryNode IDs/rows and the exact coordinator-loaded index ID."""
    merged = {row['segment_id']:dict(row, sources=list(row['sources'])) for row in observed}
    require(len(merged) == len(observed), 'Repeated physical segment ID')
    groups = {}
    for item in live:
        sid = _integer(item.get('segment_id'), 'live segment ID', 1)
        rows = _integer(item.get('loaded_insert_row_count', 0), 'live rows')
        state = item.get('state')
        require(state in ('Growing', 'Sealed'), 'Unexpected live physical state')
        if rows > capacity or (state == 'Sealed' and 0 < rows < capacity and not closed):
            raise RuntimeError('Active group sealed early or exceeded its configured capacity')
        groups.setdefault(sid, []).append(item)
    for row in merged.values():
        if (row.get('compaction_from') or (row.get('persistent_rows') or 0) > capacity
                or (row.get('query_rows') or 0) > capacity
                or (not closed and row.get('persistent_state') in ('Sealed', 'Flushing', 'Flushed')
                    and 0 < (row.get('persistent_rows') or 0) < capacity)):
            raise RuntimeError('Physical history violates the group sealing policy')
        row.update(live_query_rows=None, live_query_state=None, indexed=False)
    for sid, items in groups.items():
        sealed = [item for item in items if item['state'] == 'Sealed']
        selected = sealed or items
        counts = {_integer(item.get('loaded_insert_row_count', 0), 'live rows') for item in selected}
        require(len(counts) == 1, 'Inconsistent physical row counts across live copies')
        rows = counts.pop()
        row = merged.setdefault(sid, dict(segment_id=sid, sources=[], compaction_from=[],
            persistent_rows=None, persistent_state=None, query_rows=None, query_state=None,
            index_id=0, index_name='', expected_index_id=expected['index_id'],
            expected_index_name=expected['index_name'], expected_index_type=expected['index_type'],
            index_build_state=expected['build_state']))
        row.update(rows=rows, state='Sealed' if sealed else 'Growing', live_query_rows=rows,
            live_query_state='Sealed' if sealed else 'Growing',
            live_query_node_ids=sorted({_integer(item['node_id'], 'node ID', 1) for item in selected}),
            row_count_source='querynode.loaded_insert_row_count',
            live_index_fields=[field for item in selected for field in item.get('index_fields', [])])
        row['sources'].append('query_node')
        loaded = all(any(_integer(field.get('index_id', 0), 'index ID') == expected['index_id']
                         and _loaded(field.get('is_loaded', False)) for field in item.get('index_fields', []))
                     for item in selected)
        row['indexed'] = bool(sealed and loaded and 'query' in row['sources']
            and row.get('query_state') in ('Sealed', 'Flushed') and row.get('query_rows') == rows
            and row.get('index_id') == expected['index_id']
            and row.get('index_name') in (expected['index_name'], expected['index_type']))
    return sorted(merged.values(), key=lambda row:row['segment_id'])


def validate_group_layout(segments, expected, capacity, closed=False):
    require(type(capacity) is int and capacity > 0 and type(expected) is int and 0 <= expected <= capacity,
            'Invalid group occupancy/capacity')
    retained, seen = [], set()
    for row in segments:
        sid, rows = _integer(row.get('segment_id'), 'segment ID', 1), _integer(row.get('rows'), 'segment rows')
        require(sid not in seen, 'Repeated physical segment ID'); seen.add(sid)
        if rows > capacity or row.get('compaction_from'):
            raise RuntimeError('Group capacity or compaction invariant failed')
        if not rows:
            continue
        require(row.get('live_query_rows') == rows and row.get('live_query_state') == row.get('state'),
                'Physical group has no matching actual live row proof')
        state = row.get('state')
        require(state in ('Growing', 'Sealed', 'Flushed'), 'Unexpected physical group state')
        if not closed and rows < capacity and state != 'Growing':
            raise RuntimeError('Active partial group sealed before group completion')
        if closed or rows == capacity:
            require(state in ('Sealed', 'Flushed') and row.get('indexed') is True,
                    'Group is not yet sealed and loaded with its exact index')
        retained.append(row)
    require(sum(row['rows'] for row in retained) == expected
            and len(retained) == (1 if expected else 0), 'Expected one exact-row physical group segment')
    return dict(capacity=capacity, stored_rows=expected,
        sealed_segments=sum(row['state'] in ('Sealed', 'Flushed') for row in retained),
        growing_segments=sum(row['state'] == 'Growing' for row in retained), segments=retained)


def create_store_class(store_module, capacity, policy=POLICY, *, topk=32):
    """Return a scoped subclass; importing this module never imports a native SDK."""
    require(type(capacity) is int and capacity > 0, 'Use a positive group capacity')
    require(isinstance(policy, str) and re.fullmatch(r'[a-z][a-z0-9_]+', policy), 'Invalid group policy')
    require(type(topk) is int and topk in (4, 32), 'Use the reviewed topk4 or legacy topk32 profile')
    profile = dict(PROFILE, topk=topk)
    base, servers = store_module.GrowingMilvus, store_module.Servers
    with _HOOK_LOCK:
        original_band = getattr(store_module, 'band_configuration', None)
    require(topk == 32 or callable(original_band), 'Prepared topk4 requires the qualified band-profile constructor')

    def selected_band(value=None):
        result = original_band(value)
        require(result == {key: value for key, value in PROFILE.items() if key != 'mode'},
                'Qualified native band profile changed')
        return dict(result, topk=topk)

    class GroupServers(servers):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            source = self.source
            self.source = SimpleNamespace(**vars(source))
            self.source.config = lambda directory, slot=None:configure_server(source, directory, slot, capacity, policy)

        def __enter__(self):
            # The pinned runner repeats config verification. Redirect only that
            # exact launcher; all native/build/library/CPU checks stay verbatim.
            with _HOOK_LOCK:
                original = subprocess.Popen
                runtime = physical(self.source.ROOT)
                runner = str(runtime / 'scripts/run_milvus_local.sh')

                def launch(argv, *args, **kwargs):
                    if isinstance(argv, (list, tuple)) and len(argv) >= 3 and list(argv[:2]) == ['bash', runner]:
                        derived = _launcher(runtime, physical(argv[2]).parent, capacity, policy, verify=True)
                        argv = [argv[0], derived['path'], *argv[2:]]
                    return original(argv, *args, **kwargs)

                subprocess.Popen = launch
                try:
                    return super().__enter__()
                finally:
                    subprocess.Popen = original

    class GroupStore(base):
        def __init__(self, root, layout, cpus, timeout=1800, resume=False,
                     capacity=capacity, mode='approximate', mh_lsh_band=14):
            require(capacity == factory_capacity and type(capacity) is int and layout == 'separate_sequential'
                    and mode == 'approximate' and mh_lsh_band == 14,
                    'Use the fixed separate approximate14x8 group-store profile')
            root = physical(root)
            if (root / 'store.json').exists():
                saved = read(root / 'store.json')
                require(resume and saved.get('group_sealing', {}).get('policy') == policy
                        and saved['group_sealing'].get('capacity') == capacity,
                        'Saved store lacks this exact group sealing policy; no migration')
            self._allow_group_instance = False
            # Source/native bytes stay pinned. Select only this constructor's
            # persisted search budget, before its initial save/resume checks.
            # Serialize legacy constructors too, so they cannot observe a
            # simultaneous prepared constructor's temporary profile function.
            with _HOOK_LOCK:
                previous_band = getattr(store_module, 'band_configuration', None)
                if original_band is not None:
                    store_module.band_configuration = selected_band if topk != 32 else original_band
                try:
                    super().__init__(root, layout, cpus, timeout, resume, capacity, mode, mh_lsh_band)
                finally:
                    if original_band is not None:
                        store_module.band_configuration = previous_band
            self._validate_groups()

        @property
        def group_state(self):
            return self.state['group_sealing']

        @property
        def current_group(self):
            return self.group_state['groups'][-1] if self.group_state['groups'] else None

        def save(self):
            self.state.setdefault('group_sealing', dict(schema_version=1, policy=policy,
                capacity=factory_capacity, profile=dict(profile), groups=[]))
            return super().save()

        def _validate_groups(self):
            header = self.group_state
            require(type(header.get('schema_version')) is int and header['schema_version'] == 1
                    and header.get('policy') == policy and type(header.get('capacity')) is int
                    and header['capacity'] == factory_capacity and header.get('profile') == profile,
                    'Saved group-store profile changed')
            groups = header.get('groups')
            require(isinstance(groups, list) and len(groups) == len(self.state['instances']),
                    'Group/owner inventory differs')
            require(sum(item['rows'] for item in self.state['instances']) == self.ntotal,
                    'Group occupancy differs from stored document count')
            for index, (group, item) in enumerate(zip(groups, self.state['instances'])):
                require(type(group.get('number')) is int and group['number'] == index + 1
                        and type(group.get('owner')) is int and group['owner'] == index
                        and group.get('status') in ('open', 'closing', 'closed')
                        and (index == len(groups) - 1 or group['status'] == 'closed')
                        and type(item['rows']) is int and 0 <= item['rows'] <= self.capacity,
                        'Invalid saved group state')
                if group['status'] != 'open':
                    require(group.get('actual_rows') == item['rows']
                            and isinstance(group.get('closing_binding'), dict)
                            and (self.state.get('pending') is None or self.state['pending'].get('owner') != index),
                            'Closing group occupancy changed')
                if group['status'] == 'closed':
                    path = physical(group.get('receipt_path', ''))
                    require(path.is_file() and sha(path) == group.get('receipt_sha256'),
                            'Closed group receipt is missing or changed')
            pending = self.state.get('pending')
            if pending is not None:
                require(groups and groups[-1]['status'] == 'open' and pending.get('owner') == groups[-1]['owner'],
                        'Pending insertion is not bound to the current open group')

        def __enter__(self):
            try:
                for owner in range(len(self.state['instances'])):
                    self._connect(owner)
                self.tokens = None
                if self.state.get('pending') is not None:
                    require(self.current_group is not None and self.current_group['status'] == 'open',
                            'Pending append belongs to a closing/closed group')
                    self._finish_pending(recover=True)
                for owner, item in enumerate(self.state['instances']):
                    # Closing may have crashed before flush; finalize_group must
                    # replay flush before waiting for a partial sealed segment.
                    if item['rows'] and self.group_state['groups'][owner]['status'] != 'closing':
                        self.synchronize(owner, bytes.fromhex(item['last_signature']))
                        group = self.group_state['groups'][owner]
                        if group['status'] == 'closed':
                            self._closed_receipt(group, physical(group['receipt_path']), verify_live=True)
                return self
            except BaseException:
                self.__exit__(None, None, None)
                raise

        def _connect(self, owner):
            with _HOOK_LOCK:
                original = store_module.Servers
                store_module.Servers = GroupServers
                try:
                    result = super()._connect(owner)
                finally:
                    store_module.Servers = original
            path = physical(self.root / 'instances' / f'owner_{owner:03d}' / 'single/group_server_configuration.json')
            require(path.is_file(), 'Missing reviewed group server configuration')
            record = dict(path=str(path), sha256=sha(path))
            saved = self.state['instances'][owner].get('group_configuration')
            require(saved is None or saved == record, 'Saved server configuration changed')
            self.state['instances'][owner]['group_configuration'] = record
            self.save()
            return result

        def _new_instance(self):
            require(self._allow_group_instance, 'Only begin_group may open a new physical owner')
            return super()._new_instance()

        def begin_group(self, number):
            require(type(number) is int and number >= 1, 'Use a positive group number')
            group = self.current_group
            if group is not None and group['number'] == number:
                return copy.deepcopy(group)
            require(number == len(self.group_state['groups']) + 1
                    and (group is None or group['status'] == 'closed'), 'Close the previous group before opening the next')
            if group is not None:
                self._closed_receipt(group, physical(group['receipt_path']), verify_live=True)
            owner = len(self.state['instances'])
            self.group_state['groups'].append(dict(number=number, owner=owner, status='open'))
            self._allow_group_instance = True
            try:
                require(self._new_instance() == owner, 'Group owner changed during creation')
            except BaseException:
                if len(self.state['instances']) == owner:
                    self.group_state['groups'].pop()
                    self.save()
                raise
            finally:
                self._allow_group_instance = False
            return copy.deepcopy(self.current_group)

        def append(self, frame, sketches):
            group = self.current_group
            require(group is not None and group['status'] == 'open' and self.state.get('pending') is None,
                    'Begin an open group before appending; closed groups are immutable')
            ids = list(map(int, frame['int_id_column'].to_list()))
            require(len(ids) == len(set(ids)) and len(sketches) == len(ids), 'Invalid group insertion IDs/sketches')
            require(all(doc not in self.owner or self.owner[doc] == group['owner'] for doc in ids),
                    'Replay would move a document from another group')
            unseen = sum(doc not in self.owner for doc in ids)
            require(self.state['instances'][group['owner']]['rows'] + unseen <= self.capacity,
                    'Group would exceed capacity; no rollover or partial append is permitted')
            return super().append(frame, sketches)

        def search(self, data, topk=topk, refine_k=160):
            require(type(topk) is int and topk == profile['topk'],
                    'Use the saved group-store search budget: topk=' + str(profile['topk']))
            return super().search(data, topk, refine_k)

        def _observe(self, owner):
            module = importlib.import_module('radv2.milvus_capacity')
            client = self.clients[owner]
            handler = client._get_connection()
            expected = module._expected_index(handler, store_module.COLLECTION)
            observed = module.inspect_segments(handler, store_module.COLLECTION)
            closed = self.group_state['groups'][owner]['status'] != 'open'
            return merge_group_segments(observed, self.live_segments(owner), expected, self.capacity, closed)

        def _wait(self, owner, report=None):
            began, polls = time.monotonic(), []
            closed = self.group_state['groups'][owner]['status'] != 'open'
            expected = self.state['instances'][owner]['rows']
            while time.monotonic() - began < self.timeout:
                try:
                    observed = self._observe(owner)
                    layout = validate_group_layout(observed, expected, self.capacity, closed)
                except ValueError as error:
                    polls.append(dict(status='waiting', error=str(error), elapsed_seconds=time.monotonic() - began))
                    if report is not None:
                        write(report, dict(status='waiting', expected_rows=expected, capacity=self.capacity,
                            group_closed_or_closing=closed, polls=polls))
                    time.sleep(1.)
                except RuntimeError as error:
                    if report is not None:
                        write(report, dict(status='failed', expected_rows=expected, capacity=self.capacity,
                            group_closed_or_closing=closed, polls=polls, error=str(error)))
                    raise
                else:
                    if time.monotonic() - began > self.timeout:
                        raise RuntimeError('Group layout proof exceeded its readiness deadline')
                    polls.append(dict(status='ready', elapsed_seconds=time.monotonic() - began))
                    if report is not None:
                        write(report, dict(status='ready', expected_rows=expected, capacity=self.capacity,
                            group_closed_or_closing=closed, polls=polls, layout=layout))
                    return layout, len(polls)
            if report is not None:
                write(report, dict(status='failed', expected_rows=expected, capacity=self.capacity,
                    group_closed_or_closing=closed, polls=polls, error='layout readiness timeout'))
            raise RuntimeError('Timed out waiting for the exact group layout and loaded physical index')

        def _warmup(self, owner, signature, report):
            searcher = store_module.Searcher([self.clients[owner]], store_module.COLLECTION,
                                            False, profile['topk'], 160, False)
            try:
                result = store_module.warmup_search(searcher, [signature], self.timeout,
                    self.handles[owner].processes, report)
            finally:
                searcher.close()
            if report.exists():
                value = read(report)
                value.update(excluded_from_throughput=False, policy=policy,
                    timing_scope='Recorded group-store synchronization; parent accounting determines interval')
                write(report, value)
            return result

        def synchronize(self, owner, signature):
            began = time.perf_counter()
            rows = self.state['instances'][owner]['rows']
            name = f'{owner:03d}_{rows:09d}.json'
            report = self.root / 'group_layout_waits' / name
            layout, polls = self._wait(owner, report)
            layout_seconds = time.perf_counter() - began
            step = time.perf_counter()
            readiness = self.root / 'group_visibility' / name
            self._warmup(owner, signature, readiness)
            return dict(owner=owner, visibility_index_seconds=time.perf_counter() - began, layout=layout,
                layout_wait_seconds=layout_seconds, readiness_seconds=time.perf_counter() - step,
                layout_poll_count=polls, observation_source='live_querynode_and_coordinators',
                layout_wait_report=str(report.relative_to(self.root)), readiness_report=str(readiness.relative_to(self.root)),
                group_sealing_policy=policy)

        def layouts(self):
            return [self._wait(owner)[0] for owner in range(len(self.state['instances']))]

        def _binding(self, group):
            owner = group['owner']
            return dict(policy=policy, store_root=str(self.root), case_directory=str(self.root.parent),
                group=group['number'], owner=owner, capacity=self.capacity,
                actual_rows=self.state['instances'][owner]['rows'], collection_id=self.collection_ids[owner],
                server_configuration=self.state['instances'][owner]['group_configuration'], **profile)

        def _index_proof(self, owner, layout):
            client = self.clients[owner]
            identity = client.describe_collection(store_module.COLLECTION, timeout=self.timeout)
            require(int(identity['collection_id']) == self.collection_ids[owner], 'Native collection identity changed')
            self._validate_index(client, identity)
            index = client.describe_index(store_module.COLLECTION, 'minhash_lsh', timeout=self.timeout)
            require(all(row.get('indexed') is True for row in layout['segments']), 'Physical index is not loaded')
            return dict(collection=identity, index=index,
                physical=[dict(segment_id=row['segment_id'], rows=row['rows'],
                    index_id=row.get('expected_index_id'), indexed=row['indexed']) for row in layout['segments']])

        def _closed_receipt(self, group, path, verify_live=False):
            value = read(path)
            require(value.get('kind') == 'milvus_end_group_completion_v1' and value.get('status') == 'closed'
                    and value.get('binding') == self._binding(group)
                    and value.get('policy') == policy and value.get('group') == group['number']
                    and value.get('owner') == group['owner'] and value.get('capacity') == self.capacity
                    and value.get('actual_rows') == self.state['instances'][group['owner']]['rows']
                    and value.get('binding') == group.get('closing_binding'),
                    'Saved group completion receipt binding changed')
            require(type(value.get('timing_complete')) is bool, 'Invalid group completion timing flag')
            names = ('flush_rpc_seconds', 'load_rpc_seconds', 'layout_wait_seconds', 'readiness_seconds', 'other_seconds')
            if value['timing_complete']:
                seconds = [value.get(name) for name in (*names, 'total_seconds')]
                require(all(type(number) in (int, float) and math.isfinite(number) and number >= 0 for number in seconds)
                        and math.isclose(sum(seconds[:-1]), seconds[-1], rel_tol=1e-9, abs_tol=1e-6),
                        'Saved group completion timers do not reconcile')
            else:
                require(all(value.get(name) is None for name in (*names, 'total_seconds')),
                        'Interrupted group timing must remain unavailable')
            expected = group.get('receipt_sha256')
            require(expected is None or sha(path) == expected, 'Saved group completion receipt changed')
            if verify_live:
                layout = self._wait(group['owner'])[0]
                proof = self._index_proof(group['owner'], layout)
                require(proof['physical'] == value['index_proof']['physical'], 'Closed physical segment/index identity changed')
            return value

        def finalize_group(self, number, receipt_path):
            group = self.current_group
            require(group is not None and group['number'] == number and self.state.get('pending') is None,
                    'Finalize only the current group with no pending append')
            path = physical(receipt_path)
            require(path.is_relative_to(self.root.parent) and path != self.root / 'store.json',
                    'Keep group completion evidence under its case/audit directory')
            require(group.get('receipt_path', str(path)) == str(path), 'Group completion receipt path changed')
            if group['status'] == 'closed' or path.exists():
                require(group['status'] in ('closing', 'closed'), 'Unexpected completion evidence for an open group')
                value = self._closed_receipt(group, path, verify_live=True)
                group.update(status='closed', receipt_path=str(path), receipt_sha256=sha(path))
                self.save()
                return value
            recovered = group['status'] == 'closing'
            owner = group['owner']
            rows = self.state['instances'][owner]['rows']
            require(0 < rows <= self.capacity, 'A nonempty bounded group is required before sealing')
            binding = self._binding(group)
            require(not recovered or group.get('closing_binding') == binding,
                    'Interrupted closing native collection/profile binding changed')
            began = time.perf_counter()
            group.update(status='closing', actual_rows=rows, receipt_path=str(path), closing_binding=binding)
            group['closing_attempts'] = group.get('closing_attempts', 0) + 1
            self.save()  # Closing intent is durable before the first native flush.
            client = self.clients[owner]
            identity = client.describe_collection(store_module.COLLECTION, timeout=self.timeout)
            self._validate_index(client, identity)
            step = time.perf_counter(); client.flush(store_module.COLLECTION, timeout=self.timeout)
            flush = time.perf_counter() - step
            step = time.perf_counter(); client.load_collection(store_module.COLLECTION, timeout=self.timeout)
            load = time.perf_counter() - step
            step = time.perf_counter()
            layout_report = self.root / 'group_completion_layout' / f'group_{number:03d}.json'
            layout, polls = self._wait(owner, layout_report)
            layout_wait = time.perf_counter() - step
            step = time.perf_counter()
            readiness_report = self.root / 'group_completion_readiness' / f'group_{number:03d}.json'
            self._warmup(owner, bytes.fromhex(self.state['instances'][owner]['last_signature']), readiness_report)
            readiness = time.perf_counter() - step
            proof = self._index_proof(owner, layout)
            total = time.perf_counter() - began
            components = dict(flush_rpc_seconds=flush, load_rpc_seconds=load,
                layout_wait_seconds=layout_wait, readiness_seconds=readiness,
                other_seconds=max(0., total - flush - load - layout_wait - readiness), total_seconds=total)
            value = dict(schema_version=1, kind='milvus_end_group_completion_v1', status='closed',
                policy=policy, group=number, owner=owner, capacity=self.capacity, actual_rows=rows,
                binding=self._binding(group), physical_layout=layout, index_proof=proof,
                timing_complete=not recovered, recovered_closing=recovered,
                **{key:(None if recovered else seconds) for key, seconds in components.items()},
                retry_timings=components if recovered else None, layout_poll_count=polls,
                layout_wait_report=str(layout_report), readiness_report=str(readiness_report),
                timing_scope='Flush/load through live indexed layout and strong-query readiness; final receipt/store commit follows the timer',
                native_rebuilt=False, finalization_forced_flush=True)
            write(path, value)
            group.update(status='closed', receipt_sha256=sha(path))
            self.save()
            return value

    factory_capacity = capacity
    GroupStore.__name__ = 'GroupSealingMilvus'
    return GroupStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify-config',))
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--instance-slot', default='')
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--policy', default=POLICY)
    args = parser.parse_args()
    runtime = physical(args.runtime)
    spec = importlib.util.spec_from_file_location('qualified_group_source_check', runtime / 'scripts/check_milvus_sources.py')
    source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
    slot = int(args.instance_slot) if args.instance_slot else None
    configure_server(source, args.directory, slot, args.capacity, args.policy, verify=True)


if __name__ == '__main__':
    main()
