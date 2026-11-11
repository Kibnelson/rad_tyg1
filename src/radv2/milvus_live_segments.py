"""Read actual QueryNode segments without waiting for persisted tail statistics.

The pinned public GetQuerySegmentInfo lists sealed target segments only. The
existing metrics API exposes live physical IDs, insert counts and loaded indexes
for both sealed and growing segments. Nothing here flushes or changes a server.
"""
import json
from urllib.error import HTTPError
from urllib.request import urlopen


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f'Invalid QueryNode {name}: {value!r}')
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f'Invalid QueryNode {name}: {value!r}') from error
    if number < minimum or str(number) != str(value):
        raise ValueError(f'Invalid QueryNode {name}: {value!r}')
    return number


def loaded_flag(value):
    # Pinned metricsinfo.IndexedField uses json:",string" for this Boolean.
    if type(value) is bool:
        return value
    if value == 'true':
        return True
    if value == 'false':
        return False
    raise ValueError(f'Invalid QueryNode index loaded flag: {value!r}')


def querynode_segments(metrics_uri, collection_id, timeout=10):
    """Fail explicitly if live metadata is unavailable; never use stale counts."""
    collection_id = integer(collection_id, 'collection_id', 1)
    url = metrics_uri.rstrip('/') + f'/api/v1/_qn/segments?in=qn&collection_id={collection_id}'
    try:
        with urlopen(url, timeout=timeout) as response:
            body = response.read(8*1024*1024+1)
    except HTTPError as error:
        if error.code != 404:
            raise
        raise RuntimeError(
            f'Live QueryNode metadata API returned HTTP 404: {url}. '
            'The pinned server requires proxy.http.enabled=true and '
            'proxy.http.enableV1=true to register this metrics-port route. '
            'Update the harness and start a fresh trial with its generated config; '
            'do not fall back to stale persisted row counts.') from error
    if len(body) > 8*1024*1024:
        raise ValueError('QueryNode segment metadata exceeds the diagnostic limit')
    payload = json.loads(body)
    if payload is None:
        return []  # Pinned Go metrics marshal a nil segment slice as JSON null.
    if not isinstance(payload, list):
        raise ValueError('Expected the QueryNode segment array; check the local metrics API')
    selected = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError('Invalid QueryNode segment record')
        if integer(row.get('collection_id'), 'collection_id', 1) != collection_id:
            continue
        if row.get('state') not in ('Growing', 'Sealed'):
            raise ValueError(f'Unexpected QueryNode physical state: {row.get("state")}')
        # Zero-valued protobuf fields can be omitted by the JSON encoder.
        integer(row.get('segment_id'), 'segment_id', 1)
        integer(row.get('loaded_insert_row_count', 0), 'loaded_insert_row_count')
        integer(row.get('node_id'), 'node_id', 1)
        if not isinstance(row.get('index_fields', []), list):
            raise ValueError('Invalid QueryNode index fields')
        selected.append(row)
    return selected


def merge_live_segments(observed, live, expected, capacity=1_000_000):
    """Preserve coordinator history/index checks, use real live row counts.

    A physical ID may temporarily have both growing and sealed QueryNode copies
    during handoff. Prefer its sealed copy, require coherent same-state copies,
    and never sum replicas or invent a physical ID from a logical count.
    """
    merged = {row['segment_id']: dict(row, sources=list(row['sources'])) for row in observed}
    groups = {}
    for item in live:
        sid = integer(item.get('segment_id'), 'segment_id', 1)
        rows = integer(item.get('loaded_insert_row_count', 0), 'loaded_insert_row_count')
        state = item.get('state')
        if state not in ('Growing', 'Sealed'):
            raise ValueError(f'Unexpected live physical state: {state}')
        if rows > capacity or (state == 'Sealed' and 0 < rows < capacity):
            raise RuntimeError(f'Invalid live physical segment: {sid}; {state}; rows={rows}')
        groups.setdefault(sid, []).append(item)
    for row in merged.values():
        # Stale growing metadata is expected. Compaction and an early seal are
        # structural violations even if a growing copy is still in QueryNode.
        if (row.get('compaction_from') or (row.get('persistent_rows') or 0) > capacity
                or (row.get('query_rows') or 0) > capacity
                or (row.get('persistent_state') in ('Sealed', 'Flushing', 'Flushed')
                and 0 < row.get('persistent_rows', 0) < capacity)):
            raise RuntimeError(f'Invalid persistent segment history: {row}')
        row['live_query_rows'] = None
        row['live_query_state'] = None
        row['indexed'] = False  # Every loaded full segment also needs live proof.
    for sid, items in groups.items():
        sealed = [item for item in items if item['state'] == 'Sealed']
        selected = sealed or items
        counts = {integer(item.get('loaded_insert_row_count', 0), 'loaded_insert_row_count') for item in selected}
        if len(counts) != 1:
            raise ValueError(f'QueryNode copies have inconsistent rows for physical segment {sid}')
        rows = counts.pop()
        row = merged.setdefault(sid, dict(segment_id=sid, rows=0, state='Growing', indexed=False,
            compaction_from=[], sources=[], persistent_state=None, persistent_rows=None,
            query_state=None, query_rows=None, index_id=0, index_name='',
            expected_index_id=expected['index_id'], expected_index_name=expected['index_name'],
            expected_index_type=expected['index_type'], index_build_state=expected['build_state']))
        row.update(rows=rows, state='Sealed' if sealed else 'Growing', live_query_rows=rows,
            live_query_state='Sealed' if sealed else 'Growing',
            live_query_node_ids=sorted({integer(item['node_id'], 'node_id', 1) for item in selected}),
            row_count_source='querynode.loaded_insert_row_count', live_index_fields=[
                field for item in selected for field in item.get('index_fields', [])])
        row['sources'].append('query_node')
        loaded = all(any(integer(field.get('index_id', 0), 'index_id') == expected['index_id']
                         and loaded_flag(field.get('is_loaded',False)) for field in item.get('index_fields', []))
                     for item in selected)
        # Retain exact expected index ID and coordinator loaded-target evidence.
        row['indexed'] = bool(sealed and loaded and 'query' in row['sources']
            and row['query_state'] in ('Sealed', 'Flushed') and row['query_rows'] == rows
            and row['index_id'] == expected['index_id']
            and row['index_name'] in (expected['index_name'], expected['index_type']))
    return sorted(merged.values(), key=lambda row: row['segment_id'])
