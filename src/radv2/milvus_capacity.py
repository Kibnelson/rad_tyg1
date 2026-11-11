"""Physical-segment contract shared by insertion and the live Milvus gate."""


def capacity_slices(stored_rows, new_rows, capacity=1_000_000, wire_batch=5000):
    values = (stored_rows, new_rows, capacity, wire_batch)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
        raise ValueError('Row counters and limits must be integers')
    if min(stored_rows, new_rows) < 0 or min(capacity, wire_batch) < 1:
        raise ValueError('Use nonnegative row counters and positive limits')
    offset = 0
    while offset < new_rows:
        remaining = capacity - (stored_rows + offset) % capacity
        count = min(new_rows - offset, remaining, wire_batch)
        yield offset, count
        offset += count


def validate_layout(segments, expected_rows, capacity=1_000_000, require_indexed=True, require_live=False):
    """Validate actual physical IDs; never substitute logical partition counts."""
    if expected_rows < 0 or capacity < 1:
        raise ValueError('Invalid expected occupancy')
    ids, active, sealed, retained = set(), [], [], []
    for segment in segments:
        sid, rows = int(segment['segment_id']), int(segment['rows'])
        state = segment['state']
        if sid in ids:
            raise ValueError(f'Duplicate physical segment ID: {sid}')
        ids.add(sid)
        if rows < 0 or rows > capacity:
            raise ValueError(f'Physical segment {sid} has {rows} rows, capacity={capacity}')
        if segment.get('compaction_from'):
            raise ValueError(f'Compaction changed physical segment history: {sid}')
        if not rows:
            continue  # Empty preallocation does not add stored documents.
        if require_live and (segment.get('live_query_rows') != rows
                or segment.get('live_query_state') != state):
            raise ValueError(f'Physical segment {sid} has no matching live QueryNode observation')
        if state in ('Sealed', 'Flushing', 'Flushed'):
            if rows != capacity:
                raise ValueError(f'Early sealed segment {sid}: {rows}/{capacity}')
            if require_indexed and not segment.get('indexed'):
                raise ValueError(f'Full segment {sid} is not yet loaded with its index')
            sealed.append(sid)
        elif state == 'Growing':
            if rows == capacity:
                raise ValueError(f'Full segment {sid} is still growing; wait for sealing/indexing')
            active.append(sid)
        else:
            raise ValueError(f'Unexpected live segment state: {sid}: {state}')
        retained.append(segment)
    if len(active) > 1:
        raise ValueError(f'Multiple nonempty growing segments: {active}')
    if sum(int(s['rows']) for s in retained) != expected_rows:
        raise ValueError('Observed physical row count differs from acknowledged inserted rows')
    if len(sealed) != expected_rows // capacity or len(active) != bool(expected_rows % capacity):
        raise ValueError('Physical segment layout differs from full segments plus one active tail')
    return dict(capacity=capacity, stored_rows=expected_rows, sealed_segments=len(sealed),
                growing_segments=len(active), segments=retained)


def _expected_index(handler, collection):
    """Resolve the logical index to its ID before reading physical load reports."""
    from pymilvus.client.prepare import Prepare
    from pymilvus.client.utils import check_status
    from pymilvus.grpc_gen import common_pb2

    response = handler._stub.DescribeIndex(
        Prepare.describe_index_request(collection, 'minhash_lsh'), timeout=30)
    check_status(response.status)
    descriptions = list(response.index_descriptions)
    if len(descriptions) != 1:
        raise ValueError(f'Expected exactly one minhash_lsh index description; received {len(descriptions)}')
    index = descriptions[0]
    params = {item.key:item.value for item in index.params}
    if (index.index_name != 'minhash_lsh' or index.field_name != 'minhash_signature'
            or index.indexID <= 0 or params.get('index_type') != 'MINHASH_LSH'
            or params.get('metric_type') != 'MHJACCARD'):
        raise ValueError('Expected a positive-ID MINHASH_LSH/MHJACCARD index on minhash_signature; '
            f'received id={index.indexID}, name={index.index_name!r}, field={index.field_name!r}, '
            f'type={params.get("index_type")!r}, metric={params.get("metric_type")!r}')
    if index.state == common_pb2.IndexState.Failed:
        raise RuntimeError(f'MinHash index build failed: {index.index_state_fail_reason}')
    return dict(index_id=int(index.indexID), index_name=index.index_name,
        index_type=params['index_type'], build_state=common_pb2.IndexState.Name(index.state))


def _loaded_index_matches(row, expected):
    # GetIndexInfos in the pinned server can report the physical index type
    # as IndexName. The ID must still be the exact logical index we requested.
    return bool(expected['index_id'] > 0 and 'query' in row['sources']
        and row['query_state'] in ('Sealed', 'Flushed')
        and row['query_rows'] == row['rows']
        and row['index_id'] == expected['index_id']
        and row['index_name'] in (expected['index_name'], expected['index_type']))


def inspect_segments(handler, collection, live_segments=None):
    """Join expected index identity with physical metadata; no flush or mutation."""
    from pymilvus.client.types import SegmentState
    from pymilvus.grpc_gen import common_pb2

    expected = _expected_index(handler, collection)
    states = [SegmentState.Growing, SegmentState.Sealed, SegmentState.Flushing, SegmentState.Flushed]
    persistent = handler.get_persistent_segment_infos(collection, timeout=30, states=states)
    queried = handler.get_query_segment_info(collection, timeout=30)
    merged = {}
    for source, records in (('persistent', persistent), ('query', queried)):
        for item in records:
            if common_pb2.SegmentLevel.Name(item.level) == 'L0':
                continue
            sid = int(item.segmentID)
            row = merged.setdefault(sid, dict(segment_id=sid, rows=0, state='Growing',
                indexed=False, compaction_from=[], sources=[], persistent_state=None,
                persistent_rows=None, query_state=None, query_rows=None, index_id=0,
                index_name='', expected_index_id=expected['index_id'],
                expected_index_name=expected['index_name'], expected_index_type=expected['index_type'],
                index_build_state=expected['build_state']))
            row['sources'].append(source)
            row['rows'] = max(row['rows'], int(item.num_rows))
            state = common_pb2.SegmentState.Name(item.state)
            row[source+'_state'] = state
            row[source+'_rows'] = int(item.num_rows)
            if state != 'Growing' or row['state'] == 'Growing':
                row['state'] = state
            if source == 'query':
                row['index_id'] = int(item.indexID)
                row['index_name'] = item.index_name
                row['query_node_ids'] = list(item.nodeIds)
                row['indexed'] = _loaded_index_matches(row, expected)
            else:
                row['compaction_from'] = list(item.compaction_from)
    observed = sorted(merged.values(), key=lambda s: s['segment_id'])
    if live_segments is not None:
        from .milvus_live_segments import merge_live_segments
        observed = merge_live_segments(observed, live_segments, expected)
    return observed
