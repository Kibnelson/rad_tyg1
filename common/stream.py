"""Read ordered full-text documents; no deduplication or text truncation."""
from __future__ import annotations

import gzip
import json
from pathlib import Path


def iter_documents(path, max_documents=None):
    path = Path(path)
    if max_documents is not None and max_documents < 1:
        raise ValueError('max_documents must be positive')
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        source = (row for batch in pq.ParquetFile(path).iter_batches(batch_size=1024)
                  for row in batch.to_pylist())
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        def json_rows():
            with opener(path, 'rt', encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'{path}:{line_number}: invalid JSON') from exc
        source = json_rows()
    for ordinal, raw in enumerate(source):
        if max_documents is not None and ordinal >= max_documents:
            break
        if not isinstance(raw, dict):
            raise ValueError(f'Row {ordinal}: expected an object')
        text = raw.get('text', raw.get('contents'))
        if not isinstance(text, str):
            raise ValueError(f'Row {ordinal}: expected a string text or contents field')
        row = dict(raw)
        row['text'] = text
        row.setdefault('id', raw.get('int_id_column', ordinal))
        yield row


def iter_batches(path, batch_size=10000, max_documents=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    batch = []
    for row in iter_documents(path, max_documents):
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
