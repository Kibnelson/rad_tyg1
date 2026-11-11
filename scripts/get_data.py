#!/usr/bin/env python3
"""Download an ordered, full-text prefix from S3, FineWeb or Common Crawl WET."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


class S3RangeReader(io.RawIOBase):
    """Seekable range reader for Arrow; do not download a whole Parquet shard."""
    def __init__(self, client, bucket, key):
        super().__init__()
        self.client, self.bucket, self.key = client, bucket, key
        self.head = client.head_object(Bucket=bucket, Key=key)
        self.size, self.position = self.head['ContentLength'], 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError('Invalid seek')
        self.position = position
        return position

    def read(self, length=-1):
        if self.position >= self.size or length == 0:
            return b''
        end = self.size - 1 if length < 0 else min(self.size - 1, self.position + length - 1)
        response = self.client.get_object(Bucket=self.bucket, Key=self.key,
                                         Range=f'bytes={self.position}-{end}')
        try: body = response['Body'].read()
        finally: response['Body'].close()
        self.position += len(body)
        return body


def s3_rows(args, provenance):
    import boto3
    import pyarrow.parquet as pq
    from botocore import UNSIGNED
    from botocore.config import Config
    catalog = json.loads((ROOT / 'data/catalog.json').read_text())
    selected = catalog.get(args.dataset, {})
    uri = args.s3_uri or selected.get('uri')
    if not uri or not uri.startswith('s3://'):
        raise ValueError('Choose --dataset c4|cc_main|lm1b|realnews or provide --s3-uri')
    bucket, prefix = uri[5:].split('/', 1)
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED) if args.anonymous else None)
    if prefix.endswith('.parquet'):
        keys = [prefix]
    else:
        keys = [item['Key'] for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix)
                for item in page.get('Contents', []) if item['Key'].endswith('.parquet')]
        keys.sort(key=lambda k: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', k)])
    if not keys:
        raise ValueError('No Parquet files found at ' + uri)
    provenance.update(source='s3', uri=uri, objects=[])
    text_column = args.text_column or selected.get('text_column', 'contents')
    id_column = args.id_column or selected.get('id_column', 'int_id_column')
    for key in keys:
        with S3RangeReader(client, bucket, key) as remote:
            parquet = pq.ParquetFile(remote)
            names = parquet.schema_arrow.names
            if text_column not in names:
                raise ValueError(f'{key}: missing text column {text_column!r}')
            columns = [text_column] + ([id_column] if id_column in names else [])
            provenance['objects'].append(dict(key=key, bytes=remote.size, etag=remote.head.get('ETag')))
            for batch in parquet.iter_batches(batch_size=1024, columns=columns, use_threads=False):
                for row in batch.to_pylist():
                    yield dict(text=row[text_column], source_id=row.get(id_column), source_object=key)


def fineweb_rows(args, provenance):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = args.revision or HfApi().dataset_info('HuggingFaceFW/fineweb').sha
    provenance.update(source='fineweb', dataset='HuggingFaceFW/fineweb', config=args.config, revision=revision,
                      homepage='https://huggingface.co/datasets/HuggingFaceFW/fineweb')
    stream = load_dataset('HuggingFaceFW/fineweb', name=args.config, split='train',
                          streaming=True, revision=revision)
    for row in stream:
        yield dict(text=row['text'], source_id=row.get('id'), source_url=row.get('url'))


def common_crawl_rows(args, provenance):
    from warcio.archiveiterator import ArchiveIterator
    if not re.fullmatch(r'CC-MAIN-\d{4}-\d{2}', args.crawl):
        raise ValueError('Use a crawl identifier such as CC-MAIN-2024-10')
    base = 'https://data.commoncrawl.org/'
    listing = base + 'crawl-data/' + args.crawl + '/wet.paths.gz'
    provenance.update(source='common_crawl_wet', crawl=args.crawl, listing=listing, segments=[])
    with urlopen(listing, timeout=60) as response, gzip.GzipFile(fileobj=response) as paths:
        for encoded_path in paths:
            path = encoded_path.decode('utf-8').strip()
            if not path.startswith('crawl-data/') or '..' in path.split('/'):
                raise ValueError('Unexpected Common Crawl segment path')
            url = base + path
            provenance['segments'].append(url)
            with urlopen(url, timeout=60) as wet:
                for record in ArchiveIterator(wet):
                    if record.rec_type == 'conversion':
                        yield dict(text=record.content_stream().read().decode('utf-8', errors='replace'),
                                   source_id=record.rec_headers.get_header('WARC-Record-ID'),
                                   source_url=record.rec_headers.get_header('WARC-Target-URI'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('s3', 'fineweb', 'common-crawl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--dataset', choices=('c4', 'cc_main', 'lm1b', 'realnews'))
    parser.add_argument('--s3-uri')
    parser.add_argument('--text-column')
    parser.add_argument('--id-column')
    parser.add_argument('--anonymous', action='store_true')
    parser.add_argument('--config', default='sample-10BT')
    parser.add_argument('--revision')
    parser.add_argument('--crawl', default='CC-MAIN-2024-10')
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    receipt = args.output.with_name(args.output.name + '.provenance.json')
    if args.output.exists() or receipt.exists():
        parser.error('Output already exists; choose a new path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + '.partial')
    provenance = dict(full_text=True, text_edits=False, order='source order', requested_rows=args.rows)
    functions = {'s3': s3_rows, 'fineweb': fineweb_rows, 'common-crawl': common_crawl_rows}
    iterator = functions[args.source](args, provenance)
    count, digest = 0, hashlib.sha256()
    try:
        with partial.open('xb') as stream:
            for row in iterator:
                if not isinstance(row['text'], str):
                    raise ValueError(f'Source row {count} has no string text')
                body = (json.dumps(dict(id=count, **row), ensure_ascii=False) + '\n').encode('utf-8')
                stream.write(body); digest.update(body); count += 1
                if count == args.rows:
                    break
        if count != args.rows:
            raise ValueError(f'Source ended at {count} rows; requested {args.rows}')
        partial.rename(args.output)
        provenance.update(rows=count, sha256=digest.hexdigest(), file=args.output.name)
        receipt.write_text(json.dumps(provenance, indent=2) + '\n')
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        iterator.close()
    print(f'Saved {count:,} full-text documents: {args.output}')
    print(f'Provenance: {receipt}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Download failed: {exc}', file=sys.stderr)
        sys.exit(1)
