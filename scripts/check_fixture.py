#!/usr/bin/env python3
"""Verify the included public installation fixture without native dependencies."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
provenance = json.loads((root / 'data/SAMPLE_PROVENANCE.json').read_text())
for name, expected in provenance['files'].items():
    body = (root / 'data' / name).read_bytes()
    rows = [json.loads(line) for line in body.decode().splitlines()]
    assert len(body) == expected['bytes'] and hashlib.sha256(body).hexdigest() == expected['sha256'], name
    assert len(rows) == expected['rows']
    assert [r['id'] for r in rows] == list(range(len(rows)))
    assert all(isinstance(r['text'], str) for r in rows)
    print(f'{name}: {len(rows)} rows, SHA256 verified')
