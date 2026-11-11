"""Portable loader for the unchanged group-store mechanics."""
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(8<<20), b""):h.update(part)
    return h.hexdigest()
def record(path):
    return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def _immutable(path, value):
    """Atomically publish a completion receipt without replacing prior evidence."""
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    body=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():
        if not path.is_file() or path.read_bytes()!=body:
            raise ValueError('Preserve different completion evidence')
        return
    fd,name=tempfile.mkstemp(prefix='.milvus-completion-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.link(name,path)
    finally:
        Path(name).unlink(missing_ok=True)
@lru_cache(maxsize=1)
def _group_tools():
    spec=importlib.util.spec_from_file_location("public_milvus_group_store",Path(__file__).with_name("milvus_group_store.py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
