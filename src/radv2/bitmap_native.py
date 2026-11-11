"""Optional bit-identical native bitmap encoder; no Faiss rebuild required."""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('native/bitmap.cpp', 'external/simd/MurmurHash3.cpp', 'external/simd/MurmurHash3.h')
_LIBRARY = None
_IDENTITY = None


def source_identity():
    result=hashlib.sha256()
    for name in SOURCES:
        result.update(name.encode()+b'\0'+(ROOT/name).read_bytes())
    return result.hexdigest()


def prepare():
    global _LIBRARY, _IDENTITY
    identity=source_identity()
    if _LIBRARY is not None and identity!=_IDENTITY:
        raise RuntimeError('Native bitmap source changed; restart the process before using it')
    target=ROOT/'build/bitmap'/f'bitmap-{identity[:20]}.so'
    if _LIBRARY is None:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temporary=target.with_suffix(f'.{os.getpid()}.tmp.so')
            subprocess.run([os.environ.get('CXX','c++'),'-O3','-std=c++17','-fopenmp','-shared','-fPIC',
                '-I'+str(ROOT/'external/simd'),str(ROOT/SOURCES[0]),str(ROOT/SOURCES[1]),
                '-o',str(temporary)],check=True)
            temporary.replace(target)
        library=ctypes.CDLL(str(target))
        library.bitmap_encode.argtypes=[np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS'),ctypes.c_int64,ctypes.c_int,
            ctypes.c_uint64,ctypes.c_uint32,ctypes.c_int]
        library.bitmap_encode.restype=ctypes.c_int
        _LIBRARY=library
        _IDENTITY=identity
    return dict(source_sha256=identity,library=str(target),hash='MurmurHash3_x64_128 first uint64, four LE bytes')


def encode(sketches,spec,threads=1,metrics=None):
    values=np.asarray(sketches)
    if values.ndim!=2 or values.shape[1]!=spec.K or values.dtype.kind not in 'iu':
        raise ValueError('Native bitmap encoding requires an integer N x K matrix')
    if threads<1 or threads>2**31-1 or not 0<=spec.mmh3_seed<2**32:
        raise ValueError('Invalid native bitmap threads or uint32 hash seed')
    values=np.ascontiguousarray(values,dtype=np.uint32)
    output=np.empty((len(values),spec.M_bits//8),dtype=np.uint8)
    if _LIBRARY is None:
        prepare()
    team=_LIBRARY.bitmap_encode(values,output,len(values),spec.K,spec.M_bits,spec.mmh3_seed,int(threads))
    if metrics is not None:
        metrics.update(mode='native',requested_threads=int(threads),actual_team_threads=team)
    return output
