"""Runtime distance modes compiled together in the V2 Faiss extension."""
from functools import lru_cache
from pathlib import Path

import faiss

MODES = {
    'v2_auto': ('METRIC_MinhashBitmap', False, False),
    'cache_simd': ('METRIC_MinhashBitmapCacheSIMD', True, True),
    'cache_scalar': ('METRIC_MinhashBitmapCacheScalar', True, False),
    'nocache_simd': ('METRIC_MinhashBitmapNoCacheSIMD', False, True),
    'nocache_scalar': ('METRIC_MinhashBitmapNoCacheScalar', False, False),
}
PAPER_MODES = tuple(name for name in MODES if name != 'v2_auto')


class UnsupportedDistanceMode(RuntimeError):
    """The installed mode exists, but the current CPU cannot execute it."""


def loaded_faiss_libraries():
    """Actual mapped core libraries; the SWIG wrapper has a separate hash."""
    return tuple(sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                         if Path(line.split()[-1]).name.startswith('libfaiss')}))


def metric_for_mode(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown distance mode {mode!r}; choose from {tuple(MODES)}')
    name = MODES[mode][0]
    if not hasattr(faiss, name):
        raise RuntimeError('Rebuild V2 Faiss once with scripts/setup.py; runtime distance modes are missing')
    return getattr(faiss, name)


@lru_cache(maxsize=1)
def avx512_vpopcnt_available():
    probe = faiss.IndexBinaryHNSW(8, 4)
    probe.metric_type = metric_for_mode('v2_auto')
    return bool(probe.minhash_bitmap_uses_avx512())


def validate_mode(mode, bits):
    metric_for_mode(mode)
    _, cached, simd = MODES[mode]
    if simd and not avx512_vpopcnt_available():
        raise UnsupportedDistanceMode(f'{mode} requires AVX-512 VPOPCNTDQ; this CPU cannot execute it. '
                           'Choose a scalar mode or run on the Genoa VM.')
    if cached and bits > 65535:
        raise ValueError('Cached modes require at most 65535 bitmap bits for uint16 population counts')


def mode_info(mode):
    metric, cached, simd = MODES[mode]
    return {'name': mode, 'native_metric_name': metric, 'native_metric_value': int(metric_for_mode(mode)),
            'stored_population_cache': cached, 'cache_bytes_per_vector': 2 if cached else 0,
            'requires_avx512_vpopcntdq': simd,
            'uses_avx512_vpopcntdq': simd or (mode == 'v2_auto' and avx512_vpopcnt_available()),
            'empty_pair_distance': 0 if cached else 1}
