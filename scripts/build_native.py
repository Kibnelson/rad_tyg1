#!/usr/bin/env python3
"""Build the unchanged SIMD engines and check a real native duplicate fixture."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'vendor/simd'
BASE = 'simd_fuzzy_deduplicationv7_unsorted_multi7A_V1'


def record(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    import pybind11
    import numpy as np
    flags = set(Path('/proc/cpuinfo').read_text().split())
    if not {'avx2', 'popcnt'} <= flags:
        raise SystemExit('The native SIMD engine requires Linux x86-64 with AVX2 and POPCNT.')
    compile_flags = ['-mavx2', '-mpopcnt']
    if {'avx512f', 'avx512dq', 'avx512vl', 'avx512bw'} <= flags:
        compile_flags += ['-mavx512f', '-mavx512dq', '-mavx512vl', '-mavx512bw']
    modules = [BASE]
    if (NATIVE / 'simd_full_reference_v1.cpp').is_file():
        modules.append('simd_full_reference_v1')
    binaries = []
    for name in modules:
        target = NATIVE / (name + sysconfig.get_config_var('EXT_SUFFIX'))
        command = ['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp', *compile_flags,
                   '-I' + sysconfig.get_path('include'), '-I' + pybind11.get_include(),
                   str(NATIVE / (name + '.cpp')), str(NATIVE / 'MurmurHash3.cpp'), '-o', str(target)]
        subprocess.run(command, check=True)
        binaries.append(record(target))
    sys.path.insert(0, str(NATIVE))
    baseline = importlib.import_module(BASE)
    rng = np.random.RandomState(49037)
    signatures = rng.randint(0, 2**32, size=(8, 112), dtype=np.uint64)
    signatures[6] = signatures[0]
    removed = getattr(baseline, BASE)(signatures, threshold=.7, num_bands=14, num_threads=2)
    drops = {int(i) for group in removed for i in group}
    assert len(drops & {0, 6}) == 1 and len(drops) == 1, 'Native identity fixture failed'
    if 'simd_full_reference_v1' in modules:
        full = importlib.import_module('simd_full_reference_v1').full_reference(signatures, 6, .7, 14, 2)
        assert (0, 6) in {tuple(map(int, pair)) for pair in full['edges']}
    receipt = dict(native_engine='original_simd', actual_native_fixture_passed=True,
                   source_files=[record(p) for p in sorted(NATIVE.glob('*.cpp'))] +
                                [record(NATIVE / 'MurmurHash3.h')], binaries=binaries,
                   flags=compile_flags, python=sys.version.split()[0])
    (ROOT / 'build').mkdir(exist_ok=True)
    (ROOT / 'build/native_qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Native SIMD build and duplicate fixture passed.')


if __name__ == '__main__':
    main()
