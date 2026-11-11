#!/usr/bin/env python3
"""Build the vendored RAD Faiss and in-batch SIMD extension for this checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--skip-faiss', action='store_true')
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error('--jobs must be positive')
    import pybind11
    flags = set(Path('/proc/cpuinfo').read_text().split())
    if 'avx2' not in flags:
        raise SystemExit('The supplied in-batch SIMD extension requires an x86-64 CPU with AVX2.')
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    source = ROOT / 'external/simd'
    module = 'simd_fuzzy_deduplicationv7_unsorted_multi7A_V1'
    command = [os.environ.get('CXX', 'g++'), '-O3', '-std=c++17', '-shared', '-fPIC',
               '-fopenmp', '-mavx2', '-mpopcnt', '-I' + sysconfig.get_path('include'),
               '-I' + pybind11.get_include(), str(source / (module + '.cpp')),
               str(source / 'MurmurHash3.cpp'), '-o', str(source / (module + suffix))]
    print('Building native in-batch SIMD', flush=True)
    subprocess.run(command, check=True)
    if not args.skip_faiss:
        environment = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs))
        environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment.get('PATH', '')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps',
            str(ROOT / 'external/faiss'), '--config-settings=build-dir=' + str(ROOT / 'build/faiss'),
            '--config-settings=cmake.define.FAISS_OPT_LEVEL=generic',
            '--config-settings=cmake.define.FAISS_USE_LTO=OFF',
            '--config-settings=cmake.define.BUILD_TESTING=OFF'], check=True, env=environment)
    sys.path[:0] = [str(ROOT / 'src'), str(source)]
    from radv2.bitmap_native import prepare
    from radv2.routing import native_library
    from radv2.modes import avx512_vpopcnt_available
    prepare(); native_library()
    import faiss
    assert hasattr(faiss, 'METRIC_MinhashBitmapCacheSIMD'), 'Wrong Faiss: RAD metric missing'
    record = dict(faiss_module=faiss.__file__, faiss_opt_level='generic',
                  cache_simd_cpu_supported=avx512_vpopcnt_available(),
                  simd_compile_flags=['-mavx2', '-mpopcnt'])
    (ROOT / 'build').mkdir(exist_ok=True)
    (ROOT / 'build/native_setup.json').write_text(json.dumps(record, indent=2) + '\n')
    print('RAD native build complete.', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
