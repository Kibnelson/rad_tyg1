#!/usr/bin/env python3
"""Compile the same in-batch SIMD implementation used by the benchmark."""
from pathlib import Path
import subprocess
import sysconfig
import pybind11

ROOT = Path(__file__).resolve().parents[1]
flags = Path('/proc/cpuinfo').read_text().split()
required = ('avx2', 'popcnt', 'avx512f', 'avx512dq', 'avx512vl', 'avx512bw')
assert all(f in flags for f in required), 'This SIMD implementation requires AVX512F/DQ/VL/BW, AVX2 and POPCNT'
module = 'simd_fuzzy_deduplicationv7_unsorted_multi7A_V1'
source = ROOT / 'external/simd'
target = source / (module + sysconfig.get_config_var('EXT_SUFFIX'))
subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp',
                *['-m' + f for f in required], '-I' + sysconfig.get_path('include'),
                '-I' + pybind11.get_include(), str(source / (module + '.cpp')),
                str(source / 'MurmurHash3.cpp'), '-o', str(target)], check=True)
print('SIMD built:', target.name)
