#!/usr/bin/env python3
"""Print this Linux VM's resources and check CPU flags before a native build."""
import argparse
import os
from pathlib import Path
import platform
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=30, help='Required available logical CPUs; default 30.')
    parser.add_argument('--system', choices=('rad', 'milvus', 'simd_dpk', 'faiss_jaccard'))
    parser.add_argument('--distance-mode', choices=('cache_scalar', 'cache_simd'), default='cache_scalar')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    info = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    sets = [set(line.split(':', 1)[1].split()) for line in info.splitlines()
            if line.startswith('flags') and ':' in line]
    flags = set.intersection(*sets) if sets else set()
    base = {'avx2', 'popcnt'}
    requirements = {
        'rad (cache_scalar)': base,
        'rad (cache_simd)': base | {'avx512f', 'avx512_vpopcntdq'},
        'milvus': base | {'avx512f', 'avx512dq', 'avx512vl', 'avx512bw'},
        'simd_dpk': base,
        'faiss_jaccard': base | {'avx512f', 'avx512cd', 'avx512dq', 'avx512bw',
                                 'avx512vl', 'f16c', 'fma', 'bmi2'},
    }
    print('Architecture:', platform.machine())
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 0)
    print('Available logical CPUs:', available)
    print('Requested runtime workers:', args.threads)
    budget_ok = available >= args.threads
    print('CPU budget:', 'OK' if budget_ok else f'INSUFFICIENT; need {args.threads} available logical CPUs')
    memory = {}
    if Path('/proc/meminfo').exists():
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith(('MemTotal:', 'MemAvailable:'))}
    for key, value in memory.items():
        print(key + ' GiB:', round(value / 2**20, 1))
    print('Free disk here GiB:', round(shutil.disk_usage('.').free / 2**30, 1))
    print('Python:', platform.python_version())
    selected = ('rad (' + args.distance_mode + ')') if args.system == 'rad' else args.system
    failed = False
    for name, required in requirements.items():
        if selected and name != selected:
            continue
        missing = sorted(required - flags)
        okay = platform.machine() == 'x86_64' and not missing
        print(name + ': ' + ('CPU OK' if okay else 'CPU UNSUPPORTED; missing ' + ', '.join(missing)))
        failed |= not okay
    print('RAM/disk starting sizes: MACHINE_SPECS.md. No build, search or writes performed.')
    if selected and (failed or not budget_ok):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
