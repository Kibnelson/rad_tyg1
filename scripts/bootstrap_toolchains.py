#!/usr/bin/env python3
"""Install the build's pinned toolchains below .build, without changing profiles."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / '.build'
GO = '1.26.9'
RUST = '1.92.0'


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response, target.open('wb') as stream:
        shutil.copyfileobj(response, stream)


def main():
    BUILD.mkdir(exist_ok=True)
    with urllib.request.urlopen('https://go.dev/dl/?mode=json&include=all', timeout=90) as response:
        versions = json.load(response)
    release = next(v for v in versions if v['version'] == 'go' + GO)
    info = next(v for v in release['files'] if v['os'] == 'linux' and v['arch'] == 'amd64' and v['kind'] == 'archive')
    archive = BUILD / info['filename']
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != info['sha256']:
        fetch('https://go.dev/dl/' + info['filename'], archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == info['sha256'], 'Go archive hash differs'
    if not (BUILD / 'go/bin/go').exists():
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                assert not member.name.startswith('/') and '..' not in Path(member.name).parts
            bundle.extractall(BUILD)
    env = dict(os.environ, CARGO_HOME=str(BUILD / 'cargo'), RUSTUP_HOME=str(BUILD / 'rustup'))
    rustup = BUILD / 'cargo/bin/rustup'
    if not rustup.exists():
        url = 'https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init'
        binary, checksum = BUILD / 'rustup-init', BUILD / 'rustup-init.sha256'
        fetch(url, binary); fetch(url + '.sha256', checksum)
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == checksum.read_text().split()[0]
        binary.chmod(0o755)
        subprocess.run([str(binary), '-y', '--no-modify-path', '--profile', 'minimal', '--default-toolchain', 'none'], env=env, check=True)
    # Milvus' included Tantivy CMake target explicitly invokes cargo +1.89.
    for toolchain in (RUST, '1.89'):
        subprocess.run([str(rustup), 'toolchain', 'install', toolchain, '--profile', 'minimal'], env=env, check=True)
    entries = {'CARGO_HOME': str(BUILD / 'cargo'), 'RUSTUP_HOME': str(BUILD / 'rustup'),
               'RUSTUP_TOOLCHAIN': RUST, 'CONAN_HOME': str(BUILD / 'conan'),
               'GOPATH': str(BUILD / 'gopath'), 'GOCACHE': str(BUILD / 'gocache'),
               'GOMAXPROCS': os.environ.get('JOBS', '2'),
               'CARGO_BUILD_JOBS': os.environ.get('JOBS', '2'),
               'CONAN_CPU_COUNT': os.environ.get('JOBS', '2'),
               'CMAKE_BUILD_PARALLEL_LEVEL': os.environ.get('JOBS', '2')}
    # Conan 2 reads tools.build:jobs, rather than the legacy CPU-count variable.
    conan_home=BUILD/'conan';conan_home.mkdir(exist_ok=True)
    (conan_home/'global.conf').write_text('tools.build:jobs='+os.environ.get('JOBS','2')+'\n')
    path = ':'.join([str(ROOT / '.venv/bin'), str(BUILD / 'go/bin'), str(BUILD / 'cargo/bin'), os.environ['PATH']])
    text = '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in entries.items())
    (BUILD / 'environment.sh').write_text(text + '\nexport PATH=' + shlex.quote(path) + '\n')
    print('Pinned toolchains ready: Go', GO, 'Rust', RUST)


if __name__ == '__main__':
    main()
