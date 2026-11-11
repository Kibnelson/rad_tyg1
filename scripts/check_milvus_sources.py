#!/usr/bin/env python3
"""Check owned Milvus source files and prepare the experiment config; no build/run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    if path.is_symlink():
        return hashlib.sha256(b'symlink\0'+os.readlink(path).encode()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    lock = json.loads((ROOT/'milvus_sources.lock.json').read_text())
    states = {}
    for name, source in lock['repositories'].items():
        path = ROOT/source['path']
        if not path.is_dir():
            raise ValueError(f'Missing owned source repository: {path}')
        if (path/'.git').exists():
            remotes = subprocess.check_output(['git','-C',str(path),'remote'],text=True).splitlines()
            if remotes:
                raise ValueError(f'Local source {name} still has remotes: {remotes}')
        for relative, expected in source['file_sha256'].items():
            if digest(path/relative) != expected:
                raise ValueError(f'Owned source changed: {name}/{relative}; review and refresh its source lock')
        states[name] = dict(path=str(path),upstream_revision=source['upstream_revision'],
                            protected_files=len(source['file_sha256']))
    for relative, expected in lock.get('project_file_sha256',{}).items():
        if digest(ROOT/relative) != expected:
            raise ValueError(f'Experiment contract changed: {relative}; review its source lock')
    return dict(status='source_ready',scope='source_integrity_only',repositories=states,
                lock_sha256=digest(ROOT/'milvus_sources.lock.json'))


def instance_ports(slot):
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot <= 100:
        raise ValueError('Instance slot must be an integer between 0 and 100')
    return {name: port + slot * 100 for name, port in {
        'proxy':19530, 'proxy_internal':19529, 'proxy_http':19528, 'rootCoord':22125,
        'queryCoord':19531, 'queryNode':21123, 'dataCoord':13333,
        'dataNode':21124, 'streamingNode':22222, 'metrics':9091}.items()}


def config(directory, instance_slot=None):
    directory = Path(directory).resolve()
    inputs = {'milvus.yaml':ROOT/'external/milvus/configs/milvus.yaml',
              'glog.conf':ROOT/'external/milvus/configs/glog.conf',
              'user.yaml':ROOT/'reference/milvus/physical_1m_user.yaml'}
    for name, source in inputs.items():
        if not source.is_file():
            raise ValueError(f'Required server config file missing: {name}: {source}')
    runtime = directory.parent/'server_data'
    namespace = 'radv2-'+hashlib.sha256(str(runtime).encode()).hexdigest()[:16]
    overlay = inputs['user.yaml'].read_text()+(
        '\n# Dedicated writable paths and metadata namespaces for this server.\n'
        f'localStorage:\n  path: {json.dumps(str(runtime/"local"))}\n'
        f'rocksmq:\n  path: {json.dumps(str(runtime/"rocksmq"))}\n'
        f'log:\n  file:\n    rootPath: {json.dumps(str(runtime/"logs"))}\n'
        f'etcd:\n  rootPath: {namespace}\n'
        f'minio:\n  rootPath: {namespace}\n')
    ports = None
    if instance_slot is not None:
        ports = instance_ports(instance_slot)
        overlay += (f'proxy:\n  port: {ports["proxy"]}\n'
                    f'  internalPort: {ports["proxy_internal"]}\n  http:\n'
                    # Pinned proxy registers its metrics console routes only
                    # when HTTP is enabled. Keep REST off the gRPC transport.
                    f'    enabled: true\n    enableV1: true\n    port: {ports["proxy_http"]}\n'
                    f'common:\n  MetricsPort: {ports["metrics"]}\n')
        # Extend the existing dataCoord mapping; a duplicate YAML key would
        # otherwise discard the compaction-off settings in our profile.
        if overlay.count('dataCoord:\n') != 1:
            raise ValueError('Expected exactly one dataCoord mapping in the owned profile')
        overlay = overlay.replace('dataCoord:\n',f'dataCoord:\n  port: {ports["dataCoord"]}\n',1)
        for name in ('rootCoord','queryCoord','queryNode','dataNode','streamingNode'):
            overlay += f'{name}:\n  port: {ports[name]}\n'
    directory.mkdir(parents=True,exist_ok=True)
    # Include the native logger's flag file as well as the YAML configuration.
    # Other YAML overlays have higher priority than the base file. A fresh
    # dedicated directory avoids inheriting another experiment's defaults.
    unknown = {p.name for p in directory.iterdir()} - set(inputs)
    if unknown:
        raise ValueError(f'Use a dedicated config directory; extra files: {sorted(unknown)}')
    for name, source in inputs.items():
        target = directory/name
        content = overlay.encode() if name == 'user.yaml' else source.read_bytes()
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f'Existing server config differs: {target}')
        if not target.exists():
            target.write_bytes(content)
    return dict(directory=str(directory),files={name:digest(directory/name) for name in inputs},
                physical_row_capacity=1_000_000,compaction=False,ports=ports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-config',type=Path)
    parser.add_argument('--instance-slot',type=int)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--refresh-reviewed-sources',action='store_true',
        help='After reviewing local edits, refresh inventoried hashes without fetching or checking out source')
    args = parser.parse_args()
    if args.refresh_reviewed_sources:
        path = ROOT/'milvus_sources.lock.json'
        lock = json.loads(path.read_text())
        for source in lock['repositories'].values():
            source['file_sha256'] = {p:digest(ROOT/source['path']/p) for p in source['file_sha256']}
        lock['project_file_sha256'] = {p:digest(ROOT/p) for p in lock['project_file_sha256']}
        path.write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
    result = check()
    if args.prepare_config:
        result['server_config'] = config(args.prepare_config,args.instance_slot)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
