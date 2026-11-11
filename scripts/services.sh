#!/usr/bin/env bash
# Own only this checkout's etcd/MinIO containers. Preserve volumes on shutdown.
set -euo pipefail
PUBLIC_MILVUS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC_MILVUS_PROJECT="rad-tyg1-$(printf '%s' "$PUBLIC_MILVUS_ROOT" | sha256sum | cut -c1-12)"
PUBLIC_MILVUS_DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then PUBLIC_MILVUS_DOCKER=(sudo docker); fi
cd "$PUBLIC_MILVUS_ROOT"
case "${1:-}" in
  up)
    mkdir -p .services/etcd .services/minio
    "${PUBLIC_MILVUS_DOCKER[@]}" compose --project-name "$PUBLIC_MILVUS_PROJECT" up -d --wait
    ;;
  down)
    "${PUBLIC_MILVUS_DOCKER[@]}" compose --project-name "$PUBLIC_MILVUS_PROJECT" down
    ;;
  *) printf '%s\n' 'Use: bash scripts/services.sh up|down' >&2; exit 2 ;;
esac
