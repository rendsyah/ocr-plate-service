#!/bin/sh
set -e

chown -R nonroot:nonroot /app/.paddlex /app/ultralytics /app/matplotlib /app/storage 2>/dev/null || true

exec setpriv --reuid=nonroot --regid=nonroot --init-groups "$@"
