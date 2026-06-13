#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v pimp-remote-worker >/dev/null 2>&1; then
  echo "[PiMP] pimp-remote-worker is not installed; running installer."
  bash "$SCRIPT_DIR/install.sh"
fi

if [ "$#" -lt 1 ]; then
  cat <<'USAGE'
Usage:
  bash bootstrap_remote_worker.sh /workspace/jobs/job.zip [worker args]
  bash bootstrap_remote_worker.sh /workspace/jobs/extracted-package [worker args]

Examples:
  bash bootstrap_remote_worker.sh /workspace/jobs/darkvocals.zip --dry-run
  bash bootstrap_remote_worker.sh /workspace/jobs/darkvocals.zip
USAGE
  exit 2
fi

exec pimp-remote-worker train "$@"
