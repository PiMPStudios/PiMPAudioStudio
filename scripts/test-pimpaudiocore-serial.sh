#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_PATH="$ROOT_DIR/PiMPAudioCore"

FILTERS=()
while IFS= read -r filter; do
  FILTERS+=("$filter")
done < <(
  swift test --package-path "$PACKAGE_PATH" list \
    | awk -F'[./(]' '/^PiMPAudioCoreTests[.]/ { print $2 }' \
    | sort -u
)

for filter in "${FILTERS[@]}"; do
  printf 'Running %s\n' "$filter"
  swift test --package-path "$PACKAGE_PATH" --filter "$filter"
done

printf 'All PiMPAudioCore serial test filters passed.\n'
