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

if [[ ${#FILTERS[@]} -eq 0 ]]; then
  echo "error: no PiMPAudioCoreTests filters listed (package resolve or swift test list failed)" >&2
  exit 1
fi

for filter in "${FILTERS[@]}"; do
  printf 'Running %s\n' "$filter"
  swift test --package-path "$PACKAGE_PATH" --filter "$filter"
done

printf 'All PiMPAudioCore serial test filters passed.\n'
