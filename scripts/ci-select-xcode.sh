#!/usr/bin/env bash
# Pick an Xcode that can resolve Vendor/demucs-mlx-swift (swift-tools-version: 6.2).
# GitHub macos-15 images default /Applications/Xcode.app to 16.4 / Swift 6.1.
set -euo pipefail

candidates=(
  /Applications/Xcode_26.3.app
  /Applications/Xcode_26.2.app
  /Applications/Xcode_26.1.1.app
  /Applications/Xcode_26.1.app
  /Applications/Xcode_26.app
)

selected=""
for candidate in "${candidates[@]}"; do
  if [[ -d "$candidate" ]]; then
    selected="$candidate"
    break
  fi
done

if [[ -z "$selected" ]]; then
  echo "error: no Xcode 26.x install found (need Swift 6.2 for demucs-mlx-swift)" >&2
  ls -d /Applications/Xcode*.app 2>/dev/null || true
  xcodebuild -version || true
  swift --version || true
  exit 1
fi

sudo xcode-select -s "$selected"
echo "Using $selected"
xcodebuild -version
swift --version
