# Build And Test

## Requirements

- macOS 15+
- Apple Silicon recommended for meaningful MLX runs
- Xcode 16+
- Network for first-time Swift package resolution and optional model downloads

## Open And Run

```bash
git clone https://github.com/PiMPStudios/PiMPAudioStudio.git
cd PiMPAudioStudio
open PiMPAudioStudio.xcodeproj
```

Select the **PiMPAudioStudio** scheme and run.

### XcodeGen

`project.yml` is the declarative project definition (bundle id, versions,
packages, entitlements). If you change it, regenerate with XcodeGen when that is
part of your workflow:

```bash
xcodegen generate
```

Do not hand-edit `project.pbxproj` for settings that belong in `project.yml`
unless you know the team workflow expects it.

## Dependencies

Declared in `project.yml` / the Xcode project:

| Package | Role |
| --- | --- |
| local `PiMPAudioCore` | Inference, dataset, LoRA, registry |
| `mlx-swift` | MLX / MLXNN |
| `swift-sdk` (MCP) | Official MCP Swift SDK |

Additional vendored code may live under `Vendor/` (for example Demucs MLX).

## Tests

### App tests

```text
PiMPAudioStudioTests/
```

Covers app-level pieces such as AppState, DSP, HistoryStore, ProviderAccess,
RunPod service helpers.

### Core tests

```text
PiMPAudioCore/Tests/PiMPAudioCoreTests/
```

Heavy coverage for inference, dataset packages, LoRA trainers, model registry,
native annotation packs, and more.

### Helper scripts

```bash
scripts/test-pimpaudiocore-serial.sh
scripts/smoke-pimpaudio-mcp.sh
```

MCP smoke requires the app running with MCP enabled on the default localhost
endpoint.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) on `main` and pull requests:

| Job | What it runs |
| --- | --- |
| Remote worker | `ruff check` with `remote-worker/ruff.toml` |
| PiMPAudioCore | `scripts/test-pimpaudiocore-serial.sh` (one suite at a time — MLX GPU-state races) |
| App tests | `xcodebuild test -scheme PiMPAudioStudio -only-testing:PiMPAudioStudioTests` |

SwiftFormat config lives at `.swiftformat` for local use; it is **not** a CI gate until the tree is formatted in a dedicated pass.

```bash
ruff check remote-worker --config remote-worker/ruff.toml
scripts/test-pimpaudiocore-serial.sh
xcodebuild test -scheme PiMPAudioStudio -destination 'platform=macOS' -only-testing:PiMPAudioStudioTests
```

## Practical Local Verification

Before claiming a change is done:

1. Build the PiMPAudioStudio scheme.
2. Run relevant unit tests (or the serial Core script when races matter).
3. Smoke the affected UI path once on device (generate, edit, train preflight,
   etc.).
4. If MCP surface changed, run the smoke script.
5. If packaging changed, run codesign/spctl checks from
   [Release And Packaging](release-packaging.md).

## Common Build Issues

- **Package resolution failures** — check network / SPM cache; reset package
  caches in Xcode if needed.
- **Sandbox test surprises** — app tests may need host app context; prefer Core
  package tests for pure logic.
- **MLX memory failures in tests** — skip or gate GPU-heavy cases on machines
  without enough unified memory.

## Coding Notes

- Prefer MainActor for UI-facing services; keep heavy tensor work off the UI
  thread.
- Do not introduce a Python dependency into the app target.
- Keep secrets out of logs, traces, and remote job packages.
- Update `CHANGELOG.md` for user-visible or packaging-relevant changes.
