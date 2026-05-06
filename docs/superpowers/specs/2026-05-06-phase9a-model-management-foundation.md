# Phase 9a: Model Management Foundation — Design Spec

## Overview

Harden the model download pipeline with SHA256 verification, redesign the
first-launch experience for faster time-to-music, and prepare the registry
architecture for Phase 9b's model swapping and community model support.

**Scope:** SHA256 enforcement with auto-retry, priority-based first-launch
flow (ACE-Step only blocking, SFX/LLM background), registry split into
static built-in specs and runtime user specs with JSON persistence.

**Key improvement:** First-launch blocking download drops from ~7.6 GB
(all 4 models) to ~4 GB (ACE-Step only). Users generate music while
secondary models download in the background.

---

## SHA256 Verification

### Checksums

Populate the currently-empty `sha256` dictionaries in all 4 built-in
`ModelSpec` entries with real SHA256 hashes. Each file in a spec gets its
own hash entry:

```swift
ModelSpec(
    id: "ace-step-dit",
    huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
    files: ["model.safetensors", "config.json"],
    sizeBytes: 3_200_000_000,
    sha256: [
        "model.safetensors": "<actual-hash>",
        "config.json": "<actual-hash>"
    ]
)
```

Actual hash values computed during implementation by downloading each file
and running `SHA256.hash(data:)` via CryptoKit.

### Verification Flow

1. File downloads to temp location via `URLSession.download`
2. `DownloadClient.verifySHA256(file:expected:)` runs (already exists)
3. **Pass** — move file to final destination
4. **Fail** — delete temp file, show status "Download verification failed.
   Retrying...", re-download the same file once
5. **Second fail** — report `ModelStatus.failed("Model verification failed.
   The file may be corrupted or tampered with.")`, do not move file

### Empty Hash Handling

If a `ModelSpec` has an empty `sha256` dictionary (user-added models from
9b will not have checksums), skip verification and accept the file. This
keeps the system forward-compatible without requiring checksums for
community models.

---

## First-Launch Redesign

### Current Behavior

`FirstLaunchView` shows all 4 models, downloads sequentially, blocks until
all complete (~7.6 GB). `AppState.isSetupComplete` requires all files
present.

### New Two-Phase Behavior

**Phase 1 — Essential downloads (blocking):**

- Download `ace-step-dit` (~3.2 GB) then `ace-step-vae` (~800 MB)
  sequentially with verification after each file
- `FirstLaunchView` shows progress for these two only
- Updated copy: "Downloading core models (~4 GB)" instead of "~7.6 GB"
- On completion: `isSetupComplete = true`, transition to `MainShellView`

**Phase 2 — Background downloads (non-blocking):**

- After user enters the app, `AppState.startBackgroundDownloads()` kicks
  off sequential background downloads: Stable Audio first (more
  immediately useful), then LLM
- SFX download delegated to existing `SFXModelManager.downloadIfNeeded()`
- LLM download delegated to existing `LLMModelManager`
- Progress shown via subtle indicator in main UI (see Background Download
  Indicator below)
- Errors are non-fatal: shown as dismissable notification, user can retry
  from settings

### Setup Gate Change

`AppState.checkSetupStatus()` changes from checking all 4 models to
checking only `ace-step-dit` and `ace-step-vae`. All other models are
optional at launch.

### Feature Gating

- **SFX generation:** `SFXModelManager` already shows download prompt when
  model is not ready. No change needed.
- **Lyric writing:** `LLMModelManager` already handles not-ready state. No
  change needed.
- Both features unlock automatically when their background download
  completes.

---

## Registry Architecture

### Current State

`ModelRegistry` is an actor with `nonisolated let allSpecs: [ModelSpec]`
containing 4 hardcoded models. `ModelSpec` is `Sendable` and
`Identifiable` but not `Codable`.

### Changes

**Built-in specs become a static constant:**

```swift
static let builtInSpecs: [ModelSpec] = [
    // ace-step-dit, ace-step-vae, stable-audio-open, lyric-llm
    // With populated sha256 dictionaries
]
```

**Runtime registry for user-added models:**

```swift
private(set) var userSpecs: [ModelSpec] = []
```

Persisted to `~/Library/Application Support/Anvil/models/user-models.json`
as a JSON array. Loaded on init, saved on mutation.

**Unified access (computed property, replaces the old `let`):**

```swift
var allSpecs: [ModelSpec] {
    Self.builtInSpecs + userSpecs
}
```

**Extension points for 9b:**

```swift
func addModel(_ spec: ModelSpec) throws
func removeModel(id: String) throws
func isBuiltIn(id: String) -> Bool
```

These exist and work in 9a but nothing in the UI calls them yet.

**ModelSpec gains Codable conformance.** All fields (`id`,
`huggingFaceRepo`, `files`, `sizeBytes`, `sha256`) are simple types that
encode naturally.

**Graceful degradation on load:** If `user-models.json` is corrupt or
unreadable, log the error and start with an empty `userSpecs` array. Do
not fail app launch.

---

## Background Download Indicator

### State

New published property on `AppState`:

```swift
enum BackgroundDownloadPhase: Equatable {
    case idle
    case downloading(modelName: String, progress: Float)
    case complete
    case failed(String)
}

@Published var backgroundDownloadPhase: BackgroundDownloadPhase = .idle
```

### UI

Small progress pill or status bar line in the main UI:

- `.idle` — hidden
- `.downloading` — "Downloading Stable Audio Open... 64%"
- `.complete` — brief "All models ready" confirmation, auto-dismiss
  after 3 seconds
- `.failed` — "Download failed. Tap to retry." dismissable

### Coordination

`AppState.startBackgroundDownloads()`:

1. Check if Stable Audio model is present → if not, set phase to
   `.downloading(modelName: "Stable Audio Open", progress: 0)`, trigger
   `SFXModelManager.downloadIfNeeded()`, observe progress
2. After SFX completes (or if already present), check LLM → if not
   present, set phase to `.downloading(modelName: "Lyric Writer",
   progress: 0)`, trigger `LLMModelManager` download
3. After both complete, set phase to `.complete`
4. On any error, set phase to `.failed(message)` — non-fatal, user can
   retry

---

## File Structure

### Modified Files

| File | Changes |
|------|---------|
| `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift` | Add `Codable`, populate SHA256 checksums |
| `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` | Static `builtInSpecs`, runtime `userSpecs`, `addModel`/`removeModel`/`isBuiltIn`, JSON persistence, graceful load |
| `AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift` | Enforce verification after download, auto-retry on failure |
| `AnvilApp/AppState.swift` | `checkSetupStatus` ACE-Step only, `startBackgroundDownloads()`, `BackgroundDownloadPhase` |
| `AnvilApp/UI/FirstLaunchView.swift` | Show only ACE-Step models, updated copy and download size |

### New Files

| File | Purpose |
|------|---------|
| `AnvilApp/UI/BackgroundDownloadIndicator.swift` | Small progress pill for background download status |

### Unchanged Files

| File | Why |
|------|-----|
| `AnvilApp/Services/SFXModelManager.swift` | Already handles deferred download and feature gating |
| `AnvilApp/Services/LLMModelManager.swift` | Already handles not-ready state |
| `AnvilApp/Services/GenerationService.swift` | No changes needed |
| All pipeline actors | No changes needed |
| All DSP/Editor UI | No changes needed |

---

## Scope

### In Scope (Phase 9a)

- SHA256 checksums populated for all 4 built-in models
- Download verification enforced with one auto-retry on failure
- Empty-hash bypass for future user-added models
- First-launch redesign: ACE-Step only blocking, SFX/LLM background
- `AppState.isSetupComplete` gates only on ACE-Step models
- Background download coordination with progress indicator
- `ModelSpec` gains `Codable` conformance
- `ModelRegistry` split: static `builtInSpecs` + runtime `userSpecs`
- JSON persistence for user-added model specs
- Registry extension points: `addModel`, `removeModel`, `isBuiltIn`
- Graceful degradation: corrupt JSON → empty array, background errors →
  dismissable notification
- User-friendly error messages for verification failures

### Out of Scope (Phase 9b)

- Community model browsing/discovery UI
- Model swapping for diffusion pipelines (ACE-Step / Stable Audio)
- Hot-swap runtime (unloading/reloading weights)
- HuggingFace repo validation for user-added models
- Model deletion UI
- Settings page for model management
