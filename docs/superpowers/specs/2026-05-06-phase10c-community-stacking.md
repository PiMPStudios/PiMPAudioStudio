# Phase 10c: Community Sharing + Multi-Adapter Stacking — Design Spec

## Overview

Builds on Phase 10a/10b LoRA foundation to add community adapter discovery/download
and multi-adapter stacking at inference time.

**Scope:** Community adapter sharing (hybrid distribution) + multi-adapter stacking
(Inject-All). Adapter versioning, rating systems, and per-layer strength control
deferred to future phases.

## Goals

1. Enable users to discover and download community-shared adapters via a curated
   manifest hosted on GitHub, with actual weights on HuggingFace
2. Allow importing any adapter from a HuggingFace repo URL (open side of hybrid)
3. Support applying 2-3 LoRA adapters simultaneously with per-adapter strength
   sliders (Inject-All architecture)
4. Provide a smooth, App-Store-like download experience with progress and retry

## Architecture

### Distribution Model: Hybrid

Two paths to get community adapters:

- **Curated manifest** — A `manifest.json` hosted in a public GitHub repo
  (`PiMPStudios/anvil-adapters`) lists featured/verified adapters. The app fetches
  this on demand. Actual weights are hosted on HuggingFace.
- **Open HF import** — Users paste any HuggingFace repo URL that follows the Anvil
  adapter convention (`adapter.safetensors` + `adapter.json` at repo root). The app
  auto-resolves the files, validates metadata, and downloads.

### Multi-Adapter Stacking: Inject-All

True stacking via additive contributions in the forward pass:

```text
output = base(x) + strength_1 * loraA(x) + strength_2 * loraB(x) + ...
```

Each adapter maintains its own low-rank weight matrices. Strengths are adjustable
in real-time without recomputation. Capped at 3 simultaneous adapters (1 primary +
2 additional).

Why Inject-All over Merge-on-Apply:

- Real-time strength slider changes with zero recomputation delay
- Architecturally clean — extends existing LoRALinear naturally
- Low memory cost — LoRA weights are tiny at rank 8-16
- Easy add/remove at runtime without touching base weights

### Training Constraint

Training mode (`isTraining = true`) only supports one contribution. If the user
starts training while a stack is active, the stack is temporarily cleared (with a
toast notification) and restored after training completes.

## Data Types

### CommunityAdapter

```swift
public struct CommunityAdapter: Codable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let author: String
    public let description: String
    public let baseModelCategory: ModelCategory
    public let rank: Int
    public let adapterURL: URL
    public let metadataURL: URL
    public let tags: [String]
    public let featured: Bool
    public let minAppVersion: String
    public let createdAt: Date
    public let downloadCount: Int
}
```

### AdapterManifest

```swift
public struct AdapterManifest: Codable, Sendable {
    public let version: Int
    public let adapters: [CommunityAdapter]
}
```

### AdapterSource

```swift
public enum AdapterSource: String, Codable, Sendable, Hashable {
    case local       // trained on-device
    case community   // from curated manifest
    case huggingface // imported via URL
}
```

### LoRAAdapter (extended)

```swift
public struct LoRAAdapter: Codable, Sendable, Identifiable, Hashable {
    // ... existing fields from 10a/10b ...
    public var source: AdapterSource       // new (defaults to .local for backward compat)
    public var remoteRepoURL: URL?         // new (for HF-sourced adapters)
}
```

**Backward compatibility:** Existing adapters persisted without a `source` field
decode as `.local` (via `decodeIfPresent` with default). No migration needed.

### LoRAContribution

```swift
public struct LoRAContribution: Sendable {
    public let adapterID: String
    public let displayName: String?
    public let source: AdapterSource
    public let loraA: MLXArray   // [rank, inputDim]
    public let loraB: MLXArray   // [outputDim, rank]
    public var strength: Float
}
```

## Service Layer

### CommunityService

```swift
@MainActor
final class CommunityService: ObservableObject {
    @Published var manifest: AdapterManifest?
    @Published var downloads: [String: DownloadState] = [:]
    @Published var lastRefresh: Date?
    @Published var isRefreshing: Bool = false

    enum DownloadState {
        case downloading(progress: Double)
        case failed(Error)
        case completed
    }

    private let manifestURL = URL(string:
        "https://raw.githubusercontent.com/PiMPStudios/anvil-adapters/main/manifest.json")!
    private let cacheTTL: TimeInterval = 3600  // 1 hour
    private var activeTasks: [String: Task<Void, Never>] = [:]

    /// Fetch manifest (respects cache TTL unless force = true)
    func refreshManifest(force: Bool = false) async throws

    /// Download adapter weights + metadata, save to local registry
    func download(adapter: CommunityAdapter) async throws

    /// Cancel an in-progress download, clean up partial files
    func cancelDownload(adapterID: String)

    /// Whether an adapter is already installed locally
    func isInstalled(_ adapterID: String) -> Bool

    /// Import from a HuggingFace repo URL (not from manifest)
    func importFromHuggingFace(repoURL: URL) async throws -> LoRAAdapter
}
```

**Download behavior:**

- Inline circular progress indicator per row (App Store pattern)
- Max 3 concurrent downloads
- One auto-retry on transient failure, then manual "Retry" button
- Manifest refreshes on tab switch to Community, cached for 1 hour
- Manual refresh button with "Last updated: X min ago" timestamp

### AdapterStackManager

```swift
@MainActor
final class AdapterStackManager: ObservableObject {
    @Published var primaryAdapter: LoRAAdapter?
    @Published var additionalAdapters: [(adapter: LoRAAdapter, strength: Float)] = []

    static let maxStackSize = 3  // primary + 2 additional

    /// All active adapters (primary at strength 1.0 + additionals)
    var allContributions: [LoRAContribution] { ... }

    /// Add an adapter to the stack (fails silently if at cap)
    func addToStack(_ adapter: LoRAAdapter, strength: Float = 0.5)

    /// Remove from stack by ID
    func removeFromStack(adapterID: String)

    /// Update strength for a stacked adapter
    func setStrength(adapterID: String, strength: Float)

    /// Apply current stack to the pipeline (calls LoRAInjector)
    func applyStack(to pipeline: DiffusionPipeline) throws

    /// Clear all stacked adapters (revert to primary only or none)
    func clearStack()
}
```

## LoRALinear + LoRAInjector Evolution

### LoRALinear (multi-contribution forward pass)

```swift
public class LoRALinear: Linear {
    public var contributions: [LoRAContribution] = []
    public var isTraining: Bool = false

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = super.callAsFunction(x)

        for contribution in contributions where contribution.strength != 0.0 {
            let loraOut = x.matmul(contribution.loraA.T).matmul(contribution.loraB.T)
            out = out + MLXArray(contribution.strength) * loraOut
        }

        return out
    }
}
```

Contributions with `strength == 0.0` are skipped (no unnecessary matmuls).

### LoRAInjector stacking methods

```swift
extension LoRAInjector {
    /// Apply a full adapter stack to the model. Replaces all current contributions.
    static func applyStack(
        model: DiTDecoder,
        contributions: [LoRAContribution],
        paths: [String]
    )

    /// Remove all contributions (revert to base model behavior)
    static func clearContributions(model: DiTDecoder, paths: [String])

    /// Update strength for a single adapter across all layers
    static func setStrength(
        model: DiTDecoder,
        adapterID: String,
        strength: Float,
        paths: [String]
    )
}
```

**Migration from 10a/10b:** Existing `inject(model:weights:paths:)` still works —
creates a single LoRAContribution with strength 1.0. `extractWeights` still works
for training (extracts from contributions[0]). Stacking is inference-only.

## UI Components

### AdaptersSettingsView (evolved)

Segmented control: "My Adapters | Community"

- **My Adapters tab** — Existing 10b content (list, rename, delete, export, merge)
- **Community tab** — CommunityBrowserView

### CommunityBrowserView (new)

- Grid/list of community adapters from manifest
- Filter bar: category picker + search field
- Row shows: name, author, tags, rank badge, category badge
- Action button per row: "Get" -> circular progress -> "Installed" checkmark
- "Add from HuggingFace URL" button in toolbar
- Pull-to-refresh + "Last updated: X min ago" footer
- Version-gated adapters show "Requires Anvil {version}+" (Get button disabled)

### HuggingFaceImportSheet (new)

Progressive disclosure flow:

1. Text field for repo URL + "Fetch" button
2. On fetch: show adapter info preview (name, rank, category, estimated size)
3. If category mismatch: yellow warning banner + "Download anyway?" option
4. "Download" button -> progress -> dismiss on success

### AdapterStackView (new, in generation view)

Placed below the existing primary AdapterPicker:

- Section header: "Additional Adapters"
- For each stacked adapter: name label, strength slider (0.0-1.0), remove button
- "+ Add adapter" button (disabled at maxStackSize with tooltip)
- "Total strength: X.XX" label for power users
- Opens picker sheet filtered to compatible adapters (same baseModelCategory)

### File Organization

| File | Location |
|------|----------|
| `CommunityBrowserView.swift` | `AnvilApp/UI/Settings/` |
| `HuggingFaceImportSheet.swift` | `AnvilApp/UI/Settings/` |
| `AdapterStackView.swift` | `AnvilApp/UI/Generation/` |

## Data Flows

### Community Browse + Download

```text
User switches to Community tab
  -> CommunityService.refreshManifest() (if cache expired)
  -> Display adapter list with filters
  -> User taps "Get"
    -> CommunityService.download(adapter:)
      -> Fetch adapter.json (validation)
      -> Fetch adapter.safetensors (with progress)
      -> Save to AdapterRegistry with source: .community
    -> Row shows circular progress -> "Installed" checkmark
  -> Adapter appears in "My Adapters" tab + AdapterPicker
```

### HuggingFace Import

```text
User taps "Add from HuggingFace URL"
  -> HuggingFaceImportSheet presented
  -> User pastes repo URL, taps "Fetch"
    -> CommunityService resolves {repo}/resolve/main/adapter.json
    -> Show preview: name, rank, category, size
    -> If category mismatch: show warning banner
  -> User taps "Download"
    -> Fetch adapter.safetensors with progress
    -> Save with source: .huggingface, remoteRepoURL: original URL
  -> Sheet dismisses, adapter available in picker
```

### Multi-Adapter Stacking

```text
User selects primary adapter (existing AdapterPicker)
  -> AdapterStackManager.primaryAdapter set
User taps "+ Add adapter" in AdapterStackView
  -> Picker shows compatible adapters (same category, excludes already stacked)
  -> User picks adapter, added at strength 0.5
  -> AdapterStackManager.addToStack()
User adjusts strength slider
  -> AdapterStackManager.setStrength()
  -> LoRAInjector.setStrength() updates all LoRALinear layers in real-time
User generates
  -> Pipeline uses current stack: base + sum(strength_i * lora_i(x))
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| Manifest fetch fails (network) | Show cached version if available + "Offline — showing cached results" banner. If no cache, empty state with retry button |
| Download fails mid-transfer | Auto-retry once. On second failure: clean up partial file, show "Failed" badge + retry button |
| HF repo missing adapter.json | Error: "This repo doesn't follow the Anvil adapter format. Expected adapter.json at repo root." |
| HF repo missing adapter.safetensors | Same pattern — clear error naming the missing file |
| Category mismatch on import | Yellow warning: "This adapter is for {X}, but you have {Y} installed. Download anyway?" |
| Stack at max capacity | "+ Add" button disabled with tooltip: "Maximum 3 adapters active" |
| Remove primary while stacked | Promote first additional to primary, or clear stack if none remain |
| Training starts with active stack | Clear stack with toast: "Stack cleared during training. Will restore when complete." Restore on training end |
| Adapter deleted while in stack | Remove from stack, show brief toast: "{name} removed from stack (adapter deleted)" |
| Malformed manifest JSON | Skip malformed entries, log warning, show valid entries |
| App version < minAppVersion | Show adapter but disable "Get" with note: "Requires Anvil {version}+" |
| Downloaded file corrupted | Validate safetensors after download, delete + show error if invalid |
| Duplicate HF import | Detect by remoteRepoURL match, offer "Already installed. Re-download?" |

## Edge Cases

- `strength == 0.0` — skip contribution in forward pass (no matmul)
- All strengths at 0.0 — equivalent to base model, no adapter effect
- Duplicate adapter IDs in manifest — last entry wins
- Manifest has 0 adapters — "No community adapters available yet" empty state
- Download completes but safetensors invalid — validate, delete, show error
- User imports same HF adapter twice — detect via remoteRepoURL, prompt to re-download

## Performance Considerations

- Skip contributions with `strength == 0.0` in forward pass
- Lazy-load contribution weights: only load safetensors into memory when adapter is
  added to stack
- Unload weights when adapter is removed from stack (not just on app close)
- Manifest response is small (JSON) — no special caching beyond TTL

## Testing Strategy

### Unit-Testable (pure logic, no Metal)

1. **CommunityAdapter/AdapterManifest** — Codable round-trip, date parsing,
   category mapping
2. **CommunityService** — Manifest parsing, download state machine, cache TTL
   logic, duplicate detection (with mock URLSession)
3. **AdapterStackManager** — Add/remove/cap enforcement, strength updates,
   allContributions computation, promote-on-remove logic
4. **AdapterSource** — Codable round-trip, equality

### Unit-Testable (requires Metal)

5. **LoRALinear multi-contribution** — Additive math: verify
   `base + s1*A + s2*B` for known arrays
6. **Strength == 0 optimization** — Verify skipped (no computation)

### Build-Verified (Xcode build, no unit test)

- LoRAInjector stacking methods (applyStack, clearContributions, setStrength)
- CommunityBrowserView row states (Get/progress/installed), filter, search
- AdapterStackView slider binding, add/remove, cap disable, total strength
- HuggingFaceImportSheet URL validation, warning display, progressive flow
- AdaptersSettingsView segmented control switching

### Manual Verification

- Browse community adapters, download one, verify it appears in picker
- Import adapter from HuggingFace URL, verify provenance badge
- Stack 2 adapters with different strengths, generate, hear blended result
- Adjust strength slider in real-time, re-generate, hear difference
- Hit max stack cap, verify "+ Add" disabled
- Start training with active stack, verify toast + stack restoration
- Test offline: disable network, verify cached manifest shown
- Test failed download, verify auto-retry + manual retry flow

## File Manifest

### New Files (AnvilCore)

| File | Responsibility |
|------|---------------|
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAContribution.swift` | Contribution data type |
| `AnvilCore/Sources/AnvilCore/Community/CommunityAdapter.swift` | CommunityAdapter + AdapterManifest types |
| `AnvilCore/Sources/AnvilCore/Community/AdapterSource.swift` | AdapterSource enum |

### New Files (AnvilApp)

| File | Responsibility |
|------|---------------|
| `AnvilApp/Services/CommunityService.swift` | Manifest fetch, downloads, HF import |
| `AnvilApp/Services/AdapterStackManager.swift` | Runtime stack state + pipeline application |
| `AnvilApp/UI/Settings/CommunityBrowserView.swift` | Community adapter browsing + download UI |
| `AnvilApp/UI/Settings/HuggingFaceImportSheet.swift` | HF URL import modal |
| `AnvilApp/UI/Generation/AdapterStackView.swift` | Stacking controls with strength sliders |

### Modified Files

| File | Changes |
|------|---------|
| `AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift` | Replace single loraA/loraB with contributions array, multi-contribution forward pass |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift` | Add applyStack, clearContributions, setStrength methods |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift` | Add AdapterSource to LoRAAdapter, add remoteRepoURL field |
| `AnvilApp/UI/Settings/AdaptersSettingsView.swift` | Segmented control (My Adapters / Community), integrate CommunityBrowserView |
| `AnvilApp/UI/Generation/GenerateView.swift` | Add AdapterStackView below AdapterPicker |
| `AnvilApp/Services/TrainingService.swift` | Clear/restore stack on training start/end, toast notification |

## Out of Scope (deferred)

- Adapter versioning / update checking
- Rating/review system for community adapters
- Per-layer strength control (all layers get same strength per adapter)
- Adapter animation (morph between strengths over time)
- Upload flow (publishing adapters to community from within the app)
- Custom manifest URLs (only PiMPStudios/anvil-adapters for now)
