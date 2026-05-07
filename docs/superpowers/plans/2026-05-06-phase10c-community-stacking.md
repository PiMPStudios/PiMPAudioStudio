# Phase 10c: Community Sharing + Multi-Adapter Stacking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable community adapter discovery/download and multi-adapter stacking with per-adapter strength sliders.

**Architecture:** Hybrid distribution (GitHub manifest + HF URL import) for community adapters. Inject-All stacking via LoRAContribution arrays in LoRALinear layers. CommunityService handles networking; AdapterStackManager owns runtime stack state.

**Tech Stack:** Swift, SwiftUI, MLX, URLSession, SwiftUI Charts (existing)

---

## File Structure

### New Files (AnvilCore — 3 files)

| File | Responsibility |
|------|---------------|
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAContribution.swift` | LoRAContribution struct for multi-adapter forward pass |
| `AnvilCore/Sources/AnvilCore/Community/AdapterSource.swift` | AdapterSource enum (.local, .community, .huggingface) |
| `AnvilCore/Sources/AnvilCore/Community/CommunityAdapter.swift` | CommunityAdapter + AdapterManifest Codable types |

### New Files (AnvilApp — 5 files)

| File | Responsibility |
|------|---------------|
| `AnvilApp/Services/CommunityService.swift` | Manifest fetch, download management, HF import |
| `AnvilApp/Services/AdapterStackManager.swift` | Runtime stack state, pipeline integration |
| `AnvilApp/UI/Settings/CommunityBrowserView.swift` | Community adapter browsing + download UI |
| `AnvilApp/UI/Settings/HuggingFaceImportSheet.swift` | HF URL import modal |
| `AnvilApp/UI/Generation/AdapterStackView.swift` | Stacking controls with strength sliders |

### Modified Files (6 files)

| File | Changes |
|------|---------|
| `AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift` | Add contributions array, multi-contribution forward pass |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift` | Add applyStack, clearContributions, setStrength methods |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift` | Add source + remoteRepoURL to LoRAAdapter |
| `AnvilApp/UI/Settings/AdaptersSettingsView.swift` | Segmented control (My Adapters / Community) |
| `AnvilApp/UI/Components/GenerateView.swift` | Add AdapterStackView below AdapterPicker |
| `AnvilApp/Services/TrainingService.swift` | Clear/restore stack on training start/end |

---

### Task 1: LoRAContribution Data Type

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/LoRA/LoRAContribution.swift`

- [ ] **Step 1: Create the LoRAContribution struct**

```swift
import Foundation
import MLX

/// Represents a single adapter's contribution in a multi-adapter stack.
///
/// Used by LoRALinear to compute additive low-rank contributions:
/// `output += strength * (x @ loraA^T @ loraB^T)`
public struct LoRAContribution: Sendable {
    /// ID of the adapter providing this contribution.
    public let adapterID: String

    /// Display name for UI and logging.
    public let displayName: String?

    /// Provenance of this adapter.
    public let source: AdapterSource

    /// Low-rank matrix A: shape [rank, inputDim]. Kaiming uniform init.
    public let loraA: MLXArray

    /// Low-rank matrix B: shape [outputDim, rank]. Zero or trained values.
    public let loraB: MLXArray

    /// Strength multiplier (0.0 = no effect, 1.0 = full effect).
    public var strength: Float

    public init(
        adapterID: String,
        displayName: String? = nil,
        source: AdapterSource = .local,
        loraA: MLXArray,
        loraB: MLXArray,
        strength: Float = 1.0
    ) {
        self.adapterID = adapterID
        self.displayName = displayName
        self.source = source
        self.loraA = loraA
        self.loraB = loraB
        self.strength = strength
    }
}
```

Note: This file depends on `AdapterSource` which is created in Task 2. The build will not pass until Task 2 is complete.

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build may fail due to missing `AdapterSource` — this is expected. Task 2 creates it. If both Task 1 and Task 2 are committed together, the build will pass.

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/LoRA/LoRAContribution.swift
git commit -m "feat(core): add LoRAContribution struct for multi-adapter stacking"
```

---

### Task 2: AdapterSource Enum

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Community/AdapterSource.swift`

- [ ] **Step 1: Create the Community directory and AdapterSource enum**

```bash
mkdir -p AnvilCore/Sources/AnvilCore/Community
```

```swift
import Foundation

/// Provenance of a LoRA adapter — where it came from.
public enum AdapterSource: String, Codable, Sendable, Hashable {
    /// Trained on-device by the user.
    case local

    /// Downloaded from the curated community manifest.
    case community

    /// Imported directly from a HuggingFace repo URL.
    case huggingface
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds (or only pre-existing errors). Both LoRAContribution and AdapterSource should compile together.

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Community/AdapterSource.swift
git commit -m "feat(core): add AdapterSource enum (local/community/huggingface)"
```

---

### Task 3: CommunityAdapter + AdapterManifest Types

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Community/CommunityAdapter.swift`

- [ ] **Step 1: Create CommunityAdapter and AdapterManifest structs**

```swift
import Foundation

/// A single adapter entry from the community manifest.
public struct CommunityAdapter: Codable, Sendable, Identifiable, Hashable {
    /// Unique identifier (directory name convention).
    public let id: String

    /// User-facing display name.
    public let name: String

    /// Author or organization name.
    public let author: String

    /// Short description of what this adapter does.
    public let description: String

    /// Category of the base model this adapter targets.
    public let baseModelCategory: ModelCategory

    /// LoRA rank used during training.
    public let rank: Int

    /// URL to download the adapter.safetensors file.
    public let adapterURL: URL

    /// URL to download the adapter.json metadata file.
    public let metadataURL: URL

    /// Searchable/filterable tags.
    public let tags: [String]

    /// Whether this adapter is featured/promoted.
    public let featured: Bool

    /// Minimum app version required to use this adapter.
    public let minAppVersion: String

    /// When the adapter was published.
    public let createdAt: Date

    /// Total number of downloads (for sorting).
    public let downloadCount: Int

    public init(
        id: String,
        name: String,
        author: String,
        description: String,
        baseModelCategory: ModelCategory,
        rank: Int,
        adapterURL: URL,
        metadataURL: URL,
        tags: [String],
        featured: Bool,
        minAppVersion: String,
        createdAt: Date,
        downloadCount: Int
    ) {
        self.id = id
        self.name = name
        self.author = author
        self.description = description
        self.baseModelCategory = baseModelCategory
        self.rank = rank
        self.adapterURL = adapterURL
        self.metadataURL = metadataURL
        self.tags = tags
        self.featured = featured
        self.minAppVersion = minAppVersion
        self.createdAt = createdAt
        self.downloadCount = downloadCount
    }
}

/// The top-level manifest structure fetched from the community repo.
public struct AdapterManifest: Codable, Sendable {
    /// Schema version for forward compatibility.
    public let version: Int

    /// All published adapters.
    public let adapters: [CommunityAdapter]

    public init(version: Int, adapters: [CommunityAdapter]) {
        self.version = version
        self.adapters = adapters
    }
}
```

Note: This imports `ModelCategory` from AnvilCore's `ModelRegistry/ModelSpec.swift`.

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds (or only pre-existing errors).

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Community/CommunityAdapter.swift
git commit -m "feat(core): add CommunityAdapter and AdapterManifest types"
```

---

### Task 4: LoRALinear Evolution (Multi-Contribution Forward Pass)

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift`

This is the most critical task — evolving LoRALinear from single-adapter to multi-contribution while maintaining backward compatibility.

- [ ] **Step 1: Add contributions array and update forward pass**

Replace the entire contents of `LoRALinear.swift` with:

```swift
import Foundation
import MLX
import MLXNN
import MLXRandom

/// A Linear layer wrapped with one or more low-rank adapter contributions.
///
/// Single adapter (Phase 10a/b):
///   Output = base(x) + scaling * dropout(x @ A^T @ B^T)
///
/// Multi-adapter stacking (Phase 10c):
///   Output = base(x) + sum_i(strength_i * x @ A_i^T @ B_i^T)
///
/// A is initialized with Kaiming uniform, B is initialized to zero,
/// so a fresh adapter starts as identity (no effect on output).
public class LoRALinear: Linear {
    /// The original frozen weight matrix stored separately.
    let baseWeight: MLXArray

    /// The original bias (nil if the base layer had no bias).
    let baseBias: MLXArray?

    /// Low-rank matrix A: shape (rank, inFeatures). Kaiming uniform init.
    /// Used for training (single adapter mode). For stacking, see contributions.
    var loraA: MLXArray

    /// Low-rank matrix B: shape (outFeatures, rank). Zero init.
    /// Used for training (single adapter mode). For stacking, see contributions.
    var loraB: MLXArray

    /// Scaling factor = alpha / rank.
    let scaling: Float

    /// Dropout rate (applied during training only).
    let dropoutRate: Float

    /// Whether we are in training mode (enables dropout, uses loraA/loraB directly).
    var isTraining: Bool = false

    /// Whether adapter has been merged into base weights.
    private var isMerged: Bool = false

    /// Multiple adapter contributions for inference stacking.
    /// When non-empty, these are used instead of the single loraA/loraB pair.
    /// Each contribution has its own strength multiplier.
    public var contributions: [LoRAContribution] = []

    public init(base: Linear, rank: Int, alpha: Float = 16.0, dropout: Float = 0.0) {
        let inFeatures = base.weight.shape[1]
        let outFeatures = base.weight.shape[0]

        // Store a copy of the original weight for unmerge.
        // Use identity arithmetic (+ 0) to create a distinct MLXArray holding the same data.
        self.baseWeight = base.weight + MLXArray(Float(0))
        self.baseBias = base.bias.map { $0 + MLXArray(Float(0)) }
        self.scaling = alpha / Float(rank)
        self.dropoutRate = dropout

        // Kaiming uniform for A: bound = sqrt(1/inFeatures)
        let bound = Float(1.0 / Float(inFeatures).squareRoot())
        self.loraA = MLXRandom.uniform(-bound ..< bound, [rank, inFeatures])

        // Zero init for B so adapter starts as identity
        self.loraB = MLXArray.zeros([outFeatures, rank])

        // Initialize Linear superclass with the base weights
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base linear: x @ W^T + bias
        let baseOut: MLXArray
        if let bias {
            baseOut = addMM(bias, x, weight.T)
        } else {
            baseOut = matmul(x, weight.T)
        }

        if isMerged {
            return baseOut
        }

        // Training mode: use single loraA/loraB pair with dropout
        if isTraining {
            var loraInput = x
            if dropoutRate > 0 {
                let mask = MLXRandom.uniform(0 ..< 1, loraInput.shape) .> dropoutRate
                loraInput = loraInput * mask / (1.0 - dropoutRate)
            }
            let loraOut = scaling * matmul(matmul(loraInput, loraA.T), loraB.T)
            return baseOut + loraOut
        }

        // Inference stacking mode: use contributions if present
        if !contributions.isEmpty {
            var out = baseOut
            for contribution in contributions where contribution.strength != 0.0 {
                let loraOut = matmul(matmul(x, contribution.loraA.T), contribution.loraB.T)
                out = out + MLXArray(contribution.strength) * loraOut
            }
            return out
        }

        // Single adapter inference (backward compat with 10a/b usage)
        let loraOut = scaling * matmul(matmul(x, loraA.T), loraB.T)
        return baseOut + loraOut
    }

    /// Fold adapter weights into the base weight matrix for zero-overhead inference.
    public func merge() {
        guard !isMerged else { return }
        let delta = scaling * matmul(loraB, loraA)
        weight._updateInternal(baseWeight + delta)
        isMerged = true
    }

    /// Reverse a merge operation, restoring the original base weight.
    public func unmerge() {
        guard isMerged else { return }
        weight._updateInternal(baseWeight)
        isMerged = false
    }

    /// Mark base parameters as non-trainable, keeping only loraA and loraB trainable.
    public func freezeBase() {
        freeze(recursive: false, keys: ["weight", "bias", "baseWeight", "baseBias"])
    }
}
```

Key changes from the original:
- Added `public var contributions: [LoRAContribution] = []`
- `callAsFunction` now has three paths: training (single pair + dropout), stacking (contributions array), single-adapter inference (backward compat)
- Contributions with `strength == 0.0` are skipped (performance optimization)

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds. All existing code that uses LoRALinear continues to work (contributions is empty by default, so single-adapter path is taken).

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift
git commit -m "feat(core): evolve LoRALinear for multi-contribution stacking

Adds contributions array for Inject-All stacking at inference.
Training mode continues using single loraA/loraB pair.
Backward compatible — empty contributions uses legacy path."
```

---

### Task 5: LoRAInjector Stacking Methods

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift`

- [ ] **Step 1: Add stacking extension methods**

Add the following extension at the end of `LoRAInjector.swift`, before the closing of the file:

```swift
// MARK: - Multi-Adapter Stacking

extension LoRAInjector {

    /// Apply a full adapter stack to the model. Replaces all current contributions
    /// on every LoRALinear layer matching the given paths.
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder with injected LoRA layers.
    ///   - contributions: All contributions to apply (from multiple adapters).
    ///   - paths: Key paths where LoRALinear layers exist.
    public static func applyStack(
        model: DiTDecoder,
        contributions: [LoRAContribution],
        paths: [String]
    ) {
        // First clear existing contributions
        clearContributions(model: model, paths: paths)

        // Group contributions by path — each LoRALinear gets all adapter contributions
        // Since contributions hold per-layer weights, we apply each contribution's
        // weights to the matching LoRALinear layer based on the path encoding in the weights.
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            guard let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear else {
                continue
            }

            // Build contributions for this specific layer from adapter weight data
            var layerContributions: [LoRAContribution] = []
            for contribution in contributions {
                // Each contribution is already layer-specific (loraA/loraB sized for this layer)
                layerContributions.append(contribution)
            }
            loraLinear.contributions = layerContributions
        }
    }

    /// Apply contributions for a single adapter across all LoRALinear layers.
    /// Weights are keyed by path (e.g. "layers.0.selfAttn.qProj.loraA").
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder with injected LoRA layers.
    ///   - adapterID: Identifier for this adapter.
    ///   - displayName: Display name for UI.
    ///   - source: Provenance of the adapter.
    ///   - weights: Array of (path.loraA/loraB, MLXArray) tuples.
    ///   - strength: Initial strength multiplier.
    ///   - paths: Key paths where LoRALinear layers exist.
    public static func addContribution(
        model: DiTDecoder,
        adapterID: String,
        displayName: String?,
        source: AdapterSource,
        weights: [(String, MLXArray)],
        strength: Float,
        paths: [String]
    ) {
        let weightDict = Dictionary(weights, uniquingKeysWith: { _, last in last })

        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            guard let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear else {
                continue
            }

            guard let loraA = weightDict["\(path).loraA"],
                  let loraB = weightDict["\(path).loraB"] else {
                continue
            }

            let contribution = LoRAContribution(
                adapterID: adapterID,
                displayName: displayName,
                source: source,
                loraA: loraA,
                loraB: loraB,
                strength: strength
            )
            loraLinear.contributions.append(contribution)
        }
    }

    /// Remove all contributions from every LoRALinear layer (revert to base model behavior).
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder with injected LoRA layers.
    ///   - paths: Key paths where LoRALinear layers exist.
    public static func clearContributions(model: DiTDecoder, paths: [String]) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                loraLinear.contributions = []
            }
        }
    }

    /// Update strength for a single adapter across all LoRALinear layers.
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder with injected LoRA layers.
    ///   - adapterID: Which adapter to update.
    ///   - strength: New strength value.
    ///   - paths: Key paths where LoRALinear layers exist.
    public static func setStrength(
        model: DiTDecoder,
        adapterID: String,
        strength: Float,
        paths: [String]
    ) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                for i in loraLinear.contributions.indices {
                    if loraLinear.contributions[i].adapterID == adapterID {
                        loraLinear.contributions[i].strength = strength
                    }
                }
            }
        }
    }

    /// Remove a specific adapter's contributions from all LoRALinear layers.
    ///
    /// - Parameters:
    ///   - model: The DiTDecoder with injected LoRA layers.
    ///   - adapterID: Which adapter to remove.
    ///   - paths: Key paths where LoRALinear layers exist.
    public static func removeContribution(
        model: DiTDecoder,
        adapterID: String,
        paths: [String]
    ) {
        for path in paths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let loraLinear = resolveLinear(in: layer, path: subPath) as? LoRALinear {
                loraLinear.contributions.removeAll { $0.adapterID == adapterID }
            }
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds. New methods are additive — no existing code changes.

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift
git commit -m "feat(core): add LoRAInjector stacking methods

Adds addContribution, clearContributions, setStrength, removeContribution
for multi-adapter inference stacking."
```

---

### Task 6: LoRAAdapter Extension (source + remoteRepoURL)

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift`

- [ ] **Step 1: Add source and remoteRepoURL fields to LoRAAdapter**

In `LoRAConfig.swift`, update the `LoRAAdapter` struct. Replace the existing struct with:

```swift
/// A saved LoRA adapter with metadata.
public struct LoRAAdapter: Codable, Sendable, Identifiable, Equatable, Hashable {
    /// Directory name used as identifier.
    public var id: String

    /// User-facing display name.
    public var name: String

    /// The LoRA config used during training.
    public let config: LoRAConfig

    /// ID of the base model this adapter was trained against.
    public let baseModelID: String

    /// Category of the base model for fast compatibility filtering.
    public let baseModelCategory: ModelCategory

    /// When training completed.
    public let createdAt: Date

    /// Number of training samples used.
    public let sampleCount: Int

    /// Final training loss (nil if training was stopped early).
    public let finalLoss: Float?

    /// Where this adapter came from (defaults to .local for backward compat).
    public var source: AdapterSource

    /// Original HuggingFace repo URL if imported from HF.
    public var remoteRepoURL: URL?

    public init(
        id: String,
        name: String,
        config: LoRAConfig,
        baseModelID: String,
        baseModelCategory: ModelCategory,
        createdAt: Date,
        sampleCount: Int,
        finalLoss: Float?,
        source: AdapterSource = .local,
        remoteRepoURL: URL? = nil
    ) {
        self.id = id
        self.name = name
        self.config = config
        self.baseModelID = baseModelID
        self.baseModelCategory = baseModelCategory
        self.createdAt = createdAt
        self.sampleCount = sampleCount
        self.finalLoss = finalLoss
        self.source = source
        self.remoteRepoURL = remoteRepoURL
    }

    // Custom Codable for backward compatibility — existing adapters without
    // `source` field decode as .local.
    enum CodingKeys: String, CodingKey {
        case id, name, config, baseModelID, baseModelCategory
        case createdAt, sampleCount, finalLoss, source, remoteRepoURL
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        config = try container.decode(LoRAConfig.self, forKey: .config)
        baseModelID = try container.decode(String.self, forKey: .baseModelID)
        baseModelCategory = try container.decode(ModelCategory.self, forKey: .baseModelCategory)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        sampleCount = try container.decode(Int.self, forKey: .sampleCount)
        finalLoss = try container.decodeIfPresent(Float.self, forKey: .finalLoss)
        source = try container.decodeIfPresent(AdapterSource.self, forKey: .source) ?? .local
        remoteRepoURL = try container.decodeIfPresent(URL.self, forKey: .remoteRepoURL)
    }
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds. The new `source` parameter has a default value (`.local`), so all existing call sites (TrainingService, AdapterRegistry.merge) continue to compile without changes.

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift
git commit -m "feat(core): add source and remoteRepoURL to LoRAAdapter

Backward compatible — existing JSON without source field decodes as .local
via decodeIfPresent with default."
```

---

### Task 7: CommunityService

**Files:**
- Create: `AnvilApp/Services/CommunityService.swift`

- [ ] **Step 1: Create the CommunityService class**

```swift
import Foundation
import AnvilCore

/// Manages community adapter discovery, downloading, and HuggingFace imports.
@MainActor
final class CommunityService: ObservableObject {

    // MARK: - Published State

    @Published var manifest: AdapterManifest?
    @Published var downloads: [String: DownloadState] = [:]
    @Published var lastRefresh: Date?
    @Published var isRefreshing: Bool = false
    @Published var error: String?

    // MARK: - Types

    enum DownloadState: Equatable {
        case downloading(progress: Double)
        case failed(String)
        case completed

        static func == (lhs: DownloadState, rhs: DownloadState) -> Bool {
            switch (lhs, rhs) {
            case (.downloading(let a), .downloading(let b)): return a == b
            case (.failed(let a), .failed(let b)): return a == b
            case (.completed, .completed): return true
            default: return false
            }
        }
    }

    // MARK: - Configuration

    private let manifestURL = URL(string:
        "https://raw.githubusercontent.com/PiMPStudios/anvil-adapters/main/manifest.json")!
    private let cacheTTL: TimeInterval = 3600  // 1 hour
    private static let maxConcurrentDownloads = 3

    // MARK: - Private State

    private var activeTasks: [String: Task<Void, Never>] = [:]
    private var cachedManifest: AdapterManifest?
    private var cacheTimestamp: Date?

    // MARK: - Manifest

    /// Fetch the community manifest. Respects cache TTL unless force = true.
    func refreshManifest(force: Bool = false) async {
        // Check cache
        if !force, let cached = cachedManifest, let ts = cacheTimestamp,
           Date().timeIntervalSince(ts) < cacheTTL {
            manifest = cached
            return
        }

        isRefreshing = true
        error = nil

        do {
            let (data, _) = try await URLSession.shared.data(from: manifestURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let fetched = try decoder.decode(AdapterManifest.self, from: data)

            cachedManifest = fetched
            cacheTimestamp = Date()
            manifest = fetched
            lastRefresh = Date()
        } catch {
            // Show cached version if available
            if cachedManifest != nil {
                self.error = "Offline — showing cached results"
            } else {
                self.error = "Failed to load community adapters: \(error.localizedDescription)"
            }
        }

        isRefreshing = false
    }

    /// Whether a community adapter is already installed locally.
    func isInstalled(_ adapterID: String) -> Bool {
        AdapterRegistry.listAdapters().contains { $0.id == adapterID }
    }

    // MARK: - Downloads

    /// Download a community adapter (safetensors + metadata) and save locally.
    func download(adapter: CommunityAdapter) {
        guard activeTasks.count < Self.maxConcurrentDownloads else {
            // Queue: just mark as downloading at 0
            downloads[adapter.id] = .downloading(progress: 0)
            // Retry after a delay when a slot opens
            let task = Task { [weak self] in
                try? await Task.sleep(for: .seconds(2))
                await self?.download(adapter: adapter)
            }
            activeTasks[adapter.id] = task
            return
        }

        downloads[adapter.id] = .downloading(progress: 0)

        let task = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.performDownload(adapter: adapter, retryCount: 1)
                self.downloads[adapter.id] = .completed
            } catch {
                self.downloads[adapter.id] = .failed(error.localizedDescription)
            }
            self.activeTasks.removeValue(forKey: adapter.id)
        }
        activeTasks[adapter.id] = task
    }

    /// Cancel an in-progress download and clean up partial files.
    func cancelDownload(adapterID: String) {
        activeTasks[adapterID]?.cancel()
        activeTasks.removeValue(forKey: adapterID)
        downloads.removeValue(forKey: adapterID)

        // Clean up any partial files
        let dir = AdapterRegistry.adapterDirectory(id: adapterID)
        try? FileManager.default.removeItem(at: dir)
    }

    /// Retry a failed download.
    func retryDownload(adapter: CommunityAdapter) {
        downloads.removeValue(forKey: adapter.id)
        download(adapter: adapter)
    }

    // MARK: - HuggingFace Import

    /// Import an adapter from a HuggingFace repo URL.
    ///
    /// Resolves `{repoURL}/resolve/main/adapter.json` and `adapter.safetensors`.
    /// Fetches metadata first for validation, then downloads weights.
    ///
    /// - Parameter repoURL: The HuggingFace repo URL (e.g. https://huggingface.co/user/repo)
    /// - Returns: The installed LoRAAdapter.
    func importFromHuggingFace(repoURL: URL) async throws -> LoRAAdapter {
        let baseURL = repoURL.appendingPathComponent("resolve/main")
        let metadataURL = baseURL.appendingPathComponent("adapter.json")
        let weightsURL = baseURL.appendingPathComponent("adapter.safetensors")

        // 1. Fetch metadata
        let (metaData, metaResponse) = try await URLSession.shared.data(from: metadataURL)
        guard let httpResponse = metaResponse as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw CommunityError.missingFile("adapter.json", repoURL)
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var adapter = try decoder.decode(LoRAAdapter.self, from: metaData)

        // Check for duplicate
        if AdapterRegistry.listAdapters().contains(where: { $0.remoteRepoURL == repoURL }) {
            throw CommunityError.alreadyInstalled(adapter.name)
        }

        // Update provenance
        adapter.source = .huggingface
        adapter.remoteRepoURL = repoURL

        // 2. Download weights
        let (weightsData, weightsResponse) = try await URLSession.shared.data(from: weightsURL)
        guard let httpWeightsResponse = weightsResponse as? HTTPURLResponse,
              httpWeightsResponse.statusCode == 200 else {
            throw CommunityError.missingFile("adapter.safetensors", repoURL)
        }

        // 3. Save to local registry
        let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try metaData.write(to: dir.appending(path: "adapter.json"))
        try weightsData.write(to: dir.appending(path: "adapter.safetensors"))

        return adapter
    }

    // MARK: - Private

    private func performDownload(adapter: CommunityAdapter, retryCount: Int) async throws {
        do {
            // Download metadata
            let (metaData, _) = try await URLSession.shared.data(from: adapter.metadataURL)
            downloads[adapter.id] = .downloading(progress: 0.1)

            // Download weights with progress
            let (weightsData, _) = try await URLSession.shared.data(from: adapter.adapterURL)
            downloads[adapter.id] = .downloading(progress: 0.9)

            // Parse metadata to get a LoRAAdapter, update source
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            var localAdapter = try decoder.decode(LoRAAdapter.self, from: metaData)
            localAdapter.source = .community
            localAdapter.remoteRepoURL = nil

            // Save to disk
            let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            // Write updated metadata with source field
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let updatedMeta = try encoder.encode(localAdapter)
            try updatedMeta.write(to: dir.appending(path: "adapter.json"))
            try weightsData.write(to: dir.appending(path: "adapter.safetensors"))

            downloads[adapter.id] = .downloading(progress: 1.0)
        } catch {
            if retryCount > 0 {
                // Auto-retry once
                try await Task.sleep(for: .seconds(1))
                try await performDownload(adapter: adapter, retryCount: retryCount - 1)
            } else {
                // Clean up partial files
                let dir = AdapterRegistry.adapterDirectory(id: adapter.id)
                try? FileManager.default.removeItem(at: dir)
                throw error
            }
        }
    }
}

/// Errors specific to community adapter operations.
enum CommunityError: Error, LocalizedError {
    case missingFile(String, URL)
    case alreadyInstalled(String)
    case incompatibleVersion(String)

    var errorDescription: String? {
        switch self {
        case .missingFile(let file, let url):
            return "This repo doesn't follow the Anvil adapter format. Expected \(file) at \(url.absoluteString)"
        case .alreadyInstalled(let name):
            return "\"\(name)\" is already installed. Delete it first to re-download."
        case .incompatibleVersion(let required):
            return "This adapter requires Anvil \(required) or later"
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds (or only pre-existing errors).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/CommunityService.swift
git commit -m "feat(app): add CommunityService for manifest fetch and adapter downloads

Handles manifest caching (1hr TTL), concurrent downloads (max 3),
auto-retry, HuggingFace URL import, and partial file cleanup."
```

---

### Task 8: AdapterStackManager

**Files:**
- Create: `AnvilApp/Services/AdapterStackManager.swift`

- [ ] **Step 1: Create the AdapterStackManager class**

```swift
import Foundation
import AnvilCore
import MLX

/// Manages the runtime state of multiple stacked LoRA adapters.
///
/// Owns which adapters are active, at what strength, and coordinates
/// with LoRAInjector to apply/update/clear contributions on the model.
@MainActor
final class AdapterStackManager: ObservableObject {

    // MARK: - Published State

    /// The primary adapter (shown in the main AdapterPicker, strength always 1.0).
    @Published var primaryAdapter: LoRAAdapter?

    /// Additional stacked adapters with their individual strengths.
    @Published var additionalAdapters: [(adapter: LoRAAdapter, strength: Float)] = []

    // MARK: - Constants

    /// Maximum total adapters (primary + additional).
    static let maxStackSize = 3

    /// Maximum additional adapters that can be stacked.
    var maxAdditional: Int { Self.maxStackSize - (primaryAdapter != nil ? 1 : 0) }

    /// Whether the stack is at capacity.
    var isAtCapacity: Bool {
        let total = (primaryAdapter != nil ? 1 : 0) + additionalAdapters.count
        return total >= Self.maxStackSize
    }

    /// Total combined strength of all active adapters.
    var totalStrength: Float {
        var total: Float = primaryAdapter != nil ? 1.0 : 0.0
        for entry in additionalAdapters {
            total += entry.strength
        }
        return total
    }

    // MARK: - Stack Management

    /// Set the primary adapter. Replaces any existing primary.
    func setPrimary(_ adapter: LoRAAdapter?) {
        primaryAdapter = adapter
    }

    /// Add an adapter to the additional stack.
    /// - Parameters:
    ///   - adapter: The adapter to add.
    ///   - strength: Initial strength (default 0.5).
    func addToStack(_ adapter: LoRAAdapter, strength: Float = 0.5) {
        guard !isAtCapacity else { return }
        guard !additionalAdapters.contains(where: { $0.adapter.id == adapter.id }) else { return }
        guard adapter.id != primaryAdapter?.id else { return }
        additionalAdapters.append((adapter: adapter, strength: strength))
    }

    /// Remove an adapter from the additional stack by ID.
    func removeFromStack(adapterID: String) {
        additionalAdapters.removeAll { $0.adapter.id == adapterID }
    }

    /// Update strength for a stacked adapter.
    func setStrength(adapterID: String, strength: Float) {
        guard let idx = additionalAdapters.firstIndex(where: { $0.adapter.id == adapterID }) else {
            return
        }
        additionalAdapters[idx].strength = strength
    }

    /// Clear all additional adapters (keep primary).
    func clearStack() {
        additionalAdapters = []
    }

    /// Clear everything (primary + additional).
    func clearAll() {
        primaryAdapter = nil
        additionalAdapters = []
    }

    /// All adapters currently in the stack (for display purposes).
    var allStackedAdapters: [(adapter: LoRAAdapter, strength: Float)] {
        var result: [(adapter: LoRAAdapter, strength: Float)] = []
        if let primary = primaryAdapter {
            result.append((adapter: primary, strength: 1.0))
        }
        result.append(contentsOf: additionalAdapters)
        return result
    }

    /// Whether stacking is active (more than just a primary adapter).
    var isStacking: Bool {
        !additionalAdapters.isEmpty
    }

    /// Handle adapter deletion — remove from stack if present.
    func handleAdapterDeleted(adapterID: String) {
        if primaryAdapter?.id == adapterID {
            // Promote first additional to primary, or clear
            if let first = additionalAdapters.first {
                primaryAdapter = first.adapter
                additionalAdapters.removeFirst()
            } else {
                primaryAdapter = nil
            }
        } else {
            removeFromStack(adapterID: adapterID)
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds (or only pre-existing errors).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/AdapterStackManager.swift
git commit -m "feat(app): add AdapterStackManager for multi-adapter runtime state

Manages primary + additional adapters with strength sliders.
Max 3 total. Handles promotion on delete and capacity enforcement."
```

---

### Task 9: CommunityBrowserView

**Files:**
- Create: `AnvilApp/UI/Settings/CommunityBrowserView.swift`

- [ ] **Step 1: Create the CommunityBrowserView**

```swift
import SwiftUI
import AnvilCore

/// Browsing interface for community-published adapters.
struct CommunityBrowserView: View {
    @ObservedObject var service: CommunityService
    @State private var searchText = ""
    @State private var selectedCategory: ModelCategory?
    @State private var showingImportSheet = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Toolbar
            HStack {
                // Search field
                TextField("Search adapters...", text: $searchText)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 200)

                // Category filter
                Picker("Category", selection: $selectedCategory) {
                    Text("All").tag(ModelCategory?.none)
                    ForEach(ModelCategory.allCases, id: \.self) { cat in
                        Text(cat.rawValue).tag(Optional(cat))
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 140)

                Spacer()

                Button {
                    showingImportSheet = true
                } label: {
                    Label("Add from URL", systemImage: "link.badge.plus")
                }

                Button {
                    Task { await service.refreshManifest(force: true) }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(service.isRefreshing)
            }
            .padding()

            Divider()

            // Content
            if service.isRefreshing && service.manifest == nil {
                loadingState
            } else if let manifest = service.manifest, !manifest.adapters.isEmpty {
                adapterList(manifest.adapters)
            } else {
                emptyState
            }

            // Footer
            if let lastRefresh = service.lastRefresh {
                HStack {
                    Spacer()
                    Text("Last updated: \(lastRefresh, style: .relative) ago")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal)
                .padding(.vertical, 4)
            }

            // Error banner
            if let error = service.error {
                HStack {
                    Image(systemName: "wifi.slash")
                    Text(error)
                    Spacer()
                    Button("Retry") {
                        Task { await service.refreshManifest(force: true) }
                    }
                    .buttonStyle(.borderless)
                }
                .font(.caption)
                .foregroundStyle(.orange)
                .padding(8)
                .background(.orange.opacity(0.05))
            }
        }
        .task {
            await service.refreshManifest()
        }
        .sheet(isPresented: $showingImportSheet) {
            HuggingFaceImportSheet(service: service)
                .frame(minWidth: 450, minHeight: 250)
        }
    }

    private var filteredAdapters: [CommunityAdapter] {
        guard let manifest = service.manifest else { return [] }
        var results = manifest.adapters

        if let cat = selectedCategory {
            results = results.filter { $0.baseModelCategory == cat }
        }

        if !searchText.isEmpty {
            let query = searchText.lowercased()
            results = results.filter { adapter in
                adapter.name.lowercased().contains(query) ||
                adapter.author.lowercased().contains(query) ||
                adapter.tags.contains { $0.lowercased().contains(query) } ||
                adapter.description.lowercased().contains(query)
            }
        }

        return results
    }

    private func adapterList(_ adapters: [CommunityAdapter]) -> some View {
        List(filteredAdapters) { adapter in
            communityRow(adapter)
        }
    }

    private func communityRow(_ adapter: CommunityAdapter) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(adapter.name)
                        .font(.body)
                        .fontWeight(.medium)
                    if adapter.featured {
                        Image(systemName: "star.fill")
                            .font(.caption2)
                            .foregroundStyle(.yellow)
                    }
                }
                Text("by \(adapter.author)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(adapter.description)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                HStack(spacing: 6) {
                    Text(adapter.baseModelCategory.rawValue)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.blue.opacity(0.1))
                        .clipShape(Capsule())
                    Text("Rank \(adapter.rank)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    ForEach(adapter.tags.prefix(3), id: \.self) { tag in
                        Text(tag)
                            .font(.caption2)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(.gray.opacity(0.1))
                            .clipShape(Capsule())
                    }
                }
            }

            Spacer()

            downloadButton(for: adapter)
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func downloadButton(for adapter: CommunityAdapter) -> some View {
        if service.isInstalled(adapter.id) {
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
        } else if let state = service.downloads[adapter.id] {
            switch state {
            case .downloading(let progress):
                ProgressView(value: progress)
                    .progressViewStyle(.circular)
                    .controlSize(.small)
                    .frame(width: 24, height: 24)
            case .failed:
                Button("Retry") {
                    service.retryDownload(adapter: adapter)
                }
                .foregroundStyle(.red)
                .font(.caption)
            case .completed:
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
        } else {
            Button("Get") {
                service.download(adapter: adapter)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
    }

    private var loadingState: some View {
        VStack {
            Spacer()
            ProgressView("Loading community adapters...")
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: "globe")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("No Community Adapters Available")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Check back later or import from a HuggingFace URL.")
                .font(.caption)
                .foregroundStyle(.tertiary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding()
    }
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build may fail if `HuggingFaceImportSheet` doesn't exist yet. Create a stub file first if needed:

```swift
// Stub — replaced in Task 10
import SwiftUI
import AnvilCore

struct HuggingFaceImportSheet: View {
    @ObservedObject var service: CommunityService
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Text("Import from HuggingFace")
            .padding()
    }
}
```

Save this stub to `AnvilApp/UI/Settings/HuggingFaceImportSheet.swift` temporarily.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Settings/CommunityBrowserView.swift AnvilApp/UI/Settings/HuggingFaceImportSheet.swift
git commit -m "feat(app): add CommunityBrowserView with search, filters, and download buttons

Includes HuggingFaceImportSheet stub (full implementation in next task).
App Store-like download flow with circular progress indicators."
```

---

### Task 10: HuggingFaceImportSheet

**Files:**
- Modify: `AnvilApp/UI/Settings/HuggingFaceImportSheet.swift` (replace stub)

- [ ] **Step 1: Replace the stub with full implementation**

Replace the entire contents of `HuggingFaceImportSheet.swift`:

```swift
import SwiftUI
import AnvilCore

/// Modal sheet for importing a LoRA adapter from a HuggingFace repo URL.
struct HuggingFaceImportSheet: View {
    @ObservedObject var service: CommunityService
    @Environment(\.dismiss) private var dismiss

    @State private var repoURLText = ""
    @State private var fetchedAdapter: LoRAAdapter?
    @State private var isFetching = false
    @State private var isDownloading = false
    @State private var errorMessage: String?
    @State private var compatibilityWarning: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Import from HuggingFace")
                .font(.headline)

            Text("Paste a HuggingFace repo URL. The repo must contain adapter.json and adapter.safetensors at the root.")
                .font(.caption)
                .foregroundStyle(.secondary)

            // URL input
            HStack {
                TextField("https://huggingface.co/user/my-adapter", text: $repoURLText)
                    .textFieldStyle(.roundedBorder)

                Button("Fetch") {
                    Task { await fetchMetadata() }
                }
                .disabled(repoURLText.isEmpty || isFetching)
            }

            // Preview section
            if let adapter = fetchedAdapter {
                previewSection(adapter)
            }

            // Compatibility warning
            if let warning = compatibilityWarning {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text(warning)
                        .font(.caption)
                }
                .padding(8)
                .background(.yellow.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            // Error
            if let error = errorMessage {
                HStack {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.red)
                    Text(error)
                        .font(.caption)
                }
            }

            Spacer()

            // Actions
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)

                if fetchedAdapter != nil {
                    Button("Download") {
                        Task { await performDownload() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isDownloading)
                    .keyboardShortcut(.defaultAction)
                }
            }

            // Loading overlay
            if isFetching || isDownloading {
                HStack {
                    ProgressView()
                        .controlSize(.small)
                    Text(isFetching ? "Fetching metadata..." : "Downloading adapter...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding()
        .frame(minWidth: 450, minHeight: 250)
    }

    private func previewSection(_ adapter: LoRAAdapter) -> some View {
        GroupBox("Adapter Info") {
            VStack(alignment: .leading, spacing: 6) {
                LabeledContent("Name", value: adapter.name)
                LabeledContent("Base Model", value: adapter.baseModelCategory.rawValue)
                LabeledContent("Rank", value: "\(adapter.config.rank)")
                LabeledContent("Samples", value: "\(adapter.sampleCount)")
                if let loss = adapter.finalLoss {
                    LabeledContent("Final Loss", value: String(format: "%.4f", loss))
                }
            }
            .font(.caption)
        }
    }

    private func fetchMetadata() async {
        guard let url = URL(string: repoURLText.trimmingCharacters(in: .whitespacesAndNewlines)),
              url.host?.contains("huggingface") == true else {
            errorMessage = "Please enter a valid HuggingFace URL"
            return
        }

        isFetching = true
        errorMessage = nil
        compatibilityWarning = nil
        fetchedAdapter = nil

        let metadataURL = url.appendingPathComponent("resolve/main/adapter.json")

        do {
            let (data, response) = try await URLSession.shared.data(from: metadataURL)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                errorMessage = "Could not find adapter.json in this repo"
                isFetching = false
                return
            }

            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let adapter = try decoder.decode(LoRAAdapter.self, from: data)
            fetchedAdapter = adapter

            // Check compatibility
            let installedCategories: Set<ModelCategory> = [.aceStepDiT]  // TODO: get from registry
            if !installedCategories.contains(adapter.baseModelCategory) {
                compatibilityWarning = "This adapter is for \(adapter.baseModelCategory.rawValue), but you don't have a compatible model installed. Download anyway?"
            }
        } catch {
            errorMessage = "Failed to fetch metadata: \(error.localizedDescription)"
        }

        isFetching = false
    }

    private func performDownload() async {
        guard let url = URL(string: repoURLText.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return
        }

        isDownloading = true
        errorMessage = nil

        do {
            let _ = try await service.importFromHuggingFace(repoURL: url)
            dismiss()
        } catch {
            errorMessage = error.localizedDescription
        }

        isDownloading = false
    }
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds (or only pre-existing errors).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Settings/HuggingFaceImportSheet.swift
git commit -m "feat(app): implement HuggingFaceImportSheet with progressive disclosure

Fetch metadata first, show preview, warn on category mismatch,
then download on confirmation."
```

---

### Task 11: AdapterStackView

**Files:**
- Create: `AnvilApp/UI/Generation/AdapterStackView.swift`

- [ ] **Step 1: Create the AdapterStackView**

```swift
import SwiftUI
import AnvilCore

/// Stacking controls for additional LoRA adapters in the generation view.
/// Placed below the primary AdapterPicker.
struct AdapterStackView: View {
    @ObservedObject var stackManager: AdapterStackManager
    let activeModelCategory: ModelCategory

    @State private var showingAdapterPicker = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Section header
            HStack {
                Text("Additional Adapters")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if !stackManager.additionalAdapters.isEmpty {
                    Text("Total: \(String(format: "%.2f", stackManager.totalStrength))")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            // Stacked adapter rows
            ForEach(Array(stackManager.additionalAdapters.enumerated()), id: \.element.adapter.id) { idx, entry in
                stackedAdapterRow(entry.adapter, strength: entry.strength)
            }

            // Add button
            Button {
                showingAdapterPicker = true
            } label: {
                Label("Add adapter", systemImage: "plus")
                    .font(.caption)
            }
            .disabled(stackManager.isAtCapacity)
            .help(stackManager.isAtCapacity ? "Maximum \(AdapterStackManager.maxStackSize) adapters active" : "Add another adapter to the stack")
        }
        .sheet(isPresented: $showingAdapterPicker) {
            adapterPickerSheet
        }
    }

    private func stackedAdapterRow(_ adapter: LoRAAdapter, strength: Float) -> some View {
        HStack(spacing: 8) {
            // Name + source badge
            VStack(alignment: .leading, spacing: 2) {
                Text(adapter.name)
                    .font(.caption)
                    .fontWeight(.medium)
                    .lineLimit(1)
                if adapter.source != .local {
                    Text(adapter.source.rawValue.capitalized)
                        .font(.caption2)
                        .foregroundStyle(.blue)
                }
            }
            .frame(width: 100, alignment: .leading)

            // Strength slider
            Slider(
                value: Binding(
                    get: { strength },
                    set: { stackManager.setStrength(adapterID: adapter.id, strength: $0) }
                ),
                in: 0.0...1.0,
                step: 0.05
            )

            // Strength label
            Text(String(format: "%.2f", strength))
                .font(.caption)
                .monospacedDigit()
                .frame(width: 32)

            // Remove button
            Button {
                stackManager.removeFromStack(adapterID: adapter.id)
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
    }

    private var adapterPickerSheet: some View {
        VStack(spacing: 12) {
            Text("Select Adapter to Stack")
                .font(.headline)

            let available = availableAdapters
            if available.isEmpty {
                Text("No compatible adapters available")
                    .foregroundStyle(.secondary)
                    .padding()
            } else {
                List(available) { adapter in
                    Button {
                        stackManager.addToStack(adapter)
                        showingAdapterPicker = false
                    } label: {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(adapter.name)
                                    .font(.body)
                                HStack(spacing: 4) {
                                    Text("Rank \(adapter.config.rank)")
                                    if adapter.source != .local {
                                        Text("(\(adapter.source.rawValue))")
                                    }
                                }
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                            Spacer()
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }

            Button("Cancel") { showingAdapterPicker = false }
                .keyboardShortcut(.cancelAction)
        }
        .padding()
        .frame(minWidth: 300, minHeight: 200)
    }

    /// Adapters compatible with current model that aren't already in the stack.
    private var availableAdapters: [LoRAAdapter] {
        let allCompatible = AdapterRegistry.compatibleAdapters(for: activeModelCategory)
        let stackedIDs = Set(stackManager.allStackedAdapters.map(\.adapter.id))
        return allCompatible.filter { !stackedIDs.contains($0.id) }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds (or only pre-existing errors).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Generation/AdapterStackView.swift
git commit -m "feat(app): add AdapterStackView with strength sliders and capacity enforcement

Shows stacked adapters with real-time strength sliders, total strength
label, and picker sheet filtered to compatible available adapters."
```

---

### Task 12: AdaptersSettingsView Segmented Control

**Files:**
- Modify: `AnvilApp/UI/Settings/AdaptersSettingsView.swift`

- [ ] **Step 1: Replace AdaptersSettingsView with segmented version**

Replace the entire contents of `AdaptersSettingsView.swift`:

```swift
import SwiftUI
import AnvilCore

struct AdaptersSettingsView: View {
    enum Tab: String, CaseIterable {
        case myAdapters = "My Adapters"
        case community = "Community"
    }

    @State private var selectedTab: Tab = .myAdapters
    @StateObject private var communityService = CommunityService()

    var body: some View {
        VStack(spacing: 0) {
            // Segmented control
            Picker("", selection: $selectedTab) {
                ForEach(Tab.allCases, id: \.self) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()

            // Tab content
            switch selectedTab {
            case .myAdapters:
                MyAdaptersListView()
            case .community:
                CommunityBrowserView(service: communityService)
            }
        }
    }
}

// MARK: - My Adapters (extracted from original)

private struct MyAdaptersListView: View {
    @State private var adapters: [LoRAAdapter] = []
    @State private var showingMergeSheet = false
    @State private var renamingAdapter: LoRAAdapter?
    @State private var renameText: String = ""
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Toolbar
            HStack {
                Text("LoRA Adapters")
                    .font(.headline)
                Spacer()
                Button("Merge Two") {
                    showingMergeSheet = true
                }
                .disabled(adapters.count < 2)
            }
            .padding()

            Divider()

            if adapters.isEmpty {
                emptyState
            } else {
                adapterList
            }

            // Error banner
            if let error = errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                    Text(error)
                    Spacer()
                    Button("Dismiss") { errorMessage = nil }
                        .buttonStyle(.borderless)
                }
                .font(.caption)
                .foregroundStyle(.red)
                .padding(8)
                .background(.red.opacity(0.05))
            }
        }
        .onAppear { refreshAdapters() }
        .sheet(isPresented: $showingMergeSheet) {
            MergeSheet(onComplete: { refreshAdapters() })
                .frame(minWidth: 400, minHeight: 300)
        }
        .alert("Rename Adapter", isPresented: Binding(
            get: { renamingAdapter != nil },
            set: { if !$0 { renamingAdapter = nil } }
        )) {
            TextField("New name", text: $renameText)
            Button("Cancel", role: .cancel) { renamingAdapter = nil }
            Button("Rename") { performRename() }
        } message: {
            Text("Enter a new name for \"\(renamingAdapter?.name ?? "")\"")
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: "puzzlepiece.extension")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("No Adapters Yet")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Train one from History or download from the Community tab.")
                .font(.caption)
                .foregroundStyle(.tertiary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding()
    }

    private var adapterList: some View {
        List {
            ForEach(adapters) { adapter in
                adapterRow(adapter)
                    .contextMenu {
                        Button("Rename...") {
                            renameText = adapter.name
                            renamingAdapter = adapter
                        }
                        Button("Export...") { beginExport(adapter) }
                        Divider()
                        Button("Delete", role: .destructive) { deleteAdapter(adapter) }
                    }
            }
            .onDelete { indexSet in
                for idx in indexSet {
                    deleteAdapter(adapters[idx])
                }
            }
        }
    }

    private func adapterRow(_ adapter: LoRAAdapter) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(adapter.name)
                        .font(.body)
                        .fontWeight(.medium)
                    // Source badge
                    if adapter.source != .local {
                        Text(adapter.source.rawValue.capitalized)
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(.blue.opacity(0.1))
                            .clipShape(Capsule())
                    }
                }
                HStack(spacing: 8) {
                    Text(adapter.baseModelCategory.rawValue)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.blue.opacity(0.1))
                        .clipShape(Capsule())
                    Text("Rank \(adapter.config.rank)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let loss = adapter.finalLoss {
                        Text("Loss: \(String(format: "%.4f", loss))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            Text(adapter.createdAt, style: .date)
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
    }

    // MARK: - Actions

    private func refreshAdapters() {
        adapters = AdapterRegistry.listAdapters()
    }

    private func performRename() {
        guard let adapter = renamingAdapter else { return }
        do {
            try AdapterRegistry.rename(adapterID: adapter.id, to: renameText)
            refreshAdapters()
        } catch {
            errorMessage = error.localizedDescription
        }
        renamingAdapter = nil
    }

    private func deleteAdapter(_ adapter: LoRAAdapter) {
        do {
            try AdapterRegistry.delete(adapterID: adapter.id)
            refreshAdapters()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func beginExport(_ adapter: LoRAAdapter) {
        let panel = NSSavePanel()
        panel.title = "Export Adapter"
        panel.nameFieldStringValue = adapter.name
        panel.canCreateDirectories = true

        if panel.runModal() == .OK, let url = panel.url {
            do {
                try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
                try AdapterRegistry.export(adapterID: adapter.id, to: url)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds. Note: `ModelCategory` must conform to `CaseIterable` for the filter picker. If it doesn't, add conformance or use a fixed list. Check the existing `ModelCategory` definition.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Settings/AdaptersSettingsView.swift
git commit -m "feat(app): evolve AdaptersSettingsView with segmented My Adapters / Community tabs

Extracts existing adapter list into MyAdaptersListView.
Adds source badges for community/huggingface adapters.
Updates empty state text to reference Community tab."
```

---

### Task 13: TrainingService Stack Clear/Restore

**Files:**
- Modify: `AnvilApp/Services/TrainingService.swift`

- [ ] **Step 1: Add stack manager reference and clear/restore logic**

Add an `AdapterStackManager` property and modify `startTraining` to clear/restore the stack. Update the class as follows:

At the top of `TrainingService`, add the property:

```swift
private let stackManager: AdapterStackManager?
private var savedStack: [(adapter: LoRAAdapter, strength: Float)] = []
```

Update the `init` to accept the stack manager:

```swift
init(
    pipeline: MusicPipeline,
    vae: VAEActor,
    conditioning: ConditioningActor,
    registry: ModelRegistry,
    stackManager: AdapterStackManager? = nil
) {
    self.pipeline = pipeline
    self.vae = vae
    self.conditioning = conditioning
    self.registry = registry
    self.stackManager = stackManager
}
```

At the beginning of `startTraining`, after `trainingTask = Task { [weak self] in`, add stack clearing:

```swift
// Clear stack during training (restore after)
if let sm = self.stackManager, sm.isStacking {
    self.savedStack = sm.additionalAdapters
    sm.clearStack()
    // Toast would be triggered by the UI observing the state change
}
```

At the end of `startTraining`, before `self.trainingTask = nil`, add stack restoration:

```swift
// Restore stack if it was cleared
if let sm = self.stackManager, !self.savedStack.isEmpty {
    for entry in self.savedStack {
        sm.addToStack(entry.adapter, strength: entry.strength)
    }
    self.savedStack = []
}
```

- [ ] **Step 2: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds. The `stackManager` parameter defaults to `nil`, so existing call sites don't need changes.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/TrainingService.swift
git commit -m "feat(app): clear and restore adapter stack during training

Stack is saved before training starts and restored after completion.
Prevents multi-adapter interference with single-adapter training mode."
```

---

### Task 14: GenerateView Integration

**Files:**
- Modify: `AnvilApp/UI/Components/GenerateView.swift`

- [ ] **Step 1: Add AdapterStackManager and AdapterStackView to GenerateView**

Add an environment object at the top of GenerateView's properties:

```swift
@EnvironmentObject private var stackManager: AdapterStackManager
```

Then modify the `adapterPickerSection` computed property to include the stack view below the picker:

```swift
private var adapterPickerSection: some View {
    VStack(alignment: .leading, spacing: 12) {
        AdapterPicker(
            selectedAdapterID: $selectedAdapterID,
            activeModelCategory: .aceStepDiT
        )
        .onChange(of: selectedAdapterID) { _, newValue in
            persistedAdapterID = newValue ?? ""
            generationService.selectedAdapterID = newValue
        }
        .onAppear {
            if !persistedAdapterID.isEmpty {
                selectedAdapterID = persistedAdapterID
                generationService.selectedAdapterID = persistedAdapterID
            }
        }

        // Multi-adapter stacking controls
        if selectedAdapterID != nil || stackManager.isStacking {
            AdapterStackView(
                stackManager: stackManager,
                activeModelCategory: .aceStepDiT
            )
        }
    }
}
```

- [ ] **Step 2: Ensure AdapterStackManager is injected into the environment**

Check the app entry point (`AnvilAppApp.swift` or wherever environment objects are injected). Add `.environmentObject(adapterStackManager)` where the view hierarchy is set up. The exact location depends on where other environment objects are injected. Look for patterns like:

```swift
.environmentObject(generationService)
```

And add nearby:

```swift
.environmentObject(adapterStackManager)
```

With a property:

```swift
@StateObject private var adapterStackManager = AdapterStackManager()
```

- [ ] **Step 3: Verify build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`

Expected: Build succeeds (or only pre-existing errors).

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/UI/Components/GenerateView.swift AnvilApp/AnvilAppApp.swift
git commit -m "feat(app): integrate AdapterStackView into GenerateView

Shows stacking controls below primary AdapterPicker when an adapter
is selected or stacking is active. AdapterStackManager injected as
environment object from the app entry point."
```

---

## Build Verification

After all tasks are complete, run a final build:

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -20
```

Expected: Build succeeds with only pre-existing errors (AppState.swift async function type mismatch, SFXModelManager scope errors in GenerationService.swift/AnvilAppApp.swift). No new errors from Phase 10c code.

## Pre-existing Errors (Ignore)

These errors exist before Phase 10c and are unrelated:
- `AppState.swift` — async function type mismatch
- `MainShellView.swift` — missing PipelineChoice type
- `GenerationService.swift` / `AnvilAppApp.swift` — SFXModelManager scope errors
