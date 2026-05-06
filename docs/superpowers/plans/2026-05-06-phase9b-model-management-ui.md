# Phase 9b: Model Management UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Models settings tab for swapping diffusion models, with lazy swap runtime and HuggingFace repo validation for community models.

**Architecture:** `ModelSpec` gains a `category` field (`ModelCategory` enum). `ModelRegistry` tracks active models per category with JSON persistence. `GenerationService` detects mismatch at generation time and triggers unload → reload. New `ModelsSettingsView` and `AddModelView` in the Settings window.

**Tech Stack:** Swift, SwiftUI, AnvilCore (ModelRegistry/ModelSpec), MLX (weight loading/unloading)

---

## File Structure

### AnvilCore (Modified)

| File | Responsibility |
|------|---------------|
| `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift` | Add `ModelCategory` enum, add `category` field |
| `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` | `activeModels` tracking, JSON persistence, category-aware queries |
| `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift` | Add `unloadWeights()`, track `loadedModelID` |
| `AnvilCore/Sources/AnvilCore/Inference/VAE/VAEActor.swift` | Add `unloadWeights()`, track `loadedModelID` |

### AnvilApp (Modified)

| File | Responsibility |
|------|---------------|
| `AnvilApp/Services/GenerationService.swift` | Lazy swap detection, unload → reload flow |
| `AnvilApp/Services/SFXModelManager.swift` | Swap support: active model awareness, reload from new path |
| `AnvilApp/AnvilAppApp.swift` | Settings window `TabView` with Intelligence + Models tabs |

### AnvilApp (New)

| File | Responsibility |
|------|---------------|
| `AnvilApp/UI/Settings/ModelsSettingsView.swift` | Models tab — grouped model list, status, download/delete, active selection |
| `AnvilApp/UI/Settings/AddModelView.swift` | HF repo input, category picker, validation flow |

---

### Task 1: ModelSpec — Add ModelCategory Enum and Category Field

**Files:**

- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift`

- [ ] **Step 1: Add ModelCategory enum and category field to ModelSpec**

Replace the entire file with:

```swift
import Foundation

public enum ModelCategory: String, Codable, Sendable {
    case aceStepDiT
    case aceStepVAE
    case stableAudio
    case lyricLLM
}

public struct ModelSpec: Sendable, Identifiable, Codable {
    public let id: String
    public let huggingFaceRepo: String
    public let files: [String]
    public let sizeBytes: Int64
    /// SHA256 checksums keyed by filename. Empty dictionary means skip
    /// verification (used for user-added models without known hashes).
    public let sha256: [String: String]
    public let category: ModelCategory

    public init(
        id: String,
        huggingFaceRepo: String,
        files: [String],
        sizeBytes: Int64,
        sha256: [String: String] = [:],
        category: ModelCategory
    ) {
        self.id = id
        self.huggingFaceRepo = huggingFaceRepo
        self.files = files
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
        self.category = category
    }

    public var totalSizeFormatted: String {
        let gb = Double(sizeBytes) / 1_000_000_000
        return String(format: "%.1f GB", gb)
    }

    public func localDirectory(base: URL) -> URL {
        base.appending(path: id)
    }

    public func localPath(for file: String, base: URL) -> URL {
        localDirectory(base: base).appending(path: file)
    }
}
```

- [ ] **Step 2: Update builtInSpecs in ModelRegistry to include categories**

In `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift`,
update each `ModelSpec` in `builtInSpecs` to include the `category`
parameter:

```swift
public static let builtInSpecs: [ModelSpec] = [
    ModelSpec(
        id: "ace-step-dit",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-DiT",
        files: ["model.safetensors", "config.json"],
        sizeBytes: 3_200_000_000,
        sha256: [:],
        category: .aceStepDiT
    ),
    ModelSpec(
        id: "ace-step-vae",
        huggingFaceRepo: "ACEStep/ACE-Step-v1-5-VAE",
        files: ["vae.safetensors", "config.json"],
        sizeBytes: 800_000_000,
        sha256: [:],
        category: .aceStepVAE
    ),
    ModelSpec(
        id: "stable-audio-open",
        huggingFaceRepo: "stabilityai/stable-audio-open-1.0",
        files: ["model.safetensors", "config.json"],
        sizeBytes: 1_200_000_000,
        sha256: [:],
        category: .stableAudio
    ),
    ModelSpec(
        id: "lyric-llm",
        huggingFaceRepo: "microsoft/Phi-3-mini-4k-instruct",
        files: ["model.safetensors", "tokenizer.json", "config.json"],
        sizeBytes: 2_400_000_000,
        sha256: [:],
        category: .lyricLLM
    ),
]
```

- [ ] **Step 3: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift
git commit -m "feat(model): add ModelCategory enum and category field to ModelSpec"
```

---

### Task 2: ModelRegistry — Active Model Tracking with JSON Persistence

**Files:**

- Modify: `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift`

- [ ] **Step 1: Add activeModels tracking, persistence, and category-aware queries**

Add the following to `ModelRegistry` after the `essentialModelIDs`
constant and before the `// MARK: - Init` section:

```swift
// MARK: - Active Model Tracking

private(set) var activeModels: [ModelCategory: String] = [:]

private var activeModelsFileURL: URL {
    baseDirectory.appending(path: "active-models.json")
}
```

Update the `init` to also load active models. Change the init to:

```swift
public init(baseDirectory: URL, client: DownloadClient = DownloadClient()) {
    self.baseDirectory = baseDirectory
    self.client = client
    loadUserSpecs()
    loadActiveModels()
}
```

Add the following methods after the `// MARK: - User Model Management`
section and before `// MARK: - Persistence`:

```swift
// MARK: - Active Model Selection

public func setActiveModel(id: String, for category: ModelCategory) throws {
    guard let spec = allSpecs.first(where: { $0.id == id }) else {
        throw RegistryError.unknownModel(id)
    }
    guard spec.category == category else {
        throw RegistryError.categoryMismatch(expected: category, got: spec.category)
    }
    guard isModelReady(id) else {
        throw RegistryError.modelNotReady(id)
    }
    activeModels[category] = id
    saveActiveModels()
}

public func activeModelID(for category: ModelCategory) -> String? {
    activeModels[category]
}

public func activeSpec(for category: ModelCategory) -> ModelSpec? {
    guard let id = activeModels[category] else { return nil }
    return allSpecs.first(where: { $0.id == id })
}

public func specs(for category: ModelCategory) -> [ModelSpec] {
    allSpecs.filter { $0.category == category }
}
```

Add the persistence methods inside the existing `// MARK: - Persistence`
section:

```swift
private func loadActiveModels() {
    let url = activeModelsFileURL
    guard FileManager.default.fileExists(atPath: url.path()) else {
        // First launch — default to built-in models
        activeModels = [
            .aceStepDiT: "ace-step-dit",
            .aceStepVAE: "ace-step-vae",
            .stableAudio: "stable-audio-open",
            .lyricLLM: "lyric-llm"
        ]
        return
    }

    do {
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([String: String].self, from: data)
        activeModels = Dictionary(
            uniqueKeysWithValues: decoded.compactMap { key, value in
                guard let category = ModelCategory(rawValue: key) else { return nil }
                return (category, value)
            }
        )
    } catch {
        logger.error("Failed to load active-models.json: \(error.localizedDescription). Using defaults.")
        activeModels = [
            .aceStepDiT: "ace-step-dit",
            .aceStepVAE: "ace-step-vae",
            .stableAudio: "stable-audio-open",
            .lyricLLM: "lyric-llm"
        ]
    }
}

private func saveActiveModels() {
    let encoded = Dictionary(
        uniqueKeysWithValues: activeModels.map { ($0.key.rawValue, $0.value) }
    )
    do {
        let data = try JSONEncoder().encode(encoded)
        try data.write(to: activeModelsFileURL, options: .atomic)
    } catch {
        logger.error("Failed to save active-models.json: \(error.localizedDescription)")
    }
}
```

Add new error cases to `RegistryError`:

```swift
public enum RegistryError: Error, Sendable {
    case cannotModifyBuiltIn
    case duplicateModel(String)
    case unknownModel(String)
    case categoryMismatch(expected: ModelCategory, got: ModelCategory)
    case modelNotReady(String)
}
```

Also add a method to delete model files (for the settings UI):

```swift
// MARK: - File Management

public func deleteModelFiles(id: String) throws {
    guard let spec = allSpecs.first(where: { $0.id == id }) else {
        throw RegistryError.unknownModel(id)
    }
    let dir = spec.localDirectory(base: baseDirectory)
    if FileManager.default.fileExists(atPath: dir.path()) {
        try FileManager.default.removeItem(at: dir)
    }
    statuses[id] = .notDownloaded

    // If this was the active model, revert to built-in default
    if activeModels[spec.category] == id {
        let defaultID = Self.builtInSpecs.first(where: { $0.category == spec.category })?.id
        if let defaultID, defaultID != id, isModelReady(defaultID) {
            activeModels[spec.category] = defaultID
        } else if let defaultID {
            activeModels[spec.category] = defaultID
        }
        saveActiveModels()
    }
}
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 3: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift
git commit -m "feat(model): add active model tracking with JSON persistence and category queries"
```

---

### Task 3: DiTActor and VAEActor — Unload Weights and Model ID Tracking

**Files:**

- Modify: `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/VAE/VAEActor.swift`

- [ ] **Step 1: Add unloadWeights and loadedModelID to DiTActor**

Replace the entire `DiTActor.swift` with:

```swift
import Foundation
import MLX
import MLXNN

public actor DiTActor {
    private var decoder: DiTDecoder?
    private let decoderFactory: () -> DiTDecoder
    public private(set) var loadedModelID: String?

    public init(decoder: DiTDecoder) {
        self.decoder = decoder
        self.decoderFactory = { DiTDecoder() }
    }

    public func step(
        noisyLatents: MLXArray,
        timestep: Float,
        timestepR: Float,
        encoderStates: MLXArray,
        contextLatents: MLXArray,
        encoderAttentionMask: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> (output: MLXArray, cache: CrossAttentionCache?) {
        guard let decoder else {
            fatalError("DiTActor.step called without loaded weights")
        }
        let batchSize = noisyLatents.shape[0]
        let tArray = MLXArray.full([batchSize], values: MLXArray(timestep))
        let trArray = MLXArray.full([batchSize], values: MLXArray(timestepR))

        let (output, returnedCache) = decoder(
            hiddenStates: noisyLatents,
            timestep: tArray,
            timestepR: trArray,
            encoderHiddenStates: encoderStates,
            contextLatents: contextLatents,
            encoderAttentionMask: encoderAttentionMask,
            cache: cache,
            useCache: useCache
        )
        eval(output)
        return (output, returnedCache)
    }

    public func loadWeights(_ weights: [(String, MLXArray)], modelID: String) {
        if decoder == nil {
            decoder = decoderFactory()
        }
        decoder!.update(parameters: ModuleParameters.unflattened(weights))
        eval(decoder!.parameters())
        loadedModelID = modelID
    }

    /// Legacy overload for callers that don't track model ID yet.
    public func loadWeights(_ weights: [(String, MLXArray)]) {
        loadWeights(weights, modelID: "")
    }

    public func unloadWeights() {
        decoder = nil
        loadedModelID = nil
    }
}
```

Key changes:
1. `decoder` becomes `var` (optional) so it can be nilled for unload
2. Added `decoderFactory` closure to recreate decoder on reload
3. Added `loadedModelID` tracking
4. `loadWeights` has new overload accepting `modelID`
5. Legacy `loadWeights` kept for backward compatibility
6. `unloadWeights()` nils decoder and clears modelID
7. `step` guards on decoder presence

- [ ] **Step 2: Add unloadWeights and loadedModelID to VAEActor**

Replace the entire `VAEActor.swift` with:

```swift
import Foundation
import MLX
import MLXFast
import MLXNN

public actor VAEActor {
    private var vae: AutoEncoderOobleck?
    public private(set) var loadedModelID: String?

    public init(vae: AutoEncoderOobleck = AutoEncoderOobleck()) {
        self.vae = vae
    }

    public func decode(latents: MLXArray, precision: Precision) -> MLXArray {
        guard let vae else {
            fatalError("VAEActor.decode called without loaded weights")
        }
        let input = precision == .float16 ? latents.asType(.float16) : latents
        let audio = VAETiledDecode.decode(input, using: vae)
        return precision == .float16 ? audio.asType(.float32) : audio
    }

    public func encodeSample(audio: MLXArray, precision: Precision) -> MLXArray {
        guard let vae else {
            fatalError("VAEActor.encodeSample called without loaded weights")
        }
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeSample(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func encodeMean(audio: MLXArray, precision: Precision) -> MLXArray {
        guard let vae else {
            fatalError("VAEActor.encodeMean called without loaded weights")
        }
        let input = precision == .float16 ? audio.asType(.float16) : audio
        let latents = vae.encodeMean(input)
        return precision == .float16 ? latents.asType(.float32) : latents
    }

    public func loadWeights(_ weights: [(String, MLXArray)], modelID: String) {
        if vae == nil {
            vae = AutoEncoderOobleck()
        }
        vae!.update(parameters: ModuleParameters.unflattened(weights))
        eval(vae!.parameters())
        loadedModelID = modelID
    }

    /// Legacy overload for callers that don't track model ID yet.
    public func loadWeights(_ weights: [(String, MLXArray)]) {
        loadWeights(weights, modelID: "")
    }

    public func unloadWeights() {
        vae = nil
        loadedModelID = nil
    }
}
```

Key changes mirror DiTActor:
1. `vae` becomes optional for unload support
2. Added `loadedModelID` tracking
3. `loadWeights` overload with `modelID` parameter
4. `unloadWeights()` nils vae and clears modelID
5. All public methods guard on vae presence

- [ ] **Step 3: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 4: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift AnvilCore/Sources/AnvilCore/Inference/VAE/VAEActor.swift
git commit -m "feat(model): add unloadWeights and loadedModelID tracking to DiTActor and VAEActor"
```

---

### Task 4: GenerationService — Lazy Swap Detection

**Files:**

- Modify: `AnvilApp/Services/GenerationService.swift`

- [ ] **Step 1: Add registry dependency and lazy swap to GenerationService**

Add `registry` parameter to `GenerationService` and implement swap
detection. The key changes are:

1. Add `private let registry: ModelRegistry` property
2. Add `private var currentlyLoadedDiTModelID: String?` property
3. Add `private var currentlyLoadedVAEModelID: String?` property
4. Replace `weightsLoaded` boolean with model ID comparison
5. Add swap detection before generation

In `GenerationService`, add the registry property alongside existing
properties:

```swift
private let registry: ModelRegistry
private var currentlyLoadedDiTModelID: String?
private var currentlyLoadedVAEModelID: String?
```

Remove the `weightsLoaded` boolean.

Update the init to accept registry:

```swift
init(
    pipeline: MusicPipeline,
    conditioning: ConditioningActor,
    weightConverter: WeightConverter,
    historyStore: HistoryStore,
    outputDirectoryManager: OutputDirectoryManager,
    sfxModelManager: SFXModelManager,
    registry: ModelRegistry
) {
    self.pipeline = pipeline
    self.conditioning = conditioning
    self.weightConverter = weightConverter
    self.historyStore = historyStore
    self.outputDirectoryManager = outputDirectoryManager
    self.sfxModelManager = sfxModelManager
    self.registry = registry
}
```

Replace the weight loading check in `generateWithACEStep` (lines 171-175)
with lazy swap detection:

```swift
private func generateWithACEStep(request: GenerationRequest) async throws -> AudioResult {
    // Check DiT model swap
    let activeDiTID = await registry.activeModelID(for: .aceStepDiT)
    if activeDiTID != currentlyLoadedDiTModelID {
        if let name = await registry.activeSpec(for: .aceStepDiT)?.huggingFaceRepo {
            state = .loadingWeights(progress: 0)
        }
        try await loadDiTWeights()
    }

    // Check VAE model swap
    let activeVAEID = await registry.activeModelID(for: .aceStepVAE)
    if activeVAEID != currentlyLoadedVAEModelID {
        try await loadVAEWeights()
    }

    try Task.checkCancellation()
    state = .conditioning
    // ... rest of generation unchanged
```

Add separate weight loading methods:

```swift
private func loadDiTWeights() async throws {
    guard let spec = await registry.activeSpec(for: .aceStepDiT) else {
        throw GenerationError.modelNotConfigured("ACE-Step DiT")
    }
    let baseDir = await registry.baseDirectory
    state = .loadingWeights(progress: 0)

    // Unload existing weights if any
    await pipeline.unloadDiTWeights()

    // Load new weights from spec path
    let weightsPath = spec.localPath(for: "model.safetensors", base: baseDir)
    let weights = try weightConverter.loadSafetensors(from: weightsPath)
    await pipeline.loadDiTWeights(weights, modelID: spec.id)
    currentlyLoadedDiTModelID = spec.id
}

private func loadVAEWeights() async throws {
    guard let spec = await registry.activeSpec(for: .aceStepVAE) else {
        throw GenerationError.modelNotConfigured("ACE-Step VAE")
    }
    let baseDir = await registry.baseDirectory

    // Unload existing weights if any
    await pipeline.unloadVAEWeights()

    // Load new weights from spec path
    let vaeFile = spec.files.first(where: { $0.contains("vae") }) ?? "vae.safetensors"
    let weightsPath = spec.localPath(for: vaeFile, base: baseDir)
    let weights = try weightConverter.loadSafetensors(from: weightsPath)
    await pipeline.loadVAEWeights(weights, modelID: spec.id)
    currentlyLoadedVAEModelID = spec.id
}
```

Add the new error case to `GenerationError`:

```swift
enum GenerationError: Error, LocalizedError {
    case pipelineNotReady
    case modelNotConfigured(String)

    var errorDescription: String? {
        switch self {
        case .pipelineNotReady: return "Sound FX pipeline is not ready"
        case .modelNotConfigured(let name): return "\(name) model is not configured"
        }
    }
}
```

- [ ] **Step 2: Add weight forwarding methods to MusicPipeline**

In `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift`,
add forwarding methods so GenerationService can reach the actors:

```swift
public func loadDiTWeights(_ weights: [(String, MLXArray)], modelID: String) async {
    await diffusion.loadDiTWeights(weights, modelID: modelID)
}

public func unloadDiTWeights() async {
    await diffusion.unloadDiTWeights()
}

public func loadVAEWeights(_ weights: [(String, MLXArray)], modelID: String) async {
    await vae.loadWeights(weights, modelID: modelID)
}

public func unloadVAEWeights() async {
    await vae.unloadWeights()
}
```

Check if `DiffusionActor` needs similar forwarding methods. Find
`DiffusionActor.swift` and add if needed:

```swift
public func loadDiTWeights(_ weights: [(String, MLXArray)], modelID: String) async {
    await dit.loadWeights(weights, modelID: modelID)
}

public func unloadDiTWeights() async {
    await dit.unloadWeights()
}
```

- [ ] **Step 3: Update AppServices to pass registry to GenerationService**

In `AnvilApp/AnvilAppApp.swift`, update `makeGenerationService` to accept
and pass registry:

```swift
private static func makeGenerationService(
    historyStore: HistoryStore,
    outputManager: OutputDirectoryManager,
    sfxModelManager: SFXModelManager,
    registry: ModelRegistry
) -> GenerationService {
    let vae = VAEActor()
    let dit = DiTActor(decoder: DiTDecoder())
    let diffusion = DiffusionActor(dit: dit)
    let pipeline = MusicPipeline(vae: vae, diffusion: diffusion)
    let conditioning = ConditioningActor.forTesting(
        tokenizerURL: Bundle.main.url(forResource: "tokenizer", withExtension: "json")
            ?? FileManager.default.temporaryDirectory.appending(path: "tokenizer.json")
    )
    let converter = WeightConverter()

    return GenerationService(
        pipeline: pipeline,
        conditioning: conditioning,
        weightConverter: converter,
        historyStore: historyStore,
        outputDirectoryManager: outputManager,
        sfxModelManager: sfxModelManager,
        registry: registry
    )
}
```

Update all call sites of `makeGenerationService` in `AppServices` to
pass `registry`. `AppServices` will need a reference to the registry —
add it as a stored property and pass it through from `AnvilAppApp.init`.

- [ ] **Step 4: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 5: Commit**

```bash
git add AnvilApp/Services/GenerationService.swift AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift AnvilApp/AnvilAppApp.swift
git commit -m "feat(model): add lazy swap detection in GenerationService with weight reload"
```

Note: Also `git add` the DiffusionActor file if it was modified.

---

### Task 5: SFXModelManager — Swap Support

**Files:**

- Modify: `AnvilApp/Services/SFXModelManager.swift`

- [ ] **Step 1: Add active model awareness to SFXModelManager**

The SFX pipeline currently hardcodes `modelID` and `repoID`. To support
swapping, SFXModelManager needs to check the active model from the
registry and reload from the correct path.

Add a `registry` dependency and active model checking:

```swift
@MainActor
final class SFXModelManager: ObservableObject {
    @Published var downloadState: SFXDownloadState = .notDownloaded
    @Published var pipeline: SoundFXPipeline?

    private var loadedModelID: String?
    private let registry: ModelRegistry

    var modelPath: URL {
        modelPathFor(id: currentModelID)
    }

    private var currentModelID: String {
        // Will be resolved at runtime from registry
        "stable-audio-open"
    }

    var isReady: Bool { pipeline != nil }

    init(registry: ModelRegistry) {
        self.registry = registry
        if isModelDownloaded() {
            downloadState = .downloaded
        }
    }
```

Add a method to check if a swap is needed and handle it:

```swift
func ensureActiveModel() async throws {
    let activeID = await registry.activeModelID(for: .stableAudio)
    guard let activeID else { return }

    if activeID != loadedModelID && pipeline != nil {
        // Active model changed — unload current pipeline
        deleteWeights()
    }

    if pipeline == nil {
        let spec = await registry.activeSpec(for: .stableAudio)
        guard let spec else { return }
        let baseDir = await registry.baseDirectory
        let dir = spec.localDirectory(base: baseDir)

        // Check if model files exist at new location
        let hasFiles = spec.files.allSatisfy { file in
            FileManager.default.fileExists(
                atPath: dir.appending(path: file).path()
            )
        }

        if hasFiles {
            downloadState = .downloaded
            try await loadPipeline()
            loadedModelID = activeID
        }
    }
}
```

Update `loadPipeline` to use the active model's path instead of
hardcoded path. The model path should come from the registry's active
spec.

Update `AnvilAppApp.swift` init to pass registry when creating
`SFXModelManager`:

```swift
let sfxManager = SFXModelManager(registry: appState.registry)
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/SFXModelManager.swift AnvilApp/AnvilAppApp.swift
git commit -m "feat(model): add swap support to SFXModelManager with active model awareness"
```

---

### Task 6: ModelsSettingsView — Grouped Model List

**Files:**

- Create: `AnvilApp/UI/Settings/ModelsSettingsView.swift`

- [ ] **Step 1: Create the ModelsSettingsView**

```swift
import SwiftUI
import AnvilCore

struct ModelsSettingsView: View {
    @EnvironmentObject private var appState: AppState

    @State private var aceStepSpecs: [ModelSpec] = []
    @State private var stableAudioSpecs: [ModelSpec] = []
    @State private var activeModels: [ModelCategory: String] = [:]
    @State private var statuses: [String: ModelStatus] = [:]
    @State private var showDeleteConfirmation: String?
    @State private var showRemoveConfirmation: String?

    var body: some View {
        Form {
            aceStepSection
            stableAudioSection
            AddModelView(
                registry: appState.registry,
                onModelAdded: { refreshModels() }
            )
        }
        .formStyle(.grouped)
        .frame(minWidth: 500, minHeight: 400)
        .task { refreshModels() }
    }

    private var aceStepSection: some View {
        Section("ACE-Step (Music)") {
            ForEach(aceStepSpecs) { spec in
                ModelRow(
                    spec: spec,
                    isActive: activeModels[spec.category] == spec.id,
                    status: statuses[spec.id] ?? .notDownloaded,
                    isBuiltIn: appState.registry.isBuiltIn(id: spec.id),
                    onSetActive: { setActive(spec) },
                    onDownload: { download(spec) },
                    onDelete: { showDeleteConfirmation = spec.id },
                    onRemove: { showRemoveConfirmation = spec.id }
                )
            }
        }
        .confirmationDialog(
            "Delete model files?",
            isPresented: Binding(
                get: { showDeleteConfirmation != nil },
                set: { if !$0 { showDeleteConfirmation = nil } }
            )
        ) {
            if let id = showDeleteConfirmation {
                Button("Delete Files", role: .destructive) {
                    deleteFiles(id: id)
                }
            }
        } message: {
            Text("You can re-download later.")
        }
        .confirmationDialog(
            "Remove model?",
            isPresented: Binding(
                get: { showRemoveConfirmation != nil },
                set: { if !$0 { showRemoveConfirmation = nil } }
            )
        ) {
            if let id = showRemoveConfirmation {
                Button("Remove", role: .destructive) {
                    removeModel(id: id)
                }
            }
        } message: {
            Text("This will delete all downloaded files and remove the model from your list.")
        }
    }

    private var stableAudioSection: some View {
        Section("Stable Audio (SFX)") {
            ForEach(stableAudioSpecs) { spec in
                ModelRow(
                    spec: spec,
                    isActive: activeModels[spec.category] == spec.id,
                    status: statuses[spec.id] ?? .notDownloaded,
                    isBuiltIn: appState.registry.isBuiltIn(id: spec.id),
                    onSetActive: { setActive(spec) },
                    onDownload: { download(spec) },
                    onDelete: { showDeleteConfirmation = spec.id },
                    onRemove: { showRemoveConfirmation = spec.id }
                )
            }
        }
    }

    private func refreshModels() {
        Task {
            let registry = appState.registry
            aceStepSpecs = await registry.specs(for: .aceStepDiT)
                + await registry.specs(for: .aceStepVAE)
            stableAudioSpecs = await registry.specs(for: .stableAudio)

            var newStatuses: [String: ModelStatus] = [:]
            for spec in aceStepSpecs + stableAudioSpecs {
                newStatuses[spec.id] = await registry.status(for: spec.id)
            }
            statuses = newStatuses

            activeModels = [:]
            for cat: ModelCategory in [.aceStepDiT, .aceStepVAE, .stableAudio] {
                if let id = await registry.activeModelID(for: cat) {
                    activeModels[cat] = id
                }
            }
        }
    }

    private func setActive(_ spec: ModelSpec) {
        Task {
            try? await appState.registry.setActiveModel(id: spec.id, for: spec.category)
            refreshModels()
        }
    }

    private func download(_ spec: ModelSpec) {
        Task {
            try? await appState.registry.downloadModels([spec]) { [weak self] _, progress in
                Task { @MainActor in
                    self?.statuses[spec.id] = .downloading(progress: progress)
                }
            }
            refreshModels()
        }
    }

    private func deleteFiles(id: String) {
        Task {
            try? await appState.registry.deleteModelFiles(id: id)
            refreshModels()
        }
    }

    private func removeModel(id: String) {
        Task {
            try? await appState.registry.deleteModelFiles(id: id)
            try? await appState.registry.removeModel(id: id)
            refreshModels()
        }
    }
}

// MARK: - Model Row

private struct ModelRow: View {
    let spec: ModelSpec
    let isActive: Bool
    let status: ModelStatus
    let isBuiltIn: Bool
    let onSetActive: () -> Void
    let onDownload: () -> Void
    let onDelete: () -> Void
    let onRemove: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(displayName)
                        .font(.body)
                        .fontWeight(isActive ? .semibold : .regular)
                    if isActive {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.accent)
                            .font(.caption)
                    }
                }
                Text(spec.totalSizeFormatted)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            statusBadge

            actions
        }
        .padding(.vertical, 2)
        .listRowBackground(isActive ? Color.accentColor.opacity(0.08) : nil)
    }

    private var displayName: String {
        // Derive readable name from repo: "ACEStep/ACE-Step-v1-5-DiT" → "ACE Step v1 5 DiT"
        let name = spec.huggingFaceRepo.components(separatedBy: "/").last ?? spec.id
        return name.replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch status {
        case .ready:
            Text("Ready")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.green.opacity(0.15), in: Capsule())
                .foregroundStyle(.green)
        case .notDownloaded:
            Text("Not Downloaded")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.secondary.opacity(0.15), in: Capsule())
                .foregroundStyle(.secondary)
        case .downloading(let progress):
            ProgressView(value: progress)
                .frame(width: 60)
        case .verifying:
            Text("Verifying...")
                .font(.caption2)
                .foregroundStyle(.secondary)
        case .failed(let message):
            Text(message)
                .font(.caption2)
                .foregroundStyle(.red)
                .lineLimit(1)
        }
    }

    @ViewBuilder
    private var actions: some View {
        switch status {
        case .ready:
            HStack(spacing: 4) {
                if !isActive {
                    Button("Set Active") { onSetActive() }
                        .controlSize(.small)
                }
                Button(role: .destructive) { onDelete() } label: {
                    Image(systemName: "trash")
                }
                .controlSize(.small)
                if !isBuiltIn {
                    Button(role: .destructive) { onRemove() } label: {
                        Image(systemName: "xmark.circle")
                    }
                    .controlSize(.small)
                    .help("Remove model from list")
                }
            }
        case .notDownloaded:
            HStack(spacing: 4) {
                Button("Download") { onDownload() }
                    .controlSize(.small)
                if !isBuiltIn {
                    Button(role: .destructive) { onRemove() } label: {
                        Image(systemName: "xmark.circle")
                    }
                    .controlSize(.small)
                    .help("Remove model from list")
                }
            }
        case .failed:
            Button("Retry") { onDownload() }
                .controlSize(.small)
        default:
            EmptyView()
        }
    }
}
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Settings/ModelsSettingsView.swift
git commit -m "feat(ui): add ModelsSettingsView with grouped model list and active selection"
```

---

### Task 7: AddModelView — HuggingFace Repo Validation

**Files:**

- Create: `AnvilApp/UI/Settings/AddModelView.swift`

- [ ] **Step 1: Create AddModelView with HF validation**

```swift
import SwiftUI
import AnvilCore

struct AddModelView: View {
    let registry: ModelRegistry
    let onModelAdded: () -> Void

    @State private var repoID: String = ""
    @State private var selectedCategory: ModelCategory = .aceStepDiT
    @State private var isValidating: Bool = false
    @State private var validationError: String?
    @State private var validatedSpec: ModelSpec?
    @State private var showConfirmation: Bool = false

    private var categoryOptions: [(ModelCategory, String)] {
        [
            (.aceStepDiT, "ACE-Step DiT (Music)"),
            (.stableAudio, "Stable Audio (SFX)")
        ]
    }

    var body: some View {
        Section("Add Custom Model") {
            HStack {
                TextField("HuggingFace Repo ID (e.g., username/model-name)", text: $repoID)
                    .textFieldStyle(.roundedBorder)
                    .disabled(isValidating)

                Picker("Category", selection: $selectedCategory) {
                    ForEach(categoryOptions, id: \.0) { category, label in
                        Text(label).tag(category)
                    }
                }
                .frame(width: 200)
                .disabled(isValidating)

                Button(action: validate) {
                    if isValidating {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Text("Validate & Add")
                    }
                }
                .disabled(repoID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isValidating)
            }

            if let error = validationError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
        .alert("Add Model", isPresented: $showConfirmation) {
            Button("Add & Download") { confirmAdd() }
            Button("Cancel", role: .cancel) { validatedSpec = nil }
        } message: {
            if let spec = validatedSpec {
                Text("Add \(inferDisplayName(from: repoID))?\nSize: \(spec.totalSizeFormatted)")
            }
        }
    }

    private func validate() {
        let trimmed = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isValidating = true
        validationError = nil

        Task {
            do {
                // Check for duplicate
                let existingSpecs = await registry.allSpecs
                if existingSpecs.contains(where: { $0.huggingFaceRepo == trimmed }) {
                    validationError = "This model is already added."
                    isValidating = false
                    return
                }

                let expectedFiles = expectedFiles(for: selectedCategory)
                var totalSize: Int64 = 0

                for file in expectedFiles {
                    let urlString = "https://huggingface.co/\(trimmed)/resolve/main/\(file)"
                    guard let url = URL(string: urlString) else {
                        validationError = "Invalid repo ID format."
                        isValidating = false
                        return
                    }

                    var request = URLRequest(url: url)
                    request.httpMethod = "HEAD"

                    let (_, response) = try await URLSession.shared.data(for: request)
                    guard let http = response as? HTTPURLResponse else {
                        validationError = "Could not reach HuggingFace. Check your connection."
                        isValidating = false
                        return
                    }

                    switch http.statusCode {
                    case 200:
                        if let contentLength = http.value(forHTTPHeaderField: "Content-Length"),
                           let size = Int64(contentLength) {
                            totalSize += size
                        }
                    case 401, 403:
                        validationError = "This repository is private or gated. Only public repositories are supported for now."
                        isValidating = false
                        return
                    case 404:
                        validationError = "Missing required file: \(file)"
                        isValidating = false
                        return
                    default:
                        validationError = "Unexpected error (HTTP \(http.statusCode)) checking \(file)."
                        isValidating = false
                        return
                    }
                }

                let id = trimmed.replacingOccurrences(of: "/", with: "-").lowercased()
                let spec = ModelSpec(
                    id: id,
                    huggingFaceRepo: trimmed,
                    files: expectedFiles,
                    sizeBytes: totalSize > 0 ? totalSize : 1_000_000_000,
                    category: selectedCategory
                )

                validatedSpec = spec
                showConfirmation = true
                isValidating = false
            } catch {
                validationError = "Could not reach HuggingFace. Check your connection."
                isValidating = false
            }
        }
    }

    private func confirmAdd() {
        guard let spec = validatedSpec else { return }
        Task {
            do {
                try await registry.addModel(spec)
                // Start download
                try? await registry.downloadModels([spec]) { _, _ in }
                repoID = ""
                validatedSpec = nil
                onModelAdded()
            } catch {
                validationError = error.localizedDescription
            }
        }
    }

    private func expectedFiles(for category: ModelCategory) -> [String] {
        switch category {
        case .aceStepDiT: return ["model.safetensors", "config.json"]
        case .aceStepVAE: return ["vae.safetensors", "config.json"]
        case .stableAudio: return ["model.safetensors", "config.json"]
        case .lyricLLM: return ["model.safetensors", "tokenizer.json", "config.json"]
        }
    }

    private func inferDisplayName(from repoID: String) -> String {
        let name = repoID.components(separatedBy: "/").last ?? repoID
        return name.replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }
}
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Settings/AddModelView.swift
git commit -m "feat(ui): add AddModelView with HuggingFace repo validation"
```

---

### Task 8: Settings Window TabView Upgrade

**Files:**

- Modify: `AnvilApp/AnvilAppApp.swift`

- [ ] **Step 1: Upgrade Settings window to TabView**

In `AnvilApp/AnvilAppApp.swift`, replace the `Settings` scene (currently
lines 70-76):

```swift
Settings {
    IntelligenceSettingsView()
        .environmentObject(modelManager)
        .environmentObject(apiKeyStore)
        .environmentObject(lyricWriterManager)
        .frame(minWidth: 500, minHeight: 400)
}
```

With:

```swift
Settings {
    TabView {
        IntelligenceSettingsView()
            .tabItem { Label("Intelligence", systemImage: "brain") }
            .environmentObject(modelManager)
            .environmentObject(apiKeyStore)
            .environmentObject(lyricWriterManager)

        ModelsSettingsView()
            .tabItem { Label("Models", systemImage: "cube.box") }
            .environmentObject(appState)
    }
    .frame(minWidth: 550, minHeight: 450)
}
```

Note: `appState` is already a `@StateObject` in `AnvilAppApp`, so it's
available to pass as an environment object. `ModelsSettingsView` accesses
the registry through `appState.registry`.

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **` (or only pre-existing errors)

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/AnvilAppApp.swift
git commit -m "feat(ui): upgrade Settings window to TabView with Intelligence + Models tabs"
```

---

## Verification

After all 8 tasks complete, verify the full build one final time:

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **` with no new warnings related to model
management files.

Key behavioral changes to verify manually:

1. Cmd+, opens Settings with two tabs: Intelligence and Models
2. Models tab shows ACE-Step and Stable Audio sections with built-in models
3. Each model row shows name, size, status badge, and active indicator
4. Tapping "Set Active" on a ready model updates the active selection
5. "Add Custom Model" validates HuggingFace repo and adds to registry
6. Deleting model files changes status to Not Downloaded
7. Removing a user model removes it from the list entirely
8. After swapping active model, next generation loads new weights
