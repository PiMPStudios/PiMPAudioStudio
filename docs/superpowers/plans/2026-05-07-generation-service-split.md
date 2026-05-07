# GenerationService Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 371-line `GenerationService` into three focused `@MainActor` classes (`SFXGenerationService`, `ACEStepGenerationService`, `LoRACoordinator`) behind a thin `GenerationService` facade, with zero changes to any view or public caller.

**Architecture:** Three internal `@MainActor final class` services, each emitting typed `AsyncThrowingStream` events. The facade (`GenerationService`) iterates streams, maps events to `GenerationState`, owns all `@Published` properties, and handles file I/O. `LoRACoordinator` holds a `ModelRegistry` reference and defensively self-checks the active DiT model on every adapter cycle. All three classes live in the `AnvilApp` target.

**Tech Stack:** Swift, SwiftUI, AnvilCore actors (`MusicPipeline`, `SoundFXPipeline`, `ModelRegistry`), `AsyncThrowingStream` (matches existing pattern in `MusicPipeline`/`SoundFXPipeline`)

---

## File Structure

| Action | File | Notes |
|---|---|---|
| Create | `AnvilApp/Services/SFXGenerationService.swift` | SFX pipeline readiness + Stable Audio diffusion |
| Create | `AnvilApp/Services/ACEStepGenerationService.swift` | DiT/VAE weight loading, conditioning lifecycle, ACE-Step diffusion |
| Create | `AnvilApp/Services/LoRACoordinator.swift` | Adapter inject/remove, stack application, defensive model-change detection |
| Rewrite | `AnvilApp/Services/GenerationService.swift` | Becomes thin facade; same public API |
| Modify | `AnvilApp/AnvilAppApp.swift` | `makeGenerationService()` builds sub-services |
| Modify | `AnvilApp.xcodeproj/project.pbxproj` | Register all 3 new files in build system |

**Unchanged:** `GenerateView`, `ModelsSettingsView`, `AppState`, `AdapterStackManager`, everything in `AnvilCore`.

---

## Key Context

Before you write any code, read these existing files:

- `AnvilApp/Services/GenerationService.swift` — you are splitting this; all logic comes from here
- `AnvilApp/AnvilAppApp.swift` — the `AppServices` class and `makeGenerationService()` you will update
- `AnvilCore/Sources/AnvilCore/Inference/StableAudio/SoundFXPipeline.swift` — `generate(text:duration:config:tokenizer:)` and `makeProgressStream()`
- `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift` — `generate(text:lyrics:referenceAudio:duration:config:conditioning:)` and `makeProgressStream()`
- `AnvilApp/Services/SFXModelManager.swift` — `@MainActor`, properties `isReady`, `pipeline`, `tokenizer`, methods `ensureActiveModel()`, `downloadIfNeeded()`, `loadPipeline()`
- `AnvilApp/Services/AdapterRegistry.swift` — static methods `listAdapters()` and `loadWeights(adapterID:)`

`GenerationError` (currently at the bottom of `GenerationService.swift`) stays there — all three new files can reference it since they share the `AnvilApp` module.

**Build verification:** `xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"` — look for `BUILD SUCCEEDED`.

---

## Task 1: Create `SFXGenerationService`

**Files:**

- Create: `AnvilApp/Services/SFXGenerationService.swift`

- [ ] **Step 1: Create the file**

```swift
// AnvilApp/Services/SFXGenerationService.swift
import Foundation
import AnvilCore
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "SFXGenerationService")

/// Handles Stable Audio / SFX generation. Internal to the GenerationService facade.
/// Emits typed events via AsyncThrowingStream; the facade maps them to GenerationState.
@MainActor
final class SFXGenerationService {
    enum Event {
        case downloadingModel(progress: Float)
        case generating(step: Int, totalSteps: Int, eta: TimeInterval)
        case decoding
        case result(AudioResult)
    }

    private let sfxModelManager: SFXModelManager

    init(sfxModelManager: SFXModelManager) {
        self.sfxModelManager = sfxModelManager
    }

    func generate(request: GenerationRequest) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await sfxModelManager.ensureActiveModel()
                    if !sfxModelManager.isReady {
                        continuation.yield(.downloadingModel(progress: 0))
                        try await sfxModelManager.downloadIfNeeded()
                        try await sfxModelManager.loadPipeline()
                    }
                    guard let pipeline = sfxModelManager.pipeline else {
                        throw GenerationError.pipelineNotReady
                    }
                    guard let tokenizer = sfxModelManager.tokenizer else {
                        throw GenerationError.pipelineNotReady
                    }

                    let totalSteps = 100
                    let clampedDuration = min(request.duration, 47.0)
                    let config = SFXDiffusionConfig(
                        steps: totalSteps,
                        guidanceScale: request.guidanceScale > 1.0 ? request.guidanceScale : 7.0,
                        seed: request.seed,
                        latentShape: [1, Int(ceil(clampedDuration * 44100.0 / 1024.0)), 64]
                    )

                    continuation.yield(.generating(step: 0, totalSteps: totalSteps, eta: 0))

                    let progressStream = await pipeline.makeProgressStream()
                    let progressTask = Task {
                        for await progress in progressStream {
                            if case .diffusion(let dp) = progress {
                                let perStep = dp.currentStep > 0
                                    ? dp.elapsedTime / Double(dp.currentStep) : 0
                                let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                                continuation.yield(.generating(
                                    step: dp.currentStep,
                                    totalSteps: dp.totalSteps,
                                    eta: eta
                                ))
                            }
                        }
                    }

                    let audioResult = try await pipeline.generate(
                        text: request.text,
                        duration: clampedDuration,
                        config: config,
                        tokenizer: tokenizer
                    )
                    progressTask.cancel()

                    continuation.yield(.decoding)
                    continuation.yield(.result(audioResult))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
```

> **Why `@MainActor`:** `SFXModelManager` is `@MainActor`, so accessing its properties (`isReady`, `pipeline`, `tokenizer`) without `await` requires being on the same actor. The `Task { }` inside the stream builder inherits the `@MainActor` context. Heavy GPU compute still runs off-main via the `SoundFXPipeline` actor internals.

- [ ] **Step 2: Do not register or build yet** — Task 4 registers all three new files at once.

---

## Task 2: Create `ACEStepGenerationService`

**Files:**

- Create: `AnvilApp/Services/ACEStepGenerationService.swift`

- [ ] **Step 1: Create the file**

```swift
// AnvilApp/Services/ACEStepGenerationService.swift
import Foundation
import AnvilCore
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "ACEStepGenerationService")

/// Handles ACE-Step generation: weight loading (DiT + VAE), ConditioningActor lifecycle,
/// and diffusion. Internal to the GenerationService facade.
@MainActor
final class ACEStepGenerationService {
    enum Event {
        case loadingWeights(progress: Float)
        case conditioning
        case generating(step: Int, totalSteps: Int, eta: TimeInterval)
        case decoding
        case result(AudioResult)
    }

    private let pipeline: MusicPipeline
    private var conditioning: ConditioningActor   // var — upgraded after DiT download
    private let weightConverter: WeightConverter
    private let registry: ModelRegistry

    private var currentlyLoadedDiTModelID: String?
    private var currentlyLoadedVAEModelID: String?

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,
        weightConverter: WeightConverter,
        registry: ModelRegistry
    ) {
        self.pipeline = pipeline
        self.conditioning = conditioning
        self.weightConverter = weightConverter
        self.registry = registry
    }

    /// On-demand: checks active model IDs vs loaded IDs, loads/swaps weights if stale, then generates.
    func generate(request: GenerationRequest) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Check DiT model
                    let activeDiTID = await registry.activeModelID(for: .aceStepDiT)
                    guard let activeDiTID else {
                        throw GenerationError.modelNotConfigured("ACE-Step DiT")
                    }
                    if activeDiTID != currentlyLoadedDiTModelID {
                        continuation.yield(.loadingWeights(progress: 0))
                        try await loadDiTWeights()
                    }

                    // Check VAE model
                    let activeVAEID = await registry.activeModelID(for: .aceStepVAE)
                    guard let activeVAEID else {
                        throw GenerationError.modelNotConfigured("ACE-Step VAE")
                    }
                    if activeVAEID != currentlyLoadedVAEModelID {
                        try await loadVAEWeights()
                    }

                    try Task.checkCancellation()
                    continuation.yield(.conditioning)

                    let config = DiffusionConfig(
                        seed: request.seed,
                        method: request.method,
                        guidanceScale: request.guidanceScale,
                        precision: request.precision
                    )

                    let totalSteps = config.steps ?? 8
                    continuation.yield(.generating(step: 0, totalSteps: totalSteps, eta: 0))

                    let progressStream = await pipeline.makeProgressStream()
                    let progressTask = Task {
                        for await progress in progressStream {
                            if case .diffusion(let dp) = progress {
                                let perStep = dp.currentStep > 0
                                    ? dp.elapsedTime / Double(dp.currentStep) : 0
                                let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                                continuation.yield(.generating(
                                    step: dp.currentStep,
                                    totalSteps: dp.totalSteps,
                                    eta: eta
                                ))
                            }
                        }
                    }

                    let audioResult = try await pipeline.generate(
                        text: request.text,
                        lyrics: request.lyrics,
                        referenceAudio: nil,
                        duration: request.duration,
                        config: config,
                        conditioning: conditioning
                    )
                    progressTask.cancel()

                    continuation.yield(.decoding)
                    continuation.yield(.result(audioResult))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Unloads all model weights. Call before model swap or on memory pressure.
    func unloadWeights() async {
        await pipeline.unloadDiTWeights()
        await pipeline.unloadVAEWeights()
        currentlyLoadedDiTModelID = nil
        currentlyLoadedVAEModelID = nil
    }

    // MARK: - Weight Loading

    private func loadDiTWeights() async throws {
        guard let spec = await registry.activeSpec(for: .aceStepDiT) else {
            throw GenerationError.modelNotConfigured("ACE-Step DiT")
        }
        let baseDir = await registry.baseDirectory
        let previousModelID = currentlyLoadedDiTModelID

        await pipeline.unloadDiTWeights()
        currentlyLoadedDiTModelID = nil

        do {
            let weightsPath = spec.localPath(for: "model.safetensors", base: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightsPath)
            let weights = try await weightConverter.convertDiTWeights(stateDict: rawWeights)
            await pipeline.loadDiTWeights(weights, modelID: spec.id)
            currentlyLoadedDiTModelID = spec.id

            // Upgrade stub ConditioningActor to one backed by the real downloaded tokenizer.
            let tokenizerPath = spec.localPath(for: "tokenizer.json", base: baseDir)
            if FileManager.default.fileExists(atPath: tokenizerPath.path()),
               let real = try? ConditioningActor.forTesting(tokenizerURL: tokenizerPath) {
                conditioning = real
            }
        } catch {
            if let previousModelID {
                try? await registry.setActiveModel(id: previousModelID, for: .aceStepDiT)
            }
            throw error
        }
    }

    private func loadVAEWeights() async throws {
        guard let spec = await registry.activeSpec(for: .aceStepVAE) else {
            throw GenerationError.modelNotConfigured("ACE-Step VAE")
        }
        let baseDir = await registry.baseDirectory
        let previousModelID = currentlyLoadedVAEModelID

        await pipeline.unloadVAEWeights()
        currentlyLoadedVAEModelID = nil

        do {
            let vaeFile = spec.files.first(where: { $0.contains("vae") }) ?? "vae.safetensors"
            let weightsPath = spec.localPath(for: vaeFile, base: baseDir)
            let rawWeights = try await weightConverter.loadSafetensors(from: weightsPath)
            let weights = try await weightConverter.convertVAEWeights(stateDict: rawWeights)
            await pipeline.loadVAEWeights(weights, modelID: spec.id)
            currentlyLoadedVAEModelID = spec.id
        } catch {
            if let previousModelID {
                try? await registry.setActiveModel(id: previousModelID, for: .aceStepVAE)
            }
            throw error
        }
    }
}
```

- [ ] **Step 2: Do not register or build yet** — Task 4 registers all three new files at once.

---

## Task 3: Create `LoRACoordinator`

**Files:**

- Create: `AnvilApp/Services/LoRACoordinator.swift`

- [ ] **Step 1: Create the file**

```swift
// AnvilApp/Services/LoRACoordinator.swift
import Foundation
import AnvilCore
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "LoRACoordinator")

/// Manages LoRA adapter injection and removal on MusicPipeline.
/// Defensively detects base DiT model changes and clears adapters automatically.
/// Non-throwing — LoRA failures are silent so generation still proceeds.
@MainActor
final class LoRACoordinator {
    private let pipeline: MusicPipeline
    private let registry: ModelRegistry
    private var currentlyInjectedAdapterID: String?
    private var lastKnownBaseModelID: String?

    init(pipeline: MusicPipeline, registry: ModelRegistry) {
        self.pipeline = pipeline
        self.registry = registry
    }

    /// Diffs the desired adapter against the current one and injects/removes as needed.
    /// If the base DiT model changed since last call, clears any existing adapter first.
    /// Non-throwing: failures are logged and generation proceeds without adapter.
    func ensureAdapter(selectedID: String?) async {
        // Defensive: clear if base model changed
        let activeBaseID = await registry.activeModelID(for: .aceStepDiT)
        if activeBaseID != lastKnownBaseModelID {
            if currentlyInjectedAdapterID != nil {
                await pipeline.removeLoRA()
                currentlyInjectedAdapterID = nil
            }
            lastKnownBaseModelID = activeBaseID
        }

        let desired = selectedID
        let current = currentlyInjectedAdapterID

        guard desired != current else { return }

        // Remove the currently injected adapter if present
        if current != nil {
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }

        // Inject the new adapter if one was selected
        guard let adapterID = desired else { return }

        do {
            let adapters = AdapterRegistry.listAdapters()
            guard let adapter = adapters.first(where: { $0.id == adapterID }) else {
                logger.warning("Adapter '\(adapterID)' not found in registry — skipping injection")
                return
            }
            let _ = try await pipeline.injectLoRA(config: adapter.config)
            let weights = try AdapterRegistry.loadWeights(adapterID: adapterID)
            try await pipeline.loadLoRAWeights(weights, adapterID: adapterID)
            currentlyInjectedAdapterID = adapterID
            logger.info("Injected LoRA adapter '\(adapterID)'")
        } catch {
            logger.error("LoRA injection failed for '\(adapterID)': \(error.localizedDescription)")
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }
    }

    /// Removes any injected adapter and resets all coordinator state.
    /// Call on unloadAllWeights() and training start.
    func clearAll() async {
        if currentlyInjectedAdapterID != nil {
            await pipeline.removeLoRA()
            currentlyInjectedAdapterID = nil
        }
        lastKnownBaseModelID = nil
    }
}
```

- [ ] **Step 2: Do not register or build yet** — Task 4 is next.

---

## Task 4: Register All Three New Files in Xcode Project

The Xcode project file (`project.pbxproj`) requires manual registration — creating `.swift` files on disk is not enough. Three things must be added for each file: a `PBXBuildFile` entry, a `PBXFileReference` entry, a reference inside the `Services` group, and a reference in `PBXSourcesBuildPhase`.

**Files:**

- Modify: `AnvilApp.xcodeproj/project.pbxproj`

- [ ] **Step 1: Verify the three files exist on disk**

```bash
ls AnvilApp/Services/SFXGenerationService.swift \
   AnvilApp/Services/ACEStepGenerationService.swift \
   AnvilApp/Services/LoRACoordinator.swift
```

Expected: all three paths printed without error.

- [ ] **Step 2: Run this Python script from the repo root to add all three**

```bash
python3 - <<'PYEOF'
import re

path = "AnvilApp.xcodeproj/project.pbxproj"
with open(path, "r") as f:
    content = f.read()

# --- New entries ---
new_build_files = """\t\tEE0001010101010101010101 /* SFXGenerationService.swift in Sources */ = {isa = PBXBuildFile; fileRef = EE0002010101010101010101 /* SFXGenerationService.swift */; };
\t\tB10001010101010101010101 /* ACEStepGenerationService.swift in Sources */ = {isa = PBXBuildFile; fileRef = B10002010101010101010101 /* ACEStepGenerationService.swift */; };
\t\tC10001010101010101010101 /* LoRACoordinator.swift in Sources */ = {isa = PBXBuildFile; fileRef = C10002010101010101010101 /* LoRACoordinator.swift */; };"""

new_file_refs = """\t\tEE0002010101010101010101 /* SFXGenerationService.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = SFXGenerationService.swift; sourceTree = "<group>"; };
\t\tB10002010101010101010101 /* ACEStepGenerationService.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = ACEStepGenerationService.swift; sourceTree = "<group>"; };
\t\tC10002010101010101010101 /* LoRACoordinator.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = LoRACoordinator.swift; sourceTree = "<group>"; };"""

new_group_children = """\t\t\t\tEE0002010101010101010101 /* SFXGenerationService.swift */,
\t\t\t\tB10002010101010101010101 /* ACEStepGenerationService.swift */,
\t\t\t\tC10002010101010101010101 /* LoRACoordinator.swift */,"""

new_sources_phase = """\t\t\t\tEE0001010101010101010101 /* SFXGenerationService.swift in Sources */,
\t\t\t\tB10001010101010101010101 /* ACEStepGenerationService.swift in Sources */,
\t\t\t\tC10001010101010101010101 /* LoRACoordinator.swift in Sources */,"""

# --- 1. PBXBuildFile section: insert after the last existing build file entry ---
anchor_build = 'AA0001050101010101010101 /* GenerationService.swift in Sources */'
content = content.replace(
    anchor_build,
    new_build_files + "\n\t\t" + anchor_build
)

# --- 2. PBXFileReference section: insert after GenerationService file ref ---
anchor_ref = 'AA0002050101010101010101 /* GenerationService.swift */'
content = content.replace(
    anchor_ref,
    new_file_refs + "\n\t\t" + anchor_ref
)

# --- 3. Services group children: insert before GenerationService.swift in group ---
anchor_group = 'AA0002050101010101010101 /* GenerationService.swift */,'
content = content.replace(
    anchor_group,
    new_group_children + "\n\t\t\t\t" + anchor_group
)

# --- 4. PBXSourcesBuildPhase: insert before GenerationService.swift in sources ---
anchor_sources = 'AA0001050101010101010101 /* GenerationService.swift in Sources */,'
content = content.replace(
    anchor_sources,
    new_sources_phase + "\n\t\t\t\t" + anchor_sources
)

with open(path, "w") as f:
    f.write(content)

print("Done. Verify with: grep -c 'SFXGenerationService\|ACEStepGenerationService\|LoRACoordinator' AnvilApp.xcodeproj/project.pbxproj")
PYEOF
```

- [ ] **Step 3: Verify the script added all entries**

```bash
grep -c "SFXGenerationService\|ACEStepGenerationService\|LoRACoordinator" AnvilApp.xcodeproj/project.pbxproj
```

Expected output: `12` (4 entries × 3 files)

- [ ] **Step 4: Build to verify the new files compile**

```bash
xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"
```

Expected: `BUILD SUCCEEDED` (the new files compile but nothing uses them yet — that's fine, no "unused" errors in Swift).

- [ ] **Step 5: Commit the three new files + project registration**

```bash
git add AnvilApp/Services/SFXGenerationService.swift \
        AnvilApp/Services/ACEStepGenerationService.swift \
        AnvilApp/Services/LoRACoordinator.swift \
        AnvilApp.xcodeproj/project.pbxproj
git commit -m "feat: add SFXGenerationService, ACEStepGenerationService, LoRACoordinator

Three focused @MainActor services extracted from GenerationService.
Each emits typed AsyncThrowingStream events; wired in Task 5.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Rewrite `GenerationService` as Thin Facade

The existing `GenerationService.swift` is replaced entirely. The public API (`state`, `currentResult`, `selectedAdapterID`, `generate()`, `cancel()`, `unloadAllWeights()`) stays identical — no callers change.

**Files:**

- Rewrite: `AnvilApp/Services/GenerationService.swift`

- [ ] **Step 1: Replace the entire file contents**

```swift
// AnvilApp/Services/GenerationService.swift
import Foundation
import AnvilCore
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "GenerationService")

struct GenerationResult: Sendable {
    let fileURL: URL
    let entry: HistoryEntry
}

/// Coordinates between ACEStepGenerationService, SFXGenerationService, and LoRACoordinator.
/// Owns all @Published UI state and handles file I/O. Public API is unchanged from the
/// monolithic version — all callers (GenerateView, AppState, ModelsSettingsView) are unaffected.
@MainActor
final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?
    var selectedAdapterID: String?

    private let aceStepService: ACEStepGenerationService
    private let sfxService: SFXGenerationService
    private let loraCoordinator: LoRACoordinator
    private let historyStore: HistoryStore
    private let outputDirectoryManager: OutputDirectoryManager
    private var generationTask: Task<Void, Never>?

    init(
        aceStepService: ACEStepGenerationService,
        sfxService: SFXGenerationService,
        loraCoordinator: LoRACoordinator,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager
    ) {
        self.aceStepService = aceStepService
        self.sfxService = sfxService
        self.loraCoordinator = loraCoordinator
        self.historyStore = historyStore
        self.outputDirectoryManager = outputDirectoryManager
    }

    func generate(request: GenerationRequest) {
        generationTask?.cancel()

        generationTask = Task { [weak self] in
            guard let self else { return }

            guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                self.state = .error("Prompt cannot be empty")
                return
            }

            guard let outputDir = self.outputDirectoryManager.directory else {
                self.state = .error("No output directory selected")
                return
            }

            do {
                let audioResult: AudioResult

                if shouldUseStableAudio(request: request) {
                    audioResult = try await runSFX(request: request)
                } else {
                    await loraCoordinator.ensureAdapter(selectedID: selectedAdapterID)
                    audioResult = try await runACEStep(request: request)
                }

                try Task.checkCancellation()

                let fileName = HistoryEntry.fileName(prompt: request.text)
                let fileURL = outputDir.appending(path: fileName)

                try WAVWriter.write(
                    samples: audioResult.samples,
                    sampleRate: audioResult.sampleRate,
                    to: fileURL
                )

                let entry = HistoryEntry(
                    id: UUID(),
                    createdAt: Date(),
                    prompt: request.text,
                    lyrics: request.lyrics,
                    duration: request.duration,
                    seed: request.seed ?? 0,
                    method: request.method.rawValue,
                    guidanceScale: request.guidanceScale,
                    precision: request.precision.rawValue,
                    fileName: fileName,
                    pipeline: nil,
                    originalPrompt: nil
                )

                self.historyStore.add(entry)
                self.currentResult = GenerationResult(fileURL: fileURL, entry: entry)
                self.state = .complete
            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .error(error.localizedDescription)
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        state = .cancelled
    }

    func unloadAllWeights() async {
        await aceStepService.unloadWeights()
        await loraCoordinator.clearAll()
    }

    // MARK: - Private routing

    private func shouldUseStableAudio(request: GenerationRequest) -> Bool {
        switch request.pipelineOverride {
        case .aceStep: return false
        case .stableAudio: return true
        case .auto:
            return request.resolvedMode == .soundFX || request.resolvedMode == .ambient
        }
    }

    /// Iterates the SFX stream, maps events to GenerationState, returns AudioResult.
    private func runSFX(request: GenerationRequest) async throws -> AudioResult {
        for try await event in sfxService.generate(request: request) {
            switch event {
            case .downloadingModel(let progress):
                state = .downloadingModel(progress: progress)
            case .generating(let step, let totalSteps, let eta):
                state = .generating(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                state = .decoding
            case .result(let audioResult):
                return audioResult
            }
        }
        throw GenerationError.pipelineNotReady
    }

    /// Iterates the ACE-Step stream, maps events to GenerationState, returns AudioResult.
    private func runACEStep(request: GenerationRequest) async throws -> AudioResult {
        for try await event in aceStepService.generate(request: request) {
            switch event {
            case .loadingWeights(let progress):
                state = .loadingWeights(progress: progress)
            case .conditioning:
                state = .conditioning
            case .generating(let step, let totalSteps, let eta):
                state = .generating(step: step, totalSteps: totalSteps, eta: eta)
            case .decoding:
                state = .decoding
            case .result(let audioResult):
                return audioResult
            }
        }
        throw GenerationError.pipelineNotReady
    }
}

// MARK: - Errors

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

- [ ] **Step 2: Build to verify the rewrite compiles**

```bash
xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"
```

Expected: errors about `GenerationService` missing its old `init` — these will be fixed in Task 6. If you see errors about types not found, check that the new service files from Tasks 1–3 are registered correctly.

> **Note:** The build will fail here because `AnvilAppApp.swift` still calls the old `makeGenerationService()` which constructs `GenerationService` with the old init signature. That's expected — fix it in Task 6.

- [ ] **Step 3: Commit the facade rewrite**

```bash
git add AnvilApp/Services/GenerationService.swift
git commit -m "refactor: gut GenerationService to thin facade

Delegates ACE-Step generation to ACEStepGenerationService, SFX
generation to SFXGenerationService, and LoRA injection to
LoRACoordinator. Public API unchanged.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Update `AnvilAppApp.swift` Wiring

`AppServices.makeGenerationService()` must build the three sub-services and pass them to the new `GenerationService` init.

**Files:**

- Modify: `AnvilApp/AnvilAppApp.swift:141-168`

- [ ] **Step 1: Replace `makeGenerationService` in `AppServices`**

Find this method in `AnvilApp/AnvilAppApp.swift` (starts around line 141):

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
        // Use a stub ConditioningActor at launch — no downloaded tokenizer required.
        // GenerationService swaps in the real one after the ACE-Step model is downloaded.
        guard let conditioning = try? ConditioningActor.stub() else {
            fatalError("Failed to create stub ConditioningActor at launch")
        }
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

Replace it with:

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

        // Stub ConditioningActor at launch — no downloaded tokenizer required.
        // ACEStepGenerationService upgrades to the real one after DiT model download.
        // ConditioningActor.stub() writes a minimal tokenizer to the temp directory.
        // Failure here means the OS temp directory is unavailable — extremely unlikely.
        guard let conditioning = try? ConditioningActor.stub() else {
            fatalError("Failed to create stub ConditioningActor — OS temp directory unavailable")
        }
        let converter = WeightConverter()

        let aceStep = ACEStepGenerationService(
            pipeline: pipeline,
            conditioning: conditioning,
            weightConverter: converter,
            registry: registry
        )
        let sfx = SFXGenerationService(sfxModelManager: sfxModelManager)
        let lora = LoRACoordinator(pipeline: pipeline, registry: registry)

        return GenerationService(
            aceStepService: aceStep,
            sfxService: sfx,
            loraCoordinator: lora,
            historyStore: historyStore,
            outputDirectoryManager: outputManager
        )
    }
```

> **Key:** `MusicPipeline` is passed to both `ACEStepGenerationService` (for generation) and `LoRACoordinator` (for injection). Both hold a reference to the same actor — safe because `MusicPipeline` is itself an actor.

- [ ] **Step 2: Build to verify everything compiles**

```bash
xcodebuild -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | grep -E "error:|BUILD"
```

Expected: `BUILD SUCCEEDED` with no errors related to this refactor. Pre-existing `AppState.swift` and `SFXModelManager` scope errors are expected and can be ignored.

- [ ] **Step 3: Smoke-test the app**

Run the app in Xcode (Cmd+R). Verify:

1. App launches (Welcome to Anvil screen appears)
2. No crash on launch
3. The "Generate" tab loads without errors in the console
4. If models are downloaded: tap Generate and verify the state progression (`conditioning` → `generating` → `complete`) still works

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/AnvilAppApp.swift
git commit -m "refactor: wire GenerationService facade in AnvilAppApp

makeGenerationService() now builds ACEStepGenerationService,
SFXGenerationService, and LoRACoordinator, passing them to the
thin GenerationService facade.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Checklist (for implementer)

After completing all tasks, verify:

- [ ] `GenerationService.generate()`, `.cancel()`, `.unloadAllWeights()`, `.state`, `.currentResult`, `.selectedAdapterID` — all present with identical signatures to before
- [ ] `GenerateView`, `ModelsSettingsView`, `AppState` — zero changes needed (confirm with `git diff`)
- [ ] `GenerationError` still defined in `GenerationService.swift` and accessible from sub-services (same module)
- [ ] `SFXGenerationService` yields `.downloadingModel` event when pipeline is not ready before calling `downloadIfNeeded()`
- [ ] `ACEStepGenerationService.loadDiTWeights()` upgrades `conditioning` var if `tokenizer.json` exists at the model path
- [ ] `LoRACoordinator.ensureAdapter()` clears the existing adapter when `lastKnownBaseModelID` changes
- [ ] `LoRACoordinator.clearAll()` resets both `currentlyInjectedAdapterID` and `lastKnownBaseModelID`
