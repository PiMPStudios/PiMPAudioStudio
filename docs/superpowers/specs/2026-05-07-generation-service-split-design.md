# GenerationService Split — Implementation Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 371-line `GenerationService` into three focused actors (`ACEStepGenerationService`, `SFXGenerationService`, `LoRACoordinator`) behind a thin `GenerationService` facade, improving testability and separation of concerns with zero changes to any view or public caller.

**Architecture:** Three internal actors emit typed `AsyncThrowingStream` progress events. The facade (`@MainActor ObservableObject`) iterates these streams, maps events to `GenerationState`, handles file I/O, and publishes UI state. `LoRACoordinator` holds a `ModelRegistry` reference and defensively self-checks the active DiT model on every adapter application cycle.

**Tech Stack:** Swift, SwiftUI, MLX, AsyncThrowingStream (existing pattern from `MusicPipeline`/`SoundFXPipeline`)

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LoRA injection location | Own `LoRACoordinator` actor | Clean separation from generation; independently testable |
| Progress state flow | `AsyncThrowingStream` + facade owns `@Published` | Matches existing pipeline pattern; sub-services stay pure |
| Weight loading timing | On-demand inside `generate()` | Lazy, matches current behavior; avoids explicit `prepareWeights()` call |
| Model-swap LoRA clear | `LoRACoordinator` defensive self-check | Queries `registry.activeModelID` on every cycle; no new stream event types needed |
| `LoRACoordinator` registry dependency | Injected | Self-sufficient, keeps facade call site simple |
| Facade name | Keep `GenerationService` | Zero changes to all callers (GenerateView, AppState, ModelsSettingsView, AnvilAppApp) |

---

## File Structure

### New Files (3)

| File | Responsibility |
|---|---|
| `AnvilApp/Services/ACEStepGenerationService.swift` | DiT/VAE weight loading, `ConditioningActor` lifecycle, ACE-Step diffusion |
| `AnvilApp/Services/SFXGenerationService.swift` | SFX pipeline readiness (download + load), Stable Audio diffusion |
| `AnvilApp/Services/LoRACoordinator.swift` | Adapter injection/removal, stack application, `currentlyInjectedAdapterID` state |

### Modified Files (2)

| File | Change |
|---|---|
| `AnvilApp/Services/GenerationService.swift` | Becomes thin facade: routing, file I/O, `@Published` state, delegates all generation work |
| `AnvilApp/AnvilAppApp.swift` | `makeGenerationService()` builds sub-services and wires them |

### Unchanged (zero-diff)

All views (`GenerateView`, `ModelsSettingsView`), `AppState`, `AdapterStackManager`, `AnvilAppApp` (init wiring only), all of `AnvilCore`.

---

## Sub-Service Interfaces

### `ACEStepGenerationService` (actor)

Owns: DiT/VAE weight loading, `ConditioningActor` lifecycle (stub → real upgrade after download), ACE-Step diffusion, weight-loading rollback on failure.

```swift
actor ACEStepGenerationService {
    enum Event {
        case loadingWeights(progress: Double)
        case conditioning
        case generating(step: Int, totalSteps: Int, eta: TimeInterval)
        case decoding
        case result(AudioResult)
    }

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,     // var internally — upgraded after DiT download
        weightConverter: WeightConverter,
        registry: ModelRegistry
    )

    /// On-demand: checks active model IDs, loads/swaps weights if stale, then generates.
    func generate(request: GenerationRequest) -> AsyncThrowingStream<Event, Error>

    /// Unloads DiT and VAE weights. Called by facade on model swap or memory pressure.
    func unloadWeights() async
}
```

**Weight loading behavior:**
- On each `generate()` call, checks `registry.activeModelID(for:)` against `currentlyLoadedDiTModelID` and `currentlyLoadedVAEModelID`.
- If either differs, loads new weights before generating (emitting `.loadingWeights` events).
- On DiT weight load success, upgrades `conditioning` from stub to real `ConditioningActor` if `tokenizer.json` exists at the model path.
- On DiT weight load failure, rolls back `registry.activeModel` to the previously working model ID.

---

### `SFXGenerationService` (actor)

Owns: SFX pipeline download (if needed), pipeline load (if needed), Stable Audio diffusion.

```swift
actor SFXGenerationService {
    enum Event {
        case downloadingModel(progress: Double)
        case generating(step: Int, totalSteps: Int, eta: TimeInterval)
        case decoding
        case result(AudioResult)
    }

    init(sfxModelManager: SFXModelManager)

    func generate(request: GenerationRequest) -> AsyncThrowingStream<Event, Error>
}
```

---

### `LoRACoordinator` (actor)

Owns: `currentlyInjectedAdapterID`, adapter diff logic, LoRA injection and removal.

```swift
actor LoRACoordinator {
    init(pipeline: MusicPipeline, registry: ModelRegistry)

    /// Diffs desired vs current adapter. Clears first if the base DiT model changed since last injection.
    /// Non-throwing: LoRA failures are silent — generation proceeds without adapter.
    func ensureAdapter(selectedID: String?) async

    /// Removes any injected adapter and resets all state. Called on unloadAllWeights() and training start.
    func clearAll() async
}
```

**Defensive self-check:** On every `ensureAdapter()` call, `LoRACoordinator` queries `registry.activeModelID(for: .aceStepDiT)`. If the active DiT model changed since last injection, it clears the current adapter before proceeding. This avoids leaking LoRA weights across model swaps without needing a dedicated stream event.

---

## Facade Contract

`GenerationService` retains its exact public API — no callers change.

```swift
@MainActor final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?
    var selectedAdapterID: String?

    init(
        aceStepService: ACEStepGenerationService,
        sfxService: SFXGenerationService,
        loraCoordinator: LoRACoordinator,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager
    )

    func generate(request: GenerationRequest)
    func cancel()
    func unloadAllWeights() async
}
```

### `generate(request:)` flow

```
1. Cancel any in-flight generationTask
2. Validate: prompt non-empty, outputDir set
3. Route: shouldUseStableAudio(request)?
   ├─ true  → iterate sfxService.generate(request) stream
   │           map SFXGenerationService.Event → GenerationState
   └─ false → await loraCoordinator.ensureAdapter(selectedID: selectedAdapterID)
               iterate aceStepService.generate(request) stream
               map ACEStepGenerationService.Event → GenerationState
4. On .result(audioResult):
   - Write WAV to outputDir
   - historyStore.add(entry)
   - self.currentResult = GenerationResult(fileURL:, entry:)
   - self.state = .complete
5. catch CancellationError → self.state = .cancelled
6. catch any other error   → self.state = .error(error.localizedDescription)
```

### `unloadAllWeights()`

```swift
func unloadAllWeights() async {
    await aceStepService.unloadWeights()
    await loraCoordinator.clearAll()
}
```

---

## Wiring (`AnvilAppApp.makeGenerationService()`)

`MusicPipeline` is constructed once and passed to both `ACEStepGenerationService` and `LoRACoordinator`. Both hold actor references to the same pipeline instance — safe because `MusicPipeline` is an actor.

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
    guard let conditioning = try? ConditioningActor.stub() else {
        fatalError("Failed to create stub ConditioningActor at launch")
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

---

## Error Handling & Cancellation

- **Uniform propagation:** Sub-service stream errors bubble up through `for try await` into the facade's single `catch` block.
- **Cancellation:** `generationTask?.cancel()` propagates `CancellationError` naturally. Inner `Task.checkCancellation()` calls in `MusicPipeline` and `SoundFXPipeline` terminate the stream. No additional plumbing required in sub-services.
- **LoRA injection:** Non-throwing. `ensureAdapter()` silently clears state on failure — a broken adapter does not abort generation.
- **Weight loading rollback:** On DiT load failure, `ACEStepGenerationService` reverts `registry.activeModel` to the previously working model ID (existing behavior, moved wholesale).

---

## Testing

Each sub-service is independently constructible and testable without the facade:

```swift
// ACEStepGenerationService — same pattern as existing VAEActorTests
let service = ACEStepGenerationService(
    pipeline: MusicPipeline(vae: VAEActor(), diffusion: DiffusionActor(...)),
    conditioning: try ConditioningActor.stub(),
    weightConverter: WeightConverter(),
    registry: registry
)
// stream emits expected .conditioning, .generating, .result events with zero weights

// SFXGenerationService — isolated from ACE-Step
let service = SFXGenerationService(sfxModelManager: sfxModelManager)

// LoRACoordinator — test inject/clear independently
let coordinator = LoRACoordinator(pipeline: pipeline, registry: registry)
await coordinator.ensureAdapter(selectedID: "some-adapter-id")
// verify: currentlyInjectedAdapterID set, pipeline received injectLoRA
await coordinator.clearAll()
// verify: currentlyInjectedAdapterID nil, pipeline received removeLoRA
```

The facade requires no new unit tests. Its logic is thin routing and state mapping; sub-service correctness is covered by sub-service tests.
