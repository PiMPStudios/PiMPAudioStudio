# Changelog

All notable changes to AnvilApp are documented here.

---

## [Unreleased] — GenerationService Split

### Added

- **`ACEStepGenerationService`** (`AnvilApp/Services/`) — `@MainActor final class` owning DiT/VAE weight loading, `ConditioningActor` stub→real upgrade after tokenizer download, weight-load rollback on failure, and ACE-Step diffusion. Emits typed `Event` stream (`.loadingWeights`, `.conditioning`, `.generating`, `.decoding`, `.result`) via `AsyncThrowingStream`
- **`SFXGenerationService`** (`AnvilApp/Services/`) — `@MainActor final class` owning SFX pipeline readiness (download + load) and Stable Audio diffusion. Emits typed `Event` stream (`.downloadingModel`, `.generating`, `.decoding`, `.result`)
- **`LoRACoordinator`** (`AnvilApp/Services/`) — `@MainActor final class` owning adapter injection/removal, `currentlyInjectedAdapterID` state, and a defensive base-model self-check: queries `registry.activeModelID(for: .aceStepDiT)` on every `ensureAdapter` call and clears the adapter if the base DiT changed since last injection
- **`HistoryEntry.pipeline` field populated** — now records `"stableAudio"` or `"aceStep"` instead of `nil`, so history views can distinguish which generation path produced each entry

### Changed

- **`GenerationService`** — gutted from 371 lines to ~150 line thin facade. Public API unchanged (`state`, `currentResult`, `selectedAdapterID`, `generate(request:)`, `cancel()`, `unloadAllWeights()`); zero changes to any caller (`GenerateView`, `AppState`, `ModelsSettingsView`, etc.). Internally routes to the three sub-services and maps their stream events to `@Published` state
- **`AppServices`** (`AnvilAppApp.swift`) — `init` now takes the real `OutputDirectoryManager` instead of allocating a throwaway one. Eliminates a full GPU pipeline allocation (`VAEActor` + `DiTActor` + `DiffusionActor` + `MusicPipeline` + stub `ConditioningActor`) per cold launch that was previously orphaned when `configure()` fired on first appear
- **`AppServices.configure()`** — rewritten with explicit "skip if manager and directory both unchanged" check; first `.task` fire is now a no-op instead of a full rebuild. Tracks `currentDirectory` separately so identity-only manager swaps don't conflate with directory changes
- **`AnvilAppApp.makeGenerationService()`** — wires `ACEStepGenerationService`, `SFXGenerationService`, and `LoRACoordinator` into the facade. `MusicPipeline` is shared between `ACEStepGenerationService` and `LoRACoordinator` (safe — both hold actor references). `ConditioningActor.stub()` failure path now uses `do/catch` with error interpolation instead of `try?` so crash logs include the underlying cause
- **`GenerationError.pipelineNotReady`** — description changed from SFX-specific `"Sound FX pipeline is not ready"` to generic `"Generation pipeline did not return a result"` (now thrown by both `runSFX` and `runACEStep` fallback paths)

### Fixed

- **`DownloadError`** — added `LocalizedError` conformance with per-HTTP-status-code descriptions (401 = "Unauthorized — authentication or model access may be required", 403 = "Forbidden — license acceptance required", 404 = "Not Found", 429 = "Rate limited", 5xx = "Server error", etc.). Previously surfaced as opaque `"AnvilCore.DownloadError error 0"` in error sheets when the user clicked Download Models against unavailable HuggingFace repos
- **Xcode project (`project.pbxproj`)** — registered the three new sub-service files with fresh unique UUIDs after a collision was detected with existing project entries that would have silently dropped them from the compilation plan
- **`SFXGenerationService` cancellation gap** — added `try Task.checkCancellation()` between pipeline-readiness setup and the `pipeline.generate(...)` call so a user cancellation immediately after weights load no longer triggers the GPU pipeline before unwinding
- **`ConditioningActor.forTesting(tokenizerURL:)` misnamed for production** — renamed to `make(tokenizerURL:)` since `ACEStepGenerationService` calls it on the production path (after a model's `tokenizer.json` is downloaded). The test-only naming was misleading; the new name matches its actual responsibility. The argumentless `stub()` factory is unchanged
- **`AdapterRegistry.loadWeights` blocking the main thread** — added an `async throws` overload that runs the safetensors read on `Task.detached(priority: .userInitiated)`. `LoRACoordinator.ensureAdapter` (`@MainActor`) now `await`s it, so reading multi-MB adapter files no longer stalls the UI thread. The existing synchronous version is retained for `merge()` and other off-main callers

---

## [Unreleased] — Build Fixes & Project Hygiene

### Fixed

- **`AppState.checkSetupStatus()`** — replaced `allSatisfy` with async closure (unsupported by Swift stdlib) with a sequential `for` loop that `await`s each `registry.isModelReady` call
- **`GenerationService`** — corrected wrong error case `.modelNotReady` → `.pipelineNotReady`; added missing `try` to `pipeline.injectLoRA(config:)` and `pipeline.loadLoRAWeights(_:adapterID:)` call sites
- **`MainShellView`** — added `import AnvilCore` so `PipelineChoice` resolves to `AnvilCore.PipelineChoice` (which already defines all cases + `displayName`); removed locally-duplicated `PipelineChoice` enum that would have caused a redeclaration conflict
- **`AudioBufferConverter`** — replaced unavailable `vDSP.convertElements(of:to:)` for `Float → Int16` with the correct `vDSP_vfix16(_:_:_:_:_:)` from Accelerate
- **`ModelsSettingsView`** — split chained `+ await` expression into sequential `let` bindings (Swift requires `await` as the outermost operator); replaced removed `ShapeStyle.accent` with `Color.accentColor`

### Changed

- **Xcode project (`project.pbxproj`)** — registered 22 Swift source files that existed on disk (committed to git) but were never added to the Xcode target: all 11 `UI/Editor/` views (`EditorView`, `EditorState`, `EditorToolbar`, `EditorPromptArea`, `EditorControlsBar`, `EditorWaveformView`, `DSPPaletteView`, `EQControlView`, `PitchShiftControlView`, `ReverbControlView`, `TimeStretchControlView`), all 8 `DSP/` files (`AVAudioEngineDSP`, `AudioBufferConverter`, `DSPParameters`, `DSPProcessor`, `LPCFormantShifter`, `Resampler`, `SimpleDSP`, `WSOLAProcessor`), plus `BackgroundDownloadIndicator`, `EditorService`, `LossCurveView`, `HuggingFaceImportSheet`, and `AdapterStackView`

---

## [Unreleased] — Polish & Hardening Phase 2

### Added

- **Release GPU Memory button** (`ModelsSettingsView`) — "Memory" section in Models Settings with a "Release" button that calls `AppState.unloadAllModels()`, freeing all ACE-Step and SFX weights without restarting
- **`AppState.unloadAllModels()`** — coordinating helper that calls `GenerationService.unloadAllWeights()` and `SFXModelManager.unloadPipeline()` in one shot
- **`GenerationService.unloadAllWeights()`** — unloads ACE-Step DiT and VAE weights and resets loaded model IDs; weights reload automatically on next generation
- **`SFXDiTActorError` / `SFXVAEActorError`** — typed error enums for SFX actors matching the existing `DiTActorError` / `VAEActorError` pattern

### Changed

- **`SFXDiTActor.predict()`** — now `throws SFXDiTActorError.weightsNotLoaded` when weights have not been loaded, instead of silently running against an uninitialized model
- **`SFXVAEActor.decode()` / `encode()`** — same: now `throws SFXVAEActorError.weightsNotLoaded`; added `unloadWeights()` to both SFX actors
- **`SFXDiffusionActor`** — all `dit.predict(...)` calls updated to `try await` (6 call sites across `sample` and `inpaint`)
- **`SoundFXPipeline`** — `vae.decode(...)` updated to `try await`
- **`WeightConverter.convertVAEWeights()` / `convertDiTWeights()`** — now `throws`; `try Task.checkCancellation()` added every 50 tensor iterations so cancellation interrupts multi-GB weight conversion mid-flight
- **`SFXModelManager.loadPipeline()`** — refactored into three private helpers (`loadedDiTWeights()`, `loadedVAEWeights()`, `loadedCLAPWeights()`) for readability; main method reduced to pipeline assembly
- **`SFXModelManager.baseModelsDirectory()`** — new `private static` helper eliminates duplicated App Support URL construction; `modelPath` simplified to 3 lines; `downloadIfNeeded()` now resolves the model path before the early-return guard so `modelPath` is correct even when files are already downloaded
- **`GenerationService.ensureSFXPipelineReady()`** — extracted from `generateWithStableAudio()`, encapsulating the active-model check, optional download, and pipeline load into a single returning helper
- **`CrossAttentionCache`** — added detailed doc comment explaining the `@unchecked Sendable` threading contract (one cache per generation, sequential access only, clean ownership transfer) and a warning against concurrent use

---

## [Unreleased] — Polish & Hardening Phase 1

### Fixed

- **Streaming SHA-256 verification** (`DownloadClient`) — replaced `Data(contentsOf:)` with 1 MB chunked `FileHandle` reads to prevent OOM crashes when verifying 3 GB+ model files
- **Single-consumer `AsyncStream` bug** (`DiffusionActor`, `MusicPipeline`, `SoundFXPipeline`, `WeightConverter`, `SFXDiffusionActor`) — lazy `var progress` was exhausted after first consumer; replaced with `func makeProgressStream() -> AsyncStream<…>` that creates a fresh stream per call; `continuation?.finish()` called at generation end to prevent leaks
- **`DiTActor` crash on bad state** — three `fatalError(…)` calls replaced with `throws DiTActorError.weightsNotLoaded`; callers updated throughout `DiffusionActor` and `MusicPipeline`
- **SFX pipeline weight loading** (`SFXModelManager`) — `loadPipeline()` was a stub; fully implemented weight loading via `WeightConverter.loadSafetensors` → `SFXDiTNameMap` / `SFXVAENameMap` / `CLAPNameMap` → actor `loadWeights()`; SFX VAE uses manual name mapping (not `convertVAEWeights()` which is music-VAE-specific)
- **`CLAPTokenizer` placeholder** (`GenerationService`) — `CLAPTokenizer.forTesting()` replaced with real tokenizer loaded from `SFXModelManager.tokenizer`
- **Qwen3 text mask ignored** (`ConditioningActor`) — `textMask` computed by tokenizer was silently discarded; `Qwen3Model.callAsFunction` now accepts optional `attentionMask` and converts binary mask to additive −1e9 mask passed to all 28 attention layers
- **`VAETiledDecode` force-unwrap** — `upsampleFactor: Double?` with `!` unwrap replaced with `var upsampleFactor: Double = 0.0` pattern
- **`SFXModelManager.modelPath`** — hardcoded `"stable-audio-open"` path replaced with `resolvedModelPath` stored after fetching the active spec; correct directory used regardless of which SFX model is active

### Changed

- **`VAEActor`** — three `fatalError` calls replaced with `throws VAEActorError.weightsNotLoaded`; `VAEActorError` enum added; all callers (`MusicPipeline`, `InpaintCoordinator`, `TrainingService`, `ConditioningActor`) updated
- **`ConditioningActor.init`** — `try!` on tokenizer load replaced with `throws`; `forTesting()` factory also `throws`; call site in `AnvilAppApp` uses `guard let … = try?` with `fatalError` fallback for missing bundle resource
- **`ConditioningActor.encode()`** — `try Task.checkCancellation()` added after tokenization, after Qwen3 forward pass, and after reference audio preparation
- **`DownloadClient`** — `withTaskCancellationHandler` wraps `URLSession.download(from:)` so cancelling the parent Swift task immediately calls `session.invalidateAndCancel()` on large file downloads
- **`RotaryEmbedding`** — removed eager `cosTable`/`sinTable` pre-allocation at `maxLen=32768`; now stores only `invFreq` (`headDim/2` elements) and computes position tables on demand at `forward()` time for the actual `seqLen`
- **`SFXModelManager.unloadPipeline()`** — new method frees in-memory pipeline, tokenizer, and model ID without touching files on disk
- **`AudioPlayerService`, `OutputDirectoryManager`, `HistoryStore`** — bare `print()` error calls replaced with `os.Logger` (subsystem `com.pimp.AnvilApp`, per-class categories)

---

## [Unreleased] — Phase 10c: Community Sharing + Multi-Adapter Stacking

### Added

- **Community Adapter Browser** (`CommunityBrowserView.swift`) — App Store-like interface for discovering community-shared adapters; fetches curated manifest from GitHub (`PiMPStudios/anvil-adapters`); shows name, author, description, tags, rank/category badges; inline download with circular progress; category filter + search; pull-to-refresh with 1hr cache TTL
- **HuggingFace Import** (`HuggingFaceImportSheet.swift`) — modal for importing any adapter from a HuggingFace repo URL; validates adapter.json + adapter.safetensors at repo root; shows preview with compatibility warning for category mismatch
- **CommunityService** (`CommunityService.swift`) — manages manifest fetch, adapter downloads (max 3 concurrent, auto-retry), and HF imports; tracks download state per adapter; 1hr manifest cache
- **Multi-Adapter Stacking** (`AdapterStackView.swift`) — stacking controls below primary AdapterPicker; per-adapter strength sliders (0.0–1.0, step 0.05); total strength display; picker sheet filtered to compatible available adapters; max 3 simultaneous adapters
- **AdapterStackManager** (`AdapterStackManager.swift`) — runtime state for primary + 2 additional adapters with strength; promotion-on-delete logic; capacity enforcement; stack save/restore during training
- **LoRAContribution** (`LoRAContribution.swift`) — per-layer data type holding adapterID, displayName, source, loraA/loraB matrices, and strength for multi-adapter forward pass
- **CommunityAdapter + AdapterManifest** (`CommunityAdapter.swift`) — Codable types for the curated manifest format; includes author, tags, featured flag, minAppVersion, downloadCount
- **AdapterSource enum** (`AdapterSource.swift`) — `.local`, `.community`, `.huggingface` provenance tracking for adapters
- **LoRAInjector stacking methods** — `addContribution`, `clearContributions`, `setStrength`, `removeContribution` for per-layer multi-adapter management

### Changed

- **LoRALinear** — three-path forward pass: (1) training mode (single adapter + dropout), (2) inference stacking (contributions array with per-adapter strength × scaling), (3) legacy single-adapter backward compat
- **LoRAAdapter** — added `source: AdapterSource` (defaults to `.local` for backward compat via `decodeIfPresent`) and `remoteRepoURL: URL?` for HuggingFace-sourced adapters
- **AdaptersSettingsView** — segmented control with "My Adapters" / "Community" tabs; source badges for non-local adapters; updated empty state to reference Community tab
- **TrainingService** — clears adapter stack before training starts, restores after completion (prevents multi-adapter interference with single-adapter training mode)
- **GenerateView** — integrates AdapterStackView below primary AdapterPicker when stacking is active
- Design spec: `docs/superpowers/specs/2026-05-06-phase10c-community-stacking.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase10c-community-stacking.md`

---

## [Unreleased] — Phase 10b: Advanced LoRA

### Added

- **CosineScheduler** (`CosineScheduler.swift`) — learning rate scheduling utility with linear warmup (configurable fraction, default 10%) followed by cosine decay to 0; used by LoRATrainer to dynamically adjust AdamW learning rate each step
- **TrainingCheckpoint** (`TrainingCheckpoint.swift`) — Codable struct capturing full training state (epoch, globalStep, totalSteps, bestLoss, lossHistory, weights URL, optimizer URL, config, baseModelID) for pause/resume; `CheckpointError` for mismatch detection
- **TargetLayer enum** — `.selfAttention`, `.crossAttention`, `.mlp` cases with `moduleName` (maps to LoRAInjector property names) and `defaultPatterns` (expands to injection string patterns); `displayName` extension for UI checkboxes
- **LossCurveView** (`LossCurveView.swift`) — SwiftUI Charts mini line chart displaying per-step loss history during training; auto-scaled Y axis, 120pt height
- **AdaptersSettingsView** (`AdaptersSettingsView.swift`) — dedicated "Adapters" tab in Settings window; lists all adapters with name, base model badge, rank, loss, date; context menu rename/export/delete; swipe-to-delete; "Merge Two" button; empty state guidance
- **MergeSheet** (`MergeSheet.swift`) — adapter composition modal with picker A, picker B (filtered to same base model + rank with rank badge), alpha blend slider (0–1), name field; calls `AdapterRegistry.merge()` to create new adapter from `α×A + (1-α)×B`
- **AdapterRegistry.rename/delete/export/merge** — lifecycle methods for adapter management; rename with atomic move-then-write; delete with cleanup; export copies safetensors + metadata; merge with weighted interpolation and rank validation
- **Settings "Adapters" tab** — third tab in Settings TabView alongside Intelligence and Models

### Changed

- **LoRAConfig** — renamed `epochs` to `maxEpochs`; changed `targetLayers` from `[String]` to `Set<TargetLayer>`; added `warmupFraction` (default 0.10), `gradientAccumulationSteps` (default 1); added `injectionPatterns` computed property bridging enum to string patterns; added `Hashable` conformance
- **LoRAAdapter** — properties `id`/`name` changed from `let` to `var` for rename support; added `Hashable` conformance (required for SwiftUI Picker tags)
- **LoRATrainer** — integrates CosineScheduler for per-step LR updates; per-sample optimizer stepping with LR/progress reporting every `gradientAccumulationSteps` samples; tracks `globalStep`, `lossHistory`, `currentEpoch`; single designated init with optional `resumeFrom` checkpoint parameter; `makeCheckpoint()` for state persistence
- **TrainingProgress** — extended with `step`, `learningRate`, `estimatedTimeRemaining`, `lossHistory`, `elapsedTime` fields; backward-compatible `estimatedRemaining` accessor
- **TrainingService** — `TrainingState.training` now wraps `TrainingUIState` struct; added `.paused(TrainingCheckpoint)` state; extended progress callback for new fields
- **TrainingSheet** — target layer multi-select checkboxes; Advanced disclosure group with warmup slider and gradient accumulation picker; LossCurveView in progress state; current LR and ETA display; conditional Resume/Start buttons for paused state
- **LoRAInjector** — `inject()` iterates `config.injectionPatterns` (computed from `Set<TargetLayer>`) instead of raw `[String]`
- **ModelCategory** — added `Hashable` conformance
- Design spec: `docs/superpowers/specs/2026-05-06-phase10b-advanced-lora.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase10b-advanced-lora.md`

---

## [Unreleased] — Phase 10a: LoRA Fine-Tuning

### Added

- **LoRALinear module** (`LoRALinear.swift`) — subclasses MLXNN `Linear` with low-rank A·B adapter matrices; output = base(x) + scaling * x @ A^T @ B^T; Kaiming uniform init for A, zero init for B (starts as identity); `merge()`/`unmerge()` for folding adapter into base weights with zero inference overhead; `freezeBase()` marks base parameters non-trainable
- **LoRAInjector** (`LoRAInjector.swift`) — static utility for module tree manipulation; `inject(model:config:)` walks DiTDecoder layers and swaps target Linear properties with LoRALinear wrappers; `remove(model:paths:)` restores plain Linear; `extractWeights`/`loadWeights` for safetensors serialization
- **LoRATrainer** (`LoRATrainer.swift`) — flow-matching training loop with AdamW optimizer; samples t∈[0,1], interpolates x_t = (1-t)·x₀ + t·noise, MSE loss on predicted vs target velocity; trains only LoRA A/B matrices (~6.3M params at rank 16); tracks best weights across epochs; cooperative cancellation support
- **LoRAConfig + LoRAAdapter** (`LoRAConfig.swift`) — config struct (rank, alpha, dropout, targetLayers, learningRate, epochs, batchSize) with sensible defaults; adapter metadata struct (id, name, baseModelID, baseModelCategory, sampleCount, finalLoss, createdAt)
- **AdapterRegistry** (`AdapterRegistry.swift`) — lightweight file-system scanner for `~/Library/Application Support/Anvil/models/adapters/`; lists adapters, filters by base model category, saves/loads adapter.safetensors + adapter.json
- **TrainingService** (`TrainingService.swift`) — orchestrates full training flow: preprocesses dataset (audio→latent via VAE, text→conditioning), runs training through actor chain, saves adapter; reports progress to UI; WAV file loading for training data
- **TrainingSheet** (`TrainingSheet.swift`) — SwiftUI sheet for training config + progress; config state shows selected tracks, adapter name, rank/epochs pickers, "+ Add audio files..." import; progress state shows epoch N/M, current loss, elapsed/remaining time; "Stop Training" with save-if-possible confirmation; completion view with adapter details
- **AdapterPicker** (`AdapterPicker.swift`) — dropdown in Generate view showing "None" + compatible adapters filtered by baseModelCategory; displays name, rank, sample count per adapter; selection persisted via @AppStorage
- **History multi-select** — "Select" toolbar button toggles multi-select mode with checkmarks; selected count badge; "Train LoRA" button appears when tracks selected; presents TrainingSheet
- **Adapter lazy injection** — `GenerationService.ensureAdapter()` compares selected vs injected adapter at generation time; injects/removes/swaps as needed; falls back to base model on any adapter error; adapter removed when base model changes

### Changed

- **DiTComponents** — `DiTAttention` and `SwiGLUMLP` Linear properties changed from `let` to `var` for runtime LoRA injection
- **DiTActor** — gained `injectLoRA`, `removeLoRA`, `loadLoRAWeights`, `trainLoRA`, `cancelTraining` methods
- **DiffusionActor/MusicPipeline** — LoRA forwarding methods through the actor chain; `encodeMean` for training dataset prep; `cancelTraining` for cooperative stop
- **GenerateView** — added `adapterPickerSection` between controls bar and progress; syncs selection to GenerationService
- **Package.swift** — added `MLXOptimizers` dependency for AdamW
- Design spec: `docs/superpowers/specs/2026-05-06-phase10a-lora-fine-tuning.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase10a-lora-fine-tuning.md`

---

## [Unreleased] — Phase 9b: Model Management UI

### Changed

- **Settings window** — upgraded from single-view to `TabView` with "Intelligence" and "Models" tabs (Cmd+,)
- **GenerationService lazy swap** — replaced `weightsLoaded` boolean with model ID comparison; detects active model mismatch at generation time, unloads old weights via ARC, loads new weights from registry path, then generates
- **SFXModelManager swap support** — replaced hardcoded model ID/repo with registry dependency; `ensureActiveModel()` checks active `.stableAudio` model before generation, unloads and reloads pipeline on mismatch
- **DiTActor/VAEActor unloadable** — `decoder`/`vae` properties changed from `let` to `var` optional with `decoderFactory` closure for recreation; `unloadWeights()` nils references for ARC deallocation; `loadedModelID` tracks which model is loaded

### Added

- **ModelsSettingsView** (`ModelsSettingsView.swift`) — Models tab with ACE-Step (Music) and Stable Audio (SFX) sections; each row shows display name, formatted size, status badge (Ready/Not Downloaded/Downloading N%/Verifying/Failed), active indicator with checkmark and background highlight; Set Active/Download/Delete/Remove actions with confirmation dialogs; error alert for failed operations; task cancellation on rapid refresh
- **AddModelView** (`AddModelView.swift`) — HuggingFace repo input with category picker (ACE-Step DiT, Stable Audio); HEAD request validation for expected files per category; specific error messages (missing file, private/gated repo, network error, duplicate); size estimation from Content-Length headers; display name inference from repo ID; confirmation alert before adding; auto-download after add
- **ModelCategory enum** (`ModelCategory`) — `aceStepDiT`, `aceStepVAE`, `stableAudio`, `lyricLLM` cases for pipeline assignment; `category` field on `ModelSpec`
- **Active model tracking** — `ModelRegistry.activeModels` mapping (`[ModelCategory: String]`) persisted to `active-models.json`; `setActiveModel`, `activeModelID`, `activeSpec`, `specs(for:)` methods; defaults to built-in model IDs on first launch
- **Model file deletion** — `ModelRegistry.deleteModelFiles(id:)` removes downloaded files while preserving spec for re-download; reverts active model to built-in default on deletion
- **Error revert on swap failure** — if new model weights fail to load, active model reverts to previous working model ID; nil-decoder guard prevents fatalError
- **LocalizedError on RegistryError** — user-visible error descriptions for all registry error cases
- **ModelSpec.displayName** — computed property deriving readable name from HuggingFace repo (hyphens/underscores to spaces, title-cased)
- Design spec: `docs/superpowers/specs/2026-05-06-phase9b-model-management-ui.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase9b-model-management-ui.md`

---

## [Unreleased] — Phase 9a: Model Management Foundation

### Changed

- **First-launch redesign** — blocking download reduced from ~7.6 GB (all 4 models) to ~4 GB (ACE-Step only); users generate music while SFX and LLM models download in the background
- **Setup gate** — `AppState.checkSetupStatus` now checks only essential models (`ace-step-dit`, `ace-step-vae`) via `ModelRegistry.isModelReady`; SFX and lyric features unlock automatically when their background downloads complete
- **ModelRegistry architecture** — split from single `nonisolated let allSpecs` into `static let builtInSpecs` + actor-isolated `var userSpecs` with computed `var allSpecs`; JSON persistence for user-added model specs at `~/Library/Application Support/Anvil/models/user-models.json`

### Added

- **SHA256 download verification** (`DownloadClient.downloadAndVerify`) — enforces SHA256 checksum after each file download with one automatic retry on verification failure; empty hash dictionaries bypass verification (forward-compatible for community models in 9b)
- **BackgroundDownloadIndicator** (`BackgroundDownloadIndicator.swift`) — capsule-shaped progress pill showing background download status: spinner with model name and percentage while downloading, checkmark on completion (auto-dismisses after 3 seconds), warning with click-to-retry on failure
- **Background download coordination** (`AppState.startBackgroundDownloads`) — sequential background downloads after first launch: Stable Audio Open first (more immediately useful), then Lyric Writer LLM; progress relayed from manager `$downloadState` publishers to `BackgroundDownloadPhase`
- **Registry extension points** — `addModel`, `removeModel`, `isBuiltIn` methods on ModelRegistry (functional in 9a, UI in 9b); `RegistryError` enum for built-in protection and duplicate detection
- **ModelSpec Codable conformance** — enables JSON persistence for user-added models; `sha256` parameter defaults to empty dictionary
- Design spec: `docs/superpowers/specs/2026-05-06-phase9a-model-management-foundation.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase9a-model-management-foundation.md`

---

## [Unreleased] — Phase 8b: Custom DSP Backend

### Changed

- **Time Stretch backend** — replaced `AVAudioUnitTimePitch` with custom WSOLA (Waveform Similarity Overlap-Add) processor; time-domain stretching with cross-correlation search for optimal overlap alignment; window 1024 samples, 75% overlap, ±64 sample search range, energy gating on silent frames
- **Pitch Shift backend** — replaced `AVAudioUnitTimePitch` with three-stage pipeline: linear resampler → WSOLA time correction → LPC formant correction; eliminates "chipmunk effect" on vocals at large pitch changes
- **LPC Formant Correction** (`LPCFormantShifter.swift`) — estimates spectral envelope via Levinson-Durbin recursion (order 24), inverse-filters shifted audio to remove distorted formants, re-filters with original coefficients to restore natural vocal resonances
- **DSPProcessor region targeting** — `processRegion` helper handles selection-based byte-level splicing directly in DSPProcessor (previously delegated to AVAudioEngineDSP for time stretch/pitch shift)

### Added

- **WSOLAProcessor** (`WSOLAProcessor.swift`) — WSOLA time stretch with Hann windowing, vDSP cross-correlation search, overlap-add with envelope normalization, energy gating bypass for silent frames
- **Resampler** (`Resampler.swift`) — linear interpolation resampler for pitch ratio conversion; ratio clamped 0.25–4.0, stereo de-interleave/re-interleave
- **LPCFormantShifter** (`LPCFormantShifter.swift`) — LPC formant correction with autocorrelation via vDSP, Levinson-Durbin solver with reflection coefficient clamping (±0.99), inverse/synthesis filtering, overlap-add output
- Design spec: `docs/superpowers/specs/2026-05-06-phase8b-custom-dsp-backend.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase8b-custom-dsp-backend.md`

### Removed

- `AVAudioEngineDSP.applyTimeStretch` and `applyPitchShift` methods (replaced by custom processors; EQ and Reverb still use AVAudioEngine)

---

## [Unreleased] — Phase 8a: Time Stretch & Pitch Shift

### Added

- **Time Stretch** — rate change (0.25×–4.0×) without pitch change via `AVAudioUnitTimePitch` offline rendering; output buffer length adjusts to match new duration
- **Pitch Shift** — pitch change (±24 semitones / ±2400 cents) without duration change via `AVAudioUnitTimePitch` offline rendering; preserves original buffer length
- **TimeStretchControlView** (`TimeStretchControlView.swift`) — rate slider with snap points (0.5×, 0.75×, 1.0×, 1.25×, 1.5×, 2.0×), quality zone indicator (green/yellow/red), double-click direct numeric entry
- **PitchShiftControlView** (`PitchShiftControlView.swift`) — pitch slider with snap points at musical intervals (-12, -7, -5, 0, +5, +7, +12 semitones), quality zone indicator, direct numeric entry
- **DSPPaletteView "Transforms" section** — palette split into "EFFECTS" (Phase 7 operations) and "TRANSFORMS" (Time Stretch, Pitch Shift) with divider and sub-headers; `DSPEffect.isTransform` drives grouping
- **Quality zone indicator** — three-state display (Excellent/Good/Significant artifacts) based on parameter thresholds; green ≤2.0× or ≤±12st, yellow ≤3.0× or ≤±18st, red beyond
- **DSPProcessor dispatch** — `makeDescription` accepts `DSPParameters` for richer undo labels (e.g., "Time Stretch 0.50× (whole track)", "Pitch Shift +7 st (1.2s–2.8s)")
- Selection cleared after time stretch (duration change shifts region boundaries)
- Design spec: `docs/superpowers/specs/2026-05-06-phase8a-time-stretch-pitch-shift.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase8a-time-stretch-pitch-shift.md`

---

## [Unreleased] — Phase 7: DSP Audio Editor

### Added

- **DSPProcessor** (`DSPProcessor.swift`) — facade dispatching effects to SimpleDSP or AVAudioEngineDSP; generates human-readable operation descriptions for undo popover
- **SimpleDSP** (`SimpleDSP.swift`) — pure Swift/vDSP implementations: Trim to Selection, Delete Selection, Fade In (linear ramp), Fade Out (linear ramp), Normalize (peak via `vDSP_maxmgv`)
- **AVAudioEngineDSP** (`AVAudioEngineDSP.swift`) — offline AVAudioEngine rendering for 3-band Parametric EQ (`AVAudioUnitEQ`) and Reverb (`AVAudioUnitReverb` with Room/Hall/Plate presets); region splicing for selection-targeted effects
- **AudioBufferConverter** (`AudioBufferConverter.swift`) — format conversion between `Data` (16-bit PCM), `[Float]` (vDSP), and `AVAudioPCMBuffer` (AVAudioEngine)
- **DSPParameters** (`DSPParameters.swift`) — `AudioBuffer`, `DSPEffect`, `ApplyTarget`, `DSPParameters`, `EQBand`, `EQParameters`, `ReverbPreset`, `ReverbParameters`, `NormalizeMode`, `DSPOperation`
- **DSPPaletteView** (`DSPPaletteView.swift`) — always-available effects palette in Edit tab with 3-column button grid, expandable EQ/Reverb parameter panels, instant-apply checkmark confirmation, smart "Apply to" target indicator
- **EQControlView** (`EQControlView.swift`) — 3-band parametric EQ controls with Frequency, Gain (color-coded), and Q sliders per band
- **ReverbControlView** (`ReverbControlView.swift`) — preset row (Room/Hall/Plate) + Decay, Wet/Dry, Pre-Delay, Damping sliders; selecting preset fills sliders, adjusting slider deselects preset
- **EditorState DSP additions** — `expandedEffect`, `eqParameters`, `reverbParameters`, `audioBuffer` properties; `applyDSPEffect(_:parameters:)` method with smart target resolution; `computePeaks(from:)` for waveform display
- **EditorView integration** — DSPPaletteView below AI controls with divider; real audio file loading via `AVAudioFile`; keyboard shortcuts: Cmd+T (trim), Cmd+Delete (delete), Cmd+Shift+N (normalize), Cmd+Shift+E (EQ), Cmd+Shift+R (reverb)
- Design spec: `docs/superpowers/specs/2026-05-06-phase7-dsp-audio-editor.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase7-dsp-audio-editor.md`

---

## [Unreleased] — Phase 6: AI Audio Editor

### Added

- **InpaintCoordinator** (`InpaintCoordinator.swift`) — actor orchestrating the full inpainting pipeline: VAE encode → mask → noise injection → diffusion denoise → composite → VAE decode; manages undo stack, routes through ACE-Step or Stable Audio pipelines
- **MaskBuilder** (`MaskBuilder.swift`) — converts time selection (+ optional frequency band) into latent-space binary mask `[1, T, 1]` or `[1, T, D]`; handles sample-rate-to-frame conversion via hop size
- **LatentBlender** (`LatentBlender.swift`) — noise injection at strength level and latent compositing; seeded noise for reproducible edits
- **TimestepMapper** (`TimestepMapper.swift`) — maps strength (0.0–1.0) to diffusion start timestep; returns truncated schedule for partial denoising (fewer steps = faster)
- **EditHistoryManager** (`EditHistoryManager.swift`) — multi-level undo/redo stack (max 20 snapshots) with discard-on-branch semantics
- **InpaintTypes** (`InpaintTypes.swift`) — `EditMode`, `InpaintSelection`, `FrequencyBand`, `InpaintRequest`, `InpaintProgress`, `InpaintPreset`, `EditSnapshot`, `PipelineChoice` (moved to AnvilCore)
- **SFXVAEModel encoder path** — symmetric downsample blocks (`SFXDownsampleBlock`) mirroring the existing upsample decoder; `encode(audio:)` returns mean latents
- **VAEActor.encodeMean** — deterministic encoding for inpainting (vs. `encodeSample` which adds stochastic noise)
- **DiffusionActor.inpaint** — partial-noise diffusion start with pre-truncated schedule; reuses CFG/APG/ODE/SDE stepping from `generate()`
- **SFXDiffusionActor.inpaint** — same partial-noise pattern for Stable Audio pipeline
- **EditorView** (`EditorView.swift`) — Edit tab root: waveform canvas, frequency constraint panel, dual prompts, controls bar, empty state with file picker
- **EditorWaveformView** (`EditorWaveformView.swift`) — Canvas-based stereo waveform with drag-to-select (orange overlay), playhead, zoom-ready
- **EditorToolbar** (`EditorToolbar.swift`) — mode toggle (Inpaint/Repaint segmented), pipeline picker, undo/redo with Cmd+Z shortcuts
- **EditorPromptArea** (`EditorPromptArea.swift`) — two-column layout: Original Prompt (context, read-only style) + Region Prompt (editable, blue-themed); relabels to "Target Style" in Repaint mode
- **EditorControlsBar** (`EditorControlsBar.swift`) — preset buttons (Subtle/Moderate/Full), strength + guidance scale sliders, dynamic action button ("Inpaint Selection" / "Repaint Track")
- **EditorState** (`EditorState.swift`) — `@MainActor ObservableObject` managing all editor UI state; `buildRequest()` assembles `InpaintRequest` from current state
- **EditorService** (`EditorService.swift`) — `@MainActor ObservableObject` wrapping `InpaintCoordinator` for environment injection
- **Edit tab in sidebar** — new `.edit` entry in `MainShellView` under "Create" section
- **"Edit This" button** — added to `PlayerView` after generation; posts notification to navigate to Edit tab with audio + prompt pre-loaded
- **HistoryEntry additions** — `pipeline: String?` and `originalPrompt: String?` fields for provenance tracking
- Design spec: `docs/superpowers/specs/2026-05-06-phase6-ai-audio-editor.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase6-ai-audio-editor.md`

---

## [Unreleased] — Phase 5: Sound FX Pipeline

### Added

- **SoundFXPipeline** (`SoundFXPipeline.swift`) — top-level actor orchestrating CLAP encoding, timing conditioning, diffusion sampling, and VAE decode for Stable Audio Open; max duration 47s, progress stream via `AsyncStream<GenerationProgress>`
- **CLAPEncoder** (`CLAPEncoder.swift`) — Swift actor wrapping a CLAP text transformer (configurable layers/heads/vocabSize); produces `[1, 512]` conditioning vector from text prompt; pre-norm residual attention + GELU MLP
- **CLAPTokenizer** (`CLAPTokenizer.swift`) — word-level tokenization with character-level OOV fallback; BOS/EOS framing, pad/truncate to maxLength (77); loads vocabulary from JSON or uses `forTesting()` factory
- **TimingEncoder** (`TimingEncoder.swift`) — sinusoidal positional embeddings from start time and duration scalars; output `[1, embeddingDim]` with values bounded to [-1, 1]
- **SFXDiTModel + SFXDiTActor** — Stable Audio diffusion transformer with adaptive layer norm (adaLN), self-attention, cross-attention for text/timing conditioning, SiLU-gated MLP; actor provides `predict(latents:timestep:conditioning:)`
- **SFXVAEModel + SFXVAEActor** — upsampling VAE decoder with configurable upsample ratios; converts latent sequences to stereo audio waveforms
- **SFXDiffusionActor** — diffusion sampling loop with linear timestep schedule, classifier-free guidance (CFG) when `guidanceScale > 1.0`, cancellation support via `Task.checkCancellation()`
- **Weight Name Maps** — `CLAPNameMap`, `SFXDiTNameMap`, `SFXVAENameMap` for converting PyTorch checkpoint keys to MLX-Swift parameter names
- **SFXModelManager** (`SFXModelManager.swift`) — manages deferred download (~4GB) of Stable Audio Open weights from HuggingFace; uses `URLSession.download(from:)` for large files; tracks `SFXDownloadState` via Published property
- **PipelineChoice** enum — `.auto`, `.aceStep`, `.stableAudio` for manual pipeline override
- **GenerationService routing** — `shouldUseStableAudio(request:)` dispatches to correct pipeline based on `PipelineChoice` override or resolved `GenerationMode`; `generateWithStableAudio` / `generateWithACEStep` split
- **GenerationState.downloadingModel** — new state case for SFX model download progress
- **GenerationRequest** additions — `resolvedMode: GenerationMode` and `pipelineOverride: PipelineChoice` fields
- **UI: download badge** — mode buttons show `arrow.down.circle` when SFX model not yet downloaded
- **UI: duration cap** — slider/value capped at 47s when Stable Audio pipeline is resolved
- **UI: Inspector pipeline section** — shows active pipeline name and override picker (when SFX model downloaded)
- **SFXDiTConfig**, **SFXVAEConfig**, **SFXDiffusionConfig** types added to `Types.swift`
- Design spec: `docs/superpowers/specs/2026-05-05-phase5-soundfx-pipeline.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase5-soundfx-pipeline.md`

---

## [Unreleased] — Phase 4b: Intelligence Layer

### Added

- **LLMActor** (`LLMActor.swift`) — Swift actor for local LLM inference via MLX-Swift; streaming `generate()` and single-shot `classify()` methods
- **IntentRouter** (`IntentRouter.swift`) — debounced live classification of user prompts into `GenerationMode` (music/soundFX/ambient) using local LLM; 500ms debounce, maxTokens 20
- **PromptEnhancer** (`PromptEnhancer.swift`) — expands brief descriptions into detailed audio prompts via LLM; cancellable, mode-aware system prompt
- **LyricWriterManager** (`LyricWriterManager.swift`) — coordinates backend selection (local + API), streaming output to binding, cancellation
- **LocalLyricWriter** (`LocalLyricWriter.swift`) — on-device lyric generation via LLMActor with section markers and duration-aware line estimates
- **API Lyric Writer backends** — ClaudeLyricWriter, OpenAILyricWriter, GeminiLyricWriter, XAILyricWriter; streaming HTTP with proper cancellation
- **LLMModelManager** (`LLMModelManager.swift`) — model download (URLSession.download for large files), custom model add/remove, reactive `llmActor` publishing
- **APIKeyStore** (`APIKeyStore.swift`) — Keychain storage with env var fallback for API keys; supports Claude, OpenAI, Gemini, xAI providers
- **GenerationMode** enum — `.music`, `.soundFX`, `.ambient` with classification parsing
- **ModeSelection** enum — `.auto` / `.manual(GenerationMode)` for UI state
- **IntelligenceSettingsView** — model selection, backend management, API key configuration
- **GenerateView** additions — mode picker with auto-detect indicator, Enhance button, Write Lyrics button
- **App wiring** — reactive LLM propagation via `.onReceive(modelManager.$llmActor)`, API key refresh via `.onChange(of: apiKeyStore.availableProviders)`
- Design spec: `docs/superpowers/specs/2026-05-05-phase4b-intelligence-layer.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase4b-intelligence-layer.md`

---

## [Unreleased] — Phase 4a: Generation Flow

### Added

- **GenerationService** (`GenerationService.swift`) — `@MainActor ObservableObject` orchestrating full generation: validation → weight loading → conditioning → diffusion → VAE decode → WAV write → history entry
- **GenerationRequest** type — captures all generation parameters (text, lyrics, duration, seed, method, guidanceScale, precision)
- **GenerationState** enum — idle, loadingWeights, conditioning, generating, decoding, complete, error, cancelled
- **HistoryStore** (`HistoryStore.swift`) — JSON-backed persistent history of generated audio with search
- **HistoryEntry** type — metadata for each generation (prompt, lyrics, duration, seed, method, fileName)
- **OutputDirectoryManager** — user-selected output folder with security-scoped bookmark persistence
- **AudioPlayerService** — `AVAudioPlayer`-based playback with play/pause/stop
- **WAVWriter** extensions — write `MLXArray` samples to WAV file at specified sample rate
- **GenerateView** (`GenerateView.swift`) — prompt input, lyrics editor, controls bar, progress section, player
- **HistoryListView** — searchable list of past generations with playback
- **PlayerView** — inline audio player with waveform-style progress
- **MainShellView** — three-panel NavigationSplitView with sidebar, canvas, inspector
- **OutputDirectoryPicker** — first-use folder selection before main UI appears
- **FirstLaunchView** / **AppState** — setup flow gating
- Design spec: `docs/superpowers/specs/2026-05-05-phase4a-generation-flow.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase4a-generation-flow.md`

---

## [Unreleased] — Phase 3: Conditioning Pipeline

### Added

- **Qwen3Model** (`Qwen3Model.swift`) — 28-layer bidirectional text encoder, 1024-dim hidden, 151,669-token vocabulary; `callAsFunction(inputIds:)` returns `[B, T, 1024]`, `embedTokens(_:)` returns embedding-only lookup
- **Qwen3Layer / Qwen3Attention** (`Qwen3Layer.swift`) — pre-norm transformer layer with GQA (16 query / 8 KV heads, head_dim=128), QK-RMSNorm, RoPE via shared `RotaryEmbedding`, no causal mask
- **LyricEncoder** (`LyricEncoder.swift`) — 8-layer encoder with `Linear(1024→2048)` input projection; takes Qwen3 token embeddings, returns `[B, T_lyric, 2048]`
- **TimbreEncoder** (`TimbreEncoder.swift`) — 4-layer encoder with `Linear(64→2048)` input projection, followed by 2-layer `AttentionPooler` with 5 learnable query tokens; takes VAE latents `[B, 750, 64]`, returns timbre tokens `[B, 5, 2048]`; silence fallback (`zeros([1, 750, 64])`) for text-only generation
- **ConditionEncoder** (`ConditionEncoder.swift`) — orchestrates sub-encoders: projects text hidden states `[B, T, 1024→2048]`, concatenates text + lyric + timbre tokens along sequence axis; returns `encoderStates [B, T_combined, 2048]` and combined attention mask
- **ConditioningActor** (`ConditioningActor.swift`) — top-level Swift actor: tokenizes text and lyrics, runs Qwen3 forward pass, runs LyricEncoder, VAE-encodes reference audio (pad/truncate to 750 frames), runs TimbreEncoder, assembles `ConditioningResult`; produces `contextLatents [B, T, 128]` = zeros for text-only or `cat([latents, chunkMask], axis:-1)` for reference audio
- **AnvilTokenizer** (`AnvilTokenizer.swift`) — thin wrapper around swift-transformers `AutoTokenizer`; `encodeText(_:maxLength:)` and `encodeLyrics(_:maxLength:)` return `(ids: MLXArray, mask: MLXArray)` shaped `[1, maxLength]` as Int32; `forTesting()` factory with minimal BPE vocab for unit tests
- **Qwen3NameMap** (`Weights/Qwen3NameMap.swift`) — maps Qwen3-Embedding-0.6B HuggingFace safetensors keys to MLX-Swift parameter names; filters `lm_head.*`
- **ConditionEncoderNameMap** (`Weights/ConditionEncoderNameMap.swift`) — maps ACE-Step DiT checkpoint keys (`condition_encoder.{text_projector,lyric_encoder,timbre_encoder,timbre_attention_pooler}.*`) to Swift module paths; handles `special_token` → `attentionPooler.queryTokens`
- **WeightConverter** additions — `convertQwen3Weights(from:progress:)` and `convertConditionEncoderWeights(from:progress:)` methods
- **ModelComponent** additions — `.qwen3` and `.conditionEncoder` cases
- **ConditioningResult** type — `encoderStates`, `encoderMask`, `contextLatents` (`@unchecked Sendable`)
- **MusicPipeline** convenience method — `generate(text:lyrics:referenceAudio:duration:config:conditioning:)` runs conditioning then diffusion in one call
- swift-transformers (HuggingFace, v0.1.12+) added as dependency to `AnvilCore/Package.swift`
- Design spec: `docs/superpowers/specs/2026-05-05-phase3-conditioning-pipeline.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase3-conditioning-pipeline.md`

---

## [Unreleased] — Phase 2: Core Inference Stack

### Added

- **MusicPipeline** coordinator actor — top-level orchestrator that runs diffusion then VAE decode, returning `AudioResult` with stereo audio at 44.1 kHz
- **DiffusionActor** — full generation loop with seeded noise initialization, CFG batch doubling, APG guidance, and `AsyncStream<DiffusionProgress>` for progress reporting
- **DiffusionSampler** — ODE and SDE stepping methods, APG (Adaptive Projected Guidance) with momentum and norm thresholding, seeded noise generation
- **DiTDecoder** (`DiTModel.swift`) — 24-layer Diffusion Transformer decoder with patch embedding (Conv1d stride 2), de-patchify (ConvTransposed1d), adaptive layer norm output, and sliding window mask caching
- **DiTActor** — thread-safe actor wrapping `DiTDecoder` with `step()` and `loadWeights()` methods
- **DiT components** (`DiTComponents.swift`):
  - `RotaryEmbedding` — precomputed cos/sin tables, base frequency 1,000,000, max 32,768 tokens
  - `DiTAttention` — Grouped Query Attention (16 query / 8 KV heads), QK-RMSNorm, optional sliding window mask, cross-attention with KV cache support
  - `SwiGLUMLP` — `down(silu(gate(x)) * up(x))`, intermediate size 6144
  - `TimestepEmbedding` — sinusoidal → linear → SiLU → linear → 6×D projection for AdaLN
  - `DiTLayer` — AdaLN-modulated self-attention + cross-attention + MLP
  - `createSlidingWindowMask` — additive -1e9 mask for bidirectional window of 128 tokens
- **VAEActor** — actor wrapping `AutoEncoderOobleck` with fp16/fp32 precision switching and tiled decode delegation
- **VAE model** (`VAEModel.swift`) — `OobleckEncoder`, `OobleckDecoder`, `AutoEncoderOobleck` with `encodeSample` (diagonal Gaussian), `encodeMean`, and `decode`
- **VAETiledDecode** — overlap-discard chunked decode for latent sequences >2048 frames (chunk=2048, overlap=64, stride=1920); prevents OOM on songs >85s
- **VAE components** (`VAEComponents.swift`) — `Snake1d` activation, `OobleckResidualUnit` (dilated conv1d + skip, dilations 1/3/9), `OobleckEncoderBlock`, `OobleckDecoderBlock`
- **WeightConverter** actor — converts PyTorch `.bin` state dicts to MLX-Swift weight lists for on-device use (pure Swift, no Python subprocess)
- **VAENameMap** — `weight_norm` fusion (g+v → fused), Conv1d/ConvTransposed1d axis transposition, Snake1d parameter squeeze
- **TensorNameMap** — DiT key remapping (`proj_in.1.*` → `proj_in.*`), rotary embedding key filtering
- **CrossAttentionCache** — layer-indexed (key, value) dictionary cache; populated on first diffusion step, reused for all subsequent steps
- **TimestepSchedule** — three scheduling modes: lookup table (shift 1.0/2.0/3.0), linspace with shift transform, custom timestep snapping
- **Shared inference types** — `Precision`, `InferMethod`, `DiffusionConfig`, `DiffusionResult`, `AudioResult`, `TimeCosts`, `PipelineState`, `GenerationProgress`, `ConversionProgress`
- mlx-swift dependency added to `AnvilCore/Package.swift` (MLX, MLXNN, MLXRandom, MLXFast)
- Design spec: `docs/superpowers/specs/2026-05-05-phase2-core-inference-stack.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase2-core-inference.md`
- Audio editor roadmap entries (DSP — Phase 5; AI inpaint/repaint — Phase 6)

---

## [Unreleased] — Phase 1: Foundation

### Added

- **AnvilApp Xcode project** — SwiftUI shell targeting macOS 14+, App Store sandbox entitlements, Hardened Runtime for notarization
- **Three-panel main shell** — sidebar / canvas / inspector layout
- **First-launch download UI** — model weight download flow with progress display
- **AppState** — observable state container with injectable `ModelRegistry` for testing
- **ModelRegistry** — status tracking and download orchestration for model weights
- **DownloadClient** — HuggingFace URL building, async download with progress, SHA256 verification
- **ModelSpec** — model weight metadata type (name, URL, expected SHA256, size)
- **AnvilCore Swift package** — initialized at `AnvilCore/`, separate from the Xcode app target
- Design spec: `docs/superpowers/specs/` (initial Anvil Swift app design)
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase1-foundation.md` (implicit)

---

## Road Map

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Foundation — Xcode project, SwiftUI shell, ModelRegistry, DownloadClient |
| 2 | ✅ Complete | Core Inference Stack — VAE, DiT, Diffusion Sampler, MusicPipeline, WeightConverter |
| 3 | ✅ Complete | Conditioning Pipeline — Qwen3 text encoder, LyricEncoder, TimbreEncoder, ConditionEncoder, ConditioningActor |
| 4a | ✅ Complete | Generation Flow — GenerationService, WAV output, history, audio player, full UI |
| 4b | ✅ Complete | Intelligence Layer — IntentRouter, PromptEnhancer, LyricWriters (local + API), LLMActor |
| 5 | ✅ Complete | Sound FX Pipeline — Stable Audio Open, CLAP encoder, dual-pipeline routing, deferred download |
| 6 | ✅ Complete | AI Audio Editor — inpaint/repaint via VAE encode → mask latents → denoise |
| 7 | ✅ Complete | DSP Audio Editor — trim, fade, normalize, 3-band parametric EQ, reverb |
| 8a | ✅ Complete | Time Stretch & Pitch Shift — AVAudioUnitTimePitch, quality zones, snap-point sliders |
| 8b | ✅ Complete | Advanced DSP — custom WSOLA backend, LPC formant preservation, linear resampler |
| 9a | ✅ Complete | Model Management Foundation — SHA256 verification, first-launch redesign, registry split |
| 9b | ✅ Complete | Model Management UI — ModelsSettingsView, AddModelView, active model tracking, lazy swap runtime |
| 10a | ✅ Complete | LoRA Fine-Tuning — LoRALinear, LoRAInjector, flow-matching trainer, adapter picker, training sheet |
| 10b | ✅ Complete | Advanced LoRA — CosineScheduler, target layer picker, gradient accumulation, checkpoint/resume, loss curve, adapter merge/rename/delete/export |
| 10c | ✅ Complete | Community Sharing + Stacking — curated manifest browser, HuggingFace import, multi-adapter stacking (Inject-All) with real-time strength sliders |
