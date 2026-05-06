# Phase 9b: Model Management UI — Design Spec

## Overview

Add a Models settings tab for viewing, downloading, deleting, and swapping
diffusion models. Implement lazy model swapping at generation time with
unload/reload runtime. Support adding community models via HuggingFace
repo ID with validation.

**Scope:** Models settings view with grouped model list, active model
selection, lazy swap runtime (detect mismatch → unload → reload → generate),
HuggingFace repo validation for user-added models, model deletion UI.

**Key improvement:** Users can swap between alternative ACE-Step and Stable
Audio checkpoints without restarting the app. Community fine-tunes are one
paste away.

---

## Models Settings View

### Location

New "Models" tab in the existing macOS Settings window (Cmd+,), alongside
the existing "Intelligence" tab:

```swift
Settings {
    TabView {
        IntelligenceSettingsView()
            .tabItem { Label("Intelligence", systemImage: "brain") }
        ModelsSettingsView()
            .tabItem { Label("Models", systemImage: "cube.box") }
    }
}
```

### Layout

The view is grouped by model category with two sections:

**ACE-Step (Music)** section:

- Lists all models with `category` `.aceStepDiT` and `.aceStepVAE`
- Each row shows: model display name, formatted size, status badge
  (Ready / Not Downloaded / Downloading N%), and active indicator
  (checkmark icon for the currently selected model)
- Active model row gets a subtle background highlight for visibility
- Ready models are tappable to set as active
- Download button for not-downloaded models
- Delete button for downloaded models (deletes files, keeps spec for
  re-download)
- Remove button for user-added models only (removes spec from registry
  and deletes files). Built-in models cannot be removed — the remove
  button is hidden for built-in specs.

**Stable Audio (SFX)** section:

- Same layout and controls as ACE-Step section
- Lists models with `category` `.stableAudio`

### Model Row States

| Status | Visual | Actions |
|--------|--------|---------|
| Not Downloaded | Gray badge "Not Downloaded" | Download button |
| Downloading | Progress bar with percentage | Cancel button |
| Ready (inactive) | Green badge "Ready" | Set Active, Delete |
| Ready (active) | Green badge "Ready" + checkmark + highlight | Delete |
| Failed | Red badge with message | Retry button |

---

## Add Custom Model

### UI

Located at the bottom of `ModelsSettingsView` under a "Add Custom Model"
header:

- Text field: "HuggingFace Repo ID" placeholder
  (e.g., `username/my-ace-step-finetune`)
- Category picker: dropdown with "ACE-Step DiT (Music)" and
  "Stable Audio (SFX)" options. ACE-Step VAE is omitted from the
  picker — community fine-tunes almost never replace the VAE, and
  exposing it would confuse most users. If VAE swapping becomes
  needed, the picker can be extended in a future phase.
- "Validate & Add" button

### Validation Flow

1. User enters repo ID, selects category, taps "Validate & Add"
2. Button shows spinner, text field disabled
3. Based on category, determine expected files:
   - `.aceStepDiT` → `["model.safetensors", "config.json"]`
   - `.aceStepVAE` → `["vae.safetensors", "config.json"]`
   - `.stableAudio` → `["model.safetensors", "config.json"]`
4. HTTP `HEAD` request to
   `https://huggingface.co/{repoID}/resolve/main/{file}` for each file
5. Results:
   - **All 200** → show confirmation sheet with inferred display name
     and total size, user confirms, `ModelSpec` created with category
     and added via `registry.addModel()`, download starts automatically
   - **Any 404** → "Missing required file: {filename}"
   - **401/403** → "This repository is private or gated. Only public
     repositories are supported for now."
   - **Network error** → "Could not reach HuggingFace. Check your
     connection."

### Display Name Inference

Derived from repo ID last path component: `username/My-Cool-Model`
becomes `"My Cool Model"` (hyphens/underscores to spaces, title-cased).

### Size Estimation

Sum `Content-Length` headers from the HEAD requests. Stored in
`ModelSpec.sizeBytes`.

### Duplicate Prevention

If a model with the same HuggingFace repo already exists in the registry,
show "This model is already added" instead of re-adding.

---

## Model Categories

### ModelCategory Enum

```swift
public enum ModelCategory: String, Codable, Sendable {
    case aceStepDiT      // ACE-Step diffusion transformer
    case aceStepVAE      // ACE-Step variational autoencoder
    case stableAudio     // Stable Audio Open SFX pipeline
    case lyricLLM        // LLM for lyric writing (managed in Intelligence tab)
}
```

### ModelSpec Change

`ModelSpec` gains a `category: ModelCategory` field:

```swift
public struct ModelSpec: Sendable, Identifiable, Codable {
    public let id: String
    public let huggingFaceRepo: String
    public let files: [String]
    public let sizeBytes: Int64
    public let sha256: [String: String]
    public let category: ModelCategory
}
```

Built-in specs get hardcoded categories:

| Model ID | Category |
|----------|----------|
| `ace-step-dit` | `.aceStepDiT` |
| `ace-step-vae` | `.aceStepVAE` |
| `stable-audio-open` | `.stableAudio` |
| `lyric-llm` | `.lyricLLM` |

### Active Model Tracking

`ModelRegistry` gains:

```swift
private(set) var activeModels: [ModelCategory: String]  // category → model ID
```

Persisted to `~/Library/Application Support/Anvil/models/active-models.json`.
On first launch (no file exists), defaults to the built-in model IDs.

Methods:

```swift
func setActiveModel(id: String, for category: ModelCategory) throws
func activeModelID(for category: ModelCategory) -> String?
func activeSpec(for category: ModelCategory) -> ModelSpec?
```

`setActiveModel` validates the model exists and is downloaded before
setting it active. Persists immediately.

---

## Lazy Swap Runtime

### Strategy

Record the user's model preference in `activeModels` immediately. At
generation time, detect mismatch between the active model and what's
currently loaded. If mismatched, unload old weights, load new weights,
then generate.

### Current Generation Flow

1. User hits Generate → `GenerationService.generate()` called
2. Service checks `weightsLoaded` boolean → if false, calls
   `loadWeights()` once
3. Weights stay loaded indefinitely

### New Generation Flow

1. User hits Generate → `GenerationService.generate()` called
2. Service asks registry: `activeModelID(for: .aceStepDiT)`
3. Compares against `currentlyLoadedDiTModelID` (new tracked property)
4. **Match** → proceed to generation as normal
5. **Mismatch** → update progress: "Loading [model name]... (this may
   take a moment)", then:
   - Call `ditActor.unloadWeights()` (new method)
   - Load new weights from the active model's file path
   - Update `currentlyLoadedDiTModelID`
   - Proceed to generation
6. Same check for VAE: `activeModelID(for: .aceStepVAE)` vs
   `currentlyLoadedVAEModelID`

### Pipeline Actor Changes

`DiTActor` and `VAEActor` each gain:

```swift
private(set) var loadedModelID: String?

func unloadWeights() {
    // Nil the model reference → ARC frees memory
    // Reset loadedModelID to nil
}

func loadWeights(_ weights: [(String, MLXArray)], modelID: String) {
    // Existing load logic
    // Set loadedModelID = modelID
}
```

### SFX Pipeline

`SFXModelManager` already has `deleteWeights()` which nils the pipeline
reference. The swap pattern uses the same approach:

1. Check active model for `.stableAudio`
2. If mismatch with currently loaded SFX pipeline, call
   `deleteWeights()` then reload from new model path
3. SFX swap goes through `SFXModelManager` rather than individual actors

### Memory Behavior

- Only one set of weights per actor at any time
- `unloadWeights()` nils the model reference → ARC deallocates
- New weights loaded from disk into fresh model instance
- No double-loading, no memory pressure spikes

### Error Handling

- If new model files are missing or corrupt → set model status to
  `.failed`, revert `activeModels` to the previous working model ID,
  show error notification to user
- Generation does not proceed with partially loaded state
- Previous model remains available for immediate retry

---

## Model Deletion

### User-Added Models

"Remove" button on user model rows:

1. Confirmation dialog: "Remove [model name]? This will delete all
   downloaded files."
2. On confirm: `registry.removeModel(id:)` (removes spec from
   `userSpecs` and persists)
3. Delete model files from disk
4. If model was active, revert `activeModels` to the built-in default
   for that category

### Built-In Models

"Delete" button on built-in model rows (when downloaded):

1. Confirmation dialog: "Delete [model name] files? You can re-download
   later."
2. On confirm: delete files from disk only — spec stays in
   `builtInSpecs` (it's a static constant)
3. Model status changes to `.notDownloaded`
4. If model was active, revert `activeModels` to the built-in default
   (if another model in that category is ready) or clear active
5. "Remove" button is hidden for built-in models — they cannot be
   removed from the registry

---

## File Structure

### AnvilCore (Modified)

| File | Changes |
|------|---------|
| `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelSpec.swift` | Add `ModelCategory` enum, add `category` field to `ModelSpec` |
| `AnvilCore/Sources/AnvilCore/ModelRegistry/ModelRegistry.swift` | `activeModels` tracking with JSON persistence, `setActiveModel`, `activeModelID`, `activeSpec`, category-aware queries |
| `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift` | Add `unloadWeights()`, track `loadedModelID` |
| `AnvilCore/Sources/AnvilCore/Inference/VAE/VAEActor.swift` | Add `unloadWeights()`, track `loadedModelID` |

### AnvilApp (Modified)

| File | Changes |
|------|---------|
| `AnvilApp/AnvilAppApp.swift` | Settings window `TabView` with Intelligence + Models tabs |
| `AnvilApp/Services/GenerationService.swift` | Lazy swap detection: check active model vs loaded, unload → reload on mismatch, progress messaging |
| `AnvilApp/Services/SFXModelManager.swift` | Swap support: check active `.stableAudio` model, `deleteWeights()` + reload pattern |

### AnvilApp (New)

| File | Purpose |
|------|---------|
| `AnvilApp/UI/Settings/ModelsSettingsView.swift` | Models tab — grouped model list, status, download/delete, active selection |
| `AnvilApp/UI/Settings/AddModelView.swift` | HuggingFace repo input, category picker, validation flow |

### Unchanged

| File | Why |
|------|-----|
| `AnvilApp/UI/Settings/IntelligenceSettingsView.swift` | LLM management stays as-is |
| `AnvilApp/AppState.swift` | No changes needed — background download phase unaffected |
| `AnvilApp/UI/FirstLaunchView.swift` | First launch flow unchanged |
| `AnvilCore/Sources/AnvilCore/ModelRegistry/DownloadClient.swift` | Download/verify pipeline unchanged |

---

## Scope

### In Scope (Phase 9b)

- `ModelCategory` enum with `aceStepDiT`, `aceStepVAE`, `stableAudio`,
  `lyricLLM` cases
- `category` field on `ModelSpec`
- `ModelsSettingsView` with category-grouped model list
- Model row: name, size, status badge, active indicator with highlight
- Download/delete buttons per model
- Remove button for user-added models only (hidden for built-ins)
- Active model selection (tap ready model to activate)
- `activeModels` mapping in `ModelRegistry` with JSON persistence
- Lazy swap in `GenerationService`: mismatch detection, unload → reload
- `unloadWeights()` on `DiTActor` and `VAEActor`
- SFX swap via `SFXModelManager.deleteWeights()` + reload
- Progress messaging during swap: "Loading [name]... (this may take a
  moment)"
- Error handling: revert active model on load failure
- `AddModelView`: HF repo text field, category picker, validation
- HuggingFace HEAD validation for expected files per category
- Display name inference from repo ID
- Size estimation from Content-Length headers
- Duplicate repo prevention
- Specific error messages: missing file, private repo, network error
- Settings window upgrade to `TabView`
- Model deletion with confirmation dialogs
- Revert active model to built-in default on deletion

### Out of Scope (Future)

- Community model browsing/discovery UI (potential 9c)
- Local folder import for models
- Auto-detection of model category from repo contents
- Gated/private HuggingFace repo support
- LoRA fine-tuning (Phase 10)
