# Phase 10a: LoRA Fine-Tuning — Design Spec

## Overview

Add on-device LoRA fine-tuning for ACE-Step DiT. Users select tracks
from their generation history, configure training parameters, and
produce a personal LoRA adapter that modifies generation style. Adapters
are applied at generation time via an explicit picker in the Generate
view.

**Scope:** LoRALinear module wrapping, module tree injection/removal,
flow-matching training loop, training sheet UI from History view,
adapter picker in Generate view, lightweight file import for external
audio.

**Key improvement:** Users can train personal style adapters on-device
(jazz, lo-fi, vocal texture, etc.) from their own generated tracks and
apply them during generation — no cloud, no export, no restart.

---

## LoRA Layer Implementation

### LoRALinear Module

`LoRALinear` is an MLXNN `Module` that wraps an existing `Linear` layer
with a low-rank adapter:

```
output = base_linear(x) + scaling * dropout(x @ A^T @ B^T)
```

Properties:

- `base: Linear` — the original frozen linear layer
- `loraA: MLXArray` — shape `(rank, in_features)`, Kaiming uniform init
- `loraB: MLXArray` — shape `(out_features, rank)`, zero init
- `scaling: Float` — computed as `alpha / rank`
- `dropout: Float` — applied during training only (default 0.0)

The zero-init of B means the adapter starts as identity (no effect on
output). Training gradually moves B away from zero as the adapter
learns.

Methods:

- `callAsFunction(_ x: MLXArray) -> MLXArray` — forward pass with
  adapter contribution
- `merge()` — folds adapter into base: `W' = W + scaling * (B @ A)`.
  Zero inference overhead after merge.
- `unmerge()` — reverses merge: `W' = W - scaling * (B @ A)`. Only
  valid if base weights unchanged since merge.
- `freezeBase()` — marks base Linear parameters as non-trainable.

### LoRAInjector

Static utility for module tree manipulation:

- `inject(model:config:) -> [String]` — walks module tree, replaces
  target `Linear` layers with `LoRALinear` wrappers. Returns injected
  layer paths for tracking.
- `remove(model:paths:)` — swaps `LoRALinear` back to unwrapped
  `Linear` (extracts the base layer from the wrapper). Supports runtime
  adapter removal without restart.
- `extractWeights(model:paths:) -> [(String, MLXArray)]` — collects
  only A and B matrices from injected layers for saving.
- `loadWeights(model:paths:weights:)` — populates A/B matrices in
  existing `LoRALinear` layers from saved weights.

### Default Target Layers

For ACE-Step DiT (24 layers), default injection targets:

- `layers.*.selfAttn.qProj`
- `layers.*.selfAttn.vProj`
- `layers.*.crossAttn.qProj`
- `layers.*.crossAttn.vProj`

Total: 4 projections x 24 layers = 96 adapter pairs. At rank 16, each
pair is `(16 x 2048) + (2048 x 16) = 65,536` parameters. Total
trainable: ~6.3M parameters (~0.6% of base model).

kProj, oProj, and MLP layers are skipped by default to keep adapter
size small and training fast. Advanced target selection deferred to 10b.

---

## Training Pipeline

### LoRAConfig

```swift
public struct LoRAConfig: Codable, Sendable {
    public let rank: Int              // 8, 16, 32 — default 16
    public let alpha: Float           // scaling factor — default 16.0
    public let dropout: Float         // 0.0–0.5 — default 0.0
    public let targetLayers: [String] // layer name patterns
    public let learningRate: Float    // default 1e-4
    public let epochs: Int            // default 10
    public let batchSize: Int         // default 1 (memory constrained)
}
```

### Training Dataset

A training sample is a `(latent, conditioning)` tuple. Preparation:

1. **Audio to latents**: Load WAV from history (or imported file), pass
   through `VAEActor.encodeMean()` to get clean latent x_0.
   Preprocessed once before training begins.
2. **Text to conditioning**: Pass prompt through `ConditioningActor` to
   get encoder states, mask, and context latents. Preprocessed once.
3. **Imported files**: Same pipeline — load audio, encode to latent,
   user-provided description becomes the prompt.

Dataset held in memory during training. Typical sets are 5-20 samples
at a few MB each — no disk-backed dataset management needed in 10a.

### Training Loop (LoRATrainer)

```
for epoch in 1...config.epochs:
    for (x_0, conditioning) in dataset.shuffled():
        t = random uniform in [0, 1]
        noise = random normal, same shape as x_0
        x_t = (1 - t) * x_0 + t * noise
        v_target = noise - x_0

        v_pred = dit.forward(x_t, t, conditioning)
        loss = MSE(v_pred, v_target)

        loss.backward()
        optimizer.step()
        optimizer.zeroGrad()

    report(epoch, loss)
```

Key details:

- **Optimizer**: AdamW with configured learning rate. Only LoRA
  parameters (A, B) in the optimizer parameter group.
- **Gradient computation**: MLX `valueAndGrad` computes loss and
  gradients in one pass. Base model frozen — no gradient tracking on
  base parameters.
- **Memory**: One forward + backward pass at batch size 1. Base weights
  in memory but not accumulating gradients.
- **Progress**: Reports epoch, current loss, estimated time remaining
  after each epoch. TrainingService relays to UI.

### Post-Training

1. `LoRAInjector.extractWeights()` collects all A/B matrices
2. Save to `adapters/{name}/adapter.safetensors`
3. Save config + metadata to `adapters/{name}/adapter.json` (base model
   ID, base model category, training date, sample count, final loss)
4. Remove LoRA injection from DiT (restore clean base model)
5. Adapter appears in Generate view picker

---

## Training UI

### Entry Point

User selects tracks in History view, taps "Train LoRA" button (new).
A sheet slides up showing training configuration.

### Training Sheet — Config State

- **Training Data section**: Lists selected tracks (name + duration),
  total count and duration. "+ Add audio files..." link for lightweight
  file import (file picker + text field per imported file).
- **Configuration section**: Adapter name text field, Base Model
  (read-only, shows active ACE-Step DiT), Rank picker (8/16/32),
  Epochs picker (5/10/20/50).
- **Start Training button**: Validates inputs, begins training.

### Training Sheet — Progress State

Replaces config view after training starts:

- Adapter name as title
- "Epoch N of M" subtitle
- Progress bar (epoch-level granularity)
- Three stats: Current Loss, Elapsed Time, Estimated Remaining
- Stop Training button: if 2+ epochs completed, saves the adapter
  with weights from the lowest-loss epoch. If fewer than 2 epochs,
  shows confirmation "Training incomplete — discard?" and removes
  partial data on confirm.

### History View Changes

- Multi-select mode: long-press or toolbar toggle enables track
  selection with checkmarks
- "Train LoRA" button appears in toolbar when tracks are selected
- Selected track count badge

---

## Adapter Picker

### Location

New dropdown in the Generate view, positioned after guidance scale and
before seed. Labeled "LoRA Adapter".

### Behavior

- Shows "None" (default) plus all compatible adapters
- Compatible = adapter's `baseModelCategory` matches the currently
  active base model category
- Each adapter row shows: name, rank, training sample count
- Selected adapter gets a checkmark
- Selection persisted across app launches (stored in UserDefaults)

### Filtering on Base Model Change

When the user changes the active base model (via Models settings):

- Adapter picker filters to show only compatible adapters
- If the previously selected adapter is incompatible, selection reverts
  to "None"
- Any injected adapter is removed during base model unload

---

## Generation Integration

### Adapter Application (Lazy Inject)

At generation time in `GenerationService.generateWithACEStep()`:

1. Check `selectedAdapterID` vs `currentlyInjectedAdapterID`
2. **Match** — proceed to generation
3. **Different adapter selected** — remove current, inject new, load
   weights, update tracking
4. **"None" selected but adapter injected** — remove adapter
5. **Adapter selected but none injected** — inject and load

Follows the lazy-swap pattern from Phase 9b. Adapter inject/remove
happens through the actor chain: GenerationService → MusicPipeline →
DiffusionActor → DiTActor.

### DiTActor Changes

DiTActor gains:

```swift
public func injectLoRA(config: LoRAConfig) -> [String]
public func removeLoRA(paths: [String])
public func loadLoRAWeights(_ weights: [(String, MLXArray)], paths: [String])
public private(set) var injectedAdapterID: String?
```

These delegate to `LoRAInjector` operating on the internal
`DiTDecoder` model.

### Error Handling

- If adapter files are missing or corrupt: clear selection, show error,
  generate without adapter
- If injection fails (model architecture mismatch): same recovery
- Generation never blocks on adapter issues — falls back to base model

---

## Adapter Storage

### AdapterRegistry

Lightweight file-system scanner (not a full actor):

- Scans `~/Library/Application Support/Anvil/models/adapters/`
- Returns list of `LoRAAdapter` structs
- Called by adapter picker to populate dropdown
- Called by TrainingService to determine save location

### LoRAAdapter Struct

```swift
public struct LoRAAdapter: Codable, Sendable, Identifiable {
    public let id: String            // directory name
    public let name: String          // display name
    public let config: LoRAConfig
    public let baseModelID: String   // e.g., "ace-step-dit"
    public let baseModelCategory: ModelCategory  // e.g., .aceStepDiT
    public let createdAt: Date
    public let sampleCount: Int
    public let finalLoss: Float?
}
```

### File Layout

```
~/Library/Application Support/Anvil/models/
  adapters/
    my-jazz-style/
      adapter.safetensors    <- LoRA A/B weight matrices
      adapter.json           <- LoRAAdapter metadata
    vocal-texture/
      adapter.safetensors
      adapter.json
```

---

## File Structure

### AnvilCore (New)

| File | Responsibility |
|------|---------------|
| `AnvilCore/Sources/AnvilCore/LoRA/LoRALinear.swift` | LoRALinear Module wrapping Linear |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift` | Module tree walking, inject/remove/extract/load |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRATrainer.swift` | Flow-matching training loop with AdamW |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift` | LoRAConfig + LoRAAdapter data types |

### AnvilApp (New)

| File | Responsibility |
|------|---------------|
| `AnvilApp/Services/TrainingService.swift` | Orchestrates dataset prep, training, save |
| `AnvilApp/Services/AdapterRegistry.swift` | Scans and lists saved adapters |
| `AnvilApp/UI/Training/TrainingSheet.swift` | Training config + progress sheet |
| `AnvilApp/UI/Generate/AdapterPicker.swift` | LoRA dropdown in Generate view |

### AnvilApp (Modified)

| File | Changes |
|------|---------|
| `AnvilApp/Services/GenerationService.swift` | Adapter injection before generation, tracking |
| `AnvilApp/UI/Components/HistoryListView.swift` | Multi-select mode + "Train LoRA" button |

### AnvilCore (Modified)

| File | Changes |
|------|---------|
| `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift` | LoRA inject/remove/load methods |
| `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift` | LoRA forwarding methods |
| `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift` | LoRA forwarding methods |

---

## Scope

### In Scope (Phase 10a)

- `LoRALinear` MLXNN Module (forward, merge, unmerge, freezeBase)
- `LoRAInjector` (inject, remove, extractWeights, loadWeights)
- `LoRATrainer` (flow-matching loss, AdamW, epoch-level progress)
- `LoRAConfig` and `LoRAAdapter` data types
- `TrainingService` (orchestrates full training flow)
- `AdapterRegistry` (file-system scanner for saved adapters)
- Training Sheet UI (config + progress states)
- History view multi-select + "Train LoRA" button
- Adapter Picker dropdown in Generate view
- Adapter injection/removal in GenerationService (lazy pattern)
- DiTActor LoRA methods (inject/remove/loadWeights)
- Pipeline/DiffusionActor forwarding for LoRA operations
- Lightweight file import ("+ Add audio files..." with text per file)
- Adapter saved as safetensors + JSON metadata
- Adapter compatibility filtering by base model category
- Error recovery: fall back to base model on adapter failure
- Stop Training (saves best checkpoint if 2+ epochs done)

### Out of Scope (Phase 10b)

- Advanced training config (target layer picker, LR scheduler, alpha)
- Gradient accumulation for larger effective batch sizes
- Stop/resume training (save full optimizer state mid-run)
- Adapter composition (multiple simultaneous LoRAs)
- Adapter management UI (rename, delete, export)
- Training history / loss curve visualization
- Community adapter sharing (download from HuggingFace)
- Stable Audio LoRA (ACE-Step DiT only in 10a)
- Estimated training time display (needs profiling data)
