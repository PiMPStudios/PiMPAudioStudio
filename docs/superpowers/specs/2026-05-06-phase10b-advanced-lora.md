# Phase 10b: Advanced LoRA — Design Spec

## Overview

Builds on Phase 10a's LoRA foundation (LoRALinear, LoRAInjector, LoRATrainer,
AdapterRegistry, TrainingService, TrainingSheet, AdapterPicker) to add training
improvements and adapter management capabilities.

**Scope:** Training improvements + adapter management. Community sharing deferred
to Phase 10c.

## Goals

1. Make training more configurable and resumable (target layer picker, LR
   scheduler, gradient accumulation, stop/resume)
2. Provide visibility into training progress (loss curve visualization, ETA)
3. Enable adapter lifecycle management (rename, delete, export, merge)
4. Support adapter composition via merge-to-new (one-time weight arithmetic)

## Architecture

### Training Improvements

**LoRAConfig extensions** — The existing struct gains new fields:

- `targetLayers: Set<TargetLayer>` — which module types to inject (self-attn,
  cross-attn, mlp)
- `warmupFraction: Float` — default 0.10, feeds cosine annealing scheduler
- `gradientAccumulationSteps: Int` — default 1, accumulate before optimizer step
- `maxEpochs: Int` — renamed from `epochs` to clarify it's the cap

**CosineScheduler** — New utility struct in `AnvilCore/LoRA/`:

- Linear warmup from 0 to baseLR over `warmupFraction * totalSteps` steps
- Cosine decay from baseLR to 0 over remaining steps
- Called each step inside LoRATrainer to update optimizer LR

**LoRATrainer extensions:**

- Tracks `globalStep` counter across epochs
- Creates CosineScheduler on init, updates optimizer LR each step
- Gradient accumulation: sums gradients over N mini-batches, divides, then steps
- Checkpoint support: save/resume with full state reconstruction
- Cooperative cancellation preserved from 10a

**TrainingProgress extensions:**

- `learningRate: Float` — current LR for UI display
- `estimatedTimeRemaining: TimeInterval?` — ETA based on steps/sec average
- `lossHistory: [Float]` — appended each step for loss curve rendering

### Adapter Management

**AdapterRegistry extensions:**

- `rename(adapterID:to:)` — moves directory, updates metadata.json
- `delete(adapterID:)` — removes directory; if adapter is currently active,
  removes LoRA from pipeline first
- `export(adapterID:to:)` — copies .safetensors + metadata to user-chosen
  location
- `merge(adapterA:adapterB:alpha:name:)` — loads both weight sets, computes
  `alpha * A + (1-alpha) * B`, saves as new adapter

**Merge constraint:** Both adapters must have the same rank. UI enforces this by
disabling the merge button when ranks differ.

### New UI Components

**AdaptersSettingsView** — Dedicated "Adapters" tab in Settings window:

- List of all adapters with metadata (name, base model, rank, date)
- Swipe-to-delete (with confirmation), context menu rename
- Export button triggers NSSavePanel
- "Merge Two" button opens MergeSheet

**MergeSheet** — Modal for adapter composition:

- Picker A, Picker B (filtered to same base model + same rank; shows rank
  badge in each row for clarity)
- Alpha slider (0.0 = all B, 1.0 = all A) with labels
- Name text field for the new merged adapter
- Merge button triggers AdapterRegistry.merge()

**LossCurveView** — Mini line chart using SwiftUI Charts:

- Displays lossHistory as LineMark
- Auto-scaled Y axis (excludes zero)
- 120pt height, embedded in TrainingSheet progress state

**TrainingSheet extensions:**

- Config state: target layer multi-select, warmup fraction stepper, gradient
  accumulation stepper
- Progress state: LossCurveView, current LR, ETA display
- "Pause" button saves checkpoint, returns to config with "Resume" option

## Data Types

### CosineScheduler

```swift
public struct CosineScheduler: Sendable {
    public let baseLR: Float
    public let warmupFraction: Float
    public let totalSteps: Int

    public init(baseLR: Float, warmupFraction: Float = 0.10, totalSteps: Int)
    public func learningRate(atStep step: Int) -> Float
}
```

### TrainingCheckpoint

```swift
public struct TrainingCheckpoint: Codable, Sendable {
    public let epoch: Int
    public let globalStep: Int
    public let totalSteps: Int
    public let bestLoss: Float?
    public let lossHistory: [Float]
    public let weightsURL: URL
    public let optimizerStateURL: URL
    public let config: LoRAConfig
}
```

### TargetLayer

```swift
public enum TargetLayer: String, Codable, CaseIterable, Sendable {
    case selfAttention = "self_attn"
    case crossAttention = "cross_attn"
    case mlp = "mlp"

    /// Maps to the property name pattern used by LoRAInjector
    public var moduleName: String {
        switch self {
        case .selfAttention: return "selfAttn"
        case .crossAttention: return "crossAttn"
        case .mlp: return "mlp"
        }
    }
}
```

### LoRAConfig (updated)

```swift
public struct LoRAConfig: Codable, Sendable {
    public var rank: Int
    public var alpha: Float
    public var dropout: Float
    public var learningRate: Float
    public var batchSize: Int
    public var maxEpochs: Int
    public var targetLayers: Set<TargetLayer>
    public var warmupFraction: Float
    public var gradientAccumulationSteps: Int
}
```

### TrainingProgress (updated)

```swift
public struct TrainingProgress: Sendable {
    public let epoch: Int
    public let totalEpochs: Int
    public let step: Int
    public let loss: Float
    public let learningRate: Float
    public let estimatedTimeRemaining: TimeInterval?
    public let lossHistory: [Float]
}
```

## Data Flows

### Training (with improvements)

```text
User taps "Train LoRA" -> TrainingSheet (config state)
  -> User picks target layers, sets warmup, accumulation
  -> TrainingService.startTraining()
    -> pipeline.trainLoRA(samples:config:onProgress:)
      -> DiffusionActor -> DiTActor -> LoRATrainer
        -> CosineScheduler provides LR each step
        -> Gradient accumulation over N steps
        -> Progress callback includes LR, ETA, lossHistory
    -> TrainingSheet shows LossCurveView + progress
  -> User can pause -> checkpoint saved to disk
  -> User resumes -> LoRATrainer.resume(from:) continues
```

### Adapter Merge

```text
AdaptersSettingsView -> "Merge Two" button
  -> MergeSheet: pick A, pick B, alpha slider, name
  -> AdapterRegistry.merge(A, B, alpha, name)
    -> Load A weights, load B weights
    -> alpha * A + (1-alpha) * B per key
    -> Save as new adapter with fresh metadata
  -> List refreshes with new merged adapter
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| OOM during training | Auto-checkpoint, surface error with "Resume with smaller batch?" option |
| NaN/Inf loss (3 consecutive) | Halt, save checkpoint at last good step, report to UI |
| Checkpoint missing on resume | Validate URLs before resume; if missing, offer "Start fresh?" and auto-delete corrupt checkpoint files on confirmation |
| Close TrainingSheet mid-training | Training continues in background; re-open reconnects to progress stream |
| Delete currently-active adapter | Remove LoRA from pipeline first, then delete files |
| Merge incompatible base models | Prevented at UI level (picker B filtered to same baseModelID) |
| Merge different ranks | Blocked; merge button disabled with note "Same rank required" |
| Rename to existing name | Throws `AdapterError.nameExists`, UI shows inline error |
| Export write failure | Standard system error from NSSavePanel |

## Edge Cases

- `gradientAccumulationSteps > dataset size` — clamp to dataset size, log
  warning
- `warmupFraction = 0` — skip warmup, start at baseLR
- `warmupFraction = 1` — entire training is linear ramp (valid, no guard)
- Resume after model weights changed — checkpoint stores `baseModelID`; refuse
  resume on mismatch
- All target layers deselected — "Train" button disabled; minimum one required
- Adapter directory manually deleted — `listAdapters()` skips corrupt entries,
  logs warning
- Identical adapters merged — valid, produces identical result (no special case)

## Testing Strategy

### Unit-Testable (pure logic, no Metal)

1. **CosineScheduler** — LR at step 0 == 0, LR at warmup boundary == baseLR,
   LR at end approaches 0, monotonic in each phase
2. **TrainingCheckpoint** — Codable round-trip with equality check
3. **TargetLayer** — `moduleName` mapping correctness
4. **AdapterRegistry** (temp directory):
   - rename moves dir + updates metadata
   - delete removes dir
   - merge arithmetic: verify `alpha * A + (1-alpha) * B` for known arrays
   - listAdapters skips corrupt entries
5. **Gradient accumulation** — averaged result matches expected for known inputs

### Build-Verified (Xcode build, no unit test)

- LoRATrainer + CosineScheduler integration
- TrainingSheet UI states and transitions
- AdaptersSettingsView list operations
- MergeSheet -> AdapterRegistry pipeline
- LossCurveView rendering with Charts

### Manual Verification

- Train with gradient accumulation = 4, verify loss decreases
- Pause training, quit app, relaunch, resume from checkpoint
- Verify LR curve in LossCurveView matches cosine shape
- Merge two adapters, generate with result
- Rename adapter, verify AdapterPicker reflects new name
- Delete adapter while selected, verify graceful fallback to "None"
- Export adapter, confirm file is self-contained (safetensors + metadata)

## File Manifest

### New Files (AnvilCore)

| File | Responsibility |
|------|---------------|
| `AnvilCore/Sources/AnvilCore/LoRA/CosineScheduler.swift` | LR scheduling utility |
| `AnvilCore/Sources/AnvilCore/LoRA/TrainingCheckpoint.swift` | Checkpoint data type + serialization |

### New Files (AnvilApp)

| File | Responsibility |
|------|---------------|
| `AnvilApp/UI/Settings/AdaptersSettingsView.swift` | Adapter list + management actions |
| `AnvilApp/UI/Settings/MergeSheet.swift` | Merge-to-new modal UI |
| `AnvilApp/UI/Training/LossCurveView.swift` | Loss history line chart |

### Modified Files

| File | Changes |
|------|---------|
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAConfig.swift` | Rename `epochs` to `maxEpochs`; add targetLayers, warmupFraction, gradientAccumulationSteps; add TargetLayer enum. Update all call sites (TrainingService, TrainingSheet). |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRATrainer.swift` | CosineScheduler integration, gradient accumulation, checkpoint/resume, globalStep tracking |
| `AnvilCore/Sources/AnvilCore/LoRA/LoRAInjector.swift` | Accept Set<TargetLayer> to filter which modules get injected |
| `AnvilApp/Services/AdapterRegistry.swift` | Add rename, delete, export, merge methods |
| `AnvilApp/Services/TrainingService.swift` | Checkpoint save/resume orchestration, ETA calculation |
| `AnvilApp/UI/Training/TrainingSheet.swift` | Target layer picker, warmup/accumulation controls, LossCurveView, pause/resume, ETA |
| `AnvilApp/UI/Settings/SettingsView.swift` | Add "Adapters" tab |

## Out of Scope (deferred to 10c)

- Community adapter sharing / download
- Adapter versioning / history
- Multi-adapter stacking (apply multiple adapters simultaneously)
- Custom LR schedule curves (only cosine+warmup for now)
- Training data augmentation
