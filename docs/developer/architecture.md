# Architecture

## Purpose

PiMP Audio Studio is a sandboxed macOS application that performs local music and
SFX generation, latent editing, dataset packaging, and adapter training on Apple
Silicon. Optional remote GPU training runs on external CUDA machines through a
portable worker.

## Layering

### App target (`PiMPAudioStudio/`)

Owns:

- SwiftUI shell (`MainShellView`, Generate, Editor, Datasets, Training, History,
  Presets, Settings)
- Orchestration services (`GenerationService`, `TrainingService`,
  `EditorService`, `RunPodService`, MCP façade, lyric writers)
- App-level models (generation parameters, history entries, traces)
- DSP helpers used by the Editor
- Sandbox-facing storage managers and Keychain wrappers

### Core package (`PiMPAudioCore/`)

Owns:

- Model registry and download client
- ACE-Step music inference stack (DiT, VAE, conditioning, diffusion)
- Stable Audio Open SFX stack
- Cover/tokenizer/LM pieces
- Dataset package-v2 types, validation, native ACE annotation runtime
- LoRA / LoKR trainers and injectors
- Quality metrics and shared diagnostics hooks

### Remote worker (`remote-worker/`)

Owns:

- Portable install onto stock CUDA pods
- Conversion of PiMP job packages into Side-Step datasets
- Preprocess + train loops on PyTorch/CUDA
- Standard `remote_artifacts/**` outputs

Never executed on macOS as part of the app runtime.

## Runtime Object Graph

`PiMPAudioStudioApp` constructs long-lived `StateObject`s:

- `AppState` + `ModelRegistry`
- `OutputDirectoryManager`
- `AudioPlayerService`
- `APIKeyStore`, `ProviderAccessStore`, `RunPodService`
- `LLMModelManager`, `IntentRouter`, `PromptEnhancer`, `LyricWriterManager`
- `SFXModelManager`
- `AdapterStackManager`, `GenerationParameters`, `PresetStore`
- `AppServices` (history, generation, editor, training, trace monitor, MCP)

Environment objects flow into the root `ContentView` → first launch or main
shell.

## Major Pipelines

### Music generation (ACE-Step)

```text
Prompt/lyrics (+ optional adapters)
  -> ConditioningActor / text+lyric encoders
  -> DiffusionActor + DiT
  -> VAE decode
  -> WAV + History + optional .trace.json
```

Cover path additionally tokenizes/encodes source audio and may use LM/audio-code
conditioning stages before diffusion.

### SFX generation (Stable Audio Open)

```text
SFX prompt
  -> T5 encode + projection
  -> SFX DiT diffusion
  -> SFX VAE decode
  -> WAV + History
```

### Editor

```text
Load audio -> editable latents + waveform
  -> Inpaint/Repaint (mask + schedule-aligned noise + blend)
  -> DSP chain (trim/fade/EQ/reverb/WSOLA/pitch)
  -> Export WAV/AIFF/M4A
```

Editor mutations are serialized so overlapping MCP/UI operations cannot clobber
the buffer.

### Dataset + local train

```text
Local files / YouTube
  -> package v2 writer
  -> Dataset Review (+ optional enrichment)
  -> preprocess cache samples
  -> LoRATrainer / LoKRTrainer (MLX)
  -> adapter registry + optional audition snapshots
```

### Remote train

```text
Reviewed package + remote_job.json
  -> zip export
  -> RunPod pod lifecycle (upload/start/monitor/pull)
  -> pimp-remote-worker on CUDA
  -> remote_artifacts + local adapter import
```

## Guidance And XL Policy

`GuidancePreset` encodes CFG schedules. Soft Late forces guidance scale 6.0 and
late-only CFG. XL contexts default to Constant and auto-nudge Soft Late →
Constant when XL models/adapters become active, while still allowing deliberate
Soft Late A/Bs.

## Concurrency And Cancellation

- Generation tasks capture immutable request/adapter context at submit time.
- Cancel waits for prior work to unwind before a new run mutates shared state.
- Model unload uses an explicit `.releasing` state so new generates cannot start
  mid-teardown.
- Training stop can cancel during model load/dataset prepare, not only mid-step.
- Large export/DSP/decode work runs off the main actor.

## Boundaries And Non-Goals

| Do | Do not |
| --- | --- |
| Keep app Swift/MLX | Ship Python inside the app |
| Put secrets in Keychain | Embed tokens in zips/traces/logs |
| Confirm billable cloud actions | Auto-launch paid pods without consent |
| Use package-relative remote paths | Allow zip path traversal |
| Prefer history/audio IDs in MCP | Assume sandbox can read arbitrary paths |

## Extension Points

- New models: `ModelSpec` + registry + loader name maps
- New DSP: `DSPEffect` + processor + Editor palette presentation
- New MCP tools: `PiMPAudioMCPToolCatalog` + command façade methods
- New remote backends: job schema + worker + `remote_artifacts` contract
