# Dataset And Training Internals

## Package V2

Canonical whole-track format. Normative schema:

```text
docs/schemas/dataset-package-v2.schema.json
docs/dataset-package-v2.md
```

Swift types and validators:

```text
PiMPAudioCore/Sources/PiMPAudioCore/Dataset/
```

Core collections:

1. `source_tracks` — complete imported tracks
2. `assets` — full_mix / vocals / instrumental / clip assets
3. `training_plans` — selected assets + explicit ranges/windows

Range fields are intentionally separate:

- `asset.source_range` — complete asset duration
- `selection.training_range` — portion exposed to a backend
- `selection.sample_window_seconds` — per-step window when applicable
- `selection.range_policy` — full / backend_limit / manual / legacy

ACE-Step effective preprocess max is 240 seconds; whole-track packages record
backend-limited training ranges without destroying preserved sources.

### Compatibility

- New writers emit v2.
- V1 packages load through `DatasetPackageV1Adapter` without rewriting disk.
- V1 full-mix rows become `clip` assets (they do not prove whole sources exist).
- Training bundles (`training_bundle_*.json`, `captions.json`) are projections
  for local trainers and remote workers.

App packaging helpers:

- `LocalFilesDatasetBuilder`
- `TrainingDatasetPackageWriter`
- `TrainingDatasetPackageEditor`
- `YouTubeDatasetBuilder`

## Enrichment

`LocalDatasetEnrichmentService` orchestrates:

| Backend | Implementation |
| --- | --- |
| Lightweight native | Metadata captions + speech-swift Qwen3-ASR |
| Native ACE | Four-bit captioner/transcriber in Swift/MLX |
| Stems | `demucs-mlx-swift` htdemucs vocals+instrumental |

Native packs install via `NativeModelPackInstaller` with pinned revisions,
checksums, validation, and transactional replace.

Provenance is stored per caption/lyrics annotation (backend, revision,
quantization, raw paths, review status).

## Preprocess Cache

Training fills:

```text
preprocessed/<bundle>/samples/<clip>_<fingerprint>.safetensors
preprocessed/<bundle>/index.json
```

`preprocess_cache.json` tracks bundle readiness. Cache keys include bundle,
clip id, audio path, prompt, and lyrics.

## Local Trainers

Under `PiMPAudioCore/.../LoRA/`:

| Component | Role |
| --- | --- |
| `LoRATrainer` | Local LoRA loop |
| `LoKRTrainer` | Local LoKR loop (incl. DoRA option) |
| `LoRAInjector` / `LoKRInjector` | Parameter injection |
| `CosineScheduler` | LR schedule with non-zero first warmup step |
| `TrainingCheckpoint` | Weight + optimizer state artifacts |
| `AdapterTrainingSchedule` | Timestep / CFG dropout recipe |

Important local behavior:

- Microbatch size 1 + gradient accumulation (remote `batchSize` is not local)
- Explicit AdamW weight decay 0 + global grad clip
- Logit-normal max timestep schedule aligned with ACE-Step training
- 15% CFG dropout with learned null embedding; random draws outside grad closure
- Seeded shuffle/noise/timestep/CFG/dropout streams
- Epoch-end validation loss on fixed held-out split; dropout off for val
- Pause writes optimizer tensors + current/best weight snapshots

App orchestration: `TrainingService` + `TrainingSheet` recipes.

### Recipes (UI)

| UI name | Internals |
| --- | --- |
| Narrow | `balancedLocal` defaults (smaller rank/layers, Q/V) |
| Wide LoRA | rank 96, alpha 2×, dropout, 32 layers, q/k/v/o, more epochs |
| Wide LoKR | expanded LoKR footprint with DoRA defaults |

Memory guards block known dangerous shapes (for example broad full-attention
LoKR on XL) via preflight reports shared with MCP.

## Remote Jobs

`RemoteTrainingJob` builds provider-neutral job metadata and zip contents:

- Reviewed package assets / ranges / provenance
- `remote_job.json`
- Helper scripts and README

RunPod control plane: `RunPodService` + `RunPodLifecyclePanel`.

Worker contract details: [Remote Worker](remote-worker.md),
`docs/remote-worker-contract.md`.

## MCP Surfaces

Training-related tools include list/inspect/compare datasets and runs,
`training_preflight`, `start_training` (dry-run default), `stop_training`,
`tail_training_log`, and RunPod launch/status/terminate tools.

## Tests To Prefer

- `DatasetPackageV2Tests`
- `DatasetAnnotationBackendTests`
- `LoRATrainerTests` / adapter random + merge tests
- `NativeModelPackInstallerTests` / manifest tests
