# Datasets And Training

PiMP Audio Studio can build reviewed training packages from local audio (or
YouTube imports), enrich them with stems and lyrics, and train adapters locally
or remotely.

## Build A Dataset

Open:

```text
Build -> Datasets
```

### From Local Files

Choose **From Local Files…**, pick source audio, and name the dataset.

| Source mode | Behavior |
| --- | --- |
| **Whole Tracks** (default) | Preserves every complete input as a source; trains on full mixes with backend range limits |
| **Extract Clips** | Explicitly slices clips with length, clips-per-file, total-cap, and sampling controls |

Packages land under:

```text
<Output Directory>/TrainingDatasets/<dataset-name>_local_<timestamp>/
```

### Clip Sampling (Extract Clips Only)

| Mode | Behavior |
| --- | --- |
| Evenly Spread | Default. Samples across each file; avoids training only on openings |
| Sequential | First-window style; good for deterministic smoke tests |
| Random Seeded | Reproducible random windows; seed is stored |

Limits:

- Up to **1,000** total clips
- The builder warns when caps will not cover every source file

Evenly-spread sampling uses bin centers (not exact first/last edges) and records
source path, start seconds, start frame, frame count, and sample rate in package
metadata.

### From YouTube

**From YouTube…** downloads/import-prepares audio into training-ready files and
can open the training sheet. Prefer local files when you already own the masters.

## Package Layout (User View)

A local package typically includes:

| Item | Purpose |
| --- | --- |
| `sources/` | Byte-preserved complete imported tracks |
| `clips/` | Optional derived clips (Extract Clips mode) |
| `dataset_manifest.json` | Canonical package-v2 graph |
| `captions.json` | Compatibility captions for the active plan |
| `training_bundle_full_mix.json` | Default full-mix training entry |
| `training_bundle_vocals.json` | After enrichment with stems |
| `training_bundle_instrumental.json` | After enrichment with stems |
| `package_validation.json` | Validation / blockers |
| `preprocess_cache.json` | Shared preprocess cache status |
| `annotations/` | Raw enrichment artifacts when present |
| `stems/` | Optional Demucs stems |

New packages use whole-track **package v2**. Older v1 packages still open through
a non-destructive compatibility adapter.

Technical contract: `docs/dataset-package-v2.md` and `docs/training-datasets.md`.

## Review Before Training

Select a package in the Datasets sidebar to open **Dataset Review**.

Review:

- Captions and lyrics
- BPM, key, language, time signature
- Notes and approval status
- Include-in-training selections
- Package validation blockers

Use **Open in Training** only when the package is ready (blockers cleared).

The review workspace can import sidecars and export reviewed metadata in
ACE-compatible formats:

- `{filename}.caption.txt`
- `{filename}.lyrics.txt`
- `{filename}.json`
- package-level `metadata.csv`

Saving review metadata writes a backup under `review_backups/` first.

## Enrich Dataset

**Enrich Dataset** annotates full mixes in place. Options menu:

| Option | Behavior |
| --- | --- |
| Automatic | Native ACE when both packs are installed and memory policy allows; else lightweight native |
| Lightweight Native | Built-in metadata captions + Qwen3-ASR path |
| Native ACE (Advanced) | Four-bit ACE captioner + transcriber packs |
| Create vocal and instrumental stems | Opt-in Demucs (Swift/MLX) + stem training bundles |

All enrichment modes:

- Caption and/or transcribe full mixes (or vocals on lightweight+stems paths)
- Use Qwen3 Forced Aligner for timed ACE lyrics when available
- Analyze BPM, key, and ACE-Step time signature (instrumental preferred when stems exist)
- Store model revision, quantization, settings, elapsed time, and raw output paths

Install Native ACE packs under **Settings → Models → Dataset Annotation
(Advanced)**. Combined pack size is about **9.34 GB**. **24 GB** unified memory
is provisional minimum; **32 GB+** recommended.

Before enrichment, the app releases idle generation/cover/SFX/LoRA/LLM weights to
free memory.

## Which Bundle To Train

| Bundle | Use when |
| --- | --- |
| `training_bundle_full_mix.json` | Style / full-song adapter; default for non-stem packages |
| `training_bundle_vocals.json` | Voice adapter after stem enrichment |
| `training_bundle_instrumental.json` | Instrumental/style adapter after stem enrichment |

If you never enriched stems, use the full-mix bundle.

## Local Training

Open:

```text
Build -> Training
```

Choose **LoRA** or **LoKR**, then pick a reviewed package or **Train from
Files…**.

Local training is **Swift/MLX only**. Features:

- Narrow and Wide recipes
- LoRA ranks including 96; LoKR linear-dim recipes
- LoKR DoRA
- Deterministic seeds
- Validation loss and best-checkpoint selection
- Checkpoints, pause/resume with optimizer state
- Training-completion audition snapshots (base + gain ladder)

Local trainers use **microbatch-1 with gradient accumulation**. Batch-size fields
in remote job schemas apply to remote/cloud workers, not the native local
trainer.

### Narrow Versus Wide

| Recipe | Intent |
| --- | --- |
| Narrow | Safer local default: fewer layers/projections, less memory, faster iteration |
| Wide | Broader coverage / stronger style capture; more memory and longer runs |

Wide LoRA defaults include a much larger footprint (for example rank 96, many
epochs, full attention projections). Start Narrow unless the dataset is proven.

### Dataset Size Hints

| Clips / samples | Guidance |
| --- | --- |
| 50–200 | Smoke tests; high overfit risk |
| 200–600 | Practical local range |
| 600–1,000 | Stronger coverage; better for remote or long local runs |

For long sources, **clip/sample count** matters more than wall-clock source
duration. Thirty-two ninety-minute files with only 160 clips still train 160
windows.

### Pause And Stop

While training is active you can pause/stop. If progress exists, the app offers
**Save & Pause** (checkpoint + optimizer state) versus discard.

### Completion Audition

After a successful run, use fixed-seed audition snapshots to render base and
selected gains (commonly `1.0`, `1.6`, `2.0`), save takes to History, and open
Trace Compare.

## Remote Training

Remote GPU packages the reviewed dataset plus `remote_job.json` for a cloud
worker. See [Remote GPU Training](remote-gpu-training.md).

Remote jobs currently operate on the clip/stem package plan selected in the app.
A future Side-Step source-job backend is planned for full source tracks plus
sidecars with remote preprocessing.

## Train From History

In **History → Select**, choose takes and click **Train LoRA** to open training
with those entries as the starting set. Prefer curated dataset packages for
serious style adapters.

## Deeper Technical Reference

```text
docs/training-datasets.md
docs/dataset-package-v2.md
docs/developer/dataset-and-training-internals.md
```
