# Datasets And Training

PiMP Audio Studio can build reviewed training packages from local audio, enrich
them with stems and lyrics, and use those packages for local or remote adapter
training.

## Build A Dataset

Open:

```text
Build -> Datasets -> From Local Files...
```

Choose source audio files and a dataset name. **Whole Tracks** is the default
source mode and preserves every complete input. Choose **Extract Clips** only
when you intentionally want clip length, clips-per-file, total-cap, and sampling
controls. The app writes a package under your selected output directory:

```text
TrainingDatasets/<dataset-name>_local_<timestamp>/
```

## Clip Sampling (Optional)

Sampling appears only in Extract Clips mode and controls how source files are
sliced into derived training clips.

- `Evenly Spread` - default. Samples from across each file so long sources do
  not only train on the opening section.
- `Sequential` - old first-window behavior. Useful for quick deterministic
  tests, but less representative for long files.
- `Random Seeded` - reproducible random windows. The seed can be typed directly.

The builder supports up to 1,000 total clips and warns when your selected caps
will not cover every source file.

## Review Before Training

After build, use:

```text
Build -> Datasets -> Review Dataset
```

Review captions, lyrics, BPM, key, language, time signature, notes, and package
blockers. `Open in Training` should only be used once the package is ready.

The review workspace can import sidecars and export reviewed metadata in
ACE-compatible formats.

## Enrich Dataset

Dataset enrichment annotates complete full mixes locally. Use its compact
**Options** menu to choose:

- **Automatic** - use Native ACE when both optional packs are installed and the
  Mac meets the memory policy; otherwise use the lightweight native path.
- **Lightweight Native** - use the built-in metadata caption and Qwen3-ASR path.
- **Native ACE (Advanced)** - require the four-bit ACE captioner and transcriber.
- **Create vocal and instrumental stems** - separately opt in to Demucs stem
  separation and stem training bundles.

All modes:

- Caption and transcribe full mixes, or transcribe vocals when the lightweight
  path is used with stems.
- Use Qwen3 Forced Aligner for timed ACE lyrics where available.
- Estimates or preserves BPM/key metadata where available.
- Store model revision, quantization, prompt/settings, elapsed time, and raw
  output paths so review edits do not erase provenance.

Install the Native ACE packs under **Settings -> Models -> Dataset Annotation
(Advanced)**. **Download** retrieves the pinned PiMP Audio Studio pack from
Hugging Face, verifies every file against the pack manifest, validates the
completed pack, and then installs it transactionally. The overflow menu keeps
**Import Local Pack** available for offline transfers. Both packs use about
9.34 GB of disk; 24 GB unified memory is the provisional minimum and 32 GB or
more is recommended.

## Which Bundle To Train

Use the bundle that matches the adapter you want:

- `training_bundle_full_mix.json` - style/full-song adapter.
- `training_bundle_vocals.json` - voice/vocal adapter after enrichment.
- `training_bundle_instrumental.json` - instrumental/style adapter after
  enrichment.

If you built a full-mix dataset and never enriched it, use
`training_bundle_full_mix.json`.

## Local Training

Open:

```text
Build -> Training
```

Then choose `LoRA` or `LoKR`, select `Train from Files...`, and pick the package
or training bundle.

Local training is Swift/MLX only. It supports:

- Narrow and Wide recipes.
- LoRA rank choices including 96.
- LoKR linear-dim recipes.
- LoKR DoRA.
- Deterministic seeds.
- Validation loss.
- Checkpoints and resume.
- Optimizer-state restore.
- Training-completion audition snapshots.

Local training uses microbatch-1 with gradient accumulation. Batch-size fields
in remote job schemas are for remote/cloud workers, not the native local trainer.

## Narrow Versus Wide

`Narrow` is the safer local starting point. It trains fewer layers/projections,
uses less memory, and is easier to iterate on.

`Wide` trains broader layer/projection coverage. It can produce a stronger
adapter, but it costs more memory and is more likely to need careful settings.

For a new dataset, start Narrow unless you specifically want the larger recipe.
Use Wide when you have already proven the dataset is healthy and want stronger
style capture.

## Dataset Size Hints

More clips usually give a better spread, but they also increase training time.

Useful starting points:

- 50-200 clips - quick smoke tests, high risk of overfitting.
- 200-600 clips - practical local training range.
- 600-1,000 clips - stronger coverage, better suited to remote training or long
  local runs.

For very long source files, clip count matters more than source file duration.
If you have 32 ninety-minute sources but only create 160 clips, training sees
160 windows, not the full 48 hours of audio.

## Remote Training

Remote GPU training packages the reviewed dataset plus `remote_job.json` for a
cloud worker. See [Remote GPU Training](remote-gpu-training.md) for the RunPod
flow.

Remote jobs currently operate on the clip/stem package created by the app. A
future Side-Step source-job backend is planned for workflows that upload full
source tracks plus sidecars and let the remote worker perform Side-Step-native
preprocessing/training.

## Deeper Technical Reference

For package file layout and engineering details, see:

```text
docs/training-datasets.md
```
