# Training Dataset Packages

Anvil can build and import local LoRA training datasets without a cloud service. The current dataset builder is local-file/full-mix first, with package metadata shaped so a future remote worker can consume the same folder or zip.

## Build From Local Files

Use **Build > Datasets > From Local Files...** to slice local audio into a package under:

```text
<Output Directory>/TrainingDatasets/<source>_local_<timestamp>/
```

The package contains:

- `clips/` - generated training clips. WAV/PCM is the default and recommended format; M4A remains available as a compact option for quick tests.
- `captions.json` - clip captions for audio-dir style tooling.
- `dataset_config.json` - generic audio-directory training config.
- `dataset_manifest.json` - full package metadata, source file list, clip list, audio header metadata, and package status.
- `training_bundle.json` - Anvil training bundle for the generated full-mix clips.
- `training_bundle_full_mix.json` - explicit full-mix bundle; this is the safest file to re-import into Anvil.
- `training_bundle_vocals.json` - optional vocal-stem bundle written after local enrichment.
- `training_bundle_instrumental.json` - optional instrumental-stem bundle written after local enrichment.
- `remote_training_package.json` - provider-neutral descriptor for future RunPod/DigitalOcean GPU workers.
- `package_validation.json` - validation result written at package time.

Readable audio headers are recorded when AVFoundation can open the source/clip. Invalid stubs or unsupported headers do not fail the package; they are marked with `metadata_status`.

## Local Enrichment

After a local-file package is built, open it from **Build > Datasets** and use **Enrich Dataset** in the review workspace to enrich that package in place. AnvilApp owns this workflow directly; it does not call or require any sibling repo.

The enrichment backend is native Swift/MLX Demucs via `demucs-mlx-swift`, using `htdemucs` in vocals-plus-instrumental mode. AnvilApp writes `stems/`, `separation.json`, `training_bundle_vocals.json`, `training_bundle_instrumental.json`, refreshed manifest metadata, and package validation itself.

Native Demucs model selection can be overridden with `ANVIL_DEMUCS_MODEL_NAME`; the optional `ANVIL_DEMUCS_MODEL_DIR` points at a local Demucs MLX model directory. The default model remains `htdemucs`.

Enrichment also transcribes each generated `vocals.wav` with native `speech-swift` Qwen3-ASR. The quality path uses `Qwen3-ASR-1.7B` plus `Qwen3-ForcedAligner-0.6B` for word timestamps, then writes `lyrics.txt`, `lyrics.timed.json`, and `lyrics.lrc` next to the stem files. Those transcripts are copied into `captions.json` and `training_bundle_vocals.json`; instrumental bundles keep lyrics empty.

Enrichment preserves separation/transcription metadata when captions are edited and records a local BPM estimate from stem audio. Musical key analysis is still represented as a package status field and remains pending until a standalone local implementation is wired.

## Train From JSON

Use **Build > Training > Train from Files... > Add audio files or JSON...** when you already have a packaged dataset. From the Datasets review workspace, **Open in Training** imports the selected package directly once blockers are cleared.

Recommended JSON entry points:

- Use `training_bundle_vocals.json` for vocal-stem training. Anvil selects `vocals.wav` assets and uses transcripts/lyrics from each clip.
- Use `training_bundle_instrumental.json` for instrumental-stem training. Anvil selects instrumental assets and intentionally leaves lyrics empty.
- Use `training_bundle_full_mix.json` for Anvil local-file packages created by **From Local Files...**.
- `dataset_config.json` is also supported. If a `caption_file` is provided, or a sibling `captions.json` exists, Anvil uses those captions and preserves lyrics for vocal file paths.

The training sheet reports how many files were imported from JSON, how many include lyrics, and how many duplicate paths were skipped.

## Remote Package Shape

Use **Export Zip** from the Datasets review workspace after clips are approved. The exported zip keeps the package root intact. Remote workers should:

1. Unzip the package.
2. Read `remote_training_package.json`.
3. Resolve `entry_bundle` from the package root.
4. Resolve clip paths relative to the selected bundle's `dataset_dir`.

The initial package is full-mix only. After local enrichment, the same package can also include `stems/`, `training_bundle_vocals.json`, and `training_bundle_instrumental.json`; remote workers should pick the bundle that matches the intended training target.
