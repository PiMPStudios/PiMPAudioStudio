# Training Dataset Packages

PiMP Audio Studio can build and import local LoRA training datasets without a
cloud service. The local-file builder preserves complete sources and writes
package v2, while compatibility bundles let the current local trainer and
remote worker consume the same folder or zip.

## Package Version Status

New packages use the typed whole-track package-v2 contract. Existing v1
clip-based packages remain readable through a non-destructive compatibility
adapter and are not rewritten merely by opening them. See
[dataset-package-v2.md](dataset-package-v2.md) for the contract, training-plan,
and migration rules.

## Build From Local Files

Use **Build > Datasets > From Local Files...** to create a package under:

```text
<Output Directory>/TrainingDatasets/<source>_local_<timestamp>/
```

The package contains:

- `sources/` - byte-preserved complete imported tracks; these are canonical in
  both build modes.
- `clips/` - optional derived training clips written only in **Extract Clips**
  mode. WAV/PCM is the recommended format; M4A remains available.
- `captions.json` - compatibility annotations for assets selected by the
  default training plan.
- `dataset_config.json` - generic audio-directory training config.
- `dataset_manifest.json` - canonical v2 source tracks, assets, derivation
  relationships, annotations, and explicit training selections/ranges.
- `training_bundle.json` - compatibility projection of the default v2 training
  plan.
- `training_bundle_full_mix.json` - current default-plan entry point for PiMP
  Audio Studio and the remote worker. Its selected roles come from the v2 plan.
- `training_bundle_vocals.json` - optional vocal-stem bundle written after local enrichment.
- `training_bundle_instrumental.json` - optional instrumental-stem bundle written after local enrichment.
- `remote_training_package.json` - provider-neutral descriptor for future RunPod/DigitalOcean GPU workers.
- `package_validation.json` - validation result written at package time.
- `preprocess_cache.json` - adapter-agnostic cache manifest tracking bundle fingerprints, cache status, and sample-level tensor locations shared by LoRA and LoKR preprocessing.
- `review_backups/` - timestamped snapshots of package JSON written before review saves.

**Whole Tracks** is the default and recommended ACE-Step mode. It creates one
`full_mix` asset per preserved source and selects those assets for training.
ACE-Step's effective 240-second limit is recorded as a backend-limited training
range without truncating or replacing the preserved source.

**Extract Clips** is an explicit alternative for loops, long recordings, and
controlled sample counts. Each clip is a derived `clip` asset linked to its
preserved `full_mix`; the clips, rather than the full mixes, are selected by
that plan. Sampling can be deterministic **Evenly Spread**, **Sequential**, or
**Random Seeded**, with mode and seed stored in each clip's derivation details.

Readable audio headers are recorded when AVFoundation can open an asset.
Incomplete metadata is allowed, but package validation reports unknown ranges
and missing files.

## Annotation Provenance

Each v2 asset carries independent caption and lyrics annotations with
backend/model provenance, language, raw artifact paths, and review status.
Filename placeholders from blank style descriptions are marked as `filename`,
manual edits in Dataset Review are marked as `manual`, imported sidecars use
sidecar/CSV sources, native metadata captions use `native`, and native
Qwen3-ASR output uses `native-qwen3-asr`.

Dataset Review lists selected and opt-in assets together. It shows each asset
role, source-track relationship, parent asset, range policy, and effective
training duration. **Include in Training** edits the default v2 training plan.
Stems and other representations remain opt-in; selecting multiple
representations from one source displays an effective-weight warning. Saving
writes a backup first, then refreshes the canonical manifest, compatibility
bundles, captions, validation, remote descriptor, and preprocess manifest.

Use **Export Sidecars** in the review workspace to write reviewed metadata back beside the audio files in ACE-compatible formats:

- `{filename}.caption.txt`
- `{filename}.lyrics.txt`
- `{filename}.json`
- package-level `metadata.csv`

The exported JSON carries caption, prompt, lyrics, transcript, BPM, keyscale, language, time signature, role, notes, source fields, and the current annotation provenance block.

## Preprocess Cache

Training from a PiMP Audio Studio dataset bundle now fills the package cache instead of only declaring it. When a clip is preprocessed, PiMP Audio Studio writes a sample-level `.safetensors` file and updates a bundle index:

```text
preprocessed/<bundle>/samples/<clip>_<fingerprint>.safetensors
preprocessed/<bundle>/index.json
```

The cache key includes the bundle file, bundle name, clip id, source audio path, prompt, and lyrics, so prompt or lyric edits produce a different sample artifact. `preprocess_cache.json` reports each bundle as `ready_for_preprocess`, `partial`, `complete`, or `waiting_for_bundle`.

The current cache is sample-level training data. It is the shared handoff for local LoRA/LoKR training and remote workers, while a fuller clip-level tensor cache can still be added later if we want more random-window variation across repeated runs.

## Local Enrichment

After a local-file package is built, use **Review Dataset** from the build sheet or open it from **Build > Datasets**, then use **Enrich Dataset** in the review workspace to enrich that package in place. PiMP Audio Studio owns this workflow directly; it does not call or require any sibling repo.

The enrichment backend is native Swift/MLX Demucs via `demucs-mlx-swift`, using `htdemucs` in vocals-plus-instrumental mode. PiMP Audio Studio writes `stems/`, `separation.json`, `training_bundle_vocals.json`, `training_bundle_instrumental.json`, refreshed manifest metadata, and package validation itself.

Native Demucs model selection can be overridden with `PIMP_AUDIO_DEMUCS_MODEL_NAME`; the optional `PIMP_AUDIO_DEMUCS_MODEL_DIR` points at a local Demucs MLX model directory. The default model remains `htdemucs`.

Before enrichment starts from the app UI, PiMP Audio Studio releases idle generation, cover-song, SFX, LoRA, and LLM model weights through the same GPU-memory release path used by Settings. The enrichment loop also compacts the MLX allocator between Demucs separation and Qwen transcription work so dataset builds do not stack avoidable model/cache memory on top of the active generation pipeline.

Enrichment also transcribes each generated `vocals.wav` with native `speech-swift` Qwen3-ASR. The quality path uses `Qwen3-ASR-1.7B` plus `Qwen3-ForcedAligner-0.6B` for word timestamps, then writes `lyrics.txt`, `lyrics.timed.json`, and `lyrics.lrc` next to the stem files. Those transcripts are copied into `captions.json` and `training_bundle_vocals.json`; instrumental bundles keep lyrics empty.

Enrichment preserves separation/transcription metadata when captions are edited, promotes local stem tempo estimates into each clip's BPM field, and estimates instrumental key/keyscale with native DSP analysis. It also upgrades local-builder/manual captions into deterministic native metadata captions using the original caption seed plus BPM, key, vocal/transcript presence, and stem role context, then refreshes the full-mix training bundle so **Open in Training** uses the enriched captions. Filename placeholder captions are replaced rather than used as creative seed text. Imported sidecar/CSV/remote captions are preserved instead of being overwritten by this automatic pass.

The native captioner is metadata-based, not a full audio-language model. It gives useful training prompts immediately while keeping the local package workflow zero-Python; heavier ACE-Step captioner integration remains a future optional backend.

## Train From JSON

Use **Build > Training > Train from Files... > Add audio files or JSON...** when you already have a packaged dataset. From the Datasets review workspace, **Open in Training** imports the selected package directly once blockers are cleared.

Recommended JSON entry points:

- Use `training_bundle_vocals.json` for vocal-stem training. PiMP Audio Studio selects `vocals.wav` assets and uses transcripts/lyrics from each clip.
- Use `training_bundle_instrumental.json` for instrumental-stem training. PiMP Audio Studio selects instrumental assets and intentionally leaves lyrics empty.
- Use `training_bundle_full_mix.json` for PiMP Audio Studio local-file packages created by **From Local Files...**.
- `dataset_config.json` is also supported. If a `caption_file` is provided, or a sibling `captions.json` exists, PiMP Audio Studio uses those captions and preserves lyrics for vocal file paths.

The training sheet reports how many files were imported from JSON, how many include lyrics, and how many duplicate paths were skipped.

## Experimental LoKR Training

Use **Build > Training > LoKR** to select a reviewed dataset package and open it in the native LoKR trainer. The LoKR path uses the same package manifests, full-mix bundle, preprocessing cache, warmup, gradient accumulation, and base-model selection flow as LoRA, but writes adapter metadata with `kind: "lokr"` plus a `lokrConfig` block.

LoKR adapters can be selected for generation like LoRA adapters, but stacks are kept kind-compatible: LoRA adapters stack with LoRA, LoKR adapters stack with LoKR. Merge/resume/checkpoint behavior remains LoRA-only until the LoKR training loop has equivalent checkpoint semantics.

## Remote Package Shape

Use **Export Zip** from Dataset Review after selected assets are approved. The
same reviewed packages are also available under **Build > Training > Remote
GPU** for export-focused handoff. The exported zip keeps `sources/`, optional
derived assets, and their metadata together. Remote workers should:

1. Unzip the package.
2. Read `remote_training_package.json`.
3. Resolve `entry_bundle` from the package root.
4. Resolve selected asset paths relative to the bundle's `dataset_dir`.

The default whole-track plan selects full mixes only. After local enrichment,
the same package can include linked stems and optional stem bundles. Dataset
Review can opt those roles into the default plan without losing their source
relationships.

Remote packages include `preprocess_cache.json`, `annotations/**`, and `preprocessed/**` in their copy rules. Local bundle preprocessing can now populate `preprocessed/<bundle>/samples/*.safetensors` plus `index.json`; remote workers should read or write the same cache shape instead of inventing provider-specific tensor layouts.

Remote launch requests use the provider-neutral shape in [remote-worker-contract.md](remote-worker-contract.md). In short, PiMP Audio Studio should generate a `remote_job.json`, provider adapters should launch a worker without changing that schema, and workers should return `remote_artifacts/result.json`, `logs.jsonl`, `metrics.jsonl`, checkpoints, and final adapter artifacts in the standard layout. The first portable CUDA worker is in `remote-worker/`; copy it to a stock RunPod PyTorch pod, run `install.sh`, then run `pimp-remote-worker train <exported-job.zip>`.
