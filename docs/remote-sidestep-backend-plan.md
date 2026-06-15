# Remote Side-Step Backend Plan

This note captures the remote-training direction discussed after testing
Side-Step directly on a remote H200/DigitalOcean GPU and locally on Apple MPS.
It is a planning document only; no implementation has started from this note.

## Decision Candidate

Use Side-Step as the primary remote CUDA training backend for PiMP Audio Studio,
while keeping local training native Swift/MLX.

The macOS app should remain Swift-only and App Store friendly. Remote GPU jobs
can use Python, PyTorch, CUDA, Side-Step, and ACE-Step because they run outside
the app boundary on user-approved cloud hardware.

## Why This Came Up

The user ran a Side-Step workflow directly on a DigitalOcean H200 server using
the same long-form source collection and rich sidecars:

- 32 long source tracks, roughly 90 minutes each.
- Sidecars created with Gemini, ACE-Step captioner, ACE-Step transcriber, BPM,
  key, and structured lyric tags.
- Side-Step cloned directly on the cloud server.
- ACE-Step XL Base model downloaded from Hugging Face.
- Side-Step preprocessing and PP++/analysis run once.
- Multiple adapter recipes trained against the same tensor cache.

The surprising result was that four adapters could be trained in roughly an
hour once the server, models, preprocessing, and tensors were ready.

## What Actually Made It Fast

The speed was not only the H200. The larger win was the shape of the workload.

The inspected Desktop scripts under
`/Users/tweaver/Desktop/dark-gentlemen-training-scripts/` showed the following
pattern:

- `train-dark-gentlemen-lokr.sh` preprocesses the source set into
  `./tensors/TheDarkGentlemen`.
- The LoRA and PP++ scripts reuse that tensor cache instead of re-encoding
  audio every time.
- The logs for the generated adapters report `Loaded 32 preprocessed samples`.
- With batch size 2 and gradient accumulation 4, Side-Step reported about 4
  optimizer steps per epoch.
- 150 epochs therefore meant about 600 optimizer steps, not tens of thousands.
- Several runs early-stopped before 150 epochs:
  - LoKR early-stopped at epoch 104.
  - LoRA MLP early-stopped at epoch 83.
  - LoRA PP++ MLP early-stopped at epoch 91.
  - LoRA PP++ ran all 150 epochs.

By comparison, a PiMP remote package such as
`darkvocals_Vocals_2026-06-12-135117_remote_lokr_wide.zip` contained 160 clips.
At batch size 1 and gradient accumulation 2, that is about 80 optimizer steps
per epoch. A 30 epoch run is therefore about 2400 optimizer steps, which is
more total optimizer work than the 150 epoch / 32-sample Side-Step run.

## Chunk Duration Meaning

Side-Step's current dataset loader documents the important behavior:

- Preprocessed tensors store full target latents, conditioning states, context
  latents, and masks.
- When `chunk_duration` is set, each `__getitem__` call extracts a random
  window of that length from the T-aligned tensors.
- Different offsets are chosen across calls/epochs, acting as augmentation and
  reducing VRAM.
- The local Side-Step checkout also contains a coverage-weighted chunk sampler
  controlled by `chunk_decay_every`.

So `--chunk-duration 60` does not mean "train on one fixed 60 second clip." It
means each training sample contributes a 60 second latent window from the
preprocessed tensor. If the preprocessed tensor is long enough, later epochs can
see different sections.

One caveat from the inspected scripts: `train-dark-gentlemen-lokr.sh` used
`MAX_DURATION=240` for preprocessing. Unless the tensor cache came from a
full-length preprocess script, each long source may have been capped to four
minutes before chunk sampling. The separate `train-dark-gentlemen-pimp.sh`
script uses `--max-duration 0` and is the full-length strategy.

## Current PiMP Remote Worker

The current portable remote worker is valuable and should not be discarded.
It already:

- Uploads a reviewed PiMP package zip and portable worker folder to a stock pod.
- Converts selected PiMP training bundles into Side-Step-style dataset JSON.
- Runs preprocessing and training through the ACE-Step/Side-Step training source.
- Writes `remote_artifacts/**` logs, metrics, checkpoints, final adapter, and
  result files.
- Keeps cloud Python/CUDA outside the macOS app.

The gap is that it is still mostly a PiMP-owned training wrapper. It does not
yet expose Side-Step as the remote training product with all of its higher-level
CLI features, tensor-cache reuse, PP++ analysis flow, recipe sweep behavior, and
long-source workflow.

## Proposed Direction

Treat Side-Step as the first-class remote backend.

PiMP Audio Studio should orchestrate the job and own the user experience:

1. Build or select a reviewed dataset.
2. Choose either clip-package training or long-source Side-Step training.
3. Export a remote Side-Step job package.
4. Upload the job and worker to RunPod/DigitalOcean.
5. Install or clone a pinned Side-Step revision on the remote machine.
6. Run `sidestep preprocess` once.
7. Optionally run `sidestep analyze` for PP++/Fisher maps.
8. Run one or more training recipes against the same tensor cache.
9. Collect `best/`, `final/`, checkpoints, logs, configs, Side-Step commit hash,
   and run metadata.
10. Import the resulting Side-Step adapter through the existing PiMP importer.

## Product Shape

Add a new remote mode rather than replacing the current one immediately:

- **Remote Clip Package**: current PiMP-reviewed clip/stem package flow.
- **Remote Side-Step Source Job**: long sources plus sidecars, preprocess once,
  train one or more adapters from the tensor cache.

The Side-Step source job is the better fit for serious cloud adapter production,
especially on H100/H200 class GPUs.

## Recipe Sweep Idea

The app can offer one simple "run a sweep" action that expands to multiple
Side-Step recipes on the server:

- LoKR DoRA, XL Base, q/k/v/o, weight decomposition.
- LoRA MLP, XL Base, q/k/v/o plus MLP.
- LoRA PP++, XL Base, adaptive rank map.
- LoRA PP++ Full MLP, XL Base, adaptive rank map plus broader MLP coverage.

The important optimization is shared preprocessing:

```text
sources + sidecars
        |
        v
sidestep preprocess  ->  tensor cache
        |
        +--> sidestep analyze / PP++
        |
        +--> train recipe A
        +--> train recipe B
        +--> train recipe C
        +--> train recipe D
```

This matches what worked manually: pay the setup/preprocess cost once, then
iterate quickly across adapter recipes.

## Source Annotation And Transcriber Workflow

The Desktop folder
`/Users/tweaver/Desktop/sidestep-lyrics-backup/` contains another important
piece of the successful manual workflow:

- `update_lyrics.py`
- `sources/TheDarkGentlemen/*.txt`

The script runs the ACE-Step transcriber checkpoint locally/remotely and writes
Side-Step-compatible sidecars next to long source audio files.

### Script Fingerprint

`update_lyrics.py` is a local lyrics transcription pipeline for the
ACE-Step transcriber checkpoint:

- Audio directory: `./sources/TheDarkGentlemen`
- Model directory: `./checkpoints/acestep-transcriber`
- Model class: `Qwen2_5OmniForConditionalGeneration`
- Processor class: `Qwen2_5OmniProcessor`
- Prompt: transcribe audio in detail with section tags such as `[Verse]`,
  `[Chorus]`, and `[Intro]`.
- Chunking: long audio is split with ffmpeg into mono 16 kHz chunks.
- Default chunk size: 900 seconds with 10 seconds overlap.
- Generation settings: `max_new_tokens=2048`, `temperature=0.2`,
  `top_p=0.95`, `repetition_penalty=1.05`.
- Uses the Qwen2.5-Omni default system prompt because custom system prompts can
  degrade audio understanding.
- Calls `model.disable_talker()` because only audio-to-text transcription is
  needed.
- Uses `qwen_omni_utils.process_mm_info(...)` plus the processor chat template.
- Extracts the `# Lyrics` block from model output.
- Sanitizes section headers through Side-Step's `sanitize_headers(...)`.
- Merges fields through Side-Step's `read_sidecar`, `merge_fields`, and
  `write_sidecar` helpers using `overwrite_lyrics` policy.
- Skips already-transcribed files with substantial lyrics so the workflow can
  resume after crashes.
- Writes `[Instrumental]` and `is_instrumental=true` when very little vocal
  content is detected.
- Cleans local temp chunk directories so long runs do not fill small container
  root filesystems.

This should be treated as a remote/source-prep capability, not an in-app Python
feature. The Swift app can package sources and sidecars, launch the remote
worker, and import resulting sidecars/artifacts, but the transcriber runtime
belongs on the remote Python/CUDA side.

### Sidecar Corpus Observations

The backup contains 32 `.txt` sidecars under
`sources/TheDarkGentlemen/`, matching the 32 long source items used in the
manual Side-Step runs.

The files are not plain lyric-only text. They are structured sidecars with
fields such as:

- `caption`
- `genre`
- `bpm`
- `key`
- `signature`
- `is_instrumental`
- `custom_tag`
- optional `language`
- `lyrics`

The generated lyric blocks are long and sectioned. Across the 32 sidecars, the
observed bodies range from roughly 5.5k to 46.6k characters, with an average of
about 15.4k characters. Section tags include common song structure labels such
as `[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Instrumental Break]`,
`[Guitar Solo]`, `[Saxophone Solo]`, and `[Outro]`.

Potential cleanup/review needs:

- Some captions describe instrumental/blues-instrumental material while the
  sidecar has `is_instrumental: false` because lyrics were produced. Treat the
  generated instrumental flag as reviewable metadata, not unquestionable truth.
- Chunk overlap may produce repeated lyric or section fragments on very long
  tracks. A future cleanup pass could de-duplicate near-identical adjacent
  sections before training.
- The sidecars are rich enough to be first-class source metadata for a remote
  Side-Step source job. PiMP should avoid throwing this data away by forcing the
  user back through clip-only package creation.

### Metadata Fill Strategy

The successful manual sidecars were produced by a fallback chain rather than one
perfect annotation model:

1. **Gemini first pass**: used for early caption/BPM/key/style metadata. This
   produced useful results but was slow for very large source files because each
   file had to be uploaded and analyzed remotely.
2. **ACE-Step captioner scale pass**: used after the Gemini path became too
   slow. The ACE-Step captioner is also Qwen2.5-Omni based, but tuned for music
   caption/metadata work instead of lyric transcription.
3. **Music Flamingo gap fill**: used locally on the server to fill missing
   fields that the captioner did not reliably produce, especially missing BPM,
   key, or related musical metadata.
4. **Merge-only-missing policy**: later passes filled blanks instead of
   overwriting stronger existing fields. This matters because the best metadata
   for a file may come from different tools.

That fallback pattern is worth preserving. A future PiMP remote source-prep job
should treat annotation as a field-level enrichment pipeline, not a single
"generate all metadata" call.

Recommended source metadata merge behavior:

- Keep original user-edited fields unless the user explicitly requests overwrite.
- Fill missing `caption`, `genre`, `bpm`, `key`, `signature`, `language`,
  `is_instrumental`, and `lyrics` independently.
- Record per-field provenance: provider, model/checkpoint, timestamp, prompt or
  preset name, and whether the value was imported, generated, or manually
  reviewed.
- Prefer deterministic local/remote GPU tools for bulk annotation when source
  files are very large.
- Use cloud LLM/audio services only when their extra quality is worth the
  upload time and cost.

### Product Implication

Remote Side-Step jobs should eventually support three prep modes:

1. **Use existing sidecars**: package long sources plus `.txt` sidecars exactly
   like the manual workflow.
2. **Fill missing sidecar fields**: keep existing sidecars, run captioner,
   transcriber, and/or Music Flamingo only for missing fields, then write merged
   sidecars with provenance.
3. **Generate sidecars remotely**: run ACE-Step captioner/transcriber as a
   source-prep step on the GPU server, then feed the resulting sidecars into
   Side-Step preprocessing/training.

This makes the remote worker more than a trainer. It becomes a full source
annotation plus adapter-production pipeline, while PiMP Audio Studio remains the
review/orchestration UI.

## UX Requirements

The public user flow should stay as simple as possible:

- The app should explain what will be uploaded before upload.
- The app should never launch billable cloud hardware without explicit
  confirmation.
- API keys stay in Keychain and never go into job zips.
- The app should show the exact remote commands for manual workflows.
- The app should capture enough logs to debug provider, install, preprocess,
  train, and import failures.
- Users should not need to know Hugging Face paths, git clone commands, Python
  package names, or tensor-cache layout for the default path.

Advanced users can still override Side-Step branch/commit, recipe flags, base
model, max preprocess duration, and output paths.

## Architecture Boundaries

### In The App

- Swift UI for job setup, provider launch, status, logs, and import.
- Dataset/package review and sidecar validation.
- Job manifest generation.
- Provider credentials in Keychain.
- Adapter import/remap/registry handling.
- No Python runtime.

### In The Remote Worker

- Python, PyTorch, CUDA, Side-Step, ACE-Step, PEFT, LyCORIS.
- Source package extraction.
- Side-Step install/update pinned to a known commit or release.
- Model download/cache.
- Preprocess, analyze, train, checkpoint, and artifact collection.
- Logs and metrics in the existing `remote_artifacts/**` contract.

## Open Questions

- Which Side-Step commit should be pinned for the first supported backend?
- Should PiMP maintain a small compatibility wrapper around Side-Step CLI output,
  or call Side-Step as a pure subprocess and parse artifacts afterward?
- Should long-source jobs package raw audio into the zip, upload a folder via
  `rsync`, or support both?
- How should max preprocess duration be presented: 240 seconds, 600 seconds,
  full source, or "auto based on GPU"?
- Do we keep the current PiMP clip-package worker as a separate safe baseline?
- Should remote sweep runs return all recipes in one archive, or import only the
  selected winners?
- How much of the Side-Step local MPS tuning should be documented for Mac smoke
  tests?

## Risks

- Side-Step is Python and must stay outside the App Store app.
- Side-Step updates could change CLI flags or artifact layout, so pinning is
  mandatory.
- We should avoid copying Side-Step code into PiMP unless its license permits
  it and the copied surface is genuinely needed.
- Long-source uploads can be large; users need clear cost/time/storage warnings.
- Provider cleanup must be obvious so pods do not continue billing after tests.
- Best loss is not the final quality metric; audition snapshots and A/B imports
  remain important.

## Initial Implementation Milestones

1. Document and preserve the manual Side-Step command recipes that worked.
2. Add a new remote job type for Side-Step source jobs.
3. Package source audio plus sidecars without forcing clip expansion.
4. Add source-sidecar validation for caption, genre, BPM, key, time signature,
   instrumental flag, structured lyrics, and per-field provenance.
5. Teach the remote worker to install/use a pinned Side-Step checkout directly.
6. Add a dry-run command that prints the exact Side-Step preprocess/analyze/train
   commands before starting.
7. Add optional remote ACE-Step captioner/transcriber and Music Flamingo
   source-prep steps with fill-missing merge policy.
8. Add tensor-cache reuse and multi-recipe sweeps.
9. Add artifact pullback/import for Side-Step `best/`, `final/`, checkpoints,
   configs, logs, and commit metadata.
10. Add UI status for preprocess, analyze, per-recipe training, checkpoint
   availability, and final import.
