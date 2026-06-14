# Changelog

All notable changes to PiMP Audio Studio are documented here.

---

## [Unreleased]

### Added

- **Remote GPU now has a RunPod control-plane starter** — the Remote GPU workspace can save a RunPod API token in Keychain, link new users to RunPod account setup, list current pods, refresh live GPU type details with VRAM, live stock status, available GPU counts, and pricing, launch a pod from a template or image only after an explicit billable-resource confirmation, terminate pods, and copy SSH/upload handoff commands for the most recently exported remote job zip.
- **Remote GPU now has a portable PiMP worker** — the new `remote-worker/` package can be copied onto a stock RunPod PyTorch pod, installed once, and run as `pimp-remote-worker train <job.zip>` to convert PiMP training bundles into Side-Step dataset JSON, preprocess tensors, run LoRA/LoKR training, and write standard `remote_artifacts/**` logs, metrics, checkpoints, final adapter, and result files.
- **LyCORIS/PEFT LoKR checkpoints can import directly** — local and Hugging Face adapter import now accepts compatible `lokr_weights.safetensors` / LoKR `adapter_model.safetensors` files, converts `lokr_w1`/`lokr_w2` plus simple factored pairs and DoRA magnitudes into PiMP-native LoKR keys, preserves alpha/dim scaling from metadata when present, and clearly rejects unsupported Tucker LoKR variants.
- **Training completion can render adapter audition snapshots** — completed LoRA/LoKR adapters now offer a fixed-seed snapshot run that renders Base plus selected adapter gains, saves each take to History with listening-note tags, and opens Trace Compare for quick A/B listening.

### Changed

- **Remote GPU launch now recommends an official RunPod template** — saving a RunPod API key or refreshing launch options now loads official pod templates alongside live GPU availability, and selecting a GPU auto-fills the recommended official PyTorch pod template when the user has not chosen a custom template or image. Template ID/Image remain available under Advanced for manual overrides.
- **RunPod remote GPU actions are now observable and MCP-controllable** — RunPod API-key, GPU/template refresh, launch, and terminate events now emit sanitized `AUDIODIAG` breadcrumbs when verbose diagnostics are enabled, and MCP can refresh/status/preflight/launch/terminate RunPod pods with dry-run defaults and explicit confirmation before billable actions.
- **Dataset evenly-spread sampling now avoids source edges** — local dataset builds now sample the center of each spread bin instead of the exact first/last valid positions, and new packages record each generated clip's source file, start seconds, start frame, frame count, and sample rate in `captions.json`, `dataset_manifest.json`, and training bundles for easier audit/debugging.
- **Dataset annotation roadmap was refreshed** — the roadmap now reflects shipped dataset review, sidecar import/export, LoRA/LoKR training, remote job export, trace compare, and the PiMP Audio Studio rebrand.
- **RunPod launches now use Secure Cloud only** — the Remote GPU setup no longer asks users to choose between Secure and Community capacity; PiMP Audio Studio launches the standard Secure RunPod path and leaves custom provider-pool choices to the RunPod console.
- **Local LoRA/LoKR training now uses no hidden AdamW decay and clips gradients** — adapter trainers explicitly set optimizer weight decay to zero and apply conservative global norm clipping before each optimizer step.
- **Local LoRA/LoKR training now follows the corrected ACE-Step timestep/CFG recipe** — native Swift trainers sample the same logit-normal max timestep schedule used by fixed ACE-Step training and apply 15% CFG dropout with the learned null condition embedding while keeping random draws outside the MLX gradient closure.
- **Local LoRA/LoKR training is now seeded and validation-aware** — training setup exposes a deterministic seed, trainers use seeded shuffling/noise/timestep/CFG/dropout streams, and epoch-end validation loss uses a fixed held-out split with dropout disabled for best-checkpoint selection.
- **Training pause/resume now saves real optimizer state** — paused LoRA/LoKR runs write Adam optimizer tensors to `optimizer_state.safetensors`, save the current in-progress adapter weights, and restore optimizer moments on resume instead of writing a placeholder state file.
- **Training diagnostics and MCP payloads now include validation loss** — local training progress, run logs, loss charts, checkpoints, and MCP training inspection surfaces include validation history so epoch comparisons have a steadier signal than noisy train loss alone.
- **Adapter API docs now state real behavior and format compatibility** — `batchSize` is documented as remote-only (local trainers run microbatch-1 with gradient accumulation), native LoKR is labeled as a PiMP-native format not interchangeable with PEFT/Kohya/LyCORIS LoKR checkpoints, the conservative MLP LoRA default (gate/down only, no `upProj`) is documented against upstream's q/k/v/o presets, and APG's axis-1 projection is annotated as verified upstream parity (`apg_forward(..., dims=[1])`).

### Fixed

- **Remote GPU pod handoff now survives restarts** — the Remote GPU screen remembers the last exported job zip, offers a `Choose Job Zip...` recovery path, and clearly shows which local archive powers the upload/extract pod commands so those buttons do not appear mysteriously disabled after relaunch.
- **Remote worker now runs inside its managed venv** — `pimp-remote-worker` re-enters the installed worker Python automatically when launched from `/usr/local/bin`, preserves the venv `python` symlink with `abspath` instead of resolving to system Python, and pins `PIMP_REMOTE_PYTHON` so model-download helpers see dependencies such as `huggingface_hub`.
- **Remote worker preprocessing now matches stock RunPod Torch images** — the worker pins Diffusers to the Torch 2.4-compatible `0.35.2` release and promotes ACE-Step variant `silence_latent.pt` files into the root checkpoint folder expected by Side-Step preprocessing.
- **Remote worker stem-only datasets now preprocess safely** — selected bundle audio is staged under unique filenames before Side-Step preprocessing so vocal/instrumental stem packages with repeated names like `vocals.wav` no longer collide on `vocals.tmp.pt` or collapse metadata to one sample.
- **Remote worker training now satisfies Side-Step path safety** — generated commands run from a common safe root such as `/workspace`, so RunPod job tensors, model checkpoints, and output folders remain inside ACE-Step's `safe_path` boundary; the worker also treats printed `[FAIL]` / `TRAINING FAILED` messages as failed commands even when upstream exits with status 0.
- **Remote worker avoids poisoned bf16 tensor caches** — portable jobs now resolve `bf16_or_fp16` to fp16 preprocessing plus bf16 training by default, patch ACE-Step's fixed trainer so that precision reaches the actual training dtype/Fabric mode, skip Lightning's unsupported Fabric gradient clipping path under explicit fp16 AMP, and validate preprocessed tensors for NaN/Inf before reuse so badly poisoned caches are rebuilt while tiny encoder-state specks are repaired.
- **Remote GPU handoff is clearer on stock RunPod images** — generated remote job scripts can extract archives with Python's built-in `zipfile` fallback when `unzip` is missing, the RunPod pod row now copies a server-side extract/run command, SSH/upload commands are copy-pasteable in standard option order, and the UI/README call out that official RunPod PyTorch templates do not include the PiMP remote worker yet.
- **Remote GPU pod cleanup now terminates instead of pausing** — the destructive RunPod pod action now calls RunPod's DELETE `/pods/{podId}` endpoint, removes the pod from the local list after submission, and warns that non-network-volume data can be lost, instead of leaving an exited pod billing for storage via the pause/stop endpoint.
- **RunPod API failures are easier to diagnose** — invalid or expired API keys now show credential-specific guidance, network failures are called out separately, and RunPod server/rate-limit responses no longer appear as generic HTTP blobs.
- **MCP trace payloads now keep adapter strength numeric** — `get_trace` now serializes trace adapters with the same explicit payload builder used elsewhere, so adapter `weight` reports as `1.0`, `1.6`, etc. instead of bridging to a boolean in structured JSON.
- **Generation restart now waits for the previous run to unwind** — starting a new generation cancels the old task, waits for it to finish, and ignores stale cleanup from superseded runs so cancelled diffusion cannot clobber the new trace state, reload the LLM mid-run, or clear adapters under an active generation.
- **WAV export now correctly interleaves planar stereo buffers** — `[channels, frames]` audio is transposed to `[frames, channels]` before writing PCM, with a regression test using distinct left/right samples.
- **Corrupt history indexes are preserved before reset** — if `history.json` cannot be decoded, the unreadable file is moved to `history.json.corrupt` before the in-memory history falls back to an empty list.
- **Cover generation is now seed-reproducible** — the cover timbre path's VAE posterior draw now derives an explicit key from the generation seed (offset to stay decorrelated from the diffusion noise stream), so identical cover requests with the same seed produce identical audio for trace compare and A/B auditioning.
- **Cover sources are now trimmed before encoding** — cover source audio is sliced to the requested duration before the VAE encode and tokenize/detokenize chain, and the timbre reference is sliced to its 750-frame window before its posterior encode, removing a second full-length VAE pass and the memory spike on long source files.
- **Short cover sources now pad with the learned silence latent** — cover context and timbre latents use `silence_latent.pt` for missing source tails instead of raw zero latents, with tokenizer alignment padding repeating the last source frame so short covers stay in-distribution.
- **Cover strength wording is clearer** — the Generate UI, analyzer, and MCP schema now describe cover strength as a rework amount: lower stays closer to the source recording, higher gives the model more creative freedom.
- **Stacked adapters now keep their own alpha scaling** — LoRA/LoKR stacks now apply each adapter's saved alpha/rank or alpha/linear-dim scale instead of reusing the primary adapter's scale for every contribution.
- **LoKR DoRA training now matches PEFT norm handling more closely** — DoRA magnitude scaling now stops gradients through the directional norm denominator while leaving inference output unchanged.
- **LoKR Train from Files is available again** — the Training toolbar button now opens a LoKR training session when LoKR mode is selected instead of only enabling for LoRA.
- **Reference-audio generation no longer breaks at durations other than 30 s** — context latents are now built from the silence latent at the requested frame count (matching upstream, where reference audio feeds only the timbre encoder), instead of the fixed 750-frame reference latents that mismatched the diffusion sequence length in the DiT channel concat.
- **APG guidance momentum now matches upstream ACE-Step** — the momentum buffer uses the upstream recurrence `running = diff − 0.75·running` instead of an unbounded cumulative sum, with a regression test against the upstream formula.
- **Training window selection is now seeded** — preprocessing derives each clip's latent window from the run's training seed plus the clip identity (FNV-1a → SplitMix64) instead of `Int.random`, so identical seeds train on identical windows; the chosen window is logged in training diagnostics. Note: previously cached preprocess samples keep the window they were created with.
- **`LoRAInjector.applyStack` now filters contributions by layer path** — each `LoRALinear` receives only the contributions whose `path` matches it, instead of every layer receiving the full stack of wrong-shaped/wrong-layer tensors.
- **Cross-kind adapter injection can no longer nest wrappers** — both injectors refuse to wrap a layer already wrapped by the other adapter kind, and `DiTActor.injectLoRA`/`injectLoKR` remove an active opposite-kind adapter before injecting, with regression tests for both directions.
- **LoKR DoRA training with dropout is now mathematically consistent** — the DoRA training branch uses the PEFT-parity formulation where the magnitude-normalized correction is computed entirely on the dropout-masked input; with dropout 0 the output is unchanged and now verified equal to the inference path by a regression test.

---

## [0.8.0] — 2026-06-10 — Generation, Training, MCP, and Remote Workflows

### Added

- **Runtime identity now uses PiMP Audio Studio** — app display strings, bundle/runtime constants, Application Support paths, generated filenames, OSLog subsystems, UserDefaults keys, Keychain service names, and MCP-facing resources now use the new PiMP Audio Studio identity while reading legacy Anvil values during migration.
- **Legacy data directory migration is hardened** — first launch under the new bundle ID now migrates legacy models/adapters, presets, LLM caches, source imports, training datasets, and MCP imports into `Application Support/PiMP Audio Studio`, writes a `runtime_migration.json` report, resolves the real user home from inside the sandbox, and keeps tests from mutating real user data.
- **Source symbols and dataset schema now use PiMP Audio Studio naming** — active tokenizer symbols, source comments, dataset package JSON keys, preprocess cache metadata, remote job descriptors, and environment toggles now write `PiMPAudio` / `pimp_audio_*` names while keeping targeted legacy readers for old local packages.
- **Generation Analyzer live window** — a new analyzer scene can be opened before generation to watch the request move through routing, model selection, adapter setup, conditioning, diffusion, decode, WAV write, waveform analysis, and trace writing. Events are shown reverse-chronologically with stage labels, timestamps, progress, ETA, request metadata, model details, LoRA stack, and audio statistics.
- **Per-generation trace sidecars** — each generated WAV now gets a sibling `.trace.json` sidecar containing the resolved request, prompt, lyrics, pipeline, primary/supporting models, active adapters, timing, feature flags, waveform peaks, per-second envelope stats, and audio summary. History entries now store the trace filename plus model/adapter snapshots, so old generations can be inspected later without rerunning them.
- **Saved trace viewer from History** — history rows with trace sidecars show an analyzer shortcut and context-menu action. Opening a saved trace reconstructs the analyzer session with the original request, model stack, prompt, lyrics, audio stats, and trace metadata.
- **Two-run trace compare mode** — selecting exactly two traced history rows now offers `Compare`, opening a side-by-side analyzer view with run summaries, changed settings highlighted, prompt/lyrics columns, and audio summary columns. This makes guidance/seed/model/adapter A/B listening less guessy.
- **A/B playback in trace compare** — compare mode now has `Play A` / `Play B` controls wired through the shared audio player and uses waveform peaks stored in the trace sidecar for fast switching.
- **Edit handoff from trace compare** — compare mode now exposes `Edit A` / `Edit B`; choosing a run pauses compare playback, switches the main app to the editor, loads that WAV, and passes the trace prompt as editor context.
- **Trace compare listening notes and preferred takes** — compare mode now has per-run note fields plus `Mark Preferred`. Notes and preferred flags are saved back into the `.trace.json` sidecars and mirrored into `history.json`, so listening decisions survive relaunches and show up in History.
- **Trace audio preview strips** — saved traces and compare columns now render a compact waveform with a per-second loudness overlay, giving quick visual signal context before A/B playback.
- **Failure trace sidecars** — failed generations now preserve the live event log into a `.failed.trace.json` artifact when possible, giving failed runs enough request/model/stage context for later debugging.
- **Verbose audio diagnostics now cover editor and LoRA-heavy generation paths** — the Advanced `Verbose audio diagnostics` checkbox now emits `[AUDIODIAG]` request routing, memory snapshots, cover source-conditioning tensors, inpaint/repaint masks and latents, CFG cond/uncond deltas, and LoRA stack coverage for diagnosing adapter/guidance quality issues.
- **Verbose audio diagnostics now mirror to timestamped working-folder logs** — enabling the Advanced diagnostics checkbox still writes `[AUDIODIAG]` lines to Xcode, and now also creates `PiMPAudioStudio-AUDIODIAG-*.log` files under the app's `Diagnostics` working folder so multiple runs can be compared after generation.
- **Training diagnostics now capture high-value LoRA/LoKR breadcrumbs** — verbose diagnostics now log training model-load phases, dataset/preprocess cache hits and sample shapes, DiT injection/trainable path coverage, trainer first-sample/epoch health, save/checkpoint paths, cancellation, cleanup, and memory snapshots so major training bugs show up in the same working-folder logs as generation.
- **Adapter effect diagnostics now report audible scale** — verbose diagnostics now emit one `adapter/effect` line per active LoRA/LoKR adapter per generation, comparing the adapter delta against the base layer output so A/B tests can confirm whether an adapter is active, subtle, audible, or too tiny to matter.
- **Adapter generation gain can now boost weak adapters** — Generate and MCP adapter selection now allow runtime adapter gain up to `6.0x` while keeping `1.0x` as the normal default, making under-scaled LoRA/LoKR adapters testable without editing their files.
- **Training gradient diagnostics now isolate first-step stalls** — LoRA/LoKR training logs first `valueAndGrad` plus loss-materialization timings around the explicit adapter-parameter gradient path, so XL adapter stalls can be separated from memory or dataset issues without routing gradients through the full model parameter tree.
- **MCP can now inspect training state and logs** — the local MCP endpoint now exposes read-only `training_status`, `list_training_datasets`, `list_training_runs`, and `tail_training_log` tools plus `pimpaudio://training*` resources so agents can inspect LoRA/LoKR progress, dataset packages, adapter run logs, and recent AUDIODIAG training lines without starting or mutating training.
- **MCP training inspection now has detail views** — local agents can call read-only `inspect_training_dataset`, `inspect_training_run`, and `compare_training_runs` for dataset annotation coverage, package validation, size-aware recommendations, adapter config, run health metrics, and compact recent-run comparisons.
- **MCP can now safely preflight and control local training** — agents can call `training_preflight` to resolve reviewed datasets, recipes, adapter dimensions, projection coverage, and the local memory/config guard without starting a run. `start_training` defaults to dry-run and requires `dryRun: false` plus confirmation for actual LoRA/LoKR starts, while `stop_training` uses the same guarded TrainingService stop/checkpoint path as the UI.
- **MCP session cleanup is more resilient** — the local MCP bridge now resets stale SDK session transports before fresh initializes, treats a matching DELETE as an explicit local reset even after an abandoned stream, and the smoke test verifies DELETE followed by a fresh initialize.
- **Local training preflight now measures XL adapter risk** — before the first backward pass, LoRA/LoKR training clears MLX cache, logs active/cache/peak/limit/system-available memory, estimates the required reserve from layer/projection/rank/sample scope, and surfaces broad local Q/K/V/O shapes as large-shape warnings while memory headroom remains a hard guard.
- **Local LoRA preflight now calls out broad Q/K/V/O XL shapes** — local XL LoRA runs with full Q/K/V/O projection coverage across 12+ layers show a Training sheet and MCP preflight warning that Q/V or Last 8 is the safer local starting point.
- **Local adapter first-backward diagnostics now catch MLX stalls** — LoRA/LoKR trainers keep the previously stable explicit adapter-gradient route and log if MLX stays inside the initial `valueAndGrad` call for 30/60/120/300 seconds.
- **Local adapter dropout now uses explicit MLX keys** — local LoRA/LoKR execution keeps requested adapter dropout enabled by precomputing per-sample MLX RNG keys outside `valueAndGrad`, avoiding the confirmed implicit-RNG first-backward VJP stall without falling back to dropout 0.
- **Adapter dropout VJP testing has a Swift probe** — the new `MLXDropoutVJPProbe` SwiftPM executable compares implicit RNG, explicit-key, and precomputed-mask dropout graphs outside the full app trainer.
- **Training completion diagnostics now only mark saved adapters complete** — user-stopped LoRA/LoKR runs no longer emit an ambiguous `training/complete` breadcrumb before their `training/cancelled` or checkpoint status.
- **Training progress now records adapter health metrics** — local LoRA/LoKR runs now carry moving-average loss plus gradient, parameter, and update norms through the progress UI, verbose diagnostics, and `training_log.json` so stalled or unstable adapter training is easier to spot.
- **Training update-norm diagnostics now capture real movement** — LoRA/LoKR trainers materialize trainable adapter snapshots before optimizer updates so `updateNorm` reflects parameter changes instead of reporting zero for active runs.
- **Completed adapters can now jump straight into Generate for auditioning** — LoRA/LoKR training completion now offers Finder, adapter selection, and a deterministic `Test in Generate` handoff that selects the new adapter at strength 1.0 with seed 42 and a training caption prompt for quicker base-vs-adapter A/B checks.
- **Paused LoKR runs can now resume from checkpoints** — LoKR training now follows the same saved-weight/checkpoint resume path as LoRA, and paused LoKR adapters show the Resume Training action in Adapter settings.
- **Training setup now suggests dataset-size-aware starting points** — the LoRA/LoKR training sheet now reads imported clip durations, estimates optimizer update counts, and surfaces an applyable recommendation for rank/linear dim, epochs, layer coverage, and gradient accumulation based on both clip count and usable audio duration.
- **Training setup now uses public-friendly recipe names** — LoRA training now offers Narrow and Wide recipes, with Wide using the stronger all-32 Q/K/V/O footprint, 2x alpha scaling, dropout, and longer epoch/gradient-accumulation defaults. The secondary wide LoRA recipe has been removed, and LoKR uses the same Narrow/Wide naming.
- **LoKR training now supports DoRA weight decomposition** — native Swift/MLX LoKR can train and load ACE-Step-style `weightDecompose` adapters with saved magnitude vectors, a LoKR-only Advanced toggle, MCP `weightDecompose`/`dora` controls, and diagnostics/preflight tensor counts that show when DoRA is active.
- **Oversized local LoKR shapes are now blocked before launch** — the trainer now estimates LoKR trainable tensor count, warns and disables full q/k/v/o LoKR across 16+ local layers or above local-stable linear dim 32, blocks shapes above the local 512-tensor cap, and trims giant trainable-parameter diagnostics to a readable preview.
- **Local audio files can now be built into LoRA training datasets** — the adapter settings toolbar now includes `From Local Files…`, which slices selected audio files into `.m4a` clips under `TrainingDatasets`, writes `captions.json`, lets captions be reviewed, and opens the resulting clips in the existing trainer.
- **Local dataset builds now support larger source collections** — From Local Files now allows up to 1,000 total clips and shows whether the selected clip count/cap will cover every source file or stop early.
- **Local dataset builds now write diagnostics breadcrumbs** — With verbose audio diagnostics enabled, From Local Files now mirrors build start/end, coverage/cap decisions, per-source progress, skipped-file errors, package writes, and periodic WAV clip progress into the working-folder AUDIODIAG log.
- **ACE-Step v1.5 XL model choices are first-class** — Models settings now includes the official XL Turbo, XL Base, and XL SFT DiT checkpoints, with clearer XL labels in Settings, the sidebar model switcher, training labels, adapter import inference, and adapter compatibility checks.
- **ACE-Step LM 1.7B is available in Models** — the ACE-Step LM section now includes the official nested 1.7B 5 Hz audio-code language model from `ACE-Step/Ace-Step1.5`, and cover-smoke LM loading tries 0.6B -> 1.7B -> 4B.
- **Local dataset clip sampling can cover long source files** — From Local Files now defaults to deterministic evenly-spread clip windows across each source file, with optional Sequential and Random Seeded modes plus manifest metadata for the selected sampling mode.
- **Dataset and training workflows now live in the main sidebar** — the main window now has a **Build** section between Create and Library with dedicated Datasets and Training views, while Settings > Adapters stays focused on adapter management and merge/import operations.
- **Dataset packages now have a review workspace** — the Datasets page lists local training packages, opens package folders for review, previews clip audio, edits captions/lyrics/BPM/key/language/time signature/notes, imports ACE-style sidecars, tracks manual approval, exports reviewed packages as remote-ready zips, and only enables "Open in Training" once blockers are cleared.
- **Dataset annotation provenance is now packaged** — local dataset packages now record caption/lyrics sources, annotation backend/model metadata, review status, raw annotation artifact paths, review-save backups, and an adapter-agnostic `preprocess_cache.json` contract that both LoRA and future LoKR training can share.
- **Dataset annotation metadata now uses a shared backend contract** — package creation and review saves now build annotation provenance through the same Swift-native backend descriptor/metadata path, reducing drift between generated and edited dataset manifests.
- **Imported LRC lyric sidecars are cleaned for training** — dataset sidecar import now reads `.lyrics.lrc` / `.lrc` files and strips timestamp or metadata tags while preserving the lyric lines that should feed training.
- **Dataset preprocessing now writes reusable cache artifacts** — training from PiMP Audio Studio dataset bundles now saves encoded latent/conditioning tensors as sample-level safetensors under `preprocessed/`, updates bundle `index.json` files, and refreshes `preprocess_cache.json` so later LoRA/LoKR runs can reuse the same prepared data.
- **Reviewed datasets can export ACE-compatible sidecars** — the dataset review workspace now writes `.caption.txt`, `.lyrics.txt`, per-track `.json`, and package-level `metadata.csv` files from the reviewed clip metadata.
- **Dataset key metadata is reflected in package status** — imported or manually reviewed key/keyscale values now update manifest analysis status as metadata-backed key information instead of leaving key status stuck at `not_run`.
- **Dataset enrichment now fills clip BPM and key metadata** — Enrich Dataset now promotes native stem tempo estimates into clip-level BPM fields and estimates instrumental key/keyscale with an in-app DSP pass, preserving imported or manually reviewed values when present.
- **LoKR workspace now trains native experimental adapters** — the Training workspace's experimental LoKR tab now opens reviewed packages in a local Swift/MLX LoKR trainer, stores LoKR metadata with adapters, and keeps LoRA/LoKR adapter stacks separated at generation time.
- **Adapter training runs now write durable logs** — local LoRA/LoKR training saves a `training_log.json` sidecar beside each adapter with the adapter kind, base model, config, progress samples, loss history, final status, trained paths/layers, tensor count, and errors when available. Adapter export includes this log, and training completion plus Adapter settings can reveal the adapter folder in Finder.
- **Training workspace now opens local LoRA setup inline** — the Training view now uses its main canvas for the Train from Files flow instead of presenting a centered modal, with a tab-ready mode switch for future LoKR and remote GPU training options.
- **Remote GPU training tab now prepares package handoff** — the Training workspace's Remote GPU mode now scans local dataset packages, opens package folders, blocks exports until review blockers are cleared, and writes the same remote-ready zip archives as the Datasets workspace.
- **Remote GPU exports now build job bundles** — the Training workspace's Remote GPU mode now uses a Create Remote Job flow with LoRA/LoKR, Narrow/Wide, bundle, model, epoch, save-cadence, seed, runtime, and cost options. Exports include `remote_job.json`, `README_REMOTE_GPU.md`, and `run_remote_training.sh` alongside the reviewed dataset package so workers no longer have to infer the recipe from the archive.
- **Remote worker jobs now have a provider-neutral contract** — `remote_training_package.json` now advertises standard `remote_job.json`, result, log, and metrics paths; the new remote job schema models LoRA, LoKR, preprocess, and annotation jobs without baking in RunPod or DigitalOcean details.
- **Local adapter import now accepts native cloud results** — the adapter import sheet now auto-detects native PiMP Audio Studio folders (`adapter.json` + `adapter.safetensors`) as well as PEFT LoRA folders, so converted cloud LoKR/DoRA checkpoints can be installed from local disk without PEFT filename shims.
- **Cloud adapter recipe planning is documented** — the successful stem-specific cloud LoRA jobs are fingerprinted in `docs/cloud-adapter-training-plan.md`, including recipe settings, worker-boundary rules, and the next provider-neutral remote job fields needed for parity.
- **Training options now expose larger local LoRA runs** — the training sheet now offers rank 64/96 plus 75/100 epoch presets, and its Base Model row shows the active ACE-Step DiT variant that will be used for the adapter.
- **Training advanced controls now affect the trainer loop** — LR Warmup continues to drive the cosine scheduler, and Grad. Accumulation now averages gradients across the selected window before optimizer updates instead of only changing progress and LR reporting cadence.
- **Local dataset enrichment now has native Swift/MLX stem separation** — `Enrich Dataset` now uses an in-app Demucs backend for vocals plus instrumental stems and records the selected backend/model in each package's separation metadata, keeping the dataset workflow free of Python-backed separator tooling.
- **Local dataset enrichment now writes native lyric transcripts** — the local enrichment pass transcribes generated `vocals.wav` stems with `speech-swift` Qwen3-ASR, writes `lyrics.txt`, `lyrics.timed.json`, and `lyrics.lrc` artifacts, and carries those transcripts into vocal training bundles while keeping instrumental bundles lyric-free.
- **Local dataset enrichment now auto-captions from metadata** — Enrich Dataset now upgrades local-builder/manual captions into deterministic `native` captions using the original style seed plus BPM, key, stem role, and vocal/transcript context, while preserving imported sidecar/CSV/remote captions for review.
- **Native speech tooling raises the minimum runtime** — the app now targets macOS 15.0+ and Xcode 16+ to match the Swift/MLX speech stack used for on-device Qwen3-ASR transcription.
- **Primary LoRA strength control** — the Generate view now exposes the selected LoRA adapter's strength, routes the primary adapter through the same weighted stack path as additional adapters, persists the value in saved presets, and records the actual weight in history/trace sidecars. This makes XL PEFT adapter A/B tests line up with the `lora_scale` knob used by the Python ACE-Step path.
- **Local MCP server foundation** — Settings now includes an MCP tab that can start/stop a localhost-only endpoint at `http://127.0.0.1:38765/mcp`, optionally require a bearer token, and expose read-only tools for models, adapters, history, trace lookup, and Analyzer-style trace comparison.
- **MCP resources and smoke testing** — the local MCP endpoint now supports `resources/list` and `resources/read` for `pimpaudio://models`, `pimpaudio://model/{id}`, `pimpaudio://adapters`, `pimpaudio://adapter/{id}`, `pimpaudio://history`, `pimpaudio://history/{id}`, `pimpaudio://trace/{id}`, and `pimpaudio://audio/{id}` metadata. The `scripts/smoke-pimpaudio-mcp.sh` script checks health, initialization, tool discovery, history, trace reads, resource reads, compare calls, and progress streaming against a running local app.
- **MCP prompt and lyric helpers** — the local MCP endpoint now exposes return-only `enhance_prompt` and `write_lyrics` tools backed by the same prompt enhancer, lyric writer, and lyric sanitizer used by the Generate view. These tools return text for agent workflows without mutating the current prompt field, lyric field, history, traces, models, or audio.
- **MCP preferred-take review writes** — the local MCP endpoint now exposes `mark_preferred_take`, scoped to the same preferred flag and listening note metadata used by the Generation Analyzer. It requires an existing trace sidecar and mirrors review changes into `history.json`; audio, models, adapters, tensors, and generation settings remain untouched.
- **MCP adapter selection** — the local MCP endpoint now exposes `apply_adapter` for selecting, stacking, or clearing ACE-Step LoRA adapters. The tool validates adapter existence/category, updates the generation service, persists the primary adapter selection, and notifies the Generate view when it is open so MCP and GUI state stay aligned.
- **MCP generation submission** — the local MCP endpoint now exposes `generate_song`, `generate`, and `generation_status`. Generation requests reuse the same `GenerationService` path as the Generate button, return immediately for polling via status/history/analyzer traces, and support a `dryRun` mode for validating resolved request settings without starting a render.
- **MCP operation tracking and cancellation** — MCP-submitted generation now
  returns an `operationID` and is visible through `list_operations`,
  `get_operation`, `generation_status.currentOperation`, `pimpaudio://operations`,
  and `pimpaudio://operation/{id}`. `cancel_operation` cancels the matching active
  generation through the same cancellation path as the app UI, giving richer
  editor workflows a standards-shaped operation contract before protocol-level
  progress streams are enabled.
- **MCP song orchestration now clears then applies adapters** — `generate_song`
  now treats `clearAdapter: true` plus `adapter` / `adapters` as a stale-selection
  reset before applying the requested adapter stack, so one-call A/B generation
  requests no longer accidentally submit base-model renders.
- **MCP editor operation bridge** — `editor_status` now exposes editor
  readiness, loaded audio metadata, supported editor actions, and the active
  editor operation. `load_editor_audio` registers a background
  `editor.loadAudio` operation, loads a history/audio id or readable local path
  into the shared editor coordinator, encodes editable latents, and reports
  through the same `list_operations` / `get_operation` / `cancel_operation`
  contract as generation.
- **MCP editor inpaint submission** — the local MCP endpoint now exposes
  `submit_editor_inpaint` for confirmed editor inpaint/repaint mutations. It
  supports dry-run validation, selected time ranges, optional frequency bands,
  strength/guidance/pipeline controls, seeded runs, `editor.inpaint` operation
  polling, cancellation, and `editor_status.lastEdit` metadata after completion.
- **MCP editor DSP chains** — the local MCP endpoint now exposes
  `apply_editor_effect_chain` for confirmed DSP mutations against loaded editor
  audio. It supports dry-run validation, ordered effect chains, selection-aware
  fade/trim/delete actions, EQ/reverb/time-stretch/pitch-shift parameters,
  editable-latent refresh, and `editor.applyEffectChain` polling/cancellation.
- **MCP editor render export** — the local MCP endpoint now exposes
  `render_editor_audio` for exporting the current editor buffer or selected time
  range to WAV/AIFF. It resolves a default `MCPExports` path, refuses accidental
  overwrites, returns `editor.renderAudio` operations, and reports output file
  metadata on completion.
- **MCP workstation resources and prompts** — the local MCP endpoint now exposes
  standards-shaped workstation resources for the current project, track, clip,
  region, takes, presets, and workflow prompt definitions. It also advertises
  SDK `prompts/list` / `prompts/get` workflow prompts for chorus improvement,
  mix variants, region cleanup, review exports, and production-ready generation
  prompts.
- **MCP resource templates and completions** — the local MCP endpoint now
  advertises parameterized resource templates and SDK `completion/complete`
  suggestions for workflow prompt arguments, take/history ids, current region
  times, presets, workflows, and recent operation ids.
- **MCP client roots policy** — the SDK bridge now tracks MCP client roots when
  the initialized client advertises them, refreshes on
  `notifications/roots/list_changed`, exposes `pimpaudio://client-roots`, and uses
  root paths as low-risk output-directory completion suggestions while keeping
  history/audio ids as the preferred editor-loading path.
- **MCP quit cleanup** — app termination now explicitly stops the local MCP
  server so the localhost listener is closed during normal macOS quit.
- **MCP Streamable HTTP sessions** — the local MCP bridge now uses the Swift
  SDK's `StatefulHTTPServerTransport`. `initialize` returns `MCP-Session-Id`,
  POST requests may return SSE, `GET /mcp` opens the standalone server stream,
  and `DELETE /mcp` terminates and resets the embedded SDK session. Tool calls
  with `_meta.progressToken` emit request-scoped `notifications/progress`.
- **MCP elicitation confirmations** — guarded mutations now use SDK
  `elicitation/create` form prompts when a client advertises form elicitation
  and omits `confirm: true`. The existing `confirm: true` path remains the
  compatibility fallback for clients without elicitation support.
- **MCP normalized operation progress** — pollable generation operation records
  now expose `progress` / `total` as whole-operation completion values, with
  `stageProgress` / `stageTotal` preserving stage-local counters when available.
- **MCP guarded memory release** — the local MCP endpoint now exposes `release_memory`, which runs the same consolidated model-unload and MLX GPU-cache flush path as Settings. The tool requires `confirm: true` or MCP form elicitation approval and returns before/after active, cache, and peak memory counters for agent-visible cleanup checks.
- **MCP standards roadmap** — added `docs/mcp-standards-roadmap.md` to lock in the standards-first direction: preserve the PiMP Audio Studio facade as the app boundary, prefer the official Swift MCP SDK for transport/lifecycle/schema handling where practical, stage Swift 6/concurrency migration instead of flipping the whole app at once, and plan richer workstation resources, prompts, progress, and cancellation support.
- **Serial PiMPAudioCore test baseline** — added `scripts/test-pimpaudiocore-serial.sh` to enumerate PiMPAudioCore test suites and run each suite filter in its own `swift test` invocation while the raw whole-suite runner still trips MLX under all-at-once scheduling.
- **Swift MCP SDK spike package** — added `Spikes/MCPServerSDKSpike`, an isolated Swift 6 package that resolves `modelcontextprotocol/swift-sdk`, exposes `list_history` plus `pimpaudio://history`, and verifies SDK client discovery plus stateless HTTP handling for the same JSON-RPC method shape used by the MCP smoke script.

### Changed

- **App versioning now reflects the current product maturity** — the macOS app, checked-in Xcode project, project generator config, PiMPAudioCore info constant, and MCP fallback version now report `0.8.0` / build `8` instead of the original foundation-era `0.1.0`.
- **MCP command facade is split by domain** — `PiMPAudioCommandFacade.swift` now holds construction and service wiring only, while resources, training, generation, editor actions, operation tracking, shared parsing, and local MCP payload types live in focused extension files.
- **MCP server internals are split into focused files** — the raw localhost HTTP listener now lives in `PiMPAudioMCPHTTPServer.swift` and the static tool schema catalog lives in `PiMPAudioMCPToolCatalog.swift`, leaving `PiMPAudioMCPServer.swift` focused on lifecycle, SDK bridge wiring, session handling, and dispatch.
- **MCP docs now treat PiMP Audio Studio names as primary** — `docs/mcp-local.md`, the MCP spec/plan, and the smoke test now document `pimpaudio://` resources, `PIMP_AUDIO_MCP_*` environment variables, and snake_case tool names as the public catalog.
- **MCP progress/cancellation design is staged** — the standards roadmap now
  separates today's pollable operation records and request-scoped progress
  notifications from the later background-operation task/progress mapping. MCP
  task support remains deferred until the Swift SDK exposes task capabilities.
- **Phase 0 MCP hardening status is documented** — the MCP standards roadmap now records the cleaned-up package-test baseline, the current strict-concurrency warning inventory, and the remaining raw `swift test --package-path AnvilCore` MLX runner limitation.
- **MCP SDK replacement direction is now concrete** — the standards plan records
  the completed move to the SDK's stateful Streamable HTTP transport while
  keeping the PiMP Audio Studio command facade and localhost listener boundary intact.
- **Local MCP dispatch is now SDK-backed** — `PiMPAudioMCPServer` keeps the existing
  localhost socket listener, token gate, and `PiMPAudioCommandFacade` boundary, but
  tool/resource registration, lifecycle handling, structured tool results,
  JSON-RPC dispatch, and stateful HTTP/SSE response shaping now run through the
  official Swift MCP SDK.
- **Training roadmap now makes native implementation non-negotiable** — ACE-Step, Side-Step, StemForge, DEMON, and similar projects are documented as behavior/format/UX references only. PiMP Audio Studio should not copy license-incompatible code or add Python, `uv`, CUDA, TensorRT, or browser-server dependencies; training, annotation, preprocessing, and adapter auditioning stay Swift/MLX, Core ML, or another App Store-friendly native/compiled path.
- **Cover-song preprocessing follows the upstream source-audio hint path** —
  the native Swift cover path tokenizes source latents to 5 Hz FSQ vectors,
  detokenizes them to 25 Hz acoustic hints, then blends them into the diffusion
  context. The experimental greedy LM next-code hop is disabled after listening
  tests produced tone-like output; ACE-Step LM assets remain available for
  future discrete audio-code workflows.
- **Cover-song generation now honors selected LoRA adapters** — the cover path
  preloads its ACE-Step DiT/VAE stack before diffusion, applies the same
  selected adapter or weighted adapter stack used by normal ACE-Step generation,
  and records the adapter plan, source audio file, and cover strength in the
  Generation Analyzer trace/compare views.

### Fixed

- **Community adapter downloads can verify SHA-256 checksums** — community manifest entries now support an optional `sha256` for `adapter.safetensors`, and community plus metadata-backed HuggingFace adapter installs verify downloaded weights when a checksum is supplied before keeping them on disk.
- **Community downloads now use a real queue** — queued adapter downloads no longer occupy the active-download concurrency slots or reset progress every retry tick; only running downloads live in the active task set, and queued work starts as slots open.
- **Lyric writer failures are surfaced** — API and CLI lyric writers now use throwing streams, report request encoding, HTTP, network, and provider-exit failures, and show the error beside the lyrics box instead of silently finishing with no text.
- **LoRA/LoKR resume preserves best-checkpoint state** — resumed trainers hydrate their saved best weights when a checkpoint already has a best loss, so pause/resume metadata continues reporting usable checkpoints and lowest loss before another epoch completes.
- **New local LoRA trainings now scale alpha with rank** — LoRA configs created with a larger selected rank now default alpha to that same rank instead of keeping the old fixed `16`, so rank 64 adapters are no longer trained with a conservative `0.25` internal scale unless alpha is explicitly overridden.
- **Training layer coverage controls no longer wrap awkwardly** — the LoRA/LoKR training sheet now hides duplicate segmented-picker labels and gives Layer Coverage a stable width, so the control stays as a single clean row in the inline Training workspace.
- **Local dataset enrichment uses less peak memory** — Enrich Dataset now invokes the existing GPU-memory release path before loading Demucs/Qwen, compacts the MLX allocator between separation/transcription phases, and reuses one decoded stem sample buffer for BPM/key analysis instead of reading instrumental stems twice.
- **Editor audio export keeps sandbox write permission alive** — exporting edited WAV/AIFF/M4A files now holds security-scoped access for the full async write/export operation, normalizes the output extension, and surfaces export failures in-app instead of only logging AVFoundation `-54` errors in Xcode.
- **Local file build flow no longer hands stale captions to training** — the local dataset sheet now refreshes its preview from the package after enrichment, updates the full-mix training bundle when captions are auto-generated, and sends users into Dataset Review instead of bypassing approval with an outdated in-memory clip list.
- **Blank local-file style descriptions no longer pollute auto captions** — local dataset packages now mark filename fallback captions as `filename`, and Enrich Dataset replaces those placeholders with native metadata captions instead of treating file names as manual prompt seed text.
- **Local dataset enrichment no longer trips Metal validation during Demucs separation** — the vendored Swift/MLX Demucs kernels now use source-stable custom Metal kernel names and synchronize the GPU stream before returning CPU stem buffers, avoiding an in-flight compute pipeline release when `Enrich Dataset` runs under Xcode Metal API validation.
- **Local dataset manifests now refresh enrichment status from stem artifacts** — package validation now derives separation, transcription, BPM, stem bundle, and enrichment metadata from enriched captions and stem sidecars, so `dataset_manifest.json` cannot remain stuck at `not_run` after a successful `Enrich Dataset` pass.
- **Local dataset enrichment now evicts native audio models after a run** — completing, cancelling, or failing `Enrich Dataset` now drops the Qwen3-ASR and forced-aligner references and flushes the MLX GPU cache so Demucs/transcription memory does not stay resident after package enrichment.
- **Random seed generations are now reproducible** — "Random" seeds are resolved to a concrete integer before dispatching to any backend, then recorded into the history entry and trace sidecar. ACE-Step SDE per-step noise also receives the resolved seed plus step index, so a liked random take can be regenerated or modified later instead of being lost as `0` / missing seed metadata.
- **Imported LoRA adapters refresh the UI without relaunch** — adapter save/import/delete paths now post an adapter-registry change notification. The picker, stack view, and settings list refresh when local or HuggingFace PEFT adapters are imported, so newly imported adapters appear immediately.
- **Local PEFT adapter imports are normalized into Anvil's registry** — importing a local `adapter_config.json` + `adapter_model.safetensors` directory now reuses the same PEFT config parsing, key remapping, base-model resolution, and registry-save path as HuggingFace adapter imports.
- **Trace compare defaults to the useful differences** — the compare table is now grouped into Model, Adapters, Sampling, Prompt/Lyrics, and Audio sections, hides unchanged rows by default, and includes a `Show unchanged` toggle for full audits.
- **Trace decoding is more tolerant of older sidecars** — trace JSON now has explicit schema versioning and defaults for optional review/event metadata, so older successful traces keep loading while new traces can carry review notes, preferred state, and event snapshots.
- **Stale AnvilCore fixtures are aligned with current contracts** — tests now match the capped tokenizer mask behavior, current ConditioningActor stubs, v1.5 VAE stride, DiT encoder hidden size/cache requirements, Stable Audio module paths and transposed-conv weight layouts, editor stereo audio loading, and LLM lazy-load failure messages.

---

## [Unreleased] — Runtime + Memory Optimization Pass

### Fixed

- **Release GPU Memory now clears ACE conditioning weights too** — `ConditioningActor` can now unload Qwen3, lyric/timbre encoders, the condition encoder, and `silence_latent`; both ACE text-to-music and cover-song coordinators call it during release so the button no longer leaves the conditioning stack resident.
- **Release GPU Memory is safer around active generation** — the release path now cancels any active generation task before tearing down weights, unloads ACE, cover-song, LoRA, SFX, and local LLM resources through one consolidated path, then flushes the MLX GPU allocator off the main actor.
- **Local LLM generation now uses a KV cache** — lyric writing, prompt enhancement, and intent routing no longer re-run the entire growing prompt for every generated token. The Llama path now pre-fills once, stores per-layer key/value tensors, and decodes one token at a time against the cached context.
- **Lyric Writer no longer passes model chatter into ACE-Step** — shared lyric prompts now forbid prose summaries, chord symbols, key/tempo/duration metadata, timestamps, performance directions, and line counts; a final sanitizer strips intro/footer commentary, single timestamp markers, timestamp ranges, bracketed fade directions, and chord prefixes before the generated lyrics are handed to music conditioning. Timestamp-only lyric output is wrapped as `[Verse]` instead of falling back to raw model text.
- **Lyric Writer now repairs oversized single-verse output** — long-song prompts require explicit `[Verse 1]`, `[Chorus]`, `[Verse 2]`, `[Bridge]`-style structure, and the final sanitizer splits one giant `[Verse]` block into duration-appropriate sections before generation.
- **Generated WAV export and editor waveform drawing use fewer transient copies** — final file writing now happens off the main actor and streams header/audio bytes through a temp file instead of building a second full-size WAV `Data`; editor waveform peaks are computed directly from Int16 PCM instead of allocating a full Float copy plus per-bucket slices.
- **Generated WAV files keep a valid RIFF header after streamed export** — the optimized temp-file writer now seeks past the header before writing PCM bytes, fixing `AVAudioFile` / `AVAudioPlayer` load failures that left the player stuck at `0:00 / 0:00` with a spinner.
- **Generated audio player recovers from transient same-file load misses** — the player now retries when a URL was previously marked loaded but still has no valid duration/waveform, and it can accept late precomputed peaks for the same file instead of staying stuck on the loading spinner.
- **Editor waveform supports click-to-scrub again** — a normal click now seeks the editor preview playhead, while dragging still creates the orange inpaint selection region.
- **Cover-song source audio imports are sandbox-safe** — Generate now copies dropped or file-picker-selected source audio into Anvil's Application Support import folder using security-scoped access before submitting cover generation. This keeps Desktop/iCloud/Downloads selections readable after the picker callback returns, while `AudioFileLoader` also starts security-scoped access defensively before AVFoundation reads.
- **ACE-Step WAV export no longer applies hidden adaptive tail fades** — generated WAVs now preserve the raw VAE-decoded tail. This matches upstream's default behavior and makes the recurring 28-second ending test unambiguous: if it still stops, the model/conditioning path produced it.
- **ACE-Step context mask regression reverted** — an upstream-parity experiment changed full-generation chunk masks from `1.0` to `2.0`; listening tests showed `2.0` reached the full requested duration but collapsed the output into a hum/buzz. The Swift direct-DiT path now defaults back to the musical `1.0` mask and logs it as `AUDIODIAG cond/chunkMask`.
- **ACE-Step VAE diagnostics now include per-second audio envelope stats** — verbose audio diagnostics emit `vae/out_audio_abs_mean_per_sec` and `vae/out_audio_abs_max_per_sec` for the first 60 seconds after VAE decode, making the recurring 28-second ending visible in the actual waveform rather than only by ear/player UI.
- **ACE-Step text-to-music adds a validated two-second tail guard** — diagnostics showed the 30-second cutoff is already present in `vae/out_audio` (seconds 28-29 are near-silence), not caused by WAV export or playback. Text-to-music now internally generates a small guard band and crops back to the requested duration; user retest confirmed the generated file reaches the requested 30-second end. Verbose diagnostics log `config/tailGuard`, `diff/final_latent_abs_mean_per_sec`, and `vae/out_audio_crop`.
- **Editor undo/redo history no longer grows without bound** — `EditorState` now caps raw `AudioBuffer` history to 10 snapshots and a 512 MB byte budget per stack. This keeps repeated DSP/inpaint edits from quietly retaining unlimited full-track buffers.
- **YouTube dataset subprocesses are cancellation-safe** — `YouTubeDatasetBuilder.runProcess` now drains stdout/stderr while `yt-dlp` runs and terminates the child process when the Swift task is cancelled, avoiding background downloads and pipe back-pressure hangs.
- **LoRA training can now pause and resume from Settings** — stopping a training run after progress is made can save the current adapter weights plus a training checkpoint; paused adapters are marked `In Progress` in the adapter list and expose a right-click `Resume Training…` action that reloads the original dataset, config, checkpoint, and weights.
- **LoRA training no longer requires a dummy generation first** — retrain/resume now cold-loads the ACE-Step DiT, VAE, tokenizer, text encoders, and silence latent before dataset preparation instead of failing until the generation path has warmed the model stack.
- **Cold LoRA training uses the same memory headroom as generation** — the local LLM actor is evicted before training loads ACE-Step weights and restored afterward, preventing training from stacking DiT/VAE/text encoders on top of prompt/lyrics model memory.
- **LoRA training samples are detached before backprop** — preprocessed VAE latents and conditioning tensors are now `stopGradient`/materialized before they are stored in the dataset, preventing MLX autodiff from retaining stale VAE/text-encoder graphs and crashing during trainer `vjp`.
- **LoRA training now freezes the base DiT around adapter injection** — the trainer now differentiates only through injected `loraA`/`loraB` matrices instead of accidentally asking MLX for gradients across the full ACE-Step checkpoint. Cleanup also runs through `defer` so failed/cancelled runs do not leave LoRA wrappers active.
- **LoRA matrices are now registered MLX parameters** — `LoRALinear.loraA` and `loraB` use `@ParameterInfo`, resume loading updates them in place, and the trainer explicitly unfreezes those matrices after freezing the base DiT so `valueAndGrad` has real adapter parameters to optimize.
- **LoRA injection now goes through MLX module replacement hooks** — DiT attention/MLP projection linears are registered with `@ModuleInfo`, and adapter injection/removal uses `update(modules:)` instead of direct property assignment so MLX does not keep a stale pre-LoRA module cache.
- **LoRA training bypasses the MLX nested-parameter gradient crash** — the trainer now calls the flat-array `valueAndGrad` transform directly, passes only LoRA matrices as differentiable arguments, rebuilds the small gradient dictionary afterward, and computes MSE loss in float32. This avoids the `ModuleParameters` wrapper crash while preserving normal AdamW updates.
- **LoRA training uses a stricter on-device memory envelope** — each preprocessed training sample now contributes a random window capped at 64 latent frames, training injection is limited to the final four DiT blocks, and the optimizer materializes only trainable adapter parameters/state after each step. This prevents full-length clips from forcing MLX to retain huge DiT backward activations for 30-200 second samples.
- **Audio diagnostics no longer report false `abs_mean=inf` for float16 tensors** — diagnostic reductions now cast to float32 before computing mean/max, so finite conditioning tensors no longer look exploded just because the float16 reduction accumulator overflowed.
- **Model downloads now report aggregate progress across every file and shard** — multi-file downloads such as ACE-Step XL no longer show a misleading 0→100% loop for each individual shard; the Models UI now receives one monotonic model-level progress value.
- **The main-window model sidebar refreshes after Settings changes** — downloading, deleting, adding, or setting an ACE-Step model active now notifies the inspector so the right sidebar updates without relaunching the app.

### Changed

- **ACE-Step generation now defaults to the Soft Late guidance schedule** — Advanced settings now expose Constant, Late Emphasis, Mid-to-Late, and Soft Late CFG schedules. The presets route through the existing `cfgIntervalStart/End` diffusion path for music and cover generation; the first listening A/B favored late guidance, with Soft Late giving the best sparse-prompt result and gentler endings.
- **ACE-Step prompt conditioning now emits real SFT metadata when it can** — the prompt wrapper still matches the upstream `# Instruction / # Caption / # Metas` format, but now extracts explicit or obvious BPM, time signature, and key/mode from the enhanced prompt instead of always sending `N/A`.
- **Verbose audio diagnostics now report LoRA injection coverage** — `AUDIODIAG` logs include the active DiT layer count, expected LoRA path count, matched adapter tensors, and unused adapter weight paths, which helps compare imported PEFT/XL adapters against the Swift DiT runtime.
- **LoRA injection now keeps already-wrapped paths active** — reinjecting a stack over existing `LoRALinear` modules now returns those paths instead of silently dropping them from the contribution list. This prevents partial adapter application when a generation reuses previously wrapped layers.
- **ACE-Step LoRA now loads the active DiT before adapter injection** — generation preloads/swaps the selected ACE-Step model before applying LoRA and resets stale adapter state after the swap. This prevents XL PEFT adapters from being injected into a previously resident 24-layer SFT decoder before the 32-layer XL model is loaded.
- **XL LoRA selection now nudges the guidance preset to Constant** — when an XL-trained adapter is selected while the app is still on the default Soft Late preset, Generate switches to Constant guidance for closer ACE-Step/PyTorch PEFT parity. User-selected guidance presets are left alone.
- **Audio import avoids two extra full-channel copies** — `AudioFileLoader.bufferToMLX` now interleaves directly from `AVAudioPCMBuffer.floatChannelData` instead of first materialising separate `ch0` / `ch1` arrays.

---

## [Unreleased] — Cover-Song Generation Phase 2A (foundation)

### Added

- **`ResidualFSQ`** (`AnvilCore/Inference/AudioTokenizer/`) — Finite Scalar Quantization with `project_in` / `project_out` linear bottleneck. Mirrors upstream's `vector_quantize_pytorch.ResidualFSQ` configured with `num_quantizers=1`. Provides `quantize` (continuous → indices) and `getOutputFromIndices` (indices → continuous) for both encode and skip-encode paths.
- **`AudioTokenizerPooler`** — cls-token AttentionPooler used inside `AudioTokenizer`. Reuses `Qwen3Layer` for its 2 self-attention transformer blocks. Pools `pool_window_size` audio patches into a single token via prepended cls + self-attention.
- **`AudioTokenizer`** — top-level acoustic→codes module. Composes `audio_acoustic_proj` Linear + `AudioTokenizerPooler` + `ResidualFSQ`. Converts 25 Hz `[B, T, 64]` VAE latents into 5 Hz `[B, T/pool, 2048]` quantized representations + packed indices.
- **`AudioTokenDetokenizer`** — codes→acoustic upsampler. Linear projection + 5 learnable special_tokens for window expansion + 2 self-attention layers per window + `proj_out` back to acoustic dim.
- **`AceStepLMConfig`** — JSON decoder for each LM variant's `config.json` (hidden_size / num_hidden_layers / etc). Drives variant-aware module construction; no hardcoded constants per variant.
- **`AceStepLM`** — Qwen-architecture decoder-only transformer reusing `Qwen3Layer`. Returns final hidden states (no `lm_head`) for experimental discrete audio-code workflows; source-audio covers use the tokenizer/detokenizer path directly.
- **`WeightConverter.loadShardedSafetensors(indexURL:)`** — reads `model.safetensors.index.json`'s `weight_map`, groups keys by shard, loads each shard once, and merges into a single sorted `(name, array)` list. Required for the 4B LM variant which ships across 2 safetensors files.
- **Three new weight name maps**: `AudioTokenizerNameMap`, `AudioTokenDetokenizerNameMap`, `AceStepLMNameMap`. Each strips its own subtree prefix and applies the standard snake_case → camelCase conversion plus the Qwen-style sub-renames (`self_attn` → `attention`, `input_layernorm` → `inputNorm`, etc.).
- **Three new `WeightConverter` methods** — `convertAudioTokenizerWeights`, `convertAudioTokenDetokenizerWeights`, `convertAceStepLMWeights`. All actor-isolated, called via `await`.
- **DEBUG-only smoke test** `ACEStepGenerationService.testCoverSongFoundation()` — verifies all three module weight sets load against the user's local files and logs match counts. Kept as a foundation regression check while the runtime cover-song path matures.

### Documentation

- `mapDiTKey` doc comment now explicitly notes that `tokenizer.*` and `detokenizer.*` keys are intentionally returned as nil; their loading happens through dedicated converters that work on the raw safetensors list independently.

Phase 2B/2C are now partially implemented: `CoverSongCoordinator`, Generate UI source-audio controls, `GenerationService` routing, MCP `referenceAudioPath` submission, and native source-audio tokenization/detokenization conditioning exist. Remaining cover-song work is polish around import/storage cleanup, diagnostics, experimental discrete audio-code workflows, and listening-quality iteration.

---

## [Unreleased] — SAO SFX: variational ODE Euler step, σ_max init fix, hopLength 2048

### Fixed

- **SAO Euler step used score-function ODE instead of variational v-prediction ODE**
  (`DiffusionSampler.edmEulerStep`) — SAO uses the cosine/variational noise schedule
  `x(σ) = α·x₀ + σ_c·ε` (keeping x at unit variance), NOT the raw EDM `x₀ + σ·ε`.
  Differentiating along the ODE path and substituting `v = α·ε − σ_c·x₀` gives
  `dx/dσ = α(σ)² · v`.  The Euler step is therefore
  `x_next = x_t + α(σ_t)² · (σ_next − σ_t) · v_t`.
  Previous implementation used `d = (x_t − x₀_pred) / σ` which is correct for the
  standard Karras EDM score function, but wrong for the variational ODE.

- **Initial noise was incorrectly scaled by σ_max = 500** (`SFXDiffusionActor.sample`)
  — because x(σ_max) ≈ σ_c(500) · ε ≈ ε (σ_c ≈ 1 at σ=500), the starting distribution
  is already unit normal.  The `xt *= σ_max` line fed the DiT latents with std=500,
  completely out of its training distribution, producing all-static output.

- **SAO latent frame count used hopLength=1024 instead of 2048**
  (`SoundFXPipeline.hopLength`, `SFXGenerationService`) — the SAO VAE strides
  `[2, 4, 4, 8, 8]` multiply to 2048 samples per latent frame at 44.1 kHz.
  Using 1024 doubled the latent frame count; the VAE then decoded them at 2048
  samples/frame, producing audio twice the requested duration (~60s for a 30s request).
  Fixed in both `SoundFXPipeline.hopLength` (constant) and `SFXGenerationService`
  (latentShape computation).

---

## [Unreleased] — SAO SFX: EDM diffusion math + per-layer conditioning; resolves static audio

### Fixed

- **SAO sigma schedule was rectified-flow linear [0, 1] instead of Karras EDM [500, …, 0.3]**
  (`SFXDiffusionActor.generateKarrasSigmas`) — the old `generateLinearSchedule` produced timesteps
  in `[0, 1]`, which are the wrong units for SAO's EDM-parameterised DiT. The model expects actual
  noise levels σ ∈ [0.3, 500]. Replaced with a Karras-style schedule:
  `σ_i = (σ_max^(1/ρ) + i/(N−1)·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ`, ρ=7, N=100.
  The initial noise is now also scaled to `σ_max=500` (i.e. `xt *= 500`).

- **SAO ODE step used rectified-flow formula (`x − v·dt`) instead of EDM v-prediction Euler**
  (`DiffusionSampler.edmEulerStep`) — ACE-Step uses rectified flow where the model predicts the
  straight-line velocity; SAO uses the v-objective where `v = α·ε − σ_c·x₀`. The correct Euler
  step recovers x₀ via `x₀_pred = α·x_t − σ_c·v_t` then steps along the score direction
  `d = (x_t − x₀_pred) / σ_t`. `SFXDiffusionActor` now calls `DiffusionSampler.edmEulerStep`
  instead of the ACE-Step `odeStep`.

- **SAO global conditioning was added once before the transformer loop instead of per-layer**
  (`SFXDiTModel.swift`) — the reference architecture adds `x = x + temb.unsqueeze(1)` at the
  **start of each of the 24 DiT layers**, not just once after `project_in`. `SAODiTTransformer` now
  passes `conditioning` into every layer; `SAODiTLayer.callAsFunction` applies it as the first
  operation before `pre_norm + self_attn`.

---

## [Unreleased] — SAO SFX: four runtime crash fixes; first successful generation

### Fixed

- **SAO `projection_model` Sequential indices not stripped** (`SAOProjectionNameMap`) — `time_positional_embedding` is `nn.Sequential([FourierFeatures, nn.Linear])` in the Python checkpoint; keys arrive with `.0.weights` and `.1.weight/.1.bias` numeric indices. The mapper now remaps `.0.weights → .frequencies`, `.1.weight → .embedder.weight`, `.1.bias → .embedder.bias`. Without this, `MLXNN.UpdateError.incompatibleItems` fired at `start_number_conditioner.time_positional_embedding` because the module got an array of dicts instead of named sub-module keys.
- **SAO DiT FFN bias initialised as `false`** (`SAODiTFFN`) — PyTorch `nn.Linear` defaults to `bias=True`; both `gateUp` (GEGLU proj, shape `[12288]`) and `projOut` (down linear) had `bias: false` in Swift. Fixed to `bias: true` for both.
- **`AnvilTokenizer` routing T5/Unigram to `BPETokenizer`** — when no `tokenizer_config.json` exists alongside `tokenizer.json`, the tokenizer class was inferred as `"PreTrainedTokenizer"`, which swift-transformers maps to `BPETokenizer`. SAO's tokenizer is Unigram (SentencePiece, T5-style, `vocab_size=32100`, no merges), so `BPETokenizer` crashed with `"BPETokenizer requires merges"`. Fix: parse `model.type` from the raw JSON; if `"Unigram"` use `"T5Tokenizer"` (which routes to `UnigramTokenizer`). Also fixed `path(percentEncoded: false)` for the file-existence check.
- **SAO DiT architecture mismatches causing `[broadcast_shapes]` crash** (`SFXDiTModel.swift`) — four bugs discovered by inspecting the safetensors header without loading the full 4.8 GB file:
  - `preprocessConv` and `postprocessConv` were `Conv1d(64 → 1536)` / `Conv1d(1536 → 64)`; actual checkpoint weights are `[64, 64, 1]` — both are identity-channel 64 → 64 convolutions.
  - `SAODiTTransformer` was missing `project_in [1536, 64]` and `project_out [64, 1536]` Linears that live *inside* the transformer block in the checkpoint; also had a spurious final `LayerNorm` that does not exist in the checkpoint.
  - `SAODiTLayer` norms had no `@ModuleInfo` keys (defaulted to `norm1/norm2/norm3`); correct checkpoint names are `pre_norm`, `cross_attend_norm`, `ff_norm`.
  - `SFXDiTModel.forward` tried to broadcast `gEmb [B, 1, 1536]` onto `h [B, T, 64]` before `project_in`; fixed so `gEmb` is added *after* `projectIn` expands latents to 1536 dims inside `SAODiTTransformer`.

---

## [Unreleased] — Stable Audio Open: complete DiT architecture rewrite, pipeline live

### Added

- **`SAONumberConditioner`** (`SAOProjectionModel.swift`) — replaces the previous erroneous `SAOProjectionModel` text-projection design. Stable Audio Open uses *timing* conditioning (not text projection): two `SAOSingleConditioner` instances (start-seconds and total-seconds), each with a learned Fourier frequency bank `[128, 1]` and a `Linear(257 → 768, bias)` projection. Start/total embeds are concatenated `[B, 1536]` and passed through a `toGlobalEmbed` MLP into the DiT's global conditioning stream. T5 text `[B, L, 768]` feeds cross-attention directly without any projection head.
- **Correct SAO DiT architecture verified against actual checkpoint** — full inspection of the 745-key `stable_audio_open_1_0.safetensors` produced the definitive architecture: `hiddenSize=1536`, 24 layers, 24 Q heads / 12 KV heads (GQA 2:1), `headDim=64`, `ffIntermediateDim=6144` (GEGLU: split `gate_up → gate/up → up * gelu(gate)`), partial RoPE on first 32 of 64 head dims (`rotaryDim=32`), pre-norm LayerNorm before each sub-layer. All values corrected in `SFXDiTModel.swift`.
- **GEGLU feed-forward network** (`SAODiTFFN`) — replaces the prior SwiGLU approximation. Correct projection is a single `gate_up` weight `[2·intermediate, hidden]` split into gate and up halves at runtime; `output = up × gelu(gate)` then projected back via `down`.
- **Grouped-Query Attention in SAO cross-attention** (`SAODiTCrossAttention`) — 24 Q heads / 12 KV heads; KV expanded 2× via `MLX.repeated(_:count:axis:)` before attending.
- **Partial RoPE** (`SAODiTSelfAttention`) — rotary position encoding applied only to the first 32 dims of each 64-dim head; remaining 32 dims pass through unchanged. Frequencies computed analytically from `ropeTheta=1_000_000` at forward time (not loaded from checkpoint; `inv_freq` keys are dropped by `SFXDiTNameMap`).
- **`SAOFourierFeatures`** — learned frequency bank `[128, 1]` loaded from checkpoint (`model.model.embedding.fourier_embedder.weight`); constructs `[B, 257]` timing embedding (128 sin + 128 cos + scalar).

### Changed

- **`stable-audio-open` spec flipped to `isRunnable: true`** — SAO SFX generation is now active in the pipeline. The prior `isRunnable: false` runtime gate and `runtimeNote` about re-architecture in progress have been removed from `ModelRegistry`. Auto-mode can now route to Stable Audio when SAO weights are present.
- **`SFXDiTNameMap`** — updated to handle the complete 745-key checkpoint: strips `model.model.` prefix, drops `rotary_emb.inv_freq` keys, remaps Sequential `.0.` to `fc1` / `.1.` to `fc2`, reroutes `gate_up_proj` → GEGLU `gateUp`, maps LayerNorm `gamma`/`beta` → `weight`/`bias`, transposes Conv1d axes `[out, in, kernel] → [out, kernel, in]`.
- **`SAOProjectionNameMap`** — updated for `SAONumberConditioner` key paths (`model.conditioning.conditioners.0/1.{fourier_embedder,dense_layer}.{weight,bias}`).

### Fixed

- **Test suite repaired: 10 stale test files updated for API drift** — accumulated API changes since these tests were last edited had broken compilation:
  - `ModelSpecTests.swift` — rewrote for new `ModelSpec(id:displayName:huggingFaceRepo:files:[ModelFileSpec]:category:)` init (old `[String]` files + `sizeBytes`/`sha256` init removed)
  - `ModelRegistryTests.swift` — `"ace-step-dit"` → `"ace-step-dit-turbo"` (force-unwrap crashed on renamed model ID); `file` → `file.localName` in loop
  - `AppStateTests.swift` — same `file.localName` fix
  - `ConditioningActorTests.swift` — `forTesting()` → `stub()`; added missing `try` to `async throws encode()`
  - `DiTModelTests.swift` — added `encoderAttentionMask: nil` to all `decoder()` / `actor.step()` call sites; `actorStep()` changed to `async throws`
  - `InpaintCoordinatorTests.swift` — `loadAudio()` now throws; added `try` to all 6 call sites; 5 test functions changed to `async throws`
  - `DiffusionSamplerTests.swift` — removed obsolete `type: Float.self` parameter (MLX-Swift 0.31.3 dropped it from `MLXArray(converting:)`)
  - `WAVWriterTests.swift` — replaced `MLXArray(converting:type:)` with `MLXArray(converting: [...], [1, 4])` (flat + shape)
  - `Qwen3ModelTests.swift` — replaced `MLXArray(converting: [Int], type: .int32)` with `MLXArray([Int])` (correct Int32 init)
  - `WeightConverterTests.swift` — added `try` to `convertVAEWeights`/`convertDiTWeights` (now throws); fixed all 4 `mapDiTKey` test cases to use `"decoder."` prefix and camelCase expectations
  - `EditHistoryManagerTests.swift` — added missing `import Foundation` (`Data` not in scope)
  - `MusicPipelineTests.swift` — added `encoderMask: nil` to `pipeline.generate(...)` call (new required parameter)
- **SFX tests fixed for weight-loading requirement** — `SFXDiffusionTests` and `SoundFXPipelineTests` created actors without calling `loadWeights()` first; `SFXDiTActor.predict()` and `SFXVAEActor.decode()` both `throw .weightsNotLoaded` without loaded weights, so every test hit a runtime error instead of testing the target logic. Added `randomDiTWeights()` and `randomVAEWeights(config:)` factories that build correctly-shaped random weight arrays, and call `try await dit.loadWeights()` / `await vae.loadWeights()` before any inference call.

### Removed

- **`SAOProjectionModel` text-projection design** — the erroneous `Linear(768 → 768)` text projection + `Linear(256 → 768)` timing projection has been removed. SAO does not project T5 text embeddings; the projection model in the upstream checkpoint is a timing conditioner only. The `SAOProjectionModel` type has been replaced with `SAONumberConditioner` in the same file.

---

## [Unreleased] — Memory release, SFX routing safety, Stable Audio Open re-architecture

### Added

- **MLX GPU cache flush on Release** — the Settings → Models → Release GPU Memory button now calls `MLX.GPU.clearCache()` and `MLX.GPU.resetPeakMemory()` after the existing weight unload. MLX-Swift maintains an internal allocator pool that is NOT returned to the OS just by setting model references to nil; without `clearCache()` Activity Monitor saw no drop. User-verified: 90% RAM → 24% RAM after Release. Before/after `active` / `cache` / `peak` snapshots are logged via OSLog (`subsystem:com.pimp.AnvilApp category:GenerationService.unload`) so the delta is visible in the Xcode console.
- **SFX-routing safety guard** — `GenerationService.shouldUseStableAudio` now checks `SFXModelManager.isPipelineLoadable()` before letting Auto-mode flip to Stable Audio. If the SFX install is incomplete (typical case while SAO re-architecture is in progress), Auto silently falls back to ACE-Step with a log warning instead of crashing on a missing-file open. New `isPipelineLoadable()` returns true only when ALL files `loadPipeline()` will read are present.
- **Stable Audio Open T5 + projection architecture (plumbing complete; runtime gated)** — replaces the prior CLAP-based SFX text encoder with the actual upstream architecture used by `stabilityai/stable-audio-open-1.0`: T5-base text encoder + projection_model + Oobleck VAE.
  - New `T5Encoder.swift` — full T5-base implementation: vocab=32128, d_model=768, 12 layers, 12 heads, d_kv=64, d_ff=3072, T5v1.1 gated-GELU MLP, pre-norm blocks, RMS norm, no scale on attention scores (T5 quirk), bidirectional relative-position bucket bias propagated from block 0 through the encoder stack, additive attention mask from input mask.
  - New `T5SelfAttention.relativePositionBucket(...)` and `computePositionBias(...)` mirroring `_relative_position_bucket(bidirectional=True)` and `compute_bias(...)` in HF's `transformers/models/t5/modeling_t5.py`.
  - New `SAOProjectionModel.swift` — text + timing projection (768 → 768 text path; 256 → 768 timing path) into the DiT cross-attn embedding dim.
  - New `T5NameMap.swift` and `SAOProjectionNameMap.swift` — map HF safetensors keys (`encoder.block.N.layer.[01].*`, `text_projection.weight`, etc.) onto our Swift module paths.
  - `WeightConverter` gains `loadT5Weights` and `loadSAOProjectionWeights`.
  - `SoundFXPipeline.generate` now: tokenize via T5 → T5 encode → Fourier-feature timing via existing `TimingEncoder` → `SAOProjectionModel` → cross-attn into the DiT.
  - `SFXModelManager.loadPipeline` constructs T5 + projection via `Module.update(parameters:)`; tokenizer changed from CLAP-specific to general `AnvilTokenizer` (T5 ships its own `tokenizer.json`).
  - `SFXCrossAttention` (in `SFXDiTModel`) now handles both 2D `[B, condDim]` (legacy CLAP path, preserved for symmetry) and 3D `[B, condLen, condDim]` (new T5 + projection sequence path) by detecting `conditioning.shape.count`.
  - ModelRegistry SAO spec rewritten: drops `clap_model.safetensors` (does not exist upstream); adds `text_encoder/model.safetensors`, `text_encoder/config.json`, `tokenizer/spiece.model`, `tokenizer/tokenizer.json`, `projection_model/diffusion_pytorch_model.safetensors`, `projection_model/config.json`. VAE switched from `vae_model.ckpt` to `vae/diffusion_pytorch_model.safetensors` (same weights in MLX-loadable format). Spec marked `isRunnable: false` with a `runtimeNote` until on-device verification.
- **Diffusion Advanced section disclosure persistence** — the `audioDebugVerbose` toggle, `precision` segmented control, manual `steps` override, and `shift` slider rendered in the Advanced disclosure now read/write through `@AppStorage` / `@State` correctly across app launches (Float32 default — see "Fixed" below).

### Fixed

- **Settings → Release GPU Memory crashed the app** — SwiftUI's `Settings { }` scene is a separate scene from `WindowGroup { }`, so environment objects injected on `ContentView` do NOT propagate. `ModelsSettingsView` declares four `@EnvironmentObject` dependencies but the Settings scene only injected `appState` and `apiKeyStore`. Most of the view rendered fine because the missing two were not accessed during static body evaluation, but the Release button calls `generationService.unloadAllWeights()` and `sfxModelManager.unloadPipeline()` — tapping it fatal-errored with `SwiftUICore/EnvironmentObject.swift:93: No ObservableObject of type GenerationService found`. Now injects both `services.generationService` and `sfxModelManager` into the ModelsSettingsView tab. Inline comment in `AnvilAppApp.swift` warns future maintainers about the WindowGroup → Settings env-propagation gotcha.
- **Release GPU Memory did not actually free RAM** — even after the Settings crash was fixed, clicking Release dropped DiT/VAE references but RAM stayed pinned because MLX-Swift's GPU allocator pool wasn't being returned to the OS. Adding `MLX.GPU.clearCache()` after the existing unload chain dropped user-reported usage from 90% → 24%.
- **Generate crashed when prompt enhancer rewrote a music prompt with "atmospheric/soundscape/foreboding" wording** — `IntentRouter`'s LLM classifier flipped Auto from Music to Ambient on those mood words, routing through the Stable Audio pipeline, which then hit `MLX Error: [load_safetensors] Failed to open file …/stable-audio-open/vae.safetensors`. Two stacked bugs: (1) the SFX VAE filename mismatch (registry was downloading `vae_model.ckpt` but `loadPipeline()` wanted `vae.safetensors`), and (2) the over-eager classifier. The runtime guard above prevents the crash; the spec update + IntentRouter tuning fix the underlying causes.
- **IntentRouter false-positive on enhanced music prompts** — replaced the 1-line system prompt with a 5-rule rubric + 7 worked examples. Mood/atmosphere words alone (atmospheric, soundscape, foreboding, ominous, ethereal, haunting, dreamy, moody) no longer mean ambient; any instrument / genre / BPM / structure mention forces music; "ambient music" ≠ "ambient SFX" (an ambient pad is still music); ambiguous tie-breaks to music.
- **SFXModelManager.downloadIfNeeded was using a parallel hardcoded download list** — the old path tried flat URLs at the repo root (`vae.safetensors`, `clap_model.safetensors`) that 404'd because the upstream repo has the safetensors VAE in a `vae/` subdir and no CLAP file at all. It also did not carry HF auth bearer tokens for the gated SAO repo. Collapsed onto the registry-driven download, which already handles `remotePath → localName` mapping AND the bearer-token authentication.

### Removed

- **Legacy CLAP encoder/tokenizer (Stable Audio Open re-architecture)** — `CLAPEncoder`, `CLAPTokenizer`, `CLAPNameMap`, their tests, and `WeightConverter.loadCLAPWeights` are gone. They were designed against a CLAP-text-encoder model family (e.g. AudioLDM/MusicGen), but Stable Audio Open uses T5 + projection. The replacement T5 stack lands in this same release. `InpaintCoordinator`'s SFX inpaint branch lost its CLAP encoder dependency; it now yields a clear "Stable Audio inpainting temporarily unavailable during T5 re-architecture" message instead of compiling against the removed types. ACE-Step inpaint is unaffected.

### Known Gaps

- **Stable Audio Open spec is `isRunnable: false`** — the runtime guard prevents Auto routing to it. Explicit override will refuse to activate the spec until on-device verification of the new T5 + projection chain succeeds. Music generation via ACE-Step is unaffected.
- **`SAOProjectionModel` shape defaults are educated guesses** — `Linear(textInputDim=768, outputDim=768)` and `Linear(timingDim=256, outputDim=768)`. The actual `projection.safetensors` shapes will be discovered via the `[supplied/model/matched]` diagnostic at first load and the config defaults adjusted in a one-line follow-up.

---

## [Unreleased] — ACE-Step Quality Baseline + UI Wiring

### Added

- **Native local Lyric Writer inference path** — `LLMActor` now loads the downloaded MLX Llama 3.2 3B Instruct 4-bit safetensors and runs a minimal quantized Llama forward pass in MLX-Swift instead of treating the model as loaded with an empty placeholder. Local lyric generation now streams decoded tokens from the on-device model path.
- **Custom model validation supports sharded safetensors** — Add Custom Model now accepts repositories with `model.safetensors.index.json` plus shard files like `model-00001-of-00004.safetensors`. ACE-Step DiT generation now loads all `model*.safetensors` files listed in the model spec, while genuinely config-incompatible variants can still be added/downloaded for staging but are marked "Download only" and cannot be activated.
- **ACE-Step DiT runtime is config-driven** — `DiTDecoder` now builds from `config.json`, including XL-sized bodies such as hidden_size=2560, 32 layers, and 32 attention heads. The conditioning bridge is now `encoder_hidden_size → hidden_size`, so XL can consume the shared 2048-wide VAE/text/condition encoder output without requiring a separate shared-component download.
- **Advanced steps hint includes XL** — the Generate view now clarifies that blank/auto steps means 8 for Turbo and 50 for Base, SFT, or XL.
- **Custom model downloads now show progress immediately** — Add Custom Model refreshes the Models list as soon as the spec is saved and surfaces live download progress in both the add section and the model row, instead of looking idle during large HuggingFace downloads.
- **Download-only models no longer offer Set Active** — Ready-but-unsupported custom models such as ACE-Step XL still show delete/remove controls, but no longer show a Set Active button that the registry would reject.
- **Release GPU Memory is guarded against repeated clicks** — the Settings memory button now shows an in-progress indicator and disables itself while model weights are unloading, avoiding stacked release requests during large XL teardown.
- **ACE-Step weight release no longer runs through the UI actor** — `ACEStepGenerationService` is now an actor instead of `@MainActor`, so loading/unloading large DiT weights such as XL is coordinated off the SwiftUI main actor.
- **ACE-Step quality troubleshooting notes** — `docs/ace-step-audio-quality-investigation.md` now captures the 2026-05-07 breakthrough from static/noise to recognizable music, plus the current low-hanging follow-ups: Float32 baseline testing, SFT/Base 50-step A/Bs, prompt/meta whitespace parity, VAE decode parity, and DiT patch/unpatch parity.
- **Advanced section in GenerateView** — collapsible disclosure surfacing three diffusion knobs that were already plumbed but never rendered: precision (Float16 / Float32 segmented), inference steps (numeric override; blank = auto: 8 Turbo / 50 Base or SFT), and shift (slider 0.5–10.0 with reset to 3.0). Intended for chasing the audio-quality bug and now for controlled quality A/Bs.
- **`GenerationRequest.stepsOverride` / `shiftOverride`** — Optional Int / Float request fields. When nil, `ACEStepGenerationService` keeps the existing auto-pick (8 Turbo / 50 Base or SFT) and shift=3.0 default.
- **`GenerationParameters` `@MainActor ObservableObject`** — shared bag for the four user knobs that live in both the inspector sidebar and `GenerateView`: `duration`, `seedText`, `guidanceScale`, `pipelineOverride`. Injected via `@EnvironmentObject` so both surfaces read/write the same instance.
- **ACE-Step LM model specs** (`ace-step-lm-0.6b`, `ace-step-lm-1.7b`, `ace-step-lm-4b`) and a dedicated **ACE-Step LM** section in `ModelsSettingsView` — model-registry plumbing for future discrete audio-code workflows. The 0.6B variant is single-file (1.28 GB), the 1.7B variant is a nested single-file checkpoint in `ACE-Step/Ace-Step1.5` (3.71 GB), and the 4B is sharded (7.85 GB).
- **`ModelCategory.aceStepLM`** — new enum case wired into `AddModelView`'s expected-files switch.
- **Cover-song generation Phase 2 design spec** at `docs/superpowers/specs/2026-05-07-cover-song-generation-design.md` (480 lines) — three new modules (AudioTokenizer, AudioTokenDetokenizer, AceStepLM), CoverSongCoordinator, `srcLatentsOverride` parameter on the DiT path, UI changes, phased plan (2A → 2D), open questions, test plan.
- **Cover-song Phase 2A implementation plan** at `docs/superpowers/plans/2026-05-07-cover-song-phase-2a.md` (13 tasks) — foundation modules only (AudioTokenizer + Detokenizer + AceStepLM + sharded safetensors loader + three name maps + smoke test). Phase 2B/2C/2D deferred. Execution gated on text-to-music audio-quality bug being resolved first.

### Fixed

- **Write Lyrics could click but produce nothing locally** — the local LLM actor was never auto-loaded on generation, app startup did not always hand the initial `llmActor` to `LyricWriterManager`, and the default downloaded Llama model was not auto-activated after background download. The app now auto-activates a downloaded default model when no active local LLM is set, wires the initial actor on view appear, and lazy-loads model weights before local lyric generation.
- **Lyric Writer overfilled short clips** — lyric prompts now use a shared duration-aware plan across local/OpenAI/Claude/Gemini backends. A 30s clip asks for only 4-6 total lines with no bridge/outro/full-song structure, while longer clips scale up gradually. Max token budgets now follow duration too, so local Llama is less likely to cut off mid-line or feed ACE-Step two minutes of lyrics for a 30s generation.
- **ACE-Step model picker needed quality guidance** — the Models settings now labels Turbo as "Good / fastest", Full/Base as "Better", SFT as "Best", and VAE + Text Encoder as "Shared". The section also explains that Turbo/Full/SFT are the selectable DiT quality tiers while the VAE + Qwen text encoder is a required shared component, not another quality tier.
- **Prevent unsupported custom models from becoming active** — `ModelSpec` now carries runtime support metadata, and the registry refuses to activate models marked not runnable. This keeps downloadable-but-not-yet-wired XL DiT checkpoints from reaching generation and failing with shape mismatches. Older saved custom model specs still decode with safe defaults.
- **ACE-Step Generate view restarted into Float16** — the first recognizable-music baseline was produced under Float32, but `GenerateView` reset to Float16 on every app launch and `GenerationRequest` also defaulted to Float16. Default generation precision is now Float32 so fresh launches keep the known-good quality baseline; Float16 remains selectable as an explicit speed/RAM tradeoff.
- **LoRA adapters could silently cross DiT variants** — the picker only filters by broad `aceStepDiT` category, so an adapter trained for Turbo/Base/SFT could be selected while another DiT variant was active. `LoRACoordinator` now checks `adapter.baseModelID` against the active DiT model before injection and skips mismatches with a log warning.
- **Lyric Writer download UX cascade** — four overlapping bugs that produced confused state in `IntelligenceSettingsView`:
  1. `LLMModelManager.isModelDownloaded` returned true on a directory containing only `config.json` + `tokenizer.json` with no safetensors body, because download failures still wrote the JSON files. Now requires at least one `.safetensors` entry.
  2. The `init()` left a stale `activeModel` selection pointing at a model whose files no longer existed; now sweeps and clears.
  3. `AppState`'s download-needed gate ran a strict `.idle` check that prevented retry from `.error` and ignored the actual `isModelDownloaded` filesystem truth. Now pattern-matches against `.downloading` specifically and re-checks the filesystem.
  4. `IntelligenceSettingsView`'s green-checkmark requires both `activeModel` match and `isModelDownloaded` returning true.
- **Player did not refresh on subsequent generations** — `PlayerView.onAppear { playerService.load(url:) }` only fires the first time the view appears. SwiftUI reuses the same view instance when the parent swaps in a new `GenerationResult`, so generations after the first kept the player stuck on the original track. Added `.onChange(of: fileURL) { _, newURL in playerService.load(url: newURL) }`.
- **Sidebar Duration / Seed / Guidance fields did nothing** — `MainShellView`'s inspector "Parameters" section rendered `TextField(..., text: .constant(""))` against literal black-hole bindings. Worse, `pipelineOverride` was duplicated as separate `@State` in `MainShellView` and `GenerateView`, so even the sidebar pipeline picker never reached the request. Lifted all four knobs into the new `GenerationParameters` shared object; both surfaces now bind to it.

---

## [Unreleased] — Post-Restructure Bug Fixes (on-device validation pass)

Real-world testing of the model download restructure surfaced a chain of latent issues — long-standing path-encoding ambiguity, a URLSession async-API misuse, and weight-conversion / tensor-shape assumptions that only became reachable once real weights downloaded successfully. Each fix below has been verified live (downloads finish to 100%, conditioning loads the real tokenizer, weight conversion completes without crashing).

### Fixed

- **Live download progress UI never advanced past 0%** — three layered bugs:
  1. `DownloadProgressDelegate.didWriteData` was never called because `URLSession.downloadTask(with:completionHandler:)` suppresses session-level delegate per-task callbacks (Apple-documented behavior). Replaced with the delegate-only `downloadTask(with:)` API + a unified `DownloadOrchestrator` that bridges `didFinishDownloadingTo` / `didCompleteWithError` to a `CheckedContinuation` (with `NSLock`-guarded single-resume).
  2. Once the delegate started firing, `Task { @MainActor in self.downloadProgress[id] = progress }` could run on the same runloop tick as a SwiftUI body, producing the *"Publishing changes from within view updates is not allowed"* warning at hundreds of Hz. Switched to `DispatchQueue.main.async` (always defers to a future tick).
  3. URLSession fires `didWriteData` hundreds of times per second; throttled the delegate to ~10 Hz with `NSLock`-guarded `lastFiredAt` timestamping (always passes 100% through). Plus an explicit `onProgress(0.0)` before `task.resume()` and `onProgress(1.0)` after the final move so the bar swaps out of "Waiting…" instantly and never gets stuck shy of complete.
- **Sandboxed `Application Support` paths broke `FileManager` everywhere** — `URL.path()` (function form) percent-encodes spaces to `%20` on macOS 13+, but `FileManager` String-path APIs (`fileExists(atPath:)`, `isWritableFile(atPath:)`) want decoded filesystem paths. Codebase-wide sweep replaced 8 callsites with `path(percentEncoded: false)`:
  - `ACEStepGenerationService.loadDiTWeights` tokenizer existence check (most impactful — was silently falling back to a stub `ConditioningActor` that produced wrong-shaped tensors and crashed inference downstream)
  - `ModelRegistry.status(for:)`, orphan-directory sweep, `deleteModelFiles`, `loadUserSpecs`, `loadActiveModels`
  - `SFXModelManager` file readiness check
  - `HistoryStore.loadFromDisk`
  - `OutputDirectoryManager.hasValidDirectory`
- **`moveItem` failed with `NSFileWriteFileExistsError`** — a stale partial download at the destination wasn't removed because the same `URL.path()` percent-encoding bug made the existence check skip the file. `DownloadClient` now does `try? FileManager.removeItem(at: destination)` (URL-based, no encoding ambiguity) unconditionally before `moveItem`. Same defensive guard on the intermediate UUID-named temp move inside the orchestrator.
- **`DownloadError` was opaque on first launch** — added `LocalizedError` conformance with per-status-code descriptions (401 = "Unauthorized — authentication or model access may be required", 403 = license required, 404 = repo doesn't exist, 429 = rate limited, 5xx = server error). Previously surfaced as `"AnvilCore.DownloadError error 0"`.
- **`WeightConverter.convertDiTWeights` crashed with `[transpose] Received 3 axes for array with 2 dimensions`** — the substring predicates `isDiTConv1dWeight` / `isDiTConvTransposeWeight` matched both the top-level patch projections (Conv1d / ConvTranspose1d, 3-D weights) AND inner encoder `nn.Linear` layers (also named `proj_in`/`proj_out`, 2-D weights). The 2-D Linear weights were routed through the 3-D `transposed(0,2,1)` / `transposed(1,2,0)` axis remap and crashed inside MLX. Added `value.ndim == 3` guards so the conv axis remap fires only on actual 3-D tensors; 2-D Linear weights pass through unchanged.
- **`ConditioningActor.contextLatents` had wrong sequence length** — `DiTModel.callAsFunction` channel-concatenates `contextLatents` and the noisy `hiddenStates`, requiring matching seq dims. `ConditioningActor` was using a stale `latentFrameRate: Float = 25.0` constant (from upstream's LM-hint pathway) producing 750 frames for 30s, while `MusicPipeline.latentFrames` correctly used `ceil(44100 * duration / 2048) = 646` for the same duration. Crashed at `[concatenate] shapes (1,750,128) vs (1,646,64) along axis -1`. Replaced with `audioSampleRate` + `vaeHopLength` constants and a `latentFrames(forDuration:)` static helper that mirrors `MusicPipeline.latentFrames` exactly.

---

## [Unreleased] — Model Download Restructure (Phase 1 + 2 + 3)

### Added

- **`ModelFileSpec`** — explicit `remotePath`/`localName`/`sha256`/`sizeBytes` per file in a model spec, replacing the flat `files: [String]` array. Lets one HuggingFace repo's subdirectory layout map to flat local filenames that consumer code already expects.
- **`ModelSpec.variantFamily`** — groups related specs (e.g. ACE-Step Turbo and Full) under one UI section.
- **`ModelSpec.requiresAuth`** — flags gated repos so downloads attach `Authorization: Bearer` headers and pre-flight on token presence.
- **`ace-step-dit-base`** — Full ACE-Step v1.5 DiT variant (4.8 GB, higher quality, slower) at `ACE-Step/acestep-v15-base`.
- **`DownloadError.authRequired` / `.authForbidden`** — distinct cases for "no token configured for gated repo" vs "token rejected by HF".
- **`ModelsSettingsView` variant grouping** — Turbo and Full render as indented sibling rows under an "ACE-Step v1.5" sub-header.
- **One-shot migration shim** — maps legacy active-model ID `ace-step-dit` to `ace-step-dit-turbo` and sweeps orphan model directories. Gated on `UserDefaults` flag `com.pimp.AnvilApp.modelMigration.v1`.
- **HuggingFace access token storage** — `APIKeyStore` gains a `huggingface` provider with id `"huggingface"` and env-var fallback `HUGGINGFACE_API_KEY`. Token persists in the macOS Keychain.
- **HF token UI** in `ModelsSettingsView` — secure field + Save button + link to `huggingface.co/settings/tokens`, prefilled from existing Keychain value on view appear.
- **`ModelRegistry.tokenProvider` injection** — registry takes an optional `@Sendable () async -> String?` closure. Wired at app launch from `APIKeyStore.huggingFaceToken()`.
- **Auth pre-flight gate** — `downloadModels` resolves the bearer token for any spec with `requiresAuth: true` before the network call; missing token throws `DownloadError.authRequired(repo:)`. HTTP 401/403 with token present remaps to `DownloadError.authForbidden(repo:)`.

### Changed

- **`ModelRegistry.builtInSpecs`** — placeholder repo IDs replaced with verified `ACE-Step/Ace-Step1.5` and `ACE-Step/acestep-v15-base` paths and exact file sizes from the HF API.
- **`ace-step-dit` → `ace-step-dit-turbo` (default)** — Turbo variant is now the out-of-the-box DiT model. Lower memory, 8-step diffusion. Full variant available in Settings → Models.
- **`ace-step-vae` now bundles `tokenizer.json`** — shared between DiT variants. `ACEStepGenerationService.loadDiTWeights` now looks up the tokenizer from the active VAE spec's local directory (was: DiT spec's directory).
- **`DownloadClient`** — switched to `URLRequest`-based download API to support per-request headers; new `bearerToken: String?` parameter on `downloadAndVerify` and `download` (defaults nil).

### Fixed

- **HTTP 401 "DownloadError error 0" on first launch** — placeholder repos `ACEStep/ACE-Step-v1-5-DiT` and `ACEStep/ACE-Step-v1-5-VAE` did not exist on HuggingFace. Now resolves to real, verified repo paths.

---

## [Unreleased] — GenerationService Split

### Added

- **`ACEStepGenerationService`** (`AnvilApp/Services/`) — `@MainActor final class` owning DiT/VAE weight loading, `ConditioningActor` stub→real upgrade after tokenizer download, weight-load rollback on failure, and ACE-Step diffusion. Emits typed `Event` stream (`.loadingWeights`, `.conditioning`, `.generating`, `.decoding`, `.result`) via `AsyncThrowingStream`
- **`SFXGenerationService`** (`AnvilApp/Services/`) — `@MainActor final class` owning SFX pipeline readiness (download + load) and Stable Audio diffusion. Emits typed `Event` stream (`.downloadingModel`, `.generating`, `.decoding`, `.result`)
- **`LoRACoordinator`** (`AnvilApp/Services/`) — `@MainActor final class` owning adapter injection/removal, `currentlyInjectedAdapterID` state, and a defensive base-model self-check: queries `registry.activeModelID(for: .aceStepDiT)` on every `ensureAdapter` call and clears the adapter if the base DiT changed since last injection
- **`HistoryEntry.pipeline` field populated** — now records `"stableAudio"` or `"aceStep"` instead of `nil`, so history views can distinguish which generation path produced each entry

### Changed

- **`GenerationService`** — gutted from 371 lines to ~150 line thin facade. Public API unchanged (`state`, `currentResult`, `selectedAdapterID`, `generate(request:)`, `cancel()`, `unloadAllWeights()`); zero changes to any caller (`GenerateView`, `AppState`, `ModelsSettingsView`, etc.). Internally routes to the three sub-services and maps their stream events to `@Published` state
- **`AppServices`** (`AnvilAppApp.swift`) — `init` now takes the real `OutputDirectoryManager` instead of allocating a throwaway one. Eliminates a full GPU pipeline allocation (`VAEActor` + `DiTActor` + `DiffusionActor` + `MusicPipeline` + stub `ConditioningActor`) per cold launch that was previously orphaned when `configure()` fired on first appear
- **`AppServices.configure()`** — rewritten with explicit "skip if manager and directory both unchanged" check; first `.task` fire is now a no-op instead of a full rebuild. Tracks `currentDirectory` separately so identity-only manager swaps don't conflate with directory changes
- **`AnvilAppApp.makeGenerationService()`** — wires `ACEStepGenerationService`, `SFXGenerationService`, and `LoRACoordinator` into the facade. `MusicPipeline` is shared between `ACEStepGenerationService` and `LoRACoordinator` (safe — both hold actor references). `ConditioningActor.stub()` failure path now uses `do/catch` with error interpolation instead of `try?` so crash logs include the underlying cause
- **`GenerationError.pipelineNotReady`** — description changed from SFX-specific `"Sound FX pipeline is not ready"` to generic `"Generation pipeline did not return a result"` (now thrown by both `runSFX` and `runACEStep` fallback paths)

### Fixed

- **`DownloadError`** — added `LocalizedError` conformance with per-HTTP-status-code descriptions (401 = "Unauthorized — authentication or model access may be required", 403 = "Forbidden — license acceptance required", 404 = "Not Found", 429 = "Rate limited", 5xx = "Server error", etc.). Previously surfaced as opaque `"AnvilCore.DownloadError error 0"` in error sheets when the user clicked Download Models against unavailable HuggingFace repos
- **Xcode project (`project.pbxproj`)** — registered the three new sub-service files with fresh unique UUIDs after a collision was detected with existing project entries that would have silently dropped them from the compilation plan
- **`SFXGenerationService` cancellation gap** — added `try Task.checkCancellation()` between pipeline-readiness setup and the `pipeline.generate(...)` call so a user cancellation immediately after weights load no longer triggers the GPU pipeline before unwinding
- **`ConditioningActor.forTesting(tokenizerURL:)` misnamed for production** — renamed to `make(tokenizerURL:)` since `ACEStepGenerationService` calls it on the production path (after a model's `tokenizer.json` is downloaded). The test-only naming was misleading; the new name matches its actual responsibility. The argumentless `stub()` factory is unchanged
- **`AdapterRegistry.loadWeights` blocking the main thread** — added an `async throws` overload that runs the safetensors read on `Task.detached(priority: .userInitiated)`. `LoRACoordinator.ensureAdapter` (`@MainActor`) now `await`s it, so reading multi-MB adapter files no longer stalls the UI thread. The existing synchronous version is retained for `merge()` and other off-main callers

---

## [Unreleased] — Build Fixes & Project Hygiene

### Fixed

- **`AppState.checkSetupStatus()`** — replaced `allSatisfy` with async closure (unsupported by Swift stdlib) with a sequential `for` loop that `await`s each `registry.isModelReady` call
- **`GenerationService`** — corrected wrong error case `.modelNotReady` → `.pipelineNotReady`; added missing `try` to `pipeline.injectLoRA(config:)` and `pipeline.loadLoRAWeights(_:adapterID:)` call sites
- **`MainShellView`** — added `import AnvilCore` so `PipelineChoice` resolves to `AnvilCore.PipelineChoice` (which already defines all cases + `displayName`); removed locally-duplicated `PipelineChoice` enum that would have caused a redeclaration conflict
- **`AudioBufferConverter`** — replaced unavailable `vDSP.convertElements(of:to:)` for `Float → Int16` with the correct `vDSP_vfix16(_:_:_:_:_:)` from Accelerate
- **`ModelsSettingsView`** — split chained `+ await` expression into sequential `let` bindings (Swift requires `await` as the outermost operator); replaced removed `ShapeStyle.accent` with `Color.accentColor`

### Changed

- **Xcode project (`project.pbxproj`)** — registered 22 Swift source files that existed on disk (committed to git) but were never added to the Xcode target: all 11 `UI/Editor/` views (`EditorView`, `EditorState`, `EditorToolbar`, `EditorPromptArea`, `EditorControlsBar`, `EditorWaveformView`, `DSPPaletteView`, `EQControlView`, `PitchShiftControlView`, `ReverbControlView`, `TimeStretchControlView`), all 8 `DSP/` files (`AVAudioEngineDSP`, `AudioBufferConverter`, `DSPParameters`, `DSPProcessor`, `LPCFormantShifter`, `Resampler`, `SimpleDSP`, `WSOLAProcessor`), plus `BackgroundDownloadIndicator`, `EditorService`, `LossCurveView`, `HuggingFaceImportSheet`, and `AdapterStackView`

---

## [Unreleased] — Polish & Hardening Phase 2

### Added

- **Release GPU Memory button** (`ModelsSettingsView`) — "Memory" section in Models Settings with a "Release" button that calls `AppState.unloadAllModels()`, freeing all ACE-Step and SFX weights without restarting
- **`AppState.unloadAllModels()`** — coordinating helper that calls `GenerationService.unloadAllWeights()` and `SFXModelManager.unloadPipeline()` in one shot
- **`GenerationService.unloadAllWeights()`** — unloads ACE-Step DiT and VAE weights and resets loaded model IDs; weights reload automatically on next generation
- **`SFXDiTActorError` / `SFXVAEActorError`** — typed error enums for SFX actors matching the existing `DiTActorError` / `VAEActorError` pattern

### Changed

- **`SFXDiTActor.predict()`** — now `throws SFXDiTActorError.weightsNotLoaded` when weights have not been loaded, instead of silently running against an uninitialized model
- **`SFXVAEActor.decode()` / `encode()`** — same: now `throws SFXVAEActorError.weightsNotLoaded`; added `unloadWeights()` to both SFX actors
- **`SFXDiffusionActor`** — all `dit.predict(...)` calls updated to `try await` (6 call sites across `sample` and `inpaint`)
- **`SoundFXPipeline`** — `vae.decode(...)` updated to `try await`
- **`WeightConverter.convertVAEWeights()` / `convertDiTWeights()`** — now `throws`; `try Task.checkCancellation()` added every 50 tensor iterations so cancellation interrupts multi-GB weight conversion mid-flight
- **`SFXModelManager.loadPipeline()`** — refactored into three private helpers (`loadedDiTWeights()`, `loadedVAEWeights()`, `loadedCLAPWeights()`) for readability; main method reduced to pipeline assembly
- **`SFXModelManager.baseModelsDirectory()`** — new `private static` helper eliminates duplicated App Support URL construction; `modelPath` simplified to 3 lines; `downloadIfNeeded()` now resolves the model path before the early-return guard so `modelPath` is correct even when files are already downloaded
- **`GenerationService.ensureSFXPipelineReady()`** — extracted from `generateWithStableAudio()`, encapsulating the active-model check, optional download, and pipeline load into a single returning helper
- **`CrossAttentionCache`** — added detailed doc comment explaining the `@unchecked Sendable` threading contract (one cache per generation, sequential access only, clean ownership transfer) and a warning against concurrent use

---

## [Unreleased] — Polish & Hardening Phase 1

### Fixed

- **Streaming SHA-256 verification** (`DownloadClient`) — replaced `Data(contentsOf:)` with 1 MB chunked `FileHandle` reads to prevent OOM crashes when verifying 3 GB+ model files
- **Single-consumer `AsyncStream` bug** (`DiffusionActor`, `MusicPipeline`, `SoundFXPipeline`, `WeightConverter`, `SFXDiffusionActor`) — lazy `var progress` was exhausted after first consumer; replaced with `func makeProgressStream() -> AsyncStream<…>` that creates a fresh stream per call; `continuation?.finish()` called at generation end to prevent leaks
- **`DiTActor` crash on bad state** — three `fatalError(…)` calls replaced with `throws DiTActorError.weightsNotLoaded`; callers updated throughout `DiffusionActor` and `MusicPipeline`
- **SFX pipeline weight loading** (`SFXModelManager`) — `loadPipeline()` was a stub; fully implemented weight loading via `WeightConverter.loadSafetensors` → `SFXDiTNameMap` / `SFXVAENameMap` / `CLAPNameMap` → actor `loadWeights()`; SFX VAE uses manual name mapping (not `convertVAEWeights()` which is music-VAE-specific)
- **`CLAPTokenizer` placeholder** (`GenerationService`) — `CLAPTokenizer.forTesting()` replaced with real tokenizer loaded from `SFXModelManager.tokenizer`
- **Qwen3 text mask ignored** (`ConditioningActor`) — `textMask` computed by tokenizer was silently discarded; `Qwen3Model.callAsFunction` now accepts optional `attentionMask` and converts binary mask to additive −1e9 mask passed to all 28 attention layers
- **`VAETiledDecode` force-unwrap** — `upsampleFactor: Double?` with `!` unwrap replaced with `var upsampleFactor: Double = 0.0` pattern
- **`SFXModelManager.modelPath`** — hardcoded `"stable-audio-open"` path replaced with `resolvedModelPath` stored after fetching the active spec; correct directory used regardless of which SFX model is active

### Changed

- **`VAEActor`** — three `fatalError` calls replaced with `throws VAEActorError.weightsNotLoaded`; `VAEActorError` enum added; all callers (`MusicPipeline`, `InpaintCoordinator`, `TrainingService`, `ConditioningActor`) updated
- **`ConditioningActor.init`** — `try!` on tokenizer load replaced with `throws`; `forTesting()` factory also `throws`; call site in `AnvilAppApp` uses `guard let … = try?` with `fatalError` fallback for missing bundle resource
- **`ConditioningActor.encode()`** — `try Task.checkCancellation()` added after tokenization, after Qwen3 forward pass, and after reference audio preparation
- **`DownloadClient`** — `withTaskCancellationHandler` wraps `URLSession.download(from:)` so cancelling the parent Swift task immediately calls `session.invalidateAndCancel()` on large file downloads
- **`RotaryEmbedding`** — removed eager `cosTable`/`sinTable` pre-allocation at `maxLen=32768`; now stores only `invFreq` (`headDim/2` elements) and computes position tables on demand at `forward()` time for the actual `seqLen`
- **`SFXModelManager.unloadPipeline()`** — new method frees in-memory pipeline, tokenizer, and model ID without touching files on disk
- **`AudioPlayerService`, `OutputDirectoryManager`, `HistoryStore`** — bare `print()` error calls replaced with `os.Logger` (subsystem `com.pimp.AnvilApp`, per-class categories)

---

## [Unreleased] — Phase 10c: Community Sharing + Multi-Adapter Stacking

### Added

- **Community Adapter Browser** (`CommunityBrowserView.swift`) — App Store-like interface for discovering community-shared adapters; fetches curated manifest from GitHub (`PiMPStudios/anvil-adapters`); shows name, author, description, tags, rank/category badges; inline download with circular progress; category filter + search; pull-to-refresh with 1hr cache TTL
- **HuggingFace Import** (`HuggingFaceImportSheet.swift`) — modal for importing any adapter from a HuggingFace repo URL; validates adapter.json + adapter.safetensors at repo root; shows preview with compatibility warning for category mismatch
- **CommunityService** (`CommunityService.swift`) — manages manifest fetch, adapter downloads (max 3 concurrent, auto-retry), and HF imports; tracks download state per adapter; 1hr manifest cache
- **Multi-Adapter Stacking** (`AdapterStackView.swift`) — stacking controls below primary AdapterPicker; per-adapter strength sliders (0.0–1.0, step 0.05); total strength display; picker sheet filtered to compatible available adapters; max 3 simultaneous adapters
- **AdapterStackManager** (`AdapterStackManager.swift`) — runtime state for primary + 2 additional adapters with strength; promotion-on-delete logic; capacity enforcement; stack save/restore during training
- **LoRAContribution** (`LoRAContribution.swift`) — per-layer data type holding adapterID, displayName, source, loraA/loraB matrices, and strength for multi-adapter forward pass
- **CommunityAdapter + AdapterManifest** (`CommunityAdapter.swift`) — Codable types for the curated manifest format; includes author, tags, featured flag, minAppVersion, downloadCount
- **AdapterSource enum** (`AdapterSource.swift`) — `.local`, `.community`, `.huggingface` provenance tracking for adapters
- **LoRAInjector stacking methods** — `addContribution`, `clearContributions`, `setStrength`, `removeContribution` for per-layer multi-adapter management

### Changed

- **LoRALinear** — three-path forward pass: (1) training mode (single adapter + dropout), (2) inference stacking (contributions array with per-adapter strength × scaling), (3) legacy single-adapter backward compat
- **LoRAAdapter** — added `source: AdapterSource` (defaults to `.local` for backward compat via `decodeIfPresent`) and `remoteRepoURL: URL?` for HuggingFace-sourced adapters
- **AdaptersSettingsView** — segmented control with "My Adapters" / "Community" tabs; source badges for non-local adapters; updated empty state to reference Community tab
- **TrainingService** — clears adapter stack before training starts, restores after completion (prevents multi-adapter interference with single-adapter training mode)
- **GenerateView** — integrates AdapterStackView below primary AdapterPicker when stacking is active
- Design spec: `docs/superpowers/specs/2026-05-06-phase10c-community-stacking.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase10c-community-stacking.md`

---

## [Unreleased] — Phase 10b: Advanced LoRA

### Added

- **CosineScheduler** (`CosineScheduler.swift`) — learning rate scheduling utility with linear warmup (configurable fraction, default 10%) followed by cosine decay to 0; used by LoRATrainer to dynamically adjust AdamW learning rate each step
- **TrainingCheckpoint** (`TrainingCheckpoint.swift`) — Codable struct capturing full training state (epoch, globalStep, totalSteps, bestLoss, lossHistory, weights URL, optimizer URL, config, baseModelID) for pause/resume; `CheckpointError` for mismatch detection
- **TargetLayer enum** — `.selfAttention`, `.crossAttention`, `.mlp` cases with `moduleName` (maps to LoRAInjector property names) and `defaultPatterns` (expands to injection string patterns); `displayName` extension for UI checkboxes
- **LossCurveView** (`LossCurveView.swift`) — SwiftUI Charts mini line chart displaying per-step loss history during training; auto-scaled Y axis, 120pt height
- **AdaptersSettingsView** (`AdaptersSettingsView.swift`) — dedicated "Adapters" tab in Settings window; lists all adapters with name, base model badge, rank, loss, date; context menu rename/export/delete; swipe-to-delete; "Merge Two" button; empty state guidance
- **MergeSheet** (`MergeSheet.swift`) — adapter composition modal with picker A, picker B (filtered to same base model + rank with rank badge), alpha blend slider (0–1), name field; calls `AdapterRegistry.merge()` to create new adapter from `α×A + (1-α)×B`
- **AdapterRegistry.rename/delete/export/merge** — lifecycle methods for adapter management; rename with atomic move-then-write; delete with cleanup; export copies safetensors + metadata; merge with weighted interpolation and rank validation
- **Settings "Adapters" tab** — third tab in Settings TabView alongside Intelligence and Models

### Changed

- **LoRAConfig** — renamed `epochs` to `maxEpochs`; changed `targetLayers` from `[String]` to `Set<TargetLayer>`; added `warmupFraction` (default 0.10), `gradientAccumulationSteps` (default 1); added `injectionPatterns` computed property bridging enum to string patterns; added `Hashable` conformance
- **LoRAAdapter** — properties `id`/`name` changed from `let` to `var` for rename support; added `Hashable` conformance (required for SwiftUI Picker tags)
- **LoRATrainer** — integrates CosineScheduler for per-step LR updates; per-sample optimizer stepping with LR/progress reporting every `gradientAccumulationSteps` samples; tracks `globalStep`, `lossHistory`, `currentEpoch`; single designated init with optional `resumeFrom` checkpoint parameter; `makeCheckpoint()` for state persistence
- **TrainingProgress** — extended with `step`, `learningRate`, `estimatedTimeRemaining`, `lossHistory`, `elapsedTime` fields; backward-compatible `estimatedRemaining` accessor
- **TrainingService** — `TrainingState.training` now wraps `TrainingUIState` struct; added `.paused(TrainingCheckpoint)` state; extended progress callback for new fields
- **TrainingSheet** — target layer multi-select checkboxes; Advanced disclosure group with warmup slider and gradient accumulation picker; LossCurveView in progress state; current LR and ETA display; conditional Resume/Start buttons for paused state
- **LoRAInjector** — `inject()` iterates `config.injectionPatterns` (computed from `Set<TargetLayer>`) instead of raw `[String]`
- **ModelCategory** — added `Hashable` conformance
- Design spec: `docs/superpowers/specs/2026-05-06-phase10b-advanced-lora.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase10b-advanced-lora.md`

---

## [Unreleased] — Phase 10a: LoRA Fine-Tuning

### Added

- **LoRALinear module** (`LoRALinear.swift`) — subclasses MLXNN `Linear` with low-rank A·B adapter matrices; output = base(x) + scaling * x @ A^T @ B^T; Kaiming uniform init for A, zero init for B (starts as identity); `merge()`/`unmerge()` for folding adapter into base weights with zero inference overhead; `freezeBase()` marks base parameters non-trainable
- **LoRAInjector** (`LoRAInjector.swift`) — static utility for module tree manipulation; `inject(model:config:)` walks DiTDecoder layers and swaps target Linear properties with LoRALinear wrappers; `remove(model:paths:)` restores plain Linear; `extractWeights`/`loadWeights` for safetensors serialization
- **LoRATrainer** (`LoRATrainer.swift`) — flow-matching training loop with AdamW optimizer; samples t∈[0,1], interpolates x_t = (1-t)·x₀ + t·noise, MSE loss on predicted vs target velocity; trains only LoRA A/B matrices (~6.3M params at rank 16); tracks best weights across epochs; cooperative cancellation support
- **LoRAConfig + LoRAAdapter** (`LoRAConfig.swift`) — config struct (rank, alpha, dropout, targetLayers, learningRate, epochs, batchSize) with sensible defaults; adapter metadata struct (id, name, baseModelID, baseModelCategory, sampleCount, finalLoss, createdAt)
- **AdapterRegistry** (`AdapterRegistry.swift`) — lightweight file-system scanner for `~/Library/Application Support/Anvil/models/adapters/`; lists adapters, filters by base model category, saves/loads adapter.safetensors + adapter.json
- **TrainingService** (`TrainingService.swift`) — orchestrates full training flow: preprocesses dataset (audio→latent via VAE, text→conditioning), runs training through actor chain, saves adapter; reports progress to UI; WAV file loading for training data
- **TrainingSheet** (`TrainingSheet.swift`) — SwiftUI sheet for training config + progress; config state shows selected tracks, adapter name, rank/epochs pickers, "+ Add audio files..." import; progress state shows epoch N/M, current loss, elapsed/remaining time; "Stop Training" with save-if-possible confirmation; completion view with adapter details
- **AdapterPicker** (`AdapterPicker.swift`) — dropdown in Generate view showing "None" + compatible adapters filtered by baseModelCategory; displays name, rank, sample count per adapter; selection persisted via @AppStorage
- **History multi-select** — "Select" toolbar button toggles multi-select mode with checkmarks; selected count badge; "Train LoRA" button appears when tracks selected; presents TrainingSheet
- **Adapter lazy injection** — `GenerationService.ensureAdapter()` compares selected vs injected adapter at generation time; injects/removes/swaps as needed; falls back to base model on any adapter error; adapter removed when base model changes

### Changed

- **DiTComponents** — `DiTAttention` and `SwiGLUMLP` Linear properties changed from `let` to `var` for runtime LoRA injection
- **DiTActor** — gained `injectLoRA`, `removeLoRA`, `loadLoRAWeights`, `trainLoRA`, `cancelTraining` methods
- **DiffusionActor/MusicPipeline** — LoRA forwarding methods through the actor chain; `encodeMean` for training dataset prep; `cancelTraining` for cooperative stop
- **GenerateView** — added `adapterPickerSection` between controls bar and progress; syncs selection to GenerationService
- **Package.swift** — added `MLXOptimizers` dependency for AdamW
- Design spec: `docs/superpowers/specs/2026-05-06-phase10a-lora-fine-tuning.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase10a-lora-fine-tuning.md`

---

## [Unreleased] — Phase 9b: Model Management UI

### Changed

- **Settings window** — upgraded from single-view to `TabView` with "Intelligence" and "Models" tabs (Cmd+,)
- **GenerationService lazy swap** — replaced `weightsLoaded` boolean with model ID comparison; detects active model mismatch at generation time, unloads old weights via ARC, loads new weights from registry path, then generates
- **SFXModelManager swap support** — replaced hardcoded model ID/repo with registry dependency; `ensureActiveModel()` checks active `.stableAudio` model before generation, unloads and reloads pipeline on mismatch
- **DiTActor/VAEActor unloadable** — `decoder`/`vae` properties changed from `let` to `var` optional with `decoderFactory` closure for recreation; `unloadWeights()` nils references for ARC deallocation; `loadedModelID` tracks which model is loaded

### Added

- **ModelsSettingsView** (`ModelsSettingsView.swift`) — Models tab with ACE-Step (Music) and Stable Audio (SFX) sections; each row shows display name, formatted size, status badge (Ready/Not Downloaded/Downloading N%/Verifying/Failed), active indicator with checkmark and background highlight; Set Active/Download/Delete/Remove actions with confirmation dialogs; error alert for failed operations; task cancellation on rapid refresh
- **AddModelView** (`AddModelView.swift`) — HuggingFace repo input with category picker (ACE-Step DiT, Stable Audio); HEAD request validation for expected files per category; specific error messages (missing file, private/gated repo, network error, duplicate); size estimation from Content-Length headers; display name inference from repo ID; confirmation alert before adding; auto-download after add
- **ModelCategory enum** (`ModelCategory`) — `aceStepDiT`, `aceStepVAE`, `stableAudio`, `lyricLLM` cases for pipeline assignment; `category` field on `ModelSpec`
- **Active model tracking** — `ModelRegistry.activeModels` mapping (`[ModelCategory: String]`) persisted to `active-models.json`; `setActiveModel`, `activeModelID`, `activeSpec`, `specs(for:)` methods; defaults to built-in model IDs on first launch
- **Model file deletion** — `ModelRegistry.deleteModelFiles(id:)` removes downloaded files while preserving spec for re-download; reverts active model to built-in default on deletion
- **Error revert on swap failure** — if new model weights fail to load, active model reverts to previous working model ID; nil-decoder guard prevents fatalError
- **LocalizedError on RegistryError** — user-visible error descriptions for all registry error cases
- **ModelSpec.displayName** — computed property deriving readable name from HuggingFace repo (hyphens/underscores to spaces, title-cased)
- Design spec: `docs/superpowers/specs/2026-05-06-phase9b-model-management-ui.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase9b-model-management-ui.md`

---

## [Unreleased] — Phase 9a: Model Management Foundation

### Changed

- **First-launch redesign** — blocking download reduced from ~7.6 GB (all 4 models) to ~4 GB (ACE-Step only); users generate music while SFX and LLM models download in the background
- **Setup gate** — `AppState.checkSetupStatus` now checks only essential models (`ace-step-dit`, `ace-step-vae`) via `ModelRegistry.isModelReady`; SFX and lyric features unlock automatically when their background downloads complete
- **ModelRegistry architecture** — split from single `nonisolated let allSpecs` into `static let builtInSpecs` + actor-isolated `var userSpecs` with computed `var allSpecs`; JSON persistence for user-added model specs at `~/Library/Application Support/Anvil/models/user-models.json`

### Added

- **SHA256 download verification** (`DownloadClient.downloadAndVerify`) — enforces SHA256 checksum after each file download with one automatic retry on verification failure; empty hash dictionaries bypass verification (forward-compatible for community models in 9b)
- **BackgroundDownloadIndicator** (`BackgroundDownloadIndicator.swift`) — capsule-shaped progress pill showing background download status: spinner with model name and percentage while downloading, checkmark on completion (auto-dismisses after 3 seconds), warning with click-to-retry on failure
- **Background download coordination** (`AppState.startBackgroundDownloads`) — sequential background downloads after first launch: Stable Audio Open first (more immediately useful), then Lyric Writer LLM; progress relayed from manager `$downloadState` publishers to `BackgroundDownloadPhase`
- **Registry extension points** — `addModel`, `removeModel`, `isBuiltIn` methods on ModelRegistry (functional in 9a, UI in 9b); `RegistryError` enum for built-in protection and duplicate detection
- **ModelSpec Codable conformance** — enables JSON persistence for user-added models; `sha256` parameter defaults to empty dictionary
- Design spec: `docs/superpowers/specs/2026-05-06-phase9a-model-management-foundation.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase9a-model-management-foundation.md`

---

## [Unreleased] — Phase 8b: Custom DSP Backend

### Changed

- **Time Stretch backend** — replaced `AVAudioUnitTimePitch` with custom WSOLA (Waveform Similarity Overlap-Add) processor; time-domain stretching with cross-correlation search for optimal overlap alignment; window 1024 samples, 75% overlap, ±64 sample search range, energy gating on silent frames
- **Pitch Shift backend** — replaced `AVAudioUnitTimePitch` with three-stage pipeline: linear resampler → WSOLA time correction → LPC formant correction; eliminates "chipmunk effect" on vocals at large pitch changes
- **LPC Formant Correction** (`LPCFormantShifter.swift`) — estimates spectral envelope via Levinson-Durbin recursion (order 24), inverse-filters shifted audio to remove distorted formants, re-filters with original coefficients to restore natural vocal resonances
- **DSPProcessor region targeting** — `processRegion` helper handles selection-based byte-level splicing directly in DSPProcessor (previously delegated to AVAudioEngineDSP for time stretch/pitch shift)

### Added

- **WSOLAProcessor** (`WSOLAProcessor.swift`) — WSOLA time stretch with Hann windowing, vDSP cross-correlation search, overlap-add with envelope normalization, energy gating bypass for silent frames
- **Resampler** (`Resampler.swift`) — linear interpolation resampler for pitch ratio conversion; ratio clamped 0.25–4.0, stereo de-interleave/re-interleave
- **LPCFormantShifter** (`LPCFormantShifter.swift`) — LPC formant correction with autocorrelation via vDSP, Levinson-Durbin solver with reflection coefficient clamping (±0.99), inverse/synthesis filtering, overlap-add output
- Design spec: `docs/superpowers/specs/2026-05-06-phase8b-custom-dsp-backend.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase8b-custom-dsp-backend.md`

### Removed

- `AVAudioEngineDSP.applyTimeStretch` and `applyPitchShift` methods (replaced by custom processors; EQ and Reverb still use AVAudioEngine)

---

## [Unreleased] — Phase 8a: Time Stretch & Pitch Shift

### Added

- **Time Stretch** — rate change (0.25×–4.0×) without pitch change via `AVAudioUnitTimePitch` offline rendering; output buffer length adjusts to match new duration
- **Pitch Shift** — pitch change (±24 semitones / ±2400 cents) without duration change via `AVAudioUnitTimePitch` offline rendering; preserves original buffer length
- **TimeStretchControlView** (`TimeStretchControlView.swift`) — rate slider with snap points (0.5×, 0.75×, 1.0×, 1.25×, 1.5×, 2.0×), quality zone indicator (green/yellow/red), double-click direct numeric entry
- **PitchShiftControlView** (`PitchShiftControlView.swift`) — pitch slider with snap points at musical intervals (-12, -7, -5, 0, +5, +7, +12 semitones), quality zone indicator, direct numeric entry
- **DSPPaletteView "Transforms" section** — palette split into "EFFECTS" (Phase 7 operations) and "TRANSFORMS" (Time Stretch, Pitch Shift) with divider and sub-headers; `DSPEffect.isTransform` drives grouping
- **Quality zone indicator** — three-state display (Excellent/Good/Significant artifacts) based on parameter thresholds; green ≤2.0× or ≤±12st, yellow ≤3.0× or ≤±18st, red beyond
- **DSPProcessor dispatch** — `makeDescription` accepts `DSPParameters` for richer undo labels (e.g., "Time Stretch 0.50× (whole track)", "Pitch Shift +7 st (1.2s–2.8s)")
- Selection cleared after time stretch (duration change shifts region boundaries)
- Design spec: `docs/superpowers/specs/2026-05-06-phase8a-time-stretch-pitch-shift.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase8a-time-stretch-pitch-shift.md`

---

## [Unreleased] — Phase 7: DSP Audio Editor

### Added

- **DSPProcessor** (`DSPProcessor.swift`) — facade dispatching effects to SimpleDSP or AVAudioEngineDSP; generates human-readable operation descriptions for undo popover
- **SimpleDSP** (`SimpleDSP.swift`) — pure Swift/vDSP implementations: Trim to Selection, Delete Selection, Fade In (linear ramp), Fade Out (linear ramp), Normalize (peak via `vDSP_maxmgv`)
- **AVAudioEngineDSP** (`AVAudioEngineDSP.swift`) — offline AVAudioEngine rendering for 3-band Parametric EQ (`AVAudioUnitEQ`) and Reverb (`AVAudioUnitReverb` with Room/Hall/Plate presets); region splicing for selection-targeted effects
- **AudioBufferConverter** (`AudioBufferConverter.swift`) — format conversion between `Data` (16-bit PCM), `[Float]` (vDSP), and `AVAudioPCMBuffer` (AVAudioEngine)
- **DSPParameters** (`DSPParameters.swift`) — `AudioBuffer`, `DSPEffect`, `ApplyTarget`, `DSPParameters`, `EQBand`, `EQParameters`, `ReverbPreset`, `ReverbParameters`, `NormalizeMode`, `DSPOperation`
- **DSPPaletteView** (`DSPPaletteView.swift`) — always-available effects palette in Edit tab with 3-column button grid, expandable EQ/Reverb parameter panels, instant-apply checkmark confirmation, smart "Apply to" target indicator
- **EQControlView** (`EQControlView.swift`) — 3-band parametric EQ controls with Frequency, Gain (color-coded), and Q sliders per band
- **ReverbControlView** (`ReverbControlView.swift`) — preset row (Room/Hall/Plate) + Decay, Wet/Dry, Pre-Delay, Damping sliders; selecting preset fills sliders, adjusting slider deselects preset
- **EditorState DSP additions** — `expandedEffect`, `eqParameters`, `reverbParameters`, `audioBuffer` properties; `applyDSPEffect(_:parameters:)` method with smart target resolution; `computePeaks(from:)` for waveform display
- **EditorView integration** — DSPPaletteView below AI controls with divider; real audio file loading via `AVAudioFile`; keyboard shortcuts: Cmd+T (trim), Cmd+Delete (delete), Cmd+Shift+N (normalize), Cmd+Shift+E (EQ), Cmd+Shift+R (reverb)
- Design spec: `docs/superpowers/specs/2026-05-06-phase7-dsp-audio-editor.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase7-dsp-audio-editor.md`

---

## [Unreleased] — Phase 6: AI Audio Editor

### Added

- **InpaintCoordinator** (`InpaintCoordinator.swift`) — actor orchestrating the full inpainting pipeline: VAE encode → mask → noise injection → diffusion denoise → composite → VAE decode; manages undo stack, routes through ACE-Step or Stable Audio pipelines
- **MaskBuilder** (`MaskBuilder.swift`) — converts time selection (+ optional frequency band) into latent-space binary mask `[1, T, 1]` or `[1, T, D]`; handles sample-rate-to-frame conversion via hop size
- **LatentBlender** (`LatentBlender.swift`) — noise injection at strength level and latent compositing; seeded noise for reproducible edits
- **TimestepMapper** (`TimestepMapper.swift`) — maps strength (0.0–1.0) to diffusion start timestep; returns truncated schedule for partial denoising (fewer steps = faster)
- **EditHistoryManager** (`EditHistoryManager.swift`) — multi-level undo/redo stack (max 20 snapshots) with discard-on-branch semantics
- **InpaintTypes** (`InpaintTypes.swift`) — `EditMode`, `InpaintSelection`, `FrequencyBand`, `InpaintRequest`, `InpaintProgress`, `InpaintPreset`, `EditSnapshot`, `PipelineChoice` (moved to AnvilCore)
- **SFXVAEModel encoder path** — symmetric downsample blocks (`SFXDownsampleBlock`) mirroring the existing upsample decoder; `encode(audio:)` returns mean latents
- **VAEActor.encodeMean** — deterministic encoding for inpainting (vs. `encodeSample` which adds stochastic noise)
- **DiffusionActor.inpaint** — partial-noise diffusion start with pre-truncated schedule; reuses CFG/APG/ODE/SDE stepping from `generate()`
- **SFXDiffusionActor.inpaint** — same partial-noise pattern for Stable Audio pipeline
- **EditorView** (`EditorView.swift`) — Edit tab root: waveform canvas, frequency constraint panel, dual prompts, controls bar, empty state with file picker
- **EditorWaveformView** (`EditorWaveformView.swift`) — Canvas-based stereo waveform with drag-to-select (orange overlay), playhead, zoom-ready
- **EditorToolbar** (`EditorToolbar.swift`) — mode toggle (Inpaint/Repaint segmented), pipeline picker, undo/redo with Cmd+Z shortcuts
- **EditorPromptArea** (`EditorPromptArea.swift`) — two-column layout: Original Prompt (context, read-only style) + Region Prompt (editable, blue-themed); relabels to "Target Style" in Repaint mode
- **EditorControlsBar** (`EditorControlsBar.swift`) — preset buttons (Subtle/Moderate/Full), strength + guidance scale sliders, dynamic action button ("Inpaint Selection" / "Repaint Track")
- **EditorState** (`EditorState.swift`) — `@MainActor ObservableObject` managing all editor UI state; `buildRequest()` assembles `InpaintRequest` from current state
- **EditorService** (`EditorService.swift`) — `@MainActor ObservableObject` wrapping `InpaintCoordinator` for environment injection
- **Edit tab in sidebar** — new `.edit` entry in `MainShellView` under "Create" section
- **"Edit This" button** — added to `PlayerView` after generation; posts notification to navigate to Edit tab with audio + prompt pre-loaded
- **HistoryEntry additions** — `pipeline: String?` and `originalPrompt: String?` fields for provenance tracking
- Design spec: `docs/superpowers/specs/2026-05-06-phase6-ai-audio-editor.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase6-ai-audio-editor.md`

---

## [Unreleased] — Phase 5: Sound FX Pipeline

### Added

- **SoundFXPipeline** (`SoundFXPipeline.swift`) — top-level actor orchestrating CLAP encoding, timing conditioning, diffusion sampling, and VAE decode for Stable Audio Open; max duration 47s, progress stream via `AsyncStream<GenerationProgress>`
- **CLAPEncoder** (`CLAPEncoder.swift`) — Swift actor wrapping a CLAP text transformer (configurable layers/heads/vocabSize); produces `[1, 512]` conditioning vector from text prompt; pre-norm residual attention + GELU MLP
- **CLAPTokenizer** (`CLAPTokenizer.swift`) — word-level tokenization with character-level OOV fallback; BOS/EOS framing, pad/truncate to maxLength (77); loads vocabulary from JSON or uses `forTesting()` factory
- **TimingEncoder** (`TimingEncoder.swift`) — sinusoidal positional embeddings from start time and duration scalars; output `[1, embeddingDim]` with values bounded to [-1, 1]
- **SFXDiTModel + SFXDiTActor** — Stable Audio diffusion transformer with adaptive layer norm (adaLN), self-attention, cross-attention for text/timing conditioning, SiLU-gated MLP; actor provides `predict(latents:timestep:conditioning:)`
- **SFXVAEModel + SFXVAEActor** — upsampling VAE decoder with configurable upsample ratios; converts latent sequences to stereo audio waveforms
- **SFXDiffusionActor** — diffusion sampling loop with linear timestep schedule, classifier-free guidance (CFG) when `guidanceScale > 1.0`, cancellation support via `Task.checkCancellation()`
- **Weight Name Maps** — `CLAPNameMap`, `SFXDiTNameMap`, `SFXVAENameMap` for converting PyTorch checkpoint keys to MLX-Swift parameter names
- **SFXModelManager** (`SFXModelManager.swift`) — manages deferred download (~4GB) of Stable Audio Open weights from HuggingFace; uses `URLSession.download(from:)` for large files; tracks `SFXDownloadState` via Published property
- **PipelineChoice** enum — `.auto`, `.aceStep`, `.stableAudio` for manual pipeline override
- **GenerationService routing** — `shouldUseStableAudio(request:)` dispatches to correct pipeline based on `PipelineChoice` override or resolved `GenerationMode`; `generateWithStableAudio` / `generateWithACEStep` split
- **GenerationState.downloadingModel** — new state case for SFX model download progress
- **GenerationRequest** additions — `resolvedMode: GenerationMode` and `pipelineOverride: PipelineChoice` fields
- **UI: download badge** — mode buttons show `arrow.down.circle` when SFX model not yet downloaded
- **UI: duration cap** — slider/value capped at 47s when Stable Audio pipeline is resolved
- **UI: Inspector pipeline section** — shows active pipeline name and override picker (when SFX model downloaded)
- **SFXDiTConfig**, **SFXVAEConfig**, **SFXDiffusionConfig** types added to `Types.swift`
- Design spec: `docs/superpowers/specs/2026-05-05-phase5-soundfx-pipeline.md`
- Implementation plan: `docs/superpowers/plans/2026-05-06-phase5-soundfx-pipeline.md`

---

## [Unreleased] — Phase 4b: Intelligence Layer

### Added

- **LLMActor** (`LLMActor.swift`) — Swift actor for local LLM inference via MLX-Swift; streaming `generate()` and single-shot `classify()` methods
- **IntentRouter** (`IntentRouter.swift`) — debounced live classification of user prompts into `GenerationMode` (music/soundFX/ambient) using local LLM; 500ms debounce, maxTokens 20
- **PromptEnhancer** (`PromptEnhancer.swift`) — expands brief descriptions into detailed audio prompts via LLM; cancellable, mode-aware system prompt
- **LyricWriterManager** (`LyricWriterManager.swift`) — coordinates backend selection (local + API), streaming output to binding, cancellation
- **LocalLyricWriter** (`LocalLyricWriter.swift`) — on-device lyric generation via LLMActor with section markers and duration-aware line estimates
- **API Lyric Writer backends** — ClaudeLyricWriter, OpenAILyricWriter, GeminiLyricWriter, XAILyricWriter; streaming HTTP with proper cancellation
- **LLMModelManager** (`LLMModelManager.swift`) — model download (URLSession.download for large files), custom model add/remove, reactive `llmActor` publishing
- **APIKeyStore** (`APIKeyStore.swift`) — Keychain storage with env var fallback for API keys; supports Claude, OpenAI, Gemini, xAI providers
- **GenerationMode** enum — `.music`, `.soundFX`, `.ambient` with classification parsing
- **ModeSelection** enum — `.auto` / `.manual(GenerationMode)` for UI state
- **IntelligenceSettingsView** — model selection, backend management, API key configuration
- **GenerateView** additions — mode picker with auto-detect indicator, Enhance button, Write Lyrics button
- **App wiring** — reactive LLM propagation via `.onReceive(modelManager.$llmActor)`, API key refresh via `.onChange(of: apiKeyStore.availableProviders)`
- Design spec: `docs/superpowers/specs/2026-05-05-phase4b-intelligence-layer.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase4b-intelligence-layer.md`

---

## [Unreleased] — Phase 4a: Generation Flow

### Added

- **GenerationService** (`GenerationService.swift`) — `@MainActor ObservableObject` orchestrating full generation: validation → weight loading → conditioning → diffusion → VAE decode → WAV write → history entry
- **GenerationRequest** type — captures all generation parameters (text, lyrics, duration, seed, method, guidanceScale, precision)
- **GenerationState** enum — idle, loadingWeights, conditioning, generating, decoding, complete, error, cancelled
- **HistoryStore** (`HistoryStore.swift`) — JSON-backed persistent history of generated audio with search
- **HistoryEntry** type — metadata for each generation (prompt, lyrics, duration, seed, method, fileName)
- **OutputDirectoryManager** — user-selected output folder with security-scoped bookmark persistence
- **AudioPlayerService** — `AVAudioPlayer`-based playback with play/pause/stop
- **WAVWriter** extensions — write `MLXArray` samples to WAV file at specified sample rate
- **GenerateView** (`GenerateView.swift`) — prompt input, lyrics editor, controls bar, progress section, player
- **HistoryListView** — searchable list of past generations with playback
- **PlayerView** — inline audio player with waveform-style progress
- **MainShellView** — three-panel NavigationSplitView with sidebar, canvas, inspector
- **OutputDirectoryPicker** — first-use folder selection before main UI appears
- **FirstLaunchView** / **AppState** — setup flow gating
- Design spec: `docs/superpowers/specs/2026-05-05-phase4a-generation-flow.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase4a-generation-flow.md`

---

## [Unreleased] — Phase 3: Conditioning Pipeline

### Added

- **Qwen3Model** (`Qwen3Model.swift`) — 28-layer bidirectional text encoder, 1024-dim hidden, 151,669-token vocabulary; `callAsFunction(inputIds:)` returns `[B, T, 1024]`, `embedTokens(_:)` returns embedding-only lookup
- **Qwen3Layer / Qwen3Attention** (`Qwen3Layer.swift`) — pre-norm transformer layer with GQA (16 query / 8 KV heads, head_dim=128), QK-RMSNorm, RoPE via shared `RotaryEmbedding`, no causal mask
- **LyricEncoder** (`LyricEncoder.swift`) — 8-layer encoder with `Linear(1024→2048)` input projection; takes Qwen3 token embeddings, returns `[B, T_lyric, 2048]`
- **TimbreEncoder** (`TimbreEncoder.swift`) — 4-layer encoder with `Linear(64→2048)` input projection, followed by 2-layer `AttentionPooler` with 5 learnable query tokens; takes VAE latents `[B, 750, 64]`, returns timbre tokens `[B, 5, 2048]`; silence fallback (`zeros([1, 750, 64])`) for text-only generation
- **ConditionEncoder** (`ConditionEncoder.swift`) — orchestrates sub-encoders: projects text hidden states `[B, T, 1024→2048]`, concatenates text + lyric + timbre tokens along sequence axis; returns `encoderStates [B, T_combined, 2048]` and combined attention mask
- **ConditioningActor** (`ConditioningActor.swift`) — top-level Swift actor: tokenizes text and lyrics, runs Qwen3 forward pass, runs LyricEncoder, VAE-encodes reference audio (pad/truncate to 750 frames), runs TimbreEncoder, assembles `ConditioningResult`; produces `contextLatents [B, T, 128]` = zeros for text-only or `cat([latents, chunkMask], axis:-1)` for reference audio
- **AnvilTokenizer** (`AnvilTokenizer.swift`) — thin wrapper around swift-transformers `AutoTokenizer`; `encodeText(_:maxLength:)` and `encodeLyrics(_:maxLength:)` return `(ids: MLXArray, mask: MLXArray)` shaped `[1, maxLength]` as Int32; `forTesting()` factory with minimal BPE vocab for unit tests
- **Qwen3NameMap** (`Weights/Qwen3NameMap.swift`) — maps Qwen3-Embedding-0.6B HuggingFace safetensors keys to MLX-Swift parameter names; filters `lm_head.*`
- **ConditionEncoderNameMap** (`Weights/ConditionEncoderNameMap.swift`) — maps ACE-Step DiT checkpoint keys (`condition_encoder.{text_projector,lyric_encoder,timbre_encoder,timbre_attention_pooler}.*`) to Swift module paths; handles `special_token` → `attentionPooler.queryTokens`
- **WeightConverter** additions — `convertQwen3Weights(from:progress:)` and `convertConditionEncoderWeights(from:progress:)` methods
- **ModelComponent** additions — `.qwen3` and `.conditionEncoder` cases
- **ConditioningResult** type — `encoderStates`, `encoderMask`, `contextLatents` (`@unchecked Sendable`)
- **MusicPipeline** convenience method — `generate(text:lyrics:referenceAudio:duration:config:conditioning:)` runs conditioning then diffusion in one call
- swift-transformers (HuggingFace, v0.1.12+) added as dependency to `AnvilCore/Package.swift`
- Design spec: `docs/superpowers/specs/2026-05-05-phase3-conditioning-pipeline.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase3-conditioning-pipeline.md`

---

## [Unreleased] — Phase 2: Core Inference Stack

### Added

- **MusicPipeline** coordinator actor — top-level orchestrator that runs diffusion then VAE decode, returning `AudioResult` with stereo audio at 44.1 kHz
- **DiffusionActor** — full generation loop with seeded noise initialization, CFG batch doubling, APG guidance, and `AsyncStream<DiffusionProgress>` for progress reporting
- **DiffusionSampler** — ODE and SDE stepping methods, APG (Adaptive Projected Guidance) with momentum and norm thresholding, seeded noise generation
- **DiTDecoder** (`DiTModel.swift`) — 24-layer Diffusion Transformer decoder with patch embedding (Conv1d stride 2), de-patchify (ConvTransposed1d), adaptive layer norm output, and sliding window mask caching
- **DiTActor** — thread-safe actor wrapping `DiTDecoder` with `step()` and `loadWeights()` methods
- **DiT components** (`DiTComponents.swift`):
  - `RotaryEmbedding` — precomputed cos/sin tables, base frequency 1,000,000, max 32,768 tokens
  - `DiTAttention` — Grouped Query Attention (16 query / 8 KV heads), QK-RMSNorm, optional sliding window mask, cross-attention with KV cache support
  - `SwiGLUMLP` — `down(silu(gate(x)) * up(x))`, intermediate size 6144
  - `TimestepEmbedding` — sinusoidal → linear → SiLU → linear → 6×D projection for AdaLN
  - `DiTLayer` — AdaLN-modulated self-attention + cross-attention + MLP
  - `createSlidingWindowMask` — additive -1e9 mask for bidirectional window of 128 tokens
- **VAEActor** — actor wrapping `AutoEncoderOobleck` with fp16/fp32 precision switching and tiled decode delegation
- **VAE model** (`VAEModel.swift`) — `OobleckEncoder`, `OobleckDecoder`, `AutoEncoderOobleck` with `encodeSample` (diagonal Gaussian), `encodeMean`, and `decode`
- **VAETiledDecode** — overlap-discard chunked decode for latent sequences >2048 frames (chunk=2048, overlap=64, stride=1920); prevents OOM on songs >85s
- **VAE components** (`VAEComponents.swift`) — `Snake1d` activation, `OobleckResidualUnit` (dilated conv1d + skip, dilations 1/3/9), `OobleckEncoderBlock`, `OobleckDecoderBlock`
- **WeightConverter** actor — converts PyTorch `.bin` state dicts to MLX-Swift weight lists for on-device use (pure Swift, no Python subprocess)
- **VAENameMap** — `weight_norm` fusion (g+v → fused), Conv1d/ConvTransposed1d axis transposition, Snake1d parameter squeeze
- **TensorNameMap** — DiT key remapping (`proj_in.1.*` → `proj_in.*`), rotary embedding key filtering
- **CrossAttentionCache** — layer-indexed (key, value) dictionary cache; populated on first diffusion step, reused for all subsequent steps
- **TimestepSchedule** — three scheduling modes: lookup table (shift 1.0/2.0/3.0), linspace with shift transform, custom timestep snapping
- **Shared inference types** — `Precision`, `InferMethod`, `DiffusionConfig`, `DiffusionResult`, `AudioResult`, `TimeCosts`, `PipelineState`, `GenerationProgress`, `ConversionProgress`
- mlx-swift dependency added to `AnvilCore/Package.swift` (MLX, MLXNN, MLXRandom, MLXFast)
- Design spec: `docs/superpowers/specs/2026-05-05-phase2-core-inference-stack.md`
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase2-core-inference.md`
- Audio editor roadmap entries (DSP — Phase 5; AI inpaint/repaint — Phase 6)

---

## [Unreleased] — Phase 1: Foundation

### Added

- **AnvilApp Xcode project** — SwiftUI shell targeting macOS 14+, App Store sandbox entitlements, Hardened Runtime for notarization
- **Three-panel main shell** — sidebar / canvas / inspector layout
- **First-launch download UI** — model weight download flow with progress display
- **AppState** — observable state container with injectable `ModelRegistry` for testing
- **ModelRegistry** — status tracking and download orchestration for model weights
- **DownloadClient** — HuggingFace URL building, async download with progress, SHA256 verification
- **ModelSpec** — model weight metadata type (name, URL, expected SHA256, size)
- **AnvilCore Swift package** — initialized at `AnvilCore/`, separate from the Xcode app target
- Design spec: `docs/superpowers/specs/` (initial Anvil Swift app design)
- Implementation plan: `docs/superpowers/plans/2026-05-05-phase1-foundation.md` (implicit)

---

## Road Map

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Foundation — Xcode project, SwiftUI shell, ModelRegistry, DownloadClient |
| 2 | ✅ Complete | Core Inference Stack — VAE, DiT, Diffusion Sampler, MusicPipeline, WeightConverter |
| 3 | ✅ Complete | Conditioning Pipeline — Qwen3 text encoder, LyricEncoder, TimbreEncoder, ConditionEncoder, ConditioningActor |
| 4a | ✅ Complete | Generation Flow — GenerationService, WAV output, history, audio player, full UI |
| 4b | ✅ Complete | Intelligence Layer — IntentRouter, PromptEnhancer, LyricWriters (local + API), LLMActor |
| 5 | ✅ Complete | Sound FX Pipeline — Stable Audio Open, CLAP encoder, dual-pipeline routing, deferred download |
| 6 | ✅ Complete | AI Audio Editor — inpaint/repaint via VAE encode → mask latents → denoise |
| 7 | ✅ Complete | DSP Audio Editor — trim, fade, normalize, 3-band parametric EQ, reverb |
| 8a | ✅ Complete | Time Stretch & Pitch Shift — AVAudioUnitTimePitch, quality zones, snap-point sliders |
| 8b | ✅ Complete | Advanced DSP — custom WSOLA backend, LPC formant preservation, linear resampler |
| 9a | ✅ Complete | Model Management Foundation — SHA256 verification, first-launch redesign, registry split |
| 9b | ✅ Complete | Model Management UI — ModelsSettingsView, AddModelView, active model tracking, lazy swap runtime |
| 10a | ✅ Complete | LoRA Fine-Tuning — LoRALinear, LoRAInjector, flow-matching trainer, adapter picker, training sheet |
| 10b | ✅ Complete | Advanced LoRA — CosineScheduler, target layer picker, gradient accumulation, checkpoint/resume, loss curve, adapter merge/rename/delete/export |
| 10c | ✅ Complete | Community Sharing + Stacking — curated manifest browser, HuggingFace import, multi-adapter stacking (Inject-All) with real-time strength sliders |
