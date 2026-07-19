# PiMP Audio Studio

PiMP Audio Studio is a native macOS music-generation, editing, dataset, and
adapter-training app built with Swift, SwiftUI, and MLX.

The default app workflow runs locally on Apple Silicon. Optional remote GPU
training can be launched on user-approved cloud hardware such as RunPod, but the
macOS app itself stays Swift/MLX and App Store friendly.

## Current Capabilities

### Generation

- Text-to-music generation with ACE-Step music models.
- ACE-Step v1.5 XL model choices: XL Base, XL SFT, and XL Turbo.
- ACE-Step LM support for cover/timbre workflows with 0.6B, 1.7B, and 4B model
  options.
- Stable Audio Open SFX pipeline for short-form sound effects.
- Prompt routing and prompt enhancement for music/SFX intent.
- Local/API lyric writing through the app's intelligence layer.
- Generation Analyzer with live stage tracing, `.trace.json` sidecars, saved
  trace review, and A/B trace comparison.

### Editing

- Latent inpaint/repaint for selected audio regions.
- DSP tools: trim, fades, normalize, EQ, reverb, time stretch, and pitch shift.
- WSOLA/LPC-backed time and pitch processing for higher-quality edits.
- History playback, trace-backed review, listening notes, preferred takes, and
  selected-take deletion.

### Datasets

- Build training datasets from local audio files.
- Preserve whole tracks by default, or explicitly extract seeded/evenly-spread
  clips when a clipped dataset is required.
- Dataset review workspace with caption, lyric, BPM, key, language, time
  signature, notes, approval status, and package validation.
- Native enrichment with optional four-bit ACE captioning/transcription,
  lightweight Qwen fallback, Qwen3 forced alignment, opt-in Swift/MLX Demucs
  stems, BPM/key analysis, and full provenance metadata.
- ACE-compatible sidecar import/export, including LRC timestamp cleanup.
- Full-mix, vocal, and instrumental training bundles.

### Adapter Training

- Local Swift/MLX LoRA and LoKR training.
- Narrow and Wide recipes, dataset-size-aware recommendations, deterministic
  seeds, validation loss, checkpoints, optimizer-state resume, and metrics logs.
- LoKR DoRA support.
- Training completion handoff to fixed-seed adapter audition snapshots.
- Adapter gain controls for testing subtle or under-scaled adapters.

### Adapter Management

- Import native PiMP adapters, PEFT LoRA folders, compatible LyCORIS/LoKR
  checkpoints, and Side-Step LoRA/LoKR run folders.
- Preserve XL Base/SFT/Turbo compatibility metadata where available.
- Export, rename, delete, select, and audition adapters.
- Adapter effect diagnostics through verbose audio logs.

### Remote GPU Training

- Create remote job zips from reviewed dataset packages.
- Store RunPod API keys in macOS Keychain.
- Refresh RunPod GPU types, pricing/stock details, and official templates.
- Launch and terminate RunPod Secure Cloud pods after explicit confirmation.
- Copy upload, SSH, and extract/run commands.
- Portable `remote-worker/` package for stock CUDA pods.
- Worker output contract: `remote_artifacts/logs.jsonl`,
  `metrics.jsonl`, `result.json`, checkpoints, final adapter, and command
  scripts.

Current remote workflow note: pod launch and worker packaging are present, but
upload/run/artifact download/import are still partly manual. The next planned
remote upgrade is first-class Side-Step source jobs for long source audio plus
Side-Step `.txt` sidecars.

### MCP And Automation

- Localhost-only MCP endpoint for local agents and scripts.
- SDK-backed Streamable HTTP server with guarded mutating actions.
- Tools/resources for models, history, traces, compare, editor operations,
  generation, training inspection/control, remote GPU preflight/status, and
  workflow prompts.
- Pollable operation records for long-running generation and editor work.

### Diagnostics

- Verbose audio diagnostics mirror `[AUDIODIAG]` lines to timestamped files in
  the app's sandboxed Diagnostics folder.
- Generation, adapter-effect, dataset, training, and remote worker breadcrumbs
  are designed to make failures inspectable after the run.

## Model Setup

Models are managed in:

```text
Settings -> Models
```

Sandboxed model storage:

```text
~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/models/
```

Some downloads require a Hugging Face token and accepted model license. Tokens
are stored in macOS Keychain and are not written into dataset packages, remote
job zips, traces, or logs.

Large/gated model weights should not be bundled into public `.app` builds. The
app should guide users through downloading them after install.

## Requirements

- macOS 15.0 or newer.
- Apple Silicon Mac for local MLX generation/training.
- Xcode 16 or newer to build from source.
- 16 GB RAM recommended for generation; more is strongly preferred for local XL
  adapter training.
- Network access for initial model downloads, optional provider CLIs, optional
  Hugging Face imports, and optional remote GPU workflows.

## Building From Source

```bash
git clone https://github.com/PiMPStudios/PiMPAudioStudio.git
cd PiMPAudioStudio
open PiMPAudioStudio.xcodeproj
```

Build and run the `PiMPAudioStudio` scheme in Xcode.

The app target depends on the local `PiMPAudioCore` Swift package plus external
Swift packages declared in `project.yml` / the Xcode project.

## Documentation

User-facing docs live in:

```text
docs/user/
```

Start at `docs/user/README.md`. Covered topics include getting started, model
setup, music and SFX generation, the editor, history/presets, datasets and
training, adapters, remote GPU, intelligence/lyrics, local CLI access, MCP,
diagnostics, shortcuts, troubleshooting, and bug reports.

Developer docs live in:

```text
docs/developer/
```

Start at `docs/developer/README.md` for architecture, repo layout, build/test,
inference pipelines, dataset/training internals, adapters, remote worker, MCP,
sandbox/security, documentation integration, and release packaging.

The public release plan is to bundle selected user docs into an in-app
Documentation/Help experience so users do not need repository access. Menu bar
wiring is intentionally deferred until presentation is decided.

## Project Layout

```text
PiMPAudioStudio.xcodeproj     Xcode project
project.yml                   XcodeGen project source
PiMPAudioStudio/              macOS app target
  UI/                         SwiftUI screens and controls
  Services/                   App orchestration, model/download/training/MCP services
  DSP/                        Editor DSP helpers
  Models/                     App-level model types
PiMPAudioCore/                Swift package for inference, training, registry, DSP
  Sources/PiMPAudioCore/
  Tests/PiMPAudioCoreTests/
remote-worker/                Portable CUDA/Side-Step worker for cloud pods
Spikes/                       Isolated experiments
docs/                         User docs, developer docs, plans, roadmaps, debug notes
CHANGELOG.md                  Project changelog
```

## Current Roadmap

Near-term work called out by the docs:

1. Finish end-to-end remote GPU lifecycle: upload, run, monitor, download
   artifacts, and one-click adapter import.
2. Add Remote Side-Step source jobs for long audio files plus rich Side-Step
   sidecars, pinned Side-Step execution, tensor-cache reuse, and recipe sweeps.
3. Bundle user docs into the app and build the in-app Help window.
4. Add repeatable release packaging: version bump, signed/notarized app, DMG,
   smoke tests, and sandbox entitlement audit.
5. Continue MCP polish around long-running task progress, cancellation, and
   guarded destructive actions.
6. Continue dataset/training refinements such as saved training presets,
   checkpoint browsers, gradient sensitivity estimation, and optional remote
   ACE-Step captioner/transcriber jobs.
7. Explore larger creative directions such as live adapter auditioning,
   automation curves, prompt A/B blends, DAW/export workflows, GGUF/C++ runtime
   evaluation, and the Generative Stem Editor concept.

## Architecture Boundaries

- The macOS app remains native Swift/SwiftUI/MLX.
- No Python runtime is added to the local app.
- Remote workers may use Python, PyTorch, CUDA, ACE-Step, Side-Step, PEFT, and
  LyCORIS because they run outside the app on user-approved cloud hardware.
- API keys and provider tokens stay in Keychain or user-controlled environment
  variables; they do not belong in job zips.
- External projects are references for behavior and formats unless their license
  and the implementation plan explicitly allow code reuse.

## License

Private repository. All rights reserved.

---

Built by [PiMP Studios](https://github.com/PiMPStudios).
