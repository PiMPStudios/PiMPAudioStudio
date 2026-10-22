# PiMP Audio Studio

Native macOS MLX app for local text-to-music, sound effects, latent editing,
dataset packaging, and LoRA / LoKR adapter training. Built with Swift and
SwiftUI for Apple Silicon.

Current release identity is **0.8.0**. This is pre-1.0 software. There is no
signed public DMG yet; the supported way to run it is to build from source.

Generation, editing, and local training stay on your Mac after the required
models are downloaded. Optional remote GPU training can run on cloud hardware
you approve, such as RunPod. The Mac app itself has no Python runtime.

Licensed under [Apache License 2.0](LICENSE).

## Requirements

- macOS 15.0 or newer.
- Apple Silicon Mac for local MLX generation and training.
- **Xcode 26** or newer to build from source (Swift 6.2; required by vendored
  Demucs MLX Swift). Xcode 16 is not sufficient.
- **16 GB** unified memory can run lighter generation paths, but is tight.
  **24 GB+** is more comfortable for XL models. **32 GB+** is recommended for
  local XL adapter training and native ACE annotation packs.
- Several GB of free disk for first-launch essentials (about 6 GB for ACE-Step
  v1.5 Turbo plus the shared VAE / text encoder). Tens of GB if you also
  download XL models, Stable Audio, annotation packs, and datasets.
- Network access for first-time model downloads, optional provider CLIs,
  optional Hugging Face imports, and optional remote GPU workflows.

## Building From Source

```bash
git clone https://github.com/PiMPStudios/PiMPAudioStudio.git
cd PiMPAudioStudio
open PiMPAudioStudio.xcodeproj
```

Build and run the `PiMPAudioStudio` scheme in Xcode.

The app target depends on the local `PiMPAudioCore` Swift package plus external
Swift packages declared in `project.yml` / the Xcode project. See
[Build And Test](docs/developer/build-and-test.md) for tests and formatting.

## First Launch And Models

On first launch the app downloads the essential music models (ACE-Step v1.5
Turbo and the shared VAE / text encoder). That only happens once. Additional
models such as Stable Audio Open and a local lyric LLM can download later.

Manage models from **Settings → Models**. Weights live in the app sandbox
under Application Support (see
[First Launch And Model Setup](docs/user/model-setup.md) for the full path).

Some downloads require a Hugging Face token and an accepted model license.
Tokens are stored in macOS Keychain and are not written into dataset packages,
remote job zips, traces, or logs.

Do not bundle large or gated model weights into a public `.app`. The app
downloads them after install.

## What It Does

### Generation

- Text-to-music with ACE-Step v1.5 (Turbo / Full / SFT) and XL variants.
- ACE-Step LM options (0.6B, 1.7B, 4B) for cover and timbre workflows.
- Stable Audio Open for short-form sound effects.
- Prompt routing, prompt enhancement, and local or API lyric writing.
- Generation Analyzer with live traces, `.trace.json` sidecars, and A/B
  compare.

### Editing And History

- Latent inpaint / repaint for selected regions.
- DSP: trim, fades, normalize, EQ, reverb, time stretch, pitch shift
  (WSOLA / LPC where those paths apply).
- History playback, listening notes, preferred takes, and selected-take
  deletion.

### Datasets And Training

- Build packages from local files (whole tracks by default; optional clips).
- Review captions, lyrics, BPM, key, language, approval, and validation.
- Native enrichment: optional ACE captioner / transcriber, Qwen fallback,
  forced alignment, opt-in Demucs stems, provenance.
- Local Swift/MLX LoRA and LoKR (including DoRA), checkpoints, and audition
  snapshots.
- Import native PiMP adapters, PEFT LoRA folders, and compatible
  LyCORIS / LoKR / Side-Step run folders.

### Remote GPU

Optional, billable, confirmation-gated RunPod Secure Cloud training:

- Export a reviewed dataset as a remote job zip.
- Launch / terminate pods, then **Upload → Start → Monitor → Pull & Import**.
- Portable `remote-worker/` for stock CUDA PyTorch pods.
- Copy-paste SSH commands remain as a manual fallback.

See [Remote GPU Training](docs/user/remote-gpu-training.md).

### MCP

Optional [Model Context Protocol](https://modelcontextprotocol.io) server for
local agents. **Off by default.** Default bind is `127.0.0.1:38765`
(`http://127.0.0.1:38765/mcp`). You can bind all interfaces; that requires a
bearer token. Mutating tools default to dry-run and need confirmation.

See [MCP For Users](docs/user/mcp.md).

### Diagnostics

Verbose audio diagnostics can write timestamped `[AUDIODIAG]` logs into the
app sandbox Diagnostics folder. See
[Diagnostics And Logs](docs/user/diagnostics-and-logs.md).

## Documentation

In the running app, open **Help → PiMP Audio Studio Help** (`⌘?`). That window
loads the same user guides from `docs/user/` (copied into the app bundle at
build time).

| Audience | Start here |
| --- | --- |
| End users | [docs/user/README.md](docs/user/README.md) |
| Developers | [docs/developer/README.md](docs/developer/README.md) |
| Bug reports | [docs/user/bug-report-checklist.md](docs/user/bug-report-checklist.md) |

## Project Layout

```text
PiMPAudioStudio.xcodeproj     Xcode project
project.yml                   XcodeGen project source
PiMPAudioStudio/              macOS app target
PiMPAudioCore/                Swift package (inference, training, registry)
PiMPAudioStudioTests/         App unit tests
remote-worker/                Portable CUDA worker for cloud pods
Vendor/                       Vendored third-party Swift (original licenses)
Spikes/                       Isolated experiments
docs/                         User and developer docs
CHANGELOG.md                  Project changelog
LICENSE                       Apache License 2.0
NOTICE                        Apache attribution notice
TRADEMARKS.md                 Name and icon; not granted by the license
CONTRIBUTING.md               How to contribute
THIRD_PARTY_NOTICES.md        Direct dependency licenses
```

## Status

This is a working 0.8 macOS app, not a finished 1.0 product. Honest near-term
work:

- Repeatable Developer ID signing, notarization, and DMG packaging.
- Remaining MCP tools: train-from-history, resume training, `delete_model`.
- First-class Side-Step source jobs for long audio and sidecar workflows.

The Mac app stays Swift / SwiftUI / MLX. Remote workers may use Python,
PyTorch, and CUDA on hardware you launch. API keys stay in Keychain or your
environment; they do not belong in job zips. Third-party code and downloadable
models keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This distribution is **not** an App Store build. The sandbox includes temporary
path exceptions for optional local CLI tools (Homebrew prefixes and provider
state directories).

## Contributing And Issues

Contributions are welcome under Apache License 2.0. See
[CONTRIBUTING.md](CONTRIBUTING.md).

When you file a GitHub issue, follow the
[bug report checklist](docs/user/bug-report-checklist.md). Do not include API
keys, tokens, SSH keys, private audio, or gated model files.

## License

PiMP Audio Studio is licensed under [Apache License 2.0](LICENSE).

Copyright 2026 Professional Interactive Media Productions LLC, d/b/a PiMP
Studios.

The Apache license does not grant trademark rights. See
[TRADEMARKS.md](TRADEMARKS.md).

Third-party components and model assets remain subject to their own licenses.
See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

Built by [PiMP Studios](https://github.com/PiMPStudios).
