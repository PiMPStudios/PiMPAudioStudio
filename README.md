# Anvil

**AI-powered music generation for macOS**, built natively with Swift, SwiftUI, and [MLX](https://github.com/ml-explore/mlx-swift).

Anvil runs diffusion-based audio models entirely on-device using Apple Silicon GPU acceleration. No cloud, no subscriptions, no internet required after initial model download.

---

## Features

### Generation

- **Text-to-Music** — Describe what you want, get a full audio clip. Powered by AceStep DiT (flow-matching diffusion transformer)
- **Sound Effects** — Stable Audio Open pipeline components for short-form SFX generation
- **Dual Pipeline** — Automatic routing between music and SFX models based on prompt intent
- **Intelligent Prompting** — IntentRouter classifies prompts; PromptEnhancer expands terse ideas into detailed generation prompts
- **Lyric Writing** — Local Llama 3.2 3B Instruct and API-based (Claude, OpenAI, Gemini, xAI) lyric generation with structured, duration-aware output
- **Generation Analyzer** — Live backend trace, saved `.trace.json` sidecars, two-run compare, A/B playback, and edit handoff for judging quality changes

### Audio Editing

- **AI Inpaint/Repaint** — Select a region, describe what should replace it. VAE encode, mask latents, denoise, blend
- **DSP Processing** — Trim, fade in/out, normalize, 3-band parametric EQ, reverb
- **Time Stretch** — WSOLA algorithm with LPC formant preservation for natural-sounding tempo changes
- **Pitch Shift** — Real-time pitch shifting with configurable quality/latency tradeoff

### LoRA Fine-Tuning

- **On-Device Training** — Flow-matching training loop with AdamW optimizer, cosine LR scheduling, gradient accumulation
- **Configurable Targets** — Self-attention, cross-attention, MLP injection with per-layer control
- **Checkpoint/Resume** — Save and restore training state mid-session
- **Training Dataset Packages** — Slice local files into package manifests, export remote-ready zips, and import Anvil/anvil-audio-v2 JSON bundles with vocal transcripts
- **Adapter Management** — Rename, delete, export, merge (weighted interpolation of two adapters)
- **Community Sharing** — Browse and download community adapters from a curated manifest; import any adapter from HuggingFace
- **Multi-Adapter Stacking** — Apply up to 3 LoRA adapters simultaneously with real-time strength sliders (Inject-All architecture)

### Model Management

- **First-Launch Setup** — Guided download of base models with SHA-256 verification
- **Multiple Models** — Add/remove models, switch active model at runtime without restart
- **Deferred Downloads** — SFX pipeline downloads only when first needed

---

## Architecture

```text
AnvilApp (macOS target)
├── UI/              SwiftUI views (Generate, Editor, Settings, Training)
├── Services/        GenerationService, TrainingService, CommunityService, etc.
├── DSP/             WSOLA, LPC, EQ, resampler implementations
└── Models/          App-level data types

AnvilCore (Swift Package)
├── Inference/
│   ├── DiT/         Diffusion Transformer (AceStep)
│   ├── VAE/         Variational Autoencoder (encode/decode audio latents)
│   ├── Diffusion/   Sampler, timestep scheduling
│   ├── Conditioning/ Qwen3 text encoder, lyric/timbre encoders
│   ├── StableAudio/ Sound FX pipeline components (T5/projection, SFX DiT, SFX VAE)
│   ├── Pipeline/    MusicPipeline orchestration
│   ├── Editor/      Inpaint coordinator, mask builder, latent blender
│   └── Weights/     SafeTensors loading, name mapping, weight conversion
├── LoRA/            LoRALinear, LoRAInjector, LoRATrainer, CosineScheduler
├── Community/       CommunityAdapter, AdapterSource, manifest types
├── LLM/             LLMActor (local Llama inference for prompts/lyrics)
└── ModelRegistry/   ModelSpec, DownloadClient, registry management
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | SwiftUI, Swift Charts |
| ML Runtime | [MLX Swift](https://github.com/ml-explore/mlx-swift) (Apple Silicon GPU) |
| Tokenization | [swift-transformers](https://github.com/huggingface/swift-transformers) |
| Audio I/O | AVFoundation, AVAudioEngine |
| DSP | Custom WSOLA + LPC (pure Swift) |
| Networking | URLSession (model downloads, community manifests) |
| Persistence | FileManager + JSON + SafeTensors |

---

## Requirements

- macOS 14.0+ (Sonoma)
- Apple Silicon (M1 or later) — required for MLX GPU acceleration
- Xcode 15.4+ to build
- ~8 GB RAM minimum (16 GB recommended for generation + training)

---

## Building

```bash
git clone https://github.com/PiMPStudios/AnvilApp.git
cd AnvilApp
open AnvilApp.xcodeproj
```

Build and run with Xcode. The AnvilCore Swift package resolves automatically.

On first launch, the app guides you through downloading the base model weights (~2-4 GB depending on selected models).

---

## Project Structure

```text
AnvilApp.xcodeproj        Xcode project (macOS app target)
AnvilCore/                Swift Package (inference engine, LoRA, registry)
  Package.swift           SPM manifest (mlx-swift, swift-transformers)
  Sources/AnvilCore/      Library source
  Tests/AnvilCoreTests/   Unit tests (Metal-dependent tests require GPU)
AnvilApp/                 App source
  AnvilAppApp.swift       @main entry point, environment object wiring
  UI/                     SwiftUI views organized by feature
  Services/               Business logic and orchestration
  DSP/                    Signal processing implementations
  Models/                 App-level data types
Spikes/
  MCPServerSDKSpike/      Isolated Swift MCP SDK server transport spike
docs/
  training-datasets.md      Local dataset package build/import workflow
  mcp-local.md              Local MCP endpoint usage and curl smoke examples
  mcp-standards-roadmap.md  Standards-first MCP + Swift 6 migration direction
  superpowers/specs/      Design specifications per phase
  superpowers/plans/      Implementation plans per phase
CHANGELOG.md              Detailed changelog with road map
```

---

## Road Map

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Foundation — project setup, ModelRegistry, DownloadClient |
| 2 | ✅ | Core Inference — VAE, DiT, Diffusion Sampler, MusicPipeline |
| 3 | ✅ | Conditioning — Qwen3 text encoder, lyric/timbre encoders |
| 4a | ✅ | Generation Flow — full generation UI, WAV output, history |
| 4b | ✅ | Intelligence — IntentRouter, PromptEnhancer, LyricWriters |
| 5 | ✅ | Sound FX — Stable Audio Open, dual-pipeline routing |
| 6 | ✅ | AI Editor — inpaint/repaint via masked latent diffusion |
| 7 | ✅ | DSP Editor — trim, fade, normalize, EQ, reverb |
| 8a | ✅ | Time Stretch & Pitch Shift — AVAudioUnit + quality zones |
| 8b | ✅ | Advanced DSP — WSOLA, LPC formant preservation |
| 9a | ✅ | Model Management — SHA-256 verification, registry split |
| 9b | ✅ | Model Management UI — settings, add/remove, lazy swap |
| 10a | ✅ | LoRA Fine-Tuning — trainer, injector, adapter picker |
| 10b | ✅ | Advanced LoRA — scheduler, checkpoints, merge, export |
| 10c | ✅ | Community Sharing + Multi-Adapter Stacking |
| 11 | In Progress | Local MCP — localhost-only agent control surface for models, history, traces, compare, safe app actions, and standards-first SDK hardening |

---

## License

Private repository. All rights reserved.

---

*Built by [PiMP Studios](https://github.com/PiMPStudios)*
