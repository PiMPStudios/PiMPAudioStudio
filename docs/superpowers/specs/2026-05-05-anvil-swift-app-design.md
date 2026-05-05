# Anvil Swift App — Design Spec

**Date:** 2026-05-05
**Status:** Approved
**Author:** tweaver

---

## Overview

A native macOS application for AI audio generation — sound effects, foley, ambient, and full
songs with lyrics — built with SwiftUI and MLX-Swift, targeting the Mac App Store. The Python
`anvil-audio` repo stays unchanged as a reference implementation; this is a new standalone
product.

The app is the unified front door to all audio generation needs. A user types a natural
language description — anything from "footsteps on a metal floor" to "a dark country sad song
with a soul twang to it" — and gets back polished audio. No mode switching required by
default, though explicit control is always available.

---

## Goals

- Ship on the Mac App Store as a fully sandboxed native macOS app
- Support all audio generation use cases: music with lyrics, sound FX, foley, ambient
- Work fully offline out of the box (no API key required for any core feature)
- Let power users swap models, fine-tune with LoRA, and override routing manually
- Feel like a professional macOS app, not a wrapped Python tool

---

## Non-Goals

- iOS / iPadOS support (future consideration)
- Cloud inference or server-side processing
- Porting the Python anvil-audio codebase line-by-line (it is a reference, not source)

---

## Repo

A new standalone repository, separate from `anvil-audio`. The Python repo retains its GitHub
presence and clone audience untouched.

Suggested name: `AnvilApp` (working title — rename before App Store submission).

---

## Stack

| Layer | Technology |
|---|---|
| UI | SwiftUI, macOS 14+ (Sonoma minimum) |
| Inference | MLX-Swift |
| Concurrency | Swift Concurrency (async/await, actors) |
| Model download | URLSession with HTTP range request resume |
| API key storage | Keychain (never UserDefaults or bundle) |
| Weight format | `.safetensors` loaded directly by MLX-Swift |

---

## Project Structure

```
AnvilApp/
├── AnvilApp.xcodeproj
├── AnvilApp/                   ← SwiftUI app target
│   ├── UI/                     ← Views, screens, navigation
│   ├── Generation/             ← Pipeline wiring, intent routing
│   ├── Lyrics/                 ← LyricWriter protocol + backends
│   ├── Models/                 ← Weight download, cache management
│   └── Audio/                  ← Output handling, playback
└── AnvilCore/                  ← Swift package, no UIKit/AppKit dependency
    ├── Pipeline/               ← MusicPipeline, SoundFXPipeline, IntentRouter
    ├── LyricWriter/            ← Protocol + LocalLLMWriter + API backends
    └── ModelRegistry/          ← ModelSpec, download, verification
```

`AnvilCore` is a separate Swift package so all inference logic is testable without the UI
target.

---

## Inference Stack

ACE-Step-1.5's Python MLX layer serves as the blueprint. Each Python module has a direct
Swift equivalent in `AnvilCore/Pipeline/`.

### Music Pipeline (ACE-Step)

| ACE-Step Python | Swift equivalent |
|---|---|
| `mlx_dit_init.py` | `DITModel.swift` |
| `mlx_vae_init.py` / `mlx_vae_decode_native.py` | `VAEModel.swift` |
| `core/generation/handler/diffusion.py` | `DiffusionSampler.swift` |
| `core/generation/handler/conditioning_*.py` | `ConditioningBuilder.swift` |
| `core/generation/handler/repaint_*.py` | `RepaintController.swift` |
| `core/generation/handler/generate_music.py` | `MusicPipeline.swift` (actor) |

### Sound FX Pipeline (Stable Audio)

A parallel pipeline backed by Stable Audio Open weights, covering sound effects, foley,
and ambient generation. Uses the same `VAEModel` and `DiffusionSampler` Swift types as
the music pipeline, but loaded with Stable Audio's own weights and config — the types are
shared, not the instances.

### Generation Flow

```
User prompt
     ↓
IntentRouter (local LLM classifies: music | sound-fx | ambient)
     │
     ├── music  →  LyricWriter → ConditioningBuilder → DiffusionSampler → VAEModel
     │
     └── sound-fx / ambient  →  ConditioningBuilder → DiffusionSampler → VAEModel
                                                            ↓
                                                       AudioOutput
```

`MusicPipeline` and `SoundFXPipeline` are both `actor` types — Swift's actor model provides
data-race safety on inference state and clean cooperative cancellation via `Task.cancel()`.

### LoRA Support

ACE-Step-1.5's LoRA training pipeline (`core/lora/`, `training_v2/`) is ported as an
on-device LoRA trainer using MLX-Swift. The user provides reference audio and a style
description; the trainer produces a lightweight adapter saved alongside the base weights.
Adapters are loaded at generation time and managed by `ModelRegistry` as first-class entries.

Fine-tuning is gated to M2 Pro / Max / Ultra and above. On underpowered hardware the UI
shows a clear capability message rather than a silent failure.

---

## Intent Router

`IntentRouter` uses the same local LLM model that powers `LocalLLMWriter` — no second
model download needed. Classification is a separate system prompt + short inference call
on the same loaded weights.

**Default behaviour:** automatic. The app reads the prompt and routes without user
intervention.

**Manual override:** a mode selector (Music / Sound FX / Ambient) is always visible in the
Canvas. Selecting a mode pins routing for that session. Switching back to Auto is one tap.

This gives novice users a zero-friction experience while giving experienced users explicit
control.

---

## Lyrics Generation Layer

Lyrics generation is a first-class built-in feature, not an external API call.

```swift
public protocol LyricWriter: Sendable {
    var displayName: String { get }
    var requiresAPIKey: Bool { get }
    func write(prompt: String, style: String, duration: TimeInterval) async throws -> LyricDocument
}

public struct LyricDocument {
    public let lyrics: String        // [verse], [chorus] structure
    public let suggestedBPM: Int?
    public let suggestedKey: String?
    public let language: String
}
```

### Backends

| Backend | Default | Notes |
|---|---|---|
| `LocalLLMWriter` | Yes | Phi-3 Mini or Llama 3.2 3B via MLX-Swift, fully offline |
| `ClaudeWriter` | Optional | Anthropic API key in Keychain |
| `OpenAIWriter` | Optional | OpenAI API key in Keychain |
| `GeminiWriter` | Optional | Google API key in Keychain |

The local LLM weights download in the same first-launch flow as the music model weights.
If no API key is configured, the local model is always available as fallback. The lyrics
output is shown in an editable text view before generation so the user can tweak before
committing to a full run.

---

## Model Download & Management

### First-Launch Flow

A single setup screen before the main UI. One progress view, no options to overwhelm.

```
Welcome to Anvil

Downloading models (~X GB)
████████████░░░░░░  64%

ACE-Step DiT          ✓
ACE-Step VAE          ✓
Stable Audio          ✓
Lyric model (Phi-3)   Downloading…

This only happens once.
```

### Storage

Weights are saved to `~/Library/Application Support/Anvil/models/` — standard macOS app
data location, persists across app updates without re-downloading.

### `ModelSpec`

```swift
public struct ModelSpec {
    public let id: String
    public let huggingFaceRepo: String
    public let files: [String]
    public let sizeBytes: Int64
    public let sha256: [String: String]   // filename → expected hash
}
```

Downloads are verified by SHA256 against HuggingFace metadata. Interrupted downloads resume
via HTTP range requests. Adding a new model or checkpoint requires only a new `ModelSpec`
entry — no code changes.

### Model Swapping & Community Models

Users can download additional base model checkpoints (alternative ACE-Step variants, turbo
checkpoints, community fine-tunes) from a curated list or a direct HuggingFace URL. Switching
the active model is a single selection in the Inspector. Old weights are retained until the
user explicitly removes them.

### Updates

Non-blocking update banners appear in the UI when new model versions are available. Existing
weights stay in place and remain usable until the new download is fully verified.

---

## UI Architecture

### Main Window — Three-Panel Layout

```
┌──────────────┬─────────────────────────────┬──────────────┐
│   Sidebar    │         Canvas              │  Inspector   │
│              │                             │              │
│  Projects    │  [Auto | Music | FX | Amb]  │  BPM         │
│  History     │                             │  Key         │
│  Presets     │  Prompt input               │  Duration    │
│              │  ─────────────────          │  Seed        │
│              │  Lyrics editor (music only) │  Model       │
│              │  ─────────────────          │  Backend     │
│              │  Waveform / progress        │  LoRA        │
│              │                             │              │
│              │  [Write Lyrics]  [Generate] │              │
└──────────────┴─────────────────────────────┴──────────────┘
```

Standard macOS `NavigationSplitView`. Works with Stage Manager. Mode selector tabs sit at
the top of the Canvas — Auto is the default active tab.

### Key Components

| Component | Notes |
|---|---|
| `PromptEditor` | Multi-line text field, style tag chips |
| `LyricsEditor` | Editable, `[verse]`/`[chorus]` syntax tinting, hidden when mode is FX/Ambient |
| `WaveformView` | Live preview via `AVAudioEngine`, rendered with Metal/Canvas |
| `GenerationProgressView` | Step counter + cancel button during diffusion |
| `ModelStatusBar` | Persistent footer — active model, chip, memory pressure |

### Settings (⌘,)

Standard macOS Settings scene. Contains: API keys per backend, default model selection,
output folder, LoRA adapter management, UI preferences. Nothing buried in the main window.

---

## Error Handling

| Category | Examples | Handling |
|---|---|---|
| Setup errors | Download failed, checksum mismatch, disk full | Blocking — setup screen with recovery action |
| Generation errors | OOM during inference, corrupt weights | Non-blocking — error banner, state resets cleanly |
| API errors | Bad key, rate limit, no network | Inline in lyrics panel, prompt to switch to local model |

Cancellation is cooperative via `Task.cancel()` — stops at the next diffusion step, no
partial files written.

---

## Testing Strategy

`AnvilCore` as a Swift package enables full unit and integration testing without a running
app.

| Layer | Coverage |
|---|---|
| Unit tests | `ConditioningBuilder`, `LyricDocument` parsing, `ModelRegistry` checksum logic, all `LyricWriter` backends (mocked network), `IntentRouter` classification |
| Integration tests | Full pipeline run with synthetic weight fixtures (random weights, correct tensor shapes) |
| UI tests | First-launch flow, settings keychain storage, mode selector state, generate button transitions |

Real model weights are never committed. CI uses synthetic fixtures only.

---

## Open Questions

- Final app name (working title: Anvil)
- Which Stable Audio checkpoint ships as the default SoundFX model
- Whether `IntentRouter` classification confidence threshold is user-configurable
- Minimum hardware spec for the LoRA trainer (M2 Pro baseline TBC)
