# Getting Started

PiMP Audio Studio is a native macOS app for local music generation, sound
effects, latent editing, dataset packaging, and adapter training on Apple
Silicon. Optional remote GPU training can run on user-approved cloud hardware
such as RunPod.

## Requirements

- macOS 15.0 or newer.
- Apple Silicon Mac for local MLX generation and training.
- Enough free disk for models (several GB for essentials; tens of GB if you
  download XL models, Stable Audio, annotation packs, and datasets).
- Network access for first-time model downloads, optional provider CLIs,
  optional Hugging Face imports, and optional remote GPU workflows.

Memory guidance:

- **16 GB** unified memory can run lighter generation paths, but is tight.
- **24 GB+** is more comfortable for XL models and dataset enrichment.
- **32 GB+** is recommended for serious local XL adapter training and Native ACE
  annotation packs.

## First Launch

On first launch the app shows a welcome screen and downloads the essential
models required for basic music generation:

- ACE-Step v1.5 Turbo (music DiT)
- ACE-Step VAE + Text Encoder (shared component)

Click **Download Models**. This only needs to happen once. Additional models
such as Stable Audio Open and the local lyric LLM can download later in the
background.

You can also download, activate, or remove models later from:

```text
Settings -> Models
```

See [First Launch And Model Setup](model-setup.md).

## Choose An Output Directory

Before the main workspace opens, the app asks for an output directory. This is
where generations, history, training datasets, and related packages are stored.

Pick a folder you control, with enough free space for WAV files and training
packages. You can change the directory later through the app's output directory
controls.

Typical contents under the output folder include:

- generation history audio and traces
- `TrainingDatasets/` packages
- training and remote job exports

## Main Window Layout

### Create

- **Generate** — write prompts and lyrics, choose mode, apply adapters, generate
  music or SFX, open the Generation Analyzer.
- **Edit** — load audio, inpaint or repaint regions, apply DSP effects, export.

### Build

- **Datasets** — build packages from local files or YouTube, review captions and
  lyrics, enrich packages, open training.
- **Training** — local LoRA, local LoKR, or Remote GPU workflows.

### Library

- **History** — previous generations, playback, select/compare/export/delete,
  open traces, train from selected takes.
- **Presets** — saved generation settings from the Generate page.

### Inspector

The right-hand inspector exposes:

- Active pipeline (Auto / ACE-Step / Stable Audio when SFX models are ready)
- Duration
- Seed (blank = random)
- Guidance scale and forced overrides from guidance presets
- Active model summary

## Recommended First Hour

1. Finish essential model download.
2. Confirm output directory is set.
3. Open **Generate**, leave mode on Auto or Music, enter a short style prompt,
   and generate a 15–30 second clip.
4. Play the result from History.
5. Open **Analyzer** during or after a generation to inspect the live trace.
6. Optionally open **Edit** from a take and try a mild DSP fade or normalize.
7. Later: build a small dataset, train a Narrow LoRA, and A/B the adapter.

## Optional Features

These are powerful but not required on day one:

- Cover generation from a source audio file
- Adapter stacking and community adapters
- Native ACE dataset annotation packs
- Remote GPU training on RunPod
- Local MCP server for agent automation
- Provider lyric writers (Claude, Codex, Gemini, Grok, OpenAI-style APIs)

## Next Steps

- [Generating Music](generating.md)
- [Sound Effects](sound-effects.md)
- [Datasets And Training](datasets-and-training.md)
- [Troubleshooting](troubleshooting.md)
