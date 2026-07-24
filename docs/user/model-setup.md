# First Launch And Model Setup

PiMP Audio Studio generates and trains locally after the required model files
are downloaded. Manage models from:

```text
Settings -> Models
```

## Where Models Are Stored

The sandboxed app stores downloaded models under:

```text
~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/models/
```

You normally do not need to touch this folder. Use **Settings → Models** to
download, activate, remove, or refresh model status.

## Essential Models

First launch downloads these essentials:

| Model | Role |
| --- | --- |
| ACE-Step v1.5 Turbo | Default music DiT |
| ACE-Step VAE + Text Encoder | Shared VAE and Qwen text-encoder assets |

Without the shared VAE + text encoder component, music generation cannot run
even if a DiT shows as Ready.

## ACE-Step Music Models

### Classic ACE-Step v1.5 family

| Display name | Quality / speed note |
| --- | --- |
| ACE-Step v1.5 Turbo | Good / fastest |
| ACE-Step v1.5 Full | Better |
| ACE-Step v1.5 SFT | Best in the classic family |

### ACE-Step v1.5 XL family

| Display name | Quality / speed note |
| --- | --- |
| ACE-Step v1.5 XL Turbo | XL / fastest |
| ACE-Step v1.5 XL Base | XL / base — often useful for adapter training |
| ACE-Step v1.5 XL SFT | XL / best quality-focused XL |

Adapters are tied to the base model family they were trained against. Prefer the
same XL Base / XL SFT / XL Turbo variant used during training when auditioning
an adapter.

Default active music DiT after setup is Turbo.

## Shared Components

**ACE-Step VAE + Text Encoder** is required by ACE-Step music models. It includes:

- Audio VAE weights
- Shared Qwen text-encoder assets used by generation and training

If a music model is Ready but generation fails, confirm this shared component is
also Ready.

## ACE-Step LM Choices

The ACE-Step LM section includes:

- 0.6B
- 1.7B
- 4B

Cover generation can use LM-related paths when available. Larger LMs improve
capacity but use more memory and load more slowly. Cover workflows that only need
source-audio conditioning may not require every LM size to be present; download
what you need for your workflow.

## Stable Audio (SFX)

**Stable Audio Open 1.0** powers the short-form sound effects pipeline. It is
gated on Hugging Face and requires auth plus license acceptance.

Stable Audio is used when:

- Mode is Sound FX or Ambient, or
- Pipeline override is set to Stable Audio, or
- Auto routing classifies the prompt as SFX/ambient and Stable Audio is Ready

Stable Audio 3 is not the current app runtime.

## Dataset Annotation Packs (Advanced)

Under **Settings → Models → Dataset Annotation (Advanced)** you can install the
Native ACE captioner and transcriber packs used by advanced dataset enrichment.

- Downloads come from pinned PiMP Audio Studio Hugging Face packs.
- Each pack is checksum-verified, validated, and installed transactionally.
- Installed packs show Ready and can be deleted and re-downloaded.
- Both packs use about **9.34 GB** of disk combined.
- **24 GB** unified memory is the provisional minimum; **32 GB+** is recommended.

See [Datasets And Training](datasets-and-training.md).

## Intelligence / Lyric Local Model

A local on-device lyric / intelligence model can be managed from Settings →
Intelligence. Prompt enhancement and local lyric writing can use it when
available. Cloud providers are optional alternatives.

Supported stock models (Llama 3–compatible MLX only):

- **Llama 3.2 3B Instruct (4-bit)** — default, lighter
- **Llama 3.1 8B Instruct (4-bit)** — higher quality, more RAM

There is no custom Hugging Face add path for this runtime.

## Hugging Face Token

Some models are gated or require an accepted Hugging Face license. Save a token
in:

```text
Settings -> Models -> HuggingFace Access Token
```

The token is stored in macOS Keychain. It is **not** written into model folders,
remote training zips, traces, or diagnostics logs.

If a download fails with `401` or `403`:

1. Confirm the token is saved.
2. Confirm the token is still valid.
3. Open the model page on Hugging Face and accept the model license.
4. Retry the download.

## Memory Controls

Settings → Models includes memory-related actions such as releasing GPU / MLX
weights when idle. Use **Release GPU Memory** (or the equivalent control) before
heavy training or enrichment if you just finished generating.

MCP clients can call the same cleanup path through `release_memory` when the
local MCP server is enabled.

## What Not To Bundle

Public app builds should not include giant or gated model weights inside the
`.app` bundle. Keep the app small and let users download model files through
first-launch or Settings.

## Quick Health Check

Before troubleshooting generation, confirm:

- Selected ACE-Step DiT says **Ready**.
- **ACE-Step VAE + Text Encoder** says **Ready**.
- Selected ACE-Step LM says **Ready** if your cover/LM workflow needs it.
- **Stable Audio Open** says **Ready** if you are generating SFX.
- Native ACE packs say **Ready** if you selected Native ACE enrichment.
- Any gated model has an accepted license on Hugging Face.
