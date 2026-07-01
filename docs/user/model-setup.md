# First Launch And Model Setup

PiMP Audio Studio can generate audio locally after the required model files are
downloaded. Model downloads are managed from:

```text
Settings -> Models
```

## Where Models Are Stored

The sandboxed app stores downloaded models under:

```text
~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/models/
```

You normally do not need to touch this folder. Use `Settings -> Models` to
download, activate, remove, or refresh model status.

## Core Music Models

The ACE-Step music pipeline uses a DiT music model plus shared components.

Current first-class ACE-Step XL choices:

- `XL SFT` - quality-focused instruction-tuned XL model.
- `XL Base` - broad base XL model, often useful for adapter training/testing.
- `XL Turbo` - distilled XL model for faster generation.

The model picker also keeps older/smaller ACE-Step options where supported.

## Shared Components

The `ACE-Step VAE + Text Encoder` shared component is required by the ACE-Step
music models. It includes the audio VAE path and the shared Qwen text-encoder
assets used by generation and training.

If a music model is downloaded but generation still fails, confirm the shared
component is also `Ready`.

## ACE-Step LM Choices

The ACE-Step LM section currently includes:

- `0.6B`
- `1.7B`
- `4B`

Cover-generation LM loading tries the available ACE-Step LM choices in this
order:

```text
0.6B -> 1.7B -> 4B
```

Larger LMs can improve capacity, but they use more memory and load more slowly.

## Stable Audio

`Stable Audio Open` is the current SFX pipeline model family in the app. It is
used for sound effects and short-form SFX-style prompts when that pipeline is
selected or auto-routed.

Stable Audio 3 is research/radar material for now. It is not the current app
runtime.

## Hugging Face Token

Some models are gated or require an accepted Hugging Face license. Save a token
in:

```text
Settings -> Models -> HuggingFace Access Token
```

The token is stored in macOS Keychain. It is not written into model folders,
remote training zips, traces, or diagnostics logs.

If a download fails with `401` or `403`:

1. Confirm the token is saved.
2. Confirm the token is still valid.
3. Open the model page on Hugging Face and accept the model license.
4. Retry the download.

## What Not To Bundle

Public app builds should not include giant or gated model weights inside the
`.app` bundle. Keep the app small and let users download model files through the
first-launch or Settings flow.

## Quick Health Check

Before troubleshooting generation, confirm:

- The selected ACE-Step DiT says `Ready`.
- `ACE-Step VAE + Text Encoder` says `Ready`.
- The selected ACE-Step LM says `Ready` if you are using cover generation.
- `Stable Audio Open` says `Ready` if you are using SFX generation.
- Any gated model has an accepted license on Hugging Face.
