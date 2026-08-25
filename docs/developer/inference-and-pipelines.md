# Inference And Pipelines

## Model Categories

Defined in `ModelSpec` / `ModelCategory`:

| Category | Role |
| --- | --- |
| `aceStepDiT` | Music diffusion transformer variants |
| `aceStepVAE` | Shared VAE + Qwen text encoder assets |
| `aceStepLM` | Optional LM sizes for experimental/cover-related paths |
| `stableAudio` | Stable Audio Open SFX stack |

Local lyric / intelligence models are **not** `ModelRegistry` entries. They
live under Settings → Intelligence (`LLMModelManager`: Llama 3.2 3B / Llama 3.1
8B MLX 4-bit).

Built-in specs live in `ModelRegistry.builtInSpecs`. Essential first-launch IDs:

- `ace-step-dit-turbo`
- `ace-step-vae`

Default active DiT is Turbo.

## Music Pipeline (ACE-Step)

Primary types under `PiMPAudioCore/Inference/`:

| Area | Components |
| --- | --- |
| Conditioning | `ConditioningActor`, `ConditionEncoder`, lyric encoder, Qwen3 text path |
| DiT | `DiTModel`, `DiTActor`, attention components, cross-attn cache |
| Diffusion | `DiffusionActor`, `DiffusionSampler`, timestep schedules |
| VAE | `VAEModel`, `VAEActor`, tiled decode helpers |
| Tokenizer | Audio tokenizer/detokenizer for cover/code paths |
| Orchestration | `MusicPipeline` |

App entry: `ACEStepGenerationService` / `GenerationService` builds a
`GenerationRequest` from the UI (prompt, lyrics, duration, seed, guidance
preset, precision, steps, shift, cover strength, pipeline override, adapter
stack).

### Guidance presets

Implemented in app `GuidancePreset`:

| Preset | CFG interval | Scale override |
| --- | --- | --- |
| Constant | full schedule | none |
| Late Emphasis | `t <= 0.5` | none |
| Mid-to-Late | `t <= 0.75` | none |
| Soft Late | `t <= 0.6` | force `6.0` |

XL recipe: Soft Late auto-nudges to Constant when XL DiT/adapters become active.

### Steps defaults

When steps override is blank:

- Turbo ≈ 8 steps
- Base / SFT / XL ≈ 50 steps

### Adapters at inference

`LoRACoordinator` / injectors apply LoRA or LoKR contributions with per-adapter
weights and per-adapter alpha scaling. Effect diagnostics can log
`adapter/effect` deltas when verbose AUDIODIAG is enabled.

## SFX Pipeline (Stable Audio Open)

Under `Inference/StableAudio/`:

- T5 text encoder
- Projection model into DiT conditioning space
- SFX DiT + diffusion actors
- SFX VAE encode/decode
- `SoundFXPipeline` orchestration

App entry: `SFXGenerationService` when mode/pipeline resolves to Stable Audio.

Requires auth-gated Hugging Face download (`requiresAuth: true` on the spec).

## Cover Generation

Generate view imports source audio into the app container, then submits a
request with `referenceAudioURL` and `coverStrength`.

Important implementation details:

- Source trimmed to requested duration before VAE encode
- Timbre reference window limited to avoid double full-length encodes
- Short sources pad with learned `silence_latent.pt`
- VAE posterior draws are seed-derived for reproducibility

## Editor Latent Path

Under `Inference/Editor/`:

| Type | Role |
| --- | --- |
| `MaskBuilder` | Time/frequency masks (ACE-Step hop 1920) |
| `TimestepMapper` | Aligns strength to truncated schedule |
| `LatentBlender` | Blends regenerated region with context |
| `InpaintCoordinator` | End-to-end inpaint/repaint |
| `EditHistoryManager` | Undo stack for latent/audio edits |

Noise injection must use the first timestep of the selected schedule, not the
raw strength slider value.

## Diagnostics

`AudioDiagnostics` writes `[AUDIODIAG]` lines when verbose mode is enabled. Log
directory is set at app launch to the sandboxed Diagnostics folder.

## Adding A Model Variant

1. Add a `ModelSpec` with files, sizes, category, and auth flags.
2. Ensure weight name maps exist under `Inference/Weights/`.
3. Wire active-model selection UI in Models settings if user-visible.
4. Add readiness checks in generation/training services.
5. Document in `docs/user/model-setup.md` and CHANGELOG.
