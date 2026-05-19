# ACE-Step Audio Quality Investigation — Resumption Notes

**Last updated:** 2026-05-19 (guidance preset + prompt metadata + ending investigation pass)
**Status:** Pipeline runs end-to-end and now produces **recognizable music** for the first time. User-reported after latest fixes: "Holy shit! I heard music for the first!! Albeit not very good music but music none the less!!" Quality is still poor, but the structural failure mode has changed from static/noise/"trying" to actual music generation. Latest diagnostic signature for Turbo breakthrough: real lyric header hidden states, one silence-derived timbre token, upstream v1.5 Turbo timestep schedule ending at 0.3, final latent abs_mean ≈ 1.03 / abs_max ≈ 7.44, VAE audio abs_mean ≈ 0.155 / abs_max ≈ 1.98. XL SFT now loads and runs with config-driven DiT architecture; first seed-42 sparse-prompt output produced a sustained tone/drone rather than a good riff, so XL is alive but still needs prompt/sampling/parity A/Bs.

**Codex note (2026-05-19):** A first listening A/B of CFG interval presets
favored late guidance over constant guidance, with `Soft Late` becoming the
default. The same pass added SFT prompt metadata extraction (`bpm`,
`timesignature`, `keyscale`) and then disabled WAVWriter's automatic adaptive
tail fade for generated output. Upstream ACE-Step only applies fade when
explicitly requested, so generated WAVs should stay byte-faithful to the VAE
decode while investigating the recurring ~28-second ending in 30-second clips.

**Codex note (2026-05-19, follow-up):** The next conditioning mismatch found
after the no-fade retest was `chunk_mask_mode`. Upstream defaults
`GenerationParams.chunk_mask_mode` to `"auto"` and converts the full-generation
mask to value `2.0` in `conditioning_masks.py`; Swift had been sending `1.0`.
Swift now uses `2.0` for full-generation context masks and logs
`AUDIODIAG cond/chunkMask mode=auto value=2.0`.

> Use this doc to resume troubleshooting after a context reset. Skim "Current symptom" + "Hypotheses still open" first.

> **Quick orientation for the next session:** the runtime `AudioDiagnostics` toggle (Generate → Advanced → "Verbose audio diagnostics") emits `[AUDIODIAG]` lines at every pipeline boundary. That instrumentation cracked the structural bugs in this session. Filter the Xcode console for `AUDIODIAG`.

> **External review valuable:** GPT-5.5 read this doc + source twice during this session and identified TWO real bugs I had missed (and corrected one I'd misdiagnosed). When you hit a wall, consider feeding this doc + the relevant code to a parallel model.

---

## Project context

Porting **ACE-Step v1.5** (audio diffusion model — text→music) from PyTorch+upstream-MLX into a native Swift + MLX-Swift macOS app (`AnvilApp`). Pipeline:

```
text prompt
  → Qwen3 text encoder + LyricEncoder + TimbreEncoder
  → ConditionEncoder fuses → encoder_states
  → cross-attention into DiT (rectified-flow diffusion, 8 or 50 steps)
  → AutoEncoderOobleck VAE decodes latents
  → 48 kHz stereo WAV
```

Upstream reference repos:
- Local Python source: `~/Developer/GitRepos/ACE-Step-1.5`
- Most relevant upstream MLX code: `acestep/models/mlx/dit_generate.py`, `acestep/models/mlx/dit_model.py`, `acestep/models/mlx/vae_model.py`
- Upstream PyTorch DiT: `acestep/models/turbo/modeling_acestep_v15_turbo.py`, `acestep/models/base/modeling_acestep_v15_base.py`, `acestep/models/sft/modeling_acestep_v15_base.py`

## Current symptom (post-2026-05-07-evening breakthrough)

Generation completes in ~5–60 s. Output WAV file:

- **30.0 s, 48000 Hz, stereo Int16, ~5.76 MB** ✓
- VAE output latest **abs_max ≈ 1.98** before WAVWriter peak-normalize (higher than previous "trying" run, but now audibly music instead of underpowered texture)
- **Different prompts produce meaningfully different audio** — SHA1-diff verified, plus RMS-difference verified ("loud heavy metal" 7 dB louder than "slow dark guitar")
- **Velocity prediction healthy throughout trajectory**: latest step 0 abs_mean ≈ 0.72, growing to step 7 abs_mean ≈ 1.20, abs_max ≈ 9.42 — model takes increasingly confident corrections through the schedule
- **Per-second RMS envelope shows real musical dynamics** — slow ramp 0–8s (RMS 600–960), build 9–11s (1414–1861), drop 12–13s (445–611), build back 14–28s (587–1809). User-reported pattern matches numerics exactly.
- **ZCR in music range** (sustained-tonal 270–580 in opening, 2500–3400 mid, 1000–2300 late). Not noise.
- **L/R Pearson 0.778** — typical music range (0.5–0.9).
- **Step count alone did not fix the original static/noise failure.** Pre-breakthrough user testing at 8 / 32 / 100 steps still plateaued before the lyric/timbre fixes. Re-test Base/SFT at their auto 50-step path now that conditioning is alive.
- **Fresh app restarts defaulted back to Float16 before the cursory pass.** The first recognizable-music diagnostics were Float32, so the app now defaults GenerateView and GenerationRequest to Float32 for the quality baseline. Float16 remains useful as a later performance A/B.

User listening after the latest lyric/timbre/schedule fixes (verbatim, 2026-05-07 evening): *"Holy shit! I heard music for the first!! Albeit not very good music but music none the less!!"*

Interpretation: the model has crossed the structural threshold into recognizable music. Remaining work is a **quality/parity pass**, not a "why is everything static" investigation.

### Earlier symptoms (now resolved)

- **Pure static across all DiT variants** (start of evening session): step0/vt was nearly identical to init/xt (`vt - xt ≈ tiny perturbation`), meaning the model predicted x0 ≈ 0 regardless of prompt → diffusion converged to silence-shaped noise → VAE decoded to static.
- **"Different prompts but still all noise"** (mid-session): cross-attention was alive (RMS differs per prompt) but positionally / scale-wise wrong; output had loudness-following but no structure.
- **"Trying to be music with static layer on top"** (after silence_latent fix): user heard prompt content underneath but the noise floor was high. silence_latent eliminated the noise floor entirely; the "trying" is now exposed.
- **"Trying to be music but not music"** (after timbre/schedule parity): caused by the stale empty-lyrics bypass. We packed lyric tokens first, but empty lyrics were replaced with a `[1, 11, 2048]` zero hidden tensor and marked valid. Cross-attn ignores the mask, so the first attended conditioning token was all zeros plus downstream projection bias. Removing that bypass and encoding the real upstream lyric scaffold made music appear.

## What's verified working (don't re-investigate these)

### Weight loading (100% match for everything that has weights in v1.5 checkpoint)

Diagnostic logs in `DiTActor.loadWeights`, `VAEActor.loadWeights`, `ConditioningActor.loadWeights` print `supplied=N model=M matched=K` plus first 8 mismatches each side. Recent run:

| Module | Supplied | Model | Matched | Notes |
|---|---|---|---|---|
| DiT | 476 | 477 | 476 | Only `rotaryEmb.invFreq` unsupplied (runtime-computed) |
| Qwen3 | 310 | 311 | 310 | Same — `rotaryEmbedding.invFreq` |
| LyricEncoder | 91 | 92 | 91 | Same |
| TimbreEncoder | 47 | 48 | 47 | Only `rotaryEmbedding.invFreq` unsupplied (runtime-computed). The old Swift-only 5-query `AttentionPooler` was removed because upstream v1.5 does not use it and the checkpoint has no weights for it. |
| ConditionEncoder | 1 | 1 | 1 | `text_projector.weight` → `textProjection.weight` |
| VAE | 291 | 291 | 291 | All match |

### Healthy intermediate values

`DiffusionActor` final-latent diagnostic now shows abs_mean ≈ 1.03, abs_max ≈ 7.44 on the first recognizable-music Turbo run. This is higher than the previous "trying" run (≈0.50/2.98) and corresponds to audibly stronger musical content.

`VAEActor.decode` diagnostic on the breakthrough run shows audio output abs_mean ≈ 0.155, abs_max ≈ 1.98. WAVWriter peak-normalizes before Int16 output, so this no longer hard-clips.

### Architecture verified against upstream

- **DiT config** matches `config.json`: 24 layers, hidden_size=2048, intermediate=6144, 16 attention heads, 8 KV heads, head_dim=128, rms_norm_eps=1e-6, attention_bias=false, patch_size=2, sliding_window=128, rope_theta=1e6, audio_acoustic_hidden_dim=64, in_channels=192, layer_types alternating sliding/full
- **VAE config** matches `vae/config.json`: 48000 Hz sampling, downsampling_ratios `[2,4,4,6,10]` (total stride 1920), audio_channels=2, decoder_input_channels=64, encoder_hidden_size=128
- **Latent frame rate** matches: 48000 / 1920 = 25 fps. Both `contextLatents` and `noisyLatents` shaped `[1, 750, ...]` for 30s
- **APG implementation** byte-for-byte identical to upstream's `_mlx_apg_forward` in `acestep/models/mlx/dit_generate.py:86`: same momentum (no decay, additive accumulation), `axis=1` projection, same orthogonal-component-only formula
- **TimestepEmbedding**: scale=1000, max_period=10000, `[cos, sin]` order, silu→linear→silu→linear→silu→time_proj→reshape — matches upstream
- **scale_shift_table modulation**: `norm(x) * (1 + scale) + shift` before self-attn → `+attn*gate` residual → plain norm before cross-attn → `+cross-attn` residual (no gate or modulation on cross-attn) → `norm(x) * (1 + c_scale) + c_shift` before MLP → `+mlp*c_gate` residual. Matches upstream exactly.
- **DiT key mapping** (`mapDiTKey`): snake_case→camelCase, `decoder.` prefix strip, `proj_in.1.→proj_in.`, skip rotary_emb buffers, special-case top-level `null_condition_emb`. Verified all 476 weights resolve to real model parameter paths.
- **VAE key mapping** (`VAENameMap.mapKey`): block→blocks, snake→camel, fuse weight_g/weight_v. All 291 resolve.

## Fixes shipped (in roughly chronological order, all on `origin/main`)

Each commit message has full reasoning. Key ones for the audio path:

| Commit | Title (paraphrased) | Why |
|---|---|---|
| `f736bc2` | WAVWriter NLC stereo handling | VAE returns `[B, length, 2]`; we treated it as 1-D mono → 2× duration, alternating L/R noise |
| `07993fa` | VAE 48 kHz / `[2,4,4,6,10]` strides | Defaults were `[2,4,4,8,8]` at 44100 — wrong VAE architecture for v1.5 |
| `c6ab178` | WAVWriter peak-normalize before clip | VAE output abs_max > 1.0; hard-clip distorted audibly. Now scales by `1/abs_max` only when peak > 1.0 |
| `992be2a` | Auto step count by DiT variant | `DiffusionConfig.steps=nil` → fallback to 8-step schedule for ALL variants. Turbo→8, Full/SFT→50 |
| `d88a625` | DiT snake_case→camelCase weight key map | `mapDiTKey` only stripped `proj_in.1.`. 471 of 476 weights weren't loading |
| `6c99706` | VAE block→blocks + snake→camel weight key map | Only 11/291 VAE weights loaded before this |
| `0421733` | Wire `ConditioningActor.loadWeights` | Defined but never called — Qwen3 + lyric + timbre + condition encoders ran on default-init random weights |
| `ced3c55` | bias=true on lyric/timbre input projections, skip mismatched timbre special_token | v1.5 checkpoint provides bias for these but Linear was bias=false → load crashed |
| `fc483c9` | TimbreEncoder bypass when no reference audio | TimbreEncoder.AttentionPooler stays default-init random; running zeros through it produces 5 random tokens that polluted cross-attn. Now passes `[1, 5, 2048]` zeros directly |
| `1bc9a1c` | Drop cross-attn encoder mask | Upstream MLX hardcodes `encoder_attention_mask=None`; upstream Python shadows the param to None and rebuilds shape-only mask. Model trained without padding mask in cross-attn — applying one shifted distribution away from training |
| `9bb523b` | Bypass LyricEncoder when lyrics empty | Without cross-attn mask, LyricEncoder's 2048 PAD-input tokens get attended at full weight, drowning ~5–10 real text tokens 8:1 |
| `e8f7fe0` | shift = 3.0 | The MLX-specific upstream handler defaults shift=3.0; the 1.0 in `inference.py` is for the PyTorch backend. Tested both — same result either way |
| `34591d5` | **Use learned `null_condition_emb` for CFG unconditional branch** | We used `MLXArray.zeros(like: encHS)`. Upstream broadcasts a LEARNED `[1, 1, 2048]` token (top-level safetensors key) across encoder length. All-zeros is OOD for trained DiT; CFG `(cond - uncond)` was just adding noise on top of cond instead of isolating prompt direction. **User reported "a little bit better" after this fix** but still not matching prompt |

### 2026-05-07 afternoon/evening — instrumentation + structural bugs + course corrections

| Commit | Title | Why / impact |
|---|---|---|
| `5c82b54` | Runtime `AudioDiagnostics` | New enum gated by UserDefaults `audioDebugVerbose`. Logs `[AUDIODIAG]` at every boundary (cond outputs, init noise, schedule, per-step vt + xt, post-APG velocity, VAE in/out). UI toggle in Generate → Advanced. **Cracked four of the bugs below.** |
| `5faf163` | **Timestep schedule** trajectory tail | Used `1 - i/N` (stops at 1/N ≈ 0.125 for N=8); upstream uses `linspace(1.0, 1/num_train_timesteps, N)` (stops at ~0.001). Diffusion walked only 70% of trajectory; finalStep leaped the rest in one Euler step → noise. Now matches `scheduling_flow_match_euler_discrete.py`. |
| `89c79da` → `3629cec` | Cross-attn pre-norm REMOVED then RESTORED | I read public `attention.py:308-315` (raw hidden_states into cross_attn). **GPT-5.5 caught:** safetensors has 24 `cross_attn_norm.weight` tensors, and v1.5-specific source (`models/turbo/modeling_acestep_v15_turbo.py:512`, `models/mlx/dit_model.py:332`) DOES apply norm. Public attention.py is a newer refactor. **Reverted.** |
| `a1a22a2` → `4920923` | Cross-attn RoPE ADDED then REMOVED | I read public `customer_attention_processor.py:353-358` which rotates both Q and K. **GPT-5.5 caught:** local `models/mlx/dit_model.py:196-222` does NOT apply RoPE in cross-attn at all. Same public-vs-v1.5 divergence. **Reverted.** |
| `ae66c24` | **Tokenizer longest-padding** | `AnvilTokenizer.encode` always padded to fixed maxLength with zeros. Combined with our (correct) decision to skip cross-attn encoder mask, we fed 2298 padding-derived "ghost tokens" through cross-attn (conditionEmbedder.bias makes them non-inert). Upstream uses `padding="longest"` with maxLength as truncation cap (`conditioning_text.py:128-142`). Result: encoderStates `[1, 2309, 2048]` → `[1, ~10, 2048]` for typical short prompts. |
| `66c3162` | **SFT_GEN_PROMPT template wrap** | Tokenizing raw user prompt — DiT was trained on `# Instruction / # Caption / # Metas / <\|endoftext\|>` scaffold (`acestep/constants.py:SFT_GEN_PROMPT`) plus `# Languages / # Lyric / <\|endoftext\|>` for lyrics (`prompt_utils.py:_format_lyrics`). Now wraps both. Lyrics default to `en`, metas all N/A except duration. encoderStates seq grew ~10 → ~73 (longer wrapped prompt); abs_mean 1.5 → 2.4. |
| `461b198` | chunk_mask = ones, not zeros (smoke test) | Upstream text2music: `context_latents = cat([silence_latent_slice, chunk_mask_ones], axis=-1)` (`models/turbo/modeling_acestep_v15_turbo.py:1680, 1931`). We had all zeros. Smoke test flipped mask half to ones. contextLatents abs_mean 0 → 0.5; vt step 7 abs_mean 0.57 → 1.53 (model now confidently shaping output near t=0). |
| `9757bac` | **`silence_latent.pt` loader for text2music context body** | Final piece of `cat([silence_latent_slice, ones])`. New `SilenceLatentLoader.swift` reads the `.pt` file via `/usr/bin/unzip -p` (no third-party zip lib, no pickle parsing), extracts raw float32 at `*/data/0`, returns `[1, 15000, 64]` (transposed from stored `[1, 64, 15000]`). `ConditioningActor.silenceLatent` field; encode() slices `[:, :tFrames, :]` (tiles if duration > 600s). `ACEStepGenerationService` loads it next to model.safetensors. ModelRegistry: added to all 3 DiT specs (3.84 MB each, auto-downloads). Result: contextLatents `abs_mean=0.787, abs_max=6.73` — matches GPT-5.5's predicted `~0.78 / ~6.7`. VAE output `abs_max=0.30` (was 1.7 → 1.10 → 0.80 → 0.30 across the session). User: "starts off slow, builds up, drops down slow, builds back up. It's like it's trying." |
| `2a010ef` | **TimbreEncoder v1.5 parity** | Removed the Swift-only 5-query `AttentionPooler`. Upstream v1.5 `AceStepTimbreEncoder.forward` comments out prepending `special_token`, runs 4 Qwen layers over acoustic latents, applies norm, then takes `hidden_states[:, 0, :]` and unpacks it. Checkpoint has no weights for the old attention pooler; after removal, timbre load is clean: supplied=47, model=48, matched=47, only `rotaryEmbedding.invFreq` missing. |
| `2a010ef` | **Text2music timbre uses silence_latent** | For no reference audio, Swift now feeds the first 750 frames of learned `silence_latent` through `TimbreEncoder`, matching upstream `infer_refer_latent()` behavior. Diagnostic: `cond/timbreTokens shape=[1, 1, 2048] abs_mean=0.293 abs_max=11.59`. |
| `2a010ef` | **ConditionEncoder packing order** | Swift now packs conditioning as upstream does: `pack_sequences(lyrics, timbre)` then `pack_sequences(result, text)`, equivalent here to `[lyricHidden, timbreTokens, textProjected]` after longest-padding. Previous order was `[text, lyric, timbre]`. |
| `2a010ef` | **Turbo timestep schedule parity correction** | Reverted the FlowMatch tail-to-0.003 path for `inferSteps`. The checked-in v1.5 Turbo generation path uses fixed/valid 8-step schedules; for shift=3.0 it ends at `0.3`: `[1.0, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3]`. This alone did NOT make music, but it restores local v1.5 parity. |
| `2a010ef` | **Breakthrough: remove empty-lyrics zero bypass** | After packing lyrics first, the stale bypass became fatal: empty lyrics produced a valid `[1, 11]` scaffold token sequence, but Swift replaced its hidden states with all zeros while marking the tokens valid. Cross-attn ignores masks, so `encoderStates first5` became all zeros. Upstream always embeds and runs the lyric encoder over `# Languages\nen\n\n# Lyric\n<\|endoftext\|>`. Removing the bypass and defaulting lyric language to `en` produced the first recognizable music. Breakthrough diagnostics: `cond/lyricHidden shape=[1, 11, 2048] abs_mean=0.0555`, `cond/encoderStates first5` non-zero, final latent abs_mean=1.03, VAE abs_max=1.98. |
| local cursory pass | **Float32 as fresh-launch quality baseline** | GenerateView and GenerationRequest now default to Float32. The first musical outputs were tested with `precision=float32`; after restart the UI previously reset to Float16, making SFT/Base comparisons noisier than necessary. |
| local cursory pass | **LoRA variant guard** | LoRA selection persists in AppStorage and the picker filters by broad `aceStepDiT`, not exact variant. `LoRACoordinator` now refuses to inject an adapter unless `adapter.baseModelID == active DiT id`, avoiding accidental Turbo/Base/SFT cross-contamination. |
| `c03c4e0` → `d71364b` | VAE scale_factor (REVERTED) | Applied `latents / 0.1786 + (-1.9091)` based on upstream `music_dcae_pipeline.py:116`. Made things much worse (abs_max 1.7 → 7.3). **Misdiagnosis:** our VAE is `AutoencoderOobleck` (Stability AI's, per `vae/config.json:_class_name`), NOT MusicDCAE. Those constants belong to MusicDCAE's wrapper. Reverted. |

### 2026-05-07 late night — XL/custom model/runtime pass

| Area | What changed | Why / result |
|---|---|---|
| Lyric writer duration awareness | All lyric writer backends now share `LyricWritingPlan`: 30s clips request 4–6 total lines, no bridge/outro/full-song structure, and much smaller token budgets. | The local Llama generated ~2 minutes of lyrics for a 30s song; shorter lyrics sounded much better. This keeps ACE-Step from being asked to cram too much lyric content into short clips. |
| Custom DiT validator/downloader | `AddModelView` now accepts either single `model.safetensors` or sharded `model.safetensors.index.json` + shard files. Downloads now surface progress immediately and refresh the Models list as soon as the custom spec is saved. | `ACE-Step/acestep-v15-xl-sft` failed validation because it is sharded and has no flat `model.safetensors`. The repo now validates/downloads and appears under an `ACE-Step v1.5 XL` group. |
| Sharded DiT loading | `WeightConverter.loadSafetensors(from urls:)` and `ACEStepGenerationService.ditWeightPaths` load all `model*.safetensors` files listed in a DiT spec. | Required for any sharded custom DiT, including XL. |
| Runtime support metadata | `ModelSpec` gained `isRunnable` / `runtimeNote`, with backward-compatible decoding. Registry refuses to activate unsupported specs, and ready-but-unsupported rows hide `Set Active`. | Avoids letting downloadable-but-not-wired custom checkpoints crash generation. |
| Config-driven DiT | New `DiTConfig` reads `config.json`; `DiTDecoder(config:)` now supports variable hidden size, layer count, attention heads, KV heads, intermediate size, layer types, rope theta, patch size, etc. | XL SFT declares hidden_size=2560, intermediate_size=9728, 32 layers, 32 heads, 8 KV heads. Hardcoded 2048/24/16 runtime could never run it correctly. |
| 2048→2560 conditioning bridge | `DiTDecoder.conditionEmbedder` is now `Linear(encoderHiddenSize, hiddenSize)` instead of `Linear(hiddenSize, hiddenSize)`. `nullConditionEmb` remains encoder-width. | XL checkpoint has `decoder.condition_embedder.weight [2560,2048]` and `null_condition_emb [1,1,2048]`. This matches the shared 2048-wide VAE/Qwen/ConditionEncoder path without needing a separate XL VAE/text bundle. |
| XL first load diagnostic | XL run logged `DiTActor supplied=629 model=630 matched=629`, only `rotaryEmb.invFreq` missing. Conditioning and VAE loads remained clean. Schedule auto-resolved to 50 steps. | Confirms XL weights are structurally loading and the new bridge matches checkpoint shapes. |
| XL first audio result | Sparse prompt `slow dark guitar riff`, seed 42, 30s, auto 50 steps, guidance 7.5, shift 3.0 produced a dense sustained tone/drone with limited musical evolution. | XL runtime is alive but not yet quality-parity-proven. Sparse prompts may collapse to soundscape/drone; prompt enhancer produced a much richer brief. Next A/B should use enriched riff-forward prompt, seed 42, guidance 7.0 vs 8.0. |
| Release GPU Memory beachball | `ACEStepGenerationService` is now an actor instead of `@MainActor`; Settings release button disables itself and shows a spinner while unloading. | Clicking Release after loading XL beachballed the app under Xcode/debugserver and left a 7 GB process. Moving ACE-Step coordination off the UI actor should reduce main-thread stalls, though Metal/MLX may still pause briefly while freeing huge buffers. |

## Hypotheses still open (post-2026-05-07 music breakthrough)

Ranked by current likelihood:

### 1. Quality/parity pass beyond conditioning

The core conditioning path is now alive enough to produce recognizable music. Remaining poor quality likely lives in one of these lower-level parity surfaces:

- DiT patch embed / unembed (`projIn` / `projOut`) channel ordering and patch-size reversal
- VAE decode parity against upstream `AutoencoderOobleck` for the same latent tensor
- Self-attention details: grouped-query repeat direction, q/k norm placement, sliding/full attention mask parity
- Final latent distribution / VAE expected scale. Breakthrough latent abs_mean≈1.03, abs_max≈7.44; VAE abs_max≈1.98 before WAV normalization. This is musically stronger than previous runs but may still be off-distribution.

**Best next move:** create a cross-runtime parity harness for one small stage at a time, starting with VAE decode on a known upstream latent or DiT output on a tiny fixed tensor. Audio is now too subjective for blind guessing.

### 2. Prompt/meta quality tuning

The structural prompt wrapper is fixed, and Swift now performs a conservative
first-pass metadata extraction for `bpm`, `timesignature`, and `keyscale`
before building the upstream SFT meta block. Explicit prompts such as
`70 BPM`, `4/4`, and `A minor` should show up in `AUDIODIAG
cond/promptMetadata`. Remaining quality work should move these fields into a
proper prompt-enhancer schema/UI instead of relying only on regex heuristics.

### 3. Guidance / schedule variant A-Bs

The local v1.5 Turbo path uses the valid 8-step schedule ending at 0.3. The earlier FlowMatch tail-to-0.003 schedule produced weaker/less musical output once other conditioning bugs were fixed. Still, quality may depend on CFG interval or guidance scale. Now that music exists, small A/Bs are meaningful:

- guidance 5.0 / 7.5 / 9.5
- shift 1 / 2 / 3 using valid Turbo timesteps
- SFT/full model 50-step path after applying the same conditioning fixes

### 4. Remaining conditioning details

The biggest conditioning bugs are fixed, but still worth verifying:

- `wrapPromptForSFT` exact whitespace vs upstream `SFT_GEN_PROMPT.format(...)`
- lyric language default is now `en`; verify UI/user settings do not override it to `unknown` unexpectedly
- empty text + empty lyrics baseline should generate a much weaker/less directed output than a real prompt

### Resolved during this pass

- Timbre encoder bypass: fixed. Text2music now feeds `silence_latent[:, :750, :]` through `TimbreEncoder`.
- Swift-only 5-query timbre pooler: removed. It had no checkpoint weights and is not used by v1.5.
- ConditionEncoder order: fixed to `lyrics + timbre + text`.
- Empty-lyrics bypass: fixed. This was the breakthrough; real lyric scaffold encoding replaced all-zero hidden states.
- Turbo schedule for `inferSteps=8`: restored to local v1.5 valid schedule ending at 0.3.

## Things ruled out THIS session (2026-05-07 afternoon + evening)

- **Float16 numerical drift** — user tested f16 + 200 steps + shift=5.1; still static. Not the dominant bug.
- **Step count alone** — quality plateaued at 8 / 32 / 100 steps before the lyric/timbre fixes. Not the original static root cause; re-run SFT/Base 50-step A/Bs after Float32 defaulting.
- **Shift parameter alone** — default 3.0 matches upstream Turbo presets. May still be useful for quality A/Bs now that music exists.
- **Sample rate mismatch** — WAV files are correctly tagged 48 kHz (afinfo).
- **Cross-attention dead** — disproved by SHA1-diff and RMS-diff between prompts.
- **FlowMatch tail-to-0.003 as Turbo fix** — corrected back to local v1.5 Turbo schedule ending at 0.3. The FlowMatch tail was not the breakthrough.
- **Tokenizer fixed-length padding** — fixed in `ae66c24`. Now matches upstream `padding="longest"`.
- **Raw user prompt vs SFT template** — fixed in `66c3162`. Now wraps with `# Instruction / # Caption / # Metas / <\|endoftext\|>`.
- **chunk_mask zeros** — fixed in `461b198` (smoke test) then `9757bac` (real).
- **silence_latent body** — fixed in `9757bac`. contextLatents `abs_mean=0.787, abs_max=6.73` matches upstream prediction.
- **TimbreEncoder bypass / default-init pooler** — fixed locally. v1.5 emits one timbre token from encoded silence/reference latents; Swift now matches that and no longer has unmatched pooler weights.
- **Empty-lyrics zero hidden states** — fixed locally. This was the breakthrough that produced first recognizable music.
- **Cross-attn pre-norm absent** — INVESTIGATED via `89c79da`, REVERTED in `3629cec`. v1.5 architecture DOES use pre-norm; weights load correctly via generic snake→camel name map; we were correct before my "fix".
- **Cross-attn RoPE missing** — INVESTIGATED via `a1a22a2`, REVERTED in `4920923`. v1.5 architecture does NOT use RoPE in cross-attn (only self-attn).
- **VAE scale_factor / shift_factor** — Reverted; constants belong to MusicDCAE, our VAE is AutoencoderOobleck.

## Diagnostic numerics across the session (turbo / 8 steps / Float32 / seed=42 / "slow dark guitar riff")

| Metric | Pre-session | Post-template | Post-silence_latent | Post-timbre/schedule | Breakthrough (real lyric scaffold) |
|---|---|---|---|---|---|
| `cond/lyricHidden` | not logged | bypass/zeros when empty | bypass/zeros when empty | zeros first token | `[1,11,2048]`, abs_mean 0.0555, first5 non-zero |
| `cond/timbreTokens` | default/random or bypass | bypass zeros | bypass zeros | `[1,1,2048]`, abs_mean 0.293 | `[1,1,2048]`, abs_mean 0.293 |
| `cond/encoderStates seq` | 2309 | 73 | 73 | 69 | 69 |
| `cond/encoderStates first5` | noisy/padded | prompt-dependent | prompt-dependent | all zeros | non-zero lyric scaffold |
| `cond/encoderStates abs_mean` | 0.37 | 2.36 | 2.36 | 2.50 | 2.509 |
| `cond/contextLatents abs_mean` | 0 | 0 | 0.787 | 0.787 | 0.787 |
| `cond/contextLatents abs_max` | 0 | 0 | 6.73 | 6.73 | 6.73 |
| `schedule last` | 0.125-ish / wrong | 0.003 after FlowMatch fix | 0.003 | 0.3 | 0.3 |
| `vt step 0 abs_mean` | 0.84 | 0.81 | 0.77 | 0.769 | 0.717 |
| `vt step N (last) abs_mean` | 0.93 | 1.03 | 1.34 | 1.00 | 1.203 |
| `vt step N (last) abs_max` | 5.6 | 11.9 | 8.28 | 6.65 | 9.42 |
| `vae/in_latents abs_mean` | 0.32 | 0.68 | 0.55 | 0.496 | 1.03 |
| `vae/in_latents abs_max` | unknown | unknown | 2.95 | 2.98 | 7.44 |
| `vae/out_audio abs_mean` | unknown | unknown | 0.0295 | 0.0238 | 0.155 |
| `vae/out_audio abs_max` | 1.7 | 1.6 | 0.30 | 0.361 | 1.98 |
| user audio | "all static" | "noisy musical" | "trying to be music" | "still no music" | "heard music for the first" |

## Things explicitly ruled out (cumulative)

- WAV writer container correctness — fully verified
- VAE architecture — fully verified (AutoencoderOobleck, 48 kHz, channel_multiples [1,2,4,8,16], strides [2,4,4,6,10])
- DiT architecture defaults — fully verified
- DiT weight loading completeness — verified 100% for everything with checkpoint weights
- Conditioning weight loading — verified 100%
- APG (CFG amplification) math — byte-identical to upstream
- TimestepEmbedding math — verified identical
- scale_shift_table modulation — verified identical
- Cross-attention mask handling — now matches upstream (no mask)
- TimbreEncoder default-init pooler contamination — fixed by removing the non-v1.5 pooler and using the trained silence-latent timbre path
- LyricEncoder PAD-only contamination — original fixed-length 2048 PAD issue fixed by longest-padding; later zero-hidden bypass removed because it was wrong after longest-padding
- CFG unconditional branch — now uses learned `null_condition_emb` (was zeros)
- Step count alone — auto-picks correctly per variant; not the original static root cause
- Guidance scale — defaults to 7.5, matches upstream defaults
- Float16 numerical drift — empirically tested f16 + 200 steps + shift=5.1, still static
- Sample rate / playback speed — WAV correctly tagged 48 kHz
- Timestep schedule trajectory — corrected to local v1.5 Turbo schedule ending at 0.3 for the 8-step path
- Cross-attention pre-norm — restored; v1.5 uses `cross_attn_norm` and weights load correctly
- Cross-attention RoPE — removed; local v1.5 does not apply RoPE in cross-attention
- VAE scale_factor / shift_factor — empirically NOT applicable to our AutoencoderOobleck

## How to resume

1. **Open this doc first.** Read "Current symptom" and "Hypotheses still open."
2. **Enable AUDIODIAG.** Generate → Advanced → check "Verbose audio diagnostics." Filter Xcode console for `AUDIODIAG`. Each generation produces ~50–100 lines of tensor stats at every boundary.
3. **Reproducible test inputs:** prompt `slow dark guitar riff`, seed `42`, default 8 steps Turbo, shift 3.0, Float32. Pre-existing reference output for diff comparison: SHA1 changes per code revision but the diagnostic numbers on this exact prompt+seed are stable for a given build.
4. **A/B prompts test:** generate same seed with two very different prompts. Should produce different SHA1s + different RMS levels. If they're identical, conditioning is ignored entirely (regression).
5. **Next-action candidates** (from "Hypotheses still open"):
   - Compare VAE decode against upstream on the same latent tensor
   - Audit DiT patch embed / unembed channel ordering against upstream
   - A/B guidance and valid Turbo shift schedules now that the output is musical
   - Verify prompt/meta whitespace byte-for-byte against upstream `SFT_GEN_PROMPT.format(...)`
6. **If audio quality is back at "all static":** something regressed. First check the breakthrough diagnostics: `cond/lyricHidden` must be non-zero, `cond/timbreTokens` must be `[1,1,2048]`, `cond/contextLatents` must be ~0.787/6.73, and Turbo schedule must end at 0.3.

## Sandbox cleanup helpers

To reset to fresh-user state for testing:

```bash
ANVIL_BASE="$HOME/Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil"
rm -rf "$ANVIL_BASE/models"
defaults delete net.pimp.anvil.AnvilApp com.pimp.AnvilApp.modelMigration.v1 2>/dev/null
```

The user's local checkpoint mirrors at `~/Developer/GitRepos/ACE-Step-1.5/checkpoints/` which can be copied into the sandbox to skip downloads:

```bash
SRC=~/Developer/GitRepos/ACE-Step-1.5/checkpoints
DST="$HOME/Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil/models"

# Qwen3 text encoder
mkdir -p "$DST/ace-step-vae"
cp "$SRC/Qwen3-Embedding-0.6B/model.safetensors" "$DST/ace-step-vae/qwen3_model.safetensors"
cp "$SRC/Qwen3-Embedding-0.6B/tokenizer.json" "$DST/ace-step-vae/tokenizer.json"
cp "$SRC/vae/diffusion_pytorch_model.safetensors" "$DST/ace-step-vae/vae.safetensors"
cp "$SRC/vae/config.json" "$DST/ace-step-vae/config.json"

# DiT (turbo)
mkdir -p "$DST/ace-step-dit-turbo"
# Note: the local turbo bundle is acestep-v15-turbo/model.safetensors which is the
# WHOLE model bundle (DiT + encoders + tokenizer + null_condition_emb).
# The HF-downloaded version is also model.safetensors at the same path.
cp "$SRC/acestep-v15-turbo/model.safetensors" "$DST/ace-step-dit-turbo/model.safetensors"
cp "$SRC/acestep-v15-turbo/config.json" "$DST/ace-step-dit-turbo/config.json"
```

Or simply let the app download from HuggingFace — Phase 2/3 download infrastructure is fully working.

## Useful diagnostic snippets

### WAV amplitude + structure analysis (Python, no numpy required)

```python
import wave, struct, math
fn = "anvil_TIMESTAMP_PROMPT.wav"
with wave.open(fn, 'rb') as w:
    n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
    raw = w.readframes(n)
samples = struct.unpack(f"<{len(raw)//sw}h", raw)
left = samples[0::2]; right = samples[1::2]
mono = [(l+r)//2 for l,r in zip(left,right)]
abs_s = [abs(s) for s in samples]
print(f"peak={max(abs_s)} ({20*math.log10(max(max(abs_s),1)/32767):.1f} dBFS)")
print(f"RMS={math.sqrt(sum(s*s for s in samples)/len(samples)):.0f}")
# Per-second energy
es = [math.sqrt(sum(x*x for x in mono[i:i+sr])/sr) for i in range(0, len(mono)-sr, sr)]
zc = [sum(1 for j in range(1,sr) if (mono[i+j-1]>=0)!=(mono[i+j]>=0)) for i in range(0, len(mono)-sr, sr)]
print(f"  per-sec RMS: min={min(es):.0f} max={max(es):.0f} ratio={max(es)/max(min(es),1):.1f}")
print(f"  per-sec ZCR: mean={sum(zc)/len(zc):.0f}  (sustained tone <800; music 1000-4000; noise 5000+)")
# L/R correlation
ml = sum(left)/len(left); mr = sum(right)/len(right)
num = sum((l-ml)*(r-mr) for l,r in zip(left,right))
den = math.sqrt(sum((l-ml)**2 for l in left) * sum((r-mr)**2 for r in right))
print(f"  L/R Pearson: {num/den:.3f}  (1.0=identical, 0=independent, ~0.5–0.9 typical for music)")
```

### Inspect safetensors keys without loading

```python
import struct, json
with open("model.safetensors", 'rb') as f:
    hl = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hl).decode())
keys = [k for k in header.keys() if k != "__metadata__"]
print(f"Total keys: {len(keys)}")
prefixes = {}
for k in keys:
    p = k.split(".")[0]
    prefixes[p] = prefixes.get(p, 0) + 1
for p, c in sorted(prefixes.items()):
    print(f"  {p}: {c}")
```

## Side-quests deferred (not blocking audio investigation)

- **User-controllable diffusion settings UI** (steps, shift, method, precision, CFG interval). User asked for this; conditioning verification was prioritized. Now that almost everything is plumbed, exposing knobs is a good follow-up.
- **Cover-song generation** (5Hz LM models). User has these downloaded locally but they're for audio→audio conditioning and require:
  - LM model download specs in `ModelRegistry.builtInSpecs`
  - Audio token tokenizer (the `tokenizer.*` subtree we currently filter out)
  - LM forward pass to produce 25Hz hints
  - Wire `precomputed_lm_hints_25Hz` into the diffusion path
- **`isModelDownloaded` accuracy bug** for the LLM (`LLMModelManager`) — currently only checks `config.json` + `tokenizer.json`, missing the safetensors. Causes "downloaded" UI state while safetensors is still streaming.
- **Background download race** for `LyricWriterManager` — UI can show "Downloading 100%" with stuck phase + active model checkmark + .failed toast simultaneously.
