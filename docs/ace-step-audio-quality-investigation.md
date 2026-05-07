# ACE-Step Audio Quality Investigation — Resumption Notes

**Last updated:** 2026-05-07 (evening session — five structural bugs found, two course corrections, audio "trying to be music" but not yet recognizable)
**Status:** Pipeline runs end-to-end. Output is now **prompt-responsive with real dynamic structure** — quantified by per-second RMS envelope (varying 445–1861), ZCR in music range (270–3400), L/R Pearson 0.778, no clipping. User-reported: "starts off slow, builds up, drops down slow, builds back up. I won't say it's music, but it's like it's trying." Step count is NOT the limiter (8/32/100 steps all produce same character). **Remaining bottleneck is most likely timbre encoder bypass** — we feed `[1, 5, 2048]` zeros where the model was trained to receive non-zero tokens.

> Use this doc to resume troubleshooting after a context reset. Skim "Current symptom" + "Hypotheses still open" first.

> **Quick orientation for the next session:** the runtime `AudioDiagnostics` toggle (Generate → Advanced → "Verbose audio diagnostics") emits `[AUDIODIAG]` lines at every pipeline boundary. That instrumentation cracked four of the structural bugs in this session. Filter the Xcode console for `AUDIODIAG`.

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

## Current symptom (post-2026-05-07-evening fixes)

Generation completes in ~5–60 s. Output WAV file:

- **30.0 s, 48000 Hz, stereo Int16, ~5.76 MB** ✓
- VAE output **abs_max 0.30** (no clipping; was 1.7 → 1.10 → 0.80 → 0.30 across the session)
- **Different prompts produce meaningfully different audio** — SHA1-diff verified, plus RMS-difference verified ("loud heavy metal" 7 dB louder than "slow dark guitar")
- **Velocity prediction healthy throughout trajectory**: step 0 abs_mean ≈ 0.77, growing to step 7 abs_mean ≈ 1.34, abs_max 8.3 — model takes increasingly confident corrections as t→0 (the canonical diffusion signature)
- **Per-second RMS envelope shows real musical dynamics** — slow ramp 0–8s (RMS 600–960), build 9–11s (1414–1861), drop 12–13s (445–611), build back 14–28s (587–1809). User-reported pattern matches numerics exactly.
- **ZCR in music range** (sustained-tonal 270–580 in opening, 2500–3400 mid, 1000–2300 late). Not noise.
- **L/R Pearson 0.778** — typical music range (0.5–0.9).
- **Quality plateaus identically at 8 / 32 / 100 steps.** Confirmed by user testing. Step count is NOT the remaining bottleneck.

User listening (verbatim, 2026-05-07 evening): *"It's hard to describe. I mean, it starts off slow, and then it builds up, and then it drops down slow, and then builds back up. I won't say it's music, but I I mean, it's like it's trying, but something is still off."*

Interpretation: the model is producing prompt-responsive musical structure with valid dynamics. What's missing is **timbre / sound-bank quality** — the notes don't resolve to recognizable instruments.

### Earlier symptoms (now resolved)

- **Pure static across all DiT variants** (start of evening session): step0/vt was nearly identical to init/xt (`vt - xt ≈ tiny perturbation`), meaning the model predicted x0 ≈ 0 regardless of prompt → diffusion converged to silence-shaped noise → VAE decoded to static.
- **"Different prompts but still all noise"** (mid-session): cross-attention was alive (RMS differs per prompt) but positionally / scale-wise wrong; output had loudness-following but no structure.
- **"Trying to be music with static layer on top"** (after silence_latent fix): user heard prompt content underneath but the noise floor was high. silence_latent eliminated the noise floor entirely; the "trying" is now exposed.

## What's verified working (don't re-investigate these)

### Weight loading (100% match for everything that has weights in v1.5 checkpoint)

Diagnostic logs in `DiTActor.loadWeights`, `VAEActor.loadWeights`, `ConditioningActor.loadWeights` print `supplied=N model=M matched=K` plus first 8 mismatches each side. Recent run:

| Module | Supplied | Model | Matched | Notes |
|---|---|---|---|---|
| DiT | 476 | 477 | 476 | Only `rotaryEmb.invFreq` unsupplied (runtime-computed) |
| Qwen3 | 310 | 311 | 310 | Same — `rotaryEmbedding.invFreq` |
| LyricEncoder | 91 | 92 | 91 | Same |
| TimbreEncoder | 47 | 73 | 47 | 26 unmatched all under `attentionPooler.layers.0..1.*` and `.norm` — those have NO weights in v1.5 checkpoint at all (only `special_token` exists at top level, shape mismatched [1,1,2048] vs our [1,5,2048] — bypass below) |
| ConditionEncoder | 1 | 1 | 1 | `text_projector.weight` → `textProjection.weight` |
| VAE | 291 | 291 | 291 | All match |

### Healthy intermediate values

`DiffusionActor` final-latent diagnostic shows abs_mean ≈ 0.85, abs_max ≈ 4.5 (Turbo) or 1.0 / 5.8 (Full). Both well within typical VAE-latent magnitude range.

`VAEActor.decode` diagnostic shows audio output abs_max ≈ 1.15 (Turbo) / 1.78 (Full). Slightly above [-1, 1] but normalized down by the WAVWriter peak-normalize fix.

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
| `66c3162` | **SFT_GEN_PROMPT template wrap** | Tokenizing raw user prompt — DiT was trained on `# Instruction / # Caption / # Metas / <\|endoftext\|>` scaffold (`acestep/constants.py:SFT_GEN_PROMPT`) plus `# Languages / # Lyric / <\|endoftext\|>` for lyrics (`prompt_utils.py:_format_lyrics`). Now wraps both. Default vocal_language=unknown, metas all N/A except duration. encoderStates seq grew ~10 → ~73 (longer wrapped prompt); abs_mean 1.5 → 2.4. |
| `461b198` | chunk_mask = ones, not zeros (smoke test) | Upstream text2music: `context_latents = cat([silence_latent_slice, chunk_mask_ones], axis=-1)` (`models/turbo/modeling_acestep_v15_turbo.py:1680, 1931`). We had all zeros. Smoke test flipped mask half to ones. contextLatents abs_mean 0 → 0.5; vt step 7 abs_mean 0.57 → 1.53 (model now confidently shaping output near t=0). |
| `9757bac` | **`silence_latent.pt` loader for text2music context body** | Final piece of `cat([silence_latent_slice, ones])`. New `SilenceLatentLoader.swift` reads the `.pt` file via `/usr/bin/unzip -p` (no third-party zip lib, no pickle parsing), extracts raw float32 at `*/data/0`, returns `[1, 15000, 64]` (transposed from stored `[1, 64, 15000]`). `ConditioningActor.silenceLatent` field; encode() slices `[:, :tFrames, :]` (tiles if duration > 600s). `ACEStepGenerationService` loads it next to model.safetensors. ModelRegistry: added to all 3 DiT specs (3.84 MB each, auto-downloads). Result: contextLatents `abs_mean=0.787, abs_max=6.73` — matches GPT-5.5's predicted `~0.78 / ~6.7`. VAE output `abs_max=0.30` (was 1.7 → 1.10 → 0.80 → 0.30 across the session). User: "starts off slow, builds up, drops down slow, builds back up. It's like it's trying." |
| `c03c4e0` → `d71364b` | VAE scale_factor (REVERTED) | Applied `latents / 0.1786 + (-1.9091)` based on upstream `music_dcae_pipeline.py:116`. Made things much worse (abs_max 1.7 → 7.3). **Misdiagnosis:** our VAE is `AutoencoderOobleck` (Stability AI's, per `vae/config.json:_class_name`), NOT MusicDCAE. Those constants belong to MusicDCAE's wrapper. Reverted. |

## Hypotheses still open (post-2026-05-07-evening)

Ranked by my current likelihood estimate:

### 1. **Timbre encoder bypass for text2music — most likely remaining bug**

When the user provides no reference audio, `ConditioningActor.encode` currently substitutes a `[1, 5, 2048]` zeros tensor for the timbre tokens. The conditionEmbedder Linear has a learned bias, so those zero rows produce non-zero K/V projections in cross-attn — but the rows are *constant* (every "fake" timbre token sees the same bias output). The model was trained with **non-zero** timbre tokens for text2music: upstream `models/turbo/modeling_acestep_v15_turbo.py:1880-1885` runs `silence_latent_expanded` through the timbre / acoustic processing branch when no reference audio is provided. **We bypass that path entirely.**

Symptom match: real musical structure (notes, dynamics, ZCR in music range) but no recognizable instruments. That's exactly what "wrong timbre tokens" should produce — the rhythmic / harmonic skeleton survives because it comes from text + lyric branches; the timbre flavoring is missing because we never feed it.

**Investigation steps:**
- Trace upstream's exact text2music path: what gets passed as `refer_audio_acoustic_hidden_states_packed` and `hidden_states` when `is_covers=False` and no reference audio?
- The line `silence_latent_expanded = silence_latent[:, :src_latents.shape[1], :].expand(src_latents.shape[0], -1, -1)` in turbo modeling suggests silence_latent feeds the acoustic branch too, not just the context_latents.
- Implement: run `silence_latent` through `timbreEncoder` instead of bypassing.

### 2. ConditionEncoder fusion order / weighting

The fusion of textHidden (~73 tokens) + lyricHidden (variable) + timbreTokens (5) into the final encoder_states. We do `concatenated([text, lyric, timbre], axis: 1)` (need to verify exact order). If upstream concatenates in a different order, the downstream cross-attn positional table is misaligned with what the trained K projection expects.

### 3. Weight name mismatches — some params silently default-initialized

The DiTActor/VAEActor diagnostic prints supplied/model/matched counts. **Confirmed missing in current run:**
- DiT: `rotaryEmb.invFreq` (runtime-computed, expected)
- Qwen3: `rotaryEmbedding.invFreq` (same)
- LyricEncoder: `rotaryEmbedding.invFreq` (same)
- **TimbreEncoder: 26 params unmatched** under `attentionPooler.layers.0..1.*` — these are default-initialized RANDOM weights. We bypass the timbre encoder entirely for text2music so it shouldn't matter, but if Hypothesis #1 above is correct and we start running silence_latent through it, we'll need real weights. They aren't in the v1.5 checkpoint at all (only the special_token, which is also shape-mismatched). **Need to investigate where ACE-Step expects to load these from** — possibly a separate timbre encoder checkpoint we haven't downloaded.

### 4. Self-attention or other layer-internal bug

Audio is structured but lacks coherent phrasing. Could be a subtle bug in self-attention mixing — `nRep` for grouped-query, `transposed(0, 2, 1, 3)` ordering, qNorm/kNorm placement. Lower priority because cross-attn is verified working (different prompts produce different outputs) and self-attn share the same `Attention` class — if cross-attn produces meaningful outputs, self-attn likely is too.

### 5. Patch embed / un-embed (`projIn` / `projOut`)

Audio output dynamics look healthy and prompt-following works, suggesting the input/output projections are right. But would still verify the conv axis order (`transposed(0, 2, 1)` for Conv1d, `transposed(1, 2, 0)` for ConvTranspose1d) matches upstream byte-for-byte.

### 6. Silence-prompt baseline test

Generate with **empty text + empty lyrics**. With the SFT_GEN_PROMPT template applied, this still produces the wrapper structure with empty Caption/Lyric bodies — useful sanity check that empty-content path doesn't regress.

## Things ruled out THIS session (2026-05-07 afternoon + evening)

- **Float16 numerical drift** — user tested f16 + 200 steps + shift=5.1; still static. Not the dominant bug.
- **Step count** — quality plateaus IDENTICALLY at 8 / 32 / 100 steps (user-tested 2026-05-07 evening). Not step-count-sensitive.
- **Shift parameter** — schedule fix made shift values produce sensible trajectories. Default 3.0 matches upstream.
- **Sample rate mismatch** — WAV files are correctly tagged 48 kHz (afinfo).
- **Cross-attention dead** — disproved by SHA1-diff and RMS-diff between prompts.
- **Schedule trajectory tail** — fixed in `5faf163`, verified against upstream FlowMatchEulerDiscreteScheduler.
- **Tokenizer fixed-length padding** — fixed in `ae66c24`. Now matches upstream `padding="longest"`.
- **Raw user prompt vs SFT template** — fixed in `66c3162`. Now wraps with `# Instruction / # Caption / # Metas / <\|endoftext\|>`.
- **chunk_mask zeros** — fixed in `461b198` (smoke test) then `9757bac` (real).
- **silence_latent body** — fixed in `9757bac`. contextLatents `abs_mean=0.787, abs_max=6.73` matches upstream prediction.
- **Cross-attn pre-norm absent** — INVESTIGATED via `89c79da`, REVERTED in `3629cec`. v1.5 architecture DOES use pre-norm; weights load correctly via generic snake→camel name map; we were correct before my "fix".
- **Cross-attn RoPE missing** — INVESTIGATED via `a1a22a2`, REVERTED in `4920923`. v1.5 architecture does NOT use RoPE in cross-attn (only self-attn).
- **VAE scale_factor / shift_factor** — Reverted; constants belong to MusicDCAE, our VAE is AutoencoderOobleck.

## Diagnostic numerics across the session (turbo / 8 steps / Float32 / seed=42 / "slow dark guitar riff")

| Metric | Pre-session | Post-schedule | Post-template | Post-mask-ones | Post-silence_latent |
|---|---|---|---|---|---|
| `cond/encoderStates seq` | 2309 | 2309 | 73 | 73 | 73 |
| `cond/encoderStates abs_mean` | 0.37 | 0.37 | 2.36 | 2.36 | 2.36 |
| `cond/encoderMask abs_mean` | 0.005 | 0.005 | 1.0 | 1.0 | 1.0 |
| `cond/contextLatents abs_mean` | 0 | 0 | 0 | 0.5 | 0.787 |
| `cond/contextLatents abs_max` | 0 | 0 | 0 | 1.0 | 6.73 |
| `vt step 0 abs_mean` | 0.84 | 0.84 | 0.81 | 0.73 | 0.77 |
| `vt step N (last) abs_mean` | 0.93 | 0.94 | 1.03 | 1.53 | 1.34 |
| `vt step N (last) abs_max` | 5.6 | 12.1 | 11.9 | 11.86 | 8.28 |
| `vae/in_latents abs_mean` | 0.32 | 0.32 | 0.68 | 0.55 | 0.55 |
| `vae/out_audio abs_max` | 1.7 | 1.7 | 1.6 | 1.10 | 0.30 |
| user audio | "all static" | "static" | "noisy musical" | "static layer on top" | "trying to be music" |

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
- TimbreEncoder padding contamination — bypassed
- LyricEncoder PAD-only contamination — bypassed
- CFG unconditional branch — now uses learned `null_condition_emb` (was zeros)
- Step count — auto-picks correctly per variant; quality plateaus at 32 → not step-count-sensitive
- Guidance scale — defaults to 7.5, matches upstream defaults
- Float16 numerical drift — empirically tested f16 + 200 steps + shift=5.1, still static
- Sample rate / playback speed — WAV correctly tagged 48 kHz
- Timestep schedule trajectory — now matches upstream FlowMatchEulerDiscreteScheduler down to ~0.003
- Cross-attention pre-norm — removed (matched upstream)
- Cross-attention RoPE — Q rotated by latent rotary, K rotated by encoder rotary (matches upstream)
- VAE scale_factor / shift_factor — empirically NOT applicable to our AutoencoderOobleck

## How to resume

1. **Open this doc first.** Read "Current symptom" and "Hypotheses still open."
2. **Enable AUDIODIAG.** Generate → Advanced → check "Verbose audio diagnostics." Filter Xcode console for `AUDIODIAG`. Each generation produces ~50–100 lines of tensor stats at every boundary.
3. **Reproducible test inputs:** prompt `slow dark guitar riff`, seed `42`, default 8 steps Turbo, shift 3.0, Float32. Pre-existing reference output for diff comparison: SHA1 changes per code revision but the diagnostic numbers on this exact prompt+seed are stable for a given build.
4. **A/B prompts test:** generate same seed with two very different prompts. Should produce different SHA1s + different RMS levels. If they're identical, conditioning is ignored entirely (regression).
5. **Next-action candidates** (from "Hypotheses still open"):
   - Add unloaded-weight-name diff dump (Hypothesis 1)
   - Self-attention/projection layer diff vs upstream (Hypothesis 2)
   - patchEmbed / patchUnembed audit (Hypothesis 3)
6. **If audio quality is back at "all static":** something regressed. Check that the four shipped fixes (`5faf163` schedule, `89c79da` cross-attn norm, `a1a22a2` RoPE, plus `5c82b54` diagnostics) are present in your build.

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
