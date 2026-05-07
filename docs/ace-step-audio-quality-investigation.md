# ACE-Step Audio Quality Investigation — Resumption Notes

**Last updated:** 2026-05-07 (afternoon session — three structural bugs found via runtime instrumentation)
**Status:** Pipeline runs end-to-end. Output WAV is structurally valid (30s, 48 kHz, stereo, full amplitude). **Output is now prompt-responsive and "getting better, noisy but musical"** (user-reported). Was pure static across all configs at the start of this session — three structural model bugs were diagnosed and fixed. Remaining noise still under investigation.

> Use this doc to resume troubleshooting after a context reset. Skim "What's Verified Working" + "Hypotheses Still Open" first.

> **Quick orientation for the next session:** the runtime `AudioDiagnostics` toggle (Generate → Advanced → "Verbose audio diagnostics") emits `[AUDIODIAG]` lines at every pipeline boundary. That instrumentation is what cracked three of the bugs in this session. Filter the Xcode console for `AUDIODIAG`.

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

## Current symptom (post-2026-05-07-afternoon fixes)

Generation completes in ~5–60 s (depending on Turbo 8 steps vs Full/SFT 50 steps). Output WAV file:

- **30.0 s, 48000 Hz, stereo Int16, ~5.76 MB** ✓
- VAE output abs_max ~1.7 (slight clipping pre-WAVWriter normalize)
- **Different prompts produce meaningfully different audio character** — verified by SHA1-diffing two outputs from same seed but different prompts (different hashes), and by RMS analysis: "loud aggressive heavy metal scream" prompt produced output 7 dB louder than "slow dark guitar riff" with the same seed
- step0 velocity prediction now diverges meaningfully from input noise (`vt - xt` magnitude ~0.4, was ~0.07 pre-fixes), and grows through the trajectory (abs_max 5.3 → 12.1) — healthy diffusion behavior

User listening: "Lots of noise but if does sound different. I do hear something musical I believe." Quality plateaus at 32 steps → remaining issue is structural, not step-count-sensitive.

### Pre-fix symptom (now resolved)

Was: pure static across all DiT variants and configurations. step0/vt was nearly identical to init/xt (`vt - xt ≈ tiny perturbation`), meaning the model predicted x0 ≈ 0 regardless of prompt → diffusion converged to silence-shaped noise → VAE decoded to static.

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

### 2026-05-07 afternoon — three structural bugs found via runtime AUDIODIAG instrumentation

| Commit | Title (paraphrased) | Why |
|---|---|---|
| `5c82b54` | Runtime-toggleable `AudioDiagnostics` | New `AnvilCore/Inference/AudioDiagnostics.swift` enum gated by UserDefaults `audioDebugVerbose`. Logs `[AUDIODIAG]` at every pipeline boundary: cond outputs, init noise, schedule, per-step vt + xt, post-APG velocity, VAE in/out. UI toggle in Generate → Advanced. **This instrumentation cracked the three bugs below.** |
| `5faf163` | **Timestep schedule was missing the trajectory tail** | `compute()` used `1 - i/N` (going from 1.0 down to `1/N` ≈ 0.125 for N=8) instead of upstream's `linspace(1.0, 1/num_train_timesteps, N)` (going from 1.0 down to ~0.001). Combined with wrong spacing formula. Net: diffusion only walked 70% of the trajectory before `finalStep` tried to leap the remaining 30% in one Euler step → coarse approximation → noisy latent → static. Now matches `acestep/schedulers/scheduling_flow_match_euler_discrete.py`. Tests updated. |
| `89c79da` | **Erroneous norm before cross-attention** | `DiTLayer` applied `crossAttnNorm(h)` before crossAttn. Upstream `acestep/models/attention.py:308-315` feeds RAW `hidden_states` directly to `self.cross_attn(...)` — no pre-norm. There's no corresponding tensor in safetensors for `crossAttnNorm.weight`, so its gamma was default-initialized; the norm rescaled query magnitudes away from the cross-attn projections' training distribution → cross-attn output became near-uniform-per-batch (a constant nudge that depends on prompt's average encoder values but not spatial content). Symptom: prompts modulated loudness only, not musical structure. The unused field is left in place to avoid disturbing Module reflection. |
| `a1a22a2` | **Cross-attention had NO rotary embedding at all** | Our `DiTAttention.callAsFunction` cross-attn branch built K/V from encoder states without ever calling the rotary helper, and threw away the incoming `positionCosSin`. Upstream `customer_attention_processor.py:353-358` rotates Q with `rotary_freqs_cis` (latent positions) and K with `rotary_freqs_cis_cross` (encoder positions) — different tables per side. Our cross-attn was positionally blind. Added `encoderPositionCosSin` parameter; `DiTModel` computes a second rotary table from encoder seq length and threads through. New `applyRotaryToQOnly` / `applyRotaryToKOnly` helpers for the asymmetric case (existing `applyRotaryPosEmb` still used for self-attn). |
| `c03c4e0` → `d71364b` | **VAE scale_factor (REVERTED)** | Attempted to apply `latents = latents / 0.1786 + (-1.9091)` before VAE decode based on upstream `music_dcae_pipeline.py:116`. Made things substantially worse — VAE output abs_max went 1.7 → 7.3. Root cause of the misdiagnosis: our VAE is **`AutoencoderOobleck`** (Stability AI's audio VAE, per `vae/config.json` `_class_name`), not MusicDCAE. The scale/shift constants belong to MusicDCAE's wrapper layer; Oobleck doesn't use them. Reverted in `d71364b`. Lesson logged: **always double-check which VAE class our `vae.safetensors` actually contains before applying upstream MusicDCAE constants.** |

## Hypotheses still open (post-2026-05-07-afternoon)

Ranked by my current likelihood estimate:

### 1. Weight name mismatches — some params silently default-initialized

Same family as the just-fixed `crossAttnNorm` bug: a model field that has no corresponding safetensors tensor stays at its `MLXArray.zeros` / RMSNorm default. The DiTActor/VAEActor diagnostic prints supplied/model/matched counts, but doesn't currently list which model parameters never received a supplied tensor. **Next move:** add a one-time log dump that prints `model_params - supplied_keys` (after applying name maps) and look for any params still default-initialized. Especially audit the cross-attention QKV projections, layer-norm gammas, and the final-layer scale_shift_table.

### 2. Self-attention or other layer-internal bug

Cross-attention had a pre-norm bug AND a missing-RoPE bug — both took targeted upstream-diff to find. Worth doing a similar systematic diff of the self-attention path vs upstream's `customer_attention_processor.py`:
- `nRep` for grouped-query attention (we have 16/8=2, verify broadcast direction)
- `transposed(0, 2, 1, 3)` ordering in reshape
- qNorm/kNorm placement (per-head normalization on Q and K projections)

### 3. Patch embed / un-embed (`projIn` / `projOut`)

The conv1d-based patch embedding at DiT input boundary and the corresponding un-embedding at output. Configured patch_size=2; verify our `projIn` produces 2-stride convolutions matching upstream's `PatchEmbed.forward`, and `projOut` reverses correctly with the right channel ordering.

### 4. ConditionEncoder / lyric / timbre projector subtleties

Encoder states arrive at the DiT input with healthy magnitude (`abs_mean=0.37`, `abs_max=20.7`), but maybe their sub-encoder mix is wrong: lyric encoder vs text encoder weighting, timbre projection scale, etc. Worth dumping per-sub-encoder outputs before they're fused into `condition_embedder`.

### 5. Schedule curvature — try alternate formulas

Now using upstream's `linspace(1.0, 1/num_train, N)` then `t' = shift * t / (1 + (shift - 1) * t)`. Config has `timestep_mu=-0.4, timestep_sigma=1.0` (training distribution). Inference might use a LogNormal-derived schedule for some variants. Lower priority since the trajectory is now structurally correct (covers the full range to ~0.003).

### 6. Silence-prompt baseline test

Still useful as a sanity check: generate with **empty text prompt + empty lyrics**. If output sounds noticeably different from a real prompt, conditioning is reaching the model meaningfully (which we already believe — different prompts produce different SHA1s and different RMS).

## Things ruled out THIS session (2026-05-07 afternoon)

- **Float16 numerical drift** — user tested f16 + 200 steps + shift=5.1; still static. Not the dominant bug.
- **Step count** — quality plateaus at 32 steps with all current fixes. Not step-count-sensitive.
- **Shift parameter** — schedule fix made shift values produce sensible trajectories. Default 3.0 matches upstream.
- **Sample rate mismatch** — verified WAV files are correctly tagged 48 kHz (afinfo). Not a playback-speed issue.
- **VAE scale_factor / shift_factor (0.1786 / -1.9091)** — those constants belong to upstream's MusicDCAE wrapper. Our VAE is AutoencoderOobleck which doesn't use them. Empirical test made output substantially worse; reverted.
- **Cross-attention completely dead** — disproved by SHA1-diff: same seed + different prompts = different outputs (different RMS levels even). Cross-attn is alive; was just attenuated/positionally-blind, both fixed.

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
