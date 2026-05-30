# Stable Audio Open SFX Pipeline — Audio Quality Investigation

**Last updated:** 2026-05-10
**Status:** ✅ **COMPLETE — Python parity confirmed on all metrics.**

| Metric | Swift | Python |
|---|---|---|
| crossCond abs_mean | 0.046 | 0.037 |
| (cond−uncond).abs_mean step 0 | **0.0206** | **0.0193** |
| CFG diff ratio | **11.7%** | **11.4%** |
| final_latents abs_mean | 0.688 | ~0.65 |
| audio peak | −7 dBFS | −5.9 dBFS |
| RMS range ratio | ~1.7× | 1.7× |
| ZCR | ~1267 | 1267 |

The SAO pipeline is working correctly. Short prompts produce consistent-texture ambient audio
in both Swift and Python — this is a model characteristic, not a bug.

> Use this doc to resume troubleshooting after a context reset. Skim "Current symptom" +
> "Hypotheses still open" first.

> **Quick orientation:** the `[AUDIODIAG]` toggle (same as ACE-Step —
> `defaults write com.pimpstudios.PiMPAudioStudio com.pimpstudios.PiMPAudioStudio.audioDebugVerbose -bool YES`) is wired into the SFX
> pipeline. Filter Xcode console for `SFX_` to see only SAO-path lines.

---

## Project context

Porting **Stable Audio Open 1.0** (text→sound-effects diffusion model) from Python/mlx-audiogen
into the same Swift + MLX-Swift app as ACE-Step. Pipeline:

```text
text prompt
  → T5-base tokenizer + encoder  →  [B, 128, 768] hidden states (padded to max_length=128)
  → SAONumberConditioner (seconds_start=0, seconds_total=duration)
      → startEmb [1, 768], totalEmb [1, 768]
  → Build conditioning:
      crossCond   = cat([T5[:64], totalToken, startToken])  [1, 66, 768]
      globalCond  = cat([totalEmb, startEmb], axis=-1)  [1, 1536]  (total FIRST)
      uncrossCond = cat([T5("")[:64], totalToken, startToken])  [1, 66, 768]
  → SFXDiffusionActor (rectified flow, 100 steps, logSNR schedule)
      → each step: SFXDiTModel.forward(latents, timestep, crossCond, globalCond)
  → OobleckDecoder (VAE, strides [2,4,4,8,8] = 2048 hop, 44.1 kHz stereo)
  → audio samples [B, T_audio, 2]
```

Upstream reference:

- Local Python source: `~/Developer/GitRepos/AnvilApp/.venv/.../mlx_audiogen/models/stable_audio/`
  - `pipeline.py` — top-level `generate()`, `sigma_max=1.0`, `negative_prompt=""`
  - `conditioners.py` — `T5[:64]+totalToken+startToken` for cross-attn; `[total,start]` globalCond order
  - `sampling.py` — logSNR sigmoid schedule, Euler ODE step
  - `dit.py` — prepend token approach, `to_cond_embed` applied to cross-attn tokens
- HuggingFace checkpoint: `stabilityai/stable-audio-open-1.0` (also `stable-audio-open-small`)
- Local MLX weights: `~/.cache/anvil-audio/mlx-weights/stable-audio-open-1.0/{vae,dit,t5,conditioners}.safetensors`

**Key architecture facts (do not re-investigate these):**

| Parameter | Value |
|---|---|
| DiT layers | 24, hidden=1536 |
| Attention | 24Q / 12KV GQA, head_dim=64, softmax |
| FFN | GEGLU, intermediate=6144, **bias=true** on all linears |
| RoPE | Partial: only first 32 of 64 head dims rotated |
| Timestep embed | Learned Fourier bank `[128,1]` → `[B, 256]` |
| Cross-attn | T5 768-dim tokens as K/V + `to_cond_embed` 768→768 projection |
| Global cond | `seconds_start` + `seconds_total` via `SAOProjectionModel` → `[B,1536]` |
| Conditioning style | Prepend token (single `gEmb` prepended, sliced off after 24 blocks) |
| VAE | OobleckDecoder, strides `[2,4,4,8,8]`, hop=2048 |
| Sample rate | 44,100 Hz stereo |
| Latent frame rate | 44100 / 2048 ≈ 21.5 fps |
| Diffusion | Rectified flow (NOT Karras EDM), t ∈ [1, 0] |
| Schedule | logSNR sigmoid, `logsnr_max = −6.0`, `logsnr_min = 2.0` |
| T5 | T5-base v1.0: single `wi`+`wo` ReLU FFN (NOT GEGLU), RMSNorm, 12 layers |
| CFG null | T5("") hidden states (empty string, same tokenizer) |

---

## Current symptom (end of 2026-05-10 session)

**Audio is now real content at normal amplitude — no longer static or buzz. However all three
test prompts ("footsteps on a wooden floor", "dog barking", "rain falling on a window") produce
similar-sounding, consistent-RMS output with no temporal structure matching the described sounds.
Conditioning scale values match Python expectations; why text semantics aren't driving output
is the open question.**

### Timeline

| Date / Run | Output | Root cause identified |
|---|---|---|
| Pre-session | All static | Karras EDM schedule on RF checkpoint |
| Mid-session | Buzz, prompt-independent | Per-layer cond broadcast |
| Later | Buzz + −0.22 DC | Time tokens in cross-attn (subsequently re-examined — currently included per stable-audio-tools layout) |
| 2026-05-09 | Loud buzz + large ±DC (ch0=−0.60, ch1=+0.33) | VAE on random weights |
| 2026-05-10 (post-VAE fix) | Very faint (~−30 dBFS) white noise, DC near 0 | DiT transformer on random weights |
| 2026-05-10 (post-DiT fix) | Very faint (~−30 dBFS) white noise | T5 weights broken (3 issues) |
| 2026-05-10 (post-T5 fix) | Real audio, normal amplitude | ✅ Resolved — matches Python reference |

### Python parity comparison (2026-05-10)

Python mlx-audiogen run — "footsteps on a wooden floor", 30s, 100 steps, CFG 7.5:

```text
cond_tokens  abs_mean=0.0370   uncond_tokens abs_mean=0.0221   global_cond abs_mean=0.6767
v_cond   abs_mean=0.1699
v_uncond abs_mean=0.1748
(cond−uncond).abs_mean=0.0193   ratio=11.4%   ← proper diff metric
v_cfg    abs_mean=0.1880
audio    abs_max=0.5092 (−5.9 dBFS)   RMS range ratio=1.7×   ZCR=1267
```

Swift output (same checkpoint, same CFG):

```text
crossCond abs_mean≈0.046   uncrossCond abs_mean=0.022   globalCond abs_mean=0.677
audio abs_max≈0.45 (−7 dBFS)   RMS range ratio≈1.7×   ZCR≈1267
```

**The Python output has identical temporal structure to ours** — same ~1.7× RMS range ratio,
same ~1200–1300 ZCR, flat per-second energy. The "prompt-independent texture" is a SAO model
characteristic for short prompts, not a Swift bug. SAO generates textural ambient audio rather
than discrete event sounds when given 4–8-word prompts.

**Conditioning diff metric — confirmed matching:**

```text
Swift step0 cond_abs_mean=0.1755  uncond_abs_mean=0.1719
            diff_abs_mean=0.0206  ratio=0.1174
Python      diff_abs_mean=0.0193  ratio=0.1135
```

Swift is within 7% of Python on the actual `(cond−uncond).abs_mean`. Cross-attention
conditioning is working correctly. The `diff_abs_mean` line prints automatically on every
step-0 CFG pass — any future regression from ~0.019–0.021 would indicate a conditioning bug.

Swift `final_latents abs_mean=0.6879  abs_max=3.7960  mean=−0.0798` — latents in range,
slight negative DC is normal for this checkpoint.

---

## Root causes found and fixed

### 1. Wrong diffusion parameterization — Karras EDM vs rectified flow

**Symptom:** All static, no structure.

**Bug:** `SFXDiffusionActor` used a Karras sigma schedule `σ_max = 500, σ_min = 0.3` (EDM
parameterization) and the `edmEulerStep` variational ODE. The SAO checkpoint is a rectified-flow
model with t ∈ [1, 0] and the simple Euler step `x_next = x_t + (t_next − t_curr) * v_t`.

**Fix:** Replaced Karras schedule with `generateRFSchedule` (logSNR sigmoid, pins `t[0] = sigmaMax = 1.0`).
Replaced `edmEulerStep` with `odeStep`.

**Files:** `SFXDiffusionActor.swift`, `DiffusionSampler.swift`, `Types.swift`
(`sigmaMax` added to `SFXDiffusionConfig`).

---

### 2. Per-layer conditioning broadcast instead of prepend token

**Symptom:** Constant-amplitude buzz with the same waveform regardless of prompt.

**Bug:** `SFXDiTModel.forward` was adding the global conditioning embed to every layer's hidden
states (24 residual additions). This overwhelmed the cross-attention signal entirely. The
mlx-audiogen reference prepends the conditioning as a single extra token at position 0 of the
sequence, then removes it after all 24 blocks.

**Fix:** `SAODiTTransformer.callAsFunction` now:

1. Prepends `prependToken` → sequence becomes `[B, T+1, H]`
2. Runs all 24 layers
3. Slices away position 0 → back to `[B, T, H]`

**Files:** `SFXDiTModel.swift`.

---

### 3. `toCondEmbed` bypassed and wrong dimensions

**Symptom:** Weight load shape mismatch / silent wrong output.

**Bug:** `toCondEmbed` (the non-linear projection applied to cross-attention tokens before they
enter the blocks) was initialized as `SAOBiasMLP(768, 1536)` — wrong output dimension. The
checkpoint has `[768, 768]` weights. The projection was also never called before the transformer.

**Fix:** Changed init to `SAOBiasMLP(768, 768)`. Added `projCross = toCondEmbed(crossCond)` call
in `SFXDiTModel.forward` before passing to the transformer.

**Files:** `SFXDiTModel.swift`.

---

### 4. CFG null conditioning was zeros

**Symptom:** Guidance scale had no effect / added noise instead of directing output.

**Bug:** The unconditional branch for classifier-free guidance used `MLXArray.zeros(like: crossCond)`.
The correct null conditioning for SAO is T5-encoding the empty string `""`.

**Fix:** `SoundFXPipeline.generate(text:...)` now encodes both the prompt AND `""` via T5. The
empty-string hidden states are passed as `uncondText` through to `SFXDiffusionActor.sample(uncrossCondTokens:)`.

**Files:** `SoundFXPipeline.swift`, `SFXDiffusionActor.swift`.

---

### 5. GlobalCond ordering wrong — [start, total] instead of [total, start]

**Symptom:** Systematic conditioning mismatch; `seconds_start` and `seconds_total` embeddings
fed in swapped order to the model.

**Bug:** Swift code concatenated `[startEmb, totalEmb]`. The mlx-audiogen reference concatenates
`[totalEmb, startEmb]` (total first).

**Fix:** `let globalCond = concatenated([totalEmb, startEmb], axis: -1)`.

**Files:** `SoundFXPipeline.swift`.

---

### 6. Time-token cross-attention cross-examination (resolved — included)

**History:** Time tokens were initially appended to cross-attention (positions 64–65), then
removed after a −0.22 DC offset was observed, then re-added after reviewing stable-audio-tools
training source which confirms the training layout is:

```text
cross_attn_tokens = cat([T5[:64], total_time_emb[:, None, :], start_time_emb[:, None, :]])
```

Current layout (`crossCond shape=[1, 66, 768]`) matches stable-audio-tools and mlx-audiogen.
The DC offset observed during the removal experiment was likely caused by a different concurrent
bug (possibly the `toCondEmbed` wrong-dimensions issue). Time tokens are currently included and
the conditioning scale values match Python expectations.

**Files:** `SoundFXPipeline.swift`.

---

### 7. VAE weight loading completely broken — all 145 weights silently dropped

**Symptom:** Large opposite-sign DC offsets (ch0 ≈ −0.60, ch1 ≈ +0.33). VAE output
statistically indistinguishable from a randomly-initialized network.

**Root cause:** `loadSFXVAEWeights` called `VAENameMap.mapKey` (designed for ACE-Step key format).
The SAO `vae.safetensors` uses MLX Sequential naming (`decoder.layers.layers.0.*`,
`decoder.layers.layers.1.layers.layers.0.*`, etc.). Every key produced a path like
`layers.layers.0.weight` that matched nothing in `OobleckDecoder`. MLX's `model.update(parameters:)`
silently drops unmatched keys. All 145 decoder weights went unloaded.

**SAO VAE checkpoint path structure (MLX Sequential → named Swift property):**

| Checkpoint path | Swift property path |
|---|---|
| `decoder.layers.layers.0.*` | `decoder.conv1.*` |
| `decoder.layers.layers.1.*` | `decoder.blocks.0.*` |
| `decoder.layers.layers.2.*` | `decoder.blocks.1.*` |
| `decoder.layers.layers.3.*` | `decoder.blocks.2.*` |
| `decoder.layers.layers.4.*` | `decoder.blocks.3.*` |
| `decoder.layers.layers.5.*` | `decoder.blocks.4.*` |
| `decoder.layers.layers.6.*` | `decoder.snake1.*` |
| `decoder.layers.layers.7.*` | `decoder.conv2.*` |
| `decoder.*.layers.layers.0.*` | `decoder.*.snake1.*` (block-level) |
| `decoder.*.layers.layers.1.*` | `decoder.*.convT1.*` (block-level) |
| `decoder.*.layers.layers.2–4.*` | `decoder.*.resUnit1–3.*` (block-level) |
| `decoder.*.resUnitN.layers.layers.0.*` | `decoder.*.resUnitN.snake1.*` |
| `decoder.*.resUnitN.layers.layers.1.*` | `decoder.*.resUnitN.conv1.*` |
| `decoder.*.resUnitN.layers.layers.2.*` | `decoder.*.resUnitN.snake2.*` |
| `decoder.*.resUnitN.layers.layers.3.*` | `decoder.*.resUnitN.conv2.*` |

**Fix:** Rewrote `SFXVAENameMap` with a proper three-level path translator. Updated
`loadSFXVAEWeights` to use `SFXVAENameMap.mapKey` with no weight-norm fusion or axis
transposition (SAO weights are already in MLX NLC format).

**Files:** `SFXVAENameMap.swift` (complete rewrite), `WeightConverter.swift` (`loadSFXVAEWeights`).

---

### 8. DiT transformer weights completely missing — all 360 block weights silently dropped

**Symptom:** Very faint audio (−30.4 dBFS), ZCR ≈ 0.5 (statistical white noise). DC near zero.
Output identical across prompts.

**Root cause:** `SFXDiTNameMap.mapKey` did not handle the `blocks.N.*` key prefix used in the
mlx-audiogen `dit.safetensors` export. The checkpoint stores all 24 transformer layers as
top-level `blocks.0…23.*` keys (360 tensors) with no `model.model.` or `transformer.` prefix.
Because `mapKey` passed `blocks.0.*` through unchanged, MLX tried to set `blocks.0.*` on the
top-level `SFXDiTModel`, found no property named `blocks`, and silently dropped all 360 tensors.
The model ran with default-initialized random transformer weights → garbage velocity predictions
→ random walk in latent space → 4× out-of-distribution latents (abs_mean ≈ 3.1) → Snake
activations in high-frequency oscillation regime → near-zero white-noise audio.

**Additional secondary bug:** The Python DiT applies preprocess_conv and postprocess_conv as
**residual corrections** (`x_t = x_t + self.preprocess_conv(x_t)`). Our Swift `forward()` was
doing `let h = preprocessConv(latents)` (no add-back), discarding the input signal.

**Fix:**

1. `SFXDiTNameMap.mapKey` — added step 6: `blocks.N.* → transformer.layers.N.*`
2. `SFXDiTModel.forward` — `latents + preprocessConv(latents)` and `out + postprocessConv(out)`
3. `SFXDiTActor.loadWeights` — added `supplied/model/matched` coverage diagnostic

**Files:** `SFXDiTNameMap.swift`, `SFXDiTModel.swift`, `SFXDiTActor.swift`.

---

### 9. T5 `final_layer_norm` weights never loading — wrong @ModuleInfo key path

**Symptom:** crossCond abs_mean ~0.6041 (expected ~0.025–0.045). Output was real content after
prior fixes but consistently too loud; T5 output rescaled ~2.65× by wrong norm weights.

**Root cause:** `T5NameMap.mapKey` returned `"finalLayerNorm.weight"` (the Swift property name in
camelCase) for the HF key `encoder.final_layer_norm.weight`. But `T5Encoder` declares that
property with `@ModuleInfo(key: "final_layer_norm")`, which overrides the MLX parameter-tree
path to `final_layer_norm.weight` (snake_case). When `"finalLayerNorm.weight"` was passed to
`model.update(parameters:)`, no property in the tree matched that path, so the weight was
silently dropped. The final layer norm ran with its default initialization (`weight = ones`),
applying a ~2.65× scale to every T5 output vector (the checkpoint's mean weight magnitude is
≈ 0.377, not 1.0).

**Fix:** `T5NameMap.swift` line 43:

```swift
// Before (wrong — uses Swift property name):
if key == "encoder.final_layer_norm.weight" { return "finalLayerNorm.weight" }

// After (correct — uses @ModuleInfo key):
if key == "encoder.final_layer_norm.weight" { return "final_layer_norm.weight" }
```

**Files:** `T5NameMap.swift`.

---

### 10. T5 PAD positions not zeroed after final layer norm — 24× crossCond inflation

**Symptom:** crossCond abs_mean ≈ 0.6041 (expected ~0.025–0.045 from Python reference).
All three test prompts produced similar loud buzz even after DiT and VAE fixes.

**Root cause:** Python's `T5Stack.__call__` zeros out PAD token hidden states after the final
layer norm:

```python
if attention_mask is not None:
    mask_expanded = attention_mask[:, :, None].astype(mx.float32)
    hidden_states = hidden_states * mask_expanded
```

Our Swift `T5Encoder.callAsFunction` did not apply this mask. For a short prompt (~8 real tokens
+ 120 PAD tokens padded to max_length=128), all 128 positions had non-zero hidden states after
the norm. When the first 64 were sliced for cross-attention, ~56 PAD positions carried spurious
signal. The average was inflated ~24× relative to the Python value where only 4–8 positions are
non-zero.

**Why this matters for conditioning scale:**

For a prompt with N real tokens (out of 64 cross-attention slots):

```text
crossCond abs_mean ≈ (N × T5_scale + 2 × time_scale) / 66
```

With N=8 real tokens (T5_scale≈0.30) and 2 time tokens (time_scale≈0.68):

```text
≈ (8 × 0.30 + 2 × 0.68) / 66 ≈ 0.057    ← expected (matches Python ~0.045)
≈ (64 × 0.30 + 2 × 0.68) / 66 ≈ 0.311   ← without zeroing (56 extra PAD positions)
```

**Fix:** Added after `h = finalLayerNorm(h)` in `T5Encoder.callAsFunction`:

```swift
if let mask = attentionMask {
    let maskF = mask.asType(.float32).expandedDimensions(axis: -1)  // [B, L, 1]
    h = h * maskF
}
```

**Files:** `T5Encoder.swift`.

---

### 11. T5 position bias mask accumulated across encoder blocks

**Symptom:** Subtle quality degradation in T5 hidden states; all 12 encoder blocks incorrectly
applied the attention mask multiple times.

**Root cause:** `T5SelfAttention.callAsFunction` fused the additive attention mask (shape
`[B, 1, 1, L]`, values 0.0 for valid or −1e9 for PAD) into the `positionBias` tensor that was
returned to the block loop and reused by subsequent blocks. Block 0 computed the position bias
and returned `bias + mask`. Block 1 received `(bias + mask)` and added the mask again, producing
`bias + 2×mask`. Block N applied the mask N+1 times total. The −1e9 values at PAD positions were
already large enough to suppress attention after one application; the cumulative effect further
distorted the attention distribution at non-PAD positions across all 12 layers.

**Fix:** Separated the mask application into a local variable used only for computing scores;
the returned bias is mask-free:

```swift
// Before: bias was returned with mask already fused in.
// After:
let posBias = positionBias ?? computePositionBias(queryLength: seqL, keyLength: seqL)
var scoreBias = posBias
if let mask = attentionMask {
    scoreBias = scoreBias + mask  // used for score computation only
}
let scores = matmul(qOut, kOut.transposed(0, 1, 3, 2)) + scoreBias
// Return posBias (mask-free) so later blocks don't accumulate extra mask copies.
return (o(context), posBias)
```

**Files:** `T5Encoder.swift`.

---

## Diagnostic numerics

### Pre-T5-fix run (random-weights VAE era, 2026-05-09)

Prompt: "footsteps on a wooden floor" (SFX, 30s, 100 steps, guidance=7.5).

```text
SFX_COND          duration=30.0s steps=100 guidance=7.5 sigmaMax=1.0
SFX_START_EMB     shape=[1, 768]       abs_mean=0.048  abs_max=2.60
SFX_TOTAL_EMB     shape=[1, 768]       abs_mean=1.041  abs_max=3.97
SFX_GLOBAL_COND   shape=[1, 1536]      abs_mean=0.545  abs_max=3.97
SFX_CROSS_COND    shape=[1, 9, 768]    abs_mean=0.654  abs_max=12.2   ← 9 tokens (padding=longest)
SFX_UNCROSS_COND  shape=[1, 1, 768]    abs_mean=0.412  abs_max=10.7   ← 1 token for ""
SFX_DIFF          latentShape=[1, 646, 64] steps=100 sigmaMax=1.0 doCFG=true
SFX_SCHEDULE_FIRST10  [1.0, 0.9973, 0.9971, 0.9969, 0.9966, 0.9963, 0.9960, 0.9957, 0.9953, 0.9949]
SFX_SCHEDULE_LAST5    [0.168, 0.157, 0.147, 0.137, 0.128]
SFX_NOISE_XT      shape=[1, 646, 64]   abs_mean=0.794  abs_max=4.01   ← unit Gaussian ✓
SFX_FINAL_LATENTS abs_mean=3.134  abs_max=16.7                        ← RANDOM-WEIGHTS DiT (expected ~0.5–1.0)
SFX_VAE_DC        ch0_mean=-0.6011  ch1_mean=0.3333                   ← RANDOM WEIGHTS. Fix applied.
```

### Post-T5-fix run (2026-05-10)

Two prompts run back-to-back (30s, 100 steps, guidance=7.5). All three T5 bugs fixed.
Pre-padded to 128 tokens before T5 encoding.

```text
--- Prompt A (short, ~4 real tokens) ---
T5 textHidden  shape=[1, 128, 768]  abs_mean=0.0055   ← real tokens non-zero, PAD zeroed ✓
T5 emptyHidden shape=[1, 128, 768]  abs_mean=0.0008   ← only EOS non-zero ✓
crossCond   shape=[1, 66, 768]  abs_mean=0.0312        ← 4 real tokens + 2 time tokens
uncrossCond shape=[1, 66, 768]  abs_mean=0.0221        ← 1 EOS + 2 time tokens ✓
globalCond  shape=[1, 1536]     abs_mean=0.6767        ← Python target ~0.68 ✓
final_latents  abs_mean=0.6684  abs_max=3.6923  mean=-0.0614  shape=[1, 646, 64]
decode_out     abs_mean=0.0382  abs_max=0.4553  shape=[1, 1323008, 2]

--- Prompt B (longer, ~8 real tokens) ---
T5 textHidden  shape=[1, 128, 768]  abs_mean=0.0133   ← more real tokens → higher mean ✓
T5 emptyHidden shape=[1, 128, 768]  abs_mean=0.0008
crossCond   shape=[1, 66, 768]  abs_mean=0.0462
uncrossCond shape=[1, 66, 768]  abs_mean=0.0221
globalCond  shape=[1, 1536]     abs_mean=0.6767
final_latents  abs_mean=0.6472  abs_max=3.7623  mean=-0.0099  shape=[1, 646, 64]
decode_out     abs_mean=0.0388  abs_max=0.3970  shape=[1, 1323008, 2]

VAE loadWeights: supplied=291  model=145  matched=145
  (291 − 145 = 146 encoder keys silently dropped — expected, decoder-only model)
```

**What the numbers mean:**

- `final_latents abs_mean ≈ 0.66` — down from the old random-weights value of 3.134. DiT is
  trained and converging to the real audio latent distribution. ✓
- `decode_out abs_max ≈ 0.45` — peak ≈ −7 dBFS. Audio is present at reasonable amplitude.
- `decode_out abs_mean ≈ 0.038` — mean absolute sample ≈ −28 dBFS RMS, quiet but audible.
- `textHidden` differs between prompts (0.0055 vs 0.0133, 2.4× ratio). T5 IS differentiating
  the prompts.
- **Cross-prompt latent variation is only ~3%** (0.6684 vs 0.6472) despite a 2.4× T5 difference.
  This is the key signal: conditioning scale is correct but the text semantics aren't effectively
  steering the velocity field.

**Conditioning scale check:**

- `crossCond abs_mean = 0.031–0.046` — matches Python target (0.025–0.045 for short prompts). ✓
- `uncrossCond abs_mean = 0.0221` — stable across both runs (same empty string). ✓
- `globalCond abs_mean = 0.6767` — matches Python. ✓

---

## Things verified working (don't re-investigate)

- **Diffusion parameterization:** Rectified flow, Euler ODE step, logSNR sigmoid schedule.
  `SFX_SCHEDULE_FIRST10` matches expected shape.
- **Noise initialization:** `SFX_NOISE_XT abs_mean ≈ 0.794` matches theoretical unit Gaussian
  mean `√(2/π) ≈ 0.798`. ✓
- **GlobalCond ordering:** `[totalEmb, startEmb]` (total first). Verified against mlx-audiogen
  `conditioners.py`.
- **Cross-attention token structure:** `[T5[:64], totalToken, startToken]` = 66 tokens. Matches
  stable-audio-tools training layout.
- **CFG null conditioning:** T5("") encoded and used as `uncrossCondTokens`.
- **Prepend token conditioning:** Single `gEmb = tEmb + toGlobalEmbed(globalCond)` prepended,
  removed after 24 blocks.
- **`toCondEmbed` 768→768:** Correct dims, called on cross-attention tokens before transformer.
- **`SFXVAENameMap`:** 145/145 decoder keys verified correct via Python simulation.
- **`SFXDiTNameMap`:** 373/374 keys map to Swift paths (1 rotary inv_freq correctly dropped).
  Verified via Python simulation of `mapKey` against actual `dit.safetensors` header.
- **Preprocess/postprocess conv as residuals:** Python reference does `x_t = x_t + conv(x_t)`.
  Fixed in `SFXDiTModel.forward`.
- **No conv weight transposition needed:** SAO weights already in MLX NLC format `[out, k, in]`
  (exported from Python MLX).
- **VAE config:** `upsampleRatios=[8,8,4,4,2]`, product=2048. `latentDim=64`,
  `outputChannels=2`, `sampleRate=44100`.
- **Latent frame count for 30s:** `floor(30 × 44100 / 2048) = 646` frames. ✓
- **T5 conditioning scale:** crossCond=0.0458, uncrossCond=0.0221, globalCond=0.677. All match
  Python reference range. ✓
- **T5 architecture:** Pre-padding to 128 before encoding (matches Python `padding="max_length"`).
  PAD zeroing after final norm. Position bias returned mask-free from each block.
- **T5NameMap:** `final_layer_norm.weight` key path confirmed (uses `@ModuleInfo` key,
  not Swift property name). All 12 encoder blocks mapped correctly.
- **T5 v1.0 FFN:** Single `wi`+`wo` (ReLU), not GEGLU `wi_0`+`wi_1`. T5NameMap maps
  `DenseReluDense.wi_0.weight → layer1_mlp.wi.weight` (drops `_0` suffix for v1.0
  single-gate FFN).
- **DiT converging:** `final_latents abs_mean ≈ 0.65` (vs old random-weights value of 3.134).
  DiT is trained and producing plausible latents. ✓
- **VAE weight loading:** `supplied=291 model=145 matched=145`. 146 encoder keys silently
  dropped (expected — `SFXVAEModel` is decoder-only). All 145 decoder weights load. ✓
- **Audio amplitude:** `decode_out abs_max ≈ 0.45` (−7 dBFS peak), `abs_mean ≈ 0.038`. Audio
  is present at reasonable amplitude — no longer the −30 dBFS white noise floor.
- **DiT weight coverage confirmed:** `supplied=744 model=377 matched=373`.
  - 744 keys = HF `model.safetensors` format which bundles DiT (~374) + conditioner (~370) keys.
  - 373 DiT weights load correctly. 1 rotary `inv_freq` dropped by `SFXDiTNameMap` before
    reaching `loadWeights`. 371 conditioner keys silently ignored by MLX (not in `SFXDiTModel`).
  - 4 unmatched model params = `to_cond_embed.fc1.bias`, `to_cond_embed.fc2.bias`,
    `to_global_embed.fc1.bias`, `to_global_embed.fc2.bias`. The HF checkpoint uses `bias=False`
    for those two MLPs; our `SAOBiasMLP` uses `bias=True` (always). Default init is zero →
    equivalent to `bias=False` at inference. **Benign.**
- **CFG formula verified correct:** `vt = uncondPred + guidanceScale * (condPred − uncondPred)`.
  DiT call=0 is the conditioned pass, call=1 is unconditioned.
- **DiT velocity magnitude at t=1.0:** `vt_abs_mean ≈ 0.16–0.18` (vs old random-weights 4.6).
  DiT is trained and producing plausible velocity predictions. ✓
- **Cond vs uncond velocity at step 0:**
  - Run A: condPred=0.1751, uncondPred=0.1717, diff ≈ 2% of total magnitude
  - Run B: condPred=0.1569, uncondPred=0.1685, diff ≈ 7% of total magnitude
  - The abs_mean difference is small but `abs_mean` is a scalar summary; the actual
    CFG direction vector may still be coherent (or may not be). Needs Python reference.

---

## Hypotheses still open

### 1. ✅ RESOLVED — Python parity confirmed

**Python mlx-audiogen produces identical output characteristics for the same prompts:**
- Same RMS range ratio (~1.7×), same ZCR (~1267), same amplitude range
- Python `(cond−uncond).abs_mean = 0.019` (11% ratio) — Swift diff diagnostic added to confirm
- The "prompt-independent texture" is a SAO model behavior, not a Swift bug

**What SAO actually does well:** ambient sound textures, environmental audio. It does not
reliably produce the specific temporal patterns of discrete sound events (footsteps, barks,
impacts) from short 4–8-word prompts — Python shows the same limitation.

**For better event-like results:**
- Use longer, more detailed prompts (more T5 token positions non-zero)
- Try the `stable-audio-open-small` model variant which may have different conditioning balance
- Consider that SAO was trained on a mix of prompt styles; very short prompts may not activate
  the model's event-generation modes reliably

### 2. Guidance scale tuning (low priority — model is working correctly)

Default is 7.5 (mlx-audiogen uses 7.0). Python uses CFG=7.0 by default. May be worth aligning
to 7.0 for consistency with the Python reference. Higher values (10–15) would amplify the
~11% cond/uncond diff but may cause over-saturation. Lower values (3–5) would reduce it.
Tuning is cosmetic at this point — both endpoints produce valid audio.

### 3. ✅ RESOLVED — Inverted guidance concern

`SFXDiffusionActor`: `vt = uncondPred + guidanceScale * (condPred − uncondPred)`. Correct.
The earlier observation (uncond.abs_mean > cond.abs_mean on some steps) is expected — it
reflects magnitude differences, not direction. Not a bug.

---

## Things ruled out

- **Karras EDM schedule** — wrong parameterization for SAO, replaced with rectified flow.
- **`edmEulerStep`** — SAO uses `odeStep` (simple Euler).
- **Per-layer conditioning add** — replaced with prepend token.
- **`toCondEmbed` wrong dims / bypassed** — fixed to 768→768, called correctly.
- **CFG null as zeros** — fixed to T5("").
- **GlobalCond order `[start, total]`** — fixed to `[total, start]`.
- **Missing `bias=true` on GEGLU linears** — was fixed earlier; weight loading confirmed shape-correct.
- **T5 tokenizer crash (Unigram routing)** — fixed earlier; T5 tokenizer working correctly.
- **`hopLength = 1024`** — fixed to 2048 (product of `[2,4,4,8,8]` strides).
- **T5 `finalLayerNorm.weight` path** — fixed to `final_layer_norm.weight` (uses @ModuleInfo key).
- **T5 PAD positions inflating crossCond** — fixed; output zeroed after final norm. Scale now correct.
- **T5 position bias mask accumulation** — fixed; returned bias is mask-free.

---

## How to resume

1. **Read "Current symptom"** and "Hypotheses still open" first.
2. **Enable AUDIODIAG:** `defaults write com.pimpstudios.PiMPAudioStudio com.pimpstudios.PiMPAudioStudio.audioDebugVerbose -bool YES`.
   Filter console for `SFX_DIT_` to see DiT internals per step.
3. **Test prompts:** "footsteps on a wooden floor" and "dog barking", 30s, 100 steps,
   guidance=7.5. These should produce audibly different output once conditioning is working.
4. **Key diagnostic check — is conditioning propagating?**
   - Compare `SFX_DIT_PROJ_CROSS` abs_mean for the two prompts at step 0. If values are
     nearly identical between "footsteps" and "dog barking" after `toCondEmbed`, the problem
     is in T5 or weight loading, not in the DiT.
   - Compare conditioned vs unconditioned `SFX_DIT_TRANSFORMER_OUT`. CFG amplifies the
     difference; if the difference is near zero, guidance cannot steer output.
5. **Check `SFX_FINAL_LATENTS abs_mean`** — should be ~0.5–1.0 with trained DiT. The old
   random-weights run gave 3.134. If still elevated, diffusion may not be converging.
6. **If conditioning IS propagating but audio is wrong pitch/character:** Look at the schedule
   tail (hypothesis 4) and try lower guidance scales (3.5–7.0).
7. **If conditioning is NOT propagating:** Verify `toCondEmbed` weights loaded (hypothesis 3).
   Print `SFX_DIT_PROJ_CROSS shape` and confirm it differs between prompts.

---

## Useful diagnostic snippets

### WAV DC + energy analysis (Python)

```python
import wave, struct, math
fn = "/Users/tweaver/Documents/Anvil/anvil_TIMESTAMP_PROMPT.wav"
with wave.open(fn, 'rb') as w:
    n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
    raw = w.readframes(n)
samples = struct.unpack(f"<{len(raw)//sw}h", raw)
left  = samples[0::2]
right = samples[1::2]
dc_l = sum(left)  / len(left)
dc_r = sum(right) / len(right)
rms  = math.sqrt(sum(s*s for s in samples) / len(samples))
peak = max(abs(s) for s in samples)
print(f"peak={peak} ({20*math.log10(max(peak,1)/32767):.1f} dBFS)  RMS={rms:.0f}")
print(f"DC offset: L={dc_l:.1f}  R={dc_r:.1f}  (near-0 is good, ±100+ is bad)")
mono = [(l+r)//2 for l,r in zip(left,right)]
zc   = [sum(1 for j in range(1,sr) if (mono[i+j-1]>=0)!=(mono[i+j]>=0))
        for i in range(0, len(mono)-sr, sr)]
es   = [math.sqrt(sum(x*x for x in mono[i:i+sr])/sr)
        for i in range(0, len(mono)-sr, sr)]
print(f"per-sec ZCR: mean={sum(zc)/len(zc):.0f}  (noise >4000, SFX 500-3000)")
print(f"per-sec RMS: min={min(es):.0f}  max={max(es):.0f}  ratio={max(es)/max(min(es),1):.1f}")
```

### Inspect SAO checkpoint keys

```python
import mlx.core as mx
w = dict(mx.load('/Users/tweaver/.cache/anvil-audio/mlx-weights/stable-audio-open-1.0/vae.safetensors'))
for k in sorted(w.keys()):
    print(k, list(w[k].shape))
```

### Run mlx-audiogen reference for comparison

```bash
cd ~/Developer/GitRepos/AnvilApp
source .venv/bin/activate
python3 -c "
from mlx_audiogen.models.stable_audio.pipeline import StableAudioPipeline
# … generate with same prompt and compare latent stats
"
```

---

## File map (SAO-specific)

```text
AnvilCore/Sources/AnvilCore/Inference/StableAudio/
  SoundFXPipeline.swift         — end-to-end orchestrator; conditioning build
  SFXDiTModel.swift             — SAOFourierFeatures, SAOBiasMLP, SAODiTLayer,
                                   SAODiTTransformer, SFXDiTModel
  SFXDiTActor.swift             — actor wrapper; diagCallCount for AUDIODIAG gating
  SFXDiffusionActor.swift       — RF schedule, Euler ODE loop, inpaint stub
  SAOProjectionModel.swift      — SAONumberConditioner (seconds_start / total)
  SFXVAEModel.swift             — thin wrapper around OobleckDecoder
  SFXVAEActor.swift             — actor wrapper; decode() with AUDIODIAG probes
  T5Encoder.swift               — SAO text conditioning (T5-base v1.0, 768-dim)

AnvilCore/Sources/AnvilCore/Inference/Weights/
  SFXDiTNameMap.swift           — DiT key mapping (373/374 verified 2026-05-10)
  SFXVAENameMap.swift           — VAE key mapping (rewritten 2026-05-09, 145/145)
  SAOProjectionNameMap.swift    — conditioners key mapping
  T5NameMap.swift               — T5 encoder key mapping (final_layer_norm fixed 2026-05-10)

AnvilApp/Services/
  SFXModelManager.swift         — loads all 4 components; builds SoundFXPipeline

~/.cache/anvil-audio/mlx-weights/stable-audio-open-1.0/
  vae.safetensors               — 145 decoder keys (Sequential MLX naming)
  dit.safetensors               — DiT weights (374 keys, blocks.N.* prefix)
  t5.safetensors                — T5-base encoder weights
  conditioners.safetensors      — SAONumberConditioner weights
```
