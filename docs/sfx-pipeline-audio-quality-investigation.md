# Stable Audio Open SFX Pipeline — Audio Quality Investigation

**Last updated:** 2026-05-09
**Status:** Pipeline runs end-to-end. Has never produced real audio yet. As of the end of this session we fixed the **root cause of all output being noise: the SFX VAE was running on default-initialized random weights** — none of the 145 decoder weight tensors were loading. The next test run is expected to be the first time we hear actual sound effects.

> Use this doc to resume troubleshooting after a context reset. Skim "Current symptom" + "Hypotheses still open" first.

> **Quick orientation:** the `[AUDIODIAG]` toggle (same as ACE-Step — `defaults write com.pimp.AnvilApp audioDebugVerbose -bool YES`) is wired into the SFX pipeline. Filter Xcode console for `SFX_` to see only SAO-path lines. The diagnostic dump that led to finding the VAE weight-loading bug is preserved verbatim in "Diagnostic numerics" below.

---

## Project context

Porting **Stable Audio Open 1.0** (text→sound-effects diffusion model) from Python/mlx-audiogen into the same Swift + MLX-Swift app as ACE-Step. Pipeline:

```
text prompt
  → T5-base tokenizer + encoder  →  [B, L, 768] hidden states
  → SAONumberConditioner (seconds_start=0, seconds_total=duration)
      → startEmb [1, 768], totalEmb [1, 768]
  → Build conditioning:
      crossCond  = T5 hidden states  [B, 128, 768]
      globalCond = cat([totalEmb, startEmb], axis=-1)  [B, 1536]
  → SFXDiffusionActor (rectified flow, 100 steps, logSNR schedule)
      → each step: SFXDiTModel.forward(latents, timestep, crossCond, globalCond)
  → OobleckDecoder (VAE, strides [2,4,4,8,8] = 2048 hop, 44.1 kHz stereo)
  → audio samples [B, T_audio, 2]
```

Upstream reference:
- Local Python source: `~/Developer/GitRepos/AnvilApp/.venv/.../mlx_audiogen/models/stable_audio/`
  - `pipeline.py` — top-level generate(), sigma_max=1.0, negative_prompt=""
  - `conditioners.py` — T5[:64]+totalToken+startToken for cross-attn; [total,start] globalCond order
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
| Cross-attn | T5 768-dim tokens as K/V, `to_cond_embed` 768→768 projection |
| Global cond | `seconds_start` + `seconds_total` via `SAOProjectionModel` → `[B,1536]` |
| Conditioning style | Prepend token (NOT per-layer AdaLN broadcast) |
| VAE | OobleckDecoder, strides `[2,4,4,8,8]`, hop=2048 |
| Sample rate | 44,100 Hz stereo |
| Latent frame rate | 44100 / 2048 ≈ 21.5 fps |
| Diffusion | Rectified flow (NOT Karras EDM), t ∈ [1, 0] |
| Schedule | logSNR sigmoid, `logsnr_max = -6.0`, `logsnr_min = 2.0` |
| CFG null | T5("") hidden states (empty string, same tokenizer) |

---

## Current symptom (end of 2026-05-10 session)

**VAE fix tested — confirmed working (DC near zero). Output is now very faint white noise at −30.4 dBFS. Root cause found and fixed (commit pending): DiT transformer weights were silently dropped.**

**Timeline:**

| Run | Output | Root cause |
|---|---|---|
| Pre-session | All static | Karras EDM schedule on RF checkpoint |
| Mid-session | Buzz, prompt-independent | Per-layer cond broadcast |
| Later | Buzz + −0.22 DC | Time tokens in cross-attention |
| 2026-05-09 | Loud buzz + large ±DC (ch0=−0.60, ch1=+0.33) | VAE on random weights |
| 2026-05-10 (post-VAE fix) | Very faint (~−30 dBFS) white noise, DC near 0 | **DiT transformer on random weights** |

Next test (after DiT fix commit) is expected to produce actual sound effects.

### Earlier symptoms (chronological)

- **Pure static** (pre-session): Karras EDM schedule `σ ∈ [500, 0.3]` applied to a rectified-flow checkpoint that expects t ∈ [1, 0]. Output was indistinguishable from random noise.
- **Buzz (constant amplitude, both prompts identical)**: Per-layer conditioning broadcast (24× add to hidden states) was drowning the cross-attention signal. Fixed by switching to prepend-token approach.
- **Buzz + static, −0.22 DC offset**: Time tokens (totalToken, startToken) were being appended to the cross-attention sequence, introducing a fixed bias the model was never trained to handle via cross-attention. Fixed by using T5-only cross-attention.
- **Buzz + static, large opposite-sign DC (ch0=−0.60, ch1=+0.33)**: VAE running on random initialized weights. Fixed by rewriting `SFXVAENameMap`.

---

## Root causes found and fixed

### 1. Wrong diffusion parameterization — Karras EDM vs rectified flow

**Symptom:** All static, no structure.

**Bug:** `SFXDiffusionActor` used a Karras sigma schedule `σ_max = 500, σ_min = 0.3` (EDM parameterization) and the `edmEulerStep` variational ODE. The SAO checkpoint is a rectified-flow model with t ∈ [1, 0] and the simple Euler step `x_next = x_t + (t_next − t_curr) * v_t`.

**Fix:** Replaced Karras schedule with `generateRFSchedule` (logSNR sigmoid, pins `t[0] = sigmaMax = 1.0`). Replaced `edmEulerStep` with `odeStep`.

**Files:** `SFXDiffusionActor.swift`, `DiffusionSampler.swift`, `Types.swift` (`sigmaMax` added to `SFXDiffusionConfig`).

---

### 2. Per-layer conditioning broadcast instead of prepend token

**Symptom:** Constant-amplitude buzz with the same waveform regardless of prompt.

**Bug:** `SFXDiTModel.forward` was adding the global conditioning embed to every layer's hidden states (24 residual additions). This overwhelmed the cross-attention signal entirely. The mlx-audiogen reference prepends the conditioning as a single extra token at position 0 of the sequence, then removes it after all 24 blocks.

**Fix:** `SAODiTTransformer.callAsFunction` now:
1. Prepends `prependToken` → sequence becomes `[B, T+1, H]`
2. Runs all 24 layers
3. Slices away position 0 → back to `[B, T, H]`

**Files:** `SFXDiTModel.swift`.

---

### 3. `toCondEmbed` bypassed and wrong dimensions

**Symptom:** Weight load shape mismatch / silent wrong output.

**Bug:** `toCondEmbed` (the non-linear projection applied to cross-attention tokens before they enter the blocks) was initialized as `SAOBiasMLP(768, 1536)` — wrong output dimension. The checkpoint has `[768, 768]` weights. The projection was also never called before the transformer.

**Fix:** Changed init to `SAOBiasMLP(768, 768)`. Added `projCross = toCondEmbed(crossCond)` call in `SFXDiTModel.forward` before passing to the transformer.

**Files:** `SFXDiTModel.swift`.

---

### 4. CFG null conditioning was zeros

**Symptom:** Guidance scale had no effect / added noise instead of directing output.

**Bug:** The unconditional branch for classifier-free guidance used `MLXArray.zeros(like: crossCond)`. The correct null conditioning for SAO is T5-encoding the empty string `""`.

**Fix:** `SoundFXPipeline.generate(text:...)` now encodes both the prompt AND `""` via T5. The empty-string hidden states are passed as `uncondText` through to `SFXDiffusionActor.sample(uncrossCondTokens:)`.

**Files:** `SoundFXPipeline.swift`, `SFXDiffusionActor.swift`.

---

### 5. GlobalCond ordering wrong — [start, total] instead of [total, start]

**Symptom:** Systematic conditioning mismatch; seconds_start and seconds_total embeddings fed in swapped order to the model.

**Bug:** Swift code concatenated `[startEmb, totalEmb]`. The mlx-audiogen reference concatenates `[totalEmb, startEmb]` (total first).

**Fix:** `let globalCond = concatenated([totalEmb, startEmb], axis: -1)`.

**Files:** `SoundFXPipeline.swift`.

---

### 6. Time tokens appended to cross-attention — large DC offset

**Symptom:** −0.22 DC offset in decoded audio (consistent across both prompts), buzz and static.

**Bug:** mlx-audiogen's `conditioners.py` appends `totalToken` and `startToken` (each `[1, 1, 768]`) to the T5 hidden states before cross-attention, giving 66 tokens `[T5[:64], totalToken, startToken]`. However, stable-audio-tools (the training code) was trained with T5-only cross-attention — time tokens flow exclusively through `globalCond → to_global_embed → prepend token`. Appending them to cross-attention adds a fixed bias the model never saw during training, which integrates through 24 blocks and decodes to a DC offset.

**Fix:** `crossCond = textHidden` (full `[B, 128, 768]` T5 output, no time tokens appended). `uncrossCondTokens = uncondText` (just the T5("") hidden states).

**Note:** The mlx-audiogen port's inclusion of time tokens in cross-attention may work for *their* specific fine-tuned checkpoint; it was wrong for the original SAO 1.0 weights we load.

**Files:** `SoundFXPipeline.swift`.

---

### 8. DiT transformer weights completely missing — all 360 block weights silently dropped

**Symptom:** Very faint audio (−30.4 dBFS) with ZCR ≈ 0.5 (statistical white noise). DC near zero (confirming VAE is working). Output identical across prompts.

**Root cause:** `SFXDiTNameMap.mapKey` did not handle the `blocks.N.*` key prefix used in the mlx-audiogen `dit.safetensors` export. The checkpoint stores all 24 transformer layers as top-level `blocks.0…23.*` keys (360 tensors) with **no** `model.model.` or `transformer.` prefix. The Swift `SFXDiTTransformer` stores those layers at `transformer.layers.N.*`. Because `mapKey` passed `blocks.0.*` through unchanged, MLX tried to set `blocks.0.*` on the top-level `SFXDiTModel`, found no property named `blocks`, and silently dropped all 360 tensors. The model ran with default-initialized random transformer weights → garbage velocity predictions → random walk in latent space → 4× out-of-distribution latents (abs_mean ≈ 3.1 vs expected ~0.8) → Snake activations in high-frequency oscillation regime → near-zero white-noise audio.

**Additional secondary bug:** The Python DiT applies preprocess_conv and postprocess_conv as **residual corrections** (`x_t = x_t + self.preprocess_conv(x_t)`). Our Swift `forward()` was doing `let h = preprocessConv(latents)` (no add-back), discarding the input signal and passing only the correction to the transformer.

**Diagnosis method:** Inspected `dit.safetensors` header directly; found 374 keys, all 360 block-layer keys use `blocks.N.*` prefix (no `model.model.`). Python simulation of `mapKey` showed 360/373 loadable keys being silently dropped.

**Fix:**
1. `SFXDiTNameMap.mapKey` — added step 6: `blocks.N.* → transformer.layers.N.*`
2. `SFXDiTModel.forward` — `latents + preprocessConv(latents)` and `out + postprocessConv(out)`
3. `SFXDiTActor.loadWeights` — added `supplied/model/matched` coverage diagnostic

**Expected coverage after fix:** supplied=373, model=373, matched=373 (with 1 rotary inv_freq dropped before reaching `loadWeights`).

**Files:** `SFXDiTNameMap.swift`, `SFXDiTModel.swift`, `SFXDiTActor.swift`.

---

### 7. VAE weight loading completely broken — all 145 weights silently dropped

**Symptom:** Large opposite-sign DC offsets (ch0 ≈ −0.60, ch1 ≈ +0.33). VAE output statistically indistinguishable from a randomly-initialized network. DC offsets persisted regardless of diffusion conditioning changes.

**Root cause:** `loadSFXVAEWeights` in `WeightConverter` called `VAENameMap.mapKey`, which is designed for the ACE-Step checkpoint key format (`decoder.block.0.conv_t1.*`). The SAO `vae.safetensors` uses MLX Sequential naming (`decoder.layers.layers.0.*`, `decoder.layers.layers.1.layers.layers.0.*`, etc.). `VAENameMap.mapKey` only does `snake_case → camelCase`; it does not translate the numeric Sequential path into the named Swift properties (`conv1`, `blocks.0`, `snake1`, `convT1`, `resUnit1`, etc.). Every key produced a path like `layers.layers.0.weight` that matched nothing in `OobleckDecoder`. MLX's `model.update(parameters:)` silently drops unmatched keys. Result: all 145 decoder weights went in the bin and the model ran on default-initialized random values.

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

**Fix:** Rewrote `SFXVAENameMap` with a proper three-level path translator (decoder root → block → ResUnit). Verified all 145 decoder keys map correctly via a Python simulation against the real checkpoint before building. Updated `loadSFXVAEWeights` to use `SFXVAENameMap.mapKey` with no weight-norm fusion or axis transposition (SAO weights are already in MLX NLC format).

**Files:** `SFXVAENameMap.swift` (complete rewrite), `WeightConverter.swift` (`loadSFXVAEWeights`).

---

## Diagnostic numerics (last AUDIODIAG run — random-weights VAE, before fix)

Generated 2026-05-09. Prompt: "footsteps on a wooden floor" (SFX, 30s, 100 steps, guidance=7.5).

```
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
SFX_DIT_PREPROC   shape=[1, 646, 64]   abs_mean=0.139  abs_max=0.716
SFX_DIT_TIMESTEP  shape=[1]            abs_mean=1.0                   ← step 0, t=1.0
SFX_DIT_TEMB      shape=[1, 1536]      abs_mean=1.106  abs_max=11.1
SFX_DIT_GLOBAL_EMB shape=[1, 1, 1536]  abs_mean=1.231  abs_max=11.3
SFX_DIT_PROJ_CROSS shape=[1, 9, 768]   abs_mean=0.340  abs_max=7.55   ← conditioned
SFX_DIT_TRANSFORMER_OUT  [1, 646, 64]  abs_mean=0.462  abs_max=1.61   ← conditioned
SFX_DIT_OUT       [1, 646, 64]         abs_mean=0.361  abs_max=1.29   ← conditioned
SFX_DIT_PROJ_CROSS [1, 1, 768]         abs_mean=0.414  abs_max=3.65   ← unconditioned
SFX_DIT_TRANSFORMER_OUT  [1, 646, 64]  abs_mean=1.078  abs_max=3.25   ← unconditioned (LARGER than cond — suspicious)
SFX_DIT_OUT       [1, 646, 64]         abs_mean=0.820  abs_max=2.64   ← unconditioned
SFX_STEP[0]       sigma=1.0000  sigmaNext=0.9973
SFX_VT[0]         abs_mean=4.598  abs_max=17.18                       ← CFG-amplified velocity, plausible
SFX_XT[0]         abs_mean=0.793  abs_max=4.03                        ← barely changed (dt=0.0027)
SFX_STEP[1]       sigma=0.9973   sigmaNext=0.9971
SFX_VT[1]         abs_mean=4.662  abs_max=17.49
SFX_XT[1]         abs_mean=0.793  abs_max=4.03                        ← still barely changed
SFX_STEP[99]      sigma=0.1279   sigmaNext=0.0000
SFX_VT[99]        abs_mean=2.981  abs_max=24.2
SFX_XT[99]        abs_mean=3.134  abs_max=16.7                        ← grew significantly over 100 steps
SFX_FINAL_LATENTS abs_mean=3.134  abs_max=16.7
SFX_VAE_IN        abs_mean=3.134  abs_max=16.7
SFX_VAE_OUT       shape=[1, 1323008, 2]  abs_mean=0.526  abs_max=2.20
SFX_VAE_DC        ch0_mean=-0.6011  ch1_mean=0.3333                   ← RANDOM WEIGHTS. Fix applied.
```

**What the numbers told us:**

- `SFX_CROSS_COND shape=[1, 9, 768]` — tokenizer uses `padding=longest`, giving only 9 tokens for a short prompt. `SFX_UNCROSS_COND shape=[1, 1, 768]` — empty string produces just 1 token. This is expected and fine.
- `SFX_SCHEDULE_FIRST10` — first 10 steps all cluster near t=1.0 (step sizes ~0.0025). Most denoising happens in the later steps when dt is larger. This is by design (logSNR schedule allocates more steps at high noise).
- `SFX_DIT_TRANSFORMER_OUT` — unconditioned pass produces LARGER abs_mean (1.078) than conditioned (0.462). Suspicious but may be an artifact of the 1-token uncond vs 9-token cond cross-attention context size.
- `SFX_FINAL_LATENTS abs_mean=3.134` — latents grew from initial noise abs_mean 0.794 over 100 steps. Whether 3.134 is within the trained VAE's expected input range is unknown until we see output with real weights.
- `SFX_VAE_DC` — opposite-sign DC on left and right channels is the signature of randomly initialized Conv1d biases. This confirmed VAE weight loading was completely broken.

---

## Things verified working (don't re-investigate)

- **Diffusion parameterization:** Rectified flow, Euler ODE step, logSNR sigmoid schedule. `SFX_SCHEDULE_FIRST10` matches expected shape.
- **Noise initialization:** `SFX_NOISE_XT abs_mean ≈ 0.794` matches theoretical unit Gaussian mean `√(2/π) ≈ 0.798`. ✓
- **globalCond ordering:** `[totalEmb, startEmb]` (total first). Verified against mlx-audiogen `conditioners.py`.
- **Cross-attention token structure:** T5-only, no time tokens. Matches stable-audio-tools training.
- **CFG null conditioning:** T5("") encoded and used as `uncrossCondTokens`.
- **Prepend token conditioning:** Single `gEmb = tEmb + toGlobalEmbed(globalCond)` prepended, removed after 24 blocks.
- **`toCondEmbed` 768→768:** Correct dims, called on cross-attention tokens before transformer.
- **`SFXVAENameMap`:** 145/145 decoder keys verified correct via Python simulation.
- **`SFXDiTNameMap`:** 373/374 keys map to Swift paths (1 rotary inv_freq correctly dropped). Verified via Python simulation of `mapKey` against actual `dit.safetensors` header.
- **Preprocess/postprocess conv as residuals:** Python reference does `x_t = x_t + conv(x_t)`. Fixed in `SFXDiTModel.forward`.
- **No conv weight transposition needed:** SAO weights already in MLX NLC format `[out, k, in]` (exported from Python MLX).
- **VAE config:** `upsampleRatios=[8,8,4,4,2]`, product=2048. `latentDim=64`, `outputChannels=2`, `sampleRate=44100`.
- **Latent frame count for 30s:** `floor(30 × 44100 / 2048) = 646` frames. ✓ Matches `latentShape=[1, 646, 64]`.

---

## Hypotheses still open

### 1. First real audio — verify after DiT fix (highest priority)

Both the VAE fix (2026-05-09) and the DiT transformer fix (2026-05-10) are committed. Run both test prompts ("footsteps on a wooden floor", "waves crashing on a shore") and collect a new AUDIODIAG dump. Key things to look for:

- `[SFXDiT] loadWeights: supplied=373 model=373 matched=373` in Xcode console → confirms all weights loaded
- `SFX_VAE_DC` ch0/ch1 near 0 → VAE still working
- Output WAV should sound different for the two prompts (SHA1 diff + audible diff)
- Audio should have discernible content at reasonable amplitude (−12 to −6 dBFS range)
- `SFX_FINAL_LATENTS abs_mean` should now be much smaller than the old 3.134 (with trained DiT, latents converge to the real data distribution)

### 2. Latent scale / final latent distribution

With trained DiT, the final latents should represent encoded real audio. The old `abs_mean = 3.134` was from a random-weights DiT (random walk). After the fix, the final latents should be in the SAO VAE encoder's real output distribution (estimated ~0.5–1.0 abs_mean). If audio sounds clipped or distorted, compare `SFX_FINAL_LATENTS` against the Python mlx-audiogen reference run.

### 3. Unconditioned prediction larger than conditioned

`SFX_DIT_TRANSFORMER_OUT` uncond abs_mean (1.078) > cond abs_mean (0.462) in the old (random-weights) run. This data is now stale — the old run had a random transformer. Collect fresh AUDIODIAG from the first real run; if uncond is still larger than cond, investigate whether the empty-string T5 tokenization produces a useful null conditioning.

### 4. Guidance scale tuning

Default is 7.5. mlx-audiogen defaults to 7.0. Try 3.5 / 5.0 / 7.0 / 7.5 once real audio is confirmed.

---

## Things ruled out

- **Karras EDM schedule** — wrong parameterization for SAO, replaced with rectified flow.
- **`edmEulerStep`** — SAO uses `odeStep` (simple Euler).
- **Per-layer conditioning add** — replaced with prepend token.
- **`toCondEmbed` wrong dims / bypassed** — fixed to 768→768, called correctly.
- **CFG null as zeros** — fixed to T5("").
- **GlobalCond order `[start, total]`** — fixed to `[total, start]`.
- **Time tokens in cross-attention** — removed; causing DC offset.
- **Missing `bias=true` on GEGLU linears** — was fixed earlier; weight loading confirmed shape-correct.
- **T5 tokenizer crash (Unigram routing)** — fixed earlier; T5 tokenizer working correctly (produces 9 tokens for a short prompt).
- **`hopLength = 1024`** — fixed to 2048 (product of `[2,4,4,8,8]` strides).

---

## How to resume

1. **Read "Current symptom"** and "Hypotheses still open" first.
2. **Enable AUDIODIAG:** `defaults write com.pimp.AnvilApp audioDebugVerbose -bool YES`. Filter console for `SFX_`.
3. **Test prompts:** "footsteps on a wooden floor" (SFX mode) and "waves crashing on a shore" (ambient mode), 30s, 100 steps, guidance=7.5. These two prompts should produce audibly different output.
4. **If first run after VAE fix still sounds bad:** collect a new AUDIODIAG dump. Check:
   - `SFX_VAE_DC` near 0? (was −0.60/+0.33 → proves weights now loading)
   - `SFX_FINAL_LATENTS abs_mean` — compare against Python mlx-audiogen run on the same prompt
   - `SFX_DIT_TRANSFORMER_OUT` conditioned vs unconditioned — should cond be larger?
5. **If audio has content but is wrong pitch/tempo/character:** look at the schedule. The logSNR sigmoid schedule's `LAST5 = [0.168, 0.157, 0.147, 0.137, 0.128]` means the final step jumps from 0.128 to 0 (sigmaNext fallback). This is a 12.8% noise level wiped in one step. mlx-audiogen generates `steps+1` schedule values and uses the final one as `sigmaNext`; our code uses 0. May need to align.
6. **If audio is totally prompt-independent:** conditioning is broken. Check `SFX_CROSS_COND` is non-zero and different between prompts.
7. **Audit DiT weight loading:** add `supplied / model / matched` diagnostic to `SFXDiTActor.loadWeights` as next step after confirming real audio.

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

### Inspect SAO VAE checkpoint keys

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

```
AnvilCore/Sources/AnvilCore/Inference/StableAudio/
  SoundFXPipeline.swift         — end-to-end orchestrator; conditioning build
  SFXDiTModel.swift             — SAOFourierFeatures, SAOBiasMLP, SAODiTLayer,
                                   SAODiTTransformer, SFXDiTModel
  SFXDiTActor.swift             — actor wrapper; diagCallCount for AUDIODIAG gating
  SFXDiffusionActor.swift       — RF schedule, Euler ODE loop, inpaint stub
  SAOProjectionModel.swift      — SAONumberConditioner (seconds_start / total)
  SFXVAEModel.swift             — thin wrapper around OobleckDecoder
  SFXVAEActor.swift             — actor wrapper; decode() with AUDIODIAG probes
  T5Encoder.swift               — SAO text conditioning (T5-base, 768-dim)

AnvilCore/Sources/AnvilCore/Inference/Weights/
  SFXDiTNameMap.swift           — DiT key mapping (373/374 verified 2026-05-10)
  SFXVAENameMap.swift           — VAE key mapping (rewritten 2026-05-09, 145/145)
  SAOProjectionNameMap.swift    — conditioners key mapping
  T5NameMap.swift               — T5 encoder key mapping

AnvilApp/Services/
  SFXModelManager.swift         — loads all 4 components; builds SoundFXPipeline

~/.cache/anvil-audio/mlx-weights/stable-audio-open-1.0/
  vae.safetensors               — 145 decoder keys (Sequential MLX naming)
  dit.safetensors               — DiT weights
  t5.safetensors                — T5-base encoder weights
  conditioners.safetensors      — SAONumberConditioner weights
```
