# ACE-Step Commercial Quality Roadmap

**Date:** 2026-05-08
**Status:** Draft for discussion — Codex-reviewed 2026-05-18
**Author:** Grok analysis based on current codebase and
ace-step-audio-quality-investigation.md
**Related:** [ace-step-audio-quality-investigation.md](ace-step-audio-quality-investigation.md)

**Codex review note (2026-05-18):** This pass keeps the roadmap's overall
direction, but tightens Phase 0 into a smaller evidence-gathering milestone.
Changes are marked inline with "Codex review note" so future reviewers can see
what was adjusted and why.

---

## Executive Summary

Anvil has crossed the "recognizable music" threshold. The conditioning
stack, SFT prompt template, silence latent context, learned null condition
embedding for CFG, longest padding, and Turbo timestep schedule now produce
musically structured output instead of static or noise.

The remaining gap to commercial-grade quality (consistent prompt adherence,
harmonic coherence, transient clarity, professional loudness and stereo
image) is no longer a single structural bug. It is a combination of:

- Low-level numerical and architectural parity issues in the DiT and VAE
- A basic inference recipe that lacks the scheduled guidance, dynamic
  thresholding, and final-step refinements used in production systems
- VAE reconstruction as the final perceptual bottleneck
- Absence of a systematic perceptual evaluation loop

This document outlines a prioritized, evidence-based path from "works and
makes music" to "I would pay for this and use it professionally."

---

## Current State Assessment

### What Is Working Well

- Full weight loading parity for DiT, VAE, Qwen3, LyricEncoder, TimbreEncoder,
  and ConditionEncoder (only runtime-computed rotary buffers are intentionally
  missing).
- Conditioning pipeline produces prompt-dependent, non-zero encoder states
  with correct packing order (lyrics + timbre + text).
- APG implementation is byte-identical to the upstream MLX reference.
- Timestep schedule, scale-shift modulation, and cross-attention pre-norm all
  match v1.5 architecture.
- First recognizable musical outputs achieved with healthy velocity
  magnitudes, RMS dynamics, and prompt differentiation.

### What Still Sounds "AI-Generated"

- Limited long-range harmonic and structural coherence
- Soft transients and smeared high-frequency detail
- Narrow or unstable stereo image
- Prompt adherence that is directionally correct but rarely precise on
  specific instruments, energy, or arrangement
- Occasional over-saturation or loss of clarity at higher guidance values

These symptoms point to a **perceptual distribution mismatch** rather than a
complete failure of the forward pass.

---

## Root Causes of the Remaining Gap

### 1. VAE Reconstruction Fidelity (Highest Perceptual Impact)

The final decoded audio passes through `VAETiledDecode` and
`AutoEncoderOobleck`. Any tiling boundary artifacts, scale/phase mismatch, or
reconstruction bias directly degrades clarity, transients, and stereo imaging.

**Codex review note (2026-05-18):** Treat "VAE problem" as two separate tests.
`VAETiledDecode` only tiles when latent length exceeds 2048 frames; a normal
30-second ACE-Step generation is roughly 750 latent frames and bypasses tiling
entirely. Therefore, first compare direct non-tiled Swift decode against
upstream decode on identical 30-second latents. Then separately test a long
120-200 second latent that actually crosses the tiling threshold.

**Evidence:** No cross-runtime comparison of decoded audio from identical
latents has been performed since the breakthrough. The investigation doc
explicitly flags this as the top remaining hypothesis.

**Files:**

- `AnvilCore/Sources/AnvilCore/Inference/VAE/VAETiledDecode.swift`
- `AnvilCore/Sources/AnvilCore/Inference/VAE/VAEActor.swift`
- `AnvilCore/Sources/AnvilCore/Inference/VAE/VAEComponents.swift`

### 2. DiT Low-Level Parity (Patch Embed, GQA, Attention Details)

Small tensor layout, channel ordering, or mask construction errors that are
"close enough" to produce music will still accumulate across 24 layers and
8-50 steps, producing the characteristic "AI smear."

**Evidence:** The investigation doc lists patch embed/unembed ordering, GQA
repeat direction, q/k-norm placement, and sliding attention mask parity as
open items after conditioning was fixed.

**Files:**

- `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTModel.swift`
- `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift`
- `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift`

### 3. Inference Recipe Is Still Basic

Current `DiffusionConfig` and `DiffusionActor.generate` apply constant guidance
across all timesteps, use a fixed APG momentum path, and have no dynamic
thresholding or guidance rescale. The `cfgIntervalStart/End` fields exist but
are never varied from `[0, 1]`.

**Codex review note (2026-05-18):** This is one of the safest practical
experiments because `cfgIntervalStart/End` are already wired into
`DiffusionActor`; the missing piece is productized presets and a measured A/B
matrix. Prefer trying schedule presets after the Phase 0 harness but before
major DiT rewrites.

Commercial systems schedule guidance strength, apply norm clipping or
orthogonal projection more selectively, and often use a higher-order or
corrector step near t = 0.

**Evidence:**

- `DiffusionActor.swift:104-116` (APG applied uniformly when `doCFG`)
- `Types.swift:51-52` (cfgInterval defaults to full range)
- `GenerateView.swift` advanced section only exposes scalar shift and steps
  override

### 4. Prompt and Metadata Richness

The SFT wrapper is correct in structure, but every generation sends
`bpm: N/A`, `keyscale: N/A`, `timesignature: N/A`. The model was trained on
richer metadata. The lyric writer also produces structurally reasonable but
timing-agnostic output.

**Files:**

- `AnvilCore/Sources/AnvilCore/Inference/Conditioning/ConditioningActor.swift:67-92`
  (`wrapPromptForSFT`, `wrapLyricsForSFT`)
- `AnvilApp/Services/LyricWriter/` (all backends)

### 5. Lack of a Perceptual Evaluation Flywheel

Quality work is currently driven by subjective listening and the AUDIODIAG
tensor logs. There is no golden prompt set, no automated metrics (FAD, CLAP
score, spectral convergence), and no systematic A/B harness. This makes every
proposed fix expensive and noisy.

---

## Prioritized Roadmap

### Phase 0 — Foundation (Tight Evidence Loop, 2-5 days)

**Goal:** Stop guessing quickly. Create enough parity and evaluation coverage
to decide whether the next high-leverage work is VAE, DiT, or inference recipe,
without spending weeks building a full metrics platform.

**Codex review note (2026-05-18):** The original Phase 0 was directionally
right but too broad. Start smaller: two parity tests, eight golden prompts, and
a simple report. Expand only after the first evidence points at the bottleneck.

1. Build a **cross-runtime parity harness**.
   - Ability to load tensors exported from the upstream Python/MLX reference.
   - First target: VAE direct decode parity on a fixed 30-second latent
     (`latentFrames <= 2048`, no tiling).
   - Second target: VAE tiled decode parity on a 120-200 second latent
     (`latentFrames > 2048`, exercises `VAETiledDecode`).
   - Third target: one DiT parity probe using fixed conditioning, fixed noise,
     fixed timestep, and selected block outputs.
   - Codex implementation note (2026-05-18): initial harness added as the
     `QualityEval` Swift package executable plus reusable
     `AudioParityMetrics`. It consumes upstream `latents.safetensors` and
     upstream decoded WAV/audio tensors, decodes the same latent through
     `VAEActor.decode`, and emits a Markdown parity report.

2. Create an **internal golden evaluation set**.
   - Start with 8 prompts, not 20. Keep it small enough to rerun after every
     candidate fix.
   - Cover sparse prompt, rich prompt, lyrics, no lyrics, long duration,
     high-energy drums, guitar/bass, and ambient/texture cases.
   - For each: seed 42, default Turbo 8-step and Base 50-step runs.
   - Store reference audio, spectrograms, and metric scores.
   - Codex implementation note (2026-05-18): starter prompt set added at
     `docs/quality-eval/golden-set.json`; this is intentionally small and can
     be expanded only after the first reports prove useful.

3. Add lightweight objective metrics.
   - CLAP text-audio similarity
   - Fréchet Audio Distance (FAD) only if a small local embedding path is easy
     to wire; do not block Phase 0 on it.
   - Simple perceptual proxies: RMS envelope correlation, high-frequency
     energy, stereo correlation, transient sharpness
   - Always pair metrics with short blind listening notes; music quality is
     not reducible to one score.

**Deliverable:** A minimal `QualityEval` script/target that produces a one-page
Markdown report and stores enough artifacts to compare two branches or commits.

### Phase 1 — Perceptual Plumbing (3-6 weeks)

1. **VAE parity and improvement**
   - Exact decode comparison against upstream on identical latents.
   - If tiling introduces artifacts, implement overlap-add or learnable seam
     correction.
   - Consider a small learned post-VAE enhancer (1-2 layer Conv1D or a tiny
     DiT) trained on reconstruction pairs.

2. **DiT low-level parity sweep**
   - Patch embed / projIn channel ordering and reversal.
   - GQA KV repeat direction (`MLX.repeated` axis).
   - Exact construction of sliding vs. full attention masks.
   - q/k normalization placement inside `DiTSelfAttention`.

3. **Inference recipe upgrades**
   - Expose and vary `cfgIntervalStart/End` (e.g., guidance only applied after
     t < 0.5).
   - Implement dynamic thresholding or guidance rescale inside `apgForward`.
   - Add a higher-order or two-step corrector for the final 1-2 diffusion
     steps.
   - A/B matrix of (shift, steps, guidance schedule) against the golden set.

**Success criterion:** At least one variant shows statistically significant
improvement on 3+ golden prompts via both objective metrics and blind
listening.

### Phase 2 — Conditioning and Prompt Richness (parallel with Phase 1, 2-4 weeks)

1. Make the meta block real.
   - Add UI or inference-time estimation for BPM, key, energy, and genre tags.
   - Format them exactly as the upstream `_parse_metas` / SFT training data
     expects.
   - Codex review note (2026-05-18): this is a likely low-risk prompt-adherence
     lever because Swift currently sends `bpm: N/A`, `timesignature: N/A`, and
     `keyscale: N/A` for every generation while the prompt enhancer already
     produces tempo/key/style information that could be normalized into metas.

2. Improve lyric conditioning.
   - Add explicit timing or section markers (`[Verse 1 0-8s]`) when the lyric
     writer or user provides them.
   - Allow a short reference audio clip to supply timbre even in non-cover
     generation.

3. Prompt enhancer evolution.
   - Train or prompt the enhancer to produce "model-native" language
     (instrument + energy + arrangement + reference timbre).

### Phase 3 — Post-Processing and Mastering (2-3 weeks)

Add a lightweight, always-on mastering stage after VAE decode:

- LUFS loudness normalization (target -14 to -18)
- Gentle multiband dynamics or transient shaper
- Subtle high-frequency stereo widening or mid/side EQ
- Optional "release" preset that applies a learned enhancer

This stage alone often moves "pretty good" outputs into "sounds released"
territory and is cheap to iterate.

### Phase 4 — Evaluation Flywheel and Community Leverage (ongoing)

1. Instrument the app (opt-in) to collect generation metadata + implicit
   quality signals (kept vs. deleted, edited in the AI editor, LoRA trained on
   it, etc.).
2. Use high-quality community adapters as a data source for preference tuning
   or continued pre-training of the base DiT.
3. Build a small human preference dataset for DPO or reward modeling if
   on-device RL becomes feasible.

---

## Immediate Next Actions (Concrete and Ordered)

1. **Create the minimal parity harness** (highest leverage single task).
   - Codex implementation note (2026-05-18): first pass implemented as
     `swift run --package-path AnvilCore QualityEval ...`, with docs in
     `docs/quality-eval/README.md`.
   - New file: `AnvilCore/Sources/QualityEval/main.swift`.
   - Export a 30-second latent from upstream on the golden prompt "slow dark
     guitar riff", seed 42, and compare Swift direct decode against upstream
     decode.
   - Export one long 120-200 second latent to test the tiled decode path
     separately.
   - Compare `VAEActor.decode` output against upstream decoded WAV using L2,
     spectral difference, RMS/high-frequency/stereo proxies, and listening
     notes.

2. **Add a minimal golden set and metric runner**.
   - Store prompts + seeds in a JSON under `docs/quality-eval/golden-set.json`.
   - Start with 8 prompts. Emit a Markdown report with file paths, settings,
     metrics, and short human listening fields.

3. **Expose cfgInterval and a guidance schedule preset** in the Advanced
   disclosure in `GenerateView.swift`.
   - Start with three presets: "Constant (current)", "Late emphasis (0.4-1.0)",
     "Commercial default".

4. **Re-run a small A/B matrix** on the golden set.
   - First pass: current default vs. two guidance schedule presets.
   - Second pass only if needed: shift × steps × guidance schedule.
   - Codex review note (2026-05-18): do not let the matrix explode before the
     harness proves it can produce useful decisions.

5. **Audit the exact whitespace and token count** of `wrapPromptForSFT` against
   a fresh upstream run of `SFT_GEN_PROMPT.format(...)`.

---

## Non-Technical Levers

- **Listening discipline**: Establish a weekly blind listening panel (even 2-3
  people) using the golden set. Require written notes before looking at
  metrics.
- **Reference comparison**: Keep a small set of commercial reference tracks in
  the same genres. A/B generated audio against real music, not just against
  previous Anvil builds.
- **Scope ruthlessness**: It is easy to spend months on DiT internals while
  the VAE or post-processing would have given 70 % of the perceptual win. The
  harness will tell you where to stop.

---

## Success Definition

Commercial-grade for this product means:

- On the golden set, mean CLAP score within 0.15 of the best upstream
  reference run on the same model checkpoint.
- Blind listening: listeners cannot reliably distinguish Anvil output from a
  well-engineered commercial generation using the same base model family more
  than 60 % of the time.
- Prompt adherence: when the prompt says "heavy distorted 808 bass and crisp
  909 hats," those elements are clearly present and correctly balanced in
  >80 % of generations at default settings.

---

## References

- `docs/ace-step-audio-quality-investigation.md` — complete history of the
  structural fixes that got us to recognizable music.
- `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift`
- `AnvilCore/Sources/AnvilCore/Inference/VAE/VAETiledDecode.swift`
- `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTComponents.swift`
- `AnvilApp/UI/Components/GenerateView.swift` (advanced controls surface)
- Upstream ACE-Step v1.5 MLX reference (models/mlx/dit_generate.py and
  modeling files)

---

**Next review checkpoint:** After Phase 0 harness is running and the first VAE
parity comparison has been performed. At that point we will know whether the
biggest remaining lever is the VAE, the DiT internals, or the inference
schedule.
