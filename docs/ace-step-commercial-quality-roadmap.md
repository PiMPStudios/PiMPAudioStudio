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

- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/VAE/VAETiledDecode.swift`
- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/VAE/VAEActor.swift`
- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/VAE/VAEComponents.swift`

### 2. DiT Low-Level Parity (Patch Embed, GQA, Attention Details)

Small tensor layout, channel ordering, or mask construction errors that are
"close enough" to produce music will still accumulate across 24 layers and
8-50 steps, producing the characteristic "AI smear."

**Evidence:** The investigation doc lists patch embed/unembed ordering, GQA
repeat direction, q/k-norm placement, and sliding attention mask parity as
open items after conditioning was fixed.

**Codex implementation note (2026-05-18):** First PyTorch-vs-Swift DiT probe
now passes on a fixed SFT one-step case. Report:
`docs/quality-eval/reports/sparse-dark-guitar-dit-step.md`. Key metrics:
`proj_in` MAE `2.56e-7`, `layers.23` MAE `9.43e-5`, final `output` MAE
`1.09e-6`, all shape matches and cosine similarities effectively `1.0`.
This sharply lowers the probability that patch projection, GQA repeat, RoPE,
sliding masks, output projection, or core block math are the current dominant
quality blocker. Keep DiT parity on the watch list, but shift the next
commercial-quality work toward inference recipe, prompt/metas, and evaluation
A/Bs unless a future multi-step probe contradicts this.

**Files:**

- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/DiT/DiTModel.swift`
- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/DiT/DiTComponents.swift`
- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/DiT/DiTActor.swift`

### 3. Inference Recipe Is Still Basic

Before the guidance preset pass, `DiffusionConfig` and
`DiffusionActor.generate` applied constant guidance across all timesteps, used
a fixed APG momentum path, and had no dynamic thresholding or guidance rescale.

**Codex review note (2026-05-18):** This is one of the safest practical
experiments because `cfgIntervalStart/End` are already wired into
`DiffusionActor`; the missing piece is productized presets and a measured A/B
matrix. Prefer trying schedule presets after the Phase 0 harness but before
major DiT rewrites.

**Codex implementation note (2026-05-19):** Advanced generation settings now
include four guidance schedule presets: Constant, Late Emphasis, Mid-to-Late,
and Soft Late. These route through `cfgIntervalStart/End` for ACE-Step music
and cover generation. Soft Late also lowers effective guidance to 6.0 so the
first listening pass can test whether high constant guidance is contributing
to overcooked artifacts.

**Codex result note (2026-05-19):** First user listening A/B on
`slow dark guitar riff` and
`dark blues, vocals, slow tempo, minor key, raw guitar, atmospheric` favored
late guidance over Constant. On the sparse prompt, Soft Late was the strongest
variant and faded more naturally than Constant's hard ending. Based on that
pass, Soft Late is now the app default while Constant remains available as the
baseline comparison preset.

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

**Codex implementation note (2026-05-19):** First-pass runtime metadata
extraction is now wired in `ConditioningActor.wrapPromptForSFT`. Explicit
prompt phrases such as `70 BPM`, `4/4`, and `A minor` are normalized into the
SFT meta block, with conservative tempo heuristics for obvious descriptors
like "slow tempo". This does not replace a richer prompt-enhancer schema yet;
it removes the all-`N/A` baseline and gives future listening tests an
`AUDIODIAG cond/promptMetadata` breadcrumb.

**Files:**

- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/Conditioning/ConditioningActor.swift:67-92`
  (`wrapPromptForSFT`, `wrapLyricsForSFT`)
- `PiMPAudioStudio/Services/LyricWriter/` (all backends)

### 5. Lack of a Perceptual Evaluation Flywheel

Quality work is currently driven by subjective listening and the AUDIODIAG
tensor logs. There is no golden prompt set, no automated metrics (FAD, CLAP
score, spectral convergence), and no systematic A/B harness. This makes every
proposed fix expensive and noisy.

**Codex implementation note (2026-05-19):** The first product-facing listening
flywheel now exists in the app. Each generation writes a `.trace.json` sidecar
with request/model/adapter/audio metadata; History can reopen saved traces;
the Generation Analyzer can compare two traces side by side, highlight changed
settings, play A/B audio, save listening notes, mark a preferred take, show a
compact waveform/loudness preview, and send either run into the editor. Failed
generations also write a `.failed.trace.json` event-log artifact when possible.
This does not replace objective metrics or golden-set reports, but it closes the
practical "what changed between the good one and the wrecked one?" loop that was
missing from day-to-day quality tuning.

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
   - Codex implementation note (2026-05-18): DiT probe scaffolding added as
     `DiTProbeEval` plus reusable `TensorParityMetrics`. It runs one fixed DiT
     forward pass from safetensor inputs, captures `proj_in`,
     `condition_embed`, selected `layers.N`, `post_norm_out`, and `output`,
     then compares them against upstream reference tensors. This intentionally
     keeps normal generation untouched while giving the next reviewer a
     block-by-block parity read.
   - Codex result note (2026-05-18): first SFT one-step DiT parity report
     passes with tiny numerical deltas against upstream PyTorch. The Phase 0
     evidence now says both VAE decode and one-step DiT math are broadly
     aligned.

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
   - Codex implementation note (2026-05-19): until objective metrics are wired,
     use the Generation Analyzer trace compare view as the lightweight
     listening report for schedule/model/seed/adapter A/Bs.

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
   - Codex implementation note (2026-05-19): a conservative first pass now
     extracts BPM, time signature, and key/mode from the final prompt at
     inference time. The remaining commercial-grade version should make those
     fields explicit in the prompt enhancer output and UI rather than relying
     on regex/keyword inference.

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
     `swift run --package-path PiMPAudioCore QualityEval ...`, with docs in
     `docs/quality-eval/README.md`.
   - Codex implementation note (2026-05-18): second pass added
     `swift run --package-path PiMPAudioCore DiTProbeEval ...`, documented in
     `docs/quality-eval/README.md`. The first upstream export was generated
     locally from `../ACE-Step-1.5/scripts/export_dit_probe_case.py` and the
     resulting report is
     `docs/quality-eval/reports/sparse-dark-guitar-dit-step.md`.
   - New file: `PiMPAudioCore/Sources/QualityEval/main.swift`.
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

3. **Expose cfgInterval and guidance schedule presets** in the Advanced
   disclosure in `GenerateView.swift`.
   - Codex implementation note (2026-05-19): implemented Constant, Late
     Emphasis (`t <= 0.5`), Mid-to-Late (`t <= 0.75`), and Soft Late
     (`guidance=6.0`, `t <= 0.6`) presets. Because ACE-Step timesteps descend
     from `1.0` to `0.0`, "late" guidance means lower `t` intervals.
   - Codex result note (2026-05-19): first two-prompt listening pass picked
     Soft Late as the practical default, with Mid-to-Late and Late Emphasis
     still useful A/B options.

4. **Re-run a small A/B matrix** on the golden set.
   - First pass: Constant vs. Late Emphasis vs. Mid-to-Late vs. Soft Late.
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
- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/Diffusion/DiffusionActor.swift`
- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/VAE/VAETiledDecode.swift`
- `PiMPAudioCore/Sources/PiMPAudioCore/Inference/DiT/DiTComponents.swift`
- `PiMPAudioStudio/UI/Components/GenerateView.swift` (advanced controls surface)
- Upstream ACE-Step v1.5 MLX reference (models/mlx/dit_generate.py and
  modeling files)

---

**Next review checkpoint:** After Phase 0 harness is running and the first VAE
parity comparison has been performed. At that point we will know whether the
biggest remaining lever is the VAE, the DiT internals, or the inference
schedule.
