# Grok Analysis: Local XL LoRA MLX VJP Hang

Date: 2026-06-08

## Context
This document captures analysis of the issue described in `local-xl-lora-mlx-vjp-hang-2026-06-08.md`.

The core symptom is that local ACE-Step XL LoRA (and LoKR) training reliably hangs inside MLX autodiff on the first `valueAndGrad` invocation when broader projection coverage (Q/K/V/O) and/or higher layer counts are used. The process reaches `gradClosureReady` (forward + loss computed inside the traced closure) but never returns from the `lossGrad(...)` call. Sampling shows the training thread deep in recursive `mlx::core::vjp` graph construction (transforms.cpp:408 and related std::unordered_set work), while MLX stream/ThreadPool workers are idle and no Metal kernels are active.

Memory is high (~19.5–20.6 GB) but stable; this does not present as a classic OOM.

## Code Inspection Findings (Read-Only Review)
I examined the relevant implementation (without making modifications):

- **LoRALinear.swift (and LoKRLinear.swift equivalent)**: In `callAsFunction`, the base path is **always** executed:
  ```swift
  let baseOut = ... matmul/addMM on weight ...
  ...
  return baseOut + loraOut   // (or kronecker delta for LoKR)
  ```
  This happens even under `isTraining`. The frozen base matmuls therefore remain in the traced forward graph that `valueAndGrad` will differentiate.

- **Freeze / Unfreeze flow** (DiTActor.swift:333-334, 405-406 + LoRAInjector.swift:147-156):
  - `decoder.freeze()`
  - Then `unfreezeAdapters(...)` which does `loraLinear.unfreeze(recursive: false, keys: ["loraA", "loraB"])`
  - `freezeBase()` only freezes the base keys inside each wrapper.
  - `trainableParameters().flattened().count` is the source of the 128 / 192 / 256 numbers in the original debug doc.

- **valueAndGrad usage** (LoRATrainer.swift:154 and LoKRTrainer.swift:85):
  Currently uses the model-aware form:
  ```swift
  let lossGrad = valueAndGrad(model: model) { trainingModel, allArgs in ... return loss }
  let (loss, grads) = lossGrad(model, args)
  ```
  The expensive VJP work happens on invocation of `lossGrad(...)`, not at closure definition time (MLX laziness). This matches the observed log sequence.

- **Existing stopGradient precedent** (TrainingService.swift:1255-1266):
  The inputs to `TrainingSample` are already protected:
  ```swift
  let frozenLatent = stopGradient(windowedLatent)
  ...
  eval(arraysToMaterialize)
  ```
  A previous CHANGELOG entry also mentions detaching VAE/text graphs before backprop. The pattern is known and used for data, but **not** applied to separate frozen base outputs from trainable adapter paths inside the linear wrappers.

- No `stopGradient` calls appear inside `LoRALinear`, `LoKRLinear`, the loss closures, or the DiT attention/MLP paths.

- Each DiT layer contributes 8 projection matrices when both self- and cross-attention are wrapped with Q/K/V/O (4 per attention block). Last 16 layers on the XL config produces the 256-tensor surface that hangs.

These findings strongly support the suspicions already listed in the primary debug doc (especially #2, #4, and #5 around frozen base traversal and the need for an explicit boundary).

## Additional / Refined Ideas to Check

### 1. Graph Pruning Experiments (Highest Leverage)
- **Primary candidate**: Insert `stopGradient` around the base contribution *inside the wrappers during training*.
  - In `LoRALinear.callAsFunction` (isTraining branch) and the matching path in `LoKRLinear`:
    ```swift
    let baseOut = ...
    let loraOut = ...
    return stopGradient(baseOut) + loraOut
    ```
  - This preserves the numerical value for the loss while telling the VJP machinery that it does not need to build pullbacks through the (frozen) base matmuls for the adapter parameter gradients.
  - The gradient of the loss w.r.t. the lora inputs should still be correct because the addition happens after the stop.
- Coarser variant (easier first test): Inside the loss closure (first sample only), do `let stoppedPred = stopGradient(vPred)` before the diff computation.
- Measure: time to first loss, peak memory during the first `lossGrad` call, and whether the recursive VJP depth in samples decreases.

### 2. Enhanced First-Step Observability & Watchdogs
- Make "before/after lossGrad" + "before/after .item" + explicit `eval(loss + flatGrads)` timing fire for the absolute first backward (use a dedicated `firstBackwardDone` flag rather than the existing "first processed sample" flag).
- Add a background watchdog Task that logs at 30s / 60s / 120s:
  - "still inside first valueAndGrad after Xs"
  - Current MLX memory
  - trainable tensor count
  - Note that stream threads are idle (observable from sampling pattern)
- Log immediately before the `lossGrad(...)` invocation and capture the exact wall time for the call itself vs. the subsequent `eval` of the returned values.

### 3. Boundary / Configuration Matrix (Run These)
Follow the matrix already recommended in the primary debug doc and the pasted conversation:
- Q/K/V/O vs Q/V (or Q/K/V) on Last 8 / Last 12 / Last 16
- Rank 16 vs 64 vs 96 on the broader coverage
- LoRA vs LoKR at the same surfaces
- Self-attention-only full coverage vs cross-attention-only

Log the exact `trainableParameterKeys.count` and layer indices at `valueAndGradReady` for every attempt. This will cleanly separate "number of adapter branches", "layer depth", "rank", and "attention type".

### 4. Synthetic / Minimal Repro (Strongly Recommended)
Create a tiny standalone test or spike that does **not** require the full 24-layer XL DiT + 640 samples:
- A small stack of modules that mimics 1–2 DiTLayer attention blocks (parallel q/k/v/o Linears + adds, plus a simple MLP).
- Wrap a controllable number of them with LoRALinear/LoKRLinear.
- Freeze the bases, selectively unfreeze the adapters.
- Run a trivial flow-matching-style loss through `valueAndGrad(model: ...)`.
- Scale the number of wrapped projections and depth while keeping memory footprint small.

Goal: reproduce (or fail to reproduce) the recursive VJP hang / extreme slowdown at ~128 vs ~192 vs ~256 tensors. This isolates whether the problem is:
- Raw count of adapter parameters
- The addition of base + adapter inside the wrapper
- Depth / residual connectivity of the real DiT
- Something specific to the ACE-Step XL weight structure or conditioning paths

If the synthetic repro hangs, the `stopGradient(baseOut)` experiment can be validated in complete isolation.

### 5. Alternative Gradient Plumbing Checks
- For the first backward only, try collecting the lora arrays explicitly and using a flat (non-model) `valueAndGrad` that takes only those arrays as the differentiable variables. This gives MLX a much smaller explicit set to build VJPs for, even if the forward still contains the base ops.
- Return a scalar loss from the closure (`-> MLXArray`) instead of `[loss]`. Minor, but removes one layer of indirection in the output of the transformed function.
- After the first successful `lossGrad` on a narrow config, inspect the structure of the returned `ModuleParameters` (keys + shapes) to confirm all expected adapters actually receive gradients.

### 6. Other Quick Probes
- After `decoder.freeze()` + `unfreezeAdapters`, compare `trainableParameters().flattened()` against the full `parameters()` tree to confirm frozen bases are excluded from differentiation but still referenced in the traced graph.
- Force `eval` on selected intermediate activations (e.g., output of one full attention block) inside the closure on the first sample, before returning the loss. Observe whether this changes VJP behavior or timing.
- Capture MLX active memory + resident memory immediately before the first `lossGrad` call (in addition to the existing "gradClosureReady" memory snapshot).
- When running the matrix, also note whether Last-8 full Q/K/V/O works. If it does, the practical local limit may be higher than the current preflight guard.

## Prioritization Suggestion
1. Run the configuration matrix (especially Last 8 full coverage) with the improved first-step timing/watchdog logging.
2. Prototype the `stopGradient(baseOut)` change in the wrappers (and the coarser version in the loss closure) and measure impact.
3. Build the synthetic repro — this will save the most time long-term.
4. Only then consider deeper changes to how the model is traced or alternative autodiff entry points.

## Open Questions
- Is the recursion depth in the VJP samples correlated more with total adapter count or with the number of layers (i.e., depth of residual paths feeding the adapters)?
- Does the hang occur even if we never call `.item()` on the loss (i.e., is it purely during gradient materialization)?
- Are there any MLX Swift / MLXNN debug facilities (graph dumping, tape inspection, max recursion limits, etc.) that could be enabled during the first backward?

## Updated Ideas / Things Worth Checking Next (After Latest Dropout Bisection)

The bisection via MCP (after the model-aware `valueAndGrad` change at `0cadf12`, flat-gradient retest, and especially the `dropoutGuard` that forces effective dropout to 0) has isolated **nonzero adapter dropout** as the leading confirmed trigger.

Key evidence from the primary debug doc:
- Narrow Q/V rank-64 (previously working) hung with dropout 0.08 but completed in 0.027s first backward when clamped to 0.
- With effective dropout=0, even Q/K/V/O Last 12 (LoRA + LoKR) and an all-32-layer full Q/K/V/O LoKR (512 tensors) reached first loss and made real optimizer progress.
- The stopGradient-on-base-weights experiment did not help (as expected once we knew the random mask was the culprit).
- The flat vs model-aware route was also not the root cause.
- The random mask is generated inside the traced closure via `MLXRandom.uniform(0 ..< 1, shape) .> dropoutRate` in the `isTraining` branch of both `LoRALinear` and `LoKRLinear`.

This strongly implicates the stochastic nodes (per-adapter random masks) causing pathological recursive VJP traversal + visited-set behavior in MLX when there are 128–512 such sites.

### Refined Ideas to Check

1. **Make the Dropout Mask VJP-Safe (Highest Priority Fix)**
   - Generate the mask *outside* the `valueAndGrad` closure (in the training loop) and pass it in as a constant argument to the traced function. This removes the `MLXRandom.uniform` from the tape entirely for the backward.
   - For the very first backward only, use a deterministic mask (e.g. fixed seed, pre-generated array, or a simple threshold pattern) so no stochastic node is created during graph construction.
   - Investigate whether MLX provides a higher-level dropout that is known to be safe with `valueAndGrad` / VJP (some frameworks special-case the mask gradient).
   - In the closure, still apply the mask numerically, but the VJP should see far fewer (or no) random branches.

2. **Enhance the New First-Backward Watchdog**
   - The watchdog added in the trainers (logging `training/trainer/firstLossGradWatchdog` at 30/60/120/300s with memory) is a great addition.
   - Extend it to also report: `gradientRoute` (flat vs model-aware), `effectiveDropout`, `trainableArgumentCount`, and whether any `MLXRandom` calls were observed during the traced forward on that sample.
   - When the watchdog fires on a hang, consider also emitting the shapes of the current `args` and a quick count of active MLX memory vs resident.

3. **Re-validate Wide Shapes Systematically with Dropout Guard**
   - Now that dropout=0 unlocks wide coverage, run a clean matrix:
     - Q/K/V/O vs Q/V on Last 8 / 12 / 16 (and all 32 where feasible)
     - Rank 16/64/96 + alpha variants
     - LoRA vs LoKR at identical surfaces
     - Self-attention-only vs cross-attention-only full coverage
   - Always log the `gradientRoute` and `effectiveDropout` values at `valueAndGradReady`.
   - Goal: establish the new safe boundary for local training.

4. **Update Synthetic Repro to Target the Dropout Path**
   - The minimal DiT/LoRA repro should now explicitly include the exact dropout mask generation (`MLXRandom.uniform(...).>rate` + scaling) inside multiple parallel adapter forwards.
   - Reproduce the recursion with masks present vs. absent. This will be the fastest environment to iterate on mask-generation fixes without the full app or 640-sample dataset.

5. **Re-evaluate Graph Pruning in Light of the Real Culprit**
   - The earlier `stopGradient(baseWeight/baseBias)` experiment was reasonable but didn't address the random mask.
   - New experiments worth trying:
     - `stopGradient` on the generated mask itself (or on the `loraInput` right after the mask is applied).
     - Compute the entire adapter contribution (lora path + dropout) and add it under a stop for the base contribution in a way that still gives correct numerics for the loss.
   - Measure impact on first-backward time and sample depth when dropout is *requested but forced to 0*.

6. **Alpha / Quality Comparisons with Dropout=0**
   - As noted in the primary doc: run the same narrow Q/V shape with alpha 64 vs 128 (and vs the original successful May runs) while effective dropout is clamped to 0.
   - This separates training dynamics / final adapter quality from the VJP graph-construction problem.

7. **Longer All-32 Validation Runs**
   - Let at least one full all-32 LoKR (and ideally one LoRA) wide run with effective dropout=0 complete, if time permits.
   - Capture: total wall time, peak memory, final loss, and a saved adapter artifact.
   - This would give strong evidence that "very wide is now practical locally" once the dropout issue is mitigated.

8. **Dropout Policy and Diagnostics**
   - Decide on the permanent local policy: always force effective dropout=0 (with clear user-visible logging via the existing `training/config/dropoutGuard`), or allow nonzero but mark those recipes as "remote GPU recommended"?
   - Clean up the diagnostic noise: user-cancelled runs should not emit an ambiguous `training/complete` line immediately before `training/cancelled`.
   - Make the requested vs effective dropout values prominent in the training UI / logs when a recipe asks for >0 dropout.

9. **Check for Other Stochastic Leaks**
   - The timestep sampling (`MLXRandom.uniform(0..<1, [1])`) and noise generation happen *outside* the traced loss closure. Confirm they cannot accidentally become part of the first `valueAndGrad` tape in some code paths.
   - The encoder mask (when present) is passed in — ensure it is also protected with `stopGradient` like the other inputs.

10. **MLX-Level Visibility (If Available)**
    - When the watchdog fires or on a controlled hang, look for any MLX Swift / C++ hooks that can report tape size, number of random/stochastic nodes, or VJP recursion depth.
    - This would turn "it's hanging in vjp" into "it's hanging because of N random nodes across M adapters".

### Prioritization Recommendation for Codex
1. Implement and test a proper dropout-mask fix (idea 1) — this is the root cause.
2. Expand the matrix and alpha comparisons under the current guard (ideas 3 + 6).
3. Flesh out the synthetic repro with the mask (idea 4) so future experiments are fast.
4. Polish the watchdog and diagnostics (ideas 2 + 8).
5. Only after the above, decide on long-term policy and whether additional graph pruning is still needed.

This analysis and these ideas are based on read-only review of the updated primary debug document and relevant source. No production code was modified.

---

Further reads or targeted file inspections can be performed on request while remaining in read-only mode.
