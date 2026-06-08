# Opus Analysis: VJP Hang on Full Q/K/V/O LoRA

Date: 2026-06-08

## Most Likely Cause

VJP graph explosion through frozen base operations.

The smoking gun is in `LoRALinear.callAsFunction`. The forward pass computes:

```swift
let baseOut = matmul(x, weight.T)     // frozen, but still in the compute graph
let loraOut = scaling * matmul(matmul(loraInput, loraA.T), loraB.T)
return baseOut + loraOut              // ties both subgraphs together
```

The `+` creates a single graph node whose backward must traverse **both** subtrees.
MLXNN's `freeze()` only marks parameters as non-trainable (they won't appear in
`trainableParameters()` and the optimizer won't update them). **It does not insert a
gradient barrier.** MLX's `vjp` still has to walk the full frozen forward graph to
compute correct gradients for the adapter path, because the input `x` flows through
both branches.

## Why Q/K/V/O Scales Worse Than Q/V

Each DiT layer has 2 attention blocks (self + cross), each with 4 projections.

| Config | Injected per layer | Total for 16 layers |
|--------|-------------------|---------------------|
| Q/V | 4 LoRA branches | 64 |
| Q/K/V/O | 8 LoRA branches | 128 |

But it is worse than 2x. With Q/V only, the K and O projections are plain `Linear`.
MLX can recognize them as having no trainable descendants and prune those subtrees
from VJP traversal. With Q/K/V/O, **every** attention projection is a LoRA node,
meaning the VJP graph has no prunable subtrees in the attention blocks. The entire
DiT forward graph becomes one massive connected backward graph.

The XL model has `hiddenSize=2048`, 32 layers, `numAttentionHeads=16`, `headDim=128`.
Even at "last 12" layers, that is 12 layers x 2 attention blocks x 4 projections = 96
LoRA nodes, all connected through residual streams. MLX's recursive `vjp` at
`transforms.cpp:408` is likely hitting stack depth or combinatorial path issues on this
graph shape.

## Why It Hangs Rather Than OOMs

Diagnostics show ~20 GB active memory and 30 GB free. Not a memory problem. The process
is at 0% CPU with the VJP thread stuck in recursion. This is consistent with either:

1. **Pathological recursion depth** in MLX's C++ `vjp` implementation. The recursive
   graph walk may hit system stack limits or just be astronomically slow on a graph
   with ~100+ LoRA branch points.
2. **Graph cycle** (less likely). Possible if the `weight` property on `LoRALinear`
   (inherited from `Linear` superclass) and `baseWeight` (the frozen copy) share
   underlying MLXArray data through the `+ 0` copy. If MLX's lazy evaluation treats
   `base.weight + 0` as still linked to `base.weight` in the compute graph, VJP
   might loop.

## The `+ 0` Copy Pattern Is Suspicious

In `LoRALinear.init`:

```swift
self.baseWeight = base.weight + MLXArray(Float(0))
```

This creates a new array **node** that depends on `base.weight` in the compute graph.
Then `super.init(weight: base.weight, ...)` stores the same `base.weight` as the
`weight` property used in the forward pass. So `baseWeight` and `weight` both root from
the same source array but through different graph paths. This should not cause a cycle,
but it does mean VJP will traverse the `+ 0` operation for every frozen weight
unnecessarily.

## Suggested Investigation Priorities

### 1. `stopGradient` on `baseOut` (highest impact)

Wrapping the frozen base computation would tell MLX "this subtree has no trainable
leaves, don't traverse it." The forward value is still correct; only the backward graph
shrinks. If MLX Swift exposes `stopGradient`, applying it in `LoRALinear.callAsFunction`
would cut the VJP graph roughly in half per LoRA node.

### 2. Test Q/K/V/O at Last 4 and Last 8

If Last 8 hangs too, the threshold is somewhere between 64 paths (Q/V x 16) and 32
paths (Q/K/V/O x 4). If Last 4 works but Last 8 does not, it confirms this is a
graph-size issue, not per-node.

### 3. Detach the base forward from the adapter forward

Instead of `baseOut + loraOut` where both share input `x`, compute `baseOut` with a
detached/stopped-gradient copy of `x` so that VJP only needs to differentiate through
the adapter branch. The base output is mathematically correct either way since base
weights are frozen.

### 4. Replace the `+ 0` copy

Replace `base.weight + MLXArray(Float(0))` with a true data copy (if MLX Swift has one,
e.g. `MLXArray(base.weight)` or `.copy()`) to ensure no graph edges connect the stored
`baseWeight` to the original parameter tree.

### 5. Check LoKR for the same pattern

`LoKRLinear` has the same `baseOut + adapterOut` pattern and the same `+ 0` copy, so it
would likely hit the same hang at similar scale.

## Bottom Line

The most probable cause is that MLX's `vjp` is recursively traversing the entire frozen
DiT forward graph because there is no gradient barrier between the frozen base
computation and the adapter computation in `LoRALinear.callAsFunction`. At Q/V coverage
the graph is small enough to complete; at Q/K/V/O it crosses a threshold where the
traversal becomes pathologically slow or effectively infinite. A `stopGradient` on
`baseOut` (or equivalently, detaching `x` before the base matmul) is the most targeted
fix.

## Follow-Up (Opus 4.8): Fused SDPA Reframes the Lead Hypothesis

After reading the attention forward (`DiTAttention.callAsFunction`), which the analysis
above did not examine, the lead hypothesis shifts. The "graph got too big" framing
treated all four projections as interchangeable branch-count. They are not: K and O sit
in structurally distinct positions around a fused op whose backward is custom code.

### 1. The fused `MLXFast.scaledDotProductAttention` is the structural variable

In `DiTAttention.callAsFunction`:

```swift
let attnOut = MLXFast.scaledDotProductAttention(
    queries: q, keys: k, values: v, scale: scale, mask: attentionMask
)
```

This is a **fused primitive with a hand-written C++ vjp rule**, not a composition of
differentiable ops. The difference between the working and hanging configs flows
directly into this op: with Q/V, `kProj` is a frozen plain `Linear`; with Q/K/V/O it is
a `LoRALinear` with trainable leaves. The hang correlates with what gets wrapped around
this fused op, not with raw branch count.

### 2. The 0% CPU detail argues against the graph-explosion theory

The original analysis leaned on "pathological/slow VJP traversal." But a large-but-finite
graph traversal is **CPU-bound** (~100% on one core). The diagnostics report ~0% CPU with
worker threads parked on condition variables. That is a **block/deadlock or a wait**, not
a slow walk. Graph construction in MLX is pure CPU and does not wait on anything, so a
thread sampled "inside vjp" at 0% CPU points at either a fused-op backward that triggered
an `eval`/stream sync that deadlocked, or lock contention. Both are more consistent with a
fused-SDPA vjp issue than with sheer scale. Downgrade "graph explosion"; upgrade
"structural hang in a specific backward rule."

### 3. The discriminating experiment: per-projection bisection

The "Next Debug Ideas" above propose self-attention-only vs cross-attention-only. The
sharper cut is **per-projection**, because K and O are the two newly-wrapped projections
and they sit on opposite sides of the fused op:

```text
Q/V/O Last 16  (add O, leave K frozen)
Q/K/V Last 16  (add K, leave O frozen)
```

- If **Q/V/O hangs**, it is branch-count/graph-size; O is just a downstream matmul, so
  SDPA is exonerated.
- If **Q/V/O works but Q/K/V hangs**, it is specifically the **K-side gradient through
  the fused SDPA**, which nails it to the `scaledDotProductAttention` vjp.

This single contrast separates "too big" from "fused-attention backward bug." The
existing Last-8 / Last-12 / rank sweeps cannot, because they vary quantity while always
flipping K and O together.

### Caveat: the trigger is not "K gets differentiated at all"

Even in the working Q/V config, K still carries a gradient *tensor* in deeper trainable
layers (its input `hiddenStates` has trainable ancestors via the residual stream), so the
trigger is not "K is differentiated." It is having trainable adapter leaves *on* `kProj`
specifically. The bisection in #3 is what tells us which.

### Corollary fix if #3 implicates SDPA

Temporarily swap the fused `MLXFast.scaledDotProductAttention` call for a manual
softmax attention during training only. If the manual path does not hang, the bug is
localized to MLX's fused vjp. That allows filing it upstream while keeping the manual
path as the local-training workaround, instead of banning Q/K/V/O entirely.
