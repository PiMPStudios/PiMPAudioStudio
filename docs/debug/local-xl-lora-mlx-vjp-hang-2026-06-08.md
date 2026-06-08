# Local XL LoRA MLX VJP Hang

Date: 2026-06-08

## Summary

Local ACE-Step XL LoRA training hangs before the first optimizer step when using broad/full attention projection coverage on Apple/MLX. The app does not crash, throw, or reach the first loss value. It becomes stuck inside MLX autodiff during the first `valueAndGrad` call.

The Xcode debug dylib path, for example:

```text
/Users/tweaver/Library/Developer/Xcode/DerivedData/PiMPAudioStudio-fkgzsaakhmvyekaosjvvvswxcbaf/Build/Products/Debug/PiMPAudioStudio.app/Contents/MacOS/PiMPAudioStudio.debug.dylib
```

only indicates that the sample stack is symbolicated against the app's debug build. The important signal is the function inside that dylib: repeated samples show the active training thread stuck in `mlx::core::vjp(...)`.

## Confirmed Failing Shapes

Both failures used:

```text
Base model: ACE-Step XL SFT
Dataset: 640 cached/preprocessed samples
Rank: 96
Epochs: 150
Gradient accumulation: 4
Target layers: self-attention + cross-attention
Projection coverage: Q/K/V/O
```

Confirmed failure 1:

```text
Layer coverage: Last 16
Injected paths: 128
Trainable tensors: 256
Layers: 16...31
```

Confirmed failure 2:

```text
Layer coverage: Last 12
Injected paths: 96
Trainable tensors: 192
Layers: 20...31
```

## Exact Hang Point

Diagnostics consistently reach:

```text
training/trainer/firstLossGradBegin
training/trainer/gradClosure
training/trainer/gradClosureReady
```

Diagnostics do not reach:

```text
training/trainer/firstLossGradEnd
training/trainer/firstLoss
training/progress
```

Interpretation: the Swift loss closure completes the forward/loss graph construction, but MLX does not return from the first `valueAndGrad` invocation.

## Sample Evidence

Sampling the hung process shows the active training thread recursively inside:

```text
mlx::core::vjp(...)
transforms.cpp:408
```

The MLX worker threads are idle/waiting on condition variables. The app is not busy running Metal kernels. CPU usage is effectively 0%.

Observed sample files:

```text
/tmp/PiMPAudioStudio-hang-sample-20260608_173229.txt
/tmp/PiMPAudioStudio-hang-sample-20260608_175018.txt
```

Observed diagnostics logs:

```text
/Users/tweaver/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/Diagnostics/PiMPAudioStudio-AUDIODIAG-2026-06-08_17-30-48-503-F9BFC285.log
/Users/tweaver/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/Diagnostics/PiMPAudioStudio-AUDIODIAG-2026-06-08_17-45-59-472-14773533.log
```

## Memory Evidence

Memory does not look like the immediate trigger.

Typical guard/snapshot values:

```text
MLX active memory: about 20.6 GB
Resident memory: about 19.6 to 19.8 GB
System available memory: about 30 GB
Physical memory: 64 GB
MLX cache: 0
```

This does not resemble a normal OOM or Metal kernel memory failure. It looks more like MLX VJP graph traversal or graph-construction pathology.

## Known Working Boundary

Older successful local LoRA runs used narrower projection coverage:

```text
Layer coverage: Last 16
Projection coverage: Q/V only
Injected paths: 64
Trainable tensors: 128
Rank: 64
```

Those runs reached `firstLoss` and made epoch progress.

Current boundary:

```text
Q/V Last 16: known working
Q/K/V/O Last 12: confirmed hang
Q/K/V/O Last 16: confirmed hang
Q/K/V/O Last 8: not yet proven either way
```

## Relevant Code Path

Training uses MLXNN's model-aware gradient helper in `LoRATrainer`:

```swift
let lossGrad = valueAndGrad(model: model) { trainingModel, allArgs in
    let (vPred, _) = trainingModel(...)
    let diff = (vPred - vTarget).asType(.float32)
    let loss = (diff * diff).mean()
    return [loss]
}

let (lossValues, grads) = lossGrad(model, args)
```

The DiT model is frozen except for adapter parameters. LoRA wrappers replace selected `Linear` projections. Base weights are frozen; adapter matrices are trainable.

## Current Suspicions

Possible causes worth investigating:

1. MLX VJP graph traversal explodes or cycles on the ACE-Step XL DiT graph when many LoRA modules are injected.
2. Freezing base weights prevents base gradients, but may not prevent MLX from traversing the full frozen base computation graph.
3. Full Q/K/V/O projection coverage creates too many connected adapter branches for MLX Swift autodiff on this XL model.
4. `LoRALinear` computes `baseOut + loraOut`; even with base parameters frozen, VJP may still walk all base operations.
5. The frozen base output may need an explicit `stopGradient` boundary or an adapter-only decomposition to shrink the backward graph.
6. This may be an MLX Swift/C++ autodiff bug or pathological recursion in `vjp` for this graph shape.

## Current Mitigation

The local training preflight/UI guard now blocks full Q/K/V/O LoRA at 12+ XL layers. This avoids silently hanging the app.

Recommended local recipe for now:

```text
Projection coverage: Q/V
Layer coverage: Last 16 or less
```

Full Q/K/V/O attention coverage should be treated as a Remote GPU recipe unless we find a reliable local MLX workaround.

## Follow-Up Experiment

Implemented a conservative graph-pruning experiment in the local LoRA/LoKR wrappers:

```text
baseOut = base(x, stopGradient(baseWeight/baseBias)) + adapter(x)
```

This intentionally detaches the frozen base parameters, not the full base output. The goal is to remove unnecessary parameter graph edges while still allowing gradient flow through the activation input `x`, which earlier adapter layers need for normal PEFT training.

The trainers also now start a first-backward watchdog for the first `valueAndGrad` invocation. If MLX stays inside the call, verbose diagnostics should emit:

```text
training/trainer/firstLossGradWatchdog
```

at 30, 60, 120, and 300 seconds, with memory snapshots at each mark.

When verbose audio diagnostics are enabled, the known wide-shape UI/preflight block is downgraded to an experimental warning so the failing shape can be retested under instrumentation. With diagnostics off, the guard remains a hard block.

## Next Debug Ideas

Potential next probes:

1. Try full Q/K/V/O at Last 8 to see whether the boundary is path count, layer count, or full-attention shape in general.
2. Retest with detached frozen base parameters. If the first backward still stalls, try a diagnostic-only `stopGradient(baseOut)` branch to prove whether base-output traversal is the remaining issue.
3. Test rank 64 full Q/K/V/O Last 12 to isolate rank from projection/layer count.
4. Test self-attention-only Q/K/V/O and cross-attention-only Q/K/V/O to isolate the attention branch.
5. Build a smaller synthetic DiT/LoRA repro using MLX Swift to see if the VJP recursion reproduces outside the app.
