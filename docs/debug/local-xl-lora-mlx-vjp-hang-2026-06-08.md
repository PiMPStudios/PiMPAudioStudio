# Local XL LoRA MLX VJP Hang

Date: 2026-06-08

## Summary

Local ACE-Step XL LoRA training hangs before the first optimizer step on Apple/MLX under some local adapter-training graph shapes. The app does not crash, throw, or reach the first loss value. It becomes stuck inside MLX autodiff during the first `valueAndGrad` call.

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
Pre-0cadf12 Q/V Last 16: known working through local app testing
Post-0cadf12 Q/V Last 16 rank 64: confirmed hang through MCP
Post-0cadf12 Q/V/O Last 16 rank 96: confirmed hang through MCP
Post-0cadf12 Q/K/V Last 16 rank 96: confirmed hang through MCP
Post-0cadf12 Q/K/V/O Last 12: confirmed hang
Post-0cadf12 Q/K/V/O Last 16: confirmed hang
```

## MCP Bisection Update

After `0cadf12` switched LoRA/LoKR trainers to MLXNN's model-aware `valueAndGrad(model:)` path, controlled MCP runs showed the stall is broader than the original "wide Q/K/V/O" shape.

All of these runs reached `training/trainer/firstLossGradBegin` and `training/trainer/gradClosureReady`, then never reached `training/trainer/firstLossGradEnd`:

```text
Q/V/O Last 16, rank 96, alpha 192, dropout 0.08
Q/K/V Last 16, rank 96, alpha 192, dropout 0.08
Q/V Last 16, rank 96, alpha 192, dropout 0.08
Q/V Last 16, rank 64, alpha 128, dropout 0.08
```

Samples from the hung processes:

```text
/tmp/PiMPAudioStudio-qvo-mcp-bisect-sample-20260608_182638.txt
/tmp/PiMPAudioStudio-qkv-mcp-bisect-sample-20260608_183025.txt
/tmp/PiMPAudioStudio-qv96-mcp-control-sample-20260608_183551.txt
/tmp/PiMPAudioStudio-qv64-mcp-control-sample-20260608_183832.txt
```

This points away from K-only, O-only, or projection-count-only explanations. At this point the model-aware gradient route looked suspicious, but the later flat-gradient retest below shows it was not the whole cause.

## Flat-Gradient Retest Update

After restoring the explicit flat adapter-parameter gradient route and the pre-experiment base linear path, MCP retested this control shape:

```text
Projection coverage: Q/V
Layer coverage: Last 16
Rank: 64
Alpha: 128
Dropout: 0.08
Dataset: 640 cached/preprocessed samples
Base model id: ace-step-dit-xl-sft
```

The rebuilt app logged:

```text
training/trainer/valueAndGradReady kind=lora gradientRoute=flatAdapterParameters trainableArgumentCount=128
training/trainer/firstLossGradBegin kind=lora args=6 trainableArgumentCount=128
training/trainer/gradClosureReady kind=lora xt=[1, 64, 64] encoder=[1, 89, 2048] context=[1, 64, 128] mask=[1, 89]
```

It still did not reach `firstLossGradEnd` or `firstLoss`.

Fresh sample:

```text
/tmp/PiMPAudioStudio-current-hang-sample-20260608_185618.txt
```

The sample again shows recursive `mlx::core::vjp(...)`, so the model-aware gradient helper was not the whole cause.

Closest successful comparison:

```text
Log: PiMPAudioStudio-AUDIODIAG-2026-06-05_21-49-21-414-B6531815.log
Run: darkgentlemen 2026-05-30-171242 LoRA
Base model id: ace-step-acestep-v15-xl-sft
Samples: 200
Rank: 64
Alpha/dropout inferred from generated adapter trace: alpha=64, dropout=0
Layer coverage: Last 16
Targets: Q/V self-attention + cross-attention
Trainable tensors: 128
First sample: encoder=[1, 85, 2048]
Result: reached firstLoss after gradClosureReady in about 0.18 seconds and completed training.
```

The next clean test is therefore the same Q/V Last 16, rank 64, alpha 128, requested dropout 0.08 control with runtime dropout clamped to 0. Runtime now emits `training/config/dropoutGuard` when a recipe asks for dropout, followed by `training/start ... dropout=0.000`.

## Dropout-Guard Retest Result

After rebuilding with the local dropout guard, MCP reran the same Q/V Last 16, rank 64 control:

```text
Adapter: mcp-control-qv64-dropoutguard-lora
Requested dropout: 0.08
Effective dropout: 0
Alpha: 128
Dataset: 640 cached/preprocessed samples
Base model id: ace-step-dit-xl-sft
```

The run logged the expected guard and effective training configuration:

```text
training/config/dropoutGuard kind=lora adapterID=mcp-control-qv64-dropoutguard-lora requestedDropout=0.080 effectiveDropout=0.000
training/start kind=lora ... rank=64 alpha=128.000 dropout=0.000 lr=8.00e-05 gradAccum=4 epochs=1 layerCoverage=16
```

It reached the first backward/loss immediately and completed one epoch:

```text
training/trainer/firstLossGradEnd kind=lora elapsed=0.027s values=1 flatGrads=128
training/trainer/firstLoss kind=lora loss=2.7797143 flatGrads=128 lossGradElapsed=0.027s itemElapsed=0.195s
training/complete kind=lora name=mcp-control-qv64-dropoutguard-lora tensors=128 trainedPaths=64 finalLoss=1.04215
training/save/end kind=lora adapterID=mcp-control-qv64-dropoutguard-lora
```

This strongly implicates the nonzero adapter dropout path, specifically the random mask inserted inside each training adapter forward, as the trigger for the Q/V rank-64 first-backward VJP stall.

MCP then ran bounded wide-shape retests with verbose diagnostics enabled and effective dropout clamped to 0:

```text
Adapter: mcp-wide-qkvo-last12-dropoutguard-lora
Kind: LoRA
Projection coverage: Q/K/V/O
Layer coverage: Last 12
Rank: 96
Alpha: 192
Requested dropout: 0.08
Effective dropout: 0
Trainable tensors: 192
Trainable paths: 96
```

```text
training/trainer/firstLossGradEnd kind=lora elapsed=0.038s values=1 flatGrads=192
training/trainer/firstLoss kind=lora loss=2.3827438 flatGrads=192 lossGradElapsed=0.038s itemElapsed=0.149s
training/complete kind=lora name=mcp-wide-qkvo-last12-dropoutguard-lora tensors=192 trainedPaths=96 finalLoss=1.02993
training/save/end kind=lora adapterID=mcp-wide-qkvo-last12-dropoutguard-lora
```

```text
Adapter: mcp-wide-qkvo-last12-dropoutguard-lokr
Kind: LoKR
Projection coverage: Q/K/V/O
Layer coverage: Last 12
Linear dim: 64
Alpha: 128
Requested dropout: 0.08
Effective dropout: 0
Trainable tensors: 192
Trainable paths: 96
```

```text
training/trainer/firstLossGradEnd kind=lokr elapsed=0.037s values=1 flatGrads=192
training/trainer/firstLoss kind=lokr loss=2.1339102 flatGrads=192 lossGradElapsed=0.037s itemElapsed=0.175s
training/complete kind=lokr name=mcp-wide-qkvo-last12-dropoutguard-lokr tensors=192 trainedPaths=96 finalLoss=1.36554
training/save/end kind=lokr adapterID=mcp-wide-qkvo-last12-dropoutguard-lokr
```

Both bounded wide retests completed and saved.

MCP then pushed closer to the original LoKR Wide UI hang by probing all 32 layers with full Q/K/V/O coverage and effective dropout 0:

```text
Adapter: mcp-wide-qkvo-all32-dropoutguard-lokr
Kind: LoKR
Projection coverage: Q/K/V/O
Layer coverage: All 32
Linear dim: 64
Alpha: 128
Requested dropout: 0.08
Effective dropout: 0
Trainable tensors: 512
Trainable paths: 256
```

```text
training/trainer/firstLossGradEnd kind=lokr elapsed=0.058s values=1 flatGrads=512
training/trainer/firstLoss kind=lokr loss=3.2443097 flatGrads=512 lossGradElapsed=0.058s itemElapsed=0.256s
training/progress kind=lokr name=mcp-wide-qkvo-all32-dropoutguard-lokr epoch=1/1 step=1 loss=2.56359 avgLoss=2.56359 lr=0.00e+00 gradNorm=2.299e+00 paramNorm=5.383e+01 updateNorm=0.000e+00
```

MCP status later showed sustained optimizer progress at step 16 with nonzero update norm. The run was intentionally stopped without checkpoint because the one-epoch all-32 probe was estimated to take roughly another 17 to 30 minutes:

```text
training/stopRequested saveCheckpoint=false activeKind=lokr
training/cancelled kind=lokr adapterID=mcp-wide-qkvo-all32-dropoutguard-lokr saveCheckpoint=false
```

The all-32 stopped run still emitted `training/complete ... finalLoss=2.01800` immediately before `training/cancelled`; the app state correctly reported `cancelled` and no adapter save was requested. That diagnostic wording may be worth cleaning up separately.

These results make nonzero adapter dropout the leading confirmed trigger, not memory pressure, rank alone, Q/K/V/O coverage alone, or the explicit flat-gradient route.

## Relevant Code Path

The confirmed post-`0cadf12` hangs used MLXNN's model-aware gradient helper in `LoRATrainer`/`LoKRTrainer`:

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

The current mitigation restores the earlier explicit adapter-parameter gradient helper:

```swift
let lossGrad = valueAndGrad({ (allArgs: [MLXArray]) -> [MLXArray] in
    let parameterValues = Array(allArgs.prefix(trainableArgumentCount))
    trainingModel.update(parameters: ModuleParameters.unflattened(parameterPairs))
    let (vPred, _) = trainingModel(...)
    let diff = (vPred - vTarget).asType(.float32)
    let loss = (diff * diff).mean()
    return [loss]
}, argumentNumbers: Array(0..<trainableArgumentCount))

let trainableValues = model.trainableParameters().flattened().map(\.1)
let (lossValues, flatGrads) = lossGrad(trainableValues + args)
```

The flat-gradient retest did not restore Q/V Last 16 when dropout was nonzero. The later dropout-guard retests did restore Q/V Last 16 and bounded Last 12 Q/K/V/O for both LoRA and LoKR.

## Current Suspicions

Possible causes worth investigating:

1. Adapter dropout currently inserts an `MLXRandom.uniform` mask inside every training adapter forward. Multiple dropout-guard retests completed after clamping effective dropout to 0, while the same Q/V rank-64 shape hung with dropout 0.08.
2. This may be an MLX Swift/C++ autodiff bug or pathological recursion in `vjp` for graphs that include per-adapter random dropout masks.
3. Current `ace-step-dit-xl-sft` model identity/config may still matter for full-duration all-32-layer or higher-rank shapes, but it is not enough by itself to hang the dropout-zero retests.
4. Full Q/K/V/O projection coverage increases runtime and memory pressure, but bounded Last 12 Q/K/V/O completed for both LoRA and LoKR, and all-32 LoKR reached sustained optimizer progress, once effective dropout was 0.

## Current Mitigation

The local training preflight/UI guard still warns on full Q/K/V/O XL shapes because they are heavier and were part of the original hang reports. Verbose diagnostics can intentionally downgrade known risky shapes for controlled tests.

Local Swift/MLX execution also clamps requested adapter dropout to 0 while the dropout-path VJP stall is isolated. The requested value remains visible through the `training/config/dropoutGuard` diagnostic.

Recommended local recipe to revalidate after the explicit-gradient fallback:

```text
Projection coverage: Q/V
Layer coverage: Last 16 or less
```

Full Q/K/V/O attention coverage is now viable for bounded local tests with dropout 0. All-32-layer LoKR reached first loss and optimizer progress, but it is slow enough that full completion should be tested intentionally rather than as a default local recipe.

## Follow-Up Experiment

First tried a conservative graph-pruning experiment in the local LoRA/LoKR wrappers:

```text
baseOut = base(x, stopGradient(baseWeight/baseBias)) + adapter(x)
```

This intentionally detached the frozen base parameters, not the full base output. It did not resolve the post-`0cadf12` Q/V rank-64 hang, so the current mitigation restores the pre-experiment base linear path and the pre-`0cadf12` explicit flat adapter-gradient path.

The trainers also now start a first-backward watchdog for the first `valueAndGrad` invocation. If MLX stays inside the call, verbose diagnostics should emit:

```text
training/trainer/firstLossGradWatchdog
```

at 30, 60, 120, and 300 seconds, with memory snapshots at each mark.

When verbose audio diagnostics are enabled, the known wide-shape UI/preflight block is downgraded to an experimental warning so the failing shape can be retested under instrumentation. With diagnostics off, the guard remains a hard block.

## Next Debug Ideas

Potential next probes:

1. Let one all-32-layer LoKR wide run finish with effective dropout 0 if we need a saved adapter artifact and full-run timing.
2. Clean up diagnostics so a user-stopped run does not emit an ambiguous `training/complete` line before `training/cancelled`.
3. Compare alpha 64 vs alpha 128 with dropout 0 to separate training stability/quality from the dropout graph issue.
4. Build a smaller synthetic DiT/LoRA repro using MLX Swift to see if MLX VJP recursion reproduces with random dropout masks outside the app.
