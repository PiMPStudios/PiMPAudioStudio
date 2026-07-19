# Adapters

Adapters are small fine-tuned weight packages that change how the ACE-Step base
model behaves during generation. PiMP Audio Studio supports native adapters,
compatible PEFT LoRA folders, compatible LyCORIS/LoKR checkpoints, and
Side-Step LoRA/LoKR run folders.

## Where To Manage Adapters

```text
Settings -> Adapters
```

Tabs:

- **My Adapters** — import, rename, delete, export, retrain, merge
- **Community** — browse/download community adapters when available

On Generate, use the adapter picker and optional stack UI to select adapters for
the next run.

## Import Formats

Common accepted inputs:

- Native PiMP Audio Studio adapter folders
- PEFT LoRA folders with `adapter_config.json` and `adapter_model.safetensors`
- Compatible LyCORIS / LoKR checkpoints (`lokr_weights.safetensors` or LoKR
  `adapter_model.safetensors`)
- Side-Step LoRA / LoKR run folders (prefers best checkpoint metadata when
  present)
- Specific Side-Step checkpoint folders

Unsupported LoKR variants (for example Tucker-style tensors) are rejected with
an explanation.

Native PiMP LoKR is a PiMP-native format and is not freely interchangeable with
every PEFT/Kohya/LyCORIS LoKR checkpoint on the internet. Import converts
supported shapes into native keys when possible.

## Base Model Compatibility

Adapters are tied to the model family they were trained against. Watch the
ACE-Step variant:

- Classic Turbo / Full / SFT
- XL Base / XL SFT / XL Turbo

An adapter may import successfully but sound weak, wrong, or unstable on the
wrong variant. When testing quality, match the training base model whenever
possible.

## Adapter Gain

Adapter gain (also called strength/weight in places) controls how strongly the
adapter contribution is applied at generation time.

| Value | Use |
| --- | --- |
| `0.0x` | Off |
| `1.0x` | Normal / default |
| `1.6x` | Useful when an adapter is present but subtle |
| `2.0x` | Stronger A/B test |
| Above `2.0x` | Experimental; can distort or collapse (UI allows up to `6.0x`) |

If a trained adapter sounds identical to base at `1.0x` but clearly changes the
song at `1.6x` or `2.0x`, the adapter is working but may be under-scaled or
under-trained.

## Stacking

For clean debugging, compare one adapter at a time first. Then test combinations
only after each adapter has a known individual effect.

Kind rules:

- LoRA stacks with LoRA
- LoKR stacks with LoKR
- Avoid mixing LoRA and LoKR in the same troubleshooting run

Each stacked adapter keeps its own alpha/rank or alpha/linear-dim scaling rather
than borrowing the primary adapter's scale.

## Merge

**Settings → Adapters → Merge Two** blends two LoRA adapters. Merge uses rank
concatenation of true LoRA deltas (not independent A/B interpolation), so the
result matches weighted delta blending without cross-term artifacts. Different
ranks are allowed. LoKR merge is not the same path as LoRA merge.

## Auditioning Adapters

Keep everything else fixed:

- Same prompt
- Same lyrics
- Same seed
- Same duration
- Same base model
- Same guidance / steps / precision

Suggested sequence:

1. Generate base with no adapter
2. Generate adapter at `1.0x`
3. Generate adapter at `1.6x`
4. Generate adapter at `2.0x`
5. Open Trace Compare or History A/B playback

Training completion snapshots automate this ladder for newly finished local
runs.

## Confirming An Adapter Is Active

Enable verbose diagnostics and search the newest log for:

```text
adapter/effect
```

Those lines report whether the active adapter contributed a measurable delta
during generation.

If there are no adapter-effect lines:

- Confirm the adapter is selected
- Confirm the selected base model is compatible
- Confirm gain is not zero
- Confirm generation was restarted after changing adapter settings

## Checkpoint Testing

Do not judge only by training loss. Checkpoints with similar loss can sound
meaningfully different.

Use fixed-seed checkpoint auditions to compare:

- Mid-run checkpoint versus final
- Best validation checkpoint versus last epoch
- `1.0x` versus boosted gain

Keep the best-sounding checkpoint even when it is not the lowest-loss one.

## Community Adapters

The Community tab can list and install shared adapters into the local registry
when network access is available. Community metadata cannot redirect storage
outside the validated adapter directory. Treat community adapters like any other
import: check base-model compatibility and A/B against base.

## Related

- [Generating Music](generating.md)
- [Datasets And Training](datasets-and-training.md)
- [Diagnostics And Logs](diagnostics-and-logs.md)
