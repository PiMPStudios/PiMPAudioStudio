# Adapters

Adapters are small fine-tuned weight packages that change how the ACE-Step base
model behaves during generation. PiMP Audio Studio supports native adapters,
compatible PEFT LoRA folders, and compatible Side-Step/LoKR run folders.

## Where To Manage Adapters

Open:

```text
Settings -> Adapters
```

From there you can import, rename, delete, export, and select adapters for
generation.

## Import Formats

Common accepted inputs:

- Native PiMP Audio Studio adapter folders.
- PEFT LoRA folders containing `adapter_config.json` and
  `adapter_model.safetensors`.
- Compatible LyCORIS/LoKR checkpoints.
- Side-Step LoRA/LoKR run folders.
- Specific Side-Step checkpoint folders.

For Side-Step run folders, the app prefers the best adapter when that metadata
exists, otherwise it can import final/checkpoint outputs. Unsupported LoKR
variants, such as Tucker-style tensors, are rejected with an explanation.

## Base Model Compatibility

Adapters are tied to the model family they were trained against. Pay attention
to the ACE-Step variant:

- XL Base
- XL SFT
- XL Turbo

An adapter trained on one variant may import into the app but sound weak, wrong,
or unstable on another variant. When testing adapter quality, use the same base
model variant used during training whenever possible.

## Adapter Gain

Adapter gain controls how strongly the adapter contribution is applied at
generation time.

Practical test values:

- `1.0x` - normal/default.
- `1.6x` - useful when an adapter is present but subtle.
- `2.0x` - stronger A/B test.
- Above `2.0x` - experimental; can distort or collapse the output.

If a trained adapter sounds identical to base at `1.0x` but clearly changes the
song at `1.6x` or `2.0x`, the adapter is working but may be under-scaled or
under-trained.

## Stacking

Use adapter stacks carefully. For clean debugging, compare one adapter at a time
first. Then test combinations only after each adapter has a known individual
effect.

Keep adapter kind compatibility in mind. LoRA adapters stack cleanly with LoRA
adapters, and LoKR adapters stack cleanly with LoKR adapters. Avoid mixing LoRA
and LoKR in the same troubleshooting run unless the UI explicitly allows the
specific stack you are testing.

## Auditioning Adapters

Good adapter tests keep everything else fixed:

- Same prompt.
- Same lyrics.
- Same seed.
- Same duration.
- Same base model.
- Same guidance/settings.

Suggested sequence:

1. Generate base with no adapter.
2. Generate adapter at `1.0x`.
3. Generate adapter at `1.6x`.
4. Generate adapter at `2.0x`.
5. Open Trace Compare or History A/B playback.

Training completion snapshots automate this pattern for newly completed local
training runs.

## Confirming An Adapter Is Active

Enable verbose diagnostics and search the newest log for:

```text
adapter/effect
```

Those lines report whether the active adapter contributed a measurable delta
during generation.

If there are no adapter-effect lines:

- Confirm the adapter is selected.
- Confirm the selected base model is compatible.
- Confirm gain is not zero.
- Confirm generation was restarted after changing adapter settings.

## Checkpoint Testing

For training runs, do not judge only by loss. Checkpoints with similar loss can
sound meaningfully different.

Use fixed-seed checkpoint auditions to compare:

- Epoch 5 versus epoch 10.
- Best checkpoint versus final checkpoint.
- `1.0x` versus boosted gain.

Keep the best-sounding checkpoint even when it is not the lowest-loss one.
