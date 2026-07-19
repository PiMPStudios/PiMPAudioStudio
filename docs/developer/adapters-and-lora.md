# Adapters And LoRA

## Formats

| Kind | Storage / import notes |
| --- | --- |
| Native LoRA | PiMP adapter folder + metadata |
| Native LoKR | PiMP-native LoKR keys + `lokrConfig` metadata |
| PEFT LoRA | `adapter_config.json` + `adapter_model.safetensors` |
| LyCORIS/LoKR | Supported `lokr_w1`/`lokr_w2` (+ simple factored/DoRA); reject Tucker |
| Side-Step runs | Prefer `best/` over `final/`; preserve XL variant + target metadata |

Import/remap logic lives in app services such as `PEFTKeyRemapper` and adapter
registry code, with Core injectors applying weights at runtime.

## Registry

Adapters are stored under the models root in a reserved adapters directory.
Path rules:

- Single contained directory component
- Reject symlink escapes
- Metadata id must match on-disk directory
- Community metadata cannot redirect storage outside validated IDs

Model migration orphan cleanup must protect the adapters directory.

## Inference Injection

- LoRA and LoKR injectors patch target Linear modules by pattern.
- Stacks are kind-homogeneous.
- Each contribution uses its own alpha/rank or alpha/linear-dim scale.
- UI/MCP gains clamp through `AdapterStrengthLimits` (`0.0`…`6.0`).

### Default target patterns

Conservative local defaults favor Q/V (and limited MLP patterns where
configured). Wide recipes expand to full attention projections. Documented
divergence from some upstream q/k/v/o presets is intentional for memory and
stability on Apple Silicon.

## Training Outputs

Successful local runs write adapter folders consumed by the registry, plus:

- `training_log.json` (progress, val loss, events)
- Checkpoints and optional optimizer state
- Completion audition hooks into generation + History

## Merge

`LoRAWeightMerge` / Merge sheet blend two LoRAs via rank concatenation of scaled
`B@A` deltas so the result matches `w·ΔA + (1-w)·ΔB` without independent A/B
cross terms. Different ranks are allowed. LoKR merge is not the same operation.

## Diagnostics

Enable verbose AUDIODIAG and search `adapter/effect` to verify nonzero
contribution during generation.

## MCP

- `list_adapters`
- `apply_adapter` (select/stack/clear for next generation)
- Training inspect/compare tools for runs that produced adapters

## Compatibility Checklist For New Imports

1. Detect format and base model variant metadata.
2. Convert tensors into native key layout.
3. Preserve alpha/rank/dim and target layer intent.
4. Reject unsupported factorizations with actionable errors.
5. Round-trip: import → generate with fixed seed → `adapter/effect` present.
