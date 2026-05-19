# sparse-dark-guitar-dit-step DiT Probe Report

Generated: 2026-05-19 02:58:48 +0000

## Inputs

- Inputs: `/tmp/anvil-quality/sparse-dark-guitar-dit-step-dit-probe-inputs.safetensors`
- Reference: `/tmp/anvil-quality/sparse-dark-guitar-dit-step-dit-probe-reference.safetensors`
- Weights: `/Users/tweaver/Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil/models/ace-step-dit-sft/model.safetensors`
- Config: `/Users/tweaver/Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil/models/ace-step-dit-sft/config.json`
- Layers: `0,1,23`
- DiT variant: `ace-step-1.5` (`hiddenSize=2048`, `layers=24`, `heads=16`, `kvHeads=8`)

## Tensor Comparisons

| Tensor | Shape Match | Swift Shape | Reference Shape | Elements | MAE | RMSE | Max Abs Diff | Swift | Ref | Cosine |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proj_in` | yes | [2, 375, 2048] | [2, 375, 2048] | 1536000 | 2.56336e-07 | 6.84433e-07 | 3.8147e-05 | 0.949079 | 0.949079 | 1 |
| `condition_embed` | yes | [2, 74, 2048] | [2, 74, 2048] | 303104 | 1.63386e-06 | 4.46134e-06 | 0.000244141 | 2.73749 | 2.73749 | 1 |
| `layers.0` | yes | [2, 375, 2048] | [2, 375, 2048] | 1536000 | 1.24339e-05 | 4.96834e-05 | 0.00598145 | 12.271 | 12.271 | 1 |
| `layers.1` | yes | [2, 375, 2048] | [2, 375, 2048] | 1536000 | 1.6942e-05 | 6.56992e-05 | 0.0129395 | 13.7493 | 13.7493 | 1 |
| `layers.23` | yes | [2, 375, 2048] | [2, 375, 2048] | 1536000 | 9.42764e-05 | 0.000164413 | 0.0200195 | 39 | 39 | 1 |
| `post_norm_out` | yes | [2, 375, 2048] | [2, 375, 2048] | 1536000 | 2.16576e-07 | 4.28621e-07 | 1.07288e-05 | 0.108541 | 0.108541 | 1 |
| `output` | yes | [2, 750, 64] | [2, 750, 64] | 96000 | 1.09483e-06 | 1.42164e-06 | 1.29044e-05 | 0.871327 | 0.871327 | 1 |

## How To Read This

- A bad `proj_in` row points at input packing, patching, or Conv1d weight layout.
- A good `proj_in` row but bad early `layers.N` row points at attention, q/k norms, RoPE, GQA repeat, or masks.
- Good early rows but diverging late rows points at accumulated attention/mask drift or timestep/modulation details.
- Good `post_norm_out` but bad `output` points at `projOut` / de-patchify layout.
