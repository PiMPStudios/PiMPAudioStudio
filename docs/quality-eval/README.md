# Quality Evaluation

This folder holds the small evidence loop from
`docs/ace-step-commercial-quality-roadmap.md`. The first target is VAE parity:
decode the same upstream latent in Swift and compare it against the upstream
decoded reference audio. The second target is a narrow DiT tensor probe:
run one fixed denoising forward pass and compare selected Swift tensors against
upstream reference tensors from the same inputs.

For everyday listening A/Bs inside the app, use the Generation Analyzer rather
than this parity harness. Current Anvil builds write `.trace.json` sidecars next
to generated WAVs; History can reopen those traces and compare two runs with
changed settings, prompt/lyrics, audio summaries, A/B playback, and editor
handoff. This folder remains the lower-level parity/metric workspace.

## VAE Artifact Contract

For each parity case, export:

- `latents.safetensors` with key `latents`
- `upstream.wav` decoded from that exact latent by the upstream ACE-Step MLX
  reference

Use 48 kHz stereo audio. Keep the first direct-decode case at 30 seconds
(`latentFrames <= 2048`) so it bypasses `VAETiledDecode`, then add one
120-200 second case to exercise the tiled path.

## Export Upstream VAE Artifacts

With `PiMPStudios/ACE-Step-1.5` checked out beside this repo, export the first
direct-decode case with:

```sh
python3 ../ACE-Step-1.5/scripts/export_vae_parity_case.py \
  --output-dir /tmp/anvil-quality \
  --case-name sparse-dark-guitar-direct \
  --caption "slow dark guitar riff" \
  --lyrics "[Instrumental]" \
  --duration 30 \
  --seed 42 \
  --steps 50 \
  --guidance 7.5 \
  --shift 3.0 \
  --config-path acestep-v15-sft
```

That script writes the expected `*-latents.safetensors` and `*-upstream.wav`
pair from the same upstream generation run. The WAV is decoded directly from
the exported latent before ACE-Step generation applies its final peak
normalization, so the comparison stays focused on VAE parity.

## Run VAE Parity

From the repo root:

```sh
swift run --package-path AnvilCore QualityEval \
  --case sparse-dark-guitar-direct \
  --latent /tmp/anvil-quality/sparse-dark-guitar-direct-latents.safetensors \
  --reference /tmp/anvil-quality/sparse-dark-guitar-direct-upstream.wav \
  --report docs/quality-eval/reports/sparse-dark-guitar-direct.md
```

The VAE weights default to the app cache:

```text
~/Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil/models/ace-step-vae/vae.safetensors
```

Pass `--weights /path/to/vae.safetensors` to override it.

## DiT Probe Artifact Contract

The DiT probe uses two safetensors files: one for fixed forward inputs, one for
upstream reference tensors captured from the same forward pass.

`dit-probe-inputs.safetensors`:

- `hidden_states` — required, shape `[batch, frames, 64]`
  - aliases accepted by the Swift runner: `noisy_latents`, `latents`, `xt`
- `encoder_hidden_states` — required, shape `[batch, textSeq, 2048]`
  - alias: `encoder_states`
- `context_latents` — required, shape `[batch, frames, 128]`
  - aliases: `contextLatents`, `context`
- `timestep` — required scalar or `[1]`
  - aliases: `timesteps`, `t`
- `timestep_r` — optional scalar or `[1]`; defaults to `timestep`
  - aliases: `timestepR`, `tr`
- `encoder_attention_mask` — optional, shape `[batch, textSeq]`
  - aliases: `encoder_mask`, `encoderMask`

`dit-probe-reference.safetensors`:

- `proj_in` — output immediately after Swift/upstream patch projection
- `condition_embed` — projected text/lyric/timbre conditioning
- `layers.N` — hidden states after requested DiT block `N`
- `post_norm_out` — hidden states after final adaptive norm, before `projOut`
- `output` — final velocity prediction after de-patchify/crop

Codex implementation note (2026-05-18): `DiTProbeEval` currently compares only
matching tensor keys. Missing keys are reported as `missing-reference` or
`missing-swift`, which lets an upstream exporter start with `proj_in`,
`layers.0`, and `output` before adding deeper probes.

## Run DiT Probe

From the repo root:

```sh
swift run --package-path AnvilCore DiTProbeEval \
  --case sparse-dark-guitar-dit-step \
  --inputs /tmp/anvil-quality/dit-probe-inputs.safetensors \
  --reference /tmp/anvil-quality/dit-probe-reference.safetensors \
  --layers 0,1,23 \
  --report docs/quality-eval/reports/sparse-dark-guitar-dit.md
```

The DiT weights default to the app cache:

```text
~/Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil/models/ace-step-dit-sft/model.safetensors
```

Pass `--weights /path/to/model.safetensors` and
`--config /path/to/config.json` to override. Sharded DiT checkpoints are
accepted by passing `--weights /path/to/model.safetensors.index.json`.

The first completed DiT probe report is
`reports/sparse-dark-guitar-dit-step.md`. It compares Swift MLX against
upstream ACE-Step PyTorch for one SFT forward pass and shows near-exact parity
through `proj_in`, layers `0`, `1`, `23`, `post_norm_out`, and final `output`.

## Read The Reports

- A near-zero direct-decode diff means Swift VAE decode is probably aligned.
- A good direct-decode report but bad long-report points at `VAETiledDecode`.
- A material direct-decode diff means the VAE path needs attention before
  spending serious time on DiT quality tweaks.
- A bad DiT `proj_in` row points at input packing, patching, or Conv1d weight
  layout.
- A good `proj_in` row but bad early `layers.N` row points at attention,
  q/k norms, RoPE, GQA repeat, or masks.
- Good early DiT rows but diverging late rows points at accumulated
  attention/mask drift or timestep/modulation details.
- Good `post_norm_out` but bad `output` points at `projOut` / de-patchify
  layout.
