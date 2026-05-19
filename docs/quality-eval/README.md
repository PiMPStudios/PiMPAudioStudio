# Quality Evaluation

This folder holds the small evidence loop from
`docs/ace-step-commercial-quality-roadmap.md`. The first target is VAE parity:
decode the same upstream latent in Swift and compare it against the upstream
decoded reference audio.

## Artifact Contract

For each parity case, export:

- `latents.safetensors` with key `latents`
- `upstream.wav` decoded from that exact latent by the upstream ACE-Step MLX
  reference

Use 48 kHz stereo audio. Keep the first direct-decode case at 30 seconds
(`latentFrames <= 2048`) so it bypasses `VAETiledDecode`, then add one
120-200 second case to exercise the tiled path.

## Run

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

## Read The Report

- A near-zero direct-decode diff means Swift VAE decode is probably aligned.
- A good direct-decode report but bad long-report points at `VAETiledDecode`.
- A material direct-decode diff means the VAE path needs attention before
  spending serious time on DiT quality tweaks.
