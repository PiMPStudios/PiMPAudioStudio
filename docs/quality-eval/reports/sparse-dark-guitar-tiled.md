# VAE Parity Report: sparse-dark-guitar-tiled

**Generated:** 2026-05-19T01:52:38Z
**Precision:** float32
**Sample rate:** 48000 Hz
**Compared duration:** 120.000 s

## Artifacts

- Weights: `/Users/tweaver/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/models/ace-step-vae/vae.safetensors`
- Latent: `/tmp/pimp-audio-quality/sparse-dark-guitar-tiled-latents.safetensors` (`latents`)
- Reference: `/tmp/pimp-audio-quality/sparse-dark-guitar-tiled-upstream.wav`

## Shapes

| Tensor | Shape |
| --- | --- |
| Swift decoded audio | `[1, 5760000, 2]` |
| Upstream reference audio | `[1, 5760000, 2]` |
| Compared window | `[5760000, 2]` |

## Metrics

| Metric | Value |
| --- | ---: |
| Mean absolute error | 0.0010733579 |
| Mean squared error | 2.2664315e-06 |
| RMSE | 0.0015054672 |
| Max absolute difference | 0.064200714 |
| Swift RMS | 0.14972703 |
| Reference RMS | 0.14897089 |
| Swift high-frequency proxy | 0.019299762 |
| Reference high-frequency proxy | 0.01917509 |
| Swift stereo correlation | 0.90926611 |
| Reference stereo correlation | 0.90939307 |
| RMS envelope correlation | 0.99999219 |

## Listening Notes

- Swift:
- Reference:
- Difference:

## Initial Read

Decode is close in broad dynamics but not bit-level aligned. Inspect VAE dtype, padding, and final scaling before blaming DiT quality.
