# VAE Parity Report: sparse-dark-guitar-direct

**Generated:** 2026-05-19T01:50:06Z
**Precision:** float32
**Sample rate:** 48000 Hz
**Compared duration:** 30.000 s

## Artifacts

- Weights: `/Users/tweaver/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/models/ace-step-vae/vae.safetensors`
- Latent: `/tmp/pimp-audio-quality/sparse-dark-guitar-direct-latents.safetensors` (`latents`)
- Reference: `/tmp/pimp-audio-quality/sparse-dark-guitar-direct-upstream.wav`

## Shapes

| Tensor | Shape |
| --- | --- |
| Swift decoded audio | `[1, 1440000, 2]` |
| Upstream reference audio | `[1, 1440000, 2]` |
| Compared window | `[1440000, 2]` |

## Metrics

| Metric | Value |
| --- | ---: |
| Mean absolute error | 0.0026594712 |
| Mean squared error | 1.2963408e-05 |
| RMSE | 0.0036004733 |
| Max absolute difference | 0.085380733 |
| Swift RMS | 0.55875951 |
| Reference RMS | 0.55669093 |
| Swift high-frequency proxy | 0.02658252 |
| Reference high-frequency proxy | 0.026478246 |
| Swift stereo correlation | 0.93802601 |
| Reference stereo correlation | 0.93822348 |
| RMS envelope correlation | 0.99999845 |

## Listening Notes

- Swift:
- Reference:
- Difference:

## Initial Read

Decode is close in broad dynamics but not bit-level aligned. Inspect VAE dtype, padding, and final scaling before blaming DiT quality.
