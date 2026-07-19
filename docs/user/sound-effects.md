# Sound Effects

PiMP Audio Studio can generate short-form sound effects and ambient clips through
the **Stable Audio Open** pipeline.

## Prerequisites

1. Download **Stable Audio Open 1.0** from **Settings → Models**.
2. Accept the Hugging Face model license and save a valid HF token if required.
3. Confirm the model status is **Ready**.

Stable Audio Open is gated. Without auth + license acceptance, downloads fail
with 401/403.

## How To Generate SFX

Open **Create → Generate**, then either:

- Set mode to **Sound FX** or **Ambient**, or
- Leave **Auto** and write a clearly SFX-oriented prompt, or
- Set inspector **Pipeline Override** to **Stable Audio (SFX)**

Enter a short descriptive prompt such as:

```text
cinematic metal door slam with long reverb tail
```

Keep duration appropriate for SFX (often much shorter than full songs). Seed
works the same as music: blank = random, fixed integer = reproducible.

## Music Versus SFX Routing

| Signal | Typical pipeline |
| --- | --- |
| Mode Music | ACE-Step |
| Mode Sound FX / Ambient | Stable Audio when ready |
| Auto + music-like prompt | ACE-Step |
| Auto + SFX-like prompt | Stable Audio when ready |
| Pipeline override ACE-Step | ACE-Step |
| Pipeline override Stable Audio | Stable Audio |

If Stable Audio is not Ready, SFX modes cannot complete that path. Download the
model or switch to music generation.

## Adapters And SFX

Adapters in PiMP Audio Studio are ACE-Step fine-tunes. They apply to the music
pipeline, not Stable Audio. Clear or ignore music adapters when debugging SFX
quality.

## History And Editing

Completed SFX clips land in History like music takes. You can:

- Play them back
- Open the Generation Analyzer when a trace is present
- Send audio to the Editor for DSP polish
- Export or delete selected takes

Latent AI inpaint in the Editor may use ACE-Step or Stable Audio depending on
edit mode and readiness; DSP effects always work on the loaded buffer without a
model.

## Quality Tips

- Use concrete nouns and materials: "glass", "wood", "whoosh", "impact".
- Prefer one event per prompt for clean one-shots.
- Fix seed when comparing prompt wording.
- Keep guidance moderate; extreme values can over-sharpen or hollow the hit.
- Normalize or fade in the Editor if peaks are hot.

## Troubleshooting

- **SFX model downloading…** — wait for Stable Audio download/verification.
- **401/403** — fix Hugging Face token and accept the model license.
- **Still hearing music** — mode/pipeline still on ACE-Step; override to Stable
  Audio or set mode to Sound FX.
- **Empty or thin results** — try a more concrete prompt and a shorter duration.

See [Troubleshooting](troubleshooting.md) and [Bug Report Checklist](bug-report-checklist.md).
