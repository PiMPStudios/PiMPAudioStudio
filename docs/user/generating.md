# Generating Music

Open:

```text
Create -> Generate
```

Generate is the main text-to-audio workspace for music (and SFX when routed
there). Use the inspector on the right for duration, seed, guidance scale, and
pipeline override.

## Modes

Mode buttons at the top of Generate:

| Mode | Meaning |
| --- | --- |
| Auto | Intent router classifies the prompt as Music, Sound FX, or Ambient |
| Music | Force ACE-Step music path (subject to pipeline override) |
| Sound FX | Prefer short-form SFX / Stable Audio when ready |
| Ambient | Prefer ambient/SFX-style routing |

When Auto is selected, the UI shows the detected mode, for example
`→ Music`.

## Prompt And Lyrics

- **Prompt** — style / instrumentation / mood description. This is the main
  conditioning text for music.
- **Lyrics** — optional structured lyrics for vocal generations. Leave empty for
  instrumental-style prompts.

Drag the small grip under each box to resize **height only** (width stays full).
Heights are remembered between launches.

Helpful actions on the Generate page (when Intelligence is configured):

- **Enhance** — rewrites or strengthens the prompt through the local LLM or
  configured intelligence path.
- **Write lyrics** — fills the lyrics field through the selected lyric backend.

See [Intelligence And Lyrics](intelligence.md).

## Duration, Seed, And Guidance

These appear in the inspector and affect the next generation:

| Control | Behavior |
| --- | --- |
| Duration | Target length in seconds (music commonly 15–180s depending on model headroom) |
| Seed | Blank = random; a fixed integer enables reproducible A/B tests |
| Guidance | CFG scale; Soft Late can force scale to 6.0 and ignore the field |

### Guidance presets (Advanced)

Under **Generate → Advanced → Guidance**:

| Preset | Behavior |
| --- | --- |
| Constant | CFG across the full diffusion schedule |
| Late Emphasis | CFG only for `t <= 0.5` |
| Mid-to-Late | CFG only for `t <= 0.75` |
| Soft Late | Forces guidance **6.0**; CFG only for `t <= 0.6` |

Defaults:

- Non-XL models default toward **Soft Late**.
- XL DiT models and XL adapters auto-nudge Soft Late to **Constant** for
  full-schedule CFG closer to upstream behavior.

You can still pick Soft Late on XL deliberately for A/B tests, but the UI warns
that Soft Late often degrades XL quality.

## Advanced Controls

Expand **Advanced** on Generate for:

| Control | Notes |
| --- | --- |
| Precision | `float16` is faster / lower RAM; higher precision is slower and more accurate |
| Steps | Blank = auto (`8` Turbo, `50` Base / SFT / XL) |
| Shift | Timestep shift; default reset value is `3.0` (range about 0.5–10) |
| Verbose audio diagnostics | Writes `[AUDIODIAG]` lines to the Diagnostics folder |

See [Diagnostics And Logs](diagnostics-and-logs.md).

## Adapters During Generation

Select a primary adapter with the adapter picker. When an adapter is active:

- **Adapter Gain** scales contribution at generation time (default `1.0x`).
- Gain range is `0.0x`–`6.0x`. Practical A/B values are `1.0`, `1.6`, and `2.0`.
- Additional adapters can be stacked when kind-compatible (LoRA with LoRA, LoKR
  with LoKR).

Confirm adapter effect in verbose logs by searching for `adapter/effect`.

See [Adapters](adapters.md).

## Cover Generation

Import a source audio file into Generate. The primary button becomes
**Generate Cover** instead of **Generate**.

Cover strength controls how far the model may rework the source:

| Strength region | Meaning |
| --- | --- |
| Lower | Closer to the source recording |
| Mid | Balanced reinterpret |
| Higher | More creative freedom |

Keep seed, duration, prompt, and base model fixed when comparing cover strength
values. Cover runs are seed-reproducible when the rest of the request matches.

Source audio is copied into the app container for sandbox safety; you can clear
or replace the imported source from the Generate UI.

## Generation Progress

Typical stages:

1. Downloading model (if SFX path needs a missing download)
2. Loading weights
3. Encoding text and lyrics
4. Tokenizing source / conditioning audio codes (cover paths)
5. Diffusion steps
6. Decoding
7. Complete (audio written to History)

Use **Cancel** to stop an in-flight generation. Starting a new generation waits
for the previous run to unwind cleanly.

If local training is active, Generate may show a training-in-progress notice so
you know memory is contended.

## Analyzer

Click **Analyzer** to open the Generation Analyzer window. It shows live stage
tracing for the current run and can load saved `.trace.json` sidecars for review
and A/B comparison.

Useful analyzer / history actions include preferred-take marking and listening
notes on completed takes.

## Saving Presets

Use the save control on Generate to store the current prompt settings as a named
preset. Load presets later from **Library → Presets**.

## Pipeline Override

When Stable Audio is downloaded, the inspector **Pipeline** override offers:

- **Auto** — follow mode / intent routing
- **ACE-Step (Music)** — force music pipeline
- **Stable Audio (SFX)** — force SFX pipeline

## What Good Output Looks Like

- Prompt style is audible in instrumentation and mood.
- Lyrics (if provided) land in roughly the right places for vocal models.
- Same seed + same settings produce the same audio.
- Enabling an adapter at `1.0x`–`2.0x` produces a measurable change vs base.
- Trace file exists beside the WAV for later comparison.

## Next Steps

- [Sound Effects](sound-effects.md)
- [Audio Editor](editor.md)
- [History And Presets](history-and-presets.md)
- [Adapters](adapters.md)
