# Audio Editor

Open:

```text
Create -> Edit
```

The Editor loads audio for latent AI edits (inpaint / repaint) and traditional
DSP. You can open it from the sidebar or by sending a take from History /
generation workflows that navigate to the Editor.

## Load Audio

Use the toolbar **Open** action to import audio. Supported types include WAV,
MP3, M4A/AAC, AIFF, and FLAC when available on the system.

You can also arrive with preloaded audio when another screen opens the Editor
with a file URL and optional original prompt.

Until audio is loaded, the canvas shows an empty state.

## Edit Modes

The toolbar mode picker switches between:

| Mode | What it does |
| --- | --- |
| Inpaint | Regenerates a **selected** time region with a region prompt |
| Repaint | Regenerates more of the track according to the prompt without requiring a selection the same way |

Inpaint requires a waveform selection and a non-empty region prompt. Frequency
band limits appear in inpaint mode so you can constrain the edit to a band.

### Inpaint controls

- **Region prompt** — describe what should appear in the selection.
- **Strength** — how hard the model reworks the region (`0`–`1`).
- **Guidance scale** — CFG for the edit (`1`–`15`).
- **Presets** — Light / Moderate / Strong style shortcuts for strength/guidance.

Orange-tinted controls mark the AI-driven domain (prompt, strength, guidance,
frequency band).

### Frequency band

In inpaint mode, low/high cut sliders limit which frequencies the latent edit
targets. The UI enforces a minimum bandwidth so the band cannot collapse to
empty.

## DSP Effects

Below the AI controls, the **Effects** and **Transforms** palette applies
deterministic DSP to the loaded buffer.

### Effects

| Effect | Selection required | Notes |
| --- | --- | --- |
| Trim to Selection | Yes | Keeps only the selected region |
| Delete Selection | Yes | Removes the selected region |
| Fade In | Yes | Fade across the selection |
| Fade Out | Yes | Fade across the selection |
| Normalize | No | Peak or RMS-oriented normalize path |
| Parametric EQ | No | Multi-band EQ (default bands around 200 / 1200 / 8000 Hz) |
| Reverb | No | Room / Hall / Plate style presets with decay, wet/dry, pre-delay, damping |

### Transforms

| Effect | Notes |
| --- | --- |
| Time Stretch | Rate about `0.25`–`4.0` (`1.0` = original). WSOLA-backed. |
| Pitch Shift | About `-24` to `+24` semitones (`-2400` to `+2400` cents). LPC-aware path available. |

Effects that need parameters expand a detail panel with **Apply to selection /
whole track** and an Apply button. Simple effects apply immediately and show a
brief confirmation checkmark.

DSP runs off the main thread so the UI stays responsive. Undo / redo are
available from the toolbar when history exists.

## Export

Export the current editor buffer as:

- **WAV** (lossless)
- **AIFF** (lossless)
- **M4A** (AAC compressed)

MP3 / FLAC / OGG as export targets are intentionally not offered as native
third-party encoder dependencies; FLAC may still open for import when the system
can decode it.

## How Editor AI Differs From Generate

| Generate | Editor |
| --- | --- |
| Creates a new take from text / cover source | Mutates a loaded buffer |
| Full-length diffusion job | Region or whole-track latent edit |
| Writes a new History entry | Keeps working audio until you export / navigate |
| Adapter stack applies to music generation | Inpaint uses editor pipeline choice and loaded models |

After DSP, the editor refreshes editable latents so later inpaint/render sees the
edited audio.

## Tips

1. Start with mild strength when inpainting vocals or dense mixes.
2. Keep region prompts short and specific ("add muted guitar chugs").
3. Use DSP for loudness, fades, and timing before another AI pass when possible.
4. Export WAV masters before experimenting with irreversible chains.
5. If inpaint sounds wrong, confirm the correct pipeline model is Ready and that
   the selection covers the intended bars.

## Keyboard Shortcuts

See [Keyboard Shortcuts](keyboard-shortcuts.md) for Editor bindings such as
trim, delete selection, normalize, EQ, and reverb focus shortcuts.

## Troubleshooting

- **Apply button disabled** — missing prompt, missing selection (inpaint), or an
  edit already running.
- **Inpaint fails** — model not Ready, memory pressure, or invalid frequency
  band; check diagnostics.
- **DSP seems ignored** — confirm you clicked Apply on parameterized effects;
  selection-only effects need a selection.
- **Export fails** — path permissions or format error; try WAV to a user-chosen
  folder.

See [Troubleshooting](troubleshooting.md).
