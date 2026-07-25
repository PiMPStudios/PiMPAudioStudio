# Editor Polish / Light Mastering — Design Spec

**Status:** P0 + P1 + P2 implemented (one-shot Polish for Keep)  
**Date:** 2026-07-24  
**Related:** Phase 7 DSP (`2026-05-06-phase7-dsp-audio-editor.md`), Phase 8
transforms, `docs/user/editor.md`

---

## Overview

Add a **within-reason** polish path so a finished generation can go from “raw
model dump” to a cleaner, more listenable keep-file without leaving PiMP Audio
Studio. This is **not** a Logic/Ozone clone: no multi-track mixer, no plugin
host, no stem mastering, no reference-track matching.

**Primary product:** one-shot **Polish for Keep** with a few knobs and safe
defaults (Streaming / Louder / Gentle).

**Secondary product:** a few new **Effects** building blocks (gain target,
auto-trim, HPF, glue compressor, limiter, meters) that the one-shot chain uses
and advanced users can apply manually.

**Architecture fit:** stay offline, deterministic, float-process then write back
to the existing editor buffer model; one undo entry per Apply (Polish counts as
one step); refresh editor latents after DSP as today. Prefer Swift/vDSP and
existing AVAudioEngine offline patterns under `PiMPAudioStudio/DSP/`.

---

## Goals

1. Clean edges (silence, clicks at start/end).
2. Sensible level with headroom (not always smash to 0 dBFS).
3. Mild corrective tone (rumble, mud, air).
4. Light glue + loudness control for streaming-ish or louder targets.
5. Optional subtle space (existing reverb, very low wet by default).
6. Export-ready WAV (AIFF/M4A remain available).

### Non-goals (v1–v2)

- Full DAW insert chains, automation lanes, clip gain curves.
- Multiband dynamics, serial compressors, sidechain.
- Stereo width / mid-side imaging tools (optional later).
- Spectral repair, de-click, de-hiss, true noise gate.
- Broadcast-certified true-peak / loudness compliance.
- Real-time live monitoring while generation is running.
- Competing with Ozone for commercial masters.

AI **Inpaint / Repaint** remain for musical region fixes, not loudness.

---

## Current baseline (as of this spec)

| Capability | Today |
| --- | --- |
| Trim / delete selection | Yes |
| Fade in / out (selection, linear) | Yes |
| Normalize | Peak (to ~full scale; skips if already hot) and RMS path in code |
| Parametric EQ | 3 peaking bands (~200 / 1200 / 8000 Hz) |
| Reverb | Room / Hall / Plate |
| Time stretch / pitch | Yes (transforms) |
| Compressor / limiter | No |
| Peak target (−1 / −3 dB) | No |
| LUFS / true-peak meter | No |
| Auto silence trim | No |
| One-shot polish chain | No |

---

## User-facing design

### Placement

```text
Editor
  [ AI: Inpaint | Repaint ]
  [ Polish for Keep ▾ ]     ← new primary entry
  [ Effects ▾ ]             ← existing + new building blocks
  [ Transforms ]
```

Keep category accents and offline Apply UX consistent with the current DSP
palette (selection vs whole track, checkmark confirm, undo/redo).

### Polish for Keep panel

| Control | Options | Default |
| --- | --- | --- |
| Target | Streaming / Louder / Gentle | Streaming |
| Tone | Neutral / Brighter / Darker | Neutral |
| Glue | Off / Light / Medium (or 0–100%) | Light |
| Space | Off / Room / Hall (subtle wet) | Off |
| Trim silence & fade edges | On / Off | On |
| Apply | Whole track (v1) | Whole track |

**Readouts (always visible in panel):** sample peak dBFS, approximate integrated
LUFS (or short-term proxy if full integrated is deferred), limiter ceiling.

**Preview (nice-to-have in P2):** process a ~10–15 s middle slice without
committing; full Apply still whole-track.

### Target presets

| Target | Intent | Ceiling | Loudness aim |
| --- | --- | --- | --- |
| **Streaming** | Best default “keep” | −1.0 dB peak | ~−14 LUFS integrated (best effort) |
| **Louder** | More competitive / club-ish | −0.5 dB peak | ~−10 LUFS |
| **Gentle** | Clean + headroom for external master | Peak target −3 dB | No aggressive loudness push |

### Tone presets (mild EQ only)

| Tone | Behavior |
| --- | --- |
| Neutral | HPF ~35–40 Hz; no other forced curves |
| Brighter | Neutral + small high shelf / air (~8–12 kHz) |
| Darker | Neutral + mild cut ~250–350 Hz if needed for boxiness |

All moves should stay subtle (on the order of 1–3 dB), not “AI EQ rebuild.”

---

## Processing chain (one-shot order)

When **Apply** runs on Polish for Keep:

1. **Optional auto-trim** — drop leading/trailing below ~−50 dBFS (with
   minimum keep length and max trim caps so a quiet intro is not erased).
2. **Edge fades** — ~20–50 ms linear fade in/out (or slightly longer if trim
   was large).
3. **HPF** — ~35–40 Hz (or tone-preset variant).
4. **Corrective EQ** — peaking / shelf from Tone preset only.
5. **Glue compression** — if Glue ≠ Off: soft knee, ~2:1–3:1, medium
   attack/release, ~2–4 dB GR at Light / a bit more at Medium.
6. **Gain / loudness stage** — approach target LUFS (or peak-only for Gentle)
   without exceeding pre-limit headroom policy.
7. **Brickwall limiter** — enforce ceiling (−1.0 or −0.5 dB per target).
8. **Optional reverb** — if Space ≠ Off, very low wet (full-mix safe).

Write float pipeline → convert back to editor buffer format; single undo
snapshot before the whole chain.

Manual Effects apply one stage at a time (existing pattern).

---

## New / extended DSP building blocks

### P0 — immediate value

| Effect | Behavior |
| --- | --- |
| **Peak target normalize** | Scale so peak ≈ −3 / −1 dBFS (user pick); replace “always 0 dBFS” as the recommended path |
| **Auto trim silence** | Leading/trailing threshold + min duration; then optional auto fades |
| **HPF** | Dedicated control or first EQ band as high-pass |
| **Meters** | Peak dBFS always; rough LUFS readout (even if approximate) |

Also: keep existing trim/delete/fade/EQ/reverb; document Gentle as external-master prep.

### P1 — real “mastered-ish”

| Effect | Behavior |
| --- | --- |
| **Glue compressor** | One mono-linked stereo compressor; threshold/ratio/attack/release/makeup simplified or preset-driven |
| **Limiter** | Ceiling + optional release; no fancy multi-stage maximizer |
| **Loudness aim** | Integrated (preferred) or short-term proxy driving makeup before limit |

### P2 — productization

| Item | Behavior |
| --- | --- |
| **Polish for Keep** UI | One-shot chain + presets above |
| **Preview slice** | Optional non-destructive audition of mid section |
| **Keyboard shortcut** | e.g. focus Polish panel (TBD with shortcuts doc) |

### P3 — confidence / edge cases

| Item | Notes |
| --- | --- |
| True-peak estimate | Oversampled peak if cheap enough |
| Mono check | Temporary sum-to-mono preview |
| Soft gate | Optional; easy to damage full mixes — low priority |
| Stereo width | Explicitly later; high footgun risk |

---

## Map to classic mastering steps

| Classic step | PiMP support after this work |
| --- | --- |
| 1. Normalize / gain stage | Peak target + loudness aim (Streaming/Louder/Gentle) |
| 2. Cleanup / gate | Auto-trim + manual trim/delete; no spectral repair |
| 3. EQ corrective | HPF + mild peaking/shelf presets; improved EQ types |
| 4. Compression | Single glue compressor |
| 5. Limiter / maximizer | Simple brickwall + target presets; not full maximizer suite |
| 6. Stereo / spatial | Existing reverb only (subtle); no width tool in v1 |
| 7. Fades / polish | Auto + manual fades; no crossfade engine |

---

## Technical notes

### Processing

- Run polish math in **float** (−1…1); existing buffer may remain Int16 on
  disk/path — convert carefully; optional light dither on quantize.
- Prefer **whole-track** for one-shot v1; selection-scoped for individual
  effects where it already makes sense (fades, EQ, etc.).
- Stay **off main thread** (same as current DSP).
- After apply: refresh editable latents so later inpaint sees polished audio.

### Suggested code touchpoints (implementation time)

| Area | Likely files / modules |
| --- | --- |
| Effect enum / params | `DSP/DSPParameters.swift`, `DSPEffect` cases |
| Algorithms | `DSP/SimpleDSP.swift`, new `Compressor.swift` / `Limiter.swift` / `LoudnessMeter.swift` as needed |
| EQ path | Extend `AVAudioEngineDSP` or vDSP biquads for HPF/shelf |
| UI | Editor DSP palette, new `PolishForKeep` panel |
| Undo | Existing editor history snapshot |
| MCP (optional later) | `apply_dsp`-style tools only after UI solid |

### Metering policy

- **v1:** sample peak + approximate integrated LUFS (BS.1770-inspired gated
  estimate is enough for “aiming”).
- **v1.1:** true-peak estimate if needed for fewer intersample clips on export.

Do not block Apply on perfect compliance; show meters as guidance.

### Parameter ranges (starting points)

| Parameter | Range / default |
| --- | --- |
| Peak target | −6 … 0 dBFS; presets −3 / −1 |
| Limiter ceiling | −3 … 0 dBFS; Streaming −1.0, Louder −0.5 |
| HPF | 20–120 Hz; default ~40 Hz |
| Glue GR aim | ~2 dB Light, ~4 dB Medium |
| Edge fade | 10–100 ms; default ~30 ms |
| Silence threshold | ~−45 to −60 dBFS; default ~−50 |
| Reverb wet in Polish | ≤ ~0.08–0.12 full-mix |

Tune by ear on ACE-Step XL takes during implementation.

---

## Phased delivery

| Phase | Ship | Exit criteria |
| --- | --- | --- |
| **P0** | Peak target normalize, auto-trim + auto fades, HPF (or EQ HPF), peak (+ rough LUFS) meter | Typical take has clean edges and safe headroom without 0 dBFS smash |
| **P1** | Glue compressor + limiter + Streaming/Louder gain staging | Competitive loudness without obvious pumping/clipping on test tracks |
| **P2** | Polish for Keep one-shot UI + tone/glue/space knobs | Non-expert can polish and export WAV in one Apply |
| **P3** | Better meters, optional mono check | Users trust levels enough to skip external tools for casual keeps |

Implement **P0 → P1 → P2** as the committed product arc; P3 only if still
needed after dogfooding.

---

## Success criteria

On a typical ACE-Step (including XL) stereo take:

1. No clicky edges or long dead silence after Polish.
2. Peak at or under chosen ceiling.
3. Tone not thinner/muddier in a way that feels broken (mild EQ only).
4. Streaming target lands near −14 LUFS ± a few dB (best effort), or Gentle
   leaves headroom for external mastering.
5. One undo restores pre-Polish audio.
6. No model download/reload required; works offline on loaded buffer.
7. Export WAV remains the recommended archive format.

---

## Test plan (implementation)

- Unit tests: peak target scale math, silence trim bounds, limiter never exceeds
  ceiling on synthetic peaks, fade lengths.
- Golden ears: 3–5 real generations (sparse guitar, dense mix, vocal-ish, SFX
  if shared editor path).
- Regression: existing EQ/reverb/normalize paths still work; undo stack intact;
  inpaint after polish still runs.
- Memory: whole-track float chain on multi-minute buffers does not freeze UI.

---

## Docs follow-up (when shipping)

- Update `docs/user/editor.md` Effects table and add **Polish for Keep**.
- Update `docs/user/keyboard-shortcuts.md` if shortcuts added.
- CHANGELOG under Added / Changed for each phase shipped.
- Optional MCP: document only after UI tools are stable.

---

## Open questions

1. Should peak normalize **replace** the current 0 dBFS peak default, or sit
   beside it as “Normalize (target…)”?
2. Is selection-scoped Polish needed in v1, or whole-track only?
3. How strict should auto-trim be on tracks with intentional long silence
   (ambient)?
4. MCP `polish_track` in the same release as P2, or later?

Defaults if undecided at implement time: **replace recommended normalize with
target −1 dB**, whole-track Polish only, ambient-safe trim (max trim seconds +
high threshold), MCP later.

---

## Implementation notes

- **P0 shipped:** peak-target Normalize, Trim Silence, High-Pass, Peak +
  approximate LUFS meters, MCP effect names, unit tests.
- **P1 shipped:** Compress (glue), Limit (ceiling), Loudness (target LUFS +
  limit), dynamics UI presets, MCP parameters, unit tests.
- **P2 shipped:** one-shot **Polish for Keep** panel + chain processor, single
  undo, MCP effect `polish` / `polishForKeep`, unit tests. Preview-slice audition
  remains optional future work.
