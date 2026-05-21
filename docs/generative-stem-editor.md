# Generative Stem Editor — Vision & Concept

**Status:** Vision / High-Level Design  
**Date:** 2026-05  
**Audience:** Codex, engineering, product

---

## Overview

The **Generative Stem Editor** is a proposed evolution of Anvil's editing experience. Instead of treating the editor as a single-file AI inpainting + DSP tool (current state), it reorients the product around **stems as first-class, independently generatable and editable objects**.

The core idea is:

> Build, refine, and arrange music and sound using AI generation and inpainting *at the stem level*, then stitch and iterate until the full arrangement feels coherent.

This direction deliberately leans into Anvil's unique strengths:
- High-quality on-device music generation (ACE-Step DiT / flow-matching)
- High-quality on-device SFX/ambient generation (Stable Audio Open components)
- Strong latent-space inpainting and region editing via `InpaintCoordinator`
- Rich provenance tracking (HistoryEntry, traces, preferred takes, adapters, prompts)

It is **not** an attempt to become a general-purpose DAW. It is an **AI-native creative environment** where generation and editing are deeply intertwined.

---

## Motivation

Current state (as of mid-2026):
- Excellent single-file generation
- Powerful single-region AI inpainting / repainting (Phase 6)
- Good traditional DSP (time-stretch, pitch, EQ, reverb)
- Strong "take" system with history, traces, preferred takes, and listening notes
- No native concept of multiple simultaneous audio layers or stems

Users who want to create complex music currently have to:
1. Generate full mixes
2. Manually split them (or use external tools)
3. Do limited editing on the combined result

This loses the generative advantage.

The Generative Stem Editor vision closes that gap by making stem-level work the primary creative loop.

---

## Core Concept

In the Generative Stem Editor:

- A **Stem** is an independent audio layer (drums, bass, melody, pads, vocals, texture, SFX layer, etc.) that can be generated, regenerated, or inpainted largely on its own.
- Stems live inside a **Session** (or Project), which provides a shared timeline and context.
- The primary creative actions are:
  - Generate a new stem (with optional conditioning from other stems or the full mix)
  - Inpaint or repaint a region *within one stem* while other stems remain locked
  - Use AI to help "glue" stems together musically (transitions, call-and-response, rhythmic locking, harmonic alignment)
  - Traditional arrangement (trim, move, crossfade, volume, pan)

The fundamental difference from both traditional DAWs and the current Anvil editor is that **editing is generative by default**, not just corrective.

---

## Key Workflows

### 1. Stem-First Creation (Primary Vision)

1. User starts a new Session.
2. User generates a drum stem (or imports one).
3. User generates a bass stem, optionally providing the drum stem as strong conditioning / reference.
4. User generates melodic and textural layers, each time deciding how much context from existing stems to feed into the prompt / conditioning.
5. User enters the arrangement view, places the stems on a shared timeline, and begins refining.

This is the "build each stem then stitch them together as I see fit" workflow.

### 2. Iterative Refinement (The Killer Loop)

After initial arrangement:

- User selects a region on the **drums** stem only.
- They write a prompt: "add a rolling tom fill that leads into the next section".
- The system performs latent inpainting on just that stem's latents.
- Other stems continue playing unchanged.
- User can audition the change in context instantly.
- If it feels wrong, they can regenerate that region with different strength, guidance, or prompt, or lock the drums and edit the bass instead.

This loop (generate → arrange → targeted generative edit on one stem → audition in context → repeat) is the heart of the experience.

### 3. Full Mix → Stem Splitting → Refinement

For users who prefer to start with full generations:

- Generate a full track.
- Trigger stem separation (or manual stem extraction).
- The system creates a Session with the separated stems already aligned.
- User can now treat the stems as editable generative objects (see Workflow 2).

This is a secondary but important on-ramp.

### 4. Stitching & Musical Glue

When stems don't quite fit:

- User selects a transition area across multiple stems.
- They can use specialized "generate transition" or "align rhythmically" operations that operate across stems.
- Or they can inpaint one stem while using the audio from other stems as additional conditioning.

---

## Core Abstractions (Proposed)

These are the key new concepts Codex should internalize:

| Concept              | Description                                                                 | Relationship to Existing Code                  |
|----------------------|-----------------------------------------------------------------------------|------------------------------------------------|
| **Session**          | Container for a creative work. Holds multiple StemTracks + arrangement data + metadata | Evolves from current "project" / "current editor" MCP resources |
| **StemTrack**        | A single lane. Contains one primary audio source + its edit history + latent cache (when available) | New; will eventually sit above `EditorLoadedAudio` |
| **Stem**             | The actual audio content + associated latents + generation provenance for one layer | Can be a full `HistoryEntry` or a derived/edited version |
| **Generative Region**| A time (and optionally frequency) range on a specific StemTrack that was created or modified via AI | Generalizes current `InpaintSelection` + `InpaintRequest` |
| **Stem Context**     | The set of other stems + global prompts that should influence generation or inpainting of this stem | New conditioning concept |
| **Lock State**       | Whether a stem is "locked" (plays as-is during preview) or available for editing | UI + engine concern |
| **Arrangement**      | The timeline placement, trimming, fades, and basic mixing data for all stems in the session | New (currently everything is single-buffer) |

---

## Relationship to Existing Systems

### Current Editor + InpaintCoordinator

The existing `InpaintCoordinator`, `MaskBuilder`, `LatentBlender`, `EditHistoryManager`, and the two diffusion paths (ACE-Step + Stable Audio) are **foundational** to this vision.

In the Generative Stem Editor, these components would be lifted to operate **per-stem** rather than on a single global buffer. The coordinator will need to become aware of which stem(s) an operation targets.

### Generation Pipelines

- **ACE-Step** (music) → Primary for structured musical stems (drums, bass, melody, harmony).
- **Stable Audio** → Primary for textural, atmospheric, and SFX stems.

The system should make it natural to use the right pipeline per stem type.

### Takes, History, and Provenance

Every stem should retain rich provenance:
- Original generation prompt + enhanced prompt
- Models and adapters used
- Trace / listening notes
- Preferred take status per stem

This is a major advantage over generic DAWs.

### LoRAs / Adapters

LoRAs become even more powerful when they can be applied per-stem or per-section of a stem.

### MCP Surface

Existing concepts like `anvil://take/{id}`, `anvil://takes`, `prepare_stems_for_review`, and the current "project/track/clip" resources should evolve to support stem-aware operations.

---

## Architectural Implications (High Level)

**Do not treat this as a simple UI layer on top of the current editor.**

Major shifts required:

- Move from single `AudioBuffer` + optional latents to a multi-stem data model.
- Real (or at least much better) multi-track playback engine (current `AVAudioPlayer` usage will not scale).
- Latent storage and management per stem (memory and disk strategy becomes critical).
- Stem-aware inpainting path that can accept cross-stem conditioning.
- Non-destructive editing model (clips/regions) rather than the current destructive-or-full-snapshot approach.
- Session persistence format that can reference many audio files + their generative histories.

See the "Phased Approach" section for a pragmatic way to get there.

---

## Recommended Phased Approach (for Implementation)

Codex should not attempt to build the full vision in one pass.

**Phase A — Foundation (Session + Stem Model)**
- Introduce `Session` and `StemTrack` concepts.
- Basic multi-lane UI (even if playback is still crude).
- Ability to add existing HistoryEntry takes as stems.
- Simple timeline with trimming and basic arrangement.
- Persist sessions.

**Phase B — Per-Stem Generative Editing**
- Lift the existing inpainting pipeline to work on individual stems.
- Add "Edit this stem" entry point that loads one stem into the existing high-quality editor while the rest of the session provides context.
- Basic lock/mute/solo per stem.
- Start with offline rendering for edited stems.

**Phase C — In-Context Preview & Coherence Tools**
- Real-time (or low-latency) multi-stem playback.
- Ability to perform inpainting while hearing other stems.
- First "musical glue" tools (e.g., generate transition region informed by adjacent stems).

**Phase D — Stem-Aware Generation & Advanced Conditioning**
- Native stem generation modes in the Generate tab that feed directly into Sessions.
- Cross-stem conditioning (reference other stems' audio or latents during generation).
- Per-stem adapter and parameter control.

**Phase E — Polish & Production Features**
- Proper mixing (faders, pans, sends, per-stem effects).
- Advanced arrangement tools.
- Export options (full mix, stems, stems + metadata).

This phasing allows the team to deliver real value early while building toward the full vision.

---

## Open Questions & Hard Problems

1. **Native stem generation vs post-hoc separation** — How much should we push users toward generating stems directly vs separating full mixes?
2. **Cross-stem coherence during independent editing** — What tools help the user keep musical sense when editing one stem heavily?
3. **Latent alignment across stems** — Different pipelines have different hop sizes and latent rates. How do we keep them synchronized for inpainting?
4. **Conditioning strategy** — How much audio/latent context from sibling stems should we feed into the model during inpainting or generation?
5. **Performance & memory** — Multiple sets of latents + undo histories per stem will be expensive. What are the caching and eviction strategies?
6. **Stem taxonomy** — Do we enforce categories (drums, bass, etc.) or let them be freeform?
7. **Phase and timing drift** — When stems are edited independently, how much automatic time/phase correction do we apply?

These should be treated as first-class design problems, not afterthoughts.

---

## Non-Goals (Initial Versions)

- Full traditional DAW feature parity (automation lanes, complex routing, MIDI, video, etc.).
- Real-time low-latency monitoring suitable for live recording.
- High track counts (start with 8–12 maximum).
- Mastering-grade final output chain.
- Cloud collaboration or multi-user sessions.

The goal is an exceptional **AI-native** creative tool, not a replacement for Logic, Ableton, or Reaper.

---

## Summary for Codex

The Generative Stem Editor is the bet that Anvil's future lies in **deep, iterative, stem-level generative work** rather than competing in the general DAW space.

It reuses and extends the excellent latent inpainting work from Phase 6, the take/history system, and the dual-pipeline architecture. The hardest parts will be the data model, the multi-stem audio engine, and the conditioning/coherence problems — not the traditional DAW UI chrome.

If built well, this direction has the potential to feel genuinely new and magical in a way that "DAW plus AI buttons" never will.

---

**Next steps for deeper work (when ready):**
- Detailed data model for Session / StemTrack / GenerativeRegion
- InpaintCoordinator evolution plan for multi-stem use
- Playback engine requirements
- UI wireframes / interaction model for the timeline + stem lanes

This document is intentionally conceptual. More detailed specs can be written per phase once direction is confirmed.