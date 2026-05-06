# Phase 7: DSP Audio Editor — Design Spec

## Overview

Destructive DSP audio editing integrated into the existing Edit tab as an
always-available effects palette. Users select a region on the waveform (or target
the whole track), pick a DSP effect, adjust parameters, and apply. Each operation
snapshots the audio before applying, using the same undo system from Phase 6.

**Scope:** 5 operations — Trim to Selection, Delete Selection, Fade In/Out,
Normalize (peak), 3-band Parametric EQ, Reverb (preset + tweak).

**Architecture:** Hybrid processing — plain Swift/vDSP for simple operations
(Trim, Fade, Normalize), AVAudioEngine offline rendering for EQ and Reverb.
All code lives in the app target (`AnvilApp/DSP/`). AnvilCore remains focused
on ML inference.

---

## Interaction Model

### Effects Palette

The DSP palette is always visible in the Edit tab below the AI controls section,
separated by a divider. No mode switching required — AI and DSP tools coexist.

**Palette layout:** 3-column button grid showing all available effects:

| Row 1 | Trim to Selection | Delete Selection | Normalize |
|-------|-------------------|------------------|-----------|
| Row 2 | Fade In | Fade Out | EQ |
| Row 3 | Reverb | | |

**Instant-apply effects:** Trim to Selection, Delete Selection, Fade In, Fade Out,
and Normalize have no parameter panels — they apply immediately on click with a
brief checkmark overlay (0.5s) as confirmation.

**Selection-required effects:** Trim to Selection, Delete Selection, Fade In, and
Fade Out require an active waveform selection. Their buttons are disabled (grayed
out) when no region is selected. Normalize, EQ, and Reverb can apply to either
selection or whole track.

**Parameterized effects:** EQ and Reverb expand an inline parameter panel when
clicked. Only one panel open at a time. Last-used effect stays expanded with
previous parameter values preserved.

### Apply Target

Smart default based on current selection state:

- If a region is selected on the waveform: defaults to `.selection(start, end)`
- If no selection: defaults to `.wholeTrack`
- User can toggle via a labeled indicator showing the current target

### Keyboard Shortcuts

- `Cmd+T` — Trim to Selection
- `Cmd+Delete` — Delete Selection
- `Cmd+Shift+N` — Normalize
- `Cmd+Shift+E` — Expand/focus EQ panel
- `Cmd+Shift+R` — Expand/focus Reverb panel

---

## UI Design

### Edit Tab Layout (updated from Phase 6)

```text
EditorToolbar
ScrollView {
    EditorWaveformView
    Frequency Constraint (disclosure group)
    EditorPromptArea
    EditorControlsBar
    ─── Divider ───
    DSPPaletteView              ← NEW (Phase 7)
}
```

### DSPPaletteView

- Section header: "DSP Effects" (uppercase, 12px, secondary color)
- 3-column button grid with compact buttons
- Expanded parameter panel (blue-tinted background) when EQ or Reverb is selected
- Chevron icon (▼/▶) for collapse/expand instead of text
- "Apply to: [Selection (1.2s–2.8s)]" indicator before the Apply button
- Full-width Apply button with gradient background

### EQ Panel (expanded)

Three band sections, each showing:

- Band label (Band 1 — Low, Band 2 — Mid, Band 3 — High)
- Three values: Frequency (Hz/kHz), Gain (dB, green for boost, red for cut), Q
- Sliders for each parameter

### Reverb Panel (expanded)

- Preset row: Room / Hall / Plate (segmented buttons, selecting a preset fills
  sliders with its values; adjusting a slider deselects the preset)
- Four parameter sliders: Decay Time, Wet/Dry Mix, Pre-Delay, Damping
- Same "Apply to" indicator + Apply button

---

## Data Model

### DSPEffect

```swift
enum DSPEffect: String, CaseIterable {
    case trimToSelection
    case deleteSelection
    case fadeIn
    case fadeOut
    case normalize
    case eq
    case reverb

    var requiresParameters: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut, .normalize:
            return false
        case .eq, .reverb:
            return true
        }
    }

    var displayName: String {
        switch self {
        case .trimToSelection: return "Trim to Selection"
        case .deleteSelection: return "Delete Selection"
        case .fadeIn: return "Fade In"
        case .fadeOut: return "Fade Out"
        case .normalize: return "Normalize"
        case .eq: return "Parametric EQ"
        case .reverb: return "Reverb"
        }
    }
}
```

### ApplyTarget

```swift
enum ApplyTarget: Equatable {
    case selection(start: TimeInterval, end: TimeInterval)
    case wholeTrack
}
```

### AudioBuffer

```swift
struct AudioBuffer: Equatable {
    var data: Data          // 16-bit interleaved PCM
    var sampleRate: Int
    var channels: Int
}
```

### DSPParameters

```swift
enum DSPParameters: Equatable {
    case none
    case eq(EQParameters)
    case reverb(ReverbParameters)
}
```

### EQParameters

```swift
struct EQBand: Equatable {
    var frequency: Float    // 20–20000 Hz
    var gain: Float         // -24.0 to +24.0 dB
    var q: Float            // 0.1 to 10.0
}

struct EQParameters: Equatable {
    var bands: [EQBand]     // always 3 bands

    static let `default` = EQParameters(bands: [
        EQBand(frequency: 200, gain: 0, q: 1.0),
        EQBand(frequency: 1200, gain: 0, q: 1.4),
        EQBand(frequency: 8000, gain: 0, q: 0.8),
    ])
}
```

### ReverbPreset + ReverbParameters

```swift
enum ReverbPreset: String, CaseIterable {
    case room, hall, plate

    var parameters: ReverbParameters {
        switch self {
        case .room: return .room
        case .hall: return .hall
        case .plate: return .plate
        }
    }
}

struct ReverbParameters: Equatable {
    var decayTime: Float    // 0.1–10.0 seconds
    var wetDryMix: Float    // 0.0–1.0
    var preDelay: Float     // 0–100 ms
    var damping: Float      // 0.0–1.0

    static let room  = ReverbParameters(decayTime: 0.8, wetDryMix: 0.25, preDelay: 10, damping: 0.5)
    static let hall  = ReverbParameters(decayTime: 2.4, wetDryMix: 0.35, preDelay: 20, damping: 0.6)
    static let plate = ReverbParameters(decayTime: 1.5, wetDryMix: 0.40, preDelay: 5, damping: 0.3)
    static let `default` = ReverbParameters.hall
}
```

### NormalizeMode

```swift
enum NormalizeMode: Equatable {
    case peak
    case rms(targetDB: Float)
}
```

Phase 7 uses `.peak` exclusively. RMS normalization is reserved for future use.

### DSPOperation

```swift
struct DSPOperation: Equatable {
    let effect: DSPEffect
    let target: ApplyTarget
    let description: String
}
```

Used for undo popover descriptions (e.g., "EQ (1.2s–2.8s)", "Normalize (whole track)").

---

## Processing Architecture

### DSPProcessor (facade)

```swift
struct DSPProcessor {
    func apply(
        effect: DSPEffect,
        to buffer: AudioBuffer,
        target: ApplyTarget,
        parameters: DSPParameters = .none
    ) throws -> AudioBuffer
}
```

Dispatches to `SimpleDSP` or `AVAudioEngineDSP` based on the effect type.
Returns a new `AudioBuffer` — never mutates the input.

### SimpleDSP

Pure functions operating on `[Float]` sample arrays:

- **Trim to Selection:** Extract samples in `[startSample...endSample]`, return
  new buffer. `startSample = Int(startTime * Double(sampleRate)) * channels`
- **Delete Selection:** Concatenate `[0..<startSample] + [endSample...]`
- **Fade In:** Multiply target region samples by linear ramp `0.0 → 1.0`
- **Fade Out:** Multiply target region samples by linear ramp `1.0 → 0.0`
- **Normalize (peak):** Find peak via `vDSP_maxmgv`. If peak is already >= 0.99,
  skip processing. Otherwise divide all samples by peak value.

### AVAudioEngineDSP

Offline rendering using temporary AVAudioEngine graph:

**EQ workflow:**

1. Create `AVAudioEngine` + `AVAudioPlayerNode` + `AVAudioUnitEQ(numberOfBands: 3)`
2. Configure each band: `.parametric` filter type, frequency, gain, bandwidth (Q)
3. Connect: player → EQ → mainMixer
4. Enable manual rendering mode (`.offline`)
5. Schedule input buffer, play, render offline
6. Stop engine, return output buffer

**Reverb workflow:**

1. Create `AVAudioEngine` + `AVAudioPlayerNode` + `AVAudioUnitReverb`
2. Load factory preset closest to chosen preset (`.largeHall`, `.mediumRoom`,
   `.plate`)
3. Set `wetDryMix` (scaled to 0–100 range)
4. Connect: player → reverb → mainMixer
5. Render offline, return output buffer

**Region targeting for EQ/Reverb:** When target is `.selection`, extract the
region from the full buffer, process only that region, then splice it back into
the original buffer at the same position.

**Engine cleanup:** Always stop engine and release nodes after rendering to avoid
resource leaks.

### AudioBufferConverter

Utility for format conversion:

```swift
struct AudioBufferConverter {
    static func toFloats(_ buffer: AudioBuffer) -> [Float]
    static func fromFloats(_ samples: [Float], sampleRate: Int, channels: Int) -> AudioBuffer
    static func toPCMBuffer(_ buffer: AudioBuffer) -> AVAudioPCMBuffer
    static func fromPCMBuffer(_ pcmBuffer: AVAudioPCMBuffer) -> AudioBuffer
}
```

---

## Integration with Phase 6 Editor

### EditorState Additions

```swift
// New published properties
@Published var expandedEffect: DSPEffect? = nil
@Published var eqParameters: EQParameters = .default
@Published var reverbParameters: ReverbParameters = .default
@Published var selectedReverbPreset: ReverbPreset? = .hall

// New method
func applyDSPEffect(_ effect: DSPEffect, parameters: DSPParameters = .none) {
    let target: ApplyTarget = hasSelection
        ? .selection(start: selectionStart, end: selectionEnd)
        : .wholeTrack

    // 1. Snapshot current audio via EditHistoryManager
    // 2. Call DSPProcessor.apply(effect:to:target:parameters:)
    // 3. Update audioBuffer + recompute waveformPeaks
    // 4. Show brief checkmark confirmation (instant-apply effects)
}
```

### EditorView Changes

Add `DSPPaletteView` below `EditorControlsBar` with a `Divider()` separator.
The palette is only shown when `state.isLoaded` is true (same condition as the
rest of the editor content).

### Undo Integration

DSP operations use the same `EditHistoryManager` as AI edits. The undo popover
shows both AI and DSP operations in chronological order. Undo descriptions come
from `DSPOperation.description`.

---

## File Structure

```text
AnvilApp/
├── DSP/
│   ├── DSPProcessor.swift          // Facade: dispatches to SimpleDSP or AVAudioEngineDSP
│   ├── SimpleDSP.swift             // Trim, Delete, Fade, Normalize (Swift + vDSP)
│   ├── AVAudioEngineDSP.swift      // EQ + Reverb (offline AVAudioEngine rendering)
│   ├── DSPParameters.swift         // All parameter types + AudioBuffer
│   └── AudioBufferConverter.swift  // Data ↔ [Float] ↔ AVAudioPCMBuffer
├── UI/
│   └── Editor/
│       ├── DSPPaletteView.swift    // Effect buttons + expanded panels
│       ├── EQControlView.swift     // 3-band parametric EQ controls
│       └── ReverbControlView.swift // Preset picker + parameter sliders
```

---

## Scope

### In Scope

- Trim to Selection + Delete Selection
- Fade In / Fade Out (linear ramp)
- Normalize (peak)
- 3-band Parametric EQ (Freq/Gain/Q per band)
- Reverb (Room/Hall/Plate presets + Decay/Wet-Dry/Pre-Delay/Damping)
- DSP Palette UI in Edit tab
- Integration with existing undo/redo system
- Keyboard shortcuts for common operations
- Smart apply target (selection vs whole track)

### Out of Scope

- Time Stretch / Pitch Shift (Phase 7b or later)
- Non-destructive effect chains / live rendering
- Spectrogram visualization
- Custom fade curves (exponential, S-curve)
- RMS normalization UI
- Effect presets / user-saved parameter sets
- Batch processing / multi-region effects
