# Phase 8a: Time Stretch & Pitch Shift — Design Spec

## Overview

Time Stretch and Pitch Shift added to the existing DSP effects palette as a new
"Transforms" category. Both operations use Apple's `AVAudioUnitTimePitch` for
offline rendering via the same AVAudioEngine pattern established in Phase 7.

**Scope:** Two transform operations — Time Stretch (rate change without pitch
change) and Pitch Shift (pitch change without duration change). Both independent,
both support selection or whole-track targeting.

**Architecture:** Single `AVAudioUnitTimePitch` node per operation, dispatched
through the existing `DSPProcessor` facade. No new frameworks or dependencies.

**Phase 8b (future):** Custom Phase Vocoder / WSOLA backend with formant
preservation, replacing the Apple unit behind the same interface.

---

## Interaction Model

### Transforms in the DSP Palette

Time Stretch and Pitch Shift appear in a new "Transforms" sub-section within the
existing `DSPPaletteView`, separated from the Phase 7 effects by a divider and
sub-header.

**Palette layout (updated):**

| | Col 1 | Col 2 | Col 3 |
|---------|-------------------|------------------|-----------|
| EFFECTS | | | |
| Row 1 | Trim to Selection | Delete Selection | Normalize |
| Row 2 | Fade In | Fade Out | EQ |
| Row 3 | Reverb | | |
| | ─── divider ───── | | |
| TRANSFORMS | | | |
| Row 4 | Time Stretch | Pitch Shift | |

**Parameterized effects:** Both transforms expand an inline parameter panel when
clicked (same behavior as EQ and Reverb). Only one panel open at a time across
both sections — clicking Time Stretch collapses any open EQ/Reverb panel and
vice versa.

### Apply Target

Same smart default as Phase 7:

- If a region is selected on the waveform: defaults to `.selection(start, end)`
- If no selection: defaults to `.wholeTrack`

### Controls

Both transforms use independent controls — each has its own button, panel, and
Apply action. Users who want both apply one then the other.

### Keyboard Shortcuts

None for Phase 8a. Transform panels are parameter-heavy; shortcuts can be added
later if users request them.

---

## UI Design

### DSPPaletteView Changes

The existing palette splits into two visually distinct sections:

```text
VStack(spacing: 12) {
    // Effects section
    Text("EFFECTS")
        .font(.system(size: 10, weight: .semibold))
        .foregroundStyle(.secondary)
        .tracking(0.5)

    effectGrid  // existing 7 buttons (Phase 7)

    Divider()
        .padding(.vertical, 4)

    // Transforms section
    Text("TRANSFORMS")
        .font(.system(size: 10, weight: .semibold))
        .foregroundStyle(.secondary)
        .tracking(0.5)

    transformGrid  // Time Stretch + Pitch Shift
}
```

Section grouping driven by `DSPEffect.isTransform`:

```swift
let effects = DSPEffect.allCases.filter { !$0.isTransform }
let transforms = DSPEffect.allCases.filter { $0.isTransform }
```

### TimeStretchControlView

- **Rate slider:** 0.25x to 4.0x, default 1.0x
- **Snap points (behavior only, no visual dots):** 0.5x, 0.75x, 1.0x, 1.25x,
  1.5x, 2.0x
- **Snap threshold:** Round to nearest snap point when within 0.03 of it
- **Value label:** Shows rate with x suffix (e.g., "1.35x")
- **Direct entry:** Double-click value label to type exact number
- **Quality indicator:** Colored dot + text, updates live as slider moves
- **Apply target label:** "Apply to: Selection (1.2s - 2.8s)" or "Apply to:
  Whole Track"
- **Full-width Apply button**

### PitchShiftControlView

- **Pitch slider:** -24 to +24 semitones (internally -2400 to +2400 cents)
- **Snap points (behavior only):** -12, -7, -5, 0, +5, +7, +12 semitones
- **Snap threshold:** Round to nearest snap point when within 0.3 semitones
- **Value label:** Shows semitones + cents (e.g., "+4 st (+400c)")
- **Direct entry:** Double-click value label to type exact semitone value
- **Quality indicator:** Same colored dot + text as Time Stretch
- **Apply target label + full-width Apply button**

### Quality Zone Indicator

Three states based on current parameter value:

| Zone | Time Stretch Rate | Pitch Shift | Color | Text |
|--------|-------------------|----------------|--------|------|
| Green | 0.5x - 2.0x | +/-12 semitones | Green | "Excellent quality" |
| Yellow | 0.35x - 0.5x or 2.0x - 3.0x | +/-12 to +/-18 semitones | Yellow | "Good quality, minor artifacts possible" |
| Red | < 0.35x or > 3.0x | > +/-18 semitones | Red | "Significant artifacts likely" |

Boundary values (2.0x, +/-12 semitones) are inclusive in the green zone.

Quality zone is pure UI logic — no data model representation needed.

---

## Data Model

### TimeStretchParameters

```swift
struct TimeStretchParameters: Equatable {
    var rate: Float          // 0.25-4.0 (1.0 = original speed)
    static let `default` = TimeStretchParameters(rate: 1.0)
}
```

### PitchShiftParameters

```swift
struct PitchShiftParameters: Equatable {
    var pitch: Float         // -2400 to +2400 cents (-24 to +24 semitones)
    static let `default` = PitchShiftParameters(pitch: 0)
}
```

### DSPEffect Additions

```swift
enum DSPEffect: String, CaseIterable {
    // ... existing cases ...
    case timeStretch
    case pitchShift

    var isTransform: Bool {
        switch self {
        case .timeStretch, .pitchShift: return true
        default: return false
        }
    }

    var requiresParameters: Bool {
        switch self {
        case .eq, .reverb, .timeStretch, .pitchShift:
            return true
        default:
            return false
        }
    }

    var requiresSelection: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut:
            return true
        default:
            return false
        }
    }

    var displayName: String {
        switch self {
        // ... existing cases ...
        case .timeStretch: return "Time Stretch"
        case .pitchShift: return "Pitch Shift"
        }
    }
}
```

### DSPParameters Additions

```swift
enum DSPParameters: Equatable {
    case none
    case eq(EQParameters)
    case reverb(ReverbParameters)
    case timeStretch(TimeStretchParameters)
    case pitchShift(PitchShiftParameters)
}
```

---

## Processing Architecture

### AVAudioEngineDSP: Time Stretch

```swift
func applyTimeStretch(to buffer: AudioBuffer, target: ApplyTarget,
                      parameters: TimeStretchParameters) throws -> AudioBuffer
```

1. Create `AVAudioEngine` + `AVAudioPlayerNode` + `AVAudioUnitTimePitch`
2. Set `timePitch.rate = parameters.rate`, leave `timePitch.pitch = 0`
3. Connect: player -> timePitch -> mainMixer
4. Enable manual rendering mode (`.offline`)
5. Calculate output frame count: `Int(Double(inputFrames) / Double(rate))`
6. Create output `AVAudioPCMBuffer` with correct new frame capacity
7. Schedule input buffer, play, render offline in 4096-frame chunks
8. Set `outputBuffer.frameLength` to actual rendered frames
9. Stop engine, return output buffer

**Key difference from EQ/Reverb:** Time stretch changes the buffer length.

**Region targeting:** When target is `.selection`, extract the region, process it
(producing a different-length result), splice back into the original buffer at
the same start position. Total buffer duration changes. Selection is cleared
after apply (same as trim/delete) since region boundaries shift.

### AVAudioEngineDSP: Pitch Shift

```swift
func applyPitchShift(to buffer: AudioBuffer, target: ApplyTarget,
                     parameters: PitchShiftParameters) throws -> AudioBuffer
```

1. Same engine graph: player -> timePitch -> mainMixer
2. Set `timePitch.pitch = parameters.pitch` (in cents), leave `timePitch.rate = 1.0`
3. Offline render in 4096-frame chunks
4. No length change — pitch shift preserves duration exactly
5. Region targeting: extract, process, splice back at same position (identical
   to EQ/Reverb pattern)

### DSPProcessor Dispatch

```swift
case .timeStretch:
    guard case .timeStretch(let params) = parameters else { return buffer }
    return try avEngine.applyTimeStretch(to: buffer, target: target, parameters: params)
case .pitchShift:
    guard case .pitchShift(let params) = parameters else { return buffer }
    return try avEngine.applyPitchShift(to: buffer, target: target, parameters: params)
```

### DSPProcessor Description

`makeDescription` accepts `DSPParameters` to include parameter values in undo
labels:

- "Time Stretch 1.35x (1.2s - 2.8s)"
- "Time Stretch 0.5x (whole track)"
- "Pitch Shift +4 st (whole track)"
- "Pitch Shift -12 st (0.5s - 3.0s)"

---

## Integration with Phase 7 Editor

### EditorState Additions

```swift
// New published properties
@Published var timeStretchParameters: TimeStretchParameters = .default
@Published var pitchShiftParameters: PitchShiftParameters = .default
```

Existing infrastructure handles the rest:

- `expandedEffect: DSPEffect?` — already supports any `DSPEffect` case
- `applyDSPEffect(_:parameters:)` — already dispatches through `DSPProcessor`,
  updates `audioBuffer`, `duration`, and `waveformPeaks`
- Selection cleared after time stretch in existing `applyDSPEffect` logic
  (extend the existing trim/delete clearing check)

### EditorView Changes

No changes needed. `DSPPaletteView` is already embedded in `EditorView` and
receives `state` + `onApply` closure. The palette handles the new transforms
internally.

---

## File Structure

### Modified Files

```text
AnvilApp/DSP/
├── DSPParameters.swift          ← add TimeStretchParameters, PitchShiftParameters,
│                                   DSPEffect cases, DSPParameters cases
├── AVAudioEngineDSP.swift       ← add applyTimeStretch, applyPitchShift
└── DSPProcessor.swift           ← add dispatch cases + description formatting

AnvilApp/UI/Editor/
├── DSPPaletteView.swift         ← split into Effects/Transforms sections
└── EditorState.swift            ← add 2 @Published properties, extend selection clearing
```

### New Files

```text
AnvilApp/UI/Editor/
├── TimeStretchControlView.swift   ← rate slider + snap + quality indicator
└── PitchShiftControlView.swift    ← pitch slider + snap + quality indicator
```

---

## Scope

### In Scope (Phase 8a)

- Time Stretch via AVAudioUnitTimePitch (rate: 0.25x - 4.0x)
- Pitch Shift via AVAudioUnitTimePitch (pitch: +/-24 semitones / +/-2400 cents)
- Independent controls — each applied separately
- Quality zone indicator (green/yellow/red) based on parameter thresholds
- Linear sliders with snap-point behavior (no visual dots)
- Double-click value label for direct numeric entry
- "Transforms" sub-section in DSPPaletteView with divider and header
- Region targeting (selection or whole track) — same pattern as Phase 7
- Undo integration via existing EditHistoryManager
- Selection cleared after time stretch (duration change shifts boundaries)

### Out of Scope (Phase 8b / future)

- Custom Phase Vocoder / WSOLA backend
- Formant preservation for vocal pitch shifting
- Linked time/pitch control (vinyl speed-up mode)
- Real-time preview / audition before applying
- Visual snap dots on sliders
- Keyboard shortcuts for transform panels
- Auto-normalize after transform
- Preset stretch/pitch values (e.g., "Half Speed", "Up One Octave")
