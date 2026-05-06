# Phase 8b: Custom DSP Backend — Design Spec

## Overview

Replace the AVAudioUnitTimePitch backend from Phase 8a with custom DSP
algorithms for higher quality time stretching and pitch shifting. WSOLA for
time stretch, resampling + WSOLA + LPC formant correction for pitch shift.

**Scope:** Three new processing modules (WSOLAProcessor, Resampler,
LPCFormantShifter) replacing two AVAudioEngineDSP methods. No UI changes —
the backend swap is invisible to the user.

**Architecture:** Sequential pipeline with clean boundaries. Each processor
is a pure function over `[Float]` arrays. DSPProcessor facade routes to the
new implementations. Region targeting moves from AVAudioEngineDSP to
DSPProcessor.

**Key improvement over Phase 8a:** Formant preservation on pitch shift
eliminates the "chipmunk effect" on vocals that AVAudioUnitTimePitch produces
at large pitch changes.

---

## Processing Pipelines

### Time Stretch

```text
input [Float] → WSOLAProcessor.stretch(samples, rate) → output [Float]
```

Single stage. WSOLA handles time-domain stretching/compression without
changing pitch.

### Pitch Shift

```text
input [Float] → Resampler.resample(ratio: pitchRatio)
             → WSOLAProcessor.stretch(rate: 1.0 / pitchRatio)
             → LPCFormantShifter.correct(original: input, shifted: result)
             → output [Float]
```

Three stages:

1. **Resample** — change pitch by altering sample rate (also changes duration)
2. **WSOLA** — restore original duration (compensate for resampling)
3. **LPC Formant Correction** — restore original vocal tract resonances

Where `pitchRatio = pow(2.0, Float(cents) / 1200.0)`.

---

## Algorithm: WSOLA Time Stretch

### WSOLAProcessor

WSOLA (Waveform Similarity Overlap-Add) stretches audio in the time domain
by finding optimal overlap points that maximize waveform continuity.

**Algorithm:**

1. Divide input into overlapping analysis frames
2. For each output frame position:
   - Calculate ideal analysis position: `analysisPos = outputFrameIndex * analysisHop`
   - Search ±searchRange samples around `analysisPos` for the position that
     maximizes cross-correlation with the previous output frame's tail
   - Extract the best-matching frame, apply Hann window, overlap-add into
     output buffer
3. Output hop = `analysisHop * rate`

**API:**

```swift
struct WSOLAProcessor {
    static func stretch(_ samples: [Float], rate: Float,
                        sampleRate: Int, channels: Int) -> [Float]
}
```

**Parameters (internal constants):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window size | 1024 samples | ~23ms at 44.1kHz, good for music |
| Overlap ratio | 75% | Standard for WSOLA quality |
| Analysis hop | 256 samples | window / 4 |
| Search range | ±64 samples | ~±1.5ms, sufficient for waveform alignment |

**Edge cases:**

- Rate = 1.0: return input unchanged (skip processing)
- Very short buffers (< 2 windows): return input unchanged
- Energy gating: if frame RMS < -60dB, skip similarity search and use ideal
  position directly

**Stereo handling:** De-interleave → process each channel independently →
re-interleave.

---

## Algorithm: Resampler

### Resampler

Changes effective sample rate using linear interpolation, which shifts pitch
proportionally. Used as the first stage of pitch shifting.

**Algorithm:**

1. Calculate output length: `Int(Double(inputLength) / ratio)`
2. For each output sample, compute fractional input position:
   `inputPos = Double(outputIndex) * ratio`
3. Linear interpolation: `output = input[floor] * (1 - frac) + input[ceil] * frac`

**API:**

```swift
struct Resampler {
    static func resample(_ samples: [Float], ratio: Float, channels: Int) -> [Float]
}
```

- `ratio > 1.0` = pitch up (shorter output)
- `ratio < 1.0` = pitch down (longer output)
- Ratio = 1.0: return input unchanged
- Ratio clamped to 0.25...4.0 (matches Phase 8a slider range)

**Why linear interpolation:** The mild high-frequency roll-off from linear
interpolation is masked by subsequent WSOLA processing and LPC formant
correction. Sinc interpolation adds ~100 lines of complexity for negligible
audible benefit in this pipeline.

**Stereo handling:** De-interleave → resample each channel → re-interleave.

---

## Algorithm: LPC Formant Correction

### LPCFormantShifter

Estimates the spectral envelope (formants) of the original audio via Linear
Predictive Coding, then applies it to the pitch-shifted result. This
preserves the vocal tract resonances that make a voice sound natural.

**Algorithm — three stages per frame:**

**Stage 1: Analyze original audio**

1. Window the frame (Hann, 1024 samples)
2. Compute autocorrelation coefficients (via vDSP)
3. Solve Levinson-Durbin recursion → LPC coefficients (order 24)
4. These coefficients define the original spectral envelope

**Stage 2: Analyze shifted audio**

1. Same process on the corresponding frame of the pitch-shifted output
2. These coefficients define the shifted spectral envelope (with formant
   distortion)

**Stage 3: Apply correction**

1. Inverse-filter the shifted frame with shifted LPC coefficients → flat
   residual (removes distorted envelope)
2. Re-filter the residual with original LPC coefficients → restores natural
   vocal envelope
3. Overlap-add corrected frames into output

**API:**

```swift
struct LPCFormantShifter {
    static func correct(original: [Float], shifted: [Float],
                        sampleRate: Int, channels: Int) -> [Float]
}
```

**Parameters (internal constants):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LPC order | 24 | Standard for 44.1kHz vocal formant modeling |
| Frame size | 1024 samples | Matches WSOLA window for alignment |
| Hop size | 256 samples | 75% overlap, matches WSOLA |
| Window | Hann | Standard for LPC analysis |

**Levinson-Durbin implementation:**

Standard recursive algorithm (~40 lines) operating on autocorrelation values.
Returns LPC coefficients and prediction error. Reflection coefficients
clamped to ±0.99 for numerical stability. vDSP used for autocorrelation
computation (`vDSP_conv`), recursion in plain Swift.

**Energy gating:**

- Frame RMS below -60dB: skip LPC analysis, pass shifted frame through
  unchanged
- Prevents LPC instability on silent or near-silent frames

**Frame alignment:**

Original and shifted arrays may differ slightly in length due to
resampling/WSOLA rounding. Process `min(originalFrames, shiftedFrames)` frames.
Trailing samples passed through uncorrected.

**Stereo handling:** Process channels independently.

---

## Integration

### DSPProcessor Changes

The `apply` method's `.timeStretch` and `.pitchShift` cases change dispatch
from AVAudioEngineDSP to the new processors:

```swift
case .timeStretch:
    guard case .timeStretch(let params) = parameters else { return buffer }
    let floats = AudioBufferConverter.toFloats(buffer)
    let stretched = WSOLAProcessor.stretch(floats, rate: params.rate,
                                           sampleRate: buffer.sampleRate,
                                           channels: buffer.channels)
    return AudioBufferConverter.fromFloats(stretched,
                                          sampleRate: buffer.sampleRate,
                                          channels: buffer.channels)

case .pitchShift:
    guard case .pitchShift(let params) = parameters else { return buffer }
    let floats = AudioBufferConverter.toFloats(buffer)
    let pitchRatio = pow(2.0, params.pitch / 1200.0)
    let resampled = Resampler.resample(floats, ratio: pitchRatio,
                                       channels: buffer.channels)
    let timeCorrected = WSOLAProcessor.stretch(resampled,
                                               rate: 1.0 / pitchRatio,
                                               sampleRate: buffer.sampleRate,
                                               channels: buffer.channels)
    let corrected = LPCFormantShifter.correct(original: floats,
                                              shifted: timeCorrected,
                                              sampleRate: buffer.sampleRate,
                                              channels: buffer.channels)
    return AudioBufferConverter.fromFloats(corrected,
                                          sampleRate: buffer.sampleRate,
                                          channels: buffer.channels)
```

### Region Targeting

Extract-process-splice logic moves from AVAudioEngineDSP to a private
`processRegion` helper in DSPProcessor:

```swift
private func processRegion(
    _ buffer: AudioBuffer,
    start: TimeInterval, end: TimeInterval,
    transform: (AudioBuffer) throws -> AudioBuffer
) throws -> AudioBuffer
```

Same byte-level splicing pattern used by AVAudioEngineDSP's `applyToRegion`.
The `.timeStretch` and `.pitchShift` dispatch cases check for `.selection`
target and route through `processRegion` when needed.

### AVAudioEngineDSP Cleanup

Remove four methods:

- `applyTimeStretch(to:parameters:target:)`
- `applyTimeStretchToBuffer(_:parameters:)`
- `applyPitchShift(to:parameters:target:)`
- `applyPitchShiftToBuffer(_:parameters:)`

Keep everything else unchanged: `applyEQ`, `applyReverb`, `applyToRegion`
(still used by EQ/Reverb).

### No UI Changes

TimeStretchControlView, PitchShiftControlView, DSPPaletteView, EditorState,
and EditorView are all unchanged. The backend swap is completely transparent.

---

## File Structure

### New Files

```text
AnvilApp/DSP/
├── WSOLAProcessor.swift         ← WSOLA time stretch (~150-200 lines)
├── Resampler.swift              ← linear interpolation resampler (~60-80 lines)
└── LPCFormantShifter.swift      ← LPC formant correction (~150-180 lines)
```

### Modified Files

```text
AnvilApp/DSP/
├── DSPProcessor.swift           ← update 2 dispatch cases, add processRegion helper
└── AVAudioEngineDSP.swift       ← remove 4 time stretch/pitch shift methods
```

### Unchanged Files

```text
AnvilApp/DSP/
├── DSPParameters.swift
├── AudioBufferConverter.swift
├── SimpleDSP.swift

AnvilApp/UI/Editor/
├── DSPPaletteView.swift
├── TimeStretchControlView.swift
├── PitchShiftControlView.swift
├── EQControlView.swift
├── ReverbControlView.swift
├── EditorState.swift
├── EditorView.swift
```

---

## Scope

### In Scope (Phase 8b)

- WSOLA time stretch (window 1024, overlap 75%, search ±64, energy gating)
- Linear resampler for pitch ratio conversion (clamped 0.25-4.0)
- LPC formant correction (order 24, Levinson-Durbin, energy gating)
- Sequential pitch shift pipeline: resample → WSOLA → LPC correct
- DSPProcessor dispatch update (route to new processors)
- Region targeting moved to DSPProcessor (`processRegion` helper)
- AVAudioEngineDSP cleanup (remove time stretch / pitch shift methods)
- Stereo support (process channels independently, re-interleave)
- Edge cases: rate=1.0 bypass, short buffer fallback, energy gating on silence
- Reflection coefficient clamping (±0.99) for LPC stability

### Out of Scope (future)

- Sinc / polyphase resampler upgrade
- Phase Vocoder alternative backend
- Transient detection / preservation
- Adaptive WSOLA window sizing
- User-facing formant preservation toggle (on/off)
- Real-time preview / audition
- Cepstral or True Envelope alternative to LPC
- Quality mode toggle (Apple vs custom backend)
- GUI for internal algorithm parameters (window size, LPC order, search range)
