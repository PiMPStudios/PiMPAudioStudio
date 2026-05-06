# Phase 8b: Custom DSP Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AVAudioUnitTimePitch backend with custom WSOLA + LPC algorithms for higher quality time stretching and pitch shifting with formant preservation.

**Architecture:** Three new pure-function processors (Resampler, WSOLAProcessor, LPCFormantShifter) operating on `[Float]` arrays. DSPProcessor facade routes to new implementations transparently. No UI changes.

**Tech Stack:** Swift, Accelerate/vDSP, existing AudioBufferConverter bridge

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|------|---------------|--------|
| `AnvilApp/DSP/Resampler.swift` | Linear interpolation pitch ratio conversion | 60-80 |
| `AnvilApp/DSP/WSOLAProcessor.swift` | WSOLA time-domain time stretching | 150-200 |
| `AnvilApp/DSP/LPCFormantShifter.swift` | LPC formant correction via Levinson-Durbin | 150-180 |

### Modified Files

| File | Changes |
|------|---------|
| `AnvilApp/DSP/DSPProcessor.swift` | Add `processRegion` helper, update `.timeStretch` and `.pitchShift` dispatch |
| `AnvilApp/DSP/AVAudioEngineDSP.swift` | Remove 4 methods: `applyTimeStretch`, `applyTimeStretchToBuffer`, `applyPitchShift`, `applyPitchShiftToBuffer` |

---

### Task 1: Resampler

**Files:**

- Create: `AnvilApp/DSP/Resampler.swift`

- [ ] **Step 1: Create Resampler.swift with linear interpolation**

```swift
import Accelerate

struct Resampler {
    static func resample(_ samples: [Float], ratio: Float, channels: Int) -> [Float] {
        guard ratio != 1.0 else { return samples }

        let clampedRatio = min(max(ratio, 0.25), 4.0)
        let frameCount = samples.count / channels
        let outputFrameCount = Int(Double(frameCount) / Double(clampedRatio))
        guard outputFrameCount > 0 else { return samples }

        if channels == 1 {
            return resampleChannel(samples, inputFrames: frameCount, outputFrames: outputFrameCount, ratio: clampedRatio)
        }

        var deinterleaved = [[Float]](repeating: [Float](repeating: 0, count: frameCount), count: channels)
        for frame in 0..<frameCount {
            for ch in 0..<channels {
                deinterleaved[ch][frame] = samples[frame * channels + ch]
            }
        }

        var resampledChannels = [[Float]]()
        for ch in 0..<channels {
            resampledChannels.append(
                resampleChannel(deinterleaved[ch], inputFrames: frameCount, outputFrames: outputFrameCount, ratio: clampedRatio)
            )
        }

        let actualOutputFrames = resampledChannels[0].count
        var interleaved = [Float](repeating: 0, count: actualOutputFrames * channels)
        for frame in 0..<actualOutputFrames {
            for ch in 0..<channels {
                interleaved[frame * channels + ch] = resampledChannels[ch][frame]
            }
        }
        return interleaved
    }

    private static func resampleChannel(_ input: [Float], inputFrames: Int, outputFrames: Int, ratio: Float) -> [Float] {
        var output = [Float](repeating: 0, count: outputFrames)
        let doubleRatio = Double(ratio)

        for i in 0..<outputFrames {
            let inputPos = Double(i) * doubleRatio
            let floor = Int(inputPos)
            let frac = Float(inputPos - Double(floor))

            let idx0 = min(floor, inputFrames - 1)
            let idx1 = min(floor + 1, inputFrames - 1)

            output[i] = input[idx0] * (1.0 - frac) + input[idx1] * frac
        }
        return output
    }
}
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/Resampler.swift
git commit -m "feat(dsp): add linear interpolation resampler for pitch ratio conversion"
```

---

### Task 2: WSOLAProcessor

**Files:**

- Create: `AnvilApp/DSP/WSOLAProcessor.swift`

- [ ] **Step 1: Create WSOLAProcessor.swift with WSOLA algorithm**

```swift
import Accelerate

struct WSOLAProcessor {
    private static let windowSize = 1024
    private static let overlapRatio: Float = 0.75
    private static let analysisHop = 256
    private static let searchRange = 64
    private static let silenceThresholdDB: Float = -60.0

    static func stretch(_ samples: [Float], rate: Float, sampleRate: Int, channels: Int) -> [Float] {
        guard rate != 1.0 else { return samples }

        let frameCount = samples.count / channels
        guard frameCount >= windowSize * 2 else { return samples }

        if channels == 1 {
            return stretchChannel(samples, frameCount: frameCount, rate: rate)
        }

        var deinterleaved = [[Float]](repeating: [Float](repeating: 0, count: frameCount), count: channels)
        for frame in 0..<frameCount {
            for ch in 0..<channels {
                deinterleaved[ch][frame] = samples[frame * channels + ch]
            }
        }

        var stretchedChannels = [[Float]]()
        for ch in 0..<channels {
            stretchedChannels.append(stretchChannel(deinterleaved[ch], frameCount: frameCount, rate: rate))
        }

        let outputFrames = stretchedChannels[0].count
        var interleaved = [Float](repeating: 0, count: outputFrames * channels)
        for frame in 0..<outputFrames {
            for ch in 0..<channels {
                interleaved[frame * channels + ch] = stretchedChannels[ch][frame]
            }
        }
        return interleaved
    }

    private static func stretchChannel(_ input: [Float], frameCount: Int, rate: Float) -> [Float] {
        let synthesisHop = Int(Float(analysisHop) / rate)
        let outputLength = Int(Float(frameCount) / rate)
        var output = [Float](repeating: 0, count: outputLength)
        var window = [Float](repeating: 0, count: windowSize)
        vDSP_hann_window(&window, vDSP_Length(windowSize), Int32(vDSP_HANN_NORM))

        var analysisPos = 0
        var synthesisPos = 0

        while synthesisPos + windowSize <= outputLength && analysisPos + windowSize <= frameCount {
            let bestPos: Int
            if isFrameSilent(input, at: analysisPos) {
                bestPos = analysisPos
            } else if synthesisPos == 0 {
                bestPos = analysisPos
            } else {
                bestPos = findBestMatch(input: input, frameCount: frameCount,
                                        idealPos: analysisPos, prevOutput: output,
                                        prevEnd: synthesisPos)
            }

            for i in 0..<windowSize {
                let srcIdx = bestPos + i
                guard srcIdx < frameCount else { break }
                output[synthesisPos + i] += input[srcIdx] * window[i]
            }

            analysisPos += analysisHop
            synthesisPos += synthesisHop
        }

        normalizeOverlap(&output, synthesisHop: synthesisHop, window: window)
        return Array(output.prefix(max(synthesisPos, 0)))
    }

    private static func findBestMatch(input: [Float], frameCount: Int,
                                       idealPos: Int, prevOutput: [Float],
                                       prevEnd: Int) -> Int {
        let searchStart = max(0, idealPos - searchRange)
        let searchEnd = min(frameCount - windowSize, idealPos + searchRange)
        guard searchStart < searchEnd else { return idealPos }

        let refStart = max(0, prevEnd - analysisHop)
        let refLength = min(analysisHop, prevEnd - refStart)
        guard refLength > 0 else { return idealPos }

        var bestPos = idealPos
        var bestCorrelation: Float = -.greatestFiniteMagnitude

        for candidate in searchStart...searchEnd {
            let overlapLength = min(refLength, frameCount - candidate)
            guard overlapLength > 0 else { continue }

            var correlation: Float = 0
            vDSP_dotpr(
                Array(prevOutput[refStart..<refStart + overlapLength]), 1,
                Array(input[candidate..<candidate + overlapLength]), 1,
                &correlation,
                vDSP_Length(overlapLength)
            )

            if correlation > bestCorrelation {
                bestCorrelation = correlation
                bestPos = candidate
            }
        }
        return bestPos
    }

    private static func isFrameSilent(_ samples: [Float], at position: Int) -> Bool {
        let end = min(position + windowSize, samples.count)
        let length = end - position
        guard length > 0 else { return true }

        var rms: Float = 0
        vDSP_rmsqv(Array(samples[position..<end]), 1, &rms, vDSP_Length(length))

        let db = 20.0 * log10(max(rms, 1e-10))
        return db < silenceThresholdDB
    }

    private static func normalizeOverlap(_ output: inout [Float], synthesisHop: Int, window: [Float]) {
        let windowSize = window.count
        var envelope = [Float](repeating: 0, count: output.count)
        var pos = 0
        while pos + windowSize <= output.count {
            for i in 0..<windowSize {
                envelope[pos + i] += window[i]
            }
            pos += synthesisHop
        }

        for i in 0..<output.count {
            if envelope[i] > 0.001 {
                output[i] /= envelope[i]
            }
        }
    }
}
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/WSOLAProcessor.swift
git commit -m "feat(dsp): add WSOLA time stretch processor"
```

---

### Task 3: LPCFormantShifter

**Files:**

- Create: `AnvilApp/DSP/LPCFormantShifter.swift`

- [ ] **Step 1: Create LPCFormantShifter.swift with Levinson-Durbin and formant correction**

```swift
import Accelerate

struct LPCFormantShifter {
    private static let lpcOrder = 24
    private static let frameSize = 1024
    private static let hopSize = 256
    private static let silenceThresholdDB: Float = -60.0

    static func correct(original: [Float], shifted: [Float],
                        sampleRate: Int, channels: Int) -> [Float] {
        let origFrameCount = original.count / channels
        let shiftFrameCount = shifted.count / channels

        guard origFrameCount >= frameSize, shiftFrameCount >= frameSize else {
            return shifted
        }

        if channels == 1 {
            return correctChannel(original: original, shifted: shifted,
                                  origFrames: origFrameCount, shiftFrames: shiftFrameCount)
        }

        var origDeinterleaved = [[Float]](repeating: [Float](repeating: 0, count: origFrameCount), count: channels)
        var shiftDeinterleaved = [[Float]](repeating: [Float](repeating: 0, count: shiftFrameCount), count: channels)

        for frame in 0..<origFrameCount {
            for ch in 0..<channels {
                origDeinterleaved[ch][frame] = original[frame * channels + ch]
            }
        }
        for frame in 0..<shiftFrameCount {
            for ch in 0..<channels {
                shiftDeinterleaved[ch][frame] = shifted[frame * channels + ch]
            }
        }

        var correctedChannels = [[Float]]()
        for ch in 0..<channels {
            correctedChannels.append(
                correctChannel(original: origDeinterleaved[ch], shifted: shiftDeinterleaved[ch],
                               origFrames: origFrameCount, shiftFrames: shiftFrameCount)
            )
        }

        let outputFrames = correctedChannels[0].count
        var interleaved = [Float](repeating: 0, count: outputFrames * channels)
        for frame in 0..<outputFrames {
            for ch in 0..<channels {
                interleaved[frame * channels + ch] = correctedChannels[ch][frame]
            }
        }
        return interleaved
    }

    private static func correctChannel(original: [Float], shifted: [Float],
                                        origFrames: Int, shiftFrames: Int) -> [Float] {
        var output = [Float](repeating: 0, count: shiftFrames)
        var envelope = [Float](repeating: 0, count: shiftFrames)

        var window = [Float](repeating: 0, count: frameSize)
        vDSP_hann_window(&window, vDSP_Length(frameSize), Int32(vDSP_HANN_NORM))

        let numFrames = min((origFrames - frameSize) / hopSize, (shiftFrames - frameSize) / hopSize) + 1

        for frameIdx in 0..<numFrames {
            let origStart = frameIdx * hopSize
            let shiftStart = frameIdx * hopSize

            let origFrame = Array(original[origStart..<origStart + frameSize])
            let shiftFrame = Array(shifted[shiftStart..<shiftStart + frameSize])

            if isFrameSilent(shiftFrame) {
                for i in 0..<frameSize where shiftStart + i < shiftFrames {
                    output[shiftStart + i] += shiftFrame[i] * window[i]
                    envelope[shiftStart + i] += window[i]
                }
                continue
            }

            var windowedOrig = [Float](repeating: 0, count: frameSize)
            vDSP_vmul(origFrame, 1, window, 1, &windowedOrig, 1, vDSP_Length(frameSize))

            var windowedShift = [Float](repeating: 0, count: frameSize)
            vDSP_vmul(shiftFrame, 1, window, 1, &windowedShift, 1, vDSP_Length(frameSize))

            let origCoeffs = levinsonDurbin(autocorrelation(windowedOrig))
            let shiftCoeffs = levinsonDurbin(autocorrelation(windowedShift))

            let residual = inverseFilter(windowedShift, coefficients: shiftCoeffs)
            let corrected = synthesisFilter(residual, coefficients: origCoeffs)

            for i in 0..<frameSize where shiftStart + i < shiftFrames {
                output[shiftStart + i] += corrected[i] * window[i]
                envelope[shiftStart + i] += window[i]
            }
        }

        for i in 0..<shiftFrames {
            if envelope[i] > 0.001 {
                output[i] /= envelope[i]
            } else {
                output[i] = shifted[i]
            }
        }

        return output
    }

    private static func autocorrelation(_ frame: [Float]) -> [Float] {
        let n = frame.count
        let order = lpcOrder + 1
        var result = [Float](repeating: 0, count: order)

        for lag in 0..<order {
            var sum: Float = 0
            vDSP_dotpr(Array(frame[0..<n - lag]), 1,
                       Array(frame[lag..<n]), 1,
                       &sum, vDSP_Length(n - lag))
            result[lag] = sum
        }
        return result
    }

    private static func levinsonDurbin(_ r: [Float]) -> [Float] {
        let order = r.count - 1
        guard order > 0, r[0] > 0 else {
            return [Float](repeating: 0, count: order)
        }

        var a = [Float](repeating: 0, count: order)
        var aPrev = [Float](repeating: 0, count: order)
        var error = r[0]

        for i in 0..<order {
            var sum: Float = r[i + 1]
            for j in 0..<i {
                sum += aPrev[j] * r[i - j]
            }

            var k = -sum / error
            k = min(max(k, -0.99), 0.99)

            a[i] = k
            for j in 0..<i {
                a[j] = aPrev[j] + k * aPrev[i - 1 - j]
            }

            error *= (1.0 - k * k)
            if error < 1e-10 { break }

            aPrev = a
        }
        return a
    }

    private static func inverseFilter(_ signal: [Float], coefficients: [Float]) -> [Float] {
        let n = signal.count
        let order = coefficients.count
        var residual = [Float](repeating: 0, count: n)

        for i in 0..<n {
            var prediction: Float = 0
            for j in 0..<min(order, i) {
                prediction += coefficients[j] * signal[i - 1 - j]
            }
            residual[i] = signal[i] - prediction
        }
        return residual
    }

    private static func synthesisFilter(_ residual: [Float], coefficients: [Float]) -> [Float] {
        let n = residual.count
        let order = coefficients.count
        var output = [Float](repeating: 0, count: n)

        for i in 0..<n {
            var prediction: Float = 0
            for j in 0..<min(order, i) {
                prediction += coefficients[j] * output[i - 1 - j]
            }
            output[i] = residual[i] + prediction
        }
        return output
    }

    private static func isFrameSilent(_ frame: [Float]) -> Bool {
        var rms: Float = 0
        vDSP_rmsqv(frame, 1, &rms, vDSP_Length(frame.count))
        let db = 20.0 * log10(max(rms, 1e-10))
        return db < silenceThresholdDB
    }
}
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/LPCFormantShifter.swift
git commit -m "feat(dsp): add LPC formant correction processor"
```

---

### Task 4: DSPProcessor Update

**Files:**

- Modify: `AnvilApp/DSP/DSPProcessor.swift:55-66` (replace `.timeStretch` and `.pitchShift` dispatch)

- [ ] **Step 1: Add processRegion helper and update dispatch in DSPProcessor.swift**

Replace the `.timeStretch` and `.pitchShift` cases (lines 55-66) with:

```swift
        case .timeStretch:
            guard case .timeStretch(let params) = parameters else {
                return buffer
            }
            return try processRegion(buffer, target: target) { regionBuffer in
                let floats = AudioBufferConverter.toFloats(regionBuffer)
                let stretched = WSOLAProcessor.stretch(floats, rate: params.rate,
                                                       sampleRate: regionBuffer.sampleRate,
                                                       channels: regionBuffer.channels)
                return AudioBufferConverter.fromFloats(stretched,
                                                       sampleRate: regionBuffer.sampleRate,
                                                       channels: regionBuffer.channels)
            }

        case .pitchShift:
            guard case .pitchShift(let params) = parameters else {
                return buffer
            }
            return try processRegion(buffer, target: target) { regionBuffer in
                let floats = AudioBufferConverter.toFloats(regionBuffer)
                let pitchRatio = pow(2.0, params.pitch / 1200.0)
                let resampled = Resampler.resample(floats, ratio: pitchRatio,
                                                    channels: regionBuffer.channels)
                let timeCorrected = WSOLAProcessor.stretch(resampled,
                                                            rate: 1.0 / pitchRatio,
                                                            sampleRate: regionBuffer.sampleRate,
                                                            channels: regionBuffer.channels)
                let corrected = LPCFormantShifter.correct(original: floats,
                                                          shifted: timeCorrected,
                                                          sampleRate: regionBuffer.sampleRate,
                                                          channels: regionBuffer.channels)
                return AudioBufferConverter.fromFloats(corrected,
                                                       sampleRate: regionBuffer.sampleRate,
                                                       channels: regionBuffer.channels)
            }
```

Add this private method before the existing `formatTime` helper:

```swift
    private func processRegion(
        _ buffer: AudioBuffer,
        target: ApplyTarget,
        transform: (AudioBuffer) throws -> AudioBuffer
    ) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try transform(buffer)
        case .selection(let start, let end):
            let bytesPerSample = MemoryLayout<Int16>.size
            let startSample = SimpleDSP.sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
            let endSample = min(
                SimpleDSP.sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels),
                buffer.data.count / bytesPerSample
            )
            let startByte = startSample * bytesPerSample
            let endByte = endSample * bytesPerSample

            let regionData = Data(buffer.data[startByte..<endByte])
            let regionBuffer = AudioBuffer(data: regionData, sampleRate: buffer.sampleRate, channels: buffer.channels)
            let processed = try transform(regionBuffer)

            var result = Data()
            result.append(buffer.data[0..<startByte])
            result.append(processed.data)
            if endByte < buffer.data.count {
                result.append(buffer.data[endByte...])
            }
            return AudioBuffer(data: result, sampleRate: buffer.sampleRate, channels: buffer.channels)
        }
    }
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/DSPProcessor.swift
git commit -m "feat(dsp): route time stretch and pitch shift to custom processors"
```

---

### Task 5: AVAudioEngineDSP Cleanup

**Files:**

- Modify: `AnvilApp/DSP/AVAudioEngineDSP.swift:183-326` (remove time stretch and pitch shift methods)

- [ ] **Step 1: Remove the four time stretch / pitch shift methods**

Delete these four methods from `AVAudioEngineDSP.swift`:

- `applyTimeStretch(to:parameters:target:)` (lines 185-194)
- `applyTimeStretchToBuffer(_:parameters:)` (lines 196-254)
- `applyPitchShift(to:parameters:target:)` (lines 258-267)
- `applyPitchShiftToBuffer(_:parameters:)` (lines 269-326)

Also remove the `// MARK: - Time Stretch` and `// MARK: - Pitch Shift` section markers for those methods.

The file should end after `applyReverbToBuffer` (line 181) with just the `// MARK: - Region Splicing` section remaining (the `applyToRegion` helper is still used by EQ and Reverb).

After removal, the file structure should be:

```text
AVAudioEngineDSP.swift:
├── EngineError enum
├── // MARK: - EQ
│   ├── applyEQ(to:parameters:target:)
│   └── applyEQToBuffer(_:parameters:)
├── // MARK: - Reverb
│   ├── applyReverb(to:parameters:target:)
│   └── applyReverbToBuffer(_:parameters:)
└── // MARK: - Region Splicing
    └── applyToRegion(_:start:end:transform:)
```

- [ ] **Step 2: Verify Xcode build**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/AVAudioEngineDSP.swift
git commit -m "refactor(dsp): remove AVAudioUnitTimePitch time stretch and pitch shift methods"
```

---

## Verification

After all 5 tasks complete, verify the full build one final time:

```bash
xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp -destination 'platform=macOS' build 2>&1 | tail -5
```

Expected: `** BUILD SUCCEEDED **` with no warnings related to DSP files.

The backend swap is invisible to the user — no UI changes, same slider behavior, same parameter ranges. The only observable difference is improved audio quality on pitch shifts (formant preservation eliminates chipmunk effect).
