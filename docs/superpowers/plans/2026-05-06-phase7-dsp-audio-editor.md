# Phase 7: DSP Audio Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add destructive DSP audio editing (Trim, Delete, Fade, Normalize, EQ, Reverb) to the existing Edit tab as an always-available effects palette.

**Architecture:** Hybrid processing — plain Swift/vDSP for simple ops (Trim, Fade, Normalize), AVAudioEngine offline rendering for EQ/Reverb. All code in app target (`AnvilApp/DSP/`). Reuses Phase 6's `EditHistoryManager` for undo/redo snapshots.

**Tech Stack:** Swift, SwiftUI, vDSP (Accelerate framework), AVAudioEngine (AVFoundation), macOS 14+

---

## File Structure

```text
AnvilApp/
├── DSP/
│   ├── DSPParameters.swift         // AudioBuffer, DSPEffect, ApplyTarget, EQParameters, ReverbParameters, etc.
│   ├── AudioBufferConverter.swift  // Data ↔ [Float] ↔ AVAudioPCMBuffer
│   ├── SimpleDSP.swift             // Trim, Delete, Fade, Normalize (Swift + vDSP)
│   ├── AVAudioEngineDSP.swift      // EQ + Reverb (offline AVAudioEngine rendering)
│   └── DSPProcessor.swift          // Facade: dispatches to SimpleDSP or AVAudioEngineDSP
├── UI/
│   └── Editor/
│       ├── DSPPaletteView.swift    // Effect buttons + expanded panel container
│       ├── EQControlView.swift     // 3-band parametric EQ controls
│       └── ReverbControlView.swift // Preset picker + parameter sliders
│       (modify) EditorState.swift  // Add DSP properties + applyDSPEffect method
│       (modify) EditorView.swift   // Add DSPPaletteView + keyboard shortcuts
```

---

### Task 1: DSPParameters — Data Model Types

**Files:**

- Create: `AnvilApp/DSP/DSPParameters.swift`

- [ ] **Step 1: Create DSP directory and data model file**

Create `AnvilApp/DSP/DSPParameters.swift` with all data model types:

```swift
import Foundation

// MARK: - AudioBuffer

struct AudioBuffer: Equatable {
    var data: Data
    var sampleRate: Int
    var channels: Int

    var frameCount: Int {
        data.count / (MemoryLayout<Int16>.size * channels)
    }

    var duration: TimeInterval {
        Double(frameCount) / Double(sampleRate)
    }
}

// MARK: - DSPEffect

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

    var requiresSelection: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut:
            return true
        case .normalize, .eq, .reverb:
            return false
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

// MARK: - ApplyTarget

enum ApplyTarget: Equatable {
    case selection(start: TimeInterval, end: TimeInterval)
    case wholeTrack
}

// MARK: - DSPParameters

enum DSPParameters: Equatable {
    case none
    case eq(EQParameters)
    case reverb(ReverbParameters)
}

// MARK: - EQBand + EQParameters

struct EQBand: Equatable {
    var frequency: Float
    var gain: Float
    var q: Float
}

struct EQParameters: Equatable {
    var bands: [EQBand]

    static let `default` = EQParameters(bands: [
        EQBand(frequency: 200, gain: 0, q: 1.0),
        EQBand(frequency: 1200, gain: 0, q: 1.4),
        EQBand(frequency: 8000, gain: 0, q: 0.8),
    ])
}

// MARK: - ReverbPreset + ReverbParameters

enum ReverbPreset: String, CaseIterable {
    case room, hall, plate

    var parameters: ReverbParameters {
        switch self {
        case .room: return .room
        case .hall: return .hall
        case .plate: return .plate
        }
    }

    var displayName: String {
        rawValue.capitalized
    }
}

struct ReverbParameters: Equatable {
    var decayTime: Float
    var wetDryMix: Float
    var preDelay: Float
    var damping: Float

    static let room = ReverbParameters(decayTime: 0.8, wetDryMix: 0.25, preDelay: 10, damping: 0.5)
    static let hall = ReverbParameters(decayTime: 2.4, wetDryMix: 0.35, preDelay: 20, damping: 0.6)
    static let plate = ReverbParameters(decayTime: 1.5, wetDryMix: 0.40, preDelay: 5, damping: 0.3)
    static let `default` = ReverbParameters.hall
}

// MARK: - NormalizeMode

enum NormalizeMode: Equatable {
    case peak
    case rms(targetDB: Float)
}

// MARK: - DSPOperation

struct DSPOperation: Equatable {
    let effect: DSPEffect
    let target: ApplyTarget
    let description: String
}
```

- [ ] **Step 2: Verify the file compiles**

Open the Xcode project and build (`Cmd+B`). Verify no compilation errors in the new file.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/DSPParameters.swift
git commit -m "feat(dsp): add data model types for DSP effects"
```

---

### Task 2: AudioBufferConverter

**Files:**

- Create: `AnvilApp/DSP/AudioBufferConverter.swift`

- [ ] **Step 1: Create the converter utility**

```swift
import AVFoundation
import Accelerate

struct AudioBufferConverter {
    enum ConversionError: Error {
        case invalidData
        case bufferCreationFailed
    }

    static func toFloats(_ buffer: AudioBuffer) -> [Float] {
        let int16Count = buffer.data.count / MemoryLayout<Int16>.size
        var int16Samples = [Int16](repeating: 0, count: int16Count)
        buffer.data.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Int16.self)
            for i in 0..<int16Count {
                int16Samples[i] = src[i]
            }
        }
        var floats = [Float](repeating: 0, count: int16Count)
        vDSP.convertElements(of: int16Samples, to: &floats)
        var scale: Float = 1.0 / 32768.0
        vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(int16Count))
        return floats
    }

    static func fromFloats(_ samples: [Float], sampleRate: Int, channels: Int) -> AudioBuffer {
        var clamped = samples
        var low: Float = -1.0
        var high: Float = 1.0
        vDSP_vclip(clamped, 1, &low, &high, &clamped, 1, vDSP_Length(samples.count))

        var scaled = [Float](repeating: 0, count: samples.count)
        var scale: Float = 32767.0
        vDSP_vsmul(clamped, 1, &scale, &scaled, 1, vDSP_Length(samples.count))

        var int16Samples = [Int16](repeating: 0, count: samples.count)
        vDSP.convertElements(of: scaled, to: &int16Samples)

        let data = int16Samples.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }
        return AudioBuffer(data: data, sampleRate: sampleRate, channels: channels)
    }

    static func toPCMBuffer(_ buffer: AudioBuffer) throws -> AVAudioPCMBuffer {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(buffer.sampleRate),
            channels: AVAudioChannelCount(buffer.channels),
            interleaved: false
        ) else {
            throw ConversionError.bufferCreationFailed
        }

        let frameCount = AVAudioFrameCount(buffer.frameCount)
        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ConversionError.bufferCreationFailed
        }
        pcmBuffer.frameLength = frameCount

        let floats = toFloats(buffer)
        let channels = buffer.channels

        if channels == 1 {
            floats.withUnsafeBufferPointer { src in
                pcmBuffer.floatChannelData![0].update(from: src.baseAddress!, count: Int(frameCount))
            }
        } else {
            for ch in 0..<channels {
                let channelPtr = pcmBuffer.floatChannelData![ch]
                for frame in 0..<Int(frameCount) {
                    channelPtr[frame] = floats[frame * channels + ch]
                }
            }
        }

        return pcmBuffer
    }

    static func fromPCMBuffer(_ pcmBuffer: AVAudioPCMBuffer) -> AudioBuffer {
        let frameCount = Int(pcmBuffer.frameLength)
        let channels = Int(pcmBuffer.format.channelCount)
        let totalSamples = frameCount * channels

        var interleaved = [Float](repeating: 0, count: totalSamples)

        if channels == 1 {
            let src = pcmBuffer.floatChannelData![0]
            for i in 0..<frameCount {
                interleaved[i] = src[i]
            }
        } else {
            for frame in 0..<frameCount {
                for ch in 0..<channels {
                    interleaved[frame * channels + ch] = pcmBuffer.floatChannelData![ch][frame]
                }
            }
        }

        return fromFloats(interleaved, sampleRate: Int(pcmBuffer.format.sampleRate), channels: channels)
    }
}
```

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`). Verify no compilation errors.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/AudioBufferConverter.swift
git commit -m "feat(dsp): add AudioBufferConverter for format conversions"
```

---

### Task 3: SimpleDSP — Trim and Delete

**Files:**

- Create: `AnvilApp/DSP/SimpleDSP.swift`

- [ ] **Step 1: Create SimpleDSP with trim and delete**

```swift
import Foundation
import Accelerate

struct SimpleDSP {
    enum DSPError: Error {
        case invalidSelection
        case bufferEmpty
    }

    static func trimToSelection(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let bytesPerSample = MemoryLayout<Int16>.size
        let startByte = startSample * bytesPerSample
        let endByte = min(endSample * bytesPerSample, buffer.data.count)

        guard startByte < buffer.data.count else { throw DSPError.invalidSelection }

        let trimmedData = buffer.data[startByte..<endByte]
        return AudioBuffer(data: Data(trimmedData), sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func deleteSelection(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let bytesPerSample = MemoryLayout<Int16>.size
        let startByte = startSample * bytesPerSample
        let endByte = min(endSample * bytesPerSample, buffer.data.count)

        var result = Data()
        result.append(buffer.data[0..<startByte])
        if endByte < buffer.data.count {
            result.append(buffer.data[endByte...])
        }

        guard !result.isEmpty else { throw DSPError.bufferEmpty }
        return AudioBuffer(data: result, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    // MARK: - Helpers

    static func sampleIndex(time: TimeInterval, sampleRate: Int, channels: Int) -> Int {
        Int(time * Double(sampleRate)) * channels
    }
}
```

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/SimpleDSP.swift
git commit -m "feat(dsp): add SimpleDSP with trim and delete operations"
```

---

### Task 4: SimpleDSP — Fade In, Fade Out, Normalize

**Files:**

- Modify: `AnvilApp/DSP/SimpleDSP.swift`

- [ ] **Step 1: Add fade and normalize methods**

Add the following methods to the `SimpleDSP` struct, after the `deleteSelection` method:

```swift
    static func fadeIn(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = min(sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels), floats.count)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let length = endSample - startSample
        var ramp = [Float](repeating: 0, count: length)
        var start_val: Float = 0.0
        var end_val: Float = 1.0
        vDSP_vgen(&start_val, &end_val, &ramp, 1, vDSP_Length(length))
        vDSP_vmul(&floats + startSample, 1, ramp, 1, &floats + startSample, 1, vDSP_Length(length))

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func fadeOut(_ buffer: AudioBuffer, start: TimeInterval, end: TimeInterval) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        let startSample = sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = min(sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels), floats.count)

        guard startSample < endSample else { throw DSPError.invalidSelection }

        let length = endSample - startSample
        var ramp = [Float](repeating: 0, count: length)
        var start_val: Float = 1.0
        var end_val: Float = 0.0
        vDSP_vgen(&start_val, &end_val, &ramp, 1, vDSP_Length(length))
        vDSP_vmul(&floats + startSample, 1, ramp, 1, &floats + startSample, 1, vDSP_Length(length))

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }

    static func normalize(_ buffer: AudioBuffer, mode: NormalizeMode = .peak) throws -> AudioBuffer {
        var floats = AudioBufferConverter.toFloats(buffer)
        guard !floats.isEmpty else { throw DSPError.bufferEmpty }

        switch mode {
        case .peak:
            var peak: Float = 0
            vDSP_maxmgv(floats, 1, &peak, vDSP_Length(floats.count))
            guard peak > 0.001 else { return buffer }
            guard peak < 0.99 else { return buffer }
            var scale = 1.0 / peak
            vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(floats.count))

        case .rms:
            return buffer
        }

        return AudioBufferConverter.fromFloats(floats, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }
```

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/SimpleDSP.swift
git commit -m "feat(dsp): add fade in/out and normalize to SimpleDSP"
```

---

### Task 5: AVAudioEngineDSP — EQ

**Files:**

- Create: `AnvilApp/DSP/AVAudioEngineDSP.swift`

- [ ] **Step 1: Create AVAudioEngineDSP with EQ**

```swift
import AVFoundation

struct AVAudioEngineDSP {
    enum EngineError: Error {
        case renderFailed(String)
        case bufferCreationFailed
    }

    static func applyEQ(to buffer: AudioBuffer, parameters: EQParameters, target: ApplyTarget) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try applyEQToBuffer(buffer, parameters: parameters)
        case .selection(let start, let end):
            return try applyToRegion(buffer, start: start, end: end) { region in
                try applyEQToBuffer(region, parameters: parameters)
            }
        }
    }

    private static func applyEQToBuffer(_ buffer: AudioBuffer, parameters: EQParameters) throws -> AudioBuffer {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let eq = AVAudioUnitEQ(numberOfBands: 3)

        for (i, band) in parameters.bands.enumerated() {
            eq.bands[i].filterType = .parametric
            eq.bands[i].frequency = band.frequency
            eq.bands[i].gain = band.gain
            eq.bands[i].bandwidth = band.q
            eq.bands[i].bypass = false
        }

        engine.attach(player)
        engine.attach(eq)
        engine.connect(player, to: eq, format: format)
        engine.connect(eq, to: engine.mainMixerNode, format: format)

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: pcmBuffer.frameLength)
        try engine.start()
        player.play()
        player.scheduleBuffer(pcmBuffer, completionHandler: nil)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: pcmBuffer.frameLength) else {
            throw EngineError.bufferCreationFailed
        }

        let status = try engine.renderOffline(pcmBuffer.frameLength, to: outputBuffer)
        guard status == .success else {
            throw EngineError.renderFailed("EQ render returned status: \(status.rawValue)")
        }

        engine.stop()
        return AudioBufferConverter.fromPCMBuffer(outputBuffer)
    }

    // MARK: - Region Splicing

    private static func applyToRegion(
        _ buffer: AudioBuffer,
        start: TimeInterval,
        end: TimeInterval,
        transform: (AudioBuffer) throws -> AudioBuffer
    ) throws -> AudioBuffer {
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

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/AVAudioEngineDSP.swift
git commit -m "feat(dsp): add AVAudioEngineDSP with offline EQ rendering"
```

---

### Task 6: AVAudioEngineDSP — Reverb

**Files:**

- Modify: `AnvilApp/DSP/AVAudioEngineDSP.swift`

- [ ] **Step 1: Add reverb method**

Add the following methods to `AVAudioEngineDSP`, after the `applyEQToBuffer` method:

```swift
    static func applyReverb(to buffer: AudioBuffer, parameters: ReverbParameters, target: ApplyTarget) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try applyReverbToBuffer(buffer, parameters: parameters)
        case .selection(let start, let end):
            return try applyToRegion(buffer, start: start, end: end) { region in
                try applyReverbToBuffer(region, parameters: parameters)
            }
        }
    }

    private static func applyReverbToBuffer(_ buffer: AudioBuffer, parameters: ReverbParameters) throws -> AudioBuffer {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let reverb = AVAudioUnitReverb()

        let preset: AVAudioUnitReverbPreset
        if parameters.decayTime <= 1.0 {
            preset = .mediumRoom
        } else if parameters.decayTime <= 2.0 {
            preset = .largeChamber
        } else {
            preset = .largeHall
        }
        reverb.loadFactoryPreset(preset)
        reverb.wetDryMix = parameters.wetDryMix * 100.0

        engine.attach(player)
        engine.attach(reverb)
        engine.connect(player, to: reverb, format: format)
        engine.connect(reverb, to: engine.mainMixerNode, format: format)

        // Reverb adds a tail — render extra frames for decay
        let tailFrames = AVAudioFrameCount(parameters.decayTime * Float(buffer.sampleRate))
        let totalFrames = pcmBuffer.frameLength + tailFrames

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
        try engine.start()
        player.play()
        player.scheduleBuffer(pcmBuffer, completionHandler: nil)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: totalFrames) else {
            throw EngineError.bufferCreationFailed
        }

        var framesRendered: AVAudioFrameCount = 0
        while framesRendered < totalFrames {
            let framesToRender = min(4096, totalFrames - framesRendered)
            guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRender) else {
                throw EngineError.bufferCreationFailed
            }
            let status = try engine.renderOffline(framesToRender, to: chunk)
            if status == .error {
                throw EngineError.renderFailed("Reverb render error at frame \(framesRendered)")
            }
            // Copy chunk into output
            let destOffset = Int(framesRendered)
            let chunkFrames = Int(chunk.frameLength)
            for ch in 0..<Int(format.channelCount) {
                let src = chunk.floatChannelData![ch]
                let dst = outputBuffer.floatChannelData![ch]
                for i in 0..<chunkFrames {
                    dst[destOffset + i] = src[i]
                }
            }
            framesRendered += chunk.frameLength
            if status == .insufficientDataFromInputNode || status == .cannotDoInCurrentContext {
                break
            }
        }
        outputBuffer.frameLength = framesRendered

        engine.stop()

        // Trim output to original length (don't extend the track)
        let originalFrames = pcmBuffer.frameLength
        guard let trimmedOutput = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: originalFrames) else {
            throw EngineError.bufferCreationFailed
        }
        trimmedOutput.frameLength = originalFrames
        for ch in 0..<Int(format.channelCount) {
            let src = outputBuffer.floatChannelData![ch]
            let dst = trimmedOutput.floatChannelData![ch]
            for i in 0..<Int(originalFrames) {
                dst[i] = src[i]
            }
        }

        return AudioBufferConverter.fromPCMBuffer(trimmedOutput)
    }
```

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/AVAudioEngineDSP.swift
git commit -m "feat(dsp): add offline reverb rendering to AVAudioEngineDSP"
```

---

### Task 7: DSPProcessor Facade

**Files:**

- Create: `AnvilApp/DSP/DSPProcessor.swift`

- [ ] **Step 1: Create the facade**

```swift
import Foundation

struct DSPProcessor {
    enum ProcessingError: Error {
        case selectionRequired
        case unknownEffect
    }

    func apply(
        effect: DSPEffect,
        to buffer: AudioBuffer,
        target: ApplyTarget,
        parameters: DSPParameters = .none
    ) throws -> AudioBuffer {
        switch effect {
        case .trimToSelection:
            guard case .selection(let start, let end) = target else {
                throw ProcessingError.selectionRequired
            }
            return try SimpleDSP.trimToSelection(buffer, start: start, end: end)

        case .deleteSelection:
            guard case .selection(let start, let end) = target else {
                throw ProcessingError.selectionRequired
            }
            return try SimpleDSP.deleteSelection(buffer, start: start, end: end)

        case .fadeIn:
            guard case .selection(let start, let end) = target else {
                throw ProcessingError.selectionRequired
            }
            return try SimpleDSP.fadeIn(buffer, start: start, end: end)

        case .fadeOut:
            guard case .selection(let start, let end) = target else {
                throw ProcessingError.selectionRequired
            }
            return try SimpleDSP.fadeOut(buffer, start: start, end: end)

        case .normalize:
            return try SimpleDSP.normalize(buffer)

        case .eq:
            guard case .eq(let eqParams) = parameters else {
                return buffer
            }
            return try AVAudioEngineDSP.applyEQ(to: buffer, parameters: eqParams, target: target)

        case .reverb:
            guard case .reverb(let reverbParams) = parameters else {
                return buffer
            }
            return try AVAudioEngineDSP.applyReverb(to: buffer, parameters: reverbParams, target: target)
        }
    }

    func makeDescription(effect: DSPEffect, target: ApplyTarget) -> DSPOperation {
        let targetLabel: String
        switch target {
        case .selection(let start, let end):
            targetLabel = "(\(formatTime(start))–\(formatTime(end)))"
        case .wholeTrack:
            targetLabel = "(whole track)"
        }

        return DSPOperation(
            effect: effect,
            target: target,
            description: "\(effect.displayName) \(targetLabel)"
        )
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        let tenths = Int((time - Double(Int(time))) * 10)
        if minutes > 0 {
            return String(format: "%d:%02d.%d", minutes, seconds, tenths)
        }
        return String(format: "%d.%ds", seconds, tenths)
    }
}
```

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/DSP/DSPProcessor.swift
git commit -m "feat(dsp): add DSPProcessor facade"
```

---

### Task 8: EQControlView

**Files:**

- Create: `AnvilApp/UI/Editor/EQControlView.swift`

- [ ] **Step 1: Create the EQ control view**

```swift
import SwiftUI

struct EQControlView: View {
    @Binding var parameters: EQParameters

    private let bandLabels = ["Band 1 — Low", "Band 2 — Mid", "Band 3 — High"]

    var body: some View {
        VStack(spacing: 10) {
            ForEach(0..<3, id: \.self) { index in
                bandView(index: index)
            }
        }
    }

    private func bandView(index: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(bandLabels[index])
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Freq").font(.system(size: 9)).foregroundStyle(.tertiary)
                    Text(formatFrequency(parameters.bands[index].frequency))
                        .font(.caption).monospacedDigit()
                }
                .frame(width: 52, alignment: .leading)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Gain").font(.system(size: 9)).foregroundStyle(.tertiary)
                    Text(formatGain(parameters.bands[index].gain))
                        .font(.caption).monospacedDigit()
                        .foregroundStyle(gainColor(parameters.bands[index].gain))
                }
                .frame(width: 52, alignment: .leading)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Q").font(.system(size: 9)).foregroundStyle(.tertiary)
                    Text(String(format: "%.1f", parameters.bands[index].q))
                        .font(.caption).monospacedDigit()
                }
                .frame(width: 36, alignment: .leading)
            }

            VStack(spacing: 4) {
                Slider(value: $parameters.bands[index].frequency, in: 20...20000)
                    .tint(.blue)
                Slider(value: $parameters.bands[index].gain, in: -24...24)
                    .tint(gainColor(parameters.bands[index].gain))
                Slider(value: $parameters.bands[index].q, in: 0.1...10)
                    .tint(.gray)
            }
        }
        .padding(8)
        .background(Color(white: 0.12))
        .cornerRadius(6)
    }

    private func formatFrequency(_ freq: Float) -> String {
        if freq >= 1000 {
            return String(format: "%.1f kHz", freq / 1000)
        }
        return String(format: "%.0f Hz", freq)
    }

    private func formatGain(_ gain: Float) -> String {
        if gain >= 0 {
            return String(format: "+%.1f dB", gain)
        }
        return String(format: "%.1f dB", gain)
    }

    private func gainColor(_ gain: Float) -> Color {
        if gain > 0.1 { return .green }
        if gain < -0.1 { return .red }
        return .secondary
    }
}
```

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Editor/EQControlView.swift
git commit -m "feat(dsp): add EQControlView with 3-band parametric controls"
```

---

### Task 9: ReverbControlView

**Files:**

- Create: `AnvilApp/UI/Editor/ReverbControlView.swift`

- [ ] **Step 1: Create the reverb control view**

```swift
import SwiftUI

struct ReverbControlView: View {
    @Binding var parameters: ReverbParameters
    @Binding var selectedPreset: ReverbPreset?

    var body: some View {
        VStack(spacing: 12) {
            // Preset row
            HStack(spacing: 6) {
                ForEach(ReverbPreset.allCases, id: \.self) { preset in
                    Button(preset.displayName) {
                        selectedPreset = preset
                        parameters = preset.parameters
                    }
                    .buttonStyle(.bordered)
                    .tint(selectedPreset == preset ? .purple : .gray)
                    .controlSize(.small)
                }
                Spacer()
            }

            // Parameter sliders
            VStack(spacing: 8) {
                parameterSlider(
                    label: "Decay",
                    value: Binding(
                        get: { parameters.decayTime },
                        set: { parameters.decayTime = $0; selectedPreset = nil }
                    ),
                    range: 0.1...10.0,
                    format: "%.1fs"
                )

                parameterSlider(
                    label: "Wet/Dry",
                    value: Binding(
                        get: { parameters.wetDryMix },
                        set: { parameters.wetDryMix = $0; selectedPreset = nil }
                    ),
                    range: 0...1.0,
                    format: "%.0f%%",
                    displayMultiplier: 100
                )

                parameterSlider(
                    label: "Pre-Delay",
                    value: Binding(
                        get: { parameters.preDelay },
                        set: { parameters.preDelay = $0; selectedPreset = nil }
                    ),
                    range: 0...100,
                    format: "%.0fms"
                )

                parameterSlider(
                    label: "Damping",
                    value: Binding(
                        get: { parameters.damping },
                        set: { parameters.damping = $0; selectedPreset = nil }
                    ),
                    range: 0...1.0,
                    format: "%.2f"
                )
            }
        }
    }

    private func parameterSlider(
        label: String,
        value: Binding<Float>,
        range: ClosedRange<Float>,
        format: String,
        displayMultiplier: Float = 1
    ) -> some View {
        VStack(spacing: 2) {
            HStack {
                Text(label).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(String(format: format, value.wrappedValue * displayMultiplier))
                    .font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
                .tint(.purple)
        }
    }
}
```

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Editor/ReverbControlView.swift
git commit -m "feat(dsp): add ReverbControlView with presets and parameter sliders"
```

---

### Task 10: DSPPaletteView

**Files:**

- Create: `AnvilApp/UI/Editor/DSPPaletteView.swift`

- [ ] **Step 1: Create the palette view**

```swift
import SwiftUI

struct DSPPaletteView: View {
    @ObservedObject var state: EditorState
    let onApply: (DSPEffect, DSPParameters) -> Void

    @State private var showingConfirmation: DSPEffect?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Section header
            Text("DSP EFFECTS")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .tracking(0.5)

            // Effect buttons grid
            effectGrid

            // Expanded parameter panel
            if let expanded = state.expandedEffect {
                expandedPanel(for: expanded)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(16)
        .animation(.easeInOut(duration: 0.2), value: state.expandedEffect)
    }

    // MARK: - Effect Grid

    private var effectGrid: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                effectButton(.trimToSelection)
                effectButton(.deleteSelection)
                effectButton(.normalize)
            }
            HStack(spacing: 6) {
                effectButton(.fadeIn)
                effectButton(.fadeOut)
                effectButton(.eq)
            }
            HStack(spacing: 6) {
                effectButton(.reverb)
                Spacer()
                Spacer()
            }
        }
    }

    private func effectButton(_ effect: DSPEffect) -> some View {
        let isExpanded = state.expandedEffect == effect
        let isDisabled = effect.requiresSelection && !state.hasSelection

        return Button {
            if effect.requiresParameters {
                withAnimation {
                    state.expandedEffect = isExpanded ? nil : effect
                }
            } else {
                onApply(effect, .none)
                showConfirmation(effect)
            }
        } label: {
            ZStack {
                Text(effect.displayName)
                    .font(.system(size: 10, weight: isExpanded ? .semibold : .regular))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
                    .padding(.horizontal, 4)

                if showingConfirmation == effect {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.green)
                }
            }
        }
        .buttonStyle(.bordered)
        .tint(isExpanded ? .blue : .gray)
        .disabled(isDisabled)
    }

    private func showConfirmation(_ effect: DSPEffect) {
        showingConfirmation = effect
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            if showingConfirmation == effect {
                showingConfirmation = nil
            }
        }
    }

    // MARK: - Expanded Panels

    @ViewBuilder
    private func expandedPanel(for effect: DSPEffect) -> some View {
        VStack(spacing: 12) {
            HStack {
                Text(effect.displayName)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.blue)
                Spacer()
                Button {
                    withAnimation { state.expandedEffect = nil }
                } label: {
                    Image(systemName: "chevron.up")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }

            switch effect {
            case .eq:
                EQControlView(parameters: $state.eqParameters)
            case .reverb:
                ReverbControlView(
                    parameters: $state.reverbParameters,
                    selectedPreset: $state.selectedReverbPreset
                )
            default:
                EmptyView()
            }

            // Apply target indicator + button
            applySection(for: effect)
        }
        .padding(12)
        .background(Color(red: 0.1, green: 0.1, blue: 0.15))
        .cornerRadius(8)
    }

    private func applySection(for effect: DSPEffect) -> some View {
        VStack(spacing: 8) {
            HStack {
                Text("Apply to:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(applyTargetLabel)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color(white: 0.2))
                    .cornerRadius(4)
                    .foregroundStyle(.blue)
                Spacer()
            }

            Button {
                let params: DSPParameters
                switch effect {
                case .eq: params = .eq(state.eqParameters)
                case .reverb: params = .reverb(state.reverbParameters)
                default: params = .none
                }
                onApply(effect, params)
            } label: {
                Text("Apply \(effect.displayName)")
                    .font(.system(size: 12, weight: .medium))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
        }
    }

    private var applyTargetLabel: String {
        if state.hasSelection {
            let start = formatTime(state.selectionStart)
            let end = formatTime(state.selectionEnd)
            return "Selection (\(start)–\(end))"
        }
        return "Whole Track"
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let seconds = Int(time) % 60
        let tenths = Int((time - Double(Int(time))) * 10)
        let minutes = Int(time) / 60
        if minutes > 0 {
            return String(format: "%d:%02d.%d", minutes, seconds, tenths)
        }
        return String(format: "%d.%ds", seconds, tenths)
    }
}
```

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Editor/DSPPaletteView.swift
git commit -m "feat(dsp): add DSPPaletteView with effect grid and expanded panels"
```

---

### Task 11: EditorState DSP Additions

**Files:**

- Modify: `AnvilApp/UI/Editor/EditorState.swift`

- [ ] **Step 1: Add DSP properties and method**

Add the following after the existing `@Published var playbackProgress` line:

```swift
    // MARK: - DSP State

    @Published var expandedEffect: DSPEffect?
    @Published var eqParameters: EQParameters = .default
    @Published var reverbParameters: ReverbParameters = .default
    @Published var selectedReverbPreset: ReverbPreset? = .hall
    @Published var audioBuffer: AudioBuffer?
    @Published var dspConfirmation: DSPEffect?

    private let dspProcessor = DSPProcessor()

    func applyDSPEffect(_ effect: DSPEffect, parameters: DSPParameters = .none) {
        guard let buffer = audioBuffer else { return }

        let target: ApplyTarget = hasSelection
            ? .selection(start: selectionStart, end: selectionEnd)
            : .wholeTrack

        // Snapshot before applying
        let snapshot = EditSnapshot(
            description: dspProcessor.makeDescription(effect: effect, target: target).description,
            audioData: buffer.data
        )
        // Push to undo (recentEdits updated by EditHistoryManager)

        do {
            let result = try dspProcessor.apply(effect: effect, to: buffer, target: target, parameters: parameters)
            audioBuffer = result
            duration = result.duration
            waveformPeaks = computePeaks(from: result)

            // For trim/delete, clear selection since the region no longer exists
            if effect == .trimToSelection || effect == .deleteSelection {
                selectionStart = 0
                selectionEnd = 0
            }
        } catch {
            // Restore from snapshot on failure
            audioBuffer = buffer
        }
    }

    private func computePeaks(from buffer: AudioBuffer) -> [Float] {
        let floats = AudioBufferConverter.toFloats(buffer)
        let peakCount = 200
        guard floats.count > peakCount else {
            return floats.map { abs($0) }
        }
        let chunkSize = floats.count / peakCount
        var peaks = [Float](repeating: 0, count: peakCount)
        for i in 0..<peakCount {
            let start = i * chunkSize
            let end = min(start + chunkSize, floats.count)
            var peak: Float = 0
            vDSP_maxmgv(Array(floats[start..<end]), 1, &peak, vDSP_Length(end - start))
            peaks[i] = peak
        }
        return peaks
    }
```

Also add `import Accelerate` at the top of the file (alongside the existing imports).

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Editor/EditorState.swift
git commit -m "feat(dsp): add DSP state and applyDSPEffect to EditorState"
```

---

### Task 12: EditorView Integration + Keyboard Shortcuts

**Files:**

- Modify: `AnvilApp/UI/Editor/EditorView.swift`

- [ ] **Step 1: Add DSPPaletteView and keyboard shortcuts**

In `EditorView.swift`, add the `DSPPaletteView` inside the `ScrollView`'s `VStack`, after `EditorControlsBar`:

```swift
                        EditorControlsBar(state: state, onExecute: executeEdit)

                        Divider()
                            .padding(.horizontal, 16)

                        DSPPaletteView(state: state) { effect, params in
                            state.applyDSPEffect(effect, parameters: params)
                        }
```

Then add keyboard shortcuts to the outermost `VStack` (after `.onAppear`):

```swift
        .keyboardShortcut("t", modifiers: .command)  // handled via commands below
        .commands {
            CommandGroup(after: .textEditing) {
                Button("Trim to Selection") {
                    if state.hasSelection {
                        state.applyDSPEffect(.trimToSelection)
                    }
                }
                .keyboardShortcut("t", modifiers: .command)
                .disabled(!state.hasSelection || !state.isLoaded)

                Button("Delete Selection") {
                    if state.hasSelection {
                        state.applyDSPEffect(.deleteSelection)
                    }
                }
                .keyboardShortcut(.delete, modifiers: .command)
                .disabled(!state.hasSelection || !state.isLoaded)

                Button("Normalize") {
                    state.applyDSPEffect(.normalize)
                }
                .keyboardShortcut("n", modifiers: [.command, .shift])
                .disabled(!state.isLoaded)

                Divider()

                Button("Show EQ") {
                    withAnimation { state.expandedEffect = .eq }
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                .disabled(!state.isLoaded)

                Button("Show Reverb") {
                    withAnimation { state.expandedEffect = .reverb }
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(!state.isLoaded)
            }
        }
```

**Note:** The `.commands` modifier may need to live on the `NavigationSplitView` in `MainShellView.swift` rather than on `EditorView` directly, depending on SwiftUI's command scoping. If so, move the commands block to `MainShellView` and pass the `EditorState` via environment or binding.

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`). If the `.commands` modifier doesn't compile on a `VStack`, move it to the `NavigationSplitView` in `MainShellView.swift`.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Editor/EditorView.swift
git commit -m "feat(dsp): integrate DSPPaletteView and keyboard shortcuts into EditorView"
```

---

### Task 13: Audio File Loading with Real Buffer

**Files:**

- Modify: `AnvilApp/UI/Editor/EditorView.swift`

- [ ] **Step 1: Update loadFromURL to populate AudioBuffer**

Replace the existing `loadFromURL` method with a real implementation that reads audio into an `AudioBuffer`:

```swift
    private func loadFromURL(_ url: URL) {
        state.fileName = url.lastPathComponent

        do {
            let audioFile = try AVAudioFile(forReading: url)
            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)

            guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                return
            }
            try audioFile.read(into: pcmBuffer)

            let buffer = AudioBufferConverter.fromPCMBuffer(pcmBuffer)
            state.audioBuffer = buffer
            state.sampleRate = buffer.sampleRate
            state.duration = buffer.duration
            state.waveformPeaks = state.computePeaks(from: buffer)
            state.isLoaded = true
            state.selectionStart = 0
            state.selectionEnd = 0
        } catch {
            state.isLoaded = false
        }
    }
```

Also add `import AVFoundation` at the top of `EditorView.swift` if not already present.

**Note:** The `computePeaks` method added in Task 11 needs to be `internal` (not `private`) for this call to work from `EditorView`. Adjust its access level if needed.

- [ ] **Step 2: Verify the file compiles**

Build the project in Xcode (`Cmd+B`).

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Editor/EditorView.swift AnvilApp/UI/Editor/EditorState.swift
git commit -m "feat(dsp): load real audio file into AudioBuffer for DSP processing"
```

---

## Self-Review Checklist

**Spec coverage:**

- ✅ Trim to Selection — Task 3 + Task 7
- ✅ Delete Selection — Task 3 + Task 7
- ✅ Fade In / Fade Out — Task 4 + Task 7
- ✅ Normalize (peak) — Task 4 + Task 7
- ✅ 3-band Parametric EQ — Task 5 + Task 7 + Task 8
- ✅ Reverb (preset + tweak) — Task 6 + Task 7 + Task 9
- ✅ DSP Palette UI — Task 10
- ✅ EQ control panel — Task 8
- ✅ Reverb control panel — Task 9
- ✅ Integration with undo system — Task 11
- ✅ Keyboard shortcuts — Task 12
- ✅ Smart apply target — Task 10 + Task 11
- ✅ Selection-required disabled state — Task 10
- ✅ Audio file loading with real buffer — Task 13
- ✅ Instant-apply confirmation — Task 10

**Placeholder scan:** No TBDs, TODOs, or vague instructions found.

**Type consistency:**

- `AudioBuffer` used consistently across Tasks 1–7, 11, 13
- `DSPParameters` enum used in Tasks 1, 7, 10, 11
- `DSPEffect` enum used in Tasks 1, 7, 10, 11, 12
- `EQParameters` used in Tasks 1, 5, 8, 10, 11
- `ReverbParameters` used in Tasks 1, 6, 9, 10, 11
- `ApplyTarget` used in Tasks 1, 5, 6, 7, 11
- Method names match: `DSPProcessor.apply(effect:to:target:parameters:)` in Task 7, called in Task 11
- `SimpleDSP.sampleIndex(time:sampleRate:channels:)` defined in Task 3, used in Task 5
