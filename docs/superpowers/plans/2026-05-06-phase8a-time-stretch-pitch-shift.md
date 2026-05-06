# Phase 8a: Time Stretch & Pitch Shift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Time Stretch and Pitch Shift transforms to the DSP effects palette using AVAudioUnitTimePitch with quality zone indicators and snap-point sliders.

**Architecture:** Both operations use a single AVAudioUnitTimePitch node per apply, rendered offline through the existing AVAudioEngine pattern from Phase 7. Time Stretch changes buffer length; Pitch Shift preserves it. The DSPProcessor facade dispatches both new effects alongside existing ones.

**Tech Stack:** AVFoundation (AVAudioUnitTimePitch, AVAudioEngine offline rendering), SwiftUI, vDSP/Accelerate

---

## File Structure

### Modified Files

| File | Responsibility | Changes |
|------|---------------|---------|
| `AnvilApp/DSP/DSPParameters.swift` | All DSP data types | Add `TimeStretchParameters`, `PitchShiftParameters`, new `DSPEffect` cases, `DSPParameters` cases, `isTransform` property |
| `AnvilApp/DSP/AVAudioEngineDSP.swift` | AVAudioEngine offline rendering | Add `applyTimeStretch` and `applyPitchShift` methods |
| `AnvilApp/DSP/DSPProcessor.swift` | Effect dispatch facade | Add dispatch cases for time stretch/pitch shift, update `makeDescription` to accept `DSPParameters` |
| `AnvilApp/UI/Editor/DSPPaletteView.swift` | Effect button grid + panels | Split into "Effects" / "Transforms" sections, add transform panel routing |
| `AnvilApp/UI/Editor/EditorState.swift` | Editor state management | Add 2 `@Published` properties, extend selection-clearing logic |

### New Files

| File | Responsibility |
|------|---------------|
| `AnvilApp/UI/Editor/TimeStretchControlView.swift` | Rate slider with snap points, quality indicator, direct entry |
| `AnvilApp/UI/Editor/PitchShiftControlView.swift` | Pitch slider with snap points, quality indicator, direct entry |

---

### Task 1: Add Data Model Types to DSPParameters.swift

**Files:**

- Modify: `AnvilApp/DSP/DSPParameters.swift`

- [ ] **Step 1: Add TimeStretchParameters and PitchShiftParameters structs**

Add after the `ReverbParameters` section (after line 122) and before the `NormalizeMode` section:

```swift
// MARK: - TimeStretchParameters

struct TimeStretchParameters: Equatable {
    var rate: Float          // 0.25-4.0 (1.0 = original speed)
    static let `default` = TimeStretchParameters(rate: 1.0)
}

// MARK: - PitchShiftParameters

struct PitchShiftParameters: Equatable {
    var pitch: Float         // -2400 to +2400 cents (-24 to +24 semitones)
    static let `default` = PitchShiftParameters(pitch: 0)
}
```

- [ ] **Step 2: Add new cases to DSPEffect enum**

In the `DSPEffect` enum (line 21), add two new cases after `.reverb`:

```swift
    case timeStretch
    case pitchShift
```

- [ ] **Step 3: Add isTransform computed property to DSPEffect**

Add this new computed property after the `displayName` property (after line 58):

```swift
    var isTransform: Bool {
        switch self {
        case .timeStretch, .pitchShift: return true
        default: return false
        }
    }
```

- [ ] **Step 4: Update requiresParameters in DSPEffect**

Change the `requiresParameters` switch (lines 30-37) to:

```swift
    var requiresParameters: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut, .normalize:
            return false
        case .eq, .reverb, .timeStretch, .pitchShift:
            return true
        }
    }
```

- [ ] **Step 5: Update requiresSelection in DSPEffect**

Change the `requiresSelection` switch (lines 39-46) to:

```swift
    var requiresSelection: Bool {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut:
            return true
        case .normalize, .eq, .reverb, .timeStretch, .pitchShift:
            return false
        }
    }
```

- [ ] **Step 6: Update displayName in DSPEffect**

Add new cases to the `displayName` switch (before the closing brace):

```swift
        case .timeStretch: return "Time Stretch"
        case .pitchShift: return "Pitch Shift"
```

- [ ] **Step 7: Add new cases to DSPParameters enum**

Change the `DSPParameters` enum (lines 70-74) to:

```swift
enum DSPParameters: Equatable {
    case none
    case eq(EQParameters)
    case reverb(ReverbParameters)
    case timeStretch(TimeStretchParameters)
    case pitchShift(PitchShiftParameters)
}
```

- [ ] **Step 8: Verify build**

Open Xcode and build the project (Cmd+B). The build will fail because `DSPProcessor.swift` has a non-exhaustive switch — that's expected and will be fixed in Task 3.

- [ ] **Step 9: Commit**

```bash
git add AnvilApp/DSP/DSPParameters.swift
git commit -m "feat(dsp): add TimeStretch and PitchShift data model types"
```

---

### Task 2: Add AVAudioEngineDSP Processing Methods

**Files:**

- Modify: `AnvilApp/DSP/AVAudioEngineDSP.swift`

- [ ] **Step 1: Add applyTimeStretch method**

Add after the `applyReverb` section (after line 181) and before the `// MARK: - Region Splicing` section:

```swift
    // MARK: - Time Stretch

    static func applyTimeStretch(to buffer: AudioBuffer, parameters: TimeStretchParameters, target: ApplyTarget) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try applyTimeStretchToBuffer(buffer, parameters: parameters)
        case .selection(let start, let end):
            return try applyToRegion(buffer, start: start, end: end) { region in
                try applyTimeStretchToBuffer(region, parameters: parameters)
            }
        }
    }

    private static func applyTimeStretchToBuffer(_ buffer: AudioBuffer, parameters: TimeStretchParameters) throws -> AudioBuffer {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let timePitch = AVAudioUnitTimePitch()

        timePitch.rate = parameters.rate
        timePitch.pitch = 0

        engine.attach(player)
        engine.attach(timePitch)
        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)

        let inputFrames = pcmBuffer.frameLength
        let outputFrameCount = AVAudioFrameCount(Double(inputFrames) / Double(parameters.rate))

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
        try engine.start()
        player.play()
        player.scheduleBuffer(pcmBuffer, completionHandler: nil)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: outputFrameCount) else {
            throw EngineError.bufferCreationFailed
        }

        var framesRendered: AVAudioFrameCount = 0
        while framesRendered < outputFrameCount {
            let framesToRender = min(4096, outputFrameCount - framesRendered)
            guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRender) else {
                throw EngineError.bufferCreationFailed
            }
            let status = try engine.renderOffline(framesToRender, to: chunk)
            if status == .error {
                throw EngineError.renderFailed("Time stretch render error at frame \(framesRendered)")
            }

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
        return AudioBufferConverter.fromPCMBuffer(outputBuffer)
    }
```

- [ ] **Step 2: Add applyPitchShift method**

Add after the `applyTimeStretchToBuffer` method (immediately after the code from Step 1):

```swift
    // MARK: - Pitch Shift

    static func applyPitchShift(to buffer: AudioBuffer, parameters: PitchShiftParameters, target: ApplyTarget) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try applyPitchShiftToBuffer(buffer, parameters: parameters)
        case .selection(let start, let end):
            return try applyToRegion(buffer, start: start, end: end) { region in
                try applyPitchShiftToBuffer(region, parameters: parameters)
            }
        }
    }

    private static func applyPitchShiftToBuffer(_ buffer: AudioBuffer, parameters: PitchShiftParameters) throws -> AudioBuffer {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let timePitch = AVAudioUnitTimePitch()

        timePitch.rate = 1.0
        timePitch.pitch = parameters.pitch

        engine.attach(player)
        engine.attach(timePitch)
        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
        try engine.start()
        player.play()
        player.scheduleBuffer(pcmBuffer, completionHandler: nil)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: pcmBuffer.frameLength) else {
            throw EngineError.bufferCreationFailed
        }

        var framesRendered: AVAudioFrameCount = 0
        let totalFrames = pcmBuffer.frameLength

        while framesRendered < totalFrames {
            let framesToRender = min(4096, totalFrames - framesRendered)
            guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRender) else {
                throw EngineError.bufferCreationFailed
            }
            let status = try engine.renderOffline(framesToRender, to: chunk)
            if status == .error {
                throw EngineError.renderFailed("Pitch shift render error at frame \(framesRendered)")
            }

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
        return AudioBufferConverter.fromPCMBuffer(outputBuffer)
    }
```

- [ ] **Step 3: Verify build**

Open Xcode and build (Cmd+B). Still expect a build failure from `DSPProcessor.swift` non-exhaustive switch — that's next.

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/DSP/AVAudioEngineDSP.swift
git commit -m "feat(dsp): add time stretch and pitch shift AVAudioEngine processing"
```

---

### Task 3: Update DSPProcessor Dispatch and Descriptions

**Files:**

- Modify: `AnvilApp/DSP/DSPProcessor.swift`

- [ ] **Step 1: Add dispatch cases for time stretch and pitch shift**

In the `apply` method's switch statement, add two new cases before the closing brace (after the `.reverb` case at line 53):

```swift
        case .timeStretch:
            guard case .timeStretch(let params) = parameters else {
                return buffer
            }
            return try AVAudioEngineDSP.applyTimeStretch(to: buffer, parameters: params, target: target)

        case .pitchShift:
            guard case .pitchShift(let params) = parameters else {
                return buffer
            }
            return try AVAudioEngineDSP.applyPitchShift(to: buffer, parameters: params, target: target)
```

- [ ] **Step 2: Update makeDescription to accept DSPParameters**

Replace the `makeDescription` method (lines 57-71) with this version that includes parameter values in descriptions:

```swift
    func makeDescription(effect: DSPEffect, target: ApplyTarget, parameters: DSPParameters = .none) -> DSPOperation {
        let targetLabel: String
        switch target {
        case .selection(let start, let end):
            targetLabel = "(\(formatTime(start))–\(formatTime(end)))"
        case .wholeTrack:
            targetLabel = "(whole track)"
        }

        let paramLabel: String
        switch parameters {
        case .timeStretch(let params):
            paramLabel = " \(String(format: "%.2f", params.rate))×"
        case .pitchShift(let params):
            let semitones = Int(params.pitch / 100)
            let sign = semitones >= 0 ? "+" : ""
            paramLabel = " \(sign)\(semitones) st"
        default:
            paramLabel = ""
        }

        return DSPOperation(
            effect: effect,
            target: target,
            description: "\(effect.displayName)\(paramLabel) \(targetLabel)"
        )
    }
```

- [ ] **Step 3: Verify build**

Open Xcode and build (Cmd+B). The build should now succeed — all switch statements are exhaustive and `makeDescription` signature is backwards-compatible (default parameter value).

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/DSP/DSPProcessor.swift
git commit -m "feat(dsp): add time stretch and pitch shift dispatch to DSPProcessor"
```

---

### Task 4: Create TimeStretchControlView

**Files:**

- Create: `AnvilApp/UI/Editor/TimeStretchControlView.swift`

- [ ] **Step 1: Create the TimeStretchControlView file**

Create `AnvilApp/UI/Editor/TimeStretchControlView.swift` with this content:

```swift
import SwiftUI

struct TimeStretchControlView: View {
    @Binding var parameters: TimeStretchParameters
    @State private var isEditingValue = false
    @State private var editText = ""

    private let snapPoints: [Float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    private let snapThreshold: Float = 0.03

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text("Rate").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if isEditingValue {
                    TextField("", text: $editText)
                        .font(.caption).monospacedDigit()
                        .frame(width: 60)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { commitEdit() }
                } else {
                    Text(formatRate(parameters.rate))
                        .font(.caption).monospacedDigit()
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color(white: 0.15))
                        .cornerRadius(4)
                        .onTapGesture(count: 2) { beginEdit() }
                }
            }

            Slider(value: $parameters.rate, in: 0.25...4.0)
                .tint(qualityColor)
                .onChange(of: parameters.rate) { _, newValue in
                    parameters.rate = snapped(newValue)
                }

            HStack(spacing: 0) {
                Text("0.25×")
                Spacer()
                Text("1.0×")
                Spacer()
                Text("4.0×")
            }
            .font(.system(size: 9))
            .foregroundStyle(.tertiary)

            qualityIndicator
        }
    }

    private var qualityIndicator: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(qualityColor)
                .frame(width: 6, height: 6)
            Text(qualityText)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var qualityColor: Color {
        let rate = parameters.rate
        if rate >= 0.5 && rate <= 2.0 { return .green }
        if rate >= 0.35 && rate <= 3.0 { return .yellow }
        return .red
    }

    private var qualityText: String {
        let rate = parameters.rate
        if rate >= 0.5 && rate <= 2.0 { return "Excellent quality" }
        if rate >= 0.35 && rate <= 3.0 { return "Good quality, minor artifacts possible" }
        return "Significant artifacts likely"
    }

    private func snapped(_ value: Float) -> Float {
        for snap in snapPoints {
            if abs(value - snap) < snapThreshold {
                return snap
            }
        }
        return value
    }

    private func formatRate(_ rate: Float) -> String {
        String(format: "%.2f×", rate)
    }

    private func beginEdit() {
        editText = String(format: "%.2f", parameters.rate)
        isEditingValue = true
    }

    private func commitEdit() {
        if let value = Float(editText) {
            parameters.rate = min(max(value, 0.25), 4.0)
        }
        isEditingValue = false
    }
}
```

- [ ] **Step 2: Verify build**

Open Xcode and build (Cmd+B). Should compile — this view is standalone and not yet wired into DSPPaletteView.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Editor/TimeStretchControlView.swift
git commit -m "feat(ui): add TimeStretchControlView with snap points and quality indicator"
```

---

### Task 5: Create PitchShiftControlView

**Files:**

- Create: `AnvilApp/UI/Editor/PitchShiftControlView.swift`

- [ ] **Step 1: Create the PitchShiftControlView file**

Create `AnvilApp/UI/Editor/PitchShiftControlView.swift` with this content:

```swift
import SwiftUI

struct PitchShiftControlView: View {
    @Binding var parameters: PitchShiftParameters
    @State private var isEditingValue = false
    @State private var editText = ""

    private let snapPoints: [Float] = [-1200, -700, -500, 0, 500, 700, 1200]
    private let snapThreshold: Float = 30.0

    private var semitones: Float {
        parameters.pitch / 100.0
    }

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text("Pitch").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if isEditingValue {
                    TextField("", text: $editText)
                        .font(.caption).monospacedDigit()
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { commitEdit() }
                } else {
                    Text(formatPitch(parameters.pitch))
                        .font(.caption).monospacedDigit()
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color(white: 0.15))
                        .cornerRadius(4)
                        .onTapGesture(count: 2) { beginEdit() }
                }
            }

            Slider(value: $parameters.pitch, in: -2400...2400)
                .tint(qualityColor)
                .onChange(of: parameters.pitch) { _, newValue in
                    parameters.pitch = snapped(newValue)
                }

            HStack(spacing: 0) {
                Text("-24 st")
                Spacer()
                Text("0")
                Spacer()
                Text("+24 st")
            }
            .font(.system(size: 9))
            .foregroundStyle(.tertiary)

            qualityIndicator
        }
    }

    private var qualityIndicator: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(qualityColor)
                .frame(width: 6, height: 6)
            Text(qualityText)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var qualityColor: Color {
        let absSemitones = abs(semitones)
        if absSemitones <= 12 { return .green }
        if absSemitones <= 18 { return .yellow }
        return .red
    }

    private var qualityText: String {
        let absSemitones = abs(semitones)
        if absSemitones <= 12 { return "Excellent quality" }
        if absSemitones <= 18 { return "Good quality, minor artifacts possible" }
        return "Significant artifacts likely"
    }

    private func snapped(_ value: Float) -> Float {
        for snap in snapPoints {
            if abs(value - snap) < snapThreshold {
                return snap
            }
        }
        return value
    }

    private func formatPitch(_ cents: Float) -> String {
        let st = Int(cents / 100)
        let sign = st >= 0 ? "+" : ""
        return "\(sign)\(st) st (\(sign)\(Int(cents))¢)"
    }

    private func beginEdit() {
        editText = String(Int(semitones))
        isEditingValue = true
    }

    private func commitEdit() {
        if let value = Float(editText) {
            let clamped = min(max(value, -24), 24)
            parameters.pitch = clamped * 100
        }
        isEditingValue = false
    }
}
```

- [ ] **Step 2: Verify build**

Open Xcode and build (Cmd+B). Should compile — standalone view.

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Editor/PitchShiftControlView.swift
git commit -m "feat(ui): add PitchShiftControlView with snap points and quality indicator"
```

---

### Task 6: Update DSPPaletteView with Effects/Transforms Sections

**Files:**

- Modify: `AnvilApp/UI/Editor/DSPPaletteView.swift`

- [ ] **Step 1: Replace the body and effectGrid with sectioned layout**

Replace the `body` property (lines 9-25) with:

```swift
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("EFFECTS")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .tracking(0.5)

            effectGrid

            if let expanded = state.expandedEffect, !expanded.isTransform {
                expandedPanel(for: expanded)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            Divider()
                .padding(.vertical, 4)

            Text("TRANSFORMS")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .tracking(0.5)

            transformGrid

            if let expanded = state.expandedEffect, expanded.isTransform {
                expandedPanel(for: expanded)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(16)
        .animation(.easeInOut(duration: 0.2), value: state.expandedEffect)
    }
```

- [ ] **Step 2: Add the transformGrid computed property**

Add after the `effectGrid` computed property (after line 47):

```swift
    private var transformGrid: some View {
        HStack(spacing: 6) {
            effectButton(.timeStretch)
            effectButton(.pitchShift)
            Spacer()
        }
    }
```

- [ ] **Step 3: Update the expandedPanel switch to handle new effects**

In the `expandedPanel(for:)` method, update the switch statement (lines 114-124) to:

```swift
            switch effect {
            case .eq:
                EQControlView(parameters: $state.eqParameters)
            case .reverb:
                ReverbControlView(
                    parameters: $state.reverbParameters,
                    selectedPreset: $state.selectedReverbPreset
                )
            case .timeStretch:
                TimeStretchControlView(parameters: $state.timeStretchParameters)
            case .pitchShift:
                PitchShiftControlView(parameters: $state.pitchShiftParameters)
            default:
                EmptyView()
            }
```

- [ ] **Step 4: Update the applySection to handle new effects**

In the `applySection(for:)` method, update the params switch (lines 150-155) to:

```swift
                let params: DSPParameters
                switch effect {
                case .eq: params = .eq(state.eqParameters)
                case .reverb: params = .reverb(state.reverbParameters)
                case .timeStretch: params = .timeStretch(state.timeStretchParameters)
                case .pitchShift: params = .pitchShift(state.pitchShiftParameters)
                default: params = .none
                }
```

- [ ] **Step 5: Verify build**

Open Xcode and build (Cmd+B). Build will fail because `EditorState` doesn't yet have `timeStretchParameters` and `pitchShiftParameters` — that's Task 7.

- [ ] **Step 6: Commit**

```bash
git add AnvilApp/UI/Editor/DSPPaletteView.swift
git commit -m "feat(ui): split DSP palette into Effects and Transforms sections"
```

---

### Task 7: Update EditorState with Transform Properties

**Files:**

- Modify: `AnvilApp/UI/Editor/EditorState.swift`

- [ ] **Step 1: Add published properties for transform parameters**

In the `// MARK: - DSP State` section (after line 44, after `@Published var dspConfirmation: DSPEffect?`), add:

```swift
    @Published var timeStretchParameters: TimeStretchParameters = .default
    @Published var pitchShiftParameters: PitchShiftParameters = .default
```

- [ ] **Step 2: Extend selection-clearing logic for time stretch**

In the `applyDSPEffect` method (line 120), update the selection-clearing check from:

```swift
            if effect == .trimToSelection || effect == .deleteSelection {
```

to:

```swift
            if effect == .trimToSelection || effect == .deleteSelection || effect == .timeStretch {
```

This clears the selection after time stretch because the buffer duration changes, shifting region boundaries.

- [ ] **Step 3: Verify build**

Open Xcode and build (Cmd+B). The project should now compile successfully with all changes from Tasks 1-7.

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/UI/Editor/EditorState.swift
git commit -m "feat(editor): add time stretch and pitch shift state properties"
```

---

### Task 8: Final Integration Verification

**Files:**

- No file changes — verification only

- [ ] **Step 1: Full Xcode build**

Open Xcode and build the full project (Cmd+B). Verify clean build with zero warnings related to DSP or editor files.

- [ ] **Step 2: Verify the complete data flow**

Trace the flow mentally through the code to confirm correctness:

1. User clicks "Time Stretch" button in DSPPaletteView → `state.expandedEffect = .timeStretch`
2. Panel shows `TimeStretchControlView` with rate slider (0.25×–4.0×)
3. User adjusts slider → snap points engage → quality indicator updates
4. User clicks Apply → `onApply(.timeStretch, .timeStretch(state.timeStretchParameters))`
5. EditorState.applyDSPEffect dispatches through DSPProcessor
6. DSPProcessor routes to AVAudioEngineDSP.applyTimeStretch
7. AVAudioEngine renders offline with AVAudioUnitTimePitch (rate set, pitch 0)
8. Output buffer has new length (`inputFrames / rate`)
9. EditorState updates audioBuffer, duration, waveformPeaks
10. Selection is cleared (time stretch changes duration)

Same flow for Pitch Shift, but output buffer preserves original length and selection is not cleared.

- [ ] **Step 3: Commit all changes together if any fixups were needed**

If any build fixes were needed:

```bash
git add -A
git commit -m "fix: resolve Phase 8a integration issues"
```
