# Phase 4a: Generation Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing inference pipeline into a functional end-to-end generation flow with UI — from "type prompt" to "hear audio" with progress, cancel, history, and export.

**Architecture:** A `GenerationService` ObservableObject coordinates the full lifecycle: input validation → weight loading → conditioning → diffusion → decode → WAV export → history update. New UI views observe this service. The encoderMask plumbing fix and WAVWriter land in AnvilCore; everything else lives in the app target.

**Tech Stack:** Swift 5.10, SwiftUI, AVFoundation, MLX-Swift, Swift Testing (AnvilCore tests), Xcode build (app target)

---

## File Structure

```text
AnvilCore/Sources/AnvilCore/Inference/
├── Pipeline/
│   └── MusicPipeline.swift          (modified — add encoderMask param)
├── Diffusion/
│   └── DiffusionActor.swift         (modified — add encoderMask, throws, cancellation)
├── DiT/
│   ├── DiTActor.swift               (modified — add encoderMask param)
│   └── DiTModel.swift               (modified — use mask instead of nil)
├── Audio/
│   └── WAVWriter.swift              (new)

AnvilCore/Tests/AnvilCoreTests/Inference/
├── WAVWriterTests.swift             (new)

AnvilApp/
├── Services/
│   ├── GenerationService.swift      (new)
│   ├── AudioPlayerService.swift     (new)
│   ├── HistoryStore.swift           (new)
│   └── OutputDirectoryManager.swift (new)
├── Models/
│   ├── GenerationRequest.swift      (new)
│   ├── GenerationState.swift        (new)
│   └── HistoryEntry.swift           (new)
├── UI/
│   ├── MainShellView.swift          (modified)
│   ├── Components/
│   │   ├── GenerateView.swift       (new)
│   │   ├── PlayerView.swift         (new)
│   │   ├── WaveformView.swift       (new)
│   │   ├── HistoryListView.swift    (new)
│   │   └── OutputDirectoryPicker.swift (new)
```

---

### Task 1: Wire encoderMask Through Pipeline

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTModel.swift:89-160`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift:12-36`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift:22-119`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift:37-102`

- [ ] **Step 1: Add encoderMask parameter to DiTDecoder.callAsFunction**

In `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTModel.swift`, change the method signature (line 89) and the layer loop (line 132-144):

```swift
public func callAsFunction(
    hiddenStates: MLXArray,
    timestep: MLXArray,
    timestepR: MLXArray,
    encoderHiddenStates: MLXArray,
    contextLatents: MLXArray,
    encoderAttentionMask: MLXArray?,
    cache: CrossAttentionCache?,
    useCache: Bool
) -> (MLXArray, CrossAttentionCache?) {
```

Convert mask format before the layer loop (after line 129):

```swift
let additiveMask: MLXArray? = encoderAttentionMask.map { mask in
    let inverted = (1.0 - mask) * Float(-1e9)
    return inverted.expandedDimensions(axes: [1, 2])
}
```

In the layer loop, replace `encoderAttentionMask: nil` with `encoderAttentionMask: additiveMask`:

```swift
for layer in layers {
    let selfAttnMask = layer.layerType == "sliding_attention" ? slidingMask : nil
    h = layer(
        hiddenStates: h,
        positionCosSin: (cos, sin),
        temb: timestepProj,
        selfAttnMask: selfAttnMask,
        encoderHiddenStates: encStates,
        encoderAttentionMask: additiveMask,
        cache: cache,
        useCache: useCache
    )
}
```

- [ ] **Step 2: Add encoderMask parameter to DiTActor.step**

In `AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift`, update the `step` method:

```swift
public func step(
    noisyLatents: MLXArray,
    timestep: Float,
    timestepR: Float,
    encoderStates: MLXArray,
    contextLatents: MLXArray,
    encoderAttentionMask: MLXArray?,
    cache: CrossAttentionCache?,
    useCache: Bool
) -> (output: MLXArray, cache: CrossAttentionCache?) {
    let batchSize = noisyLatents.shape[0]
    let tArray = MLXArray.full([batchSize], values: MLXArray(timestep))
    let trArray = MLXArray.full([batchSize], values: MLXArray(timestepR))

    let (output, returnedCache) = decoder(
        hiddenStates: noisyLatents,
        timestep: tArray,
        timestepR: trArray,
        encoderHiddenStates: encoderStates,
        contextLatents: contextLatents,
        encoderAttentionMask: encoderAttentionMask,
        cache: cache,
        useCache: useCache
    )
    eval(output)
    return (output, returnedCache)
}
```

- [ ] **Step 3: Add encoderMask parameter to DiffusionActor.generate**

In `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift`, update the signature and the `dit.step` call:

```swift
public func generate(
    encoderStates: MLXArray,
    contextLatents: MLXArray,
    encoderMask: MLXArray?,
    shape: (batch: Int, frames: Int, channels: Int),
    config: DiffusionConfig
) async -> DiffusionResult {
```

In the CFG block (around line 44-48), also duplicate the mask:

```swift
var encMask = encoderMask
if doCFG {
    let nullExpanded = MLXArray.zeros(like: encHS)
    encHS = concatenated([encHS, nullExpanded], axis: 0)
    ctx = concatenated([ctx, ctx], axis: 0)
    encMask = encoderMask.map { concatenated([$0, $0], axis: 0) }
}
```

Update the `dit.step` call (around line 65):

```swift
let (vt, _) = await dit.step(
    noisyLatents: xIn,
    timestep: currentT,
    timestepR: currentT,
    encoderStates: encHS,
    contextLatents: ctx,
    encoderAttentionMask: encMask,
    cache: cache,
    useCache: !doCFG
)
```

- [ ] **Step 4: Add encoderMask to MusicPipeline.generate (low-level overload)**

In `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift`, update the first generate method:

```swift
public func generate(
    encoderStates: MLXArray,
    contextLatents: MLXArray,
    encoderMask: MLXArray?,
    duration: TimeInterval,
    config: DiffusionConfig
) async throws -> AudioResult {
```

And the diffusion call:

```swift
let diffResult = await diffusion.generate(
    encoderStates: encoderStates,
    contextLatents: contextLatents,
    encoderMask: encoderMask,
    shape: (batch: batchSize, frames: frames, channels: 64),
    config: config
)
```

- [ ] **Step 5: Update MusicPipeline.generate convenience overload**

Pass `cond.encoderMask` through:

```swift
public func generate(
    text: String,
    lyrics: String,
    referenceAudio: MLXArray?,
    duration: TimeInterval,
    config: DiffusionConfig,
    conditioning: ConditioningActor
) async throws -> AudioResult {
    let cond = await conditioning.encode(
        text: text,
        lyrics: lyrics,
        referenceAudio: referenceAudio,
        duration: duration,
        precision: config.precision
    )
    return try await generate(
        encoderStates: cond.encoderStates,
        contextLatents: cond.contextLatents,
        encoderMask: cond.encoderMask,
        duration: duration,
        config: config
    )
}
```

- [ ] **Step 6: Verify build succeeds**

Run: `swift build --package-path AnvilCore 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 7: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/DiT/DiTModel.swift \
        AnvilCore/Sources/AnvilCore/Inference/DiT/DiTActor.swift \
        AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift \
        AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift
git commit -m "feat: wire encoderMask through pipeline to DiT cross-attention"
```

---

### Task 2: Add Cancellation Support to DiffusionActor

**Files:**
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift`
- Modify: `AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift`

- [ ] **Step 1: Make DiffusionActor.generate throw**

Change the signature from `async -> DiffusionResult` to `async throws -> DiffusionResult`:

```swift
public func generate(
    encoderStates: MLXArray,
    contextLatents: MLXArray,
    encoderMask: MLXArray?,
    shape: (batch: Int, frames: Int, channels: Int),
    config: DiffusionConfig
) async throws -> DiffusionResult {
```

- [ ] **Step 2: Add cancellation check in the diffusion loop**

At the top of the `for stepIdx in 0..<numSteps` loop (line 56), add:

```swift
for stepIdx in 0..<numSteps {
    try Task.checkCancellation()

    let currentT = schedule[stepIdx]
    // ... rest of loop unchanged
```

- [ ] **Step 3: Update MusicPipeline to propagate throws**

The `diffusion.generate(...)` call in MusicPipeline now throws, so add `try`:

```swift
let diffResult = try await diffusion.generate(
    encoderStates: encoderStates,
    contextLatents: contextLatents,
    encoderMask: encoderMask,
    shape: (batch: batchSize, frames: frames, channels: 64),
    config: config
)
```

Both `MusicPipeline.generate` overloads already have `throws` in their signature, so no signature change needed.

- [ ] **Step 4: Verify build succeeds**

Run: `swift build --package-path AnvilCore 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Diffusion/DiffusionActor.swift \
        AnvilCore/Sources/AnvilCore/Inference/Pipeline/MusicPipeline.swift
git commit -m "feat: add cancellation support to DiffusionActor via Task.checkCancellation"
```

---

### Task 3: WAVWriter

**Files:**
- Create: `AnvilCore/Sources/AnvilCore/Inference/Audio/WAVWriter.swift`
- Create: `AnvilCore/Tests/AnvilCoreTests/Inference/WAVWriterTests.swift`

- [ ] **Step 1: Write the test file**

Create `AnvilCore/Tests/AnvilCoreTests/Inference/WAVWriterTests.swift`:

```swift
import Testing
import Foundation
import MLX
@testable import AnvilCore

@Suite("WAVWriter")
struct WAVWriterTests {
    @Test("Writes valid WAV header and PCM data")
    func writesValidWAV() throws {
        let numFrames = 44100
        let samples = MLXArray.zeros([2, numFrames])
        let url = FileManager.default.temporaryDirectory
            .appending(path: "test_output.wav")

        try WAVWriter.write(samples: samples, sampleRate: 44100, to: url)

        let data = try Data(contentsOf: url)
        // WAV header is 44 bytes + 2 channels * numFrames * 2 bytes per sample
        let expectedSize = 44 + (2 * numFrames * 2)
        #expect(data.count == expectedSize)

        // Check RIFF header
        let riff = String(data: data[0..<4], encoding: .ascii)
        #expect(riff == "RIFF")

        // Check WAVE format
        let wave = String(data: data[8..<12], encoding: .ascii)
        #expect(wave == "WAVE")

        // Check sample rate at byte offset 24 (little-endian UInt32)
        let sampleRate = data[24...27].withUnsafeBytes { $0.load(as: UInt32.self) }
        #expect(sampleRate == 44100)

        // Check channels at byte offset 22 (little-endian UInt16)
        let channels = data[22...23].withUnsafeBytes { $0.load(as: UInt16.self) }
        #expect(channels == 2)

        try FileManager.default.removeItem(at: url)
    }

    @Test("Clamps samples to [-1, 1] range")
    func clampsSamples() throws {
        let samples = MLXArray(converting: [[2.0, -2.0, 0.5, -0.5]], type: Float.self)
        let url = FileManager.default.temporaryDirectory
            .appending(path: "test_clamp.wav")

        try WAVWriter.write(samples: samples, sampleRate: 44100, to: url)

        let data = try Data(contentsOf: url)
        // Read first two Int16 samples after 44-byte header
        let firstSample = data[44...45].withUnsafeBytes { $0.load(as: Int16.self) }
        let secondSample = data[46...47].withUnsafeBytes { $0.load(as: Int16.self) }
        #expect(firstSample == 32767)   // clamped from 2.0
        #expect(secondSample == -32767) // clamped from -2.0

        try FileManager.default.removeItem(at: url)
    }
}
```

- [ ] **Step 2: Verify tests fail (WAVWriter doesn't exist yet)**

Run: `swift build --package-path AnvilCore 2>&1 | grep -c "error"`
Expected: Multiple errors about WAVWriter not found

- [ ] **Step 3: Implement WAVWriter**

Create `AnvilCore/Sources/AnvilCore/Inference/Audio/WAVWriter.swift`:

```swift
import Foundation
import MLX

public struct WAVWriter {
    public static func write(
        samples: MLXArray,
        sampleRate: Int = 44100,
        to url: URL
    ) throws {
        let channels: Int
        let numFrames: Int

        if samples.ndim == 1 {
            channels = 1
            numFrames = samples.shape[0]
        } else {
            channels = samples.shape[0]
            numFrames = samples.shape[1]
        }

        let clamped = samples.clipped(min: -1.0, max: 1.0)
        let scaled = (clamped * Float(32767)).asType(.int16)
        eval(scaled)

        let interleaved: MLXArray
        if channels == 1 {
            interleaved = scaled.reshaped([-1])
        } else {
            interleaved = scaled.transposed(0, 1).reshaped([-1])
        }
        eval(interleaved)

        let pcmData = interleaved.asData(type: Int16.self)

        let bitsPerSample: UInt16 = 16
        let blockAlign = UInt16(channels) * (bitsPerSample / 8)
        let byteRate = UInt32(sampleRate) * UInt32(blockAlign)
        let dataSize = UInt32(pcmData.count)
        let fileSize = 36 + dataSize

        var header = Data(capacity: 44)
        header.append(contentsOf: "RIFF".utf8)
        header.append(littleEndian: fileSize)
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        header.append(littleEndian: UInt32(16))
        header.append(littleEndian: UInt16(1))
        header.append(littleEndian: UInt16(channels))
        header.append(littleEndian: UInt32(sampleRate))
        header.append(littleEndian: byteRate)
        header.append(littleEndian: blockAlign)
        header.append(littleEndian: bitsPerSample)
        header.append(contentsOf: "data".utf8)
        header.append(littleEndian: dataSize)

        var fileData = header
        fileData.append(pcmData)
        try fileData.write(to: url)
    }
}

private extension Data {
    mutating func append(littleEndian value: UInt16) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 2))
    }

    mutating func append(littleEndian value: UInt32) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 4))
    }
}

private extension MLXArray {
    func asData(type: Int16.Type) -> Data {
        let count = size
        let flat = reshaped([-1])
        var buffer = [Int16](repeating: 0, count: count)
        for i in 0..<count {
            buffer[i] = flat[i].item(Int16.self)
        }
        return Data(bytes: buffer, count: count * MemoryLayout<Int16>.size)
    }
}
```

- [ ] **Step 4: Verify build and tests pass**

Run: `swift build --package-path AnvilCore 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add AnvilCore/Sources/AnvilCore/Inference/Audio/WAVWriter.swift \
        AnvilCore/Tests/AnvilCoreTests/Inference/WAVWriterTests.swift
git commit -m "feat: add WAVWriter for PCM 16-bit stereo WAV export"
```

---

### Task 4: Model Types (GenerationRequest, GenerationState, HistoryEntry)

**Files:**
- Create: `AnvilApp/Models/GenerationRequest.swift`
- Create: `AnvilApp/Models/GenerationState.swift`
- Create: `AnvilApp/Models/HistoryEntry.swift`

- [ ] **Step 1: Create GenerationRequest**

Create `AnvilApp/Models/GenerationRequest.swift`:

```swift
import Foundation
import AnvilCore

struct GenerationRequest: Sendable {
    let text: String
    let lyrics: String
    let referenceAudioURL: URL?
    let duration: TimeInterval
    let seed: Int?
    let method: InferMethod
    let guidanceScale: Float
    let precision: Precision

    init(
        text: String,
        lyrics: String = "",
        referenceAudioURL: URL? = nil,
        duration: TimeInterval = 30.0,
        seed: Int? = nil,
        method: InferMethod = .ode,
        guidanceScale: Float = 1.0,
        precision: Precision = .float16
    ) {
        self.text = text
        self.lyrics = lyrics
        self.referenceAudioURL = referenceAudioURL
        self.duration = duration
        self.seed = seed
        self.method = method
        self.guidanceScale = guidanceScale
        self.precision = precision
    }
}
```

- [ ] **Step 2: Create GenerationState**

Create `AnvilApp/Models/GenerationState.swift`:

```swift
import Foundation

enum GenerationState: Equatable, Sendable {
    case idle
    case loadingWeights(progress: Float)
    case conditioning
    case generating(step: Int, totalSteps: Int, eta: TimeInterval)
    case decoding
    case complete
    case error(String)
    case cancelled
}
```

- [ ] **Step 3: Create HistoryEntry**

Create `AnvilApp/Models/HistoryEntry.swift`:

```swift
import Foundation

struct HistoryEntry: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    let createdAt: Date
    let prompt: String
    let lyrics: String
    let duration: TimeInterval
    let seed: Int
    let method: String
    let guidanceScale: Float
    let precision: String
    let fileName: String

    static func fileName(prompt: String, date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = formatter.string(from: date)

        let words = prompt
            .lowercased()
            .components(separatedBy: .whitespaces)
            .prefix(5)
            .joined(separator: "-")
            .replacingOccurrences(
                of: "[^a-z0-9\\-]",
                with: "",
                options: .regularExpression
            )
            .prefix(50)

        return "anvil_\(timestamp)_\(words).wav"
    }
}
```

- [ ] **Step 4: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 5: Commit**

```bash
git add AnvilApp/Models/
git commit -m "feat: add GenerationRequest, GenerationState, and HistoryEntry models"
```

---

### Task 5: OutputDirectoryManager

**Files:**
- Create: `AnvilApp/Services/OutputDirectoryManager.swift`

- [ ] **Step 1: Implement OutputDirectoryManager**

Create `AnvilApp/Services/OutputDirectoryManager.swift`:

```swift
import SwiftUI

@MainActor
final class OutputDirectoryManager: ObservableObject {
    @Published var directory: URL?

    private static let bookmarkKey = "outputDirectoryBookmark"

    init() {
        directory = resolveBookmark()
    }

    func pickDirectory() async -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Choose where Anvil saves generated audio"
        panel.prompt = "Select Folder"

        let response = panel.runModal()
        guard response == .OK, let url = panel.url else { return nil }

        saveBookmark(for: url)
        directory = url
        return url
    }

    func hasValidDirectory() -> Bool {
        guard let dir = directory else { return false }
        return FileManager.default.isWritableFile(atPath: dir.path())
    }

    private func saveBookmark(for url: URL) {
        do {
            let bookmark = try url.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            UserDefaults.standard.set(bookmark, forKey: Self.bookmarkKey)
        } catch {
            print("Failed to save bookmark: \(error)")
        }
    }

    private func resolveBookmark() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: Self.bookmarkKey) else {
            return nil
        }
        do {
            var isStale = false
            let url = try URL(
                resolvingBookmarkData: data,
                options: [.withSecurityScope],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
            guard url.startAccessingSecurityScopedResource() else { return nil }
            if isStale {
                saveBookmark(for: url)
            }
            return url
        } catch {
            return nil
        }
    }
}
```

- [ ] **Step 2: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/OutputDirectoryManager.swift
git commit -m "feat: add OutputDirectoryManager with security-scoped bookmarks"
```

---

### Task 6: HistoryStore

**Files:**
- Create: `AnvilApp/Services/HistoryStore.swift`

- [ ] **Step 1: Implement HistoryStore**

Create `AnvilApp/Services/HistoryStore.swift`:

```swift
import Foundation

@MainActor
final class HistoryStore: ObservableObject {
    @Published var entries: [HistoryEntry] = []

    private let directory: URL
    private var historyFileURL: URL { directory.appending(path: "history.json") }

    init(directory: URL) {
        self.directory = directory
        loadFromDisk()
    }

    func add(_ entry: HistoryEntry) {
        entries.insert(entry, at: 0)
        saveToDisk()
    }

    func delete(_ entry: HistoryEntry) {
        entries.removeAll { $0.id == entry.id }
        let fileURL = directory.appending(path: entry.fileName)
        try? FileManager.default.removeItem(at: fileURL)
        saveToDisk()
    }

    func deleteAll() {
        for entry in entries {
            let fileURL = directory.appending(path: entry.fileName)
            try? FileManager.default.removeItem(at: fileURL)
        }
        entries.removeAll()
        saveToDisk()
    }

    func fileURL(for entry: HistoryEntry) -> URL {
        directory.appending(path: entry.fileName)
    }

    private func loadFromDisk() {
        guard FileManager.default.fileExists(atPath: historyFileURL.path()) else { return }
        do {
            let data = try Data(contentsOf: historyFileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            entries = try decoder.decode([HistoryEntry].self, from: data)
        } catch {
            entries = []
        }
    }

    private func saveToDisk() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(entries)
            try data.write(to: historyFileURL, options: .atomic)
        } catch {
            print("Failed to save history: \(error)")
        }
    }
}
```

- [ ] **Step 2: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/HistoryStore.swift
git commit -m "feat: add HistoryStore with JSON persistence and file deletion"
```

---

### Task 7: AudioPlayerService

**Files:**
- Create: `AnvilApp/Services/AudioPlayerService.swift`

- [ ] **Step 1: Implement AudioPlayerService**

Create `AnvilApp/Services/AudioPlayerService.swift`:

```swift
import AVFoundation
import SwiftUI

struct WaveformData: Sendable {
    let peaks: [Float]
    let sampleRate: Int
    let channelCount: Int
}

@MainActor
final class AudioPlayerService: ObservableObject {
    @Published var isPlaying: Bool = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var volume: Float = 0.7
    @Published var waveform: WaveformData?

    private var player: AVAudioPlayer?
    private var timer: Timer?

    func load(url: URL) {
        stop()
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
            player?.volume = volume
            duration = player?.duration ?? 0
            currentTime = 0
            waveform = computeWaveform(url: url)
        } catch {
            print("Failed to load audio: \(error)")
        }
    }

    func play() {
        guard let player else { return }
        player.play()
        isPlaying = true
        startTimer()
    }

    func pause() {
        player?.pause()
        isPlaying = false
        stopTimer()
    }

    func togglePlayback() {
        if isPlaying { pause() } else { play() }
    }

    func seek(to time: TimeInterval) {
        player?.currentTime = time
        currentTime = time
    }

    func setVolume(_ newVolume: Float) {
        volume = newVolume
        player?.volume = newVolume
    }

    func stop() {
        player?.stop()
        isPlaying = false
        currentTime = 0
        stopTimer()
    }

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let player = self.player else { return }
                self.currentTime = player.currentTime
                if !player.isPlaying {
                    self.isPlaying = false
                    self.stopTimer()
                }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func computeWaveform(url: URL, bucketCount: Int = 400) -> WaveformData? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        do { try file.read(into: buffer) } catch { return nil }

        guard let channelData = buffer.floatChannelData else { return nil }
        let channelCount = Int(format.channelCount)
        let totalFrames = Int(buffer.frameLength)
        let framesPerBucket = max(totalFrames / bucketCount, 1)

        var peaks = [Float](repeating: 0, count: bucketCount)
        for bucket in 0..<bucketCount {
            let start = bucket * framesPerBucket
            let end = min(start + framesPerBucket, totalFrames)
            var maxVal: Float = 0
            for frame in start..<end {
                for ch in 0..<channelCount {
                    let val = abs(channelData[ch][frame])
                    if val > maxVal { maxVal = val }
                }
            }
            peaks[bucket] = maxVal
        }

        let maxPeak = peaks.max() ?? 1.0
        if maxPeak > 0 {
            peaks = peaks.map { $0 / maxPeak }
        }

        return WaveformData(
            peaks: peaks,
            sampleRate: Int(format.sampleRate),
            channelCount: channelCount
        )
    }
}
```

- [ ] **Step 2: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/AudioPlayerService.swift
git commit -m "feat: add AudioPlayerService with waveform computation and playhead tracking"
```

---

### Task 8: GenerationService

**Files:**
- Create: `AnvilApp/Services/GenerationService.swift`

- [ ] **Step 1: Implement GenerationService**

Create `AnvilApp/Services/GenerationService.swift`:

```swift
import Foundation
import AnvilCore
import MLX

struct GenerationResult: Sendable {
    let fileURL: URL
    let entry: HistoryEntry
}

@MainActor
final class GenerationService: ObservableObject {
    @Published var state: GenerationState = .idle
    @Published var currentResult: GenerationResult?

    private var generationTask: Task<Void, Never>?
    private let pipeline: MusicPipeline
    private let conditioning: ConditioningActor
    private let weightConverter: WeightConverter
    private let historyStore: HistoryStore
    private let outputDirectoryManager: OutputDirectoryManager
    private var weightsLoaded = false

    init(
        pipeline: MusicPipeline,
        conditioning: ConditioningActor,
        weightConverter: WeightConverter,
        historyStore: HistoryStore,
        outputDirectoryManager: OutputDirectoryManager
    ) {
        self.pipeline = pipeline
        self.conditioning = conditioning
        self.weightConverter = weightConverter
        self.historyStore = historyStore
        self.outputDirectoryManager = outputDirectoryManager
    }

    func generate(request: GenerationRequest) {
        generationTask?.cancel()

        generationTask = Task { [weak self] in
            guard let self else { return }

            guard !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                self.state = .error("Prompt cannot be empty")
                return
            }

            guard let outputDir = self.outputDirectoryManager.directory else {
                self.state = .error("No output directory selected")
                return
            }

            do {
                if !self.weightsLoaded {
                    self.state = .loadingWeights(progress: 0)
                    try await self.loadWeights()
                    self.weightsLoaded = true
                }

                try Task.checkCancellation()
                self.state = .conditioning

                let config = DiffusionConfig(
                    seed: request.seed,
                    method: request.method,
                    guidanceScale: request.guidanceScale,
                    precision: request.precision
                )

                self.state = .generating(step: 0, totalSteps: config.steps ?? 8, eta: 0)

                let result = try await self.pipeline.generate(
                    text: request.text,
                    lyrics: request.lyrics,
                    referenceAudio: nil,
                    duration: request.duration,
                    config: config,
                    conditioning: self.conditioning
                )

                try Task.checkCancellation()
                self.state = .decoding

                let fileName = HistoryEntry.fileName(prompt: request.text)
                let fileURL = outputDir.appending(path: fileName)

                try WAVWriter.write(
                    samples: result.samples,
                    sampleRate: result.sampleRate,
                    to: fileURL
                )

                let entry = HistoryEntry(
                    id: UUID(),
                    createdAt: Date(),
                    prompt: request.text,
                    lyrics: request.lyrics,
                    duration: request.duration,
                    seed: request.seed ?? 0,
                    method: request.method.rawValue,
                    guidanceScale: request.guidanceScale,
                    precision: request.precision.rawValue,
                    fileName: fileName
                )

                self.historyStore.add(entry)
                self.currentResult = GenerationResult(fileURL: fileURL, entry: entry)
                self.state = .complete
            } catch is CancellationError {
                self.state = .cancelled
            } catch {
                self.state = .error(error.localizedDescription)
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        state = .cancelled
    }

    private func loadWeights() async throws {
        // Weight loading will be connected to actual model files in a future task.
        // For now this is a placeholder that sets progress to 1.0.
        state = .loadingWeights(progress: 1.0)
    }
}
```

- [ ] **Step 2: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/Services/GenerationService.swift
git commit -m "feat: add GenerationService coordinating full generation lifecycle"
```

---

### Task 9: OutputDirectoryPicker View

**Files:**
- Create: `AnvilApp/UI/Components/OutputDirectoryPicker.swift`

- [ ] **Step 1: Implement OutputDirectoryPicker**

Create `AnvilApp/UI/Components/OutputDirectoryPicker.swift`:

```swift
import SwiftUI

struct OutputDirectoryPicker: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)

            Text("Choose Output Folder")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Select where Anvil saves your generated audio files.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)

            Button("Select Folder") {
                Task {
                    await outputManager.pickDirectory()
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(40)
    }
}
```

- [ ] **Step 2: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Components/OutputDirectoryPicker.swift
git commit -m "feat: add OutputDirectoryPicker view for first-launch folder selection"
```

---

### Task 10: WaveformView and PlayerView

**Files:**
- Create: `AnvilApp/UI/Components/WaveformView.swift`
- Create: `AnvilApp/UI/Components/PlayerView.swift`

- [ ] **Step 1: Implement WaveformView**

Create `AnvilApp/UI/Components/WaveformView.swift`:

```swift
import SwiftUI

struct WaveformView: View {
    let peaks: [Float]
    let progress: Double
    let onSeek: (Double) -> Void

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                let barWidth = size.width / CGFloat(peaks.count)
                let midY = size.height / 2

                for (index, peak) in peaks.enumerated() {
                    let x = CGFloat(index) * barWidth
                    let barHeight = CGFloat(peak) * size.height * 0.8
                    let progressFraction = Double(index) / Double(peaks.count)
                    let color: Color = progressFraction <= progress ? .accentColor : .secondary.opacity(0.4)

                    let rect = CGRect(
                        x: x,
                        y: midY - barHeight / 2,
                        width: max(barWidth - 1, 1),
                        height: barHeight
                    )
                    context.fill(Path(roundedRect: rect, cornerRadius: 1), with: .color(color))
                }

                let playheadX = size.width * progress
                let playheadRect = CGRect(x: playheadX - 1, y: 0, width: 2, height: size.height)
                context.fill(Path(playheadRect), with: .color(.white))
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let fraction = max(0, min(1, value.location.x / geo.size.width))
                        onSeek(fraction)
                    }
            )
        }
    }
}
```

- [ ] **Step 2: Implement PlayerView**

Create `AnvilApp/UI/Components/PlayerView.swift`:

```swift
import SwiftUI

struct PlayerView: View {
    @EnvironmentObject private var playerService: AudioPlayerService

    let fileURL: URL

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Button(action: { playerService.togglePlayback() }) {
                    Image(systemName: playerService.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 32))
                        .foregroundStyle(.accent)
                }
                .buttonStyle(.plain)

                if let waveform = playerService.waveform {
                    WaveformView(
                        peaks: waveform.peaks,
                        progress: playerService.duration > 0
                            ? playerService.currentTime / playerService.duration
                            : 0,
                        onSeek: { fraction in
                            let time = fraction * playerService.duration
                            playerService.seek(to: time)
                        }
                    )
                    .frame(height: 40)
                } else {
                    ProgressView()
                        .frame(height: 40)
                }

                Text(timeString(playerService.currentTime) + " / " + timeString(playerService.duration))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(minWidth: 70)
            }

            HStack(spacing: 12) {
                Image(systemName: "speaker.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Slider(value: Binding(
                    get: { playerService.volume },
                    set: { playerService.setVolume($0) }
                ), in: 0...1)
                .frame(width: 80)

                Spacer()

                Button("Export WAV") {
                    exportToFinder()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([fileURL])
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .onAppear {
            playerService.load(url: fileURL)
        }
    }

    private func timeString(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    private func exportToFinder() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = fileURL.lastPathComponent
        panel.begin { response in
            guard response == .OK, let destination = panel.url else { return }
            try? FileManager.default.copyItem(at: fileURL, to: destination)
        }
    }
}
```

- [ ] **Step 3: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/UI/Components/WaveformView.swift \
        AnvilApp/UI/Components/PlayerView.swift
git commit -m "feat: add WaveformView and PlayerView for inline audio playback"
```

---

### Task 11: HistoryListView

**Files:**
- Create: `AnvilApp/UI/Components/HistoryListView.swift`

- [ ] **Step 1: Implement HistoryListView**

Create `AnvilApp/UI/Components/HistoryListView.swift`:

```swift
import SwiftUI

struct HistoryListView: View {
    @EnvironmentObject private var historyStore: HistoryStore
    @EnvironmentObject private var playerService: AudioPlayerService

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            if historyStore.entries.isEmpty {
                emptyState
            } else {
                list
            }
        }
    }

    private var header: some View {
        HStack {
            Text("HISTORY")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
            Spacer()
            if !historyStore.entries.isEmpty {
                Button("Delete All", role: .destructive) {
                    historyStore.deleteAll()
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform")
                .font(.title)
                .foregroundStyle(.quaternary)
            Text("No generations yet")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var list: some View {
        List {
            ForEach(historyStore.entries) { entry in
                HistoryRow(entry: entry)
                    .onTapGesture {
                        let url = historyStore.fileURL(for: entry)
                        playerService.load(url: url)
                    }
            }
            .onDelete { indices in
                for index in indices {
                    historyStore.delete(historyStore.entries[index])
                }
            }
        }
        .listStyle(.plain)
    }
}

struct HistoryRow: View {
    let entry: HistoryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.prompt)
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.tail)

            HStack(spacing: 8) {
                Text("\(Int(entry.duration))s")
                Text("•")
                Text(entry.createdAt, style: .date)
                Text("•")
                Text("seed: \(entry.seed)")
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}
```

- [ ] **Step 2: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Components/HistoryListView.swift
git commit -m "feat: add HistoryListView with swipe-to-delete and tap-to-play"
```

---

### Task 12: GenerateView

**Files:**
- Create: `AnvilApp/UI/Components/GenerateView.swift`

- [ ] **Step 1: Implement GenerateView**

Create `AnvilApp/UI/Components/GenerateView.swift`:

```swift
import SwiftUI
import AnvilCore

struct GenerateView: View {
    @EnvironmentObject private var generationService: GenerationService
    @EnvironmentObject private var playerService: AudioPlayerService

    @State private var promptText: String = ""
    @State private var lyricsText: String = ""
    @State private var duration: Double = 30
    @State private var seed: String = ""
    @State private var guidanceScale: Double = 1.0
    @State private var method: InferMethod = .ode
    @State private var precision: Precision = .float16

    var body: some View {
        VStack(spacing: 16) {
            promptSection
            lyricsSection
            controlsBar
            progressSection
            playerSection
        }
        .padding()
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Prompt")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: $promptText)
                .font(.body)
                .frame(minHeight: 60, maxHeight: 100)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
        }
    }

    private var lyricsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Lyrics")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: $lyricsText)
                .font(.body)
                .frame(minHeight: 80, maxHeight: 120)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
        }
    }

    private var controlsBar: some View {
        HStack {
            Button("Generate") {
                let request = GenerationRequest(
                    text: promptText,
                    lyrics: lyricsText,
                    duration: duration,
                    seed: Int(seed),
                    method: method,
                    guidanceScale: Float(guidanceScale),
                    precision: precision
                )
                generationService.generate(request: request)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canGenerate)

            if case .generating = generationService.state {
                Button("Cancel", role: .destructive) {
                    generationService.cancel()
                }
                .buttonStyle(.bordered)
            }

            Spacer()

            if let eta = estimatedTime {
                Text("~\(Int(eta))s remaining")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var progressSection: some View {
        switch generationService.state {
        case .idle, .complete, .cancelled:
            EmptyView()
        case .loadingWeights(let progress):
            ProgressView("Loading weights...", value: Double(progress), total: 1.0)
        case .conditioning:
            ProgressView("Encoding text and lyrics...")
                .progressViewStyle(.linear)
        case .generating(let step, let total, _):
            VStack(spacing: 4) {
                ProgressView(value: Double(step), total: Double(total))
                Text("Step \(step) / \(total)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .decoding:
            ProgressView("Decoding audio...")
                .progressViewStyle(.linear)
        case .error(let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
        }
    }

    @ViewBuilder
    private var playerSection: some View {
        if let result = generationService.currentResult {
            PlayerView(fileURL: result.fileURL)
        }
    }

    private var canGenerate: Bool {
        let isIdle = generationService.state == .idle
            || generationService.state == .complete
            || generationService.state == .cancelled
        return isIdle && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var estimatedTime: TimeInterval? {
        if case .generating(_, _, let eta) = generationService.state, eta > 0 {
            return eta
        }
        return nil
    }
}
```

- [ ] **Step 2: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 3: Commit**

```bash
git add AnvilApp/UI/Components/GenerateView.swift
git commit -m "feat: add GenerateView with prompt/lyrics editors, progress, and player"
```

---

### Task 13: Wire MainShellView to Services

**Files:**
- Modify: `AnvilApp/UI/MainShellView.swift`
- Modify: `AnvilApp/AnvilAppApp.swift`

- [ ] **Step 1: Rewrite MainShellView to use new views and services**

Replace the contents of `AnvilApp/UI/MainShellView.swift`:

```swift
import SwiftUI

struct MainShellView: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @State private var selectedSidebarItem: SidebarItem = .generate

    var body: some View {
        if !outputManager.hasValidDirectory() {
            OutputDirectoryPicker()
        } else {
            NavigationSplitView {
                sidebar
            } content: {
                canvas
            } detail: {
                inspector
            }
            .navigationSplitViewStyle(.balanced)
        }
    }

    private var sidebar: some View {
        List(selection: $selectedSidebarItem) {
            Section("Create") {
                Label("Generate", systemImage: "waveform")
                    .tag(SidebarItem.generate)
            }
            Section("Library") {
                Label("History", systemImage: "clock")
                    .tag(SidebarItem.history)
                Label("Presets", systemImage: "slider.horizontal.3")
                    .tag(SidebarItem.presets)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Anvil")
    }

    @ViewBuilder
    private var canvas: some View {
        switch selectedSidebarItem {
        case .generate:
            GenerateView()
        case .history:
            HistoryListView()
        case .presets:
            Text("Presets coming in Phase 4b")
                .foregroundStyle(.secondary)
        }
    }

    private var inspector: some View {
        Form {
            Section("Parameters") {
                LabeledContent("Duration") { TextField("30s", text: .constant("")) }
                LabeledContent("Seed") { TextField("Random", text: .constant("")) }
                LabeledContent("Guidance") { TextField("1.0", text: .constant("")) }
            }
            Section("Model") {
                Text("ACE-Step DiT v1.5")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 250)
    }
}

enum SidebarItem: Hashable {
    case generate
    case history
    case presets
}
```

- [ ] **Step 2: Update AnvilAppApp to inject services into environment**

Read `AnvilApp/AnvilAppApp.swift` first, then update it to create and inject all services:

```swift
import SwiftUI
import AnvilCore

@main
struct AnvilAppApp: App {
    @StateObject private var appState = AppState()
    @StateObject private var outputManager = OutputDirectoryManager()
    @StateObject private var playerService = AudioPlayerService()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .environmentObject(outputManager)
                .environmentObject(playerService)
                .environmentObject(historyStore)
                .environmentObject(generationService)
        }
        .defaultSize(width: 1200, height: 800)

        Settings {
            Text("Settings coming soon")
                .padding()
        }
    }

    private var historyStore: HistoryStore {
        HistoryStore(directory: outputManager.directory ?? FileManager.default.temporaryDirectory)
    }

    private var generationService: GenerationService {
        let vae = VAEActor()
        let dit = DiTActor(decoder: DiTDecoder())
        let diffusion = DiffusionActor(dit: dit)
        let pipeline = MusicPipeline(vae: vae, diffusion: diffusion)
        let conditioning = ConditioningActor.forTesting(
            tokenizerURL: Bundle.main.url(forResource: "tokenizer", withExtension: "json")
                ?? FileManager.default.temporaryDirectory.appending(path: "tokenizer.json")
        )
        let converter = WeightConverter()
        return GenerationService(
            pipeline: pipeline,
            conditioning: conditioning,
            weightConverter: converter,
            historyStore: historyStore,
            outputDirectoryManager: outputManager
        )
    }
}

struct ContentView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        if appState.isSetupComplete {
            MainShellView()
        } else {
            FirstLaunchView()
        }
    }
}
```

**Note:** The `generationService` construction above uses placeholder initialization. The actual weight loading and model initialization will be refined when real model files are available. The key thing is the dependency injection wiring is correct.

- [ ] **Step 3: Verify Xcode build succeeds**

Run: `xcodebuild -project AnvilApp.xcodeproj -scheme AnvilApp build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add AnvilApp/UI/MainShellView.swift AnvilApp/AnvilAppApp.swift
git commit -m "feat: wire MainShellView to GenerationService, HistoryStore, and PlayerService"
```

---

## Summary

| Task | Component | Key Output |
|------|-----------|------------|
| 1 | encoderMask wiring | Mask flows from ConditionEncoder → DiT cross-attention |
| 2 | Cancellation support | DiffusionActor.generate throws on Task cancel |
| 3 | WAVWriter | PCM 16-bit stereo WAV export from MLXArray |
| 4 | Model types | GenerationRequest, GenerationState, HistoryEntry |
| 5 | OutputDirectoryManager | Security-scoped bookmark persistence |
| 6 | HistoryStore | JSON-backed persistent generation history |
| 7 | AudioPlayerService | AVAudioPlayer wrapper with waveform computation |
| 8 | GenerationService | Full lifecycle coordinator |
| 9 | OutputDirectoryPicker | First-launch folder selection UI |
| 10 | WaveformView + PlayerView | Inline audio playback with scrubbing |
| 11 | HistoryListView | Browsable history with delete |
| 12 | GenerateView | Prompt/lyrics editors + progress + player |
| 13 | MainShellView wiring | Connect everything via environment objects |
