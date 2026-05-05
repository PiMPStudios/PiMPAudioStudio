# Phase 4a: Generation Flow — Design Spec

## Overview

Wire the existing inference pipeline into a functional end-to-end generation flow
with UI. This phase delivers: encoderMask plumbing through the diffusion stack,
a `GenerationService` coordinating inference, an inline audio player, persistent
generation history, and the UI connections to drive it all from MainShellView.

**Scope boundary:** Phase 4a does NOT include IntentRouter, LyricWriter, prompt
enhancement, or any intelligence layer. It delivers a working "type prompt → hear
audio" experience with progress feedback, cancel support, and history.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | GenerationService as middle layer | Keeps AppState focused on app-level state; testable in isolation |
| Audio output | Play inline + export WAV | Best UX for iterative creative work |
| Export format | WAV default (others in post-editing phases) | Zero-effort from raw samples; lossless |
| Progress UI | Progress bar + ETA + Cancel | Users need to know how long and ability to abort |
| Output directory | User picks on first launch | Respectful; no hidden sandbox folders |
| Audio player | Play/pause, waveform scrub, volume | Enough to evaluate output without over-building |
| History | Persistent JSON, individual + bulk delete | Creative tools need session continuity |
| File naming | `anvil_YYYYMMDD_HHmmss_<words>.wav` | Human-readable in Finder without the app |

## Component 1: encoderMask Wiring

Thread the existing `encoderMask` from `ConditioningResult` through the pipeline
to `DiTDecoder`'s cross-attention layers. No new logic — pure parameter plumbing.

### Changes Required

```text
MusicPipeline.generate(encoderStates:contextLatents:duration:config:)
  → add encoderMask: MLXArray? parameter
  → pass to DiffusionActor.generate()

DiffusionActor.generate(encoderStates:contextLatents:shape:config:)
  → add encoderMask: MLXArray? parameter
  → pass to DiTActor.step() on each diffusion step

DiTActor.step(noisyLatents:timestep:timestepR:encoderStates:contextLatents:cache:useCache:)
  → add encoderMask: MLXArray? parameter
  → pass to DiTDecoder.callAsFunction()

DiTDecoder.callAsFunction(...)
  → replace hardcoded `encoderAttentionMask: nil` with the passed mask
```

The convenience `MusicPipeline.generate(text:lyrics:...)` method passes
`cond.encoderMask` through automatically.

### Mask Format

The mask from `ConditionEncoder` is shape `[B, T_combined]` with values:
- `1` for real tokens (attend)
- `0` for padding tokens (ignore)

`MLXFast.scaledDotProductAttention` expects the mask in additive format
(`0` = attend, `-inf` = ignore). The conversion happens at the DiTDecoder level
before passing to layers:

```swift
let additiveMask = encoderMask.map { mask in
    (1.0 - mask) * Float(-1e9)  // [B, T] → expand to [B, 1, 1, T] for broadcast
}
```

## Component 2: GenerationService

`@MainActor` ObservableObject that coordinates the full generation lifecycle.

### Responsibilities

1. Validate inputs (non-empty prompt, valid duration)
2. Load weights if not already loaded (lazy, on first generate)
3. Run ConditioningActor.encode → MusicPipeline.generate pipeline
4. Publish progress updates from DiffusionActor's AsyncStream
5. Handle cancellation via Swift Task cancellation
6. Save output WAV to user directory
7. Update HistoryStore on completion

### State

```swift
@MainActor
public class GenerationService: ObservableObject {
    @Published public var state: GenerationState = .idle
    @Published public var currentResult: GenerationResult?

    private var generationTask: Task<Void, Never>?
    private let pipeline: MusicPipeline
    private let conditioning: ConditioningActor
    private let historyStore: HistoryStore
    private var outputDirectory: URL?
}

public enum GenerationState: Equatable, Sendable {
    case idle
    case loadingWeights(progress: Float)
    case conditioning
    case generating(step: Int, totalSteps: Int, eta: TimeInterval)
    case decoding
    case complete
    case error(String)
    case cancelled
}

public struct GenerationResult: Sendable {
    public let fileURL: URL
    public let entry: HistoryEntry
}
```

### API

```swift
public func generate(request: GenerationRequest) async
public func cancel()
public func setOutputDirectory(_ url: URL)
```

### GenerationRequest

```swift
public struct GenerationRequest: Sendable {
    public let text: String
    public let lyrics: String
    public let referenceAudioURL: URL?
    public let duration: TimeInterval
    public let seed: Int?
    public let method: InferMethod
    public let guidanceScale: Float
    public let precision: Precision

    public init(
        text: String,
        lyrics: String = "",
        referenceAudioURL: URL? = nil,
        duration: TimeInterval = 30.0,
        seed: Int? = nil,
        method: InferMethod = .ode,
        guidanceScale: Float = 1.0,
        precision: Precision = .float16
    )
}
```

### ETA Calculation

Track elapsed time per diffusion step. ETA = average step time * remaining steps.
Update each step for accuracy as generation proceeds.

### Cancellation

```swift
public func cancel() {
    generationTask?.cancel()
    state = .cancelled
}
```

DiffusionActor checks `Task.isCancelled` between steps. This changes
`DiffusionActor.generate` from `async -> DiffusionResult` to
`async throws -> DiffusionResult`:

```swift
for step in 0..<totalSteps {
    try Task.checkCancellation()
    // ... run step ...
}
```

### Weight Loading

Weights load lazily on first generate call. Progress is reported through
`.loadingWeights(progress:)` state. Once loaded, subsequent generates skip this
phase. Weights stay resident until the app quits (no unloading for Phase 4a).

## Component 3: AudioPlayerService

`@MainActor` ObservableObject wrapping `AVAudioPlayer` for inline playback.

### API

```swift
@MainActor
public class AudioPlayerService: ObservableObject {
    @Published public var isPlaying: Bool = false
    @Published public var currentTime: TimeInterval = 0
    @Published public var duration: TimeInterval = 0
    @Published public var volume: Float = 0.7

    public func load(url: URL)
    public func play()
    public func pause()
    public func togglePlayback()
    public func seek(to time: TimeInterval)
    public func setVolume(_ volume: Float)
}
```

### Waveform Data

On `load(url:)`, compute a waveform thumbnail:

1. Read WAV file samples (raw PCM Int16 or Float32)
2. Divide into N buckets (N = display width, ~400 points)
3. For each bucket, compute peak amplitude (max absolute value)
4. Store as `[Float]` array normalized to 0.0–1.0

```swift
public struct WaveformData {
    public let peaks: [Float]  // normalized 0.0–1.0
    public let sampleRate: Int
    public let channelCount: Int
}

@Published public var waveform: WaveformData?
```

### Playhead Updates

Use a `CADisplayLink` (or `Timer` at 30Hz) while playing to update
`currentTime`. Stop the timer on pause/stop.

## Component 4: HistoryStore

`@MainActor` ObservableObject backed by `history.json` in the output directory.

### Data Model

```swift
public struct HistoryEntry: Identifiable, Codable, Sendable {
    public let id: UUID
    public let createdAt: Date
    public let prompt: String
    public let lyrics: String
    public let duration: TimeInterval
    public let seed: Int
    public let method: String
    public let guidanceScale: Float
    public let precision: String
    public let fileName: String  // relative to output directory
}
```

### API

```swift
@MainActor
public class HistoryStore: ObservableObject {
    @Published public var entries: [HistoryEntry] = []

    public init(directory: URL)
    public func add(_ entry: HistoryEntry)
    public func delete(_ entry: HistoryEntry)
    public func deleteAll()
    public func fileURL(for entry: HistoryEntry) -> URL
}
```

### Storage Format

`history.json` — JSON array of `HistoryEntry`. Written atomically on each
mutation. Located in the user's chosen output directory alongside WAV files.

### File Naming Convention

```text
anvil_20260505_143022_a-dreamy-synthwave-track.wav
```

Format: `anvil_<date>_<time>_<first-5-words-slugified>.wav`

Slug rules: lowercase, spaces to hyphens, strip non-alphanumeric, truncate to
50 chars max.

### Deletion

- `delete(_:)` — removes entry from JSON, deletes WAV file from disk
- `deleteAll()` — removes all entries and their WAV files

## Component 5: WAV Export

Utility to write raw `MLXArray` audio samples to a WAV file.

### API

```swift
public struct WAVWriter {
    public static func write(
        samples: MLXArray,
        sampleRate: Int = 44100,
        to url: URL
    ) throws
}
```

### Format

- PCM 16-bit signed integer (standard WAV)
- 2 channels (stereo), interleaved
- 44.1 kHz sample rate
- Standard RIFF/WAV header (44 bytes)

### Conversion

MLXArray samples are Float32 in range [-1.0, 1.0]. Convert to Int16:

```swift
let int16Samples = (samples.clipped(min: -1.0, max: 1.0) * 32767.0).asType(.int16)
```

## Component 6: UI Views

### Modified Views

**MainShellView** — Wire sidebar selection to show History or Generate views.
Connect Generate button to `GenerationService.generate(request:)`.

**Canvas area** — Shows either:
- Generate form + progress (during/before generation)
- Audio player (after generation completes)
- Both: form stays visible, player appears below it

### New Views

**GenerateView** — The generate form (prompt, lyrics text editors) + Generate
button + progress indicator + cancel button. Reads parameters from inspector
panel fields.

**PlayerView** — Waveform display with playhead, play/pause button, time labels,
volume slider, Export WAV button, Reveal in Finder button.

**HistoryListView** — Scrollable list of `HistoryEntry` items. Each row shows:
prompt (truncated), duration, date, seed. Tap to load into player. Swipe to
delete individual. "Delete All" button in header.

**OutputDirectoryPicker** — Shown on first launch (or if directory is invalid).
Uses `NSOpenPanel` with directory selection mode. Stores URL as a security-scoped
bookmark in UserDefaults for persistence across launches.

### View State Flow

```text
User taps Generate
  → GenerateView reads form fields → builds GenerationRequest
  → calls generationService.generate(request:)
  → GenerateView observes generationService.state
  → shows progress bar / ETA / Cancel
  → on .complete: playerService.load(url:), show PlayerView
  → historyStore auto-updated by GenerationService
```

## Component 7: Output Directory & Security-Scoped Bookmarks

macOS sandboxed apps lose file access on relaunch. Use security-scoped bookmarks
to persist access to the user-chosen output directory.

### Flow

1. First launch (or missing bookmark): show `OutputDirectoryPicker`
2. User selects directory via `NSOpenPanel`
3. Create security-scoped bookmark from URL
4. Store bookmark data in UserDefaults
5. On subsequent launches: resolve bookmark → start accessing security-scoped
   resource

### API

```swift
public class OutputDirectoryManager: ObservableObject {
    @Published public var directory: URL?

    public func pickDirectory() async -> URL?
    public func resolveBookmark() -> URL?
    public func hasValidDirectory() -> Bool
}
```

## Package Structure

```text
AnvilApp/
├── Services/
│   ├── GenerationService.swift
│   ├── AudioPlayerService.swift
│   ├── HistoryStore.swift
│   └── OutputDirectoryManager.swift
├── UI/
│   ├── MainShellView.swift          (modified)
│   ├── Components/
│   │   ├── GenerateView.swift       (new)
│   │   ├── PlayerView.swift         (new)
│   │   ├── WaveformView.swift       (new)
│   │   ├── HistoryListView.swift    (new)
│   │   └── OutputDirectoryPicker.swift (new)

AnvilCore/Sources/AnvilCore/Inference/
├── Pipeline/
│   └── MusicPipeline.swift          (modified — encoderMask param)
├── Diffusion/
│   └── DiffusionActor.swift         (modified — encoderMask param, cancellation)
├── DiT/
│   ├── DiTActor.swift               (modified — encoderMask param)
│   └── DiTModel.swift               (modified — use mask instead of nil)
├── Audio/
│   └── WAVWriter.swift              (new)
```

## Testing Strategy

1. **encoderMask wiring** — Unit test: pass a non-nil mask through
   MusicPipeline, verify DiTModel receives it (mock-based or shape assertion).
2. **GenerationService** — Test state transitions: idle → conditioning →
   generating → decoding → complete. Test cancel mid-generation.
3. **WAVWriter** — Write known samples, read back WAV file, verify header and
   PCM data match.
4. **HistoryStore** — Add entries, verify JSON persistence, delete and verify
   file removal.
5. **AudioPlayerService** — Load a WAV file, verify duration and waveform data
   computation.
6. **WaveformData** — Feed known audio samples, verify peak computation.
7. **OutputDirectoryManager** — Test bookmark creation and resolution.

## Dependencies

- **AVFoundation** (existing, for audio playback)
- **UniformTypeIdentifiers** (for file type handling)
- No new external packages

## Non-Goals (Phase 4b / Later)

- IntentRouter (Phase 4b)
- LyricWriter / prompt enhancement (Phase 4b)
- OGG/MP3/FLAC export (post-editing phases)
- A/B comparison playback (Phase 5+)
- Loop playback (Phase 5+)
- Audio editor integration (Phase 5)
