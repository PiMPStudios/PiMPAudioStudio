import AVFoundation
import PiMPAudioCore
import SwiftUI

/// Identifies a finished DSP apply. The token makes two consecutive applies of the
/// same effect distinguishable so the UI can react to each one.
struct CompletedDSPApply: Equatable {
    let effect: DSPEffect
    let token: Int
}

@MainActor
class EditorState: ObservableObject {
    @Published var editMode: EditMode = .inpaint
    @Published var isLoaded = false
    @Published var fileName: String = ""
    @Published var duration: TimeInterval = 0
    @Published var sampleRate: Int = 44100

    @Published var selectionStart: TimeInterval = 0
    @Published var selectionEnd: TimeInterval = 0
    @Published var frequencyBandEnabled = false
    @Published var lowCutHz: Float = 200
    @Published var highCutHz: Float = 4000

    /// Shared slider span for frequency-band cuts.
    static let frequencyCutRange: ClosedRange<Float> = 20 ... 20000
    /// Minimum non-empty bandwidth enforced by the paired low/high bindings.
    static let minimumFrequencyBandwidthHz: Float = 50

    @Published var originalPrompt: String = ""
    @Published var regionPrompt: String = ""

    @Published var selectedPreset: InpaintPreset? = .moderate
    @Published var strength: Float = 0.6
    @Published var guidanceScale: Float = 7.0
    @Published var pipeline: PipelineChoice = .auto

    @Published var progress: InpaintProgress?
    @Published var isProcessing = false
    /// True while a DSP effect is running off the main actor.
    @Published var isDSPBusy = false
    @Published var operationErrorMessage: String?

    private var dspTask: Task<Void, Never>?

    @Published var canUndo = false
    @Published var canRedo = false
    @Published var recentEdits: [EditSnapshot] = []

    @Published var waveformPeaks: [Float] = []
    @Published var playbackProgress: Double = 0

    // MARK: - DSP State

    @Published var expandedEffect: DSPEffect?
    @Published var eqParameters: EQParameters = .default
    @Published var reverbParameters: ReverbParameters = .default
    @Published var selectedReverbPreset: ReverbPreset? = .hall
    @Published var audioBuffer: AudioBuffer?
    @Published var timeStretchParameters: TimeStretchParameters = .default
    @Published var pitchShiftParameters: PitchShiftParameters = .default
    @Published var normalizeParameters: NormalizeParameters = .default
    @Published var trimSilenceParameters: TrimSilenceParameters = .default
    @Published var highPassParameters: HighPassParameters = .default
    @Published var compressorParameters: CompressorParameters = .default
    @Published var limiterParameters: LimiterParameters = .default
    @Published var loudnessParameters: LoudnessParameters = .default
    @Published var polishRecipe: PolishForKeepRecipe = .default
    @Published var softGateParameters: SoftGateParameters = .default
    /// Brief confirmation after one-shot polish applies.
    @Published var polishAppliedConfirmation = false
    /// Set when a DSP apply actually finishes, so the palette confirms real results
    /// instead of flashing a checkmark the moment the button is pressed.
    @Published private(set) var lastCompletedApply: CompletedDSPApply?
    /// When true, editor preview playback folds stereo to mono (does not edit the buffer).
    @Published var previewInMono = false {
        didSet {
            if oldValue != previewInMono {
                invalidatePreviewPlayer()
            }
        }
    }

    /// Cached level readout; refreshed when the loaded buffer changes.
    @Published private(set) var levelMetrics: LoudnessMeter.Metrics?

    /// Bumps whenever the working PCM buffer changes from load / DSP / undo /
    /// AI apply. EditorView uses this to know when to re-sync the coordinator
    /// without re-encoding on every Inpaint of an unchanged buffer.
    private(set) var bufferRevision: Int = 0
    /// Bumps when a new document is adopted so in-flight DSP/AI/load work can
    /// drop stale results instead of overwriting the newly opened file.
    private(set) var documentGeneration: Int = 0

    private let dspProcessor = DSPProcessor()

    func noteBufferMutation() {
        bufferRevision &+= 1
    }

    // MARK: - Preview playback

    @Published var isPlayingPreview = false
    @Published var previewCurrentTime: TimeInterval = 0
    private var previewPlayer: AVAudioPlayer?
    private var previewTimer: Timer?
    private var previewFileURL: URL?
    private var peakComputationTask: Task<Void, Never>?
    private var metricsTask: Task<Void, Never>?
    /// Distinguishes consecutive applies of the same effect for `lastCompletedApply`.
    private var applyToken = 0

    // DSP undo/redo — stores raw AudioBuffer snapshots.
    // AI inpainting also pushes here (via saveUndoSnapshot) so both paths
    // share a single undo button.
    private var undoStack: [AudioBuffer] = []
    private var redoStack: [AudioBuffer] = []
    private let maxHistorySnapshots = 10
    private let maxHistoryBytes = 512 * 1024 * 1024

    deinit {
        previewTimer?.invalidate()
        peakComputationTask?.cancel()
        metricsTask?.cancel()
        if let previewFileURL {
            try? FileManager.default.removeItem(at: previewFileURL)
        }
    }

    /// Hop to the *next* main runloop turn before mutating `@Published` state.
    ///
    /// `Task { @MainActor }` / `MainActor.run` can still execute on the current
    /// tick while SwiftUI is evaluating body, which triggers:
    /// "Publishing changes from within view updates is not allowed".
    /// `DispatchQueue.main.async` always defers (same pattern as AppState download progress).
    func deferUIUpdate(_ body: @escaping @MainActor (EditorState) -> Void) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            body(self)
        }
    }

    /// Recompute peak / true-peak / LUFS off the main actor when the buffer changes.
    func refreshLevelMetrics() {
        metricsTask?.cancel()
        guard let buffer = audioBuffer else {
            deferUIUpdate { $0.levelMetrics = nil }
            return
        }
        metricsTask = Task { [weak self] in
            let metrics = await Task.detached(priority: .utility) {
                LoudnessMeter.measure(buffer)
            }.value
            guard !Task.isCancelled else { return }
            self?.deferUIUpdate { state in
                state.levelMetrics = metrics
            }
        }
    }

    var hasSelection: Bool {
        selectionEnd > selectionStart
    }

    func applyPreset(_ preset: InpaintPreset) {
        selectedPreset = preset
        strength = preset.strength
        guidanceScale = preset.guidanceScale
    }

    func clearPresetIfManual() {
        selectedPreset = nil
    }

    var actionButtonTitle: String {
        switch editMode {
        case .inpaint: "Inpaint Selection"
        case .repaint: "Repaint Track"
        }
    }

    var regionPromptLabel: String {
        switch editMode {
        case .inpaint: "Region Prompt"
        case .repaint: "Target Style"
        }
    }

    /// Paired binding: raising low cut never crosses or meets high cut.
    var lowCutHzBinding: Binding<Float> {
        Binding(
            get: { self.lowCutHz },
            set: { self.setLowCutHz($0) }
        )
    }

    /// Paired binding: lowering high cut never crosses or meets low cut.
    var highCutHzBinding: Binding<Float> {
        Binding(
            get: { self.highCutHz },
            set: { self.setHighCutHz($0) }
        )
    }

    func setLowCutHz(_ value: Float) {
        let maxLow = Self.frequencyCutRange.upperBound - Self.minimumFrequencyBandwidthHz
        let clamped = min(max(value, Self.frequencyCutRange.lowerBound), maxLow)
        lowCutHz = clamped
        if highCutHz <= lowCutHz {
            highCutHz = min(lowCutHz + Self.minimumFrequencyBandwidthHz, Self.frequencyCutRange.upperBound)
        }
    }

    func setHighCutHz(_ value: Float) {
        let minHigh = Self.frequencyCutRange.lowerBound + Self.minimumFrequencyBandwidthHz
        let clamped = min(max(value, minHigh), Self.frequencyCutRange.upperBound)
        highCutHz = clamped
        if lowCutHz >= highCutHz {
            lowCutHz = max(highCutHz - Self.minimumFrequencyBandwidthHz, Self.frequencyCutRange.lowerBound)
        }
    }

    func buildRequest() throws -> InpaintRequest {
        let selection: InpaintSelection
        if editMode == .repaint {
            selection = InpaintSelection(startTime: 0, endTime: duration)
        } else {
            let band = frequencyBandEnabled
                ? try FrequencyBand(lowCutHz: lowCutHz, highCutHz: highCutHz)
                : nil
            selection = InpaintSelection(
                startTime: selectionStart,
                endTime: selectionEnd,
                frequencyBand: band
            )
        }

        return InpaintRequest(
            editMode: editMode,
            selection: selection,
            regionPrompt: regionPrompt,
            originalPrompt: originalPrompt.isEmpty ? nil : originalPrompt,
            strength: strength,
            guidanceScale: guidanceScale,
            pipeline: pipeline,
            seed: nil
        )
    }

    // MARK: - DSP Operations

    /// One-shot keep polish (trim → HPF → tone → glue → space → loudness/limit) as a single undo step.
    /// - Parameter recipe: Optional override (e.g. Polish panel local draft). Defaults to `polishRecipe`.
    func applyPolishForKeep(_ recipe: PolishForKeepRecipe? = nil) {
        guard let buffer = audioBuffer else { return }
        guard !isDSPBusy, !isProcessing else { return }

        let resolved = recipe ?? polishRecipe
        polishRecipe = resolved
        dspTask?.cancel()
        isDSPBusy = true
        operationErrorMessage = nil
        polishAppliedConfirmation = false

        let sourceRevision = bufferRevision
        dspTask = Task { [weak self] in
            defer {
                Task { @MainActor [weak self] in
                    self?.isDSPBusy = false
                }
            }

            do {
                let result = try await Task.detached(priority: .userInitiated) {
                    try PolishForKeepProcessor.apply(buffer, recipe: resolved)
                }.value

                guard !Task.isCancelled else { return }

                await MainActor.run { [weak self] in
                    guard let self else { return }
                    guard shouldCommitProcessedBuffer(sourceRevision: sourceRevision) else { return }
                    pushUndoSnapshot(buffer)
                    redoStack.removeAll()
                    canUndo = true
                    canRedo = false

                    invalidatePreviewPlayer()
                    audioBuffer = result
                    duration = result.duration
                    noteBufferMutation()
                    refreshWaveformPeaks(from: result)
                    refreshLevelMetrics()
                    selectionStart = 0
                    selectionEnd = 0
                    polishAppliedConfirmation = true
                }

                try? await Task.sleep(for: .milliseconds(800))
                await MainActor.run { [weak self] in
                    self?.polishAppliedConfirmation = false
                }
            } catch is CancellationError {
                return
            } catch {
                await MainActor.run { [weak self] in
                    self?.operationErrorMessage = Self.errorMessage(for: error)
                }
            }
        }
    }

    /// Applies a DSP effect off the main actor so long tracks do not freeze the UI.
    func applyDSPEffect(_ effect: DSPEffect, parameters: DSPParameters = .none) {
        guard let buffer = audioBuffer else { return }
        guard !isDSPBusy, !isProcessing else { return }

        let target: ApplyTarget = hasSelection
            ? .selection(start: selectionStart, end: selectionEnd)
            : .wholeTrack
        let processor = dspProcessor

        dspTask?.cancel()
        isDSPBusy = true
        operationErrorMessage = nil

        let sourceRevision = bufferRevision
        dspTask = Task { [weak self] in
            defer {
                Task { @MainActor [weak self] in
                    self?.isDSPBusy = false
                }
            }

            do {
                let result = try await Task.detached(priority: .userInitiated) {
                    try processor.apply(
                        effect: effect,
                        to: buffer,
                        target: target,
                        parameters: parameters
                    )
                }.value

                guard !Task.isCancelled else { return }

                await MainActor.run { [weak self] in
                    guard let self else { return }
                    guard shouldCommitProcessedBuffer(sourceRevision: sourceRevision) else { return }
                    // Commit: save pre-op buffer so it can be restored by undo.
                    pushUndoSnapshot(buffer)
                    redoStack.removeAll()
                    canUndo = true
                    canRedo = false

                    invalidatePreviewPlayer()
                    audioBuffer = result
                    duration = result.duration
                    noteBufferMutation()
                    refreshWaveformPeaks(from: result)
                    refreshLevelMetrics()

                    if effect == .trimToSelection
                        || effect == .deleteSelection
                        || effect == .timeStretch
                        || effect == .trimSilence {
                        selectionStart = 0
                        selectionEnd = 0
                    }

                    applyToken += 1
                    lastCompletedApply = CompletedDSPApply(
                        effect: effect,
                        token: applyToken
                    )
                }
            } catch is CancellationError {
                return
            } catch {
                await MainActor.run { [weak self] in
                    self?.operationErrorMessage = Self.errorMessage(for: error)
                }
            }
        }
    }

    // MARK: - Undo / Redo

    /// Push the current buffer onto the undo stack without applying any effect.
    /// Call this before starting an AI inpaint so the result is also undoable.
    func saveUndoSnapshot() {
        guard let buffer = audioBuffer else { return }
        pushUndoSnapshot(buffer)
        redoStack.removeAll()
        canUndo = true
        canRedo = false
    }

    func undoDSP() {
        guard let previous = undoStack.popLast() else { return }
        if let current = audioBuffer {
            pushRedoSnapshot(current)
        }
        invalidatePreviewPlayer()
        audioBuffer = previous
        duration = previous.duration
        noteBufferMutation()
        refreshWaveformPeaks(from: previous)
        refreshLevelMetrics()
        canUndo = !undoStack.isEmpty
        canRedo = !redoStack.isEmpty
    }

    func redoDSP() {
        guard let next = redoStack.popLast() else { return }
        if let current = audioBuffer {
            pushUndoSnapshot(current)
        }
        invalidatePreviewPlayer()
        audioBuffer = next
        duration = next.duration
        noteBufferMutation()
        refreshWaveformPeaks(from: next)
        refreshLevelMetrics()
        canUndo = !undoStack.isEmpty
        canRedo = !redoStack.isEmpty
    }

    /// Wipe undo/redo history — call when a new file is loaded.
    func resetUndoHistory() {
        undoStack.removeAll()
        redoStack.removeAll()
        canUndo = false
        canRedo = false
    }

    private func pushUndoSnapshot(_ buffer: AudioBuffer) {
        undoStack.append(buffer)
        trimHistoryStack(&undoStack)
    }

    private func pushRedoSnapshot(_ buffer: AudioBuffer) {
        redoStack.append(buffer)
        trimHistoryStack(&redoStack)
    }

    private func trimHistoryStack(_ stack: inout [AudioBuffer]) {
        while stack.count > maxHistorySnapshots {
            stack.removeFirst()
        }

        while historyByteCount(stack) > maxHistoryBytes, stack.count > 1 {
            stack.removeFirst()
        }
    }

    private func historyByteCount(_ stack: [AudioBuffer]) -> Int {
        stack.reduce(0) { $0 + $1.data.count }
    }

    // MARK: - Preview playback

    func playPreview() {
        ensurePreviewPlayer()
        previewPlayer?.play()
        isPlayingPreview = true
        startPreviewTimer()
    }

    func pausePreview() {
        previewPlayer?.pause()
        isPlayingPreview = false
        stopPreviewTimer()
    }

    func seekPreview(to fraction: Double) {
        ensurePreviewPlayer()
        guard let player = previewPlayer else { return }
        let clampedFraction = max(0, min(1, fraction))
        let t = clampedFraction * player.duration
        player.currentTime = t
        previewCurrentTime = t
        playbackProgress = clampedFraction
    }

    private func ensurePreviewPlayer() {
        guard previewPlayer == nil, let buffer = audioBuffer else { return }
        do {
            let playbackBuffer = previewInMono && buffer.channels >= 2
                ? LoudnessMeter.foldToMono(buffer)
                : buffer
            let fileURL = try makePreviewWAVFile(from: playbackBuffer)
            previewFileURL = fileURL
            previewPlayer = try AVAudioPlayer(contentsOf: fileURL, fileTypeHint: AVFileType.wav.rawValue)
            previewPlayer?.prepareToPlay()
        } catch {
            operationErrorMessage = "Could not prepare preview playback: \(Self.errorMessage(for: error))"
        }
    }

    /// Nil out the player whenever the buffer changes — playPreview reloads lazily.
    private func invalidatePreviewPlayer() {
        previewPlayer?.stop()
        previewPlayer = nil
        removePreviewFile()
        isPlayingPreview = false
        previewCurrentTime = 0
        playbackProgress = 0
        stopPreviewTimer()
    }

    private func startPreviewTimer() {
        previewTimer?.invalidate()
        previewTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let player = previewPlayer else { return }
                previewCurrentTime = player.currentTime
                playbackProgress = player.duration > 0
                    ? player.currentTime / player.duration : 0
                if !player.isPlaying, isPlayingPreview {
                    isPlayingPreview = false
                    previewCurrentTime = 0
                    playbackProgress = 0
                    stopPreviewTimer()
                }
            }
        }
    }

    private func stopPreviewTimer() {
        previewTimer?.invalidate()
        previewTimer = nil
    }

    private func removePreviewFile() {
        if let previewFileURL {
            try? FileManager.default.removeItem(at: previewFileURL)
        }
        previewFileURL = nil
    }

    /// Write a temporary preview WAV without building a second full-size Data blob.
    private func makePreviewWAVFile(from buffer: AudioBuffer) throws -> URL {
        let pcm = buffer.data
        guard buffer.sampleRate > 0 else { throw PreviewError.invalidBuffer("sample rate is zero") }
        guard buffer.channels > 0 else { throw PreviewError.invalidBuffer("channel count is zero") }
        guard pcm.count % (buffer.channels * MemoryLayout<Int16>.size) == 0 else {
            throw PreviewError.invalidBuffer("PCM data is not aligned to the channel count")
        }
        guard pcm.count <= Int(UInt32.max - 36) else {
            throw PreviewError.invalidBuffer("audio is too large for a standard WAV preview")
        }

        let fileURL = FileManager.default.temporaryDirectory
            .appending(path: "pimpaudio-editor-preview-\(UUID().uuidString).wav")
        let header = Self.makeWAVHeader(
            dataSize: UInt32(pcm.count),
            sampleRate: UInt32(buffer.sampleRate),
            channels: UInt16(buffer.channels)
        )

        _ = FileManager.default.createFile(atPath: fileURL.path(percentEncoded: false), contents: nil)
        do {
            let handle = try FileHandle(forWritingTo: fileURL)
            defer { try? handle.close() }
            try handle.write(contentsOf: header)
            try handle.write(contentsOf: pcm)
            return fileURL
        } catch {
            try? FileManager.default.removeItem(at: fileURL)
            throw error
        }
    }

    private static func makeWAVHeader(dataSize: UInt32, sampleRate: UInt32, channels: UInt16) -> Data {
        let sr = sampleRate
        let ch = channels
        let bps = UInt16(16)
        let rate = sr * UInt32(ch) * UInt32(bps) / 8
        let align = ch * bps / 8

        var h = Data()
        h.reserveCapacity(44)
        func le(_ v: some FixedWidthInteger) {
            var x = v.littleEndian; withUnsafeBytes(of: &x) { h.append(contentsOf: $0) }
        }
        h.append(contentsOf: "RIFF".utf8); le(36 + dataSize)
        h.append(contentsOf: "WAVE".utf8)
        h.append(contentsOf: "fmt ".utf8); le(UInt32(16))
        le(UInt16(1)); le(ch); le(sr); le(rate); le(align); le(bps)
        h.append(contentsOf: "data".utf8); le(dataSize)
        return h
    }

    func refreshWaveformPeaks(from buffer: AudioBuffer, clearExisting: Bool = false) {
        if clearExisting {
            deferUIUpdate { $0.waveformPeaks = [] }
        }
        peakComputationTask?.cancel()
        peakComputationTask = Task { [weak self, buffer] in
            let peaks = await Task.detached(priority: .userInitiated) {
                Self.computePeaks(from: buffer)
            }.value
            guard !Task.isCancelled else { return }
            self?.deferUIUpdate { state in
                state.waveformPeaks = peaks
            }
        }
    }

    /// Apply a freshly decoded buffer on the next runloop tick (History → Edit / Open).
    func applyLoadedBuffer(_ buffer: AudioBuffer, fileName: String) {
        invalidateInFlightBufferWork()
        deferUIUpdate { state in
            state.fileName = fileName
            state.operationErrorMessage = nil
            state.audioBuffer = buffer
            state.sampleRate = buffer.sampleRate
            state.duration = buffer.duration
            state.isLoaded = true
            state.selectionStart = 0
            state.selectionEnd = 0
            state.resetUndoHistory()
            state.invalidatePreviewPlayer()
            state.noteBufferMutation()
            // Peaks/metrics schedule their own deferred publishes when ready.
            state.refreshWaveformPeaks(from: buffer, clearExisting: true)
            state.refreshLevelMetrics()
        }
    }

    /// Cancel DSP and bump document/buffer identity so in-flight results cannot
    /// commit onto a different file or an undone buffer.
    func invalidateInFlightBufferWork() {
        dspTask?.cancel()
        documentGeneration &+= 1
        noteBufferMutation()
        isProcessing = false
    }

    private func shouldCommitProcessedBuffer(sourceRevision: Int) -> Bool {
        !Task.isCancelled && bufferRevision == sourceRevision
    }

    nonisolated static func computePeaks(from buffer: AudioBuffer) -> [Float] {
        let sampleCount = buffer.data.count / MemoryLayout<Int16>.size
        let peakCount = 200
        guard sampleCount > 0 else {
            return []
        }

        let outputCount = min(peakCount, sampleCount)
        let chunkSize = max(sampleCount / outputCount, 1)
        var peaks = [Float](repeating: 0, count: outputCount)

        buffer.data.withUnsafeBytes { rawBuffer in
            let samples = rawBuffer.bindMemory(to: Int16.self)
            for bucket in 0 ..< outputCount {
                let start = bucket * chunkSize
                let end = bucket == outputCount - 1
                    ? sampleCount
                    : min(start + chunkSize, sampleCount)
                var maxMagnitude = 0
                for index in start ..< end {
                    let sample = Int(samples[index])
                    let magnitude = sample == Int(Int16.min) ? 32768 : abs(sample)
                    if magnitude > maxMagnitude {
                        maxMagnitude = magnitude
                    }
                }
                peaks[bucket] = Float(maxMagnitude) / 32768.0
            }
        }

        return peaks
    }

    private static func errorMessage(for error: Error) -> String {
        if let localizedError = error as? LocalizedError,
           let description = localizedError.errorDescription {
            return description
        }
        return error.localizedDescription
    }

    private enum PreviewError: LocalizedError {
        case invalidBuffer(String)

        var errorDescription: String? {
            switch self {
            case let .invalidBuffer(reason):
                "The editor buffer is invalid: \(reason)."
            }
        }
    }
}
