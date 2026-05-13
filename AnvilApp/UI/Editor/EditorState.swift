import SwiftUI
import AVFoundation
import AnvilCore
import Accelerate

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

    @Published var originalPrompt: String = ""
    @Published var regionPrompt: String = ""

    @Published var selectedPreset: InpaintPreset? = .moderate
    @Published var strength: Float = 0.6
    @Published var guidanceScale: Float = 7.0
    @Published var pipeline: PipelineChoice = .auto

    @Published var progress: InpaintProgress?
    @Published var isProcessing = false

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
    @Published var dspConfirmation: DSPEffect?
    @Published var timeStretchParameters: TimeStretchParameters = .default
    @Published var pitchShiftParameters: PitchShiftParameters = .default

    private let dspProcessor = DSPProcessor()

    // MARK: - Preview playback
    @Published var isPlayingPreview = false
    @Published var previewCurrentTime: TimeInterval = 0
    private var previewPlayer: AVAudioPlayer?
    private var previewTimer: Timer?

    // DSP undo/redo — stores raw AudioBuffer snapshots.
    // AI inpainting also pushes here (via saveUndoSnapshot) so both paths
    // share a single undo button.
    private var undoStack: [AudioBuffer] = []
    private var redoStack: [AudioBuffer] = []
    private let maxHistorySnapshots = 10
    private let maxHistoryBytes = 512 * 1024 * 1024

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
        case .inpaint: return "Inpaint Selection"
        case .repaint: return "Repaint Track"
        }
    }

    var regionPromptLabel: String {
        switch editMode {
        case .inpaint: return "Region Prompt"
        case .repaint: return "Target Style"
        }
    }

    func buildRequest() -> InpaintRequest {
        let selection: InpaintSelection
        if editMode == .repaint {
            selection = InpaintSelection(startTime: 0, endTime: duration)
        } else {
            let band = frequencyBandEnabled
                ? FrequencyBand(lowCutHz: lowCutHz, highCutHz: highCutHz)
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

    func applyDSPEffect(_ effect: DSPEffect, parameters: DSPParameters = .none) {
        guard let buffer = audioBuffer else { return }

        let target: ApplyTarget = hasSelection
            ? .selection(start: selectionStart, end: selectionEnd)
            : .wholeTrack

        do {
            let result = try dspProcessor.apply(effect: effect, to: buffer, target: target, parameters: parameters)

            // Commit: save pre-op buffer so it can be restored by undo.
            pushUndoSnapshot(buffer)
            redoStack.removeAll()
            canUndo = true
            canRedo = false

            invalidatePreviewPlayer()
            audioBuffer = result
            duration = result.duration
            waveformPeaks = computePeaks(from: result)

            if effect == .trimToSelection || effect == .deleteSelection || effect == .timeStretch {
                selectionStart = 0
                selectionEnd = 0
            }
        } catch {
            // Effect failed — buffer unchanged, undo stack not modified.
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
        waveformPeaks = computePeaks(from: previous)
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
        waveformPeaks = computePeaks(from: next)
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
        if previewPlayer == nil, let buffer = audioBuffer {
            let data = makeWAVData(from: buffer)
            previewPlayer = try? AVAudioPlayer(data: data, fileTypeHint: AVFileType.wav.rawValue)
            previewPlayer?.prepareToPlay()
        }
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
        guard let player = previewPlayer else { return }
        let t = fraction * player.duration
        player.currentTime = t
        previewCurrentTime = t
        playbackProgress = fraction
    }

    /// Nil out the player whenever the buffer changes — playPreview reloads lazily.
    private func invalidatePreviewPlayer() {
        previewPlayer?.stop()
        previewPlayer = nil
        isPlayingPreview = false
        previewCurrentTime = 0
        playbackProgress = 0
        stopPreviewTimer()
    }

    private func startPreviewTimer() {
        previewTimer?.invalidate()
        previewTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let player = self.previewPlayer else { return }
                self.previewCurrentTime = player.currentTime
                self.playbackProgress = player.duration > 0
                    ? player.currentTime / player.duration : 0
                if !player.isPlaying && self.isPlayingPreview {
                    self.isPlayingPreview = false
                    self.previewCurrentTime = 0
                    self.playbackProgress = 0
                    self.stopPreviewTimer()
                }
            }
        }
    }

    private func stopPreviewTimer() {
        previewTimer?.invalidate()
        previewTimer = nil
    }

    /// Build a minimal 44-byte WAV header + Int16 PCM body from an AudioBuffer.
    /// Used for in-memory playback via AVAudioPlayer(data:).
    private func makeWAVData(from buffer: AudioBuffer) -> Data {
        let pcm = buffer.data
        let sr    = UInt32(buffer.sampleRate)
        let ch    = UInt16(buffer.channels)
        let bps   = UInt16(16)
        let rate  = sr * UInt32(ch) * UInt32(bps) / 8
        let align = ch * bps / 8
        let dataSize = UInt32(pcm.count)

        var h = Data()
        func le<T: FixedWidthInteger>(_ v: T) {
            var x = v.littleEndian; withUnsafeBytes(of: &x) { h.append(contentsOf: $0) }
        }
        h.append(contentsOf: "RIFF".utf8); le(36 + dataSize)
        h.append(contentsOf: "WAVE".utf8)
        h.append(contentsOf: "fmt ".utf8); le(UInt32(16))
        le(UInt16(1)); le(ch); le(sr); le(rate); le(align); le(bps)
        h.append(contentsOf: "data".utf8); le(dataSize)
        h.append(pcm)
        return h
    }

    func computePeaks(from buffer: AudioBuffer) -> [Float] {
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
            let slice = Array(floats[start..<end])
            var peak: Float = 0
            vDSP_maxmgv(slice, 1, &peak, vDSP_Length(slice.count))
            peaks[i] = peak
        }
        return peaks
    }
}
