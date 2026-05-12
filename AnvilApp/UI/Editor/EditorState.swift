import SwiftUI
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

    // DSP undo/redo — stores raw AudioBuffer snapshots.
    // AI inpainting also pushes here (via saveUndoSnapshot) so both paths
    // share a single undo button.
    private var undoStack: [AudioBuffer] = []
    private var redoStack: [AudioBuffer] = []

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
            undoStack.append(buffer)
            redoStack.removeAll()
            canUndo = true
            canRedo = false

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
        undoStack.append(buffer)
        redoStack.removeAll()
        canUndo = true
        canRedo = false
    }

    func undoDSP() {
        guard let previous = undoStack.popLast() else { return }
        if let current = audioBuffer {
            redoStack.append(current)
        }
        audioBuffer = previous
        duration = previous.duration
        waveformPeaks = computePeaks(from: previous)
        canUndo = !undoStack.isEmpty
        canRedo = !redoStack.isEmpty
    }

    func redoDSP() {
        guard let next = redoStack.popLast() else { return }
        if let current = audioBuffer {
            undoStack.append(current)
        }
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
