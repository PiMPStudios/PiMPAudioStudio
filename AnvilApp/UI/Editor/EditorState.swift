import SwiftUI
import AnvilCore

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
}
