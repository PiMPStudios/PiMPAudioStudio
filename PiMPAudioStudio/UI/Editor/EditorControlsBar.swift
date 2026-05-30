import SwiftUI
import PiMPAudioCore

struct EditorControlsBar: View {
    @ObservedObject var state: EditorState
    let onExecute: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            // Presets
            HStack(spacing: 8) {
                ForEach(InpaintPreset.allCases, id: \.self) { preset in
                    Button(preset.rawValue.capitalized) {
                        state.applyPreset(preset)
                    }
                    .buttonStyle(.bordered)
                    .tint(state.selectedPreset == preset ? .blue : .gray)
                    .controlSize(.small)
                }
                Spacer()
            }

            // Sliders
            HStack(spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Strength").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.2f", state.strength))
                            .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                    }
                    Slider(value: $state.strength, in: 0...1) { _ in
                        state.clearPresetIfManual()
                    }
                    .tint(.orange)
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Guidance Scale").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.1f", state.guidanceScale))
                            .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                    }
                    Slider(value: $state.guidanceScale, in: 1...15) { _ in
                        state.clearPresetIfManual()
                    }
                    .tint(.blue)
                }
            }

            // Action button
            Button(action: onExecute) {
                HStack {
                    if state.isProcessing {
                        ProgressView().controlSize(.small)
                    }
                    Text(state.actionButtonTitle).fontWeight(.medium)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .disabled(!canExecute)
        }
        .padding(16)
        .background(.ultraThinMaterial)
    }

    private var canExecute: Bool {
        guard state.isLoaded, !state.isProcessing, !state.regionPrompt.isEmpty else { return false }
        if state.editMode == .inpaint { return state.hasSelection }
        return true
    }
}
