import SwiftUI

struct ReverbControlView: View {
    @Binding var parameters: ReverbParameters
    @Binding var selectedPreset: ReverbPreset?

    var body: some View {
        VStack(spacing: 12) {
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
