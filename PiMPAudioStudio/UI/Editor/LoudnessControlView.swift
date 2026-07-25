import SwiftUI

struct LoudnessControlView: View {
    @Binding var parameters: LoudnessParameters

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                presetButton("Streaming −14", .streaming)
                presetButton("Louder −10", .louder)
            }

            labeledSlider(
                title: "Target loudness",
                valueText: String(format: "%.0f LUFS", parameters.targetLUFS),
                value: $parameters.targetLUFS,
                range: LoudnessParameters.targetRange
            )
            labeledSlider(
                title: "Peak ceiling",
                valueText: String(format: "%.1f dB", parameters.ceilingDBFS),
                value: $parameters.ceilingDBFS,
                range: LimiterParameters.ceilingRange
            )
            labeledSlider(
                title: "Max boost",
                valueText: String(format: "%.0f dB", parameters.maxBoostDB),
                value: $parameters.maxBoostDB,
                range: 0...24
            )

            Text("Raises or lowers level toward the target using the same approximate LUFS meter as the strip, then limits peaks to the ceiling. Best-effort aiming, not broadcast certification.")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func presetButton(_ title: String, _ preset: LoudnessParameters) -> some View {
        Button(title) { parameters = preset }
            .buttonStyle(.bordered)
            .tint(
                abs(parameters.targetLUFS - preset.targetLUFS) < 0.5
                    ? DSPEffectCategory.dynamics.accentColor
                    : .gray
            )
            .controlSize(.small)
    }

    private func labeledSlider(
        title: String,
        valueText: String,
        value: Binding<Float>,
        range: ClosedRange<Float>
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(valueText).font(.caption).monospacedDigit()
            }
            Slider(value: value, in: range)
                .tint(DSPEffectCategory.dynamics.accentColor)
        }
    }
}
