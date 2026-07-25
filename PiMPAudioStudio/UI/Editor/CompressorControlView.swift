import SwiftUI

struct CompressorControlView: View {
    @Binding var parameters: CompressorParameters

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                presetButton("Light", CompressorParameters.light)
                presetButton("Medium", CompressorParameters.medium)
            }

            labeledSlider(
                title: "Threshold",
                valueText: String(format: "%.0f dB", parameters.thresholdDBFS),
                value: $parameters.thresholdDBFS,
                range: -40...(-6)
            )
            labeledSlider(
                title: "Ratio",
                valueText: String(format: "%.1f:1", parameters.ratio),
                value: $parameters.ratio,
                range: 1.5...8
            )
            labeledSlider(
                title: "Attack",
                valueText: String(format: "%.0f ms", parameters.attackMilliseconds),
                value: $parameters.attackMilliseconds,
                range: 1...80
            )
            labeledSlider(
                title: "Release",
                valueText: String(format: "%.0f ms", parameters.releaseMilliseconds),
                value: $parameters.releaseMilliseconds,
                range: 20...400
            )
            labeledSlider(
                title: "Makeup",
                valueText: String(format: "%+.1f dB", parameters.makeupGainDB),
                value: $parameters.makeupGainDB,
                range: 0...12
            )

            Text("Stereo-linked glue compression for evening dynamics. Light is a safe starting point for full mixes.")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func presetButton(_ title: String, _ preset: CompressorParameters) -> some View {
        Button(title) { parameters = preset }
            .buttonStyle(.bordered)
            .tint(parameters == preset ? DSPEffectCategory.dynamics.accentColor : .gray)
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
