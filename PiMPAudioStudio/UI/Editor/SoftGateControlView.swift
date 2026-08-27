import SwiftUI

struct SoftGateControlView: View {
    @Binding var parameters: SoftGateParameters

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button("Gentle") {
                parameters = .gentle
            }
            .buttonStyle(.bordered)
            .tint(
                abs(parameters.thresholdDBFS - SoftGateParameters.gentle.thresholdDBFS) < 0.5
                    ? DSPEffectCategory.dynamics.accentColor
                    : .gray
            )
            .controlSize(.small)

            labeledSlider(
                title: "Threshold",
                valueText: String(format: "%.0f dB", parameters.thresholdDBFS),
                value: $parameters.thresholdDBFS,
                range: SoftGateParameters.thresholdRange
            )
            labeledSlider(
                title: "Ratio",
                valueText: String(format: "%.1f:1", parameters.ratio),
                value: $parameters.ratio,
                range: 1.5 ... 6
            )
            labeledSlider(
                title: "Attack",
                valueText: String(format: "%.0f ms", parameters.attackMilliseconds),
                value: $parameters.attackMilliseconds,
                range: 1 ... 40
            )
            labeledSlider(
                title: "Release",
                valueText: String(format: "%.0f ms", parameters.releaseMilliseconds),
                value: $parameters.releaseMilliseconds,
                range: 20 ... 300
            )
            labeledSlider(
                title: "Max range",
                valueText: String(format: "%.0f dB", parameters.rangeDB),
                value: $parameters.rangeDB,
                range: 3 ... 24
            )

            Text(
                "Light downward expansion of the noise floor. Easy to thin a full mix — keep threshold low and range modest. Prefer Trim Silence for dead air at the edges."
            )
            .font(.system(size: 10))
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
        }
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
