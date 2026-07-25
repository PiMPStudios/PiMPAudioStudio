import SwiftUI

struct LimiterControlView: View {
    @Binding var parameters: LimiterParameters

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                presetButton("Streaming −1", .streaming)
                presetButton("Louder −0.5", .louder)
            }

            labeledSlider(
                title: "Ceiling",
                valueText: String(format: "%.1f dB", parameters.ceilingDBFS),
                value: $parameters.ceilingDBFS,
                range: LimiterParameters.ceilingRange
            )
            labeledSlider(
                title: "Release",
                valueText: String(format: "%.0f ms", parameters.releaseMilliseconds),
                value: $parameters.releaseMilliseconds,
                range: 10...200
            )

            Text("Brickwall-style peak limiter. Use after glue or loudness staging so peaks stay under the ceiling.")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func presetButton(_ title: String, _ preset: LimiterParameters) -> some View {
        Button(title) { parameters = preset }
            .buttonStyle(.bordered)
            .tint(
                abs(parameters.ceilingDBFS - preset.ceilingDBFS) < 0.05
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
