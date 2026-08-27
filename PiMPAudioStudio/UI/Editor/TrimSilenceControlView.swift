import SwiftUI

struct TrimSilenceControlView: View {
    @Binding var parameters: TrimSilenceParameters

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            labeledSlider(
                title: "Silence threshold",
                valueText: String(format: "%.0f dB", parameters.thresholdDBFS),
                value: $parameters.thresholdDBFS,
                range: -70 ... -30
            )
            labeledSlider(
                title: "Edge fade",
                valueText: String(format: "%.0f ms", parameters.edgeFadeMilliseconds),
                value: $parameters.edgeFadeMilliseconds,
                range: 0 ... 100
            )
            labeledSlider(
                title: "Max lead trim",
                valueText: String(format: "%.1f s", parameters.maxLeadingSeconds),
                value: $parameters.maxLeadingSeconds,
                range: 0.5 ... 15
            )
            labeledSlider(
                title: "Max trail trim",
                valueText: String(format: "%.1f s", parameters.maxTrailingSeconds),
                value: $parameters.maxTrailingSeconds,
                range: 0.5 ... 15
            )

            Text(
                "Removes quiet head/tail only up to the max trim limits, then applies short edge fades so exports do not click."
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
                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(valueText)
                    .font(.caption)
                    .monospacedDigit()
            }
            Slider(value: value, in: range)
                .tint(DSPEffectCategory.edit.accentColor)
        }
    }
}
