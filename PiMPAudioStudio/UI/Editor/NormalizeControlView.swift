import SwiftUI

struct NormalizeControlView: View {
    @Binding var parameters: NormalizeParameters

    private let presets: [(String, Float)] = [
        ("−3 dB", -3),
        ("−1 dB", -1),
        ("0 dB", 0),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Peak target")
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack(spacing: 6) {
                ForEach(presets, id: \.1) { label, value in
                    Button(label) {
                        parameters.targetPeakDBFS = value
                    }
                    .buttonStyle(.bordered)
                    .tint(abs(parameters.targetPeakDBFS - value) < 0.05
                          ? DSPEffectCategory.edit.accentColor
                          : .gray)
                    .controlSize(.small)
                }
            }

            HStack {
                Text(String(format: "%.1f dBFS", parameters.targetPeakDBFS))
                    .font(.caption)
                    .monospacedDigit()
                    .frame(width: 72, alignment: .leading)
                Slider(
                    value: $parameters.targetPeakDBFS,
                    in: NormalizeParameters.allowedRange
                )
                .tint(DSPEffectCategory.edit.accentColor)
            }

            Text("Scales the track so the sample peak lands at the target (headroom for later polish). −1 dB is the recommended keep default.")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
