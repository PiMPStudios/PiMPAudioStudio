import SwiftUI

struct HighPassControlView: View {
    @Binding var parameters: HighPassParameters

    private let presets: [(String, Float)] = [
        ("30 Hz", 30),
        ("40 Hz", 40),
        ("60 Hz", 60),
        ("80 Hz", 80),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Cutoff")
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack(spacing: 6) {
                ForEach(presets, id: \.1) { label, value in
                    Button(label) {
                        parameters.cutoffHz = value
                    }
                    .buttonStyle(.bordered)
                    .tint(abs(parameters.cutoffHz - value) < 0.5
                          ? DSPEffectCategory.spectral.accentColor
                          : .gray)
                    .controlSize(.small)
                }
            }

            HStack {
                Text(String(format: "%.0f Hz", parameters.cutoffHz))
                    .font(.caption)
                    .monospacedDigit()
                    .frame(width: 56, alignment: .leading)
                Slider(
                    value: $parameters.cutoffHz,
                    in: HighPassParameters.allowedRange
                )
                .tint(DSPEffectCategory.spectral.accentColor)
            }

            Text("Removes sub-rumble below the cutoff. ~40 Hz is a safe default for most full mixes.")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
