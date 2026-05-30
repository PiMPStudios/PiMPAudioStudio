import SwiftUI

struct EQControlView: View {
    @Binding var parameters: EQParameters

    private let bandLabels = ["Band 1 — Low", "Band 2 — Mid", "Band 3 — High"]

    var body: some View {
        VStack(spacing: 10) {
            ForEach(0..<3, id: \.self) { index in
                bandView(index: index)
            }
        }
    }

    private func bandView(index: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(bandLabels[index])
                .font(.caption2)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Freq").font(.system(size: 9)).foregroundStyle(.tertiary)
                    Text(formatFrequency(parameters.bands[index].frequency))
                        .font(.caption).monospacedDigit()
                }
                .frame(width: 52, alignment: .leading)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Gain").font(.system(size: 9)).foregroundStyle(.tertiary)
                    Text(formatGain(parameters.bands[index].gain))
                        .font(.caption).monospacedDigit()
                        .foregroundStyle(gainColor(parameters.bands[index].gain))
                }
                .frame(width: 52, alignment: .leading)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Q").font(.system(size: 9)).foregroundStyle(.tertiary)
                    Text(String(format: "%.1f", parameters.bands[index].q))
                        .font(.caption).monospacedDigit()
                }
                .frame(width: 36, alignment: .leading)
            }

            VStack(spacing: 4) {
                Slider(value: $parameters.bands[index].frequency, in: 20...20000)
                    .tint(.blue)
                Slider(value: $parameters.bands[index].gain, in: -24...24)
                    .tint(gainColor(parameters.bands[index].gain))
                Slider(value: $parameters.bands[index].q, in: 0.1...10)
                    .tint(.gray)
            }
        }
        .padding(8)
        .background(Color(white: 0.12))
        .cornerRadius(6)
    }

    private func formatFrequency(_ freq: Float) -> String {
        if freq >= 1000 {
            return String(format: "%.1f kHz", freq / 1000)
        }
        return String(format: "%.0f Hz", freq)
    }

    private func formatGain(_ gain: Float) -> String {
        if gain >= 0 {
            return String(format: "+%.1f dB", gain)
        }
        return String(format: "%.1f dB", gain)
    }

    private func gainColor(_ gain: Float) -> Color {
        if gain > 0.1 { return .green }
        if gain < -0.1 { return .red }
        return .secondary
    }
}
