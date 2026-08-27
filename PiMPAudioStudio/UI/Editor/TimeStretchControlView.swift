import SwiftUI

struct TimeStretchControlView: View {
    @Binding var parameters: TimeStretchParameters
    @State private var isEditingValue = false
    @State private var editText = ""

    private let snapPoints: [Float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    private let snapThreshold: Float = 0.03

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text("Rate").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if isEditingValue {
                    TextField("", text: $editText)
                        .font(.caption).monospacedDigit()
                        .frame(width: 60)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { commitEdit() }
                } else {
                    Text(formatRate(parameters.rate))
                        .font(.caption).monospacedDigit()
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            DSPEffectCategory.transform.accentColor.opacity(0.12),
                            in: RoundedRectangle(cornerRadius: 4)
                        )
                        .onTapGesture(count: 2) { beginEdit() }
                }
            }

            Slider(value: $parameters.rate, in: 0.25 ... 4.0)
                .tint(qualityColor)
                .onChange(of: parameters.rate) { _, newValue in
                    parameters.rate = snapped(newValue)
                }

            HStack(spacing: 0) {
                Text("0.25×")
                Spacer()
                Text("1.0×")
                Spacer()
                Text("4.0×")
            }
            .font(.system(size: 9))
            .foregroundStyle(.tertiary)

            qualityIndicator
        }
    }

    private var qualityIndicator: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(qualityColor)
                .frame(width: 6, height: 6)
            Text(qualityText)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private var qualityColor: Color {
        let rate = parameters.rate
        if rate >= 0.5, rate <= 2.0 {
            return .green
        }
        if rate >= 0.35, rate <= 3.0 {
            return .yellow
        }
        return .red
    }

    private var qualityText: String {
        let rate = parameters.rate
        if rate >= 0.5, rate <= 2.0 {
            return "Excellent quality"
        }
        if rate >= 0.35, rate <= 3.0 {
            return "Good quality, minor artifacts possible"
        }
        return "Significant artifacts likely"
    }

    private func snapped(_ value: Float) -> Float {
        for snap in snapPoints {
            if abs(value - snap) < snapThreshold {
                return snap
            }
        }
        return value
    }

    private func formatRate(_ rate: Float) -> String {
        String(format: "%.2f×", rate)
    }

    private func beginEdit() {
        editText = String(format: "%.2f", parameters.rate)
        isEditingValue = true
    }

    private func commitEdit() {
        if let value = Float(editText) {
            parameters.rate = min(max(value, 0.25), 4.0)
        }
        isEditingValue = false
    }
}
