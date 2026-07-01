import SwiftUI

struct PitchShiftControlView: View {
    @Binding var parameters: PitchShiftParameters
    @State private var isEditingValue = false
    @State private var editText = ""

    private let snapPoints: [Float] = [-1200, -700, -500, 0, 500, 700, 1200]
    private let snapThreshold: Float = 30.0

    private var semitones: Float {
        parameters.pitch / 100.0
    }

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text("Pitch").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if isEditingValue {
                    TextField("", text: $editText)
                        .font(.caption).monospacedDigit()
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { commitEdit() }
                } else {
                    Text(formatPitch(parameters.pitch))
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

            Slider(value: $parameters.pitch, in: -2400...2400)
                .tint(qualityColor)
                .onChange(of: parameters.pitch) { _, newValue in
                    parameters.pitch = snapped(newValue)
                }

            HStack(spacing: 0) {
                Text("-24 st")
                Spacer()
                Text("0")
                Spacer()
                Text("+24 st")
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
        let absSemitones = abs(semitones)
        if absSemitones <= 12 { return .green }
        if absSemitones <= 18 { return .yellow }
        return .red
    }

    private var qualityText: String {
        let absSemitones = abs(semitones)
        if absSemitones <= 12 { return "Excellent quality" }
        if absSemitones <= 18 { return "Good quality, minor artifacts possible" }
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

    private func formatPitch(_ cents: Float) -> String {
        let st = Int((cents / 100).rounded())
        let sign = st >= 0 ? "+" : ""
        return "\(sign)\(st) st (\(sign)\(Int(cents))¢)"
    }

    private func beginEdit() {
        editText = String(Int(semitones))
        isEditingValue = true
    }

    private func commitEdit() {
        if let value = Float(editText) {
            let clamped = min(max(value, -24), 24)
            parameters.pitch = clamped * 100
        }
        isEditingValue = false
    }
}
