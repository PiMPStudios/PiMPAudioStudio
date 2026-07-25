import SwiftUI

struct PolishForKeepView: View {
    @ObservedObject var state: EditorState
    let onApply: () -> Void

    private let accent = Color.orange

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            if let metrics = state.levelMetrics {
                meterRow(metrics)
            }

            pickerRow(
                title: "Target",
                selection: $state.polishRecipe.target,
                options: PolishForKeepRecipe.Target.allCases
            ) { $0.displayName }

            Text(state.polishRecipe.target.detail)
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)

            pickerRow(
                title: "Tone",
                selection: $state.polishRecipe.tone,
                options: PolishForKeepRecipe.Tone.allCases
            ) { $0.displayName }

            pickerRow(
                title: "Glue",
                selection: $state.polishRecipe.glue,
                options: PolishForKeepRecipe.Glue.allCases
            ) { $0.displayName }

            pickerRow(
                title: "Space",
                selection: $state.polishRecipe.space,
                options: PolishForKeepRecipe.Space.allCases
            ) { $0.displayName }

            Toggle(isOn: $state.polishRecipe.trimSilenceAndFades) {
                Text("Trim silence & fade edges")
                    .font(.caption)
            }
            .toggleStyle(.switch)
            .controlSize(.small)

            HStack {
                Text("Apply to:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Whole Track")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(.secondary.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                Spacer()
            }

            Button {
                onApply()
            } label: {
                Text("Apply Polish for Keep")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .tint(accent)
            .disabled(state.isDSPBusy || !state.isLoaded)

            Text("Runs trim → high-pass → tone → glue → space → loudness/limit as one undo step. Use individual Effects below for finer control.")
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(accent.opacity(0.25), lineWidth: 1))
    }

    private var header: some View {
        HStack {
            Image(systemName: "sparkles")
                .font(.caption)
                .foregroundStyle(accent)
            Text("POLISH FOR KEEP")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(accent)
                .tracking(0.5)
            Spacer()
        }
    }

    private func meterRow(_ metrics: LoudnessMeter.Metrics) -> some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Peak")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(LoudnessMeter.formatDBFS(metrics.peakDBFS))
                    .font(.system(size: 12, weight: .medium).monospacedDigit())
                    .foregroundStyle(metrics.peakDBFS > -0.3 ? Color.orange : Color.primary)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Loudness ≈")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(LoudnessMeter.formatLUFS(metrics.approximateLUFS))
                    .font(.system(size: 12, weight: .medium).monospacedDigit())
            }
            Spacer()
        }
        .padding(8)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }

    private func pickerRow<T: Hashable>(
        title: String,
        selection: Binding<T>,
        options: [T],
        label: @escaping (T) -> String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Picker(title, selection: selection) {
                ForEach(options, id: \.self) { option in
                    Text(label(option)).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }
}
