import SwiftUI

struct PolishForKeepView: View {
    @ObservedObject var state: EditorState
    /// Receives the recipe to apply (local draft, not a mid-gesture `@Published` write).
    let onApply: (PolishForKeepRecipe) -> Void

    private let accent = Color.orange

    /// Local draft so segmented pickers update immediately without publishing
    /// `EditorState` during the control's view-update pass (avoids 1×N observer warnings).
    @State private var recipe: PolishForKeepRecipe = .default

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            if let metrics = state.levelMetrics {
                meterRow(metrics)
            } else {
                Text("Measuring levels…")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            pickerRow(
                title: "Target",
                selection: $recipe.target,
                options: PolishForKeepRecipe.Target.allCases
            ) { $0.displayName }

            Text(recipe.target.detail)
                .font(.system(size: 10))
                .foregroundStyle(.tertiary)

            pickerRow(
                title: "Tone",
                selection: $recipe.tone,
                options: PolishForKeepRecipe.Tone.allCases
            ) { $0.displayName }

            pickerRow(
                title: "Glue",
                selection: $recipe.glue,
                options: PolishForKeepRecipe.Glue.allCases
            ) { $0.displayName }

            pickerRow(
                title: "Space",
                selection: $recipe.space,
                options: PolishForKeepRecipe.Space.allCases
            ) { $0.displayName }

            Toggle(isOn: $recipe.trimSilenceAndFades) {
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
                onApply(recipe)
            } label: {
                Text("Apply Polish for Keep")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .tint(accent)
            .disabled(state.isDSPBusy || !state.isLoaded)

            Text(
                "Runs trim → high-pass → tone → glue → space → loudness/limit as one undo step. Use individual Effects below for finer control."
            )
            .font(.system(size: 10))
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(accent.opacity(0.25), lineWidth: 1))
        .onAppear {
            recipe = state.polishRecipe
        }
        .onChange(of: state.polishRecipe) { _, newValue in
            if recipe != newValue {
                recipe = newValue
            }
        }
        .onChange(of: recipe) { _, newValue in
            guard state.polishRecipe != newValue else { return }
            // Publish on the next runloop tick — not during the picker update pass.
            state.deferUIUpdate { $0.polishRecipe = newValue }
        }
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
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Peak")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(LoudnessMeter.formatDBFS(metrics.peakDBFS))
                    .font(.system(size: 12, weight: .medium).monospacedDigit())
                    .foregroundStyle(metrics.peakDBFS > -0.3 ? Color.orange : Color.primary)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("True peak ≈")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(LoudnessMeter.formatDBFS(metrics.truePeakDBFS))
                    .font(.system(size: 12, weight: .medium).monospacedDigit())
                    .foregroundStyle(metrics.truePeakDBFS > -0.1 ? Color.orange : Color.primary)
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
