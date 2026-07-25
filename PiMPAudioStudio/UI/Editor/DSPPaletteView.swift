import SwiftUI

struct DSPPaletteView: View {
    @ObservedObject var state: EditorState
    let onApply: (DSPEffect, DSPParameters) -> Void

    @State private var showingConfirmation: DSPEffect?
    @State private var confirmationTask: Task<Void, Never>?

    var body: some View {
        paletteContent
            // Confirm the finished apply, not the button press — a long track keeps
            // processing well after the tap, and a failed apply must not show a check.
            .onChange(of: state.lastCompletedApply) { _, completed in
                guard let completed else { return }
                showConfirmation(completed.effect)
            }
    }

    private var paletteContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            levelMeterStrip

            PolishForKeepView(state: state) { recipe in
                state.applyPolishForKeep(recipe)
            }

            if state.polishAppliedConfirmation {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("Polish applied — one undo reverts the full chain")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .transition(.opacity)
            }

            Divider()
                .padding(.vertical, 2)

            Text("EFFECTS")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .tracking(0.5)

            effectGrid

            if let expanded = state.expandedEffect, !expanded.isTransform {
                expandedPanel(for: expanded)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            Divider()
                .padding(.vertical, 4)

            Text("TRANSFORMS")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
                .tracking(0.5)

            transformGrid

            if let expanded = state.expandedEffect, expanded.isTransform {
                expandedPanel(for: expanded)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(16)
        .animation(.easeInOut(duration: 0.2), value: state.expandedEffect)
        .onDisappear {
            confirmationTask?.cancel()
        }
    }

    // MARK: - Level meters

    @ViewBuilder
    private var levelMeterStrip: some View {
        if let metrics = state.levelMetrics {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 14) {
                    metricChip(
                        title: "Peak",
                        value: LoudnessMeter.formatDBFS(metrics.peakDBFS),
                        warn: metrics.peakDBFS > -0.3
                    )
                    metricChip(
                        title: "True peak ≈",
                        value: LoudnessMeter.formatDBFS(metrics.truePeakDBFS),
                        warn: metrics.truePeakDBFS > -0.1
                    )
                    metricChip(
                        title: "Loudness ≈",
                        value: LoudnessMeter.formatLUFS(metrics.approximateLUFS),
                        warn: false
                    )
                    if let monoPeak = metrics.monoPeakDBFS {
                        metricChip(
                            title: "Mono peak",
                            value: LoudnessMeter.formatDBFS(monoPeak),
                            warn: monoPeak > -0.3
                        )
                    }
                    Spacer(minLength: 0)
                }

                if metrics.isStereo {
                    Toggle(isOn: $state.previewInMono) {
                        Text("Preview mono (sum L+R — check compatibility, does not edit)")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                    }
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                }
            }
            .padding(10)
            .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func metricChip(title: String, value: String, warn: Bool) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12, weight: .medium).monospacedDigit())
                .foregroundStyle(warn ? Color.orange : Color.primary)
        }
    }

    // MARK: - Effect Grid

    /// Three equal columns so a final orphan (Soft Gate) stays left-aligned at ⅓ width
    /// instead of stretching full-bleed when placeholders are Spacers.
    private static let paletteColumns = [
        GridItem(.flexible(), spacing: 6),
        GridItem(.flexible(), spacing: 6),
        GridItem(.flexible(), spacing: 6),
    ]

    private var effectGrid: some View {
        LazyVGrid(columns: Self.paletteColumns, spacing: 6) {
            effectButton(.trimToSelection)
            effectButton(.deleteSelection)
            effectButton(.trimSilence)
            effectButton(.fadeIn)
            effectButton(.fadeOut)
            effectButton(.normalize)
            effectButton(.highPass)
            effectButton(.eq)
            effectButton(.reverb)
            effectButton(.compress)
            effectButton(.limit)
            effectButton(.loudness)
            effectButton(.softGate)
        }
    }

    private var transformGrid: some View {
        LazyVGrid(columns: Self.paletteColumns, spacing: 6) {
            effectButton(.timeStretch)
            effectButton(.pitchShift)
        }
    }

    private func effectButton(_ effect: DSPEffect) -> some View {
        let isExpanded = state.expandedEffect == effect
        let isDisabled = effect.requiresSelection && !state.hasSelection
        let accent = effect.category.accentColor

        return Button {
            if effect.requiresParameters {
                withAnimation {
                    state.expandedEffect = isExpanded ? nil : effect
                }
            } else {
                onApply(effect, .none)
            }
        } label: {
            VStack(spacing: 3) {
                ZStack {
                    Image(systemName: effect.iconName)
                        .font(.system(size: 13, weight: .medium))
                        .opacity(showingConfirmation == effect ? 0 : 1)

                    if showingConfirmation == effect {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.green)
                            .transition(.scale.combined(with: .opacity))
                    }
                }
                Text(effect.displayName)
                    .font(.system(size: 9.5, weight: isExpanded ? .semibold : .regular))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .padding(.horizontal, 4)
        }
        .buttonStyle(.bordered)
        .tint(isExpanded ? accent : .gray)
        .disabled(isDisabled || state.isDSPBusy)
    }

    private func showConfirmation(_ effect: DSPEffect) {
        confirmationTask?.cancel()
        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
            showingConfirmation = effect
        }
        confirmationTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(600))
            guard !Task.isCancelled, showingConfirmation == effect else {
                return
            }
            withAnimation(.easeOut(duration: 0.25)) {
                showingConfirmation = nil
            }
        }
    }

    // MARK: - Expanded Panels

    @ViewBuilder
    private func expandedPanel(for effect: DSPEffect) -> some View {
        let accent = effect.category.accentColor
        VStack(spacing: 12) {
            HStack {
                Image(systemName: effect.iconName)
                    .font(.caption)
                    .foregroundStyle(accent)
                Text(effect.displayName)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(accent)
                Spacer()
                Button {
                    withAnimation { state.expandedEffect = nil }
                } label: {
                    Image(systemName: "chevron.up")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }

            switch effect {
            case .normalize:
                NormalizeControlView(parameters: $state.normalizeParameters)
            case .trimSilence:
                TrimSilenceControlView(parameters: $state.trimSilenceParameters)
            case .highPass:
                HighPassControlView(parameters: $state.highPassParameters)
            case .compress:
                CompressorControlView(parameters: $state.compressorParameters)
            case .limit:
                LimiterControlView(parameters: $state.limiterParameters)
            case .loudness:
                LoudnessControlView(parameters: $state.loudnessParameters)
            case .softGate:
                SoftGateControlView(parameters: $state.softGateParameters)
            case .eq:
                EQControlView(parameters: $state.eqParameters)
            case .reverb:
                ReverbControlView(
                    parameters: $state.reverbParameters,
                    selectedPreset: $state.selectedReverbPreset
                )
            case .timeStretch:
                TimeStretchControlView(parameters: $state.timeStretchParameters)
            case .pitchShift:
                PitchShiftControlView(parameters: $state.pitchShiftParameters)
            default:
                EmptyView()
            }

            applySection(for: effect)
        }
        .padding(12)
        .background(accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(accent.opacity(0.25), lineWidth: 1))
    }

    private func applySection(for effect: DSPEffect) -> some View {
        let accent = effect.category.accentColor
        return VStack(spacing: 8) {
            HStack {
                Text("Apply to:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(applyTargetLabel(for: effect))
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(.secondary.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                    .foregroundStyle(.primary)
                Spacer()
            }

            Button {
                onApply(effect, parameters(for: effect))
            } label: {
                Text("Apply \(effect.displayName)")
                    .font(.system(size: 12, weight: .medium))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .tint(accent)
            .disabled(state.isDSPBusy)
        }
    }

    private func parameters(for effect: DSPEffect) -> DSPParameters {
        switch effect {
        case .normalize: return .normalize(state.normalizeParameters)
        case .trimSilence: return .trimSilence(state.trimSilenceParameters)
        case .highPass: return .highPass(state.highPassParameters)
        case .compress: return .compress(state.compressorParameters)
        case .limit: return .limit(state.limiterParameters)
        case .loudness: return .loudness(state.loudnessParameters)
        case .softGate: return .softGate(state.softGateParameters)
        case .eq: return .eq(state.eqParameters)
        case .reverb: return .reverb(state.reverbParameters)
        case .timeStretch: return .timeStretch(state.timeStretchParameters)
        case .pitchShift: return .pitchShift(state.pitchShiftParameters)
        default: return .none
        }
    }

    private func applyTargetLabel(for effect: DSPEffect) -> String {
        if effect == .trimSilence {
            return "Whole Track"
        }
        if state.hasSelection {
            let start = formatTime(state.selectionStart)
            let end = formatTime(state.selectionEnd)
            return "Selection (\(start)–\(end))"
        }
        return "Whole Track"
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let seconds = Int(time) % 60
        let tenths = Int((time - Double(Int(time))) * 10)
        let minutes = Int(time) / 60
        if minutes > 0 {
            return String(format: "%d:%02d.%d", minutes, seconds, tenths)
        }
        return String(format: "%d.%ds", seconds, tenths)
    }
}
