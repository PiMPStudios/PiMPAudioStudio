import SwiftUI

struct DSPPaletteView: View {
    @ObservedObject var state: EditorState
    let onApply: (DSPEffect, DSPParameters) -> Void

    @State private var showingConfirmation: DSPEffect?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("DSP EFFECTS")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .tracking(0.5)

            effectGrid

            if let expanded = state.expandedEffect {
                expandedPanel(for: expanded)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(16)
        .animation(.easeInOut(duration: 0.2), value: state.expandedEffect)
    }

    // MARK: - Effect Grid

    private var effectGrid: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                effectButton(.trimToSelection)
                effectButton(.deleteSelection)
                effectButton(.normalize)
            }
            HStack(spacing: 6) {
                effectButton(.fadeIn)
                effectButton(.fadeOut)
                effectButton(.eq)
            }
            HStack(spacing: 6) {
                effectButton(.reverb)
                Spacer()
                Spacer()
            }
        }
    }

    private func effectButton(_ effect: DSPEffect) -> some View {
        let isExpanded = state.expandedEffect == effect
        let isDisabled = effect.requiresSelection && !state.hasSelection

        return Button {
            if effect.requiresParameters {
                withAnimation {
                    state.expandedEffect = isExpanded ? nil : effect
                }
            } else {
                onApply(effect, .none)
                showConfirmation(effect)
            }
        } label: {
            ZStack {
                Text(effect.displayName)
                    .font(.system(size: 10, weight: isExpanded ? .semibold : .regular))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
                    .padding(.horizontal, 4)

                if showingConfirmation == effect {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(.green)
                }
            }
        }
        .buttonStyle(.bordered)
        .tint(isExpanded ? .blue : .gray)
        .disabled(isDisabled)
    }

    private func showConfirmation(_ effect: DSPEffect) {
        showingConfirmation = effect
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            if showingConfirmation == effect {
                showingConfirmation = nil
            }
        }
    }

    // MARK: - Expanded Panels

    @ViewBuilder
    private func expandedPanel(for effect: DSPEffect) -> some View {
        VStack(spacing: 12) {
            HStack {
                Text(effect.displayName)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.blue)
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
            case .eq:
                EQControlView(parameters: $state.eqParameters)
            case .reverb:
                ReverbControlView(
                    parameters: $state.reverbParameters,
                    selectedPreset: $state.selectedReverbPreset
                )
            default:
                EmptyView()
            }

            applySection(for: effect)
        }
        .padding(12)
        .background(Color(red: 0.1, green: 0.1, blue: 0.15))
        .cornerRadius(8)
    }

    private func applySection(for effect: DSPEffect) -> some View {
        VStack(spacing: 8) {
            HStack {
                Text("Apply to:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(applyTargetLabel)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color(white: 0.2))
                    .cornerRadius(4)
                    .foregroundStyle(.blue)
                Spacer()
            }

            Button {
                let params: DSPParameters
                switch effect {
                case .eq: params = .eq(state.eqParameters)
                case .reverb: params = .reverb(state.reverbParameters)
                default: params = .none
                }
                onApply(effect, params)
            } label: {
                Text("Apply \(effect.displayName)")
                    .font(.system(size: 12, weight: .medium))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
        }
    }

    private var applyTargetLabel: String {
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
