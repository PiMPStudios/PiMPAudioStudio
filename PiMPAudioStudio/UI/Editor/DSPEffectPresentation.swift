import SwiftUI

/// Visual grouping for `DSPEffect`. Gives every effect a category so an
/// accent color always means the same thing across the editor — edit ops,
/// spectral (EQ), spatial (reverb), or time-domain transforms — instead of
/// an incidental mix of blue-when-expanded/gray-when-not.
enum DSPEffectCategory {
    case edit
    case spectral
    case dynamics
    case spatial
    case transform

    var accentColor: Color {
        switch self {
        case .edit: .mint
        case .spectral: .blue
        case .dynamics: .orange
        case .spatial: .purple
        case .transform: .indigo
        }
    }
}

extension DSPEffect {
    var category: DSPEffectCategory {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut, .normalize, .trimSilence:
            .edit
        case .highPass, .eq:
            .spectral
        case .compress, .limit, .loudness, .softGate, .polishForKeep:
            .dynamics
        case .reverb:
            .spatial
        case .timeStretch, .pitchShift:
            .transform
        }
    }

    var iconName: String {
        switch self {
        case .trimToSelection: "scissors"
        case .deleteSelection: "trash"
        case .fadeIn: "arrow.up.right"
        case .fadeOut: "arrow.down.right"
        case .normalize: "gauge"
        case .trimSilence: "timeline.selection"
        case .highPass: "waveform.badge.minus"
        case .compress: "waveform.path.ecg"
        case .limit: "arrow.down.to.line"
        case .loudness: "speaker.wave.3"
        case .softGate: "door.left.hand.closed"
        case .polishForKeep: "sparkles"
        case .eq: "slider.horizontal.3"
        case .reverb: "waveform.path"
        case .timeStretch: "timer"
        case .pitchShift: "tuningfork"
        }
    }
}
