import SwiftUI

/// Visual grouping for `DSPEffect`. Gives every effect a category so an
/// accent color always means the same thing across the editor — edit ops,
/// spectral (EQ), spatial (reverb), or time-domain transforms — instead of
/// an incidental mix of blue-when-expanded/gray-when-not.
enum DSPEffectCategory {
    case edit
    case spectral
    case spatial
    case transform

    var accentColor: Color {
        switch self {
        case .edit: return .mint
        case .spectral: return .blue
        case .spatial: return .purple
        case .transform: return .indigo
        }
    }
}

extension DSPEffect {
    var category: DSPEffectCategory {
        switch self {
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut, .normalize:
            return .edit
        case .eq:
            return .spectral
        case .reverb:
            return .spatial
        case .timeStretch, .pitchShift:
            return .transform
        }
    }

    var iconName: String {
        switch self {
        case .trimToSelection: return "scissors"
        case .deleteSelection: return "trash"
        case .fadeIn: return "arrow.up.right"
        case .fadeOut: return "arrow.down.right"
        case .normalize: return "gauge"
        case .eq: return "slider.horizontal.3"
        case .reverb: return "waveform.path"
        case .timeStretch: return "timer"
        case .pitchShift: return "tuningfork"
        }
    }
}
