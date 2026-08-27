import Foundation

enum GenerationMode: String, CaseIterable, Sendable, Identifiable {
    case music
    case soundFX
    case ambient

    var id: String {
        rawValue
    }

    var displayName: String {
        switch self {
        case .music: "Music"
        case .soundFX: "Sound FX"
        case .ambient: "Ambient"
        }
    }

    init?(fromClassification text: String) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch cleaned {
        case "music": self = .music
        case "soundfx", "sound_fx", "sound fx", "sfx": self = .soundFX
        case "ambient": self = .ambient
        default: return nil
        }
    }
}

enum ModeSelection: Equatable, Sendable {
    case auto
    case manual(GenerationMode)

    var resolvedMode: GenerationMode? {
        switch self {
        case .auto: nil
        case let .manual(mode): mode
        }
    }
}
