import Foundation
import AnvilCore

enum GuidancePreset: String, CaseIterable, Codable, Identifiable, Sendable, Equatable {
    case constant
    case lateEmphasis
    case midToLate
    case softLate

    static let defaultSelection: GuidancePreset = .softLate

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .constant: return "Constant"
        case .lateEmphasis: return "Late Emphasis"
        case .midToLate: return "Mid-to-Late"
        case .softLate: return "Soft Late"
        }
    }

    var detail: String {
        switch self {
        case .constant:
            return "CFG across the full schedule"
        case .lateEmphasis:
            return "CFG only for t <= 0.5"
        case .midToLate:
            return "CFG only for t <= 0.75"
        case .softLate:
            return "Guidance 6.0, CFG only for t <= 0.6"
        }
    }

    var cfgInterval: (start: Float, end: Float) {
        switch self {
        case .constant:
            return (0.0, 1.0)
        case .lateEmphasis:
            return (0.0, 0.5)
        case .midToLate:
            return (0.0, 0.75)
        case .softLate:
            return (0.0, 0.6)
        }
    }

    var guidanceScaleOverride: Float? {
        switch self {
        case .softLate:
            return 6.0
        case .constant, .lateEmphasis, .midToLate:
            return nil
        }
    }

    func effectiveGuidanceScale(userValue: Float, fallback: Float = 7.5) -> Float {
        if let guidanceScaleOverride {
            return guidanceScaleOverride
        }
        return userValue > 1.0 ? userValue : fallback
    }
}

/// Shared, observable bag for the user-tweakable generation knobs that need to
/// be reachable from both the GenerateView body and the inspector sidebar.
/// Holding them here (rather than as @State on a single view) keeps the two
/// surfaces in sync — typing a seed in the sidebar and clicking Generate in the
/// main view work together because they read/write the same instance.
@MainActor
final class GenerationParameters: ObservableObject {
    @Published var duration: Double = 30
    @Published var seedText: String = ""
    @Published var guidanceScale: Double = 7.5
    @Published var guidancePreset: GuidancePreset = GuidancePreset.defaultSelection
    @Published var pipelineOverride: PipelineChoice = .auto

    /// Parsed seed; nil = random.
    var seed: Int? {
        Int(seedText.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
