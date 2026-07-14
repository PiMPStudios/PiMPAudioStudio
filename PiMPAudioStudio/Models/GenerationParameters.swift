import Foundation
import PiMPAudioCore

enum GuidancePreset: String, CaseIterable, Codable, Identifiable, Sendable, Equatable {
    case constant
    case lateEmphasis
    case midToLate
    case softLate

    /// Non-XL default. Soft Late won sparse-prompt A/Bs on Turbo/Base/SFT.
    static let defaultSelection: GuidancePreset = .softLate

    /// XL prefers full-schedule CFG (Constant) for closer upstream/PEFT parity
    /// and to avoid late-only Soft Late kinks on 50-step XL schedules.
    static let xlDefaultSelection: GuidancePreset = .constant

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
            return "Forces guidance 6.0; CFG only for t <= 0.6 (late steps)"
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

    /// True when this preset ignores the user guidance-scale field.
    var overridesGuidanceScale: Bool {
        guidanceScaleOverride != nil
    }

    func effectiveGuidanceScale(userValue: Float, fallback: Float = 7.5) -> Float {
        if let guidanceScaleOverride {
            return guidanceScaleOverride
        }
        return userValue > 1.0 ? userValue : fallback
    }

    /// App/MCP default: Soft Late for non-XL, Constant for XL DiT / XL adapters.
    static func defaultSelection(isXLContext: Bool) -> GuidancePreset {
        isXLContext ? xlDefaultSelection : defaultSelection
    }

    /// Nudge Soft Late → Constant in XL contexts. Leaves any other user-chosen
    /// preset (including a deliberate re-selection of Soft Late after Constant)
    /// alone only when `current` is already non-Soft-Late; Soft Late is always
    /// treated as the auto-default for this nudge because it is `defaultSelection`.
    ///
    /// Call this when XL becomes active (model or adapter). Do not call it on
    /// every generation if the user may intentionally pick Soft Late for an XL A/B.
    static func applyingXLRecipeIfNeeded(
        current: GuidancePreset,
        isXLContext: Bool
    ) -> GuidancePreset {
        guard isXLContext, current == .softLate else { return current }
        return .constant
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
