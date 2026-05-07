import Foundation
import AnvilCore

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
    @Published var pipelineOverride: PipelineChoice = .auto

    /// Parsed seed; nil = random.
    var seed: Int? {
        Int(seedText.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
