import Foundation

/// Provenance of a LoRA adapter — where it came from.
public enum AdapterSource: String, Codable, Sendable, Hashable {
    /// Trained on-device by the user.
    case local

    /// Downloaded from the curated community manifest.
    case community

    /// Imported directly from a HuggingFace repo URL.
    case huggingface
}
