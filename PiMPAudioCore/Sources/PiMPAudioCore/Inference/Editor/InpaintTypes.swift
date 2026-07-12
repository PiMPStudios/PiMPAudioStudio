import Foundation

// MARK: - EditMode

public enum EditMode: String, Sendable, CaseIterable {
    /// Inpaint: regenerate only the masked region, preserving surrounding audio.
    case inpaint
    /// Repaint: regenerate the masked region with stronger deviation from the original.
    case repaint
}

// MARK: - FrequencyBand

public enum FrequencyBandError: Error, Sendable, Equatable, LocalizedError {
    case invalid(lowCutHz: Float, highCutHz: Float)

    public var errorDescription: String? {
        switch self {
        case .invalid(let lowCutHz, let highCutHz):
            return "Frequency band must provide lowCutHz >= 0, highCutHz > lowCutHz, and highCutHz <= 24000 (got \(lowCutHz)–\(highCutHz) Hz)."
        }
    }
}

public struct FrequencyBand: Sendable, Equatable {
    /// Lower cut-off frequency in Hz (must be >= 0).
    public let lowCutHz: Float
    /// Upper cut-off frequency in Hz (must be > lowCutHz and <= 24000).
    public let highCutHz: Float

    /// Creates a frequency band, throwing when the range is empty or out of bounds.
    /// Prefer this over process-fatal validation so UI and MCP paths can recover.
    public init(lowCutHz: Float, highCutHz: Float) throws {
        guard lowCutHz >= 0,
              highCutHz > lowCutHz,
              highCutHz <= 24_000 else {
            throw FrequencyBandError.invalid(lowCutHz: lowCutHz, highCutHz: highCutHz)
        }
        self.lowCutHz = lowCutHz
        self.highCutHz = highCutHz
    }
}

// MARK: - InpaintSelection

public struct InpaintSelection: Sendable, Equatable {
    /// Start of the selected region in seconds.
    public let startTime: TimeInterval
    /// End of the selected region in seconds.
    public let endTime: TimeInterval
    /// Optional frequency band constraint for the selection.
    public let frequencyBand: FrequencyBand?

    public init(startTime: TimeInterval, endTime: TimeInterval, frequencyBand: FrequencyBand? = nil) {
        self.startTime = startTime
        self.endTime = endTime
        self.frequencyBand = frequencyBand
    }
}

// MARK: - PipelineChoice

public enum PipelineChoice: String, Codable, Sendable, CaseIterable {
    case auto
    case aceStep
    case stableAudio

    public var displayName: String {
        switch self {
        case .auto: return "Auto"
        case .aceStep: return "ACE-Step (Music)"
        case .stableAudio: return "Stable Audio (SFX)"
        }
    }
}

// MARK: - InpaintRequest

public struct InpaintRequest: Sendable {
    public let editMode: EditMode
    public let selection: InpaintSelection
    /// Prompt describing the desired content in the edited region.
    public let regionPrompt: String
    /// Optional prompt describing the original full audio context.
    public let originalPrompt: String?
    /// Denoising strength (0.0 – 1.0); higher values deviate more from source.
    public let strength: Float
    public let guidanceScale: Float
    public let pipeline: PipelineChoice
    public let seed: Int?

    public init(
        editMode: EditMode,
        selection: InpaintSelection,
        regionPrompt: String,
        originalPrompt: String? = nil,
        strength: Float,
        guidanceScale: Float,
        pipeline: PipelineChoice,
        seed: Int? = nil
    ) {
        self.editMode = editMode
        self.selection = selection
        self.regionPrompt = regionPrompt
        self.originalPrompt = originalPrompt
        self.strength = strength
        self.guidanceScale = guidanceScale
        self.pipeline = pipeline
        self.seed = seed
    }
}

// MARK: - InpaintProgress

public enum InpaintProgress: Sendable {
    case encoding
    case buildingMask
    case diffusion(DiffusionProgress)
    case decoding
    case complete
    case failed(String)
}

// MARK: - InpaintPreset

public enum InpaintPreset: String, CaseIterable, Sendable {
    case subtle
    case moderate
    case full

    public var strength: Float {
        switch self {
        case .subtle: return 0.3
        case .moderate: return 0.6
        case .full: return 0.9
        }
    }

    public var guidanceScale: Float {
        switch self {
        case .subtle: return 5.0
        case .moderate: return 7.0
        case .full: return 10.0
        }
    }
}

// MARK: - EditSnapshot

public struct EditSnapshot: Sendable {
    public let id: UUID
    public let timestamp: Date
    public let description: String
    /// Raw audio bytes (e.g. PCM or encoded audio).
    public let audioData: Data
    /// Encoded latent representation, if available.
    public let latents: Data?
    /// The request that produced this snapshot, if any.
    public let request: InpaintRequest?

    public init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        description: String,
        audioData: Data,
        latents: Data? = nil,
        request: InpaintRequest? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.description = description
        self.audioData = audioData
        self.latents = latents
        self.request = request
    }
}
