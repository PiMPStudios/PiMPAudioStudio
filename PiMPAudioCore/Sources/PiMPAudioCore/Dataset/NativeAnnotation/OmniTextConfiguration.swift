import Foundation

public struct OmniTextConfiguration: Codable, Equatable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let rmsNormEpsilon: Float
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let ropeTraditional: Bool
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int

    public init(
        hiddenSize: Int = 3584,
        hiddenLayers: Int = 28,
        intermediateSize: Int = 18944,
        attentionHeads: Int = 28,
        keyValueHeads: Int = 4,
        rmsNormEpsilon: Float = 1e-6,
        vocabularySize: Int = 152_064,
        ropeTheta: Float = 1_000_000,
        ropeTraditional: Bool = false,
        tieWordEmbeddings: Bool = false,
        maxPositionEmbeddings: Int = 32768
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads
        self.rmsNormEpsilon = rmsNormEpsilon
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }

    public static let qwen25Omni7B = OmniTextConfiguration()

    public var headDimension: Int {
        hiddenSize / attentionHeads
    }

    public func validate() throws {
        guard hiddenSize > 0,
              hiddenLayers > 0,
              intermediateSize > 0,
              attentionHeads > 0,
              keyValueHeads > 0,
              hiddenSize.isMultiple(of: attentionHeads),
              attentionHeads.isMultiple(of: keyValueHeads),
              vocabularySize > 0,
              maxPositionEmbeddings > 0
        else {
            throw OmniTextConfigurationError.invalidDimensions
        }
        guard rmsNormEpsilon > 0, ropeTheta > 0 else {
            throw OmniTextConfigurationError.invalidNumerics
        }
    }
}

public struct OmniSpecialTokenConfiguration: Equatable, Sendable {
    public let audioToken: Int
    public let audioStartToken: Int
    public let audioEndToken: Int
    public let endTokenIDs: Set<Int>

    public init(
        audioToken: Int = 151_646,
        audioStartToken: Int = 151_647,
        audioEndToken: Int = 151_648,
        endTokenIDs: Set<Int> = [151_643, 151_645]
    ) {
        self.audioToken = audioToken
        self.audioStartToken = audioStartToken
        self.audioEndToken = audioEndToken
        self.endTokenIDs = endTokenIDs
    }
}

public enum OmniTextConfigurationError: LocalizedError {
    case invalidDimensions
    case invalidNumerics

    public var errorDescription: String? {
        switch self {
        case .invalidDimensions:
            "The Omni Thinker text configuration contains invalid dimensions."
        case .invalidNumerics:
            "The Omni Thinker text configuration contains invalid numeric values."
        }
    }
}

extension OmniTextConfiguration {
    static func fromCheckpoint(
        at modelDirectory: URL
    ) throws -> (configuration: Self, specialTokens: OmniSpecialTokenConfiguration) {
        struct Root: Decodable {
            struct Thinker: Decodable {
                struct Text: Decodable {
                    let hiddenSize: Int
                    let hiddenLayers: Int
                    let intermediateSize: Int
                    let attentionHeads: Int
                    let keyValueHeads: Int
                    let rmsNormEpsilon: Float
                    let vocabularySize: Int
                    let ropeTheta: Float?
                    let ropeTraditional: Bool?
                    let tieWordEmbeddings: Bool?
                    let maxPositionEmbeddings: Int

                    enum CodingKeys: String, CodingKey {
                        case hiddenSize = "hidden_size"
                        case hiddenLayers = "num_hidden_layers"
                        case intermediateSize = "intermediate_size"
                        case attentionHeads = "num_attention_heads"
                        case keyValueHeads = "num_key_value_heads"
                        case rmsNormEpsilon = "rms_norm_eps"
                        case vocabularySize = "vocab_size"
                        case ropeTheta = "rope_theta"
                        case ropeTraditional = "rope_traditional"
                        case tieWordEmbeddings = "tie_word_embeddings"
                        case maxPositionEmbeddings = "max_position_embeddings"
                    }
                }

                let textConfig: Text
                let audioTokenIndex: Int
                let audioStartTokenID: Int
                let audioEndTokenID: Int
                let endTokenID: Int
                let padTokenID: Int

                enum CodingKeys: String, CodingKey {
                    case textConfig = "text_config"
                    case audioTokenIndex = "audio_token_index"
                    case audioStartTokenID = "audio_start_token_id"
                    case audioEndTokenID = "audio_end_token_id"
                    case endTokenID = "eos_token_id"
                    case padTokenID = "pad_token_id"
                }
            }

            let thinkerConfig: Thinker

            enum CodingKeys: String, CodingKey {
                case thinkerConfig = "thinker_config"
            }
        }

        let configURL = modelDirectory.appendingPathComponent("config.json")
        let thinker = try JSONDecoder()
            .decode(Root.self, from: Data(contentsOf: configURL))
            .thinkerConfig
        let text = thinker.textConfig
        let configuration = Self(
            hiddenSize: text.hiddenSize,
            hiddenLayers: text.hiddenLayers,
            intermediateSize: text.intermediateSize,
            attentionHeads: text.attentionHeads,
            keyValueHeads: text.keyValueHeads,
            rmsNormEpsilon: text.rmsNormEpsilon,
            vocabularySize: text.vocabularySize,
            ropeTheta: text.ropeTheta ?? 1_000_000,
            ropeTraditional: text.ropeTraditional ?? false,
            tieWordEmbeddings: text.tieWordEmbeddings ?? false,
            maxPositionEmbeddings: text.maxPositionEmbeddings
        )
        try configuration.validate()
        let specialTokens = OmniSpecialTokenConfiguration(
            audioToken: thinker.audioTokenIndex,
            audioStartToken: thinker.audioStartTokenID,
            audioEndToken: thinker.audioEndTokenID,
            endTokenIDs: [thinker.endTokenID, thinker.padTokenID]
        )
        return (configuration, specialTokens)
    }
}
