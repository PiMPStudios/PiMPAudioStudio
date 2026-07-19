import Foundation

public struct OmniAudioConfiguration: Codable, Equatable, Sendable {
    public let sampleRate: Int
    public let featureSize: Int
    public let fftSize: Int
    public let hopLength: Int
    public let modelDimension: Int
    public let attentionWindow: Int
    public let attentionHeads: Int
    public let encoderLayers: Int
    public let maxSourcePositions: Int
    public let outputDimension: Int

    public init(
        sampleRate: Int = 16_000,
        featureSize: Int = 128,
        fftSize: Int = 400,
        hopLength: Int = 160,
        modelDimension: Int = 1_280,
        attentionWindow: Int = 100,
        attentionHeads: Int = 20,
        encoderLayers: Int = 32,
        maxSourcePositions: Int = 1_500,
        outputDimension: Int = 3_584
    ) {
        self.sampleRate = sampleRate
        self.featureSize = featureSize
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.modelDimension = modelDimension
        self.attentionWindow = attentionWindow
        self.attentionHeads = attentionHeads
        self.encoderLayers = encoderLayers
        self.maxSourcePositions = maxSourcePositions
        self.outputDimension = outputDimension
    }

    public static let qwen25Omni = OmniAudioConfiguration()

    public var headDimension: Int {
        modelDimension / attentionHeads
    }

    public var inputFramesPerAttentionWindow: Int {
        attentionWindow * 2
    }

    public func validate() throws {
        guard sampleRate > 0, featureSize > 0, fftSize > 0, hopLength > 0 else {
            throw OmniAudioConfigurationError.invalidFrontend
        }
        guard modelDimension > 0,
              attentionHeads > 0,
              modelDimension.isMultiple(of: attentionHeads),
              encoderLayers > 0,
              attentionWindow > 0,
              outputDimension > 0
        else {
            throw OmniAudioConfigurationError.invalidTower
        }
    }
}

public enum OmniAudioConfigurationError: LocalizedError {
    case invalidFrontend
    case invalidTower
    case checkpointMismatch(field: String, expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidFrontend:
            return "The audio feature configuration contains an invalid dimension."
        case .invalidTower:
            return "The audio tower configuration contains an invalid dimension."
        case .checkpointMismatch(let field, let expected, let actual):
            return "Checkpoint \(field) is \(actual); expected \(expected)."
        }
    }
}

extension OmniAudioConfiguration {
    static func fromCheckpoint(at modelDirectory: URL) throws -> Self {
        struct Root: Decodable {
            struct Thinker: Decodable {
                struct Audio: Decodable {
                    let numMelBins: Int
                    let dModel: Int
                    let nWindow: Int
                    let encoderAttentionHeads: Int
                    let encoderLayers: Int
                    let maxSourcePositions: Int
                    let outputDim: Int

                    enum CodingKeys: String, CodingKey {
                        case numMelBins = "num_mel_bins"
                        case dModel = "d_model"
                        case nWindow = "n_window"
                        case encoderAttentionHeads = "encoder_attention_heads"
                        case encoderLayers = "encoder_layers"
                        case maxSourcePositions = "max_source_positions"
                        case outputDim = "output_dim"
                    }
                }

                let audioConfig: Audio

                enum CodingKeys: String, CodingKey {
                    case audioConfig = "audio_config"
                }
            }

            let thinkerConfig: Thinker

            enum CodingKeys: String, CodingKey {
                case thinkerConfig = "thinker_config"
            }
        }

        struct Preprocessor: Decodable {
            let featureSize: Int
            let samplingRate: Int
            let hopLength: Int
            let fftSize: Int

            enum CodingKeys: String, CodingKey {
                case featureSize = "feature_size"
                case samplingRate = "sampling_rate"
                case hopLength = "hop_length"
                case fftSize = "n_fft"
            }
        }

        let configURL = modelDirectory.appendingPathComponent("config.json")
        let preprocessorURL = modelDirectory.appendingPathComponent(
            "preprocessor_config.json"
        )
        let data = try Data(contentsOf: configURL)
        let audio = try JSONDecoder().decode(Root.self, from: data).thinkerConfig.audioConfig
        let preprocessor = try JSONDecoder().decode(
            Preprocessor.self,
            from: Data(contentsOf: preprocessorURL)
        )
        guard audio.numMelBins == preprocessor.featureSize else {
            throw OmniAudioConfigurationError.checkpointMismatch(
                field: "num_mel_bins",
                expected: preprocessor.featureSize,
                actual: audio.numMelBins
            )
        }
        let configuration = Self(
            sampleRate: preprocessor.samplingRate,
            featureSize: audio.numMelBins,
            fftSize: preprocessor.fftSize,
            hopLength: preprocessor.hopLength,
            modelDimension: audio.dModel,
            attentionWindow: audio.nWindow,
            attentionHeads: audio.encoderAttentionHeads,
            encoderLayers: audio.encoderLayers,
            maxSourcePositions: audio.maxSourcePositions,
            outputDimension: audio.outputDim
        )
        try configuration.validate()
        return configuration
    }
}
