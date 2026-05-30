// PiMPAudioCore/Sources/PiMPAudioCore/Inference/LM/AceStepLMConfig.swift
import Foundation

/// Decoded form of an ACE-Step LM variant's `config.json`.
///
/// Both the 0.6B and 4B variants ship a Qwen-architecture LM. The model size
/// differs but the schema is identical, so we drive layer construction from
/// these values rather than hardcoding constants per variant.
public struct AceStepLMConfig: Sendable, Codable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
    }

    /// Loads from `config.json` on disk. The path comes from the
    /// `ace-step-lm-*` model spec's `localDirectory(base:)`.
    public static func load(from configURL: URL) throws -> AceStepLMConfig {
        let data = try Data(contentsOf: configURL)
        // Some checkpoints omit head_dim and rope_theta; fall back if needed.
        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let hiddenSize = (raw["hidden_size"] as? Int) ?? 1024
        let numLayers = (raw["num_hidden_layers"] as? Int) ?? 28
        let intermediate = (raw["intermediate_size"] as? Int) ?? 3072
        let nHeads = (raw["num_attention_heads"] as? Int) ?? 16
        let nKV = (raw["num_key_value_heads"] as? Int) ?? 8
        let hDim = (raw["head_dim"] as? Int) ?? (hiddenSize / nHeads)
        let eps = (raw["rms_norm_eps"] as? Double).map(Float.init) ?? 1e-6
        let vocab = (raw["vocab_size"] as? Int) ?? 64003
        let maxPos = (raw["max_position_embeddings"] as? Int) ?? 32768
        let rope = (raw["rope_theta"] as? Double).map(Float.init) ?? 1_000_000.0

        return AceStepLMConfig(
            hiddenSize: hiddenSize,
            numHiddenLayers: numLayers,
            intermediateSize: intermediate,
            numAttentionHeads: nHeads,
            numKeyValueHeads: nKV,
            headDim: hDim,
            rmsNormEps: eps,
            vocabSize: vocab,
            maxPositionEmbeddings: maxPos,
            ropeTheta: rope
        )
    }
}
