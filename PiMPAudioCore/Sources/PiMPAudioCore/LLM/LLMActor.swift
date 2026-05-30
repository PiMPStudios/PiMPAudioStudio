import Foundation
import Hub
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Tokenizers

// MARK: - Errors

/// Errors thrown by ``LLMActor`` during model loading and generation.
public enum LLMError: Error, LocalizedError {
    case modelNotFound(URL)
    case missingFile(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let url):
            return "Model directory not found: \(url.path)"
        case .missingFile(let name):
            return "Required file missing: \(name)"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        }
    }
}

// MARK: - Config

/// Minimal representation of a model's `config.json` for LLM loading.
public struct LLMConfig: Codable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int?
    public let maxPositionEmbeddings: Int?
    public let rmsNormEps: Float
    public let ropeTheta: Float?
    public let quantization: QuantizationConfig?
    public let quantizationConfig: QuantizationConfig?
    public let tieWordEmbeddings: Bool?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case quantization
        case quantizationConfig = "quantization_config"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.vocabSize = try c.decode(Int.self, forKey: CodingKeys.vocabSize)
        self.hiddenSize = try c.decode(Int.self, forKey: CodingKeys.hiddenSize)
        self.numHiddenLayers = try c.decode(Int.self, forKey: CodingKeys.numHiddenLayers)
        self.numAttentionHeads = try c.decode(Int.self, forKey: CodingKeys.numAttentionHeads)
        self.numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: CodingKeys.numKeyValueHeads) ?? numAttentionHeads
        self.headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? (hiddenSize / numAttentionHeads)
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize)
        self.maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxPositionEmbeddings)
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-5
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: CodingKeys.ropeTheta)
        self.quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: CodingKeys.quantization)
        self.quantizationConfig = try c.decodeIfPresent(QuantizationConfig.self, forKey: CodingKeys.quantizationConfig)
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings)
    }
}

public struct QuantizationConfig: Codable, Sendable {
    public let groupSize: Int
    public let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

// MARK: - KV Cache

final class LLMKeyValueCache {
    private var entries: [Int: (key: MLXArray, value: MLXArray)] = [:]

    var sequenceLength: Int {
        entries[0]?.key.shape[2] ?? 0
    }

    func get(layer: Int) -> (key: MLXArray, value: MLXArray)? {
        entries[layer]
    }

    func update(key: MLXArray, value: MLXArray, layer: Int) {
        entries[layer] = (key, value)
    }

    var arrays: [MLXArray] {
        entries.keys.sorted().flatMap { layer -> [MLXArray] in
            guard let entry = entries[layer] else { return [] }
            return [entry.key, entry.value]
        }
    }

    func reset() {
        entries.removeAll()
    }
}

// MARK: - Actor

/// Actor providing local LLM inference using MLX-Swift.
///
/// Loads model weights and tokenizer from a local directory and exposes
/// streaming (`generate`) and non-streaming (`classify`) text generation.
public actor LLMActor {
    private let modelPath: URL
    private var tokenizer: (any Tokenizer)?
    private var config: LLMConfig?
    private var arrays: [String: MLXArray]?

    /// Llama 3 EOS token IDs.
    private static let eosTokens: Set<Int> = [128001, 128008, 128009]

    /// Whether the model and tokenizer have been loaded successfully.
    public var isLoaded: Bool {
        tokenizer != nil && arrays != nil && config != nil
    }

    // MARK: - Init

    /// Create an LLMActor pointing at a model directory.
    ///
    /// - Parameter modelPath: Directory containing config.json, tokenizer.json,
    ///   and safetensors weight files.
    public init(modelPath: URL) {
        self.modelPath = modelPath
    }

    // MARK: - Loading

    /// Load the model weights and tokenizer from the configured directory.
    ///
    /// Verifies that the directory exists and contains the required files
    /// (`config.json`, `tokenizer.json`) before attempting to load.
    public func loadModel() async throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false

        guard fm.fileExists(atPath: modelPath.path, isDirectory: &isDir), isDir.boolValue else {
            throw LLMError.modelNotFound(modelPath)
        }

        let configURL = modelPath.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: configURL.path) else {
            throw LLMError.missingFile("config.json")
        }

        let tokenizerURL = modelPath.appendingPathComponent("tokenizer.json")
        guard fm.fileExists(atPath: tokenizerURL.path) else {
            throw LLMError.missingFile("tokenizer.json")
        }

        // Parse config.json
        let configData = try Data(contentsOf: configURL)
        self.config = try JSONDecoder().decode(LLMConfig.self, from: configData)

        // Load tokenizer
        let tokenizerData = try Data(contentsOf: tokenizerURL)
        let tokenizerParsed = try JSONSerialization.jsonObject(with: tokenizerData, options: [])
        guard let tokenizerDict = tokenizerParsed as? [NSString: Any] else {
            throw LLMError.missingFile("tokenizer.json (invalid format)")
        }
        let tokenizerConfig = Config(tokenizerDict)

        // Check for tokenizer_config.json (optional)
        let tokenizerConfigURL = modelPath.appendingPathComponent("tokenizer_config.json")
        let baseConfig: Config
        if fm.fileExists(atPath: tokenizerConfigURL.path) {
            let data = try Data(contentsOf: tokenizerConfigURL)
            let parsed = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dict = parsed as? [NSString: Any] else {
                throw LLMError.missingFile("tokenizer_config.json (invalid format)")
            }
            baseConfig = Config(dict)
        } else {
            baseConfig = Config(["tokenizer_class": "PreTrainedTokenizer"] as [NSString: Any])
        }

        self.tokenizer = try AutoTokenizer.from(
            tokenizerConfig: baseConfig,
            tokenizerData: tokenizerConfig
        )

        var loadedArrays: [String: MLXArray] = [:]
        let entries = try fm.contentsOfDirectory(at: modelPath, includingPropertiesForKeys: nil)
        let weightFiles = entries
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !weightFiles.isEmpty else {
            throw LLMError.missingFile("model.safetensors")
        }
        for file in weightFiles {
            let fileArrays = try MLX.loadArrays(url: file)
            for (key, value) in fileArrays {
                loadedArrays[key] = value
            }
        }
        guard loadedArrays["model.embed_tokens.weight"] != nil,
              loadedArrays["model.norm.weight"] != nil else {
            throw LLMError.missingFile("Llama model weights")
        }
        self.arrays = loadedArrays
    }

    // MARK: - Generation

    /// Stream tokens from the LLM given system and user prompts.
    ///
    /// Returns an immediately-finishing stream if the model is not loaded.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system instruction.
    ///   - userPrompt: The user message.
    ///   - maxTokens: Maximum tokens to generate. Default `512`.
    ///   - temperature: Sampling temperature. Default `0.7`.
    /// - Returns: An `AsyncStream` of token strings.
    public func generate(
        systemPrompt: String,
        userPrompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7
    ) -> AsyncStream<String> {
        let prompt = Self.formatPrompt(system: systemPrompt, user: userPrompt)

        return AsyncStream { continuation in
            Task { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    try await self.ensureLoaded()
                    guard let tokenizer = await self.tokenizer else {
                        continuation.finish()
                        return
                    }
                    let inputIds = tokenizer.encode(text: prompt)
                    try await self.runGeneration(
                        inputIds: inputIds,
                        maxTokens: maxTokens,
                        temperature: temperature,
                        tokenizer: tokenizer,
                        continuation: continuation
                    )
                } catch {
                    continuation.yield("\n[Lyric writer unavailable: \(error.localizedDescription)]")
                    continuation.finish()
                }
            }
        }
    }

    /// Non-streaming classification with temperature 0.
    ///
    /// Returns an empty string if the model is not loaded.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system instruction.
    ///   - userPrompt: The user message.
    /// - Returns: The complete generated text.
    public func classify(
        systemPrompt: String,
        userPrompt: String
    ) async -> String {
        // generate() calls ensureLoaded() internally — no early guard needed.
        // The model is loaded on first use rather than requiring a separate load step.
        var result = ""
        let stream = generate(
            systemPrompt: systemPrompt,
            userPrompt: userPrompt,
            maxTokens: 20,
            temperature: 0.0
        )
        for await token in stream {
            result += token
        }
        return result
    }

    // MARK: - Private

    private func ensureLoaded() async throws {
        if !isLoaded {
            try await loadModel()
        }
    }

    private func runGeneration(
        inputIds: [Int],
        maxTokens: Int,
        temperature: Float,
        tokenizer: any Tokenizer,
        continuation: AsyncStream<String>.Continuation
    ) throws {
        let cache = LLMKeyValueCache()
        var nextInputIds = inputIds
        var key = MLXRandom.key(UInt64(Date().timeIntervalSince1970))

        for _ in 0..<maxTokens {
            if Task.isCancelled { break }

            let logits = try forward(inputIds: nextInputIds, cache: cache)
            let lastIndex = logits.shape[1] - 1
            var nextLogits = logits[0, lastIndex, 0...]
            if temperature > 0 {
                nextLogits = nextLogits / temperature
            }

            let nextArray: MLXArray
            if temperature <= 0 {
                nextArray = nextLogits.argMax()
            } else {
                nextArray = MLXRandom.categorical(nextLogits, key: key)
                key = MLXRandom.split(key: key, into: 2)[1]
            }
            eval(nextArray)

            let nextToken = nextArray.item(Int.self)
            if Self.eosTokens.contains(nextToken) { break }

            nextInputIds = [nextToken]
            continuation.yield(tokenizer.decode(tokens: [nextToken]))
        }
        cache.reset()
        continuation.finish()
    }

    private func forward(inputIds: [Int], cache: LLMKeyValueCache?) throws -> MLXArray {
        guard let config, let arrays else {
            throw LLMError.generationFailed("model is not loaded")
        }
        let ids = MLXArray(inputIds.map(Int32.init)).reshaped([1, inputIds.count])
        var hidden = try embed(ids, arrays: arrays, config: config)
        let cachedLength = cache?.sequenceLength ?? 0
        let mask = causalMask(queryLength: inputIds.count, keyLength: cachedLength + inputIds.count, queryOffset: cachedLength)
        let rotary = rotaryEmbedding(seqLen: inputIds.count, positionOffset: cachedLength, config: config)

        for layer in 0..<config.numHiddenLayers {
            hidden = try llamaLayer(
                hidden,
                layer: layer,
                mask: mask,
                rotary: rotary,
                arrays: arrays,
                config: config,
                cache: cache
            )
        }

        hidden = try rmsNorm(hidden, weightKey: "model.norm.weight", arrays: arrays, eps: config.rmsNormEps)
        let lastHidden = hidden[0..., (hidden.shape[1] - 1)..<hidden.shape[1], 0...]
        let logits = try outputLogits(lastHidden, arrays: arrays, config: config)
        if let cache {
            eval(cache.arrays + [logits])
        }
        return logits
    }

    private func llamaLayer(
        _ hidden: MLXArray,
        layer: Int,
        mask: MLXArray?,
        rotary: (cos: MLXArray, sin: MLXArray),
        arrays: [String: MLXArray],
        config: LLMConfig,
        cache: LLMKeyValueCache?
    ) throws -> MLXArray {
        let prefix = "model.layers.\(layer)"
        let normed = try rmsNorm(hidden, weightKey: "\(prefix).input_layernorm.weight", arrays: arrays, eps: config.rmsNormEps)
        let attn = try attention(
            normed,
            prefix: "\(prefix).self_attn",
            layer: layer,
            mask: mask,
            rotary: rotary,
            arrays: arrays,
            config: config,
            cache: cache
        )
        var h = hidden + attn

        let post = try rmsNorm(h, weightKey: "\(prefix).post_attention_layernorm.weight", arrays: arrays, eps: config.rmsNormEps)
        let gate = try qLinear(post, prefix: "\(prefix).mlp.gate_proj", arrays: arrays, config: config)
        let up = try qLinear(post, prefix: "\(prefix).mlp.up_proj", arrays: arrays, config: config)
        let mlp = try qLinear(silu(gate) * up, prefix: "\(prefix).mlp.down_proj", arrays: arrays, config: config)
        h = h + mlp
        return h
    }

    private func attention(
        _ hidden: MLXArray,
        prefix: String,
        layer: Int,
        mask: MLXArray?,
        rotary: (cos: MLXArray, sin: MLXArray),
        arrays: [String: MLXArray],
        config: LLMConfig,
        cache: LLMKeyValueCache?
    ) throws -> MLXArray {
        let b = hidden.shape[0]
        let seqLen = hidden.shape[1]
        let numHeads = config.numAttentionHeads
        let numKVHeads = config.numKeyValueHeads
        let headDim = config.headDim
        let nRep = numHeads / numKVHeads

        var q = try qLinear(hidden, prefix: "\(prefix).q_proj", arrays: arrays, config: config)
            .reshaped([b, seqLen, numHeads, headDim])
            .transposed(0, 2, 1, 3)
        var k = try qLinear(hidden, prefix: "\(prefix).k_proj", arrays: arrays, config: config)
            .reshaped([b, seqLen, numKVHeads, headDim])
            .transposed(0, 2, 1, 3)
        let v = try qLinear(hidden, prefix: "\(prefix).v_proj", arrays: arrays, config: config)
            .reshaped([b, seqLen, numKVHeads, headDim])
            .transposed(0, 2, 1, 3)

        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: rotary.cos, sin: rotary.sin)

        var allKeys = k
        var allValues = v
        if let cached = cache?.get(layer: layer) {
            allKeys = concatenated([cached.key, k], axis: 2)
            allValues = concatenated([cached.value, v], axis: 2)
        }
        cache?.update(key: allKeys, value: allValues, layer: layer)

        let kExpanded = repeatKV(allKeys, nRep: nRep)
        let vExpanded = repeatKV(allValues, nRep: nRep)
        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: kExpanded,
            values: vExpanded,
            scale: pow(Float(headDim), -0.5),
            mask: mask
        )
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, seqLen, numHeads * headDim])
        return try qLinear(merged, prefix: "\(prefix).o_proj", arrays: arrays, config: config)
    }

    private func embed(_ ids: MLXArray, arrays: [String: MLXArray], config: LLMConfig) throws -> MLXArray {
        guard let weight = arrays["model.embed_tokens.weight"],
              let scales = arrays["model.embed_tokens.scales"] else {
            throw LLMError.missingFile("model.embed_tokens")
        }
        let biases = arrays["model.embed_tokens.biases"]
        return dequantized(
            weight[ids],
            scales: scales[ids],
            biases: biases == nil ? nil : biases![ids],
            groupSize: quantization(config).groupSize,
            bits: quantization(config).bits
        )
    }

    private func outputLogits(_ hidden: MLXArray, arrays: [String: MLXArray], config: LLMConfig) throws -> MLXArray {
        if arrays["lm_head.weight"] != nil {
            return try qLinear(hidden, prefix: "lm_head", arrays: arrays, config: config)
        }
        guard let weight = arrays["model.embed_tokens.weight"],
              let scales = arrays["model.embed_tokens.scales"] else {
            throw LLMError.missingFile("lm_head/model.embed_tokens")
        }
        return quantizedMM(
            hidden,
            weight,
            scales: scales,
            biases: arrays["model.embed_tokens.biases"],
            transpose: true,
            groupSize: quantization(config).groupSize,
            bits: quantization(config).bits
        )
    }

    private func qLinear(
        _ x: MLXArray,
        prefix: String,
        arrays: [String: MLXArray],
        config: LLMConfig
    ) throws -> MLXArray {
        guard let weight = arrays["\(prefix).weight"],
              let scales = arrays["\(prefix).scales"] else {
            throw LLMError.missingFile(prefix)
        }
        return quantizedMM(
            x,
            weight,
            scales: scales,
            biases: arrays["\(prefix).biases"],
            transpose: true,
            groupSize: quantization(config).groupSize,
            bits: quantization(config).bits
        )
    }

    private func rmsNorm(
        _ x: MLXArray,
        weightKey: String,
        arrays: [String: MLXArray],
        eps: Float
    ) throws -> MLXArray {
        guard let weight = arrays[weightKey] else {
            throw LLMError.missingFile(weightKey)
        }
        let variance = (x * x).mean(axis: -1, keepDims: true)
        return x * rsqrt(variance + eps) * weight
    }

    private func causalMask(queryLength: Int, keyLength: Int, queryOffset: Int) -> MLXArray? {
        guard keyLength > 1 else { return nil }
        if queryLength == 1 && queryOffset == keyLength - 1 {
            return nil
        }
        let queryPositions = MLXArray(Array(queryOffset..<(queryOffset + queryLength)).map(Float.init))
        let keyPositions = MLXArray(Array(0..<keyLength).map(Float.init))
        let row = queryPositions.expandedDimensions(axis: 1)
        let col = keyPositions.expandedDimensions(axis: 0)
        let zeros = MLXArray.zeros([queryLength, keyLength])
        let negInf = MLXArray.full([queryLength, keyLength], values: MLXArray(Float(-1e9)))
        return which(col .<= row, zeros, negInf).reshaped([1, 1, queryLength, keyLength])
    }

    private func rotaryEmbedding(
        seqLen: Int,
        positionOffset: Int,
        config: LLMConfig
    ) -> (cos: MLXArray, sin: MLXArray) {
        let headDim = config.headDim
        let base = config.ropeTheta ?? 500_000.0
        let invFreq = 1.0 / pow(
            MLXArray(base),
            MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) })
        )
        let positions = MLXArray(Array(positionOffset..<(positionOffset + seqLen)).map(Float.init))
        let freqs = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let freqsFull = concatenated([freqs, freqs], axis: 1)
        let c = cos(freqsFull).reshaped([1, 1, seqLen, headDim])
        let s = sin(freqsFull).reshaped([1, 1, seqLen, headDim])
        return (c, s)
    }

    private func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
        if nRep == 1 { return x }
        let (b, nKV, seqLen, dim) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let expanded = x.expandedDimensions(axis: 2)
        let broadcasted = broadcast(expanded, to: [b, nKV, nRep, seqLen, dim])
        return broadcasted.reshaped([b, nKV * nRep, seqLen, dim])
    }

    private func quantization(_ config: LLMConfig) -> QuantizationConfig {
        config.quantizationConfig ?? config.quantization ?? QuantizationConfig(groupSize: 64, bits: 4)
    }

    /// Format a prompt using the Llama 3 chat template.
    static func formatPrompt(system: String, user: String) -> String {
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
        \(system)<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
        \(user)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        """
    }
}
