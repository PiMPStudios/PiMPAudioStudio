import Foundation
import Hub
import MLX
import MLXNN
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
    public let intermediateSize: Int?
    public let maxPositionEmbeddings: Int?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
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
    private var weights: [String: MLXArray]?

    /// Llama 3 EOS token IDs.
    private static let eosTokens: Set<Int> = [128001, 128009]

    /// Whether the model and tokenizer have been loaded successfully.
    public var isLoaded: Bool {
        tokenizer != nil && weights != nil
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

        // Placeholder: In production this would load safetensors weights
        // and build the model graph. For now, store an empty dict to signal
        // that "loading" succeeded.
        self.weights = [:]
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
        guard isLoaded, let tokenizer = self.tokenizer else {
            return AsyncStream { $0.finish() }
        }

        let prompt = Self.formatPrompt(system: systemPrompt, user: userPrompt)
        let inputIds = tokenizer.encode(text: prompt)

        return AsyncStream { continuation in
            Task { [weak self] in
                guard let self = self else {
                    continuation.finish()
                    return
                }
                await self.runGeneration(
                    inputIds: inputIds,
                    maxTokens: maxTokens,
                    temperature: temperature,
                    tokenizer: tokenizer,
                    continuation: continuation
                )
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
        guard isLoaded else { return "" }

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

    private func runGeneration(
        inputIds: [Int],
        maxTokens: Int,
        temperature: Float,
        tokenizer: any Tokenizer,
        continuation: AsyncStream<String>.Continuation
    ) {
        // Placeholder generation loop.
        // In production this would:
        // 1. Convert inputIds to MLXArray
        // 2. Run forward passes through the model
        // 3. Sample from logits with temperature
        // 4. Decode token IDs back to strings
        //
        // For now, immediately finish since no real model weights are loaded.
        for _ in 0..<maxTokens {
            if Task.isCancelled { break }

            // Placeholder: real implementation would sample next token here
            // let nextTokenId = sampleToken(logits: logits, temperature: temperature)
            // if Self.eosTokens.contains(nextTokenId) { break }
            // let tokenStr = tokenizer.decode(tokens: [nextTokenId])
            // continuation.yield(tokenStr)

            break // No real model to generate from
        }
        continuation.finish()
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
