import Foundation
import MLX
import MLXNN

public struct OmniGenerationOptions: Equatable, Sendable {
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let repetitionPenalty: Float?

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0,
        topP: Float = 0,
        repetitionPenalty: Float? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
    }

    public static let greedy = OmniGenerationOptions()
}

public struct OmniAnnotationResult: Sendable {
    public let task: OmniAnnotationTask
    public let rawText: String
    public let text: String
    public let language: String?
    public let generatedTokenIDs: [Int]
    public let promptTokenCount: Int
    public let elapsedSeconds: Double
    public let memory: Memory.Snapshot

    public init(
        task: OmniAnnotationTask,
        rawText: String,
        text: String,
        language: String?,
        generatedTokenIDs: [Int],
        promptTokenCount: Int,
        elapsedSeconds: Double,
        memory: Memory.Snapshot
    ) {
        self.task = task
        self.rawText = rawText
        self.text = text
        self.language = language
        self.generatedTokenIDs = generatedTokenIDs
        self.promptTokenCount = promptTokenCount
        self.elapsedSeconds = elapsedSeconds
        self.memory = memory
    }
}

public struct OmniFirstTokenProjection: Sendable {
    public let tokenID: Int
    public let topTokenIDs: [Int]
    public let topLogProbabilities: [Float]
    public let promptTokenCount: Int

    public init(
        tokenID: Int,
        topTokenIDs: [Int],
        topLogProbabilities: [Float],
        promptTokenCount: Int
    ) {
        self.tokenID = tokenID
        self.topTokenIDs = topTokenIDs
        self.topLogProbabilities = topLogProbabilities
        self.promptTokenCount = promptTokenCount
    }
}

public struct OmniEngineLoadReport: Sendable {
    public let thinker: OmniThinkerWeightLoadReport
    public let audioTower: OmniWeightLoadReport
    public let memory: Memory.Snapshot

    public init(
        thinker: OmniThinkerWeightLoadReport,
        audioTower: OmniWeightLoadReport,
        memory: Memory.Snapshot
    ) {
        self.thinker = thinker
        self.audioTower = audioTower
        self.memory = memory
    }
}

public final class OmniAnnotationEngine {
    private var thinker: OmniQwen2ForCausalLM?
    private var audioTower: Qwen25OmniAudioTower?
    private var promptBuilder: OmniPromptBuilder?
    private let featureExtractor: OmniAudioFeatureExtractor

    public private(set) var loadReport: OmniEngineLoadReport?
    public private(set) var modelDirectory: URL?
    public private(set) var modelPackTask: OmniAnnotationTask?

    public init() throws {
        self.featureExtractor = try OmniAudioFeatureExtractor()
    }

    public var isLoaded: Bool {
        thinker != nil && audioTower != nil && promptBuilder != nil
    }

    @discardableResult
    public func load(
        from modelDirectory: URL,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) async throws -> OmniEngineLoadReport {
        evict()
        try cancellationCheck()
        let manifest = try NativeModelPackValidator.loadIfPresent(
            at: modelDirectory,
            cancellationCheck: cancellationCheck
        )
        let thinkerLoad = try OmniThinkerWeightLoader.load(
            from: modelDirectory,
            validatedManifest: manifest,
            cancellationCheck: cancellationCheck
        )
        do {
            try cancellationCheck()
            let towerLoad = try OmniWeightLoader.loadAudioTower(
                from: modelDirectory,
                validatedManifest: manifest,
                cancellationCheck: cancellationCheck
            )
            try cancellationCheck()
            let builder = try await OmniPromptBuilder.load(
                from: modelDirectory,
                specialTokens: thinkerLoad.specialTokens
            )
            thinker = thinkerLoad.model
            audioTower = towerLoad.tower
            promptBuilder = builder
            self.modelDirectory = modelDirectory
            self.modelPackTask = manifest?.task
            let report = OmniEngineLoadReport(
                thinker: thinkerLoad.report,
                audioTower: towerLoad.report,
                memory: Memory.snapshot()
            )
            loadReport = report
            return report
        } catch {
            evict()
            throw error
        }
    }

    public func preparePrompt(
        task: OmniAnnotationTask,
        audioEmbeddingCount: Int
    ) throws -> OmniPreparedPrompt {
        guard let promptBuilder else {
            throw OmniAnnotationEngineError.notLoaded
        }
        return try promptBuilder.prepare(
            task: task,
            audioEmbeddingCount: audioEmbeddingCount
        )
    }

    public func projectedAudioEmbeddings(
        samples: [Float],
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> MLXArray {
        guard let audioTower else {
            throw OmniAnnotationEngineError.notLoaded
        }
        let logMel = try featureExtractor.logMelSpectrogram(waveform: samples)
        return try audioTower(
            logMel,
            cancellationCheck: cancellationCheck
        ).projectedAudioEmbeddings
    }

    public func firstTokenProjection(
        samples: [Float],
        task: OmniAnnotationTask,
        topK: Int = 16,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> OmniFirstTokenProjection {
        let audioEmbeddings = try projectedAudioEmbeddings(
            samples: samples,
            cancellationCheck: cancellationCheck
        )
        let prompt = try preparePrompt(
            task: task,
            audioEmbeddingCount: audioEmbeddings.dim(1)
        )
        return try firstTokenProjection(
            audioEmbeddings: audioEmbeddings,
            prompt: prompt,
            topK: topK,
            cancellationCheck: cancellationCheck
        )
    }

    public func firstTokenProjection(
        audioEmbeddings: MLXArray,
        prompt: OmniPreparedPrompt,
        topK: Int = 16,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> OmniFirstTokenProjection {
        guard let thinker else {
            throw OmniAnnotationEngineError.notLoaded
        }
        let cache = thinker.newCache(
            allocationStep: prompt.expandedTokenIDs.count
        )
        let embeddings = try prompt.mixedEmbeddings(
            using: thinker.model,
            audioEmbeddings: audioEmbeddings
        )
        let logits = try thinker.logits(
            embeddings: embeddings,
            cache: cache,
            cancellationCheck: cancellationCheck
        )
        let finalLogits = logits[
            0,
            (logits.dim(1) - 1),
            0...
        ].asType(.float32)
        let logProbabilities = logSoftmax(finalLogits, axis: -1)
        eval(logProbabilities)
        let values = logProbabilities.asArray(Float.self)
        let sorted = values.indices.sorted {
            values[$0] > values[$1]
        }
        let count = min(max(1, topK), sorted.count)
        let indices = Array(sorted.prefix(count))
        return OmniFirstTokenProjection(
            tokenID: indices[0],
            topTokenIDs: indices,
            topLogProbabilities: indices.map { values[$0] },
            promptTokenCount: embeddings.dim(1)
        )
    }

    public func annotate(
        audioURL: URL,
        task: OmniAnnotationTask,
        options: OmniGenerationOptions = .greedy,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> OmniAnnotationResult {
        let audio = try OmniAudioFileLoader.loadMono16k(from: audioURL)
        return try annotate(
            samples: audio.samples,
            task: task,
            options: options,
            cancellationCheck: cancellationCheck
        )
    }

    public func annotate(
        samples: [Float],
        task: OmniAnnotationTask,
        options: OmniGenerationOptions = .greedy,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> OmniAnnotationResult {
        guard let thinker, let promptBuilder else {
            throw OmniAnnotationEngineError.notLoaded
        }
        if let modelPackTask, modelPackTask != task {
            throw OmniAnnotationEngineError.modelPackTaskMismatch(
                expected: modelPackTask,
                actual: task
            )
        }
        try options.validate()
        let started = ContinuousClock.now
        let audioEmbeddings = try projectedAudioEmbeddings(
            samples: samples,
            cancellationCheck: cancellationCheck
        )
        let prompt = try promptBuilder.prepare(
            task: task,
            audioEmbeddingCount: audioEmbeddings.dim(1)
        )
        let mixed = try prompt.mixedEmbeddings(
            using: thinker.model,
            audioEmbeddings: audioEmbeddings
        )
        let cache = thinker.newCache(
            allocationStep: max(256, mixed.dim(1))
        )
        var logits = try thinker.logits(
            embeddings: mixed,
            cache: cache,
            cancellationCheck: cancellationCheck
        )
        var generated: [Int] = []
        generated.reserveCapacity(options.maxTokens)

        for _ in 0..<options.maxTokens {
            try cancellationCheck()
            let finalLogits = logits[
                0,
                (logits.dim(1) - 1),
                0...
            ]
            let next = try Self.selectToken(
                logits: finalLogits,
                generated: generated,
                options: options
            )
            if promptBuilder.specialTokens.endTokenIDs.contains(next) {
                break
            }
            generated.append(next)
            let nextInput = MLXArray([Int32(next)], [1, 1])
            logits = try thinker.logits(
                tokenIDs: nextInput,
                cache: cache,
                cancellationCheck: cancellationCheck
            )
        }

        let raw = promptBuilder.decode(generated)
        let parsed = Self.parse(text: raw, task: task)
        return OmniAnnotationResult(
            task: task,
            rawText: raw,
            text: parsed.text,
            language: parsed.language,
            generatedTokenIDs: generated,
            promptTokenCount: mixed.dim(1),
            elapsedSeconds: started.duration(to: .now).secondsValue,
            memory: Memory.snapshot()
        )
    }

    public func evict() {
        thinker = nil
        audioTower = nil
        promptBuilder = nil
        loadReport = nil
        modelDirectory = nil
        modelPackTask = nil
        Memory.clearCache()
    }

    private static func selectToken(
        logits: MLXArray,
        generated: [Int],
        options: OmniGenerationOptions
    ) throws -> Int {
        if options.temperature == 0,
           options.topP == 0,
           options.repetitionPenalty == nil
        {
            return Int(
                argMax(logits, axis: -1)
                    .asType(.int32)
                    .item(Int32.self)
            )
        }

        var values = logits.asType(.float32).asArray(Float.self)
        if let penalty = options.repetitionPenalty, penalty > 1 {
            for token in Set(generated) where values.indices.contains(token) {
                values[token] =
                    values[token] < 0
                    ? values[token] * penalty
                    : values[token] / penalty
            }
        }
        if options.temperature <= 0 {
            guard let best = values.indices.max(by: {
                values[$0] < values[$1]
            }) else {
                throw OmniAnnotationEngineError.emptyLogits
            }
            return best
        }

        let scaled = values.map { $0 / options.temperature }
        let maximum = scaled.max() ?? 0
        var probabilities = scaled.map { exp($0 - maximum) }
        let total = probabilities.reduce(0, +)
        guard total.isFinite, total > 0 else {
            throw OmniAnnotationEngineError.invalidProbabilities
        }
        probabilities = probabilities.map { $0 / total }

        var candidates = probabilities.indices.sorted {
            probabilities[$0] > probabilities[$1]
        }
        if options.topP > 0, options.topP < 1 {
            var cumulative: Float = 0
            candidates = Array(
                candidates.prefix {
                    let include = cumulative < options.topP
                    cumulative += probabilities[$0]
                    return include
                }
            )
        }
        let candidateTotal = candidates.reduce(Float(0)) {
            $0 + probabilities[$1]
        }
        var draw = Float.random(in: 0..<candidateTotal)
        for candidate in candidates {
            draw -= probabilities[candidate]
            if draw <= 0 {
                return candidate
            }
        }
        guard let final = candidates.last else {
            throw OmniAnnotationEngineError.invalidProbabilities
        }
        return final
    }

    private static func parse(
        text raw: String,
        task: OmniAnnotationTask
    ) -> (text: String, language: String?) {
        guard task == .transcribe else {
            return (raw, nil)
        }
        let lines = raw.components(separatedBy: .newlines)
        var language: String?
        if let marker = lines.firstIndex(where: {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
                .hasPrefix("# language")
        }) {
            language = lines.dropFirst(marker + 1)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .first(where: { !$0.isEmpty })?
                .lowercased()
        }

        if let marker = lines.firstIndex(where: {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
                .hasPrefix("# lyrics")
        }) {
            let lyrics = lines.dropFirst(marker + 1)
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !lyrics.isEmpty {
                return (lyrics, language)
            }
        }
        return (raw, language)
    }
}

public enum OmniAnnotationEngineError: LocalizedError {
    case notLoaded
    case emptyLogits
    case invalidProbabilities
    case invalidGenerationOptions(String)
    case modelPackTaskMismatch(
        expected: OmniAnnotationTask,
        actual: OmniAnnotationTask
    )

    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "The native Omni annotation engine is not loaded."
        case .emptyLogits:
            return "The language model produced no logits."
        case .invalidProbabilities:
            return "The language-model sampling probabilities are invalid."
        case .invalidGenerationOptions(let detail):
            return "Invalid generation options: \(detail)"
        case .modelPackTaskMismatch(let expected, let actual):
            return "The \(expected.rawValue) model pack cannot run a "
                + "\(actual.rawValue) request."
        }
    }
}

private extension OmniGenerationOptions {
    func validate() throws {
        guard maxTokens > 0 else {
            throw OmniAnnotationEngineError.invalidGenerationOptions(
                "maxTokens must be greater than zero"
            )
        }
        guard temperature >= 0 else {
            throw OmniAnnotationEngineError.invalidGenerationOptions(
                "temperature cannot be negative"
            )
        }
        guard topP >= 0, topP <= 1 else {
            throw OmniAnnotationEngineError.invalidGenerationOptions(
                "topP must be between zero and one"
            )
        }
        if let repetitionPenalty, repetitionPenalty < 1 {
            throw OmniAnnotationEngineError.invalidGenerationOptions(
                "repetitionPenalty must be at least one"
            )
        }
    }
}

private extension Duration {
    var secondsValue: Double {
        let components = self.components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }
}
