import Foundation
import MLX
import MLXNN

public struct OmniThinkerWeightLoadReport: Sendable {
    public let modelDirectory: URL
    public let shardNames: [String]
    public let tensorCount: Int
    public let loadSeconds: Double

    public init(
        modelDirectory: URL,
        shardNames: [String],
        tensorCount: Int,
        loadSeconds: Double
    ) {
        self.modelDirectory = modelDirectory
        self.shardNames = shardNames
        self.tensorCount = tensorCount
        self.loadSeconds = loadSeconds
    }
}

public enum OmniThinkerWeightLoaderError: LocalizedError {
    case missingIndex(URL)
    case missingShard(URL)
    case missingTensor(String)
    case incorrectTensorCount(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case let .missingIndex(url):
            "Missing safetensors index at \(url.path)."
        case let .missingShard(url):
            "Missing checkpoint shard at \(url.path)."
        case let .missingTensor(name):
            "Checkpoint shard does not contain \(name)."
        case let .incorrectTensorCount(expected, actual):
            "Loaded \(actual) Thinker text tensors; expected \(expected)."
        }
    }
}

public enum OmniThinkerWeightLoader {
    public static let checkpointPrefix = "thinker."
    public static let expectedParameterCount = 339

    private struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    @discardableResult
    public static func load(
        from modelDirectory: URL,
        evaluateWeights: Bool = true,
        validatedManifest: NativeModelPackManifest? = nil,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> (
        model: OmniQwen2ForCausalLM,
        specialTokens: OmniSpecialTokenConfiguration,
        report: OmniThinkerWeightLoadReport
    ) {
        let started = ContinuousClock.now
        let parsed = try OmniTextConfiguration.fromCheckpoint(
            at: modelDirectory
        )
        let model = OmniQwen2ForCausalLM(configuration: parsed.configuration)
        let report = try loadWeights(
            into: model,
            from: modelDirectory,
            evaluateWeights: evaluateWeights,
            validatedManifest: validatedManifest,
            started: started,
            cancellationCheck: cancellationCheck
        )
        return (model, parsed.specialTokens, report)
    }

    @discardableResult
    public static func loadWeights(
        into model: OmniQwen2ForCausalLM,
        from modelDirectory: URL,
        evaluateWeights: Bool = true,
        validatedManifest: NativeModelPackManifest? = nil,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> OmniThinkerWeightLoadReport {
        try loadWeights(
            into: model,
            from: modelDirectory,
            evaluateWeights: evaluateWeights,
            validatedManifest: validatedManifest,
            started: ContinuousClock.now,
            cancellationCheck: cancellationCheck
        )
    }

    private static func loadWeights(
        into model: OmniQwen2ForCausalLM,
        from modelDirectory: URL,
        evaluateWeights: Bool,
        validatedManifest: NativeModelPackManifest?,
        started: ContinuousClock.Instant,
        cancellationCheck: OmniCancellationCheck
    ) throws -> OmniThinkerWeightLoadReport {
        let manifest = try validatedManifest
            ?? NativeModelPackValidator.loadIfPresent(at: modelDirectory)
        if let manifest {
            NativeModelQuantizer.quantize(
                model: model,
                using: manifest.quantization
            )
        }
        let indexURL = modelDirectory.appendingPathComponent(
            "model.safetensors.index.json"
        )
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            throw OmniThinkerWeightLoaderError.missingIndex(indexURL)
        }
        let index = try JSONDecoder().decode(
            SafetensorsIndex.self,
            from: Data(contentsOf: indexURL)
        )
        let indexed = index.weightMap.filter {
            $0.key.hasPrefix("thinker.model.")
                || $0.key.hasPrefix("thinker.lm_head.")
        }
        let shardNames = Array(Set(indexed.values)).sorted()
        var selected: [(String, MLXArray)] = []
        selected.reserveCapacity(expectedParameterCount)

        for shardName in shardNames {
            try cancellationCheck()
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw OmniThinkerWeightLoaderError.missingShard(shardURL)
            }
            let shard = try MLX.loadArrays(url: shardURL)
            let names = indexed
                .filter { $0.value == shardName }
                .map(\.key)
                .sorted()
            for fullName in names {
                try cancellationCheck()
                guard let value = shard[fullName] else {
                    throw OmniThinkerWeightLoaderError.missingTensor(fullName)
                }
                selected.append(
                    (
                        String(fullName.dropFirst(checkpointPrefix.count)),
                        value
                    )
                )
            }
        }

        let expectedCount =
            manifest?.tensorCounts.thinker
                ?? model.parameters().flattened().count
        guard selected.count == expectedCount else {
            throw OmniThinkerWeightLoaderError.incorrectTensorCount(
                expected: expectedCount,
                actual: selected.count
            )
        }
        try model.update(
            parameters: ModuleParameters.unflattened(selected),
            verify: .all
        )
        if evaluateWeights {
            eval(model.parameters())
        }
        let duration = started.duration(to: .now)
        return OmniThinkerWeightLoadReport(
            modelDirectory: modelDirectory,
            shardNames: shardNames,
            tensorCount: selected.count,
            loadSeconds: duration.timeInterval
        )
    }
}

private extension Duration {
    var timeInterval: Double {
        let components = components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }
}
