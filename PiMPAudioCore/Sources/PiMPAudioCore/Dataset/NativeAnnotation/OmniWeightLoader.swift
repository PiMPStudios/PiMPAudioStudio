import Foundation
import MLX
import MLXNN

public enum OmniWeightLoaderError: LocalizedError {
    case missingIndex(URL)
    case missingShard(URL)
    case missingTensor(String)
    case incorrectTensorCount(expected: Int, actual: Int)
    case unsupportedConvShape(name: String, shape: [Int])

    public var errorDescription: String? {
        switch self {
        case let .missingIndex(url):
            "Missing safetensors index at \(url.path)."
        case let .missingShard(url):
            "Missing checkpoint shard at \(url.path)."
        case let .missingTensor(name):
            "Checkpoint shard does not contain \(name)."
        case let .incorrectTensorCount(expected, actual):
            "Loaded \(actual) audio-tower tensors; expected \(expected)."
        case let .unsupportedConvShape(name, shape):
            "Unsupported \(name) Conv1d weight shape \(shape)."
        }
    }
}

public struct OmniWeightLoadReport: Sendable {
    public let modelDirectory: URL
    public let shardNames: [String]
    public let tensorCount: Int
    public let convertedConvWeights: [String]

    public init(
        modelDirectory: URL,
        shardNames: [String],
        tensorCount: Int,
        convertedConvWeights: [String]
    ) {
        self.modelDirectory = modelDirectory
        self.shardNames = shardNames
        self.tensorCount = tensorCount
        self.convertedConvWeights = convertedConvWeights
    }
}

public enum OmniWeightLoader {
    public static let transcriberRevision =
        "53062cf7d0e09d463d7fc4c218ce2ecb71400f29"
    public static let captionerRevision =
        "7109e9af1c6cfe3371730310b3a5cb773145b11e"
    public static let towerPrefix = "thinker.audio_tower."
    public static let audioBosEosTensor = "audio_bos_eos_token.weight"
    public static let expectedTowerParameterCount = 488

    private struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    public static func defaultTranscriberSnapshotURL(
        fileManager: FileManager = .default
    ) -> URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent("models--ACE-Step--acestep-transcriber")
            .appendingPathComponent("snapshots")
            .appendingPathComponent(transcriberRevision)
    }

    public static func defaultCaptionerSnapshotURL(
        fileManager: FileManager = .default
    ) -> URL {
        fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent("models--ACE-Step--acestep-captioner")
            .appendingPathComponent("snapshots")
            .appendingPathComponent(captionerRevision)
    }

    @discardableResult
    public static func loadAudioTower(
        from modelDirectory: URL,
        validatedManifest: NativeModelPackManifest? = nil,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> (tower: Qwen25OmniAudioTower, report: OmniWeightLoadReport) {
        let configuration = try OmniAudioConfiguration.fromCheckpoint(
            at: modelDirectory
        )
        let tower = try Qwen25OmniAudioTower(configuration: configuration)
        let report = try loadAudioTowerWeights(
            into: tower,
            from: modelDirectory,
            validatedManifest: validatedManifest,
            cancellationCheck: cancellationCheck
        )
        return (tower, report)
    }

    @discardableResult
    public static func loadAudioTowerWeights(
        into tower: Qwen25OmniAudioTower,
        from modelDirectory: URL,
        validatedManifest: NativeModelPackManifest? = nil,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> OmniWeightLoadReport {
        let manifest = try validatedManifest
            ?? NativeModelPackValidator.loadIfPresent(at: modelDirectory)
        if let manifest {
            NativeModelQuantizer.quantize(
                model: tower,
                using: manifest.quantization
            )
        }
        let indexURL = modelDirectory.appendingPathComponent(
            "model.safetensors.index.json"
        )
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            throw OmniWeightLoaderError.missingIndex(indexURL)
        }
        let index = try JSONDecoder().decode(
            SafetensorsIndex.self,
            from: Data(contentsOf: indexURL)
        )

        let indexedTowerWeights = index.weightMap.filter {
            $0.key.hasPrefix(towerPrefix)
        }
        let shardNames = Array(Set(indexedTowerWeights.values)).sorted()
        var selected: [(String, MLXArray)] = []
        selected.reserveCapacity(expectedTowerParameterCount)
        var converted: [String] = []

        for shardName in shardNames {
            try cancellationCheck()
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw OmniWeightLoaderError.missingShard(shardURL)
            }
            let shard = try MLX.loadArrays(url: shardURL)
            let names = indexedTowerWeights
                .filter { $0.value == shardName }
                .map(\.key)
                .sorted()

            for fullName in names {
                try cancellationCheck()
                guard let raw = shard[fullName] else {
                    throw OmniWeightLoaderError.missingTensor(fullName)
                }
                let localName = String(fullName.dropFirst(towerPrefix.count))
                if localName == audioBosEosTensor {
                    continue
                }

                let value: MLXArray
                if localName == "conv1.weight" || localName == "conv2.weight" {
                    value = try convertConv1dWeight(raw, name: localName)
                    if value.shape != raw.shape {
                        converted.append(
                            "\(localName):\(raw.shape)->\(value.shape)"
                        )
                    }
                } else {
                    value = raw
                }
                selected.append((localName, value))
            }
        }

        let expectedCount =
            manifest?.tensorCounts.audioTower
                ?? tower.parameters().flattened().count
        guard selected.count == expectedCount else {
            throw OmniWeightLoaderError.incorrectTensorCount(
                expected: expectedCount,
                actual: selected.count
            )
        }
        try tower.update(
            parameters: ModuleParameters.unflattened(selected),
            verify: .all
        )
        return OmniWeightLoadReport(
            modelDirectory: modelDirectory,
            shardNames: shardNames,
            tensorCount: selected.count,
            convertedConvWeights: converted
        )
    }

    public static func convertConv1dWeight(
        _ weight: MLXArray,
        name: String
    ) throws -> MLXArray {
        guard weight.ndim == 3 else {
            throw OmniWeightLoaderError.unsupportedConvShape(
                name: name,
                shape: weight.shape
            )
        }
        if weight.shape[1] == 3 {
            return weight
        }
        if weight.shape[2] == 3 {
            return weight.swappedAxes(1, 2)
        }
        throw OmniWeightLoaderError.unsupportedConvShape(
            name: name,
            shape: weight.shape
        )
    }
}

/// Owns a loaded tower and exposes an explicit eviction boundary suitable for
/// a long-lived macOS application process.
public final class OmniAudioTowerHandle {
    private var storage: Qwen25OmniAudioTower?

    public init(tower: Qwen25OmniAudioTower) {
        storage = tower
    }

    public var isLoaded: Bool {
        storage != nil
    }

    public func withTower<T>(
        _ body: (Qwen25OmniAudioTower) throws -> T
    ) throws -> T {
        guard let storage else {
            throw OmniAudioTowerHandleError.evicted
        }
        return try body(storage)
    }

    public func evict() {
        storage = nil
        Memory.clearCache()
    }
}

public enum OmniAudioTowerHandleError: LocalizedError {
    case evicted

    public var errorDescription: String? {
        "The native Omni audio tower has been evicted."
    }
}
