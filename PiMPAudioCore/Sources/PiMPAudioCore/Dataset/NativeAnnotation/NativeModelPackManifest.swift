import CryptoKit
import Foundation

public struct NativeModelPackManifest: Codable, Equatable, Sendable {
    public static let filename = "native-model-pack.json"
    public static let currentSchemaVersion = 1
    public static let currentTensorMappingVersion = 1

    public struct Quantization: Codable, Equatable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public let mode: String

        public init(bits: Int, groupSize: Int, mode: String = "affine") {
            self.bits = bits
            self.groupSize = groupSize
            self.mode = mode
        }
    }

    public struct TensorCounts: Codable, Equatable, Sendable {
        public let thinker: Int
        public let audioTower: Int

        public init(thinker: Int, audioTower: Int) {
            self.thinker = thinker
            self.audioTower = audioTower
        }

        public var total: Int {
            thinker + audioTower
        }
    }

    public struct FileRecord: Codable, Equatable, Sendable {
        public let role: String
        public let path: String
        public let byteCount: Int64
        public let sha256: String

        public init(
            role: String,
            path: String,
            byteCount: Int64,
            sha256: String
        ) {
            self.role = role
            self.path = path
            self.byteCount = byteCount
            self.sha256 = sha256
        }
    }

    public let schemaVersion: Int
    public let packID: String
    public let task: OmniAnnotationTask
    public let upstreamModelID: String
    public let upstreamRevision: String
    public let upstreamLicense: String
    public let tensorMappingVersion: Int
    public let quantization: Quantization
    public let tensorCounts: TensorCounts
    public let expectedSizeBytes: Int64
    public let excludedComponents: [String]
    public let files: [FileRecord]

    public init(
        schemaVersion: Int = currentSchemaVersion,
        packID: String,
        task: OmniAnnotationTask,
        upstreamModelID: String,
        upstreamRevision: String,
        upstreamLicense: String,
        tensorMappingVersion: Int = currentTensorMappingVersion,
        quantization: Quantization,
        tensorCounts: TensorCounts,
        expectedSizeBytes: Int64,
        excludedComponents: [String],
        files: [FileRecord]
    ) {
        self.schemaVersion = schemaVersion
        self.packID = packID
        self.task = task
        self.upstreamModelID = upstreamModelID
        self.upstreamRevision = upstreamRevision
        self.upstreamLicense = upstreamLicense
        self.tensorMappingVersion = tensorMappingVersion
        self.quantization = quantization
        self.tensorCounts = tensorCounts
        self.expectedSizeBytes = expectedSizeBytes
        self.excludedComponents = excludedComponents
        self.files = files
    }
}

public enum NativeModelPackError: LocalizedError {
    case invalidManifest(String)
    case missingFile(URL)
    case unsafePath(String)
    case sizeMismatch(path: String, expected: Int64, actual: Int64)
    case checksumMismatch(path: String, expected: String, actual: String)
    case unreadableIndex(URL)

    public var errorDescription: String? {
        switch self {
        case .invalidManifest(let detail):
            return "Invalid native model pack: \(detail)"
        case .missingFile(let url):
            return "Native model pack is missing \(url.lastPathComponent)."
        case .unsafePath(let path):
            return "Native model pack contains an unsafe path: \(path)."
        case .sizeMismatch(let path, let expected, let actual):
            return "Native model pack file \(path) is \(actual) bytes; expected \(expected)."
        case .checksumMismatch(let path, let expected, let actual):
            return "Native model pack file \(path) has SHA-256 \(actual); expected \(expected)."
        case .unreadableIndex(let url):
            return "Native model pack index is unreadable at \(url.path)."
        }
    }
}

public enum NativeModelPackValidator {
    public static let requiredFiles: [String: String] = [
        "weights.thinker": "thinker.safetensors",
        "weights.audio_tower": "audio-tower.safetensors",
        "weights.index": "model.safetensors.index.json",
        "configuration": "config.json",
        "generation": "generation_config.json",
        "tokenizer": "tokenizer.json",
        "tokenizer.configuration": "tokenizer_config.json",
        "tokenizer.special_tokens": "special_tokens_map.json",
        "tokenizer.added_tokens": "added_tokens.json",
        "tokenizer.vocabulary": "vocab.json",
        "tokenizer.merges": "merges.txt",
        "prompt.chat_template": "chat_template.jinja",
        "prompt.annotation": "annotation-prompts.json",
        "processor.audio": "preprocessor_config.json",
    ]

    private struct Index: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    private struct AnnotationPrompts: Decodable {
        let schemaVersion: Int
        let caption: String
        let transcribe: String
    }

    public static func loadIfPresent(
        at directory: URL,
        cancellationCheck: OmniCancellationCheck = {
            try Task.checkCancellation()
        }
    ) throws -> NativeModelPackManifest? {
        let url = directory.appendingPathComponent(
            NativeModelPackManifest.filename
        )
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return try loadAndValidate(
            at: directory,
            cancellationCheck: cancellationCheck
        )
    }

    public static func loadManifestMetadata(
        at directory: URL
    ) throws -> NativeModelPackManifest {
        let manifestURL = directory.appendingPathComponent(
            NativeModelPackManifest.filename
        )
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw NativeModelPackError.missingFile(manifestURL)
        }
        let manifest = try JSONDecoder().decode(
            NativeModelPackManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        try validateMetadata(manifest)
        return manifest
    }

    @discardableResult
    public static func loadAndValidate(
        at directory: URL,
        cancellationCheck: OmniCancellationCheck = {
            try Task.checkCancellation()
        }
    ) throws -> NativeModelPackManifest {
        try cancellationCheck()
        let manifest = try loadManifestMetadata(at: directory)

        var roles: [String: NativeModelPackManifest.FileRecord] = [:]
        var paths = Set<String>()
        var totalBytes: Int64 = 0
        for record in manifest.files {
            try cancellationCheck()
            try validateRelativePath(record.path)
            guard roles[record.role] == nil else {
                throw NativeModelPackError.invalidManifest(
                    "duplicate file role \(record.role)"
                )
            }
            guard paths.insert(record.path).inserted else {
                throw NativeModelPackError.invalidManifest(
                    "duplicate file path \(record.path)"
                )
            }
            guard record.byteCount >= 0 else {
                throw NativeModelPackError.invalidManifest(
                    "negative byte count for \(record.path)"
                )
            }
            guard record.sha256.count == 64,
                  record.sha256.allSatisfy({ $0.isHexDigit })
            else {
                throw NativeModelPackError.invalidManifest(
                    "invalid SHA-256 for \(record.path)"
                )
            }
            let fileURL = directory.appendingPathComponent(record.path)
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                throw NativeModelPackError.missingFile(fileURL)
            }
            let values = try fileURL.resourceValues(
                forKeys: [.fileSizeKey, .isRegularFileKey]
            )
            guard values.isRegularFile == true else {
                throw NativeModelPackError.invalidManifest(
                    "\(record.path) is not a regular file"
                )
            }
            let actualSize = Int64(values.fileSize ?? -1)
            guard actualSize == record.byteCount else {
                throw NativeModelPackError.sizeMismatch(
                    path: record.path,
                    expected: record.byteCount,
                    actual: actualSize
                )
            }
            let actualHash = try sha256(
                of: fileURL,
                cancellationCheck: cancellationCheck
            )
            guard actualHash == record.sha256.lowercased() else {
                throw NativeModelPackError.checksumMismatch(
                    path: record.path,
                    expected: record.sha256.lowercased(),
                    actual: actualHash
                )
            }
            roles[record.role] = record
            totalBytes += actualSize
        }

        for (role, path) in requiredFiles {
            guard roles[role]?.path == path else {
                throw NativeModelPackError.invalidManifest(
                    "required role \(role) must reference \(path)"
                )
            }
        }
        guard totalBytes == manifest.expectedSizeBytes else {
            throw NativeModelPackError.invalidManifest(
                "expected size \(manifest.expectedSizeBytes) does not match "
                    + "the \(totalBytes)-byte file set"
            )
        }

        let promptsURL = directory.appendingPathComponent(
            requiredFiles["prompt.annotation"]!
        )
        let prompts = try JSONDecoder().decode(
            AnnotationPrompts.self,
            from: Data(contentsOf: promptsURL)
        )
        guard prompts.schemaVersion == 1,
              prompts.caption == OmniAnnotationTask.caption.prompt,
              prompts.transcribe == OmniAnnotationTask.transcribe.prompt
        else {
            throw NativeModelPackError.invalidManifest(
                "annotation prompts do not match prompt contract v1"
            )
        }

        let indexURL = directory.appendingPathComponent(
            requiredFiles["weights.index"]!
        )
        guard let index = try? JSONDecoder().decode(
            Index.self,
            from: Data(contentsOf: indexURL)
        ) else {
            throw NativeModelPackError.unreadableIndex(indexURL)
        }
        try validateIndex(index, manifest: manifest)
        return manifest
    }

    public static func sha256(
        of url: URL,
        cancellationCheck: OmniCancellationCheck = {
            try Task.checkCancellation()
        }
    ) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }
        var hasher = SHA256()
        while true {
            try cancellationCheck()
            let data = try handle.read(upToCount: 1_048_576) ?? Data()
            if data.isEmpty {
                break
            }
            hasher.update(data: data)
        }
        return hasher.finalize().map {
            String(format: "%02x", $0)
        }.joined()
    }

    public static func record(
        role: String,
        path: String,
        in directory: URL
    ) throws -> NativeModelPackManifest.FileRecord {
        try validateRelativePath(path)
        let url = directory.appendingPathComponent(path)
        let values = try url.resourceValues(
            forKeys: [.fileSizeKey, .isRegularFileKey]
        )
        guard values.isRegularFile == true, let size = values.fileSize else {
            throw NativeModelPackError.missingFile(url)
        }
        return NativeModelPackManifest.FileRecord(
            role: role,
            path: path,
            byteCount: Int64(size),
            sha256: try sha256(of: url)
        )
    }

    private static func validateMetadata(
        _ manifest: NativeModelPackManifest
    ) throws {
        guard manifest.schemaVersion
                == NativeModelPackManifest.currentSchemaVersion
        else {
            throw NativeModelPackError.invalidManifest(
                "unsupported schema version \(manifest.schemaVersion)"
            )
        }
        guard manifest.tensorMappingVersion
                == NativeModelPackManifest.currentTensorMappingVersion
        else {
            throw NativeModelPackError.invalidManifest(
                "unsupported tensor mapping version "
                    + "\(manifest.tensorMappingVersion)"
            )
        }
        guard !manifest.packID.isEmpty,
              manifest.packID.allSatisfy({
                  $0.isLetter || $0.isNumber || $0 == "-" || $0 == "."
              })
        else {
            throw NativeModelPackError.invalidManifest("invalid pack ID")
        }
        let expected: (modelID: String, revision: String)
        switch manifest.task {
        case .caption:
            expected = (
                "ACE-Step/acestep-captioner",
                OmniWeightLoader.captionerRevision
            )
        case .transcribe:
            expected = (
                "ACE-Step/acestep-transcriber",
                OmniWeightLoader.transcriberRevision
            )
        }
        guard manifest.upstreamModelID == expected.modelID,
              manifest.upstreamRevision == expected.revision
        else {
            throw NativeModelPackError.invalidManifest(
                "upstream model or revision is not approved for "
                    + manifest.task.rawValue
            )
        }
        guard manifest.upstreamLicense.uppercased() == "MIT" else {
            throw NativeModelPackError.invalidManifest(
                "upstream license must be MIT"
            )
        }
        guard [4, 8].contains(manifest.quantization.bits),
              manifest.quantization.groupSize == 64,
              manifest.quantization.mode == "affine"
        else {
            throw NativeModelPackError.invalidManifest(
                "unsupported quantization settings"
            )
        }
        guard manifest.tensorCounts.thinker > 0,
              manifest.tensorCounts.audioTower > 0
        else {
            throw NativeModelPackError.invalidManifest(
                "tensor counts must be positive"
            )
        }
        let exclusions = Set(manifest.excludedComponents)
        let requiredExclusions = Set(["talker", "token2wav", "vision"])
        guard exclusions.isSuperset(of: requiredExclusions) else {
            throw NativeModelPackError.invalidManifest(
                "talker, token2wav, and vision exclusions are required"
            )
        }
    }

    private static func validateIndex(
        _ index: Index,
        manifest: NativeModelPackManifest
    ) throws {
        let thinkerPrefix = "thinker.model."
        let headPrefix = "thinker.lm_head."
        let audioPrefix = OmniWeightLoader.towerPrefix
        var thinkerCount = 0
        var audioCount = 0
        for (name, shard) in index.weightMap {
            if name.hasPrefix(thinkerPrefix) || name.hasPrefix(headPrefix) {
                thinkerCount += 1
                guard shard == requiredFiles["weights.thinker"] else {
                    throw NativeModelPackError.invalidManifest(
                        "Thinker tensor \(name) references \(shard)"
                    )
                }
            } else if name.hasPrefix(audioPrefix) {
                audioCount += 1
                guard shard == requiredFiles["weights.audio_tower"] else {
                    throw NativeModelPackError.invalidManifest(
                        "audio tensor \(name) references \(shard)"
                    )
                }
            } else {
                throw NativeModelPackError.invalidManifest(
                    "excluded or unknown tensor \(name)"
                )
            }
        }
        guard thinkerCount == manifest.tensorCounts.thinker,
              audioCount == manifest.tensorCounts.audioTower,
              thinkerCount + audioCount == index.weightMap.count
        else {
            throw NativeModelPackError.invalidManifest(
                "index tensor counts do not match the manifest"
            )
        }
    }

    private static func validateRelativePath(_ path: String) throws {
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.contains("\\"),
              !path.contains("\0")
        else {
            throw NativeModelPackError.unsafePath(path)
        }
        let components = path.split(
            separator: "/",
            omittingEmptySubsequences: false
        )
        guard components.allSatisfy({
            !$0.isEmpty && $0 != "." && $0 != ".."
        }) else {
            throw NativeModelPackError.unsafePath(path)
        }
    }
}
