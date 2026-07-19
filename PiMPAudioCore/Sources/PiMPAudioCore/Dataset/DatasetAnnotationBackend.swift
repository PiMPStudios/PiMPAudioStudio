import Foundation

public enum DatasetAnnotationTask: String, Codable, Hashable, Sendable {
    case caption
    case transcribe
}

public struct DatasetAnnotationRequest: Identifiable, Hashable, Sendable {
    public let id: String
    public let audioURL: URL

    public init(id: String, audioURL: URL) {
        self.id = id
        self.audioURL = audioURL
    }
}

public struct DatasetAnnotationGenerationOptions: Equatable, Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var repetitionPenalty: Float?

    public init(
        maxTokens: Int = 512,
        temperature: Float = 0,
        topP: Float = 0,
        repetitionPenalty: Float? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
    }

    public static let greedy = DatasetAnnotationGenerationOptions()

    public var persistedSettings: [String: String] {
        var settings = [
            "max_tokens": String(maxTokens),
            "temperature": String(temperature),
            "top_p": String(topP),
        ]
        if let repetitionPenalty {
            settings["repetition_penalty"] = String(repetitionPenalty)
        }
        return settings
    }
}

public struct DatasetAnnotationBackendDescriptor: Equatable, Sendable {
    public var id: String
    public var version: String
    public var modelID: String
    public var modelRevision: String
    public var promptVersion: String
    public var quantization: String

    public init(
        id: String,
        version: String = "",
        modelID: String = "",
        modelRevision: String = "",
        promptVersion: String = "",
        quantization: String = ""
    ) {
        self.id = id
        self.version = version
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.promptVersion = promptVersion
        self.quantization = quantization
    }
}

public struct DatasetAnnotationBackendResult: Sendable {
    public let requestID: String
    public let task: DatasetAnnotationTask
    public let text: String
    public let rawText: String
    public let language: String?
    public let descriptor: DatasetAnnotationBackendDescriptor
    public let settings: [String: String]
    public let generatedTokenCount: Int
    public let promptTokenCount: Int
    public let elapsedSeconds: Double
    public let diagnostics: OmniOutputDiagnostics

    public init(
        requestID: String,
        task: DatasetAnnotationTask,
        text: String,
        rawText: String,
        language: String?,
        descriptor: DatasetAnnotationBackendDescriptor,
        settings: [String: String],
        generatedTokenCount: Int,
        promptTokenCount: Int,
        elapsedSeconds: Double,
        diagnostics: OmniOutputDiagnostics
    ) {
        self.requestID = requestID
        self.task = task
        self.text = text
        self.rawText = rawText
        self.language = language
        self.descriptor = descriptor
        self.settings = settings
        self.generatedTokenCount = generatedTokenCount
        self.promptTokenCount = promptTokenCount
        self.elapsedSeconds = elapsedSeconds
        self.diagnostics = diagnostics
    }
}

public enum DatasetAnnotationProgressStage: String, Equatable, Sendable {
    case validatingModelPack
    case loadingModel
    case annotating
    case evictingModel
}

public struct DatasetAnnotationProgress: Equatable, Sendable {
    public let task: DatasetAnnotationTask
    public let stage: DatasetAnnotationProgressStage
    public let completedCount: Int
    public let totalCount: Int
    public let currentRequestID: String?

    public init(
        task: DatasetAnnotationTask,
        stage: DatasetAnnotationProgressStage,
        completedCount: Int,
        totalCount: Int,
        currentRequestID: String? = nil
    ) {
        self.task = task
        self.stage = stage
        self.completedCount = completedCount
        self.totalCount = totalCount
        self.currentRequestID = currentRequestID
    }
}

public enum DatasetAnnotationHardwareTier: String, Codable, Sendable {
    case unsupported
    case provisional
    case recommended

    public static func classify(physicalMemoryBytes: UInt64) -> Self {
        let gibibytes = Double(physicalMemoryBytes) / 1_073_741_824
        if gibibytes < 24 {
            return .unsupported
        }
        if gibibytes < 32 {
            return .provisional
        }
        return .recommended
    }
}

public struct DatasetAnnotationModelAvailability: Sendable {
    public let task: DatasetAnnotationTask
    public let directory: URL
    public let isInstalled: Bool
    public let descriptor: DatasetAnnotationBackendDescriptor?
    public let unavailableReason: String?

    public init(
        task: DatasetAnnotationTask,
        directory: URL,
        isInstalled: Bool,
        descriptor: DatasetAnnotationBackendDescriptor? = nil,
        unavailableReason: String? = nil
    ) {
        self.task = task
        self.directory = directory
        self.isInstalled = isInstalled
        self.descriptor = descriptor
        self.unavailableReason = unavailableReason
    }
}

public struct DatasetAnnotationBackendAvailability: Sendable {
    public let backendID: String
    public let hardwareTier: DatasetAnnotationHardwareTier
    public let models: [DatasetAnnotationModelAvailability]

    public init(
        backendID: String,
        hardwareTier: DatasetAnnotationHardwareTier,
        models: [DatasetAnnotationModelAvailability]
    ) {
        self.backendID = backendID
        self.hardwareTier = hardwareTier
        self.models = models
    }

    public var isFullyInstalled: Bool {
        models.allSatisfy(\.isInstalled)
    }
}

public protocol DatasetAnnotationBackend: Sendable {
    var backendID: String { get }

    func availability() async -> DatasetAnnotationBackendAvailability

    func annotate(
        requests: [DatasetAnnotationRequest],
        task: DatasetAnnotationTask,
        options: DatasetAnnotationGenerationOptions,
        onProgress: @Sendable @escaping (DatasetAnnotationProgress) -> Void
    ) async throws -> [DatasetAnnotationBackendResult]

    func evict() async
}
